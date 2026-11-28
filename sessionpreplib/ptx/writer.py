"""Pure-Python offline Pro Tools ``.ptx`` session writer (in-memory and atomic file-backed)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import struct
from typing import Any
import uuid

from ._document import PtxBuffer
from .codec import (
    atomic_write_bytes,
    iter_top_chunks,
    normalize_ptx_uuid16,
    validate_index_0002,
)
from .models import (
    VALID_PROTOOLS_BIT_DEPTHS,
    VALID_PROTOOLS_SAMPLE_RATES,
    PtxChunk,
    PtxCorruptFileError,
    PtxValidationError,
)
from .parser import PtxSession
from .patch_builder import PtxPatchBuilder, empty_patch_summary
from .patch_session import patch_session_settings_on_buffer
from .writer_blueprints import resolve_track_format_codes
from .writer_chunks import serialize_directory_tree_103a
from .writer_index import apply_rebuilt_index_0002
from .writer_media_mutations import (
    ClipWriteSpec,
    MediaSurvivalPlan,
    MidiSurvivalPlan,
    mutate_0f3d_volumes,
    mutate_1004_audio_files,
    mutate_1015_and_1054_media,
    mutate_1058_midi_tracks,
    mutate_262a_clips,
    plan_existing_media_survival,
    plan_existing_midi_survival,
)
from .writer_primitives import (
    build_zchunk,
    generate_ptx_uuid,
    parse_index_0002,
    serialize_folder_tree_210c,
)
from .writer_specs import (
    OfflineAudioTrackRequest,
    OfflineClipSpec as OfflineClipSpec,
    OfflineFadeRequest as OfflineFadeRequest,
    OfflineGroupRequest as OfflineGroupRequest,
    OfflineMarkerRequest as OfflineMarkerRequest,
    OfflineNonAudioTrackRequest,
    OfflinePtxWriteRequest,
    OfflinePtxWriteResult,
    resolve_output_bus,
    validate_template_supported,
    validate_write_request,
)
from .writer_track_mutations import (
    NewAudioTrackSpec,
    TrackSlotOp,
    mutate_202b_groups_for_track_map,
    mutate_2030_markers_for_track_map,
    mutate_2107_tracks,
    mutate_210c_folders,
    mutate_2519_tracks,
    mutate_2587_ui_state,
    mutate_2624_mixer,
)

_FOLDER_TRACK_TYPES = frozenset({"RoutingFolder", "BasicFolder"})


@dataclass(frozen=True)
class _StructuralMutationPlan:
    old_track_count: int
    request: OfflinePtxWriteRequest
    new_specs: tuple[NewAudioTrackSpec, ...]
    clips: tuple[ClipWriteSpec, ...]
    layout: tuple[TrackSlotOp, ...]
    old_to_new: Mapping[int, int]
    folder_insertions: Mapping[bytes, Sequence[bytes]]
    deleted_uuids: frozenset[bytes]
    survival: MediaSurvivalPlan
    midi_survival: MidiSurvivalPlan


def _chunk_u32_count(chunk_raw: bytes, label: str) -> int:
    if len(chunk_raw) < 13:
        raise PtxCorruptFileError(f"Truncated {label} container chunk")
    return int(struct.unpack_from("<I", chunk_raw, 9)[0])


def _empty_optional_chunk(tag: int) -> bytes:
    if tag == 0x0F3D:
        return build_zchunk(0x0F3D, struct.pack("<I", 0), version=1)
    if tag == 0x1004:
        empty_103a = serialize_directory_tree_103a([], version=1)
        return build_zchunk(0x1004, struct.pack("<I", 0) + empty_103a + struct.pack("<I", 0) + empty_103a, version=3)
    if tag == 0x2637:
        return build_zchunk(0x2637, struct.pack("<I", 0), version=1)
    if tag == 0x262A:
        return build_zchunk(0x262A, struct.pack("<I", 0), version=1)
    if tag == 0x1054:
        return build_zchunk(0x1054, struct.pack("<I", 0) + build_zchunk(0x4828, b"\x00\x00\x00\x00", version=2), version=2)
    if tag == 0x1015:
        return build_zchunk(0x1015, struct.pack("<I", 0) + build_zchunk(0x4829, b"\x00\x00\x00\x00", version=2), version=2)
    if tag == 0x1058:
        return build_zchunk(0x1058, struct.pack("<I", 0), version=1)
    if tag == 0x2030:
        return build_zchunk(0x2030, struct.pack("<I", 0) + (b"\x00" * 8), version=5)
    if tag == 0x210C:
        return serialize_folder_tree_210c([], version=1)
    raise PtxValidationError(f"Unsupported synthesized chunk tag 0x{tag:04x}")


def _make_clip_spec(
    name: str,
    length_samples: int,
    audio_file_index: int,
    channel_index: int,
    *,
    start_sample: int = 0,
    sync_point_samples: int = 0,
    orig_timestamp: int = 0,
    user_timestamp: int = 0,
) -> ClipWriteSpec:
    return ClipWriteSpec(
        clip_name=name, length_samples=length_samples,
        audio_file_index=audio_file_index, channel_index=channel_index,
        clip_guid16=uuid.uuid4().bytes_le, pt_uuid16=generate_ptx_uuid(),
        start_sample=start_sample, sync_point_samples=sync_point_samples,
        orig_timestamp=orig_timestamp, user_timestamp=user_timestamp,
    )


def _resolve_parent_uuid(
    parent_uuid_hex: str | None,
    parent_name: str | None,
    folder_uuid_by_name_lc: Mapping[str, bytes],
) -> bytes | None:
    if parent_uuid_hex is not None:
        return normalize_ptx_uuid16(parent_uuid_hex)
    if parent_name is not None:
        return folder_uuid_by_name_lc[parent_name.strip().lower()]
    return None


def _prepare_clips_and_specs(
    session: PtxSession,
    requests: Sequence[OfflineAudioTrackRequest],
    folder_uuid_by_name_lc: Mapping[str, bytes],
    *,
    base_audio_file_index: int = 0,
    base_clip_index: int = 0,
    base_lane_index: int = 0,
    base_playlist_index: int = 0,
) -> tuple[tuple[NewAudioTrackSpec, ...], tuple[ClipWriteSpec, ...]]:
    clips: list[ClipWriteSpec] = []
    specs: list[NewAudioTrackSpec] = []
    next_lane = base_lane_index
    for req_offset, req in enumerate(requests):
        _lw, _mw, _ew, resolved_ch = resolve_track_format_codes(req.channels, req.track_format)
        out_bus = resolve_output_bus(session, req.output_bus_name) if req.output_bus_name else None
        lane_indices = tuple(range(next_lane, next_lane + resolved_ch))
        next_lane += resolved_ch
        clip = req.resolved_clip()
        if clip is None:
            clip_indices: tuple[int, ...] = ()
            tl_start, c_muted, c_tlock, c_elock, c_color = req.start_sample, False, False, False, 0xFFFE
        else:
            target_af_idx = base_audio_file_index + clip.audio_file_index
            suffixes = ("",) if resolved_ch == 1 else ((".L", ".R") if resolved_ch == 2 else tuple(f".Ch{k + 1}" for k in range(resolved_ch)))
            c_list: list[int] = []
            for ch_i, suf in enumerate(suffixes):
                c_list.append(base_clip_index + len(clips))
                clips.append(_make_clip_spec(
                    f"{clip.clip_base_name}{suf}", clip.length_samples, target_af_idx, ch_i,
                    start_sample=clip.source_start_sample, sync_point_samples=clip.sync_point_samples,
                    orig_timestamp=clip.orig_timestamp, user_timestamp=clip.user_timestamp,
                ))
            clip_indices = tuple(c_list)
            tl_start, c_muted, c_tlock, c_elock, c_color = (
                clip.timeline_start_sample,
                clip.is_muted,
                bool(clip.is_time_locked or clip.is_edit_locked),
                clip.is_edit_locked,
                clip.color_index,
            )
        parent_uid = _resolve_parent_uuid(req.parent_folder_uuid_hex, req.parent_folder_name, folder_uuid_by_name_lc)
        specs.append(NewAudioTrackSpec(
            track_name=req.track_name, track_uuid16=generate_ptx_uuid(),
            playlist_uuid16=generate_ptx_uuid(), channels=resolved_ch,
            parent_folder_uuid16=parent_uid, output_bus=out_bus,
            color_index=req.color_index, clip_indices=clip_indices,
            lane_indices=lane_indices, playlist_index=base_playlist_index + req_offset,
            start_sample=tl_start, track_format=req.track_format,
            clip_muted=c_muted, clip_time_locked=c_tlock, clip_edit_locked=c_elock, clip_color_index=c_color,
        ))
    return tuple(specs), tuple(clips)


def _prepare_non_audio_specs(
    session: PtxSession,
    extra_tracks: Sequence[OfflineNonAudioTrackRequest],
    folder_uuid_by_name_lc: Mapping[str, bytes],
    preassigned_uuids: Mapping[int, bytes],
    *,
    base_midi_index: int = 0,
) -> tuple[NewAudioTrackSpec, ...]:
    specs: list[NewAudioTrackSpec] = []
    next_midi = base_midi_index
    for idx, extra in enumerate(extra_tracks):
        out_bus = resolve_output_bus(session, extra.output_bus_name) if extra.output_bus_name else None
        parent_uid = _resolve_parent_uuid(extra.parent_folder_uuid_hex, extra.parent_folder_name, folder_uuid_by_name_lc)
        channels = 1 if extra.track_type in ("Vca", "BasicFolder", "Midi") else resolve_track_format_codes(extra.channels, extra.track_format)[3]
        midi_idx: int | None = None
        midi_uid = b"\x00" * 16
        if extra.track_type in ("Midi", "Instrument"):
            midi_idx = next_midi
            next_midi += 1
            midi_uid = generate_ptx_uuid()
        specs.append(NewAudioTrackSpec(
            track_name=extra.track_name, track_uuid16=preassigned_uuids[idx],
            playlist_uuid16=b"\x00" * 16, channels=channels,
            parent_folder_uuid16=parent_uid, output_bus=out_bus,
            color_index=extra.color_index, clip_indices=(), lane_indices=(),
            playlist_index=0, start_sample=0, track_type=extra.track_type,
            is_folder_open=extra.is_folder_open, track_format=extra.track_format,
            midi_index=midi_idx, midi_uuid16=midi_uid,
        ))
    return tuple(specs)


def _compute_unified_layout(
    session: PtxSession,
    new_specs: Sequence[NewAudioTrackSpec],
    delete_names: Sequence[str],
) -> tuple[tuple[TrackSlotOp, ...], dict[int, int], dict[bytes, list[bytes]], frozenset[bytes], tuple[str, ...]]:
    """Build final ordered track slots, old->new index map, folder insertions, and deleted UUIDs."""
    delete_lc = {n.lower() for n in delete_names}
    by_folder: dict[bytes, list[NewAudioTrackSpec]] = {}
    top_level_new: list[NewAudioTrackSpec] = []
    for spec in new_specs:
        if spec.parent_folder_uuid16 is None:
            top_level_new.append(spec)
        else:
            by_folder.setdefault(spec.parent_folder_uuid16, []).append(spec)

    layout: list[TrackSlotOp] = []
    old_to_new: dict[int, int] = {}
    folder_insertions: dict[bytes, list[bytes]] = {}
    deleted_uuids: set[bytes] = set()
    deleted_actual: list[str] = []
    placed_uuids: set[bytes] = set()

    def _emit_children(folder_uuid: bytes, stack: set[bytes]) -> None:
        children = by_folder.get(folder_uuid)
        if not children:
            return
        folder_insertions[folder_uuid] = [c.track_uuid16 for c in children]
        for child in children:
            if child.track_uuid16 in stack:
                raise PtxValidationError(f"Cycle detected in new folder hierarchy at {child.track_name!r}")
            layout.append(TrackSlotOp(new_track=child))
            placed_uuids.add(child.track_uuid16)
            if child.track_type in _FOLDER_TRACK_TYPES:
                _emit_children(child.track_uuid16, stack | {child.track_uuid16})

    for old_idx, trk in enumerate(session.tracks):
        t_uuid = bytes.fromhex(trk.uuid_hex) if trk.uuid_hex else b""
        if trk.name.lower() in delete_lc:
            if t_uuid:
                deleted_uuids.add(t_uuid)
            deleted_actual.append(trk.name)
            continue
        old_to_new[old_idx] = len(layout)
        layout.append(TrackSlotOp(old_index=old_idx))
        if t_uuid:
            _emit_children(t_uuid, {t_uuid})
    for spec in top_level_new:
        layout.append(TrackSlotOp(new_track=spec))
        placed_uuids.add(spec.track_uuid16)
        if spec.track_type in _FOLDER_TRACK_TYPES:
            _emit_children(spec.track_uuid16, {spec.track_uuid16})
    if len(placed_uuids) != len(new_specs):
        raise PtxValidationError(f"Placed {len(placed_uuids)}/{len(new_specs)} requested tracks; check parent folder references")
    return tuple(layout), old_to_new, folder_insertions, frozenset(deleted_uuids), tuple(deleted_actual)


def _assemble_chunks_with_insertions(
    dec_header: bytes,
    top_chunks: Sequence[PtxChunk],
    replacements: Mapping[int, bytes],
) -> bytearray:
    """Emit top-level chunks replacing last occurrence of mutated tags and inserting synthesized chunks."""
    last_idx_by_tag: dict[int, int] = {chunk.tag: idx for idx, chunk in enumerate(top_chunks)}
    insertions_after: dict[int, list[bytes]] = {}
    insertions_before: dict[int, list[bytes]] = {}

    def _schedule_missing(tag: int, *, after: Sequence[int] = (), before: Sequence[int] = ()) -> None:
        if tag in last_idx_by_tag or tag not in replacements:
            return
        for anchor in after:
            if anchor in last_idx_by_tag:
                insertions_after.setdefault(last_idx_by_tag[anchor], []).append(replacements[tag])
                return
        for anchor in before:
            if anchor in last_idx_by_tag:
                insertions_before.setdefault(last_idx_by_tag[anchor], []).append(replacements[tag])
                return
        insertions_before.setdefault(last_idx_by_tag.get(0x0002, len(top_chunks)), []).append(replacements[tag])

    _schedule_missing(0x0F3D, after=(0x2032, 0x1028), before=(0x1004, 0x2027))
    _schedule_missing(0x1004, after=(0x0F3D, 0x2032, 0x1028), before=(0x2027, 0x2096))
    _schedule_missing(0x2637, after=(0x204D,), before=(0x262C, 0x262A, 0x2721, 0x1054, 0x1015, 0x1055, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x262A, after=(0x2637, 0x2632, 0x204D), before=(0x2721, 0x1054, 0x1015, 0x1055, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x1054, after=(0x2437, 0x2029), before=(0x1015, 0x1055, 0x210C, 0x2107))
    _schedule_missing(0x1015, after=(0x1054, 0x2437), before=(0x1055, 0x1058, 0x210C, 0x2107))
    _schedule_missing(0x1058, after=(0x1055, 0x1015, 0x1054), before=(0x2604, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x2030, after=(0x271A, 0x2505, 0x2066), before=(0x2550, 0x2587, 0x0002))
    _schedule_missing(0x210C, before=(0x2107, 0x2624))

    out = bytearray(dec_header)
    for idx, chunk in enumerate(top_chunks):
        for raw_ins in insertions_before.get(idx, ()):
            out.extend(raw_ins)
        out.extend(replacements[chunk.tag] if (chunk.tag in replacements and last_idx_by_tag[chunk.tag] == idx) else chunk.raw)
        for raw_ins in insertions_after.get(idx, ()):
            out.extend(raw_ins)
    for raw_ins in insertions_before.get(len(top_chunks), ()):
        out.extend(raw_ins)
    return out


def _apply_structural_mutations(dec_base: bytes, top_chunks: Sequence[PtxChunk], plan: _StructuralMutationPlan) -> bytearray:
    template_index = parse_index_0002(dec_base)
    if template_index is None:
        raise PtxCorruptFileError("Template .ptx lacks trailing 0x0002 chunk-offset index")
    by_tag = {c.tag: c.raw for c in top_chunks}
    req = plan.request
    new_1004, root_path = mutate_1004_audio_files(
        by_tag.get(0x1004) or _empty_optional_chunk(0x1004),
        req.audio_files, session_dir=req.session_dir, volume_serial_u32=req.volume_serial_u32,
    )
    new_1015, new_1054 = mutate_1015_and_1054_media(
        by_tag.get(0x1015) or _empty_optional_chunk(0x1015),
        by_tag.get(0x1054) or _empty_optional_chunk(0x1054),
        plan.new_specs, survival=plan.survival,
    )
    replacements: dict[int, bytes] = {
        0x1015: new_1015,
        0x1054: new_1054,
        0x202B: mutate_202b_groups_for_track_map(by_tag[0x202B], plan.old_to_new, len(plan.layout)),
        0x2107: mutate_2107_tracks(by_tag[0x2107], plan.layout, expected_old_count=plan.old_track_count),
        0x2519: mutate_2519_tracks(by_tag[0x2519], plan.layout, expected_old_count=plan.old_track_count),
        0x2624: mutate_2624_mixer(
            by_tag[0x2624], plan.layout,
            expected_old_count=plan.old_track_count,
            old_playlist_to_new=plan.survival.old_playlist_to_new,
            old_midi_to_new=plan.midi_survival.old_midi_to_new,
        ),
    }
    if 0x1004 in by_tag or req.audio_files:
        replacements[0x1004] = new_1004
    if 0x262A in by_tag or plan.clips:
        replacements[0x262A] = mutate_262a_clips(by_tag.get(0x262A) or _empty_optional_chunk(0x262A), plan.clips)
    has_midi_change = any(nt.track_type in ("Midi", "Instrument") for nt in plan.new_specs) or (
        0x1058 in by_tag and len(plan.midi_survival.entries) != _chunk_u32_count(by_tag[0x1058], "0x1058")
    )
    if has_midi_change:
        replacements[0x1058] = mutate_1058_midi_tracks(
            by_tag.get(0x1058) or _empty_optional_chunk(0x1058), plan.new_specs, midi_survival=plan.midi_survival,
        )
    if 0x210C in by_tag or plan.folder_insertions:
        replacements[0x210C] = mutate_210c_folders(by_tag.get(0x210C) or _empty_optional_chunk(0x210C), plan.folder_insertions, set(plan.deleted_uuids))
    if 0x2637 not in by_tag and req.clip_gains_db:
        replacements[0x2637] = _empty_optional_chunk(0x2637)
    if 0x2030 not in by_tag and req.markers:
        replacements[0x2030] = _empty_optional_chunk(0x2030)
    if 0x2587 in by_tag:
        replacements[0x2587] = mutate_2587_ui_state(by_tag[0x2587], plan.layout, expected_old_count=plan.old_track_count)
    if 0x2030 in by_tag:
        remapped_2030 = mutate_2030_markers_for_track_map(by_tag[0x2030], plan.old_to_new)
        if remapped_2030 != by_tag[0x2030]:
            replacements[0x2030] = remapped_2030
    if req.audio_files:
        replacements[0x0F3D] = mutate_0f3d_volumes(by_tag.get(0x0F3D) or _empty_optional_chunk(0x0F3D), req.volume_serial_u32, req.volume_label, root_path)

    out = _assemble_chunks_with_insertions(dec_base[:0x1F], top_chunks, replacements)
    apply_rebuilt_index_0002(out, template_index, dec_base)
    return out


def _queue_group_requests(builder: PtxPatchBuilder, group_requests: Sequence[Any]) -> None:
    for grp in group_requests:
        if grp.action == "create":
            builder.add_group(
                grp.name, grp.member_tracks, vca_track=grp.vca_track, group_type=grp.group_type,
                is_edit_active=grp.is_edit_active, is_mix_active=grp.is_mix_active,
            )
        elif grp.action == "add_members":
            kwargs: dict[str, Any] = {
                "add_tracks": grp.member_tracks,
                "is_edit_active": grp.is_edit_active,
                "is_mix_active": grp.is_mix_active,
            }
            if grp.vca_track:
                kwargs["vca_track"] = grp.vca_track
                kwargs["group_type"] = grp.group_type
            builder.modify_group(grp.name, **kwargs)
        elif grp.action == "link_vca" and grp.vca_track:
            builder.modify_group(
                grp.name, vca_track=grp.vca_track, group_type=grp.group_type,
                is_edit_active=grp.is_edit_active, is_mix_active=grp.is_mix_active,
            )


def _apply_post_patches_on_buffer(
    ptx_buf: PtxBuffer,
    request: OfflinePtxWriteRequest,
    session_settings_summary: dict[str, Any],
) -> tuple[PtxBuffer, dict[str, Any]]:
    has_work = bool(
        request.clip_gains_db or request.slot_a_trims_db or request.track_volumes_db
        or request.track_states or request.markers or request.fades or request.group_requests
    )
    if not has_work:
        summary = empty_patch_summary()
        summary["session_settings"] = dict(session_settings_summary)
        return ptx_buf, summary
    builder = PtxPatchBuilder.from_buffer(ptx_buf)
    if request.clip_gains_db:
        builder.set_clip_gains(request.clip_gains_db)
    if request.slot_a_trims_db:
        builder.set_slot_a_trim_gains(request.slot_a_trims_db, activate_if_inactive=True)
    if request.track_volumes_db:
        builder.set_track_volumes(request.track_volumes_db)
    if request.track_states:
        builder.set_track_states(request.track_states)
    for f in request.fades:
        builder.add_fade(
            f.track_name, f.fade_type, f.start_sample, f.total_samples,
            pre_samples=f.pre_samples, shape=f.shape, crossfade_link=f.crossfade_link,
        )
    for m in request.markers:
        builder.add_marker(
            m.name, m.start_pos,
            number=m.number, end_pos=m.end_pos, reference=m.reference,
            time_properties=m.time_properties, comments=m.comments,
            color_index=m.color_index, color_rgb=m.color_rgb, track=m.track_name,
            zoom_settings=m.zoom_settings, pre_post_roll_times=m.pre_post_roll_times,
            track_visibility=m.track_visibility, track_heights=m.track_heights,
            group_enables=m.group_enables, window_configuration=m.window_configuration,
        )
    _queue_group_requests(builder, request.group_requests)
    out_buf, summary = builder.build_buffer(validate=False)
    summary["session_settings"].update(session_settings_summary)
    return out_buf, summary


def _verify_written_buffer(
    ptx_buf: PtxBuffer,
    expected_track_count: int,
    created_names: Sequence[str],
    expected_audio_file_count: int,
    expected_clip_count: int,
    expected_playlist_count: int,
    *,
    baseline_warnings: Sequence[str] = (),
) -> None:
    dec = bytes(ptx_buf.data)
    if not validate_index_0002(dec):
        raise PtxCorruptFileError("Offline .ptx verification failed: trailing 0x0002 index missing or invalid")
    by_tag = {c.tag: c.raw for c in iter_top_chunks(dec, strict=True)}
    session = PtxSession(dec, xor_delta=ptx_buf.xor_delta)
    baseline_set = set(baseline_warnings)
    new_warnings = [w for w in session.warnings if w not in baseline_set]
    if new_warnings:
        raise PtxCorruptFileError(f"Offline .ptx verification produced new parser warnings: {new_warnings}")
    if len(session.tracks) != expected_track_count:
        raise PtxCorruptFileError(
            f"Offline .ptx verification failed: expected {expected_track_count} tracks, parsed {len(session.tracks)}"
        )
    parsed_names = {t.name for t in session.tracks}
    missing = [n for n in created_names if n not in parsed_names]
    if missing:
        raise PtxCorruptFileError(f"Offline .ptx verification failed: missing created tracks {missing}")
    af_hdr = _chunk_u32_count(by_tag[0x1004], "0x1004") if 0x1004 in by_tag else 0
    if af_hdr != expected_audio_file_count:
        raise PtxCorruptFileError(f"Offline .ptx verification failed: expected {expected_audio_file_count} 0x1004 files, got {af_hdr}")
    clip_hdr = _chunk_u32_count(by_tag[0x262A], "0x262A") if 0x262A in by_tag else 0
    if clip_hdr != expected_clip_count:
        raise PtxCorruptFileError(f"Offline .ptx verification failed: expected {expected_clip_count} 0x262A clips, got {clip_hdr}")
    pl_hdr = _chunk_u32_count(by_tag[0x1015], "0x1015") if 0x1015 in by_tag else 0
    if pl_hdr != expected_playlist_count or len(session.playlists) != expected_playlist_count:
        raise PtxCorruptFileError(
            f"Offline .ptx verification failed: expected {expected_playlist_count} playlists, got header={pl_hdr} parsed={len(session.playlists)}"
        )


def _prepare_validated_base(
    template_bytes: bytes,
    request: OfflinePtxWriteRequest,
) -> tuple[PtxBuffer, list[PtxChunk], dict[str, Any]]:
    eff_settings = request.effective_session_settings()
    if eff_settings.sample_rate is None or isinstance(eff_settings.sample_rate, bool) or eff_settings.sample_rate not in VALID_PROTOOLS_SAMPLE_RATES:
        raise PtxValidationError(f"Unsupported Pro Tools sample rate: {eff_settings.sample_rate!r}")
    if eff_settings.bit_depth is None or isinstance(eff_settings.bit_depth, bool) or eff_settings.bit_depth not in VALID_PROTOOLS_BIT_DEPTHS:
        raise PtxValidationError(f"Unsupported Pro Tools bit depth: {eff_settings.bit_depth!r}")
    ptx_buf = PtxBuffer.from_encrypted(template_bytes)
    validate_template_supported(iter_top_chunks(bytes(ptx_buf.data), strict=True))
    settings_summary = patch_session_settings_on_buffer(ptx_buf, eff_settings)
    return ptx_buf, iter_top_chunks(bytes(ptx_buf.data), strict=True), settings_summary


def _plan_folder_uuids(
    session: PtxSession,
    request: OfflinePtxWriteRequest,
    delete_lc: set[str],
) -> tuple[dict[str, bytes], dict[int, bytes]]:
    folder_uuid_by_name_lc = {
        trk.name.lower(): bytes.fromhex(trk.uuid_hex)
        for trk in session.tracks
        if trk.name.lower() not in delete_lc and trk.track_type in _FOLDER_TRACK_TYPES and trk.uuid_hex
    }
    preassigned_extra_uuids: dict[int, bytes] = {}
    for idx, extra in enumerate(request.extra_tracks):
        uid = normalize_ptx_uuid16(extra.track_uuid_hex) if extra.track_uuid_hex is not None else generate_ptx_uuid()
        preassigned_extra_uuids[idx] = uid
        if extra.track_type in _FOLDER_TRACK_TYPES:
            folder_uuid_by_name_lc[extra.track_name.strip().lower()] = uid
    return folder_uuid_by_name_lc, preassigned_extra_uuids


def build_offline_ptx_bytes(
    template_bytes: bytes,
    request: OfflinePtxWriteRequest,
) -> tuple[bytes, OfflinePtxWriteResult]:
    """Synthesize a complete encrypted Pro Tools ``.ptx`` buffer purely in memory from *template_bytes*."""
    base_buf, top_chunks, settings_summary = _prepare_validated_base(template_bytes, request)
    dec_base, session = bytes(base_buf.data), base_buf.session
    baseline_warnings = tuple(session.warnings)
    validate_write_request(session, request)

    by_tag = {c.tag: c.raw for c in top_chunks}
    delete_lc = {n.lower() for n in request.delete_track_names}
    deleted_old_indices = [i for i, trk in enumerate(session.tracks) if trk.name.lower() in delete_lc]
    folder_uuid_by_name_lc, preassigned_extra_uuids = _plan_folder_uuids(session, request, delete_lc)

    survival = plan_existing_media_survival(
        by_tag.get(0x1015) or _empty_optional_chunk(0x1015),
        by_tag.get(0x1054) or _empty_optional_chunk(0x1054),
        request.delete_track_names, c2624_raw=by_tag[0x2624], deleted_old_indices=deleted_old_indices,
    )
    base_af_count = _chunk_u32_count(by_tag[0x1004], "0x1004") if 0x1004 in by_tag else 0
    base_clip_count = _chunk_u32_count(by_tag[0x262A], "0x262A") if 0x262A in by_tag else 0
    audio_specs, clips = _prepare_clips_and_specs(
        session, request.tracks, folder_uuid_by_name_lc,
        base_audio_file_index=base_af_count, base_clip_index=base_clip_count,
        base_lane_index=len(survival.lanes), base_playlist_index=len(survival.playlists),
    )
    midi_survival = plan_existing_midi_survival(
        by_tag.get(0x1058) or _empty_optional_chunk(0x1058), by_tag[0x2624], deleted_old_indices,
    )
    extra_specs = _prepare_non_audio_specs(
        session, request.extra_tracks, folder_uuid_by_name_lc, preassigned_extra_uuids,
        base_midi_index=len(midi_survival.entries),
    )
    new_specs = (*audio_specs, *extra_specs)
    layout, old_to_new, folder_insertions, deleted_uuids, deleted_actual = _compute_unified_layout(
        session, new_specs, request.delete_track_names,
    )
    mut_plan = _StructuralMutationPlan(
        old_track_count=len(session.tracks), request=request, new_specs=new_specs, clips=clips,
        layout=layout, old_to_new=old_to_new, folder_insertions=folder_insertions,
        deleted_uuids=deleted_uuids, survival=survival, midi_survival=midi_survival,
    )
    mutated_buf = PtxBuffer(_apply_structural_mutations(dec_base, top_chunks, mut_plan), base_buf.xor_delta, base_buf.k0)
    final_buf, summary = _apply_post_patches_on_buffer(mutated_buf, request, settings_summary)

    created_names = tuple(spec.track_name for spec in new_specs)
    _verify_written_buffer(
        final_buf, len(layout), created_names, base_af_count + len(request.audio_files),
        base_clip_count + len(clips), len(survival.playlists) + len(audio_specs),
        baseline_warnings=baseline_warnings,
    )
    return final_buf.to_encrypted(), OfflinePtxWriteResult(
        ptx_path=request.output_ptx_path, created_tracks=created_names,
        deleted_tracks=deleted_actual, patch_summary=summary,
    )


def write_offline_ptx_session(request: OfflinePtxWriteRequest) -> OfflinePtxWriteResult:
    """Create a complete Pro Tools ``.ptx`` session on disk directly from a ``.ptxt``/``.ptx`` template offline."""
    if request.ptxt_path is None:
        raise PtxValidationError("write_offline_ptx_session requires ptxt_path; use build_offline_ptx_bytes for in-memory bytes")
    if request.output_ptx_path is None:
        raise PtxValidationError("write_offline_ptx_session requires output_ptx_path; use build_offline_ptx_bytes for in-memory bytes")
    final_enc, result = build_offline_ptx_bytes(request.ptxt_path.read_bytes(), request)
    atomic_write_bytes(request.output_ptx_path, final_enc)
    return result
