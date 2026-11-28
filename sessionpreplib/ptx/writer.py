"""High-level pure offline Pro Tools ``.ptx`` session writer from a ``.ptxt`` or ``.ptx`` template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Any
import uuid

from .codec import atomic_write_bytes, decrypt_ptx, encrypt_ptx, iter_top_chunks, validate_index_0002
from .models import PtxChunk
from .parser import PtxSession
from .patch_builder import PtxPatchBuilder
from .writer_chunks import patch_session_audio_format_1028
from .writer_index import apply_rebuilt_index_0002
from .writer_mutations import (
    AudioFileWriteSpec,
    ClipWriteSpec,
    NewAudioTrackSpec,
    TrackSlotOp,
    mutate_0f3d_volumes,
    mutate_1004_audio_files,
    mutate_1015_and_1054_media,
    mutate_202b_groups_for_track_map,
    mutate_2107_tracks,
    mutate_210c_folders,
    mutate_2519_tracks,
    mutate_2587_ui_state,
    mutate_2624_mixer,
    mutate_262a_clips,
    plan_existing_media_survival,
)
from .writer_primitives import generate_ptx_uuid, parse_index_0002

_VALID_GROUP_ACTIONS = frozenset({"create", "add_members", "link_vca"})


@dataclass(frozen=True)
class OfflineAudioTrackRequest:
    """One Audio track to synthesize inside a Pro Tools template."""

    track_name: str
    channels: int
    parent_folder_uuid_hex: str
    output_bus_name: str | None
    audio_file_index: int
    length_samples: int
    clip_base_name: str
    color_index: int = 0xFFFE


@dataclass(frozen=True)
class OfflineGroupRequest:
    """One Pro Tools Mix/Edit group (optionally VCA-linked) to apply after track creation."""

    name: str
    action: str  # "create" | "add_members" | "link_vca"
    member_tracks: tuple[str, ...]
    vca_track: str | None = None
    group_type: str = "Mix/Edit"


@dataclass(frozen=True)
class OfflinePtxWriteRequest:
    """Complete self-contained offline ``.ptx`` generation request (no GUI/processor coupling)."""

    ptxt_path: Path
    output_ptx_path: Path
    session_dir: Path
    sample_rate: int
    bit_depth: int
    audio_files: tuple[AudioFileWriteSpec, ...]
    tracks: tuple[OfflineAudioTrackRequest, ...]
    delete_track_names: tuple[str, ...] = ()
    clip_gains_db: Mapping[str, float] = field(default_factory=dict)
    slot_a_trims_db: Mapping[str, float] = field(default_factory=dict)
    track_volumes_db: Mapping[str, float] = field(default_factory=dict)
    group_requests: tuple[OfflineGroupRequest, ...] = ()
    volume_serial_u32: int = 0x53505250
    volume_label: str = "SessionPrep"


@dataclass(frozen=True)
class OfflinePtxWriteResult:
    """Summary of a completed offline ``.ptx`` write and roundtrip verification."""

    ptx_path: Path
    created_tracks: tuple[str, ...]
    deleted_tracks: tuple[str, ...]
    patch_summary: dict[str, Any]


def _validate_template_supported(top_chunks: Sequence[PtxChunk]) -> None:
    tags = {c.tag for c in top_chunks}
    missing = [hex(t) for t in (0x1004, 0x1015, 0x1054, 0x202B, 0x2107, 0x210C, 0x2519, 0x2624, 0x262A) if t not in tags]
    if missing:
        raise ValueError(f"Pro Tools template is missing required chunks: {', '.join(missing)}")


def _chunk_u32_count(chunk_raw: bytes, label: str) -> int:
    if len(chunk_raw) < 13:
        raise RuntimeError(f"Truncated {label} container chunk")
    return int(struct.unpack_from("<I", chunk_raw, 9)[0])


def _validate_write_request(session: PtxSession, request: OfflinePtxWriteRequest) -> None:
    delete_lc = {n.lower() for n in request.delete_track_names}
    folder_uuids_alive: set[bytes] = set()
    surviving_names_lc: set[str] = set()
    for trk in session.tracks:
        if trk.name.lower() not in delete_lc:
            surviving_names_lc.add(trk.name.lower())
            if trk.uuid_hex:
                folder_uuids_alive.add(bytes.fromhex(trk.uuid_hex))
        if trk.parent_folder_name and trk.parent_folder_name.lower() in delete_lc and trk.name.lower() not in delete_lc:
            raise ValueError(
                f"Cannot delete folder {trk.parent_folder_name!r} while surviving child track {trk.name!r} remains"
            )
    seen_new_lc: set[str] = set()
    for req in request.tracks:
        low_name = req.track_name.lower()
        if low_name in surviving_names_lc or low_name in seen_new_lc:
            raise ValueError(f"Duplicate or colliding track name {req.track_name!r}")
        seen_new_lc.add(low_name)
        if req.channels not in (1, 2):
            raise ValueError(f"Unsupported channel count {req.channels!r} for track {req.track_name!r} (expected 1 or 2)")
        if not (0 <= req.audio_file_index < len(request.audio_files)):
            raise ValueError(
                f"Track {req.track_name!r} audio_file_index {req.audio_file_index} out of range (0..{len(request.audio_files) - 1})"
            )
        if req.channels > request.audio_files[req.audio_file_index].channels:
            raise ValueError(
                f"Track {req.track_name!r} requires {req.channels} channels but audio file {req.audio_file_index} has "
                f"{request.audio_files[req.audio_file_index].channels}"
            )
        if req.length_samples < 0:
            raise ValueError(f"Track {req.track_name!r} length_samples must be non-negative: {req.length_samples}")
        try:
            parent_uid = bytes.fromhex(req.parent_folder_uuid_hex)
        except ValueError as exc:
            raise ValueError(f"Track {req.track_name!r} has invalid parent_folder_uuid_hex") from exc
        if len(parent_uid) != 16 or parent_uid not in folder_uuids_alive:
            raise ValueError(
                f"Track {req.track_name!r} targets missing or deleted parent folder UUID {req.parent_folder_uuid_hex!r}"
            )
    for grp in request.group_requests:
        if grp.action not in _VALID_GROUP_ACTIONS:
            raise ValueError(f"Unsupported offline group action {grp.action!r} for group {grp.name!r}")


def _make_clip_spec(name: str, length_samples: int, audio_file_index: int, channel_index: int) -> ClipWriteSpec:
    return ClipWriteSpec(
        clip_name=name, length_samples=length_samples,
        audio_file_index=audio_file_index, channel_index=channel_index,
        clip_guid16=uuid.uuid4().bytes_le, pt_uuid16=generate_ptx_uuid(),
    )


def _prepare_clips_and_specs(
    session: PtxSession,
    requests: Sequence[OfflineAudioTrackRequest],
    *,
    base_audio_file_index: int = 0,
    base_clip_index: int = 0,
    base_lane_index: int = 0,
    base_playlist_index: int = 0,
) -> tuple[list[NewAudioTrackSpec], list[ClipWriteSpec]]:
    clips: list[ClipWriteSpec] = []
    specs: list[NewAudioTrackSpec] = []
    next_lane = base_lane_index
    for req_offset, req in enumerate(requests):
        out_bus: tuple[int, bytes, str] | None = None
        if req.output_bus_name:
            sp = session.signal_path_by_name(req.output_bus_name)
            if sp is not None and sp.uuid_hex:
                out_bus = (sp.index, bytes.fromhex(sp.uuid_hex), sp.name)

        target_af_idx = base_audio_file_index + req.audio_file_index
        if req.channels == 2:
            c_idx_l = base_clip_index + len(clips)
            clips.append(_make_clip_spec(f"{req.clip_base_name}.L", req.length_samples, target_af_idx, 0))
            c_idx_r = base_clip_index + len(clips)
            clips.append(_make_clip_spec(f"{req.clip_base_name}.R", req.length_samples, target_af_idx, 1))
            clip_indices: tuple[int, ...] = (c_idx_l, c_idx_r)
            lane_indices: tuple[int, ...] = (next_lane, next_lane + 1)
            next_lane += 2
        else:
            c_idx = base_clip_index + len(clips)
            clips.append(_make_clip_spec(req.clip_base_name, req.length_samples, target_af_idx, 0))
            clip_indices = (c_idx,)
            lane_indices = (next_lane,)
            next_lane += 1

        specs.append(NewAudioTrackSpec(
            track_name=req.track_name,
            track_uuid16=generate_ptx_uuid(),
            playlist_uuid16=generate_ptx_uuid(),
            channels=req.channels,
            parent_folder_uuid16=bytes.fromhex(req.parent_folder_uuid_hex),
            output_bus=out_bus,
            color_index=req.color_index,
            clip_indices=clip_indices,
            lane_indices=lane_indices,
            playlist_index=base_playlist_index + req_offset,
        ))
    return specs, clips


def _compute_unified_layout(
    session: PtxSession,
    new_specs: Sequence[NewAudioTrackSpec],
    delete_names: Sequence[str],
) -> tuple[list[TrackSlotOp], dict[int, int], dict[bytes, list[bytes]], set[bytes], list[str]]:
    """Build final ordered track slots, old->new index map, folder insertions, and deleted UUIDs."""
    delete_lc = {n.lower() for n in delete_names}
    by_folder: dict[bytes, list[NewAudioTrackSpec]] = {}
    for spec in new_specs:
        by_folder.setdefault(spec.parent_folder_uuid16, []).append(spec)

    layout: list[TrackSlotOp] = []
    old_to_new: dict[int, int] = {}
    folder_insertions: dict[bytes, list[bytes]] = {}
    deleted_uuids: set[bytes] = set()
    deleted_actual: list[str] = []
    placed_count = 0

    for old_idx, trk in enumerate(session.tracks):
        t_uuid = bytes.fromhex(trk.uuid_hex) if trk.uuid_hex else b""
        if trk.name.lower() in delete_lc:
            if t_uuid:
                deleted_uuids.add(t_uuid)
            deleted_actual.append(trk.name)
            continue
        old_to_new[old_idx] = len(layout)
        layout.append(TrackSlotOp(old_index=old_idx))
        children = by_folder.get(t_uuid)
        if children:
            folder_insertions[t_uuid] = [c.track_uuid16 for c in children]
            for child in children:
                layout.append(TrackSlotOp(new_track=child))
                placed_count += 1
    if placed_count != len(new_specs):
        raise ValueError(f"Placed {placed_count}/{len(new_specs)} requested tracks; check parent folder UUIDs")
    return layout, old_to_new, folder_insertions, deleted_uuids, deleted_actual


def _apply_structural_mutations(
    dec_base: bytes,
    top_chunks: Sequence[PtxChunk],
    old_track_count: int,
    request: OfflinePtxWriteRequest,
    new_specs: Sequence[NewAudioTrackSpec],
    clips: Sequence[ClipWriteSpec],
    layout: Sequence[TrackSlotOp],
    old_to_new: Mapping[int, int],
    folder_insertions: Mapping[bytes, Sequence[bytes]],
    deleted_uuids: set[bytes],
    surviving_playlists: Sequence[bytes],
    surviving_lanes: Sequence[bytes],
    old_playlist_to_new: Mapping[int, int],
    c4829_trailer: bytes,
    c4828_trailer: bytes,
) -> bytes:
    template_index = parse_index_0002(dec_base)
    if template_index is None:
        raise RuntimeError("Template .ptx lacks trailing 0x0002 chunk-offset index")

    by_tag = {c.tag: c.raw for c in reversed(top_chunks)}
    new_1004, root_path = mutate_1004_audio_files(
        by_tag[0x1004], request.audio_files, session_dir=request.session_dir, volume_serial_u32=request.volume_serial_u32,
    )
    new_1015, new_1054 = mutate_1015_and_1054_media(
        by_tag[0x1015], by_tag[0x1054], new_specs,
        surviving_playlists=surviving_playlists, surviving_lanes=surviving_lanes,
        c4829_trailer=c4829_trailer, c4828_trailer=c4828_trailer,
    )
    replacements: dict[int, bytes] = {
        0x1004: new_1004,
        0x1015: new_1015,
        0x1054: new_1054,
        0x202B: mutate_202b_groups_for_track_map(by_tag[0x202B], old_to_new, len(layout)),
        0x2107: mutate_2107_tracks(by_tag[0x2107], layout, expected_old_count=old_track_count),
        0x210C: mutate_210c_folders(by_tag[0x210C], folder_insertions, deleted_uuids),
        0x2519: mutate_2519_tracks(by_tag[0x2519], layout, expected_old_count=old_track_count),
        0x2624: mutate_2624_mixer(
            by_tag[0x2624], layout, expected_old_count=old_track_count, old_playlist_to_new=old_playlist_to_new,
        ),
        0x262A: mutate_262a_clips(by_tag[0x262A], clips),
    }
    if 0x2587 in by_tag:
        replacements[0x2587] = mutate_2587_ui_state(by_tag[0x2587], layout, expected_old_count=old_track_count)
    if request.audio_files and 0x0F3D in by_tag:
        replacements[0x0F3D] = mutate_0f3d_volumes(
            by_tag[0x0F3D], request.volume_serial_u32, request.volume_label, root_path,
        )

    out = bytearray(dec_base[:0x1F])
    replaced_tags: set[int] = set()
    for chunk in top_chunks:
        if chunk.tag in replacements and chunk.tag not in replaced_tags:
            out.extend(replacements[chunk.tag])
            replaced_tags.add(chunk.tag)
        else:
            out.extend(chunk.raw)
    apply_rebuilt_index_0002(out, template_index, dec_base)
    return bytes(out)


def _apply_signal_flow_and_groups(
    ptx_encrypted: bytes,
    request: OfflinePtxWriteRequest,
) -> tuple[bytes, dict[str, Any]]:
    builder = PtxPatchBuilder.from_bytes(ptx_encrypted)
    if request.clip_gains_db:
        builder.set_clip_gains(request.clip_gains_db)
    if request.slot_a_trims_db:
        builder.set_slot_a_trim_gains(request.slot_a_trims_db, activate_if_inactive=True)
    if request.track_volumes_db:
        builder.set_track_volumes(request.track_volumes_db)
    for grp in request.group_requests:
        if grp.action == "create":
            builder.add_group(
                grp.name, grp.member_tracks, vca_track=grp.vca_track, group_type=grp.group_type,
            )
        elif grp.action == "add_members":
            kwargs: dict[str, Any] = {"add_tracks": grp.member_tracks}
            if grp.vca_track:
                kwargs["vca_track"] = grp.vca_track
                kwargs["group_type"] = grp.group_type
            builder.modify_group(grp.name, **kwargs)
        elif grp.action == "link_vca" and grp.vca_track:
            builder.modify_group(grp.name, vca_track=grp.vca_track, group_type=grp.group_type)
    return builder.build()


def _verify_written_ptx(
    ptx_encrypted: bytes,
    expected_track_count: int,
    created_names: Sequence[str],
    expected_audio_file_count: int,
    expected_clip_count: int,
    expected_playlist_count: int,
) -> None:
    dec, delta = decrypt_ptx(ptx_encrypted)
    if not validate_index_0002(dec):
        raise RuntimeError("Offline .ptx verification failed: trailing 0x0002 index missing or invalid")
    by_tag = {c.tag: c.raw for c in reversed(iter_top_chunks(dec, strict=True))}
    session = PtxSession(dec, xor_delta=delta)
    if session.warnings:
        raise RuntimeError(f"Offline .ptx verification produced parser warnings: {session.warnings}")
    if len(session.tracks) != expected_track_count:
        raise RuntimeError(
            f"Offline .ptx verification failed: expected {expected_track_count} tracks, parsed {len(session.tracks)}"
        )
    parsed_names = {t.name for t in session.tracks}
    missing = [n for n in created_names if n not in parsed_names]
    if missing:
        raise RuntimeError(f"Offline .ptx verification failed: missing created tracks {missing}")
    af_hdr = _chunk_u32_count(by_tag[0x1004], "0x1004")
    if af_hdr != expected_audio_file_count:
        raise RuntimeError(f"Offline .ptx verification failed: expected {expected_audio_file_count} 0x1004 files, got {af_hdr}")
    clip_hdr = _chunk_u32_count(by_tag[0x262A], "0x262A")
    if clip_hdr != expected_clip_count:
        raise RuntimeError(f"Offline .ptx verification failed: expected {expected_clip_count} 0x262A clips, got {clip_hdr}")
    pl_hdr = _chunk_u32_count(by_tag[0x1015], "0x1015")
    if pl_hdr != expected_playlist_count or len(session.playlists) != expected_playlist_count:
        raise RuntimeError(
            f"Offline .ptx verification failed: expected {expected_playlist_count} playlists, got header={pl_hdr} parsed={len(session.playlists)}"
        )


def write_offline_ptx_session(request: OfflinePtxWriteRequest) -> OfflinePtxWriteResult:
    """Create a complete Pro Tools ``.ptx`` session directly from a ``.ptxt``/``.ptx`` template offline."""
    tpl_raw = request.ptxt_path.read_bytes()
    tpl_raw = patch_session_audio_format_1028(
        tpl_raw, sample_rate=request.sample_rate, bit_depth=request.bit_depth, audio_format_code=0x03,
    )
    dec_base, delta = decrypt_ptx(tpl_raw)
    k0 = tpl_raw[0x14] ^ 0x5A
    top_chunks = iter_top_chunks(dec_base, strict=True)
    _validate_template_supported(top_chunks)
    session = PtxSession(dec_base, xor_delta=delta)
    _validate_write_request(session, request)

    by_tag = {c.tag: c.raw for c in reversed(top_chunks)}
    delete_lc = {n.lower() for n in request.delete_track_names}
    deleted_old_indices = [i for i, trk in enumerate(session.tracks) if trk.name.lower() in delete_lc]
    surviving_pl, surviving_lanes, old_pl_to_new, c4829, c4828 = plan_existing_media_survival(
        by_tag[0x1015], by_tag[0x1054], request.delete_track_names,
        c2624_raw=by_tag[0x2624], deleted_old_indices=deleted_old_indices,
    )
    base_af_count = _chunk_u32_count(by_tag[0x1004], "0x1004")
    base_clip_count = _chunk_u32_count(by_tag[0x262A], "0x262A")
    new_specs, clips = _prepare_clips_and_specs(
        session, request.tracks,
        base_audio_file_index=base_af_count, base_clip_index=base_clip_count,
        base_lane_index=len(surviving_lanes), base_playlist_index=len(surviving_pl),
    )
    layout, old_to_new, folder_insertions, deleted_uuids, deleted_actual = _compute_unified_layout(
        session, new_specs, request.delete_track_names,
    )
    mutated_dec = _apply_structural_mutations(
        dec_base, top_chunks, len(session.tracks), request, new_specs, clips,
        layout, old_to_new, folder_insertions, deleted_uuids,
        surviving_pl, surviving_lanes, old_pl_to_new, c4829, c4828,
    )
    mutated_enc = encrypt_ptx(mutated_dec, delta, k0=k0)
    final_enc, summary = _apply_signal_flow_and_groups(mutated_enc, request)

    created_names = tuple(spec.track_name for spec in new_specs)
    _verify_written_ptx(
        final_enc, len(layout), created_names,
        base_af_count + len(request.audio_files),
        base_clip_count + len(clips),
        len(surviving_pl) + len(created_names),
    )
    atomic_write_bytes(request.output_ptx_path, final_enc)
    return OfflinePtxWriteResult(
        ptx_path=request.output_ptx_path,
        created_tracks=created_names,
        deleted_tracks=tuple(deleted_actual),
        patch_summary=summary,
    )
