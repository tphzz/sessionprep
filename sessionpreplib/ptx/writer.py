"""High-level pure offline Pro Tools ``.ptx`` session writer from a ``.ptxt`` template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import secrets
from typing import Any

from .codec import atomic_write_bytes, decrypt_ptx, encrypt_ptx, iter_top_chunks, validate_index_0002
from .models import PtxChunk
from .parser import PtxSession
from .patch_builder import PtxPatchBuilder
from .writer_blueprints import build_volume_table_0f3d
from .writer_chunks import patch_session_audio_format_1028
from .writer_index import apply_rebuilt_index_0002
from .writer_mutations import (
    AudioFileWriteSpec,
    ClipWriteSpec,
    NewAudioTrackSpec,
    TrackSlotOp,
    build_1004_audio_files,
    build_1015_playlists,
    build_1054_lanes,
    build_262a_clips,
    mutate_202b_groups_for_track_map,
    mutate_2107_tracks,
    mutate_210c_folders,
    mutate_2519_tracks,
    mutate_2587_ui_state,
    mutate_2624_mixer,
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


def _validate_template_supported(session: PtxSession, top_chunks: Sequence[PtxChunk]) -> None:
    tags = {c.tag for c in top_chunks}
    missing = [hex(t) for t in (0x1004, 0x1015, 0x1054, 0x202B, 0x2107, 0x210C, 0x2519, 0x2624, 0x262A) if t not in tags]
    if missing:
        raise ValueError(f"Pro Tools template is missing required chunks: {', '.join(missing)}")
    if session.clips or session.audio_files:
        raise ValueError(
            "Offline .ptx writer currently requires a clean session template without pre-existing audio files or clips."
        )


def _validate_write_request(session: PtxSession, request: OfflinePtxWriteRequest) -> None:
    delete_lc = {n.lower() for n in request.delete_track_names}
    folder_uuids_alive: set[bytes] = set()
    for trk in session.tracks:
        if trk.uuid_hex and trk.name.lower() not in delete_lc:
            folder_uuids_alive.add(bytes.fromhex(trk.uuid_hex))
        if trk.parent_folder_name and trk.parent_folder_name.lower() in delete_lc and trk.name.lower() not in delete_lc:
            raise ValueError(
                f"Cannot delete folder {trk.parent_folder_name!r} while surviving child track {trk.name!r} remains"
            )
    for req in request.tracks:
        if req.channels not in (1, 2):
            raise ValueError(f"Unsupported channel count {req.channels!r} for track {req.track_name!r} (expected 1 or 2)")
        if not (0 <= req.audio_file_index < len(request.audio_files)):
            raise ValueError(
                f"Track {req.track_name!r} audio_file_index {req.audio_file_index} out of range (0..{len(request.audio_files) - 1})"
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


def _prepare_clips_and_specs(
    session: PtxSession,
    requests: Sequence[OfflineAudioTrackRequest],
) -> tuple[list[NewAudioTrackSpec], list[ClipWriteSpec]]:
    clips: list[ClipWriteSpec] = []
    specs: list[NewAudioTrackSpec] = []
    next_lane = 0
    for req in requests:
        out_bus: tuple[int, bytes, str] | None = None
        if req.output_bus_name:
            sp = session.signal_path_by_name(req.output_bus_name)
            if sp is not None and sp.uuid_hex:
                out_bus = (sp.index, bytes.fromhex(sp.uuid_hex), sp.name)

        clip_indices: tuple[int, ...]
        lane_indices: tuple[int, ...]
        if req.channels == 2:
            c_idx_l = len(clips)
            clips.append(ClipWriteSpec(
                clip_name=f"{req.clip_base_name}.L", length_samples=req.length_samples,
                audio_file_index=req.audio_file_index, channel_index=0,
                clip_guid16=secrets.token_bytes(16), pt_uuid16=generate_ptx_uuid(),
            ))
            c_idx_r = len(clips)
            clips.append(ClipWriteSpec(
                clip_name=f"{req.clip_base_name}.R", length_samples=req.length_samples,
                audio_file_index=req.audio_file_index, channel_index=1,
                clip_guid16=secrets.token_bytes(16), pt_uuid16=generate_ptx_uuid(),
            ))
            clip_indices = (c_idx_l, c_idx_r)
            lane_indices = (next_lane, next_lane + 1)
            next_lane += 2
        else:
            c_idx = len(clips)
            clips.append(ClipWriteSpec(
                clip_name=req.clip_base_name, length_samples=req.length_samples,
                audio_file_index=req.audio_file_index, channel_index=0,
                clip_guid16=secrets.token_bytes(16), pt_uuid16=generate_ptx_uuid(),
            ))
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
    old_track_count: int,
    request: OfflinePtxWriteRequest,
    new_specs: Sequence[NewAudioTrackSpec],
    clips: Sequence[ClipWriteSpec],
    layout: Sequence[TrackSlotOp],
    old_to_new: Mapping[int, int],
    folder_insertions: Mapping[bytes, Sequence[bytes]],
    deleted_uuids: set[bytes],
) -> bytes:
    top_chunks = iter_top_chunks(dec_base, strict=True)
    template_index = parse_index_0002(dec_base)
    if template_index is None:
        raise RuntimeError("Template .ptx lacks trailing 0x0002 chunk-offset index")

    new_1004, root_path = build_1004_audio_files(
        next(c.raw for c in top_chunks if c.tag == 0x1004),
        request.audio_files,
        session_dir=request.session_dir,
        volume_serial_u32=request.volume_serial_u32,
    )
    replacements: dict[int, bytes] = {
        0x1004: new_1004,
        0x1015: build_1015_playlists(next(c.raw for c in top_chunks if c.tag == 0x1015), new_specs),
        0x1054: build_1054_lanes(next(c.raw for c in top_chunks if c.tag == 0x1054), new_specs),
        0x202B: mutate_202b_groups_for_track_map(
            next(c.raw for c in top_chunks if c.tag == 0x202B), old_to_new, len(layout),
        ),
        0x2107: mutate_2107_tracks(
            next(c.raw for c in top_chunks if c.tag == 0x2107), layout, expected_old_count=old_track_count,
        ),
        0x210C: mutate_210c_folders(
            next(c.raw for c in top_chunks if c.tag == 0x210C), folder_insertions, deleted_uuids,
        ),
        0x2519: mutate_2519_tracks(
            next(c.raw for c in top_chunks if c.tag == 0x2519), layout, expected_old_count=old_track_count,
        ),
        0x2624: mutate_2624_mixer(
            next(c.raw for c in top_chunks if c.tag == 0x2624), layout, expected_old_count=old_track_count,
        ),
        0x262A: build_262a_clips(next(c.raw for c in top_chunks if c.tag == 0x262A), clips),
    }
    if any(c.tag == 0x2587 for c in top_chunks):
        replacements[0x2587] = mutate_2587_ui_state(
            next(c.raw for c in top_chunks if c.tag == 0x2587), layout, expected_old_count=old_track_count,
        )
    if request.audio_files and any(c.tag == 0x0F3D for c in top_chunks):
        replacements[0x0F3D] = build_volume_table_0f3d(
            request.volume_serial_u32, request.volume_label, root_path,
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
) -> None:
    dec, delta = decrypt_ptx(ptx_encrypted)
    if not validate_index_0002(dec):
        raise RuntimeError("Offline .ptx verification failed: trailing 0x0002 index missing or invalid")
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
    if len(session.audio_files) != expected_audio_file_count:
        raise RuntimeError(
            f"Offline .ptx verification failed: expected {expected_audio_file_count} audio files, parsed {len(session.audio_files)}"
        )
    if len(session.clips) != expected_clip_count:
        raise RuntimeError(
            f"Offline .ptx verification failed: expected {expected_clip_count} clips, parsed {len(session.clips)}"
        )
    if len(session.playlists) != len(created_names):
        raise RuntimeError(
            f"Offline .ptx verification failed: expected {len(created_names)} playlists, parsed {len(session.playlists)}"
        )


def write_offline_ptx_session(request: OfflinePtxWriteRequest) -> OfflinePtxWriteResult:
    """Create a complete Pro Tools ``.ptx`` session directly from a ``.ptxt`` template offline."""
    tpl_raw = request.ptxt_path.read_bytes()
    tpl_raw = patch_session_audio_format_1028(
        tpl_raw, sample_rate=request.sample_rate, bit_depth=request.bit_depth, audio_format_code=0x03,
    )
    dec_base, delta = decrypt_ptx(tpl_raw)
    k0 = tpl_raw[0x14] ^ 0x5A
    top_chunks = iter_top_chunks(dec_base, strict=True)
    session = PtxSession(dec_base, xor_delta=delta)
    _validate_template_supported(session, top_chunks)
    _validate_write_request(session, request)

    new_specs, clips = _prepare_clips_and_specs(session, request.tracks)
    layout, old_to_new, folder_insertions, deleted_uuids, deleted_actual = _compute_unified_layout(
        session, new_specs, request.delete_track_names,
    )
    mutated_dec = _apply_structural_mutations(
        dec_base, len(session.tracks), request, new_specs, clips, layout, old_to_new, folder_insertions, deleted_uuids,
    )
    mutated_enc = encrypt_ptx(mutated_dec, delta, k0=k0)
    final_enc, summary = _apply_signal_flow_and_groups(mutated_enc, request)

    created_names = tuple(spec.track_name for spec in new_specs)
    _verify_written_ptx(final_enc, len(layout), created_names, len(request.audio_files), len(clips))
    atomic_write_bytes(request.output_ptx_path, final_enc)
    return OfflinePtxWriteResult(
        ptx_path=request.output_ptx_path,
        created_tracks=created_names,
        deleted_tracks=tuple(deleted_actual),
        patch_summary=summary,
    )
