"""High-level pure offline Pro Tools ``.ptx`` session writer from a ``.ptxt`` or ``.ptx`` template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any
import uuid

from .codec import (
    atomic_write_bytes,
    decrypt_ptx,
    encrypt_ptx,
    iter_top_chunks,
    normalize_ptx_uuid16,
    validate_index_0002,
    walk_sequential_zchunks,
)
from .models import (
    VALID_PROTOOLS_BIT_DEPTHS,
    VALID_PROTOOLS_SAMPLE_RATES,
    PtxChunk,
    PtxCorruptFileError,
    PtxUnsupportedTemplateError,
    PtxValidationError,
)
from .parser import PtxSession
from .patch_builder import PtxPatchBuilder
from .writer_blueprints import TRACK_TYPE_TO_251A, OutputBusSpec, resolve_track_format_codes
from .writer_chunks import (
    patch_session_audio_format_1028_decrypted,
    patch_session_timecode_204d_decrypted,
    patch_session_ui_state_decrypted,
    serialize_directory_tree_103a,
)
from .writer_index import apply_rebuilt_index_0002
from .writer_mutations import (
    AudioFileWriteSpec,
    ClipWriteSpec,
    MediaSurvivalPlan,
    MidiSurvivalPlan,
    NewAudioTrackSpec,
    TrackSlotOp,
    mutate_0f3d_volumes,
    mutate_1004_audio_files,
    mutate_1015_and_1054_media,
    mutate_1058_midi_tracks,
    mutate_202b_groups_for_track_map,
    mutate_2030_markers_for_track_map,
    mutate_2107_tracks,
    mutate_210c_folders,
    mutate_2519_tracks,
    mutate_2587_ui_state,
    mutate_2624_mixer,
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

_VALID_GROUP_ACTIONS = frozenset({"create", "add_members", "link_vca"})
_EMPTY_MAP: Mapping[str, float] = MappingProxyType({})
_EMPTY_STATE_MAP: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
_FOLDER_TRACK_TYPES = frozenset({"RoutingFolder", "BasicFolder"})
_NON_AUDIO_TRACK_TYPES = frozenset(t for t in TRACK_TYPE_TO_251A if t != "Audio")


@dataclass(frozen=True)
class OfflineAudioTrackRequest:
    """One Audio track to synthesize inside a Pro Tools template (Mono, Stereo, or Surround)."""

    track_name: str
    channels: int = 2
    parent_folder_uuid_hex: str | None = None
    output_bus_name: str | None = None
    audio_file_index: int | None = None
    length_samples: int = 0
    clip_base_name: str = ""
    color_index: int = 0xFFFE
    start_sample: int = 0
    track_format: str | None = None
    clip_start_sample: int = 0
    clip_sync_point_samples: int = 0
    clip_orig_timestamp: int = 0
    clip_user_timestamp: int = 0
    clip_muted: bool = False
    clip_time_locked: bool = False
    clip_edit_locked: bool = False
    clip_color_index: int = 0xFFFE


@dataclass(frozen=True)
class OfflineNonAudioTrackRequest:
    """One non-Audio track (Aux, RoutingFolder, Master, Vca, BasicFolder, Midi, Instrument) to synthesize."""

    track_name: str
    track_type: str  # "Aux" | "RoutingFolder" | "Master" | "Vca" | "BasicFolder" | "Midi" | "Instrument"
    channels: int = 2
    parent_folder_uuid_hex: str | None = None
    output_bus_name: str | None = None
    color_index: int = 0xFFFE
    is_folder_open: bool = True
    track_format: str | None = None


@dataclass(frozen=True)
class OfflineMarkerRequest:
    """One Pro Tools Memory Location / Marker to add after track creation."""

    name: str
    start_pos: int
    number: int | None = None
    end_pos: int | None = None
    reference: str = "Absolute"
    time_properties: str = "Marker"
    comments: str = ""
    color_index: int = 0
    color_rgb: tuple[int, int, int] | None = None
    track_name: str | None = None
    zoom_settings: bool = False
    pre_post_roll_times: bool = False
    track_visibility: bool = False
    track_heights: bool = False
    group_enables: bool = False
    window_configuration: bool = False


@dataclass(frozen=True)
class OfflineFadeRequest:
    """One fade-in, fade-out, or crossfade to place on an Audio track after track creation."""

    track_name: str
    fade_type: str  # "in" | "out" | "crossfade"
    start_sample: int
    total_samples: int
    pre_samples: int | None = None
    shape: str = "standard"
    crossfade_link: str = "equal_power"


@dataclass(frozen=True)
class OfflineGroupRequest:
    """One Pro Tools Mix/Edit group (optionally VCA-linked) to apply after track creation."""

    name: str
    action: str  # "create" | "add_members" | "link_vca"
    member_tracks: tuple[str, ...]
    vca_track: str | None = None
    group_type: str = "Mix/Edit"
    is_edit_active: bool = False
    is_mix_active: bool = False


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
    extra_tracks: tuple[OfflineNonAudioTrackRequest, ...] = ()
    delete_track_names: tuple[str, ...] = ()
    clip_gains_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    slot_a_trims_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    track_volumes_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    track_states: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: _EMPTY_STATE_MAP)
    markers: tuple[OfflineMarkerRequest, ...] = ()
    fades: tuple[OfflineFadeRequest, ...] = ()
    group_requests: tuple[OfflineGroupRequest, ...] = ()
    volume_serial_u32: int = 0x53505250
    volume_label: str = "SessionPrep"
    audio_format: str | None = None
    is_interleaved: bool | None = None
    audio_pull_rate: str | None = None
    video_pull_rate: str | None = None
    fps_code: int | None = None
    start_frames: int | None = None
    feet_frames_rate: str | None = None
    session_length: str | None = None
    main_counter_format: str | None = None
    sub_counter_format: str | None = None
    edit_mode: str | None = None
    grid_relative: bool | None = None
    edit_tool: str | None = None
    smart_tool_enabled: bool | None = None
    pre_roll_enabled: bool | None = None
    pre_roll_samples: int | None = None
    post_roll_enabled: bool | None = None
    post_roll_samples: int | None = None
    timeline_in_sample: int | None = None
    timeline_out_sample: int | None = None

    @property
    def template_path(self) -> Path:
        return self.ptxt_path


@dataclass(frozen=True)
class OfflinePtxWriteResult:
    """Summary of a completed offline ``.ptx`` write and roundtrip verification."""

    ptx_path: Path
    created_tracks: tuple[str, ...]
    deleted_tracks: tuple[str, ...]
    patch_summary: dict[str, Any]


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
    if tag == 0x210C:
        return serialize_folder_tree_210c([], version=1)
    raise ValueError(f"Unsupported synthesized chunk tag 0x{tag:04x}")


def _validate_template_supported(top_chunks: Sequence[PtxChunk]) -> None:
    by_tag = {c.tag: c for c in top_chunks}
    missing = [hex(t) for t in (0x1028, 0x2519, 0x2107, 0x2624, 0x202B) if t not in by_tag]
    if missing:
        raise PtxUnsupportedTemplateError(f"Pro Tools template is missing required chunks: {', '.join(missing)}")
    if by_tag[0x2519].version < 8:
        raise PtxUnsupportedTemplateError(
            f"Unsupported legacy Pro Tools template 0x2519 version {by_tag[0x2519].version} (expected >= 8)"
        )
    c2624_raw = by_tag[0x2624].raw
    if len(c2624_raw) >= 13:
        strips, _ = walk_sequential_zchunks(c2624_raw, 13, struct.unpack_from("<I", c2624_raw, 9)[0])
        for st in strips:
            if struct.unpack_from("<H", st, 7)[0] == 0x261C and struct.unpack_from("<H", st, 1)[0] < 6:
                ver = struct.unpack_from("<H", st, 1)[0]
                raise PtxUnsupportedTemplateError(f"Unsupported legacy Pro Tools template 0x261C version {ver} (expected >= 6)")


def _resolve_output_bus(session: PtxSession, bus_name: str) -> OutputBusSpec:
    sp = session.signal_path_by_name(bus_name)
    if sp is None:
        low = bus_name.lower()
        sp = next((p for p in session.signal_paths if p.name.lower() == low and p.uuid_hex), None)
    if sp is None or not sp.uuid_hex:
        raise PtxValidationError(f"Output bus {bus_name!r} not found or lacks a valid UUID in template")
    return OutputBusSpec(
        ptsl_index=sp.index,
        uuid16=bytes.fromhex(sp.uuid_hex),
        name=sp.name,
        channel_count=max(1, int(sp.channel_count or 2)),
        parent_uuid16=(bytes.fromhex(sp.parent_uuid_hex) if sp.is_subpath and sp.parent_uuid_hex else None),
    )


def _validate_track_request(
    session: PtxSession,
    request: OfflinePtxWriteRequest,
    req: OfflineAudioTrackRequest,
    folder_uuids_alive: set[bytes],
    used_names_lc: set[str],
) -> None:
    if not req.track_name or not req.track_name.strip():
        raise PtxValidationError("Track name must not be empty")
    low_name = req.track_name.lower()
    if low_name in used_names_lc:
        raise PtxValidationError(f"Duplicate or colliding track name {req.track_name!r}")
    used_names_lc.add(low_name)
    try:
        _lw, _mw, _ew, resolved_ch = resolve_track_format_codes(req.channels, req.track_format)
    except ValueError as exc:
        raise PtxValidationError(f"Track {req.track_name!r}: {exc}") from exc
    if isinstance(req.start_sample, bool) or not (0 <= req.start_sample <= 0x00FF_FFFF_FFFF_FFFF):
        raise PtxValidationError(f"Track {req.track_name!r} start_sample out of range: {req.start_sample!r}")
    if isinstance(req.color_index, bool) or not (0 <= req.color_index <= 0xFFFF):
        raise PtxValidationError(f"Track {req.track_name!r} color_index out of range: {req.color_index!r}")
    if isinstance(req.clip_color_index, bool) or not (0 <= req.clip_color_index <= 0xFFFF):
        raise PtxValidationError(f"Track {req.track_name!r} clip_color_index out of range: {req.clip_color_index!r}")
    for lbl, val in (
        ("clip_start_sample", req.clip_start_sample),
        ("length_samples", req.length_samples),
        ("clip_sync_point_samples", req.clip_sync_point_samples),
        ("clip_orig_timestamp", req.clip_orig_timestamp),
        ("clip_user_timestamp", req.clip_user_timestamp),
    ):
        if isinstance(val, bool) or not (0 <= val <= 0xFFFF_FFFF_FFFF_FFFF):
            raise PtxValidationError(f"Track {req.track_name!r} {lbl} out of u64 range: {val!r}")
    if req.audio_file_index is not None:
        if not (0 <= req.audio_file_index < len(request.audio_files)):
            raise PtxValidationError(
                f"Track {req.track_name!r} audio_file_index {req.audio_file_index} out of range (0..{len(request.audio_files) - 1})"
            )
        af = request.audio_files[req.audio_file_index]
        if resolved_ch > af.channels:
            raise PtxValidationError(
                f"Track {req.track_name!r} requires {resolved_ch} channels but audio file {req.audio_file_index} has {af.channels}"
            )
        if req.length_samples <= 0 or req.clip_start_sample + req.length_samples > af.length_samples:
            raise PtxValidationError(
                f"Track {req.track_name!r} clip range [{req.clip_start_sample}..{req.clip_start_sample + req.length_samples}] invalid for audio file length {af.length_samples}"
            )
        if not req.clip_base_name or not req.clip_base_name.strip():
            raise PtxValidationError(f"Track {req.track_name!r} clip_base_name must not be empty")
    if req.parent_folder_uuid_hex is not None:
        try:
            parent_uid = normalize_ptx_uuid16(req.parent_folder_uuid_hex)
        except ValueError as exc:
            raise PtxValidationError(f"Track {req.track_name!r} has invalid parent_folder_uuid_hex") from exc
        if parent_uid not in folder_uuids_alive:
            raise PtxValidationError(
                f"Track {req.track_name!r} targets missing, non-folder, or deleted parent folder UUID {req.parent_folder_uuid_hex!r}"
            )
    if req.output_bus_name:
        _resolve_output_bus(session, req.output_bus_name)


def _validate_non_audio_track_request(
    session: PtxSession,
    req: OfflineNonAudioTrackRequest,
    folder_uuids_alive: set[bytes],
    used_names_lc: set[str],
) -> None:
    if not req.track_name or not req.track_name.strip():
        raise PtxValidationError("Non-Audio track name must not be empty")
    low_name = req.track_name.lower()
    if low_name in used_names_lc:
        raise PtxValidationError(f"Duplicate or colliding track name {req.track_name!r}")
    used_names_lc.add(low_name)
    if req.track_type not in _NON_AUDIO_TRACK_TYPES:
        raise PtxValidationError(f"Unsupported non-Audio track_type {req.track_type!r} for track {req.track_name!r}")
    if req.track_type == "Instrument":
        try:
            _lw, _mw, _ew, resolved_ch = resolve_track_format_codes(req.channels, req.track_format)
        except ValueError as exc:
            raise PtxValidationError(f"Track {req.track_name!r}: {exc}") from exc
        if resolved_ch not in (1, 2):
            raise PtxValidationError(f"Instrument track {req.track_name!r} only supports Mono or Stereo, got {req.track_format or req.channels!r}")
    elif req.track_type not in ("Vca", "BasicFolder", "Midi"):
        try:
            resolve_track_format_codes(req.channels, req.track_format)
        except ValueError as exc:
            raise PtxValidationError(f"Track {req.track_name!r}: {exc}") from exc
    if isinstance(req.color_index, bool) or not (0 <= req.color_index <= 0xFFFF):
        raise PtxValidationError(f"Track {req.track_name!r} color_index out of range: {req.color_index!r}")
    if req.parent_folder_uuid_hex is not None:
        try:
            parent_uid = normalize_ptx_uuid16(req.parent_folder_uuid_hex)
        except ValueError as exc:
            raise PtxValidationError(f"Track {req.track_name!r} has invalid parent_folder_uuid_hex") from exc
        if parent_uid not in folder_uuids_alive:
            raise PtxValidationError(
                f"Track {req.track_name!r} targets missing, non-folder, or deleted parent folder UUID {req.parent_folder_uuid_hex!r}"
            )
    if req.output_bus_name:
        _resolve_output_bus(session, req.output_bus_name)


def _validate_write_request(session: PtxSession, request: OfflinePtxWriteRequest) -> None:
    delete_lc = {n.lower() for n in request.delete_track_names}
    deleted_uuids = {bytes.fromhex(t.uuid_hex) for t in session.tracks if t.uuid_hex and t.name.lower() in delete_lc}
    folder_uuids_alive: set[bytes] = set()
    used_names_lc: set[str] = set()
    for trk in session.tracks:
        if trk.name.lower() not in delete_lc:
            used_names_lc.add(trk.name.lower())
            if trk.uuid_hex and trk.track_type in _FOLDER_TRACK_TYPES:
                folder_uuids_alive.add(bytes.fromhex(trk.uuid_hex))
            parent_deleted = bool(trk.parent_folder_name and trk.parent_folder_name.lower() in delete_lc)
            if not parent_deleted and trk.parent_folder_uuid:
                try:
                    parent_deleted = normalize_ptx_uuid16(trk.parent_folder_uuid) in deleted_uuids
                except ValueError:
                    parent_deleted = False
            if parent_deleted:
                raise PtxValidationError(
                    f"Cannot delete folder {trk.parent_folder_name!r} while surviving child track {trk.name!r} remains"
                )
    for req in request.tracks:
        _validate_track_request(session, request, req, folder_uuids_alive, used_names_lc)
    for extra in request.extra_tracks:
        _validate_non_audio_track_request(session, extra, folder_uuids_alive, used_names_lc)
    for grp in request.group_requests:
        if grp.action not in _VALID_GROUP_ACTIONS:
            raise PtxValidationError(f"Unsupported offline group action {grp.action!r} for group {grp.name!r}")
        if grp.action == "link_vca" and (not grp.vca_track or not grp.vca_track.strip()):
            raise PtxValidationError(f"Group {grp.name!r} with action='link_vca' requires a non-empty vca_track")


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


def _prepare_clips_and_specs(
    session: PtxSession,
    requests: Sequence[OfflineAudioTrackRequest],
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
        out_bus = _resolve_output_bus(session, req.output_bus_name) if req.output_bus_name else None
        lane_indices = tuple(range(next_lane, next_lane + resolved_ch))
        next_lane += resolved_ch

        if req.audio_file_index is None:
            clip_indices: tuple[int, ...] = ()
        elif resolved_ch == 1:
            target_af_idx = base_audio_file_index + req.audio_file_index
            c_idx = base_clip_index + len(clips)
            clips.append(_make_clip_spec(
                req.clip_base_name, req.length_samples, target_af_idx, 0,
                start_sample=req.clip_start_sample, sync_point_samples=req.clip_sync_point_samples,
                orig_timestamp=req.clip_orig_timestamp, user_timestamp=req.clip_user_timestamp,
            ))
            clip_indices = (c_idx,)
        else:
            target_af_idx = base_audio_file_index + req.audio_file_index
            suffixes = (".L", ".R") if resolved_ch == 2 else tuple(f".Ch{k + 1}" for k in range(resolved_ch))
            c_list: list[int] = []
            for ch_i, suf in enumerate(suffixes):
                c_list.append(base_clip_index + len(clips))
                clips.append(_make_clip_spec(
                    f"{req.clip_base_name}{suf}", req.length_samples, target_af_idx, ch_i,
                    start_sample=req.clip_start_sample, sync_point_samples=req.clip_sync_point_samples,
                    orig_timestamp=req.clip_orig_timestamp, user_timestamp=req.clip_user_timestamp,
                ))
            clip_indices = tuple(c_list)

        parent_uid = normalize_ptx_uuid16(req.parent_folder_uuid_hex) if req.parent_folder_uuid_hex is not None else None
        specs.append(NewAudioTrackSpec(
            track_name=req.track_name, track_uuid16=generate_ptx_uuid(),
            playlist_uuid16=generate_ptx_uuid(), channels=resolved_ch,
            parent_folder_uuid16=parent_uid, output_bus=out_bus,
            color_index=req.color_index, clip_indices=clip_indices,
            lane_indices=lane_indices, playlist_index=base_playlist_index + req_offset,
            start_sample=req.start_sample, track_format=req.track_format,
            clip_muted=req.clip_muted, clip_time_locked=req.clip_time_locked,
            clip_edit_locked=req.clip_edit_locked, clip_color_index=req.clip_color_index,
        ))
    return tuple(specs), tuple(clips)


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
    for spec in top_level_new:
        layout.append(TrackSlotOp(new_track=spec))
        placed_count += 1
    if placed_count != len(new_specs):
        raise PtxValidationError(f"Placed {placed_count}/{len(new_specs)} requested tracks; check parent folder UUIDs")
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
        idx_0002 = last_idx_by_tag.get(0x0002, len(top_chunks))
        insertions_before.setdefault(idx_0002, []).append(replacements[tag])

    _schedule_missing(0x0F3D, after=(0x2032, 0x1028), before=(0x1004, 0x2027))
    _schedule_missing(0x1004, after=(0x0F3D, 0x2032, 0x1028), before=(0x2027, 0x2096))
    _schedule_missing(0x2637, after=(0x204D,), before=(0x262C, 0x262A, 0x2721, 0x1054, 0x1015, 0x1055, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x262A, after=(0x2637, 0x2632, 0x204D), before=(0x2721, 0x1054, 0x1015, 0x1055, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x1054, after=(0x2437, 0x2029), before=(0x1015, 0x1055, 0x210C, 0x2107))
    _schedule_missing(0x1015, after=(0x1054, 0x2437), before=(0x1055, 0x1058, 0x210C, 0x2107))
    _schedule_missing(0x1058, after=(0x1055, 0x1015, 0x1054), before=(0x2604, 0x210C, 0x2107, 0x2624))
    _schedule_missing(0x210C, before=(0x2107, 0x2624))

    out = bytearray(dec_header)
    for idx, chunk in enumerate(top_chunks):
        for raw_ins in insertions_before.get(idx, ()):
            out.extend(raw_ins)
        if chunk.tag in replacements and last_idx_by_tag[chunk.tag] == idx:
            out.extend(replacements[chunk.tag])
        else:
            out.extend(chunk.raw)
        for raw_ins in insertions_after.get(idx, ()):
            out.extend(raw_ins)
    for raw_ins in insertions_before.get(len(top_chunks), ()):
        out.extend(raw_ins)
    return out


def _apply_structural_mutations(
    dec_base: bytes,
    top_chunks: Sequence[PtxChunk],
    plan: _StructuralMutationPlan,
) -> bytes:
    template_index = parse_index_0002(dec_base)
    if template_index is None:
        raise PtxCorruptFileError("Template .ptx lacks trailing 0x0002 chunk-offset index")

    by_tag = {c.tag: c.raw for c in top_chunks}  # last-wins, matching parser.py & writer_index.py
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
    return bytes(out)


def _prepare_non_audio_specs(
    session: PtxSession,
    extra_tracks: Sequence[OfflineNonAudioTrackRequest],
    *,
    base_midi_index: int = 0,
) -> tuple[NewAudioTrackSpec, ...]:
    specs: list[NewAudioTrackSpec] = []
    next_midi = base_midi_index
    for extra in extra_tracks:
        out_bus = _resolve_output_bus(session, extra.output_bus_name) if extra.output_bus_name else None
        parent_uid = normalize_ptx_uuid16(extra.parent_folder_uuid_hex) if extra.parent_folder_uuid_hex is not None else None
        if extra.track_type in ("Vca", "BasicFolder", "Midi"):
            channels = 1
        else:
            _lw, _mw, _ew, channels = resolve_track_format_codes(extra.channels, extra.track_format)
        midi_idx: int | None = None
        midi_uid = b"\x00" * 16
        if extra.track_type in ("Midi", "Instrument"):
            midi_idx = next_midi
            next_midi += 1
            midi_uid = generate_ptx_uuid()
        specs.append(NewAudioTrackSpec(
            track_name=extra.track_name, track_uuid16=generate_ptx_uuid(),
            playlist_uuid16=b"\x00" * 16, channels=channels,
            parent_folder_uuid16=parent_uid, output_bus=out_bus,
            color_index=extra.color_index, clip_indices=(), lane_indices=(),
            playlist_index=0, start_sample=0, track_type=extra.track_type,
            is_folder_open=extra.is_folder_open, track_format=extra.track_format,
            midi_index=midi_idx, midi_uuid16=midi_uid,
        ))
    return tuple(specs)


def _apply_signal_flow_and_groups(
    ptx_encrypted: bytes,
    request: OfflinePtxWriteRequest,
) -> tuple[bytes, dict[str, Any]]:
    has_work = bool(
        request.clip_gains_db or request.slot_a_trims_db or request.track_volumes_db
        or request.track_states or request.markers or request.fades or request.group_requests
    )
    if not has_work:
        return ptx_encrypted, {
            "cleared_inputs": {}, "clip_gains": {}, "slot_a_trims": {}, "volumes": {},
            "groups": [], "session_settings": {}, "track_states": {}, "renamed_tracks": {},
            "renamed_clips": {}, "clip_attributes": [], "fades": [], "markers": [],
        }
    builder = PtxPatchBuilder.from_bytes(ptx_encrypted)
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
    for grp in request.group_requests:
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
    return builder.build()


def _verify_written_ptx(
    ptx_encrypted: bytes,
    expected_track_count: int,
    created_names: Sequence[str],
    expected_audio_file_count: int,
    expected_clip_count: int,
    expected_playlist_count: int,
    *,
    baseline_warnings: Sequence[str] = (),
) -> None:
    dec, delta = decrypt_ptx(ptx_encrypted)
    try:
        idx_ok = validate_index_0002(dec)
    except RuntimeError as exc:
        raise PtxCorruptFileError(f"Offline .ptx verification failed: {exc}") from exc
    if not idx_ok:
        raise PtxCorruptFileError("Offline .ptx verification failed: trailing 0x0002 index missing or invalid")
    by_tag = {c.tag: c.raw for c in iter_top_chunks(dec, strict=True)}
    session = PtxSession(dec, xor_delta=delta)
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


def _prepare_validated_base(request: OfflinePtxWriteRequest) -> tuple[bytes, int, int, list[PtxChunk]]:
    """Decrypt, validate template schema and format parameters, and patch ``0x1028`` / ``0x204D`` in place."""
    if isinstance(request.sample_rate, bool) or request.sample_rate not in VALID_PROTOOLS_SAMPLE_RATES:
        raise PtxValidationError(f"Unsupported Pro Tools sample rate: {request.sample_rate!r}")
    if isinstance(request.bit_depth, bool) or request.bit_depth not in VALID_PROTOOLS_BIT_DEPTHS:
        raise PtxValidationError(f"Unsupported Pro Tools bit depth: {request.bit_depth!r}")
    tpl_raw = request.ptxt_path.read_bytes()
    dec_raw, delta = decrypt_ptx(tpl_raw)
    k0 = tpl_raw[0x14] ^ 0x5A
    _validate_template_supported(iter_top_chunks(dec_raw, strict=True))
    dec_buf = bytearray(dec_raw)
    try:
        patch_session_audio_format_1028_decrypted(
            dec_buf,
            sample_rate=request.sample_rate,
            bit_depth=request.bit_depth,
            audio_format=request.audio_format,
            is_interleaved=request.is_interleaved,
            audio_pull_rate=request.audio_pull_rate,
            video_pull_rate=request.video_pull_rate,
        )
        if any(v is not None for v in (request.fps_code, request.start_frames, request.feet_frames_rate, request.session_length)):
            patch_session_timecode_204d_decrypted(
                dec_buf,
                fps_code=request.fps_code,
                start_frames=request.start_frames,
                feet_frames_rate=request.feet_frames_rate,
                session_length=request.session_length,
            )
        if any(v is not None for v in (
            request.main_counter_format, request.sub_counter_format, request.edit_mode,
            request.grid_relative, request.edit_tool, request.smart_tool_enabled,
            request.pre_roll_enabled, request.pre_roll_samples, request.post_roll_enabled,
            request.post_roll_samples, request.timeline_in_sample, request.timeline_out_sample,
        )):
            patch_session_ui_state_decrypted(
                dec_buf,
                main_counter_format=request.main_counter_format,
                sub_counter_format=request.sub_counter_format,
                edit_mode=request.edit_mode,
                grid_relative=request.grid_relative,
                edit_tool=request.edit_tool,
                smart_tool_enabled=request.smart_tool_enabled,
                pre_roll_enabled=request.pre_roll_enabled,
                pre_roll_samples=request.pre_roll_samples,
                post_roll_enabled=request.post_roll_enabled,
                post_roll_samples=request.post_roll_samples,
                timeline_in_sample=request.timeline_in_sample,
                timeline_out_sample=request.timeline_out_sample,
            )
    except ValueError as exc:
        raise PtxValidationError(str(exc)) from exc
    dec_base = bytes(dec_buf)
    return dec_base, delta, k0, iter_top_chunks(dec_base, strict=True)


def write_offline_ptx_session(request: OfflinePtxWriteRequest) -> OfflinePtxWriteResult:
    """Create a complete Pro Tools ``.ptx`` session directly from a ``.ptxt``/``.ptx`` template offline."""
    dec_base, delta, k0, top_chunks = _prepare_validated_base(request)
    session = PtxSession(dec_base, xor_delta=delta)
    baseline_warnings = tuple(session.warnings)
    _validate_write_request(session, request)

    by_tag = {c.tag: c.raw for c in top_chunks}
    delete_lc = {n.lower() for n in request.delete_track_names}
    deleted_old_indices = [i for i, trk in enumerate(session.tracks) if trk.name.lower() in delete_lc]
    survival = plan_existing_media_survival(
        by_tag.get(0x1015) or _empty_optional_chunk(0x1015),
        by_tag.get(0x1054) or _empty_optional_chunk(0x1054),
        request.delete_track_names,
        c2624_raw=by_tag[0x2624], deleted_old_indices=deleted_old_indices,
    )
    base_af_count = _chunk_u32_count(by_tag[0x1004], "0x1004") if 0x1004 in by_tag else 0
    base_clip_count = _chunk_u32_count(by_tag[0x262A], "0x262A") if 0x262A in by_tag else 0
    audio_specs, clips = _prepare_clips_and_specs(
        session, request.tracks,
        base_audio_file_index=base_af_count, base_clip_index=base_clip_count,
        base_lane_index=len(survival.lanes), base_playlist_index=len(survival.playlists),
    )
    midi_survival = plan_existing_midi_survival(
        by_tag.get(0x1058) or _empty_optional_chunk(0x1058),
        by_tag[0x2624], deleted_old_indices,
    )
    extra_specs = _prepare_non_audio_specs(session, request.extra_tracks, base_midi_index=len(midi_survival.entries))
    new_specs = (*audio_specs, *extra_specs)
    layout, old_to_new, folder_insertions, deleted_uuids, deleted_actual = _compute_unified_layout(
        session, new_specs, request.delete_track_names,
    )
    mut_plan = _StructuralMutationPlan(
        old_track_count=len(session.tracks), request=request, new_specs=new_specs, clips=clips,
        layout=layout, old_to_new=old_to_new, folder_insertions=folder_insertions,
        deleted_uuids=deleted_uuids, survival=survival, midi_survival=midi_survival,
    )
    mutated_dec = _apply_structural_mutations(dec_base, top_chunks, mut_plan)
    mutated_enc = encrypt_ptx(mutated_dec, delta, k0=k0)
    final_enc, summary = _apply_signal_flow_and_groups(mutated_enc, request)

    created_names = tuple(spec.track_name for spec in new_specs)
    _verify_written_ptx(
        final_enc, len(layout), created_names,
        base_af_count + len(request.audio_files),
        base_clip_count + len(clips),
        len(survival.playlists) + len(audio_specs),
        baseline_warnings=baseline_warnings,
    )
    atomic_write_bytes(request.output_ptx_path, final_enc)
    return OfflinePtxWriteResult(
        ptx_path=request.output_ptx_path,
        created_tracks=created_names,
        deleted_tracks=deleted_actual,
        patch_summary=summary,
    )
