"""Declarative request/result models and upfront validators for offline Pro Tools ``.ptx`` session writing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from .codec import normalize_ptx_uuid16, walk_sequential_zchunks
from .models import (
    TICK_MASK_56,
    PtxChunk,
    PtxSessionSettings,
    PtxTrackStatePatch,
    PtxUnsupportedTemplateError,
    PtxValidationError,
)
from .parser import PtxSession
from .writer_blueprints import TRACK_TYPE_TO_251A, OutputBusSpec, resolve_track_format_codes
from .writer_media_mutations import AudioFileWriteSpec

_VALID_GROUP_ACTIONS = frozenset({"create", "add_members", "link_vca"})
_EMPTY_MAP: Mapping[str, float] = MappingProxyType({})
_EMPTY_STATE_MAP: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]] = MappingProxyType({})
_FOLDER_TRACK_TYPES = frozenset({"RoutingFolder", "BasicFolder"})
_NON_AUDIO_TRACK_TYPES = frozenset(t for t in TRACK_TYPE_TO_251A if t != "Audio")


@dataclass(frozen=True)
class OfflineClipSpec:
    """Typed clip placement specification with unambiguous timeline vs. source-file offsets."""

    audio_file_index: int
    clip_base_name: str
    length_samples: int
    timeline_start_sample: int = 0
    source_start_sample: int = 0
    sync_point_samples: int = 0
    orig_timestamp: int = 0
    user_timestamp: int = 0
    is_muted: bool = False
    is_time_locked: bool | None = None
    is_edit_locked: bool = False
    color_index: int = 0xFFFE


@dataclass(frozen=True)
class OfflineAudioTrackRequest:
    """One Audio track to synthesize inside a Pro Tools template (Mono, Stereo, or Surround)."""

    track_name: str
    channels: int | None = None
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
    parent_folder_name: str | None = None
    clip: OfflineClipSpec | None = None

    def resolved_clip(self) -> OfflineClipSpec | None:
        """Return the normalized ``OfflineClipSpec`` when this track carries an audio clip."""
        if self.clip is not None:
            if self.audio_file_index is not None:
                raise PtxValidationError(
                    f"Track {self.track_name!r}: specify either clip or audio_file_index, not both"
                )
            return self.clip
        if self.audio_file_index is None:
            return None
        return OfflineClipSpec(
            audio_file_index=self.audio_file_index,
            clip_base_name=self.clip_base_name,
            length_samples=self.length_samples,
            timeline_start_sample=self.start_sample,
            source_start_sample=self.clip_start_sample,
            sync_point_samples=self.clip_sync_point_samples,
            orig_timestamp=self.clip_orig_timestamp,
            user_timestamp=self.clip_user_timestamp,
            is_muted=self.clip_muted,
            is_time_locked=True if self.clip_edit_locked else self.clip_time_locked,
            is_edit_locked=self.clip_edit_locked,
            color_index=self.clip_color_index,
        )


@dataclass(frozen=True)
class OfflineNonAudioTrackRequest:
    """One non-Audio track (Aux, RoutingFolder, Master, Vca, BasicFolder, Midi, Instrument) to synthesize."""

    track_name: str
    track_type: str
    channels: int | None = None
    parent_folder_uuid_hex: str | None = None
    output_bus_name: str | None = None
    color_index: int = 0xFFFE
    is_folder_open: bool = True
    track_format: str | None = None
    parent_folder_name: str | None = None
    track_uuid_hex: str | None = None


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
    fade_type: str
    start_sample: int
    total_samples: int
    pre_samples: int | None = None
    shape: str = "standard"
    crossfade_link: str = "equal_power"


@dataclass(frozen=True)
class OfflineGroupRequest:
    """One Pro Tools Mix/Edit group (optionally VCA-linked) to apply after track creation."""

    name: str
    action: str
    member_tracks: tuple[str, ...]
    vca_track: str | None = None
    group_type: str = "Mix/Edit"
    is_edit_active: bool = False
    is_mix_active: bool = False


@dataclass(frozen=True)
class OfflinePtxWriteRequest:
    """Complete self-contained offline ``.ptx`` generation request (file-backed or in-memory)."""

    ptxt_path: Path | None = None
    output_ptx_path: Path | None = None
    session_dir: Path = Path(".")
    sample_rate: int = 48000
    bit_depth: int = 24
    audio_files: tuple[AudioFileWriteSpec, ...] = ()
    tracks: tuple[OfflineAudioTrackRequest, ...] = ()
    extra_tracks: tuple[OfflineNonAudioTrackRequest, ...] = ()
    delete_track_names: tuple[str, ...] = ()
    clip_gains_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    slot_a_trims_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    track_volumes_db: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    track_states: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]] = field(default_factory=lambda: _EMPTY_STATE_MAP)
    markers: tuple[OfflineMarkerRequest, ...] = ()
    fades: tuple[OfflineFadeRequest, ...] = ()
    group_requests: tuple[OfflineGroupRequest, ...] = ()
    volume_serial_u32: int = 0x53505250
    volume_label: str = "SessionPrep"
    session_settings: PtxSessionSettings | None = None
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
    template_path: Path | None = None

    def __post_init__(self) -> None:
        if self.ptxt_path is not None and self.template_path is not None and self.ptxt_path != self.template_path:
            raise PtxValidationError(
                f"Conflicting ptxt_path ({self.ptxt_path}) and template_path ({self.template_path})"
            )
        resolved = self.ptxt_path if self.ptxt_path is not None else self.template_path
        object.__setattr__(self, "ptxt_path", resolved)
        object.__setattr__(self, "template_path", resolved)

    def effective_session_settings(self) -> PtxSessionSettings:
        """Merge ``session_settings`` with flat fields (flat optional fields override; ``session_settings`` sample_rate/bit_depth take precedence)."""
        base = self.session_settings
        def _pick(name: str, default: Any = None) -> Any:
            flat_val = getattr(self, name)
            if flat_val is not None:
                return flat_val
            return getattr(base, name) if base is not None else default

        return PtxSessionSettings(
            sample_rate=base.sample_rate if (base is not None and base.sample_rate is not None) else self.sample_rate,
            bit_depth=base.bit_depth if (base is not None and base.bit_depth is not None) else self.bit_depth,
            audio_format=_pick("audio_format"),
            is_interleaved=_pick("is_interleaved"),
            audio_pull_rate=_pick("audio_pull_rate"),
            video_pull_rate=_pick("video_pull_rate"),
            fps_code=_pick("fps_code"),
            start_frames=_pick("start_frames"),
            feet_frames_rate=_pick("feet_frames_rate"),
            session_length=_pick("session_length"),
            main_counter_format=_pick("main_counter_format"),
            sub_counter_format=_pick("sub_counter_format"),
            edit_mode=_pick("edit_mode"),
            grid_relative=_pick("grid_relative"),
            edit_tool=_pick("edit_tool"),
            smart_tool_enabled=_pick("smart_tool_enabled"),
            pre_roll_enabled=_pick("pre_roll_enabled"),
            pre_roll_samples=_pick("pre_roll_samples"),
            post_roll_enabled=_pick("post_roll_enabled"),
            post_roll_samples=_pick("post_roll_samples"),
            timeline_in_sample=_pick("timeline_in_sample"),
            timeline_out_sample=_pick("timeline_out_sample"),
        )


@dataclass(frozen=True)
class OfflinePtxWriteResult:
    """Summary of a completed offline ``.ptx`` write and roundtrip verification."""

    ptx_path: Path | None
    created_tracks: tuple[str, ...]
    deleted_tracks: tuple[str, ...]
    patch_summary: dict[str, Any]


def validate_template_supported(top_chunks: Sequence[PtxChunk]) -> None:
    """Ensure the template contains all required chunks and supported schema versions."""
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


def resolve_output_bus(session: PtxSession, bus_name: str) -> OutputBusSpec:
    """Resolve *bus_name* against *session.signal_paths* into an ``OutputBusSpec``."""
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


def _validate_parent_ref(
    track_name: str,
    parent_uuid_hex: str | None,
    parent_name: str | None,
    folder_uuids_alive: set[bytes],
    folder_names_alive_lc: set[str],
) -> None:
    if parent_uuid_hex is not None and parent_name is not None:
        raise PtxValidationError(f"Track {track_name!r}: specify either parent_folder_uuid_hex or parent_folder_name, not both")
    if parent_uuid_hex is not None:
        parent_uid = normalize_ptx_uuid16(parent_uuid_hex)
        if parent_uid not in folder_uuids_alive:
            raise PtxValidationError(
                f"Track {track_name!r} targets missing, non-folder, or deleted parent folder UUID {parent_uuid_hex!r}"
            )
    if parent_name is not None and parent_name.strip().lower() not in folder_names_alive_lc:
        raise PtxValidationError(f"Track {track_name!r} targets missing, non-folder, or deleted parent folder {parent_name!r}")


def _validate_audio_track_request(
    session: PtxSession,
    request: OfflinePtxWriteRequest,
    req: OfflineAudioTrackRequest,
    folder_uuids_alive: set[bytes],
    folder_names_alive_lc: set[str],
    used_names_lc: set[str],
) -> None:
    if not req.track_name or not req.track_name.strip():
        raise PtxValidationError("Track name must not be empty")
    low_name = req.track_name.lower()
    if low_name in used_names_lc:
        raise PtxValidationError(f"Duplicate or colliding track name {req.track_name!r}")
    used_names_lc.add(low_name)
    _lw, _mw, _ew, resolved_ch = resolve_track_format_codes(req.channels, req.track_format)
    if isinstance(req.color_index, bool) or not (0 <= req.color_index <= 0xFFFF):
        raise PtxValidationError(f"Track {req.track_name!r} color_index out of range: {req.color_index!r}")
    clip = req.resolved_clip()
    if clip is not None:
        if isinstance(clip.timeline_start_sample, bool) or not (0 <= clip.timeline_start_sample <= TICK_MASK_56):
            raise PtxValidationError(f"Track {req.track_name!r} start_sample out of range: {clip.timeline_start_sample!r}")
        if isinstance(clip.color_index, bool) or not (0 <= clip.color_index <= 0xFFFF):
            raise PtxValidationError(f"Track {req.track_name!r} clip_color_index out of range: {clip.color_index!r}")
        if clip.is_edit_locked and clip.is_time_locked is False:
            raise PtxValidationError(f"Track {req.track_name!r}: clip is_edit_locked=True requires is_time_locked=True")
        for lbl, val in (
            ("clip_start_sample", clip.source_start_sample),
            ("length_samples", clip.length_samples),
            ("clip_sync_point_samples", clip.sync_point_samples),
            ("clip_orig_timestamp", clip.orig_timestamp),
            ("clip_user_timestamp", clip.user_timestamp),
        ):
            if isinstance(val, bool) or not (0 <= val <= 0xFFFF_FFFF_FFFF_FFFF):
                raise PtxValidationError(f"Track {req.track_name!r} {lbl} out of u64 range: {val!r}")
        if isinstance(clip.audio_file_index, bool) or not (0 <= clip.audio_file_index < len(request.audio_files)):
            raise PtxValidationError(
                f"Track {req.track_name!r} audio_file_index {clip.audio_file_index} out of range (0..{len(request.audio_files) - 1})"
            )
        af = request.audio_files[clip.audio_file_index]
        if resolved_ch > af.channels:
            raise PtxValidationError(
                f"Track {req.track_name!r} requires {resolved_ch} channels but audio file {clip.audio_file_index} has {af.channels}"
            )
        if clip.length_samples <= 0 or clip.source_start_sample + clip.length_samples > af.length_samples:
            raise PtxValidationError(
                f"Track {req.track_name!r} clip range [{clip.source_start_sample}..{clip.source_start_sample + clip.length_samples}] "
                f"invalid for audio file length {af.length_samples}"
            )
        if not clip.clip_base_name or not clip.clip_base_name.strip():
            raise PtxValidationError(f"Track {req.track_name!r} clip_base_name must not be empty")
    elif isinstance(req.start_sample, bool) or not (0 <= req.start_sample <= TICK_MASK_56):
        raise PtxValidationError(f"Track {req.track_name!r} start_sample out of range: {req.start_sample!r}")
    _validate_parent_ref(req.track_name, req.parent_folder_uuid_hex, req.parent_folder_name, folder_uuids_alive, folder_names_alive_lc)
    if req.output_bus_name:
        resolve_output_bus(session, req.output_bus_name)


def _validate_non_audio_track_request(
    session: PtxSession,
    req: OfflineNonAudioTrackRequest,
    folder_uuids_alive: set[bytes],
    folder_names_alive_lc: set[str],
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
        _lw, _mw, _ew, resolved_ch = resolve_track_format_codes(req.channels, req.track_format)
        if resolved_ch not in (1, 2):
            raise PtxValidationError(
                f"Instrument track {req.track_name!r} only supports Mono or Stereo, got {req.track_format or req.channels!r}"
            )
    elif req.track_type not in ("Vca", "BasicFolder", "Midi"):
        resolve_track_format_codes(req.channels, req.track_format)
    if isinstance(req.color_index, bool) or not (0 <= req.color_index <= 0xFFFF):
        raise PtxValidationError(f"Track {req.track_name!r} color_index out of range: {req.color_index!r}")
    if req.track_uuid_hex is not None:
        normalize_ptx_uuid16(req.track_uuid_hex)
    _validate_parent_ref(req.track_name, req.parent_folder_uuid_hex, req.parent_folder_name, folder_uuids_alive, folder_names_alive_lc)
    if req.output_bus_name:
        resolve_output_bus(session, req.output_bus_name)


def validate_write_request(session: PtxSession, request: OfflinePtxWriteRequest) -> None:
    """Validate all tracks, folder targets, deletions, and group requests upfront."""
    delete_lc = {n.lower() for n in request.delete_track_names}
    deleted_uuids = {bytes.fromhex(t.uuid_hex) for t in session.tracks if t.uuid_hex and t.name.lower() in delete_lc}
    folder_uuids_alive: set[bytes] = set()
    folder_names_alive_lc: set[str] = set()
    used_names_lc: set[str] = set()
    for trk in session.tracks:
        if trk.name.lower() not in delete_lc:
            used_names_lc.add(trk.name.lower())
            if trk.track_type in _FOLDER_TRACK_TYPES:
                folder_names_alive_lc.add(trk.name.lower())
                if trk.uuid_hex:
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
    for extra in request.extra_tracks:
        if extra.track_type in _FOLDER_TRACK_TYPES and extra.track_name.strip():
            folder_names_alive_lc.add(extra.track_name.strip().lower())
            if extra.track_uuid_hex is not None:
                folder_uuids_alive.add(normalize_ptx_uuid16(extra.track_uuid_hex))
    for req in request.tracks:
        _validate_audio_track_request(session, request, req, folder_uuids_alive, folder_names_alive_lc, used_names_lc)
    for extra in request.extra_tracks:
        _validate_non_audio_track_request(session, extra, folder_uuids_alive, folder_names_alive_lc, used_names_lc)
    for grp in request.group_requests:
        if grp.action not in _VALID_GROUP_ACTIONS:
            raise PtxValidationError(f"Unsupported offline group action {grp.action!r} for group {grp.name!r}")
        if grp.action == "link_vca" and (not grp.vca_track or not grp.vca_track.strip()):
            raise PtxValidationError(f"Group {grp.name!r} with action='link_vca' requires a non-empty vca_track")
