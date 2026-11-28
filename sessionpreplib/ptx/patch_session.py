"""Session-level ``.ptx`` patchers: audio format, timecode, UI/transport state, and memory locations/markers."""

from __future__ import annotations

from dataclasses import asdict
import struct
from typing import Any

from ._document import PtxBuffer, assemble_top_chunks
from .codec import splice_top_chunk, walk_sequential_zchunks
from .models import (
    PtxCorruptFileError,
    PtxNotFoundError,
    PtxSessionSettings,
    PtxValidationError,
    TrackRef,
)
from .parser import PtxSession
from .writer_chunks import (
    patch_session_audio_format_1028_decrypted,
    patch_session_timecode_204d_decrypted,
    patch_session_ui_state_decrypted,
    serialize_marker_2077,
)
from .writer_index import apply_inserted_top_chunk_index_0002
from .writer_primitives import build_zchunk, parse_index_0002

_AUDIO_FIELDS = (
    "sample_rate", "bit_depth", "audio_format", "is_interleaved", "audio_pull_rate", "video_pull_rate",
)
_TC_FIELDS = ("fps_code", "start_frames", "feet_frames_rate", "session_length")
_UI_FIELDS = (
    "main_counter_format", "sub_counter_format", "edit_mode", "grid_relative",
    "edit_tool", "smart_tool_enabled", "pre_roll_enabled", "pre_roll_samples",
    "post_roll_enabled", "post_roll_samples", "timeline_in_sample", "timeline_out_sample",
)


def _merge_session_settings(
    settings: PtxSessionSettings | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = asdict(settings) if settings is not None else {}
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def patch_session_settings_on_buffer(
    ptx_buf: PtxBuffer,
    settings: PtxSessionSettings | None = None,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format: str | None = None,
    is_interleaved: bool | None = None,
    audio_pull_rate: str | None = None,
    video_pull_rate: str | None = None,
    fps_code: int | None = None,
    start_frames: int | None = None,
    feet_frames_rate: str | None = None,
    session_length: str | None = None,
    main_counter_format: str | None = None,
    sub_counter_format: str | None = None,
    edit_mode: str | None = None,
    grid_relative: bool | None = None,
    edit_tool: str | None = None,
    smart_tool_enabled: bool | None = None,
    pre_roll_enabled: bool | None = None,
    pre_roll_samples: int | None = None,
    post_roll_enabled: bool | None = None,
    post_roll_samples: int | None = None,
    timeline_in_sample: int | None = None,
    timeline_out_sample: int | None = None,
) -> dict[str, Any]:
    """Patch ``0x1028``, ``0x204D``, and UI counter/edit/roll/selection settings in-place on *ptx_buf*."""
    merged = _merge_session_settings(
        settings,
        {
            "sample_rate": sample_rate, "bit_depth": bit_depth, "audio_format": audio_format,
            "is_interleaved": is_interleaved, "audio_pull_rate": audio_pull_rate,
            "video_pull_rate": video_pull_rate, "fps_code": fps_code, "start_frames": start_frames,
            "feet_frames_rate": feet_frames_rate, "session_length": session_length,
            "main_counter_format": main_counter_format, "sub_counter_format": sub_counter_format,
            "edit_mode": edit_mode, "grid_relative": grid_relative, "edit_tool": edit_tool,
            "smart_tool_enabled": smart_tool_enabled, "pre_roll_enabled": pre_roll_enabled,
            "pre_roll_samples": pre_roll_samples, "post_roll_enabled": post_roll_enabled,
            "post_roll_samples": post_roll_samples, "timeline_in_sample": timeline_in_sample,
            "timeline_out_sample": timeline_out_sample,
        },
    )
    changes: dict[str, Any] = {}
    audio_kw = {k: merged.get(k) for k in _AUDIO_FIELDS}
    if any(v is not None for v in audio_kw.values()):
        patch_session_audio_format_1028_decrypted(ptx_buf.data, **audio_kw)
        changes.update({k: v for k, v in audio_kw.items() if v is not None})
    tc_kw = {k: merged.get(k) for k in _TC_FIELDS}
    if any(v is not None for v in tc_kw.values()):
        patch_session_timecode_204d_decrypted(ptx_buf.data, **tc_kw)
        changes.update({k: v for k, v in tc_kw.items() if v is not None})
    ui_kw = {k: merged.get(k) for k in _UI_FIELDS}
    if any(v is not None for v in ui_kw.values()):
        applied = patch_session_ui_state_decrypted(ptx_buf.data, **ui_kw)
        changes.update({k: v for k, v in ui_kw.items() if v is not None and k in applied})
    if changes:
        ptx_buf.invalidate()
    return changes


def patch_session_settings(
    ptx_raw: bytes,
    settings: PtxSessionSettings | None = None,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format: str | None = None,
    is_interleaved: bool | None = None,
    audio_pull_rate: str | None = None,
    video_pull_rate: str | None = None,
    fps_code: int | None = None,
    start_frames: int | None = None,
    feet_frames_rate: str | None = None,
    session_length: str | None = None,
    main_counter_format: str | None = None,
    sub_counter_format: str | None = None,
    edit_mode: str | None = None,
    grid_relative: bool | None = None,
    edit_tool: str | None = None,
    smart_tool_enabled: bool | None = None,
    pre_roll_enabled: bool | None = None,
    pre_roll_samples: int | None = None,
    post_roll_enabled: bool | None = None,
    post_roll_samples: int | None = None,
    timeline_in_sample: int | None = None,
    timeline_out_sample: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Patch ``0x1028``, ``0x204D``, and UI counter/edit/roll/selection settings in place."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    changes = patch_session_settings_on_buffer(
        ptx_buf, settings,
        sample_rate=sample_rate, bit_depth=bit_depth, audio_format=audio_format,
        is_interleaved=is_interleaved, audio_pull_rate=audio_pull_rate,
        video_pull_rate=video_pull_rate, fps_code=fps_code, start_frames=start_frames,
        feet_frames_rate=feet_frames_rate, session_length=session_length,
        main_counter_format=main_counter_format, sub_counter_format=sub_counter_format,
        edit_mode=edit_mode, grid_relative=grid_relative, edit_tool=edit_tool,
        smart_tool_enabled=smart_tool_enabled, pre_roll_enabled=pre_roll_enabled,
        pre_roll_samples=pre_roll_samples, post_roll_enabled=post_roll_enabled,
        post_roll_samples=post_roll_samples, timeline_in_sample=timeline_in_sample,
        timeline_out_sample=timeline_out_sample,
    )
    return ptx_buf.to_encrypted(), changes


def _split_2030_markers(c2030_raw: bytes) -> tuple[int, list[bytes], bytes]:
    if len(c2030_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x2030 markers chunk in .ptx")
    ver = struct.unpack_from("<H", c2030_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2030_raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(c2030_raw, 13, cnt, expected_tag=0x2077)
    trailer = c2030_raw[end_pos:] if end_pos < len(c2030_raw) else (b"\x00" * 8)
    return ver, items, trailer


def _build_2030_chunk(items: list[bytes], version: int = 5, trailer: bytes = b"\x00" * 8) -> bytes:
    return build_zchunk(0x2030, struct.pack("<I", len(items)) + b"".join(items) + trailer, version=version)


def _resolve_marker_target(
    session: PtxSession,
    number: int | None,
    track: TrackRef | None,
) -> tuple[int, int, str]:
    used_nums = {m.number for m in session.markers}
    if number is None:
        resolved_num = next((n for n in range(1, 1000) if n not in used_nums), None)
        if resolved_num is None:
            raise PtxValidationError("All 999 Pro Tools memory location numbers are in use")
    else:
        if isinstance(number, bool):
            raise PtxValidationError(f"Invalid marker number: {number!r}")
        if number in used_nums:
            raise PtxValidationError(f"Memory location number {number} already exists in session")
        resolved_num = number
    if track is None:
        return resolved_num, -1, ""
    if isinstance(track, bool):
        raise PtxValidationError(f"Invalid marker track reference: {track!r}")
    if isinstance(track, int):
        if not (0 <= track < len(session.tracks)):
            raise PtxNotFoundError(f"Marker track index {track} out of range")
        return resolved_num, track, session.tracks[track].name
    trk = session.track_by_name(str(track))
    if trk is None:
        raise PtxNotFoundError(f"Track {track!r} not found in .ptx session")
    return resolved_num, trk.index, trk.name


def patch_add_marker_on_buffer(
    ptx_buf: PtxBuffer,
    name: str,
    start_pos: int,
    *,
    number: int | None = None,
    end_pos: int | None = None,
    reference: str = "Absolute",
    time_properties: str = "Marker",
    comments: str = "",
    color_index: int = 0,
    color_rgb: tuple[int, int, int] | None = None,
    track: TrackRef | None = None,
    zoom_settings: bool = False,
    pre_post_roll_times: bool = False,
    track_visibility: bool = False,
    track_heights: bool = False,
    group_enables: bool = False,
    window_configuration: bool = False,
) -> dict[str, Any]:
    """Add a Memory Location / Marker to ``0x2030`` in-place on *ptx_buf* (synthesizing ``0x2030`` if absent)."""
    session = ptx_buf.session
    resolved_num, track_idx, track_name = _resolve_marker_target(session, number, track)
    new_2077 = serialize_marker_2077(
        resolved_num, name, start_pos,
        end_pos=end_pos, reference=reference, time_properties=time_properties,
        comments=comments, color_index=color_index, color_rgb=color_rgb, track_index=track_idx,
        zoom_settings=zoom_settings, pre_post_roll_times=pre_post_roll_times,
        track_visibility=track_visibility, track_heights=track_heights,
        group_enables=group_enables, window_configuration=window_configuration,
    )
    c2030 = session.chunk_map.get(0x2030)
    if c2030 is not None:
        ver, items, trailer = _split_2030_markers(c2030.raw)
        items.append(new_2077)
        splice_top_chunk(ptx_buf.data, 0x2030, _build_2030_chunk(items, version=ver, trailer=trailer))
    else:
        old_dec = bytes(ptx_buf.data)
        template_idx = parse_index_0002(old_dec) if 0x0002 in session.chunk_map else None
        new_2030 = _build_2030_chunk([new_2077], version=5, trailer=b"\x00" * 8)
        new_buf = assemble_top_chunks(old_dec, {}, insertions=((0x2030, new_2030, (0x271A, 0x2505, 0x2066)),))
        if template_idx is not None:
            apply_inserted_top_chunk_index_0002(new_buf, template_idx, 0x2030, new_2030)
        ptx_buf.replace_data(new_buf)
    ptx_buf.invalidate()
    return {
        "action": "add_marker", "number": resolved_num, "name": name,
        "location": "Track" if track_idx >= 0 else "MainRuler", "track_name": track_name,
    }


def patch_add_marker(
    ptx_raw: bytes,
    name: str,
    start_pos: int,
    *,
    number: int | None = None,
    end_pos: int | None = None,
    reference: str = "Absolute",
    time_properties: str = "Marker",
    comments: str = "",
    color_index: int = 0,
    color_rgb: tuple[int, int, int] | None = None,
    track: TrackRef | None = None,
    zoom_settings: bool = False,
    pre_post_roll_times: bool = False,
    track_visibility: bool = False,
    track_heights: bool = False,
    group_enables: bool = False,
    window_configuration: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Add a Memory Location / Marker to ``0x2030`` (Main Ruler or Track marker)."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_add_marker_on_buffer(
        ptx_buf, name, start_pos,
        number=number, end_pos=end_pos, reference=reference, time_properties=time_properties,
        comments=comments, color_index=color_index, color_rgb=color_rgb, track=track,
        zoom_settings=zoom_settings, pre_post_roll_times=pre_post_roll_times,
        track_visibility=track_visibility, track_heights=track_heights,
        group_enables=group_enables, window_configuration=window_configuration,
    )
    return ptx_buf.to_encrypted(), info


def patch_delete_marker_on_buffer(ptx_buf: PtxBuffer, number: int) -> dict[str, Any]:
    """Delete an existing Memory Location / Marker by its 1-based number in-place on *ptx_buf*."""
    if isinstance(number, bool):
        raise PtxValidationError(f"Invalid marker number: {number!r}")
    session = ptx_buf.session
    c2030 = session.chunk_map.get(0x2030)
    if c2030 is None:
        raise PtxNotFoundError(f"Marker #{number} not found in .ptx session")
    ver, items, trailer = _split_2030_markers(c2030.raw)
    kept: list[bytes] = []
    removed = False
    for it in items:
        if len(it) >= 11 and struct.unpack_from("<H", it, 9)[0] == number:
            removed = True
        else:
            kept.append(it)
    if not removed:
        raise PtxNotFoundError(f"Marker #{number} not found in .ptx session")
    splice_top_chunk(ptx_buf.data, 0x2030, _build_2030_chunk(kept, version=ver, trailer=trailer))
    ptx_buf.invalidate()
    return {"action": "delete_marker", "number": number}


def patch_delete_marker(ptx_raw: bytes, number: int) -> tuple[bytes, dict[str, Any]]:
    """Delete an existing Memory Location / Marker by its 1-based number from ``0x2030``."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_delete_marker_on_buffer(ptx_buf, number)
    return ptx_buf.to_encrypted(), info
