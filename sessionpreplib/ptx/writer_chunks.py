"""Verified Pro Tools ``.ptx`` chunk serializers & offline palette mapper for offline writer prep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import struct
from typing import Any
import uuid

from .codec import decrypt_ptx, encrypt_ptx, find_zchunk_offset, iter_top_chunks
from .models import (
    AUDIO_FORMAT_MAP,
    AUDIO_PULL_4_MAP,
    AUDIO_PULL_01_MAP,
    COUNTER_FORMAT_MAP,
    CROSSFADE_LINK_MAP,
    EDIT_MODE_MAP,
    EDIT_MODE_MASK_MAP,
    EDIT_TOOL_MAP,
    FADE_SHAPE_MAP,
    FEET_FRAMES_RATE_MAP,
    FPS_CODE_MAP,
    MARKER_GENERAL_PROP_BITS,
    MARKER_REFERENCE_MAP,
    MARKER_TIME_PROPS,
    PT12_COLOR_PALETTE,
    TICK_MASK_56,
    VALID_PROTOOLS_BIT_DEPTHS,
    VALID_PROTOOLS_SAMPLE_RATES,
)
from .writer_primitives import build_zchunk, encode_lp_str

_VALID_AUDIO_RATES = VALID_PROTOOLS_SAMPLE_RATES
_VALID_BIT_DEPTHS = VALID_PROTOOLS_BIT_DEPTHS
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_AUDIO_FORMAT_TO_CODE = {v.upper(): k for k, v in AUDIO_FORMAT_MAP.items()}
_FEET_FRAMES_TO_CODE = {v: k for k, v in FEET_FRAMES_RATE_MAP.items()}
_PULL_4_TO_CODE = {v: k for k, v in AUDIO_PULL_4_MAP.items() if v}
_PULL_01_TO_CODE = {v: k for k, v in AUDIO_PULL_01_MAP.items() if v}
_MARKER_REF_TO_CODE = {v.lower(): k for k, v in MARKER_REFERENCE_MAP.items()}
_MARKER_PROP_TO_CODE = {v.lower(): k for k, v in MARKER_TIME_PROPS.items()}
_COUNTER_FMT_TO_CODE = {v.lower(): k for k, v in COUNTER_FORMAT_MAP.items()}
_EDIT_MODE_TO_CODE = {v.lower(): k for k, v in EDIT_MODE_MAP.items()}
_EDIT_TOOL_TO_CODE = {v.lower(): k for k, v in EDIT_TOOL_MAP.items()}
_FADE_SHAPE_TO_CODE = {v.lower(): k for k, v in FADE_SHAPE_MAP.items()}
_CROSSFADE_LINK_TO_CODE = {v.lower(): k for k, v in CROSSFADE_LINK_MAP.items()}
_SESSION_LENGTH_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([:;])(\d{2})$")
_MARKER_VIEW_DEFAULT_68 = bytes.fromhex(
    "000000000000f0bfffffffffffffffffffffffffffbfffffffffffffffbf"
    "00000000000000400000000000000040ffffffff090000000000000000000000000100000000"
)
_MARKER_MID_FLAGS_40 = (
    (b"\x00" * 8) + (b"\xff" * 4) + (b"\x00" * 12) + (b"\xff" * 4) + (b"\x00" * 12)
)


def _encode_min_le(value: int, field_name: str) -> bytes:
    if isinstance(value, bool) or value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"{field_name} out of u64 range: {value}")
    if value == 0:
        return b""
    width = (value.bit_length() + 7) // 8
    return value.to_bytes(width, "little")


@dataclass(frozen=True)
class PtxDirTreeEntry:
    """One node inside a ``0x1004`` -> ``0x103a`` audio directory tree."""

    kind: int  # 0=volume root, 1=directory, 2=audio file
    parent_index: int  # 0xFFFFFFFF for volume root, else 0-based index
    name: str
    tag4: bytes = b"\x00\x00\x00\x00"  # 4-byte ASCII/LE type tag (e.g. b"EVAW", b"fFXM", or volume hash)


def parse_pull_rate_codes(label: str) -> tuple[int, int]:
    """Parse a pull-rate label (e.g. ``'None'``, ``'+0.1%'``, ``'+4% -0.1%'``) into ``(pull_4, pull_01)``."""
    cleaned = label.strip()
    if not cleaned or cleaned.lower() == "none":
        return 0, 0
    pull_4 = 0
    pull_01 = 0
    for token in cleaned.split():
        if token in _PULL_4_TO_CODE and pull_4 == 0:
            pull_4 = _PULL_4_TO_CODE[token]
        elif token in _PULL_01_TO_CODE and pull_01 == 0:
            pull_01 = _PULL_01_TO_CODE[token]
        else:
            raise ValueError(f"Unsupported Pro Tools pull rate: {label!r}")
    return pull_4, pull_01


def _resolve_audio_format_code(audio_format: str | None, audio_format_code: int | None) -> int | None:
    if audio_format is not None and audio_format_code is not None:
        raise ValueError("Provide either audio_format or audio_format_code, not both")
    if audio_format is not None:
        code = _AUDIO_FORMAT_TO_CODE.get(audio_format.strip().upper())
        if code is None:
            raise ValueError(f"Unsupported Pro Tools audio format: {audio_format!r} (expected WAV or AIFF)")
        return code
    if audio_format_code is not None and (isinstance(audio_format_code, bool) or not (0 <= audio_format_code <= 0xFF)):
        raise ValueError(f"audio_format_code out of u8 range: {audio_format_code}")
    return audio_format_code


def patch_session_audio_format_1028_decrypted(
    buf: bytearray,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format_code: int | None = None,
    audio_format: str | None = None,
    is_interleaved: bool | None = None,
    audio_pull_rate: str | None = None,
    video_pull_rate: str | None = None,
) -> None:
    """Patch ``0x1028`` session format, bit depth, pull rates, and interleaved state in-place."""
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate not in _VALID_AUDIO_RATES):
        raise ValueError(f"Unsupported Pro Tools sample rate: {sample_rate}")
    if bit_depth is not None and (isinstance(bit_depth, bool) or bit_depth not in _VALID_BIT_DEPTHS):
        raise ValueError(f"Unsupported Pro Tools bit depth: {bit_depth}")
    fmt_code = _resolve_audio_format_code(audio_format, audio_format_code)
    ap_codes = parse_pull_rate_codes(audio_pull_rate) if audio_pull_rate is not None else None
    vp_codes = parse_pull_rate_codes(video_pull_rate) if video_pull_rate is not None else None
    if vp_codes is not None and vp_codes[0] != 0:
        raise ValueError(f"Video pull rate only supports +/-0.1% or None in 0x1028, got {video_pull_rate!r}")

    chunks = iter_top_chunks(bytes(buf), strict=True)
    c1028 = next((c for c in reversed(chunks) if c.tag == 0x1028), None)
    if c1028 is None or len(c1028.raw) < 15:
        raise RuntimeError("Missing or truncated 0x1028 session audio format chunk in .ptx")

    base = c1028.offset
    end = base + len(c1028.raw)
    has_tail = len(c1028.raw) >= 31
    if fmt_code is not None:
        buf[base + 9] = fmt_code
    if bit_depth is not None:
        if bit_depth == 32 and has_tail:
            buf[base + 10] = 24
            buf[end - 8] = 1
            buf[end - 6] = 32
        else:
            buf[base + 10] = bit_depth
            if has_tail:
                buf[end - 8] = 0
                buf[end - 6] = bit_depth
    if sample_rate is not None:
        struct.pack_into("<I", buf, base + 11, sample_rate)
    if ap_codes is not None:
        if len(c1028.raw) < 16 or not has_tail:
            raise RuntimeError("0x1028 chunk too short to patch audio pull rate")
        buf[base + 15] = ap_codes[1]
        buf[end - 15] = ap_codes[0]
    if vp_codes is not None:
        if not has_tail:
            raise RuntimeError("0x1028 chunk too short to patch video pull rate")
        buf[end - 14] = vp_codes[1]
    if is_interleaved is not None:
        if not has_tail:
            raise RuntimeError("0x1028 chunk too short to patch interleaved state")
        buf[end - 7] = 1 if is_interleaved else 0


def patch_session_audio_format_1028(
    ptx_raw: bytes,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format_code: int | None = None,
    audio_format: str | None = None,
    is_interleaved: bool | None = None,
    audio_pull_rate: str | None = None,
    video_pull_rate: str | None = None,
) -> bytes:
    """Patch ``0x1028`` session sample rate, bit depth, format, pull rates, and interleaved state."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    patch_session_audio_format_1028_decrypted(
        buf,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        audio_format_code=audio_format_code,
        audio_format=audio_format,
        is_interleaved=is_interleaved,
        audio_pull_rate=audio_pull_rate,
        video_pull_rate=video_pull_rate,
    )
    return encrypt_ptx(bytes(buf), delta, k0=k0)


def _validate_session_length(session_length: str, effective_fps_code: int | None) -> bytes:
    cleaned = session_length.strip()
    m = _SESSION_LENGTH_RE.match(cleaned)
    if m is None:
        raise ValueError(f"Invalid session_length {session_length!r} (expected HH:MM:SS:FF or HH:MM:SS;FF)")
    hh, mm, ss, sep, ff = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
    fps_entry = FPS_CODE_MAP.get(effective_fps_code) if effective_fps_code is not None else None
    max_ff = fps_entry[1] if fps_entry is not None else 30
    if hh < 6 or hh > 24 or (hh == 24 and (mm > 0 or ss > 0 or ff > 0)) or mm >= 60 or ss >= 60 or ff >= max_ff:
        raise ValueError(f"session_length out of Pro Tools valid range (06:00:00:00..24:00:00:00): {session_length!r}")
    if fps_entry is not None:
        sep = ";" if fps_entry[3] else ":"
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}".encode("ascii")


def patch_session_timecode_204d_decrypted(
    buf: bytearray,
    *,
    fps_code: int | None = None,
    start_frames: int | None = None,
    feet_frames_rate: str | None = None,
    session_length: str | None = None,
) -> None:
    """Patch ``0x204D`` timecode frame rate, session start frames, length, and Feet+Frames rate in-place."""
    if fps_code is not None and (isinstance(fps_code, bool) or fps_code not in FPS_CODE_MAP):
        raise ValueError(f"Unsupported Pro Tools fps_code: {fps_code!r} (valid: {sorted(FPS_CODE_MAP)})")
    if start_frames is not None and (isinstance(start_frames, bool) or not (0 <= start_frames <= 0xFFFF_FFFF)):
        raise ValueError(f"start_frames out of u32 range: {start_frames!r}")
    ff_code: int | None = None
    if feet_frames_rate is not None:
        ff_code = _FEET_FRAMES_TO_CODE.get(feet_frames_rate.strip())
        if ff_code is None:
            raise ValueError(f"Unsupported Pro Tools feet_frames_rate: {feet_frames_rate!r} (expected 23.976, 24, or 25)")

    chunks = iter_top_chunks(bytes(buf), strict=True)
    c204d = next((c for c in reversed(chunks) if c.tag == 0x204D), None)
    if c204d is None or len(c204d.raw) < 22:
        raise RuntimeError("Missing or truncated 0x204D session timecode chunk in .ptx")

    body_off = c204d.offset + 9
    body_len = len(c204d.raw) - 9
    eff_fps = fps_code if fps_code is not None else struct.unpack_from("<I", buf, body_off)[0]
    len_bytes = _validate_session_length(session_length, eff_fps) if session_length is not None else None
    if fps_code is not None:
        struct.pack_into("<I", buf, body_off, fps_code)
        if body_len >= 44:
            struct.pack_into("<I", buf, body_off + 40, fps_code)
        sep = ord(";") if FPS_CODE_MAP[fps_code][3] else ord(":")
        for tc_off in (29, 56):
            if (
                body_len >= tc_off + 11
                and struct.unpack_from("<I", buf, body_off + tc_off - 4)[0] == 11
                and buf[body_off + tc_off + 8] in (ord(":"), ord(";"))
            ):
                buf[body_off + tc_off + 8] = sep
    if len_bytes is not None:
        if body_len < 40 or struct.unpack_from("<I", buf, body_off + 25)[0] != 11:
            raise RuntimeError("0x204D chunk lacks 11-byte session_length field")
        buf[body_off + 29 : body_off + 40] = len_bytes
        if body_len >= 67 and struct.unpack_from("<I", buf, body_off + 52)[0] == 11:
            buf[body_off + 56 : body_off + 67] = len_bytes
    if start_frames is not None:
        struct.pack_into("<I", buf, body_off + 9, start_frames)
        if body_len >= 48:
            struct.pack_into("<I", buf, body_off + 44, start_frames)
    if ff_code is not None:
        if body_len <= 17:
            raise RuntimeError("0x204D chunk too short to patch feet_frames_rate")
        buf[body_off + 17] = ff_code


def patch_session_timecode_204d(
    ptx_raw: bytes,
    *,
    fps_code: int | None = None,
    start_frames: int | None = None,
    feet_frames_rate: str | None = None,
    session_length: str | None = None,
) -> bytes:
    """Patch ``0x204D`` timecode frame rate, start frames, session length, and/or Feet+Frames rate in place."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    patch_session_timecode_204d_decrypted(
        buf,
        fps_code=fps_code,
        start_frames=start_frames,
        feet_frames_rate=feet_frames_rate,
        session_length=session_length,
    )
    return encrypt_ptx(bytes(buf), delta, k0=k0)


def _resolve_edit_mode_and_grid(edit_mode: str | None, grid_relative: bool | None) -> tuple[int | None, bool | None]:
    if edit_mode is None:
        return None, grid_relative
    low = edit_mode.strip().lower()
    if low == "gridabsolute":
        return 3, False if grid_relative is None else grid_relative
    if low == "gridrelative":
        return 3, True if grid_relative is None else grid_relative
    code = _EDIT_MODE_TO_CODE.get(low)
    if code is None:
        raise ValueError(f"Unsupported Pro Tools edit_mode: {edit_mode!r}")
    return code, grid_relative


def _resolve_edit_tool_and_smart(edit_tool: str | None, smart_tool: bool | None) -> tuple[int | None, bool | None]:
    if edit_tool is None:
        return None, smart_tool
    low = edit_tool.strip().lower()
    if low in ("smarttool", "smart"):
        return None, True
    code = _EDIT_TOOL_TO_CODE.get(low)
    if code is None:
        raise ValueError(f"Unsupported Pro Tools edit_tool: {edit_tool!r}")
    return code, (False if smart_tool is None else smart_tool)


def _patch_2056_block(
    buf: bytearray,
    off: int,
    length: int,
    *,
    mc_code: int | None,
    em_code: int | None,
    tool_code: int | None,
    smart_flag: bool | None,
) -> None:
    if smart_flag is not None and length >= 10:
        buf[off + 9] = 1 if smart_flag else 0
    if tool_code is not None and length >= 23:
        buf[off + 22] = tool_code
    if em_code is not None and length >= 45:
        buf[off + 26] = em_code
        buf[off + 44] = EDIT_MODE_MASK_MAP[em_code]
    if mc_code is not None and length >= 31:
        buf[off + 30] = mc_code


def _patch_grid_relative_chunks(buf: bytearray, by_tag: dict[int, tuple[int, int]], grid_rel: bool) -> None:
    val = 1 if grid_rel else 0
    if 0x2016 in by_tag:
        off, ln = by_tag[0x2016]
        if ln >= 45:
            buf[off + ln - 45] = val
    for tag in (0x2066, 0x2530):
        if tag in by_tag:
            off, ln = by_tag[tag]
            if ln > 275:
                buf[off + 275] = val
    if 0x2587 in by_tag:
        off, ln = by_tag[0x2587]
        p66 = find_zchunk_offset(buf, 0x2066, off + 9, off + ln)
        if p66 >= 0:
            l66 = 7 + struct.unpack_from("<I", buf, p66 + 3)[0]
            if l66 > 275:
                buf[p66 + 275] = val
            elif l66 >= 197:
                buf[p66 + l66 - 4] = val


def _validate_ui_state_args(
    main_counter_format: str | None,
    sub_counter_format: str | None,
    edit_mode: str | None,
    grid_relative: bool | None,
    edit_tool: str | None,
    smart_tool_enabled: bool | None,
    pre_roll_samples: int | None,
    post_roll_samples: int | None,
    timeline_in_sample: int | None,
    timeline_out_sample: int | None,
) -> tuple[int | None, int | None, int | None, bool | None, int | None, bool | None]:
    mc_code = _COUNTER_FMT_TO_CODE.get(main_counter_format.strip().lower()) if main_counter_format is not None else None
    if main_counter_format is not None and mc_code is None:
        raise ValueError(f"Unsupported main_counter_format: {main_counter_format!r}")
    sc_code = _COUNTER_FMT_TO_CODE.get(sub_counter_format.strip().lower()) if sub_counter_format is not None else None
    if sub_counter_format is not None and sc_code is None:
        raise ValueError(f"Unsupported sub_counter_format: {sub_counter_format!r}")
    em_code, grid_rel = _resolve_edit_mode_and_grid(edit_mode, grid_relative)
    tool_code, smart_flag = _resolve_edit_tool_and_smart(edit_tool, smart_tool_enabled)
    for lbl, val in (
        ("pre_roll_samples", pre_roll_samples),
        ("post_roll_samples", post_roll_samples),
        ("timeline_in_sample", timeline_in_sample),
        ("timeline_out_sample", timeline_out_sample),
    ):
        if val is not None and (isinstance(val, bool) or not (0 <= val <= TICK_MASK_56)):
            raise ValueError(f"{lbl} out of 56-bit range: {val!r}")
    if timeline_in_sample is not None and timeline_out_sample is not None and timeline_out_sample < timeline_in_sample:
        raise ValueError("timeline_out_sample must be >= timeline_in_sample")
    return mc_code, sc_code, em_code, grid_rel, tool_code, smart_flag


def _patch_roll_and_selection(
    buf: bytearray,
    by_tag: dict[int, tuple[int, int]],
    *,
    pre_roll_enabled: bool | None,
    pre_roll_samples: int | None,
    post_roll_enabled: bool | None,
    post_roll_samples: int | None,
    timeline_in_sample: int | None,
    timeline_out_sample: int | None,
) -> None:
    if 0x2025 in by_tag and any(x is not None for x in (pre_roll_enabled, pre_roll_samples, post_roll_enabled, post_roll_samples)):
        off, ln = by_tag[0x2025]
        if ln >= 27:
            if pre_roll_enabled is not None:
                buf[off + 9] = 1 if pre_roll_enabled else 0
            if pre_roll_samples is not None:
                struct.pack_into("<Q", buf, off + 10, pre_roll_samples)
            if post_roll_enabled is not None:
                buf[off + 18] = 1 if post_roll_enabled else 0
            if post_roll_samples is not None:
                struct.pack_into("<Q", buf, off + 19, post_roll_samples)
    if 0x2016 in by_tag and (timeline_in_sample is not None or timeline_out_sample is not None):
        off, ln = by_tag[0x2016]
        if ln >= 129:
            cur_in, cur_out = struct.unpack_from("<QQ", buf, off + 89)
            cur_is_tick = (cur_in >> 56) == 0x40 or (cur_out >> 56) == 0x40
            if cur_is_tick:
                in_val = timeline_in_sample if timeline_in_sample is not None else int(timeline_out_sample or 0)
                out_val = timeline_out_sample if timeline_out_sample is not None else in_val
            else:
                in_val = timeline_in_sample if timeline_in_sample is not None else cur_in
                out_val = timeline_out_sample if timeline_out_sample is not None else max(cur_out, in_val)
            if out_val < in_val:
                raise ValueError(f"timeline_out_sample ({out_val}) must be >= timeline_in_sample ({in_val})")
            struct.pack_into("<QQQQQ", buf, off + 89, in_val, out_val, in_val, out_val, in_val)


def patch_session_ui_state_decrypted(
    buf: bytearray,
    *,
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
) -> None:
    """Patch counters, edit mode/tool, pre/post roll, and timeline selection in-place."""
    mc_code, sc_code, em_code, grid_rel, tool_code, smart_flag = _validate_ui_state_args(
        main_counter_format, sub_counter_format, edit_mode, grid_relative,
        edit_tool, smart_tool_enabled, pre_roll_samples, post_roll_samples,
        timeline_in_sample, timeline_out_sample,
    )
    by_tag = {c.tag: (c.offset, len(c.raw)) for c in iter_top_chunks(bytes(buf), strict=True)}
    if 0x2024 in by_tag and (em_code is not None or mc_code is not None):
        off, ln = by_tag[0x2024]
        if em_code is not None and ln >= 14:
            buf[off + 9], buf[off + 13] = em_code, EDIT_MODE_MASK_MAP[em_code]
        if mc_code is not None and ln >= 11:
            buf[off + 10] = mc_code
    if any(x is not None for x in (mc_code, em_code, tool_code, smart_flag)):
        if 0x2056 in by_tag:
            _patch_2056_block(buf, *by_tag[0x2056], mc_code=mc_code, em_code=em_code, tool_code=tool_code, smart_flag=smart_flag)
        for parent_tag in (0x2016, 0x2587):
            if parent_tag in by_tag:
                poff, plen = by_tag[parent_tag]
                p56 = find_zchunk_offset(buf, 0x2056, poff + 9, poff + plen)
                if p56 >= 0:
                    l56 = 7 + struct.unpack_from("<I", buf, p56 + 3)[0]
                    _patch_2056_block(buf, p56, l56, mc_code=mc_code, em_code=em_code, tool_code=tool_code, smart_flag=smart_flag)
    if sc_code is not None:
        if 0x201F in by_tag and by_tag[0x201F][1] > 41:
            buf[by_tag[0x201F][0] + 41] = sc_code
        if 0x2016 in by_tag and by_tag[0x2016][1] > 134:
            buf[by_tag[0x2016][0] + 134] = sc_code
    if grid_rel is not None:
        _patch_grid_relative_chunks(buf, by_tag, grid_rel)
    _patch_roll_and_selection(
        buf, by_tag,
        pre_roll_enabled=pre_roll_enabled, pre_roll_samples=pre_roll_samples,
        post_roll_enabled=post_roll_enabled, post_roll_samples=post_roll_samples,
        timeline_in_sample=timeline_in_sample, timeline_out_sample=timeline_out_sample,
    )


def _validate_marker_args(
    number: int,
    start_pos: int,
    end_pos: int | None,
    reference: str,
    time_properties: str,
    color_index: int,
    color_rgb: tuple[int, int, int] | None,
    track_index: int,
    marker_uuid16: bytes | None,
) -> tuple[int, int, int, bytes, bytes]:
    if isinstance(number, bool) or not (1 <= number <= 999):
        raise ValueError(f"Marker number must be 1..999, got {number!r}")
    ref_code = _MARKER_REF_TO_CODE.get(reference.strip().lower())
    if ref_code is None:
        raise ValueError(f"Unsupported marker reference: {reference!r}")
    prop_code = _MARKER_PROP_TO_CODE.get(time_properties.strip().lower())
    if prop_code is None:
        raise ValueError(f"Unsupported marker time_properties: {time_properties!r}")
    end_val = start_pos if end_pos is None else end_pos
    for lbl, pos_val in (("start_pos", start_pos), ("end_pos", end_val)):
        if isinstance(pos_val, bool) or not (0 <= pos_val <= TICK_MASK_56):
            raise ValueError(f"Marker {lbl} out of 56-bit range: {pos_val!r}")
    if prop_code == 2 and end_val < start_pos:
        raise ValueError(f"Selection marker end_pos ({end_val}) must be >= start_pos ({start_pos})")
    if isinstance(color_index, bool) or not (0 <= color_index <= 0xFFFF):
        raise ValueError(f"Marker color_index out of u16 range: {color_index!r}")
    if isinstance(track_index, bool) or track_index < -1 or track_index > 0x7FFF_FFFF:
        raise ValueError(f"Marker track_index out of range: {track_index!r}")
    uid = uuid.uuid4().bytes_le if marker_uuid16 is None else marker_uuid16
    if len(uid) != 16:
        raise ValueError(f"marker_uuid16 must be 16 bytes, got {len(uid)}")
    rgb_body = b"\x00" * 7
    if color_rgb is not None:
        if len(color_rgb) != 3 or any(isinstance(c, bool) or not (0 <= c <= 255) for c in color_rgb):
            raise ValueError(f"color_rgb must be 3 ints in 0..255, got {color_rgb!r}")
        r, g, b = color_rgb
        rgb_body = bytes([1, 0, r, 0, g, 0, b])
    return ref_code, prop_code, end_val, uid, rgb_body


def serialize_marker_2077(
    number: int,
    name: str,
    start_pos: int,
    *,
    end_pos: int | None = None,
    reference: str = "Absolute",
    time_properties: str = "Marker",
    comments: str = "",
    color_index: int = 0,
    color_rgb: tuple[int, int, int] | None = None,
    track_index: int = -1,
    marker_uuid16: bytes | None = None,
    zoom_settings: bool = False,
    pre_post_roll_times: bool = False,
    track_visibility: bool = False,
    track_heights: bool = False,
    group_enables: bool = False,
    window_configuration: bool = False,
    version: int = 18,
) -> bytes:
    """Serialize a Pro Tools ``0x2030`` -> ``0x2077`` (v18) Memory Location / Marker Z-chunk."""
    ref_code, prop_code, end_val, uid, rgb_body = _validate_marker_args(
        number, start_pos, end_pos, reference, time_properties,
        color_index, color_rgb, track_index, marker_uuid16,
    )
    ref_byte = ref_code | sum(
        MARKER_GENERAL_PROP_BITS[k]
        for k, v in (
            ("zoom_settings", zoom_settings), ("pre_post_roll_times", pre_post_roll_times),
            ("track_visibility", track_visibility), ("track_heights", track_heights),
            ("group_enables", group_enables), ("window_configuration", window_configuration),
        ) if v
    )
    tick_flag = (0x40 << 56) if ref_code in (1, 2) else 0
    c2506 = build_zchunk(0x2506, b"\xff\xff\xff\xff\x00\x00\x00\x00", version=3)
    c4826_primary = build_zchunk(0x4826, rgb_body, version=1)
    c4826_secondary = build_zchunk(0x4826, b"\x00" * 7, version=1)
    c4827 = build_zchunk(0x4827, b"\x00" * 28, version=1)
    comment_lp = encode_lp_str(comments)
    loc_type = 2 if track_index >= 0 else 0
    body = (
        struct.pack("<HBBH", number, ref_byte, prop_code, 0)
        + encode_lp_str(name) + struct.pack("<QQ", start_pos | tick_flag, end_val | tick_flag)
        + _MARKER_VIEW_DEFAULT_68 + comment_lp + _MARKER_MID_FLAGS_40
        + struct.pack("<HI", color_index, 9) + (c2506 * 9) + b"\x00\x00\x00\x00" + comment_lp
        + (b"\x00" * 13) + b"\xff\xff" + uid + (b"\xff" * 8) + struct.pack("<i", track_index)
        + c4826_primary + c4826_secondary + c4827 + struct.pack("<II", loc_type, 0)
    )
    return build_zchunk(0x2077, body, version=version)


def serialize_directory_tree_103a(entries: Sequence[PtxDirTreeEntry], *, version: int = 1) -> bytes:
    """Serialize ``0x1004`` -> ``0x103a`` directory/file entries into a Z-chunk."""
    body = bytearray(struct.pack("<I", len(entries)))
    for entry in entries:
        if not (0 <= entry.kind <= 0xFF):
            raise ValueError(f"0x103a entry kind out of u8 range: {entry.kind}")
        if not (0 <= entry.parent_index <= 0xFFFF_FFFF):
            raise ValueError(f"0x103a parent_index out of u32 range: {entry.parent_index}")
        if len(entry.tag4) != 4:
            raise ValueError(f"0x103a entry tag4 must be 4 bytes, got {len(entry.tag4)}")
        body.append(entry.kind)
        body.extend(struct.pack("<I", entry.parent_index))
        body.extend(encode_lp_str(entry.name))
        body.extend(entry.tag4)
    return build_zchunk(0x103A, bytes(body), version=version)


def serialize_audio_desc_1001(
    sample_rate: int,
    channels: int,
    bit_depth: int,
    length_samples: int,
    *,
    format_code: int = 0x11,
    tail16: bytes = b"\x00" * 16,
    version: int = 2,
) -> bytes:
    """Serialize a ``0x1003`` -> ``0x1001`` audio file format descriptor Z-chunk."""
    if sample_rate <= 0 or sample_rate > 0xFFFF_FFFF:
        raise ValueError(f"Invalid sample_rate for 0x1001: {sample_rate}")
    if not (1 <= channels <= 64):
        raise ValueError(f"Invalid channel count for 0x1001: {channels}")
    if bit_depth not in _VALID_BIT_DEPTHS:
        raise ValueError(f"Invalid bit_depth for 0x1001: {bit_depth}")
    if length_samples < 0 or length_samples > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"Invalid length_samples for 0x1001: {length_samples}")
    if not (0 <= format_code <= 0xFF):
        raise ValueError(f"Invalid format_code for 0x1001: {format_code}")
    if len(tail16) != 16:
        raise ValueError(f"0x1001 tail16 must be 16 bytes, got {len(tail16)}")
    body = struct.pack("<IBBQB", sample_rate, channels, bit_depth, length_samples, format_code) + tail16
    return build_zchunk(0x1001, body, version=version)


def serialize_placement_104f(
    clip_index: int,
    start_sample: int,
    *,
    flags: int = 3,
    color_index: int = 0xFFFE,
    is_muted: bool = False,
    is_time_locked: bool = False,
    is_edit_locked: bool = False,
    version: int = 10,
) -> bytes:
    """Serialize a single 44-byte ``0x104f`` clip or fade placement Z-chunk."""
    if isinstance(clip_index, bool) or not (0 <= clip_index <= 0xFFFF_FFFF):
        raise ValueError(f"clip_index out of u32 range: {clip_index}")
    if isinstance(start_sample, bool) or not (0 <= start_sample <= 0x00FF_FFFF_FFFF_FFFF):
        raise ValueError(f"start_sample out of 56-bit range: {start_sample}")
    if isinstance(flags, bool) or not (0 <= flags <= 0xFF):
        raise ValueError(f"flags out of u8 range: {flags}")
    if isinstance(color_index, bool) or not (0 <= color_index <= 0xFFFF):
        raise ValueError(f"color_index out of u16 range: {color_index}")
    mute_b = 1 if is_muted else 0
    elock_b = 1 if is_edit_locked else 0
    tlock_b = 1 if (is_time_locked or is_edit_locked) else 0
    body = (
        bytes([mute_b, elock_b])
        + struct.pack("<I", clip_index)
        + b"\x00"
        + struct.pack("<Q", start_sample)
        + struct.pack("<BH", flags, color_index)
        + b"\x00\x00\x00\x00"
        + b"\xff" * 8
        + bytes([tlock_b, 0, 0, 0, 0])
    )
    return build_zchunk(0x104F, body, version=version)


def serialize_fade_262f(
    fade_type: str,
    total_samples: int,
    *,
    pre_samples: int | None = None,
    shape: str = "standard",
    crossfade_link: str = "equal_power",
    version: int = 2,
) -> bytes:
    """Serialize a ``0x2630`` -> ``0x262F`` (v2) fade-in, fade-out, or crossfade Z-chunk."""
    if isinstance(total_samples, bool) or total_samples <= 0 or total_samples > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"Fade total_samples must be positive u64, got {total_samples!r}")
    ftype = fade_type.strip().lower()
    if ftype == "in":
        b1 = _encode_min_le(total_samples, "total_samples")
        shape_code = _FADE_SHAPE_TO_CODE.get(shape.strip().lower())
        if shape_code is None:
            raise ValueError(f"Unsupported fade shape: {shape!r}")
        tail = bytes([3, 1, 0, 0, 0, 0, shape_code, 1, 0, 0, 0xFF, 0xFF])
        body = (b"\x00" * 5) + bytes([len(b1) << 4, 0, 0]) + b1 + tail
    elif ftype == "out":
        b1 = _encode_min_le(total_samples, "total_samples")
        shape_code = _FADE_SHAPE_TO_CODE.get(shape.strip().lower())
        if shape_code is None:
            raise ValueError(f"Unsupported fade shape: {shape!r}")
        hdr = (b"\x00" * 5) + bytes([(len(b1) << 4) | len(b1), 0, 0])
        out_shape = 2 if shape_code == 0 else shape_code
        body = hdr + b1 + b1 + bytes([2, 1]) + (b"\x00" * 10) + bytes([out_shape, 0xFF, 0xFF])
    elif ftype == "crossfade":
        pre = total_samples // 2 if pre_samples is None else pre_samples
        if isinstance(pre, bool) or pre < 0 or pre > total_samples:
            raise ValueError(f"Crossfade pre_samples ({pre}) must be in 0..{total_samples}")
        link_code = _CROSSFADE_LINK_TO_CODE.get(crossfade_link.strip().lower())
        if link_code is None:
            raise ValueError(f"Unsupported crossfade_link: {crossfade_link!r}")
        b_pre = _encode_min_le(pre, "pre_samples")
        b_tot = _encode_min_le(total_samples, "total_samples")
        hdr = (b"\x00" * 5) + bytes([(len(b_pre) << 4) | len(b_tot), 0, 0])
        body = hdr + b_pre + b_tot + bytes([1, link_code]) + (b"\x00" * 10) + b"\x02\xff\xff\x00\x00\x00\x00\x02\xff\xff"
    else:
        raise ValueError(f"Unsupported fade_type {fade_type!r} (expected 'in', 'out', or 'crossfade')")
    return build_zchunk(0x262F, body, version=version)


def serialize_lane_1052(
    lane_name: str,
    placements: Sequence[tuple[Any, ...]],
    *,
    wrap_1050: bool = True,
    wrap_version: int = 3,
    lane_version: int = 3,
    ticks_timebase: bool = False,
) -> bytes:
    """Serialize a ``0x1054`` -> ``0x1052`` channel lane with clip/fade placement tuples."""
    body = bytearray(encode_lp_str(lane_name))
    body.extend(struct.pack("<I", len(placements)))
    for item in placements:
        clip_idx, start_smp, flags, color_idx = int(item[0]), int(item[1]), int(item[2]), int(item[3])
        is_muted = bool(item[4]) if len(item) > 4 else False
        is_tlock = bool(item[5]) if len(item) > 5 else False
        is_elock = bool(item[6]) if len(item) > 6 else False
        is_fade = bool(item[7]) if len(item) > 7 else False
        c4f = serialize_placement_104f(
            clip_idx, start_smp, flags=flags, color_index=color_idx,
            is_muted=is_muted, is_time_locked=is_tlock, is_edit_locked=is_elock,
        )
        if wrap_1050:
            fade_b = 1 if is_fade else 0
            suffix = bytes([fade_b, 1, 1]) if wrap_version >= 3 else bytes([fade_b, 1])
            body.extend(build_zchunk(0x1050, c4f + suffix, version=wrap_version))
        else:
            body.extend(c4f)
    if lane_version >= 3:
        body.extend(b"\x02\x00" if ticks_timebase else b"\x01\x00")
    return build_zchunk(0x1052, bytes(body), version=lane_version)


def serialize_playlist_1014(
    playlist_name: str,
    lane_indices: Sequence[int],
    playlist_uuid16: bytes,
    *,
    flags: int | None = None,
    is_stereo: bool | None = None,
    format_code: int | None = None,
    version: int = 9,
) -> bytes:
    """Serialize a ``0x1015`` -> ``0x1014`` playlist entry Z-chunk (v9 Mono, Stereo, and Surround)."""
    if len(playlist_uuid16) != 16:
        raise ValueError(f"playlist_uuid16 must be 16 bytes, got {len(playlist_uuid16)}")
    for li in lane_indices:
        if isinstance(li, bool) or not (0 <= li <= 0xFFFF):
            raise ValueError(f"lane_index out of u16 range: {li}")
    ext_code = format_code
    if ext_code is None:
        stereo = bool(is_stereo) if is_stereo is not None else (bool(flags & 1) if flags is not None else len(lane_indices) >= 2)
        ext_code = 1 if stereo else 0
    if ext_code == 17 and len(lane_indices) != 10:
        raise ValueError(f"7.1.2 playlist requires 10 lane indices, got {len(lane_indices)}")
    if ext_code == 30 and len(lane_indices) != 12:
        raise ValueError(f"7.1.4 playlist requires 12 lane indices, got {len(lane_indices)}")
    if ext_code not in (17, 30) and len(lane_indices) > 8:
        raise ValueError(f"Playlist format_code {ext_code} supports at most 8 lanes, got {len(lane_indices)}")
    first_lanes = list(lane_indices[:8])
    extra_lanes = list(lane_indices[8:])
    if ext_code in (14, 17, 30):
        fbyte = 0x0A
    elif ext_code == 13:
        fbyte = 0x09
    else:
        fbyte = flags if flags is not None else ext_code
    if not (0 <= fbyte <= 0xFF):
        raise ValueError(f"playlist flags out of u8 range: {fbyte}")
    body = bytearray(encode_lp_str(playlist_name))
    body.append(fbyte)
    body.extend(struct.pack("<I", len(first_lanes)))
    if first_lanes:
        body.extend(struct.pack(f"<{len(first_lanes)}H", *first_lanes))
    body.extend(playlist_uuid16)
    if version >= 9 and lane_indices:
        if ext_code == 17:  # 7.1.2 (8 bed + 2 height)
            body.extend(struct.pack("<HBI", 0x0E, 0x11, len(extra_lanes)) + struct.pack(f"<{len(extra_lanes)}H", *extra_lanes))
            body.extend(struct.pack(f"<{len(lane_indices)}I", *lane_indices) + struct.pack("<IH", 0x11, 0))
        elif ext_code == 30:  # 7.1.4 (8 bed + 4 height)
            body.extend(struct.pack("<HBI", 0x0A, 0x0E, 0) + struct.pack(f"<{len(first_lanes)}I", *first_lanes))
            body.extend(struct.pack("<BI", 0x1E, len(extra_lanes)) + struct.pack(f"<{len(extra_lanes)}I", *extra_lanes) + b"\x00")
        else:
            body.extend(struct.pack("<HBI", ext_code, ext_code, 0) + struct.pack(f"<{len(lane_indices)}I", *lane_indices) + struct.pack("<IH", ext_code, 0))
    else:
        body.extend(b"\x00" * (17 if version >= 9 else 7))
    return build_zchunk(0x1014, bytes(body), version=version)


def serialize_clip_core_2628_dims(
    start_sample: int,
    length_samples: int,
    sync_point_samples: int = 0,
    *,
    orig_timestamp: int = 0,
    user_timestamp: int = 0,
) -> bytes:
    """Encode the 5-byte nibble header + minimal-width LE fields for a ``0x2628`` clip core."""
    b_start = _encode_min_le(start_sample, "start_sample")
    b_sync = _encode_min_le(sync_point_samples, "sync_point_samples")
    b_len = _encode_min_le(length_samples, "length_samples")
    b_orig = _encode_min_le(orig_timestamp, "orig_timestamp")
    b_user = _encode_min_le(user_timestamp, "user_timestamp")
    hdr = bytes([
        0x00,
        (len(b_start) << 4),
        (len(b_len) << 4) | len(b_sync),
        (len(b_orig) << 4) | len(b_user),
        0x00,
    ])
    return hdr + b_start + b_sync + b_len + b_orig + b_user


def closest_pt_color_index(argb_hex: str) -> int | None:
    """Map an ``#AARRGGBB`` or ``#RRGGBB`` color string to the closest ``PT12_COLOR_PALETTE`` index offline."""
    cleaned = argb_hex.strip().lstrip("#")
    if len(cleaned) == 8:
        cleaned = cleaned[2:]
    if len(cleaned) != 6 or not all(ch in _HEX_CHARS for ch in cleaned):
        return None
    r0 = int(cleaned[0:2], 16)
    g0 = int(cleaned[2:4], 16)
    b0 = int(cleaned[4:6], 16)
    best_idx: int | None = None
    best_dist: int | None = None
    for idx, (r1, g1, b1) in PT12_COLOR_PALETTE.items():
        # Perceptually weighted squared RGB distance (fast, stdlib-only)
        dr, dg, db = r0 - r1, g0 - g1, b0 - b1
        dist = 2 * dr * dr + 4 * dg * dg + 3 * db * db
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
