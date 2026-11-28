"""Verified Pro Tools ``.ptx`` chunk serializers & offline palette mapper for offline writer prep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import struct

from .codec import decrypt_ptx, encrypt_ptx, iter_top_chunks
from .models import (
    PT12_COLOR_PALETTE,
    VALID_PROTOOLS_BIT_DEPTHS,
    VALID_PROTOOLS_SAMPLE_RATES,
)
from .writer_primitives import build_zchunk, encode_lp_str

_VALID_AUDIO_RATES = VALID_PROTOOLS_SAMPLE_RATES
_VALID_BIT_DEPTHS = VALID_PROTOOLS_BIT_DEPTHS
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


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


def patch_session_audio_format_1028_decrypted(
    buf: bytearray,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format_code: int | None = None,
) -> None:
    """Patch ``0x1028`` session sample rate, bit depth, and/or format code in-place on decrypted bytes."""
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate not in _VALID_AUDIO_RATES):
        raise ValueError(f"Unsupported Pro Tools sample rate: {sample_rate}")
    if bit_depth is not None and (isinstance(bit_depth, bool) or bit_depth not in _VALID_BIT_DEPTHS):
        raise ValueError(f"Unsupported Pro Tools bit depth: {bit_depth}")
    if audio_format_code is not None and (isinstance(audio_format_code, bool) or not (0 <= audio_format_code <= 0xFF)):
        raise ValueError(f"audio_format_code out of u8 range: {audio_format_code}")

    chunks = iter_top_chunks(bytes(buf), strict=True)
    c1028 = next((c for c in reversed(chunks) if c.tag == 0x1028), None)
    if c1028 is None or len(c1028.raw) < 15:
        raise RuntimeError("Missing or truncated 0x1028 session audio format chunk in .ptx")

    base = c1028.offset
    if audio_format_code is not None:
        buf[base + 9] = audio_format_code
    if bit_depth is not None:
        buf[base + 10] = bit_depth
    if sample_rate is not None:
        struct.pack_into("<I", buf, base + 11, sample_rate)


def patch_session_audio_format_1028(
    ptx_raw: bytes,
    *,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    audio_format_code: int | None = None,
) -> bytes:
    """Patch ``0x1028`` session sample rate, bit depth, and/or audio format code in place."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    patch_session_audio_format_1028_decrypted(
        buf, sample_rate=sample_rate, bit_depth=bit_depth, audio_format_code=audio_format_code,
    )
    return encrypt_ptx(bytes(buf), delta, k0=k0)


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
    version: int = 10,
) -> bytes:
    """Serialize a single 44-byte ``0x104f`` clip placement Z-chunk."""
    if not (0 <= clip_index <= 0xFFFF_FFFF):
        raise ValueError(f"clip_index out of u32 range: {clip_index}")
    if not (0 <= start_sample <= 0x00FF_FFFF_FFFF_FFFF):
        raise ValueError(f"start_sample out of 56-bit range: {start_sample}")
    if not (0 <= flags <= 0xFF):
        raise ValueError(f"flags out of u8 range: {flags}")
    if not (0 <= color_index <= 0xFFFF):
        raise ValueError(f"color_index out of u16 range: {color_index}")
    body = (
        b"\x00\x00"
        + struct.pack("<I", clip_index)
        + b"\x00"
        + struct.pack("<Q", start_sample)
        + struct.pack("<BH", flags, color_index)
        + b"\x00\x00\x00\x00"
        + b"\xff" * 8
        + b"\x00" * 5
    )
    return build_zchunk(0x104F, body, version=version)


def serialize_lane_1052(
    lane_name: str,
    placements: Sequence[tuple[int, int, int, int]],
    *,
    wrap_1050: bool = True,
    wrap_version: int = 3,
    lane_version: int = 3,
) -> bytes:
    """Serialize a ``0x1054`` -> ``0x1052`` channel lane with ``(clip_idx, start_smp, flags, color)`` events."""
    body = bytearray(encode_lp_str(lane_name))
    body.extend(struct.pack("<I", len(placements)))
    for clip_idx, start_smp, flags, color_idx in placements:
        c4f = serialize_placement_104f(clip_idx, start_smp, flags=flags, color_index=color_idx)
        if wrap_1050:
            suffix = b"\x00\x01\x01" if wrap_version >= 3 else b"\x00\x01"
            body.extend(build_zchunk(0x1050, c4f + suffix, version=wrap_version))
        else:
            body.extend(c4f)
    if lane_version >= 3:
        body.extend(b"\x01\x00")
    return build_zchunk(0x1052, bytes(body), version=lane_version)


def serialize_playlist_1014(
    playlist_name: str,
    lane_indices: Sequence[int],
    playlist_uuid16: bytes,
    *,
    flags: int | None = None,
    is_stereo: bool | None = None,
    version: int = 9,
) -> bytes:
    """Serialize a ``0x1015`` -> ``0x1014`` playlist entry Z-chunk (exact v9 stereo/mono layout)."""
    if len(playlist_uuid16) != 16:
        raise ValueError(f"playlist_uuid16 must be 16 bytes, got {len(playlist_uuid16)}")
    stereo = bool(is_stereo) if is_stereo is not None else (bool(flags & 1) if flags is not None else len(lane_indices) >= 2)
    flag_byte = flags if flags is not None else (1 if stereo else 0)
    if not (0 <= flag_byte <= 0xFF):
        raise ValueError(f"playlist flags out of u8 range: {flag_byte}")
    for li in lane_indices:
        if isinstance(li, bool) or not (0 <= li <= 0xFFFF):
            raise ValueError(f"lane_index out of u16 range: {li}")
    body = bytearray(encode_lp_str(playlist_name))
    body.append(flag_byte)
    body.extend(struct.pack("<I", len(lane_indices)))
    if lane_indices:
        body.extend(struct.pack(f"<{len(lane_indices)}H", *lane_indices))
    body.extend(playlist_uuid16)
    if version >= 9 and lane_indices:
        if stereo:
            body.extend(struct.pack("<HH", 1, 1) + (b"\x00" * 3))
            body.extend(struct.pack(f"<{len(lane_indices)}I", *lane_indices))
            body.extend(struct.pack("<IH", 1, 0))
        else:
            body.extend(b"\x00" * 7)
            body.extend(struct.pack(f"<{len(lane_indices)}I", *lane_indices))
            body.extend(struct.pack("<IH", 0, 0))
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
