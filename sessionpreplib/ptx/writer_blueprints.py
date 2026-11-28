"""Canonical Pro Tools ``.ptx`` chunk blueprints and serializers for offline session writing."""

from __future__ import annotations

import base64
from collections.abc import Sequence
import secrets
import struct
import zlib

from .writer_chunks import (
    serialize_audio_desc_1001,
    serialize_clip_core_2628_dims,
)
from .writer_primitives import build_zchunk, encode_lp_str

# Neutralized canonical Audio track 0x261C mixer strip blueprints (zero session/user/path data).
_STEREO_261C_B64 = (
    "eNqLYmPQl2JgkFGL4mdIl2BgkFaLYmZYw8DAoCsQxckwB8iQVOMGkvHxwSGuQa7+8fHxjAyMDEQCohRGMTIogdQ6M5AOGMGY8f9/"
    "RqApp4AcdaBrgUyQk1XVgAwmIENNbUiKsDFMZmNg4FWL4mOQBPpTUwAemgm//v+f8x+CU4B8UjACoItQx1TkeCE1HolQh6TICIoZ"
    "GYAp1QHI4FNDVYAdMAOxU2gwIzDRQAADmOABJiNg2CsAZbnUGN0YGUTgDmNCdwQV1SX8AmYAJgYjoBiPGjEa5vwnzmBi1aUwjCx1"
    "1I5AxgGyF1xykpJwRlpEjyYcytUhJTR2EOUB5nGBmGwMOYzk1U745MivnYgzlf61Eyz8BBhAGRUjWLEJYw92oOGjgU73QGcaDXT6"
    "BzrzaKDTP9BZRgOd/oHOOhro9A90ttFAp3+gs48GOv0DnWM00Okf6JyjgT4AnSNGLpIGjjkIyEMGBuu5GBrP1PzfkyDpsL1zR3bR"
    "xvRqjoYE9gm793sufKo43/Ujd4+Gy9YjnQkznLu1vs+sCDm7+8dkBzsv0a9ek9a/vJb3K/ioxztX+6kh1Z5TvC4kB/x7UZudFWB5"
    "4UW6777kwNopNQXrJK/rPOXreOi03r79Qyxn2O2wG98ikltPum2cnLzSdvqpe5n6vxkcvN+79glFzHQ2OLPCt4+hIKWt3WF+gWpR"
    "3PYzx0Fhj2tEPoqFYTKQCvPANh7PCPUliKbSACoTEDJiNZ+JgRNIxXkwQYYYGBGO+f8/iouhHcjmVgAyUoEMLoUoUYZ/QClRhShG"
    "Bh4gw0IBOipurgB1Xwyygx7YMwItUAeyWBShCpkV3/+/vx+EjZAUwviM//5DlDEmQgc9OG2RlLEjjZKABuVNVFggAjaw+Qp0R4H9"
    "EsUKDsQGVZAKfiDLGqECEhgQDAkW7GqgChhxWSQMdqo4zAR0aSbsurC5mORpFmQuAK5w/qM="
)

_MONO_261C_B64 = (
    "eNqLYmPQlWJgkFGL4mdIlWBgkFaLYmZYxcDAoCsQxckwC8iQVOMEkvHxvv5+/vHx8YwMjAxEAqIURjEyKIHUOjOQDhgh+P9/RqAp"
    "p4AcdTVukIHcQKaqGpDBBGSoqQ1JETaGyWwMDLxqUXwMkkA/agrAQzPh1///MHYKFkyMHKYIdUxFiRdS45GwOiRFRlDMyABMpw5A"
    "Bp8aqgLsgBmInUKDGRn/QwEDmOABJmtg2CsAZbnUGN0YGUTgDmNCdwQV1SX8AmYAJgYjoBiP2kA4IGWEqaN2+DEOkL3gknM04Ywm"
    "HFqqQ0po7CDKA8zjAjHZGHIYR2snwrUTLPwEGEAZFSNYsQljD3ag4aOBTvdAZxoNdPoHOvNooNM/0FlGA53+gc46Guj0D3S20UCn"
    "f6CzjwY6/QOdYzTQ6R/onKOBPgCdI0YukgaOOQjIQwYG67kYmMXC9oYt/+fTon1b3i737uR5YvnmE1/+9tmrdckp4sCOX31u2YtT"
    "dTo8F/Q9n8UV/lv/zbNbR3q7+/3m2PEr7mDPdD8zKcJdt1DBa0GV/055tyf/fb5/fJeacM5/YWjaf78Tilf8U6puzfkd67jbOeXd"
    "3uvrqj2nZkrkyhe4LkzObCq0lz7omLHx1U+lLa4diuv177QsWX1g08ZXyrNv+i95Fr/3q+WVBfiG2aNYGCYDqTAPbOPxjFBfgmgq"
    "DaAyASEjVvOZGEBTCHEeTJAhBkZIooEoieJiaAeyuBWAjFQgg0shSpThH1BKVCGKkYEHyLBQgI6KmytA/RCD7KAH9oxAC9SBLBZF"
    "qEJmxff/7+8HYSMkhTA+47//EGWMidBBD05bJGXsSKMkoEF5ExUWiIANbL4Cq6OiWMGB2KAKUsEPZFkjVEB8CsGQYMGuhgGRj7Fa"
    "JAx2qjjMBHRpJuy6sLmYkvzNCADDfMwo"
)

# Canonical 283-byte 0x2589 per-track UI state entry for a freshly created Audio track.
_UI_2589_B64 = (
    "eNqL4mYQYWRg6FSVZmBksGWIYmTgZ2BgsFYAMpiADHMFoCQDmMlpC2KJ41ICkQaCB/YMGACkIoqVwQFINajiMOH/fwgGkf//47AF"
    "ooCB4cN+JIU8QKYFQiEO83GKomuHAFsGAMQ7ICs="
)

# Fixed 0x4403 + MdChun clip-metadata trailer present in Pro Tools v12+ 0x2629 whole-file clips.
_CLIP_2629_MDCHUN_TRAILER = bytes.fromhex(
    "5a01001e00000003445a000014000000001000000000000000000000000000000000000000000001007f"
    "4d644368756e01004f0000004d644e4c4203001600000000000000000000000b00000000000000030012"
    "000000000000004d64455301000f0000004d6454454c01000400000000000000"
)

_COLOR_200B_MARKER = bytes.fromhex("ffdfbf3200000001")
_STEREO_261C_RAW = zlib.decompress(base64.b64decode(_STEREO_261C_B64))
_MONO_261C_RAW = zlib.decompress(base64.b64decode(_MONO_261C_B64))
AUDIO_UI_2589_RAW = zlib.decompress(base64.b64decode(_UI_2589_B64))


def _check_bytes_len(data: bytes, expected: int, name: str) -> None:
    if len(data) != expected:
        raise ValueError(f"{name} must be {expected} bytes, got {len(data)}")


def _check_uint(value: int, max_val: int, name: str) -> None:
    if isinstance(value, bool) or not (0 <= value <= max_val):
        raise ValueError(f"{name} out of range 0..{max_val}: {value!r}")


def _adjust_zchunk_size(buf: bytearray, chunk_offset: int, delta: int) -> None:
    if delta == 0:
        return
    old_sz = struct.unpack_from("<I", buf, chunk_offset + 3)[0]
    new_sz = old_sz + delta
    if new_sz < 2 or new_sz > 0xFFFF_FFFF:
        raise ValueError(f"Adjusted Z-chunk size out of range at 0x{chunk_offset:x}: {new_sz}")
    struct.pack_into("<I", buf, chunk_offset + 3, new_sz)


def build_output_bus_260e(bus_ptsl_index: int, bus_uuid16: bytes, bus_name: str) -> bytes:
    """Serialize a Pro Tools ``0x260E`` (v9) Main Output bus descriptor Z-chunk."""
    _check_bytes_len(bus_uuid16, 16, "bus_uuid16")
    bus_id = bus_ptsl_index + 1
    _check_uint(bus_id, 0xFFFF, "bus_id")
    body = (
        struct.pack("<H", bus_id)
        + b"\x01\x01"
        + bus_uuid16
        + (b"\x00" * 16)
        + encode_lp_str(bus_name)
        + bytes.fromhex("0101ffffffffffffffff00ffffffff000c0100")
    )
    return build_zchunk(0x260E, body, version=9)


def _find_zchunk_offset(buf: bytearray | bytes, tag: int, start: int = 0) -> int:
    needle = struct.pack("<H", tag)
    pos = start + 7
    while pos < len(buf):
        hit = buf.find(needle, pos)
        if hit == -1:
            return -1
        off = hit - 7
        if off >= start and buf[off] == 0x5A:
            ver = struct.unpack_from("<H", buf, off + 1)[0]
            sz = struct.unpack_from("<I", buf, off + 3)[0]
            if ver < 0x1000 and 2 <= sz and off + 7 + sz <= len(buf):
                return off
        pos = hit + 1
    return -1


def _patch_261c_output_bus(
    buf: bytearray,
    output_bus: tuple[int, bytes, str] | None,
) -> None:
    """Patch (or remove) both ``0x260E`` output bus chunks inside a ``0x261C`` Audio strip."""
    c261b_off = 9
    c260d0_off = _find_zchunk_offset(buf, 0x260D, c261b_off + 9)
    c4856_off = _find_zchunk_offset(buf, 0x4856, c261b_off + 9)
    e1_off = _find_zchunk_offset(buf, 0x260E, c260d0_off + 9)
    e2_off = _find_zchunk_offset(buf, 0x260E, c4856_off + 9)
    if min(c260d0_off, c4856_off, e1_off, e2_off) < 0:
        raise RuntimeError("Malformed 0x261C blueprint: missing 0x260D/0x4856/0x260E")

    repl = build_output_bus_260e(*output_bus) if output_bus is not None else b""
    e2_len = 7 + struct.unpack_from("<I", buf, e2_off + 3)[0]
    if output_bus is None:
        struct.pack_into("<I", buf, e2_off - 4, 0)
    buf[e2_off : e2_off + e2_len] = repl
    delta2 = len(repl) - e2_len
    for anc in (c4856_off, c261b_off, 0):
        _adjust_zchunk_size(buf, anc, delta2)

    e1_len = 7 + struct.unpack_from("<I", buf, e1_off + 3)[0]
    if output_bus is None:
        struct.pack_into("<H", buf, e1_off - 2, 0)
    buf[e1_off : e1_off + e1_len] = repl
    delta1 = len(repl) - e1_len
    for anc in (c260d0_off, c261b_off, 0):
        _adjust_zchunk_size(buf, anc, delta1)


def _patch_261c_identity(buf: bytearray, track_name: str, track_uuid16: bytes, placeholder: bytes) -> None:
    c261b_off, c102d_off, c2619_off = 9, 18, 27
    pat = struct.pack("<I", len(placeholder)) + placeholder
    name_pos = buf.find(pat, c2619_off, c2619_off + 64)
    if name_pos < 0:
        raise RuntimeError("Malformed 0x261C blueprint: missing track name placeholder")
    enc_name = encode_lp_str(track_name)
    buf[name_pos : name_pos + len(pat)] = enc_name
    delta = len(enc_name) - len(pat)
    for anc in (c2619_off, c102d_off, c261b_off, 0):
        _adjust_zchunk_size(buf, anc, delta)

    u1_off = name_pos + len(enc_name) + 3
    buf[u1_off : u1_off + 16] = track_uuid16
    buf[u1_off + 16 : u1_off + 32] = track_uuid16
    c4301_off = _find_zchunk_offset(buf, 0x4301, u1_off + 32)
    if c4301_off < 0:
        raise RuntimeError("Malformed 0x261C blueprint: missing 0x4301 inside 0x102D")
    c4301_end = c4301_off + 7 + struct.unpack_from("<I", buf, c4301_off + 3)[0]
    buf[c4301_end : c4301_end + 16] = track_uuid16


def build_audio_mixer_strip_261c(
    track_name: str,
    track_uuid16: bytes,
    channels: int,
    output_bus: tuple[int, bytes, str] | None,
    *,
    track_ordinal_1based: int,
    id_base: int,
    color_index: int = 0xFFFE,
) -> bytes:
    """Synthesize a complete ``0x261C`` Audio track mixer strip with No Input and target Output bus."""
    _check_bytes_len(track_uuid16, 16, "track_uuid16")
    if channels not in (1, 2):
        raise ValueError(f"Audio mixer strip channels must be 1 or 2, got {channels!r}")
    _check_uint(track_ordinal_1based, 0xFFFF_FFFF, "track_ordinal_1based")
    _check_uint(id_base, 0xFFFF_FFF5, "id_base")
    _check_uint(color_index, 0xFFFF, "color_index")

    is_stereo = channels == 2
    buf = bytearray(_STEREO_261C_RAW if is_stereo else _MONO_261C_RAW)
    placeholder = b"__STEREO___" if is_stereo else b"__MONO___"

    p_col = buf.find(_COLOR_200B_MARKER)
    c2434 = _find_zchunk_offset(buf, 0x2434)
    p485a = _find_zchunk_offset(buf, 0x485A, 0x1500)
    p10 = buf.find(b"\x01\x01\x0a\x00", p485a) if p485a >= 0 else -1
    if min(p_col, c2434, p10) < 0:
        raise RuntimeError("Malformed 0x261C blueprint: missing color/0x2434/control-id anchors")
    struct.pack_into("<H", buf, p_col + len(_COLOR_200B_MARKER), color_index)
    struct.pack_into("<I", buf, c2434 + 18, track_ordinal_1based)
    struct.pack_into("<10I", buf, p10 + 4, *(id_base + i for i in range(10)))

    _patch_261c_output_bus(buf, output_bus)
    _patch_261c_identity(buf, track_name, track_uuid16, placeholder)
    return bytes(buf)


def build_track_summary_2519(
    track_name: str,
    track_uuid16: bytes,
    channels: int,
    visible_order_1based: int,
) -> bytes:
    """Build one Audio track summary row for Table 1 of ``0x2519``."""
    _check_bytes_len(track_uuid16, 16, "track_uuid16")
    _check_uint(visible_order_1based, 0xFFFF, "visible_order_1based")
    is_stereo = 1 if channels >= 2 else 0
    return (
        struct.pack("<H", 0)
        + encode_lp_str(track_name)
        + bytes([is_stereo, 0])
        + track_uuid16
        + struct.pack("<HB", visible_order_1based, 0)
    )


def build_track_entry_251a(
    track_name: str,
    track_uuid16: bytes,
    channels: int,
    visible_order_1based: int,
    *,
    color_index: int = 0xFFFE,
    has_clips: bool = True,
) -> bytes:
    """Build one ``0x251A`` (v10) Audio track directory entry for Pass 1 & Pass 2 of ``0x2519``."""
    _check_bytes_len(track_uuid16, 16, "track_uuid16")
    _check_uint(visible_order_1based, 0xFFFF, "visible_order_1based")
    _check_uint(color_index, 0xFFFF, "color_index")
    is_stereo = 1 if channels >= 2 else 0
    c4420 = build_zchunk(0x4420, b"\x00\x00", version=1)
    body = (
        struct.pack("<H", 0)
        + encode_lp_str(track_name)
        + bytes([is_stereo, 0])
        + track_uuid16
        + struct.pack("<H", visible_order_1based)
        + (b"\x00" * 4)
        + track_uuid16
        + bytes([is_stereo, 0])
        + c4420
        + bytes([0, 0, is_stereo, 0, 1, 1, 1 if has_clips else 0, 0, 0])
        + struct.pack("<H", color_index)
    )
    return build_zchunk(0x251A, body, version=10)


def build_track_uuid_entry_210b(
    track_name: str,
    track_uuid16: bytes,
    *,
    type_code: int = 0,
) -> bytes:
    """Build one ``0x210B`` (v1) entry inside ``0x2107``."""
    _check_bytes_len(track_uuid16, 16, "track_uuid16")
    _check_uint(type_code, 0xFF, "type_code")
    body = bytes([0, 0, 0, type_code]) + encode_lp_str(track_name) + track_uuid16 + (b"\x00" * 16)
    return build_zchunk(0x210B, body, version=1)


def build_playlist_1014_v9(
    playlist_name: str,
    lane_indices: Sequence[int],
    playlist_uuid16: bytes,
    *,
    is_stereo: bool,
    playlist_ordinal_1based: int,
) -> bytes:
    """Build a Pro Tools v12+ ``0x1014`` (v9) playlist entry matching live PT layout."""
    _check_bytes_len(playlist_uuid16, 16, "playlist_uuid16")
    _check_uint(playlist_ordinal_1based, 0xFFFF_FFFF, "playlist_ordinal_1based")
    for li in lane_indices:
        _check_uint(li, 0xFFFF, "lane_index")
    body = bytearray(encode_lp_str(playlist_name))
    body.append(1 if is_stereo else 0)
    body.extend(struct.pack("<I", len(lane_indices)))
    if lane_indices:
        body.extend(struct.pack(f"<{len(lane_indices)}H", *lane_indices))
    body.extend(playlist_uuid16)
    if is_stereo:
        body.extend(b"\x01\x00\x01\x00")
        body.extend(b"\x00" * 7)
        body.extend(struct.pack("<II", playlist_ordinal_1based, 1))
        body.extend(b"\x00\x00")
    else:
        body.extend(b"\x00" * 7)
        body.extend(struct.pack("<I", playlist_ordinal_1based))
        body.extend(b"\x00" * 6)
    return build_zchunk(0x1014, bytes(body), version=9)


def build_audio_file_entry_1003(
    sample_rate: int,
    channels: int,
    bit_depth: int,
    length_samples: int,
    *,
    file_id8: bytes,
    smpte_umid32: bytes,
    filetime_u64: int,
    file_index: int,
) -> bytes:
    """Build a ``0x1004`` -> ``0x1003`` (v9) WAV audio file descriptor with BWF/UMID linkage."""
    _check_bytes_len(file_id8, 8, "file_id8")
    _check_bytes_len(smpte_umid32, 32, "smpte_umid32")
    _check_uint(filetime_u64, 0xFFFF_FFFF_FFFF_FFFF, "filetime_u64")
    _check_uint(file_index, 0xFFFF_FFFE, "file_index")
    tail16 = b"\x00\x00\x00\x00\x00\x00\x00\x2a" + file_id8
    c1001 = serialize_audio_desc_1001(
        sample_rate, channels, bit_depth, length_samples,
        format_code=0x03, tail16=tail16, version=2,
    )
    c1033 = build_zchunk(0x1033, b"", version=2)
    c4301 = build_zchunk(0x4301, b"\x00" * 32, version=1)
    guid16 = secrets.token_bytes(16)
    ch_flag = 1 if channels >= 2 else 0
    b2106 = (
        (b"\x00" * 29)
        + struct.pack("<Q", filetime_u64)
        + b"\x01\x00\x2c\x00"
        + (b"\x00" * 51)
        + struct.pack("<Q", filetime_u64)
        + b"\x00\x00\x00"
        + struct.pack("<I", ch_flag)
        + (b"\x00" * 15)
        + guid16
        + c4301
        + (b"\x00" * 17)
        + bytes([ch_flag])
        + smpte_umid32
        + (b"\x00" * 7)
        + struct.pack("<I", ch_flag)
        + (b"\x00" * 4)
    )
    c2106 = build_zchunk(0x2106, b2106, version=38)
    tree_node_index = file_index + 1  # 0x103A entry 0 is "Audio Files", entries 1..N are audio files
    return build_zchunk(
        0x1003,
        struct.pack("<I", tree_node_index) + c1001 + c1033 + c2106 + b"\x00\x00\x00\x00",
        version=9,
    )


def build_whole_file_clip_2629(
    clip_name: str,
    length_samples: int,
    *,
    audio_file_index: int,
    channel_index: int,
    clip_guid16: bytes,
    pt_uuid16: bytes,
    color_index: int = 0xFFFE,
) -> bytes:
    """Build a ``0x262A`` -> ``0x2629`` (v11) whole-file clip descriptor."""
    _check_bytes_len(clip_guid16, 16, "clip_guid16")
    _check_bytes_len(pt_uuid16, 16, "pt_uuid16")
    _check_uint(audio_file_index, 0xFFFF_FFFF, "audio_file_index")
    _check_uint(channel_index, 0xFFFF, "channel_index")
    _check_uint(color_index, 0xFFFF, "color_index")
    b2628 = (
        encode_lp_str(clip_name)
        + serialize_clip_core_2628_dims(0, length_samples)
        + struct.pack("<H", color_index)
        + bytes.fromhex("00000000ffff04000400")
        + (b"\x00" * 21)
        + bytes.fromhex("ffffffff0000")
        + clip_guid16
    )
    c2628 = build_zchunk(0x2628, b2628, version=5)
    after_2628 = (
        struct.pack("<IHff", audio_file_index, channel_index, 1.0, 1.0)
        + (b"\x00" * 8)
        + b"\xff"
        + clip_guid16
        + (b"\x00" * 9)
    )
    body = c2628 + after_2628 + _CLIP_2629_MDCHUN_TRAILER + pt_uuid16 + b"\x01" + struct.pack("<II", audio_file_index, 0)
    return build_zchunk(0x2629, body, version=11)


def build_volume_table_0f3d(
    volume_serial_u32: int,
    volume_label: str,
    root_path: str,
) -> bytes:
    """Build a top-level ``0x0F3D`` volume table containing one ``0x0F3C`` entry."""
    _check_uint(volume_serial_u32, 0xFFFF_FFFF, "volume_serial_u32")
    c0f3c = build_zchunk(
        0x0F3C,
        encode_lp_str(volume_label) + struct.pack("<I", volume_serial_u32) + encode_lp_str(root_path),
        version=1,
    )
    return build_zchunk(0x0F3D, struct.pack("<I", 1) + c0f3c, version=1)
