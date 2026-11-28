"""Structured Pro Tools ``.ptx`` chunk serializers for offline session writing (zero binary blobs)."""

from __future__ import annotations

from collections.abc import Sequence
import struct
import uuid

from .writer_chunks import (
    serialize_audio_desc_1001,
    serialize_clip_core_2628_dims,
)
from .writer_primitives import build_zchunk, encode_lp_str

_PAN_MONO: tuple[int, ...] = (0, 0, 100, 100, 100, 100, 100, 0, 0, 100, 100, 100, 100, 100, 0, 0)
_PAN_STEREO_MAIN: tuple[int, ...] = (-100, -100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0)
_PAN_STEREO_SEND: tuple[int, ...] = (-100, -100, 100, 100, 100, 100, 100, -100, -100, 100, 100, 100, 100, 100, 0, 0)

_AUTO_PAN_BANK_MONO: tuple[int, ...] = (0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 1, 0, 0, 0)
_AUTO_PAN_BANK_STEREO_L: tuple[int, ...] = (0xFF9C, 0xFF9C, 100, 100, 100, 100, 100, 100, 0, 0, 1, 0, 0, 0)
_AUTO_PAN_BANK_STEREO_R: tuple[int, ...] = (100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 1, 0, 0, 0)

_DEFAULT_TRACK_HEIGHT_PX = 0x61  # Pro Tools Medium track height (97 px)
_DEFAULT_COLOR_SENTINEL = bytes.fromhex("efffdfbf")
_DEFAULT_VOLUME_CURVE_RAW = 0x0000FA60  # Unity / default initial automation curve value


def _check_bytes_len(data: bytes, expected: int, name: str) -> None:
    if len(data) != expected:
        raise ValueError(f"{name} must be {expected} bytes, got {len(data)}")


def _check_uint(value: int, max_val: int, name: str) -> None:
    if isinstance(value, bool) or not (0 <= value <= max_val):
        raise ValueError(f"{name} out of range 0..{max_val}: {value!r}")


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


# ── Small UI state Z-chunks (used by 0x2589 and 0x200B) ──────────────


def _z_2037() -> bytes:
    return build_zchunk(0x2037, b"", version=1)


def _z_2038(flag: int = 0) -> bytes:
    return build_zchunk(0x2038, _z_2037() + bytes([flag]), version=1)


def _z_203b(val: int = 1) -> bytes:
    return build_zchunk(0x203B, _z_2037() + struct.pack("<I", val), version=1)


def _z_2580() -> bytes:
    gap = struct.pack("<HHIHI", 0xFFFF, 0, 0xFFFF, 0, 0xFFFF_FFFF)
    return build_zchunk(0x2580, b"\x00" + _z_203b(1) + gap + _z_203b(1) + b"\x00\xff\xff", version=5)


def build_audio_ui_entry_2589(*, is_visible: bool = True) -> bytes:
    """Build a canonical 283-byte ``0x2589`` (v11) per-track UI state entry for an Audio track."""
    body = (
        struct.pack("<HHH", 0x001B, 1 if is_visible else 0, 0x003D)
        + _z_203b(1)
        + b"\x00" + struct.pack("<IHHHH", 2, 0x3D09, 0, 0, 0x0017)
        + _z_203b(1)
        + b"\x00" + struct.pack("<H", 0x3D09) + b"\x00" + struct.pack("<d", 0.5) + (b"\x00" * 18) + struct.pack("<I", 1)
        + _z_2580()
        + struct.pack("<dIi", -1.0, 0, -1)
        + _z_2038(0)
        + _z_203b(1)
        + _z_203b(1)
        + _z_2038(0)
        + struct.pack("<HHHH", 0, 0, 0, 0x003D)
    )
    return build_zchunk(0x2589, body, version=11)


# ── Output bus and mixer strip (0x261C) structured builders ───────────


def build_output_bus_260e(bus_ptsl_index: int, bus_uuid16: bytes, bus_name: str) -> bytes:
    """Serialize a Pro Tools ``0x260E`` (v9) Main Output bus descriptor Z-chunk."""
    _check_bytes_len(bus_uuid16, 16, "bus_uuid16")
    _check_uint(bus_ptsl_index, 0xFFFE, "bus_ptsl_index")
    bus_id = bus_ptsl_index + 1
    body = (
        struct.pack("<H", bus_id)
        + b"\x01\x01"  # present=1, active=1
        + bus_uuid16
        + (b"\x00" * 16)
        + encode_lp_str(bus_name)
        + b"\x01\x01" + struct.pack("<q", -1) + b"\x00" + struct.pack("<i", -1) + b"\x00\x0c\x01\x00"
    )
    return build_zchunk(0x260E, body, version=9)


def _build_auto_curve_260a(value: int, *, ticks_timebase: bool = False) -> bytes:
    """Serialize a single-breakpoint ``0x260A`` (v1) automation parameter curve at time 0."""
    body = (
        struct.pack("<HHI", 0x4601, 1, 20)
        + bytes([1 if ticks_timebase else 0, 0])
        + struct.pack("<II", 1, 2)  # 1 breakpoint, curve type 2
        + (b"\x00" * 8)  # timestamp = 0
        + struct.pack("<I", value & 0xFFFF_FFFF)
    )
    return build_zchunk(0x260A, body, version=1)


def _build_pan_bank_260c(values: Sequence[int]) -> bytes:
    return build_zchunk(0x260C, b"".join(b"\x01" + _build_auto_curve_260a(v) for v in values), version=2)


def _build_mixer_dsp_1029(*, is_stereo: bool, is_send: bool, is_muted: bool = False) -> bytes:
    """Serialize ``0x1029`` (v14) mixer DSP initial + live state (volume, pan array, mute, divergence)."""
    mute_b = 1 if is_muted else 0
    pans = _PAN_MONO if not is_stereo else (_PAN_STEREO_SEND if is_send else _PAN_STEREO_MAIN)
    vol_and_pans = struct.pack("<i16i", -1440, *pans)  # 68 B: default vol + 16 surround/stereo pan slots
    body = (
        bytes([1, 0, 0, 0, 0, mute_b, 0, 0, 0])
        + vol_and_pans
        + b"\x00\x00"
        + struct.pack("<ii", 100, 100)
        + bytes([0, 0, 0, 0, mute_b, 0, 0, 0])
        + vol_and_pans
        + (b"\x00" * 8)
        + struct.pack("<II", 1, 1) + (b"\x00" * 40)
        + struct.pack("<II", 1, 1) + (b"\x00" * 32)
        + b"\x01\x01" + (b"\x00" * 10)
        + struct.pack("<ii", 50, 50)
    )
    return build_zchunk(0x1029, body, version=14)


def _build_main_channel_260d(*, is_stereo: bool, output_bus_260e: bytes | None) -> bytes:
    c1029 = _build_mixer_dsp_1029(is_stereo=is_stereo, is_send=False)
    out_block = (struct.pack("<H", 1) + output_bus_260e) if output_bus_260e else struct.pack("<H", 0)
    scalars = (
        b"\x01" + _build_auto_curve_260a(0)  # bypass/solo
        + b"\x01" + _build_auto_curve_260a(0)  # mute
        + b"\x01" + _build_auto_curve_260a(_DEFAULT_VOLUME_CURVE_RAW)  # volume
    )
    pan_l = _build_pan_bank_260c(_AUTO_PAN_BANK_STEREO_L if is_stereo else _AUTO_PAN_BANK_MONO)
    pan_r = _build_pan_bank_260c(_AUTO_PAN_BANK_STEREO_R if is_stereo else _AUTO_PAN_BANK_MONO)
    c485a = build_zchunk(0x485A, b"\x00\x00\x00\x00\x00", version=2)
    body = c1029 + out_block + scalars + pan_l + pan_r + b"\x01" + _build_auto_curve_260a(0) + (b"\x00" * 8) + c485a
    return build_zchunk(0x260D, body, version=6)


def _build_empty_send_260d(*, is_stereo: bool) -> bytes:
    c1029 = _build_mixer_dsp_1029(is_stereo=is_stereo, is_send=True)
    empty_260c = build_zchunk(0x260C, b"\x00" * 14, version=2)
    c485a = build_zchunk(0x485A, b"\x00\x00\x00\x00\x01", version=2)
    return build_zchunk(0x260D, c1029 + (b"\x00" * 5) + empty_260c + empty_260c + (b"\x00" * 9) + c485a, version=6)


def _build_track_identity_102d(track_name: str, track_uuid16: bytes) -> bytes:
    c4301 = build_zchunk(0x4301, b"\x00" * 32, version=1)
    b2619 = (
        encode_lp_str(track_name)
        + b"\x01\x00\x01"
        + track_uuid16
        + track_uuid16
        + (b"\x00" * 8) + b"\x01" + (b"\x00" * 36)
        + c4301
        + track_uuid16
        + b"\x00\x00"
    )
    c2619 = build_zchunk(0x2619, b2619, version=9)
    return build_zchunk(0x102D, c2619 + b"\x00\x01\x00\x00\x00\x01\x00", version=3)


def _build_empty_inserts_2627(slot_count: int = 11) -> bytes:
    empty_slot = build_zchunk(0x2625, build_zchunk(0x2626, b"", version=1), version=1)
    return build_zchunk(0x2627, struct.pack("<H", slot_count) + empty_slot * slot_count, version=1)


def _build_output_mirror_4856(output_bus_260e: bytes | None) -> bytes:
    prefix = (b"\x00" * 8) + struct.pack("<I", 1) + (b"\x00" * 8) + struct.pack("<Ii", 1, -1)
    if output_bus_260e is None:
        mid = struct.pack("<I", 1) + (b"\x00" * 5) + struct.pack("<IiI", 1, -1, 0)
        return build_zchunk(0x4856, prefix + struct.pack("<I", 0) + mid + b"\x01", version=4)
    c485e = build_zchunk(0x485E, b"\x02\x00\x02\x00\x00\x00\x00", version=2)
    mid = struct.pack("<HHI", 2, 2, 1) + (b"\x00" * 5) + struct.pack("<IiI", 1, -1, 1)
    return build_zchunk(0x4856, prefix + struct.pack("<I", 1) + output_bus_260e + mid + c485e + b"\x01", version=4)


def _build_track_view_200b(
    *,
    is_stereo: bool,
    color_index: int,
    view_mode_id: int = 4,
    track_ordinal_1based: int = 0,
) -> bytes:
    c2103 = build_zchunk(0x2103, b"", version=1)
    c2104 = build_zchunk(
        0x2104,
        c2103
        + _DEFAULT_COLOR_SENTINEL * 2 + struct.pack("<III", 50, 0, 0)
        + _DEFAULT_COLOR_SENTINEL + struct.pack("<I", 50),
        version=2,
    )
    c2434 = build_zchunk(0x2434, struct.pack("<I", view_mode_id) + b"\x00\x00\x00\x3c\x00", version=1)
    b2015 = (
        _z_2038(0)
        + bytes([1 if is_stereo else 0, 1]) + struct.pack("<IH", 0, 0x005C) + (b"\x00" * 5) + struct.pack("<d", 0.5) + b"\x01"
        + c2104
        + b"\x01" + struct.pack("<HH", color_index, 1)
        + struct.pack("<IBI", 2, 1, _DEFAULT_TRACK_HEIGHT_PX)
        + struct.pack("<IH", 2, 0x3D09) + (b"\x00" * 11) + struct.pack("<I", 7) + (b"\x00" * 4) + b"\x01"
        + c2434
        + struct.pack("<I", track_ordinal_1based)
        + _z_2038(0)
        + b"\x00" + struct.pack("<I", 1)
        + _z_2580()
        + struct.pack("<II", 0, 0)
    )
    c2015 = build_zchunk(0x2015, b2015, version=21)
    b200a = (
        c2015 + b"\x00\x00"
        + _z_2038(0)
        + b"\x13\x00\x00" + struct.pack("<H", 0x3D09) + b"\x00" + struct.pack("<Hi", 0x0017, -1)
        + _z_2038(0)
        + struct.pack("<I", 2)
        + _z_2038(0)
        + _z_2038(0)
    )
    return build_zchunk(0x200B, build_zchunk(0x200A, b200a, version=10) + (b"\x00" * 25), version=10)


def build_audio_mixer_strip_261c(
    track_name: str,
    track_uuid16: bytes,
    channels: int,
    output_bus: tuple[int, bytes, str] | None,
    *,
    track_ordinal_1based: int,
    id_base: int,
    playlist_index: int = 0,
    color_index: int = 0xFFFE,
    send_uuids16: Sequence[bytes] | None = None,
) -> bytes:
    """Synthesize a complete ``0x261C`` Audio track mixer strip cleanly from structured subchunks."""
    _check_bytes_len(track_uuid16, 16, "track_uuid16")
    if channels not in (1, 2):
        raise ValueError(f"Audio mixer strip channels must be 1 or 2, got {channels!r}")
    _check_uint(track_ordinal_1based, 0xFFFF_FFFF, "track_ordinal_1based")
    _check_uint(id_base, 0xFFFF_FFF5, "id_base")
    _check_uint(playlist_index, 0xFFFF, "playlist_index")
    _check_uint(color_index, 0xFFFF, "color_index")

    is_stereo = channels == 2
    out_260e = build_output_bus_260e(*output_bus) if output_bus is not None else None
    send_ids = tuple(send_uuids16) if send_uuids16 is not None else tuple(uuid.uuid4().bytes_le for _ in range(10))
    if len(send_ids) != 10 or any(len(u) != 16 for u in send_ids):
        raise ValueError("send_uuids16 must contain ten 16-byte UUIDs")

    send_260d = _build_empty_send_260d(is_stereo=is_stereo)
    sends_blob = bytearray(struct.pack("<I", 10))
    for slot_idx in range(10):
        if slot_idx > 0:
            sends_blob.extend(struct.pack("<H", slot_idx))
        sends_blob.extend(send_260d)

    ch_b = 1 if is_stereo else 0
    ctrl_and_sends = (
        b"\x01\x01" + struct.pack("<H", 10) + struct.pack("<10I", *(id_base + i for i in range(10)))
        + struct.pack("<II", 0, 8) + (b"\x00" * 28)
        + struct.pack("<I", 0xFFFF_FFFF) + b"\x00\xff\x7f"
        + struct.pack("<H", 10) + b"".join(send_ids)
        + bytes([0, ch_b, 0, 0, 1, 0, ch_b]) + (b"\x00" * 21)
    )
    b261b = (
        _build_track_identity_102d(track_name, track_uuid16)
        + bytes([ch_b, 0xFF, 0xFF, 1])  # width_flag, 0xFFFF, active=1
        + _build_empty_inserts_2627(11)
        + _build_main_channel_260d(is_stereo=is_stereo, output_bus_260e=out_260e)
        + bytes(sends_blob)
        + ctrl_and_sends
        + _build_output_mirror_4856(out_260e)
    )
    c261b = build_zchunk(0x261B, b261b, version=15)
    c200b = _build_track_view_200b(
        is_stereo=is_stereo, color_index=color_index, track_ordinal_1based=track_ordinal_1based,
    )
    mid_gap = struct.pack("<IHHHH", playlist_index, 1, playlist_index, 0, 0xFFFF)
    tail = (b"\x00" * 17) + struct.pack("<I", playlist_index) + b"\x01" + (b"\x00" * 14) + b"\x01"
    return build_zchunk(0x261C, c261b + mid_gap + c200b + tail, version=6)


# ── Track directory, playlists, audio files, clips, volumes ──────────


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
) -> bytes:
    """Build a Pro Tools v12+ ``0x1014`` (v9) playlist entry with ``<H`` + ``<I`` lane references."""
    _check_bytes_len(playlist_uuid16, 16, "playlist_uuid16")
    expected_lanes = 2 if is_stereo else 1
    if len(lane_indices) != expected_lanes:
        raise ValueError(f"Expected {expected_lanes} lane index(es) for {'stereo' if is_stereo else 'mono'} playlist, got {len(lane_indices)}")
    for li in lane_indices:
        _check_uint(li, 0xFFFF, "lane_index")
    body = bytearray(encode_lp_str(playlist_name))
    body.append(1 if is_stereo else 0)
    body.extend(struct.pack("<I", len(lane_indices)))
    body.extend(struct.pack(f"<{len(lane_indices)}H", *lane_indices))
    body.extend(playlist_uuid16)
    if is_stereo:
        body.extend(struct.pack("<HH", 1, 1) + (b"\x00" * 3))
        body.extend(struct.pack("<II", lane_indices[0], lane_indices[1]))
        body.extend(struct.pack("<IH", 1, 0))
    else:
        body.extend(b"\x00" * 7)
        body.extend(struct.pack("<I", lane_indices[0]))
        body.extend(struct.pack("<IH", 0, 0))
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
    tree_node_index: int,
) -> bytes:
    """Build a ``0x1004`` -> ``0x1003`` (v9) WAV audio file descriptor with BWF/UMID linkage."""
    _check_bytes_len(file_id8, 8, "file_id8")
    _check_bytes_len(smpte_umid32, 32, "smpte_umid32")
    _check_uint(filetime_u64, 0xFFFF_FFFF_FFFF_FFFF, "filetime_u64")
    _check_uint(tree_node_index, 0xFFFF_FFFF, "tree_node_index")
    tail16 = b"\x00\x00\x00\x00\x00\x00\x00\x2a" + file_id8
    c1001 = serialize_audio_desc_1001(
        sample_rate, channels, bit_depth, length_samples,
        format_code=0x03, tail16=tail16, version=2,
    )
    c1033 = build_zchunk(0x1033, b"", version=2)
    c4301 = build_zchunk(0x4301, b"\x00" * 32, version=1)
    guid16 = uuid.uuid4().bytes_le
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
    return build_zchunk(
        0x1003,
        struct.pack("<I", tree_node_index) + c1001 + c1033 + c2106 + b"\x00\x00\x00\x00",
        version=9,
    )


def _build_clip_mdchun_trailer_4403() -> bytes:
    """Build the default empty ``0x4403`` + ``MdChun`` (``MdNLB``/``MdES``/``MdTEL``) clip metadata header.

    ``MdChun``'s 32-bit size spans its child sub-records plus the 17-byte ``pt_uuid16 + 0x01`` tail that
    immediately follows this header inside ``0x2629``.
    """
    c1000 = build_zchunk(0x1000, b"\x00" * 18, version=0)
    c4403 = build_zchunk(0x4403, c1000 + b"\x00", version=1)
    md_nlb_body = (b"\x00" * 8) + struct.pack("<IIHH", 0x0B, 0, 3, 0x12) + (b"\x00" * 2)
    md_nlb = b"MdNLB" + struct.pack("<HI", 3, len(md_nlb_body)) + md_nlb_body
    md_tel = b"MdTEL" + struct.pack("<HI", 1, 4) + (b"\x00" * 4)
    md_es = b"MdES" + struct.pack("<HI", 1, len(md_tel)) + md_tel
    md_chun_body = md_nlb + (b"\x00" * 4) + md_es
    return c4403 + b"\x00\x00\x01\x00\x7f" + b"MdChun" + struct.pack("<HI", 1, len(md_chun_body) + 17) + md_chun_body


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
        + struct.pack("<IHHH", 0, 0xFFFF, 4, 4)
        + (b"\x00" * 21)
        + struct.pack("<IH", 0xFFFF_FFFF, 0)
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
    body = (
        c2628
        + after_2628
        + _build_clip_mdchun_trailer_4403()
        + pt_uuid16
        + b"\x01"
        + struct.pack("<II", audio_file_index, 0)
    )
    return build_zchunk(0x2629, body, version=11)


def build_volume_entry_0f3c(
    volume_serial_u32: int,
    volume_label: str,
    root_path: str,
) -> bytes:
    """Build one ``0x0F3C`` (v1) volume descriptor entry for ``0x0F3D``."""
    _check_uint(volume_serial_u32, 0xFFFF_FFFF, "volume_serial_u32")
    return build_zchunk(
        0x0F3C,
        encode_lp_str(volume_label) + struct.pack("<I", volume_serial_u32) + encode_lp_str(root_path),
        version=1,
    )


def build_volume_table_0f3d(
    volume_serial_u32: int,
    volume_label: str,
    root_path: str,
) -> bytes:
    """Build a top-level ``0x0F3D`` volume table containing one ``0x0F3C`` entry."""
    return build_zchunk(0x0F3D, struct.pack("<I", 1) + build_volume_entry_0f3c(volume_serial_u32, volume_label, root_path), version=1)
