"""Per-chunk binary mutators for adding/removing tracks, audio, clips, and placements in ``.ptx``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import struct

from .codec import find_zchunk_offset, read_zchunk_at, walk_sequential_zchunks
from .parser import split_202b
from .writer_blueprints import (
    OutputBusSpec,
    build_audio_file_entry_1003,
    build_audio_mixer_strip_261c,
    build_audio_ui_entry_2589,
    build_playlist_1014_v9,
    build_track_entry_251a,
    build_track_summary_2519,
    build_track_uuid_entry_210b,
    build_volume_entry_0f3c,
    build_whole_file_clip_2629,
)
from .writer_chunks import (
    PtxDirTreeEntry,
    serialize_directory_tree_103a,
    serialize_lane_1052,
)
from .writer_primitives import build_zchunk, serialize_folder_tree_210c

_read_zchunk_at = read_zchunk_at
_walk_sequential_zchunks = walk_sequential_zchunks


@dataclass(frozen=True)
class NewAudioTrackSpec:
    """Internal spec for one newly synthesized Audio track during ``.ptx`` writing."""

    track_name: str
    track_uuid16: bytes
    playlist_uuid16: bytes
    channels: int
    parent_folder_uuid16: bytes | None
    output_bus: OutputBusSpec | tuple[int, bytes, str] | None
    color_index: int
    clip_indices: tuple[int, ...]  # () for empty track, 1 (Mono) or 2 (Stereo L, R) in 0x262A
    lane_indices: tuple[int, ...]  # 1 (Mono) or 2 (Stereo L, R) in 0x1054
    playlist_index: int = 0  # 0-based index into 0x1015 Audio Playlists
    start_sample: int = 0


@dataclass(frozen=True)
class MediaSurvivalPlan:
    """Compacted existing playlists, lanes, and playlist-index mapping for surviving template audio."""

    playlists: tuple[bytes, ...]
    lanes: tuple[bytes, ...]
    old_playlist_to_new: Mapping[int, int]
    c4829_trailer: bytes
    c4828_trailer: bytes


@dataclass(frozen=True)
class TrackSlotOp:
    """Unified output track slot: either keep existing template track *old_index* or insert *new_track*."""

    old_index: int | None = None
    new_track: NewAudioTrackSpec | None = None

    def __post_init__(self) -> None:
        if (self.old_index is None) == (self.new_track is None):
            raise ValueError("TrackSlotOp must set exactly one of old_index or new_track")


@dataclass(frozen=True)
class AudioFileWriteSpec:
    """Staged WAV file info needed to populate ``0x1004``."""

    filename: str
    sample_rate: int
    channels: int
    bit_depth: int
    length_samples: int
    file_id8: bytes
    smpte_umid32: bytes
    filetime_u64: int


@dataclass(frozen=True)
class ClipWriteSpec:
    """Whole-file clip info needed to populate ``0x262A``."""

    clip_name: str
    length_samples: int
    audio_file_index: int
    channel_index: int
    clip_guid16: bytes
    pt_uuid16: bytes


def _parse_2519_sections(raw: bytes) -> tuple[bytes, list[bytes], list[bytes], list[bytes], bytes]:
    """Deterministically parse ``0x2519`` into ``(clean_prefix, rows, pass1, pass2, tail)``."""
    if len(raw) < 27 or raw[0] != 0x5A:
        raise RuntimeError("Truncated 0x2519 track directory chunk")
    pos = 9
    hdr10 = raw[pos : pos + 10]
    pos += 10
    cnt_a = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    if cnt_a > (len(raw) - pos) // 2:
        raise RuntimeError(f"0x2519 header cnt_a ({cnt_a}) exceeds chunk bounds")
    pos += cnt_a * 2
    # Clear pre-existing track-selection indices so Pro Tools does not reference shifted/deleted track slots.
    clean_prefix = hdr10 + b"\x00\x00\x00\x00"
    if pos + 4 > len(raw):
        raise RuntimeError("Truncated 0x2519 Table 1 track count")
    tcnt = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    rows: list[bytes] = []
    for _ in range(tcnt):
        if pos + 6 > len(raw):
            raise RuntimeError("Truncated 0x2519 Table 1 track row")
        row_start = pos
        pos += 2
        nlen = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if nlen + 21 > len(raw) - pos:
            raise RuntimeError("Truncated 0x2519 Table 1 track name/UUID")
        pos += nlen + 21
        rows.append(raw[row_start:pos])
    pass1, pos = walk_sequential_zchunks(raw, pos, tcnt, expected_tag=0x251A)
    if pos + 4 > len(raw):
        raise RuntimeError("Truncated 0x2519 Pass 2 track count")
    tcnt2 = struct.unpack_from("<I", raw, pos)[0]
    if tcnt2 != tcnt:
        raise RuntimeError(f"0x2519 Pass 2 count {tcnt2} != Pass 1 count {tcnt}")
    pos += 4
    pass2, pos = walk_sequential_zchunks(raw, pos, tcnt2, expected_tag=0x251A)
    return clean_prefix, rows, pass1, pass2, raw[pos:]


def _patch_2519_summary_order(row: bytes, visible_order: int) -> bytes:
    if not (0 <= visible_order <= 0xFFFF):
        raise OverflowError(f"0x2519 visible_order exceeds u16 range: {visible_order}")
    buf = bytearray(row)
    struct.pack_into("<H", buf, len(buf) - 3, visible_order)
    return bytes(buf)


def _patch_251a_order(raw_251a: bytes, visible_order: int) -> bytes:
    if not (0 <= visible_order <= 0xFFFF):
        raise OverflowError(f"0x251A visible_order exceeds u16 range: {visible_order}")
    buf = bytearray(raw_251a)
    if len(buf) < 15:
        raise RuntimeError("Truncated 0x251A track entry")
    nlen = struct.unpack_from("<I", buf, 11)[0]
    ord_off = 15 + nlen + 2 + 16
    if ord_off + 2 > len(buf):
        raise RuntimeError("Truncated 0x251A visible-order field")
    struct.pack_into("<H", buf, ord_off, visible_order)
    return bytes(buf)


def mutate_2519_tracks(
    c2519_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
) -> bytes:
    """Rebuild ``0x2519`` (Table 1 + Pass 1 & Pass 2 ``0x251A``), preserving hidden tracks' ``visible_order == 0``."""
    version = struct.unpack_from("<H", c2519_raw, 1)[0]
    prefix, old_rows, old_p1, old_p2, tail_raw = _parse_2519_sections(c2519_raw)
    if expected_old_count is not None and len(old_rows) != expected_old_count:
        raise RuntimeError(f"0x2519 track count {len(old_rows)} != expected {expected_old_count}")

    new_rows: list[bytes] = []
    new_p1: list[bytes] = []
    new_p2: list[bytes] = []
    next_visible_ord = 1
    for slot in layout:
        if slot.old_index is not None:
            old_row = old_rows[slot.old_index]
            old_p1_item = old_p1[slot.old_index]
            old_vis = struct.unpack_from("<H", old_row, len(old_row) - 3)[0]
            p1_ver = struct.unpack_from("<H", old_p1_item, 1)[0]
            is_hidden = (old_vis == 0) or (p1_ver >= 8 and len(old_p1_item) >= 40 and not bool(old_p1_item[-7]))
            vis_ord = 0 if is_hidden else next_visible_ord
            if vis_ord > 0:
                next_visible_ord += 1
            new_rows.append(_patch_2519_summary_order(old_row, vis_ord))
            new_p1.append(_patch_251a_order(old_p1_item, vis_ord))
            new_p2.append(_patch_251a_order(old_p2[slot.old_index], vis_ord))
        elif slot.new_track is not None:
            nt = slot.new_track
            vis_ord = next_visible_ord
            next_visible_ord += 1
            new_rows.append(build_track_summary_2519(nt.track_name, nt.track_uuid16, nt.channels, vis_ord))
            entry = build_track_entry_251a(
                nt.track_name, nt.track_uuid16, nt.channels, vis_ord,
                color_index=nt.color_index, has_clips=bool(nt.clip_indices),
            )
            new_p1.append(entry)
            new_p2.append(entry)

    count_le = struct.pack("<I", len(layout))
    body = prefix + count_le + b"".join(new_rows) + b"".join(new_p1) + count_le + b"".join(new_p2) + tail_raw
    return build_zchunk(0x2519, body, version=version)


def mutate_2107_tracks(
    c2107_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
) -> bytes:
    """Rebuild ``0x2107`` (track UUID + name registry) deterministically for *layout*."""
    if len(c2107_raw) < 22:
        raise RuntimeError("Truncated 0x2107 track UUID registry chunk")
    version = struct.unpack_from("<H", c2107_raw, 1)[0]
    hdr_prefix = c2107_raw[9:18]
    old_count = struct.unpack_from("<I", c2107_raw, 18)[0]
    old_items, end_pos = _walk_sequential_zchunks(c2107_raw, 22, old_count, expected_tag=0x210B)
    if expected_old_count is not None and len(old_items) != expected_old_count:
        raise RuntimeError(f"0x2107 track count {len(old_items)} != expected {expected_old_count}")
    new_items: list[bytes] = []
    for slot in layout:
        if slot.old_index is not None:
            new_items.append(old_items[slot.old_index])
        elif slot.new_track is not None:
            new_items.append(build_track_uuid_entry_210b(slot.new_track.track_name, slot.new_track.track_uuid16))
    body = hdr_prefix + struct.pack("<I", len(new_items)) + b"".join(new_items) + c2107_raw[end_pos:]
    return build_zchunk(0x2107, body, version=version)


def mutate_210c_folders(
    c210c_raw: bytes,
    folder_insertions: Mapping[bytes, Sequence[bytes]],
    deleted_uuids: set[bytes],
) -> bytes:
    """Update ``0x210C`` folder hierarchy with inserted child UUIDs and pruned folders/children."""
    if len(c210c_raw) < 13:
        raise RuntimeError("Truncated 0x210C folder hierarchy chunk")
    version = struct.unpack_from("<H", c210c_raw, 1)[0]
    pos = 9
    folder_count = struct.unpack_from("<I", c210c_raw, pos)[0]
    pos += 4
    folders: list[tuple[bytes, list[bytes]]] = []
    seen_folders: set[bytes] = set()
    for _ in range(folder_count):
        if pos + 21 > len(c210c_raw):
            raise RuntimeError("Truncated 0x210C folder header")
        pos += 1
        f_uuid = c210c_raw[pos : pos + 16]
        pos += 16
        ccnt = struct.unpack_from("<I", c210c_raw, pos)[0]
        pos += 4
        if ccnt > (len(c210c_raw) - pos) // 17:
            raise RuntimeError(f"0x210C child count {ccnt} exceeds chunk bounds")
        existing_children: list[bytes] = []
        for _ in range(ccnt):
            pos += 1
            c_uuid = c210c_raw[pos : pos + 16]
            pos += 16
            if c_uuid not in deleted_uuids:
                existing_children.append(c_uuid)
        seen_folders.add(f_uuid)
        if f_uuid in deleted_uuids:
            continue
        added = list(folder_insertions.get(f_uuid, ()))
        folders.append((f_uuid, added + existing_children))
    for f_uuid, added_children in folder_insertions.items():
        if f_uuid not in seen_folders and f_uuid not in deleted_uuids and added_children:
            folders.append((f_uuid, list(added_children)))
    return serialize_folder_tree_210c(folders, version=version)


def mutate_2587_ui_state(
    c2587_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
) -> bytes:
    """Update ``0x2587 -> 0x2551 -> 0x258b -> 0x258a`` per-track ``0x2589`` entries deterministically."""
    c2551_off = find_zchunk_offset(c2587_raw, 0x2551, 9)
    if c2551_off < 0:
        return c2587_raw
    c2551_end = c2551_off + 7 + struct.unpack_from("<I", c2587_raw, c2551_off + 3)[0]
    c258b_off = find_zchunk_offset(c2587_raw, 0x258B, c2551_off + 9, c2551_end)
    if c258b_off < 0:
        return c2587_raw
    c258b_end = c258b_off + 7 + struct.unpack_from("<I", c2587_raw, c258b_off + 3)[0]
    c258a_off = find_zchunk_offset(c2587_raw, 0x258A, c258b_off + 9, c258b_end)
    if c258a_off < 0:
        return c2587_raw

    _tag81, _raw81, after_81 = read_zchunk_at(c2587_raw, c258a_off + 9, expected_tag=0x2581)
    if after_81 + 4 > len(c2587_raw):
        raise RuntimeError("Truncated 0x258A track UI entry count")
    old_count = struct.unpack_from("<I", c2587_raw, after_81)[0]
    first_off = after_81 + 4
    old_2589, last_end = walk_sequential_zchunks(c2587_raw, first_off, old_count, expected_tag=0x2589)
    if expected_old_count is not None and len(old_2589) != expected_old_count:
        raise RuntimeError(f"0x258A UI track count {len(old_2589)} != expected {expected_old_count}")

    default_audio_ui = build_audio_ui_entry_2589()
    new_items = [
        old_2589[slot.old_index] if slot.old_index is not None else default_audio_ui
        for slot in layout
    ]
    new_blob = b"".join(new_items)
    buf = bytearray(c2587_raw)
    struct.pack_into("<I", buf, after_81, len(new_items))
    buf[first_off:last_end] = new_blob
    delta = len(new_blob) - (last_end - first_off)
    for anc_off in (c258a_off, c258b_off, c2551_off, 0):
        old_sz = struct.unpack_from("<I", buf, anc_off + 3)[0]
        new_sz = old_sz + delta
        if new_sz < 2 or new_sz > 0xFFFF_FFFF:
            raise ValueError(f"0x2587 ancestor size out of u32 range at 0x{anc_off:x}: {new_sz}")
        struct.pack_into("<I", buf, anc_off + 3, new_sz)
    return bytes(buf)


def _max_2624_control_id(strips: Sequence[bytes]) -> int:
    max_id = 10000
    for strip in strips:
        last_485a_end = -1
        scan = 9
        while True:
            hit = find_zchunk_offset(strip, 0x485A, scan)
            if hit < 0:
                break
            last_485a_end = hit + 7 + struct.unpack_from("<I", strip, hit + 3)[0]
            scan = last_485a_end
        if last_485a_end < 0 or last_485a_end + 53 > len(strip):
            continue
        if (
            strip[last_485a_end : last_485a_end + 4] == b"\x01\x01\x0a\x00"
            and strip[last_485a_end + 44 : last_485a_end + 53] == b"\x00\x00\x00\x00\x08\x00\x00\x00\x00"
        ):
            for val in struct.unpack_from("<10I", strip, last_485a_end + 4):
                if val < 0xFFFF_FF00 and val > max_id:
                    max_id = val
    return max_id + 16


def _parse_261c_mid_gap(strip: bytes) -> tuple[int, int, int, int, int, int]:
    """Parse ``0x261C`` mid-gap between ``0x261B`` and ``0x200B``.

    Returns ``(mid_off, active_pl, pl_cnt, entry_stride, p200b, tail_off)``.
    """
    _tag_b, _raw_b, mid_off = read_zchunk_at(strip, 9, expected_tag=0x261B)
    p200b = find_zchunk_offset(strip, 0x200B, mid_off)
    if p200b < mid_off + 6:
        raise RuntimeError("Missing or malformed 0x200B after 0x261B inside 0x261C")
    _tag_200b, _raw_200b, tail_off = read_zchunk_at(strip, p200b, expected_tag=0x200B)
    active_pl, pl_cnt = struct.unpack_from("<IH", strip, mid_off)
    mid_len = p200b - mid_off
    trailer_ok = mid_len >= 8 and struct.unpack_from("<H", strip, p200b - 2)[0] == 0xFFFF
    if trailer_ok and mid_len == 6 + pl_cnt * 2 + 4:
        return mid_off, active_pl, pl_cnt, 2, p200b, tail_off
    if trailer_ok and mid_len == 6 + pl_cnt * 4 + 2:
        return mid_off, active_pl, pl_cnt, 4, p200b, tail_off
    raise RuntimeError(f"Unrecognized 0x261C mid_gap layout (len={mid_len}, pl_cnt={pl_cnt})")


def _extract_261c_playlist_indices(strip: bytes) -> set[int]:
    """Return all ``0x1015`` playlist indices referenced by an Audio track's ``0x261C`` strip."""
    if len(strip) < 25 or strip[0] != 0x5A or struct.unpack_from("<H", strip, 7)[0] != 0x261C:
        return set()
    mid_off, active_pl, pl_cnt, stride, _p200b, _tail_off = _parse_261c_mid_gap(strip)
    fmt = f"<{pl_cnt}H" if stride == 2 else f"<{pl_cnt}I"
    entries = struct.unpack_from(fmt, strip, mid_off + 6) if pl_cnt else ()
    return {active_pl, *entries}


def _patch_200b_view_ordinal(buf: bytearray, p200b: int, tail_off: int, track_ordinal_1based: int) -> None:
    p2434 = find_zchunk_offset(buf, 0x2434, p200b + 9, tail_off)
    if 0 <= p2434 < tail_off:
        after_2434 = p2434 + 7 + struct.unpack_from("<I", buf, p2434 + 3)[0]
        if after_2434 + 4 <= tail_off:
            struct.pack_into("<I", buf, after_2434, track_ordinal_1based)


def _remap_261c_strip_indices(
    strip: bytes,
    old_pl_to_new: Mapping[int, int] | None,
    track_ordinal_1based: int,
) -> bytes:
    """Remap ``0x2015`` track ordinals on surviving strips and ``0x1015`` playlist indices on ``0x261C`` strips."""
    if len(strip) < 25 or strip[0] != 0x5A:
        return strip
    tag = struct.unpack_from("<H", strip, 7)[0]
    if tag != 0x261C:
        search_start = 9
        if strip[9] == 0x5A and struct.unpack_from("<H", strip, 16)[0] == 0x261B:
            _t, _r, search_start = read_zchunk_at(strip, 9, expected_tag=0x261B)
        p200b = find_zchunk_offset(strip, 0x200B, search_start)
        if p200b < 0:
            return strip
        _t, _r, tail_off = read_zchunk_at(strip, p200b, expected_tag=0x200B)
        buf = bytearray(strip)
        _patch_200b_view_ordinal(buf, p200b, tail_off, track_ordinal_1based)
        return bytes(buf)
    mid_off, active_pl, pl_cnt, stride, p200b, tail_off = _parse_261c_mid_gap(strip)
    buf = bytearray(strip)
    if old_pl_to_new is not None:
        if active_pl not in old_pl_to_new:
            raise RuntimeError(f"Surviving 0x261C strip references deleted/missing playlist index {active_pl}")
        new_active = old_pl_to_new[active_pl]
        struct.pack_into("<I", buf, mid_off, new_active)
        max_entry = 0xFFFF if stride == 2 else 0xFFFF_FFFF
        entry_fmt = "<H" if stride == 2 else "<I"
        for k in range(pl_cnt):
            off_k = mid_off + 6 + k * stride
            old_i = struct.unpack_from(entry_fmt, buf, off_k)[0]
            if old_i not in old_pl_to_new:
                raise RuntimeError(f"Surviving 0x261C strip references deleted/missing playlist index {old_i}")
            new_i = old_pl_to_new[old_i]
            if not (0 <= new_i <= max_entry):
                raise OverflowError(f"Remapped 0x261C playlist index {new_i} exceeds {entry_fmt} range")
            struct.pack_into(entry_fmt, buf, off_k, new_i)
        if tail_off + 21 <= len(buf):
            struct.pack_into("<I", buf, tail_off + 17, new_active)
    _patch_200b_view_ordinal(buf, p200b, tail_off, track_ordinal_1based)
    return bytes(buf)


def mutate_2624_mixer(
    c2624_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
    old_playlist_to_new: Mapping[int, int] | None = None,
) -> bytes:
    """Rebuild ``0x2624`` mixer strip container deterministically for *layout*."""
    if len(c2624_raw) < 13:
        raise RuntimeError("Truncated 0x2624 mixer container chunk")
    version = struct.unpack_from("<H", c2624_raw, 1)[0]
    old_count = struct.unpack_from("<I", c2624_raw, 9)[0]
    old_strips, end_pos = _walk_sequential_zchunks(c2624_raw, 13, old_count)
    if expected_old_count is not None and len(old_strips) != expected_old_count:
        raise RuntimeError(f"0x2624 strip count {len(old_strips)} != expected {expected_old_count}")

    next_id = _max_2624_control_id(old_strips)
    new_strips: list[bytes] = []
    for ord_1based, slot in enumerate(layout, start=1):
        if slot.old_index is not None:
            new_strips.append(_remap_261c_strip_indices(old_strips[slot.old_index], old_playlist_to_new, ord_1based))
        elif slot.new_track is not None:
            nt = slot.new_track
            if next_id + 10 > 0xFFFF_FFFE:
                raise OverflowError(f"0x2624 control ID counter overflow: {next_id}")
            new_strips.append(
                build_audio_mixer_strip_261c(
                    nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                    track_ordinal_1based=ord_1based, id_base=next_id,
                    playlist_index=nt.playlist_index, color_index=nt.color_index,
                )
            )
            next_id += 10
    body = struct.pack("<I", len(new_strips)) + b"".join(new_strips) + c2624_raw[end_pos:]
    return build_zchunk(0x2624, body, version=version)


def _remap_single_group_202a(
    g_raw: bytes,
    old_to_new: Mapping[int, int],
    new_track_count: int,
) -> bytes:
    if len(g_raw) < 13:
        raise RuntimeError("Truncated 0x202A group chunk")
    ver = struct.unpack_from("<H", g_raw, 1)[0]
    nlen = struct.unpack_from("<I", g_raw, 9)[0]
    cnt_off = 13 + nlen + 7
    if cnt_off + 4 > len(g_raw):
        raise RuntimeError("Truncated 0x202A group header")
    name = g_raw[13 : 13 + nlen].decode("utf-8", "replace")
    mcnt = struct.unpack_from("<I", g_raw, cnt_off)[0]
    members_end = cnt_off + 4 + mcnt * 2
    if members_end > len(g_raw):
        raise RuntimeError("Truncated 0x202A group member array")

    if name == "<ALL>":
        new_members = list(range(new_track_count))
    else:
        old_members = struct.unpack_from(f"<{mcnt}H", g_raw, cnt_off + 4) if mcnt else ()
        new_members = [old_to_new[m] for m in old_members if m in old_to_new]
    packed_members = struct.pack(f"<{len(new_members)}H", *new_members) if new_members else b""

    tail = bytearray(g_raw[members_end:])
    if len(tail) >= 12:
        raw_vca = struct.unpack_from("<H", tail, 10)[0]
        if raw_vca != 0xFFFF:
            struct.pack_into("<H", tail, 10, old_to_new.get(raw_vca, 0xFFFF))

    new_body = g_raw[9:cnt_off] + struct.pack("<I", len(new_members)) + packed_members + bytes(tail)
    return build_zchunk(0x202A, new_body, version=ver)


def mutate_202b_groups_for_track_map(
    c202b_raw: bytes,
    old_to_new: Mapping[int, int],
    new_track_count: int,
) -> bytes:
    """Remap all ``0x202A`` group member & VCA indices in ``0x202B`` (both edit and mix lists)."""
    version = struct.unpack_from("<H", c202b_raw, 1)[0]
    edit_list, mix_list, trailer = split_202b(c202b_raw)
    new_edit = [_remap_single_group_202a(g, old_to_new, new_track_count) for g in edit_list]
    new_mix = [_remap_single_group_202a(g, old_to_new, new_track_count) for g in mix_list]
    body = (
        struct.pack("<I", len(new_edit))
        + b"".join(new_edit)
        + struct.pack("<I", len(new_mix))
        + b"".join(new_mix)
        + trailer
    )
    return build_zchunk(0x202B, body, version=version)


def _split_session_dir_parts(session_dir: Path) -> tuple[str, list[str]]:
    raw_str = str(session_dir.resolve())
    if raw_str.startswith("\\\\?\\UNC\\"):
        raw_str = "\\\\" + raw_str[8:]
    elif raw_str.startswith("\\\\?\\"):
        raw_str = raw_str[4:]
    parts = list(Path(raw_str).parts)
    if not parts:
        return "/", []
    root = parts[0]
    rest = parts[1:]
    if len(root) >= 2 and root[1] == ":":
        root = root[0].lower() + ":\\"
    elif root == "/" and len(rest) >= 2 and rest[0] == "Volumes":
        root = rest[1]
        rest = rest[2:]
    return root, rest


def _parse_103a_entries(c103a_raw: bytes) -> list[PtxDirTreeEntry]:
    if len(c103a_raw) < 13 or c103a_raw[0] != 0x5A:
        raise RuntimeError("Truncated 0x103A directory tree chunk")
    ecnt = struct.unpack_from("<I", c103a_raw, 9)[0]
    pos = 13
    entries: list[PtxDirTreeEntry] = []
    for _ in range(ecnt):
        if pos + 13 > len(c103a_raw):
            raise RuntimeError("Truncated 0x103A entry header")
        kind = c103a_raw[pos]
        parent = struct.unpack_from("<I", c103a_raw, pos + 1)[0]
        nlen = struct.unpack_from("<I", c103a_raw, pos + 5)[0]
        pos += 9
        if pos + nlen + 4 > len(c103a_raw):
            raise RuntimeError("Truncated 0x103A entry name/tag")
        name = c103a_raw[pos : pos + nlen].decode("utf-8", "replace")
        tag4 = c103a_raw[pos + nlen : pos + nlen + 4]
        pos += nlen + 4
        entries.append(PtxDirTreeEntry(kind=kind, parent_index=parent, name=name, tag4=tag4))
    return entries


def _append_session_tree_nodes(
    entries: list[PtxDirTreeEntry],
    staged_files: Sequence[AudioFileWriteSpec],
    root_path: str,
    dir_parts: Sequence[str],
    volume_serial_u32: int,
) -> list[int]:
    """Append volume, directory chain, ``Audio Files``, and file nodes; return new file node indices."""
    seen_staged: set[str] = set()
    for sf_item in staged_files:
        low_fn = sf_item.filename.lower()
        if low_fn in seen_staged:
            raise ValueError(f"Duplicate audio filename {sf_item.filename!r} in staged audio files")
        seen_staged.add(low_fn)

    vol_tag4 = struct.pack("<I", volume_serial_u32 & 0xFFFF_FFFF)
    if not entries:
        vol_idx = 1 + len(staged_files)
        session_dir_idx = vol_idx + len(dir_parts)
        entries.append(PtxDirTreeEntry(kind=1, parent_index=session_dir_idx, name="Audio Files"))
        file_indices: list[int] = []
        for sf_item in staged_files:
            file_indices.append(len(entries))
            entries.append(PtxDirTreeEntry(kind=2, parent_index=0, name=sf_item.filename, tag4=b"EVAW"))
        entries.append(PtxDirTreeEntry(kind=0, parent_index=0xFFFF_FFFF, name=root_path, tag4=vol_tag4))
        for idx, part in enumerate(dir_parts):
            entries.append(PtxDirTreeEntry(kind=1, parent_index=vol_idx + idx, name=part))
        return file_indices

    # Existing entries at 0..E-1 stay at their original indices so existing 0x1003 tree_node_index refs never shift.
    parent_idx = next(
        (i for i, e in enumerate(entries) if e.kind == 0 and e.name.lower() == root_path.lower() and e.tag4 == vol_tag4),
        -1,
    )
    if parent_idx < 0:
        parent_idx = len(entries)
        entries.append(PtxDirTreeEntry(kind=0, parent_index=0xFFFF_FFFF, name=root_path, tag4=vol_tag4))
    for part in (*dir_parts, "Audio Files"):
        found = next(
            (i for i, e in enumerate(entries) if e.kind == 1 and e.parent_index == parent_idx and e.name.lower() == part.lower()),
            -1,
        )
        if found >= 0:
            parent_idx = found
        else:
            new_idx = len(entries)
            entries.append(PtxDirTreeEntry(kind=1, parent_index=parent_idx, name=part))
            parent_idx = new_idx
    existing_files = {
        e.name.lower() for e in entries if e.kind == 2 and e.parent_index == parent_idx
    }
    file_indices = []
    for sf_item in staged_files:
        low_fn = sf_item.filename.lower()
        if low_fn in existing_files:
            raise ValueError(f"Duplicate audio filename {sf_item.filename!r} under session Audio Files directory")
        existing_files.add(low_fn)
        file_indices.append(len(entries))
        entries.append(PtxDirTreeEntry(kind=2, parent_index=parent_idx, name=sf_item.filename, tag4=b"EVAW"))
    return file_indices


def mutate_1004_audio_files(
    c1004_raw: bytes,
    staged_files: Sequence[AudioFileWriteSpec],
    *,
    session_dir: Path,
    volume_serial_u32: int,
) -> tuple[bytes, str]:
    """Append staged audio files into ``0x1004`` while preserving any pre-existing template audio files."""
    if len(c1004_raw) < 13:
        raise RuntimeError("Truncated 0x1004 audio file table chunk")
    version = struct.unpack_from("<H", c1004_raw, 1)[0]
    root_path, dir_parts = _split_session_dir_parts(session_dir)
    if not staged_files:
        return c1004_raw, root_path

    old_file_count = struct.unpack_from("<I", c1004_raw, 9)[0]
    _tag_tree, old_103a_raw, pos = _read_zchunk_at(c1004_raw, 13, expected_tag=0x103A)
    old_103a_ver = struct.unpack_from("<H", old_103a_raw, 1)[0]
    old_1003_list, after_1003 = _walk_sequential_zchunks(c1004_raw, pos, old_file_count, expected_tag=0x1003)
    trailing_tail = c1004_raw[after_1003:] if after_1003 < len(c1004_raw) else (
        b"\x00\x00\x00\x00" + serialize_directory_tree_103a([], version=1)
    )

    entries = _parse_103a_entries(old_103a_raw)
    new_node_indices = _append_session_tree_nodes(entries, staged_files, root_path, dir_parts, volume_serial_u32)
    new_103a = serialize_directory_tree_103a(entries, version=old_103a_ver)
    new_1003_list = [
        build_audio_file_entry_1003(
            item.sample_rate, item.channels, item.bit_depth, item.length_samples,
            file_id8=item.file_id8, smpte_umid32=item.smpte_umid32,
            filetime_u64=item.filetime_u64, tree_node_index=node_idx,
        )
        for item, node_idx in zip(staged_files, new_node_indices)
    ]
    total_files = old_file_count + len(new_1003_list)
    body = struct.pack("<I", total_files) + new_103a + b"".join(old_1003_list) + b"".join(new_1003_list) + trailing_tail
    return build_zchunk(0x1004, body, version=version), root_path


def mutate_0f3d_volumes(
    c0f3d_raw: bytes,
    volume_serial_u32: int,
    volume_label: str,
    root_path: str,
) -> bytes:
    """Ensure ``0x0F3D`` contains a ``0x0F3C`` entry for ``(volume_serial_u32, root_path)``."""
    if len(c0f3d_raw) < 13:
        raise RuntimeError("Truncated 0x0F3D volume table chunk")
    version = struct.unpack_from("<H", c0f3d_raw, 1)[0]
    old_count = struct.unpack_from("<I", c0f3d_raw, 9)[0]
    old_entries, end_pos = _walk_sequential_zchunks(c0f3d_raw, 13, old_count, expected_tag=0x0F3C)
    target_serial = volume_serial_u32 & 0xFFFF_FFFF
    target_root = root_path.lower()
    for entry in old_entries:
        if len(entry) >= 17:
            llen = struct.unpack_from("<I", entry, 9)[0]
            ser_off = 13 + llen
            if ser_off + 8 <= len(entry):
                ser = struct.unpack_from("<I", entry, ser_off)[0]
                rlen = struct.unpack_from("<I", entry, ser_off + 4)[0]
                if ser_off + 8 + rlen > len(entry):
                    raise RuntimeError("Truncated 0x0F3C volume root path")
                rpath = entry[ser_off + 8 : ser_off + 8 + rlen].decode("utf-8", "replace").lower()
                if ser == target_serial and rpath == target_root:
                    return c0f3d_raw
    new_entries = [*old_entries, build_volume_entry_0f3c(target_serial, volume_label, root_path)]
    body = struct.pack("<I", len(new_entries)) + b"".join(new_entries) + c0f3d_raw[end_pos:]
    return build_zchunk(0x0F3D, body, version=version)


def mutate_262a_clips(c262a_raw: bytes, new_clips: Sequence[ClipWriteSpec]) -> bytes:
    """Append new whole-file ``0x2629`` clips into ``0x262A`` while preserving existing clips."""
    if len(c262a_raw) < 13:
        raise RuntimeError("Truncated 0x262A clip container chunk")
    version = struct.unpack_from("<H", c262a_raw, 1)[0]
    old_count = struct.unpack_from("<I", c262a_raw, 9)[0]
    old_entries, end_pos = _walk_sequential_zchunks(c262a_raw, 13, old_count, expected_tag=0x2629)
    added = [
        build_whole_file_clip_2629(
            c.clip_name, c.length_samples,
            audio_file_index=c.audio_file_index, channel_index=c.channel_index,
            clip_guid16=c.clip_guid16, pt_uuid16=c.pt_uuid16,
        )
        for c in new_clips
    ]
    all_entries = [*old_entries, *added]
    body = struct.pack("<I", len(all_entries)) + b"".join(all_entries) + c262a_raw[end_pos:]
    return build_zchunk(0x262A, body, version=version)


def _extract_counted_items_and_trailer(
    container_raw: bytes,
    item_tag: int,
    trailer_tag: int,
    default_trailer: bytes,
) -> tuple[list[bytes], bytes]:
    if len(container_raw) < 13:
        return [], default_trailer
    cnt = struct.unpack_from("<I", container_raw, 9)[0]
    items, after = _walk_sequential_zchunks(container_raw, 13, cnt, expected_tag=item_tag)
    if after + 9 <= len(container_raw) and container_raw[after] == 0x5A:
        tag, raw_sub, _end = _read_zchunk_at(container_raw, after)
        if tag == trailer_tag:
            return items, raw_sub
    return items, default_trailer


def _parse_playlist_1014_meta(raw_1014: bytes) -> tuple[str, int, tuple[int, ...], int]:
    """Return ``(name, is_stereo_byte, lane_indices, lane_array_offset)`` for a ``0x1014`` chunk."""
    if len(raw_1014) < 18 or raw_1014[0] != 0x5A:
        raise RuntimeError("Truncated 0x1014 playlist chunk")
    nlen = struct.unpack_from("<I", raw_1014, 9)[0]
    st_off = 13 + nlen
    if st_off + 5 > len(raw_1014):
        raise RuntimeError("Truncated 0x1014 playlist header")
    name = raw_1014[13:st_off].decode("utf-8", "replace")
    is_st = raw_1014[st_off]
    lcnt = struct.unpack_from("<I", raw_1014, st_off + 1)[0]
    arr_off = st_off + 5
    if lcnt > (len(raw_1014) - arr_off) // 2:
        raise RuntimeError(f"Truncated 0x1014 lane array for playlist {name!r}")
    lanes = struct.unpack_from(f"<{lcnt}H", raw_1014, arr_off) if lcnt else ()
    return name, is_st, lanes, arr_off


def _remap_playlist_1014_lanes(raw_1014: bytes, old_lane_to_new: Mapping[int, int]) -> bytes:
    """Remap both ``<H`` and v9 trailing ``<I`` lane indices in a surviving ``0x1014`` playlist."""
    name, _is_st, old_lanes, arr_off = _parse_playlist_1014_meta(raw_1014)
    missing = [li for li in old_lanes if li not in old_lane_to_new]
    if missing:
        raise RuntimeError(f"Surviving 0x1014 playlist {name!r} references missing lane(s): {missing}")
    new_lanes = tuple(old_lane_to_new[li] for li in old_lanes)
    if new_lanes == old_lanes:
        return raw_1014
    if any(not (0 <= li <= 0xFFFF) for li in new_lanes):
        raise OverflowError(f"Remapped 0x1014 lane index out of u16 range for playlist {name!r}")
    buf = bytearray(raw_1014)
    struct.pack_into(f"<{len(new_lanes)}H", buf, arr_off, *new_lanes)
    ver = struct.unpack_from("<H", buf, 1)[0]
    tail_off = arr_off + len(old_lanes) * 2 + 16 + 7  # after <H array + 16B UUID + 7B header
    if ver >= 9 and old_lanes and tail_off + len(old_lanes) * 4 <= len(buf):
        if struct.unpack_from(f"<{len(old_lanes)}I", buf, tail_off) == old_lanes:
            struct.pack_into(f"<{len(new_lanes)}I", buf, tail_off, *new_lanes)
    return bytes(buf)


def _partition_2624_playlist_indices(
    c2624_raw: bytes,
    deleted_old_indices: Sequence[int],
) -> tuple[set[int], set[int]]:
    """Return ``(deleted_pl_indices, surviving_pl_indices)`` from ``0x2624`` Audio strips."""
    if len(c2624_raw) < 13:
        raise RuntimeError("Truncated 0x2624 mixer container chunk")
    old_count = struct.unpack_from("<I", c2624_raw, 9)[0]
    old_strips, _ = _walk_sequential_zchunks(c2624_raw, 13, old_count)
    del_set = set(deleted_old_indices)
    dropped: set[int] = set()
    surviving: set[int] = set()
    for idx, strip in enumerate(old_strips):
        pl_ids = _extract_261c_playlist_indices(strip)
        if idx in del_set:
            dropped.update(pl_ids)
        else:
            surviving.update(pl_ids)
    return dropped, surviving


def plan_existing_media_survival(
    c1015_raw: bytes,
    c1054_raw: bytes,
    deleted_track_names: Sequence[str],
    *,
    c2624_raw: bytes | None = None,
    deleted_old_indices: Sequence[int] = (),
) -> MediaSurvivalPlan:
    """Compact existing ``0x1014`` playlists & ``0x1052`` lanes for any deleted template audio tracks."""
    default_4829 = build_zchunk(0x4829, b"\x00\x00\x00\x00", version=2)
    default_4828 = build_zchunk(0x4828, b"\x00\x00\x00\x00", version=2)
    old_playlists, c4829 = _extract_counted_items_and_trailer(c1015_raw, 0x1014, 0x4829, default_4829)
    old_lanes, c4828 = _extract_counted_items_and_trailer(c1054_raw, 0x1052, 0x4828, default_4828)

    if c2624_raw is not None:
        dropped_pl, surviving_pl = _partition_2624_playlist_indices(c2624_raw, deleted_old_indices)
        removable_pl = dropped_pl - surviving_pl
        use_name_fallback = False
    else:
        removable_pl = set()
        use_name_fallback = True
    del_lc = {n.lower() for n in deleted_track_names}

    kept_pl_raw: list[bytes] = []
    old_pl_to_new: dict[int, int] = {}
    dropped_lanes: set[int] = set()
    kept_lanes_used: set[int] = set()
    for pl_idx, pl in enumerate(old_playlists):
        name, _st, lanes, _off = _parse_playlist_1014_meta(pl)
        should_drop = (pl_idx in removable_pl) or (use_name_fallback and name.lower() in del_lc)
        if should_drop:
            dropped_lanes.update(lanes)
        else:
            old_pl_to_new[pl_idx] = len(kept_pl_raw)
            kept_pl_raw.append(pl)
            kept_lanes_used.update(lanes)
    removable_lanes = dropped_lanes - kept_lanes_used

    surviving_lanes: list[bytes] = []
    old_lane_to_new: dict[int, int] = {}
    for old_idx, lane_raw in enumerate(old_lanes):
        if old_idx in removable_lanes:
            continue
        old_lane_to_new[old_idx] = len(surviving_lanes)
        surviving_lanes.append(lane_raw)
    remapped_playlists = tuple(_remap_playlist_1014_lanes(pl, old_lane_to_new) for pl in kept_pl_raw)
    return MediaSurvivalPlan(
        playlists=remapped_playlists,
        lanes=tuple(surviving_lanes),
        old_playlist_to_new=old_pl_to_new,
        c4829_trailer=c4829,
        c4828_trailer=c4828,
    )


def mutate_1015_and_1054_media(
    c1015_raw: bytes,
    c1054_raw: bytes,
    new_tracks: Sequence[NewAudioTrackSpec],
    *,
    survival: MediaSurvivalPlan,
) -> tuple[bytes, bytes]:
    """Rebuild ``0x1015`` playlists & ``0x1054`` lanes by appending *new_tracks* after surviving entries."""
    ver_1015 = struct.unpack_from("<H", c1015_raw, 1)[0]
    ver_1054 = struct.unpack_from("<H", c1054_raw, 1)[0]
    added_playlists = [
        build_playlist_1014_v9(
            nt.track_name, nt.lane_indices, nt.playlist_uuid16, is_stereo=(nt.channels >= 2),
        )
        for nt in new_tracks
    ]
    added_lanes: list[bytes] = []
    for nt in new_tracks:
        if nt.clip_indices and len(nt.clip_indices) != len(nt.lane_indices):
            raise ValueError(
                f"Track {nt.track_name!r} clip_indices count ({len(nt.clip_indices)}) != lane_indices count ({len(nt.lane_indices)})"
            )
        if nt.clip_indices:
            for clip_idx in nt.clip_indices:
                added_lanes.append(
                    serialize_lane_1052(
                        nt.track_name,
                        [(clip_idx, nt.start_sample, 3, 0xFFFE)],
                        wrap_1050=True, wrap_version=3, lane_version=3,
                    )
                )
        else:
            for _ in nt.lane_indices:
                added_lanes.append(
                    serialize_lane_1052(
                        nt.track_name, (), wrap_1050=True, wrap_version=3, lane_version=3,
                    )
                )
    all_playlists = [*survival.playlists, *added_playlists]
    all_lanes = [*survival.lanes, *added_lanes]
    out_1015 = build_zchunk(
        0x1015, struct.pack("<I", len(all_playlists)) + b"".join(all_playlists) + survival.c4829_trailer, version=ver_1015,
    )
    out_1054 = build_zchunk(
        0x1054, struct.pack("<I", len(all_lanes)) + b"".join(all_lanes) + survival.c4828_trailer, version=ver_1054,
    )
    return out_1015, out_1054
