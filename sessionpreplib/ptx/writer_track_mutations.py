"""Track-layout binary mutators (``0x2519``, ``0x2107``, ``0x210C``, ``0x2587``, ``0x2624``, ``0x202B``, ``0x2030``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import struct

from .codec import (
    find_subchunk_spans,
    find_zchunk_offset,
    read_zchunk_at,
    resize_zchunk_ancestors,
    walk_sequential_zchunks,
)
from .models import PtxCorruptFileError, PtxUnsupportedTemplateError, PtxValidationError
from .parser import split_202b
from .writer_blueprints import (
    TRACK_TYPE_TO_251A,
    OutputBusSpec,
    build_audio_mixer_strip_261c,
    build_audio_ui_entry_2589,
    build_aux_or_rf_mixer_strip_261e,
    build_basic_folder_mixer_strip_263b,
    build_instrument_mixer_strip_2621,
    build_master_mixer_strip_261d,
    build_midi_mixer_strip_2620,
    build_track_entry_251a,
    build_track_summary_2519,
    build_track_uuid_entry_210b,
    build_vca_mixer_strip_261f,
)
from .writer_primitives import build_zchunk, serialize_folder_tree_210c


@dataclass(frozen=True)
class NewAudioTrackSpec:
    """Internal spec for one newly synthesized track (Audio, Aux, RoutingFolder, Master, Vca, BasicFolder, Midi, Instrument)."""

    track_name: str
    track_uuid16: bytes
    playlist_uuid16: bytes
    channels: int
    parent_folder_uuid16: bytes | None
    output_bus: OutputBusSpec | tuple[int, bytes, str] | None
    color_index: int
    clip_indices: tuple[int, ...]  # () for empty/non-Audio track, 1..12 in 0x262A
    lane_indices: tuple[int, ...]  # 1..12 in 0x1054 for Audio; () for non-Audio
    playlist_index: int = 0  # 0-based index into 0x1015 Audio Playlists
    start_sample: int = 0
    track_type: str = "Audio"
    is_folder_open: bool = True
    track_format: str | None = None
    midi_index: int | None = None
    midi_uuid16: bytes = b"\x00" * 16
    clip_muted: bool = False
    clip_time_locked: bool = False
    clip_edit_locked: bool = False
    clip_color_index: int = 0xFFFE


@dataclass(frozen=True)
class TrackSlotOp:
    """Unified output track slot: either keep existing template track *old_index* or insert *new_track*."""

    old_index: int | None = None
    new_track: NewAudioTrackSpec | None = None

    def __post_init__(self) -> None:
        if (self.old_index is None) == (self.new_track is None):
            raise PtxValidationError("TrackSlotOp must set exactly one of old_index or new_track")


@dataclass(frozen=True)
class MidiStripPlaylists:
    """Parsed MIDI/Instrument playlist-reference block inside a ``0x2620`` or ``0x2621`` strip."""

    strip_tag: int
    offset: int
    stride: int  # 4 for 0x2620 (<I count + <N*I), 2 for 0x2621 (<H count + <N*H)
    active_index: int
    playlist_indices: tuple[int, ...]


def parse_2519_sections(raw: bytes) -> tuple[bytes, list[bytes], list[bytes], list[bytes], bytes]:
    """Deterministically parse ``0x2519`` into ``(clean_prefix, rows, pass1, pass2, tail)``."""
    if len(raw) < 27 or raw[0] != 0x5A:
        raise PtxCorruptFileError("Truncated 0x2519 track directory chunk")
    pos = 9
    hdr10 = raw[pos : pos + 10]
    pos += 10
    cnt_a = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    if cnt_a > (len(raw) - pos) // 2:
        raise PtxCorruptFileError(f"0x2519 header cnt_a ({cnt_a}) exceeds chunk bounds")
    pos += cnt_a * 2
    clean_prefix = hdr10 + b"\x00\x00\x00\x00"
    if pos + 4 > len(raw):
        raise PtxCorruptFileError("Truncated 0x2519 Table 1 track count")
    tcnt = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    rows: list[bytes] = []
    for _ in range(tcnt):
        if pos + 6 > len(raw):
            raise PtxCorruptFileError("Truncated 0x2519 Table 1 track row")
        row_start = pos
        pos += 2
        nlen = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if nlen + 21 > len(raw) - pos:
            raise PtxCorruptFileError("Truncated 0x2519 Table 1 track name/UUID")
        pos += nlen + 21
        rows.append(raw[row_start:pos])
    pass1, pos = walk_sequential_zchunks(raw, pos, tcnt, expected_tag=0x251A)
    if pos + 4 > len(raw):
        raise PtxCorruptFileError("Truncated 0x2519 Pass 2 track count")
    tcnt2 = struct.unpack_from("<I", raw, pos)[0]
    if tcnt2 != tcnt:
        raise PtxCorruptFileError(f"0x2519 Pass 2 count {tcnt2} != Pass 1 count {tcnt}")
    pos += 4
    pass2, pos = walk_sequential_zchunks(raw, pos, tcnt2, expected_tag=0x251A)
    return clean_prefix, rows, pass1, pass2, raw[pos:]


def _patch_2519_summary_order(row: bytes, visible_order: int) -> bytes:
    if not (0 <= visible_order <= 0xFFFF):
        raise PtxValidationError(f"0x2519 visible_order exceeds u16 range: {visible_order}")
    buf = bytearray(row)
    struct.pack_into("<H", buf, len(buf) - 3, visible_order)
    return bytes(buf)


def _patch_251a_order(raw_251a: bytes, visible_order: int) -> bytes:
    if not (0 <= visible_order <= 0xFFFF):
        raise PtxValidationError(f"0x251A visible_order exceeds u16 range: {visible_order}")
    buf = bytearray(raw_251a)
    if len(buf) < 15:
        raise PtxCorruptFileError("Truncated 0x251A track entry")
    nlen = struct.unpack_from("<I", buf, 11)[0]
    ord_off = 15 + nlen + 2 + 16
    if ord_off + 2 > len(buf):
        raise PtxCorruptFileError("Truncated 0x251A visible-order field")
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
    prefix, old_rows, old_p1, old_p2, tail_raw = parse_2519_sections(c2519_raw)
    if expected_old_count is not None and len(old_rows) != expected_old_count:
        raise PtxCorruptFileError(f"0x2519 track count {len(old_rows)} != expected {expected_old_count}")

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
            new_rows.append(
                build_track_summary_2519(
                    nt.track_name, nt.track_uuid16, nt.channels, vis_ord,
                    track_type=nt.track_type, track_format=nt.track_format,
                )
            )
            entry = build_track_entry_251a(
                nt.track_name, nt.track_uuid16, nt.channels, vis_ord,
                color_index=nt.color_index, has_clips=bool(nt.clip_indices),
                track_type=nt.track_type, track_format=nt.track_format,
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
        raise PtxCorruptFileError("Truncated 0x2107 track UUID registry chunk")
    version = struct.unpack_from("<H", c2107_raw, 1)[0]
    hdr_prefix = c2107_raw[9:18]
    old_count = struct.unpack_from("<I", c2107_raw, 18)[0]
    old_items, end_pos = walk_sequential_zchunks(c2107_raw, 22, old_count, expected_tag=0x210B)
    if expected_old_count is not None and len(old_items) != expected_old_count:
        raise PtxCorruptFileError(f"0x2107 track count {len(old_items)} != expected {expected_old_count}")
    new_items: list[bytes] = []
    for slot in layout:
        if slot.old_index is not None:
            new_items.append(old_items[slot.old_index])
        elif slot.new_track is not None:
            tcode = TRACK_TYPE_TO_251A.get(slot.new_track.track_type, (0, False))[0]
            new_items.append(
                build_track_uuid_entry_210b(slot.new_track.track_name, slot.new_track.track_uuid16, type_code=tcode)
            )
    body = hdr_prefix + struct.pack("<I", len(new_items)) + b"".join(new_items) + c2107_raw[end_pos:]
    return build_zchunk(0x2107, body, version=version)


def mutate_210c_folders(
    c210c_raw: bytes,
    folder_insertions: Mapping[bytes, Sequence[bytes]],
    deleted_uuids: set[bytes],
) -> bytes:
    """Update ``0x210C`` folder hierarchy with inserted child UUIDs and pruned folders/children."""
    if len(c210c_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x210C folder hierarchy chunk")
    version = struct.unpack_from("<H", c210c_raw, 1)[0]
    pos = 9
    folder_count = struct.unpack_from("<I", c210c_raw, pos)[0]
    pos += 4
    folders: list[tuple[bytes, list[bytes]]] = []
    seen_folders: set[bytes] = set()
    for _ in range(folder_count):
        if pos + 21 > len(c210c_raw):
            raise PtxCorruptFileError("Truncated 0x210C folder header")
        pos += 1
        f_uuid = c210c_raw[pos : pos + 16]
        pos += 16
        ccnt = struct.unpack_from("<I", c210c_raw, pos)[0]
        pos += 4
        if ccnt > (len(c210c_raw) - pos) // 17:
            raise PtxCorruptFileError(f"0x210C child count {ccnt} exceeds chunk bounds")
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
        raise PtxCorruptFileError("Truncated 0x258A track UI entry count")
    old_count = struct.unpack_from("<I", c2587_raw, after_81)[0]
    first_off = after_81 + 4
    old_2589, last_end = walk_sequential_zchunks(c2587_raw, first_off, old_count, expected_tag=0x2589)
    if expected_old_count is not None and len(old_2589) != expected_old_count:
        raise PtxCorruptFileError(f"0x258A UI track count {len(old_2589)} != expected {expected_old_count}")

    old_by_idx: dict[int, bytes] = {}
    for pos_i, entry in enumerate(old_2589):
        if len(entry) >= 11:
            tidx = struct.unpack_from("<H", entry, 9)[0]
            if tidx < len(old_2589) and tidx not in old_by_idx:
                old_by_idx[tidx] = entry
            elif pos_i not in old_by_idx:
                old_by_idx[pos_i] = entry

    new_items: list[bytes] = []
    for new_idx, slot in enumerate(layout):
        if slot.old_index is not None:
            surv = bytearray(old_by_idx.get(slot.old_index, old_2589[slot.old_index]))
            if len(surv) >= 11:
                struct.pack_into("<H", surv, 9, new_idx)
            new_items.append(bytes(surv))
        elif slot.new_track is not None:
            new_items.append(
                build_audio_ui_entry_2589(track_index=new_idx, is_folder_open=slot.new_track.is_folder_open)
            )
    new_blob = b"".join(new_items)
    buf = bytearray(c2587_raw)
    struct.pack_into("<I", buf, after_81, len(new_items))
    buf[first_off:last_end] = new_blob
    delta = len(new_blob) - (last_end - first_off)
    resize_zchunk_ancestors(buf, (c258a_off, c258b_off, c2551_off, 0), delta)
    return bytes(buf)


def _max_2624_control_id(strips: Sequence[bytes]) -> int:
    max_id = 10000
    for strip in strips:
        spans = find_subchunk_spans(strip, 0x485A)
        if not spans:
            continue
        last_off, last_raw = spans[-1]
        last_485a_end = last_off + len(last_raw)
        if last_485a_end + 53 > len(strip):
            continue
        if (
            strip[last_485a_end : last_485a_end + 4] == b"\x01\x01\x0a\x00"
            and (
                strip[last_485a_end + 44 : last_485a_end + 53] == b"\x00\x00\x00\x00\x08\x00\x00\x00\x00"
                or strip[last_485a_end + 44 : last_485a_end + 52] == b"\x00" * 8
            )
        ):
            for val in struct.unpack_from("<10I", strip, last_485a_end + 4):
                if val < 0xFFFF_FF00 and val > max_id:
                    max_id = val
    return max_id + 16


def _parse_261c_mid_gap(strip: bytes) -> tuple[int, int, int, int, int, int]:
    """Parse ``0x261C`` mid-gap between ``0x261B`` and ``0x200B``."""
    _tag_b, _raw_b, mid_off = read_zchunk_at(strip, 9, expected_tag=0x261B)
    p200b = find_zchunk_offset(strip, 0x200B, mid_off)
    if p200b < mid_off + 6:
        raise PtxCorruptFileError("Missing or malformed 0x200B after 0x261B inside 0x261C")
    _tag_200b, _raw_200b, tail_off = read_zchunk_at(strip, p200b, expected_tag=0x200B)
    active_pl, pl_cnt = struct.unpack_from("<IH", strip, mid_off)
    mid_len = p200b - mid_off
    trailer_ok = mid_len >= 8 and struct.unpack_from("<H", strip, p200b - 2)[0] == 0xFFFF
    if trailer_ok and mid_len == 6 + pl_cnt * 2 + 4:
        return mid_off, active_pl, pl_cnt, 2, p200b, tail_off
    if trailer_ok and mid_len == 6 + pl_cnt * 4 + 2:
        return mid_off, active_pl, pl_cnt, 4, p200b, tail_off
    raise PtxCorruptFileError(f"Unrecognized 0x261C mid_gap layout (len={mid_len}, pl_cnt={pl_cnt})")


def extract_261c_playlist_indices(strip: bytes) -> set[int]:
    """Return all ``0x1015`` playlist indices referenced by an Audio track's ``0x261C`` strip."""
    if len(strip) < 25 or strip[0] != 0x5A or struct.unpack_from("<H", strip, 7)[0] != 0x261C:
        return set()
    mid_off, active_pl, pl_cnt, stride, _p200b, _tail_off = _parse_261c_mid_gap(strip)
    fmt = f"<{pl_cnt}H" if stride == 2 else f"<{pl_cnt}I"
    entries = struct.unpack_from(fmt, strip, mid_off + 6) if pl_cnt else ()
    return {active_pl, *entries}


def extract_261c_active_playlist_index(strip: bytes) -> int | None:
    """Return the active ``0x1015`` playlist index of an Audio ``0x261C`` strip, or ``None``."""
    if len(strip) < 25 or strip[0] != 0x5A or struct.unpack_from("<H", strip, 7)[0] != 0x261C:
        return None
    _mid_off, active_pl, _pl_cnt, _stride, _p200b, _tail_off = _parse_261c_mid_gap(strip)
    return active_pl


def parse_midi_strip_playlists(strip: bytes) -> MidiStripPlaylists | None:
    """Parse active and all alternate ``0x1058`` playlist indices on a ``0x2620`` or ``0x2621`` strip."""
    if len(strip) < 25 or strip[0] != 0x5A:
        return None
    tag = struct.unpack_from("<H", strip, 7)[0]
    if tag == 0x2620:
        _t, _r, gap_off = read_zchunk_at(strip, 9, expected_tag=0x102D)
        p2611 = find_zchunk_offset(strip, 0x2611, gap_off)
        if p2611 < gap_off + 9:
            return None
        active_idx, pl_cnt = struct.unpack_from("<II", strip, gap_off)
        gap_len = p2611 - gap_off
        if pl_cnt >= 1 and gap_len == 8 + pl_cnt * 4 + 1:
            entries = struct.unpack_from(f"<{pl_cnt}I", strip, gap_off + 8)
            return MidiStripPlaylists(tag, gap_off, 4, active_idx, tuple(entries))
        return None
    if tag == 0x2621:
        spans = find_subchunk_spans(strip, 0x485A)
        if not spans:
            return None
        last_485a_end = spans[-1][0] + len(spans[-1][1])
        off = last_485a_end + 52
        p2010 = find_zchunk_offset(strip, 0x2010, off)
        if p2010 < off + 9:
            return None
        active_idx, pl_cnt = struct.unpack_from("<IH", strip, off)
        gap_len = p2010 - off
        if pl_cnt >= 1 and gap_len == 6 + pl_cnt * 2 + 3:
            entries = struct.unpack_from(f"<{pl_cnt}H", strip, off + 6)
            return MidiStripPlaylists(tag, off, 2, active_idx, tuple(entries))
        if pl_cnt >= 1 and gap_len == 6 + pl_cnt * 4 + 1:
            entries = struct.unpack_from(f"<{pl_cnt}I", strip, off + 6)
            return MidiStripPlaylists(tag, off, 4, active_idx, tuple(entries))
    return None


def _remap_midi_strip_index(buf: bytearray, old_midi_to_new: Mapping[int, int] | None) -> None:
    if old_midi_to_new is None:
        return
    tag = struct.unpack_from("<H", buf, 7)[0]
    if tag not in (0x2620, 0x2621):
        return
    parsed = parse_midi_strip_playlists(bytes(buf))
    if parsed is None:
        if any(k != v for k, v in old_midi_to_new.items()):
            raise PtxUnsupportedTemplateError(f"Unrecognized 0x{tag:04x} MIDI playlist layout during index remapping")
        return
    if parsed.active_index not in old_midi_to_new or any(i not in old_midi_to_new for i in parsed.playlist_indices):
        raise PtxCorruptFileError(
            f"Surviving 0x{tag:04x} strip references deleted/missing 0x1058 index: "
            f"active={parsed.active_index}, playlists={parsed.playlist_indices}"
        )
    new_active = old_midi_to_new[parsed.active_index]
    new_entries = tuple(old_midi_to_new[i] for i in parsed.playlist_indices)
    max_val = 0xFFFF_FFFF if parsed.stride == 4 else 0xFFFF
    if any(not (0 <= x <= max_val) for x in (new_active, *new_entries)):
        raise PtxValidationError(f"Remapped MIDI strip index out of range: {new_entries}")
    if parsed.strip_tag == 0x2620:
        struct.pack_into(f"<II{len(new_entries)}I", buf, parsed.offset, new_active, len(new_entries), *new_entries)
        p206f = find_zchunk_offset(bytes(buf), 0x206F, parsed.offset + 9)
        if p206f >= 0:
            t_off = p206f + 7 + struct.unpack_from("<I", buf, p206f + 3)[0]
            if t_off + 6 <= len(buf):
                struct.pack_into("<I", buf, t_off + 1, new_active)
    else:
        fmt_ch = "H" if parsed.stride == 2 else "I"
        struct.pack_into(f"<IH{len(new_entries)}{fmt_ch}", buf, parsed.offset, new_active, len(new_entries), *new_entries)


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
    *,
    old_midi_to_new: Mapping[int, int] | None = None,
) -> bytes:
    """Remap ``0x2015`` track ordinals, ``0x1015`` playlist indices, and ``0x1058`` MIDI indices on surviving strips."""
    if len(strip) < 25 or strip[0] != 0x5A:
        return strip
    tag = struct.unpack_from("<H", strip, 7)[0]
    if tag != 0x261C:
        buf = bytearray(strip)
        scan_start = 9
        if tag in (0x261E, 0x261D) and len(strip) >= 18 and strip[9] == 0x5A and struct.unpack_from("<H", strip, 16)[0] == 0x261B:
            scan_start = 9 + 7 + struct.unpack_from("<I", strip, 12)[0]
        _patch_200b_view_ordinal(buf, scan_start - 9, len(buf), track_ordinal_1based)
        if tag in (0x2620, 0x2621):
            _remap_midi_strip_index(buf, old_midi_to_new)
        return bytes(buf)

    mid_off, active_pl, pl_cnt, stride, p200b, tail_off = _parse_261c_mid_gap(strip)
    buf = bytearray(strip)
    _patch_200b_view_ordinal(buf, p200b, tail_off, track_ordinal_1based)
    if old_pl_to_new is not None:
        if active_pl not in old_pl_to_new:
            raise PtxCorruptFileError(f"Surviving Audio strip references missing playlist index {active_pl}")
        new_active = old_pl_to_new[active_pl]
        struct.pack_into("<I", buf, mid_off, new_active)
        if pl_cnt:
            fmt = f"<{pl_cnt}H" if stride == 2 else f"<{pl_cnt}I"
            old_entries = struct.unpack_from(fmt, buf, mid_off + 6)
            new_entries = []
            for pl_i in old_entries:
                if pl_i not in old_pl_to_new:
                    raise PtxCorruptFileError(f"Surviving Audio strip references missing playlist index {pl_i}")
                new_entries.append(old_pl_to_new[pl_i])
            if stride == 2 and any(not (0 <= x <= 0xFFFF) for x in new_entries):
                raise PtxValidationError("Remapped 0x261C playlist index exceeds u16 range")
            struct.pack_into(fmt, buf, mid_off + 6, *new_entries)
        if tail_off + 21 <= len(buf) and struct.unpack_from("<I", buf, tail_off + 17)[0] == active_pl:
            struct.pack_into("<I", buf, tail_off + 17, new_active)
    return bytes(buf)


def _require_control_ids(next_id: int) -> None:
    if next_id + 10 > 0xFFFF_FFFF:
        raise PtxValidationError(f"0x2624 automation control ID space exhausted at {next_id}")


def _build_new_mixer_strip(nt: NewAudioTrackSpec, ord_1based: int, next_id: int) -> tuple[bytes, int]:
    tt = nt.track_type
    if tt == "Audio":
        _require_control_ids(next_id)
        return (
            build_audio_mixer_strip_261c(
                nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                track_ordinal_1based=ord_1based, id_base=next_id,
                playlist_index=nt.playlist_index, color_index=nt.color_index, track_format=nt.track_format,
            ),
            next_id + 10,
        )
    if tt in ("Aux", "RoutingFolder"):
        _require_control_ids(next_id)
        return (
            build_aux_or_rf_mixer_strip_261e(
                nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                is_routing_folder=(tt == "RoutingFolder"),
                track_ordinal_1based=ord_1based, id_base=next_id,
                color_index=nt.color_index, track_format=nt.track_format,
            ),
            next_id + 10,
        )
    if tt == "Master":
        _require_control_ids(next_id)
        return (
            build_master_mixer_strip_261d(
                nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                track_ordinal_1based=ord_1based, id_base=next_id,
                color_index=nt.color_index, track_format=nt.track_format,
            ),
            next_id + 10,
        )
    if tt == "Instrument":
        _require_control_ids(next_id)
        if nt.midi_index is None:
            raise PtxValidationError(f"Instrument track {nt.track_name!r} requires midi_index")
        return (
            build_instrument_mixer_strip_2621(
                nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                track_ordinal_1based=ord_1based, id_base=next_id,
                midi_index=nt.midi_index, color_index=nt.color_index,
            ),
            next_id + 10,
        )
    if tt == "Midi":
        if nt.midi_index is None:
            raise PtxValidationError(f"Midi track {nt.track_name!r} requires midi_index")
        return build_midi_mixer_strip_2620(
            nt.track_name, nt.track_uuid16,
            track_ordinal_1based=ord_1based, midi_index=nt.midi_index, color_index=nt.color_index,
        ), next_id
    if tt == "Vca":
        return build_vca_mixer_strip_261f(nt.track_name, nt.track_uuid16, track_ordinal_1based=ord_1based, color_index=nt.color_index), next_id
    if tt == "BasicFolder":
        return build_basic_folder_mixer_strip_263b(nt.track_name, nt.track_uuid16, track_ordinal_1based=ord_1based, color_index=nt.color_index), next_id
    raise PtxValidationError(f"Unsupported track_type for 0x2624 mixer strip synthesis: {tt!r}")


def mutate_2624_mixer(
    c2624_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
    old_playlist_to_new: Mapping[int, int] | None = None,
    old_midi_to_new: Mapping[int, int] | None = None,
) -> bytes:
    """Rebuild ``0x2624`` mixer strip container deterministically for *layout*."""
    if len(c2624_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x2624 mixer container chunk")
    version = struct.unpack_from("<H", c2624_raw, 1)[0]
    old_count = struct.unpack_from("<I", c2624_raw, 9)[0]
    old_strips, end_pos = walk_sequential_zchunks(c2624_raw, 13, old_count)
    if expected_old_count is not None and len(old_strips) != expected_old_count:
        raise PtxCorruptFileError(f"0x2624 strip count {len(old_strips)} != expected {expected_old_count}")

    next_id = _max_2624_control_id(old_strips)
    new_strips: list[bytes] = []
    for ord_1based, slot in enumerate(layout, start=1):
        if slot.old_index is not None:
            new_strips.append(
                _remap_261c_strip_indices(
                    old_strips[slot.old_index], old_playlist_to_new, ord_1based, old_midi_to_new=old_midi_to_new,
                )
            )
        elif slot.new_track is not None:
            strip_bytes, next_id = _build_new_mixer_strip(slot.new_track, ord_1based, next_id)
            new_strips.append(strip_bytes)
    body = struct.pack("<I", len(new_strips)) + b"".join(new_strips) + c2624_raw[end_pos:]
    return build_zchunk(0x2624, body, version=version)


def _remap_single_group_202a(g_raw: bytes, old_to_new: Mapping[int, int], new_track_count: int) -> bytes:
    if len(g_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x202A group chunk")
    ver = struct.unpack_from("<H", g_raw, 1)[0]
    nlen = struct.unpack_from("<I", g_raw, 9)[0]
    cnt_off = 13 + nlen + 7
    if cnt_off + 4 > len(g_raw):
        raise PtxCorruptFileError("Truncated 0x202A group header")
    name = g_raw[13 : 13 + nlen].decode("utf-8", "replace")
    mcnt = struct.unpack_from("<I", g_raw, cnt_off)[0]
    members_end = cnt_off + 4 + mcnt * 2
    if members_end > len(g_raw):
        raise PtxCorruptFileError("Truncated 0x202A group member array")

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


def mutate_2030_markers_for_track_map(c2030_raw: bytes, old_to_new: Mapping[int, int]) -> bytes:
    """Remap Track marker ``track_index`` fields in ``0x2030`` and drop markers on deleted tracks."""
    if len(c2030_raw) < 13:
        return c2030_raw
    ver = struct.unpack_from("<H", c2030_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2030_raw, 9)[0]
    if cnt == 0:
        return c2030_raw
    items, end_pos = walk_sequential_zchunks(c2030_raw, 13, cnt, expected_tag=0x2077)
    kept: list[bytes] = []
    for item in items:
        spans = find_subchunk_spans(item, 0x4826)
        if len(spans) >= 2 and spans[-2][0] + len(spans[-2][1]) == spans[-1][0]:
            p4826 = spans[-2][0]
        elif spans:
            p4826 = spans[-1][0]
        else:
            p4826 = -1
        if p4826 >= 35:
            old_tidx = struct.unpack_from("<i", item, p4826 - 4)[0]
            if old_tidx >= 0:
                if old_tidx not in old_to_new:
                    continue
                new_tidx = old_to_new[old_tidx]
                if new_tidx != old_tidx:
                    mbuf = bytearray(item)
                    struct.pack_into("<i", mbuf, p4826 - 4, new_tidx)
                    item = bytes(mbuf)
        kept.append(item)
    return build_zchunk(0x2030, struct.pack("<I", len(kept)) + b"".join(kept) + c2030_raw[end_pos:], version=ver)
