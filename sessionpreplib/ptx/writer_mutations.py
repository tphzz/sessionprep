"""Per-chunk binary mutators for adding/removing tracks, audio, clips, and placements in ``.ptx``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import struct

from .parser import split_202b
from .writer_blueprints import (
    AUDIO_UI_2589_RAW,
    _find_zchunk_offset,
    build_audio_file_entry_1003,
    build_audio_mixer_strip_261c,
    build_playlist_1014_v9,
    build_track_entry_251a,
    build_track_summary_2519,
    build_track_uuid_entry_210b,
    build_whole_file_clip_2629,
)
from .writer_chunks import (
    PtxDirTreeEntry,
    serialize_directory_tree_103a,
    serialize_lane_1052,
)
from .writer_primitives import build_zchunk, serialize_folder_tree_210c


@dataclass(frozen=True)
class NewAudioTrackSpec:
    """Internal spec for one newly synthesized Audio track during ``.ptx`` writing."""

    track_name: str
    track_uuid16: bytes
    playlist_uuid16: bytes
    channels: int
    parent_folder_uuid16: bytes
    output_bus: tuple[int, bytes, str] | None  # (ptsl_index, bus_uuid16, bus_name)
    color_index: int
    clip_indices: tuple[int, ...]  # 1 index (Mono) or 2 indices (Stereo L, R) in 0x262A
    lane_indices: tuple[int, ...]  # 1 index (Mono) or 2 indices (Stereo L, R) in 0x1054


@dataclass(frozen=True)
class TrackSlotOp:
    """Unified output track slot: either keep existing template track *old_index* or insert *new_track*."""

    old_index: int | None = None
    new_track: NewAudioTrackSpec | None = None


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


def _read_zchunk_at(raw: bytes, pos: int, *, expected_tag: int | None = None) -> tuple[int, bytes, int]:
    if pos < 0 or pos + 9 > len(raw) or raw[pos] != 0x5A:
        raise RuntimeError(f"Expected Z-chunk header at offset 0x{pos:x}")
    sz = struct.unpack_from("<I", raw, pos + 3)[0]
    tag = struct.unpack_from("<H", raw, pos + 7)[0]
    end = pos + 7 + sz
    if sz < 2 or end > len(raw):
        raise RuntimeError(f"Truncated Z-chunk 0x{tag:04x} at offset 0x{pos:x}")
    if expected_tag is not None and tag != expected_tag:
        raise RuntimeError(f"Expected Z-chunk 0x{expected_tag:04x} at 0x{pos:x}, found 0x{tag:04x}")
    return tag, raw[pos:end], end


def _walk_sequential_zchunks(
    raw: bytes,
    start: int,
    count: int,
    *,
    expected_tag: int | None = None,
) -> tuple[list[bytes], int]:
    items: list[bytes] = []
    pos = start
    for _ in range(count):
        _tag, chunk_raw, pos = _read_zchunk_at(raw, pos, expected_tag=expected_tag)
        items.append(chunk_raw)
    return items, pos


def _parse_2519_sections(raw: bytes) -> tuple[bytes, list[bytes], list[bytes], list[bytes], bytes]:
    """Deterministically parse ``0x2519`` into ``(clean_prefix, rows, pass1, pass2, tail)``."""
    if len(raw) < 27 or raw[0] != 0x5A:
        raise RuntimeError("Truncated 0x2519 track directory chunk")
    pos = 9
    hdr10 = raw[pos : pos + 10]  # v0 (u16) + v1 (u32) + v2 (u32)
    pos += 10
    cnt_a = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    if cnt_a > (len(raw) - pos) // 2:
        raise RuntimeError(f"0x2519 header cnt_a ({cnt_a}) exceeds chunk bounds")
    pos += cnt_a * 2
    # Reset cnt_a to 0 so stale track-selection indices are never left pointing out of bounds.
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
    pass1, pos = _walk_sequential_zchunks(raw, pos, tcnt, expected_tag=0x251A)
    if pos + 4 > len(raw):
        raise RuntimeError("Truncated 0x2519 Pass 2 track count")
    tcnt2 = struct.unpack_from("<I", raw, pos)[0]
    if tcnt2 != tcnt:
        raise RuntimeError(f"0x2519 Pass 2 count {tcnt2} != Pass 1 count {tcnt}")
    pos += 4
    pass2, pos = _walk_sequential_zchunks(raw, pos, tcnt2, expected_tag=0x251A)
    return clean_prefix, rows, pass1, pass2, raw[pos:]


def _patch_2519_summary_order(row: bytes, visible_order: int) -> bytes:
    buf = bytearray(row)
    struct.pack_into("<H", buf, len(buf) - 3, visible_order)
    return bytes(buf)


def _patch_251a_order(raw_251a: bytes, visible_order: int) -> bytes:
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
    """Rebuild ``0x2519`` (Table 1 + Pass 1 & Pass 2 ``0x251A``) for *layout*."""
    version = struct.unpack_from("<H", c2519_raw, 1)[0]
    prefix, old_rows, old_p1, old_p2, tail_raw = _parse_2519_sections(c2519_raw)
    if expected_old_count is not None and len(old_rows) != expected_old_count:
        raise RuntimeError(f"0x2519 track count {len(old_rows)} != expected {expected_old_count}")

    new_rows: list[bytes] = []
    new_p1: list[bytes] = []
    new_p2: list[bytes] = []
    for visible_ord, slot in enumerate(layout, start=1):
        if slot.old_index is not None:
            new_rows.append(_patch_2519_summary_order(old_rows[slot.old_index], visible_ord))
            new_p1.append(_patch_251a_order(old_p1[slot.old_index], visible_ord))
            new_p2.append(_patch_251a_order(old_p2[slot.old_index], visible_ord))
        elif slot.new_track is not None:
            nt = slot.new_track
            new_rows.append(build_track_summary_2519(nt.track_name, nt.track_uuid16, nt.channels, visible_ord))
            entry = build_track_entry_251a(
                nt.track_name, nt.track_uuid16, nt.channels, visible_ord,
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
    c2551_off = _find_zchunk_offset(c2587_raw, 0x2551, 9)
    c258b_off = _find_zchunk_offset(c2587_raw, 0x258B, c2551_off + 9) if c2551_off >= 0 else -1
    c258a_off = _find_zchunk_offset(c2587_raw, 0x258A, c258b_off + 9) if c258b_off >= 0 else -1
    if min(c2551_off, c258b_off, c258a_off) < 0:
        return c2587_raw

    _tag81, _raw81, after_81 = _read_zchunk_at(c2587_raw, c258a_off + 9, expected_tag=0x2581)
    if after_81 + 4 > len(c2587_raw):
        raise RuntimeError("Truncated 0x258A track UI entry count")
    old_count = struct.unpack_from("<I", c2587_raw, after_81)[0]
    first_off = after_81 + 4
    old_2589, last_end = _walk_sequential_zchunks(c2587_raw, first_off, old_count, expected_tag=0x2589)
    if expected_old_count is not None and len(old_2589) != expected_old_count:
        raise RuntimeError(f"0x258A UI track count {len(old_2589)} != expected {expected_old_count}")

    new_items = [
        old_2589[slot.old_index] if slot.old_index is not None else AUDIO_UI_2589_RAW
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
        p485a = _find_zchunk_offset(strip, 0x485A, 9)
        if p485a < 0:
            continue
        p10 = strip.find(b"\x01\x01\x0a\x00", p485a)
        if p10 >= 0 and p10 + 44 <= len(strip):
            for val in struct.unpack_from("<10I", strip, p10 + 4):
                if val < 0xFFFF_FF00 and val > max_id:
                    max_id = val
    return max_id + 16


def mutate_2624_mixer(
    c2624_raw: bytes,
    layout: Sequence[TrackSlotOp],
    *,
    expected_old_count: int | None = None,
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
            new_strips.append(old_strips[slot.old_index])
        elif slot.new_track is not None:
            nt = slot.new_track
            if next_id + 10 > 0xFFFF_FFFE:
                raise OverflowError(f"0x2624 control ID counter overflow: {next_id}")
            new_strips.append(
                build_audio_mixer_strip_261c(
                    nt.track_name, nt.track_uuid16, nt.channels, nt.output_bus,
                    track_ordinal_1based=ord_1based, id_base=next_id,
                    color_index=nt.color_index,
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
    resolved = session_dir.resolve()
    parts = list(resolved.parts)
    if not parts:
        return "/", []
    root = parts[0]
    if len(root) >= 2 and root[1] == ":":
        root = root[0].lower() + ":\\"
    return root, parts[1:]


def build_1004_audio_files(
    c1004_raw: bytes,
    staged_files: Sequence[AudioFileWriteSpec],
    *,
    session_dir: Path,
    volume_serial_u32: int,
) -> tuple[bytes, str]:
    """Build ``0x1004`` audio-file directory tree (``0x103A``) and descriptors (``0x1003``)."""
    version = struct.unpack_from("<H", c1004_raw, 1)[0]
    root_path, dir_parts = _split_session_dir_parts(session_dir)
    if not staged_files:
        return c1004_raw, root_path

    vol_idx = 1 + len(staged_files)
    session_dir_idx = vol_idx + len(dir_parts)
    entries: list[PtxDirTreeEntry] = [
        PtxDirTreeEntry(kind=1, parent_index=session_dir_idx, name="Audio Files"),
    ]
    for sf_item in staged_files:
        entries.append(PtxDirTreeEntry(kind=2, parent_index=0, name=sf_item.filename, tag4=b"EVAW"))
    entries.append(
        PtxDirTreeEntry(
            kind=0,
            parent_index=0xFFFF_FFFF,
            name=root_path,
            tag4=struct.pack("<I", volume_serial_u32 & 0xFFFF_FFFF),
        )
    )
    for idx, part in enumerate(dir_parts):
        entries.append(PtxDirTreeEntry(kind=1, parent_index=vol_idx + idx, name=part))

    c103a = serialize_directory_tree_103a(entries, version=1)
    c1003_list = [
        build_audio_file_entry_1003(
            item.sample_rate, item.channels, item.bit_depth, item.length_samples,
            file_id8=item.file_id8, smpte_umid32=item.smpte_umid32,
            filetime_u64=item.filetime_u64, file_index=file_idx,
        )
        for file_idx, item in enumerate(staged_files)
    ]
    empty_trailing_103a = serialize_directory_tree_103a([], version=1)
    body = (
        struct.pack("<I", len(staged_files))
        + c103a
        + b"".join(c1003_list)
        + b"\x00\x00\x00\x00"
        + empty_trailing_103a
    )
    return build_zchunk(0x1004, body, version=version), root_path


def build_262a_clips(c262a_raw: bytes, clips: Sequence[ClipWriteSpec]) -> bytes:
    """Build ``0x262A`` whole-file clip container."""
    version = struct.unpack_from("<H", c262a_raw, 1)[0]
    entries = [
        build_whole_file_clip_2629(
            c.clip_name, c.length_samples,
            audio_file_index=c.audio_file_index, channel_index=c.channel_index,
            clip_guid16=c.clip_guid16, pt_uuid16=c.pt_uuid16,
        )
        for c in clips
    ]
    body = struct.pack("<I", len(entries)) + b"".join(entries)
    return build_zchunk(0x262A, body, version=version)


def _extract_counted_trailing_subchunk(
    container_raw: bytes,
    item_tag: int,
    trailer_tag: int,
    default_trailer: bytes,
) -> bytes:
    if len(container_raw) < 13:
        return default_trailer
    cnt = struct.unpack_from("<I", container_raw, 9)[0]
    _items, after = _walk_sequential_zchunks(container_raw, 13, cnt, expected_tag=item_tag)
    if after + 9 <= len(container_raw) and container_raw[after] == 0x5A:
        tag, raw_sub, _end = _read_zchunk_at(container_raw, after)
        if tag == trailer_tag:
            return raw_sub
    return default_trailer


def build_1015_playlists(c1015_raw: bytes, new_tracks: Sequence[NewAudioTrackSpec]) -> bytes:
    """Build ``0x1015`` playlists container preserving trailing ``0x4829`` subchunk."""
    version = struct.unpack_from("<H", c1015_raw, 1)[0]
    default_4829 = build_zchunk(0x4829, b"\x00\x00\x00\x00", version=2)
    c4829 = _extract_counted_trailing_subchunk(c1015_raw, 0x1014, 0x4829, default_4829)
    entries = [
        build_playlist_1014_v9(
            nt.track_name, nt.lane_indices, nt.playlist_uuid16,
            is_stereo=(nt.channels >= 2), playlist_ordinal_1based=idx,
        )
        for idx, nt in enumerate(new_tracks, start=1)
    ]
    body = struct.pack("<I", len(entries)) + b"".join(entries) + c4829
    return build_zchunk(0x1015, body, version=version)


def build_1054_lanes(c1054_raw: bytes, new_tracks: Sequence[NewAudioTrackSpec]) -> bytes:
    """Build ``0x1054`` channel lanes with sample-0 whole-file clip placements."""
    version = struct.unpack_from("<H", c1054_raw, 1)[0]
    default_4828 = build_zchunk(0x4828, b"\x00\x00\x00\x00", version=2)
    c4828 = _extract_counted_trailing_subchunk(c1054_raw, 0x1052, 0x4828, default_4828)
    lanes: list[bytes] = []
    for nt in new_tracks:
        for clip_idx in nt.clip_indices:
            lanes.append(
                serialize_lane_1052(
                    nt.track_name,
                    [(clip_idx, 0, 3, 0xFFFE)],
                    wrap_1050=True, wrap_version=3, lane_version=3,
                )
            )
    body = struct.pack("<I", len(lanes)) + b"".join(lanes) + c4828
    return build_zchunk(0x1054, body, version=version)
