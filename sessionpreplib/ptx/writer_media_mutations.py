"""Media, clip, playlist, lane, and MIDI binary mutators (``0x0F3D``, ``0x1004``, ``0x262A``, ``0x1015``, ``0x1054``, ``0x1058``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import struct

from .codec import read_zchunk_at, walk_sequential_zchunks
from .models import PtxCorruptFileError, PtxUnsupportedTemplateError, PtxValidationError
from .writer_blueprints import (
    build_audio_file_entry_1003,
    build_midi_track_1057,
    build_playlist_1014_v9,
    build_volume_entry_0f3c,
    build_whole_file_clip_2629,
)
from .writer_chunks import (
    PtxDirTreeEntry,
    serialize_directory_tree_103a,
    serialize_lane_1052,
)
from .writer_primitives import build_zchunk
from .writer_track_mutations import (
    NewAudioTrackSpec,
    extract_261c_playlist_indices,
    parse_midi_strip_playlists,
)


@dataclass(frozen=True)
class MediaSurvivalPlan:
    """Compacted existing playlists, lanes, and playlist-index mapping for surviving template audio."""

    playlists: tuple[bytes, ...]
    lanes: tuple[bytes, ...]
    old_playlist_to_new: Mapping[int, int]
    c4829_trailer: bytes
    c4828_trailer: bytes


@dataclass(frozen=True)
class MidiSurvivalPlan:
    """Compacted existing ``0x1057`` MIDI entries and old->new index mapping for surviving MIDI/Instrument tracks."""

    entries: tuple[bytes, ...]
    old_midi_to_new: Mapping[int, int]
    trailer: bytes = b""


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
    """Whole-file or sub-clip info needed to populate ``0x262A``."""

    clip_name: str
    length_samples: int
    audio_file_index: int
    channel_index: int
    clip_guid16: bytes
    pt_uuid16: bytes
    start_sample: int = 0
    sync_point_samples: int = 0
    orig_timestamp: int = 0
    user_timestamp: int = 0


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
        raise PtxCorruptFileError("Truncated 0x103A directory tree chunk")
    ecnt = struct.unpack_from("<I", c103a_raw, 9)[0]
    pos = 13
    entries: list[PtxDirTreeEntry] = []
    for _ in range(ecnt):
        if pos + 13 > len(c103a_raw):
            raise PtxCorruptFileError("Truncated 0x103A entry header")
        kind = c103a_raw[pos]
        parent = struct.unpack_from("<I", c103a_raw, pos + 1)[0]
        nlen = struct.unpack_from("<I", c103a_raw, pos + 5)[0]
        pos += 9
        if pos + nlen + 4 > len(c103a_raw):
            raise PtxCorruptFileError("Truncated 0x103A entry name/tag")
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
            raise PtxValidationError(f"Duplicate audio filename {sf_item.filename!r} in staged audio files")
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
    existing_files = {e.name.lower() for e in entries if e.kind == 2 and e.parent_index == parent_idx}
    file_indices = []
    for sf_item in staged_files:
        low_fn = sf_item.filename.lower()
        if low_fn in existing_files:
            raise PtxValidationError(f"Duplicate audio filename {sf_item.filename!r} under session Audio Files directory")
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
        raise PtxCorruptFileError("Truncated 0x1004 audio file table chunk")
    version = struct.unpack_from("<H", c1004_raw, 1)[0]
    root_path, dir_parts = _split_session_dir_parts(session_dir)
    if not staged_files:
        return c1004_raw, root_path

    old_file_count = struct.unpack_from("<I", c1004_raw, 9)[0]
    _tag_tree, old_103a_raw, pos = read_zchunk_at(c1004_raw, 13, expected_tag=0x103A)
    old_103a_ver = struct.unpack_from("<H", old_103a_raw, 1)[0]
    old_1003_list, after_1003 = walk_sequential_zchunks(c1004_raw, pos, old_file_count, expected_tag=0x1003)
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
        raise PtxCorruptFileError("Truncated 0x0F3D volume table chunk")
    version = struct.unpack_from("<H", c0f3d_raw, 1)[0]
    old_count = struct.unpack_from("<I", c0f3d_raw, 9)[0]
    old_entries, end_pos = walk_sequential_zchunks(c0f3d_raw, 13, old_count, expected_tag=0x0F3C)
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
                    raise PtxCorruptFileError("Truncated 0x0F3C volume root path")
                rpath = entry[ser_off + 8 : ser_off + 8 + rlen].decode("utf-8", "replace").lower()
                if ser == target_serial and rpath == target_root:
                    return c0f3d_raw
    new_entries = [*old_entries, build_volume_entry_0f3c(target_serial, volume_label, root_path)]
    body = struct.pack("<I", len(new_entries)) + b"".join(new_entries) + c0f3d_raw[end_pos:]
    return build_zchunk(0x0F3D, body, version=version)


def mutate_262a_clips(c262a_raw: bytes, new_clips: Sequence[ClipWriteSpec]) -> bytes:
    """Append new whole-file or sub-clip ``0x2629`` entries into ``0x262A`` while preserving existing clips."""
    if len(c262a_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x262A clip container chunk")
    version = struct.unpack_from("<H", c262a_raw, 1)[0]
    old_count = struct.unpack_from("<I", c262a_raw, 9)[0]
    old_entries, end_pos = walk_sequential_zchunks(c262a_raw, 13, old_count, expected_tag=0x2629)
    added = [
        build_whole_file_clip_2629(
            c.clip_name, c.length_samples,
            audio_file_index=c.audio_file_index, channel_index=c.channel_index,
            clip_guid16=c.clip_guid16, pt_uuid16=c.pt_uuid16,
            start_sample=c.start_sample, sync_point_samples=c.sync_point_samples,
            orig_timestamp=c.orig_timestamp, user_timestamp=c.user_timestamp,
        )
        for c in new_clips
    ]
    all_entries = [*old_entries, *added]
    body = struct.pack("<I", len(all_entries)) + b"".join(all_entries) + c262a_raw[end_pos:]
    return build_zchunk(0x262A, body, version=version)


def _extract_counted_items_and_trailer(
    container_raw: bytes, item_tag: int, trailer_tag: int, default_trailer: bytes
) -> tuple[list[bytes], bytes]:
    if len(container_raw) < 13:
        return [], default_trailer
    cnt = struct.unpack_from("<I", container_raw, 9)[0]
    items, after = walk_sequential_zchunks(container_raw, 13, cnt, expected_tag=item_tag)
    if after + 9 <= len(container_raw) and container_raw[after] == 0x5A:
        tag, raw_sub, _end = read_zchunk_at(container_raw, after)
        if tag == trailer_tag:
            return items, raw_sub
    return items, default_trailer


def _parse_playlist_1014_meta(raw_1014: bytes) -> tuple[str, int, tuple[int, ...], int]:
    """Return ``(name, is_stereo_byte, all_lane_indices, lane_array_offset)`` for a ``0x1014`` chunk."""
    if len(raw_1014) < 18 or raw_1014[0] != 0x5A:
        raise PtxCorruptFileError("Truncated 0x1014 playlist chunk")
    ver = struct.unpack_from("<H", raw_1014, 1)[0]
    nlen = struct.unpack_from("<I", raw_1014, 9)[0]
    st_off = 13 + nlen
    if st_off + 5 > len(raw_1014):
        raise PtxCorruptFileError("Truncated 0x1014 playlist header")
    name = raw_1014[13:st_off].decode("utf-8", "replace")
    is_st = raw_1014[st_off]
    lcnt = struct.unpack_from("<I", raw_1014, st_off + 1)[0]
    arr_off = st_off + 5
    if lcnt > (len(raw_1014) - arr_off) // 2:
        raise PtxCorruptFileError(f"Truncated 0x1014 lane array for playlist {name!r}")
    lanes = list(struct.unpack_from(f"<{lcnt}H", raw_1014, arr_off)) if lcnt else []
    ext_hdr_off = arr_off + lcnt * 2 + 16
    if ver >= 9 and ext_hdr_off + 7 <= len(raw_1014):
        ext_a, ext_b, extra_cnt = struct.unpack_from("<HBI", raw_1014, ext_hdr_off)
        p_ext = ext_hdr_off + 7
        if ext_b == 0x11 and 0 < extra_cnt <= 32 and p_ext + extra_cnt * 2 <= len(raw_1014):
            lanes.extend(struct.unpack_from(f"<{extra_cnt}H", raw_1014, p_ext))
        elif ext_a == 0x0A and ext_b == 0x0E and extra_cnt == 0:
            p_hi = p_ext + lcnt * 4
            if p_hi + 5 <= len(raw_1014) and raw_1014[p_hi] == 0x1E:
                h_cnt = struct.unpack_from("<I", raw_1014, p_hi + 1)[0]
                if 0 < h_cnt <= 32 and p_hi + 5 + h_cnt * 4 <= len(raw_1014):
                    lanes.extend(struct.unpack_from(f"<{h_cnt}I", raw_1014, p_hi + 5))
    return name, is_st, tuple(lanes), arr_off


def _remap_playlist_1014_lanes(raw_1014: bytes, old_lane_to_new: Mapping[int, int]) -> bytes:
    """Remap all ``<H`` and v9 trailing ``<I`` lane indices (including 7.1.2 / 7.1.4 height lanes)."""
    name, _is_st, old_lanes, arr_off = _parse_playlist_1014_meta(raw_1014)
    missing = [li for li in old_lanes if li not in old_lane_to_new]
    if missing:
        raise PtxCorruptFileError(f"Surviving 0x1014 playlist {name!r} references missing lane(s): {missing}")
    new_lanes = tuple(old_lane_to_new[li] for li in old_lanes)
    if new_lanes == old_lanes:
        return raw_1014
    if any(not (0 <= li <= 0xFFFF) for li in new_lanes):
        raise PtxValidationError(f"Remapped 0x1014 lane index out of u16 range for playlist {name!r}")
    buf = bytearray(raw_1014)
    lcnt = struct.unpack_from("<I", buf, arr_off - 4)[0]
    struct.pack_into(f"<{lcnt}H", buf, arr_off, *new_lanes[:lcnt])
    ver = struct.unpack_from("<H", buf, 1)[0]
    ext_hdr_off = arr_off + lcnt * 2 + 16
    if ver >= 9 and old_lanes and ext_hdr_off + 7 <= len(buf):
        ext_a, ext_b, extra_cnt = struct.unpack_from("<HBI", buf, ext_hdr_off)
        p_ext = ext_hdr_off + 7
        if ext_b == 0x11 and extra_cnt == len(old_lanes) - lcnt:
            struct.pack_into(f"<{extra_cnt}H", buf, p_ext, *new_lanes[lcnt:])
            p_i = p_ext + extra_cnt * 2
            if p_i + len(old_lanes) * 4 <= len(buf) and struct.unpack_from(f"<{len(old_lanes)}I", buf, p_i) == old_lanes:
                struct.pack_into(f"<{len(new_lanes)}I", buf, p_i, *new_lanes)
        elif ext_a == 0x0A and ext_b == 0x0E and extra_cnt == 0 and len(old_lanes) > lcnt:
            if p_ext + lcnt * 4 <= len(buf) and struct.unpack_from(f"<{lcnt}I", buf, p_ext) == old_lanes[:lcnt]:
                struct.pack_into(f"<{lcnt}I", buf, p_ext, *new_lanes[:lcnt])
            p_hi = p_ext + lcnt * 4
            if p_hi + 5 <= len(buf) and buf[p_hi] == 0x1E:
                h_cnt = struct.unpack_from("<I", buf, p_hi + 1)[0]
                if h_cnt == len(old_lanes) - lcnt and p_hi + 5 + h_cnt * 4 <= len(buf):
                    struct.pack_into(f"<{h_cnt}I", buf, p_hi + 5, *new_lanes[lcnt:])
        elif p_ext + len(old_lanes) * 4 <= len(buf) and struct.unpack_from(f"<{len(old_lanes)}I", buf, p_ext) == old_lanes:
            struct.pack_into(f"<{len(new_lanes)}I", buf, p_ext, *new_lanes)
    return bytes(buf)


def _partition_2624_playlist_indices(c2624_raw: bytes, deleted_old_indices: Sequence[int]) -> tuple[set[int], set[int]]:
    if len(c2624_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x2624 mixer container chunk")
    old_count = struct.unpack_from("<I", c2624_raw, 9)[0]
    old_strips, _ = walk_sequential_zchunks(c2624_raw, 13, old_count)
    del_set = set(deleted_old_indices)
    dropped: set[int] = set()
    surviving: set[int] = set()
    for idx, strip in enumerate(old_strips):
        pl_ids = extract_261c_playlist_indices(strip)
        (dropped if idx in del_set else surviving).update(pl_ids)
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
        playlists=remapped_playlists, lanes=tuple(surviving_lanes),
        old_playlist_to_new=old_pl_to_new, c4829_trailer=c4829, c4828_trailer=c4828,
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
    audio_tracks = [nt for nt in new_tracks if nt.track_type == "Audio"]
    added_playlists = [
        build_playlist_1014_v9(
            nt.track_name, nt.lane_indices, nt.playlist_uuid16,
            is_stereo=(nt.channels >= 2), track_format=nt.track_format,
        )
        for nt in audio_tracks
    ]
    added_lanes: list[bytes] = []
    for nt in audio_tracks:
        if nt.clip_indices and len(nt.clip_indices) != len(nt.lane_indices):
            raise PtxValidationError(
                f"Track {nt.track_name!r} clip_indices count ({len(nt.clip_indices)}) != lane_indices count ({len(nt.lane_indices)})"
            )
        if nt.clip_indices:
            for clip_idx in nt.clip_indices:
                added_lanes.append(
                    serialize_lane_1052(
                        nt.track_name,
                        [(clip_idx, nt.start_sample, 3, nt.clip_color_index, nt.clip_muted, nt.clip_time_locked, nt.clip_edit_locked)],
                        wrap_1050=True, wrap_version=3, lane_version=3,
                    )
                )
        else:
            for _ in nt.lane_indices:
                added_lanes.append(serialize_lane_1052(nt.track_name, (), wrap_1050=True, wrap_version=3, lane_version=3))
    all_playlists = [*survival.playlists, *added_playlists]
    all_lanes = [*survival.lanes, *added_lanes]
    out_1015 = build_zchunk(
        0x1015, struct.pack("<I", len(all_playlists)) + b"".join(all_playlists) + survival.c4829_trailer, version=ver_1015,
    )
    out_1054 = build_zchunk(
        0x1054, struct.pack("<I", len(all_lanes)) + b"".join(all_lanes) + survival.c4828_trailer, version=ver_1054,
    )
    return out_1015, out_1054


def plan_existing_midi_survival(
    c1058_raw: bytes,
    c2624_raw: bytes,
    deleted_old_indices: Sequence[int],
) -> MidiSurvivalPlan:
    """Compact existing ``0x1057`` MIDI entries for any deleted template MIDI/Instrument tracks (incl. ``pl_cnt > 1``)."""
    if len(c1058_raw) < 13:
        return MidiSurvivalPlan(entries=(), old_midi_to_new={}, trailer=b"")
    old_cnt = struct.unpack_from("<I", c1058_raw, 9)[0]
    old_entries, end_pos = walk_sequential_zchunks(c1058_raw, 13, old_cnt, expected_tag=0x1057)
    trailer = c1058_raw[end_pos:]
    if not old_entries or not deleted_old_indices or len(c2624_raw) < 13:
        return MidiSurvivalPlan(
            entries=tuple(old_entries),
            old_midi_to_new={i: i for i in range(len(old_entries))},
            trailer=trailer,
        )
    s_cnt = struct.unpack_from("<I", c2624_raw, 9)[0]
    old_strips, _ = walk_sequential_zchunks(c2624_raw, 13, s_cnt)
    del_set = set(deleted_old_indices)
    dropped: set[int] = set()
    surviving: set[int] = set()
    for idx, strip in enumerate(old_strips):
        tag = struct.unpack_from("<H", strip, 7)[0] if len(strip) >= 9 else 0
        if tag not in (0x2620, 0x2621):
            continue
        parsed = parse_midi_strip_playlists(strip)
        if parsed is None:
            raise PtxUnsupportedTemplateError(
                f"Unrecognized 0x{tag:04x} MIDI/Instrument playlist layout on track slot {idx} during track deletion"
            )
        all_ids = {parsed.active_index, *parsed.playlist_indices}
        (dropped if idx in del_set else surviving).update(all_ids)
    removable = dropped - surviving
    kept: list[bytes] = []
    old_to_new: dict[int, int] = {}
    for i, entry in enumerate(old_entries):
        if i in removable:
            continue
        old_to_new[i] = len(kept)
        kept.append(entry)
    return MidiSurvivalPlan(entries=tuple(kept), old_midi_to_new=old_to_new, trailer=trailer)


def mutate_1058_midi_tracks(
    c1058_raw: bytes,
    new_tracks: Sequence[NewAudioTrackSpec],
    *,
    midi_survival: MidiSurvivalPlan | None = None,
) -> bytes:
    """Rebuild ``0x1058`` with surviving and newly synthesized MIDI/Instrument ``0x1057`` entries."""
    if len(c1058_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x1058 MIDI playlist container chunk")
    version = struct.unpack_from("<H", c1058_raw, 1)[0]
    if midi_survival is not None:
        base_entries = list(midi_survival.entries)
        trailer = midi_survival.trailer
    else:
        old_count = struct.unpack_from("<I", c1058_raw, 9)[0]
        base_entries, end_pos = walk_sequential_zchunks(c1058_raw, 13, old_count, expected_tag=0x1057)
        trailer = c1058_raw[end_pos:]
    midi_tracks = [nt for nt in new_tracks if nt.track_type in ("Midi", "Instrument")]
    added = [build_midi_track_1057(nt.track_name, nt.midi_uuid16) for nt in midi_tracks]
    all_entries = [*base_entries, *added]
    body = struct.pack("<I", len(all_entries)) + b"".join(all_entries) + trailer
    return build_zchunk(0x1058, body, version=version)
