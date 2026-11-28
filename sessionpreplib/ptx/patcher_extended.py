"""Extended ``.ptx`` patchers: session settings, memory locations/markers, track states, and track renames."""

from __future__ import annotations

from collections.abc import Mapping
import struct
from typing import Any

from .codec import (
    decrypt_ptx,
    encrypt_ptx,
    find_subchunks,
    find_zchunk_offset,
    iter_top_chunks,
    parse_control_curve,
    splice_top_chunk,
    walk_sequential_zchunks,
)
from .models import (
    TRACK_HEIGHT_PRESET_PX,
    UUID_PREFIX,
    ZERO_TICKS,
    PtxTrack,
)
from .parser import PtxSession
from .writer_chunks import (
    patch_session_audio_format_1028_decrypted,
    patch_session_timecode_204d_decrypted,
    patch_session_ui_state_decrypted,
    serialize_fade_262f,
    serialize_marker_2077,
    serialize_placement_104f,
)
from .writer_index import apply_rebuilt_index_0002
from .writer_mutations import _parse_2519_sections
from .writer_primitives import build_zchunk, encode_lp_str, parse_index_0002

_TRACK_STATE_KEYS = frozenset({
    "mute", "solo", "solo_safe", "inactive", "hidden", "folder_open", "color_index", "height", "timebase",
})
_HEIGHT_PRESETS_LC = {k.lower(): (k, v) for k, v in TRACK_HEIGHT_PRESET_PX.items()}
_NO_MUTE_TYPES = frozenset({"Master", "BasicFolder"})
_NO_SOLO_TYPES = frozenset({"Master", "BasicFolder"})
_NO_SOLO_SAFE_TYPES = frozenset({"Master", "BasicFolder"})


def _resolve_height_px(height: str | int) -> tuple[int, int]:
    if isinstance(height, bool):
        raise ValueError(f"Invalid track height: {height!r}")
    if isinstance(height, int):
        if not (16 <= height <= 2000):
            raise ValueError(f"Track height_px out of range 16..2000: {height}")
        return height, height
    hit = _HEIGHT_PRESETS_LC.get(str(height).strip().lower())
    if hit is None:
        raise ValueError(f"Unsupported track height preset: {height!r} (valid: {list(TRACK_HEIGHT_PRESET_PX)})")
    return hit[1]


def patch_session_settings(
    ptx_raw: bytes,
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
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    changes: dict[str, Any] = {}
    audio_kw = {
        "sample_rate": sample_rate, "bit_depth": bit_depth, "audio_format": audio_format,
        "is_interleaved": is_interleaved, "audio_pull_rate": audio_pull_rate, "video_pull_rate": video_pull_rate,
    }
    if any(v is not None for v in audio_kw.values()):
        patch_session_audio_format_1028_decrypted(buf, **audio_kw)  # type: ignore[arg-type]
        changes.update({k: v for k, v in audio_kw.items() if v is not None})
    tc_kw = {
        "fps_code": fps_code, "start_frames": start_frames,
        "feet_frames_rate": feet_frames_rate, "session_length": session_length,
    }
    if any(v is not None for v in tc_kw.values()):
        patch_session_timecode_204d_decrypted(buf, **tc_kw)  # type: ignore[arg-type]
        changes.update({k: v for k, v in tc_kw.items() if v is not None})
    ui_kw = {
        "main_counter_format": main_counter_format, "sub_counter_format": sub_counter_format,
        "edit_mode": edit_mode, "grid_relative": grid_relative,
        "edit_tool": edit_tool, "smart_tool_enabled": smart_tool_enabled,
        "pre_roll_enabled": pre_roll_enabled, "pre_roll_samples": pre_roll_samples,
        "post_roll_enabled": post_roll_enabled, "post_roll_samples": post_roll_samples,
        "timeline_in_sample": timeline_in_sample, "timeline_out_sample": timeline_out_sample,
    }
    if any(v is not None for v in ui_kw.values()):
        patch_session_ui_state_decrypted(buf, **ui_kw)  # type: ignore[arg-type]
        changes.update({k: v for k, v in ui_kw.items() if v is not None})
    return encrypt_ptx(bytes(buf), delta, k0=k0), changes


def _split_2030_markers(c2030_raw: bytes) -> tuple[int, list[bytes], bytes]:
    if len(c2030_raw) < 13:
        raise RuntimeError("Truncated 0x2030 markers chunk in .ptx")
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
    track: str | int | None,
) -> tuple[int, int, str]:
    used_nums = {m.number for m in session.markers}
    if number is None:
        resolved_num = next((n for n in range(1, 1000) if n not in used_nums), None)
        if resolved_num is None:
            raise ValueError("All 999 Pro Tools memory location numbers are in use")
    else:
        if number in used_nums:
            raise ValueError(f"Memory location number {number} already exists in session")
        resolved_num = number
    if track is None:
        return resolved_num, -1, ""
    if isinstance(track, int) and not isinstance(track, bool):
        if not (0 <= track < len(session.tracks)):
            raise IndexError(f"Marker track index {track} out of range")
        return resolved_num, track, session.tracks[track].name
    trk = session.track_by_name(str(track))
    if trk is None:
        raise KeyError(f"Track {track!r} not found in .ptx session")
    return resolved_num, trk.index, trk.name


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
    track: str | int | None = None,
    zoom_settings: bool = False,
    pre_post_roll_times: bool = False,
    track_visibility: bool = False,
    track_heights: bool = False,
    group_enables: bool = False,
    window_configuration: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Add a Memory Location / Marker to ``0x2030`` (Main Ruler or Track marker)."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    resolved_num, track_idx, track_name = _resolve_marker_target(session, number, track)
    c2030 = session.chunk_map.get(0x2030)
    if c2030 is None:
        raise RuntimeError("Missing 0x2030 markers chunk in .ptx")
    ver, items, trailer = _split_2030_markers(c2030.raw)
    new_2077 = serialize_marker_2077(
        resolved_num, name, start_pos,
        end_pos=end_pos, reference=reference, time_properties=time_properties,
        comments=comments, color_index=color_index, color_rgb=color_rgb, track_index=track_idx,
        zoom_settings=zoom_settings, pre_post_roll_times=pre_post_roll_times,
        track_visibility=track_visibility, track_heights=track_heights,
        group_enables=group_enables, window_configuration=window_configuration,
    )
    items.append(new_2077)
    buf = bytearray(dec)
    splice_top_chunk(buf, 0x2030, _build_2030_chunk(items, version=ver, trailer=trailer))
    return encrypt_ptx(bytes(buf), delta, k0=k0), {
        "action": "add_marker", "number": resolved_num, "name": name,
        "location": "Track" if track_idx >= 0 else "MainRuler", "track_name": track_name,
    }


def patch_delete_marker(ptx_raw: bytes, number: int) -> tuple[bytes, dict[str, Any]]:
    """Delete an existing Memory Location / Marker by its 1-based number from ``0x2030``."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    c2030 = session.chunk_map.get(0x2030)
    if c2030 is None:
        raise RuntimeError("Missing 0x2030 markers chunk in .ptx")
    ver, items, trailer = _split_2030_markers(c2030.raw)
    kept: list[bytes] = []
    removed = False
    for it in items:
        if len(it) >= 11 and struct.unpack_from("<H", it, 9)[0] == number:
            removed = True
        else:
            kept.append(it)
    if not removed:
        raise KeyError(f"Marker #{number} not found in .ptx session")
    buf = bytearray(dec)
    splice_top_chunk(buf, 0x2030, _build_2030_chunk(kept, version=ver, trailer=trailer))
    return encrypt_ptx(bytes(buf), delta, k0=k0), {"action": "delete_marker", "number": number}


def _find_second_curve_offset(container: bytes, curves: list[bytes]) -> int:
    p0 = container.find(curves[0], 9)
    if p0 < 0:
        return -1
    return container.find(curves[1], p0 + len(curves[0]))


def _patch_static_mute_curve(buf: bytearray, abs_curve_off: int, curve: bytes, flag: int) -> None:
    parsed = parse_control_curve(curve, 9)
    if parsed is None:
        raise RuntimeError("Malformed 0x260A mute control curve in .ptx")
    _tb, pts, _end = parsed
    if len(pts) != 1 or len(curve) < 37 or struct.unpack_from("<H", curve, 23)[0] != 2:
        raise RuntimeError("Cannot patch static mute on a track with multi-point mute automation")
    struct.pack_into("<h", buf, abs_curve_off + 35, flag)


def _patch_strip_mute(buf: bytearray, blk_off: int, blk: bytes, mute: bool) -> None:
    flag = 1 if mute else 0
    d_list = find_subchunks(blk, 0x260D)
    if d_list:
        d0 = d_list[0]
        d0_off = blk_off + blk.find(d0)
        cpos = find_zchunk_offset(d0, 0x260C, 9)
        head = d0[:cpos] if cpos != -1 else d0
        curves = find_subchunks(head, 0x260A)
        if len(curves) < 2:
            raise RuntimeError("Track mixer strip has no 0x260A mute curve")
        p1 = _find_second_curve_offset(head, curves)
        if p1 < 0:
            raise RuntimeError("Could not locate 0x260A mute curve offset in 0x260D")
        _patch_static_mute_curve(buf, d0_off + p1, curves[1], flag)
        p1029 = find_subchunks(d0, 0x1029)
        if p1029 and len(p1029[0]) >= 101:
            off_29 = d0_off + d0.find(p1029[0])
            buf[off_29 + 14] = flag  # 0x1029 body[+5]
            buf[off_29 + 100] = flag  # 0x1029 body[+91]
        return
    s2610 = find_subchunks(blk, 0x2610)
    if s2610:
        s10_off = blk_off + blk.find(s2610[0])
        curves = find_subchunks(s2610[0], 0x260A)
        if len(curves) >= 2:
            p1 = _find_second_curve_offset(s2610[0], curves)
            if p1 >= 0:
                _patch_static_mute_curve(buf, s10_off + p1, curves[1], flag)
                return
    raise RuntimeError("Track mixer strip has neither 0x260D nor 0x2610 mute section")


def _patch_strip_2015_fields(
    buf: bytearray,
    blk_off: int,
    blk: bytes,
    *,
    solo_safe: bool | None = None,
    hidden: bool | None = None,
    color_index: int | None = None,
    height_pair: tuple[int, int] | None = None,
    timebase_ticks: bool | None = None,
) -> None:
    s15 = find_subchunks(blk, 0x2015)
    if not s15:
        return
    b15 = s15[-1]
    p15 = blk.rfind(b15)
    abs15 = blk_off + p15
    if solo_safe is not None and p15 + len(b15) < len(blk):
        buf[abs15 + len(b15)] = 1 if solo_safe else 0
    if len(b15) < 18 or b15[9] != 0x5A:
        return
    f_sz = struct.unpack_from("<I", b15, 12)[0]
    f_tag = struct.unpack_from("<H", b15, 16)[0]
    fend = 9 + 7 + f_sz
    if f_tag not in (0x2038, 0x203A, 0x203B) or f_sz < 2 or fend + 12 > len(b15):
        return
    if hidden is not None and fend + 2 <= len(b15):
        buf[abs15 + fend + 1] = 0 if hidden else 1
    clen = struct.unpack_from("<I", b15, fend + 8)[0]
    if clen >= 4096 or fend + 21 + clen >= len(b15):
        return
    q = fend + 21 + clen
    if b15[q] == 1 and q + 8 <= len(b15) and b15[q + 1] == 0x5A:
        sub_sz = struct.unpack_from("<I", b15, q + 4)[0]
        q2 = q + 1 + 7 + sub_sz
    else:
        q2 = q + 1
    if timebase_ticks is not None and q2 < len(b15):
        buf[abs15 + q2] = 2 if timebase_ticks else 1
    if color_index is not None and q2 + 3 <= len(b15):
        struct.pack_into("<H", buf, abs15 + q2 + 1, color_index & 0xFFFF)
    if height_pair is not None and q2 + 14 <= len(b15):
        struct.pack_into("<I", buf, abs15 + q2 + 10, height_pair[0])
        if fend + 8 <= len(b15):
            struct.pack_into("<H", buf, abs15 + fend + 6, height_pair[1])


def _patch_strip_inactive(buf: bytearray, blk_off: int, blk: bytes, inact: bool) -> None:
    active_b = 0 if inact else 1
    s2d = find_subchunks(blk, 0x102D)
    if s2d:
        after_2d = blk.find(s2d[0]) + len(s2d[0])
        # Only patch the 4-byte [width, 0xFF, 0xFF, active] gap when followed directly by 0x2627 (never on VCA 0x261F).
        if (
            after_2d + 13 <= len(blk)
            and blk[after_2d] != 0x5A
            and blk[after_2d + 4] == 0x5A
            and struct.unpack_from("<H", blk, after_2d + 11)[0] == 0x2627
        ):
            buf[blk_off + after_2d + 3] = active_b
    s4856 = find_subchunks(blk, 0x4856)
    if s4856:
        p4856 = blk.rfind(s4856[-1])
        if p4856 >= 25:
            buf[blk_off + p4856 - 25] = 1 if inact else 0
        for sub_e in find_subchunks(s4856[-1], 0x260E):
            pe = s4856[-1].find(sub_e)
            if pe != -1 and len(sub_e) > 12:
                buf[blk_off + p4856 + pe + 12] = active_b
    d_list = find_subchunks(blk, 0x260D)
    if d_list:
        d0_off = blk.find(d_list[0])
        for sub_e in find_subchunks(d_list[0], 0x260E):
            pe = d_list[0].find(sub_e)
            if pe != -1 and len(sub_e) > 12:
                buf[blk_off + d0_off + pe + 12] = active_b


def _patch_strip_timebase(buf: bytearray, blk_off: int, blk: bytes, is_ticks: bool) -> None:
    tb_byte = 1 if is_ticks else 0
    origin = ZERO_TICKS if is_ticks else 0
    scan = 9
    end = len(blk)
    targets: list[int] = []
    while scan < end:
        hit = find_zchunk_offset(blk, 0x260A, scan, end)
        if hit < 0:
            break
        sz = struct.unpack_from("<I", blk, hit + 3)[0]
        c_end = hit + 7 + sz
        if sz >= 30 and c_end <= end and blk[hit + 9 : hit + 11] == b"\x01\x46":
            parsed = parse_control_curve(blk[hit:c_end], 9)
            if parsed is None:
                raise RuntimeError("Malformed 0x260A automation curve while patching track timebase")
            _tb, pts, _end = parsed
            raw_pos0 = struct.unpack_from("<Q", blk, hit + 27)[0]
            if len(pts) != 1 or raw_pos0 not in (0, ZERO_TICKS):
                raise ValueError(
                    "Cannot patch track timebase on a track with non-origin or multi-point automation curves"
                )
            targets.append(hit)
        scan = c_end if c_end > hit else hit + 1
    for hit in targets:
        buf[blk_off + hit + 17] = tb_byte
        struct.pack_into("<Q", buf, blk_off + hit + 27, origin)


def _patch_1054_lane_timebases(
    buf: bytearray,
    session: PtxSession,
    tb_by_track_name_lc: Mapping[str, bool],
) -> None:
    if not tb_by_track_name_lc:
        return
    c1054 = session.chunk_map.get(0x1054)
    if c1054 is None or len(c1054.raw) < 13:
        return
    lane_tb: dict[int, bool] = {}
    for pl in session.playlists:
        tname = (pl.track_name or pl.name).lower()
        if tname in tb_by_track_name_lc:
            for li in pl.lane_indices:
                lane_tb[li] = tb_by_track_name_lc[tname]
    if not lane_tb:
        return
    cnt = struct.unpack_from("<I", buf, c1054.offset + 9)[0]
    pos = c1054.offset + 13
    end = c1054.offset + len(c1054.raw)
    for lane_idx in range(cnt):
        if pos + 9 > end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        l_end = pos + 7 + sz
        if l_end > end:
            break
        if lane_idx in lane_tb and struct.unpack_from("<H", buf, pos + 1)[0] >= 3 and l_end >= pos + 11:
            buf[l_end - 2 : l_end] = b"\x02\x00" if lane_tb[lane_idx] else b"\x01\x00"
        pos = l_end


def _apply_single_strip_states(buf: bytearray, blk_off: int, blk: bytes, st: Mapping[str, Any]) -> None:
    if "mute" in st:
        _patch_strip_mute(buf, blk_off, blk, bool(st["mute"]))
    if "solo" in st:
        s2d = find_subchunks(blk[:512], 0x102D)
        if s2d:
            p2d = blk.find(s2d[0], 9)
            s19 = find_subchunks(s2d[0], 0x2619)
            if p2d != -1 and s19:
                p19 = s2d[0].find(s19[0], 9)
                if p19 != -1 and p2d + p19 + len(s19[0]) < len(blk):
                    buf[blk_off + p2d + p19 + len(s19[0])] = 1 if st["solo"] else 0
    if "inactive" in st:
        _patch_strip_inactive(buf, blk_off, blk, bool(st["inactive"]))
    tb_ticks = (str(st["timebase"]).strip().lower() == "ticks") if "timebase" in st else None
    if tb_ticks is not None:
        _patch_strip_timebase(buf, blk_off, blk, tb_ticks)
    height_pair = _resolve_height_px(st["height"]) if "height" in st else None
    _patch_strip_2015_fields(
        buf, blk_off, blk,
        solo_safe=bool(st["solo_safe"]) if "solo_safe" in st else None,
        hidden=bool(st["hidden"]) if "hidden" in st else None,
        color_index=int(st["color_index"]) if "color_index" in st else None,
        height_pair=height_pair,
        timebase_ticks=tb_ticks,
    )


def _patch_strip_states_in_2624(
    buf: bytearray,
    c2624_off: int,
    chunk_end: int,
    states_by_uuid: Mapping[bytes, Mapping[str, Any]],
) -> set[bytes]:
    tcount = struct.unpack_from("<I", buf, c2624_off + 9)[0]
    pos = c2624_off + 13
    matched: set[bytes] = set()
    for _ in range(tcount):
        if pos + 9 > chunk_end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        blk_end = pos + 7 + sz
        if sz < 2 or blk_end > chunk_end:
            break
        blk = bytes(buf[pos:blk_end])
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        uid = blk[uoff : uoff + 16] if (uoff != -1 and uoff + 16 <= len(blk)) else b""
        st = states_by_uuid.get(uid)
        if st is not None:
            _apply_single_strip_states(buf, pos, blk, st)
            matched.add(uid)
        pos = blk_end
    return matched


def _patch_2519_states_in_place(
    buf: bytearray,
    c2519_off: int,
    c2519_raw: bytes,
    states_by_idx: Mapping[int, Mapping[str, Any]],
    expected_track_count: int,
) -> None:
    _prefix, rows, p1, p2, _tail = _parse_2519_sections(c2519_raw)
    if len(rows) != expected_track_count:
        raise RuntimeError(f"0x2519 track count {len(rows)} != session track count {expected_track_count}")
    any_hidden_change = any("hidden" in st for st in states_by_idx.values())
    vis_orders: list[int] | None = None
    if any_hidden_change:
        vis_orders = []
        next_vis = 1
        for idx, (row, e1) in enumerate(zip(rows, p1)):
            e1_ver = struct.unpack_from("<H", e1, 1)[0]
            old_vis = struct.unpack_from("<H", row, len(row) - 3)[0]
            cur_hid = (old_vis == 0) or (e1_ver >= 8 and len(e1) >= 40 and not bool(e1[-7]))
            st = states_by_idx.get(idx)
            is_hid = bool(st["hidden"]) if (st is not None and "hidden" in st) else cur_hid
            if is_hid:
                vis_orders.append(0)
            else:
                if next_vis > 0xFFFF:
                    raise OverflowError(f"0x2519 visible_order exceeds u16 range: {next_vis}")
                vis_orders.append(next_vis)
                next_vis += 1
    pos = c2519_off + 19
    cnt_a = struct.unpack_from("<I", buf, pos)[0]
    pos += 4 + cnt_a * 2 + 4
    for idx, row in enumerate(rows):
        if vis_orders is not None:
            struct.pack_into("<H", buf, pos + len(row) - 3, vis_orders[idx])
        pos += len(row)
    for pass_entries in (p1, p2):
        for idx, entry in enumerate(pass_entries):
            e_ver = struct.unpack_from("<H", entry, 1)[0]
            nlen = struct.unpack_from("<I", buf, pos + 11)[0]
            if vis_orders is not None and 15 + nlen + 20 <= len(entry):
                struct.pack_into("<H", buf, pos + 15 + nlen + 18, vis_orders[idx])
            st = states_by_idx.get(idx)
            if st is not None and e_ver >= 8 and len(entry) >= 40:
                if "hidden" in st:
                    buf[pos + len(entry) - 7] = 0 if st["hidden"] else 1
                if "inactive" in st:
                    buf[pos + len(entry) - 6] = 0 if st["inactive"] else 1
                if "color_index" in st:
                    struct.pack_into("<H", buf, pos + len(entry) - 2, int(st["color_index"]) & 0xFFFF)
            pos += len(entry)
        pos += 4


def _patch_2587_states_in_place(
    buf: bytearray,
    c2587_off: int,
    c2587_len: int,
    states_by_idx: Mapping[int, Mapping[str, Any]],
) -> None:
    end = c2587_off + c2587_len
    scan = c2587_off + 9
    while scan < end:
        hit = find_zchunk_offset(buf, 0x2589, scan, end)
        if hit < 0:
            break
        sz = struct.unpack_from("<I", buf, hit + 3)[0]
        e_end = hit + 7 + sz
        if sz >= 16 and e_end <= end:
            tidx = struct.unpack_from("<H", buf, hit + 9)[0]
            st = states_by_idx.get(tidx)
            if st is not None:
                if "hidden" in st:
                    hid = bool(st["hidden"])
                    struct.pack_into("<H", buf, hit + 11, 0 if hid else 1)
                    buf[e_end - 8] = 1 if hid else 0
                    buf[e_end - 6] = 1 if hid else 0
                if "folder_open" in st:
                    buf[e_end - 7] = 0 if st["folder_open"] else 1
                if "height" in st:
                    px, _ = _resolve_height_px(st["height"])
                    struct.pack_into("<H", buf, hit + 13, px)
                    struct.pack_into("<H", buf, e_end - 2, px)
        scan = e_end


def _validate_track_state_map(
    session: PtxSession,
    track_states: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[bytes, Mapping[str, Any]], dict[int, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    by_uuid: dict[bytes, Mapping[str, Any]] = {}
    by_idx: dict[int, Mapping[str, Any]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tname, st in track_states.items():
        unknown = set(st) - _TRACK_STATE_KEYS
        if unknown:
            raise ValueError(f"Unsupported track state key(s) for {tname!r}: {sorted(unknown)}")
        if "height" in st:
            _resolve_height_px(st["height"])
        if "timebase" in st and str(st["timebase"]).strip().lower() not in ("samples", "ticks"):
            raise ValueError(f"Unsupported track timebase for {tname!r}: {st['timebase']!r} (expected 'Samples' or 'Ticks')")
        if "color_index" in st:
            ci = st["color_index"]
            if isinstance(ci, bool) or not (0 <= int(ci) <= 0xFFFF):
                raise ValueError(f"Track color_index out of u16 range for {tname!r}: {ci!r}")
        trk: PtxTrack | None = session.track_by_name(tname)
        if trk is None:
            raise KeyError(f"Track {tname!r} not found in .ptx session")
        uid = bytes.fromhex(trk.uuid_hex)
        if uid in by_uuid:
            raise ValueError(f"Duplicate track_states target for {trk.name!r}")
        if "folder_open" in st and trk.track_type not in ("RoutingFolder", "BasicFolder"):
            raise ValueError(f"Track {trk.name!r} is type {trk.track_type!r}, not a folder track")
        if "mute" in st and trk.track_type in _NO_MUTE_TYPES:
            raise ValueError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support mute state")
        if "solo" in st and trk.track_type in _NO_SOLO_TYPES:
            raise ValueError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support solo state")
        if "solo_safe" in st and trk.track_type in _NO_SOLO_SAFE_TYPES:
            raise ValueError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support solo_safe state")
        if "timebase" in st and trk.track_type == "BasicFolder":
            raise ValueError(f"Track {trk.name!r} (type 'BasicFolder') does not support timebase state")
        by_uuid[uid] = st
        by_idx[trk.index] = st
        summary[trk.name] = dict(st)
    return by_uuid, by_idx, summary


def patch_track_states(
    ptx_raw: bytes,
    track_states: Mapping[str, Mapping[str, Any]],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Patch per-track mute, solo, solo_safe, inactive, hidden, folder_open, color_index, and height in place."""
    if not track_states:
        raise ValueError("track_states must be a non-empty {track_name: {state: value}} mapping")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    by_uuid, by_idx, summary = _validate_track_state_map(session, track_states)
    buf = bytearray(dec)

    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(buf):
        raise RuntimeError("Missing or truncated 0x2624 mixer chunk in .ptx")
    matched = _patch_strip_states_in_2624(buf, c2624.offset, min(len(buf), c2624.offset + 7 + c2624.size), by_uuid)
    if len(matched) != len(by_uuid):
        raise RuntimeError(f"Located {len(matched)}/{len(by_uuid)} target mixer strips in 0x2624")

    c2519 = session.chunk_map.get(0x2519)
    if c2519 is not None:
        _patch_2519_states_in_place(buf, c2519.offset, c2519.raw, by_idx, len(session.tracks))

    c2587 = session.chunk_map.get(0x2587)
    if c2587 is not None:
        _patch_2587_states_in_place(buf, c2587.offset, len(c2587.raw), by_idx)

    tb_audio = {
        trk.name.lower(): (str(by_idx[trk.index]["timebase"]).strip().lower() == "ticks")
        for trk in session.tracks
        if trk.index in by_idx and "timebase" in by_idx[trk.index] and trk.track_type == "Audio"
    }
    _patch_1054_lane_timebases(buf, session, tb_audio)

    return encrypt_ptx(bytes(buf), delta, k0=k0), summary


def _rewrite_2107_names(c2107_raw: bytes, rename_by_uuid: Mapping[bytes, str]) -> bytes:
    if len(c2107_raw) < 22:
        raise RuntimeError("Truncated 0x2107 track UUID registry chunk")
    ver = struct.unpack_from("<H", c2107_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2107_raw, 18)[0]
    items, end_pos = walk_sequential_zchunks(c2107_raw, 22, cnt, expected_tag=0x210B)
    out_items: list[bytes] = []
    matched: set[bytes] = set()
    for it in items:
        if len(it) < 17:
            raise RuntimeError("Truncated 0x210B track UUID entry")
        old_nlen = struct.unpack_from("<I", it, 13)[0]
        uid_off = 17 + old_nlen
        if uid_off + 16 > len(it):
            raise RuntimeError("Truncated 0x210B name/UUID payload")
        uid = it[uid_off : uid_off + 16]
        new_name = rename_by_uuid.get(uid)
        if new_name is None:
            out_items.append(it)
            continue
        iver = struct.unpack_from("<H", it, 1)[0]
        out_items.append(build_zchunk(0x210B, it[9:13] + encode_lp_str(new_name) + it[uid_off:], version=iver))
        matched.add(uid)
    if len(matched) != len(rename_by_uuid):
        raise RuntimeError(f"Located {len(matched)}/{len(rename_by_uuid)} target entries in 0x2107")
    return build_zchunk(0x2107, c2107_raw[9:18] + struct.pack("<I", len(out_items)) + b"".join(out_items) + c2107_raw[end_pos:], version=ver)


def _rewrite_2519_names(c2519_raw: bytes, rename_by_idx: Mapping[int, str], expected_count: int) -> bytes:
    _clean, rows, p1, p2, tail = _parse_2519_sections(c2519_raw)
    if len(rows) != expected_count:
        raise RuntimeError(f"0x2519 track count {len(rows)} != session track count {expected_count}")
    ver = struct.unpack_from("<H", c2519_raw, 1)[0]
    cnt_a = struct.unpack_from("<I", c2519_raw, 19)[0]
    orig_prefix = c2519_raw[9 : 23 + cnt_a * 2]

    def _ren_row(r: bytes, nm: str) -> bytes:
        old_nlen = struct.unpack_from("<I", r, 2)[0]
        if 6 + old_nlen > len(r):
            raise RuntimeError("Truncated 0x2519 summary row name")
        return r[:2] + encode_lp_str(nm) + r[6 + old_nlen :]

    def _ren_251a(e: bytes, nm: str) -> bytes:
        ever = struct.unpack_from("<H", e, 1)[0]
        old_nlen = struct.unpack_from("<I", e, 11)[0]
        if 15 + old_nlen > len(e):
            raise RuntimeError("Truncated 0x251A track entry name")
        return build_zchunk(0x251A, e[9:11] + encode_lp_str(nm) + e[15 + old_nlen :], version=ever)

    new_rows = [_ren_row(r, rename_by_idx[i]) if i in rename_by_idx else r for i, r in enumerate(rows)]
    new_p1 = [_ren_251a(e, rename_by_idx[i]) if i in rename_by_idx else e for i, e in enumerate(p1)]
    new_p2 = [_ren_251a(e, rename_by_idx[i]) if i in rename_by_idx else e for i, e in enumerate(p2)]
    cnt_le = struct.pack("<I", len(new_rows))
    body = orig_prefix + cnt_le + b"".join(new_rows) + b"".join(new_p1) + cnt_le + b"".join(new_p2) + tail
    return build_zchunk(0x2519, body, version=ver)


def _rewrite_strip_name(strip: bytes, new_name: str) -> bytes:
    p19 = find_zchunk_offset(strip, 0x2619, 9)
    if p19 < 0 or p19 + 13 > len(strip):
        raise RuntimeError("Missing or truncated 0x2619 track identity inside mixer strip")
    old_19_sz = struct.unpack_from("<I", strip, p19 + 3)[0]
    end_19 = p19 + 7 + old_19_sz
    old_nlen = struct.unpack_from("<I", strip, p19 + 9)[0]
    if p19 + 13 + old_nlen > end_19:
        raise RuntimeError("Truncated 0x2619 track name length inside mixer strip")
    new_lp = encode_lp_str(new_name)
    delta = len(new_lp) - (4 + old_nlen)
    # Locate all enclosing ancestor headers (strip at 0, optional 0x261B, optional 0x102D) BEFORE mutating buf.
    ancestors = [0]
    for anc_tag in (0x261B, 0x102D):
        aoff = find_zchunk_offset(strip, anc_tag, 9)
        if 9 <= aoff < p19:
            asz = struct.unpack_from("<I", strip, aoff + 3)[0]
            if aoff + 7 + asz >= end_19:
                ancestors.append(aoff)
    buf = bytearray(strip)
    buf[p19 + 9 : p19 + 13 + old_nlen] = new_lp
    for target_off in (*ancestors, p19):
        new_sz = struct.unpack_from("<I", buf, target_off + 3)[0] + delta
        if not (2 <= new_sz <= 0xFFFF_FFFF):
            raise OverflowError(f"Renamed Z-chunk size out of u32 range at 0x{target_off:x}: {new_sz}")
        struct.pack_into("<I", buf, target_off + 3, new_sz)
    return bytes(buf)


def _rewrite_2624_names(c2624_raw: bytes, rename_by_uuid: Mapping[bytes, str]) -> bytes:
    if len(c2624_raw) < 13:
        raise RuntimeError("Truncated 0x2624 mixer container chunk")
    ver = struct.unpack_from("<H", c2624_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2624_raw, 9)[0]
    strips, end_pos = walk_sequential_zchunks(c2624_raw, 13, cnt)
    new_strips: list[bytes] = []
    matched: set[bytes] = set()
    for st in strips:
        uoff = st.find(UUID_PREFIX, 0, min(256, len(st)))
        uid = st[uoff : uoff + 16] if (uoff != -1 and uoff + 16 <= len(st)) else b""
        new_nm = rename_by_uuid.get(uid)
        if new_nm is not None:
            new_strips.append(_rewrite_strip_name(st, new_nm))
            matched.add(uid)
        else:
            new_strips.append(st)
    if len(matched) != len(rename_by_uuid):
        raise RuntimeError(f"Located {len(matched)}/{len(rename_by_uuid)} target mixer strips in 0x2624")
    return build_zchunk(0x2624, struct.pack("<I", len(new_strips)) + b"".join(new_strips) + c2624_raw[end_pos:], version=ver)


def _rewrite_1015_playlist_names(c1015_raw: bytes, audio_renames_lc: Mapping[str, str]) -> bytes | None:
    if not audio_renames_lc or len(c1015_raw) < 13:
        return None
    ver = struct.unpack_from("<H", c1015_raw, 1)[0]
    cnt = struct.unpack_from("<I", c1015_raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(c1015_raw, 13, cnt, expected_tag=0x1014)
    changed = False
    out_items: list[bytes] = []
    for pl in items:
        if len(pl) < 13:
            out_items.append(pl)
            continue
        nlen = struct.unpack_from("<I", pl, 9)[0]
        if 13 + nlen > len(pl):
            out_items.append(pl)
            continue
        old_name = pl[13 : 13 + nlen].decode("utf-8", "replace")
        new_name = audio_renames_lc.get(old_name.lower())
        if new_name is not None:
            pver = struct.unpack_from("<H", pl, 1)[0]
            out_items.append(build_zchunk(0x1014, encode_lp_str(new_name) + pl[13 + nlen :], version=pver))
            changed = True
        else:
            out_items.append(pl)
    if not changed:
        return None
    return build_zchunk(0x1015, struct.pack("<I", len(out_items)) + b"".join(out_items) + c1015_raw[end_pos:], version=ver)


def _rewrite_1058_midi_names(c1058_raw: bytes, midi_renames_lc: Mapping[str, str]) -> bytes | None:
    if not midi_renames_lc or len(c1058_raw) < 13:
        return None
    ver = struct.unpack_from("<H", c1058_raw, 1)[0]
    cnt = struct.unpack_from("<I", c1058_raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(c1058_raw, 13, cnt, expected_tag=0x1057)
    changed = False
    out_items: list[bytes] = []
    for it in items:
        if len(it) < 13:
            out_items.append(it)
            continue
        nlen = struct.unpack_from("<I", it, 9)[0]
        if 13 + nlen > len(it):
            out_items.append(it)
            continue
        old_name = it[13 : 13 + nlen].decode("utf-8", "replace")
        new_name = midi_renames_lc.get(old_name.lower())
        if new_name is not None:
            iver = struct.unpack_from("<H", it, 1)[0]
            out_items.append(build_zchunk(0x1057, encode_lp_str(new_name) + it[13 + nlen :], version=iver))
            changed = True
        else:
            out_items.append(it)
    if not changed:
        return None
    return build_zchunk(0x1058, struct.pack("<I", len(out_items)) + b"".join(out_items) + c1058_raw[end_pos:], version=ver)


def _validate_rename_map(
    session: PtxSession,
    renames: Mapping[str, str],
) -> tuple[dict[int, str], dict[bytes, str], dict[str, str], dict[str, str], dict[str, str]]:
    rename_by_idx: dict[int, str] = {}
    rename_by_uuid: dict[bytes, str] = {}
    audio_renames_lc: dict[str, str] = {}
    midi_renames_lc: dict[str, str] = {}
    summary: dict[str, str] = {}
    for old_nm, new_nm in renames.items():
        cleaned = new_nm.strip()
        if not cleaned:
            raise ValueError(f"New track name for {old_nm!r} must not be empty")
        trk = session.track_by_name(old_nm)
        if trk is None:
            raise KeyError(f"Track {old_nm!r} not found in .ptx session")
        if trk.index in rename_by_idx:
            raise ValueError(f"Duplicate rename target for track {trk.name!r}")
        rename_by_idx[trk.index] = cleaned
        rename_by_uuid[bytes.fromhex(trk.uuid_hex)] = cleaned
        if trk.track_type == "Audio":
            audio_renames_lc[trk.name.lower()] = cleaned
        elif trk.track_type in ("Midi", "Instrument"):
            midi_renames_lc[trk.name.lower()] = cleaned
        summary[trk.name] = cleaned
    final_names_lc: set[str] = set()
    for trk in session.tracks:
        nm_lc = rename_by_idx.get(trk.index, trk.name).lower()
        if nm_lc in final_names_lc:
            raise ValueError(f"Track rename collision on {rename_by_idx.get(trk.index, trk.name)!r}")
        final_names_lc.add(nm_lc)
    return rename_by_idx, rename_by_uuid, audio_renames_lc, midi_renames_lc, summary


def patch_rename_tracks(
    ptx_raw: bytes,
    renames: Mapping[str, str],
) -> tuple[bytes, dict[str, str]]:
    """Rename tracks across ``0x2107``, ``0x2519``, ``0x2624``, ``0x1015``, ``0x1058`` and rebuild ``0x0002``."""
    if not renames:
        raise ValueError("renames must be a non-empty {old_name: new_name} mapping")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    rename_by_idx, rename_by_uuid, audio_renames_lc, midi_renames_lc, summary = _validate_rename_map(session, renames)

    template_idx = parse_index_0002(dec)
    top_chunks = iter_top_chunks(dec, strict=True)
    last_idx = {c.tag: i for i, c in enumerate(top_chunks)}
    c2107 = session.chunk_map.get(0x2107)
    c2519 = session.chunk_map.get(0x2519)
    c2624 = session.chunk_map.get(0x2624)
    if c2107 is None or c2519 is None or c2624 is None:
        raise RuntimeError("Missing required 0x2107/0x2519/0x2624 chunk in .ptx for track rename")
    replacements: dict[int, bytes] = {
        0x2107: _rewrite_2107_names(c2107.raw, rename_by_uuid),
        0x2519: _rewrite_2519_names(c2519.raw, rename_by_idx, len(session.tracks)),
        0x2624: _rewrite_2624_names(c2624.raw, rename_by_uuid),
    }
    c1015 = session.chunk_map.get(0x1015)
    if c1015 is not None:
        new_1015 = _rewrite_1015_playlist_names(c1015.raw, audio_renames_lc)
        if new_1015 is not None:
            replacements[0x1015] = new_1015
    c1058 = session.chunk_map.get(0x1058)
    if c1058 is not None:
        new_1058 = _rewrite_1058_midi_names(c1058.raw, midi_renames_lc)
        if new_1058 is not None:
            replacements[0x1058] = new_1058

    out = bytearray(dec[:0x1F])
    for i, ch in enumerate(top_chunks):
        if ch.tag in replacements and last_idx[ch.tag] == i:
            out.extend(replacements[ch.tag])
        else:
            out.extend(ch.raw)
    if template_idx is not None:
        apply_rebuilt_index_0002(out, template_idx, dec)
    return encrypt_ptx(bytes(out), delta, k0=k0), summary


def _rewrite_clip_2629_name(c2629: bytes, new_name: str) -> bytes:
    p28 = find_zchunk_offset(c2629, 0x2628, 9)
    if p28 < 0 or p28 + 13 > len(c2629):
        raise RuntimeError("Missing or truncated 0x2628 inside 0x2629 clip")
    old_28_sz = struct.unpack_from("<I", c2629, p28 + 3)[0]
    end_28 = p28 + 7 + old_28_sz
    old_nlen = struct.unpack_from("<I", c2629, p28 + 9)[0]
    if p28 + 13 + old_nlen > end_28:
        raise RuntimeError("Truncated 0x2628 clip name length")
    new_lp = encode_lp_str(new_name)
    delta = len(new_lp) - (4 + old_nlen)
    buf = bytearray(c2629)
    buf[p28 + 9 : p28 + 13 + old_nlen] = new_lp
    for off in (0, p28):
        new_sz = struct.unpack_from("<I", buf, off + 3)[0] + delta
        if not (2 <= new_sz <= 0xFFFF_FFFF):
            raise OverflowError(f"Renamed clip Z-chunk size out of u32 range: {new_sz}")
        struct.pack_into("<I", buf, off + 3, new_sz)
    return bytes(buf)


def patch_rename_clips(
    ptx_raw: bytes,
    renames: Mapping[str, str],
) -> tuple[bytes, dict[str, str]]:
    """Rename one or more audio clips inside ``0x262A`` -> ``0x2629`` -> ``0x2628`` and rebuild ``0x0002``."""
    if not renames:
        raise ValueError("renames must be a non-empty {old_name: new_name} mapping")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    c262a = session.chunk_map.get(0x262A)
    if c262a is None or len(c262a.raw) < 13:
        raise RuntimeError("Missing or truncated 0x262A clip container chunk in .ptx")
    rename_by_idx: dict[int, str] = {}
    summary: dict[str, str] = {}
    for old_nm, new_nm in renames.items():
        cleaned = new_nm.strip()
        if not cleaned:
            raise ValueError(f"New clip name for {old_nm!r} must not be empty")
        matches = [c for c in session.clips if c.name == old_nm]
        if not matches:
            low = old_nm.lower()
            matches = [c for c in session.clips if c.name.lower() == low]
        if not matches:
            raise KeyError(f"Clip {old_nm!r} not found in .ptx session")
        for clip in matches:
            rename_by_idx[clip.index] = cleaned
            summary[clip.name] = cleaned
    ver = struct.unpack_from("<H", c262a.raw, 1)[0]
    cnt = struct.unpack_from("<I", c262a.raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(c262a.raw, 13, cnt, expected_tag=0x2629)
    new_items = [_rewrite_clip_2629_name(it, rename_by_idx[i]) if i in rename_by_idx else it for i, it in enumerate(items)]
    new_262a = build_zchunk(0x262A, struct.pack("<I", len(new_items)) + b"".join(new_items) + c262a.raw[end_pos:], version=ver)
    template_idx = parse_index_0002(dec)
    top_chunks = iter_top_chunks(dec, strict=True)
    last_262a = max(i for i, c in enumerate(top_chunks) if c.tag == 0x262A)
    buf = bytearray(dec[:0x1F])
    for i, ch in enumerate(top_chunks):
        buf.extend(new_262a if i == last_262a else ch.raw)
    if template_idx is not None:
        apply_rebuilt_index_0002(buf, template_idx, dec)
    return encrypt_ptx(bytes(buf), delta, k0=k0), summary


def _resolve_audio_track_lanes(session: PtxSession, track_name: str) -> tuple[PtxTrack, set[int]]:
    trk = session.track_by_name(track_name)
    if trk is None or trk.track_type != "Audio":
        raise KeyError(f"Audio track {track_name!r} not found in .ptx session")
    pl = next((p for p in session.playlists if not p.is_alternate and (p.track_name or p.name).lower() == trk.name.lower()), None)
    if pl is None or not pl.lane_indices:
        raise RuntimeError(f"Audio track {trk.name!r} has no active playlist lanes in 0x1015")
    return trk, set(pl.lane_indices)


def _patch_104f_in_lane(
    buf: bytearray,
    pos: int,
    l_end: int,
    start_sample: int,
    *,
    is_muted: bool | None,
    is_time_locked: bool | None,
    is_edit_locked: bool | None,
    color_index: int | None,
) -> int:
    patched = 0
    scan = pos + 9
    while scan < l_end:
        hit = find_zchunk_offset(buf, 0x104F, scan, l_end)
        if hit < 0:
            break
        fsz = struct.unpack_from("<I", buf, hit + 3)[0]
        f_end = hit + 7 + fsz
        is_fade = f_end < l_end and buf[f_end] == 1
        if fsz >= 37 and f_end <= l_end and not is_fade:
            if (struct.unpack_from("<Q", buf, hit + 16)[0] & 0x00FF_FFFF_FFFF_FFFF) == start_sample:
                if is_muted is not None:
                    buf[hit + 9] = 1 if is_muted else 0
                if is_edit_locked is not None:
                    buf[hit + 10] = 1 if is_edit_locked else 0
                    if is_edit_locked:
                        buf[hit + 39] = 1
                if is_time_locked is not None:
                    buf[hit + 39] = 1 if (is_time_locked or bool(buf[hit + 10])) else 0
                if color_index is not None:
                    struct.pack_into("<H", buf, hit + 25, color_index)
                patched += 1
        scan = f_end if f_end > hit else hit + 1
    return patched


def patch_clip_attributes(
    ptx_raw: bytes,
    track_name: str,
    start_sample: int,
    *,
    is_muted: bool | None = None,
    is_time_locked: bool | None = None,
    is_edit_locked: bool | None = None,
    color_index: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Patch ``0x104F`` clip placement mute, time-lock, edit-lock, and/or color_index in place."""
    if color_index is not None and (isinstance(color_index, bool) or not (0 <= color_index <= 0xFFFF)):
        raise ValueError(f"color_index out of u16 range: {color_index!r}")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    trk, target_lanes = _resolve_audio_track_lanes(session, track_name)
    c1054 = session.chunk_map.get(0x1054)
    if c1054 is None or len(c1054.raw) < 13:
        raise RuntimeError("Missing or truncated 0x1054 timeline lanes chunk in .ptx")
    buf = bytearray(dec)
    cnt = struct.unpack_from("<I", buf, c1054.offset + 9)[0]
    pos, end = c1054.offset + 13, c1054.offset + len(c1054.raw)
    patched_count = 0
    for lane_idx in range(cnt):
        if pos + 9 > end or buf[pos] != 0x5A:
            break
        l_end = pos + 7 + struct.unpack_from("<I", buf, pos + 3)[0]
        if l_end > end:
            break
        if lane_idx in target_lanes:
            patched_count += _patch_104f_in_lane(
                buf, pos, l_end, start_sample,
                is_muted=is_muted, is_time_locked=is_time_locked,
                is_edit_locked=is_edit_locked, color_index=color_index,
            )
        pos = l_end
    if patched_count == 0:
        raise KeyError(f"No clip placement found on track {trk.name!r} at start_sample={start_sample}")
    return encrypt_ptx(bytes(buf), delta, k0=k0), {
        "track_name": trk.name, "start_sample": start_sample, "lanes_patched": patched_count,
    }


def _insert_fade_into_lane_1052(lane_raw: bytes, fade_index: int, start_sample: int) -> bytes:
    ver = struct.unpack_from("<H", lane_raw, 1)[0]
    nlen = struct.unpack_from("<I", lane_raw, 9)[0]
    cnt_off = 13 + nlen
    if cnt_off + 4 > len(lane_raw):
        raise RuntimeError("Truncated 0x1052 lane header")
    old_cnt = struct.unpack_from("<I", lane_raw, cnt_off)[0]
    old_items, after = walk_sequential_zchunks(lane_raw, cnt_off + 4, old_cnt, expected_tag=0x1050)
    tail = lane_raw[after:]
    fade_104f = serialize_placement_104f(fade_index, start_sample, flags=1, color_index=0xFFFE)
    fade_1050 = build_zchunk(0x1050, fade_104f + b"\x01\x01\x01", version=3)

    def _ev_key(w1050: bytes) -> tuple[int, int]:
        p4f = find_zchunk_offset(w1050, 0x104F, 9)
        if p4f >= 0 and p4f + 24 <= len(w1050):
            smp = struct.unpack_from("<Q", w1050, p4f + 16)[0] & 0x00FF_FFFF_FFFF_FFFF
            f_end = p4f + 7 + struct.unpack_from("<I", w1050, p4f + 3)[0]
            is_f = 0 if (f_end < len(w1050) and w1050[f_end] == 1) else 1
            return smp, is_f
        return 0, 1

    new_items = sorted([*old_items, fade_1050], key=_ev_key)
    body = lane_raw[9:cnt_off] + struct.pack("<I", len(new_items)) + b"".join(new_items) + tail
    return build_zchunk(0x1052, body, version=ver)


def _splice_2630_and_1054(dec: bytes, new_2630: bytes, new_1054: bytes) -> bytearray:
    top_chunks = iter_top_chunks(dec, strict=True)
    last_idx = {c.tag: i for i, c in enumerate(top_chunks)}
    out = bytearray(dec[:0x1F])
    inserted_2630 = 0x2630 in last_idx
    anchor_after = next((t for t in (0x2504, 0x262A) if t in last_idx), None)
    for i, ch in enumerate(top_chunks):
        if not inserted_2630 and anchor_after is None and ch.tag in (0x1054, 0x0002) and last_idx[ch.tag] == i:
            out.extend(new_2630)
            inserted_2630 = True
        if ch.tag == 0x2630 and last_idx[0x2630] == i:
            out.extend(new_2630)
        elif ch.tag == 0x1054 and last_idx[0x1054] == i:
            out.extend(new_1054)
        else:
            out.extend(ch.raw)
        if not inserted_2630 and anchor_after is not None and ch.tag == anchor_after and last_idx[ch.tag] == i:
            out.extend(new_2630)
            inserted_2630 = True
    if not inserted_2630:
        out.extend(new_2630)
    return out


def patch_add_fade(
    ptx_raw: bytes,
    track_name: str,
    fade_type: str,
    start_sample: int,
    total_samples: int,
    *,
    pre_samples: int | None = None,
    shape: str = "standard",
    crossfade_link: str = "equal_power",
) -> tuple[bytes, dict[str, Any]]:
    """Append a fade-in, fade-out, or crossfade to ``0x2630`` and place it on *track_name* in ``0x1054``."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    trk, target_lanes = _resolve_audio_track_lanes(session, track_name)
    c1054 = session.chunk_map.get(0x1054)
    if c1054 is None or len(c1054.raw) < 13:
        raise RuntimeError("Missing or truncated 0x1054 timeline lanes chunk in .ptx")
    fade_chunk = serialize_fade_262f(
        fade_type, total_samples, pre_samples=pre_samples, shape=shape, crossfade_link=crossfade_link,
    )
    c2630 = session.chunk_map.get(0x2630)
    if c2630 is not None and len(c2630.raw) >= 13:
        fver, fcnt = struct.unpack_from("<H", c2630.raw, 1)[0], struct.unpack_from("<I", c2630.raw, 9)[0]
        old_fades, f_end = walk_sequential_zchunks(c2630.raw, 13, fcnt, expected_tag=0x262F)
        fade_idx = len(old_fades)
        new_2630 = build_zchunk(0x2630, struct.pack("<I", fade_idx + 1) + b"".join([*old_fades, fade_chunk]) + c2630.raw[f_end:], version=fver)
    else:
        fade_idx = 0
        new_2630 = build_zchunk(0x2630, struct.pack("<I", 1) + fade_chunk, version=1)
    lver, lcnt = struct.unpack_from("<H", c1054.raw, 1)[0], struct.unpack_from("<I", c1054.raw, 9)[0]
    old_lanes, l_end = walk_sequential_zchunks(c1054.raw, 13, lcnt, expected_tag=0x1052)
    new_lanes = [_insert_fade_into_lane_1052(ln, fade_idx, start_sample) if i in target_lanes else ln for i, ln in enumerate(old_lanes)]
    new_1054 = build_zchunk(0x1054, struct.pack("<I", len(new_lanes)) + b"".join(new_lanes) + c1054.raw[l_end:], version=lver)
    template_idx = parse_index_0002(dec)
    out = _splice_2630_and_1054(dec, new_2630, new_1054)
    if template_idx is not None:
        apply_rebuilt_index_0002(out, template_idx, dec)
    return encrypt_ptx(bytes(out), delta, k0=k0), {
        "action": "add_fade", "track_name": trk.name, "fade_type": fade_type,
        "fade_index": fade_idx, "start_sample": start_sample, "total_samples": total_samples,
    }
