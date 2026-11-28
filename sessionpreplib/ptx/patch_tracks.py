"""Track-level ``.ptx`` patchers: per-track states, timebases, and cross-chunk track renames."""

from __future__ import annotations

from collections.abc import Mapping
import struct
from typing import Any

from ._document import PtxBuffer, assemble_top_chunks, iter_mixer_strips, patch_static_0x260a_curve
from .codec import (
    find_subchunk_spans,
    find_zchunk_offset,
    parse_control_curve,
    resize_zchunk_ancestors,
    walk_sequential_zchunks,
)
from .models import (
    TRACK_HEIGHT_PRESET_PX,
    UUID_PREFIX,
    ZERO_TICKS,
    PtxCorruptFileError,
    PtxNotFoundError,
    PtxTrack,
    PtxTrackStatePatch,
    PtxValidationError,
)
from .parser import PtxSession
from .writer_index import apply_rebuilt_index_0002
from .writer_primitives import build_zchunk, encode_lp_str, parse_index_0002
from .writer_track_mutations import (
    extract_261c_active_playlist_index,
    parse_2519_sections,
    parse_midi_strip_playlists,
)

_TRACK_STATE_KEYS = frozenset({
    "mute", "solo", "solo_safe", "inactive", "hidden", "folder_open", "color_index", "height", "timebase",
})
_TRACK_STATE_ALIASES: dict[str, str] = {
    "is_muted": "mute",
    "is_soloed": "solo",
    "is_solo_safe": "solo_safe",
    "is_inactive": "inactive",
    "is_hidden": "hidden",
    "is_folder_open": "folder_open",
}
_HEIGHT_PRESETS_LC = {k.lower(): (k, v) for k, v in TRACK_HEIGHT_PRESET_PX.items()}
_NO_MUTE_TYPES = frozenset({"Master", "BasicFolder"})
_NO_SOLO_TYPES = frozenset({"Master", "BasicFolder"})
_NO_SOLO_SAFE_TYPES = frozenset({"Master", "BasicFolder"})


def _resolve_height_px(height: str | int) -> tuple[int, int]:
    if isinstance(height, bool):
        raise PtxValidationError(f"Invalid track height: {height!r}")
    if isinstance(height, int):
        if not (16 <= height <= 2000):
            raise PtxValidationError(f"Track height_px out of range 16..2000: {height}")
        return height, height
    hit = _HEIGHT_PRESETS_LC.get(str(height).strip().lower())
    if hit is None:
        raise PtxValidationError(f"Unsupported track height preset: {height!r} (valid: {list(TRACK_HEIGHT_PRESET_PX)})")
    return hit[1]


def _normalize_track_state_entry(tname: str, raw_st: PtxTrackStatePatch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_st, PtxTrackStatePatch):
        return raw_st.to_state_dict()
    normalized: dict[str, Any] = {}
    for k, v in raw_st.items():
        canon = _TRACK_STATE_ALIASES.get(k, k)
        if canon not in _TRACK_STATE_KEYS:
            raise PtxValidationError(f"Unsupported track state key(s) for {tname!r}: [{k!r}]")
        if canon in normalized:
            raise PtxValidationError(f"Conflicting track state keys for {tname!r}: {k!r} and {canon!r}")
        normalized[canon] = v
    return normalized


def _patch_strip_mute(buf: bytearray, blk_off: int, blk: bytes, mute: bool) -> None:
    flag = 1 if mute else 0
    d_spans = find_subchunk_spans(blk, 0x260D)
    if d_spans:
        d0_rel, d0 = d_spans[0]
        d0_off = blk_off + d0_rel
        cpos = find_zchunk_offset(d0, 0x260C, 9)
        head_end = cpos if cpos != -1 else len(d0)
        curves = find_subchunk_spans(d0, 0x260A, 9, head_end)
        if len(curves) < 2:
            raise PtxCorruptFileError("Track mixer strip has no 0x260A mute curve")
        c1_rel, c1_raw = curves[1]
        patch_static_0x260a_curve(buf, d0_off + c1_rel, c1_raw, flag, param_label="Track mute")
        p1029_spans = find_subchunk_spans(d0, 0x1029)
        if p1029_spans and len(p1029_spans[0][1]) >= 101:
            off_29 = d0_off + p1029_spans[0][0]
            buf[off_29 + 14] = flag
            buf[off_29 + 100] = flag
        return
    s2610_spans = find_subchunk_spans(blk, 0x2610)
    if s2610_spans:
        s10_rel, s10 = s2610_spans[0]
        curves = find_subchunk_spans(s10, 0x260A)
        if len(curves) >= 2:
            c1_rel, c1_raw = curves[1]
            patch_static_0x260a_curve(buf, blk_off + s10_rel + c1_rel, c1_raw, flag, param_label="Track mute")
            return
    raise PtxCorruptFileError("Track mixer strip has neither 0x260D nor 0x2610 mute section")


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
    s15_spans = find_subchunk_spans(blk, 0x2015)
    if not s15_spans:
        return
    p15, b15 = s15_spans[-1]
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
    strip_tag = struct.unpack_from("<H", blk, 7)[0] if len(blk) >= 9 else 0
    s2d_spans = find_subchunk_spans(blk, 0x102D)
    if s2d_spans:
        after_2d = s2d_spans[0][0] + len(s2d_spans[0][1])
        if (
            after_2d + 13 <= len(blk)
            and blk[after_2d] != 0x5A
            and blk[after_2d + 4] == 0x5A
            and struct.unpack_from("<H", blk, after_2d + 11)[0] == 0x2627
        ):
            buf[blk_off + after_2d + 3] = active_b
    if strip_tag == 0x2621:
        p2010 = find_zchunk_offset(blk, 0x2010, 9)
        if p2010 > 0:
            buf[blk_off + p2010 - 1] = active_b
    s4856_spans = find_subchunk_spans(blk, 0x4856)
    if s4856_spans:
        p4856, raw_4856 = s4856_spans[-1]
        inact_back = 31 if strip_tag == 0x2621 else 25
        if p4856 >= inact_back:
            buf[blk_off + p4856 - inact_back] = 1 if inact else 0
        for pe, sub_e in find_subchunk_spans(raw_4856, 0x260E):
            if len(sub_e) > 12:
                buf[blk_off + p4856 + pe + 12] = active_b
    d_spans = find_subchunk_spans(blk, 0x260D)
    if d_spans:
        d0_off, d0_raw = d_spans[0]
        for pe, sub_e in find_subchunk_spans(d0_raw, 0x260E):
            if len(sub_e) > 12:
                buf[blk_off + d0_off + pe + 12] = active_b


def _patch_strip_timebase(buf: bytearray, blk_off: int, blk: bytes, is_ticks: bool) -> None:
    tb_byte = 1 if is_ticks else 0
    origin = ZERO_TICKS if is_ticks else 0
    targets: list[int] = []
    for hit, c_raw in find_subchunk_spans(blk, 0x260A):
        if len(c_raw) >= 37 and c_raw[9:11] == b"\x01\x46":
            parsed = parse_control_curve(c_raw, 9)
            if parsed is None:
                raise PtxCorruptFileError("Malformed 0x260A automation curve while patching track timebase")
            _tb, pts, _end = parsed
            raw_pos0 = struct.unpack_from("<Q", c_raw, 27)[0]
            if len(pts) != 1 or raw_pos0 not in (0, ZERO_TICKS):
                raise PtxValidationError(
                    "Cannot patch track timebase on a track with non-origin or multi-point automation curves"
                )
            targets.append(hit)
    for hit in targets:
        buf[blk_off + hit + 17] = tb_byte
        struct.pack_into("<Q", buf, blk_off + hit + 27, origin)


def _patch_1054_lane_timebases(buf: bytearray, session: PtxSession, tb_by_track_name_lc: Mapping[str, bool]) -> None:
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
    pos, end = c1054.offset + 13, c1054.offset + len(c1054.raw)
    for lane_idx in range(cnt):
        if pos + 9 > end or buf[pos] != 0x5A:
            break
        l_end = pos + 7 + struct.unpack_from("<I", buf, pos + 3)[0]
        if l_end > end:
            break
        if lane_idx in lane_tb and struct.unpack_from("<H", buf, pos + 1)[0] >= 3 and l_end >= pos + 11:
            buf[l_end - 2 : l_end] = b"\x02\x00" if lane_tb[lane_idx] else b"\x01\x00"
        pos = l_end


def _apply_single_strip_states(buf: bytearray, blk_off: int, blk: bytes, st: Mapping[str, Any]) -> None:
    if "mute" in st:
        _patch_strip_mute(buf, blk_off, blk, bool(st["mute"]))
    if "solo" in st:
        s2d_spans = find_subchunk_spans(blk[:512], 0x102D)
        if s2d_spans:
            p2d, raw_2d = s2d_spans[0]
            s19_spans = find_subchunk_spans(raw_2d, 0x2619)
            if s19_spans:
                p19, raw_19 = s19_spans[0]
                if p2d + p19 + len(raw_19) < len(blk):
                    buf[blk_off + p2d + p19 + len(raw_19)] = 1 if st["solo"] else 0
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


def _patch_2519_states_in_place(
    buf: bytearray,
    c2519_off: int,
    c2519_raw: bytes,
    states_by_idx: Mapping[int, Mapping[str, Any]],
    expected_track_count: int,
) -> None:
    _prefix, rows, p1, p2, _tail = parse_2519_sections(c2519_raw)
    if len(rows) != expected_track_count:
        raise PtxCorruptFileError(f"0x2519 track count {len(rows)} != session track count {expected_track_count}")
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
                    raise PtxValidationError(f"0x2519 visible_order exceeds u16 range: {next_vis}")
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
    buf: bytearray, c2587_off: int, c2587_len: int, states_by_idx: Mapping[int, Mapping[str, Any]]
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
    track_states: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]],
) -> tuple[dict[bytes, Mapping[str, Any]], dict[int, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    by_uuid: dict[bytes, Mapping[str, Any]] = {}
    by_idx: dict[int, Mapping[str, Any]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tname, raw_st in track_states.items():
        st = _normalize_track_state_entry(tname, raw_st)
        if "height" in st:
            _resolve_height_px(st["height"])
        if "timebase" in st and str(st["timebase"]).strip().lower() not in ("samples", "ticks"):
            raise PtxValidationError(f"Unsupported track timebase for {tname!r}: {st['timebase']!r} (expected 'Samples' or 'Ticks')")
        if "color_index" in st:
            ci = st["color_index"]
            if isinstance(ci, bool) or not (0 <= int(ci) <= 0xFFFF):
                raise PtxValidationError(f"Track color_index out of u16 range for {tname!r}: {ci!r}")
        trk: PtxTrack | None = session.track_by_name(tname)
        if trk is None:
            raise PtxNotFoundError(f"Track {tname!r} not found in .ptx session")
        uid = bytes.fromhex(trk.uuid_hex)
        if uid in by_uuid:
            raise PtxValidationError(f"Duplicate track_states target for {trk.name!r}")
        if "folder_open" in st and trk.track_type not in ("RoutingFolder", "BasicFolder"):
            raise PtxValidationError(f"Track {trk.name!r} is type {trk.track_type!r}, not a folder track")
        if "mute" in st and trk.track_type in _NO_MUTE_TYPES:
            raise PtxValidationError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support mute state")
        if "solo" in st and trk.track_type in _NO_SOLO_TYPES:
            raise PtxValidationError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support solo state")
        if "solo_safe" in st and trk.track_type in _NO_SOLO_SAFE_TYPES:
            raise PtxValidationError(f"Track {trk.name!r} (type {trk.track_type!r}) does not support solo_safe state")
        if "timebase" in st and trk.track_type == "BasicFolder":
            raise PtxValidationError(f"Track {trk.name!r} (type 'BasicFolder') does not support timebase state")
        by_uuid[uid] = st
        by_idx[trk.index] = st
        summary[trk.name] = dict(st)
    return by_uuid, by_idx, summary


def patch_track_states_on_buffer(
    ptx_buf: PtxBuffer,
    track_states: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Patch per-track states in-place on a decrypted ``PtxBuffer``."""
    if not track_states:
        raise PtxValidationError("track_states must be a non-empty {track_name: state} mapping")
    session = ptx_buf.session
    by_uuid, by_idx, summary = _validate_track_state_map(session, track_states)
    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(ptx_buf.data):
        raise PtxCorruptFileError("Missing or truncated 0x2624 mixer chunk in .ptx")
    matched: set[bytes] = set()
    for span in iter_mixer_strips(ptx_buf.data, c2624.offset):
        if span.uuid16 is not None and span.uuid16 in by_uuid:
            blk = bytes(ptx_buf.data[span.offset : span.end])
            _apply_single_strip_states(ptx_buf.data, span.offset, blk, by_uuid[span.uuid16])
            matched.add(span.uuid16)
    if len(matched) != len(by_uuid):
        raise PtxCorruptFileError(f"Located {len(matched)}/{len(by_uuid)} target mixer strips in 0x2624")
    c2519 = session.chunk_map.get(0x2519)
    if c2519 is not None:
        _patch_2519_states_in_place(ptx_buf.data, c2519.offset, c2519.raw, by_idx, len(session.tracks))
    c2587 = session.chunk_map.get(0x2587)
    if c2587 is not None:
        _patch_2587_states_in_place(ptx_buf.data, c2587.offset, len(c2587.raw), by_idx)
    tb_audio = {
        trk.name.lower(): (str(by_idx[trk.index]["timebase"]).strip().lower() == "ticks")
        for trk in session.tracks
        if trk.index in by_idx and "timebase" in by_idx[trk.index] and trk.track_type == "Audio"
    }
    _patch_1054_lane_timebases(ptx_buf.data, session, tb_audio)
    ptx_buf.invalidate()
    return summary


def patch_track_states(
    ptx_raw: bytes,
    track_states: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Patch per-track mute, solo, solo_safe, inactive, hidden, folder_open, color_index, height, and timebase."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    summary = patch_track_states_on_buffer(ptx_buf, track_states)
    return ptx_buf.to_encrypted(), summary


def _rewrite_2107_names(c2107_raw: bytes, rename_by_uuid: Mapping[bytes, str]) -> bytes:
    if len(c2107_raw) < 22:
        raise PtxCorruptFileError("Truncated 0x2107 track UUID registry chunk")
    ver = struct.unpack_from("<H", c2107_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2107_raw, 18)[0]
    items, end_pos = walk_sequential_zchunks(c2107_raw, 22, cnt, expected_tag=0x210B)
    out_items: list[bytes] = []
    matched: set[bytes] = set()
    for it in items:
        if len(it) < 17:
            raise PtxCorruptFileError("Truncated 0x210B track UUID entry")
        old_nlen = struct.unpack_from("<I", it, 13)[0]
        uid_off = 17 + old_nlen
        if uid_off + 16 > len(it):
            raise PtxCorruptFileError("Truncated 0x210B name/UUID payload")
        uid = it[uid_off : uid_off + 16]
        new_name = rename_by_uuid.get(uid)
        if new_name is None:
            out_items.append(it)
            continue
        iver = struct.unpack_from("<H", it, 1)[0]
        out_items.append(build_zchunk(0x210B, it[9:13] + encode_lp_str(new_name) + it[uid_off:], version=iver))
        matched.add(uid)
    if len(matched) != len(rename_by_uuid):
        raise PtxCorruptFileError(f"Located {len(matched)}/{len(rename_by_uuid)} target entries in 0x2107")
    return build_zchunk(0x2107, c2107_raw[9:18] + struct.pack("<I", len(out_items)) + b"".join(out_items) + c2107_raw[end_pos:], version=ver)


def _rewrite_2519_names(c2519_raw: bytes, rename_by_idx: Mapping[int, str], expected_count: int) -> bytes:
    _clean, rows, p1, p2, tail = parse_2519_sections(c2519_raw)
    if len(rows) != expected_count:
        raise PtxCorruptFileError(f"0x2519 track count {len(rows)} != session track count {expected_count}")
    ver = struct.unpack_from("<H", c2519_raw, 1)[0]
    cnt_a = struct.unpack_from("<I", c2519_raw, 19)[0]
    orig_prefix = c2519_raw[9 : 23 + cnt_a * 2]

    def _ren_row(r: bytes, nm: str) -> bytes:
        old_nlen = struct.unpack_from("<I", r, 2)[0]
        if 6 + old_nlen > len(r):
            raise PtxCorruptFileError("Truncated 0x2519 summary row name")
        return r[:2] + encode_lp_str(nm) + r[6 + old_nlen :]

    def _ren_251a(e: bytes, nm: str) -> bytes:
        ever = struct.unpack_from("<H", e, 1)[0]
        old_nlen = struct.unpack_from("<I", e, 11)[0]
        if 15 + old_nlen > len(e):
            raise PtxCorruptFileError("Truncated 0x251A track entry name")
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
        raise PtxCorruptFileError("Missing or truncated 0x2619 track identity inside mixer strip")
    end_19 = p19 + 7 + struct.unpack_from("<I", strip, p19 + 3)[0]
    old_nlen = struct.unpack_from("<I", strip, p19 + 9)[0]
    if p19 + 13 + old_nlen > end_19:
        raise PtxCorruptFileError("Truncated 0x2619 track name length inside mixer strip")
    new_lp = encode_lp_str(new_name)
    delta = len(new_lp) - (4 + old_nlen)
    ancestors = [0]
    for anc_tag in (0x261B, 0x102D):
        aoff = find_zchunk_offset(strip, anc_tag, 9)
        if 9 <= aoff < p19 and aoff + 7 + struct.unpack_from("<I", strip, aoff + 3)[0] >= end_19:
            ancestors.append(aoff)
    buf = bytearray(strip)
    buf[p19 + 9 : p19 + 13 + old_nlen] = new_lp
    resize_zchunk_ancestors(buf, (*ancestors, p19), delta)
    return bytes(buf)


def _rewrite_2624_and_collect_active_playlists(
    c2624_raw: bytes,
    rename_by_uuid: Mapping[bytes, str],
) -> tuple[bytes, dict[int, str], dict[int, str]]:
    if len(c2624_raw) < 13:
        raise PtxCorruptFileError("Truncated 0x2624 mixer container chunk")
    ver = struct.unpack_from("<H", c2624_raw, 1)[0]
    cnt = struct.unpack_from("<I", c2624_raw, 9)[0]
    strips, end_pos = walk_sequential_zchunks(c2624_raw, 13, cnt)
    new_strips: list[bytes] = []
    matched: set[bytes] = set()
    audio_pl_renames: dict[int, str] = {}
    midi_pl_renames: dict[int, str] = {}
    for st in strips:
        uoff = st.find(UUID_PREFIX, 0, min(256, len(st)))
        uid = st[uoff : uoff + 16] if (uoff != -1 and uoff + 16 <= len(st)) else b""
        new_nm = rename_by_uuid.get(uid)
        if new_nm is not None:
            tag = struct.unpack_from("<H", st, 7)[0]
            if tag == 0x261C:
                pl_idx = extract_261c_active_playlist_index(st)
                if pl_idx is not None:
                    audio_pl_renames[pl_idx] = new_nm
            elif tag in (0x2620, 0x2621):
                parsed = parse_midi_strip_playlists(st)
                if parsed is not None:
                    midi_pl_renames[parsed.active_index] = new_nm
            new_strips.append(_rewrite_strip_name(st, new_nm))
            matched.add(uid)
        else:
            new_strips.append(st)
    if len(matched) != len(rename_by_uuid):
        raise PtxCorruptFileError(f"Located {len(matched)}/{len(rename_by_uuid)} target mixer strips in 0x2624")
    rebuilt = build_zchunk(0x2624, struct.pack("<I", len(new_strips)) + b"".join(new_strips) + c2624_raw[end_pos:], version=ver)
    return rebuilt, audio_pl_renames, midi_pl_renames


def _rewrite_counted_item_names(
    container_raw: bytes,
    container_tag: int,
    item_tag: int,
    renames_by_idx: Mapping[int, str],
    fallback_renames_lc: Mapping[str, str],
) -> bytes | None:
    if (not renames_by_idx and not fallback_renames_lc) or len(container_raw) < 13:
        return None
    ver = struct.unpack_from("<H", container_raw, 1)[0]
    cnt = struct.unpack_from("<I", container_raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(container_raw, 13, cnt, expected_tag=item_tag)
    changed = False
    out_items: list[bytes] = []
    for idx, it in enumerate(items):
        if len(it) < 13:
            out_items.append(it)
            continue
        nlen = struct.unpack_from("<I", it, 9)[0]
        if 13 + nlen > len(it):
            out_items.append(it)
            continue
        new_name = renames_by_idx.get(idx)
        if new_name is None and not renames_by_idx:
            old_name = it[13 : 13 + nlen].decode("utf-8", "replace")
            new_name = fallback_renames_lc.get(old_name.lower())
        if new_name is not None:
            iver = struct.unpack_from("<H", it, 1)[0]
            out_items.append(build_zchunk(item_tag, encode_lp_str(new_name) + it[13 + nlen :], version=iver))
            changed = True
        else:
            out_items.append(it)
    if not changed:
        return None
    return build_zchunk(container_tag, struct.pack("<I", len(out_items)) + b"".join(out_items) + container_raw[end_pos:], version=ver)


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
            raise PtxValidationError(f"New track name for {old_nm!r} must not be empty")
        trk = session.track_by_name(old_nm)
        if trk is None:
            raise PtxNotFoundError(f"Track {old_nm!r} not found in .ptx session")
        if trk.index in rename_by_idx:
            raise PtxValidationError(f"Duplicate rename target for track {trk.name!r}")
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
            raise PtxValidationError(f"Track rename collision on {rename_by_idx.get(trk.index, trk.name)!r}")
        final_names_lc.add(nm_lc)
    return rename_by_idx, rename_by_uuid, audio_renames_lc, midi_renames_lc, summary


def patch_rename_tracks_on_buffer(ptx_buf: PtxBuffer, renames: Mapping[str, str]) -> dict[str, str]:
    """Rename tracks across ``0x2107``, ``0x2519``, ``0x2624``, ``0x1015``, ``0x1058`` in-place on *ptx_buf*."""
    if not renames:
        raise PtxValidationError("renames must be a non-empty {old_name: new_name} mapping")
    session = ptx_buf.session
    rename_by_idx, rename_by_uuid, audio_renames_lc, midi_renames_lc, summary = _validate_rename_map(session, renames)
    old_dec = bytes(ptx_buf.data)
    template_idx = parse_index_0002(old_dec)
    c2107, c2519, c2624 = session.chunk_map.get(0x2107), session.chunk_map.get(0x2519), session.chunk_map.get(0x2624)
    if c2107 is None or c2519 is None or c2624 is None:
        raise PtxCorruptFileError("Missing required 0x2107/0x2519/0x2624 chunk in .ptx for track rename")
    new_2624, audio_pl_by_idx, midi_pl_by_idx = _rewrite_2624_and_collect_active_playlists(c2624.raw, rename_by_uuid)
    replacements: dict[int, bytes] = {
        0x2107: _rewrite_2107_names(c2107.raw, rename_by_uuid),
        0x2519: _rewrite_2519_names(c2519.raw, rename_by_idx, len(session.tracks)),
        0x2624: new_2624,
    }
    c1015 = session.chunk_map.get(0x1015)
    if c1015 is not None:
        new_1015 = _rewrite_counted_item_names(c1015.raw, 0x1015, 0x1014, audio_pl_by_idx, audio_renames_lc)
        if new_1015 is not None:
            replacements[0x1015] = new_1015
    c1058 = session.chunk_map.get(0x1058)
    if c1058 is not None:
        new_1058 = _rewrite_counted_item_names(c1058.raw, 0x1058, 0x1057, midi_pl_by_idx, midi_renames_lc)
        if new_1058 is not None:
            replacements[0x1058] = new_1058
    out = assemble_top_chunks(old_dec, replacements)
    if template_idx is not None:
        apply_rebuilt_index_0002(out, template_idx, old_dec)
    ptx_buf.replace_data(out)
    return summary


def patch_rename_tracks(ptx_raw: bytes, renames: Mapping[str, str]) -> tuple[bytes, dict[str, str]]:
    """Rename tracks across ``0x2107``, ``0x2519``, ``0x2624``, ``0x1015``, ``0x1058`` and rebuild ``0x0002``."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    summary = patch_rename_tracks_on_buffer(ptx_buf, renames)
    return ptx_buf.to_encrypted(), summary
