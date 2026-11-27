"""Mixer strip, automation lane, AAX plugin state, and ARA2 parsers for Pro Tools ``.ptx``."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
import re
import struct
from typing import Protocol

from .codec import find_subchunks, format_rfc_uuid, lookup_pt_color_rgb, parse_control_curve, read_lp_str
from .models import (
    AUTOMATION_MODE_MAP,
    SEND_SLOTS,
    UUID_PREFIX,
    PtxAraBinding,
    PtxAraPlugin,
    PtxAutomationLane,
    PtxAutomationPoint,
    PtxInsert,
    PtxPluginAutomationLane,
    PtxPluginInfo,
    PtxPluginState,
    PtxSend,
    PtxSignalPath,
    PtxSignalPathRef,
    PtxTrack,
)


class SignalPathResolver(Protocol):
    """Callable resolving a 16-byte signal-path UUID into a ``PtxSignalPathRef``."""

    def __call__(self, raw_uuid: bytes, fallback_name: str = "") -> PtxSignalPathRef | None: ...


@dataclass
class OutputSlotParams:
    """Decoded parameters and automation lanes from a ``0x260d`` output/send slot."""

    volume_db: float | None = None
    is_muted: bool = False
    lfe_db: float | None = None
    pan_left: int | None = None
    pan_right: int | None = None
    timebase: str = "Samples"
    volume_automation: PtxAutomationLane | None = None
    mute_automation: PtxAutomationLane | None = None
    lfe_or_trim_automation: PtxAutomationLane | None = None
    pan_left_automation: PtxAutomationLane | None = None
    pan_right_automation: PtxAutomationLane | None = None


def parse_ara_plugin_4807(sub: bytes, index: int = 0) -> PtxAraPlugin | None:
    """Parse a ``0x4807`` ARA2 plugin descriptor chunk."""
    if len(sub) < 25:
        return None
    m4 = sub[9:13][::-1].decode("ascii", errors="replace")
    p4 = sub[13:17][::-1].decode("ascii", errors="replace")
    t4 = sub[17:21][::-1].decode("ascii", errors="replace")
    name, _ = read_lp_str(sub, 21, max_len=160)
    if not name:
        return None
    return PtxAraPlugin(index=index, name=name, manufacturer_4cc=m4, product_4cc=p4, type_4cc=t4)


def parse_ara_registry_4823(raw: bytes) -> list[PtxAraPlugin]:
    """Walk top-level ``0x4823`` sequentially (global blob -> count -> ``0x4807`` + state pairs)."""
    plugins: list[PtxAraPlugin] = []
    if len(raw) < 17:
        return plugins
    global_len = struct.unpack_from("<I", raw, 9)[0]
    pos = 13 + global_len
    if pos + 4 > len(raw):
        return plugins
    count = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    for idx in range(min(count, 4096)):
        if pos + 9 > len(raw) or raw[pos] != 0x5A:
            break
        ver = struct.unpack_from("<H", raw, pos + 1)[0]
        csz = struct.unpack_from("<I", raw, pos + 3)[0]
        tag = struct.unpack_from("<H", raw, pos + 7)[0]
        sub_end = pos + 7 + csz
        if ver >= 0x1000 or csz < 2 or sub_end > len(raw) or tag != 0x4807:
            break
        plug = parse_ara_plugin_4807(raw[pos:sub_end], index=idx)
        pos = sub_end
        if pos + 4 > len(raw):
            if plug is not None:
                plugins.append(plug)
            break
        st_len = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if st_len > len(raw) - pos:
            if plug is not None:
                plugins.append(plug)
            break
        blob = raw[pos : pos + st_len]
        pos += st_len
        if plug is not None:
            if st_len > 0:
                plug.has_state = True
                plug.state_size = st_len
                plug.state_b64 = base64.b64encode(blob).decode("ascii")
                if len(blob) >= 56:
                    plug.state_label = blob[24:56].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            plugins.append(plug)
    return plugins


def parse_ara_bindings_4820(raw: bytes) -> list[PtxAraBinding]:
    """Parse top-level ``0x4820`` -> ``0x480c`` ARA2 document and audio-source bindings."""
    bindings: list[PtxAraBinding] = []
    for s0c in find_subchunks(raw, 0x480C):
        chunk_id, _ = read_lp_str(s0c, 9, max_len=256)
        binding = PtxAraBinding(chunk_id=chunk_id)
        s07 = find_subchunks(s0c, 0x4807)
        if s07:
            plug = parse_ara_plugin_4807(s07[0])
            if plug is not None:
                binding.plugin_name = plug.name
                binding.manufacturer_4cc = plug.manufacturer_4cc
                binding.product_4cc = plug.product_4cc
                binding.type_4cc = plug.type_4cc
        for s0b in find_subchunks(s0c, 0x480B):
            if len(s0b) >= 29:
                as_id, _ = read_lp_str(s0b, 25, max_len=256)
                if as_id:
                    binding.audio_source_ids.append(as_id)
            for s0a in find_subchunks(s0b, 0x480A):
                if len(s0a) >= 29:
                    am_uuid = format_rfc_uuid(s0a[9:25])
                    am_id, off = read_lp_str(s0a, 25, max_len=256)
                    ds_id, _ = read_lp_str(s0a, off, max_len=256)
                    if am_uuid:
                        binding.audio_modification_uuids.append(am_uuid)
                    if am_id:
                        binding.audio_modification_ids.append(am_id)
                    if ds_id:
                        binding.document_state_ids.append(ds_id)
        bindings.append(binding)
    return bindings


def parse_complete_controls_state(raw: bytes) -> dict[str, int | float]:
    """Parse Avid ``Complete Controls State`` tagged parameters (``l_`` i32 / ``d_`` f64 BE)."""
    marker = raw.find(b"Complete Controls State")
    if marker == -1:
        return {}
    p1 = raw.find(b"l_", marker)
    p2 = raw.find(b"d_", marker)
    starts = [p for p in (p1, p2) if p != -1]
    if not starts:
        return {}
    pos = min(starts)
    controls: dict[str, int | float] = {}
    while pos + 8 <= len(raw):
        kind = raw[pos : pos + 2]
        if kind not in (b"l_", b"d_"):
            break
        tag = raw[pos + 2 : pos + 8].rstrip(b"\x00").decode("ascii", errors="replace")
        if kind == b"l_":
            if pos + 12 > len(raw):
                break
            controls[tag] = int(struct.unpack_from(">i", raw, pos + 8)[0])
            pos += 12
        else:
            if pos + 16 > len(raw):
                break
            val = float(struct.unpack_from(">d", raw, pos + 8)[0])
            if math.isfinite(val):
                controls[tag] = val
            pos += 16
    return controls


def parse_plugin_state_1038(s1038: bytes) -> PtxPluginState | None:
    """Parse an AAX plugin preset/state payload from insert chunk ``0x1038``."""
    if len(s1038) < 73:
        return None
    m4 = s1038[25:29].decode("ascii", errors="replace")
    p4 = s1038[29:33].decode("ascii", errors="replace")
    t4 = s1038[33:37].decode("ascii", errors="replace")
    st4 = s1038[37:41].decode("ascii", errors="replace")
    label = s1038[41:73].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    payload = s1038[73:]
    return PtxPluginState(
        manufacturer_4cc=m4,
        product_4cc=p4,
        plugin_type_4cc=t4,
        state_type_4cc=st4,
        state_label=label,
        payload_size=len(payload),
        payload_b64=base64.b64encode(payload).decode("ascii"),
        controls=parse_complete_controls_state(s1038),
    )


def parse_plugin_automation_2614(slot_blk: bytes) -> list[PtxPluginAutomationLane]:
    """Decode plugin parameter automation lanes from ``0x2614`` -> ``0x260f`` -> ``0x260b``."""
    b14_list = find_subchunks(slot_blk, 0x2614)
    if not b14_list:
        return []
    lanes: list[PtxPluginAutomationLane] = []
    for lane in find_subchunks(b14_list[0], 0x260F):
        cb_list = find_subchunks(lane, 0x260B)
        if not cb_list:
            continue
        cb = cb_list[0]
        parsed = parse_control_curve(cb, 9, signed_int32=True)
        if parsed is None:
            continue
        timebase, raw_pts, _ = parsed
        cb_pos = lane.find(cb, 9)
        if cb_pos == -1:
            continue
        pname, _ = read_lp_str(lane, cb_pos + len(cb), max_len=256)
        ptag = "?"
        if len(lane) >= 8 and struct.unpack_from("<I", lane, len(lane) - 8)[0] == 4:
            ptag = lane[-4:].decode("ascii", errors="replace")
        norm_pts = [PtxAutomationPoint(pt.position, pt.value / 2147483648.0) for pt in raw_pts if math.isfinite(pt.value)]
        lanes.append(PtxPluginAutomationLane(ptag, pname, timebase, norm_pts))
    return lanes


def _decode_260a_lane(chunk_260a: bytes, parameter: str, scale: float) -> tuple[float | None, PtxAutomationLane | None]:
    parsed = parse_control_curve(chunk_260a, 9)
    if parsed is not None:
        timebase, raw_pts, _ = parsed
        scaled = [PtxAutomationPoint(pt.position, round(pt.value * scale, 4)) for pt in raw_pts if math.isfinite(pt.value)]
        if not scaled:
            return None, None
        return scaled[0].value, PtxAutomationLane(parameter, timebase, scaled)
    if len(chunk_260a) >= 39:
        raw_val = struct.unpack_from("<h", chunk_260a, len(chunk_260a) - 4)[0]
        tb = "Ticks" if chunk_260a[17] == 1 else "Samples"
        val = round(raw_val * scale, 4)
        return val, PtxAutomationLane(parameter, tb, [PtxAutomationPoint(0, val)])
    return None, None


def extract_260d_params(chunk_260d: bytes) -> OutputSlotParams:
    """Decode static values and multi-point automation lanes from a ``0x260d`` block."""
    c_list = find_subchunks(chunk_260d, 0x260C)
    first_c_pos = chunk_260d.find(c_list[0], 9) if c_list else -1
    head_part = chunk_260d[:first_c_pos] if first_c_pos != -1 else chunk_260d
    direct_a = find_subchunks(head_part, 0x260A)
    res = OutputSlotParams()
    if len(direct_a) >= 1:
        val, lane = _decode_260a_lane(direct_a[0], "volume", 0.1)
        if val is not None:
            res.volume_db = round(val, 1)
        if lane is not None:
            res.volume_automation = lane
            res.timebase = lane.timebase
    if len(direct_a) >= 2:
        val, lane = _decode_260a_lane(direct_a[1], "mute", 1.0)
        if val is not None:
            res.is_muted = bool(val)
        res.mute_automation = lane
    if len(direct_a) >= 3:
        val, lane = _decode_260a_lane(direct_a[2], "lfe_or_trim", 0.1)
        if val is not None:
            res.lfe_db = round(val, 1)
        res.lfe_or_trim_automation = lane
    if len(c_list) >= 1:
        pa = find_subchunks(c_list[0], 0x260A)
        if pa:
            val, lane = _decode_260a_lane(pa[0], "pan_left", 1.0)
            res.pan_left = int(round(val)) if (val is not None and math.isfinite(val)) else None
            res.pan_left_automation = lane
    if len(c_list) >= 2:
        pa = find_subchunks(c_list[1], 0x260A)
        if pa:
            val, lane = _decode_260a_lane(pa[0], "pan_right", 1.0)
            res.pan_right = int(round(val)) if (val is not None and math.isfinite(val)) else None
            res.pan_right_automation = lane
    return res


def parse_2015_metadata(blk: bytes, *, start: int = 9) -> tuple[str, int | None, bool, bytes | None]:
    """Extract ``(comment, color_index, is_solo_safe, first_2038)`` from a track's ``0x2015`` block."""
    s15 = find_subchunks(blk, 0x2015, start=start)
    if not s15:
        return "", None, False, None
    b15 = s15[-1]
    p15 = blk.rfind(b15)
    is_solo_safe = p15 != -1 and p15 + len(b15) < len(blk) and blk[p15 + len(b15)] == 1
    s38 = find_subchunks(b15, 0x2038)
    if not s38:
        return "", None, is_solo_safe, None
    p38 = b15.find(s38[0], 9)
    if p38 == -1:
        return "", None, is_solo_safe, s38[0]
    fend = p38 + len(s38[0])
    comment = ""
    clen = 0
    if fend + 12 > len(b15):
        return "", None, is_solo_safe, s38[0]
    cand_len = struct.unpack_from("<I", b15, fend + 8)[0]
    if cand_len > 0:
        if cand_len >= 4096 or fend + 12 + cand_len > len(b15):
            return "", None, is_solo_safe, s38[0]
        clen = cand_len
        comment = b15[fend + 12 : fend + 12 + clen].decode("utf-8", errors="replace")
    q = fend + 21 + clen
    color_idx: int | None = None
    if q < len(b15):
        if b15[q] == 1 and q + 8 <= len(b15) and b15[q + 1] == 0x5A:
            sub_sz = struct.unpack_from("<I", b15, q + 4)[0]
            q2 = q + 1 + 7 + sub_sz
        else:
            q2 = q + 1
        if q2 + 3 <= len(b15):
            col = struct.unpack_from("<h", b15, q2 + 1)[0]
            if col >= 0:
                color_idx = int(col)
    return comment, color_idx, is_solo_safe, s38[0]


def parse_track_flags(track: PtxTrack, blk: bytes, ara_plugins: list[PtxAraPlugin]) -> None:
    """Populate track solo, solo-safe, automation mode, record, monitor, comment, color, and ARA2 state."""
    s2619 = find_subchunks(blk[:512], 0x2619)
    if s2619:
        p19 = blk.find(s2619[0], 9)
        if p19 != -1 and p19 + len(s2619[0]) + 2 <= len(blk):
            e19 = p19 + len(s2619[0])
            track.is_soloed = bool(blk[e19])
            am_code = blk[e19 + 1]
            track.automation_mode_code = am_code
            track.automation_mode = AUTOMATION_MODE_MAP.get(am_code, f"Mode_{am_code}")
    else:
        s4301 = find_subchunks(blk[:512], 0x4301)
        if s4301:
            h4301 = blk.find(s4301[0], 9)
            if h4301 != -1 and h4301 + 61 <= len(blk):
                track.is_soloed = bool(blk[h4301 + 59])
                am_code = blk[h4301 + 60]
                track.automation_mode_code = am_code
                track.automation_mode = AUTOMATION_MODE_MAP.get(am_code, f"Mode_{am_code}")
    comment, col_2015, solo_safe, b2038 = parse_2015_metadata(blk, start=9)
    if b2038 is None:
        tail_blk = blk[max(0, len(blk) - 900) :]
        s2038 = find_subchunks(tail_blk, 0x2038, start=0)
        b2038 = s2038[0] if s2038 else None
    if b2038 is not None and len(b2038) >= 20:
        track.is_record_enabled = bool(b2038[18])
        track.is_record_safe = bool(b2038[19])
    if comment:
        track.comment = comment
    track.is_solo_safe = solo_safe
    if track.color_index == 0xFFFE and col_2015 is not None:
        track.color_index = col_2015
    track.color_rgb = lookup_pt_color_rgb(track.color_index)
    if track.track_type == "Audio" and len(blk) > 33:
        track.is_input_monitor = bool(blk[-33])
    tail_ara = blk[max(0, len(blk) - 900) :]
    s4807 = find_subchunks(tail_ara, 0x4807, start=0)
    if s4807:
        ara = parse_ara_plugin_4807(s4807[0])
        if ara is not None:
            for reg in ara_plugins:
                if reg.name == ara.name and reg.has_state:
                    ara.has_state = reg.has_state
                    ara.state_size = reg.state_size
                    ara.state_label = reg.state_label
                    ara.state_b64 = reg.state_b64
                    break
            track.ara_plugin = ara


def parse_vca_mixer(track: PtxTrack, blk: bytes) -> None:
    """Decode volume and mute automation lanes for a VCA ``0x261f`` mixer strip."""
    s2610 = find_subchunks(blk, 0x2610)
    if not s2610:
        return
    a_list = find_subchunks(s2610[0], 0x260A)
    if len(a_list) >= 1:
        val, lane = _decode_260a_lane(a_list[0], "volume", 0.1)
        if val is not None:
            track.volume_db = round(val, 1)
        if lane is not None:
            track.volume_automation = lane
            track.timebase = lane.timebase
    if len(a_list) >= 2:
        val, lane = _decode_260a_lane(a_list[1], "mute", 1.0)
        if val is not None:
            track.is_muted = bool(val)
        track.mute_automation = lane


def parse_track_input(
    track: PtxTrack,
    blk: bytes,
    resolve_uuid: SignalPathResolver,
    io_paths: list[PtxSignalPath],
) -> None:
    """Resolve track input from pre-``0x4856`` UUID slot or pre-``0x2627`` raw I/O index fallback."""
    s2627 = find_subchunks(blk, 0x2627)
    if s2627:
        p27 = blk.find(s2627[0], 9)
        if p27 >= 3:
            raw_in = struct.unpack_from("<H", blk, p27 - 3)[0]
            if raw_in != 0xFFFF:
                track.input_path_raw_index = raw_in
    s4856_list = find_subchunks(blk, 0x4856)
    if s4856_list:
        h4856 = blk.rfind(s4856_list[-1])
        pre = blk[max(0, h4856 - 480) : h4856]
        for match in re.finditer(re.escape(UUID_PREFIX), pre):
            ref = resolve_uuid(pre[match.start() : match.start() + 16])
            if ref is not None:
                track.input_path = ref
                return
        return
    if track.input_path_raw_index is not None and track.input_path_raw_index < len(io_paths):
        sp = io_paths[track.input_path_raw_index]
        track.input_path = PtxSignalPathRef(sp.name, sp.kind, sp.uuid_hex, None)


def _extract_260e_ref_and_active(
    chunk_260e: bytes,
    resolve_uuid: SignalPathResolver,
) -> tuple[PtxSignalPathRef | None, bool]:
    is_active = not (len(chunk_260e) > 12 and chunk_260e[12] == 0)
    fallback_name = ""
    if len(chunk_260e) >= 49:
        fallback_name, _ = read_lp_str(chunk_260e, 45, max_len=256)
    uoff = chunk_260e.find(UUID_PREFIX)
    if uoff != -1 and uoff + 16 <= len(chunk_260e):
        ref = resolve_uuid(chunk_260e[uoff : uoff + 16], fallback_name)
        if ref is not None:
            return ref, is_active
    if fallback_name:
        return PtxSignalPathRef(fallback_name, "unknown", "", None), is_active
    return None, is_active


def parse_track_outputs_and_sends(
    track: PtxTrack,
    blk: bytes,
    resolve_uuid: SignalPathResolver,
    *,
    derive_inactive_from_output: bool = False,
) -> None:
    """Decode Main Output and Sends A..J (``0x260d`` blocks) including automation lanes."""
    for di, d_chunk in enumerate(find_subchunks(blk, 0x260D)):
        e_list = find_subchunks(d_chunk, 0x260E)
        if not e_list:
            continue
        dest, is_active = _extract_260e_ref_and_active(e_list[0], resolve_uuid)
        if dest is None:
            continue
        p = extract_260d_params(d_chunk)
        if di == 0:
            track.main_output = dest
            track.is_output_active = is_active
            if derive_inactive_from_output and not is_active:
                track.is_inactive = True
            track.volume_db = p.volume_db
            track.is_muted = p.is_muted
            track.lfe_db = p.lfe_db
            track.pan_left = p.pan_left
            track.pan_right = p.pan_right
            track.timebase = p.timebase
            track.volume_automation = p.volume_automation
            track.mute_automation = p.mute_automation
            track.lfe_or_trim_automation = p.lfe_or_trim_automation
            track.pan_left_automation = p.pan_left_automation
            track.pan_right_automation = p.pan_right_automation
        else:
            slot_idx = di - 1
            slot = SEND_SLOTS[slot_idx] if slot_idx < len(SEND_SLOTS) else str(slot_idx)
            s1029 = find_subchunks(d_chunk, 0x1029)
            is_pre = bool(s1029 and len(s1029[0]) > 9 and s1029[0][9] == 0)
            track.sends.append(
                PtxSend(
                    slot, slot_idx, dest, p.volume_db, p.is_muted, is_pre,
                    p.lfe_db, p.pan_left, p.pan_right,
                    is_active=is_active,
                    volume_automation=p.volume_automation,
                    mute_automation=p.mute_automation,
                    lfe_or_trim_automation=p.lfe_or_trim_automation,
                    pan_left_automation=p.pan_left_automation,
                    pan_right_automation=p.pan_right_automation,
                )
            )


def _resolve_insert_plugin(
    slot_blk: bytes, s2626: bytes, plugins: list[PtxPluginInfo]
) -> tuple[int | None, str, str, str]:
    p26 = slot_blk.find(s2626, 9)
    if p26 != -1 and p26 + len(s2626) + 2 <= len(slot_blk):
        cand = struct.unpack_from("<H", slot_blk, p26 + len(s2626))[0]
        if cand < len(plugins):
            pinfo = plugins[cand]
            fourcc = f"{pinfo.manufacturer_4cc}/{pinfo.product_4cc}/{pinfo.type_4cc}"
            return cand, pinfo.name, pinfo.bundle_id, fourcc
    return None, "", "", ""


def _parse_insert_flags_and_state(
    slot_blk: bytes, *, default_active: bool
) -> tuple[bool, bool, PtxPluginState | None]:
    is_active = default_active
    is_bypassed = False
    s2613 = find_subchunks(slot_blk, 0x2613)
    s1038 = find_subchunks(slot_blk, 0x1038)
    state = parse_plugin_state_1038(s1038[0]) if s1038 else None
    if s2613:
        b13 = s2613[0]
        if default_active and s1038:
            p38 = b13.find(s1038[0], 9)
            if p38 != -1 and p38 + len(s1038[0]) < len(b13):
                is_active = bool(b13[p38 + len(s1038[0])])
        p13 = slot_blk.find(b13, 9)
        if p13 != -1 and p13 + len(b13) < len(slot_blk):
            is_bypassed = bool(slot_blk[p13 + len(b13)])
    else:
        s2614 = find_subchunks(slot_blk, 0x2614)
        if s2614:
            p14 = slot_blk.find(s2614[0], 9)
            if p14 >= 1:
                is_bypassed = bool(slot_blk[p14 - 1])
    return is_active, is_bypassed, state


def parse_track_inserts(
    track: PtxTrack,
    blk: bytes,
    plugins: list[PtxPluginInfo],
    resolve_uuid: SignalPathResolver,
) -> None:
    """Decode track inserts A..J (``0x2627`` -> ``0x2625``/``0x2616``/``0x2617``)."""
    s2627 = find_subchunks(blk, 0x2627)
    if not s2627:
        return
    b27 = s2627[0]
    pos = 11 if (len(b27) > 11 and b27[11] == 0x5A) else 9
    si = 0
    while pos + 9 <= len(b27) and b27[pos] == 0x5A:
        ver = struct.unpack_from("<H", b27, pos + 1)[0]
        csz = struct.unpack_from("<I", b27, pos + 3)[0]
        tag = struct.unpack_from("<H", b27, pos + 7)[0]
        end = pos + 7 + csz
        if ver >= 0x1000 or csz < 2 or end > len(b27):
            break
        if tag == 0x2625:
            si += 1
        elif tag == 0x2616:
            _append_insert_slot(track, si, b27[pos:end], plugins, resolve_uuid, default_active=True)
            si += 1
        elif tag == 0x2617:
            inner = find_subchunks(b27[pos:end], 0x2616)
            if inner:
                _append_insert_slot(track, si, inner[0], plugins, resolve_uuid, default_active=False)
            si += 1
        pos = end


def _append_insert_slot(
    track: PtxTrack,
    slot_index: int,
    slot_blk: bytes,
    plugins: list[PtxPluginInfo],
    resolve_uuid: SignalPathResolver,
    *,
    default_active: bool,
) -> None:
    s2626 = find_subchunks(slot_blk, 0x2626)
    if s2626:
        plug_idx, plug_name, bundle_id, fourcc = _resolve_insert_plugin(slot_blk, s2626[0], plugins)
    else:
        plug_idx, plug_name, bundle_id, fourcc = None, "(Hardware / Unknown Insert)", "", ""
    sc_chunks = find_subchunks(slot_blk, 0x260E)
    sc_ref = _extract_260e_ref_and_active(sc_chunks[0], resolve_uuid)[0] if sc_chunks else None
    is_active, is_bypassed, state = _parse_insert_flags_and_state(slot_blk, default_active=default_active)
    automation = parse_plugin_automation_2614(slot_blk)
    slot = SEND_SLOTS[slot_index] if slot_index < len(SEND_SLOTS) else str(slot_index)
    track.inserts.append(
        PtxInsert(
            slot, slot_index, plug_idx, plug_name, bundle_id, fourcc,
            is_active, is_bypassed, sc_ref, state=state, automation=automation,
        )
    )
    # Only direct 0x2616 slots (default_active=True) are eligible for Slot A Trim patching/donors.
    if default_active and slot_index == 0 and plug_idx is not None and plug_idx < len(plugins):
        from .plugin_trim import inspect_slot_a_trim
        track.slot_a_trim = inspect_slot_a_trim(
            slot_blk, plugins[plug_idx],
            is_active=is_active, is_bypassed=is_bypassed,
            track_format=track.format, has_sidechain=(sc_ref is not None),
        )
