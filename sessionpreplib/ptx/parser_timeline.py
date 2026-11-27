"""Timeline, tempo/meter/markers, media, clips, fades, MIDI, and video parsers for Pro Tools ``.ptx``."""

from __future__ import annotations

import base64
import math
import re
import struct

from .codec import (
    decode_ticks_relative,
    find_subchunks,
    format_ptsl_uuid,
    format_rfc_uuid,
    format_timecode,
    parse_control_curve,
    read_lp_str,
)
from .models import (
    CROSSFADE_LINK_MAP,
    FADE_SHAPE_MAP,
    FADE_TYPE_MAP,
    FPS_CODE_MAP,
    MARKER_TIME_PROPS,
    MIDI_CONTROL_LANE_TYPE_MAP,
    TICK_MASK_56,
    TICKS_PER_QUARTER,
    UUID_PREFIX,
    ZERO_TICKS,
    PtxAudioFile,
    PtxChunk,
    PtxClip,
    PtxClipGainBreakpoint,
    PtxClipGainCurve,
    PtxClipPlacement,
    PtxCompoundClip,
    PtxCompoundClipMember,
    PtxFade,
    PtxFadePlacement,
    PtxMarker,
    PtxMeterEvent,
    PtxMidiChunk,
    PtxMidiClip,
    PtxMidiControlEvent,
    PtxMidiControlLane,
    PtxMidiNote,
    PtxMidiPlacement,
    PtxPlaylist,
    PtxSessionInfo,
    PtxTempoEvent,
    PtxTrack,
    PtxVideoClip,
    PtxVideoPlacement,
    PtxWarpMarker,
)
from .parser_mixer import parse_ara_plugin_4807

_MAC_TOP_DIRS = frozenset({"Users", "Applications", "Library", "System", "private", "Volumes"})
_VIDEO_EXTS = frozenset({"mov", "mp4", "m4v", "avi", "mxf", "mkv"})


def _join_session_path(parts: list[str]) -> str:
    cleaned = [p.rstrip("/\\") for p in parts if p]
    if not cleaned:
        return ""
    if re.match(r"^[A-Za-z]:$", cleaned[0]) or cleaned[0].startswith("\\"):
        return "\\".join(cleaned)
    if len(cleaned) >= 2 and cleaned[1] in _MAC_TOP_DIRS:
        return "/" + "/".join(cleaned[1:])
    return "/Volumes/" + "/".join(cleaned)


def parse_session_start_204d(raw: bytes, info: PtxSessionInfo) -> None:
    """Populate timecode frame rate and session start offset from ``0x204d``."""
    body = raw[9:]
    if len(body) < 13:
        return
    fps_code = struct.unpack_from("<I", body, 0)[0]
    start_frames = struct.unpack_from("<I", body, 9)[0]
    info.fps_code = fps_code
    info.start_frames = start_frames
    fps_info = FPS_CODE_MAP.get(fps_code)
    if fps_info is not None:
        real_fps, _nominal, label, is_drop = fps_info
        info.frame_rate = real_fps
        info.frame_rate_label = label
        info.is_drop_frame = is_drop
        info.start_seconds = round(start_frames / real_fps, 6)
        info.start_timecode = format_timecode(start_frames, fps_code)


def parse_tempo_2028(raw: bytes) -> tuple[list[PtxTempoEvent], int, float | None]:
    """Decode ``0x2028`` tempo map into ``(events, declared_count, fallback_bpm)``."""
    body = raw[9:]
    declared = struct.unpack_from("<I", body, 11)[0] if len(body) >= 15 else 0
    events: list[PtxTempoEvent] = []
    pos = raw.find(b"Const\x01\x00", 9)
    while pos != -1 and pos + 48 <= len(raw):
        tick_raw = struct.unpack_from("<Q", raw, pos + 30)[0]
        bpm = struct.unpack_from("<d", raw, pos + 40)[0]
        if 1.0 <= bpm <= 1000.0 and tick_raw >= ZERO_TICKS:
            events.append(PtxTempoEvent(tick=tick_raw - ZERO_TICKS, bpm=round(bpm, 4)))
        step = struct.unpack_from("<I", raw, pos + 7)[0] if pos + 11 <= len(raw) else 0
        nxt = pos + 11 + step if step > 0 else pos + 7
        pos = raw.find(b"Const\x01\x00", max(pos + 7, nxt))
    events.sort(key=lambda e: (e.tick, e.bpm))
    _integrate_tempo_seconds(events)
    if events:
        return events, declared, events[0].bpm
    fallback = _fallback_tempo_bpm(raw)
    return events, declared, fallback


def _integrate_tempo_seconds(events: list[PtxTempoEvent]) -> None:
    cur_sec = 0.0
    for idx, ev in enumerate(events):
        if idx > 0:
            prev = events[idx - 1]
            cur_sec += (ev.tick - prev.tick) / float(TICKS_PER_QUARTER) * 60.0 / prev.bpm
        ev.seconds = round(cur_sec, 6)


def _fallback_tempo_bpm(raw: bytes) -> float | None:
    sent = raw.find(b"\x00\x10\xa5\xd4\xe8\x00")
    if sent == -1 or sent + 22 > len(raw):
        return None
    for off in range(sent + 6, len(raw) - 7):
        val = struct.unpack_from("<d", raw, off)[0]
        if 20.0 <= val <= 400.0 and abs(val * 100 - round(val * 100)) < 1e-4:
            return round(val, 4)
    return None


def ticks_to_seconds(events: list[PtxTempoEvent], tick: int, default_bpm: float = 120.0) -> float:
    """Convert a relative Pro Tools tick offset to seconds via piecewise-constant tempo integration."""
    if not events:
        return tick / float(TICKS_PER_QUARTER) * 60.0 / (default_bpm or 120.0)
    active = events[0]
    if tick < active.tick:
        return tick / float(TICKS_PER_QUARTER) * 60.0 / active.bpm
    for ev in events:
        if ev.tick <= tick:
            active = ev
        else:
            break
    return active.seconds + (tick - active.tick) / float(TICKS_PER_QUARTER) * 60.0 / active.bpm


def parse_meter_2029(raw: bytes) -> list[PtxMeterEvent]:
    """Decode ``0x2029`` meter / time-signature events."""
    body = raw[9:]
    if len(body) < 15:
        return []
    count = struct.unpack_from("<I", body, 11)[0]
    pos = 15
    meters: list[PtxMeterEvent] = []
    max_entries = min(count, (len(body) - pos) // 36)
    for _ in range(max_entries):
        tick_raw = struct.unpack_from("<Q", body, pos)[0]
        click_val, num, den = struct.unpack_from("<III", body, pos + 8)
        if tick_raw >= ZERO_TICKS and num > 0 and den > 0:
            meters.append(PtxMeterEvent(tick=tick_raw - ZERO_TICKS, numerator=num, denominator=den, click_value=click_val))
        pos += 36
    return meters


def parse_markers_2030(chunks: list[PtxChunk], sample_rate: int) -> list[PtxMarker]:
    """Decode Memory Locations / Markers from ``0x2030`` -> ``0x2077`` chunks."""
    markers: list[PtxMarker] = []
    seen: set[tuple[int, str, int, int]] = set()
    for ch in chunks:
        targets = [ch.raw] if ch.tag == 0x2030 else find_subchunks(ch.raw, 0x2030)
        for m_raw in targets:
            for sub in find_subchunks(m_raw, 0x2077):
                marker = _decode_marker_2077(sub, sample_rate)
                if marker is None:
                    continue
                key = (marker.number, marker.name, marker.start_sample, marker.end_sample)
                if key not in seen:
                    seen.add(key)
                    markers.append(marker)
    markers.sort(key=lambda m: (m.start_sample, m.number))
    return markers


def _decode_marker_2077(sub: bytes, sample_rate: int) -> PtxMarker | None:
    if len(sub) < 35:
        return None
    num = struct.unpack_from("<H", sub, 9)[0]
    tprop = MARKER_TIME_PROPS.get(sub[11], f"Prop_{sub[11]}")
    ref = "Absolute" if sub[12] == 1 else "BarBeat"
    mname, off = read_lp_str(sub, 15, max_len=256)
    if off + 16 > len(sub):
        return None
    start_smp, end_smp = struct.unpack_from("<QQ", sub, off)
    comments = ""
    s2506 = find_subchunks(sub, 0x2506)
    if s2506:
        last_end = sub.rfind(s2506[-1]) + len(s2506[-1])
        if last_end + 8 <= len(sub):
            comments, _ = read_lp_str(sub, last_end + 4, max_len=1024)
    return PtxMarker(
        number=num,
        name=mname,
        start_sample=start_smp,
        end_sample=end_smp,
        time_properties=tprop,
        reference=ref,
        comments=comments,
        start_seconds=round(start_smp / sample_rate, 4) if sample_rate > 0 else None,
        end_seconds=round(end_smp / sample_rate, 4) if (sample_rate > 0 and end_smp > 0) else None,
    )


def _parse_103a_entries(raw: bytes) -> list[tuple[int, int, str, str]]:
    entries: list[tuple[int, int, str, str]] = []
    for s3a in find_subchunks(raw, 0x103A):
        if len(s3a) < 13:
            continue
        count = struct.unpack_from("<I", s3a, 9)[0]
        pos = 13
        for _ in range(min(count, 65536)):
            if pos + 9 > len(s3a):
                break
            kind = s3a[pos]
            parent = struct.unpack_from("<I", s3a, pos + 1)[0]
            name, after = read_lp_str(s3a, pos + 5, max_len=512)
            if after + 4 > len(s3a):
                break
            tag4 = s3a[after : after + 4].decode("ascii", errors="replace").strip("\x00")
            entries.append((kind, parent, name, tag4))
            pos = after + 4
    return entries


def _resolve_103a_path(entries: list[tuple[int, int, str, str]], idx: int) -> list[str]:
    parts: list[str] = []
    visited: set[int] = set()
    cur = idx
    for _ in range(64):
        if not (0 <= cur < len(entries)) or cur in visited:
            break
        visited.add(cur)
        _kind, parent, name, _tag4 = entries[cur]
        parts.append(name)
        if parent == 0xFFFFFFFF or parent >= len(entries):
            break
        cur = parent
    parts.reverse()
    return parts


def parse_audio_files_1004(raw: bytes) -> tuple[str, list[PtxAudioFile]]:
    """Decode ``0x1004`` audio file directory tree (``0x103a``) and file descriptors (``0x1003``/``0x1001``)."""
    entries = _parse_103a_entries(raw)
    root_parts: list[str] = []
    for idx, (kind, _parent, name, _tag4) in enumerate(entries):
        if kind == 1 and name == "Audio Files":
            full = _resolve_103a_path(entries, idx)
            root_parts = full[:-1]
            break
    session_root = _join_session_path(root_parts) if root_parts else ""
    s1003 = find_subchunks(raw, 0x1003)
    if entries and s1003:
        files = _build_audio_files_from_tree(s1003, entries, root_parts)
        return session_root, files
    return session_root, _parse_audio_files_regex_fallback(raw, s1003)


def _build_audio_files_from_tree(
    s1003: list[bytes],
    entries: list[tuple[int, int, str, str]],
    root_parts: list[str],
) -> list[PtxAudioFile]:
    files: list[PtxAudioFile] = []
    for i, b3 in enumerate(s1003):
        tree_idx, sr, chs, bits, length, fmt_code = _decode_audio_1003(b3, i)
        fname, directory, full_path, rel_path, tag4 = f"file_{i}", "", "", None, ""
        if 0 <= tree_idx < len(entries):
            parts = _resolve_103a_path(entries, tree_idx)
            fname = parts[-1] if parts else fname
            directory = parts[-2] if len(parts) >= 2 else ""
            full_path = _join_session_path(parts)
            tag4 = entries[tree_idx][3]
            if root_parts and len(parts) > len(root_parts) and parts[: len(root_parts)] == root_parts:
                rel_path = "/".join(parts[len(root_parts) :])
        dur = round(length / sr, 4) if (sr > 0 and length > 0) else None
        files.append(
            PtxAudioFile(i, tree_idx, fname, directory, sr, chs, bits, length, dur, fmt_code, full_path, rel_path, tag4)
        )
    return files


def _parse_audio_files_regex_fallback(raw: bytes, s1003: list[bytes]) -> list[PtxAudioFile]:
    directory = ""
    filenames: list[str] = []
    s103a = find_subchunks(raw, 0x103A)
    if s103a:
        blob = s103a[0]
        if re.search(rb"Audio Files", blob):
            directory = "Audio Files"
        for m in re.finditer(rb"[\x20-\x7e]{1,200}\.(?:wav|WAV|aif|aiff|AIF|AIFF|mxf|MXF|mp3|m4a|flac|sd2)", blob):
            fn = m.group().decode("utf-8", errors="replace").strip()
            if fn not in filenames:
                filenames.append(fn)
    files: list[PtxAudioFile] = []
    for i in range(max(len(filenames), len(s1003))):
        fname = filenames[i] if i < len(filenames) else f"file_{i}"
        file_num, sr, chs, bits, length, fmt_code = (
            _decode_audio_1003(s1003[i], i + 1) if i < len(s1003) else (i + 1, 0, 0, 0, 0, 0)
        )
        dur = round(length / sr, 4) if (sr > 0 and length > 0) else None
        files.append(PtxAudioFile(i, file_num, fname, directory, sr, chs, bits, length, dur, fmt_code))
    return files


def _decode_audio_1003(b3: bytes, default_num: int) -> tuple[int, int, int, int, int, int]:
    file_num = struct.unpack_from("<I", b3, 9)[0] if len(b3) >= 13 else default_num
    s1001 = find_subchunks(b3, 0x1001)
    if s1001 and len(s1001[0]) >= 24:
        b1 = s1001[0]
        return file_num, struct.unpack_from("<I", b1, 9)[0], b1[13], b1[14], struct.unpack_from("<Q", b1, 15)[0], b1[23]
    return file_num, 0, 0, 0, 0, 0


def parse_clip_gain_curves_2637(raw: bytes, sample_rate: int) -> list[PtxClipGainCurve]:
    """Decode all single-point and multi-point clip-gain curves from top-level ``0x2637``."""
    if len(raw) < 13:
        return []
    cnt = struct.unpack_from("<I", raw, 9)[0]
    pos = 13
    curves: list[PtxClipGainCurve] = []
    for ci in range(cnt):
        if pos + 8 > len(raw):
            break
        flags = raw[pos]
        parsed = parse_control_curve(raw, pos)
        if parsed is None:
            break
        timebase, raw_pts, next_pos = parsed
        bps: list[PtxClipGainBreakpoint] = []
        for pt in raw_pts:
            gdb = round(float(pt.value), 4) if math.isfinite(pt.value) else 0.0
            secs = round(pt.position / sample_rate, 4) if (sample_rate > 0 and timebase == "Samples") else None
            bps.append(PtxClipGainBreakpoint(sample=pt.position, gain_db=gdb, seconds=secs))
        curves.append(PtxClipGainCurve(index=ci, flags=flags, timebase=timebase, breakpoints=bps))
        pos = next_pos
    return curves


def _read_uint_le(buf: bytes, pos: int, width: int) -> tuple[int, int]:
    if width <= 0 or width > 8 or pos + width > len(buf):
        return 0, pos
    return int.from_bytes(buf[pos : pos + width], "little"), pos + width


def _parse_region_core_2628(
    b28: bytes,
) -> tuple[str, int, int, int, int, int, list[PtxWarpMarker]] | None:
    name, after_name = read_lp_str(b28, 9, max_len=256)
    if not name:
        return None
    offset_smp = sync_smp = len_smp = orig_ts = user_ts = 0
    if after_name + 5 <= len(b28):
        ob = (b28[after_name + 1] >> 4) & 0x0F
        lb = (b28[after_name + 2] >> 4) & 0x0F
        sb = b28[after_name + 2] & 0x0F
        otb = (b28[after_name + 3] >> 4) & 0x0F
        utb = b28[after_name + 3] & 0x0F
        pos = after_name + 5
        offset_smp, pos = _read_uint_le(b28, pos, ob)
        sync_smp, pos = _read_uint_le(b28, pos, sb)
        len_smp, pos = _read_uint_le(b28, pos, lb)
        orig_ts, pos = _read_uint_le(b28, pos, otb)
        user_ts, _ = _read_uint_le(b28, pos, utb)
    warp_markers: list[PtxWarpMarker] = []
    for s36 in find_subchunks(b28, 0x2636):
        k = s36.find(b"\x01\x46\x01", 9)
        if k != -1:
            parsed = parse_control_curve(s36, k)
            if parsed is not None:
                warp_markers = [
                    PtxWarpMarker(pt.position, int(pt.value))
                    for pt in parsed[1]
                    if math.isfinite(pt.value)
                ]
                warp_markers.sort(key=lambda wm: wm.source_sample)
                break
    return name, offset_smp, sync_smp, len_smp, orig_ts, user_ts, warp_markers


def _eval_warp_at(markers: list[PtxWarpMarker], x: int) -> float:
    if x <= markers[0].source_sample:
        a, b = markers[0], markers[1]
    elif x >= markers[-1].source_sample:
        a, b = markers[-2], markers[-1]
    else:
        a, b = markers[0], markers[1]
        for idx in range(len(markers) - 1):
            if markers[idx].source_sample <= x <= markers[idx + 1].source_sample:
                a, b = markers[idx], markers[idx + 1]
                break
    if b.source_sample == a.source_sample:
        return float(a.timeline_sample)
    return a.timeline_sample + (x - a.source_sample) * (b.timeline_sample - a.timeline_sample) / float(
        b.source_sample - a.source_sample
    )


def parse_clips_262a(
    raw: bytes,
    audio_files: list[PtxAudioFile],
    curves: list[PtxClipGainCurve],
) -> list[PtxClip]:
    """Decode audio clips from ``0x262a`` -> ``0x2629`` -> ``0x2628``."""
    clips: list[PtxClip] = []
    for ci, sub in enumerate(find_subchunks(raw, 0x2629)):
        clip = _decode_clip_2629(ci, sub, audio_files, curves)
        if clip is not None:
            clips.append(clip)
    return clips


def _decode_clip_2629(
    index: int,
    sub: bytes,
    audio_files: list[PtxAudioFile],
    curves: list[PtxClipGainCurve],
) -> PtxClip | None:
    s28 = find_subchunks(sub, 0x2628)
    if not s28:
        return None
    b28 = s28[0]
    core = _parse_region_core_2628(b28)
    if core is None:
        return None
    cname, start_smp, sync_smp, len_smp, orig_ts, user_ts, warp_markers = core
    b28_ver = struct.unpack_from("<H", b28, 1)[0] if len(b28) >= 3 else 0
    _nm, after_name = read_lp_str(b28, 9, max_len=256)
    clip_id = format_rfc_uuid(b28[-16:]) if (b28_ver >= 5 and len(b28) >= 16) else ""
    cg_idx = _extract_clip_gain_idx(b28, b28_ver, after_name, len(curves))
    b28_pos = sub.find(b28, 9)
    b28_end = (b28_pos + len(b28)) if b28_pos != -1 else len(sub)
    file_idx, file_name, ch_idx, len_smp = _resolve_clip_file(sub, b28_end, audio_files, len_smp)
    gain_lin, gain_db, cg_bps = _resolve_clip_gain(cg_idx, curves)
    tl_len, rate = _compute_warped_length(start_smp, len_smp, warp_markers)
    cg_size, cg_b64, ara_name, ara_uuid = _parse_clip_extras(sub)
    return PtxClip(
        index, cname, clip_id, file_idx, file_name, ch_idx, start_smp, len_smp, gain_lin, gain_db, cg_idx,
        sync_point_samples=sync_smp,
        original_timestamp_samples=orig_ts,
        user_timestamp_samples=user_ts,
        has_clip_gain_automation=bool(cg_bps) or (cg_size > 37),
        clip_gain_breakpoints=cg_bps,
        warp_markers=warp_markers,
        timeline_length_samples=tl_len,
        playback_rate=rate,
        ara_plugin_name=ara_name,
        ara_modification_uuid=ara_uuid,
        clip_gain_chunk_size=cg_size,
        clip_gain_raw_b64=cg_b64,
    )


def _extract_clip_gain_idx(b28: bytes, b28_ver: int, after_name: int, curve_count: int) -> int:
    if b28_ver >= 5 and len(b28) >= 31 and after_name <= len(b28) - 22:
        return int(struct.unpack_from("<i", b28, len(b28) - 22)[0])
    if b28_ver < 5 and len(b28) >= 13 and after_name <= len(b28) - 4:
        cand = int(struct.unpack_from("<i", b28, len(b28) - 4)[0])
        if 0 <= cand < curve_count:
            return cand
    return -1


def _resolve_clip_file(
    sub: bytes, b28_end: int, audio_files: list[PtxAudioFile], len_smp: int
) -> tuple[int | None, str, int, int]:
    file_idx: int | None = None
    file_name = ""
    ch_idx = 0
    if b28_end + 6 <= len(sub):
        fi, chi = struct.unpack_from("<IH", sub, b28_end)
        if fi < len(audio_files):
            file_idx, file_name = fi, audio_files[fi].filename
            len_smp = len_smp or audio_files[fi].length_samples
        ch_idx = chi
    return file_idx, file_name, ch_idx, len_smp


def _resolve_clip_gain(
    cg_idx: int, curves: list[PtxClipGainCurve]
) -> tuple[float, float, list[PtxClipGainBreakpoint]]:
    if 0 <= cg_idx < len(curves):
        bps = list(curves[cg_idx].breakpoints)
        if bps:
            gdb = round(bps[0].gain_db, 2)
            glin = 0.0 if gdb <= -144.0 else round(10.0 ** (gdb / 20.0), 6)
            return glin, gdb, bps
    return 1.0, 0.0, []


def _compute_warped_length(start_smp: int, len_smp: int, warp: list[PtxWarpMarker]) -> tuple[int, float]:
    if len(warp) >= 2 and len_smp > 0:
        a = _eval_warp_at(warp, start_smp)
        b = _eval_warp_at(warp, start_smp + len_smp)
        if b > a:
            return round(b - a), len_smp / (b - a)
    return len_smp, 1.0


def _parse_clip_extras(sub: bytes) -> tuple[int, str, str, str]:
    cg_list = find_subchunks(sub, 0x4403)
    cg_size = len(cg_list[0]) if cg_list else 0
    cg_b64 = base64.b64encode(cg_list[0]).decode("ascii") if (cg_list and cg_size > 37) else ""
    ara_name = ""
    ara_uuid = ""
    s4807 = find_subchunks(sub, 0x4807)
    if s4807:
        ara_cp = parse_ara_plugin_4807(s4807[0])
        if ara_cp is not None:
            ara_name = ara_cp.name
        if cg_list:
            p44 = sub.find(cg_list[0], 9)
            after = (p44 + len(cg_list[0])) if p44 != -1 else len(sub)
            if after + 17 <= len(sub) and sub[after] == 1:
                ara_uuid = format_rfc_uuid(sub[after + 1 : after + 17])
    return cg_size, cg_b64, ara_name, ara_uuid


def parse_fades_2630(raw: bytes) -> list[PtxFade]:
    """Decode fade-ins, fade-outs, and crossfades from ``0x2630`` -> ``0x262f``."""
    fades: list[PtxFade] = []
    for idx, sub in enumerate(find_subchunks(raw, 0x262F)):
        body = sub[9:]
        if len(body) < 9:
            continue
        n1, n2 = (body[5] >> 4) & 0x0F, body[5] & 0x0F
        if n1 > 8 or n2 > 8 or 8 + n1 + n2 >= len(body):
            continue
        v1, p = _read_uint_le(body, 8, n1)
        v2, p = _read_uint_le(body, p, n2)
        tcode = body[p]
        ftype = FADE_TYPE_MAP.get(tcode, "unknown")
        pre = 0 if tcode == 3 else v1
        total = v1 if tcode == 3 else (v2 if v2 != 0 else v1)
        fade = PtxFade(index=idx, fade_type=ftype, pre_samples=pre, total_samples=total)
        _decode_fade_shape(body, p + 1, tcode, fade)
        fades.append(fade)
    return fades


def _decode_fade_shape(body: bytes, r0: int, tcode: int, fade: PtxFade) -> None:
    if tcode == 1 and r0 < len(body):
        link = body[r0]
        fade.crossfade_link_code = link
        fade.crossfade_link_name = CROSSFADE_LINK_MAP.get(link, f"link_{link}")
        if link == 3 and r0 + 21 <= len(body):
            fade.out_shape_code = body[r0 + 11]
            fade.out_curve_amount = struct.unpack_from("<h", body, r0 + 12)[0]
            fade.in_shape_code = body[r0 + 18]
            fade.in_curve_amount = struct.unpack_from("<h", body, r0 + 19)[0]
    elif tcode == 3 and r0 + 8 <= len(body):
        fade.shape_code = body[r0 + 5]
        fade.shape_name = FADE_SHAPE_MAP.get(fade.shape_code, f"shape_{fade.shape_code}")
        fade.curve_amount = struct.unpack_from("<h", body, r0 + 6)[0]
    elif tcode == 2 and r0 + 14 <= len(body):
        fade.shape_code = body[r0 + 11]
        fade.shape_name = FADE_SHAPE_MAP.get(fade.shape_code, f"shape_{fade.shape_code}")
        fade.curve_amount = struct.unpack_from("<h", body, r0 + 12)[0]


def parse_compound_clips(c262c_raw: bytes | None, c2428_raw: bytes | None) -> list[PtxCompoundClip]:
    """Decode compound clips / clip groups from ``0x262c`` and ``0x2428``."""
    compounds: list[PtxCompoundClip] = []
    if c262c_raw:
        for idx, sub in enumerate(find_subchunks(c262c_raw, 0x262B)):
            s28 = find_subchunks(sub, 0x2628)
            core = _parse_region_core_2628(s28[0]) if s28 else None
            if core is not None:
                compounds.append(PtxCompoundClip(idx, core[0], core[1], core[3]))
    if c2428_raw:
        member_lists: list[list[PtxCompoundClipMember]] = []
        for s54 in find_subchunks(c2428_raw, 0x1054):
            for s52 in find_subchunks(s54, 0x1052):
                members: list[PtxCompoundClipMember] = []
                for s4f in find_subchunks(s52, 0x104F):
                    if len(s4f) >= 25:
                        cidx = struct.unpack_from("<I", s4f, 11)[0]
                        pos = struct.unpack_from("<Q", s4f, 16)[0] & TICK_MASK_56
                        flags = s4f[24]
                        members.append(
                            PtxCompoundClipMember(
                                cidx, pos, flags,
                                flags != 0 and (flags & 2) == 0,
                                flags != 0 and (flags & 1) == 0,
                            )
                        )
                member_lists.append(members)
        for idx, comp in enumerate(compounds):
            if idx < len(member_lists):
                comp.members = member_lists[idx]
    return compounds


def parse_timeline_lanes_1054(
    raw: bytes,
    clips: list[PtxClip],
    fades: list[PtxFade],
    compound_clips: list[PtxCompoundClip],
    sample_rate: int,
) -> list[tuple[str, list[PtxClipPlacement], list[PtxFadePlacement]]]:
    """Decode ``0x1054`` -> ``0x1052`` channel lanes into clip and fade placements."""
    clips_by_idx = {c.index: c for c in clips}
    fades_by_idx = {f.index: f for f in fades}
    comps_by_idx = {cp.index: cp for cp in compound_clips}
    lanes: list[tuple[str, list[PtxClipPlacement], list[PtxFadePlacement]]] = []
    for s52 in find_subchunks(raw, 0x1052):
        lname, _ = read_lp_str(s52, 9, max_len=256)
        lane_clips: list[PtxClipPlacement] = []
        lane_fades: list[PtxFadePlacement] = []
        s50s = find_subchunks(s52, 0x1050)
        if s50s:
            for s50 in s50s:
                search_pos = 9
                for s4f in find_subchunks(s50, 0x104F):
                    fpos = s50.find(s4f, search_pos)
                    fend = (fpos + len(s4f)) if fpos != -1 else len(s50)
                    search_pos = fend
                    is_fade = fend < len(s50) and s50[fend] == 1
                    _decode_lane_event(s4f, is_fade, clips_by_idx, fades_by_idx, comps_by_idx, sample_rate, lane_clips, lane_fades)
        else:
            for s4f in find_subchunks(s52, 0x104F):
                _decode_lane_event(s4f, False, clips_by_idx, fades_by_idx, comps_by_idx, sample_rate, lane_clips, lane_fades)
        lanes.append((lname, lane_clips, lane_fades))
    return lanes


def _decode_lane_event(
    s4f: bytes,
    is_fade: bool,
    clips_by_idx: dict[int, PtxClip],
    fades_by_idx: dict[int, PtxFade],
    comps_by_idx: dict[int, PtxCompoundClip],
    sample_rate: int,
    lane_clips: list[PtxClipPlacement],
    lane_fades: list[PtxFadePlacement],
) -> None:
    if len(s4f) < 27:
        return
    idx = struct.unpack_from("<I", s4f, 11)[0]
    start_smp = struct.unpack_from("<Q", s4f, 16)[0] & TICK_MASK_56
    flags = s4f[24]
    color_idx = struct.unpack_from("<H", s4f, 25)[0]
    if is_fade:
        fade = fades_by_idx.get(idx)
        lane_fades.append(
            PtxFadePlacement(
                channel=0, fade_index=idx, start_sample=start_smp,
                fade_type=fade.fade_type if fade else "unknown",
                pre_samples=fade.pre_samples if fade else 0,
                total_samples=fade.total_samples if fade else 0,
                flags=flags,
            )
        )
        return
    is_comp = len(s4f) > 27 and s4f[27] == 1
    lp_chunks = find_subchunks(s4f, 0x242B)
    loop_count = lp_chunks[0][36] if (lp_chunks and len(lp_chunks[0]) > 36) else None
    if is_comp and idx in comps_by_idx:
        comp = comps_by_idx[idx]
        cname, afname, clen, clip_obj = comp.name, "", comp.length_samples, None
    else:
        clip_obj = clips_by_idx.get(idx)
        cname = clip_obj.name if clip_obj else f"clip_{idx}"
        afname = clip_obj.audio_file_name if clip_obj else ""
        clen = (clip_obj.timeline_length_samples or clip_obj.length_samples) if clip_obj else 0
    lane_clips.append(
        PtxClipPlacement(
            0, idx, cname, afname, start_smp, clen, start_smp + clen, color_idx,
            start_seconds=round(start_smp / sample_rate, 4) if sample_rate > 0 else None,
            duration_seconds=round(clen / sample_rate, 4) if (sample_rate > 0 and clen > 0) else None,
            flags=flags,
            is_muted=(flags != 0 and (flags & 2) == 0),
            is_covered=(flags != 0 and (flags & 1) == 0),
            is_compound=is_comp,
            loop_count=loop_count,
            has_clip_gain_automation=clip_obj.has_clip_gain_automation if clip_obj else False,
            clip_gain_breakpoints=list(clip_obj.clip_gain_breakpoints) if clip_obj else [],
            ara_plugin_name=clip_obj.ara_plugin_name if clip_obj else "",
            ara_modification_uuid=clip_obj.ara_modification_uuid if clip_obj else "",
        )
    )


def parse_playlists_1015(
    raw: bytes,
    lanes: list[tuple[str, list[PtxClipPlacement], list[PtxFadePlacement]]],
    tracks: list[PtxTrack],
) -> list[PtxPlaylist]:
    """Decode ``0x1015`` playlists, attach main clips/fades to Audio tracks, and link alternate playlists."""
    tracks_by_name = {t.name: t for t in tracks if t.track_type == "Audio"}
    playlists: list[PtxPlaylist] = []
    for pi, s14 in enumerate(find_subchunks(raw, 0x1014)):
        pl = _decode_playlist_1014(pi, s14, lanes)
        if pl is None:
            continue
        trk = tracks_by_name.get(pl.name)
        if trk is not None:
            pl.track_name = trk.name
            if not trk.clips:
                trk.clips = list(pl.placements)
            if not trk.fade_placements:
                trk.fade_placements = list(pl.fade_placements)
        else:
            m = re.match(r"^(.*)\.\d+$", pl.name)
            if m and m.group(1) in tracks_by_name:
                parent_trk = tracks_by_name[m.group(1)]
                pl.is_alternate = True
                pl.track_name = parent_trk.name
                parent_trk.alternate_playlists.append(pl)
        playlists.append(pl)
    return playlists


def _decode_playlist_1014(
    index: int,
    s14: bytes,
    lanes: list[tuple[str, list[PtxClipPlacement], list[PtxFadePlacement]]],
) -> PtxPlaylist | None:
    pname, off = read_lp_str(s14, 9, max_len=256)
    if not pname or off + 5 > len(s14):
        return None
    ch_count = struct.unpack_from("<I", s14, off + 1)[0]
    lp = off + 5
    lane_idxs = list(struct.unpack_from(f"<{ch_count}H", s14, lp)) if (0 < ch_count <= 64 and lp + 2 * ch_count <= len(s14)) else []
    uoff = s14.find(UUID_PREFIX, lp)
    pl_uuid = format_ptsl_uuid(s14[uoff : uoff + 16]) if (uoff != -1 and uoff + 16 <= len(s14)) else ""
    placements: list[PtxClipPlacement] = []
    fade_placements: list[PtxFadePlacement] = []
    for ch_i, lidx in enumerate(lane_idxs):
        if 0 <= lidx < len(lanes):
            for plc in lanes[lidx][1]:
                placements.append(
                    PtxClipPlacement(
                        ch_i, plc.clip_index, plc.clip_name, plc.audio_file_name,
                        plc.start_sample, plc.length_samples, plc.end_sample, plc.color_index,
                        plc.start_seconds, plc.duration_seconds, plc.flags,
                        plc.is_muted, plc.is_covered, plc.is_compound, plc.loop_count,
                        plc.has_clip_gain_automation, list(plc.clip_gain_breakpoints),
                        plc.ara_plugin_name, plc.ara_modification_uuid,
                    )
                )
            for fpl in lanes[lidx][2]:
                fade_placements.append(
                    PtxFadePlacement(
                        ch_i, fpl.fade_index, fpl.start_sample, fpl.fade_type,
                        fpl.pre_samples, fpl.total_samples, fpl.flags,
                    )
                )
    return PtxPlaylist(
        index, pname, pl_uuid, ch_count if ch_count <= 64 else len(lane_idxs),
        lane_idxs, placements, fade_placements=fade_placements,
    )


def parse_midi_data(
    c2000_raw: bytes | None,
    c2634_raw: bytes | None,
    c1058_raw: bytes | None,
    tempo_events: list[PtxTempoEvent],
    default_bpm: float = 120.0,
) -> tuple[list[PtxMidiChunk], list[PtxMidiClip], dict[str, list[PtxMidiPlacement]]]:
    """Decode MIDI event chunks (``0x2000``), MIDI clips (``0x2634``), and placements (``0x1058``)."""
    chunks = _parse_midi_chunks_2000(c2000_raw) if c2000_raw else []
    clips = _parse_midi_clips_2634(c2634_raw) if c2634_raw else []
    playlists = _parse_midi_playlists_1058(c1058_raw, clips, tempo_events, default_bpm) if c1058_raw else {}
    return chunks, clips, playlists


def _parse_midi_chunks_2000(raw: bytes) -> list[PtxMidiChunk]:
    chunks: list[PtxMidiChunk] = []
    pos = 13
    while pos + 12 <= len(raw) and raw[pos : pos + 6] == b"MdChun":
        msz = struct.unpack_from("<I", raw, pos + 8)[0]
        mend = pos + 12 + msz
        if mend > len(raw):
            break
        notes = _parse_mdnlb_notes(raw, pos + 12, mend)
        lanes = _parse_mdcds_lanes(raw, pos + 12, mend)
        chunks.append(PtxMidiChunk(index=len(chunks), notes=notes, control_lanes=lanes))
        pos = mend
    return chunks


def _parse_mdnlb_notes(raw: bytes, start: int, end: int) -> list[PtxMidiNote]:
    if start + 15 > end or raw[start : start + 5] != b"MdNLB":
        return []
    n_notes = struct.unpack_from("<I", raw, start + 11)[0]
    r = start + 15
    notes: list[PtxMidiNote] = []
    for _ in range(min(n_notes, (end - r) // 35)):
        tick = decode_ticks_relative(struct.unpack_from("<Q", raw, r)[0] & TICK_MASK_56)
        pitch = raw[r + 8] & 0x7F
        dur = struct.unpack_from("<Q", raw, r + 9)[0] & TICK_MASK_56
        vel = raw[r + 17] & 0x7F
        notes.append(PtxMidiNote(tick=tick, pitch=pitch, duration_ticks=dur, velocity=vel))
        r += 35
    return notes


def _parse_mdcds_lanes(raw: bytes, start: int, end: int) -> list[PtxMidiControlLane]:
    lanes: list[PtxMidiControlLane] = []
    k = raw.find(b"MdCDS", start, end)
    while k != -1 and k + 11 <= end:
        csz = struct.unpack_from("<I", raw, k + 7)[0]
        cend = min(end, k + 11 + csz)
        ss = raw.find(b"Mdss", k, cend)
        cd = raw.find(b"MdCD\x01", k + 5, cend)
        if ss != -1 and cd != -1 and ss + 15 <= cend and cd + 14 <= cend:
            ctype = raw[ss + 10]
            cnum = struct.unpack_from("<I", raw, ss + 11)[0]
            nev = struct.unpack_from("<I", raw, cd + 10)[0]
            r = cd + 14
            evs: list[PtxMidiControlEvent] = []
            for _ in range(min(nev, (cend - r) // 12)):
                tick = decode_ticks_relative(struct.unpack_from("<Q", raw, r)[0] & TICK_MASK_56)
                val = struct.unpack_from("<I", raw, r + 8)[0]
                if ctype == 1 and val >= 0x8000_0000:
                    val -= 0x1_0000_0000
                evs.append(PtxMidiControlEvent(tick=tick, value=val))
                r += 12
            lanes.append(
                PtxMidiControlLane(ctype, MIDI_CONTROL_LANE_TYPE_MAP.get(ctype, f"type_{ctype}"), cnum, evs)
            )
        k = raw.find(b"MdCDS", max(k + 5, cend), end)
    return lanes


def _parse_midi_clips_2634(raw: bytes) -> list[PtxMidiClip]:
    clips: list[PtxMidiClip] = []
    for idx, sub in enumerate(find_subchunks(raw, 0x2633)):
        s28 = find_subchunks(sub, 0x2628)
        if not s28:
            continue
        core = _parse_region_core_2628(s28[0])
        if core is None:
            continue
        p28 = sub.find(s28[0], 9)
        end28 = (p28 + len(s28[0])) if p28 != -1 else len(sub)
        chunk_idx = struct.unpack_from("<I", sub, end28)[0] if end28 + 4 <= len(sub) else 0
        clips.append(PtxMidiClip(idx, core[0], decode_ticks_relative(core[1]), core[3], chunk_idx))
    return clips


def _parse_midi_playlists_1058(
    raw: bytes,
    clips: list[PtxMidiClip],
    tempo_events: list[PtxTempoEvent],
    default_bpm: float,
) -> dict[str, list[PtxMidiPlacement]]:
    clips_by_idx = {c.index: c for c in clips}
    playlists: dict[str, list[PtxMidiPlacement]] = {}
    for s57 in find_subchunks(raw, 0x1057):
        tname, _ = read_lp_str(s57, 9, max_len=256)
        if not tname:
            continue
        dest = playlists.setdefault(tname, [])
        for s4f in find_subchunks(s57, 0x104F):
            if len(s4f) < 25:
                continue
            cidx = struct.unpack_from("<I", s4f, 11)[0]
            start_tick = decode_ticks_relative(struct.unpack_from("<Q", s4f, 16)[0] & TICK_MASK_56)
            flags = s4f[24]
            mc = clips_by_idx.get(cidx)
            clen = mc.length_ticks if mc else 0
            t0 = ticks_to_seconds(tempo_events, start_tick, default_bpm)
            t1 = ticks_to_seconds(tempo_events, start_tick + clen, default_bpm) if clen > 0 else t0
            dest.append(
                PtxMidiPlacement(
                    clip_index=cidx,
                    clip_name=mc.name if mc else f"midi_clip_{cidx}",
                    start_tick=start_tick,
                    length_ticks=clen,
                    flags=flags,
                    is_muted=(flags != 0 and (flags & 2) == 0),
                    start_seconds=round(t0, 4),
                    duration_seconds=round(t1 - t0, 4) if clen > 0 else None,
                )
            )
    return playlists


def parse_video_data(
    c1006_raw: bytes | None,
    c262e_raw: bytes | None,
    c1055_raw: bytes | None,
    sample_rate: int,
) -> tuple[list[str], list[PtxVideoClip], dict[str, list[PtxVideoPlacement]]]:
    """Decode video filenames (``0x1006``), video clips (``0x262e``), and placements (``0x1055``)."""
    video_files: list[str] = []
    if c1006_raw:
        p = 9
        while p + 4 < len(c1006_raw):
            slen = struct.unpack_from("<I", c1006_raw, p)[0]
            if 4 < slen < 256 and p + 4 + slen <= len(c1006_raw):
                cand = c1006_raw[p + 4 : p + 4 + slen]
                if not any(b < 32 for b in cand):
                    text = cand.decode("utf-8", errors="replace")
                    ext = text.rsplit(".", 1)[-1].lower() if "." in text else ""
                    if ext in _VIDEO_EXTS and text not in video_files:
                        video_files.append(text)
                        p += 4 + slen
                        continue
            p += 1
    video_clips: list[PtxVideoClip] = []
    if c262e_raw:
        for idx, sub in enumerate(find_subchunks(c262e_raw, 0x262D)):
            s28 = find_subchunks(sub, 0x2628)
            core = _parse_region_core_2628(s28[0]) if s28 else None
            if core is not None:
                video_clips.append(PtxVideoClip(idx, core[0], core[1], core[3]))
    clips_by_idx = {vc.index: vc for vc in video_clips}
    video_playlists: dict[str, list[PtxVideoPlacement]] = {}
    if c1055_raw:
        for s53 in find_subchunks(c1055_raw, 0x1053):
            tname, _ = read_lp_str(s53, 9, max_len=256)
            if not tname:
                continue
            dest = video_playlists.setdefault(tname, [])
            for s4f in find_subchunks(s53, 0x104F):
                if len(s4f) < 24:
                    continue
                cidx = struct.unpack_from("<I", s4f, 11)[0]
                start_smp = struct.unpack_from("<Q", s4f, 16)[0] & TICK_MASK_56
                vc = clips_by_idx.get(cidx)
                dest.append(
                    PtxVideoPlacement(
                        cidx, vc.name if vc else f"video_clip_{cidx}", start_smp,
                        round(start_smp / sample_rate, 4) if sample_rate > 0 else None,
                    )
                )
    return video_files, video_clips, video_playlists
