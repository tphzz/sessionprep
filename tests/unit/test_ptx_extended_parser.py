"""Unit tests for the extended general-purpose Pro Tools ``.ptx`` (v10+) model and parser."""

from __future__ import annotations

import json
import struct

import pytest

from sessionpreplib.ptx import (
    TICKS_PER_QUARTER,
    ZERO_TICKS,
    PtxSession,
    decrypt_ptx,
    encrypt_ptx,
    find_subchunks,
    format_timecode,
    lookup_pt_color_rgb,
    parse_chunk_tree,
    parse_control_curve,
)
from sessionpreplib.ptx.models import PTX_MAGIC, UUID_PREFIX


def _zchunk(tag: int, body: bytes, version: int = 1) -> bytes:
    return struct.pack("<BHIH", 0x5A, version, len(body) + 2, tag) + body


def _lp(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _uuid(seed: int) -> bytes:
    return UUID_PREFIX + struct.pack(">II", 0x2000_0000 + seed, 0x3000_0000 + seed)


def _control_curve(
    tb_code: int,
    vsize: int,
    points: list[tuple[int, float | int]],
    *,
    flags: int = 1,
    signed_int32: bool = False,
) -> bytes:
    pos0, val0 = points[0]
    raw_pos0 = (pos0 + ZERO_TICKS) if tb_code == 1 else pos0

    def _pack_val(v: float | int) -> bytes:
        if vsize == 2:
            return struct.pack("<h", int(v))
        if vsize == 4:
            return struct.pack("<i" if signed_int32 else "<f", int(v) if signed_int32 else float(v))
        return struct.pack("<Q", int(v))

    payload = bytearray()
    payload.extend(struct.pack("<HIHHQ", tb_code, len(points), vsize, len(points) - 1, raw_pos0))
    payload.extend(_pack_val(val0))
    for pos, val in points[1:]:
        payload.extend(struct.pack("<I", pos - pos0))
        payload.extend(_pack_val(val))
    return struct.pack("<BBHI", flags, 0x46, 1, len(payload)) + bytes(payload)


def _wrap_ptx(top_chunks: list[bytes], *, xor_val: int = 0x90, delta: int = 0x50) -> bytes:
    out = bytearray(PTX_MAGIC + bytes([0x00, 0x05, xor_val]))
    out.extend(struct.pack("<BHII", 0x5A, 1, 4, 0))
    for ch in top_chunks:
        out.extend(ch)
    return encrypt_ptx(bytes(out), delta, k0=0)


def _v10_track(name: str, type_code: int, uid: bytes, *, stereo: bool = False) -> bytes:
    body = (
        bytes([type_code, 0]) + _lp(name) + bytes([1 if stereo else 0]) + uid
        + b"\x00\x00\x00\x00\x00\x00\x01\x01\x01\x00\x00\xfe\xff"
    )
    return _zchunk(0x251A, body, version=10)


def _v6_track(name: str, type_code: int, uid: bytes, *, disp_idx: int = 1) -> bytes:
    body = (
        bytes([type_code, 0]) + _lp(name) + bytes([0, 0]) + uid
        + struct.pack("<I", disp_idx) + b"\x00\x00" + uid + b"\x00\x00"
        + _zchunk(0x4420, b"\x00\x00")
    )
    return _zchunk(0x251A, body, version=6)


def test_codec_algebraic_decrypt_find_subchunks_bounds_and_legacy_rejection():
    dense = bytes((i * 37 + 13) & 0xFF for i in range(9000))
    raw = _wrap_ptx([_zchunk(0x7FFE, dense)], xor_val=0x90, delta=0x50)
    dec, delta = decrypt_ptx(raw)
    assert delta == 0x50 and dec[0x1F] == 0x5A
    assert parse_chunk_tree(dec)[0].tag == 0x7FFE

    # find_subchunks must never match a chunk whose header starts before `start`
    buf = b"\x00\x00" + _zchunk(0x4807, b"\x00" * 16) + _zchunk(0x4807, b"\x11" * 16)
    assert len(find_subchunks(buf, 0x4807, start=0)) == 2
    assert len(find_subchunks(buf, 0x4807, start=5)) == 1

    with pytest.raises(ValueError, match="Big-endian"):
        decrypt_ptx(PTX_MAGIC + b"\x01\x05\x90\x5A" + b"\x00" * 32)
    with pytest.raises(ValueError, match="Legacy"):
        decrypt_ptx(PTX_MAGIC + b"\x00\x01\x00\x5A" + b"\x00" * 32)


def test_control_curve_timecode_and_color_helpers():
    raw_i16 = _control_curve(1, 2, [(0, -60), (TICKS_PER_QUARTER, 0)])
    parsed = parse_control_curve(raw_i16, 0)
    assert parsed is not None
    assert parsed[0] == "Ticks" and parsed[2] == len(raw_i16)
    assert [(p.position, p.value) for p in parsed[1]] == [(0, -60.0), (TICKS_PER_QUARTER, 0.0)]

    # Bypassed/flags=0 control curve is accepted; negative offset & truncated input return None
    raw_zero_flags = _control_curve(0, 4, [(0, -2.5)], flags=0)
    parsed_zf = parse_control_curve(raw_zero_flags, 0)
    assert parsed_zf is not None and parsed_zf[1][0].value == pytest.approx(-2.5)
    assert parse_control_curve(raw_zero_flags, -1) is None
    assert parse_control_curve(b"\x01\x46\x01\x00\xff\x00\x00\x00", 0) is None

    assert format_timecode(89808, 2) == "00:59:52:08"
    assert format_timecode(1798, 5) == "00:00:59;28"
    assert format_timecode(1800, 5) == "00:01:00;02"
    assert format_timecode(17982, 5) == "00:10:00;00"
    assert format_timecode(100, 999) == ""
    assert lookup_pt_color_rgb(32) == (188, 0, 13) and lookup_pt_color_rgb(0xFFFE) is None


def test_parse_session_start_tempo_meter_and_markers():
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c204d = _zchunk(0x204D, struct.pack("<I5sI", 2, b"\x00" * 5, 89808), version=5)

    def _tempo_const(tick_rel: int, bpm: float) -> bytes:
        rec = bytearray(48)
        rec[0:7] = b"Const\x01\x00"
        struct.pack_into("<I", rec, 7, 37)
        struct.pack_into("<Q", rec, 30, ZERO_TICKS + tick_rel)
        struct.pack_into("<d", rec, 40, bpm)
        return bytes(rec)

    b2028 = b"Tempo\x02\x00\x00\x00\x00\x00" + struct.pack("<I", 3) + _tempo_const(0, 120.0) + _tempo_const(TICKS_PER_QUARTER * 4, 60.0)
    c2028 = _zchunk(0x2028, b2028, version=3)
    m0 = struct.pack("<QIII", ZERO_TICKS, 1, 4, 4) + b"\x00" * 16
    m1 = struct.pack("<QIII", ZERO_TICKS + TICKS_PER_QUARTER * 4, 2, 7, 8) + b"\x00" * 16
    c2029 = _zchunk(0x2029, b"Meter\x02\x00\x00\x00\x00\x00" + struct.pack("<I", 2) + m0 + m1, version=1)
    m_chorus = _zchunk(0x2077, struct.pack("<HBBH", 2, 3, 1, 0) + _lp("Chorus") + struct.pack("<QQ", 96000, 144000) + _zchunk(0x2506, b"\x00\x00") + b"\x00\x00\x00\x00" + _lp("Big hook"))
    m_intro = _zchunk(0x2077, struct.pack("<HBBH", 1, 1, 0, 0) + _lp("Intro") + struct.pack("<QQ", 0, 48000))
    c2030 = _zchunk(0x2030, struct.pack("<I", 2) + m_chorus + m_intro, version=5)

    sess = PtxSession.from_bytes(_wrap_ptx([c1028, c204d, c2028, c2029, c2030]))
    assert sess.info.fps_code == 2 and sess.info.start_timecode == "00:59:52:08"
    assert sess.has_unparsed_tempo_ramps is True and any("tempo ramps" in w for w in sess.warnings)
    assert sess.ticks_to_seconds(TICKS_PER_QUARTER * 6) == pytest.approx(4.0, abs=1e-5)
    assert [(m.numerator, m.denominator) for m in sess.meter_events] == [(4, 4), (7, 8)]
    assert [m.name for m in sess.markers] == ["Intro", "Chorus"]
    assert sess.markers[1].comments == "Big hook" and sess.markers[1].start_seconds == 2.0


def test_parse_audio_file_trees_windows_macos_and_cycle_guard():
    # Windows tree + cyclic parent pointer guard
    win_entries = [
        (0, 0xFFFFFFFF, "C:", b"\x00\x00\x00\x00"),
        (1, 0, "Song", b"\x00\x00\x00\x00"),
        (1, 1, "Audio Files", b"\x00\x00\x00\x00"),
        (2, 2, "Guit\u00e4r.mxf", b"fFXM"),
        (1, 5, "CycleA", b"\x00\x00\x00\x00"),
        (2, 4, "Loop.wav", b"EVAW"),
    ]
    b103a = bytearray(struct.pack("<I", len(win_entries)))
    for kind, parent, name, tag4 in win_entries:
        b103a.extend(struct.pack("<BI", kind, parent) + _lp(name) + tag4)
    c1001 = _zchunk(0x1001, struct.pack("<IBBQB", 48000, 1, 24, 480000, 17))
    c1004 = _zchunk(0x1004, _zchunk(0x103A, bytes(b103a)) + _zchunk(0x1003, struct.pack("<I", 3) + c1001) + _zchunk(0x1003, struct.pack("<I", 5) + c1001))
    sess = PtxSession.from_bytes(_wrap_ptx([_zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000)), c1004]))
    assert sess.info.session_root == r"C:\Song"
    assert sess.audio_files[0].filename == "Guit\u00e4r.mxf"
    assert sess.audio_files[0].relative_path == "Audio Files/Guit\u00e4r.mxf"
    assert sess.audio_files[1].filename == "Loop.wav" and sess.audio_files[1].path.count("CycleA") == 1

    # macOS external volume path (/Volumes/ExtDrive/...)
    mac_entries = [
        (0, 0xFFFFFFFF, "ExtDrive", b"\x00\x00\x00\x00"),
        (1, 0, "Audio Files", b"\x00\x00\x00\x00"),
        (2, 1, "Take1.wav", b"EVAW"),
    ]
    b_mac = bytearray(struct.pack("<I", len(mac_entries)))
    for kind, parent, name, tag4 in mac_entries:
        b_mac.extend(struct.pack("<BI", kind, parent) + _lp(name) + tag4)
    c1004_mac = _zchunk(0x1004, _zchunk(0x103A, bytes(b_mac)) + _zchunk(0x1003, struct.pack("<I", 2) + c1001))
    s_mac = PtxSession.from_bytes(_wrap_ptx([_zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000)), c1004_mac]))
    assert s_mac.audio_files[0].path == "/Volumes/ExtDrive/Audio Files/Take1.wav"


def test_parse_clips_warp_clip_gain_flags0_and_fades():
    # Curve 0 has flags=0 (bypassed/disabled) so curve 1 must still be parsed cleanly!
    rec0 = _control_curve(0, 4, [(0, -1.0)], flags=0)
    rec1 = _control_curve(0, 4, [(0, -3.5), (24000, 2.0)], flags=1)
    c2637 = _zchunk(0x2637, struct.pack("<I", 2) + rec0 + rec1)

    # Unsorted delta warp markers + packed offset=10000, sync=500, length=48000 + ARA2
    warp = _zchunk(0x2636, _control_curve(0, 8, [(10000, 10000), (58000, 106000), (34000, 58000)]))
    packed = bytes([0x01, 0x30, 0x32, 0x00, 0x00]) + (10000).to_bytes(3, "little") + (500).to_bytes(2, "little") + (48000).to_bytes(3, "little")
    b28 = _zchunk(0x2628, _lp("GuitClip") + packed + warp + struct.pack("<i", 1) + b"\x00\x00" + _uuid(50), version=5)
    c2629 = _zchunk(0x2629, b28 + struct.pack("<IHf", 0, 0, 1.0) + _zchunk(0x4403, b"\x00" * 40) + b"\x01" + _uuid(51) + _zchunk(0x4807, b"elecGLPMXXXX" + _lp("Melodyne")))

    # Fades: fade-in, fade-out, linked crossfade, unlinked crossfade
    f_in = _zchunk(0x262F, b"\x00" * 5 + bytes([0x30, 0, 0]) + (2400).to_bytes(3, "little") + bytes([3]) + b"\x00" * 5 + struct.pack("<Bh", 1, 1200))
    f_out = _zchunk(0x262F, b"\x00" * 5 + bytes([0x33, 0, 0]) + (3000).to_bytes(3, "little") + (3000).to_bytes(3, "little") + bytes([2]) + b"\x00" * 11 + struct.pack("<Bh", 0, -400))
    f_xf = _zchunk(0x262F, b"\x00" * 5 + bytes([0x33, 0, 0]) + (1200).to_bytes(3, "little") + (2400).to_bytes(3, "little") + bytes([1, 3]) + b"\x00" * 10 + struct.pack("<Bh", 0, -500) + b"\x00" * 4 + struct.pack("<Bh", 1, 700))

    sess = PtxSession.from_bytes(_wrap_ptx([_zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000)), c2637, _zchunk(0x262A, c2629), _zchunk(0x2630, f_in + f_out + f_xf)]))
    assert len(sess.clip_gain_curves) == 2 and sess.clip_gain_curves[0].flags == 0
    clip = sess.clips[0]
    assert clip.start_sample == 10000 and clip.sync_point_samples == 500 and clip.length_samples == 48000
    assert clip.timeline_length_samples == 96000 and clip.playback_rate == pytest.approx(0.5)
    assert clip.gain_db == pytest.approx(-3.5) and clip.ara_plugin_name == "Melodyne"
    assert [(f.fade_type, f.shape_name, f.curve_amount) for f in sess.fades[:2]] == [("in", "s_curve", 1200), ("out", "standard", -400)]
    assert sess.fades[2].crossfade_link_name == "none" and sess.fades[2].out_curve_amount == -500 and sess.fades[2].in_curve_amount == 700


def test_parse_compounds_loops_fade_separation_and_alt_playlists():
    comp_28 = _zchunk(0x2628, _lp("CompGrp") + bytes([0, 0x20, 0x30, 0, 0]) + (100).to_bytes(2, "little") + (96000).to_bytes(3, "little"), version=2)
    c262c = _zchunk(0x262C, _zchunk(0x262B, comp_28))
    comp_ev = _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 100) + bytes([0x00]) + struct.pack("<H", 0xFFFE))
    c2428 = _zchunk(0x2428, _zchunk(0x1054, _zchunk(0x1052, _lp("CompGrp") + comp_ev)))

    loop_242b = _zchunk(0x242B, b"\x00" * 27 + bytes([4]))
    ev_fade = _zchunk(0x1050, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 0) + bytes([0x01]) + struct.pack("<H", 0xFFFE)) + b"\x01\x01")
    ev_clip = _zchunk(0x1050, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 0) + bytes([0x03]) + struct.pack("<H", 0xFFFE) + b"\x00" + loop_242b) + b"\x00\x01")
    ev_comp = _zchunk(0x1050, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 100000) + bytes([0x03]) + struct.pack("<H", 0xFFFE) + b"\x01") + b"\x00\x01")
    ev_muted = _zchunk(0x1050, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 48000) + bytes([0x01]) + struct.pack("<H", 0xFFFE)) + b"\x00\x01")
    c1054 = _zchunk(0x1054, _zchunk(0x1052, _lp("Guitar") + ev_fade + ev_clip + ev_comp) + _zchunk(0x1052, _lp("Guitar.01") + ev_muted))
    c1015 = _zchunk(0x1015, _zchunk(0x1014, _lp("Guitar") + b"\x00" + struct.pack("<IH", 1, 0) + _uuid(2)) + _zchunk(0x1014, _lp("Guitar.01") + b"\x00" + struct.pack("<IH", 1, 1) + _uuid(3)))
    c2519 = _zchunk(0x2519, _v10_track("Guitar", 0, _uuid(1)), version=9)

    sess = PtxSession.from_bytes(_wrap_ptx([_zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000)), c262c, c2428, c2519, c1054, c1015]))
    assert sess.compound_clips[0].members[0].is_muted is False
    trk = sess.track_by_name("Guitar")
    assert trk is not None and len(trk.fade_placements) == 1 and len(trk.clips) == 2
    assert trk.clips[0].loop_count == 4 and trk.clips[1].is_compound is True
    assert len(trk.alternate_playlists) == 1 and trk.alternate_playlists[0].placements[0].is_muted is True


def test_parse_midi_and_video_tracks():
    note = bytearray(35)
    struct.pack_into("<Q", note, 0, ZERO_TICKS + 480000)
    note[8], note[17] = 60, 100
    struct.pack_into("<Q", note, 9, 240000)
    mdnlb = b"MdNLB\x03\x00\x00\x00\x00\x00" + struct.pack("<I", 1) + bytes(note)

    def _mdcds(ctype: int, cnum: int, evs: list[tuple[int, int]]) -> bytes:
        ss = b"Mdss" + b"\x00" * 6 + bytes([ctype]) + struct.pack("<I", cnum)
        cd = b"MdCD\x01" + b"\x00" * 5 + struct.pack("<I", len(evs)) + b"".join(struct.pack("<QI", ZERO_TICKS + tk, val & 0xFFFFFFFF) for tk, val in evs)
        return b"MdCDS\x01\x00" + struct.pack("<I", len(ss) + len(cd)) + ss + cd

    mbody = mdnlb + _mdcds(1, 0, [(0, -4096)]) + _mdcds(2, 64, [(480000, 127)])
    c2000 = _zchunk(0x2000, struct.pack("<I", 1) + b"MdChun\x01\x00" + struct.pack("<I", len(mbody)) + mbody)
    m28 = _zchunk(0x2628, _lp("Keys-01") + bytes([0, 0x50, 0x30, 0, 0]) + ZERO_TICKS.to_bytes(5, "little") + (TICKS_PER_QUARTER * 2).to_bytes(3, "little"), version=2)
    c2634 = _zchunk(0x2634, _zchunk(0x2633, m28 + struct.pack("<I", 0)))
    c1058 = _zchunk(0x1058, _zchunk(0x1057, _lp("Keys") + _zchunk(0x1056, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", ZERO_TICKS) + bytes([0x03])))))
    c1006 = _zchunk(0x1006, _lp("Reel1.mov") + b"\x00\x00" + _lp("Reel2.mp4"))
    c262e = _zchunk(0x262E, _zchunk(0x262D, _zchunk(0x2628, _lp("Pic") + bytes([0, 0x20, 0x20, 0, 0]) + (25).to_bytes(2, "little") + (250).to_bytes(2, "little"), version=2)))
    c1055 = _zchunk(0x1055, _zchunk(0x1053, _lp("Video 1") + _zchunk(0x1051, _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 48000) + bytes([0x03])))))
    c2519 = _zchunk(0x2519, _v10_track("Keys", 7, _uuid(1)) + _v10_track("Video 1", 3, _uuid(2)), version=9)

    sess = PtxSession.from_bytes(_wrap_ptx([_zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000)), c2519, c2000, c2634, c1058, c1006, c262e, c1055]))
    assert sess.midi_chunks[0].notes[0].pitch == 60 and sess.midi_chunks[0].control_lanes[0].events[0].value == -4096
    assert sess.track_by_name("Keys").midi_clips[0].duration_seconds == pytest.approx(1.0)
    assert sess.video_files == ["Reel1.mov", "Reel2.mp4"] and sess.track_by_name("Video 1").video_clips[0].start_seconds == 1.0


def test_parse_mixer_automation_0c26_collision_long_comment_and_pt12_flags():
    out_uid, trk_uid, vca_uid, hid_uid = _uuid(90), _uuid(10), _uuid(11), _uuid(12)
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c1022 = _zchunk(0x1022, struct.pack("<I", 1) + _zchunk(0x1021, bytes([0, 0]) + _lp("Mic 1") + struct.pack("<IH", 1, 1) + struct.pack("<HB", 0xFFFF, 1) + _uuid(91)))
    c2603 = _zchunk(0x2603, struct.pack("<I", 1) + _zchunk(0x2602, bytes([1, 1]) + _lp("Main Out") + struct.pack("<IHH", 2, 1, 2) + struct.pack("<HBB", 0xFFFF, 1, 0) + out_uid + b"\x00" * 16))
    c2519 = _zchunk(0x2519, _v6_track("Vocal", 0, trk_uid, disp_idx=1) + _v6_track("VCA1", 3, vca_uid, disp_idx=2) + _v6_track("Hidden", 0, hid_uid, disp_idx=0), version=7)

    # Regression #1: timestamp 9740 (0x0C 0x26 LE) inside volume 0x260A must NOT truncate direct lanes!
    vol_a = _zchunk(0x260A, _control_curve(0, 2, [(0, -60), (0x260C, 20)]) + b"\x00\x00")
    mute_a = _zchunk(0x260A, _control_curve(0, 2, [(0, 0)]) + b"\x00\x00")
    pan_a = _zchunk(0x260A, _control_curve(0, 2, [(0, -50), (48000, 50)]) + b"\x00\x00")
    d0 = _zchunk(0x260D, _zchunk(0x260E, b"\x00\x00\x00\x00" + out_uid) + vol_a + mute_a + _zchunk(0x260C, pan_a))
    # Regression #3: >900-byte track comment + optional 0x2104 subchunk inside 0x2015
    long_comment = "X" * 950
    s2038 = _zchunk(0x2038, b"\x00" * 9 + bytes([1, 1]) + b"\x00" * 5)
    s2015 = _zchunk(0x2015, s2038 + b"\x00" * 8 + _lp(long_comment) + b"\x00" * 9 + b"\x01" + _zchunk(0x2104, b"\x00\x00") + b"\x00" + struct.pack("<h", 32))
    # Pre-0x4856 hardware input fallback at 0x2627 - 3 (index 0 -> Mic 1)
    strip_voc = _zchunk(0x261C, _zchunk(0x2619, _lp("Vocal") + trk_uid) + bytes([1, 3]) + struct.pack("<HB", 0, 1) + _zchunk(0x2627, struct.pack("<H", 0)) + d0 + s2015 + bytes([1]) + b"\x00" * 34)
    strip_vca = _zchunk(0x261F, vca_uid + _zchunk(0x2610, _zchunk(0x260A, _control_curve(0, 2, [(0, -30), (48000, 0)]) + b"\x00\x00")))
    c2624 = _zchunk(0x2624, struct.pack("<I", 2) + strip_voc + strip_vca)

    sess = PtxSession.from_bytes(_wrap_ptx([c1028, c1022, c2603, c2519, c2624]))
    voc = sess.track_by_name("Vocal")
    assert voc is not None and voc.volume_db == -6.0 and len(voc.volume_automation.points) == 2
    assert voc.pan_left == -50 and voc.is_output_active is False and voc.is_inactive is True
    assert voc.comment == long_comment and voc.color_index == 32 and voc.is_solo_safe is True
    assert voc.is_record_enabled is True and voc.is_record_safe is True
    assert voc.input_path is not None and voc.input_path.name == "Mic 1"
    assert sess.track_by_name("VCA1").track_type == "Vca" and sess.track_by_name("VCA1").volume_db == -3.0
    assert sess.track_by_name("Hidden").is_hidden is True


def test_parse_inserts_2617_guard_nan_safety_ara2_sequential_and_json():
    trk_uid, out_uid = _uuid(10), _uuid(90)
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c1018 = _zchunk(0x1018, struct.pack("<I", 1) + _zchunk(0x1017, b"\x00" + _lp("Trim") + b"igiDmirT11RT" + b"com.avid.aax.trim"))
    c2603 = _zchunk(0x2603, struct.pack("<I", 1) + _zchunk(0x2602, bytes([1, 1]) + _lp("Out") + struct.pack("<IHH", 2, 1, 2) + struct.pack("<HBB", 0xFFFF, 1, 0) + out_uid + b"\x00" * 16))
    c2519 = _zchunk(0x2519, _v10_track("Vox", 0, trk_uid), version=9)

    # Regression #7: 0x4807-like bytes inside ARA2 state blob must NOT spawn a phantom plugin
    decoy_4807 = _zchunk(0x4807, b"xxxxYYYYZZZZ" + _lp("Phantom"))
    ara_blob = b"\x00" * 24 + b"Melodyne Plug-In Data\x00" + b"\x00" * 10 + decoy_4807
    c4823 = _zchunk(0x4823, struct.pack("<II", 0, 1) + _zchunk(0x4807, b"elecGLPMXXXX" + _lp("Melodyne")) + struct.pack("<I", len(ara_blob)) + ara_blob)
    c4820 = _zchunk(0x4820, _zchunk(0x480C, _lp("chunk.1") + _zchunk(0x4807, b"elecGLPMXXXX" + _lp("Melodyne")) + _zchunk(0x480B, b"\x00" * 16 + _lp("src-1") + _zchunk(0x480A, _uuid(77) + _lp("mod-1") + _lp("doc-1")))))

    # Regression #4 & #5: 0x2617 Slot A wrapper does NOT populate slot_a_trim; empty 0x2617 advances slot index; NaN float ignored
    ctrl_state = b"Complete Controls State\x00" + b"l_hpfs\x00\x00" + struct.pack(">i", 2) + b"d_bad \x00\x00" + struct.pack(">d", float("nan")) + b"d_frq1\x00\x00" + struct.pack(">d", 120.5)
    s1038 = _zchunk(0x1038, b"\x00" * 16 + b"DigiTrimTR11All " + b"Complete Controls State".ljust(32, b"\x00") + ctrl_state)
    s260f = _zchunk(0x260F, _zchunk(0x260B, _control_curve(0, 4, [(0, 0), (48000, 1 << 30)], signed_int32=True)) + _lp("Gain") + struct.pack("<I", 4) + b"gai2")
    s2616 = _zchunk(0x2616, _zchunk(0x2615, _zchunk(0x2626, b"") + struct.pack("<H", 0)) + _zchunk(0x2613, s1038 + b"\x01") + b"\x01" + _zchunk(0x2614, s260f))
    s2627 = _zchunk(0x2627, struct.pack("<H", 3) + _zchunk(0x2617, s2616) + _zchunk(0x2617, b"\x00\x00") + s2616)
    nan_pan = _zchunk(0x260C, _zchunk(0x260A, _control_curve(0, 4, [(0, float("nan"))]) + b"\x00\x00"))
    d0 = _zchunk(0x260D, _zchunk(0x260E, b"\x00\x00\x00\x01" + out_uid) + _zchunk(0x260A, _control_curve(0, 2, [(0, 0)]) + b"\x00\x00") + nan_pan)
    c2624 = _zchunk(0x2624, struct.pack("<I", 1) + _zchunk(0x261C, trk_uid + s2627 + d0 + _zchunk(0x4807, b"elecGLPMXXXX" + _lp("Melodyne")) + b"\x00" * 34))

    sess = PtxSession.from_bytes(_wrap_ptx([c1028, c1018, c2603, c2519, c4823, c4820, c2624]))
    assert [a.name for a in sess.ara_plugins] == ["Melodyne"] and sess.ara_bindings[0].audio_source_ids == ["src-1"]
    vox = sess.track_by_name("Vox")
    assert vox is not None and vox.slot_a_trim is None and vox.pan_left is None
    assert [(ins.slot, ins.slot_index, ins.is_active) for ins in vox.inserts] == [("A", 0, False), ("C", 2, True)]
    assert "bad" not in vox.inserts[0].state.controls and vox.inserts[0].state.controls["frq1"] == pytest.approx(120.5)
    assert vox.inserts[0].automation[0].points[1].value == pytest.approx(0.5)
    assert json.loads(json.dumps(sess.to_dict(), allow_nan=False))["info"]["sample_rate"] == 48000
    assert sess.io_configuration()["counts"]["total_routable_2603"] == 1
