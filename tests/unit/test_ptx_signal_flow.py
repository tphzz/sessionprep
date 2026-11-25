"""DSP and routing unit tests for Pro Tools signal-flow DAG summing and bus fader staging."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from sessionpreplib.ptx import (
    LeafAudioInput,
    PtxSession,
    build_annotated_signal_flow,
    pan_gains,
    patch_add_group,
)
from sessionpreplib.ptx.codec import encrypt_ptx
from sessionpreplib.ptx.models import PTX_MAGIC
from tests.unit.test_ptx_core import (
    _chunk_1029,
    _chunk_260a,
    _mixer_track_261e,
    _mixer_vca_261f,
    _sp_entry,
    _track_251a,
    _uuid,
    _zchunk,
)


def _mixer_with_pan_and_send(
    uid: bytes,
    input_uid: bytes | None,
    output_uid: bytes,
    *,
    vol_db: float = 0.0,
    muted: bool = False,
    pan_l: int | None = None,
    pan_r: int | None = None,
    send_dest: bytes | None = None,
    send_db: float = 0.0,
    send_pre: bool = False,
) -> bytes:
    val = int(round(vol_db * 10.0))
    pan_chunks = b""
    if pan_l is not None:
        pan_chunks += _zchunk(0x260C, _chunk_260a(pan_l))
    if pan_r is not None:
        pan_chunks += _zchunk(0x260C, _chunk_260a(pan_r))
    d0 = _zchunk(
        0x260D,
        _zchunk(0x260E, b"\x00\x00\x00\x01" + output_uid)
        + _chunk_1029(val)
        + _chunk_260a(val)
        + _chunk_260a(1 if muted else 0)
        + pan_chunks,
    )
    d1 = b""
    if send_dest is not None:
        sval = int(round(send_db * 10.0))
        d1 = _zchunk(
            0x260D,
            _zchunk(0x260E, b"\x00\x00\x00\x01" + send_dest)
            + _chunk_1029(sval, post_fader=not send_pre)
            + _chunk_260a(sval)
            + _chunk_260a(0),
        )
    pre_input = (input_uid if input_uid else b"") + _zchunk(0x4856, b"\x00" * 8)
    return _zchunk(0x261E, uid + pre_input + d0 + d1 + (b"\x00" * 40), version=1)


def _build_console_ptx(
    *,
    parallel_vol: float = -144.0,
    create_cycle: bool = False,
    mono_kik_bus: bool = False,
    k1_mono: bool = False,
    k1_pan: int | None = None,
    mus_pan: tuple[int | None, int | None] = (None, None),
    vca_db: float = 0.0,
    vca_inactive: bool = False,
    k1_muted: bool = False,
    send_pre: bool = False,
) -> bytes:
    b_out = _uuid(1)
    b_kik, b_ckik, b_snr, b_csnr = _uuid(2), _uuid(3), _uuid(4), _uuid(5)
    b_alldrm, b_mus, b_mix1, b_pa, b_fx = _uuid(6), _uuid(7), _uuid(8), _uuid(9), _uuid(10)

    u_k1, u_k2, u_s1, u_pad = _uuid(20), _uuid(21), _uuid(22), _uuid(23)
    u_kik, u_snr, u_mus = _uuid(30), _uuid(31), _uuid(32)
    u_ckik, u_csnr, u_alldrm, u_mix1, u_pa, u_fx, u_mst, u_vca = (
        _uuid(40), _uuid(41), _uuid(42), _uuid(43), _uuid(44), _uuid(45), _uuid(46), _uuid(47)
    )

    sps = [
        ("Main Out", 1, b_out), ("KIK", 2, b_kik), ("cKIK", 2, b_ckik), ("SNR", 2, b_snr),
        ("cSNR", 2, b_csnr), ("ALLDRM", 2, b_alldrm), ("MUS", 2, b_mus), ("MIX1", 2, b_mix1),
        ("RYT-PA", 2, b_pa), ("FXBUS", 2, b_fx),
    ]
    c0003 = _zchunk(0x0003, b"Pro Tools Studio\x00\x002026.11.1\x00\x00Release\x00\x00Session\x00\x00Windows\x00")
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c2603 = _zchunk(0x2603, struct.pack("<I", len(sps)) + b"".join(_sp_entry(n, k, True, u) for n, k, u in sps), version=2)

    vca_chunk = bytearray(_track_251a("KIK-VCA", 9, u_vca, is_stereo=False))
    if vca_inactive:
        vca_chunk[-6] = 0

    tracks = [
        _track_251a("K1", 0, u_k1, is_stereo=not k1_mono),
        _track_251a("K2", 0, u_k2, is_stereo=True),
        _track_251a("S1", 0, u_s1, is_stereo=True),
        _track_251a("Pad", 0, u_pad, is_stereo=True),
        _track_251a("KIK", 2, u_kik, is_folder=True, is_stereo=not mono_kik_bus),
        _track_251a("SNR", 2, u_snr, is_folder=True),
        _track_251a("MUS", 2, u_mus, is_folder=True),
        _track_251a("cKIK", 2, u_ckik, is_folder=False),
        _track_251a("cSNR", 2, u_csnr, is_folder=False),
        _track_251a("ALLDRM", 2, u_alldrm, is_folder=False),
        _track_251a("MIX1", 2, u_mix1, is_folder=False),
        _track_251a("RYT-PA", 2, u_pa, is_folder=False),
        _track_251a("FX", 2, u_fx, is_folder=False),
        _track_251a("Master", 5, u_mst, is_folder=False),
        bytes(vca_chunk),
    ]
    c2519 = _zchunk(0x2519, b"".join(tracks), version=9)

    mix1_out = b_kik if create_cycle else b_out
    mixers = [
        _mixer_with_pan_and_send(u_k1, None, b_kik, muted=k1_muted, pan_l=k1_pan, send_dest=b_fx, send_db=0.0, send_pre=send_pre),
        _mixer_track_261e(u_k2, None, b_kik, 0.0),
        _mixer_track_261e(u_s1, None, b_snr, 0.0),
        _mixer_track_261e(u_pad, None, b_mus, 0.0),
        _mixer_with_pan_and_send(u_kik, b_kik, b_ckik, pan_l=0 if mono_kik_bus else None),
        _mixer_track_261e(u_snr, b_snr, b_csnr, 0.0),
        _mixer_with_pan_and_send(u_mus, b_mus, b_mix1, pan_l=mus_pan[0], pan_r=mus_pan[1]),
        _mixer_track_261e(u_ckik, b_ckik, b_alldrm, 0.0),
        _mixer_track_261e(u_csnr, b_csnr, b_alldrm, 0.0),
        _mixer_track_261e(u_alldrm, b_alldrm, b_mix1, 0.0),
        _mixer_track_261e(u_mix1, b_mix1, mix1_out, 0.0),
        _mixer_track_261e(u_pa, b_pa, b_alldrm, parallel_vol),
        _mixer_track_261e(u_fx, b_fx, b_mix1, 0.0),
        _mixer_track_261e(u_mst, None, b_out, 0.0),
        _mixer_vca_261f(u_vca, vca_db),
    ]
    c2624 = _zchunk(0x2624, struct.pack("<I", len(mixers)) + b"".join(mixers))
    c202b = _zchunk(0x202B, struct.pack("<II", 0, 0) + b"\x00\x00", version=4)
    pad = _zchunk(0x7FFE, b"\x00" * 9000)

    out = bytearray(PTX_MAGIC + b"\x00\x05\x90")
    out.extend(struct.pack("<BHII", 0x5A, 1, 4, 0))
    for ch in (c0003, c1028, c2603, c2519, c2624, c202b, pad):
        out.extend(ch)
    return encrypt_ptx(bytes(out), 0x33)


def _transient_hit(sr: int, duration_s: float, peak_lin: float, hit_times: list[float]) -> np.ndarray:
    n = int(sr * duration_s)
    sig = np.zeros(n, dtype=np.float64)
    decay = int(sr * 0.03)
    env = np.exp(-np.linspace(0, 8, decay))
    t = np.arange(decay) / sr
    pulse = env * np.sin(2 * np.pi * 120 * t)
    pulse = pulse * (peak_lin / np.max(np.abs(pulse)))
    for sec in hit_times:
        start = int(sec * sr)
        end = min(n, start + decay)
        sig[start:end] += pulse[: end - start]
    return np.column_stack((sig, sig))


def _sustained_tone(sr: int, duration_s: float, rms_lin: float, freq: float = 220.0, *, mono: bool = False) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    sig = rms_lin * math.sqrt(2.0) * np.sin(2 * np.pi * freq * t)
    return sig if mono else np.column_stack((sig, sig))


def test_pan_law_all_depths_including_zero_db_balance_and_continuity():
    for depth in (0.0, -2.5, -3.0, -4.5, -6.0):
        assert pan_gains(-100, depth) == pytest.approx((1.0, 0.0), abs=1e-9)
        assert pan_gains(100, depth) == pytest.approx((0.0, 1.0), abs=1e-9)
        gl, gr = pan_gains(0, depth)
        assert gl == pytest.approx(gr, abs=1e-9)
        assert 20.0 * math.log10(gl) == pytest.approx(depth, abs=1e-4)
    # Continuous near -3.0 dB (no jump between -2.95 and -3.05)
    l_hi, _ = pan_gains(-40, -2.95)
    l_lo, _ = pan_gains(-40, -3.05)
    assert abs(20.0 * math.log10(l_hi / l_lo)) < 0.1


def test_correlated_kick_sum_stages_kik_folder_and_leaves_single_input_aux_and_master_unstaged():
    sr = 48000
    ptx = PtxSession.from_bytes(_build_console_ptx())
    peak_m6 = 10.0 ** (-6.0 / 20.0)
    k_hit = _transient_hit(sr, 2.0, peak_m6, [0.2, 0.8, 1.4])
    s_hit = _transient_hit(sr, 2.0, peak_m6, [0.5, 1.1, 1.7])
    pad_sig = _sustained_tone(sr, 2.0, 10.0 ** (-18.0 / 20.0), freq=220.0)

    leaf_audio = {
        "K1": LeafAudioInput(k_hit, sr, clip_gain_db=0.0, fader_db=0.0),
        "K2": LeafAudioInput(k_hit, sr, clip_gain_db=0.0, fader_db=0.0),
        "S1": LeafAudioInput(s_hit, sr, clip_gain_db=0.0, fader_db=0.0),
        "Pad": LeafAudioInput(pad_sig, sr, clip_gain_db=0.0, fader_db=0.0),
    }
    graph = build_annotated_signal_flow(ptx, leaf_audio, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True)
    assert graph.cycles == []

    kik = graph.nodes["KIK"]
    assert kik.classification == "Transient" and kik.is_transient is True
    assert kik.pre_fader_peak_db == pytest.approx(0.0, abs=0.15)
    assert kik.staged is True and kik.recommended_fader_db == pytest.approx(-6.0, abs=0.2)
    assert graph.nodes["cKIK"].staged is False and graph.nodes["cSNR"].staged is False
    assert graph.nodes["ALLDRM"].staged is True and set(graph.nodes["ALLDRM"].upstream_tracks) == {"cKIK", "cSNR"}
    assert graph.nodes["MIX1"].staged is True
    assert graph.nodes["Master"].staged is False and graph.nodes["Master"].pre_fader_peak_db is not None
    assert "Master" not in graph.recommended_bus_faders() and "RYT-PA" not in graph.recommended_bus_faders()


def test_uncorrelated_rms_sum_plus_3db_and_exact_out_of_phase_cancellation():
    sr = 48000
    ptx = PtxSession.from_bytes(_build_console_ptx())
    rms_m18 = 10.0 ** (-18.0 / 20.0)
    # Orthogonal frequencies -> uncorrelated +3.01 dB RMS sum
    t1 = _sustained_tone(sr, 2.0, rms_m18, freq=220.0)
    t2 = _sustained_tone(sr, 2.0, rms_m18, freq=330.0)
    g_unc = build_annotated_signal_flow(
        ptx, {"K1": LeafAudioInput(t1, sr), "K2": LeafAudioInput(t2, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True,
    )
    assert g_unc.nodes["KIK"].pre_fader_rms_db == pytest.approx(-15.0, abs=0.25)
    assert g_unc.nodes["KIK"].recommended_fader_db == pytest.approx(-3.0, abs=0.3)

    # Exact out-of-phase cancellation (+x + -x == 0) -> Silent, unstaged, not counted downstream
    g_cancel = build_annotated_signal_flow(
        ptx, {"K1": LeafAudioInput(t1, sr), "K2": LeafAudioInput(-t1, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True,
    )
    assert g_cancel.nodes["KIK"].classification == "Silent"
    assert g_cancel.nodes["KIK"].staged is False
    assert g_cancel.nodes["cKIK"].upstream_tracks == []


def test_mono_leaf_pre_pan_metering_and_mono_bus_no_lr_tilt():
    sr = 48000
    mono_sig = _sustained_tone(sr, 2.0, 10.0 ** (-18.0 / 20.0), freq=220.0, mono=True)

    # 1. Mono leaf -> Stereo bus: pre-fader leaf meter is PRE-PAN (-18 dBFS), stereo bus receives -3 dB pan law (-21 dBFS)
    ptx_stereo_bus = PtxSession.from_bytes(_build_console_ptx(k1_mono=True, k1_pan=0, mono_kik_bus=False))
    g1 = build_annotated_signal_flow(ptx_stereo_bus, {"K1": LeafAudioInput(mono_sig, sr)}, {"target_peak": -6.0, "target_rms": -18.0})
    assert g1.nodes["K1"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)
    assert g1.nodes["KIK"].pre_fader_rms_db == pytest.approx(-21.01, abs=0.25)

    # 2. Mono leaf -> Mono bus -> Stereo cKIK: unity into Mono KIK (-18 dBFS), then clean -3 dB center pan into Stereo cKIK (-21 dBFS, no L/R tilt!)
    ptx_mono_bus = PtxSession.from_bytes(_build_console_ptx(k1_mono=True, k1_pan=0, mono_kik_bus=True))
    g2 = build_annotated_signal_flow(
        ptx_mono_bus, {"K1": LeafAudioInput(mono_sig, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=False,
    )
    assert g2.nodes["KIK"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)
    assert g2.nodes["cKIK"].pre_fader_rms_db == pytest.approx(-21.01, abs=0.25)


def test_stereo_bus_narrow_pan_updates_post_fader_levels_accurately():
    sr = 48000
    # Stereo bus MUS panned (0, 0) center sums L+R in-phase -> +3.01 dB (2 * 0.7071 = +3.01 dB)
    ptx = PtxSession.from_bytes(_build_console_ptx(mus_pan=(0, 0)))
    pad_sig = _sustained_tone(sr, 2.0, 10.0 ** (-18.0 / 20.0), freq=220.0)
    g = build_annotated_signal_flow(ptx, {"Pad": LeafAudioInput(pad_sig, sr)}, stage_bus_faders=False)
    assert g.nodes["MUS"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)
    assert g.nodes["MUS"].post_fader_rms_db == pytest.approx(-15.0, abs=0.25)


def test_attenuate_only_preserves_quiet_folder_and_headroom_caps_boost():
    sr = 48000
    ptx = PtxSession.from_bytes(_build_console_ptx())
    quiet = _sustained_tone(sr, 2.0, 10.0 ** (-28.0 / 20.0), freq=330.0)

    # attenuate_only=True keeps quiet folder at 0.0 dB (never undoes inverse balance!)
    g_att = build_annotated_signal_flow(
        ptx, {"Pad": LeafAudioInput(quiet, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=True,
    )
    assert g_att.nodes["MUS"].recommended_fader_db == 0.0

    # attenuate_only=False respects fader_headroom_db=8.0 -> +4.0 dB ceiling instead of slamming +10/+12 dB
    g_boost = build_annotated_signal_flow(
        ptx, {"Pad": LeafAudioInput(quiet, sr)},
        {"target_peak": -6.0, "target_rms": -18.0, "fader_headroom_db": 8.0}, stage_bus_faders=True, attenuate_only=False,
    )
    assert g_boost.nodes["MUS"].recommended_fader_db == pytest.approx(4.0, abs=0.1)

    # Sustained bus with a sharp peak excursion is peak-limited to target_peak (never pushes peak above -6 dBFS)
    spiky = _sustained_tone(sr, 2.0, 10.0 ** (-22.0 / 20.0), freq=220.0)
    spiky[sr // 2, :] = 10.0 ** (-5.0 / 20.0)  # single -5 dBFS peak while RMS stays ~ -22 dBFS
    g_plim = build_annotated_signal_flow(
        ptx, {"Pad": LeafAudioInput(spiky, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=False,
    )
    assert g_plim.nodes["MUS"].classification == "Sustained"
    assert g_plim.nodes["MUS"].recommended_fader_db == pytest.approx(-1.0, abs=0.2)
    assert g_plim.nodes["MUS"].post_fader_peak_db <= -5.8


def test_vca_attenuation_inactive_vca_resampling_mixed_lengths_aes17_nan_and_sends():
    sr = 48000
    # Active VCA at -6 dB wired to K1 attenuates KIK sum by -6 dB
    raw_vca = _build_console_ptx(vca_db=-6.0, vca_inactive=False)
    raw_vca, _ = patch_add_group(raw_vca, "KIK-G", ["K1"], vca_track="KIK-VCA")
    ptx_vca = PtxSession.from_bytes(raw_vca)
    tone_44 = _sustained_tone(44100, 1.0, 10.0 ** (-18.0 / 20.0), freq=220.0)
    tone_48_long = _sustained_tone(sr, 2.0, 10.0 ** (-18.0 / 20.0), freq=330.0)
    g_vca = build_annotated_signal_flow(
        ptx_vca,
        {"K1": LeafAudioInput(tone_44, 44100), "Pad": LeafAudioInput(tone_48_long, sr)},
        {"target_peak": -6.0, "target_rms": -18.0, "dbfs_convention": "aes17", "force_transient": "bus"},
        stage_bus_faders=False,
    )
    assert g_vca.dbfs_convention == "aes17"
    assert g_vca.nodes["K1"].vca_offset_db == -6.0
    assert g_vca.nodes["KIK"].pre_fader_rms_db == pytest.approx(-24.0, abs=0.35)
    assert g_vca.nodes["MUS"].classification == "Sustained"  # force_transient="bus" did not corrupt bus node

    # Inactive VCA at -6 dB does NOT attenuate K1
    raw_inact = _build_console_ptx(vca_db=-6.0, vca_inactive=True)
    raw_inact, _ = patch_add_group(raw_inact, "KIK-G", ["K1"], vca_track="KIK-VCA")
    g_inact = build_annotated_signal_flow(PtxSession.from_bytes(raw_inact), {"K1": LeafAudioInput(tone_48_long, sr)}, stage_bus_faders=False)
    assert g_inact.nodes["K1"].vca_offset_db == 0.0

    # Non-finite NaN leaf is isolated cleanly without crashing other tracks
    bad_buf = tone_48_long.copy()
    bad_buf[100, 0] = float("nan")
    g_nan = build_annotated_signal_flow(
        PtxSession.from_bytes(_build_console_ptx()),
        {"K1": LeafAudioInput(bad_buf, sr), "Pad": LeafAudioInput(tone_48_long, sr)},
        stage_bus_faders=False,
    )
    assert g_nan.nodes["K1"].classification == "Silent"
    assert g_nan.nodes["MUS"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.25)

    # Muted track must not leak into pre-fader send
    ptx_muted_send = PtxSession.from_bytes(_build_console_ptx(k1_muted=True, send_pre=True))
    g_send = build_annotated_signal_flow(
        ptx_muted_send, {"K1": LeafAudioInput(tone_48_long, sr)}, stage_bus_faders=False, include_sends=True,
    )
    assert g_send.nodes["FX"].upstream_tracks == []


def test_signal_flow_detects_routing_cycle_without_hanging():
    ptx = PtxSession.from_bytes(_build_console_ptx(create_cycle=True))
    graph = build_annotated_signal_flow(ptx, {}, {"target_peak": -6.0, "target_rms": -18.0})
    assert len(graph.cycles) == 1
    assert "KIK" in graph.cycles[0] and "MIX1" in graph.cycles[0]


def _make_trim_session_fixture():
    from types import SimpleNamespace
    from sessionpreplib.ptx.models import PtxSend, PtxSignalPathRef, PtxTrack, SlotATrimInfo

    def _tc_trim(gain_db: float = 0.0, *, active: bool = False, bypassed: bool = False) -> SlotATrimInfo:
        return SlotATrimInfo("trackcontrol", "TrackControl", "DMGA/TrCt/RTST", "com.dmgaudio.TrackControl.aax", gain_db, -200.0, 18.0, active, bypassed)

    def _trk(
        idx: int, name: str, ttype: str, *, in_bus: str = "", out_bus: str = "",
        trim: SlotATrimInfo | None = None, vol: float = 0.0, inactive: bool = False, muted: bool = False,
        sends: list[PtxSend] | None = None,
    ) -> PtxTrack:
        return PtxTrack(
            index=idx, name=name, uuid_hex=f"{idx:032x}", uuid_braced=f"{{{idx:032x}}}",
            track_type=ttype, format="Stereo", is_folder=(ttype == "RoutingFolder"),
            is_inactive=inactive, is_muted=muted, pan_left=-100, pan_right=100, volume_db=vol,
            input_path=PtxSignalPathRef(in_bus, "bus", f"i{idx}") if in_bus else None,
            main_output=PtxSignalPathRef(out_bus, "bus", f"o{idx}") if out_bus else None,
            sends=sends or [], slot_a_trim=trim,
        )

    def _wrap(tracks: list[PtxTrack]):
        return SimpleNamespace(
            info=SimpleNamespace(sample_rate=48000), tracks=tracks, groups=[],
            track_by_name=lambda n: next((t for t in tracks if t.name.lower() == n.lower()), None),
            group_by_name=lambda _n: None,
        )

    return _tc_trim, _trk, _wrap, PtxSend, PtxSignalPathRef


def test_recursive_two_level_slot_a_trim_preserves_balance_and_drives_bus_inserts():
    _tc_trim, _trk, _wrap, _PtxSend, _PtxRef = _make_trim_session_fixture()
    tracks = [
        _trk(0, "Kick", "Audio", out_bus="KIK"), _trk(1, "Bass", "Audio", out_bus="EB"), _trk(2, "Vox", "Audio", out_bus="VOX"),
        _trk(3, "KIK", "RoutingFolder", in_bus="KIK", out_bus="cKIK"),
        _trk(4, "EB", "RoutingFolder", in_bus="EB", out_bus="cBAS"),
        _trk(5, "VOX", "RoutingFolder", in_bus="VOX", out_bus="cVOX"),
        _trk(6, "cKIK", "Aux", in_bus="cKIK", out_bus="ALLDRM", trim=_tc_trim()),
        _trk(7, "cBAS", "Aux", in_bus="cBAS", out_bus="ALLRYT", trim=_tc_trim()),
        _trk(8, "cVOX", "Aux", in_bus="cVOX", out_bus="ALLMUS", trim=_tc_trim(active=True, bypassed=True)),
        _trk(9, "ALLDRM", "Aux", in_bus="ALLDRM", out_bus="ALLRYT", trim=_tc_trim()),
        _trk(10, "ALLRYT", "Aux", in_bus="ALLRYT", out_bus="MIX1", trim=_tc_trim()),
        _trk(11, "ALLMUS", "Aux", in_bus="ALLMUS", out_bus="MIX1", trim=_tc_trim()),
        _trk(12, "MIX1", "Aux", in_bus="MIX1", out_bus="Out 1-2"),
        _trk(13, "Master", "Master", out_bus="Out 1-2"),
    ]
    ptx, sr = _wrap(tracks), 48000
    leaves = {
        "Kick": LeafAudioInput(_sustained_tone(sr, 1.0, 10.0 ** (-12.0 / 20.0), freq=80.0), sr),
        "Bass": LeafAudioInput(_sustained_tone(sr, 1.0, 10.0 ** (-26.0 / 20.0), freq=130.0), sr),
        "Vox": LeafAudioInput(_sustained_tone(sr, 1.0, 10.0 ** (-10.0 / 20.0), freq=520.0), sr),
    }
    g = build_annotated_signal_flow(ptx, leaves, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=True)
    for rf in ("KIK", "EB", "VOX"):
        assert g.nodes[rf].staging_strategy == "none" and g.nodes[rf].recommended_fader_db == 0.0
    assert g.nodes["cBAS"].staging_strategy == "trim_and_fader"
    assert g.nodes["cBAS"].recommended_trim_db == pytest.approx(8.0, abs=0.2)
    assert g.nodes["cBAS"].recommended_fader_db == pytest.approx(-8.0, abs=0.2)
    assert g.nodes["cBAS"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)
    assert g.nodes["cVOX"].staging_strategy == "none" and "cVOX" not in g.recommended_slot_a_trims()
    assert g.nodes["ALLMUS"].recommended_trim_db + g.nodes["ALLMUS"].recommended_fader_db == pytest.approx(0.0, abs=1e-6)
    assert g.nodes["MIX1"].staging_strategy == "fader_only" and g.nodes["Master"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)

    g_raw = build_annotated_signal_flow(ptx, leaves, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=False)
    assert g.nodes["ALLRYT"].post_fader_rms_db == pytest.approx(g_raw.nodes["ALLRYT"].post_fader_rms_db, abs=0.1)
    assert g.nodes["ALLMUS"].post_fader_rms_db == pytest.approx(g_raw.nodes["ALLMUS"].post_fader_rms_db, abs=0.1)


def test_slot_a_trim_serial_mix_chains_unity_overrides_and_non_zero_template_fader():
    _tc_trim, _trk, _wrap, PtxSend, PtxRef = _make_trim_session_fixture()
    sr = 48000
    tracks = [
        _trk(0, "Kick", "Audio", out_bus="KIK"), _trk(1, "Vox", "Audio", out_bus="VOX"),
        _trk(2, "KIK", "RoutingFolder", in_bus="KIK", out_bus="MIX1"),
        _trk(3, "VOX", "RoutingFolder", in_bus="VOX", out_bus="cVOX"),
        _trk(4, "cVOX", "Aux", in_bus="cVOX", out_bus="MIX1", trim=_tc_trim(), vol=-12.0, sends=[
            PtxSend("A", 0, PtxRef("PRE_FX", "bus", "s0"), 0.0, False, True),
            PtxSend("B", 1, PtxRef("POST_FX", "bus", "s1"), 0.0, False, False),
        ]),
        _trk(5, "PRE_FX", "Aux", in_bus="PRE_FX", out_bus="FX_OUT"),
        _trk(6, "POST_FX", "Aux", in_bus="POST_FX", out_bus="FX_OUT"),
        _trk(7, "MIX1", "Aux", in_bus="MIX1", out_bus="MIX2"),
        _trk(8, "MIX2", "Aux", in_bus="MIX2", out_bus="MIX3", trim=_tc_trim()),
        _trk(9, "MIX3", "Aux", in_bus="MIX3", out_bus="Out 1-2", trim=_tc_trim()),
        _trk(10, "PRINT", "Aux", in_bus="Out 1-2", out_bus="Out 1-2", trim=_tc_trim(), inactive=True),
        _trk(11, "Master", "Master", out_bus="Out 1-2"),
    ]
    leaves = {
        "Kick": LeafAudioInput(_sustained_tone(sr, 1.0, 10.0 ** (-8.0 / 20.0), freq=80.0), sr),
        "Vox": LeafAudioInput(_sustained_tone(sr, 1.0, 10.0 ** (-24.0 / 20.0), freq=520.0), sr),
    }
    g = build_annotated_signal_flow(
        _wrap(tracks), leaves, {"target_peak": -6.0, "target_rms": -18.0},
        stage_bus_faders=True, attenuate_only=True, include_sends=True,
    )
    assert g.nodes["cVOX"].recommended_trim_db == pytest.approx(6.0, abs=0.2)
    assert g.nodes["cVOX"].recommended_fader_db == pytest.approx(-18.0, abs=0.2)
    assert g.nodes["PRE_FX"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)
    assert g.nodes["POST_FX"].pre_fader_rms_db == pytest.approx(-36.0, abs=0.2)
    assert g.nodes["MIX1"].staging_strategy == "none"
    assert g.nodes["MIX2"].staging_strategy == "trim_only" and g.nodes["MIX2"].recommended_trim_db < 0.0
    assert g.nodes["Master"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.2)

    g_u = build_annotated_signal_flow(_wrap(tracks), leaves, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=False)
    assert g_u.nodes["cVOX"].staging_strategy == "trim_only" and g_u.nodes["cVOX"].recommended_fader_db == -12.0
    g_ovr = build_annotated_signal_flow(
        _wrap(tracks), leaves, {"target_peak": -6.0, "target_rms": -18.0},
        stage_bus_faders=True, attenuate_only=True, bus_fader_overrides={"cVOX": -15.0},
    )
    assert g_ovr.nodes["cVOX"].is_overridden is True and g_ovr.nodes["cVOX"].recommended_trim_db == pytest.approx(6.0, abs=0.2)
    assert g_ovr.nodes["cVOX"].recommended_fader_db == -15.0


def test_slot_a_trim_inactive_isolation_and_staging_off_template_trim():
    _tc_trim, _trk, _wrap, _PtxSend, _PtxRef = _make_trim_session_fixture()
    sr = 48000
    tracks = [
        _trk(0, "K1", "Audio", out_bus="KIK"), _trk(1, "K2", "Audio", out_bus="KIK"),
        _trk(2, "KIK", "RoutingFolder", in_bus="KIK", out_bus="Out 1-2"),
        _trk(3, "UTIL", "Aux", in_bus="UTIL", out_bus="Out 1-2", trim=_tc_trim(gain_db=4.0, active=True), inactive=True),
        _trk(4, "ACT_TRIM", "Aux", in_bus="KIK", out_bus="Out 1-2", trim=_tc_trim(gain_db=4.0, active=True)),
        _trk(5, "INACT_TRIM", "Aux", in_bus="KIK", out_bus="Out 1-2", trim=_tc_trim(gain_db=4.0, active=False)),
        _trk(6, "Master", "Master", out_bus="Out 1-2"),
    ]
    leaves = {
        "K1": LeafAudioInput(_sustained_tone(sr, 0.5, 10.0 ** (-10.0 / 20.0), freq=80.0), sr),
        "K2": LeafAudioInput(_sustained_tone(sr, 0.5, 10.0 ** (-10.0 / 20.0), freq=120.0), sr),
    }
    g_on = build_annotated_signal_flow(_wrap(tracks[:4] + [tracks[6]]), leaves, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=True)
    assert g_on.nodes["KIK"].staging_strategy == "fader_only" and g_on.nodes["KIK"].recommended_fader_db < 0.0
    g_off = build_annotated_signal_flow(_wrap(tracks), leaves, {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=False)
    assert g_off.nodes["ACT_TRIM"].pre_fader_rms_db == pytest.approx(g_off.nodes["ACT_TRIM"].pre_insert_rms_db + 4.0, abs=0.1)
    assert g_off.nodes["INACT_TRIM"].pre_fader_rms_db == pytest.approx(g_off.nodes["INACT_TRIM"].pre_insert_rms_db, abs=0.1)
