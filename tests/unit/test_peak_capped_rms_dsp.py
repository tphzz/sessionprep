"""Headless DSP verification matrix (M1-M13) for Peak-Capped RMS normalization."""

from __future__ import annotations

import html
import json
import math
import re

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import (
    AES17_OFFSET,
    find_governing_and_next_peak,
    get_gated_rms_data,
    get_rms_window_means,
    get_window_samples,
    load_track,
    measure_true_peak_db,
)
from sessionpreplib.config import (
    build_structured_defaults,
    default_config,
    migrate_legacy_config,
    migrate_legacy_override,
    validate_config_fields,
    validate_structured_config,
)
from sessionpreplib.detectors import DCOffsetDetector, SilenceDetector
from sessionpreplib.gain_staging import apply_group_gains
from sessionpreplib.models import SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionpreplib.ptx.models import PtxSignalPathRef, PtxTrack, SlotATrimInfo
from sessionpreplib.ptx.signal_flow import LeafAudioInput, build_annotated_signal_flow
from sessionpreplib.repair_verification import RepairVerification
from sessionpreplib.reports import save_json

RATES = (44100, 48000, 96000, 192000)


def _track(samples: np.ndarray, sr: int, name: str = "Track.wav", subtype: str = "DOUBLE") -> TrackContext:
    arr = np.asarray(samples, dtype=np.float64)
    ch = 1 if arr.ndim == 1 else arr.shape[1]
    return TrackContext(
        filename=name,
        filepath="",
        audio_data=arr,
        samplerate=sr,
        channels=ch,
        total_samples=arr.shape[0],
        bitdepth="64-bit float" if subtype == "DOUBLE" else "24-bit",
        subtype=subtype,
        duration_sec=arr.shape[0] / sr,
    )


def _run_norm(track: TrackContext, **cfg):
    proc = BimodalNormalizeProcessor()
    proc.configure({**default_config(), **cfg})
    return proc, proc.process(track)


def _apply_norm(proc: BimodalNormalizeProcessor, samples: np.ndarray, sr: int, res) -> np.ndarray:
    return proc.apply(_track(samples, sr), res)


def _analytic_peak_rms_db(samples: np.ndarray, sr: int, win_ms: int = 400, p: float = 95.0) -> tuple[float, float]:
    """Independent NumPy analytic oracle (no library helpers)."""
    arr = np.asarray(samples, dtype=np.float64)
    peak_db = 20.0 * math.log10(float(np.max(np.abs(arr))))
    power = arr ** 2 if arr.ndim == 1 else np.mean(arr ** 2, axis=1)
    ws = min(max(1, round(win_ms * sr / 1000.0)), len(power))
    cs = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    means = (cs[ws:] - cs[:-ws]) / ws
    mmax = float(np.max(means))
    active = means[means >= mmax * 1e-4]
    rms_db = 10.0 * math.log10(float(np.percentile(active, p)))
    return peak_db, rms_db


# ── M1: Pop/Rock Compressed Drum Loop ────────────────────────────────────────

@pytest.mark.parametrize("sr", RATES)
def test_m1_pop_rock_compressed_drum_loop_across_rates(sr):
    rng = np.random.default_rng(101)
    dur = 8.0  # 4 bars @ 120 BPM
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    # Dense 8th-note compressed drum loop: steady bed + compressed hits -> ~9 dB crest
    sig = 0.16 * np.sin(2.0 * np.pi * 110.0 * t)
    for beat_s in np.arange(0.0, dur, 0.25):
        i0 = int(beat_s * sr)
        L = min(int(0.08 * sr), n - i0)
        tt = np.arange(L, dtype=np.float64) / sr
        hit = np.exp(-tt / 0.025) * (0.5 * np.sin(2.0 * np.pi * 65.0 * tt) + 0.35 * rng.uniform(-1, 1, L))
        sig[i0:i0 + L] += hit
    # Soft-clip/compress to ~8.5 dB crest with peak ~ -5 dBFS
    sig = np.tanh(sig * 3.8)
    sig *= 10.0 ** (-5.0 / 20.0) / float(np.max(np.abs(sig)))
    exp_pk, exp_rms = _analytic_peak_rms_db(sig, sr)
    assert 6.5 <= (exp_pk - exp_rms) <= 9.0

    proc, res = _run_norm(_track(sig, sr, "DrumLoop.wav"))
    assert res.classification == "Auto (RMS)"
    assert res.data["governing"] == "RMS"
    assert res.gain_db == pytest.approx(-18.0 - exp_rms, abs=0.05)
    rendered = _apply_norm(proc, sig, sr, res)
    post_pk, post_rms = _analytic_peak_rms_db(rendered, sr)
    assert post_rms == pytest.approx(-18.0, abs=0.1)
    assert post_pk <= -8.9


# ── M2: R&B 808 Sub Bass ─────────────────────────────────────────────────────

@pytest.mark.parametrize("sr", RATES)
def test_m2_rnb_808_sub_bass_across_rates(sr):
    notes = (41.2, 49.0, 55.0, 41.2)
    segs = []
    for hz in notes:
        n = int(0.75 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        f_inst = hz + 120.0 * np.exp(-t / 0.015)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
        env = np.minimum(1.0, t / 0.004) * np.exp(-t / 0.55)
        segs.append(0.35 * env * np.sin(phase))
    sig = np.concatenate(segs)
    proc, res = _run_norm(_track(sig, sr, "808_Sub.wav"))
    assert res.classification == "Auto (RMS)"
    assert res.data["governing"] == "RMS"
    rendered = _apply_norm(proc, sig, sr, res)
    _, post_rms = _analytic_peak_rms_db(rendered, sr)
    assert post_rms == pytest.approx(-18.0, abs=0.1)


def test_m2_rnb_808_cross_rate_gain_spread():
    gains = []
    for sr in RATES:
        t = np.arange(int(2.0 * sr), dtype=np.float64) / sr
        phase = 2.0 * np.pi * np.cumsum(41.2 + 100.0 * np.exp(-t / 0.015)) / sr
        sig = 0.25 * np.sin(phase)
        _, res = _run_norm(_track(sig, sr, "808.wav"))
        gains.append(res.gain_db)
    assert max(gains) - min(gains) <= 0.1


# ── M3: Isolated Plosive Vocal + ±100 ms Next-Peak ───────────────────────────

@pytest.mark.parametrize("sr", RATES)
def test_m3_isolated_plosive_vocal_and_next_peak(sr):
    t = np.arange(int(3.0 * sr), dtype=np.float64) / sr
    # Sustained vocal bed around -22 dBFS RMS (amp ~ -19 dBFS peak)
    bed = 10.0 ** (-19.0 / 20.0) * np.sin(2.0 * np.pi * 240.0 * t)
    # Single mid-hop 8 ms plosive at 1.423 s peaking at -4.0 dBFS
    p0 = int(1.423 * sr)
    plen = int(0.008 * sr)
    plosive = 10.0 ** (-4.0 / 20.0) * np.sin(np.pi * np.arange(plen) / plen) * np.sin(2.0 * np.pi * 80.0 * np.arange(plen) / sr)
    sig = bed.copy()
    sig[p0:p0 + plen] = plosive

    proc, res = _run_norm(_track(sig, sr, "LeadVocal.wav"))
    assert res.classification == "Auto (Peak)"
    assert res.data["governing"] == "Peak"
    rendered = _apply_norm(proc, sig, sr, res)
    assert 20.0 * math.log10(float(np.max(np.abs(rendered)))) == pytest.approx(-6.0, abs=0.05)
    assert res.data["next_peak_db"] is not None
    assert res.data["detected_peak_db"] - res.data["next_peak_db"] >= 6.0


@pytest.mark.parametrize("sr", RATES)
def test_m3_next_peak_edge_cases(sr):
    # 150 BPM 16th notes = 100 ms spacing -> at 105 ms spacing, next peak equals governing peak (0.0 dB delta)
    sig = np.zeros(int(2.0 * sr), dtype=np.float64)
    for k in range(1, 15):
        sig[int(k * 0.105 * sr)] = 0.5
    info = find_governing_and_next_peak(sig, sr, exclusion_ms=100.0)
    assert info["next_peak_db"] is not None
    assert abs(info["peak_db"] - info["next_peak_db"]) < 1e-6

    # Exact radius boundary: sample at peak_idx + radius is masked, peak_idx + radius + 1 is reported
    radius = max(1, round(100.0 * sr / 1000.0))
    bnd = np.zeros(3 * radius + 10, dtype=np.float64)
    k0 = radius + 2
    bnd[k0] = 0.8
    bnd[k0 + radius] = 0.6  # inside inclusive ±radius exclusion -> masked
    bnd[k0 + radius + 1] = 0.4  # first sample outside exclusion -> reported
    binfo = find_governing_and_next_peak(bnd, sr, exclusion_ms=100.0)
    assert binfo["peak_sample"] == k0
    assert binfo["next_peak_sample"] == k0 + radius + 1
    assert binfo["next_peak_db"] == pytest.approx(20.0 * math.log10(0.4), abs=1e-9)

    # Stereo-linked masking: L peak at k masks R peak at k+10 ms, reports R peak at k+150 ms
    st = np.zeros((sr, 2), dtype=np.float64)
    st[sr // 4, 0] = 0.9
    st[sr // 4 + int(0.010 * sr), 1] = 0.7  # masked across channels
    st[sr // 4 + int(0.150 * sr), 1] = 0.3  # distinct next peak
    st_info = find_governing_and_next_peak(st, sr, exclusion_ms=100.0)
    assert st_info["next_peak_db"] == pytest.approx(20.0 * math.log10(0.3), abs=1e-9)

    # Two clipped 0.0 dBFS hits separated by > 100 ms -> next_peak_db == 0.0 dBFS
    clipped = np.zeros(sr, dtype=np.float64)
    clipped[sr // 4] = 1.0
    clipped[sr // 2] = -1.0
    cinfo = find_governing_and_next_peak(clipped, sr, exclusion_ms=100.0)
    assert cinfo["peak_db"] == pytest.approx(0.0, abs=1e-9)
    assert cinfo["next_peak_db"] == pytest.approx(0.0, abs=1e-9)

    # Centered peak in <= 200 ms clip -> no unmasked samples -> next_peak_db is None
    short = np.zeros(int(0.18 * sr), dtype=np.float64)
    short[len(short) // 2] = 0.5
    sinfo = find_governing_and_next_peak(short, sr, exclusion_ms=100.0)
    assert sinfo["next_peak_db"] is None


# ── M4: Noise-Floor Wind-Up & max_gain_db (Test 16) ──────────────────────────

def test_m4_noise_floor_wind_up_and_max_gain_cap():
    sr = 48000
    rng = np.random.default_rng(7)
    n = 2 * sr

    # -75 dBFS noise -> Noise Floor hold (0.0 dB)
    quiet_noise = rng.uniform(-1.0, 1.0, n)
    quiet_noise *= 10.0 ** (-75.0 / 20.0) / float(np.max(np.abs(quiet_noise)))
    _, r_nf = _run_norm(_track(quiet_noise, sr, "NoiseFloor.wav"))
    assert r_nf.classification == "Noise Floor" and r_nf.gain_db == 0.0

    # 16-bit & 24-bit TPDF dither -> held at 0.0 dB
    for bits in (16, 24):
        lsb = 2.0 ** -(bits - 1)
        dither = (rng.uniform(-0.5, 0.5, n) + rng.uniform(-0.5, 0.5, n)) * lsb
        _, r_d = _run_norm(_track(dither, sr, f"Dither{bits}.wav"))
        assert r_d.classification == "Noise Floor" and r_d.gain_db == 0.0

    # -55 dBFS steady room mic -> clamped to max_gain_db (+24.0 dB)
    room = 10.0 ** (-55.0 / 20.0) * np.sin(2.0 * np.pi * 200.0 * np.arange(n) / sr)
    _, r_rm = _run_norm(_track(room, sr, "RoomMic.wav"))
    assert r_rm.gain_db == pytest.approx(24.0, abs=1e-9)
    assert r_rm.data["boost_clamped"] is True

    # -65 dBFS bleed + one -20 dBFS tick -> peak-capped at +14.0 dB (finite, no wind-up)
    bleed = 10.0 ** (-65.0 / 20.0) * np.sin(2.0 * np.pi * 300.0 * np.arange(n) / sr)
    bleed[sr] = 10.0 ** (-20.0 / 20.0)
    _, r_bl = _run_norm(_track(bleed, sr, "BleedTick.wav"))
    assert r_bl.data["governing"] == "Peak"
    assert r_bl.gain_db == pytest.approx(14.0, abs=0.05)

    # Exact silence -> 0.0 dB Silent
    _, r_sil = _run_norm(_track(np.zeros(n), sr, "Silent.wav"))
    assert r_sil.classification == "Silent" and r_sil.gain_db == 0.0

    # -70.01 dBFS vs -69.99 dBFS Noise Floor Hold boundary
    below = 10.0 ** (-70.01 / 20.0) * np.sin(2.0 * np.pi * 200.0 * np.arange(n) / sr)
    above = 10.0 ** (-69.99 / 20.0) * np.sin(2.0 * np.pi * 200.0 * np.arange(n) / sr)
    _, r_below = _run_norm(_track(below, sr, "BelowFloor.wav"))
    _, r_above = _run_norm(_track(above, sr, "AboveFloor.wav"))
    assert r_below.classification == "Noise Floor" and r_below.gain_db == 0.0
    assert r_above.classification != "Noise Floor" and r_above.gain_db == pytest.approx(24.0, abs=1e-6)

    # max_gain_db=0.0 (attenuate-only) and 36.0 + Peak/RMS override clamping
    _, r_att = _run_norm(_track(room, sr, "AttOnly.wav"), max_gain_db=0.0)
    assert r_att.gain_db == 0.0 and r_att.data["boost_clamped"] is True
    _, r_36 = _run_norm(_track(room, sr, "Cap36.wav"), max_gain_db=36.0)
    assert r_36.gain_db == pytest.approx(36.0, abs=1e-9) and r_36.data["boost_clamped"] is True
    for ov in ("Peak", "RMS"):
        tr_ov = _track(room, sr, f"Cap_{ov}.wav")
        tr_ov.classification_override = ov
        _, r_ov = _run_norm(tr_ov, max_gain_db=12.0)
        assert r_ov.gain_db == pytest.approx(12.0, abs=1e-9) and r_ov.data["boost_clamped"] is True

    # True 1/f pink noise at -75 dBFS peak -> Noise Floor Hold
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.fft.rfft(white) / np.maximum(np.sqrt(freqs), 1.0)
    spec[0] = 0.0
    pink = np.fft.irfft(spec, n)
    pink *= 10.0 ** (-75.0 / 20.0) / float(np.max(np.abs(pink)))
    _, r_pk_nf = _run_norm(_track(pink, sr, "PinkFloor.wav"))
    assert r_pk_nf.classification == "Noise Floor" and r_pk_nf.gain_db == 0.0


# ── M5: Panned Stereo & L<->R Swap (Test 10) ─────────────────────────────────

@pytest.mark.parametrize("stereo_mode", ["avg", "sum"])
def test_m5_panned_stereo_and_lr_swap_symmetry(stereo_mode):
    sr = 48000
    t = np.arange(2 * sr, dtype=np.float64) / sr
    tone = 10.0 ** (-18.0 / 20.0) * math.sqrt(2.0) * np.sin(2.0 * np.pi * 440.0 * t)
    # Hard-panned L=-18 dBFS RMS, R=0
    hard_lr = np.column_stack((tone, np.zeros_like(tone)))
    hard_rl = hard_lr[:, ::-1].copy()
    p1, r1 = _run_norm(_track(hard_lr, sr), stereo_mode=stereo_mode)
    p2, r2 = _run_norm(_track(hard_rl, sr), stereo_mode=stereo_mode)
    expected_rms = -18.0 - (10.0 * math.log10(2.0) if stereo_mode == "avg" else 0.0)
    assert r1.data["detected_rms_db"] == pytest.approx(expected_rms, abs=0.1)
    assert abs(r1.gain_db - r2.gain_db) <= 1e-12
    assert abs(r1.data["detected_peak_db"] - r2.data["detected_peak_db"]) <= 1e-12

    # 12 dB imbalanced stereo L<->R swap invariance (< -120 dBFS diff)
    imb_lr = np.column_stack((tone, 0.25 * tone))
    imb_rl = imb_lr[:, ::-1].copy()
    pi1, ri1 = _run_norm(_track(imb_lr, sr), stereo_mode=stereo_mode)
    pi2, ri2 = _run_norm(_track(imb_rl, sr), stereo_mode=stereo_mode)
    assert abs(ri1.gain_db - ri2.gain_db) <= 1e-12
    out1 = _apply_norm(pi1, imb_lr, sr, ri1)
    out2 = _apply_norm(pi2, imb_rl, sr, ri2)[:, ::-1]
    max_diff = float(np.max(np.abs(out1 - out2)))
    assert max_diff < 1e-6  # < -120 dBFS


# ── M6: Exact Tie-Break & Target Spread ──────────────────────────────────────

@pytest.mark.parametrize(("t_peak", "t_rms", "kink"), [(-6.0, -18.0, 12.0), (-3.0, -18.0, 15.0)])
@pytest.mark.parametrize("delta", [-0.01, 0.0, +0.01])
def test_m6_exact_tie_break_and_target_spread(t_peak, t_rms, kink, delta):
    sr = 48000
    n = 12 * sr  # > 10s so < 5% of 400 ms windows touch the single spike -> P95 stays exact
    rms_amp = 0.1
    sig = np.tile([rms_amp, -rms_amp], n // 2)
    crest_db = kink + delta
    sig[n // 2] = rms_amp * (10.0 ** (crest_db / 20.0))
    _, res = _run_norm(_track(sig, sr), target_peak=t_peak, target_rms=t_rms)
    if delta >= 0.0:
        assert res.data["governing"] == "Peak"
        assert res.gain_db == pytest.approx(t_peak - 20.0 * math.log10(sig[n // 2]), abs=0.01)
    else:
        assert res.data["governing"] == "RMS"
        assert res.gain_db == pytest.approx(t_rms - 20.0 * math.log10(rms_amp), abs=0.01)


def test_m6_inverted_targets_rejected_cleanly():
    flat_errs = validate_config_fields({**default_config(), "target_rms": -4.0, "target_peak": -6.0})
    assert any(e.key == "target_rms" for e in flat_errs)
    struct = build_structured_defaults()
    struct["processors"]["bimodal_normalize"]["target_rms"] = -4.0
    struct["processors"]["bimodal_normalize"]["target_peak"] = -6.0
    struct_errs = validate_structured_config(struct)
    assert any("target_rms" in e.key for e in struct_errs)


# ── M7: Manual RMS Override & Lazy BS.1770 dBTP Warnings ─────────────────────

@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
@pytest.mark.parametrize(("post_pk_target", "expected_warn", "expect_tp"), [
    (-6.5, "none", False),
    (-5.6, "none", False),
    (-5.4, "none", True),
    (-0.5, "amber", True),
    (+0.5, "red", True),
    (+4.0, "red", True),
])
def test_m7_manual_rms_override_and_lazy_true_peak_warnings(subtype, post_pk_target, expected_warn, expect_tp):
    sr = 48000
    n = 12 * sr  # P95 unaffected by isolated spike
    rms_amp = 10.0 ** (-24.0 / 20.0)
    sig = np.tile([rms_amp, -rms_amp], n // 2)
    sig[n // 2] = 10.0 ** ((post_pk_target - 6.0) / 20.0)
    tr = _track(sig, sr, "Vocal.wav", subtype=subtype)
    tr.classification_override = "RMS"
    _, res = _run_norm(tr)
    assert res.data["mode"] == "RMS"
    assert res.gain_db == pytest.approx(6.0, abs=0.01)
    assert res.data["peak_warning_level"] == expected_warn
    assert (res.data["post_true_peak_db"] is not None) is expect_tp


@pytest.mark.parametrize("sr", (*RATES, 176400))
def test_m7_measure_true_peak_calibrated_across_all_rates(sr):
    n = max(4096, sr // 2)
    t = np.arange(n, dtype=np.float64) / sr
    win = np.hanning(n)
    # 1 kHz windowed tone at -6.0 dBFS -> exact -6.0 dBTP within ±0.1 dB
    tone = 10.0 ** (-6.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t) * win
    assert measure_true_peak_db(tone, sr) == pytest.approx(-6.0, abs=0.1)
    # Stereo right-channel ISP at fs/4 @ 45° (applies where oversampling >= 2x, i.e. < 192 kHz)
    if sr < 192000:
        k = np.arange(4096, dtype=np.float64)
        burst = np.sin(0.5 * np.pi * k + np.pi / 4.0) * np.hanning(4096)
        burst *= 10.0 ** (-3.6 / 20.0) / float(np.max(np.abs(burst)))
        st = np.column_stack((np.zeros_like(burst), burst))
        tp = measure_true_peak_db(st, sr)
        assert tp == pytest.approx(-0.59, abs=0.15)


def test_m7_fs_over_4_inter_sample_peak_and_lazy_gate_bounds(tmp_path):
    from sessionpreplib.processors.bimodal_normalize import update_post_gain_metrics

    sr = 48000
    n = 12 * sr  # P95 unaffected by 20 ms burst (< 5% of windows)
    rms_amp = 10.0 ** (-24.0 / 20.0)
    sig = np.tile([rms_amp, -rms_amp], n // 2)
    burst_n = 960
    k = np.arange(burst_n, dtype=np.float64)
    # Peak sample amplitude = -7.2 dBFS pre-gain -> -1.2 dBFS post-gain (+6 dB), +1.81 dBTP interior
    amp_pre = 10.0 ** ((-1.2 - 6.0) / 20.0) * math.sqrt(2.0)
    sig[n // 2:n // 2 + burst_n] = amp_pre * np.sin(0.5 * np.pi * k + np.pi / 4.0) * np.hanning(burst_n)
    tr = _track(sig, sr, "ISP.wav", subtype="PCM_24")
    tr.classification_override = "RMS"
    proc, res = _run_norm(tr)
    assert res.data["post_peak_db"] == pytest.approx(-1.2, abs=0.1)
    assert res.data["post_true_peak_db"] == pytest.approx(1.81, abs=0.15)
    assert res.data["peak_warning_level"] == "red"
    # HTML must NOT falsely say "(-1.2 dBFS (>0))" or claim fixed-point PCM clip for sub-0 dBFS sample peak
    rendered_html = html.unescape(re.sub("<[^>]*>", " ", proc.render_html(res))).replace("−", "-")
    assert "-1.2 dBFS (>0)" not in rendered_html
    assert "Prepare to fixed-point PCM would clip" not in rendered_html
    assert "+1.8 dBTP" in rendered_html

    # Lazy gate at -5.5 dBFS catches fs/4 @ 45° (-3.6 dBFS -> Amber) and [-1,+1,+1,-1] (-3.6 dBFS -> Red)
    s_fs4 = np.sin(0.5 * np.pi * np.arange(4096) + np.pi / 4.0)
    s_fs4 *= 10.0 ** (-3.6 / 20.0) / float(np.max(np.abs(s_fs4)))
    tr_a = _track(s_fs4, sr, "FS4.wav")
    _, res_a = _run_norm(tr_a)
    update_post_gain_metrics(res_a, tr_a, 0.0)
    assert res_a.data["post_true_peak_db"] is not None and res_a.data["peak_warning_level"] == "amber"

    s_tr = np.zeros(4096, dtype=np.float64)
    s_tr[2048:2052] = np.array([-1.0, 1.0, 1.0, -1.0]) * (10.0 ** (-3.6 / 20.0))
    tr_r = _track(s_tr, sr, "Trans.wav")
    _, res_r = _run_norm(tr_r)
    update_post_gain_metrics(res_r, tr_r, 0.0)
    assert res_r.data["post_true_peak_db"] is not None and res_r.data["peak_warning_level"] == "red"

    # Pipeline.execute() rejects PCM_24 clip (> 0 dBFS) via OutputHeadroomError while allowing FLOAT
    for subtype, expect_err in (("PCM_24", True), ("FLOAT", False)):
        wav = tmp_path / f"over_{subtype}.wav"
        over_sig = np.tile([rms_amp, -rms_amp], (sr * 6) // 2)
        over_sig[sr * 3] = 10.0 ** ((-4.0) / 20.0)  # +6 dB RMS gain -> +2.0 dBFS peak
        sf.write(wav, over_sig, sr, subtype=subtype)
        t_io = load_track(str(wav))
        t_io.classification_override = "RMS"
        pipe = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()], default_config(), max_workers=1)
        sess = SessionContext([t_io], default_config())
        pipe.analyze_phase1(sess)
        pipe.analyze_phase2(sess)
        pipe.plan(sess)
        pipe.execute(sess, str(tmp_path / f"out_{subtype}"))
        assert bool(sess.execution_receipts.execute_errors) is expect_err


# ── M8: Linked Groups (Auto & Overrides, Test 10) ────────────────────────────

@pytest.mark.parametrize("override_a,override_b", [(None, None), ("RMS", None), (None, "RMS")])
def test_m8_linked_group_auto_and_overrides(override_a, override_b):
    sr = 48000
    n = 2 * sr
    # Track A: crest ~6 dB (RMS -20, peak -14 -> Auto gain +2.0 dB)
    sig_a = np.tile([0.1, -0.1], n // 2)
    sig_a[sr] = 0.2
    # Track B: crest ~18 dB (RMS -26, peak -8 -> Auto peak-capped gain +2.0 or lower: peak -4 -> gain -2.0 dB)
    sig_b = np.tile([0.05, -0.05], n // 2)
    sig_b[sr] = 10.0 ** (-4.0 / 20.0)  # peak -4 dBFS -> peak gain = -2.0 dB, RMS gain = +8.0 dB
    tr_a = _track(sig_a, sr, "KickIn.wav")
    tr_b = _track(sig_b, sr, "KickOut.wav")
    tr_c = _track(sig_a, sr, "SkipTrack.wav")
    tr_d = _track(np.zeros(n), sr, "SilentTrack.wav")
    tr_e = _track(10.0 ** (-75.0 / 20.0) * np.ones(n), sr, "NoiseFloorTrack.wav")
    for t in (tr_a, tr_b, tr_c, tr_d, tr_e):
        t.group = "Kick"
    tr_a.classification_override = override_a
    tr_b.classification_override = override_b
    tr_c.classification_override = "Skip"

    proc = BimodalNormalizeProcessor()
    proc.configure(default_config())
    session = SessionContext([tr_a, tr_b, tr_c, tr_d, tr_e], default_config(), processors=[proc])
    for t in session.tracks:
        t.processor_results[proc.id] = proc.process(t)
    apply_group_gains(session, linked_groups={"Kick"}, headroom_db=0.0, anchor=None)

    ra, rb, rc, rd, re_nf = [t.processor_results[proc.id] for t in session.tracks]
    assert abs(ra.gain_db - rb.gain_db) == 0.0
    assert rc.gain_db == 0.0 and rd.gain_db == 0.0 and re_nf.gain_db == 0.0
    # Analytic expected group gain: KickIn RMS=-20 -> +2.0 dB; KickOut Auto Peak=-4 -> -2.0 dB, RMS override -> +8.0 dB
    expected_gain = -2.0 if override_b is None else 2.0
    assert ra.gain_db == pytest.approx(expected_gain, abs=0.05)
    assert rb.data["post_peak_db"] == pytest.approx(-4.0 + expected_gain, abs=0.05)
    if override_b is None:
        assert ra.data["group_limited"] is True
        assert ra.data["rms_anchor_active"] is False
        assert ra.data["group_governing_track"] == "KickOut.wav"
    else:
        assert rb.data["group_limited"] is True
        assert rb.data["rms_anchor_active"] is False
        assert rb.data["group_governing_track"] == "KickIn.wav"
        # KickOut RMS override individually (+8 dB -> +4 dBFS red) is pulled to +2 dB -> -2 dBFS (warning cleared!)
        assert rb.data["peak_warning_level"] == "none"


# ── M9: Recall, Determinism & Legacy Migration (Test 12) ─────────────────────

def test_m9_recall_determinism_and_legacy_migration(tmp_path):
    sr = 48000
    rng = np.random.default_rng(2026)
    n = 3 * sr
    t = np.arange(n, dtype=np.float64) / sr
    # Multi-segment deterministic fixture: seeded noise + tones + transients
    sig = 0.08 * np.sin(2.0 * np.pi * 220.0 * t) + 0.05 * np.sin(2.0 * np.pi * 880.0 * t)
    sig += 0.03 * rng.standard_normal(n)
    sig[sr] = 0.65
    sig[2 * sr] = -0.45
    # Independent analytic golden reference render (Test 12: < -100 dBFS against pinned reference)
    exp_pk, exp_rms = _analytic_peak_rms_db(sig, sr)
    exp_gain_db = min(-6.0 - exp_pk, -18.0 - exp_rms)
    golden_ref = sig * (10.0 ** (exp_gain_db / 20.0))

    p_wav = tmp_path / "Golden.wav"
    sf.write(p_wav, sig, sr, subtype="DOUBLE")

    outputs = []
    for workers in (1, 4):
        tr = load_track(str(p_wav))
        pipe = Pipeline([SilenceDetector(), DCOffsetDetector()], [BimodalNormalizeProcessor()], default_config(), max_workers=workers)
        sess = SessionContext([tr], default_config())
        pipe.analyze_phase1(sess)
        pipe.analyze_phase2(sess)
        pipe.plan(sess)
        out_dir = tmp_path / f"w{workers}"
        pipe.execute(sess, str(out_dir))
        y, _ = sf.read(out_dir / p_wav.name)
        outputs.append(y)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert float(np.max(np.abs(outputs[0] - golden_ref))) < 1e-5  # < -100 dBFS vs golden reference

    # Legacy override & config migration
    assert migrate_legacy_override("Transient") == "Peak"
    assert migrate_legacy_override("Sustained") is None
    assert migrate_legacy_override("Skip") == "Skip"
    legacy_cfg = {
        "detectors": {"audio_classifier": {"crest_threshold": 14.0, "decay_db_threshold": 10.0}},
        "processors": {"bimodal_normalize": {"force_transient": ["kick"], "force_sustained": ["pad"]}},
    }
    cleaned, notes = migrate_legacy_config(legacy_cfg)
    assert "audio_classifier" not in cleaned.get("detectors", {})
    assert "force_sustained" not in cleaned["processors"]["bimodal_normalize"]
    assert cleaned["processors"]["bimodal_normalize"]["force_peak"] == "kick"
    assert notes

    # dbfs_convention toggle does NOT invalidate RepairVerification evidence keys (neither settings nor layout)
    v_std = RepairVerification([SilenceDetector(), DCOffsetDetector()], {**default_config(), "dbfs_convention": "standard"})
    v_aes = RepairVerification([SilenceDetector(), DCOffsetDetector()], {**default_config(), "dbfs_convention": "aes17"})
    assert v_std.settings_key == v_aes.settings_key
    assert v_std.layout_settings_key == v_aes.layout_settings_key


# ── M10: Meter Truth, AES17 & cumsum Precision (Tests 6 & 15) ────────────────

@pytest.mark.parametrize("sr", RATES)
def test_m10_meter_truth_window_scaling_and_aes17(sr):
    # Window scaling + step-response timing within ±1 sample across 4 rates
    ws = round(0.4 * sr)
    step_sig = np.zeros(int(1.2 * sr), dtype=np.float64)
    step_sig[ws:] = 0.5
    tr_step = _track(step_sig, sr)
    assert abs(get_window_samples(tr_step, 400) - ws) <= 1
    wm_step = get_rms_window_means(tr_step, 400, "avg")
    # First window reaching full 0.25 mean square must sit at exact ws ± 1 sample
    first_full = int(np.flatnonzero(wm_step >= 0.25 - 1e-12)[0])
    assert abs(first_full - ws) <= 1

    # Short 10 ms clip records exact effective_window_ms = 10.0 ms
    short_tr = _track(0.2 * np.ones(round(0.010 * sr)), sr, "Short10ms.wav")
    _, r_short = _run_norm(short_tr)
    assert r_short.data["effective_window_ms"] == pytest.approx(10.0, abs=0.1)

    # Calibrated 5 frequencies (25, 100, 1000, 10000, 18000 Hz) + white & pink noise within ±0.1 dB
    t = np.arange(int(1.2 * sr), dtype=np.float64) / sr
    for hz in (25.0, 100.0, 1000.0, 10000.0, 18000.0):
        sig = 0.25 * np.sin(2.0 * np.pi * hz * t)
        _, res = _run_norm(_track(sig, sr, f"Sine_{int(hz)}.wav"))
        assert res.data["detected_rms_db"] == pytest.approx(20.0 * math.log10(0.25 / math.sqrt(2.0)), abs=0.1)

    rng = np.random.default_rng(10)
    white = rng.standard_normal(int(1.2 * sr))
    white *= 10.0 ** (-18.0 / 20.0) / math.sqrt(float(np.mean(white ** 2)))
    _, r_w = _run_norm(_track(white, sr, "White.wav"))
    assert r_w.data["detected_rms_db"] == pytest.approx(-18.0, abs=0.15)

    # AES17 exact 10*log10(2) shifts ONLY RMS (+3.0103 dB), NOT Peak, Next-Peak, dBTP, or DC
    sig = 0.25 * np.sin(2.0 * np.pi * 1000.0 * t)
    tr = _track(sig, sr, "CalSine.wav")
    p_std, r_std = _run_norm(tr, dbfs_convention="standard")
    p_aes, r_aes = _run_norm(tr, dbfs_convention="aes17")
    assert r_std.gain_db == pytest.approx(r_aes.gain_db, abs=1e-12)
    assert AES17_OFFSET == pytest.approx(10.0 * math.log10(2.0), abs=1e-12)
    html_std = html.unescape(re.sub("<[^>]*>", " ", p_std.render_html(r_std))).replace("−", "-")
    html_aes = html.unescape(re.sub("<[^>]*>", " ", p_aes.render_html(r_aes))).replace("−", "-")
    assert "-12.0 dBFS" in html_std and "-12.0 dBFS" in html_aes
    assert "-18.0 dBFS" in html_std and "-15.0 dBFS" in html_aes


def test_m10_block_window_sums_eliminates_cumsum_tail_cancellation():
    # 3 minutes of 0 dBFS full-scale followed by a -100 dBFS quiet tail at 48 kHz
    sr = 48000
    n_loud = 180 * sr
    n_tail = 2 * sr
    sig = np.empty(n_loud + n_tail, dtype=np.float64)
    sig[:n_loud] = 1.0
    sig[n_loud:] = 10.0 ** (-100.0 / 20.0)
    tr = _track(sig, sr, "LongTail.wav")
    wm = get_rms_window_means(tr, 400, "avg")
    tail_db = 10.0 * math.log10(float(wm[-1]))
    assert tail_db == pytest.approx(-100.0, abs=0.01)


def test_m10_dc_offset_aes17_invariance():
    from sessionpreplib.summary import build_diagnostic_summary

    sr = 48000
    dc_sig = np.full(sr, 0.5, dtype=np.float64)  # exact -6.0206 dBFS
    tr = _track(dc_sig, sr, "dc.wav")
    det = DCOffsetDetector()
    cfg = {**default_config(), "dbfs_convention": "aes17", "dc_offset_warn_db": -40.0}
    det.configure(cfg)
    res = det.analyze(tr)
    tr.detector_results["dc_offset"] = res
    assert "-6.0 dBFS" in res.summary
    assert "-3.0 dBFS" not in res.summary
    summ = build_diagnostic_summary(SessionContext([tr], cfg, detectors=[det]))
    dc_items = [it for grp in summ["attention"] for it in grp.get("items", []) if "DC offset" in it]
    assert dc_items and "-6.0 dBFS" in dc_items[0] and "-3.0 dBFS" not in dc_items[0]


def test_m10_gated_rms_all_zero_returns_neg_inf():
    tr = _track(np.zeros(48000), 48000, "Zero.wav")
    gated = get_gated_rms_data(tr, 400, "avg", 40.0)
    assert gated["max_window_db"] == float("-inf")
    assert gated["active_means"].size == 0


# ── M11: Bad Input, Denormals & Float Range (Tests 7 & 16) ───────────────────

@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_m11_non_finite_input_quarantined_and_json_clean(tmp_path, bad_val):
    sr = 48000
    sig = 0.2 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr) / sr)
    sig[100] = bad_val
    tr = _track(sig, sr, "Bad.wav")
    proc, res = _run_norm(tr)
    assert res.classification == "Error" and res.gain_db == 0.0
    assert res.error is not None
    tr.processor_results[proc.id] = res
    sess = SessionContext([tr], default_config(), processors=[proc])
    json_path = tmp_path / "out.json"
    save_json(sess, default_config(), str(json_path), [])
    # Strict JSON parser fails if NaN or Infinity leaked
    json.loads(json_path.read_text(encoding="utf-8"), parse_constant=lambda s: pytest.fail(f"Non-finite JSON: {s}"))


def test_m11_subnormals_flushed_and_float_range_linear():
    from sessionpreplib.models import ProcessorResult

    sr = 48000
    proc = BimodalNormalizeProcessor()
    proc.configure(default_config())
    sub = np.array([1e-310, -1e-310, 1e-200, 0.25], dtype=np.float64)
    out = _apply_norm(proc, sub, sr, ProcessorResult(proc.id, -6.0, "Auto (RMS)", ""))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 0.0  # subnormals flushed to exact 0.0

    # +24 dBFS and -120 dBFS float sines scale linearly without NaN/Inf; overflow raises ValueError
    t = np.arange(2 * sr, dtype=np.float64) / sr
    hot = 10.0 ** (+24.0 / 20.0) * math.sqrt(2.0) * np.sin(2.0 * np.pi * 440.0 * t)
    _, r_hot = _run_norm(_track(hot, sr, "HotFloat.wav"))
    y_hot = _apply_norm(proc, hot, sr, r_hot)
    _, post_rms = _analytic_peak_rms_db(y_hot, sr)
    assert post_rms == pytest.approx(-18.0, abs=0.1)
    assert abs(float(np.mean(y_hot))) < 1e-3  # DC mean < -60 dBFS

    # -120 dBFS float sine held at 0 dB by Noise Floor Hold, and linear scaling under explicit apply()
    quiet_120 = 10.0 ** (-120.0 / 20.0) * np.sin(2.0 * np.pi * 440.0 * t)
    _, r_120 = _run_norm(_track(quiet_120, sr, "Quiet120.wav"))
    assert r_120.classification == "Noise Floor" and r_120.gain_db == 0.0
    y_120 = _apply_norm(proc, quiet_120, sr, ProcessorResult(proc.id, +24.0, "Auto (RMS)", ""))
    assert np.all(np.isfinite(y_120)) and 20.0 * math.log10(float(np.max(np.abs(y_120)))) == pytest.approx(-96.0, abs=0.05)

    # Zero-mean white noise DC stays < -60 dBFS after normalization
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(sr)
    noise -= np.mean(noise)
    noise *= 0.2
    _, r_nz = _run_norm(_track(noise, sr, "ZeroMeanNoise.wav"))
    y_nz = _apply_norm(proc, noise, sr, r_nz)
    assert abs(float(np.mean(y_nz))) < 1e-3

    # Overflowing gain raises ValueError (not OverflowError), non-finite gain raises ValueError
    with pytest.raises(ValueError):
        _apply_norm(proc, hot[:100], sr, ProcessorResult(proc.id, 7000.0, "Auto (RMS)", ""))
    with pytest.raises(ValueError):
        _apply_norm(proc, hot[:100], sr, ProcessorResult(proc.id, float("nan"), "Auto (RMS)", ""))

    # get_gated_rms_data raises ValueError on NaN rather than silently dropping NaN windows
    bad = np.full(1000, 0.5, dtype=np.float64)
    bad[10] = float("nan")
    with pytest.raises(ValueError):
        get_gated_rms_data(_track(bad, sr), 400, "avg", 40.0)


# ── M12: Neutral, Bypass, Zero Latency & Ceiling Stress (Tests 8, 9, 13, 14) ─

@pytest.mark.parametrize("sr", RATES)
def test_m12_neutral_bypass_zero_latency(sr):
    from sessionpreplib.models import ProcessorResult

    rng = np.random.default_rng(120)
    t = np.arange(sr, dtype=np.float64) / sr
    sine = 10.0 ** (-6.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    noise = rng.standard_normal(sr)
    noise *= 10.0 ** (-18.0 / 20.0) / math.sqrt(float(np.mean(noise ** 2)))
    proc = BimodalNormalizeProcessor()
    proc.configure(default_config())
    for sig in (sine, noise):
        # Neutral 0.0 dB is bit-exact
        np.testing.assert_array_equal(_apply_norm(proc, sig, sr, ProcessorResult(proc.id, 0.0, "Auto (RMS)", "")), sig)
        # Bypass (Skip) returns bit-exact input
        tr_skip = _track(sig, sr)
        tr_skip.classification_override = "Skip"
        r_skip = proc.process(tr_skip)
        assert r_skip.gain_db == 0.0
        np.testing.assert_array_equal(_apply_norm(proc, sig, sr, r_skip), sig)
    # Zero latency: Dirac impulse at k stays at exact k with positive polarity & exact 0.0 off-impulse
    imp = np.zeros(sr, dtype=np.float64)
    k = sr // 3
    imp[k] = 0.5
    _, r_imp = _run_norm(_track(imp, sr))
    y_imp = _apply_norm(proc, imp, sr, r_imp)
    assert int(np.argmax(np.abs(y_imp))) == k
    assert y_imp[k] > 0.0
    mask = np.ones(sr, dtype=bool)
    mask[k] = False
    assert np.all(y_imp[mask] == 0.0)


@pytest.mark.parametrize("sr", RATES)
@pytest.mark.parametrize("kind", ["fs_div_4", "square", "white", "burst_60hz"])
def test_m12_ceiling_stress_never_exceeds_minus_5_9_dbfs(sr, kind):
    n = sr
    rng = np.random.default_rng(12)
    amp = 10.0 ** (+6.0 / 20.0)
    if kind == "fs_div_4":
        sig = amp * np.sin(0.5 * np.pi * np.arange(n) + np.pi / 4.0)
    elif kind == "square":
        sig = amp * np.sign(np.sin(2.0 * np.pi * 1000.0 * np.arange(n) / sr))
    elif kind == "white":
        sig = amp * rng.uniform(-1.0, 1.0, n)
    else:
        sig = np.zeros(n, dtype=np.float64)
        L = int(0.05 * sr)
        sig[n // 2:n // 2 + L] = amp * np.sin(2.0 * np.pi * 60.0 * np.arange(L) / sr)
    proc, res = _run_norm(_track(sig, sr, f"{kind}.wav"))
    out = _apply_norm(proc, sig, sr, res)
    post_pk = 20.0 * math.log10(float(np.max(np.abs(out))))
    assert post_pk <= -5.9


# ── M13: Bus Gain-Staging Regression Suite ───────────────────────────────────

def test_m13_musical_bus_staging_peak_capped_rms():
    from types import SimpleNamespace

    sr = 48000
    t = np.arange(2 * sr, dtype=np.float64) / sr
    tc = SlotATrimInfo("trackcontrol", "TrackControl", "DMGA/TrCt/RTST", "com.dmgaudio.TrackControl.aax", 0.0, -200.0, 18.0, False, False, True)

    def _mk(idx, name, ttype, in_bus="", out_bus="", trim=None):
        return PtxTrack(
            index=idx, name=name, uuid_hex=f"{idx:032x}", uuid_braced=f"{{{idx:032x}}}",
            track_type=ttype, format="Stereo", is_folder=(ttype == "RoutingFolder"),
            is_inactive=False, is_muted=False, pan_left=-100, pan_right=100, volume_db=0.0,
            input_path=PtxSignalPathRef(in_bus, "bus", f"i{idx}") if in_bus else None,
            main_output=PtxSignalPathRef(out_bus, "bus", f"o{idx}") if out_bus else None,
            sends=[], slot_a_trim=trim,
        )

    tracks = [
        _mk(0, "Kick", "Audio", out_bus="DRM"),
        _mk(1, "Bass", "Audio", out_bus="BAS"),
        _mk(2, "cDRM", "Aux", in_bus="DRM", out_bus="MIX", trim=tc),
        _mk(3, "cBAS", "Aux", in_bus="BAS", out_bus="MIX", trim=tc),
        _mk(4, "MIX", "Aux", in_bus="MIX", out_bus="Out 1-2"),
        _mk(5, "Master", "Master", out_bus="Out 1-2"),
    ]
    ptx = SimpleNamespace(
        info=SimpleNamespace(sample_rate=sr), tracks=tracks, groups=[],
        track_by_name=lambda n: next((x for x in tracks if x.name.lower() == n.lower()), None),
        group_by_name=lambda _n: None,
    )
    # Spiky drum stem (crest > 15 dB) -> cDRM governed by Peak
    kick = np.zeros((2 * sr, 2), dtype=np.float64)
    kick[sr // 4:sr // 4 + 400, :] = 0.9
    # Sustained 808 bass stem (crest ~3 dB) -> cBAS governed by RMS
    bass = np.column_stack((0.35 * np.sin(2 * np.pi * 55 * t), 0.35 * np.sin(2 * np.pi * 55 * t)))
    g = build_annotated_signal_flow(
        ptx, {"Kick": LeafAudioInput(kick, sr), "Bass": LeafAudioInput(bass, sr)},
        {"target_peak": -6.0, "target_rms": -18.0},
        stage_bus_faders=True, attenuate_only=True,
    )
    assert g.nodes["cDRM"].governing == "Peak"
    assert g.nodes["cBAS"].governing == "RMS"
    assert g.nodes["cDRM"].pre_fader_peak_db <= -5.9
    assert g.nodes["cBAS"].pre_fader_rms_db == pytest.approx(-18.0, abs=0.1)
    assert g.nodes["Master"].governing is None
    assert "File-level balance verified" in g.balance_validation["summary"]
    assert measure_true_peak_db(bass, sr) == pytest.approx(20.0 * math.log10(0.35), abs=0.1)


# ── Post-Review DSP Regressions (Passes 1, 2, 3) ────────────────────────────

def test_dsp_regression_tail_exceedance_mid_hop_bursts_and_is_relevant():
    from sessionpreplib.detectors import TailExceedanceDetector
    from sessionpreplib.models import DetectorResult, Severity

    sr = 48000
    # Case 1: window=20ms, hop=50ms, 25ms +12 dB burst strictly between 1.000s and 1.050s hops
    n1 = 10 * sr
    t1 = np.arange(n1, dtype=np.float64) / sr
    s1 = 10.0 ** (-20.0 / 20.0) * math.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * t1)
    s1[int(1.022 * sr):int(1.047 * sr)] *= 10.0 ** (12.0 / 20.0)
    det1 = TailExceedanceDetector()
    det1.configure({**default_config(), "window": 20, "tail_hop_ms": 50, "tail_min_exceed_db": 3.0})
    r1 = det1.analyze(_track(s1, sr))
    assert r1.severity == Severity.ATTENTION
    assert len(r1.data["tail_regions"]) >= 1
    assert r1.data["tail_summary"]["max_exceed_db"] == pytest.approx(12.0, abs=0.1)

    # Case 2: default window=400ms, hop=10ms, 400ms +3.02 dB burst starting mid-hop at 5.005s
    n2 = 20 * sr
    t2 = np.arange(n2, dtype=np.float64) / sr
    s2 = 10.0 ** (-20.0 / 20.0) * math.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * t2)
    s2[int(5.005 * sr):int(5.405 * sr)] *= 10.0 ** (3.02 / 20.0)
    det2 = TailExceedanceDetector()
    det2.configure({**default_config(), "window": 400, "tail_hop_ms": 10, "tail_min_exceed_db": 3.0})
    r2 = det2.analyze(_track(s2, sr))
    assert r2.severity == Severity.ATTENTION
    assert r2.data["tail_summary"]["max_exceed_db"] == pytest.approx(3.02, abs=0.05)

    # is_relevant matrix
    dummy = DetectorResult("tail_exceedance", Severity.ATTENTION, "1 region", {})
    tr = _track(s2[:sr], sr)
    _, pr_rms = _run_norm(tr)
    tr.processor_results["bimodal_normalize"] = pr_rms
    assert det2.is_relevant(dummy, tr) is True
    tr.processor_skip.add("bimodal_normalize")
    assert det2.is_relevant(dummy, tr) is False
    tr.processor_skip.clear()
    pr_rms.data["group_limited"] = True
    pr_rms.data["rms_anchor_active"] = False
    assert det2.is_relevant(dummy, tr) is False


def test_dsp_regression_repaired_track_true_peak_accuracy(tmp_path):
    from sessionpreplib.repairs import render_corrected

    # (a) 96 kHz ultrasonic-attenuating repair + manual_gain_db=+10 dB
    sr = 96000
    t = np.arange(sr, dtype=np.float64) / sr
    win = np.hanning(sr)
    sig_us = (0.25 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.70 * np.sin(2.0 * np.pi * 30000.0 * t)) * win
    p_us = tmp_path / "us.wav"
    sf.write(p_us, sig_us, sr, subtype="DOUBLE")
    tr_us = load_track(str(p_us))
    tr_us.repair_overrides = {"ultrasonic": True}
    pipe = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()], default_config(), max_workers=1)
    sess = SessionContext([tr_us], default_config())
    pipe.analyze_phase1(sess)
    pipe.analyze_phase2(sess)
    pipe.plan(sess)
    tr_us.manual_gain_db = 10.0
    apply_group_gains(sess, linked_groups=None, headroom_db=0.0, anchor=None)
    repaired_us = render_corrected(tr_us.audio_data, sr, tr_us.repair_recipe) * (10.0 ** (10.0 / 20.0))
    actual_tp = measure_true_peak_db(repaired_us, sr)
    reported_tp = tr_us.processor_results["bimodal_normalize"].data["post_true_peak_db"]
    assert reported_tp == pytest.approx(actual_tp, abs=0.1)

    # (b) 48 kHz 60 Hz square wave with low_cut_hz=40 Hz + manual_gain_db=+6 dB
    sr2 = 48000
    t2 = np.arange(sr2, dtype=np.float64) / sr2
    sq = 10.0 ** (-10.0 / 20.0) * np.sign(np.sin(2.0 * np.pi * 60.0 * t2)) * np.hanning(sr2)
    p_sq = tmp_path / "sq.wav"
    sf.write(p_sq, sq, sr2, subtype="DOUBLE")
    tr_sq = load_track(str(p_sq))
    tr_sq.repair_overrides = {"low_cut": True, "low_cut_hz": 40.0}
    sess2 = SessionContext([tr_sq], default_config())
    pipe.analyze_phase1(sess2)
    pipe.analyze_phase2(sess2)
    pipe.plan(sess2)
    tr_sq.manual_gain_db = 6.0
    apply_group_gains(sess2, linked_groups=None, headroom_db=0.0, anchor=None)
    repaired_sq = render_corrected(tr_sq.audio_data, sr2, tr_sq.repair_recipe) * (10.0 ** (6.0 / 20.0))
    actual_tp2 = measure_true_peak_db(repaired_sq, sr2)
    reported_tp2 = tr_sq.processor_results["bimodal_normalize"].data["post_true_peak_db"]
    assert reported_tp2 == pytest.approx(actual_tp2, abs=0.1)


def test_dsp_regression_unloaded_skip_roundtrip_and_isp_retention(tmp_path):
    sr = 48000
    sig = np.tile([10.0 ** (-24.0 / 20.0), -(10.0 ** (-24.0 / 20.0))], (12 * sr) // 2)
    k = np.arange(512, dtype=np.float64)
    # Pre-gain sample peak -7.2 dBFS -> +6 dB RMS gain pushes sample peak to -1.2 dBFS & true peak to +1.8 dBTP (red)
    sig[sr:sr + 512] = 10.0 ** (-7.2 / 20.0) * math.sqrt(2.0) * np.sin(0.5 * np.pi * k + np.pi / 4.0)
    wav = tmp_path / "isp_reload.wav"
    sf.write(wav, sig, sr, subtype="DOUBLE")
    tr = load_track(str(wav))
    proc = BimodalNormalizeProcessor()
    proc.configure(default_config())
    r_auto = proc.process(tr)
    tr.processor_results[proc.id] = r_auto

    # Unload PCM -> Skip -> Auto must succeed with exact recall
    tr.audio_data = None
    tr.classification_override = "Skip"
    tr.processor_results[proc.id] = proc.process(tr)
    tr.classification_override = None
    r_back = proc.process(tr)
    assert r_back.gain_db == pytest.approx(r_auto.gain_db, abs=1e-9)

    # RMS override -> Peak -> RMS with audio_data=None preserves post_true_peak_db and red warning
    tr.classification_override = "RMS"
    r_rms1 = proc.process(tr)
    tr.processor_results[proc.id] = r_rms1
    assert r_rms1.data["peak_warning_level"] == "red"
    assert r_rms1.data["post_true_peak_db"] is not None
    tr.classification_override = "Peak"
    tr.processor_results[proc.id] = proc.process(tr)
    tr.classification_override = "RMS"
    r_rms2 = proc.process(tr)
    assert r_rms2.data["peak_warning_level"] == "red"
    assert r_rms2.data["post_true_peak_db"] == pytest.approx(r_rms1.data["post_true_peak_db"], abs=0.1)


def test_dsp_regression_skip_and_noise_floor_ignore_manual_gain(tmp_path):
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    s_skip = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    s_nf = 10.0 ** (-80.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    p_skip, p_nf = tmp_path / "skip.wav", tmp_path / "nf.wav"
    sf.write(p_skip, s_skip, sr, subtype="DOUBLE")
    sf.write(p_nf, s_nf, sr, subtype="DOUBLE")

    tr_skip, tr_nf = load_track(str(p_skip)), load_track(str(p_nf))
    tr_skip.classification_override = "Skip"
    tr_skip.manual_gain_db = 6.0
    tr_nf.manual_gain_db = 6.0
    pipe = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()], default_config(), max_workers=1)
    sess = SessionContext([tr_skip, tr_nf], default_config())
    pipe.analyze_phase1(sess)
    pipe.analyze_phase2(sess)
    pipe.plan(sess)
    out_dir = tmp_path / "out"
    pipe.execute(sess, str(out_dir))
    y_skip, _ = sf.read(out_dir / "skip.wav")
    y_nf, _ = sf.read(out_dir / "nf.wav")
    np.testing.assert_array_equal(y_skip, s_skip)
    np.testing.assert_array_equal(y_nf, s_nf)
    assert sess.execution_receipts.executed_files["skip.wav"]["fader_db"] == 0.0
    assert sess.execution_receipts.executed_files["nf.wav"]["fader_db"] == 0.0

    # Non-finite manual_gain_db on Skip never leaks NaN into rendered audio
    tr_skip.manual_gain_db = float("nan")
    _, y_nan, _ = pipe._render_track(load_track(str(p_skip)) if tr_skip.audio_data is None else tr_skip)
    assert np.all(np.isfinite(y_nan))


def test_dsp_regression_html_overshoot_group_and_signal_flow_quarantine():
    from types import SimpleNamespace
    from sessionpreplib.ptx.signal_flow_analysis import compute_signal_flow_graph_layout

    sr = 48000
    # Dense sine under Peak override (+9.0 dB above target) must NOT say "At target"
    sig = 10.0 ** (-10.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(sr) / sr)
    tr_pk = _track(sig, sr, "DensePeak.wav")
    tr_pk.classification_override = "Peak"
    proc, r_pk = _run_norm(tr_pk)
    txt_pk = html.unescape(re.sub("<[^>]*>", " ", proc.render_html(r_pk))).replace("−", "-")
    assert "At target" not in txt_pk and "above target" in txt_pk

    # Group-limited track must NOT print "Governs" on its own RMS row
    tr_a = _track(10.0 ** (-19.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(sr) / sr), sr, "A.wav")
    tr_b = _track(10.0 ** (-4.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(sr) / sr), sr, "B.wav")
    tr_a.group = tr_b.group = "G"
    sess = SessionContext([tr_a, tr_b], default_config(), processors=[proc])
    tr_a.processor_results[proc.id] = proc.process(tr_a)
    tr_b.processor_results[proc.id] = proc.process(tr_b)
    apply_group_gains(sess, linked_groups={"G"}, headroom_db=0.0, anchor=None)
    txt_a = html.unescape(re.sub("<[^>]*>", " ", proc.render_html(tr_a.processor_results[proc.id]))).replace("−", "-")
    assert "Group-limited" in txt_a and "Governs" not in txt_a

    # Signal flow non-finite leaf -> validation status == "error" (NOT "verified") and visible in graph
    tc = SlotATrimInfo("trackcontrol", "TrackControl", "DMGA/TrCt/RTST", "com.dmgaudio.TrackControl.aax", 0.0, -200.0, 18.0, False, False, True)

    def _mk(idx, name, ttype, in_bus="", out_bus="", trim=None):
        return PtxTrack(
            index=idx, name=name, uuid_hex=f"{idx:032x}", uuid_braced=f"{{{idx:032x}}}",
            track_type=ttype, format="Stereo", is_folder=(ttype == "RoutingFolder"),
            is_inactive=False, is_muted=False, pan_left=-100, pan_right=100, volume_db=0.0,
            input_path=PtxSignalPathRef(in_bus, "bus", f"i{idx}") if in_bus else None,
            main_output=PtxSignalPathRef(out_bus, "bus", f"o{idx}") if out_bus else None,
            sends=[], slot_a_trim=trim,
        )

    tracks = [
        _mk(0, "Kick", "Audio", out_bus="MIX"),
        _mk(1, "Snare", "Audio", out_bus="MIX"),
        _mk(2, "MIX", "Aux", in_bus="MIX", out_bus="PRINT", trim=tc),
        _mk(3, "PRINT", "RoutingFolder", in_bus="PRINT", out_bus="Out 1-2"),
        _mk(4, "Master", "Master", out_bus="Out 1-2"),
    ]
    ptx = SimpleNamespace(
        info=SimpleNamespace(sample_rate=sr), tracks=tracks, groups=[],
        track_by_name=lambda n: next((x for x in tracks if x.name.lower() == n.lower()), None),
        group_by_name=lambda _n: None,
    )
    good = np.full((sr // 4, 2), 0.2, dtype=np.float64)
    bad = good.copy()
    bad[10, 0] = float("nan")
    g = build_annotated_signal_flow(
        ptx, {"Kick": LeafAudioInput(good, sr), "Snare": LeafAudioInput(bad, sr)},
        {"target_peak": -6.0, "target_rms": -18.0}, stage_bus_faders=True, attenuate_only=True,
    )
    assert g.balance_validation["status"] == "error"
    assert g.balance_validation["converged"] is False
    assert "Snare" in g.balance_validation["non_finite_nodes"]
    assert g.nodes["PRINT"].staged is False  # post-terminal RoutingFolder stays exempt
    layout = compute_signal_flow_graph_layout(g, active_only=True)
    assert "Snare" in layout.nodes
