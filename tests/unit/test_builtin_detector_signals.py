"""Signal and metadata examples with known outcomes for the built-in detectors."""

import json
from fractions import Fraction

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import discover_track, load_track, subsonic_stft_analysis, windowed_stereo_correlation
from sessionpreplib.config import build_structured_defaults, default_config, flatten_structured_config
from sessionpreplib.detectors import (
    AudioClassifierDetector,
    ClippingDetector,
    DCOffsetDetector,
    DualMonoDetector,
    FormatConsistencyDetector,
    LengthConsistencyDetector,
    OneSidedSilenceDetector,
    SilenceDetector,
    StereoCompatDetector,
    SubsonicDetector,
    TailExceedanceDetector,
    UltrasonicDetector,
)
from sessionpreplib.models import SessionContext, Severity, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.reports import save_json


def _track(samples, rate=8000):
    samples = np.asarray(samples, dtype=np.float64)
    return TrackContext(
        filename="signal.wav", filepath="", audio_data=samples, samplerate=rate,
        channels=1 if samples.ndim == 1 else samples.shape[1], total_samples=len(samples),
        bitdepth="64-bit float", subtype="DOUBLE", duration_sec=len(samples) / rate,
    )


def _measure(detector, samples, rate=8000, **config):
    detector.configure(config)
    return detector.analyze(_track(samples, rate))


def test_silence_requires_every_channel_to_be_zero():
    stereo = np.zeros((12, 2))
    silent = _measure(SilenceDetector(), stereo)
    assert silent.severity == Severity.ATTENTION
    assert silent.data["topology_action"] == "drop"
    issue, = silent.issues
    assert (issue.sample_start, issue.sample_end, issue.channel) == (0, 11, None)

    stereo[-1, 1] = 2 ** -23  # One 24-bit PCM step on the right.
    active = _measure(SilenceDetector(), stereo)
    assert active.severity == Severity.CLEAN
    assert "topology_action" not in active.data


def test_dual_mono_rejects_a_level_change_even_with_perfect_correlation():
    tone = np.tile([0.25, -0.25], 64)
    duplicate = _measure(DualMonoDetector(), np.column_stack((tone, tone)))
    assert duplicate.severity == Severity.INFO
    assert duplicate.data["topology_channel"] == 0

    quieter = _measure(DualMonoDetector(), np.column_stack((tone, 0.5 * tone)))
    assert quieter.severity == Severity.CLEAN
    assert not quieter.data["dual_mono"]
    assert "topology_action" not in quieter.data


@pytest.mark.parametrize("active_channel", [0, 1])
def test_one_sided_silence_preserves_the_channel_carrying_audio(active_channel):
    stereo = np.zeros((128, 2))
    tone = np.tile([0.25, -0.25], 64)
    stereo[:, active_channel] = tone
    result = _measure(OneSidedSilenceDetector(), stereo)
    assert result.severity == Severity.ATTENTION
    assert result.data["topology_channel"] == active_channel
    issue, = result.issues
    assert issue.channel == 1 - active_channel

    stereo[:, 1 - active_channel] = 0.5 * tone
    active = _measure(OneSidedSilenceDetector(), stereo)
    assert active.severity == Severity.CLEAN
    assert "topology_action" not in active.data


def test_clipping_counts_each_qualifying_run_even_when_report_is_limited():
    samples = np.zeros((16, 2))
    samples[:2, 0] = 1.0
    short = _measure(ClippingDetector(), samples)
    assert short.severity == Severity.CLEAN
    assert not short.issues

    samples[4:7, 0] = -1.0
    samples[-3:, 1] = 1.0
    clipped = _measure(ClippingDetector(), samples, clip_report_max_ranges=1)
    assert clipped.severity == Severity.PROBLEM
    assert clipped.data["runs"] == 2
    assert clipped.data["affected_channels"] == [0, 1]
    issue, = clipped.issues
    assert (issue.sample_start, issue.sample_end, issue.channel) == (4, 6, 0)


def test_dc_offset_uses_each_channels_signed_mean():
    balanced = np.tile([0.25, -0.25], 128)
    clean = _measure(DCOffsetDetector(), np.column_stack((balanced, -balanced)))
    assert clean.severity == Severity.CLEAN
    biased = _measure(DCOffsetDetector(), np.column_stack((balanced, balanced - 0.125)))
    assert biased.severity == Severity.ATTENTION
    assert biased.data["signed_dc"] == [0.0, -0.125]
    assert biased.data["dc_db"] == pytest.approx(20 * np.log10(0.125))


@pytest.mark.parametrize(("right_gain", "correlation", "loss_db", "severity"), [
    (1.0, 1.0, 0.0, Severity.CLEAN),
    (-0.5, -1.0, 10.0, Severity.INFO),
])
def test_stereo_compatibility_measures_mono_fold_down(right_gain, correlation, loss_db, severity):
    left = np.tile([0.25, -0.25], 4000)
    stereo = np.column_stack((left, right_gain * left))
    result = _measure(StereoCompatDetector(), stereo, corr_windowed=False)
    assert result.severity == severity
    assert result.data["lr_corr"] == pytest.approx(correlation)
    assert result.data["mono_loss_db"] == pytest.approx(loss_db)


@pytest.mark.parametrize("windowed", [False, True])
def test_full_cancellation_still_exceeds_the_loss_limit_when_correlation_is_at_its_threshold(windowed):
    left = np.tile([0.25, -0.25], 4000)
    result = _measure(StereoCompatDetector(), np.column_stack((left, -left)),
                      corr_warn=-1.0, corr_windowed=windowed)
    assert result.data["mono_loss_db"] == np.inf
    assert result.data["mono_warn"]
    assert result.severity == (Severity.ATTENTION if windowed else Severity.INFO)
    assert "∞ dB" in result.summary
    if windowed:
        region, = result.data["windowed_regions"]
        assert region["max_mono_loss_db"] == result.data["max_window_mono_loss_db"] == np.inf
        assert "∞ dB" in result.issues[0].description


def test_local_cancellation_can_exceed_the_loss_threshold_with_a_clean_whole_file():
    left = np.tile([0.25, -0.25], 1200)
    right = left.copy()
    right[800:1600] *= -1
    result = _measure(StereoCompatDetector(), np.column_stack((left, right)),
                      corr_warn=-1.0, corr_window_ms=100)
    assert result.data["mono_loss_db"] == pytest.approx(10 * np.log10(3 / 2))
    assert not result.data["corr_warn"] and not result.data["mono_warn"]
    issue, = result.issues
    assert (issue.sample_start, issue.sample_end) == (800, 1599)
    assert result.data["max_window_mono_loss_db"] == np.inf


def test_small_residual_after_near_cancellation_has_finite_measured_loss():
    gain = -(1 - 2**-30)
    left = np.tile([0.25, -0.25], 1200)
    result = _measure(StereoCompatDetector(), np.column_stack((left, gain * left)))
    expected = 10 * np.log10(((1 + gain**2) / 2) / ((1 + gain) / 2)**2)
    assert result.data["mono_loss_db"] == pytest.approx(expected, abs=1e-6)
    assert result.data["max_window_mono_loss_db"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("residual", [0.0, 1e-17, 1e-165])
def test_mono_loss_measures_the_fold_before_removing_dc(residual):
    # A representable residual on an otherwise zero sample can be smaller
    # than one ULP of either channel's DC offset.
    left = np.tile([0.0, 0.25], 6000)
    right = -left
    right[::2] = residual
    result = _measure(StereoCompatDetector(), np.column_stack((left, right)), 48000)
    # Each channel has an AC amplitude of 0.125. The fold alternates
    # between residual/2 and zero, with AC amplitude residual/4.
    expected = 20 * np.log10(0.5 / residual) if residual else np.inf
    assert result.data["mono_loss_db"] == pytest.approx(expected)
    assert result.data["max_window_mono_loss_db"] == pytest.approx(expected)


def test_pcm16_fold_with_constant_dc_has_no_ac_signal():
    left = np.random.default_rng(19).integers(-2000, 2001, 4800)
    stereo = np.column_stack((left, 257 - left)) / 32768
    assert np.all(stereo.mean(axis=1) == 257 / 65536)
    result = _measure(StereoCompatDetector(), stereo, 48000, corr_window_ms=100)
    assert result.data["mono_loss_db"] == np.inf
    assert result.data["max_window_mono_loss_db"] == np.inf


def test_smallest_finite_double_residual_keeps_its_level_in_db():
    residual = np.nextafter(0.0, 1.0)
    samples = np.array([[0.0, residual], [0.25, -0.25]])
    result = _measure(StereoCompatDetector(), samples, 48000)
    expected_db = 20 * (np.log10(0.5) - np.log10(residual))
    assert result.data["mono_loss_db"] == pytest.approx(expected_db, abs=1e-8)
    assert result.data["max_window_mono_loss_db"] == pytest.approx(expected_db, abs=1e-8)


def test_region_limit_keeps_the_full_null_when_two_passages_have_equal_correlation():
    tone = np.tile([0.25, -0.25], 400)
    gap = np.zeros_like(tone)
    left = np.concatenate((tone, gap, tone))
    right = np.concatenate((-0.25 * tone, gap, -tone))
    result = _measure(StereoCompatDetector(), np.column_stack((left, right)),
                      corr_window_ms=100, corr_max_regions=1)
    assert result.data["max_window_mono_loss_db"] == np.inf
    issue, = result.issues
    assert (issue.sample_start, issue.sample_end) == (1600, 2399)


@pytest.mark.parametrize("both_active", [False, True])
def test_float_stereo_with_a_large_dc_sum_preserves_the_exact_fold_residual(tmp_path, both_active):
    cycle = ([[0, .5], [.25, .25], [-.25, .75], [1e-17, .5]] if both_active
             else [[1e-17, .5], [-1e-17, .5]])
    path = tmp_path / "Deep fold.wav"
    sf.write(path, np.tile(cycle, (12000 // len(cycle), 1)), 48000, subtype="FLOAT")
    track = load_track(str(path))
    left, right = [[Fraction(float(v)) for v in side] for side in track.audio_data[:len(cycle)].T]

    def variance(values):
        mean = sum(values) / len(values)
        return sum((x - mean)**2 for x in values) / len(values)

    folded = [(l + r) / 2 for l, r in zip(left, right)]
    expected = 10 * np.log10(float((variance(left) + variance(right)) / (2 * variance(folded))))
    detector = StereoCompatDetector()
    detector.configure({})
    result = detector.analyze(track)
    assert result.data["mono_loss_db"] == pytest.approx(expected, abs=1e-8)
    if both_active:
        assert result.data["max_window_mono_loss_db"] == pytest.approx(expected, abs=1e-8)
    else:
        assert result.severity == Severity.CLEAN and result.data["lr_corr"] is None


def test_nulls_separated_by_silence_keep_each_span_and_use_the_same_loss_in_summary():
    tone = np.tile([0.25, -0.25], 400)
    gap = np.zeros_like(tone)
    left = np.concatenate((tone, gap, tone, gap, tone[:200]))
    track = _track(np.column_stack((left, -left)))
    detector = StereoCompatDetector()
    detector.configure({"corr_window_ms": 100})
    result = detector.analyze(track)
    assert [(i.sample_start, i.sample_end) for i in result.issues] == [
        (0, 799), (1600, 2399), (3200, 3399),
    ]
    track.detector_results[detector.id] = result
    summary = build_diagnostic_summary(SessionContext([track], {}))
    details = [item for group in summary["attention"] for item in group["items"]]
    assert any("mono loss ∞ dB" in item for item in details)


@pytest.mark.parametrize(("gain", "expected_db"), [(-1.0, None), (-0.5, 10.0), (1.0, 0.0)])
def test_json_report_distinguishes_full_cancellation_from_finite_loss(tmp_path, gain, expected_db):
    left = np.tile([0.25, -0.25], 16)
    track = _track(np.column_stack((left, gain * left)))
    detector = StereoCompatDetector()
    detector.configure({})
    track.detector_results[detector.id] = detector.analyze(track)
    path = tmp_path / "report.json"
    save_json(SessionContext([track], {}), {}, str(path), [])
    payload = json.loads(path.read_text(encoding="utf-8"),
                         parse_constant=lambda s: pytest.fail(f"Invalid JSON number: {s}"))
    diagnostics = payload["tracks"][0]["diagnostics"]
    assert diagnostics["mono_fully_cancelled"] is (gain == -1.0)
    if expected_db is None:
        assert diagnostics["mono_loss_db"] is None
    else:
        assert diagnostics["mono_loss_db"] == pytest.approx(expected_db)


def test_short_antiphase_and_dc_only_final_window():
    short = np.array([0.25, -0.25])
    result = _measure(StereoCompatDetector(), np.column_stack((short, -short)), corr_warn=-1.0)
    assert result.data["mono_loss_db"] == np.inf
    assert result.issues[0].sample_end == 1
    # A constant final partial window contains no AC after removing its real mean.
    left = np.r_[np.tile([0.25, -0.25], 400), np.full(29, 0.5)]
    right = left.copy()
    right[-29:] = -0.5
    result = _measure(StereoCompatDetector(), np.column_stack((left, right)), corr_window_ms=100)
    assert result.severity == Severity.CLEAN
    expected_corr = np.corrcoef(left, right)[0, 1]
    stereo_power = (np.var(left) + np.var(right)) / 2
    expected_loss = 10 * np.log10(stereo_power / np.var((left + right) / 2))
    assert result.data["lr_corr"] == pytest.approx(expected_corr)
    assert result.data["mono_loss_db"] == pytest.approx(expected_loss)
    _, _, windows = windowed_stereo_correlation(left, right, 8000, window_ms=100)
    start, end, corr, loss = windows[-1]
    assert (start, end) == (800, 828)
    assert np.isnan(corr) and np.isnan(loss)


def test_default_stereo_windows_localize_each_quarter_second_of_cancellation():
    left = np.tile([0.25, 0, -0.25, 0], 12000)
    stereo = np.column_stack((left, left * np.repeat([1, -1, 1, -1], 12000)))
    for config in ({}, default_config(), flatten_structured_config(build_structured_defaults())):
        result = _measure(StereoCompatDetector(), stereo, 48000, **config)
        assert result.data["lr_corr"] == pytest.approx(0, abs=1e-12)
        assert result.data["mono_loss_db"] == pytest.approx(10 * np.log10(2))
        assert [(i.sample_start, i.sample_end) for i in result.issues] == [(12000, 23999), (36000, 47999)]
    assert _measure(StereoCompatDetector(), stereo, 48000, corr_window_ms=500).severity == Severity.CLEAN


@pytest.mark.parametrize("window_ms", [100, 250, 500])
@pytest.mark.parametrize("windowed", [True, False])
def test_whole_file_loss_includes_low_side_energy_independently_of_local_windows(window_ms, windowed):
    rate = 48000
    t = np.arange(2 * rate) / rate
    mid = 0.1 * np.sin(2 * np.pi * 300 * t)
    side_ratio = 10**0.605 - 1
    side = 0.1 * np.sqrt(side_ratio) * np.sin(2 * np.pi * 25 * t)
    result = _measure(StereoCompatDetector(), np.column_stack((mid + side, mid - side)), rate,
                      corr_warn=-1.0, corr_window_ms=window_ms, corr_windowed=windowed)
    assert result.data["lr_corr"] == pytest.approx((1 - side_ratio) / (1 + side_ratio))
    assert result.data["mono_loss_db"] == pytest.approx(6.05)
    assert result.data["mono_warn"]
    if not windowed or window_ms == 100:
        assert result.severity == Severity.INFO


@pytest.mark.parametrize("active", [0, 1])
def test_hard_pan_only_warns_when_its_measured_mono_loss_exceeds_the_threshold(active):
    t = np.arange(96000) / 48000
    stereo = np.full((len(t), 2), 0.04)
    stereo[:, active] = 0.1 * np.sin(2 * np.pi * 1000 * t)
    for limit in (6, 2):
        result = _measure(StereoCompatDetector(), stereo, 48000, corr_warn=0.2, mono_loss_warn_db=limit)
        assert result.data["lr_corr"] is None and not result.data["corr_warn"]
        assert result.data["mono_loss_db"] == pytest.approx(10 * np.log10(2))
        if limit == 6:
            assert result.severity == Severity.CLEAN and not result.issues
        else:
            region, = result.data["windowed_regions"]
            assert region["min_corr"] is None
            assert region["max_mono_loss_db"] == pytest.approx(10 * np.log10(2))
            assert (region["sample_start"], region["sample_end"]) == (0, len(t) - 1)
            assert "unavailable" in result.issues[0].description
    stereo[:, 1 - active] = 0.1 * np.cos(2 * np.pi * 1000 * t)
    orthogonal = _measure(StereoCompatDetector(), stereo, 48000, corr_warn=0.2)
    assert orthogonal.severity == Severity.ATTENTION
    assert orthogonal.data["min_window_corr"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("frequency", [None, 2, 12])
def test_broad_subsonic_meter_ignores_bias_and_retains_real_low_frequencies(frequency):
    rate = 48000
    t = np.arange(2 * rate) / rate
    signal = 0.1 * np.sin(2 * np.pi * 1000 * t)
    if frequency:
        signal += 0.04 * np.sin(2 * np.pi * frequency * t + np.pi / 4)
    for bias in (0.0, 0.04, -0.05):
        result = _measure(SubsonicDetector(), signal + bias, rate)
        if frequency:
            assert result.severity == Severity.INFO
            assert result.data["subsonic_ratio_db"] == pytest.approx(10 * np.log10(.04**2 / (.04**2 + .1**2)), abs=0.1)
        else:
            assert result.severity == Severity.CLEAN and not result.issues
            assert result.data["subsonic_ratio_db"] < -100
        assert (_measure(DCOffsetDetector(), signal + bias, rate).severity == Severity.ATTENTION) == bool(bias)


@pytest.mark.parametrize("frames", [1001, 24000, 48517])
def test_dc_is_removed_before_padding_overlapping_subsonic_windows(frames):
    rate = 48000
    ac = np.sin(2 * np.pi * 12 * np.arange(frames) / rate) / 8
    ratio, windows = subsonic_stft_analysis(ac, rate, 30, window_ms=500, hop_ms=250)
    biased, shifted = subsonic_stft_analysis(ac + .05, rate, 30, window_ms=500, hop_ms=250)
    assert biased == pytest.approx(ratio, abs=1e-10)
    np.testing.assert_allclose(shifted, windows, rtol=0, atol=1e-10)
    assert windows[-1][1] == frames - 1
    constant, gated = subsonic_stft_analysis(np.full(frames, .05), rate, 30, window_ms=500, hop_ms=250)
    assert constant == -np.inf
    assert all(level == -np.inf for _, _, level in gated)


@pytest.mark.parametrize(("subtype", "suffix"), [("PCM_U8", ".wav"), ("PCM_S8", ".aiff")])
@pytest.mark.parametrize(("code", "runs"), [(127, 4000), (-128, 4000), (126, 0), (-127, 0)])
def test_encoded_eight_bit_clipping_uses_each_polaritys_rail(tmp_path, subtype, suffix, code, runs):
    samples = np.tile(np.array([0, 38, 89, code, code, code, 89, 38, 0, -25, -51, -25]) / 128, 4000)
    path = tmp_path / ("rail" + suffix)
    sf.write(path, samples, 48000, subtype=subtype)
    track = load_track(str(path))
    np.testing.assert_array_equal(track.audio_data.ravel(), samples)
    detector = ClippingDetector()
    detector.configure({})
    result = detector.analyze(track)
    assert result.data["runs"] == runs
    if runs:
        assert (result.issues[0].sample_start, result.issues[0].sample_end) == (3, 5)


def test_subsonic_finding_identifies_the_channel_containing_low_band_energy():
    time = np.arange(8000) / 8000
    bed = 0.2 * np.sin(2 * np.pi * 400 * time)
    clean = _measure(SubsonicDetector(), np.column_stack((bed, bed)))
    assert clean.severity == Severity.CLEAN
    right = bed + 0.2 * np.sin(2 * np.pi * 16 * time)
    finding = _measure(SubsonicDetector(), np.column_stack((bed, right)))
    assert finding.severity == Severity.INFO
    assert not finding.data["per_channel"][0]["warn"]
    assert finding.data["per_channel"][1]["ratio_db"] == pytest.approx(10 * np.log10(0.5), abs=0.1)
    assert {(i.channel, i.freq_min_hz, i.freq_max_hz) for i in finding.issues} == {(1, 0.0, 30.0)}


def test_quiet_subsonic_fallback_ignores_bias_in_the_other_channel_and_silent_gaps():
    rate = 48000
    t = np.arange(2 * rate) / rate
    rumble = .001 * np.sin(2 * np.pi * 12 * t)
    rumble[rate // 2:rate] = 0
    stereo = np.column_stack((np.full(t.size, .05), rumble + .05))
    result = _measure(SubsonicDetector(), stereo, rate)
    assert [(i.sample_start, i.sample_end, i.channel) for i in result.issues] == [
        (0, 23999, 1), (48000, 95999, 1),
    ]
    assert all("active audio" in i.description for i in result.issues)


def test_long_non_fft_friendly_track_keeps_the_whole_record_frequency_grid():
    n, rate = 300007, 48000
    phase = 2 * np.pi * np.arange(n) / n
    last_in_band = int(30 * n / rate)
    # Adjacent coherent DFT bins straddle the cutoff. The final incomplete
    # processing block must contribute with the same phase as the first.
    samples = .05 + .03 * np.cos(last_in_band * phase) + .1 * np.sin((last_in_band + 1) * phase)
    expected = 10 * np.log10(.03**2 / (.03**2 + .1**2))
    for window in (100, 500):
        ratio, _ = subsonic_stft_analysis(samples, rate, 30, window_ms=window)
        assert ratio == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
def test_decimal_cutoff_includes_the_boundary_bin_and_excludes_its_lower_neighbor(rate):
    frames = 5 * rate // 2
    phase = 2 * np.pi * np.arange(frames) / frames
    samples = .04 * np.sin(41 * phase) + .1 * np.cos(2500 * phase)
    cutoff = 16.4  # bin 41 of a 2.5-second recording
    expected = 10 * np.log10(.04**2 / (.04**2 + .1**2))
    for hz in (cutoff, np.nextafter(cutoff, np.inf)):
        result = _measure(SubsonicDetector(), samples, rate, subsonic_hz=hz)
        assert result.severity == Severity.INFO
        assert result.data["subsonic_ratio_db"] == pytest.approx(expected, abs=1e-8)
    result = _measure(SubsonicDetector(), samples, rate, subsonic_hz=np.nextafter(cutoff, -np.inf))
    assert result.severity == Severity.CLEAN and result.data["subsonic_ratio_db"] < -120


@pytest.mark.parametrize("phase", [0, np.pi / 4, np.pi / 2])
@pytest.mark.parametrize(("hz", "level"), [(1, -20.5), (1, -19.5), (2, -19.8), (29, -19.5), (30, -19.5), (31, -8)])
def test_whole_subsonic_reading_matches_coherent_band_energy_at_every_phase(hz, level, phase):
    rate = 48000
    t = np.arange(4 * rate) / rate
    share = 10**(level / 10)
    low = 0.1 * np.sqrt(share / (1 - share)) * np.sin(2 * np.pi * hz * t + phase)
    samples = 0.05 + 0.1 * np.cos(2 * np.pi * 1000 * t) + low
    result = _measure(SubsonicDetector(), samples, rate)
    ratio = result.data["subsonic_ratio_db"]
    if hz <= 30:
        assert ratio == pytest.approx(level, abs=1e-8)
    else:
        assert ratio < -120
    assert (result.severity == Severity.INFO) == (hz <= 30 and level >= -20)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT"])
def test_higher_resolution_clipping_retains_its_near_full_scale_threshold(tmp_path, subtype):
    path = tmp_path / "limits.wav"
    sf.write(path, np.repeat([.99995, .9998, -.99995, -.9998], 3), 48000, subtype=subtype)
    detector = ClippingDetector()
    detector.configure({})
    result = detector.analyze(load_track(str(path)))
    assert [(i.sample_start, i.sample_end) for i in result.issues] == [(0, 2), (6, 8)]


def test_ultrasonic_requires_sustained_energy_at_an_eligible_sample_rate():
    rate = 96000
    time = np.arange(rate // 2) / rate
    audible = 0.1 * np.sin(2 * np.pi * 1000 * time)
    ultrasonic = 0.1 * np.sin(2 * np.pi * 30000 * time)
    result = _measure(UltrasonicDetector(), np.column_stack((audible, ultrasonic)), rate)
    assert result.severity == Severity.INFO
    issue, = result.issues
    assert (issue.sample_start, issue.sample_end, issue.channel) == (0, len(time) - 1, 1)
    assert (issue.freq_min_hz, issue.freq_max_hz) == (24100, rate / 2)

    short = _measure(UltrasonicDetector(), ultrasonic[:rate // 10], rate)
    assert short.applicable and short.severity == Severity.CLEAN
    ineligible = _measure(UltrasonicDetector(), audible, 48000)
    assert not ineligible.applicable and not ineligible.issues


def test_classifier_distinguishes_a_steady_sine_from_an_isolated_impulse():
    time = np.arange(8000) / 8000
    tone = _measure(AudioClassifierDetector(), 0.5 * np.sin(2 * np.pi * 250 * time))
    assert tone.data["classification"] == "Sustained"
    assert tone.data["crest"] == pytest.approx(10 * np.log10(2))
    assert tone.data["peak_db"] == pytest.approx(20 * np.log10(0.5))

    samples = np.zeros(len(time))
    samples[1000] = 0.5
    impulse = _measure(AudioClassifierDetector(), samples)
    assert impulse.data["classification"] == "Transient"
    assert impulse.data["peak_db"] == pytest.approx(tone.data["peak_db"])


def _measure_tail(samples, **options):
    track = _track(samples)
    config = {"window": 50, "rms_percentile": 80, **options}
    detectors = [TailExceedanceDetector(), AudioClassifierDetector(), SilenceDetector()]
    pipeline = Pipeline(detectors, config=config)
    session = SessionContext([track], config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    return track.detector_results["tail_exceedance"]


def test_tail_exceedance_finds_the_loud_section_relative_to_the_measured_anchor():
    bed = np.tile([0.125, -0.125], 32000)
    assert _measure_tail(bed).severity == Severity.CLEAN
    bed[20000:22000] *= 4
    result = _measure_tail(bed)
    assert result.severity == Severity.ATTENTION
    assert result.data["tail_summary"]["max_exceed_db"] == pytest.approx(20 * np.log10(4))
    issue, = result.issues
    # Each 50 ms analysis window covers 400 frames.
    assert 20000 - 400 < issue.sample_start <= 20000
    assert 21999 <= issue.sample_end < 21999 + 400


def test_tail_regions_do_not_include_the_silence_between_two_loud_sections():
    bed = np.tile([0.125, -0.125], 32000)
    bed[20000:22000] *= 4
    bed[22000:34000] = 0
    bed[34000:36000] *= 4
    result = _measure_tail(bed)
    issues = result.issues
    assert len(issues) == 2
    assert [(i.sample_start, i.sample_end) for i in issues] == [(19680, 22319), (33680, 36319)]
    assert result.data["tail_summary"]["total_duration_sec"] == pytest.approx(5280 / 8000)


def test_total_tail_duration_counts_overlapping_window_support_only_once():
    rate = 8000
    signal = np.zeros(26 * rate)
    signal[:22 * rate] = 0.002 * np.sin(2 * np.pi * 1000 * np.arange(22 * rate) / rate)
    signal[24 * rate] = signal[24 * rate + 3280] = 0.8
    result = _measure_tail(signal, window=400, rms_percentile=95)
    assert [(i.sample_start, i.sample_end) for i in result.issues] == [
        (188880, 195199), (192160, 198479),
    ]
    assert result.data["tail_summary"]["total_duration_sec"] == pytest.approx(1.2)


def test_tail_scan_with_widely_spaced_hops_leaves_unmeasured_gaps_out_of_regions():
    bed = np.tile([0.125, -0.125], 32000)
    bed[20000:22000] *= 4
    result = _measure_tail(bed, window=10, tail_hop_ms=100)
    assert [(i.sample_start, i.sample_end) for i in result.issues] == [
        (20000, 20079), (20800, 20879), (21600, 21679),
    ]


def test_session_detectors_distinguish_format_from_duration_across_sample_rates(tmp_path):
    tracks = []
    for name, rate, frames, subtype in [
        ("one.wav", 48000, 4800, "PCM_24"), ("two.wav", 48000, 4800, "PCM_24"),
        ("other-format.wav", 96000, 9600, "PCM_16"), ("short.wav", 48000, 4799, "PCM_24"),
    ]:
        path = tmp_path / name
        sf.write(path, np.zeros(frames), rate, subtype=subtype)
        tracks.append(discover_track(str(path)))
    session = SessionContext(tracks, {})
    formats = {r.data["filename"]: r for r in FormatConsistencyDetector().analyze(session)}
    lengths = {r.data["filename"]: r for r in LengthConsistencyDetector().analyze(session)}

    assert [name for name, r in formats.items() if r.severity == Severity.PROBLEM] == ["other-format.wav"]
    mismatch = formats["other-format.wav"].data
    assert (mismatch["expected_sr"], mismatch["actual_sr"]) == (48000, 96000)
    assert (mismatch["expected_bd"], mismatch["actual_bd"]) == ("24-bit", "16-bit")
    assert lengths["other-format.wav"].severity == Severity.CLEAN
    assert formats["short.wav"].severity == Severity.CLEAN
    short = lengths["short.wav"]
    assert short.severity == Severity.INFO
    assert (short.data["expected_samples"], short.data["actual_samples"]) == (4800, 4799)
