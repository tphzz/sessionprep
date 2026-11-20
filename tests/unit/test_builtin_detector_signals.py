"""Signal and metadata examples with known outcomes for the built-in detectors."""

import json

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import discover_track
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
    assert result.data["mono_loss_db"] == pytest.approx(0.0)


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
