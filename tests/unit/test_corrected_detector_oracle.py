"""Verify the two detector records against measurements of the actual signals."""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import periodogram

from sessionpreplib.audio import load_track
from sessionpreplib.detector import TrackDetector
from sessionpreplib.detectors import ClippingDetector, default_detectors
from sessionpreplib.models import DetectorResult, SessionContext, Severity
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import RepairRecipe, render_corrected
from sessionpreplib.topology import TopologyEntry, TopologyMapping, TopologySource, extract_channel
from tests.fs_helpers import rewrite_preserving_mtime as _rewrite_same_stamp


def _plan(tmp_path, audio, rate, choices, *, config=None, manual_gain_db=None):
    path = tmp_path / "Track.wav"
    sf.write(path, audio, rate, subtype="DOUBLE")
    track = load_track(str(path))
    track.repair_overrides.update(choices)
    track.manual_gain_db = manual_gain_db
    config = {} if config is None else config
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    session = SessionContext([track], config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    assert track.corrected_analysis is not None
    np.testing.assert_array_equal(track.audio_data, audio)
    return track


def _assert_matches_direct(track, audio, recipe):
    """Give each source its own detector instances, audio cache, and dependencies."""
    assert track.repair_recipe == recipe
    corrected = render_corrected(audio, track.samplerate, recipe)
    for samples, recorded in ((audio, track.detector_results), (corrected, track.corrected_analysis.results)):
        reference = replace(track, audio_data=samples, _cache={}, detector_results={}, processor_results={})
        for detector in Pipeline(default_detectors()).track_detectors:
            expected = detector.analyze(reference)
            reference.detector_results[detector.id] = expected
            assert expected.error is None
            np.testing.assert_equal(asdict(recorded[detector.id]), asdict(expected), err_msg=detector.id)


def _locations(result):
    return [(i.sample_start, i.sample_end, i.channel, i.freq_min_hz, i.freq_max_hz) for i in result.issues]


def test_dc_removal_clears_bias_without_earning_a_false_subsonic_verdict(tmp_path):
    rate = 8000
    time = np.arange(8 * rate) / rate
    left = 0.2 * np.sin(2 * np.pi * 440 * time)
    right = left.copy()
    reversed_at = (1, 3, 5, 7)
    for start in reversed_at:
        right[start * rate:start * rate + rate // 4] *= -1
    stereo = np.column_stack((left + 0.05, right + 0.05))
    track = _plan(tmp_path, stereo, rate, {"dc_remove": True})

    before = track.detector_results
    after = track.corrected_analysis.results
    assert before["dc_offset"].severity == Severity.ATTENTION
    assert before["subsonic"].severity == Severity.CLEAN and not before["subsonic"].issues
    for detector_id in ("dc_offset", "subsonic"):
        assert after[detector_id].applicable and after[detector_id].severity == Severity.CLEAN
        assert after[detector_id].issues == []
    # The phase switches and the repair's startup transient can have real low
    # energy. Neither file contains a sustained low-frequency fault.
    for findings in (before, after):
        assert findings["subsonic"].data["subsonic_ratio_db"] < -30
    expected = [(s * rate, s * rate + rate // 4 - 1) for s in reversed_at]
    for findings in (before, after):
        stereo_result = findings["stereo_compat"]
        assert stereo_result.applicable and stereo_result.severity == Severity.ATTENTION
        assert sorted((i.sample_start, i.sample_end) for i in stereo_result.issues) == expected


def test_partial_subsonic_repair_keeps_only_the_affected_channel_and_band(tmp_path):
    rate = 48000
    time = np.arange(6 * rate) / rate
    bed = 0.2 * np.sin(2 * np.pi * 440 * time)
    left = bed + 0.12 + 0.03 * np.sin(2 * np.pi * time)
    audio = np.column_stack((left, bed + 0.18 * np.sin(2 * np.pi * 15 * time)))
    track = _plan(tmp_path, audio, rate, {"dc_remove": True})
    _assert_matches_direct(track, audio, RepairRecipe(dc_remove=True))

    before = track.detector_results["subsonic"]
    after = track.corrected_analysis.results["subsonic"]
    assert before.severity == after.severity == Severity.INFO
    assert [value["warn"] for value in before.data["per_channel"].values()] == [True, True]
    assert [value["warn"] for value in after.data["per_channel"].values()] == [False, True]
    assert before.data["per_channel"][0]["ratio_db"] > -20
    assert after.data["per_channel"][0]["ratio_db"] < -20
    assert sorted(_locations(before)) == [(0, 287999, 0, 0.0, 30.0), (0, 287999, 1, 0.0, 30.0)]
    assert _locations(after) == [(0, 287999, 1, 0.0, 30.0)]
    assert track.detector_results["dc_offset"].severity == Severity.ATTENTION
    assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    assert not track.corrected_analysis.results["ultrasonic"].applicable


def test_repair_can_introduce_a_clipping_run_before_output_trim(tmp_path):
    rate = 48000
    time = np.arange(6 * rate) / rate
    audio = np.where(time < 3.0, 0.8, -0.8)
    track = _plan(tmp_path, audio, rate, {"dc_remove": True}, manual_gain_db=-18.0)

    before = track.detector_results["clipping"]
    after = track.corrected_analysis.results["clipping"]
    assert before.severity == Severity.CLEAN and before.issues == []
    assert after.severity == Severity.PROBLEM and after.data["runs"] == 1
    assert _locations(after) == [(144000, 145795, None, None, None)]
    before_peak = 20 * np.log10(float(np.max(np.abs(audio))))
    after_peak = 20 * np.log10(float(np.max(np.abs(render_corrected(audio, rate, track.repair_recipe)))))
    assert before_peak == pytest.approx(20 * np.log10(0.8))
    assert after_peak == pytest.approx(20 * np.log10(1.5997906), abs=0.00001)
    gain = track.processor_results["bimodal_normalize"]
    assert gain.gain_db == -18.0
    assert gain.data["detected_peak_db"] == pytest.approx(after_peak)


def test_corrected_eight_bit_source_is_metered_before_reencoding(tmp_path):
    path = tmp_path / "rail.wav"
    samples = np.tile([-1 / 128, 0.0], 48000)
    samples[48000:48003] = 127 / 128
    sf.write(path, samples, 48000, subtype="PCM_U8")
    track = load_track(str(path))
    track.repair_overrides["dc_remove"] = True
    pipeline = Pipeline(default_detectors(), default_processors(), {}, max_workers=1)
    session = SessionContext([track], {})
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    corrected = render_corrected(samples, 48000, track.repair_recipe)
    assert np.all(corrected[48000:48003] > 127 / 128)
    assert np.max(np.abs(corrected)) < .9999
    assert track.detector_results["clipping"].data["runs"] == 1
    assert track.corrected_analysis.results["clipping"].data["runs"] == 0
    assert track.subtype == "PCM_U8"


def test_right_only_output_does_not_inherit_discarded_left_clipping(tmp_path):
    rate = 48000
    time = np.arange(rate) / rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    left = tone + 0.03
    left[6001:6009] = 1.5
    right = tone - 0.06
    audio = np.column_stack((left, right))
    source = _plan(tmp_path, audio, rate, {"dc_remove": True})
    route = TopologySource(source.filename, extract_channel(1))
    entry = TopologyEntry("Right.wav", 1, [route])
    session = SessionContext([source], {}, topology=TopologyMapping([entry]))
    Pipeline(default_detectors()).prepare(session, str(tmp_path / "prepared"))
    assert session.config["_prepare_errors"] == []
    output, = session.output_tracks

    corrected = render_corrected(audio, rate, source.repair_recipe)[:, 1]
    meter = ClippingDetector()
    meter.configure({})
    for path, expected in ((output.filepath, right), (output.processed_filepath, corrected)):
        rendered = load_track(path)
        assert rendered.channels == 1
        np.testing.assert_allclose(rendered.audio_data, expected, atol=2e-8, rtol=0)
        assert meter.analyze(rendered).issues == []

    for results in (source.detector_results, source.corrected_analysis.results):
        assert _locations(results["clipping"]) == [(6001, 6008, 0, None, None)]
    if "clipping" in output.detector_results:
        assert output.detector_results["clipping"].severity == Severity.CLEAN
        assert output.detector_results["clipping"].issues == []
    if output.corrected_analysis is not None:
        assert output.corrected_analysis.results["clipping"].severity == Severity.CLEAN
        assert output.corrected_analysis.results["clipping"].issues == []


@pytest.mark.parametrize("change", ["resident-edit", "resident-edit-and-reanalyze", "same-stat-file-replacement"])
def test_changed_samples_cannot_reuse_either_sources_peak_reading(tmp_path, change):
    rate = 8000
    time = np.arange(3 * rate) / rate
    audio = 0.1 + 0.2 * np.sin(2 * np.pi * 440 * time)
    track = _plan(tmp_path, audio, rate, {"dc_remove": True})
    previous = track.corrected_analysis
    original_dc = track.detector_results["dc_offset"].data["dc_db"]
    corrected_peak = track.processor_results["bimodal_normalize"].data["detected_peak_db"]
    expected_corrected_peak = 20 * np.log10(float(np.max(np.abs(render_corrected(audio, rate, track.repair_recipe)))))
    assert original_dc == pytest.approx(20 * np.log10(0.1), abs=0.05)
    assert corrected_peak == pytest.approx(expected_corrected_peak, abs=0.01)

    if change == "same-stat-file-replacement":
        path = Path(track.filepath)
        _rewrite_same_stamp(path, lambda: sf.write(path, audio * 0.1, rate, subtype="DOUBLE"))
        track.audio_data = None
    else:
        track.audio_data *= 0.1

    session = SessionContext([track], {})
    pipeline = Pipeline(default_detectors(), default_processors())
    if change == "resident-edit-and-reanalyze":
        pipeline.analyze_phase1(session)
        pipeline.analyze_phase2(session)
    pipeline.plan(session)
    if track.corrected_analysis is None:
        # Source replacement can invalidate the plan and request reanalysis.
        assert track._cache.get("repair_error")
    else:
        assert track.corrected_analysis.key != previous.key
        assert track.processor_results["bimodal_normalize"].data["detected_peak_db"] == pytest.approx(
            corrected_peak - 20.0, abs=0.05
        )
        assert track.detector_results["dc_offset"].data["dc_db"] == pytest.approx(original_dc - 20.0, abs=0.05)


def test_prepare_rejects_source_replaced_between_signature_check_and_read(tmp_path, monkeypatch):
    rate = 8000
    source = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    track = _plan(tmp_path, source, rate, {"dc_remove": True})
    track.audio_data = None

    def replace_then_load(path):
        target = Path(path)
        _rewrite_same_stamp(target, lambda: sf.write(target, 4 * source, rate, subtype="DOUBLE"))
        return load_track(path)

    monkeypatch.setattr("sessionpreplib.repair_plan.load_track", replace_then_load)
    session = SessionContext([track], {})
    directory = tmp_path / "prepared"
    Pipeline(default_detectors()).prepare(session, str(directory))
    assert session.config["_prepared_files"] == []
    assert len(session.config["_prepare_errors"]) == 1
    assert "changed" in session.config["_prepare_errors"][0][1].lower()
    assert not (directory / track.filename).exists()


def test_residual_dc_is_still_flagged_when_a_short_track_ends_before_settling(tmp_path):
    time = np.arange(2205) / 44100
    audio = 0.2 + 0.03 * np.sin(2 * np.pi * 440 * time)
    track = _plan(tmp_path, audio, 44100, {"dc_remove": True}, config={"rms_anchor": "max"})
    before = track.detector_results["dc_offset"]
    after = track.corrected_analysis.results["dc_offset"]
    assert before.severity == after.severity == Severity.ATTENTION
    assert after.applicable
    assert before.data["dc_db"] == pytest.approx(-13.9794, abs=0.001)
    assert after.data["dc_db"] == pytest.approx(-16.5717, abs=0.001)
    assert _locations(after) == [(0, 2204, None, None, None)]
    assert not track.corrected_analysis.results["tail_exceedance"].applicable


def test_local_rumble_clearance_does_not_create_a_whole_track_finding(tmp_path):
    rate = 8000
    time = np.arange(40 * rate) / rate
    bed = 0.03 * np.sin(2 * np.pi * 440 * time)
    rumble = 0.008 * np.sin(2 * np.pi * 8 * time) * (time < 3)
    audio = np.column_stack((bed, -bed - rumble))
    track = _plan(tmp_path, audio, rate, {"low_cut": True, "low_cut_hz": 10})

    before = track.detector_results["subsonic"]
    after = track.corrected_analysis.results["subsonic"]
    assert before.severity == after.severity == Severity.CLEAN
    assert before.issues == after.issues == []
    corrected = render_corrected(audio, rate, track.repair_recipe)
    for source, result in ((audio, before), (corrected, after)):
        frequencies, power = periodogram(source, fs=rate, window="boxcar", axis=0)
        ac = frequencies > 0
        ratio = power[ac & (frequencies <= 30)].sum(axis=0) / power[ac].sum(axis=0)
        assert result.data["subsonic_ratio_db"] == pytest.approx(10 * np.log10(ratio.max()), abs=1e-8)
    original_local = track.repair_measurements["original"]["low_band"]
    corrected_local = track.repair_measurements["corrected"]["low_band"]
    assert original_local["regions"] == [[0, 27999, 1]]
    assert corrected_local["regions"] == []


def test_complete_recipe_updates_ultrasonic_regions_and_classifier_together(tmp_path):
    rate = 96000
    time = np.arange(2 * rate) / rate
    bed = 0.1 * np.sin(2 * np.pi * 440 * time)
    carrier = 0.05 * np.sin(2 * np.pi * 31000 * time)
    audio = np.column_stack((bed + 0.08, bed + carrier))
    choices = {"dc_remove": True, "low_cut": True, "low_cut_hz": 10, "ultrasonic": True}
    track = _plan(tmp_path, audio, rate, choices)
    _assert_matches_direct(track, audio, RepairRecipe(dc_remove=True, low_cut_hz=10, ultrasonic=True))

    before = track.detector_results["ultrasonic"]
    after = track.corrected_analysis.results["ultrasonic"]
    assert before.applicable and after.applicable
    assert before.severity == Severity.INFO and after.severity == Severity.CLEAN
    assert _locations(before) == [(0, 191999, 1, 24100, 48000.0)]
    assert _locations(after) == []


@pytest.mark.parametrize("disable_after_check", [False, True], ids=["disabled-initially", "disabled-after-plan"])
def test_disabled_ultrasonic_check_stays_out_of_both_sources_during_manual_repair(tmp_path, disable_after_check):
    rate = 96000
    time = np.arange(rate) / rate
    audio = 0.1 * np.sin(2 * np.pi * 31000 * time)
    config = {"ultrasonic_enabled": disable_after_check}
    track = _plan(tmp_path, audio, rate, {"ultrasonic": True}, config=config)
    if disable_after_check:
        assert track.detector_results["ultrasonic"].severity == Severity.INFO
        config["ultrasonic_enabled"] = False
        Pipeline(default_detectors(), config=config).plan(SessionContext([track], config))
    assert "ultrasonic" not in track.detector_results
    assert "ultrasonic" not in track.corrected_analysis.results
    before = track.repair_measurements["original"]["ultrasonic"]
    after = track.repair_measurements["corrected"]["ultrasonic"]
    assert before["regions"] == [[0, rate - 1, 0]]
    assert after["regions"] == []


def test_both_sources_use_the_current_threshold_when_phase_one_ran_earlier(tmp_path):
    rate = 8000
    tone = np.sin(2 * np.pi * 440 * np.arange(3 * rate) / rate)
    path = tmp_path / "Low right channel.wav"
    sf.write(path, np.column_stack((0.2 * tone, 0.00006 * tone)), rate, subtype="DOUBLE")
    track = load_track(str(path))
    track.repair_overrides["dc_remove"] = True
    config = {"one_sided_silence_db": -80.0}
    session = SessionContext([track], config)
    Pipeline(default_detectors(), config=config).analyze_phase1(session)
    result = track.detector_results["one_sided_silence"]
    assert result.severity == Severity.ATTENTION
    assert result.data["r_rms_db"] == pytest.approx(-87.4473, abs=0.001)

    session.config["one_sided_silence_db"] = -90.0
    pipeline = Pipeline(default_detectors(), config=session.config)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    assert track.detector_results["one_sided_silence"].severity == Severity.CLEAN
    assert track.corrected_analysis.results["one_sided_silence"].severity == Severity.CLEAN


class _GatedBiasMeter(TrackDetector):
    id = "gated_bias"

    @classmethod
    def html_help(cls):
        return "Measure DC bias above the meter's operating range floor."

    def analyze(self, track):
        bias = abs(float(np.mean(track.audio_data)))
        if bias < 0.1:
            return DetectorResult(self.id, Severity.CLEAN, "Below operating range", {}, applicable=False)
        return DetectorResult(self.id, Severity.INFO, "Bias measured", {"bias": bias})


class _BiasWarning(TrackDetector):
    id = "bias_warning"
    depends_on = [_GatedBiasMeter.id]

    @classmethod
    def html_help(cls):
        return "Warn when the measured DC bias exceeds 0.15 FS."

    def analyze(self, track):
        bias = track.detector_results[_GatedBiasMeter.id].data.get("bias", 0.0)
        severity = Severity.ATTENTION if bias > 0.15 else Severity.CLEAN
        return DetectorResult(self.id, severity, "Bias check", {"bias": bias})


def test_dependent_finding_cannot_clear_if_its_repaired_measurement_is_ineligible(tmp_path):
    rate = 8000
    time = np.arange(3 * rate) / rate
    path = tmp_path / "Biased tone.wav"
    sf.write(path, 0.2 + 0.02 * np.sin(2 * np.pi * 400 * time), rate, subtype="DOUBLE")
    track = load_track(str(path))
    track.repair_overrides["dc_remove"] = True
    pipeline = Pipeline([_BiasWarning(), _GatedBiasMeter()])
    session = SessionContext([track], {})
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    original = track.detector_results[_BiasWarning.id]
    assert original.applicable and original.severity == Severity.ATTENTION
    assert original.data["bias"] == pytest.approx(0.2)
    corrected = track.corrected_analysis.results
    assert not corrected[_GatedBiasMeter.id].applicable
    assert not corrected[_BiasWarning.id].applicable
