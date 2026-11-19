"""Exercise corrected-detector reuse, invalidation and failure through the planner."""
from collections import Counter
import os

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.detector import TrackDetector
from sessionpreplib.detectors import ClippingDetector, DCOffsetDetector, SilenceDetector, default_detectors
from sessionpreplib.models import DetectorResult, ParamSpec, SessionContext, Severity
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import render_corrected


class _PeakProbe(TrackDetector):
    id = "test_peak_probe"
    name = "Peak probe"

    @classmethod
    def html_help(cls):
        return "Measures sample peak."

    @classmethod
    def config_params(cls):
        return [*super().config_params(), ParamSpec("test_peak_limit", float, 0.01, "Peak limit")]

    def configure(self, config):
        super().configure(config)
        self.limit = config.get("test_peak_limit", 0.01)

    def analyze(self, track):
        peak = float(np.max(np.abs(track.audio_data)))
        severity = Severity.ATTENTION if peak > self.limit else Severity.CLEAN
        return DetectorResult(self.id, severity, "Sample peak", {"peak": peak, "limit": self.limit})


class _DependentProbe(TrackDetector):
    id = "test_dependent_probe"
    name = "Dependent probe"
    depends_on = [_PeakProbe.id]

    @classmethod
    def html_help(cls):
        return "Uses the peak probe's measurement."

    def analyze(self, track):
        # A dependent detector may return CLEAN for a missing measurement.
        # Verification must keep such an answer from being mistaken for a repair.
        peak = track.detector_results[_PeakProbe.id].data.get("peak", 0.0)
        severity = Severity.ATTENTION if peak > 0.01 else Severity.CLEAN
        return DetectorResult(self.id, severity, "Dependent peak", {"peak": peak})


@pytest.fixture
def track(tmp_path):
    rate = 16000
    time = np.arange(2 * rate) / rate
    path = tmp_path / "source.wav"
    sf.write(path, 0.125 + 0.25 * np.sin(2 * np.pi * 330 * time), rate, subtype="DOUBLE")
    return load_track(str(path))


def _observe_detectors(monkeypatch, track, fail=None, *, kinds=(_PeakProbe, _DependentProbe, ClippingDetector)):
    calls = Counter()
    for kind in kinds:
        analyze = kind.analyze

        def observed(self, target, original_method=analyze):
            source = "original" if np.array_equal(target.audio_data, track.audio_data) else "corrected"
            calls[self.id, source] += 1
            if fail is not None and fail(self.id, source):
                raise RuntimeError("peak meter unavailable")
            return original_method(self, target)

        monkeypatch.setattr(kind, "analyze", observed)
    return calls


def _pipeline(config):
    # Register the dependent first to exercise detector-declared ordering.
    detectors = [*default_detectors(), _DependentProbe(), _PeakProbe()]
    return Pipeline(detectors, default_processors(), config, max_workers=1)


def _analyze(track):
    session = SessionContext([track], {})
    pipeline = _pipeline(session.config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    return pipeline, session


def test_active_replay_reuses_measurements_and_bypass_does_no_corrected_work(track, monkeypatch):
    calls = _observe_detectors(monkeypatch, track)
    pipeline, session = _analyze(track)
    original_calls = calls.copy()
    pipeline.plan(session)
    assert track.corrected_analysis is None
    assert calls == original_calls

    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)
    checked = track.corrected_analysis
    assert checked is not None
    assert checked.results[_DependentProbe.id].data["peak"] == checked.results[_PeakProbe.id].data["peak"]
    for kind in (_PeakProbe, _DependentProbe, ClippingDetector):
        assert calls[kind.id, "original"] == 1
        assert calls[kind.id, "corrected"] == 1

    checked_calls = calls.copy()
    pipeline.plan(session)
    track.manual_gain_db = -9.0
    session.config["clipping_report_as"] = "attention"
    _pipeline(session.config).plan(session)
    assert track.corrected_analysis.key == checked.key
    assert calls == checked_calls
    assert track.processor_results["bimodal_normalize"].gain_db == -9.0

    track.repair_overrides["dc_remove"] = False
    pipeline.plan(session)
    assert not track.repair_recipe.active
    assert track.corrected_analysis is None
    assert calls == checked_calls


def test_changed_detector_settings_and_effective_recipe_invalidate_measurements(track, monkeypatch):
    calls = _observe_detectors(monkeypatch, track)
    pipeline, session = _analyze(track)
    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)
    first = track.corrected_analysis
    assert first.results[_PeakProbe.id].severity == Severity.ATTENTION

    session.config["test_peak_limit"] = 0.99
    pipeline = _pipeline(session.config)
    pipeline.plan(session)
    changed_limit = track.corrected_analysis
    assert changed_limit.key != first.key
    assert track.detector_results[_PeakProbe.id].severity == Severity.CLEAN
    assert track.detector_results[_PeakProbe.id].data["limit"] == 0.99
    assert changed_limit.results[_PeakProbe.id].severity == Severity.CLEAN
    assert changed_limit.results[_PeakProbe.id].data["limit"] == 0.99
    assert calls[_PeakProbe.id, "corrected"] == 2

    track.repair_overrides.update(low_cut=True, low_cut_hz=18.0)
    pipeline.plan(session)
    changed_audio = track.corrected_analysis
    expected_audio = render_corrected(track.audio_data, track.samplerate, track.repair_recipe)
    assert changed_audio.key != changed_limit.key
    assert changed_audio.results[_PeakProbe.id].data["peak"] == pytest.approx(np.max(np.abs(expected_audio)))
    assert calls[_PeakProbe.id, "corrected"] == 3


def test_standalone_phase_two_reuses_its_original_checks_and_their_prerequisites(track, monkeypatch):
    calls = _observe_detectors(monkeypatch, track, kinds=(SilenceDetector, DCOffsetDetector))
    pipeline = _pipeline({})
    session = SessionContext([track], {})
    pipeline.analyze_phase2(session)
    assert track.detector_results["dc_offset"].severity == Severity.ATTENTION
    assert calls == {("silence", "original"): 1, ("dc_offset", "original"): 1}
    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)
    assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    assert calls == {(kind.id, source): 1 for kind in (SilenceDetector, DCOffsetDetector)
                     for source in ("original", "corrected")}


def test_changing_the_registered_detector_set_releases_removed_results(track):
    pipeline, session = _analyze(track)
    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)
    original_set = track.corrected_analysis
    assert _DependentProbe.id in original_set.results

    reduced = Pipeline(default_detectors(), default_processors(), session.config, max_workers=1)
    reduced.plan(session)
    assert track.corrected_analysis.key != original_set.key
    assert _PeakProbe.id not in track.corrected_analysis.results
    assert _DependentProbe.id not in track.corrected_analysis.results
    assert ClippingDetector.id in track.corrected_analysis.results


def test_replaced_source_invalidates_previous_verification(track):
    pipeline, session = _analyze(track)
    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)
    assert track.corrected_analysis.results[_PeakProbe.id].applicable

    old = os.stat(track.filepath)
    sf.write(track.filepath, -track.audio_data, track.samplerate, subtype="DOUBLE")
    os.utime(track.filepath, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
    pipeline.plan(session)
    assert "changed" in track._cache["repair_error"].lower()
    assert track.corrected_analysis is None


def test_detector_failure_isolated_and_cannot_clear_a_dependent_finding(track, monkeypatch):
    unavailable = True

    def fail(detector_id, source):
        return unavailable and detector_id == _PeakProbe.id and source == "corrected"

    calls = _observe_detectors(monkeypatch, track, fail)
    pipeline, session = _analyze(track)
    track.repair_overrides["dc_remove"] = True
    pipeline.plan(session)

    assert track.detector_results[_DependentProbe.id].severity == Severity.ATTENTION
    results = track.corrected_analysis.results
    assert "peak meter unavailable" in results[_PeakProbe.id].error
    dependent = results[_DependentProbe.id]
    assert dependent.error or not dependent.applicable
    assert results[ClippingDetector.id].applicable
    assert results[ClippingDetector.id].error is None
    assert track.processor_results["bimodal_normalize"].error is None

    unavailable = False
    pipeline.plan(session)
    recovered = track.corrected_analysis.results
    assert recovered[_PeakProbe.id].error is None
    assert recovered[_DependentProbe.id].applicable
    assert recovered[_DependentProbe.id].severity == Severity.ATTENTION
    assert recovered[_DependentProbe.id].data["peak"] == recovered[_PeakProbe.id].data["peak"]
    assert calls[_PeakProbe.id, "corrected"] == 2
