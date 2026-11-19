"""A repair earns a resolved finding only on a successful, applicable measurement."""
import pytest

from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import CorrectedAnalysis, DetectorResult, IssueLocation, SessionContext, Severity, TrackContext
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.repair_verification import CheckOutcome, audition_results, detector_check, remaining_results
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture
def track():
    return TrackContext("synth.wav", "synth.wav", None, 44100, 2, 44100, "24-bit", "PCM_24", 1,
                        repair_recipe=RepairRecipe(dc_remove=True))


def result(detector, severity, **overrides):
    return DetectorResult(detector, severity, detector, {}, **overrides)


@pytest.mark.parametrize(("before", "after", "outcome"), [
    (Severity.ATTENTION, Severity.CLEAN, CheckOutcome.RESOLVED),
    (Severity.INFO, Severity.INFO, CheckOutcome.REMAINING),
    (Severity.CLEAN, Severity.PROBLEM, CheckOutcome.NEW),
    (Severity.CLEAN, Severity.CLEAN, CheckOutcome.CLEAN),
])
def test_verdict_depends_on_measured_results(track, before, after, outcome):
    original = result("dc_offset", before)
    corrected = result("dc_offset", after)
    track.detector_results["dc_offset"] = original
    track.corrected_analysis = CorrectedAnalysis("key", {"dc_offset": corrected})
    check = detector_check(track, "dc_offset")
    assert (check.original, check.corrected, check.outcome) == (original, corrected, outcome)
    assert check.remaining_result is corrected


@pytest.mark.parametrize(("after", "outcome"), [
    (None, CheckOutcome.PENDING),
    (result("dc_offset", Severity.CLEAN, applicable=False), CheckOutcome.INAPPLICABLE),
    (result("dc_offset", Severity.PROBLEM, error="Cannot measure"), CheckOutcome.FAILED),
])
def test_unverified_detector_cannot_clear_existing_finding(track, after, outcome):
    original = result("dc_offset", Severity.ATTENTION)
    track.detector_results["dc_offset"] = original
    if after:
        track.corrected_analysis = CorrectedAnalysis("key", {"dc_offset": after})
    check = detector_check(track, "dc_offset")
    assert check.outcome == outcome
    assert check.remaining_result is original


def test_source_switch_and_summary_use_the_same_measurements(track):
    detectors = default_detectors()
    dc = DetectorResult("dc_offset", Severity.ATTENTION, "DC offset", {"dc_warn": True, "dc_db": -37.6})
    low = DetectorResult("subsonic", Severity.INFO, "Low frequency", {"subsonic_warn": True, "subsonic_ratio_db": -10.6})
    ranges = [(start, start + 99) for start in (10, 1000, 2000, 3000)]
    stereo = DetectorResult("stereo_compat", Severity.ATTENTION, "Window correlation",
                            {"corr_warn": False, "mono_warn": False, "windowed_regions": ranges},
                            issues=[IssueLocation(a, b, None, Severity.ATTENTION, "stereo", "Correlation") for a, b in ranges])
    track.detector_results = {"dc_offset": dc, "subsonic": low, "stereo_compat": stereo}
    track.corrected_analysis = CorrectedAnalysis("key", {"dc_offset": result("dc_offset", Severity.CLEAN),
                                                         "subsonic": result("subsonic", Severity.CLEAN),
                                                         "stereo_compat": stereo})
    session = SessionContext([track], {}, detectors=detectors)
    for corrected in (True, False, True):
        selected = audition_results(track, corrected=corrected)
        assert selected["subsonic"].severity == (Severity.CLEAN if corrected else Severity.INFO)
        assert selected["stereo_compat"].issues == stereo.issues
        summary = build_diagnostic_summary(session, remaining=True)
        assert [g["detector_id"] for g in summary["attention"]] == ["stereo_compat"]
        assert summary["attention"][0]["item_refs"][0]["detail"] == "4 marked locations"
        assert not summary["information"] and not summary["problems"] and summary["clean_count"] == 0
    assert audition_results(track, corrected=True, pending=True) == {}
    assert remaining_results(track, detectors, pending=True)["subsonic"] is low
    track.repair_recipe = RepairRecipe()
    assert detector_check(track, "subsonic").outcome == CheckOutcome.ORIGINAL
    assert audition_results(track, corrected=True) is track.detector_results


def test_missing_or_failed_baseline_cannot_earn_a_success_or_new_label(track):
    for original in (None, result("dc_offset", Severity.PROBLEM, error="Original unreadable"),
                     result("dc_offset", Severity.CLEAN, applicable=False)):
        track.detector_results = {"dc_offset": original} if original else {}
        track.corrected_analysis = CorrectedAnalysis("key", {"dc_offset": result("dc_offset", Severity.CLEAN)})
        assert detector_check(track, "dc_offset").outcome == CheckOutcome.INAPPLICABLE
        track.corrected_analysis = CorrectedAnalysis("key", {"dc_offset": result("dc_offset", Severity.ATTENTION)})
        assert detector_check(track, "dc_offset").outcome == CheckOutcome.REMAINING


def test_summary_withholds_clean_count_while_repaired_audio_is_unverified(track):
    session = SessionContext([track], {}, detectors=default_detectors())
    pending = build_diagnostic_summary(session, remaining=True)
    assert pending["clean_count"] == 0 and pending["verification_pending"] == [track.filename]
    track._cache["repair_error"] = "Cannot read source"
    failed = build_diagnostic_summary(session, remaining=True)
    assert failed["clean_count"] == 0 and failed["verification_failed"] == [track.filename]
