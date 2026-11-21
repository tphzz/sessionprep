"""A track offers repairs for measured findings, saved settings and recovery."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from sessionprepgui.detail import resolve_audition_results
from sessionprepgui.detail.repair_options import track_repair_options
from sessionprepgui.detail.report import prepare_track_report
from sessionpreplib.models import CorrectedAnalysis, DetectorResult, SessionContext, Severity, TrackContext
from sessionpreplib.repair_verification import CheckOutcome
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture
def track():
    return TrackContext("Synth.wav", "missing.wav", None, 96000, 2, 96000, "24-bit", "PCM_24", 1)


def finding(name, **data):
    return DetectorResult(name, Severity.ATTENTION, name, data)


@pytest.mark.parametrize(("key", "data", "rate", "expected"), [
    ("stereo_compat", {"corr_warn": True}, 96000, set()),
    ("dc_offset", {"dc_warn": True}, 44100, {"dc_remove"}),
    ("dc_offset", {"dc_warn": False}, 44100, set()),
    ("subsonic", {"subsonic_warn": True}, 44100, {"low_cut"}),
    ("ultrasonic", {"regions": []}, 96000, set()),
    ("ultrasonic", {"regions": [[0, 999, 0]]}, 49999, set()),
    ("ultrasonic", {"regions": [[0, 999, 0]]}, 50000, {"ultrasonic"}),
])
def test_only_matching_evidence_opens_controls(track, key, data, rate, expected):
    track.samplerate = rate
    track.detector_results[key] = finding(key, **data)
    config = {"repair_auto_dc": True, "repair_auto_low_cut": True, "repair_auto_ultrasonic": True}
    session = SessionContext([track], config)
    assert track_repair_options(track, session).rows == expected
    assert prepare_track_report(track, session)["can_review_repairs"] is bool(expected)
    for setting in ({f"{key}_enabled": False}, {f"{key}_report_as": "skip"}):
        assert not track_repair_options(track, SessionContext([track], config | setting)).available


@pytest.mark.parametrize(("key", "value", "row"), [
    ("dc_remove", False, "dc_remove"), ("low_cut_hz", 10, "low_cut"), ("low_cut", True, "low_cut"),
    ("ultrasonic", False, "ultrasonic"),
])
def test_saved_track_setting_always_retains_its_own_reset(track, key, value, row):
    track.samplerate = 44100
    track.repair_overrides[key] = value
    options = track_repair_options(track)
    assert options.rows == {row} and not options.targets


def test_new_corrected_finding_and_stale_or_failed_verification(track):
    track.detector_results["ultrasonic"] = replace(finding("ultrasonic", regions=[]), severity=Severity.CLEAN)
    track.repair_recipe = RepairRecipe(dc_remove=True)
    track.corrected_analysis = CorrectedAnalysis("verified", {"ultrasonic": finding("ultrasonic", regions=[[0, 10, 0]])})
    assert track_repair_options(track).rows == {"dc_remove", "ultrasonic"}
    track._cache["repair_verification_pending"] = True
    assert track_repair_options(track).rows == {"dc_remove"}
    track._cache.clear()
    track._cache["repair_error"] = "Source unavailable"
    options = track_repair_options(track)
    assert options.rows == {"dc_remove"} and options.errors == ["Source unavailable"]
    track._cache.clear()
    track.repair_recipe = RepairRecipe()
    assert not track_repair_options(track).available


def test_multiple_active_repairs_do_not_claim_an_unproven_cause(track):
    track.repair_recipe = RepairRecipe(dc_remove=True, ultrasonic=True)
    track.detector_results["subsonic"] = finding("subsonic", subsonic_warn=True)
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": replace(
        finding("subsonic", subsonic_warn=False), severity=Severity.CLEAN)})
    options = track_repair_options(track)
    assert options.rows == {"dc_remove", "ultrasonic"}
    assert "subsonic" not in options.targets and not options.attribution("subsonic")


@pytest.mark.parametrize("global_result", [None, DetectorResult("subsonic", Severity.PROBLEM, "failed", {}, error="failed")])
@pytest.mark.parametrize("source", ["original", "corrected"])
def test_local_measurements_survive_missing_global_detector_in_file_and_plot(track, global_result, source):
    session = SessionContext([track], {})
    if global_result:
        track.detector_results["subsonic"] = global_result
    track.repair_recipe = RepairRecipe(dc_remove=True)
    track.corrected_analysis = CorrectedAnalysis("verified", dict(track.detector_results))
    track.repair_measurements = {side: {"low_band": {"regions": [[10, 50, 1]] if side == source else []}}
                                 for side in ("original", "corrected")}
    assert track_repair_options(track, session).targets["subsonic-local"] == "low_cut"
    rows = {r["id"]: r for r in prepare_track_report(track, session, show_clean=False)["rows"]}
    local = rows["subsonic-local"]
    assert local["review_repair"]
    assert local["outcome"] == (CheckOutcome.NEW if source == "corrected" else
                                CheckOutcome.FAILED if global_result else CheckOutcome.PENDING)
    assert ("subsonic" in rows) is bool(global_result)
    for side in ("original", "corrected"):
        result = resolve_audition_results(track, session, corrected=(side == "corrected")).get("subsonic-local")
        assert bool(result and result.issues) is (side == source)
    session.config["subsonic_enabled"] = False
    assert "subsonic-local" not in track_repair_options(track, session).targets
    assert "subsonic-local" not in resolve_audition_results(track, session, corrected=(source == "corrected"))
    assert all(r["id"] != "subsonic-local" for r in prepare_track_report(track, session)["rows"])


def test_output_recovery_is_accessible_only_on_the_affected_track(track):
    other = replace(track, filename="Other.wav")
    session = SessionContext([track, other], {"_prepare_errors": [(track.filename, "Could not write output")]})
    options = track_repair_options(track, session)
    assert options.available and not options.rows and options.errors == ["Could not write output"]
    assert not track_repair_options(other, session).available
    session.config = {"_output_gain_adjustments": {track.filename: {"peak_db": 1.3, "reduction_db": 1.4}}}
    session.prepare_state = "stale"
    session.output_tracks = [replace(track, processed_filepath="previous.wav")]
    options = track_repair_options(track, session)
    assert options.available and options.has_stale_output and options.gain_adjustment["reduction_db"] == 1.4
    assert not track_repair_options(other, session).available


@pytest.mark.parametrize("state", [{"repair_verification_pending": True}, {"repair_error": "Source missing"}])
def test_first_pending_or_failed_edit_does_not_copy_original_local_overlay_to_corrected(track, state):
    track.repair_measurements = {"original": {"low_band": {"regions": [[11, 66, 0]]}}}
    track._cache.update(state)
    session = SessionContext([track], {})
    assert not resolve_audition_results(track, session, corrected=True)
    assert resolve_audition_results(track, session, corrected=False)["subsonic-local"].issues
