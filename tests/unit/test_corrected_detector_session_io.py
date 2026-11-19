"""Persist paired detector measurements through real project JSON files."""
import json
import math
import os

import numpy as np
import pytest
import soundfile as sf

from sessionprepgui.session.io import load_session, save_session
from sessionpreplib.audio import SourceChangedError, load_track
from sessionpreplib.detectors import ClippingDetector, default_detectors
from sessionpreplib.models import SessionContext, Severity
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors


def _pipeline():
    return Pipeline(default_detectors(), default_processors(), max_workers=1)


@pytest.fixture
def analyzed(tmp_path):
    rate = 16000
    time = np.arange(3 * rate) / rate
    left = 0.12 + 0.16 * np.sin(2 * np.pi * 15 * time) + 0.07 * np.sin(2 * np.pi * 440 * time)
    source = tmp_path / "stereo.wav"
    sf.write(source, np.column_stack((left, np.zeros_like(left))), rate, subtype="DOUBLE")
    track = load_track(str(source))
    track.repair_overrides["dc_remove"] = True
    pipeline = _pipeline()
    session = SessionContext([track], {})
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    return track, pipeline, session


def _write_project(tmp_path, track):
    project = tmp_path / "sample.spsession"
    save_session(str(project), {"source_dir": str(tmp_path), "tracks": [track]})
    return project


def _forbid_nonstandard_json(constant):
    pytest.fail(f"Project contains the non-JSON number {constant}")


def test_replacement_during_decode_invalidates_source_evidence(tmp_path, monkeypatch):
    path = tmp_path / "changing.wav"
    sf.write(path, np.zeros(128), 8000)
    revisions = iter(["before", "after"])
    monkeypatch.setattr("sessionpreplib.audio.file_revision", lambda *_: next(revisions))
    with pytest.raises(SourceChangedError, match="Source changed while it was being read"):
        load_track(str(path))


@pytest.mark.parametrize("change,expected", [
    ("invalid replacement", SourceChangedError), ("removed", FileNotFoundError),
    ("unchanged invalid file", sf.LibsndfileError),
])
def test_decode_failure_distinguishes_changed_source_from_invalid_audio(tmp_path, monkeypatch, change, expected):
    path = tmp_path / "decode.wav"
    sf.write(path, np.zeros(128), 8000)
    if change == "unchanged invalid file":
        path.write_bytes(b"invalid header")
    open_audio = sf.SoundFile

    def open_after_revision_check(*args, **kwargs):
        if change == "invalid replacement":
            path.write_bytes(b"invalid header")
        elif change == "removed":
            path.unlink()
        return open_audio(*args, **kwargs)

    monkeypatch.setattr(sf, "SoundFile", open_after_revision_check)
    with pytest.raises(expected):
        load_track(str(path))


def test_saved_comparison_restores_channels_locations_and_an_immediate_cache_hit(tmp_path, analyzed, monkeypatch):
    original, pipeline, session = analyzed
    pipeline.plan(session)
    project = _write_project(tmp_path, original)
    json.loads(project.read_text(encoding="utf-8"), parse_constant=_forbid_nonstandard_json)

    calls = []
    for detector in pipeline.track_detectors:
        kind = type(detector)
        analyze = kind.analyze

        def counted(self, track, original_method=analyze):
            calls.append(self.id)
            return original_method(self, track)

        monkeypatch.setattr(kind, "analyze", counted)

    restored = load_session(str(project))["tracks"][0]
    assert restored.audio_data is None
    assert restored.corrected_analysis.key == original.corrected_analysis.key
    assert not restored.corrected_analysis.results["ultrasonic"].applicable
    for saved, loaded in ((original.detector_results, restored.detector_results),
                          (original.corrected_analysis.results, restored.corrected_analysis.results)):
        before = saved["subsonic"]
        after = loaded["subsonic"]
        assert set(after.data["per_channel"]) == {0, 1}
        assert after.data["per_channel"][1]["ratio_db"] == -math.inf
        assert after.data == before.data
        assert after.issues == before.issues
        assert any(i.channel == 0 and i.freq_min_hz == 0 and i.freq_max_hz == 30 for i in after.issues)

    _pipeline().plan(SessionContext([restored], {}))
    assert restored.corrected_analysis.key == original.corrected_analysis.key
    assert (restored.processor_results["bimodal_normalize"].gain_db
            == original.processor_results["bimodal_normalize"].gain_db)
    assert calls == []


def test_legacy_project_remeasures_its_original_before_certifying_a_repair(tmp_path, analyzed):
    track, pipeline, session = analyzed
    pipeline.plan(session)
    project = _write_project(tmp_path, track)
    raw = json.loads(project.read_text(encoding="utf-8"))
    raw["version"] = 6
    stored = raw["tracks"][track.filename]
    for field in ("corrected_analysis", "detector_analysis_key", "detector_sample_digest",
                  "source_file_signature", "source_file_revision"):
        stored.pop(field)
    stale_dc = stored["detector_results"]["dc_offset"]
    stale_dc.update(severity="clean", summary="no DC offset issue", data={"dc_warn": False, "dc_db": -120}, issues=[])
    project.write_text(json.dumps(raw), encoding="utf-8")

    reopened = load_session(str(project))["tracks"][0]
    assert reopened.corrected_analysis is None
    _pipeline().plan(SessionContext([reopened], {}))
    assert reopened.detector_results["dc_offset"].severity == Severity.ATTENTION
    corrected = reopened.corrected_analysis.results["dc_offset"]
    assert corrected.applicable and not corrected.error and corrected.severity == Severity.CLEAN
    assert reopened.source_file_signature == track.source_file_signature


@pytest.mark.parametrize("mtime_change", [0, 2_000_000_000], ids=["preserved-mtime", "changed-mtime"])
def test_saved_check_is_withheld_when_the_source_at_the_same_path_was_replaced(tmp_path, analyzed, mtime_change):
    track, pipeline, session = analyzed
    pipeline.plan(session)
    project = _write_project(tmp_path, track)
    old_stat = os.stat(track.filepath)
    sf.write(track.filepath, -track.audio_data, track.samplerate, subtype="DOUBLE")
    os.utime(track.filepath, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + mtime_change))

    reopened = load_session(str(project))["tracks"][0]
    assert reopened.corrected_analysis is None
    _pipeline().plan(SessionContext([reopened], {}))
    assert reopened.corrected_analysis is None
    assert "changed" in reopened._cache["repair_error"].lower()


def test_verified_project_checks_the_routed_file_when_an_input_has_the_same_name(tmp_path, analyzed):
    routed, pipeline, session = analyzed
    pipeline.plan(session)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sf.write(inputs / routed.filename, np.zeros_like(routed.audio_data), routed.samplerate, subtype="DOUBLE")
    project = tmp_path / "routed.spsession"
    save_session(str(project), {"source_dir": str(inputs), "topology_applied": True,
                                "topology_dir": str(tmp_path), "tracks": [routed]})

    reopened = load_session(str(project))["tracks"][0]
    assert reopened.filepath == routed.filepath
    assert reopened.corrected_analysis.key == routed.corrected_analysis.key
    os.remove(routed.filepath)
    missing = load_session(str(project))["tracks"][0]
    assert missing.status == "Error"
    assert missing.corrected_analysis is None


def test_comparison_of_edited_memory_is_not_restored_as_a_measurement_of_the_disk_wav(tmp_path, analyzed):
    track, pipeline, session = analyzed
    pipeline.plan(session)
    disk_peak = track.corrected_analysis.results["audio_classifier"].data["peak_db"]
    track.audio_data *= 0.05
    pipeline.plan(session)
    buffer_peak = track.corrected_analysis.results["audio_classifier"].data["peak_db"]
    assert buffer_peak == pytest.approx(disk_peak + 20 * np.log10(0.05))

    reopened = load_session(str(_write_project(tmp_path, track)))["tracks"][0]
    assert reopened.corrected_analysis is None
    _pipeline().plan(SessionContext([reopened], {}))
    assert reopened.corrected_analysis.results["audio_classifier"].data["peak_db"] == pytest.approx(disk_peak)


def test_empty_wav_has_no_out_of_bounds_region_and_can_restore_its_check(tmp_path):
    source = tmp_path / "empty.wav"
    sf.write(source, np.empty(0), 48000, subtype="PCM_24")
    track = load_track(str(source))
    track.repair_overrides["dc_remove"] = True
    pipeline = _pipeline()
    session = SessionContext([track], {})
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    silence = track.corrected_analysis.results["silence"]
    assert silence.severity == Severity.ATTENTION
    assert silence.issues == []
    reopened = load_session(str(_write_project(tmp_path, track)))["tracks"][0]
    assert reopened.corrected_analysis.results["silence"] == silence


@pytest.mark.parametrize("damage", ["wrong container", "invalid severity", "mismatched detector", "invalid location"])
def test_unreadable_optional_comparison_keeps_the_project_usable(tmp_path, analyzed, damage):
    track, pipeline, session = analyzed
    pipeline.plan(session)
    project = _write_project(tmp_path, track)
    raw = json.loads(project.read_text(encoding="utf-8"))
    snapshot = raw["tracks"][track.filename]["corrected_analysis"]
    if damage == "wrong container":
        snapshot["results"] = []
    elif damage == "invalid severity":
        snapshot["results"]["dc_offset"]["severity"] = "invalid"
    elif damage == "mismatched detector":
        snapshot["results"]["dc_offset"] = snapshot["results"]["subsonic"]
    else:
        snapshot["results"]["subsonic"]["issues"][0]["sample_start"] = "not a sample position"
    project.write_text(json.dumps(raw), encoding="utf-8")

    reopened = load_session(str(project))["tracks"][0]
    assert reopened.repair_overrides == {"dc_remove": True}
    assert reopened.corrected_analysis is None
    _pipeline().plan(SessionContext([reopened], {}))
    assert reopened.corrected_analysis.results["dc_offset"].detector_id == "dc_offset"
    assert reopened.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN


def test_per_detector_failure_survives_reopen_and_an_explicit_retry_can_recover(tmp_path, analyzed, monkeypatch):
    track, pipeline, session = analyzed
    analyze = ClippingDetector.analyze
    unavailable = True

    def fail_on_corrected(self, current):
        if unavailable and not np.array_equal(current.audio_data, track.audio_data):
            raise RuntimeError("analysis interrupted")
        return analyze(self, current)

    monkeypatch.setattr(ClippingDetector, "analyze", fail_on_corrected)
    pipeline.plan(session)
    project = _write_project(tmp_path, track)
    restored = load_session(str(project))["tracks"][0]
    after = restored.corrected_analysis.results
    assert "analysis interrupted" in after["clipping"].error
    assert not after["clipping"].applicable
    assert after["dc_offset"].applicable and after["dc_offset"].severity == Severity.CLEAN

    unavailable = False
    _pipeline().plan(SessionContext([restored], {}))
    retried = restored.corrected_analysis.results["clipping"]
    assert retried.applicable and retried.error is None and retried.severity == Severity.CLEAN
