"""Exercise invalidation and persistence using real audio and the Phase 2 window."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from test_corrected_detector_gui_lifecycle import (
    _edit, _hold_completed_plan, _idle, _open, _save_fixture, _wait, app, window,
)
from sessionprepgui.detail.verification import comparison_readings
from sessionprepgui.helpers import track_analysis_label
from sessionprepgui.session.io import load_session, save_session
from sessionpreplib.audio import load_track
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repair_verification import CheckOutcome, detector_check


@pytest.mark.parametrize("source_change", ["replace", "remove"])
def test_waveform_source_failure_revokes_published_evidence(window, tmp_path, monkeypatch, source_change):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=True))
    _idle(window)
    window.track_table_controller.select_row(window.track_table_controller.find_table_row(track.filename))
    window.detail_controller.show_track_detail(track)
    preview = window._repair_preview
    _wait(lambda: preview.displayed and preview.worker is None)
    assert detector_check(track, "dc_offset").outcome == CheckOutcome.RESOLVED
    assert "Correction applied" in window._file_report.toPlainText()
    assert window.state.summary["clean_count"] == 1

    source = Path(track.filepath)
    if source_change == "remove":
        source.unlink()
    else:
        samples, rate = sf.read(source, dtype="float64")
        sf.write(source, -0.05 + 0.1 * samples, rate, subtype="DOUBLE")
    window._wf_panel.source_selector.buttons[1].click()
    _wait(lambda: preview.worker is None and window.detail_controller.waveform_unavailable_generation is not None)

    assert track.corrected_analysis is None
    assert detector_check(track, "dc_offset").outcome == CheckOutcome.FAILED
    report = window._file_report.toPlainText()
    assert "Unable to verify" in report and "Correction applied" not in report
    assert "Unable to verify" in track_analysis_label(track, window.state.session.detectors)[0]
    assert window.state.summary["clean_count"] == 0
    window.detail_controller.open_track_repairs(track)
    dc = window._repair_editor.rows["dc_remove"]
    assert dc.verdict.text() == "Unable to verify"
    assert "0.33%" not in dc.evidence.text()
    assert "→ — dBFS" in window._repair_editor.peak.text()


@pytest.mark.parametrize("key,value", [("dc_remove", False), ("low_cut", True)])
def test_repair_meters_change_before_new_plan_is_published(window, tmp_path, monkeypatch, key, value):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=True))
    _idle(window)
    window.detail_controller.open_track_repairs(track)
    editor = window._repair_editor
    dc = editor.rows["dc_remove"]
    assert "12.50%" in dc.evidence.text() and "0.33%" in dc.evidence.text()

    with _hold_completed_plan(monkeypatch) as (ready, _):
        _edit(window, track, key, value)
        _wait(ready.is_set)
        assert track.corrected_analysis is None
        assert "0.33%" not in dc.evidence.text()
        assert dc.evidence.text().count("12.50%") == 1 and "—" in dc.evidence.text()
        if key == "dc_remove":
            assert editor.peak_box.isHidden()
        else:
            assert dc.verdict.text() == "Checking…"
            assert not editor.peak_box.isHidden() and "→ — dBFS" in editor.peak.text()
    _idle(window)


def _verified_audio(source, samples, rate, config):
    sf.write(source, samples, rate, subtype="DOUBLE")
    track = load_track(str(source))
    track.repair_overrides["dc_remove"] = True
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    session = SessionContext([track], config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    assert track.corrected_analysis is not None
    return track


def test_verified_tail_ending_on_the_final_sample_survives_reopening(tmp_path):
    rate = 8000
    time = np.arange(20 * rate) / rate
    samples = 0.08 * np.sin(2 * np.pi * 440 * time) + 0.02
    samples[-rate // 5:] *= 8
    track = _verified_audio(tmp_path / "final-tail.wav", samples, rate, {})

    project = tmp_path / "final-tail.spsession"
    save_session(str(project), {"source_dir": str(tmp_path), "tracks": [track]})
    restored = load_session(str(project))["tracks"][0]
    assert restored.corrected_analysis is not None
    assert restored.corrected_analysis.key == track.corrected_analysis.key
    for source_track in (track, restored):
        for results in (source_track.detector_results, source_track.corrected_analysis.results):
            spans = [(issue.sample_start, issue.sample_end) for issue in results["tail_exceedance"].issues]
            assert spans == [(155280, len(samples) - 1)]


@pytest.mark.parametrize("detector_id", ["clipping", "dual_mono"])
def test_affected_channels_are_independent_of_overlay_location_count(tmp_path, detector_id):
    rate = 8000
    time = np.arange(2 * rate) / rate
    left = 0.04 + 0.1 * np.sin(2 * np.pi * 182 * time)
    right = left.copy()
    if detector_id == "clipping":
        left[100:110] = 1.1
        right[1000:1010] = 1.2
    config = {"clip_report_max_ranges": 1}
    track = _verified_audio(tmp_path / "stereo.wav", np.column_stack((left, right)), rate, config)
    check = detector_check(track, detector_id)
    assert check.outcome == CheckOutcome.REMAINING
    rows, *_ = comparison_readings(check, track, config)
    assert ("Affected channels", "Left and right", "Left and right") in rows
    if detector_id == "clipping":
        assert ("Clipped ranges", "2", "2") in rows
        assert ("Locations", "1", "1") in rows
        for result in (check.original, check.corrected):
            assert [(i.sample_start, i.sample_end, i.channel) for i in result.issues] == [(100, 109, 0)]
    else:
        assert ("Locations", "0", "0") in rows
        assert check.original.data["dual_mono"] and check.corrected.data["dual_mono"]
