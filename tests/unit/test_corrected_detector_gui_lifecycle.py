"""Drive paired verification through the Qt project, edit and worker entry points."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.session.io import save_session
from sessionpreplib.audio import load_track
from sessionpreplib.daw_processors import protools
from sessionpreplib.detectors import DCOffsetDetector, default_detectors
from sessionpreplib.models import SessionContext, Severity
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.repairs import RepairRecipe


def _wait(predicate):
    deadline = monotonic() + 8
    while not predicate() and monotonic() < deadline:
        QApplication.processEvents()
        sleep(0.005)
    assert predicate(), "Timed out waiting for Qt to publish the result"


def _idle(window):
    _wait(lambda: window.lifecycle_controller.repair_plan_worker is None and not window.lifecycle_controller.repair_plan_pending)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    monkeypatch.setattr(protools, "check_ptsl_connectivity", lambda *args, **kwargs: (False, "offline"))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    settings.save_config(settings.build_defaults())
    win = SessionPrepWindow()
    try:
        yield win
    finally:
        _idle(win)
        dispose_window(win)


def _save_fixture(window, folder, *, active):
    folder.mkdir()
    rate = 16000
    time = np.arange(3 * rate) / rate
    path = folder / "Bias.wav"
    sf.write(path, 0.125 + 0.2 * np.sin(2 * np.pi * 400 * time), rate, subtype="DOUBLE")
    track = load_track(str(path))
    if active:
        track.repair_overrides["dc_remove"] = True
    config = window.state.flat_config()
    session = SessionContext([track], config)
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    project = folder / "record.spsession"
    save_session(str(project), {"source_dir": str(folder), "tracks": [track], "active_config_preset": "Default",
                                "session_config": window.state.effective_session_config(), "session_config_overrides": {}})
    return project


def _open(window, monkeypatch, project):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(project), ""))
    window.lifecycle_controller.load_session_dialog()
    return window.state.session.tracks[0]


def _edit(window, track, key, value):
    row = window.track_table_controller.find_table_row(track.filename)
    processing = window._track_table.cellWidget(row, 7)
    next(a for a in processing.menu().actions() if a.text() == "Track repairs…").trigger()
    editor = window._repair_editor
    assert editor.track is track
    choice = editor.rows[key].choice
    choice.setCurrentIndex(choice.findData(value))


@contextmanager
def _hold_completed_plan(monkeypatch):
    """Control delivery order after the real planner has finished the signal checks."""
    ready, release = Event(), Event()
    plans = []
    plan = Pipeline.plan

    def hold(self, session, **kwargs):
        result = plan(self, session, **kwargs)
        if not ready.is_set():
            plans.append(session.tracks[0])
            ready.set()
            if not release.wait(10):
                raise RuntimeError("Timed out delivering a completed test plan")
        return result

    monkeypatch.setattr(Pipeline, "plan", hold)
    try:
        yield ready, plans
    finally:
        release.set()


def _count_dc_calls(monkeypatch):
    calls = []
    analyze = DCOffsetDetector.analyze

    def count(self, track):
        calls.append(get_ident())
        return analyze(self, track)

    monkeypatch.setattr(DCOffsetDetector, "analyze", count)
    return calls


def _set_dc_threshold(window, monkeypatch, value):
    def edit(dialog):
        field = dict(dialog._cfg_widgets["detectors.dc_offset"])["dc_offset_warn_db"]
        field.setValue(value)
        dialog._save_btn.click()
        assert dialog.saved
        return dialog.result()

    with monkeypatch.context() as local:
        local.setattr(PreferencesDialog, "exec", edit)
        local.setattr(QMessageBox, "information", lambda *args: QMessageBox.Ok)
        window._on_preferences()


def test_normal_repair_worker_publishes_verdict_and_file_navigation_reuses_it(window, tmp_path, monkeypatch):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=False))
    calls = _count_dc_calls(monkeypatch)
    _edit(window, track, "dc_remove", True)
    _idle(window)
    assert track.corrected_analysis is not None
    assert track.detector_results["dc_offset"].severity == Severity.ATTENTION
    assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    assert calls and get_ident() not in calls
    verified_calls = list(calls)
    snapshot = track.corrected_analysis
    window._repair_editor.back_button.click()
    preview = window._repair_preview
    _wait(lambda: preview.displayed and preview.worker is None)
    for corrected in (True, False, True):
        window._wf_panel.source_selector.buttons[int(corrected)].click()
        _wait(lambda: not preview.pending and preview.worker is None)
        assert window._wf_panel.corrected == corrected
        assert track.corrected_analysis is snapshot
    row = window.track_table_controller.find_table_row(track.filename)
    window.track_table_controller.select_row(row)
    window.detail_controller.show_track_detail(track)
    assert calls == verified_calls


def test_preferences_recheck_both_sources_of_the_open_project(window, tmp_path, monkeypatch):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=True))
    saved_key = track.corrected_analysis.key
    calls = _count_dc_calls(monkeypatch)
    _set_dc_threshold(window, monkeypatch, -3.0)
    _idle(window)
    assert window.state.session.tracks[0] is track
    assert track.corrected_analysis.key != saved_key
    assert track.detector_results["dc_offset"].severity == Severity.CLEAN
    assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    assert len(calls) == 2 and get_ident() not in calls


def test_bypassing_the_last_repair_restores_source_findings_before_publication(window, tmp_path, monkeypatch):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=True))
    assert track.corrected_analysis is not None
    original = build_diagnostic_summary(window.state.session)
    with _hold_completed_plan(monkeypatch) as (ready, _):
        _edit(window, track, "dc_remove", False)
        _wait(ready.is_set)
        assert track.corrected_analysis is None
        assert window.state.summary["attention"] == original["attention"]
    _idle(window)
    assert not track.repair_recipe.active and track.corrected_analysis is None


@pytest.mark.parametrize("enable", [True, False])
def test_interrupted_repair_finishes_if_full_reanalysis_fails(window, tmp_path, monkeypatch, enable):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=not enable))

    def fail_load(*args, **kwargs):
        raise OSError("Full analysis could not load the session")

    monkeypatch.setattr("sessionprepgui.analysis.worker.load_session", fail_load)
    with _hold_completed_plan(monkeypatch) as (ready, _):
        _edit(window, track, "dc_remove", enable)
        _wait(ready.is_set)
        assert window._analyze_action.isEnabled()
        window._analyze_action.trigger()
        _wait(lambda: window.lifecycle_controller.analyze_worker is None)
    _idle(window)
    assert window.state.session.tracks[0] is track
    assert window._save_session_action.isEnabled() and window._prepare_action.isEnabled()
    assert "repair_verification_pending" not in track._cache
    assert track.repair_recipe.active == enable
    assert (track.corrected_analysis is not None) == enable


@pytest.mark.parametrize("restored", ["same audio", "different audio", "different sample rate", "unknown digest"])
def test_file_retry_recovers_only_the_original_audio(window, tmp_path, monkeypatch, restored):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=False))
    path = Path(track.filepath)
    previous = track.source_file_signature
    moved = path.with_suffix(".away")
    path.rename(moved)
    _edit(window, track, "dc_remove", True)
    _idle(window)
    assert track._cache.get("repair_error") and track.corrected_analysis is None

    moved.rename(path)
    audio, rate = sf.read(path, dtype="float64")
    if restored == "different audio":
        sf.write(path, audio * 0.5, rate, subtype="DOUBLE")
    elif restored == "different sample rate":
        sf.write(path, audio, rate * 2, subtype="DOUBLE")
    elif restored == "unknown digest":
        track.source_sample_digest = None
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, previous[1] + 2_000_000_000))
    window.lifecycle_controller.queue_repair_plan([track])
    _idle(window)
    assert track._cache.get("repair_error") and track.source_file_signature == previous
    window._file_report.verificationRetryRequested.emit()
    _idle(window)
    if restored == "same audio":
        assert not track._cache.get("repair_error") and track.corrected_analysis is not None
        assert track.source_file_signature == (path.stat().st_size, path.stat().st_mtime_ns)
        assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    else:
        assert track._cache.get("repair_error") and track.corrected_analysis is None
        assert track.source_file_signature == previous


def test_gui_owned_samples_cannot_change_after_verification_while_publication_is_queued(window, tmp_path, monkeypatch):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=False))
    audio, _ = sf.read(track.filepath, dtype="float64")
    owner = audio.copy()
    track.audio_data = owner.view()
    with _hold_completed_plan(monkeypatch) as (ready, plans):
        _edit(window, track, "dc_remove", True)
        _wait(ready.is_set)
        with pytest.raises(ValueError, match="read-only"):
            track.audio_data *= 0.01
        with pytest.raises(ValueError, match="read-only"):
            owner *= 0.01
        key = plans[0].corrected_analysis.key
    _idle(window)
    assert track.corrected_analysis.key == key


@pytest.mark.parametrize("change", [
    "new edit", "replaced track", "rewritten wav", "replaced resident buffer", "loaded different buffer",
])
def test_completed_verification_cannot_overwrite_newer_state(window, tmp_path, monkeypatch, change):
    track = _open(window, monkeypatch, _save_fixture(window, tmp_path / "audio", active=False))
    buffer_changes = ("replaced resident buffer", "loaded different buffer")
    if change == "replaced resident buffer":
        track.audio_data, _ = sf.read(track.filepath, dtype="float64")
    elif change == "loaded different buffer":
        assert track.audio_data is None
    replacement = None
    deliveries = []
    refresh = window.track_table_controller.update_track_row

    def observed(filename):
        current = window.state.session.tracks[0]
        if current.corrected_analysis is not None:
            deliveries.append(current.corrected_analysis.key)
        return refresh(filename)

    monkeypatch.setattr(window.track_table_controller, "update_track_row", observed)
    with _hold_completed_plan(monkeypatch) as (ready, plans):
        _edit(window, track, "dc_remove", True)
        _wait(ready.is_set)
        old = plans[0].corrected_analysis
        assert old is not None and old.results["dc_offset"].severity == Severity.CLEAN
        if change == "new edit":
            _edit(window, track, "low_cut", True)
        elif change == "replaced track":
            replacement = load_track(track.filepath)
            window.state.session.tracks[0] = replacement
        elif change in buffer_changes:
            signal, _ = sf.read(track.filepath, dtype="float64")
            track.audio_data = signal - np.mean(signal)
        else:
            path = Path(track.filepath)
            signal, rate = sf.read(path, dtype="float64")
            sf.write(path, signal * 0.5, rate, subtype="DOUBLE")
    _idle(window)
    assert old.key not in deliveries
    if change == "new edit":
        assert track.repair_recipe == RepairRecipe(dc_remove=True, low_cut_hz=10)
        assert track.corrected_analysis is not None and track.corrected_analysis.key != old.key
    elif change == "replaced track":
        assert replacement.corrected_analysis is None
    elif change in buffer_changes:
        assert track.detector_results["dc_offset"].severity == Severity.CLEAN
        assert track.corrected_analysis is not None and track.corrected_analysis.key != old.key
    else:
        assert track.corrected_analysis is None or track.corrected_analysis.key != old.key


@pytest.mark.parametrize("saved_state", ["verified", "legacy", "incomplete", "changed threshold"])
def test_open_project_repairs_missing_verification_and_reuses_complete_results(window, tmp_path, monkeypatch, saved_state):
    project = _save_fixture(window, tmp_path / "audio", active=True)
    state = json.loads(project.read_text(encoding="utf-8"))
    record = state["tracks"]["Bias.wav"]
    saved_key = record["corrected_analysis"]["key"]
    if saved_state == "legacy":
        state["version"] = 6
        for field in ("corrected_analysis", "detector_analysis_key", "detector_sample_digest", "source_file_revision"):
            record.pop(field, None)
    elif saved_state == "incomplete":
        del record["corrected_analysis"]["results"]["dc_offset"]
    elif saved_state == "changed threshold":
        _set_dc_threshold(window, monkeypatch, -3.0)
    if saved_state in ("legacy", "incomplete"):
        project.write_text(json.dumps(state), encoding="utf-8")
    calls = _count_dc_calls(monkeypatch)
    track = _open(window, monkeypatch, project)
    _idle(window)
    assert track.corrected_analysis is not None
    assert (track.corrected_analysis.key != saved_key) == (saved_state == "changed threshold")
    assert track.corrected_analysis.results["dc_offset"].severity == Severity.CLEAN
    expected = Severity.CLEAN if saved_state == "changed threshold" else Severity.ATTENTION
    assert track.detector_results["dc_offset"].severity == expected
    if saved_state == "verified":
        assert calls == []
    else:
        assert calls and get_ident() not in calls
