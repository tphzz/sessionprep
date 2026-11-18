"""Exercise repair inheritance, asynchronous edits and PCM recovery through Qt."""
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QDialog, QDoubleSpinBox, QFileDialog, QMessageBox, QPushButton,
)

from sessionprepgui import settings
from sessionprepgui.analysis.worker import PrepareWorker, RepairPlanWorker
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.session.io import load_session, save_session
from sessionpreplib import pipeline as pipeline_module
from sessionpreplib.audio import load_track
from sessionpreplib.daw_processor import resolve_transfer_audio
from sessionpreplib.daw_processors import protools
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import RepairRecipe, render_corrected


def wait_for(condition, seconds=8):
    end = monotonic() + seconds
    while not condition() and monotonic() < end:
        QApplication.processEvents()
        sleep(0.005)
    assert condition(), "Qt did not reach the expected state"


def idle(window):
    try:
        wait_for(lambda: window._prepare_worker is None and window._repair_plan_worker is None
                 and not window._repair_plan_pending)
    except AssertionError:
        worker = window._repair_plan_worker or window._prepare_worker
        pytest.fail(f"Pending {window._repair_plan_pending}, worker {worker}, "
                    f"finished {worker and worker.isFinished()}, error {getattr(worker, 'failure', None)}")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(app, tmp_path, monkeypatch, request):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    monkeypatch.setattr(protools, "check_ptsl_connectivity", lambda *args, **kwargs: (False, "Test DAW offline"))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: QMessageBox.Ok)
    config = settings.build_defaults()
    preset = config["config_presets"]["Default"]
    preset["processors"]["bimodal_normalize"]["bimodal_normalize_enabled"] = getattr(request, "param", False)
    settings.save_config(config)
    window = SessionPrepWindow()
    source = tmp_path / "audio"
    source.mkdir()
    rate = 8000
    t = np.arange(3 * rate) / rate
    sf.write(source / "Bias.wav", 0.06 + 0.125 * np.sin(2 * np.pi * 150 * t), rate, subtype="PCM_16")
    sf.write(source / "Step.wav", np.where(t < 1.5, -0.875, 0.75), rate, subtype="PCM_16")
    tracks = [load_track(str(source / name)) for name in ("Bias.wav", "Step.wav")]
    flat = window._flat_config()
    session = SessionContext(tracks, flat)
    pipeline = Pipeline(default_detectors(), default_processors(), flat, max_workers=1)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    pipeline.prepare(session, str(source / config["app"]["phase2_output_folder"]))
    window._restore_session_state({
        "source_dir": str(source), "active_config_preset": "Default", "session_config": preset,
        "session_config_overrides": {}, "tracks": tracks, "output_tracks": session.output_tracks,
        "prepare_state": "ready", "transfer_manifest": session.transfer_manifest,
    })
    try:
        yield window
    finally:
        window.close()


def dc_checkbox(widgets):
    return dict(widgets["processors.corrective"])["repair_auto_dc"]


def save_dc_preference(window, monkeypatch, value):
    def edit(dialog):
        dc_checkbox(dialog._cfg_widgets).setChecked(value)
        dialog._save_btn.click()
        assert dialog.saved
        return dialog.result()

    with monkeypatch.context() as local:
        local.setattr(PreferencesDialog, "exec", edit)
        window._on_preferences()
    idle(window)


def open_editor(window, track):
    button = window._track_table.cellWidget(window._find_table_row(track.filename), 7)
    action, = [a for a in button.menu().actions() if a.text() == "Track repairs…"]
    action.trigger()
    assert window._repair_editor.track is track
    return window._repair_editor


def test_project_and_track_inherit_explicit_values_and_reopen(project, tmp_path, monkeypatch):
    window = project
    first, second = window._session.tracks
    field = dc_checkbox(window._session_widgets)
    field.setChecked(True)
    field.setChecked(False)
    idle(window)
    save_dc_preference(window, monkeypatch, True)
    assert not field.isChecked() and not first.repair_recipe.active
    assert window._capture_session_state()["session_config_overrides"] == {
        "processors.corrective.repair_auto_dc": False,
    }

    reset = next(b for b in field.parentWidget().findChildren(QPushButton)
                 if b.accessibleName() == "Use preset value for DC removal")
    reset.click()
    idle(window)
    assert field.isChecked() and all(t.repair_recipe.dc_remove for t in (first, second))
    assert window._capture_session_state()["session_config_overrides"] == {}
    editor = open_editor(window, first)
    choice = editor.rows["dc_remove"].choice
    choice.setCurrentIndex(choice.findData(True))
    idle(window)
    assert first.repair_overrides == {"dc_remove": True}
    assert not second.repair_overrides

    saved = tmp_path / "values.spsession"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(saved), ""))
    window._on_save_session()
    assert load_session(str(saved))["tracks"][0].repair_overrides == {"dc_remove": True}
    save_dc_preference(window, monkeypatch, False)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(saved), ""))
    window._on_load_session()
    idle(window)
    first, second = window._session.tracks
    assert not dc_checkbox(window._session_widgets).isChecked()
    assert first.repair_overrides == {"dc_remove": True} and first.repair_recipe.dc_remove
    assert not second.repair_overrides and not second.repair_recipe.active
    open_editor(window, first).rows["dc_remove"].reset.click()
    idle(window)
    assert first.repair_overrides == {} and not first.repair_recipe.active


def test_rapid_edits_only_publish_the_latest_queued_recipe(project, monkeypatch):
    window = project
    track = window._session.tracks[0]
    editor = open_editor(window, track)
    row = editor.rows["low_cut"]
    dc = editor.rows["dc_remove"]
    entered, release = Event(), Event()
    run = RepairPlanWorker.run

    def gated(worker):
        if not entered.is_set():
            entered.set()
            if not release.wait(6):
                worker.failure = "Test worker timed out"
                return
        run(worker)

    monkeypatch.setattr(RepairPlanWorker, "run", gated)
    try:
        row.choice.setCurrentIndex(row.choice.findData(True))
        wait_for(entered.is_set)
        assert not window._save_session_action.isEnabled() and not window._prepare_action.isEnabled()
        row.choice.setCurrentIndex(row.choice.findData(False))
        dc.choice.setCurrentIndex(dc.choice.findData(True))
    finally:
        release.set()
    idle(window)
    assert track.repair_recipe == RepairRecipe(dc_remove=True)
    assert track.repair_overrides == {"low_cut": False, "dc_remove": True}
    assert window._session.prepare_state == "stale"
    assert window._save_session_action.isEnabled() and window._prepare_action.isEnabled()


def test_switching_project_detaches_a_completed_plan_before_it_can_publish(project, tmp_path, monkeypatch):
    window = project
    old_source = window._session.tracks[0]
    source = tmp_path / "another project"
    source.mkdir()
    # Both projects use the same track name so a stale filename-based update would match.
    path = source / old_source.filename
    sf.write(path, np.sin(2 * np.pi * np.arange(24000) / 80) / 8, 8000, subtype="PCM_16")
    restored = load_track(str(path))
    config = window._flat_config()
    new_session = SessionContext([restored], config)
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    pipeline.analyze_phase2(new_session)
    pipeline.plan(new_session)
    saved = tmp_path / "another.spsession"
    save_session(str(saved), {"source_dir": str(source), "tracks": [restored], "active_config_preset": "Default",
                             "session_config_overrides": {}, "project_name": "Another project"})
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(saved), ""))

    ready, release, finished = Event(), Event(), Event()
    plan = Pipeline.plan

    def hold_result(pipeline, snapshot, **kwargs):
        result = plan(pipeline, snapshot, **kwargs)
        if snapshot.tracks[0].filepath == old_source.filepath:
            ready.set()
            try:
                if not release.wait(6):
                    raise RuntimeError("Test worker timed out")
            finally:
                finished.set()
        return result

    monkeypatch.setattr(Pipeline, "plan", hold_result)
    try:
        choice = open_editor(window, old_source).rows["dc_remove"].choice
        choice.setCurrentIndex(choice.findData(True))
        wait_for(ready.is_set)
        assert window._repair_plan_worker.tracks[0].repair_recipe.dc_remove
        window._on_load_session()
        assert window._session.project_name == "Another project"
        assert not finished.is_set()
        current, = window._session.tracks
        assert current.filepath == str(path) and not current.repair_recipe.active
        low = open_editor(window, current).rows["low_cut"].choice
        low.setCurrentIndex(low.findData(True))
        idle(window)
        assert current.repair_recipe == RepairRecipe(low_cut_hz=10)
    finally:
        release.set()
        if ready.is_set():
            wait_for(finished.is_set)
    wait_for(lambda: not getattr(window, "_zombie_workers", set()))
    QApplication.processEvents()
    assert window._session.tracks[0] is current
    assert current.repair_overrides == {"low_cut": True} and current.repair_recipe == RepairRecipe(low_cut_hz=10)
    assert not old_source.repair_recipe.active


def test_edit_while_preparing_requires_a_new_render_before_transfer(project, monkeypatch):
    window = project
    session = window._session
    track = session.tracks[0]
    row = open_editor(window, track).rows["dc_remove"]
    entered, release = Event(), Event()
    run = PrepareWorker.run

    def gated(worker):
        entered.set()
        if release.wait(6):
            run(worker)
        else:
            worker.error.emit("Test worker timed out")

    monkeypatch.setattr(PrepareWorker, "run", gated)
    window._prepare_action.trigger()
    try:
        wait_for(entered.is_set)
        row.choice.setCurrentIndex(row.choice.findData(True))
        assert not window._save_session_action.isEnabled()
    finally:
        release.set()
    idle(window)
    assert session.prepare_state == "stale" and track.repair_recipe.dc_remove
    output = next(t for t in session.output_tracks if t.filename == track.filename)
    assert output.repair_overrides == {} and not output.repair_recipe.active
    np.testing.assert_array_equal(sf.read(output.processed_filepath)[0], track.audio_data)
    session.config["_use_processed"] = True
    with pytest.raises(ValueError, match="out of date"):
        resolve_transfer_audio(session, output)

    window._prepare_action.trigger()
    idle(window)
    assert session.prepare_state == "ready"
    output = next(t for t in session.output_tracks if t.filename == track.filename)
    selected = resolve_transfer_audio(session, output)
    expected = render_corrected(track.audio_data, track.samplerate, track.repair_recipe)
    np.testing.assert_allclose(sf.read(selected.filepath)[0], expected, atol=2**-15)


@pytest.mark.parametrize("project", [True], indirect=True, ids=["normalization-enabled"])
def test_gain_changed_during_output_write_requires_a_new_prepare(project, monkeypatch):
    window = project
    session = window._session
    track = session.tracks[0]
    choice = open_editor(window, track).rows["dc_remove"].choice
    choice.setCurrentIndex(choice.findData(True))
    idle(window)
    assert track.repair_recipe.dc_remove
    previous_gain = track.processor_results["bimodal_normalize"].gain_db
    ready, release = Event(), Event()
    write = pipeline_module._write_atomic

    def hold_first_write(destination, audio, sample_rate, subtype):
        if destination.name == track.filename and not ready.is_set():
            ready.set()
            if not release.wait(6):
                raise RuntimeError("Test writer timed out")
        return write(destination, audio, sample_rate, subtype)

    monkeypatch.setattr(pipeline_module, "_write_atomic", hold_first_write)
    window._prepare_action.trigger()
    try:
        wait_for(ready.is_set)
        assert not window._analyze_action.isEnabled() and not window._save_session_action.isEnabled()
        window._on_analyze()
        assert window._worker is None and window._session is session
        gain = window._track_table.cellWidget(window._find_table_row(track.filename), 4)
        assert gain.isEnabled()
        gain.setValue(-12.0)
        assert track.manual_gain_db == -12.0
    finally:
        release.set()
    idle(window)
    assert window._analyze_action.isEnabled() and window._save_session_action.isEnabled()
    output = next(t for t in session.output_tracks if t.filename == track.filename)
    wet = render_corrected(track.audio_data, track.samplerate, track.repair_recipe)
    np.testing.assert_allclose(sf.read(output.processed_filepath)[0], wet * 10 ** (previous_gain / 20), atol=2**-15)
    assert session.prepare_state == "stale"
    session.config["_use_processed"] = True
    with pytest.raises(ValueError, match="out of date"):
        resolve_transfer_audio(session, output)

    window._prepare_action.trigger()
    idle(window)
    assert session.prepare_state == "ready"
    output = next(t for t in session.output_tracks if t.filename == track.filename)
    selected = resolve_transfer_audio(session, output)
    np.testing.assert_allclose(sf.read(selected.filepath)[0], wet * 10 ** (-12 / 20), atol=2**-15)
    assert selected.fader_db == pytest.approx(track.processor_results["bimodal_normalize"].data["fader_offset"])


def test_pcm_error_manual_dialog_cancel_apply_and_retry(project, monkeypatch):
    window = project
    track = window._session.tracks[1]
    path = Path(window._session.output_tracks[1].processed_filepath)
    original_wet = path.read_bytes()
    row = open_editor(window, track).rows["dc_remove"]
    row.choice.setCurrentIndex(row.choice.findData(True))
    idle(window)
    assert track.repair_recipe == RepairRecipe(dc_remove=True)
    assert not window._flat_config()["bimodal_normalize_enabled"]
    actions = iter(("Cancel", "Apply gain"))
    proposed = []
    real_box, real_dialog = QMessageBox.exec, QDialog.exec

    def choose_adjust(box):
        assert box.windowTitle() == "Output gain needs adjustment"
        button, = [b for b in box.buttons() if b.text() == "Adjust Gain"]
        QTimer.singleShot(0, button.click)
        return real_box(box)

    def choose_gain(dialog):
        assert dialog.windowTitle() == "Manual output gain"
        proposed.append(dialog.findChild(QDoubleSpinBox).value())
        label = next(actions)
        button, = [b for b in dialog.findChildren(QPushButton) if b.text().replace("&", "") == label]
        QTimer.singleShot(0, button.click)
        return real_dialog(dialog)

    monkeypatch.setattr(QMessageBox, "exec", choose_adjust)
    monkeypatch.setattr(QDialog, "exec", choose_gain)
    window._prepare_action.trigger()
    idle(window)
    assert track.manual_gain_db is None and path.read_bytes() == original_wet
    window._prepare_action.trigger()
    idle(window)
    assert len(proposed) == 2 and proposed[0] == proposed[1] < 0
    assert track.manual_gain_db == proposed[1] and path.read_bytes() == original_wet
    window._prepare_action.trigger()
    idle(window)
    session = window._session
    assert session.prepare_state == "ready" and not session.config["_prepare_errors"]
    prepared = next(t for t in session.output_tracks if t.filename == track.filename)
    wet = render_corrected(track.audio_data, track.samplerate, track.repair_recipe)
    np.testing.assert_allclose(sf.read(path)[0], wet * 10 ** (proposed[1] / 20), atol=2**-15)
    for processed, expected in [(False, Path(track.filepath)), (True, path)]:
        window._use_processed_cb.setChecked(processed)
        selection = resolve_transfer_audio(session, prepared)
        assert Path(selection.filepath) == expected and selection.fader_db == 0
