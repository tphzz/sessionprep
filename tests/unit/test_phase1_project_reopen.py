"""Phase 1 source findings must survive separately from the user's output layout."""

import os
import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.analysis.worker import AnalyzeWorker, Phase1AnalyzeWorker, TopologyApplyWorker
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.session.io import load_session, save_session
from sessionpreplib.topology import extract_channel, passthrough_routes


def _write_source(path, *, duplicate=True):
    left = np.tile([0.25, -0.25, 0.125, -0.125], 1200)
    sf.write(path, np.column_stack((left, left if duplicate else -left)), 48000,
             subtype="PCM_24")


@pytest.fixture
def project(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    folder = tmp_path / "audio"
    folder.mkdir()
    _write_source(folder / "Keys.wav")
    scans = []

    def run_phase1(worker):
        scans.append([t.filename for t in worker.session_context.tracks])
        worker.run()

    monkeypatch.setattr(Phase1AnalyzeWorker, "start", run_phase1)
    monkeypatch.setattr(TopologyApplyWorker, "start", TopologyApplyWorker.run)
    monkeypatch.setattr(AnalyzeWorker, "start", AnalyzeWorker.run)
    window = SessionPrepWindow()
    monkeypatch.setattr(window.persistence_controller, "start_peak_build", lambda *_args: None)
    try:
        window.persistence_controller.load_directory(str(folder))
        assert scans == [["Keys.wav"]]
        yield window, folder, scans
    finally:
        window.batch_controller.dock.clear_all()
        dispose_window(window)
        app.processEvents()


def _edited_route(window):
    entry, = window.state.topo_topology.entries
    entry.sources[0].routes = extract_channel(1)
    window.topology_controller.notify_topology_changed()


def test_apply_and_reanalyze_preserve_unowned_audio_without_analyzing_it(project):
    window, source, _ = project
    layout = source / "sp_01_tracklayout"
    layout.mkdir(exist_ok=True)
    sentinel = layout / "External take.wav"
    _write_source(sentinel, duplicate=False)
    original = sentinel.read_bytes()
    window.topology_controller.apply_topology()
    window.analysis_controller.analyze()
    assert [track.filename for track in window.state.session.tracks] == ["Keys.wav"]
    assert sentinel.read_bytes() == original


@pytest.mark.parametrize("batch", [False, True], ids=["save-session", "save-batch"])
def test_desktop_save_error_preserves_previous_document_and_retry(project, tmp_path, monkeypatch, batch):
    from sessionprepgui.batch.panel import BatchItem

    window, _, _ = project
    path = tmp_path / ("queue.spbatch" if batch else "project.spsession")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), ""))
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _, title, msg: messages.append((title, msg)))
    if batch:
        item = BatchItem("a", "Before", "dawproject", "DAWproject", "", window.persistence_controller.capture_session_state())
        window.batch_controller.dock.add_item(item)
        save = window.batch_controller.save_queue
    else:
        window.daw_controller.page.project_name_edit.setText("Before")
        save = window.persistence_controller.save_session_dialog
    save()
    previous = path.read_bytes()
    if batch:
        window.batch_controller.dock.add_item(BatchItem("b", "After", "dawproject", "DAWproject", "",
                                              window.persistence_controller.capture_session_state()))
    else:
        window.daw_controller.page.project_name_edit.setText("After")

    def short_write(_state, stream, **_kwargs):
        stream.write("{")
        raise OSError("short write")

    with monkeypatch.context() as fault:
        fault.setattr(json, "dump", short_write)
        save()
    assert messages and "short write" in messages[0][1]
    assert path.read_bytes() == previous
    save()
    if batch:
        assert [record["project_name"] for record in json.loads(path.read_text())] == ["Before", "After"]
    else:
        assert load_session(str(path))["project_name"] == "After"


@pytest.mark.parametrize("applied", [False, True], ids=["before-apply", "after-apply"])
@pytest.mark.parametrize("legacy", [False, True], ids=["current", "legacy"])
def test_saved_layout_and_source_finding_are_independent(project, tmp_path, applied, legacy):
    window, _, scans = project
    _edited_route(window)
    if applied:
        window.topology_controller.apply_topology()
        assert window.state.session.tracks[0].channels == 1
    state = window.persistence_controller.capture_session_state()
    if legacy:
        state.pop("phase1_sources", None)
    path = str(tmp_path / "session.spsession")
    save_session(path, state)
    window.persistence_controller.restore_session_state(load_session(path))
    # Old projects can still use their saved Phase 2/3 results immediately.
    assert len(scans) == 1
    window._phase_tabs.setCurrentIndex(0)
    entry, = window.state.topo_topology.entries
    assert entry.sources[0].routes == extract_channel(1)
    assert "dual-mono" in window.topology_controller.page.input_tree.topLevelItem(0).text(0)
    assert window.topology_controller.page.reset_action.isEnabled()
    assert window.topology_controller.page.reanalyze_action.isEnabled()
    window.topology_controller.page.reset_action.trigger()
    entry, = window.state.topo_topology.entries
    assert entry.output_channels == 1
    assert entry.sources[0].routes == extract_channel(0)
    window._phase_tabs.setCurrentIndex(1)
    window._phase_tabs.setCurrentIndex(0)
    assert len(scans) == (2 if legacy else 1)


def test_source_replacement_invalidates_only_its_saved_evidence(project, tmp_path):
    window, folder, scans = project
    _write_source(folder / "Vox.wav")
    window.persistence_controller.load_directory(str(folder))
    path = str(tmp_path / "session.spsession")
    save_session(path, window.persistence_controller.capture_session_state())
    source = folder / "Keys.wav"
    old = source.stat()
    replacement = tmp_path / "replacement.wav"
    _write_source(replacement, duplicate=False)
    os.replace(replacement, source)
    os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert source.stat().st_size == old.st_size

    scans.clear()
    window.persistence_controller.restore_session_state(load_session(path))
    window._phase_tabs.setCurrentIndex(0)
    assert scans == [["Keys.wav"]]
    # Preserve the saved layout until the user explicitly requests new defaults.
    assert [entry.output_channels for entry in window.state.topo_topology.entries] == [1, 1]
    window.topology_controller.page.reset_action.trigger()
    entries = {entry.output_filename: entry for entry in window.state.topo_topology.entries}
    assert entries["Keys.wav"].sources[0].routes == passthrough_routes(2)
    assert entries["Vox.wav"].sources[0].routes == extract_channel(0)


@pytest.mark.parametrize("changed_before_save", [False, True])
def test_measurement_uses_the_epsilon_in_effect_at_scan_time(project, tmp_path, changed_before_save):
    window, folder, scans = project
    left = np.tile([0.25, -0.25], 2400)
    sf.write(folder / "Keys.wav", np.column_stack((left, left + 0.000005)), 48000, subtype="DOUBLE")
    window.persistence_controller.load_directory(str(folder))
    assert window.state.topo_source_tracks[0].detector_results["dual_mono"].data["dual_mono"]
    preset = window.state.config["config_presets"]["Default"]["detectors"]["dual_mono"]
    if changed_before_save:
        preset["dual_mono_eps"] = 1e-7
    path = str(tmp_path / "session.spsession")
    save_session(path, window.persistence_controller.capture_session_state())
    preset["dual_mono_eps"] = 1e-7
    scans.clear()
    window.persistence_controller.restore_session_state(load_session(path))
    window._phase_tabs.setCurrentIndex(0)
    assert scans == [["Keys.wav"]]
    window.topology_controller.page.reset_action.trigger()
    assert window.state.topo_topology.entries[0].output_channels == 2


def test_finished_scan_of_previous_folder_cannot_replace_current_scan(project, tmp_path, monkeypatch):
    window, first, _ = project
    second = tmp_path / "second"
    second.mkdir()
    _write_source(second / "Different.wav", duplicate=False)
    monkeypatch.setattr(Phase1AnalyzeWorker, "start", lambda worker: None)
    window.persistence_controller.load_directory(str(first))
    old = window.analysis_controller.p1_worker
    QThread.start(old)
    assert old.wait(10_000)
    # Its completion is queued on the GUI thread when the user opens another folder.
    window.persistence_controller.load_directory(str(second))
    current = window.analysis_controller.p1_worker
    QApplication.processEvents()
    assert window.analysis_controller.p1_worker is current
    assert [t.filename for t in window.state.session.tracks] == ["Different.wav"]
    assert window.state.topo_topology is None
    current.run()
    assert window.state.topo_topology.entries[0].output_channels == 2


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_format_findings_follow_full_input_set_when_membership_changes(project, tmp_path, change):
    window, folder, scans = project
    sf.write(folder / "A.wav", [0.25, -0.25], 44_100, subtype="PCM_24")
    _write_source(folder / "B.wav")
    window.persistence_controller.load_directory(str(folder))
    initial = {t.filename: t.detector_results["format_consistency"] for t in window.state.topo_source_tracks}
    assert initial["A.wav"].data["expected_sr"] == 48_000
    path = str(tmp_path / "session.spsession")
    save_session(path, window.persistence_controller.capture_session_state())
    if change == "replace":
        sf.write(folder / "Keys.wav", [0.25, -0.25], 44_100, subtype="PCM_24")
    else:
        (folder / "Keys.wav").unlink()
    scans.clear()
    window.persistence_controller.restore_session_state(load_session(path))
    window._phase_tabs.setCurrentIndex(0)
    results = {t.filename: t.detector_results["format_consistency"] for t in window.state.topo_source_tracks}
    assert results["B.wav"].data["expected_sr"] == 44_100
    assert "mismatch" not in results["A.wav"].summary
    assert scans == ([["Keys.wav"]] if change == "replace" else [])
