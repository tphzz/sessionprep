"""A completed worker's queued signals belong to the project that started it."""
import json
from threading import Event

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.analysis.worker import AnalyzeWorker, PrepareWorker, TopologyApplyWorker
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionpreplib.audio import load_track
from sessionpreplib.topology import build_default_topology


@pytest.mark.parametrize("operation", ["analyze", "prepare", "apply"])
@pytest.mark.parametrize("action", ["open", "close"])
def test_pending_worker_results_after_project_open_or_close(tmp_path, monkeypatch, operation, action):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    old, new = (tmp_path / name for name in ("old", "new"))
    for folder in (old, new):
        folder.mkdir()
    sf.write(old / "Old.wav", np.tile([.2, -.1], 4000), 8000)
    path = tmp_path / "new.spsession"
    path.write_text(json.dumps({"version": 7, "source_dir": str(new), "project_name": "New"}))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    window = SessionPrepWindow()
    window.persistence_controller.restore_session_state({"source_dir": str(old), "tracks": [load_track(str(old / "Old.wav"))]})
    queued, release = Event(), Event()
    worker_class = {"analyze": AnalyzeWorker, "prepare": PrepareWorker, "apply": TopologyApplyWorker}[operation]
    cancel_method = "requestInterruption" if operation == "prepare" else "cancel"
    stop = getattr(worker_class, cancel_method)
    run = worker_class.run

    def finish_then_wait(worker):
        run(worker)
        worker.progress.emit(worker, "OLD PROJECT PROGRESS")
        queued.set()
        release.wait(8)

    monkeypatch.setattr(worker_class, "run", finish_then_wait)
    def release_on_cancel(worker):
        stop(worker)
        release.set()

    monkeypatch.setattr(worker_class, cancel_method, release_on_cancel)
    try:
        if operation == "analyze":
            window.analysis_controller.analyze()
            worker = window.analysis_controller.analyze_worker
        elif operation == "prepare":
            window.analysis_controller.prepare()
            worker = window.analysis_controller.prepare_worker
        else:
            window.state.topo_topology = build_default_topology(window.state.topo_source_tracks)
            window.topology_controller.apply_topology()
            worker = window.topology_controller.apply_worker
        # Let the real worker finish its DSP and post results while the GUI
        # is busy. Switching projects happens before the event queue drains.
        assert queued.wait(6)
        if action == "open":
            window.persistence_controller.load_session_dialog()
        else:
            assert window.close()
        fresh = window.state.session
        progress = []
        monkeypatch.setattr(window, "progress_message", progress.append)
        release.set()
        assert worker.wait(3000)
        app.processEvents()
        assert window.state.session is fresh
        assert "OLD PROJECT PROGRESS" not in progress
        if action == "open":
            assert fresh.project_name == "New" and not fresh.tracks and not fresh.output_tracks
            assert window.phase2_page.track_table.rowCount() == 0
            assert not window._phase_tabs.isTabEnabled(2)
        else:
            assert not window._playback.is_playing and window.analysis_controller.analyze_worker is None
    finally:
        release.set()
        dispose_window(window)
