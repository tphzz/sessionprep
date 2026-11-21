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
    window._restore_session_state({"source_dir": str(old), "tracks": [load_track(str(old / "Old.wav"))]})
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
            window._on_analyze()
            worker = window._worker
        elif operation == "prepare":
            window._on_prepare()
            worker = window._prepare_worker
        else:
            window._topo_topology = build_default_topology(window._topo_source_tracks)
            window._on_topo_apply()
            worker = window._topo_apply_worker
        # Let the real worker finish its DSP and post results while the GUI
        # is busy. Switching projects happens before the event queue drains.
        assert queued.wait(6)
        if action == "open":
            window._on_load_session()
        else:
            assert window.close()
        fresh = window._session
        progress = []
        monkeypatch.setattr(window, "_progress_message", progress.append)
        release.set()
        assert worker.wait(3000)
        app.processEvents()
        assert window._session is fresh
        assert "OLD PROJECT PROGRESS" not in progress
        if action == "open":
            assert fresh.project_name == "New" and not fresh.tracks and not fresh.output_tracks
            assert window._track_table.rowCount() == 0
            assert not window._phase_tabs.isTabEnabled(2)
        else:
            assert not window._playback.is_playing and window._worker is None
    finally:
        release.set()
        dispose_window(window)
