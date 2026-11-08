"""Applying a track layout reports progress in the progress strip only."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from sessionprepgui import mainwindow, settings
from sessionprepgui.analysis import worker as worker_module
from sessionpreplib.models import SessionContext


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


class _FakeApplyWorker(QObject):
    progress = Signal(str)
    progress_value = Signal(int, int)
    apply_finished = Signal()
    error = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def start(self):  # no files are written in this test
        pass


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *args: None)
    monkeypatch.setattr(mainwindow.SessionPrepWindow, "_configure_daw_processors",
                        lambda self: setattr(self, "_daw_processors", []))
    monkeypatch.setattr(worker_module, "TopologyApplyWorker", _FakeApplyWorker)
    owner = mainwindow.SessionPrepWindow()
    owner._session = SessionContext([], {})
    owner._source_dir = "unused-source"
    yield owner
    owner.close()


def test_apply_keeps_the_toolbar_summary_and_uses_the_progress_strip(qapp, window, monkeypatch):
    started = []
    monkeypatch.setattr(window, "_progress_start", lambda message: started.append(message))
    window._topo_status_label.setText("6 input → 6 output tracks")
    window._on_topo_apply()
    assert window._topo_status_label.text() == "6 input → 6 output tracks"
    assert started == ["Applying topology…"]


def test_failed_apply_leaves_no_stale_toolbar_message(qapp, window, monkeypatch):
    monkeypatch.setattr(window, "_progress_start", lambda message: None)
    monkeypatch.setattr(window, "_progress_fail", lambda message: None)
    window._topo_status_label.setText("6 input → 6 output tracks")
    window._on_topo_apply()
    window._on_topo_apply_error("disk full")
    assert window._topo_status_label.text() == "6 input → 6 output tracks"
