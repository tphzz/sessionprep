"""Applying a track layout reports progress in the progress strip only."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

from qt_helpers import dispose_window
from sessionprepgui import mainwindow, settings
from sessionprepgui.analysis import worker as worker_module
from sessionpreplib.models import SessionContext


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


class _FakeApplyWorker(QThread):
    progress = Signal(object, str)
    progress_value = Signal(object, int, int)
    overwrite_confirmation_required = Signal(object, str, object)
    apply_finished = Signal(object)
    error = Signal(object, str)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def start(self):  # no files are written in this test
        pass


@pytest.fixture
def window(qapp, monkeypatch):
    from sessionprepgui.daw import DawTransferController
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *args: None)
    monkeypatch.setattr(DawTransferController, "configure_daw_processors",
                        lambda self: setattr(self._state, "daw_processors", []))
    monkeypatch.setattr(worker_module, "TopologyApplyWorker", _FakeApplyWorker)
    owner = mainwindow.SessionPrepWindow()
    owner.state.session = SessionContext([], {})
    owner.state.source_dir = "unused-source"
    yield owner
    dispose_window(owner)


def test_apply_keeps_the_toolbar_summary_and_uses_the_progress_strip(qapp, window, monkeypatch):
    started = []
    monkeypatch.setattr(window, "progress_start", lambda message, **kw: started.append(message))
    window.topology_controller.page.status_label.setText("6 input → 6 output tracks")
    window.topology_controller.apply_topology()
    assert window.topology_controller.page.status_label.text() == "6 input → 6 output tracks"
    assert started == ["Applying topology…"]


def test_failed_apply_leaves_no_stale_toolbar_message(qapp, window, monkeypatch):
    monkeypatch.setattr(window, "progress_start", lambda message, **kw: None)
    monkeypatch.setattr(window, "progress_fail", lambda message: None)
    window.topology_controller.page.status_label.setText("6 input → 6 output tracks")
    window.topology_controller.apply_topology()
    worker = window.topology_controller.apply_worker
    worker.error.emit(worker, "disk full")
    assert window.topology_controller.page.status_label.text() == "6 input → 6 output tracks"
