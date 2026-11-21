from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from qt_helpers import dispose_window
from sessionprepgui.analysis.worker import AnalyzeWorker, PrepareWorker, TopologyApplyWorker
from sessionprepgui.batch import BatchQueueDock
from sessionpreplib.models import SessionContext
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.tracks.table_widgets import (
    _PHASE_ANALYSIS,
    _PHASE_SETUP,
    _PHASE_TOPOLOGY,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _window(qapp):
    return SessionPrepWindow()


def _select_phase(window, phase_index):
    window._phase_tabs.setTabEnabled(phase_index, True)
    window._phase_tabs.setCurrentIndex(phase_index)


def _progress_layout_contains(window, phase_index):
    return (
        window.progress_coordinator.phase_layouts[phase_index].indexOf(
            window.progress_coordinator.phase_panels[phase_index]) >= 0
    )


def _progress_panel(window, phase_index):
    return window.progress_coordinator.phase_panels[phase_index]


def test_phase_progress_panels_exist_and_start_hidden(qapp):
    window = _window(qapp)
    try:
        assert set(window.progress_coordinator.phase_layouts) == {
            _PHASE_TOPOLOGY, _PHASE_ANALYSIS, _PHASE_SETUP}
        assert set(window.progress_coordinator.phase_panels) == {
            _PHASE_TOPOLOGY, _PHASE_ANALYSIS, _PHASE_SETUP}
        for phase_index in (
                _PHASE_TOPOLOGY, _PHASE_ANALYSIS, _PHASE_SETUP):
            assert _progress_layout_contains(window, phase_index)
            progress = _progress_panel(window, phase_index)
            assert not progress.isVisible()
            assert progress.testAttribute(Qt.WA_StyledBackground)
            assert "background-color" in progress.styleSheet()
            assert "border-top" in progress.styleSheet()
        assert not hasattr(window, "_global_progress")
        assert not hasattr(window, "_prepare_progress")
        assert not hasattr(window, "_topo_progress")
        assert not hasattr(window, "_transfer_progress")
        assert not hasattr(window, "_progress_label")
        assert not hasattr(window, "_progress_bar")
    finally:
        dispose_window(window)


def test_analysis_progress_uses_global_panel_not_status_bar(qapp):
    window = _window(qapp)
    try:
        _select_phase(window, _PHASE_ANALYSIS)
        window.progress_start("Analyzing...")
        window._status_bar.showMessage("Idle")

        worker = window.analysis_controller.analyze_worker = AnalyzeWorker(".", {})
        worker.progress.connect(window.analysis_controller._on_worker_progress)
        worker.progress_value.connect(window.analysis_controller._on_worker_progress_value)
        worker.progress.emit(worker, "Analyzing Kick.wav - clipping")
        worker.progress_value.emit(worker, 3, 10)

        progress = _progress_panel(window, _PHASE_ANALYSIS)
        assert progress._label.text() == (
            "Analyzing Kick.wav - clipping")
        assert progress._bar.maximum() == 10
        assert progress._bar.value() == 3
        assert window._status_bar.currentMessage() == "Idle"
        assert _progress_layout_contains(window, _PHASE_ANALYSIS)
    finally:
        dispose_window(window)


def test_prepare_progress_uses_global_panel_not_status_bar(qapp):
    window = _window(qapp)
    try:
        _select_phase(window, _PHASE_ANALYSIS)
        window.progress_start("Preparing...")
        window._status_bar.showMessage("Idle")

        worker = window.analysis_controller.prepare_worker = PrepareWorker(SessionContext([], {}), [], ".")
        worker.progress.connect(window.analysis_controller._on_prepare_progress)
        worker.progress_value.connect(window.analysis_controller._on_prepare_progress_value)
        worker.progress.emit(worker, "Writing Snare.wav")
        worker.progress_value.emit(worker, 4, 12)

        progress = _progress_panel(window, _PHASE_ANALYSIS)
        assert progress._label.text() == "Writing Snare.wav"
        assert progress._bar.maximum() == 12
        assert progress._bar.value() == 4
        assert window._status_bar.currentMessage() == "Idle"
    finally:
        dispose_window(window)


def test_topology_apply_progress_uses_global_panel(qapp):
    window = _window(qapp)
    try:
        _select_phase(window, _PHASE_TOPOLOGY)
        window.progress_start("Applying topology...")

        worker = window.topology_controller.apply_worker = TopologyApplyWorker(SessionContext([], {}), ".")
        worker.progress.connect(window.topology_controller._on_apply_progress)
        worker.progress_value.connect(window.topology_controller._on_apply_progress_value)
        worker.progress.emit(worker, "Writing 01_Kick.wav")
        worker.progress_value.emit(worker, 1, 8)

        progress = _progress_panel(window, _PHASE_TOPOLOGY)
        assert progress._label.text() == "Writing 01_Kick.wav"
        assert progress._bar.maximum() == 8
        assert progress._bar.value() == 1
        assert _progress_layout_contains(window, _PHASE_TOPOLOGY)
    finally:
        dispose_window(window)


def test_daw_transfer_progress_uses_global_panel_not_status_bar(qapp):
    window = _window(qapp)
    try:
        _select_phase(window, _PHASE_SETUP)
        window.progress_start("Transferring...")
        window._status_bar.showMessage("Idle")

        window.daw_controller._on_transfer_progress("Creating Pro Tools session")
        window.daw_controller._on_transfer_progress_value(2, 5)

        progress = _progress_panel(window, _PHASE_SETUP)
        assert progress._label.text() == (
            "Creating Pro Tools session")
        assert progress._bar.maximum() == 5
        assert progress._bar.value() == 2
        assert window._status_bar.currentMessage() == "Idle"
        assert _progress_layout_contains(window, _PHASE_SETUP)
    finally:
        dispose_window(window)


def test_progress_panel_switches_phase_during_running_operation(qapp):
    window = _window(qapp)
    try:
        window.show()
        qapp.processEvents()
        _select_phase(window, _PHASE_ANALYSIS)
        window.progress_start("Preparing...")
        window.progress_value(4, 12)

        _select_phase(window, _PHASE_SETUP)
        progress = _progress_panel(window, _PHASE_SETUP)

        assert not _progress_panel(window, _PHASE_ANALYSIS).isVisible()
        assert progress.isVisible()
        assert progress._label.text() == "Preparing..."
        assert progress._bar.maximum() == 12
        assert progress._bar.value() == 4
    finally:
        dispose_window(window)


@pytest.mark.parametrize("phase_index", [
    _PHASE_TOPOLOGY,
    _PHASE_ANALYSIS,
    _PHASE_SETUP,
])
def test_phase_progress_panel_shrinks_content_above(qapp, phase_index):
    window = _window(qapp)
    try:
        _select_phase(window, phase_index)
        window.resize(1200, 800)
        window.show()
        qapp.processEvents()

        layout = window.progress_coordinator.phase_layouts[phase_index]
        content = layout.itemAt(layout.count() - 2).widget()
        before_height = content.geometry().height()

        window.progress_start("Working...")
        qapp.processEvents()

        progress = _progress_panel(window, phase_index)
        assert progress.isVisible()
        assert content.geometry().height() < before_height
        assert progress.geometry().top() >= content.geometry().bottom()
    finally:
        dispose_window(window)


def test_batch_dock_uses_progress_panel(qapp):
    dock = BatchQueueDock()
    try:
        dock.show()
        qapp.processEvents()

        assert hasattr(dock, "_progress_panel")
        assert not hasattr(dock, "_progress_bar")
        assert not dock._progress_panel.isVisible()

        dock.set_running_state(True)
        dock.update_progress_message("Batch item")
        dock.update_progress(2, 5)

        assert dock._progress_panel.isVisible()
        assert dock._progress_panel._label.text() == "Batch item"
        assert dock._progress_panel._bar.maximum() == 5
        assert dock._progress_panel._bar.value() == 2

        dock.set_running_state(False)
        assert not dock._progress_panel.isVisible()
    finally:
        dock.close()
