from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox, QWidget

from sessionprepgui.daw import DawTransferController
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionpreplib.models import SessionContext, TransferEntry


class _Shell:
    def __init__(self):
        self.parent_widget = QWidget()
        self.messages: list[str] = []

    def show_status(self, message: str):
        self.messages.append(message)

    def progress_start(self, text: str, *, owner=None): pass
    def progress_message(self, text: str): pass
    def progress_value(self, current: int, total: int): pass
    def progress_finish(self, text: str): pass
    def progress_fail(self, text: str): pass
    def clear_progress(self): pass
    def set_phase_enabled(self, phase_index: int, enabled: bool): pass
    def switch_to_phase(self, phase_index: int): pass
    def current_phase(self) -> int: return 2
    def is_phase_enabled(self, phase_index: int) -> bool: return True
    def register_phase_progress_layout(self, phase_index: int, layout): pass
    def dialog_parent(self): return self.parent_widget
    def is_batch_mode(self) -> bool: return False
    def is_closing_after_daw(self) -> bool: return False
    def resume_close_after_daw(self): pass
    def set_window_title(self, title: str): pass


def _make_harness():
    state = WorkspaceState(config=build_defaults())
    shell = _Shell()
    counters = {"setup_table": 0}
    ctrl = DawTransferController(
        state,
        shell,
        on_capture_session_state=lambda: {},
        on_enqueue_batch_item=lambda _i: True,
        on_is_prepare_running=lambda: False,
    )
    ctrl.setup_table_refresh_requested.connect(
        lambda: counters.__setitem__("setup_table", counters["setup_table"] + 1)
    )
    state.active_daw_processor = SimpleNamespace(id="protools_0", name="Pro Tools", enabled=True)
    state.session = SessionContext(
        tracks=[],
        config={},
        transfer_manifest=[TransferEntry("track-1", "track-1.wav", "Track 1")],
        daw_state={
            "protools_0": {
                "assignments": {"track-1": "folder-1", "duplicate-1": "folder-1"},
                "track_order": {"folder-1": ["track-1", "duplicate-1"], "folder-2": ["duplicate-2"]},
            }
        },
    )
    return ctrl, state, shell, counters


def test_prune_assignments_to_manifest_removes_stale_entries(_font_app):
    ctrl, state, shell, _ = _make_harness()
    try:
        ctrl.prune_assignments_to_manifest()
        dp_state = state.session.daw_state["protools_0"]
        assert dp_state["assignments"] == {"track-1": "folder-1"}
        assert dp_state["track_order"] == {"folder-1": ["track-1"]}
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_unassign_all_clears_assignment_state(_font_app, monkeypatch):
    ctrl, state, shell, counters = _make_harness()
    try:
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
        ctrl.unassign_all()
        dp_state = state.session.daw_state["protools_0"]
        assert dp_state["assignments"] == {}
        assert dp_state["track_order"] == {}
        assert counters["setup_table"] == 1
        assert shell.messages == ["All DAW folder assignments cleared."]
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()
