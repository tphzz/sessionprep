"""Isolated unit tests for SessionConfigController (Phase 2 Config tab & impact routing)."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from sessionpreplib.models import SessionContext
from sessionprepgui.analysis.config_controller import SessionConfigController
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState


class _StubShell:
    def __init__(self) -> None:
        self.parent_widget = QWidget()
        self.status_messages: list[str] = []

    def show_status(self, message: str) -> None:
        self.status_messages.append(message)

    def progress_start(self, text: str, *, owner=None) -> None: pass
    def progress_message(self, text: str) -> None: pass
    def progress_value(self, current: int, total: int) -> None: pass
    def progress_finish(self, text: str) -> None: pass
    def progress_fail(self, text: str) -> None: pass
    def clear_progress(self) -> None: pass
    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None: pass
    def switch_to_phase(self, phase_index: int) -> None: pass
    def current_phase(self) -> int: return 1
    def is_phase_enabled(self, phase_index: int) -> bool: return True
    def register_phase_progress_layout(self, phase_index: int, layout) -> None: pass
    def dialog_parent(self) -> QWidget: return self.parent_widget
    def is_batch_mode(self) -> bool: return False
    def is_closing_after_daw(self) -> bool: return False
    def resume_close_after_daw(self) -> None: pass
    def set_window_title(self, title: str) -> None: pass


def test_session_config_controller_tracks_overrides_and_revert(_font_app):
    state = WorkspaceState(config=build_defaults())
    state.session = SessionContext(tracks=[], config={})
    shell = _StubShell()
    stale_calls: list[bool] = []
    ctrl = SessionConfigController(state, shell)
    ctrl.prepare_stale_requested.connect(lambda: stale_calls.append(True))
    try:
        assert ctrl.session_config_dirty_keys() == set()
        ctrl.on_session_config_value_changed("analysis.window", 250)
        assert "analysis" in ctrl.session_config_dirty_keys()
        assert stale_calls == [True]
        assert "press Reanalyze" in shell.status_messages[-1]

        ctrl.on_session_config_reset()
        assert state.session_overrides == {}
        assert ctrl.session_config_dirty_keys() == set()
        assert shell.status_messages[-1] == "Session config reverted to preset."
    finally:
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_session_config_controller_routes_daw_only_change_without_reanalysis(_font_app):
    state = WorkspaceState(config=build_defaults())
    state.session = SessionContext(tracks=[], config={})
    shell = _StubShell()
    daw_refreshed: list[bool] = []
    ctrl = SessionConfigController(state, shell)
    ctrl.daw_refresh_requested.connect(lambda: daw_refreshed.append(True))
    try:
        ctrl.on_session_config_value_changed("daw_processors.protools.protools_project_dir", "/tmp/pt")
        assert daw_refreshed == [True]
        assert "Phase 3 DAW settings updated" in shell.status_messages[-1]
    finally:
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()
