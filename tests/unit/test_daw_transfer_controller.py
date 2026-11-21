"""Isolated unit tests for DawTransferController (Phase 3 DAW Transfer)."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QMessageBox, QWidget

from sessionpreplib.models import SessionContext, TransferEntry
from sessionprepgui.daw.controller import DawTransferController
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState


class _StubShell:
    def __init__(self) -> None:
        self.parent_widget = QWidget()
        self.status_messages: list[str] = []
        self.progress_events: list[tuple[str, object]] = []
        self.batch_mode = False
        self.closing = False

    def show_status(self, message: str) -> None:
        self.status_messages.append(message)

    def progress_start(self, text: str, *, owner=None) -> None:
        self.progress_events.append(("start", text))

    def progress_message(self, text: str) -> None:
        self.progress_events.append(("message", text))

    def progress_value(self, current: int, total: int) -> None:
        self.progress_events.append(("value", (current, total)))

    def progress_finish(self, text: str) -> None:
        self.progress_events.append(("finish", text))

    def progress_fail(self, text: str) -> None:
        self.progress_events.append(("fail", text))

    def clear_progress(self) -> None:
        pass

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None:
        pass

    def switch_to_phase(self, phase_index: int) -> None:
        pass

    def current_phase(self) -> int:
        return 2

    def is_phase_enabled(self, phase_index: int) -> bool:
        return True

    def register_phase_progress_layout(self, phase_index: int, layout) -> None:
        pass

    def dialog_parent(self) -> QWidget:
        return self.parent_widget

    def is_batch_mode(self) -> bool:
        return self.batch_mode

    def is_closing_after_daw(self) -> bool:
        return self.closing

    def resume_close_after_daw(self) -> None:
        pass

    def set_window_title(self, title: str) -> None:
        pass


def _make_controller(state: WorkspaceState, shell: _StubShell):
    calls = {"setup_table": 0, "cleared": 0, "enqueued": []}
    ctrl = DawTransferController(
        state,
        shell,
        on_capture_session_state=lambda: {"captured": True},
        on_enqueue_batch_item=lambda item: (calls["enqueued"].append(item), True)[1],
        on_is_prepare_running=lambda: False,
    )
    ctrl.setup_table_refresh_requested.connect(
        lambda: calls.__setitem__("setup_table", calls["setup_table"] + 1)
    )
    ctrl.clear_workspace_requested.connect(
        lambda: calls.__setitem__("cleared", calls["cleared"] + 1)
    )
    return ctrl, calls


def test_daw_controller_prune_and_unassign_all(_font_app, monkeypatch):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    ctrl, calls = _make_controller(state, shell)
    state.active_daw_processor = SimpleNamespace(id="protools_0", name="Pro Tools", enabled=True, fader_ceiling_db=12.0)
    state.session = SessionContext(
        tracks=[],
        config={},
        transfer_manifest=[TransferEntry("track-1", "track-1.wav", "Track 1")],
        daw_state={
            "protools_0": {
                "folders": [{"id": "folder-1", "name": "Drums", "parent_id": None, "index": 0}],
                "assignments": {"track-1": "folder-1", "duplicate-1": "folder-1"},
                "track_order": {"folder-1": ["track-1", "duplicate-1"], "folder-2": ["duplicate-2"]},
            }
        },
    )
    try:
        ctrl.prune_assignments_to_manifest()
        dp_state = state.session.daw_state["protools_0"]
        assert dp_state["assignments"] == {"track-1": "folder-1"}
        assert dp_state["track_order"] == {"folder-1": ["track-1"]}

        monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
        ctrl.unassign_all()
        assert dp_state["assignments"] == {}
        assert dp_state["track_order"] == {}
        assert calls["setup_table"] == 1
        assert shell.status_messages == ["All DAW folder assignments cleared."]
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_daw_controller_discards_stale_fetch_after_template_or_workspace_change(_font_app):
    state = WorkspaceState(config=build_defaults())
    proc = SimpleNamespace(id="dawproject_0", name="DAWproject", template_identity="tpl-v1")
    state.daw_template_bindings = {"dawproject_0": "tpl-v2"}
    session = SessionContext(tracks=[], config={})
    state.session = session
    shell = _StubShell()
    ctrl, _ = _make_controller(state, shell)
    try:
        ctrl.fetch_worker = SimpleNamespace(_session=session, _processor=proc)
        ctrl._on_daw_fetch_result(True, "Fetched", SessionContext(tracks=[], config={}, daw_state={"dawproject_0": {"folders": [{"id": "f1", "name": "A"}]}}))
        assert ctrl.fetch_worker is None
        assert "dawproject_0" not in session.daw_state
        assert ("fail", "DAW template changed. Fetch again.") in shell.progress_events
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_daw_controller_auto_assign_maps_groups_to_folders(_font_app):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    ctrl, calls = _make_controller(state, shell)
    state.active_daw_processor = SimpleNamespace(id="pt_0", name="Pro Tools", enabled=True, fader_ceiling_db=12.0)
    state.session_groups = [{"name": "Drums", "daw_target": "Drums"}]
    state.session = SessionContext(
        tracks=[],
        config={},
        transfer_manifest=[
            TransferEntry("kick.wav", "kick.wav", "kick", group="Drums"),
            TransferEntry("vox.wav", "vox.wav", "vox", group=None),
        ],
        daw_state={
            "pt_0": {
                "folders": [{"id": "f_drums", "name": "Drums", "parent_id": None, "index": 0}],
                "assignments": {},
                "track_order": {},
            }
        },
    )
    try:
        ctrl.auto_assign()
        assert state.session.daw_state["pt_0"]["assignments"] == {"kick.wav": "f_drums"}
        assert calls["setup_table"] >= 1
        assert "assigned 1 track(s) to 1 folder(s)" in shell.status_messages[-1]
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()
