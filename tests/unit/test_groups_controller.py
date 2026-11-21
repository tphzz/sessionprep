"""Isolated unit tests for GroupsController (Phase 2 Groups tab & rules)."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QTableWidgetItem, QWidget

from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.tracks.groups_controller import GroupsController, match_track_to_group
from sessionprepgui.tracks.table_widgets import _DraggableTrackTable


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


def test_match_track_to_group_prefers_longest_match():
    groups = [
        {"name": "Drums", "match_method": "contains", "match_pattern": "drum, kick"},
        {"name": "Kick In", "match_method": "contains", "match_pattern": "kick_in"},
        {"name": "Bass", "match_method": "regex", "match_pattern": r"bass[_\s]?di"},
    ]
    assert match_track_to_group("01_Kick_In.wav", groups) == "Kick In"
    assert match_track_to_group("02_Bass_DI.wav", groups) == "Bass"
    assert match_track_to_group("03_Vocals.wav", groups) is None


def test_groups_controller_auto_group_and_linked_gain_update(_font_app, monkeypatch):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    table = _DraggableTrackTable()
    table.setColumnCount(8)
    proc = BimodalNormalizeProcessor()
    proc.configure({})

    t1 = TrackContext("kick.wav", "/tmp/kick.wav", None, 48000, 1, 48000, "24-bit", "PCM_24", 1.0)
    t2 = TrackContext("snare.wav", "/tmp/snare.wav", None, 48000, 1, 48000, "24-bit", "PCM_24", 1.0)
    t1.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", -4.0, "Transient", "Peak", {"gain_db_individual": -4.0, "original_gain_db": -4.0}
    )
    t2.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", -8.0, "Transient", "Peak", {"gain_db_individual": -8.0, "original_gain_db": -8.0}
    )
    state.session = SessionContext([t1, t2], {}, processors=[proc])
    state.session_groups = [
        {"name": "Drums", "color": "Red", "gain_linked": True, "daw_target": "Drums",
         "match_method": "contains", "match_pattern": "kick, snare"}
    ]
    table.setRowCount(2)
    table.setItem(0, 0, QTableWidgetItem("kick.wav"))
    table.setItem(1, 0, QTableWidgetItem("snare.wav"))

    ctrl = GroupsController(
        state,
        shell,
        track_table=table,
        on_find_table_row=lambda fn: 0 if fn == "kick.wav" else 1,
    )
    setup_refreshes: list[bool] = []
    ctrl.setup_table_refresh_requested.connect(lambda: setup_refreshes.append(True))
    try:
        ctrl.populate_groups_tab()
        ctrl.create_group_combo(0, t1)
        ctrl.create_group_combo(1, t2)
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        ctrl.auto_group()
        assert t1.group == "Drums"
        assert t2.group == "Drums"
        assert setup_refreshes == [True]
        assert "assigned 2 of 2 tracks" in shell.status_messages[-1]
        assert t1.processor_results["bimodal_normalize"].gain_db == t2.processor_results["bimodal_normalize"].gain_db
    finally:
        ctrl.page.deleteLater()
        table.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_groups_controller_match_combo_validates_shifted_row_after_delete(_font_app):
    state = WorkspaceState(config=build_defaults())
    state.session_groups = [
        {"name": "Drums", "color": "Red", "gain_linked": False, "daw_target": "",
         "match_method": "contains", "match_pattern": "kick"},
        {"name": "Bass", "color": "Blue", "gain_linked": False, "daw_target": "",
         "match_method": "contains", "match_pattern": "[unclosed"},
    ]
    shell = _StubShell()
    table = _DraggableTrackTable()
    table.setColumnCount(8)
    ctrl = GroupsController(state, shell, track_table=table)
    try:
        ctrl.populate_groups_tab()
        assert ctrl.page.table.rowCount() == 2
        # Delete row 0 ("Drums"); "Bass" shifts from row 1 to row 0.
        ctrl.page.table.selectRow(0)
        ctrl.remove_selected_group_row()
        assert ctrl.page.table.rowCount() == 1
        assert ctrl.page.table.item(0, 0).text() == "Bass"

        # Switching match method on the shifted row must validate row 0, not stale row 1.
        match_combo = ctrl.page.table.cellWidget(0, 4)
        match_combo.setCurrentText("regex")
        pat_item = ctrl.page.table.item(0, 5)
        assert pat_item.toolTip().startswith("Invalid regex:")
    finally:
        ctrl.page.deleteLater()
        table.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()
