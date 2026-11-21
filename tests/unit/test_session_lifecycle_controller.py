"""Isolated unit tests for SessionPersistenceController & AnalysisPrepareController."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QStackedWidget, QTabWidget, QWidget

from sessionpreplib.models import SessionContext, TrackContext
from sessionprepgui.analysis.analysis_controller import AnalysisPrepareController
from sessionprepgui.analysis.persistence_controller import SessionPersistenceController
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.tracks.table_widgets import _DraggableTrackTable


class _StubShell:
    def __init__(self) -> None:
        self.parent_widget = QWidget()
        self.status_messages: list[str] = []
        self.progress_events: list[tuple[str, object]] = []
        self.enabled_phases: dict[int, bool] = {}
        self.switched_phases: list[int] = []
        self.titles: list[str] = []

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
        self.progress_events.append(("clear", None))

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None:
        self.enabled_phases[phase_index] = enabled

    def switch_to_phase(self, phase_index: int) -> None:
        self.switched_phases.append(phase_index)

    def current_phase(self) -> int:
        return 0

    def is_phase_enabled(self, phase_index: int) -> bool:
        return self.enabled_phases.get(phase_index, False)

    def register_phase_progress_layout(self, phase_index: int, layout) -> None: pass
    def dialog_parent(self) -> QWidget: return self.parent_widget
    def is_batch_mode(self) -> bool: return False
    def is_closing_after_daw(self) -> bool: return False
    def resume_close_after_daw(self) -> None: pass

    def set_window_title(self, title: str) -> None:
        self.titles.append(title)


def _make_controllers(state: WorkspaceState, shell: _StubShell):
    table = _DraggableTrackTable()
    table.setColumnCount(8)
    tabs = QTabWidget()
    for title in ("Summary", "File", "Groups", "Config"):
        tabs.addTab(QWidget(), title)
    stack = QStackedWidget()
    stack.addWidget(tabs)
    analyze_act = QAction("Reanalyze")
    auto_grp_act = QAction("Auto-Group")
    prepare_act = QAction("Prepare")
    save_act = QAction("Save")

    phase2_page = SimpleNamespace(
        track_table=table,
        summary_view=SimpleNamespace(setHtml=lambda h: None, clear=lambda: None),
        file_report=SimpleNamespace(clear=lambda: None),
        detail_tabs=tabs,
        right_stack=stack,
        analyze_action=analyze_act,
        auto_group_action=auto_grp_act,
        prepare_action=prepare_act,
    )
    topology_ctrl = SimpleNamespace(
        clear_workspace=lambda: None,
        populate=lambda: None,
        on_peak_file_done=lambda k, d: None,
        pending_sources=[],
        apply_worker=None,
        page=SimpleNamespace(
            input_tree=SimpleNamespace(set_source_dir=lambda d: None),
            apply_action=QAction("Apply"),
            reanalyze_action=QAction("Reanalyze"),
            reset_action=QAction("Reset"),
            recursive_cb=SimpleNamespace(setChecked=lambda v: None),
        ),
    )
    daw_ctrl = SimpleNamespace(
        clear_workspace=lambda: None,
        refresh_from_config=lambda: None,
        update_lifecycle_buttons=lambda: None,
        update_use_processed_action=lambda: None,
        page=SimpleNamespace(
            project_name_edit=SimpleNamespace(text=lambda: "MySong", blockSignals=lambda b: None, setText=lambda t: None),
            use_processed_cb=SimpleNamespace(isChecked=lambda: False, setChecked=lambda b: None),
            right_stack=QStackedWidget(),
        ),
    )
    detail_ctrl = SimpleNamespace(
        stop=lambda: None,
        clear_workspace=lambda: None,
        render_summary=lambda: None,
        refresh_verified_summary=lambda: None,
        refresh_track_repair_editor=lambda t=None: None,
        refresh_repair_preview=lambda t: None,
        on_peak_file_done=lambda k, d: None,
    )
    table_ctrl = SimpleNamespace(
        batch_filenames=set(),
        rebuild_track_table_rows=lambda tracks: None,
        populate_table=lambda s: None,
        populate_setup_table=lambda: None,
        auto_fit_track_table=lambda: None,
        find_table_row=lambda fn: -1,
        update_track_row=lambda fn: None,
        refresh_file_tab=lambda t: None,
    )
    groups_ctrl = SimpleNamespace(
        clear_workspace=lambda: None,
        merge_groups_from_preset=lambda: None,
        populate_group_preset_combo=lambda: None,
        populate_groups_tab=lambda: None,
        refresh_group_combos=lambda: None,
        apply_linked_group_levels=lambda: None,
    )
    config_ctrl = SimpleNamespace(
        populate_config_preset_combo=lambda: None,
        load_session_widgets=lambda cfg: None,
        refresh_phase2_dirty_indicators=lambda: None,
        init_session_config=lambda: None,
    )

    analysis = AnalysisPrepareController(
        state,
        shell,
        phase2_page=phase2_page,
        topology_ctrl=topology_ctrl,
        detail_ctrl=detail_ctrl,
        table_ctrl=table_ctrl,
        groups_ctrl=groups_ctrl,
        config_ctrl=config_ctrl,
        get_project_name=lambda: daw_ctrl.page.project_name_edit.text().strip(),
    )
    persistence = SessionPersistenceController(
        state,
        shell,
        phase2_page=phase2_page,
        topology_ctrl=topology_ctrl,
        daw_ctrl=daw_ctrl,
        detail_ctrl=detail_ctrl,
        table_ctrl=table_ctrl,
        groups_ctrl=groups_ctrl,
        config_ctrl=config_ctrl,
        save_session_action=save_act,
        on_is_save_blocked=analysis.is_save_blocked,
    )
    persistence.workspace_clearing.connect(analysis.cancel_all_workers)
    return persistence, analysis, phase2_page, [table, stack, daw_ctrl.page.right_stack, shell.parent_widget]


def test_persistence_and_analysis_capture_and_mark_prepare_stale(_font_app, tmp_path):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, phase2, widgets = _make_controllers(state, shell)
    try:
        track = TrackContext("kick.wav", str(tmp_path / "kick.wav"), None, 48000, 1, 48000, "24-bit", "PCM_24", 1.0)
        state.source_dir = str(tmp_path)
        state.session = SessionContext([track], {}, prepare_state="ready")

        captured = persistence.capture_session_state()
        assert captured["source_dir"] == str(tmp_path)
        assert captured["project_name"] == "MySong"
        assert captured["prepare_state"] == "ready"

        analysis.mark_prepare_stale()
        assert state.session.prepare_state == "stale"
        assert phase2.prepare_action.text() == "Prepare (!)"
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()


def test_persistence_clear_workspace_resets_state_and_empty_dir_status(_font_app, tmp_path):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, phase2, widgets = _make_controllers(state, shell)
    try:
        persistence.load_directory(str(tmp_path))
        assert f"No readable audio files in {tmp_path}" in shell.status_messages[-1]
        assert not phase2.analyze_action.isEnabled()

        wav = tmp_path / "bass.wav"
        sf.write(wav, np.zeros(4800), 48000)
        tracks = persistence.discover_source_tracks(str(tmp_path), recursive=False)
        assert [t.filename for t in tracks] == ["bass.wav"]
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()


def test_repair_plan_worker_progress_value_matches_analysis_slot(_font_app, tmp_path):
    from sessionprepgui.analysis.worker import RepairPlanWorker

    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, _phase2, widgets = _make_controllers(state, shell)
    try:
        wav = tmp_path / "snare.wav"
        sf.write(wav, np.zeros(4800), 48000)
        track = TrackContext("snare.wav", str(wav), np.zeros(4800), 48000, 1, 4800, "16-bit", "PCM_16", 0.1)
        session = SessionContext([track], state.flat_config())
        worker = RepairPlanWorker(session, [track], state.flat_config(), parent=analysis)
        analysis.repair_plan_worker = worker
        worker.progress_value.connect(analysis._on_worker_progress_value)
        worker._on_planned()
        _font_app.processEvents()
        assert ("value", (1, 1)) in shell.progress_events
        analysis.repair_plan_worker = None
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()


def test_persistence_parents_peak_worker_and_drains_on_dispose(_font_app, tmp_path):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, _phase2, widgets = _make_controllers(state, shell)
    try:
        wav = tmp_path / "pad.wav"
        sf.write(wav, np.zeros(4800), 48000)
        track = TrackContext("pad.wav", str(wav), None, 48000, 1, 4800, "16-bit", "PCM_16", 0.1)
        persistence.start_peak_build([track], str(tmp_path))
        worker = persistence.peak_build_worker
        assert worker is not None
        assert worker.parent() is persistence
        persistence.cancel_peak_worker()
        persistence.dispose()
        assert not worker.isRunning()
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()
