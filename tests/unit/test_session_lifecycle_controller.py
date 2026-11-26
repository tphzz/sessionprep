"""Isolated unit tests for SessionPersistenceController & AnalysisPrepareController."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PySide6.QtCore import QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QStackedWidget, QTabWidget, QWidget

from qt_helpers import dispose_window
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
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
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
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
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
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
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
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        _font_app.processEvents()


def test_default_project_name_helper_handles_edges(tmp_path):
    import os
    from sessionprepgui.analysis.persistence_controller import _default_project_name

    assert _default_project_name(str(tmp_path / "My_Song") + os.sep) == "My_Song"
    assert _default_project_name(os.path.join(str(tmp_path), "Spektakulatius_Full")) == "Spektakulatius_Full"
    if os.name == "nt":
        assert _default_project_name(r"D:\multitracks\Spektakulatius_Full") == "Spektakulatius_Full"
    assert _default_project_name(os.path.join(str(tmp_path), "Bad:Name?Test")) == "BadNameTest"
    assert _default_project_name("") == ""
    assert _default_project_name("   ") == ""
    assert _default_project_name(os.path.abspath(os.sep)) == ""
    assert _default_project_name("..") == _default_project_name(os.path.abspath(".."))
    assert _default_project_name(".") == _default_project_name(os.path.abspath("."))
    assert _default_project_name(os.path.join(str(tmp_path), "...")) == ""


def test_load_directory_defaults_phase3_project_name_and_preserves_on_reanalyze(_font_app, tmp_path):
    from PySide6.QtWidgets import QLineEdit

    song_dir = tmp_path / "OurLoveIsHereToStay"
    song_dir.mkdir()
    sf.write(song_dir / "kick.wav", np.zeros(4800), 48000)

    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, _phase2, widgets = _make_controllers(state, shell)
    edit = QLineEdit()
    persistence._daw_ctrl.page.project_name_edit = edit
    # Mirror MainWindow wiring so clear_workspace clears the edit field
    persistence.workspace_cleared.connect(lambda _k: edit.clear())
    widgets.append(edit)
    try:
        persistence.load_directory(str(song_dir))
        assert edit.text() == "OurLoveIsHereToStay"
        assert state.session is not None
        assert state.session.project_name == "OurLoveIsHereToStay"

        # User edits project name -> Reanalyze (keep_session_config=True) preserves custom name
        edit.setText("Custom Mix Name")
        persistence.load_directory(str(song_dir), keep_session_config=True)
        assert edit.text() == "Custom Mix Name"
        assert state.session.project_name == "Custom Mix Name"

        # Opening a different folder resets to the new directory's basename
        other_dir = tmp_path / "AnotherTrack"
        other_dir.mkdir()
        sf.write(other_dir / "snare.wav", np.zeros(4800), 48000)
        persistence.load_directory(str(other_dir))
        assert edit.text() == "AnotherTrack"
        assert state.session.project_name == "AnotherTrack"

        # Opening an empty directory leaves project_name_edit cleared
        empty_dir = tmp_path / "EmptySong"
        empty_dir.mkdir()
        persistence.load_directory(str(empty_dir))
        assert edit.text() == ""
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        _font_app.processEvents()


def test_prepare_and_analyze_stop_phase2_playback(_font_app, tmp_path, monkeypatch):
    from sessionprepgui.analysis import analysis_controller as ac_mod

    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    persistence, analysis, _phase2, widgets = _make_controllers(state, shell)
    stops: list[bool] = []
    analysis._detail_ctrl.stop = lambda: stops.append(True)

    class _NoopWorker:
        def __init__(self, *args, **kwargs):
            self.progress = SimpleNamespace(connect=lambda *a, **k: None)
            self.progress_value = SimpleNamespace(connect=lambda *a, **k: None)
            self.track_analyzed = SimpleNamespace(connect=lambda *a, **k: None)
            self.track_planned = SimpleNamespace(connect=lambda *a, **k: None)
            self.completed = SimpleNamespace(connect=lambda *a, **k: None)
            self.error = SimpleNamespace(connect=lambda *a, **k: None)
            self.finished = SimpleNamespace(connect=lambda *a, **k: None)
            self.layout_files = None
            self.interrupted_repairs = None

        def start(self):
            pass

        def cancel(self):
            pass

        def disconnect(self, *_args):
            pass

        def isRunning(self):
            return False

        def deleteLater(self):
            pass

    monkeypatch.setattr(ac_mod, "PrepareWorker", _NoopWorker)
    monkeypatch.setattr(ac_mod, "AnalyzeWorker", _NoopWorker)
    try:
        state.source_dir = str(tmp_path)
        state.session = SessionContext(tracks=[], config={})

        analysis.prepare()
        assert len(stops) == 1
        analysis.prepare_worker = None

        analysis.analyze()
        assert len(stops) == 2
        analysis.analyze_worker = None
    finally:
        analysis.dispose()
        persistence.dispose()
        analysis.deleteLater()
        persistence.deleteLater()
        for w in widgets:
            w.deleteLater()
        _font_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        _font_app.processEvents()


def test_phase_switches_stop_active_playback(_font_app, monkeypatch):
    from sessionprepgui.mainwindow import SessionPrepWindow
    from sessionprepgui.tracks.table_widgets import _PHASE_ANALYSIS, _PHASE_SETUP, _PHASE_TOPOLOGY

    win = SessionPrepWindow()
    try:
        win.set_phase_enabled(_PHASE_ANALYSIS, True)
        win.set_phase_enabled(_PHASE_SETUP, True)

        playing = [False]
        stop_calls: list[str] = []

        monkeypatch.setattr(type(win._playback), "is_playing", property(lambda _self: playing[0]))
        monkeypatch.setattr(
            win._playback,
            "stop",
            lambda: (playing.__setitem__(0, False), stop_calls.append("playback")),
        )

        # Phase 1 -> Phase 2 stops Phase 1 preview without touching Phase 2 cursor
        dummy = np.zeros(4800, dtype=np.float32)
        win.phase2_page.wf_panel.waveform.set_audio(dummy, 48000)
        win.topology_controller.page.wf_panel.waveform.set_audio(dummy, 48000)
        win.phase2_page.wf_panel.waveform.set_cursor(1234)
        playing[0] = True
        win.switch_to_phase(_PHASE_ANALYSIS)
        assert not playing[0] and len(stop_calls) == 1
        assert win.phase2_page.wf_panel.waveform._cursor_sample == 1234
        assert win._space_shortcut.isEnabled()

        # Phase 2 -> Phase 3 stops Phase 2 playback, clears pending_play, disables Space shortcut
        win.topology_controller.page.wf_panel.waveform.set_cursor(567)
        win._repair_preview.pending_play = (100, "as_is", None)
        playing[0] = True
        win.switch_to_phase(_PHASE_SETUP)
        assert not playing[0] and len(stop_calls) == 2
        assert win._repair_preview.pending_play is None
        assert win.topology_controller.page.wf_panel.waveform._cursor_sample == 567
        assert not win._space_shortcut.isEnabled()

        # Back to Phase 2 from Phase 3, play again, return to Phase 3 -> stops again
        win.switch_to_phase(_PHASE_ANALYSIS)
        assert win._space_shortcut.isEnabled()
        playing[0] = True
        win.switch_to_phase(_PHASE_SETUP)
        assert not playing[0] and len(stop_calls) == 3

        # Phase 2 -> Phase 1 and direct Phase 1 -> Phase 3 both stop playback
        win.switch_to_phase(_PHASE_ANALYSIS)
        playing[0] = True
        win.switch_to_phase(_PHASE_TOPOLOGY)
        assert not playing[0] and len(stop_calls) == 4
        playing[0] = True
        win.switch_to_phase(_PHASE_SETUP)
        assert not playing[0] and len(stop_calls) == 5
    finally:
        dispose_window(win)


def test_topology_apply_overwrite_and_clear_stop_preview(_font_app, tmp_path, monkeypatch):
    from sessionprepgui.analysis import worker as worker_mod
    from sessionprepgui.mainwindow import SessionPrepWindow

    win = SessionPrepWindow()
    try:
        playing = [False]
        monkeypatch.setattr(type(win._playback), "is_playing", property(lambda _self: playing[0]))
        monkeypatch.setattr(win._playback, "stop", lambda: playing.__setitem__(0, False))

        class _NoopApplyWorker:
            def __init__(self, *args, **kwargs):
                for name in ("finished", "progress", "progress_value",
                             "overwrite_confirmation_required", "apply_finished", "error"):
                    setattr(self, name, SimpleNamespace(connect=lambda *a, **k: None))

            def setParent(self, _p): pass
            def start(self): pass
            def cancel(self): pass
            def deleteLater(self): pass

        monkeypatch.setattr(worker_mod, "TopologyApplyWorker", _NoopApplyWorker)
        win.state.source_dir = str(tmp_path)
        win.state.session = SessionContext(tracks=[], config={})

        # Initial Apply stops Phase 1 preview
        playing[0] = True
        win.topology_controller.apply_topology()
        assert not playing[0]
        win.topology_controller.apply_worker = None

        # Overwrite confirmation retry also stops Phase 1 preview
        out_dir, _ = win.topology_controller._layout_output_dir()
        win.topology_controller._pending_overwrite = (out_dir, {"a.wav": "rev1"})
        playing[0] = True
        win.topology_controller._confirm_overwrite_apply()
        assert not playing[0]
        win.topology_controller.apply_worker = None

        # clear_workspace stops playback and resets Phase 1 stop_btn/play_btn
        win.topology_controller.page.wf_panel.stop_btn.setEnabled(True)
        playing[0] = True
        win.topology_controller.clear_workspace()
        assert not playing[0]
        assert not win.topology_controller.page.wf_panel.stop_btn.isEnabled()
        assert not win.topology_controller.page.wf_panel.play_btn.isEnabled()
    finally:
        dispose_window(win)
