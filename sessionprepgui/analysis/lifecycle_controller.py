"""Session lifecycle controller: open/save/load, Phase 1/2 analysis, peaks, repairs, Prepare."""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR
import logging
import os
from typing import Any

from PySide6.QtCore import QObject, QThread, SIGNAL, SLOT, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
)

from sessionpreplib.audio import discover_audio_files, discover_track
from sessionpreplib.config import (
    REPAIR_PARAMS,
    classify_structured_config_change,
    default_config,
    infer_session_overrides,
    resolve_session_config,
)
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import LifecyclePhase, SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.repair_plan import check_source_signature
from sessionpreplib.repairs import RepairRecipe
from sessionpreplib.topology import build_default_topology
from sessionpreplib.utils import protools_sort_key

from ..controls import FormDoubleSpinBox, set_action_busy
from ..detail.report_style import wrap_report_html
from ..helpers import esc, track_analysis_label
from ..prefs.config_pages import _config_section_values
from ..session.io import (
    capture_phase1_sources,
    load_session as _load_session_file,
    restore_layout_sources,
    restore_phase1_source,
    save_session as _save_session_file,
)
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import COLORS, FILE_COLOR_ERROR, FILE_COLOR_OK
from ..tracks.table_widgets import (
    _PAGE_TABS,
    _PHASE_ANALYSIS,
    _PHASE_SETUP,
    _PHASE_TOPOLOGY,
    _SETUP_RIGHT_TREE,
    _TAB_FILE,
    _TAB_GROUPS,
    _TAB_SESSION,
    _TAB_SUMMARY,
    _SortableItem,
    _make_analysis_cell,
)
from ..waveform.compute import PeakBuildWorker
from ..waveform.peakcache import get_source_mtime, load_peaks, peak_source_key, peaks_path_for
from .worker import AnalyzeWorker, Phase1AnalyzeWorker, PrepareWorker, RepairPlanWorker

log = logging.getLogger(__name__)


class SessionLifecycleController(QObject):
    """Owns workspace open/save/load/clear and all Phase 1/2 background workers."""

    workspace_opened = Signal(str)
    analysis_completed = Signal(object, object)
    prepare_completed = Signal(object)

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        topology_ctrl: Any,
        daw_ctrl: Any,
        detail_ctrl: Any,
        table_ctrl: Any,
        groups_ctrl: Any,
        config_ctrl: Any,
        track_table: Any,
        summary_view: Any,
        file_report: Any,
        detail_tabs: Any,
        right_stack: Any,
        analyze_action: Any,
        auto_group_action: Any,
        prepare_action: Any,
        save_session_action: Any,
        repair_preview: Any = None,
        on_progress_owner: Callable[[], Any] | None = None,
        on_update_tools_menu: Callable[[], None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._topology_ctrl = topology_ctrl
        self._daw_ctrl = daw_ctrl
        self._detail_ctrl = detail_ctrl
        self._table_ctrl = table_ctrl
        self._groups_ctrl = groups_ctrl
        self._config_ctrl = config_ctrl
        self._track_table = track_table
        self._summary_view = summary_view
        self._file_report = file_report
        self._detail_tabs = detail_tabs
        self._right_stack = right_stack
        self._analyze_action = analyze_action
        self._auto_group_action = auto_group_action
        self._prepare_action = prepare_action
        self._save_session_action = save_session_action
        self._repair_preview = repair_preview
        self._on_progress_owner = on_progress_owner
        self._on_update_tools_menu = on_update_tools_menu

        self.p1_worker: Any = None
        self.analyze_worker: Any = None
        self.prepare_worker: Any = None
        self.repair_plan_worker: Any = None
        self.peak_build_worker: Any = None
        self.repair_plan_pending: set[str] = set()
        self.repair_plan_anchors: set[str] = set()
        self.repair_plan_revalidate: set[str] = set()
        self._zombie_workers: set[Any] = set()

    # ── Worker lifecycle & shutdown ──────────────────────────────────────

    def cancel_worker(self, attr_name: str) -> None:
        """Safely detach and cancel an owned background worker."""
        worker = getattr(self, attr_name, None)
        if isinstance(worker, (Phase1AnalyzeWorker, AnalyzeWorker)):
            setattr(self, attr_name, None)
            worker.cancel()
            if not worker.isRunning():
                worker.deleteLater()
            return
        if isinstance(worker, (RepairPlanWorker, PrepareWorker)):
            setattr(self, attr_name, None)
            worker.disconnect(self)
            if isinstance(worker, RepairPlanWorker):
                worker.cancel()
            else:
                worker.requestInterruption()
            worker.finished.connect(worker.deleteLater)
            if worker.isFinished():
                worker.deleteLater()
            return
        if worker is not None:
            if hasattr(worker, "isRunning") and worker.isRunning():
                try:
                    worker.disconnect(self)
                except RuntimeError:
                    pass
                cancel_fn = getattr(worker, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    worker.requestInterruption()
                self._zombie_workers.add(worker)

                def _cleanup(*_args: Any, w: Any = worker) -> None:
                    self._zombie_workers.discard(w)

                if hasattr(worker, "finished"):
                    worker.finished.connect(_cleanup)
                if hasattr(worker, "error"):
                    worker.error.connect(_cleanup)
                if hasattr(worker, "isFinished") and worker.isFinished():
                    self._zombie_workers.discard(worker)
            setattr(self, attr_name, None)

    def dispose(self) -> None:
        """Stop all Phase 1/2 workers cleanly before window close."""
        for kind in (Phase1AnalyzeWorker, AnalyzeWorker, RepairPlanWorker):
            for child in self.findChildren(kind):
                child_any: Any = child
                child_any.cancel()
                child_any.wait()
        self.repair_plan_worker = None
        for prep in self.findChildren(PrepareWorker):
            prep.requestInterruption()
            prep.wait()
        self.prepare_worker = None

        for attr in ("p1_worker", "analyze_worker", "peak_build_worker"):
            w: Any = getattr(self, attr, None)
            if isinstance(w, QThread) and w.isRunning():
                cancel_fn = getattr(w, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    w.requestInterruption()
                w.wait()
            setattr(self, attr, None)

        for zombie in list(self._zombie_workers):
            if isinstance(zombie, QThread) and zombie.isRunning():
                cancel_fn = getattr(zombie, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    zombie.requestInterruption()
                zombie.wait()
        self._zombie_workers.clear()

    # ── Clear workspace ──────────────────────────────────────────────────

    def clear_workspace(self, *, keep_session_config: bool = False) -> None:
        """Clear all UI state and cancel in-flight workers."""
        self._detail_ctrl.stop()
        self._topology_ctrl.clear_workspace()
        for attr in ("p1_worker", "analyze_worker", "prepare_worker", "repair_plan_worker", "peak_build_worker"):
            self.cancel_worker(attr)
        self.repair_plan_pending = set()
        self.repair_plan_anchors = set()
        self.repair_plan_revalidate = set()
        self._table_ctrl.batch_filenames = set()
        self._track_table.setSortingEnabled(True)
        set_action_busy(self._prepare_action, False)

        self._shell.clear_progress()
        self._state.reset_workspace(keep_session_config=keep_session_config)
        self._daw_ctrl.clear_workspace()
        self._detail_ctrl.clear_workspace()
        self._groups_ctrl.clear_workspace()

        self._analyze_action.setEnabled(False)
        self._save_session_action.setEnabled(False)
        self._shell.switch_to_phase(_PHASE_TOPOLOGY)
        self._shell.set_phase_enabled(_PHASE_ANALYSIS, False)
        self._shell.set_phase_enabled(_PHASE_SETUP, False)
        self._track_table.setRowCount(0)
        self._track_table.set_source_dir(None)
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, False)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, False)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._shell.set_window_title("SessionPrep")

    # ── Open folder & Phase 1 scan ───────────────────────────────────────

    @Slot()
    def open_directory_dialog(self) -> None:
        start_dir = self._state.config.get("app", {}).get("default_project_dir", "") or ""
        path = QFileDialog.getExistingDirectory(
            self._shell.dialog_parent(),
            "Select Session Directory",
            start_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self.load_directory(path)

    @Slot()
    def reanalyze_topology(self) -> None:
        if self._state.source_dir:
            self.load_directory(self._state.source_dir, keep_session_config=True)

    def load_directory(self, path: str, *, keep_session_config: bool = False) -> None:
        log.info("Open folder: %s", path)
        self.clear_workspace(keep_session_config=keep_session_config)
        self._state.source_dir = path
        self._track_table.set_source_dir(path)
        self._topology_ctrl.page.input_tree.set_source_dir(path)
        self._config_ctrl.populate_config_preset_combo()
        self._config_ctrl.load_session_widgets(self._state.effective_session_config())
        self._daw_ctrl.refresh_from_config()
        if self._on_update_tools_menu is not None:
            self._on_update_tools_menu()

        tracks = self._discover_source_tracks(path, recursive=self._state.recursive_scan)
        if not tracks:
            self._shell.show_status(f"No readable audio files in {path}")
            self._analyze_action.setEnabled(False)
            return

        self._state.topo_source_tracks = list(tracks)
        self._state.topo_topology = None
        self._state.session = SessionContext(tracks=tracks, config={})
        self.workspace_opened.emit(path)
        self.run_phase1_analysis()

    def _discover_source_tracks(self, directory: str, *, recursive: bool) -> list[TrackContext]:
        app_cfg = self._state.config.get("app", {})
        skip_folders = {
            app_cfg.get("phase1_output_folder", "sp_01_tracklayout"),
            app_cfg.get("phase2_output_folder", "sp_02_prepared"),
            app_cfg.get("peak_cache_folder", "sp_peaks"),
        }
        tracks: list[TrackContext] = []
        for rel in discover_audio_files(directory, recursive=recursive, skip_folders=skip_folders):
            try:
                tc = discover_track(os.path.join(directory, rel))
                tc.filename = rel
                tracks.append(tc)
            except Exception:
                pass
        return tracks

    def run_phase1_analysis(self) -> None:
        self._analyze_action.setEnabled(False)
        self._topology_ctrl.page.apply_action.setEnabled(False)
        self._shell.progress_start("Analyzing Format & Layout\u2026")
        assert self._state.session is not None
        self.start_phase1_worker(self._state.session)

    def start_phase1_worker(self, session: SessionContext) -> None:
        config = self._state.flat_config()
        config["_source_dir"] = self._state.source_dir
        worker = Phase1AnalyzeWorker(session, config)
        worker.setParent(self)
        QObject.connect(worker, SIGNAL("finished()"), worker, SLOT("deleteLater()"))
        self.p1_worker = worker
        if not self._topology_ctrl.pending_sources:
            worker.progress.connect(self._on_phase1_progress, Qt.ConnectionType.QueuedConnection)
            worker.progress_value.connect(self._on_phase1_progress_value, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_phase1_done)
        worker.error.connect(self._on_phase1_error, Qt.ConnectionType.QueuedConnection)
        worker.start()

    @Slot(str)
    def _on_phase1_progress(self, message: str) -> None:
        if self.p1_worker is not None and self.sender() is self.p1_worker:
            self._shell.progress_message(message)

    @Slot(int, int)
    def _on_phase1_progress_value(self, current: int, total: int) -> None:
        if self.p1_worker is not None and self.sender() is self.p1_worker:
            self._shell.progress_value(current, total)

    @Slot(object)
    def _on_phase1_done(self, session: SessionContext) -> None:
        worker = self.p1_worker
        if worker is None or worker.session_context is not session:
            return
        self.p1_worker = None
        if not worker.isRunning():
            worker.deleteLater()
        if self._topology_ctrl.pending_sources:
            self._on_reopened_sources_checked(session)
            return

        self._state.session = session
        self._state.topo_topology = build_default_topology(session.tracks)
        self._topology_ctrl.populate()

        self._analyze_action.setEnabled(True)
        self._topology_ctrl.page.reanalyze_action.setEnabled(True)
        self._topology_ctrl.page.reset_action.setEnabled(True)
        self._shell.show_status(
            f"Discovered {len(session.tracks)} file(s) from {self._state.source_dir} \u2014 "
            "review layout, then click Apply"
        )
        self._shell.progress_finish(f"Discovered {len(session.tracks)} file(s)")
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._save_session_action.setEnabled(True)
        self._shell.set_window_title("SessionPrep")
        self.start_peak_build(session.tracks, self._state.source_dir)

    def check_reopened_layout_sources(self) -> None:
        if not self._topology_ctrl.pending_sources or self.p1_worker is not None:
            return
        tracks = [replace(t, _cache={}, detector_results={}) for t in self._topology_ctrl.pending_sources]
        self._shell.show_status("Checking Track Layout source files…")
        self.start_phase1_worker(SessionContext(tracks=tracks, config={}))

    def _on_reopened_sources_checked(self, checked: SessionContext) -> None:
        refreshed = {t.filename: t for t in checked.tracks}
        for track in checked.tracks:
            track.audio_data = None
            track._cache.clear()
        self._state.topo_source_tracks = [
            refreshed.get(t.filename, t) for t in self._state.topo_source_tracks
        ]
        self._topology_ctrl.pending_sources = []
        pipeline = Pipeline(default_detectors(), config=self._state.flat_config())
        pipeline.analyze_session_detectors(
            SessionContext(self._state.topo_source_tracks, {}), LifecyclePhase.PHASE1
        )
        if self._topology_ctrl.apply_worker is None:
            self._topology_ctrl.populate()
            self._topology_ctrl.page.reset_action.setEnabled(True)
        else:
            self._topology_ctrl.page.input_tree.populate(
                self._state.topo_source_tracks, self._state.topo_topology
            )
        failed = sum(t.status != "OK" for t in checked.tracks)
        self._shell.show_status(
            f"Track Layout checked. {failed} unreadable source file(s)."
            if failed
            else "Track Layout source checks complete."
        )

    @Slot(str)
    def _on_phase1_error(self, message: str) -> None:
        worker: Any = self.p1_worker
        sender: Any = self.sender()
        if worker is None or sender is not worker:
            return
        self.p1_worker = None
        if not worker.isRunning():
            worker.deleteLater()
        if self._topology_ctrl.pending_sources:
            self._shell.show_status(f"Could not check Track Layout sources: {message}")
        else:
            self._on_analyze_error(message)

    # ── Save & load session ──────────────────────────────────────────────

    @Slot()
    def save_session_dialog(self) -> None:
        if (
            not self._state.session
            or not self._state.source_dir
            or self.prepare_worker is not None
            or self.repair_plan_worker is not None
        ):
            return
        log.info("Save session")
        default_path = os.path.join(self._state.source_dir, "session.spsession")
        path, _ = QFileDialog.getSaveFileName(
            self._shell.dialog_parent(),
            "Save Session",
            default_path,
            "SessionPrep Session (*.spsession);;All Files (*)",
        )
        if not path:
            return
        try:
            _save_session_file(path, self.capture_session_state())
            self._shell.show_status(f"Session saved to {path}")
        except Exception as exc:
            QMessageBox.critical(
                self._shell.dialog_parent(), "Save Session Failed", f"Could not save session:\n\n{exc}"
            )

    def capture_session_state(self) -> dict[str, Any]:
        session = self._state.session
        if session:
            session.project_name = self._daw_ctrl.page.project_name_edit.text().strip()
        active_dp = self._state.active_daw_processor
        return {
            "source_dir": self._state.source_dir,
            "active_config_preset": self._state.active_config_preset_name,
            "session_config": self._state.effective_session_config(),
            "session_config_overrides": copy.deepcopy(self._state.session_overrides),
            "session_groups": self._state.session_groups,
            "daw_state": session.daw_state if session else {},
            "active_daw_processor_id": active_dp.id if active_dp else None,
            "tracks": session.tracks if session else [],
            "phase1_sources": capture_phase1_sources(self._state.topo_source_tracks),
            "output_tracks": session.output_tracks if session else [],
            "topology": self._state.topo_topology,
            "transfer_manifest": session.transfer_manifest if session else [],
            "topology_applied": self._state.topology_dir is not None,
            "topology_dir": self._state.topology_dir,
            "prepare_state": session.prepare_state if session else "none",
            "prepare_errors": session.config.get("_prepare_errors", []) if session else [],
            "output_gain_adjustments": session.config.get("_output_gain_adjustments", {}) if session else {},
            "base_transfer_manifest": session.base_transfer_manifest if session else [],
            "use_processed": self._daw_ctrl.page.use_processed_cb.isChecked(),
            "recursive_scan": self._state.recursive_scan,
            "project_name": session.project_name if session else "",
        }

    @Slot()
    def load_session_dialog(self) -> None:
        start_dir = (
            self._state.source_dir
            or self._state.config.get("app", {}).get("default_project_dir", "")
            or ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self._shell.dialog_parent(),
            "Load Session",
            start_dir,
            "SessionPrep Session (*.spsession);;All Files (*)",
        )
        if not path:
            return
        log.info("Load session: %s", path)
        try:
            data = _load_session_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self._shell.dialog_parent(), "Load Session Failed", f"Could not load session:\n\n{exc}"
            )
            return
        self.restore_session_state(data)

    def restore_session_state(self, data: dict[str, Any]) -> bool:
        source_dir = data["source_dir"]
        if not os.path.isdir(source_dir):
            QMessageBox.warning(
                self._shell.dialog_parent(),
                "Load Session",
                f"The session's audio directory no longer exists:\n\n{source_dir}\n\n"
                "Please move the files back or open the directory manually.",
            )
            return False

        source_tracks = self._restore_workspace_inputs(data, source_dir)
        effective, daw_state = self._restore_config_and_daw(data)
        flat = self._state.flat_config()
        pipeline = Pipeline(default_detectors(), default_processors(), flat)
        self._restore_phase1_source_records(data, source_tracks, pipeline)

        session, analysis_changed, repair_defaults_changed = self._build_restored_session(
            data, source_dir, effective, flat, daw_state, pipeline
        )
        self._state.session = session
        self._finish_restore_ui(data, session, source_dir, source_tracks, pipeline, analysis_changed, repair_defaults_changed, flat)
        return True

    def _restore_workspace_inputs(self, data: dict[str, Any], source_dir: str) -> list[TrackContext]:
        self.clear_workspace()
        self._state.source_dir = source_dir
        self._track_table.set_source_dir(source_dir)
        self._topology_ctrl.page.input_tree.set_source_dir(source_dir)
        was_recursive = data.get("recursive_scan", False)
        self._state.recursive_scan = was_recursive
        self._topology_ctrl.page.recursive_cb.setChecked(was_recursive)
        source_tracks = self._discover_source_tracks(source_dir, recursive=was_recursive)
        self._state.topo_source_tracks = source_tracks
        self._state.topo_topology = data.get("topology")
        return source_tracks

    def _restore_config_and_daw(self, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        preset_name = data.get("active_config_preset", "Default")
        self._state.active_config_preset_name = self._state.existing_config_preset_name(preset_name)
        overrides = data.get("session_config_overrides")
        self._state.session_overrides = (
            copy.deepcopy(overrides)
            if isinstance(overrides, dict)
            else infer_session_overrides(self._state.active_preset(), data.get("session_config"))
        )
        self._state.session_groups = data.get("session_groups", [])
        self._config_ctrl.populate_config_preset_combo()
        effective = self._state.effective_session_config()
        self._config_ctrl.load_session_widgets(effective)
        self._daw_ctrl.configure_daw_processors()
        daw_state, active_daw_id = self._daw_ctrl.restore_daw_bindings(data)
        self._daw_ctrl.populate_daw_combo(active_daw_id)
        return effective, daw_state

    def _restore_phase1_source_records(
        self, data: dict[str, Any], source_tracks: list[TrackContext], pipeline: Pipeline
    ) -> None:
        records = data.get("phase1_sources")
        records = records if isinstance(records, dict) else {}
        self._topology_ctrl.pending_sources = [
            t for t in source_tracks
            if not restore_phase1_source(t, records.get(t.filename), pipeline.repair_verification)
        ]
        pipeline.analyze_session_detectors(SessionContext(source_tracks, {}), LifecyclePhase.PHASE1)
        self._topology_ctrl.page.reset_action.setEnabled(
            bool(source_tracks) and not self._topology_ctrl.pending_sources
        )
        self._topology_ctrl.page.reanalyze_action.setEnabled(bool(source_tracks))

    def _build_restored_session(
        self,
        data: dict[str, Any],
        source_dir: str,
        effective: dict[str, Any],
        flat: dict[str, Any],
        daw_state: dict[str, Any],
        pipeline: Pipeline,
    ) -> tuple[SessionContext, bool, bool]:
        session_config_flat = dict(default_config())
        session_config_flat.update(flat)
        session_config_flat["_source_dir"] = source_dir
        session_config_flat["_prepare_errors"] = data.get("prepare_errors", [])
        session_config_flat["_output_gain_adjustments"] = data.get("output_gain_adjustments", {})

        prepare_state = data.get("prepare_state", "none")
        saved_repairs = _config_section_values(data.get("session_config") or {}, "processors.corrective")
        current_repairs = _config_section_values(effective, "processors.corrective")
        repair_defaults_changed = any(
            saved_repairs.get(p.key, p.default) != current_repairs.get(p.key, p.default)
            for p in REPAIR_PARAMS
        )
        since_saved = infer_session_overrides(effective, data.get("session_config"))
        impact = classify_structured_config_change(resolve_session_config(effective, since_saved), effective)
        analysis_changed = impact.requires_phase1 or impact.requires_phase2
        if prepare_state == "ready" and (analysis_changed or repair_defaults_changed):
            prepare_state = "stale"

        session = SessionContext(
            tracks=data["tracks"],
            config=session_config_flat,
            groups={},
            detectors=pipeline.repair_verification.detectors + pipeline.session_detectors,
            processors=pipeline.audio_processors,
            daw_state=daw_state,
            prepare_state=prepare_state,
            transfer_manifest=data.get("transfer_manifest", []),
            base_transfer_manifest=data.get("base_transfer_manifest", []),
            project_name=data.get("project_name", ""),
        )
        return session, analysis_changed, repair_defaults_changed

    def _finish_restore_ui(
        self,
        data: dict[str, Any],
        session: SessionContext,
        source_dir: str,
        source_tracks: list[TrackContext],
        pipeline: Pipeline,
        analysis_changed: bool,
        repair_defaults_changed: bool,
        flat: dict[str, Any],
    ) -> None:
        edit = self._daw_ctrl.page.project_name_edit
        edit.blockSignals(True)
        edit.setText(session.project_name)
        edit.blockSignals(False)

        self._state.summary = build_diagnostic_summary(session, remaining=True)
        self._rebuild_track_table_rows(session.tracks)
        self._groups_ctrl.populate_groups_tab()
        self._groups_ctrl.populate_group_preset_combo()
        self._table_ctrl.populate_table(session)
        self._detail_ctrl.render_summary()

        app_cfg = self._state.config.get("app", {})
        layout_dir = restore_layout_sources(data, app_cfg.get("phase1_output_folder", "sp_01_tracklayout"))
        if layout_dir and os.path.isdir(layout_dir):
            self._state.topology_dir = layout_dir
        session.output_tracks = (
            data["output_tracks"]
            if data.get("output_tracks")
            else self._rebuild_output_tracks_from_layout(session, source_dir)
        )

        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._topology_ctrl.populate()
        self._shell.set_phase_enabled(_PHASE_ANALYSIS, True)
        self._shell.switch_to_phase(_PHASE_ANALYSIS)

        has_manifest = bool(session.transfer_manifest)
        self._shell.set_phase_enabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._table_ctrl.populate_setup_table()
        self._analyze_action.setEnabled(True)
        self._save_session_action.setEnabled(True)
        self.update_prepare_button()

        use_processed = data.get("use_processed", False)
        session.config["_use_processed"] = use_processed
        self._daw_ctrl.page.use_processed_cb.setChecked(use_processed)
        self._daw_ctrl.update_use_processed_action()
        self._daw_ctrl.update_lifecycle_buttons()
        if has_manifest and self._state.active_daw_processor:
            dp_state = session.daw_state.get(self._state.active_daw_processor.id, {})
            if dp_state.get("folders"):
                self._daw_ctrl.populate_folder_tree()
                self._daw_ctrl.page.right_stack.setCurrentIndex(_SETUP_RIGHT_TREE)
        self._table_ctrl.auto_fit_track_table()

        ok_count = sum(1 for t in session.tracks if t.status == "OK")
        hint = "" if has_manifest else " — run Prepare to enable Session Setup"
        self._shell.show_status(f"Session loaded: {ok_count}/{len(session.tracks)} tracks OK{hint}")
        self._shell.set_window_title("SessionPrep")
        self.start_peak_build(source_tracks, source_dir, (session.tracks, self._state.topology_dir))

        if repair_defaults_changed:
            self.queue_repair_plan(session.tracks)
        else:
            unverified = [
                t for t in session.tracks
                if (t.repair_recipe.active and not pipeline.repair_verification.has_current_evidence(t))
                or (analysis_changed and self.repair_settings_can_apply(t, flat))
            ]
            if unverified:
                self.queue_repair_plan(unverified, verification_only=True)

    def _rebuild_output_tracks_from_layout(self, session: SessionContext, source_dir: str) -> list[TrackContext]:
        if not self._state.topology_dir or not self._state.topo_topology:
            return []
        import soundfile as sf

        prep_folder = self._state.config.get("app", {}).get("phase2_output_folder", "sp_02_prepared")
        prep_dir = os.path.join(source_dir, prep_folder)
        manifest_group = {e.output_filename: e.group for e in session.transfer_manifest}
        rebuilt: list[TrackContext] = []
        for entry in self._state.topo_topology.entries:
            topo_path = os.path.join(self._state.topology_dir, entry.output_filename)
            if not os.path.isfile(topo_path):
                continue
            try:
                info = sf.info(topo_path)
            except Exception:
                continue
            proc_path = os.path.join(prep_dir, entry.output_filename)
            out_tc = TrackContext(
                filename=entry.output_filename,
                filepath=topo_path,
                audio_data=None,
                samplerate=info.samplerate,
                channels=info.channels,
                total_samples=info.frames,
                bitdepth=str(info.subtype_info) if hasattr(info, "subtype_info") else "",
                subtype=info.subtype,
                duration_sec=info.duration,
                processed_filepath=proc_path if os.path.isfile(proc_path) else None,
            )
            out_tc.group = manifest_group.get(entry.output_filename)
            rebuilt.append(out_tc)
        return rebuilt

    # ── Phase 2 Analysis ─────────────────────────────────────────────────

    def _snapshot_track_settings(self) -> dict[str, dict[str, Any]]:
        track_settings: dict[str, dict[str, Any]] = {}
        if not self._state.session:
            return track_settings
        for t in self._state.session.tracks:
            overrides: dict[str, Any] = {}
            if t.classification_override:
                overrides["classification_override"] = t.classification_override
            if t.rms_anchor_override:
                overrides["rms_anchor_override"] = t.rms_anchor_override
            if t.processor_skip or "audio_classifier" in t.detector_results:
                overrides["processor_skip"] = set(t.processor_skip)
            if t.repair_overrides:
                overrides["repair_overrides"] = dict(t.repair_overrides)
            if t.manual_gain_db is not None:
                overrides["manual_gain_db"] = t.manual_gain_db
            if t.group is not None:
                overrides["group"] = t.group
            if overrides:
                track_settings[t.filename] = overrides
        return track_settings

    @Slot()
    def analyze(self) -> None:
        if self.prepare_worker is not None or not self._state.source_dir:
            return
        analyze_dir = self._state.topology_dir or self._state.source_dir
        log.info("Analyze: %s", analyze_dir)
        if self.repair_plan_worker is not None:
            self.repair_plan_pending.update(self.repair_plan_worker.targets)
        interrupted_repairs = (
            self.repair_plan_pending,
            self.repair_plan_anchors,
            self.repair_plan_revalidate,
        )
        self.cancel_worker("repair_plan_worker")
        self.repair_plan_pending = set()
        self.repair_plan_anchors = set()
        self.repair_plan_revalidate = set()
        self._table_ctrl.batch_filenames = set()
        self._track_table.setSortingEnabled(True)

        track_settings = self._snapshot_track_settings()
        self._analyze_action.setEnabled(False)
        self._state.current_track = None
        self._file_report.clear()
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._track_table.setVisible(False)
        if not self._detail_tabs.isTabEnabled(_TAB_SESSION):
            self._config_ctrl.init_session_config()

        self._shell.progress_start("Analyzing\u2026")
        config = self._state.flat_config()
        config["_source_dir"] = self._state.source_dir

        worker: Any = AnalyzeWorker(
            analyze_dir, config, recursive=self._state.recursive_scan, track_settings=track_settings
        )
        worker.setParent(self)
        worker.finished.connect(worker.deleteLater)
        if self._state.topology_dir and self._state.topo_topology is not None:
            worker.layout_files = [
                e.output_filename
                for e in self._state.topo_topology.entries
                if e.output_channels and e.sources
            ]
        worker.interrupted_repairs = interrupted_repairs
        worker.progress.connect(self._on_worker_progress)
        worker.progress_value.connect(self._on_worker_progress_value)
        worker.track_analyzed.connect(self._on_track_analyzed)
        worker.track_planned.connect(self._on_track_planned)
        worker.completed.connect(self._on_analyze_done)
        worker.error.connect(self._on_phase2_error)
        self.analyze_worker = worker
        worker.start()

    @Slot(object, str)
    def _on_worker_progress(self, worker: Any, message: str) -> None:
        if worker is self.analyze_worker:
            self._shell.progress_message(message)

    @Slot(object, int, int)
    def _on_worker_progress_value(self, worker: Any, current: int, total: int) -> None:
        if worker is self.analyze_worker or worker is self.repair_plan_worker:
            self._shell.progress_value(current, total)

    @Slot(object, str, object)
    def _on_track_analyzed(self, worker: Any, filename: str, track: Any) -> None:
        if worker is not self.analyze_worker:
            return
        row = self._table_ctrl.find_table_row(filename)
        if row < 0:
            return
        ch_item = _SortableItem(str(track.channels), track.channels)
        ch_item.setForeground(QColor(COLORS["metadata"]))
        self._track_table.setItem(row, 1, ch_item)
        _plain, html, _color, sort_key = track_analysis_label(track)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._track_table.setItem(row, 2, item)
        self._track_table.setCellWidget(row, 2, lbl)

    @Slot(object, str, object)
    def _on_track_planned(self, worker: Any, filename: str, track: Any) -> None:
        if worker is not self.analyze_worker:
            return
        row = self._table_ctrl.find_table_row(filename)
        if row < 0:
            return
        dets = self._state.session.detectors if self._state.session else None
        for col in (3, 4, 5):
            self._track_table.removeCellWidget(row, col)
        self._table_ctrl._populate_track_row(row, track, dets)
        self._table_ctrl.auto_fit_track_table()

    def _rebuild_track_table_rows(self, tracks: list[TrackContext]) -> None:
        self._track_table.setSortingEnabled(False)
        self._track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            item = _SortableItem(track.filename, protools_sort_key(track.filename))
            item.setForeground(FILE_COLOR_OK if track.status == "OK" else FILE_COLOR_ERROR)
            self._track_table.setItem(row, 0, item)
            for col in range(1, self._track_table.columnCount()):
                cell = _SortableItem("", "")
                cell.setForeground(QColor(COLORS["dim"]))
                self._track_table.setItem(row, col, cell)
        self._track_table.setSortingEnabled(True)

    @Slot(object, object, object)
    def _on_analyze_done(self, worker: Any, session: SessionContext, summary: dict[str, Any]) -> None:
        if worker is not self.analyze_worker:
            return
        prev = self._state.session
        if prev and prev.topology and not self._state.topo_topology:
            self._state.topo_topology = prev.topology
        session.project_name = self._daw_ctrl.page.project_name_edit.text().strip()
        if prev and prev.output_tracks:
            session.output_tracks = prev.output_tracks
            session.prepare_state = "stale"
            session.transfer_manifest = prev.transfer_manifest
            session.base_transfer_manifest = prev.base_transfer_manifest
            session.daw_state = prev.daw_state

        self._state.session = session
        self._state.summary = summary
        self._analyze_action.setEnabled(True)
        self._track_table.setVisible(True)
        if not worker.isRunning():
            worker.deleteLater()
        self.analyze_worker = None

        if not self._state.session_groups:
            self._state.active_session_preset = "Default"
            self._groups_ctrl.merge_groups_from_preset()
            self._groups_ctrl.populate_group_preset_combo()
        else:
            self._groups_ctrl.populate_groups_tab()
            self._groups_ctrl.refresh_group_combos()

        self._rebuild_track_table_rows(session.tracks)
        self._table_ctrl.populate_table(session)
        self._detail_ctrl.render_summary()
        self._post_analyze_finalize(session, summary)

    def _post_analyze_finalize(self, session: SessionContext, summary: dict[str, Any]) -> None:
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._topology_ctrl.populate()

        has_manifest = bool(session.transfer_manifest)
        self._shell.set_phase_enabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._table_ctrl.populate_setup_table()
        if session.prepare_state == "ready":
            session.prepare_state = "stale"
        self.update_prepare_button()
        self._save_session_action.setEnabled(True)

        ok_count = sum(1 for t in session.tracks if t.status == "OK")
        msg = f"Analysis complete: {ok_count}/{len(session.tracks)} tracks OK"
        log.info("Analyze complete: %d/%d tracks OK", ok_count, len(session.tracks))
        self._shell.show_status(msg)
        self._shell.progress_finish(msg)

        if self._state.topology_dir:
            self.start_peak_build(
                session.tracks,
                self._state.topology_dir,
                (self._state.topo_source_tracks, self._state.source_dir),
            )
        elif self._state.source_dir:
            self.start_peak_build(session.tracks, self._state.source_dir)
        if self.repair_plan_pending:
            self.queue_repair_plan(
                [t for t in session.tracks if t.filename in self.repair_plan_pending]
            )
        self.analysis_completed.emit(session, summary)

    # ── Peak cache ───────────────────────────────────────────────────────

    def start_peak_build(self, tracks: list[Any], base_dir: str | None, *additional_groups: Any) -> None:
        peaks_folder = self._state.config.get("app", {}).get("peak_cache_folder", "sp_peaks")
        items: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for group, directory in ((tracks, base_dir), *additional_groups):
            if not directory:
                continue
            peaks_dir = os.path.join(directory, peaks_folder)
            for track in group:
                filepath = track.filepath or os.path.join(directory, track.filename)
                key = peak_source_key(filepath)
                if not key or key in seen:
                    continue
                seen.add(key)
                if not os.path.isfile(filepath):
                    self._state.peak_cache.pop(key, None)
                    continue
                pp = peaks_path_for(peaks_dir, os.path.abspath(filepath))
                existing = load_peaks(pp, expected_mtime=get_source_mtime(filepath))
                if existing is not None:
                    self._state.peak_cache[key] = existing
                else:
                    self._state.peak_cache.pop(key, None)
                    items.append((filepath, track.filename, pp))
        if not seen:
            return

        previous = self.peak_build_worker
        self.cancel_worker("peak_build_worker")
        if not items:
            owner = self._on_progress_owner() if self._on_progress_owner else None
            if previous is not None and previous is owner:
                count = sum(key in self._state.peak_cache for key in seen)
                self._shell.progress_finish(f"Peak cache built for {count} file(s)")
            return

        worker = PeakBuildWorker(items, parent=self)
        self._shell.progress_start(f"Building peak cache: 0/{len(items)}", owner=worker)
        worker.file_done.connect(self._on_peak_file_done, Qt.ConnectionType.QueuedConnection)
        worker.progress.connect(self._on_peak_progress, Qt.ConnectionType.QueuedConnection)
        worker.progress_value.connect(self._on_peak_progress_value, Qt.ConnectionType.QueuedConnection)
        worker.all_done.connect(self._on_peak_build_done, Qt.ConnectionType.QueuedConnection)
        self.peak_build_worker = worker
        worker.start()

    def _is_active_peak_owner(self, worker: Any) -> bool:
        owner = self._on_progress_owner() if self._on_progress_owner else None
        return worker is self.peak_build_worker and worker is owner

    @Slot(object, str)
    def _on_peak_progress(self, worker: Any, message: str) -> None:
        if self._is_active_peak_owner(worker):
            self._shell.progress_message(message)

    @Slot(object, int, int)
    def _on_peak_progress_value(self, worker: Any, current: int, total: int) -> None:
        if self._is_active_peak_owner(worker):
            self._shell.progress_value(current, total)

    @Slot(object, int)
    def _on_peak_build_done(self, worker: Any, count: int) -> None:
        if self._is_active_peak_owner(worker):
            log.info("Peak cache build complete (%d files)", count)
            self._shell.progress_finish(f"Peak cache built for {count} file(s)")

    @Slot(object, str, object)
    def _on_peak_file_done(self, worker: Any, filepath: str, peak_data: Any) -> None:
        if worker is not self.peak_build_worker:
            return
        key = peak_source_key(filepath)
        if not key:
            return
        self._state.peak_cache[key] = peak_data
        self._topology_ctrl.on_peak_file_done(key, peak_data)
        self._detail_ctrl.on_peak_file_done(key, peak_data)

    def prioritize_peak(self, source_key: str) -> None:
        worker = self.peak_build_worker
        if worker is not None and worker.isRunning():
            worker.prioritize(source_key)

    @Slot(object, str)
    def _on_phase2_error(self, worker: Any, message: str) -> None:
        if worker is self.analyze_worker:
            self._on_analyze_error(message)

    def _on_analyze_error(self, message: str) -> None:
        self._analyze_action.setEnabled(True)
        self._track_table.setVisible(True)
        if self.analyze_worker is not None:
            pending, anchors, sources = self.analyze_worker.interrupted_repairs
            self.repair_plan_pending.update(pending)
            self.repair_plan_anchors.update(anchors)
            self.repair_plan_revalidate.update(sources)
            if not self.analyze_worker.isRunning():
                self.analyze_worker.deleteLater()
            self.analyze_worker = None

        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._summary_view.setHtml(
            wrap_report_html(
                f'<div style="color:{COLORS["problems"]}; font-weight:bold;">Analysis Error</div>'
                f'<div style="margin-top:8px;">{esc(message)}</div>'
            )
        )
        self._shell.show_status(f"Error: {message}")
        self._shell.progress_fail(message)
        if self.repair_plan_pending:
            self.queue_repair_plan([])

    # ── Repair plan coalescing ───────────────────────────────────────────

    def set_track_repair_choice(self, track: Any, key: str, value: Any) -> None:
        session = self._state.session
        if session is None or not any(t is track for t in session.tracks) or track.status != "OK":
            return
        if value is None:
            track.repair_overrides.pop(key, None)
        else:
            track.repair_overrides[key] = value
        self.queue_repair_plan([track])

    @staticmethod
    def repair_settings_can_apply(track: Any, config: dict[str, Any]) -> bool:
        return any(
            track.repair_overrides.get(key, config.get(setting, False))
            for key, setting in (
                ("dc_remove", "repair_auto_dc"),
                ("low_cut", "repair_auto_low_cut"),
                ("ultrasonic", "repair_auto_ultrasonic"),
            )
        )

    def queue_repair_plan(
        self,
        tracks: list[Any],
        *,
        refresh_anchors: bool = False,
        verification_only: bool = False,
        revalidate_source: bool = False,
    ) -> None:
        session = self._state.session
        if session is None:
            return
        config = {**session.config, **self._state.flat_config()}
        changed: set[str] = set()
        for track in tracks:
            if track.status != "OK":
                continue
            changed.add(track.filename)
            if revalidate_source and track._cache.get("repair_error"):
                self.repair_plan_revalidate.add(track.filename)
            if track.corrected_analysis is not None:
                track._cache["unvalidated_corrected_analysis"] = track.corrected_analysis
            track.corrected_analysis = None
            track._cache.pop("repair_error", None)
            if self.repair_settings_can_apply(track, config):
                track._cache["repair_verification_pending"] = True
            else:
                track.repair_recipe = RepairRecipe()
                track._cache.pop("repair_verification_pending", None)
                track._cache.pop("unvalidated_corrected_analysis", None)

        self.repair_plan_pending.update(changed)
        if self._repair_preview and self._repair_preview.track and self._repair_preview.track.filename in changed:
            self._repair_preview.planning()
        if refresh_anchors:
            self.repair_plan_anchors.update(changed)
        if not self.repair_plan_pending:
            return
        if changed:
            if not verification_only:
                self.mark_prepare_stale()
            self._detail_ctrl.refresh_verified_summary()
            for name in changed:
                self._table_ctrl.update_track_row(name)
            cur = self._state.current_track
            if cur and cur.filename in changed:
                self._table_ctrl.refresh_file_tab(cur)
                self._detail_ctrl.refresh_track_repair_editor(cur)

        self._save_session_action.setEnabled(False)
        self._prepare_action.setEnabled(False)
        if self.repair_plan_worker is not None or self.prepare_worker is not None or self.analyze_worker is not None:
            return
        queued = [t for t in session.tracks if t.filename in self.repair_plan_pending]
        self.repair_plan_pending.clear()
        worker = RepairPlanWorker(
            session,
            queued,
            config,
            self,
            refresh_anchors=self.repair_plan_anchors,
            revalidate_sources=self.repair_plan_revalidate,
        )
        if self._table_ctrl.batch_filenames:
            worker.progress_value.connect(self._on_worker_progress_value)
        worker.completed.connect(self._repair_plan_finished, Qt.ConnectionType.QueuedConnection)
        self.repair_plan_worker = worker
        worker.start()

    def _commit_planned_track(self, worker: Any, planned: Any, current: dict[str, Any]) -> Any | None:
        t = current.get(planned.filename)
        if planned.filename in self.repair_plan_pending:
            return None
        target, source_audio = worker.targets[planned.filename]
        if t is target and t.audio_data is not source_audio:
            self.repair_plan_pending.add(planned.filename)
            return None
        self.repair_plan_anchors.discard(planned.filename)
        self.repair_plan_revalidate.discard(planned.filename)
        if t is not target:
            return None

        problem = worker.failure or planned._cache.get("repair_error")
        if not problem:
            try:
                if t.filepath != planned.filepath or t.status != "OK":
                    raise ValueError(f"{t.filename} has changed since analysis. Reanalyze this source.")
                check_source_signature(planned, allow_digest=False)
            except (OSError, ValueError) as exc:
                problem = str(exc)

        t._cache.pop("repair_verification_pending", None)
        t._cache.pop("unvalidated_corrected_analysis", None)
        if problem:
            t._cache["repair_error"] = problem
            t.corrected_analysis = None
            t.processor_results = {}
            self.mark_prepare_stale()
            self._shell.show_status(f"{t.filename}: {problem}")
            return t

        if (
            t.repair_recipe != planned.repair_recipe
            or {k: r.gain_db for k, r in t.processor_results.items()}
            != {k: r.gain_db for k, r in planned.processor_results.items()}
        ):
            self.mark_prepare_stale()
        t._cache.pop("repair_error", None)
        if t.detector_sample_digest != planned.detector_sample_digest:
            t._cache.clear()
        for attr in (
            "repair_recipe",
            "repair_measurements",
            "processor_results",
            "detector_results",
            "detector_analysis_key",
            "detector_sample_digest",
            "source_sample_digest",
            "source_file_signature",
            "source_file_revision",
            "corrected_analysis",
        ):
            setattr(t, attr, getattr(planned, attr))
        return t

    @Slot(object)
    def _repair_plan_finished(self, worker: Any) -> None:
        if self.repair_plan_worker is not worker or self._state.session is not worker.session:
            return
        worker.wait()
        self.repair_plan_worker = None
        worker.deleteLater()
        session = self._state.session
        assert session is not None
        current = {t.filename: t for t in session.tracks}
        committed = [
            t for planned in worker.tracks if (t := self._commit_planned_track(worker, planned, current)) is not None
        ]

        session.config.update(self._state.flat_config())
        self._groups_ctrl.apply_linked_group_levels()
        self._detail_ctrl.refresh_verified_summary()
        for t in committed:
            self._table_ctrl.update_track_row(t.filename)
            row = self._table_ctrl.find_table_row(t.filename)
            button = self._track_table.cellWidget(row, 7) if row >= 0 else None
            if button:
                self._table_ctrl.update_processing_button_label(button, t, session.processors)
            if t is self._state.current_track:
                self._table_ctrl.refresh_file_tab(t)
                self._detail_ctrl.refresh_track_repair_editor(t)
                self._detail_ctrl.refresh_repair_preview(t)

        if self.repair_plan_pending:
            self.queue_repair_plan([])
        else:
            self._save_session_action.setEnabled(True)
            self.update_prepare_button()
            if self._table_ctrl.batch_filenames:
                self._table_ctrl.on_batch_done()

    # ── Prepare pipeline ─────────────────────────────────────────────────

    @Slot()
    def prepare(self) -> None:
        session = self._state.session
        if not session or not self._state.source_dir:
            return
        if self.prepare_worker is not None or self.repair_plan_worker is not None:
            return
        log.info("Prepare: %s", self._state.source_dir)
        output_folder = self._state.config.get("app", {}).get("phase2_output_folder", "sp_02_prepared")
        output_dir = os.path.join(self._state.source_dir, output_folder)
        session.config.update(self._state.flat_config())
        processors = list(session.processors) if session.processors else []

        set_action_busy(self._prepare_action, True, "Preparing…")
        self._prepare_action.setEnabled(False)
        self._shell.progress_start("Preparing\u2026")

        worker = PrepareWorker(session, processors, output_dir)
        worker.setParent(self)
        self.prepare_worker = worker
        self._save_session_action.setEnabled(False)
        self._analyze_action.setEnabled(False)
        self._daw_ctrl.update_lifecycle_buttons()
        worker.progress.connect(self._on_prepare_progress)
        worker.progress_value.connect(self._on_prepare_progress_value)
        worker.completed.connect(self._on_prepare_done, Qt.ConnectionType.QueuedConnection)
        worker.start()

    @Slot(object, str)
    def _on_prepare_progress(self, worker: Any, message: str) -> None:
        if worker is self.prepare_worker:
            self._shell.progress_message(message)

    @Slot(object, int, int)
    def _on_prepare_progress_value(self, worker: Any, current: int, total: int) -> None:
        if worker is self.prepare_worker:
            self._shell.progress_value(current, total)

    @Slot(object)
    def _on_prepare_done(self, worker: Any) -> None:
        session = self._state.session
        if self.prepare_worker is not worker or worker.owner is not session or session is None:
            return
        worker.wait()
        if worker.failure:
            session.prepare_state = "stale"
            self._on_prepare_error(worker.failure)
            return

        had_recovery = bool(
            session.config.get("_prepare_errors") or session.config.get("_output_gain_adjustments")
        )
        worker.publish()
        outdated = worker.outdated
        self.prepare_worker = None
        worker.deleteLater()
        set_action_busy(self._prepare_action, False)
        self._analyze_action.setEnabled(not self._table_ctrl.batch_filenames)
        self._save_session_action.setEnabled(not self.repair_plan_pending)
        self._daw_ctrl.update_lifecycle_buttons()
        self.update_prepare_button()
        self._daw_ctrl.update_use_processed_action()

        self._refresh_after_prepare(session, had_recovery)
        self._report_prepare_outcome(session, outdated)
        self._table_ctrl.populate_setup_table()
        if self.repair_plan_pending:
            self.queue_repair_plan([])
        self.prepare_completed.emit(session)

    def _refresh_after_prepare(self, session: SessionContext, had_recovery: bool) -> None:
        if (
            had_recovery
            or session.config.get("_prepare_errors")
            or session.config.get("_output_gain_adjustments")
        ):
            tracks = {t.filename: t for t in session.tracks}
            for row in range(self._track_table.rowCount()):
                button = self._track_table.cellWidget(row, 7)
                track = tracks.get(button.property("track_filename")) if button else None
                if track is not None:
                    self._table_ctrl.update_processing_button_label(button, track, session.processors)
            if self._state.current_track:
                self._table_ctrl.refresh_file_tab(self._state.current_track)
        if session.transfer_manifest:
            self._shell.set_phase_enabled(_PHASE_SETUP, True)
        if self._state.current_track:
            self._detail_ctrl.refresh_track_repair_editor(self._state.current_track)

    def _report_prepare_outcome(self, session: SessionContext, outdated: bool) -> None:
        prepared = len(session.config.get("_prepared_files", []))
        errors = session.config.get("_prepare_errors", [])
        if outdated:
            msg = "Settings changed during Prepare. Run Prepare again for the current settings."
            self._shell.progress_finish(msg)
            self._shell.show_status(msg)
        elif errors:
            msg = f"Prepare incomplete: {prepared} file(s) written, {len(errors)} error(s)"
            self._shell.progress_fail(msg)
            self._shell.show_status(msg)
            detail = "\n".join(f"• {fn}: {err}" for fn, err in errors)
            adjustments = session.config.get("_output_gain_adjustments", {})
            track = next((t for t in session.tracks if t.filename in adjustments), None)
            parent = self._shell.dialog_parent()
            if track is None:
                QMessageBox.warning(parent, "Prepare — errors", detail)
            else:
                box = QMessageBox(
                    QMessageBox.Icon.Warning,
                    "Output gain needs adjustment",
                    detail,
                    QMessageBox.StandardButton.Close,
                    parent,
                )
                adjust = box.addButton("Adjust Gain", QMessageBox.ButtonRole.ActionRole)
                box.exec()
                if box.clickedButton() is adjust:
                    self.adjust_failed_output_gain(track)
        else:
            msg = f"Prepare complete: {prepared} file(s) written"
            log.info("Prepare complete: %d file(s)", prepared)
            self._shell.progress_finish(msg)
            self._shell.show_status(msg)
            if session.transfer_manifest and not self.repair_plan_pending:
                self._shell.switch_to_phase(_PHASE_SETUP)

    def adjust_failed_output_gain(self, track: Any) -> None:
        session = self._state.session
        if session is None:
            return
        adjustment = session.config.get("_output_gain_adjustments", {}).get(track.filename)
        if not adjustment:
            return
        norm = track.processor_results.get("bimodal_normalize")
        normalizing = (
            norm is not None
            and "bimodal_normalize" not in track.processor_skip
            and norm.classification not in ("Skip", "Silent")
        )
        current = norm.gain_db if normalizing else float(track.manual_gain_db or 0)
        previous = adjustment.get("gain_db", current)
        dialog = QDialog(self._shell.dialog_parent())
        dialog.setWindowTitle("Adjust Gain" if normalizing else "Manual output gain")
        form = QFormLayout(dialog)
        form.addRow(QLabel(track.filename))

        gain = FormDoubleSpinBox()
        target = adjustment.get("target_gain_db")
        if target is None:
            safe = Decimal(str(previous)) - Decimal(str(adjustment["reduction_db"]))
            target = float(safe.quantize(Decimal("0.1"), rounding=ROUND_FLOOR))
        gain.setRange(min(-160, target), max(60, target))
        gain.setDecimals(1)
        gain.setSingleStep(0.1)
        gain.setValue(target)
        form.addRow("Output gain (dB)", gain)

        forecast = QLabel()

        def show_peak() -> None:
            peak = adjustment["peak_db"] + gain.value() - previous
            forecast.setText(f"At this gain, the prepared peak is {peak:+.2f} dBFS.")

        show_peak()
        gain.valueChanged.connect(show_peak)
        form.addRow(forecast)
        form.addRow(QLabel("Original and Corrected audition both stay at unity before output gain."))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        apply_btn = buttons.addButton("Apply gain", QDialogButtonBox.ButtonRole.AcceptRole)
        apply_btn.setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            track.manual_gain_db = gain.value()
            self.queue_repair_plan([track])

    @Slot(str)
    def _on_prepare_error(self, message: str) -> None:
        set_action_busy(self._prepare_action, False)
        log.error("Prepare failed: %s", message)
        if self.prepare_worker is not None:
            self.prepare_worker.deleteLater()
            self.prepare_worker = None
        self._prepare_action.setEnabled(True)
        self._analyze_action.setEnabled(not self._table_ctrl.batch_filenames)
        self._save_session_action.setEnabled(not self.repair_plan_pending)
        self._daw_ctrl.update_lifecycle_buttons()
        self.update_prepare_button()
        self._daw_ctrl.update_use_processed_action()
        self._shell.progress_fail(message)
        self._shell.show_status(f"Prepare failed: {message}")
        if self.repair_plan_pending:
            self.queue_repair_plan([])

    def update_prepare_button(self) -> None:
        if self.prepare_worker is not None or self.repair_plan_worker is not None:
            return
        if not self._state.session:
            self._prepare_action.setEnabled(False)
            self._prepare_action.setText("Prepare")
            self._auto_group_action.setEnabled(False)
            return
        state = self._state.session.prepare_state
        self._prepare_action.setEnabled(True)
        self._auto_group_action.setEnabled(True)
        if state == "ready":
            self._prepare_action.setText("Prepare \u2713")
        elif state == "stale":
            self._prepare_action.setText("Prepare (!)")
        else:
            self._prepare_action.setText("Prepare")

    def mark_prepare_stale(self) -> None:
        if self.prepare_worker is not None:
            self.prepare_worker.outdated = True
        if self._state.session and self._state.session.prepare_state == "ready":
            self._state.session.prepare_state = "stale"
            self.update_prepare_button()
            self._daw_ctrl.update_use_processed_action()
