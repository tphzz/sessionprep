"""Session persistence controller: open/save/load .spsession, workspace clear, peak cache."""

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
import os
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMessageBox
from shiboken6 import isValid

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
)
from ..waveform.compute import PeakBuildWorker
from ..waveform.peakcache import get_source_mtime, load_peaks, peak_source_key, peaks_path_for
from .analysis_controller import repair_settings_can_apply

if TYPE_CHECKING:
    from ..daw.controller import DawTransferController
    from ..detail.controller import TrackDetailController
    from ..topology.controller import TopologyController
    from ..tracks.groups_controller import GroupsController
    from ..tracks.phase2_page import Phase2Page
    from ..tracks.table_controller import TrackTableController
    from .config_controller import SessionConfigController

log = logging.getLogger(__name__)


class SessionPersistenceController(QObject):
    """Owns folder opening, .spsession save/load/restore, workspace reset, and peak cache builds."""

    workspace_clearing = Signal()
    workspace_cleared = Signal(bool)                # keep_session_config
    workspace_opened = Signal(str)                  # source_dir
    phase1_analysis_requested = Signal()
    peak_file_ready = Signal(str, object)           # (source_key, peak_data)
    repair_plan_requested = Signal(object, bool)    # (tracks, verification_only)
    prepare_button_refresh_requested = Signal()
    tools_menu_refresh_requested = Signal()

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        phase2_page: Phase2Page,
        topology_ctrl: TopologyController,
        daw_ctrl: DawTransferController,
        detail_ctrl: TrackDetailController,
        table_ctrl: TrackTableController,
        groups_ctrl: GroupsController,
        config_ctrl: SessionConfigController,
        save_session_action: QAction,
        on_progress_owner: Callable[[], Any] | None = None,
        on_is_save_blocked: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._phase2 = phase2_page
        self._topology_ctrl = topology_ctrl
        self._daw_ctrl = daw_ctrl
        self._detail_ctrl = detail_ctrl
        self._table_ctrl = table_ctrl
        self._groups_ctrl = groups_ctrl
        self._config_ctrl = config_ctrl
        self._save_session_action = save_session_action
        self._on_progress_owner = on_progress_owner
        self._on_is_save_blocked = on_is_save_blocked

        self.peak_build_worker: PeakBuildWorker | None = None

    # ── Worker lifecycle & shutdown ──────────────────────────────────────

    def cancel_peak_worker(self) -> None:
        worker = self.peak_build_worker
        self.peak_build_worker = None
        if worker is None or not isValid(worker):
            return
        try:
            worker.disconnect(self)
        except RuntimeError:
            pass
        worker.cancel()
        if not worker.isRunning():
            worker.deleteLater()

    def dispose(self) -> None:
        """Stop all peak-cache worker threads cleanly before window close."""
        for child in self.findChildren(PeakBuildWorker):
            if isValid(child) and child.isRunning():
                child.cancel()
                child.wait()
        w = self.peak_build_worker
        if isinstance(w, QThread) and isValid(w) and w.isRunning():
            w.cancel()
            w.wait()
        self.peak_build_worker = None

    # ── Clear workspace ──────────────────────────────────────────────────

    @Slot()
    def clear_workspace(self, *, keep_session_config: bool = False) -> None:
        """Clear all UI state and cancel in-flight workers."""
        self.workspace_clearing.emit()
        self.cancel_peak_worker()
        self._table_ctrl.batch_filenames = set()
        self._phase2.track_table.setSortingEnabled(True)

        self._shell.clear_progress()
        self._state.reset_workspace(keep_session_config=keep_session_config)
        self.workspace_cleared.emit(keep_session_config)

        self._phase2.analyze_action.setEnabled(False)
        self._save_session_action.setEnabled(False)
        self._shell.switch_to_phase(_PHASE_TOPOLOGY)
        self._shell.set_phase_enabled(_PHASE_ANALYSIS, False)
        self._shell.set_phase_enabled(_PHASE_SETUP, False)
        self._phase2.track_table.setRowCount(0)
        self._phase2.track_table.set_source_dir(None)
        self._phase2.detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._phase2.detail_tabs.setTabEnabled(_TAB_GROUPS, False)
        self._phase2.detail_tabs.setTabEnabled(_TAB_SESSION, False)
        self._phase2.detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._phase2.right_stack.setCurrentIndex(_PAGE_TABS)
        self._shell.set_window_title("SessionPrep")

    # ── Open folder & discovery ──────────────────────────────────────────

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
        self._phase2.track_table.set_source_dir(path)
        self._topology_ctrl.page.input_tree.set_source_dir(path)
        self._config_ctrl.populate_config_preset_combo()
        self._config_ctrl.load_session_widgets(self._state.effective_session_config())
        self._daw_ctrl.refresh_from_config()
        self.tools_menu_refresh_requested.emit()

        tracks = self.discover_source_tracks(path, recursive=self._state.recursive_scan)
        if not tracks:
            self._shell.show_status(f"No readable audio files in {path}")
            self._phase2.analyze_action.setEnabled(False)
            return

        self._state.topo_source_tracks = list(tracks)
        self._state.topo_topology = None
        self._state.session = SessionContext(tracks=tracks, config={})
        self.workspace_opened.emit(path)
        self.phase1_analysis_requested.emit()

    def discover_source_tracks(self, directory: str, *, recursive: bool) -> list[TrackContext]:
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

    # ── Save & load session ──────────────────────────────────────────────

    @Slot()
    def save_session_dialog(self) -> None:
        if (
            not self._state.session
            or not self._state.source_dir
            or (self._on_is_save_blocked is not None and self._on_is_save_blocked())
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
        self._finish_restore_ui(
            data, session, source_dir, source_tracks, pipeline, analysis_changed, repair_defaults_changed, flat
        )
        return True

    def _restore_workspace_inputs(self, data: dict[str, Any], source_dir: str) -> list[TrackContext]:
        self.clear_workspace()
        self._state.source_dir = source_dir
        self._phase2.track_table.set_source_dir(source_dir)
        self._topology_ctrl.page.input_tree.set_source_dir(source_dir)
        was_recursive = data.get("recursive_scan", False)
        self._state.recursive_scan = was_recursive
        self._topology_ctrl.page.recursive_cb.setChecked(was_recursive)
        source_tracks = self.discover_source_tracks(source_dir, recursive=was_recursive)
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
        self._table_ctrl.rebuild_track_table_rows(session.tracks)
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

        self._restore_phase_tabs_and_daw(data, session)
        self._table_ctrl.auto_fit_track_table()

        ok_count = sum(1 for t in session.tracks if t.status == "OK")
        hint = "" if session.transfer_manifest else " — run Prepare to enable Session Setup"
        self._shell.show_status(f"Session loaded: {ok_count}/{len(session.tracks)} tracks OK{hint}")
        self._shell.set_window_title("SessionPrep")
        self.start_peak_build(source_tracks, source_dir, (session.tracks, self._state.topology_dir))

        if repair_defaults_changed:
            self.repair_plan_requested.emit(session.tracks, False)
        else:
            unverified = [
                t for t in session.tracks
                if (t.repair_recipe.active and not pipeline.repair_verification.has_current_evidence(t))
                or (analysis_changed and repair_settings_can_apply(t, flat))
            ]
            if unverified:
                self.repair_plan_requested.emit(unverified, True)

    def _restore_phase_tabs_and_daw(self, data: dict[str, Any], session: SessionContext) -> None:
        self._phase2.right_stack.setCurrentIndex(_PAGE_TABS)
        self._phase2.detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._phase2.detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._phase2.detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._topology_ctrl.populate()
        self._shell.set_phase_enabled(_PHASE_ANALYSIS, True)
        self._shell.switch_to_phase(_PHASE_ANALYSIS)

        has_manifest = bool(session.transfer_manifest)
        self._shell.set_phase_enabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._table_ctrl.populate_setup_table()
        self._phase2.analyze_action.setEnabled(True)
        self._save_session_action.setEnabled(True)
        self.prepare_button_refresh_requested.emit()

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

    # ── Peak cache ───────────────────────────────────────────────────────

    @Slot(object, object, object)
    def on_peak_build_requested(self, tracks: Any, base_dir: Any, additional_groups: Any) -> None:
        self.start_peak_build(tracks, base_dir, *additional_groups)

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
        self.cancel_peak_worker()
        if not items:
            owner = self._on_progress_owner() if self._on_progress_owner else None
            if previous is not None and previous is owner:
                count = sum(key in self._state.peak_cache for key in seen)
                self._shell.progress_finish(f"Peak cache built for {count} file(s)")
            return

        worker = PeakBuildWorker(items, parent=self)
        worker.finished.connect(worker.deleteLater)
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
        active_owner = self._is_active_peak_owner(worker)
        if worker is self.peak_build_worker:
            self.peak_build_worker = None
        if active_owner:
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
        self.peak_file_ready.emit(key, peak_data)

    @Slot(str)
    def prioritize_peak(self, source_key: str) -> None:
        worker = self.peak_build_worker
        if worker is not None and isValid(worker) and worker.isRunning():
            worker.prioritize(source_key)
