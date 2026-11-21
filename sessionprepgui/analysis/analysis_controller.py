"""Phase 1/2 analysis, repair-plan coalescing, and Prepare worker controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR
import logging
import os
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
)
from shiboken6 import isValid

from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import LifecyclePhase, SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.repair_plan import check_source_signature
from sessionpreplib.repairs import RepairRecipe
from sessionpreplib.topology import build_default_topology

from ..controls import FormDoubleSpinBox, set_action_busy
from ..detail.report_style import wrap_report_html
from ..helpers import esc, track_analysis_label
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import COLORS
from ..tracks.table_widgets import (
    TrackCol,
    _PAGE_TABS,
    _PHASE_SETUP,
    _TAB_FILE,
    _TAB_GROUPS,
    _TAB_SESSION,
    _TAB_SUMMARY,
    _SortableItem,
    _make_analysis_cell,
)
from .worker import AnalyzeWorker, Phase1AnalyzeWorker, PrepareWorker, RepairPlanWorker

if TYPE_CHECKING:
    from ..detail.controller import TrackDetailController
    from ..detail.repair_preview import RepairPreview
    from ..topology.controller import TopologyController
    from ..tracks.groups_controller import GroupsController
    from ..tracks.phase2_page import Phase2Page
    from ..tracks.table_controller import TrackTableController
    from .config_controller import SessionConfigController

log = logging.getLogger(__name__)


def repair_settings_can_apply(track: TrackContext, config: dict[str, Any]) -> bool:
    """Return True if any corrective repair toggle can apply to *track* under *config*."""
    return any(
        track.repair_overrides.get(key, config.get(setting, False))
        for key, setting in (
            ("dc_remove", "repair_auto_dc"),
            ("low_cut", "repair_auto_low_cut"),
            ("ultrasonic", "repair_auto_ultrasonic"),
        )
    )


class AnalysisPrepareController(QObject):
    """Owns Phase 1 analysis, Phase 2 analysis, repair-plan coalescing, and Prepare workers."""

    phase1_completed = Signal(object)               # SessionContext
    analysis_completed = Signal(object, object)     # (SessionContext, diagnostic_summary)
    prepare_completed = Signal(object)              # SessionContext
    prepare_state_changed = Signal()
    save_enabled_changed = Signal(bool)
    peak_build_requested = Signal(object, object, object)  # (tracks, base_dir, additional_groups)

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        phase2_page: Phase2Page,
        topology_ctrl: TopologyController,
        detail_ctrl: TrackDetailController,
        table_ctrl: TrackTableController,
        groups_ctrl: GroupsController,
        config_ctrl: SessionConfigController,
        repair_preview: RepairPreview | None = None,
        get_project_name: Callable[[], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._phase2 = phase2_page
        self._topology_ctrl = topology_ctrl
        self._detail_ctrl = detail_ctrl
        self._table_ctrl = table_ctrl
        self._groups_ctrl = groups_ctrl
        self._config_ctrl = config_ctrl
        self._repair_preview = repair_preview
        self._get_project_name = get_project_name

        self.p1_worker: Phase1AnalyzeWorker | None = None
        self.analyze_worker: AnalyzeWorker | None = None
        self.prepare_worker: PrepareWorker | None = None
        self.repair_plan_worker: RepairPlanWorker | None = None
        self.repair_plan_pending: set[str] = set()
        self.repair_plan_anchors: set[str] = set()
        self.repair_plan_revalidate: set[str] = set()
        self._zombie_workers: set[Any] = set()

    # ── Status queries ───────────────────────────────────────────────────

    def is_analysis_running(self) -> bool:
        w = self.analyze_worker
        return bool(w is not None and isValid(w) and w.isRunning())

    def is_repair_running(self) -> bool:
        return self.repair_plan_worker is not None

    def is_repair_pending_or_running(self) -> bool:
        return bool(self.repair_plan_worker or self.repair_plan_pending)

    def is_prepare_running(self) -> bool:
        return self.prepare_worker is not None

    def is_save_blocked(self) -> bool:
        return self.prepare_worker is not None or self.repair_plan_worker is not None

    # ── Worker lifecycle & shutdown ──────────────────────────────────────

    def cancel_worker(self, attr_name: str) -> None:
        """Safely detach and cancel an owned background worker."""
        worker = getattr(self, attr_name, None)
        if worker is None:
            return
        if isinstance(worker, QObject) and not isValid(worker):
            setattr(self, attr_name, None)
            return
        if isinstance(worker, (Phase1AnalyzeWorker, AnalyzeWorker)):
            setattr(self, attr_name, None)
            try:
                worker.disconnect(self)
            except RuntimeError:
                pass
            worker.cancel()
            if not worker.isRunning():
                worker.deleteLater()
            return
        if isinstance(worker, (RepairPlanWorker, PrepareWorker)):
            setattr(self, attr_name, None)
            try:
                worker.disconnect(self)
            except RuntimeError:
                pass
            if isinstance(worker, RepairPlanWorker):
                worker.cancel()
            else:
                worker.requestInterruption()
            worker.finished.connect(worker.deleteLater)
            if worker.isFinished():
                worker.deleteLater()
            return
        self._detach_generic_worker(attr_name, worker)

    def _detach_generic_worker(self, attr_name: str, worker: Any) -> None:
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

    @Slot()
    def cancel_all_workers(self) -> None:
        """Cancel all Phase 1/2 analysis, repair, and Prepare workers."""
        for attr in ("p1_worker", "analyze_worker", "prepare_worker", "repair_plan_worker"):
            self.cancel_worker(attr)
        self.repair_plan_pending = set()
        self.repair_plan_anchors = set()
        self.repair_plan_revalidate = set()
        set_action_busy(self._phase2.prepare_action, False)

    def dispose(self) -> None:
        """Stop all analysis/repair/prepare workers cleanly before window close."""
        for kind in (Phase1AnalyzeWorker, AnalyzeWorker, RepairPlanWorker):
            for child in self.findChildren(kind):
                if not isValid(child):
                    continue
                child_any: Any = child
                child_any.cancel()
                child_any.wait()
        self.repair_plan_worker = None
        for prep in self.findChildren(PrepareWorker):
            if not isValid(prep):
                continue
            prep.requestInterruption()
            prep.wait()
        self.prepare_worker = None

        for attr in ("p1_worker", "analyze_worker"):
            w: Any = getattr(self, attr, None)
            if isinstance(w, QThread) and isValid(w) and w.isRunning():
                cancel_fn = getattr(w, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    w.requestInterruption()
                w.wait()
            setattr(self, attr, None)

        for zombie in list(self._zombie_workers):
            if isinstance(zombie, QThread) and isValid(zombie) and zombie.isRunning():
                cancel_fn = getattr(zombie, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    zombie.requestInterruption()
                zombie.wait()
        self._zombie_workers.clear()

    # ── Phase 1 Analysis ─────────────────────────────────────────────────

    @Slot()
    def run_phase1_analysis(self) -> None:
        self._phase2.analyze_action.setEnabled(False)
        self._topology_ctrl.page.apply_action.setEnabled(False)
        self._shell.progress_start("Analyzing Format & Layout\u2026")
        assert self._state.session is not None
        self.start_phase1_worker(self._state.session)

    def start_phase1_worker(self, session: SessionContext) -> None:
        config = self._state.flat_config()
        config["_source_dir"] = self._state.source_dir
        worker = Phase1AnalyzeWorker(session, config, parent=self)
        worker.finished.connect(worker.deleteLater)
        self.p1_worker = worker
        if not self._topology_ctrl.pending_sources:
            worker.progress.connect(self._on_phase1_progress, Qt.ConnectionType.QueuedConnection)
            worker.progress_value.connect(self._on_phase1_progress_value, Qt.ConnectionType.QueuedConnection)
        worker.completed.connect(self._on_phase1_done)
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
        if isValid(worker) and not worker.isRunning():
            worker.deleteLater()
        if self._topology_ctrl.pending_sources:
            self._on_reopened_sources_checked(session)
            return

        self._state.session = session
        self._state.topo_topology = build_default_topology(session.tracks)
        self._topology_ctrl.populate()

        self._phase2.analyze_action.setEnabled(True)
        self._topology_ctrl.page.reanalyze_action.setEnabled(True)
        self._topology_ctrl.page.reset_action.setEnabled(True)
        self._shell.show_status(
            f"Discovered {len(session.tracks)} file(s) from {self._state.source_dir} \u2014 "
            "review layout, then click Apply"
        )
        self._shell.progress_finish(f"Discovered {len(session.tracks)} file(s)")
        self._phase2.right_stack.setCurrentIndex(_PAGE_TABS)
        self.save_enabled_changed.emit(True)
        self._shell.set_window_title("SessionPrep")
        self.peak_build_requested.emit(session.tracks, self._state.source_dir, ())
        self.phase1_completed.emit(session)

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
        if isValid(worker) and not worker.isRunning():
            worker.deleteLater()
        if self._topology_ctrl.pending_sources:
            self._shell.show_status(f"Could not check Track Layout sources: {message}")
        else:
            self._on_analyze_error(message)

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
        self._phase2.track_table.setSortingEnabled(True)

        track_settings = self._snapshot_track_settings()
        self._phase2.analyze_action.setEnabled(False)
        self._state.current_track = None
        self._phase2.file_report.clear()
        self._phase2.detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._phase2.track_table.setVisible(False)
        if not self._phase2.detail_tabs.isTabEnabled(_TAB_SESSION):
            self._config_ctrl.init_session_config()

        self._shell.progress_start("Analyzing\u2026")
        config = self._state.flat_config()
        config["_source_dir"] = self._state.source_dir

        worker: Any = AnalyzeWorker(
            analyze_dir,
            config,
            recursive=self._state.recursive_scan,
            track_settings=track_settings,
            parent=self,
        )
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
        if worker is not self.analyze_worker or not self._phase2.track_table.isVisible():
            return
        row = self._table_ctrl.find_table_row(filename)
        if row < 0:
            return
        ch_item = _SortableItem(str(track.channels), track.channels)
        ch_item.setForeground(QColor(COLORS["metadata"]))
        self._phase2.track_table.setItem(row, TrackCol.CHANNELS, ch_item)
        _plain, html, _color, sort_key = track_analysis_label(track)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._phase2.track_table.setItem(row, TrackCol.ANALYSIS, item)
        self._phase2.track_table.setCellWidget(row, TrackCol.ANALYSIS, lbl)

    @Slot(object, str, object)
    def _on_track_planned(self, worker: Any, filename: str, track: Any) -> None:
        if worker is not self.analyze_worker or not self._phase2.track_table.isVisible():
            return
        row = self._table_ctrl.find_table_row(filename)
        if row < 0:
            return
        dets = self._state.session.detectors if self._state.session else None
        for col in (TrackCol.CLASSIFICATION, TrackCol.GAIN, TrackCol.ANCHOR):
            self._phase2.track_table.removeCellWidget(row, col)
        self._table_ctrl.populate_track_row(row, track, dets)
        self._table_ctrl.auto_fit_track_table()

    @Slot(object, object, object)
    def _on_analyze_done(self, worker: Any, session: SessionContext, summary: dict[str, Any]) -> None:
        if self.analyze_worker is None or worker is not self.analyze_worker:
            return
        prev = self._state.session
        if prev and prev.topology and not self._state.topo_topology:
            self._state.topo_topology = prev.topology
        if self._get_project_name is not None:
            session.project_name = self._get_project_name()
        if prev and prev.output_tracks:
            session.output_tracks = prev.output_tracks
            session.prepare_state = "stale"
            session.transfer_manifest = prev.transfer_manifest
            session.base_transfer_manifest = prev.base_transfer_manifest
            session.daw_state = prev.daw_state

        self._state.session = session
        self._state.summary = summary
        self._phase2.analyze_action.setEnabled(True)
        self._phase2.track_table.setVisible(True)
        if isValid(worker) and not worker.isRunning():
            worker.deleteLater()
        self.analyze_worker = None

        if not self._state.session_groups:
            self._state.active_session_preset = "Default"
            self._groups_ctrl.merge_groups_from_preset()
            self._groups_ctrl.populate_group_preset_combo()
        else:
            self._groups_ctrl.populate_groups_tab()
            self._groups_ctrl.refresh_group_combos()

        self._table_ctrl.rebuild_track_table_rows(session.tracks)
        self._table_ctrl.populate_table(session)
        self._detail_ctrl.render_summary()
        self._post_analyze_finalize(session, summary)

    def _post_analyze_finalize(self, session: SessionContext, summary: dict[str, Any]) -> None:
        self._phase2.right_stack.setCurrentIndex(_PAGE_TABS)
        self._phase2.detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._phase2.detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._phase2.detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._config_ctrl.refresh_phase2_dirty_indicators()
        self._topology_ctrl.populate()

        has_manifest = bool(session.transfer_manifest)
        self._shell.set_phase_enabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._table_ctrl.populate_setup_table()
        if session.prepare_state == "ready":
            session.prepare_state = "stale"
        self.update_prepare_button()
        self.save_enabled_changed.emit(True)

        ok_count = sum(1 for t in session.tracks if t.status == "OK")
        msg = f"Analysis complete: {ok_count}/{len(session.tracks)} tracks OK"
        log.info("Analyze complete: %d/%d tracks OK", ok_count, len(session.tracks))
        self._shell.show_status(msg)
        self._shell.progress_finish(msg)

        if self._state.topology_dir:
            self.peak_build_requested.emit(
                session.tracks,
                self._state.topology_dir,
                ((self._state.topo_source_tracks, self._state.source_dir),),
            )
        elif self._state.source_dir:
            self.peak_build_requested.emit(session.tracks, self._state.source_dir, ())
        if self.repair_plan_pending:
            self.queue_repair_plan(
                [t for t in session.tracks if t.filename in self.repair_plan_pending]
            )
        self.analysis_completed.emit(session, summary)

    @Slot(object, str)
    def _on_phase2_error(self, worker: Any, message: str) -> None:
        if worker is self.analyze_worker:
            self._on_analyze_error(message)

    def _on_analyze_error(self, message: str) -> None:
        self._phase2.analyze_action.setEnabled(True)
        self._phase2.track_table.setVisible(True)
        worker = self.analyze_worker
        if worker is not None:
            interrupted = getattr(worker, "interrupted_repairs", None)
            if interrupted:
                pending, anchors, sources = interrupted
                self.repair_plan_pending.update(pending)
                self.repair_plan_anchors.update(anchors)
                self.repair_plan_revalidate.update(sources)
            if isValid(worker) and not worker.isRunning():
                worker.deleteLater()
            self.analyze_worker = None

        self._phase2.right_stack.setCurrentIndex(_PAGE_TABS)
        self._phase2.detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._phase2.summary_view.setHtml(
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
        changed = self._prepare_repair_queue_entries(tracks, config, revalidate_source=revalidate_source)

        self.repair_plan_pending.update(changed)
        if self._repair_preview and self._repair_preview.track and self._repair_preview.track.filename in changed:
            self._repair_preview.planning()
        if refresh_anchors:
            self.repair_plan_anchors.update(changed)
        if not self.repair_plan_pending:
            return
        if changed:
            self._notify_repair_queue_ui(changed, verification_only=verification_only)

        self.save_enabled_changed.emit(False)
        self._phase2.prepare_action.setEnabled(False)
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

    def _prepare_repair_queue_entries(
        self, tracks: list[Any], config: dict[str, Any], *, revalidate_source: bool
    ) -> set[str]:
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
            if repair_settings_can_apply(track, config):
                track._cache["repair_verification_pending"] = True
            else:
                track.repair_recipe = RepairRecipe()
                track._cache.pop("repair_verification_pending", None)
                track._cache.pop("unvalidated_corrected_analysis", None)
        return changed

    def _notify_repair_queue_ui(self, changed: set[str], *, verification_only: bool) -> None:
        if not verification_only:
            self.mark_prepare_stale()
        self._detail_ctrl.refresh_verified_summary()
        for name in changed:
            self._table_ctrl.update_track_row(name)
        cur = self._state.current_track
        if cur and cur.filename in changed:
            self._table_ctrl.refresh_file_tab(cur)
            self._detail_ctrl.refresh_track_repair_editor(cur)

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
        if self.repair_plan_worker is None or self.repair_plan_worker is not worker or self._state.session is not worker.session:
            return
        w: Any = worker
        w.wait()
        self.repair_plan_worker = None
        w.deleteLater()
        session = self._state.session
        assert session is not None
        current = {t.filename: t for t in session.tracks}
        committed = [
            t for planned in w.tracks if (t := self._commit_planned_track(w, planned, current)) is not None
        ]

        session.config.update(self._state.flat_config())
        self._groups_ctrl.apply_linked_group_levels()
        self._detail_ctrl.refresh_verified_summary()
        for t in committed:
            self._table_ctrl.update_track_row(t.filename)
            row = self._table_ctrl.find_table_row(t.filename)
            button = self._phase2.track_table.cellWidget(row, TrackCol.PROCESSING) if row >= 0 else None
            if button:
                self._table_ctrl.update_processing_button_label(button, t, session.processors)
            if t is self._state.current_track:
                self._table_ctrl.refresh_file_tab(t)
                self._detail_ctrl.refresh_track_repair_editor(t)
                self._detail_ctrl.refresh_repair_preview(t)

        if self.repair_plan_pending:
            self.queue_repair_plan([])
        else:
            self.save_enabled_changed.emit(True)
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

        set_action_busy(self._phase2.prepare_action, True, "Preparing…")
        self._phase2.prepare_action.setEnabled(False)
        self._shell.progress_start("Preparing\u2026")

        worker = PrepareWorker(session, processors, output_dir, parent=self)
        self.prepare_worker = worker
        self.save_enabled_changed.emit(False)
        self._phase2.analyze_action.setEnabled(False)
        self.prepare_state_changed.emit()
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
        if self.prepare_worker is None or self.prepare_worker is not worker or worker.owner is not session or session is None:
            return
        w: Any = worker
        w.wait()
        if w.failure:
            session.prepare_state = "stale"
            self._on_prepare_error(w.failure)
            return

        had_recovery = bool(
            session.config.get("_prepare_errors") or session.config.get("_output_gain_adjustments")
        )
        w.publish()
        outdated = w.outdated
        self.prepare_worker = None
        w.deleteLater()
        set_action_busy(self._phase2.prepare_action, False)
        self._phase2.analyze_action.setEnabled(not self._table_ctrl.batch_filenames)
        self.save_enabled_changed.emit(not self.repair_plan_pending)
        self.update_prepare_button()
        self.prepare_state_changed.emit()

        self._refresh_after_prepare(session, had_recovery)
        self._report_prepare_outcome(session, outdated)
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
            for row in range(self._phase2.track_table.rowCount()):
                button = self._phase2.track_table.cellWidget(row, TrackCol.PROCESSING)
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
        set_action_busy(self._phase2.prepare_action, False)
        log.error("Prepare failed: %s", message)
        if self.prepare_worker is not None:
            if isValid(self.prepare_worker):
                self.prepare_worker.deleteLater()
            self.prepare_worker = None
        self._phase2.prepare_action.setEnabled(True)
        self._phase2.analyze_action.setEnabled(not self._table_ctrl.batch_filenames)
        self.save_enabled_changed.emit(not self.repair_plan_pending)
        self.update_prepare_button()
        self.prepare_state_changed.emit()
        self._shell.progress_fail(message)
        self._shell.show_status(f"Prepare failed: {message}")
        if self.repair_plan_pending:
            self.queue_repair_plan([])

    @Slot()
    def update_prepare_button(self) -> None:
        if self.prepare_worker is not None or self.repair_plan_worker is not None:
            return
        if not self._state.session:
            self._phase2.prepare_action.setEnabled(False)
            self._phase2.prepare_action.setText("Prepare")
            self._phase2.auto_group_action.setEnabled(False)
            return
        state = self._state.session.prepare_state
        self._phase2.prepare_action.setEnabled(True)
        self._phase2.auto_group_action.setEnabled(True)
        if state == "ready":
            self._phase2.prepare_action.setText("Prepare \u2713")
        elif state == "stale":
            self._phase2.prepare_action.setText("Prepare (!)")
        else:
            self._phase2.prepare_action.setText("Prepare")

    @Slot()
    def mark_prepare_stale(self) -> None:
        if self.prepare_worker is not None:
            self.prepare_worker.outdated = True
        if self._state.session and self._state.session.prepare_state == "ready":
            self._state.session.prepare_state = "stale"
            self.update_prepare_button()
            self.prepare_state_changed.emit()
