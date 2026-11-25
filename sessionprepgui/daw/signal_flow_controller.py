"""Controller and background QThread worker for Phase 3 Pro Tools Signal Flow preview."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
import logging
import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from shiboken6 import isValid

from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.daw_processors.protools_ptx import GAIN_MODE_CLIP_AND_FADER
from sessionpreplib.daw_processors.protools_ptx_preview import (
    TemplateTopologySummary,
    inspect_protools_template_topology,
    preview_protools_signal_flow,
)
from sessionpreplib.models import SessionContext, TrackContext

from ..change_marker import set_tab_marker
from ..state import WorkspaceState
from .signal_flow_panel import SignalFlowPanel

log = logging.getLogger(__name__)


def _snapshot_track(track: TrackContext) -> TrackContext:
    return replace(
        track,
        processor_results=deepcopy(track.processor_results),
        processor_skip=set(track.processor_skip),
        detector_results={},
    )


def _snapshot_session_for_preview(session: SessionContext, flat_cfg: dict[str, Any]) -> SessionContext:
    merged_cfg = dict(flat_cfg)
    for key in ("_use_processed",):
        if key in session.config:
            merged_cfg[key] = session.config[key]
    tracks = [_snapshot_track(t) for t in session.tracks]
    out_tracks = [_snapshot_track(t) for t in session.output_tracks]
    return replace(
        session,
        tracks=tracks,
        output_tracks=out_tracks,
        config=merged_cfg,
        transfer_manifest=[replace(e) for e in session.transfer_manifest],
        daw_state={},
    )


def _disconnect_worker(worker: SignalFlowWorker) -> None:
    for sig in (worker.progress, worker.completed, worker.failed, worker.cancelled):
        try:
            sig.disconnect()
        except (RuntimeError, TypeError):
            pass


def _reconcile_overrides(
    overrides: dict[str, float],
    graph_dict: dict[str, Any] | None,
) -> dict[str, float]:
    """Keep only active overrides that differ from each staged bus's auto_fader_db."""
    if not overrides:
        return {}
    if not isinstance(graph_dict, dict):
        return dict(overrides)
    nodes = graph_dict.get("nodes") or {}
    nodes_lc = {k.lower(): n for k, n in nodes.items() if isinstance(n, dict)}
    cleaned: dict[str, float] = {}
    for bus_name, val in overrides.items():
        node = nodes_lc.get(bus_name.lower())
        if node is None or not node.get("staged"):
            continue
        auto_db = float(node.get("auto_fader_db", 0.0))
        fval = round(float(val), 1)
        if math.isfinite(fval) and abs(fval - auto_db) >= 0.05:
            cleaned[bus_name] = fval
    return cleaned


class SignalFlowWorker(QThread):
    """Runs offline ``.ptxt`` signal-flow DAG summing off the GUI thread."""

    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(
        self,
        ptxt_path: Path,
        session_snapshot: SessionContext,
        assignments: dict[str, str],
        folders: list[dict[str, Any]],
        options: dict[str, Any],
        bus_overrides: dict[str, float],
        *,
        session_ref: Any,
        processor_id: str,
        silent_refresh: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_ref = session_ref
        self.processor_id = processor_id
        self.silent_refresh = silent_refresh
        self._ptxt_path = ptxt_path
        self._session = session_snapshot
        self._assignments = dict(assignments)
        self._folders = deepcopy(folders)
        self._options = dict(options)
        self._bus_overrides = dict(bus_overrides)

    def run(self) -> None:
        try:
            graph = preview_protools_signal_flow(
                self._ptxt_path,
                self._session,
                self._assignments,
                self._folders,
                auto_route=bool(self._options.get("auto_route", True)),
                gain_mode=str(self._options.get("gain_mode", GAIN_MODE_CLIP_AND_FADER)),
                auto_group_folders=bool(self._options.get("auto_group_folders", True)),
                link_template_vcas=bool(self._options.get("link_template_vcas", True)),
                group_type=str(self._options.get("group_type", "Mix/Edit")),
                stage_bus_faders=bool(self._options.get("stage_bus_faders", True)),
                pan_law_db=float(self._options.get("pan_law_db", -3.0)),
                bus_fader_overrides=self._bus_overrides or None,
                progress_cb=lambda cur, tot, nm: self.progress.emit(
                    cur, tot, f"Summing signal flow — evaluating {tot} nodes ({cur} / {tot}) · {nm}…"
                ),
                cancel_cb=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self.completed.emit(graph.to_dict())
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit("Offline Signal Flow preview failed", str(exc))


class SignalFlowController(QObject):
    """Owns offline template topology inspection, SignalFlowWorker, and bus fader overrides."""

    topology_updated = Signal(object)

    def __init__(
        self,
        state: WorkspaceState,
        panel: SignalFlowPanel,
        right_tabs: Any,
        *,
        get_pt_options: Callable[[], dict[str, Any]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._panel = panel
        self._right_tabs = right_tabs
        self._get_pt_options = get_pt_options

        self._worker: SignalFlowWorker | None = None
        self._topology_cache: dict[tuple[str, float], TemplateTopologySummary] = {}
        self._current_topology: TemplateTopologySummary | None = None
        self._stale_ids: set[str] = set()
        self._cancelled_ids: set[str] = set()
        self._pending_focus_bus: str | None = None

        self._override_timer = QTimer(self)
        self._override_timer.setSingleShot(True)
        self._override_timer.setInterval(220)
        self._override_timer.timeout.connect(self._flush_override_recompute)

        panel.compute_requested.connect(self.compute_preview)
        panel.cancel_requested.connect(self._on_user_cancel_requested)
        panel.reset_all_overrides_requested.connect(self.reset_all_overrides)
        panel.bus_override_changed.connect(self.set_bus_override)
        panel.bus_override_reset.connect(self.reset_bus_override)
        panel.back_to_routing_requested.connect(lambda: self._right_tabs.setCurrentIndex(0))
        right_tabs.currentChanged.connect(self._on_right_tab_changed)

    @property
    def current_topology(self) -> TemplateTopologySummary | None:
        return self._current_topology

    def is_running(self) -> bool:
        return self._worker is not None and isValid(self._worker) and self._worker.isRunning()

    def dispose(self) -> None:
        self._override_timer.stop()
        self.cancel_preview(wait=True, restore_ui=False)
        for worker in self.findChildren(SignalFlowWorker):
            if isValid(worker) and worker.isRunning():
                _disconnect_worker(worker)
                worker.requestInterruption()
                worker.wait()

    def clear_workspace(self) -> None:
        self._override_timer.stop()
        self.cancel_preview(wait=False, restore_ui=False)
        self._stale_ids.clear()
        self._cancelled_ids.clear()
        self._pending_focus_bus = None
        self._current_topology = None
        self._update_tab_chrome(None, {})
        self._panel.show_empty(assigned_count=0, folder_count=0)

    def on_daw_processor_changed(self) -> None:
        self._override_timer.stop()
        self._pending_focus_bus = None
        self.cancel_preview(wait=False, restore_ui=False)

    def active_protools_processor(self) -> ProToolsDawProcessor | None:
        dp = self._state.active_daw_processor
        return dp if isinstance(dp, ProToolsDawProcessor) else None

    def refresh_template_topology(self) -> TemplateTopologySummary | None:
        dp = self.active_protools_processor()
        if dp is None:
            self._current_topology = None
            return None
        ptxt_path = dp.resolve_template_ptxt_path()
        if ptxt_path is None or not ptxt_path.is_file():
            self._current_topology = None
            return None
        try:
            key = (str(ptxt_path), ptxt_path.stat().st_mtime)
            if key not in self._topology_cache:
                self._topology_cache[key] = inspect_protools_template_topology(ptxt_path)
            self._current_topology = self._topology_cache[key]
        except Exception as exc:
            log.warning("Offline .ptxt topology parse failed for %s: %s", ptxt_path, exc)
            self._current_topology = None
        self.topology_updated.emit(self._current_topology)
        return self._current_topology

    def _dp_state(self) -> dict[str, Any]:
        session = self._state.session
        dp = self.active_protools_processor()
        if session is None or dp is None:
            return {}
        return session.daw_state.setdefault(dp.id, {})

    def _assignment_counts(self) -> tuple[int, int]:
        assignments = self._dp_state().get("assignments") or {}
        if not assignments:
            return 0, 0
        return len(assignments), len(set(assignments.values()))

    def mark_stale(self) -> None:
        dp = self.active_protools_processor()
        if dp is None:
            return
        self._override_timer.stop()
        was_running = self.is_running()
        if was_running:
            self.cancel_preview(wait=False, restore_ui=False)
        self._cancelled_ids.discard(dp.id)
        assigned, folders = self._assignment_counts()
        dp_state = self._dp_state()
        if assigned == 0:
            dp_state.pop("signal_flow", None)
            self._stale_ids.discard(dp.id)
            self._panel.show_empty(assigned_count=0, folder_count=0)
            self._update_tab_chrome(None, {})
            return
        self._stale_ids.add(dp.id)
        if dp_state.get("signal_flow"):
            if was_running or self._panel.stack.currentIndex() == 1:
                self.sync_from_session()
            else:
                self._panel.set_stale(True)
        else:
            tmpl = self._current_topology.ptxt_path if self._current_topology else ""
            self._panel.show_empty(assigned_count=assigned, folder_count=folders, template_name=Path(tmpl).name if tmpl else "")

    def sync_from_session(self) -> None:
        dp = self.active_protools_processor()
        if dp is None:
            self._update_tab_chrome(None, {})
            return
        self.refresh_template_topology()
        dp_state = self._dp_state()
        graph_dict = dp_state.get("signal_flow")
        overrides = dp_state.get("bus_fader_overrides") or {}
        opts = self._get_pt_options()
        staging_on = bool(opts.get("stage_bus_faders", True))
        if isinstance(graph_dict, dict):
            self._panel.show_graph(graph_dict, staging_enabled=staging_on)
            self._panel.set_stale(dp.id in self._stale_ids)
        else:
            assigned, folders = self._assignment_counts()
            tmpl = Path(self._current_topology.ptxt_path).name if self._current_topology else ""
            self._panel.show_empty(assigned_count=assigned, folder_count=folders, template_name=tmpl)
        self._update_tab_chrome(
            graph_dict if isinstance(graph_dict, dict) else None,
            overrides if staging_on else {},
            staging_enabled=staging_on,
        )

    def apply_authoritative_graph(self, graph_dict: dict[str, Any] | None) -> None:
        dp = self.active_protools_processor()
        if dp is not None:
            self._stale_ids.discard(dp.id)
        dp_state = self._dp_state()
        overrides = _reconcile_overrides(dp_state.get("bus_fader_overrides") or {}, graph_dict)
        if dp_state:
            dp_state["bus_fader_overrides"] = overrides
        opts = self._get_pt_options()
        staging_on = bool(opts.get("stage_bus_faders", True))
        if isinstance(graph_dict, dict):
            self._panel.show_graph(graph_dict, staging_enabled=staging_on)
            self._panel.set_stale(False)
        self._update_tab_chrome(graph_dict, overrides if staging_on else {}, staging_enabled=staging_on)

    @Slot(int)
    def _on_right_tab_changed(self, index: int) -> None:
        dp = self.active_protools_processor()
        if index != 1 or dp is None:
            return
        dp_state = self._dp_state()
        assigned, _ = self._assignment_counts()
        if assigned > 0 and not dp_state.get("signal_flow") and dp.id not in self._cancelled_ids and not self.is_running():
            self.compute_preview()

    @Slot()
    def _on_user_cancel_requested(self) -> None:
        self.cancel_preview(wait=False, restore_ui=True, user_initiated=True)

    @Slot()
    def _flush_override_recompute(self) -> None:
        self.compute_preview(silent=True)

    @Slot()
    def compute_preview(self, *, silent: bool = False) -> None:
        dp = self.active_protools_processor()
        session = self._state.session
        if dp is None or session is None:
            return
        assigned, _ = self._assignment_counts()
        if assigned == 0:
            self._panel.show_empty(assigned_count=0, folder_count=0)
            return
        ptxt_path = dp.resolve_template_ptxt_path()
        if ptxt_path is None or not ptxt_path.is_file():
            self._panel.show_error(
                "Offline Signal Flow preview unavailable — template file not found",
                f"Could not locate '{ptxt_path or dp.name}'. Routing & Folders remains usable via PTSL cache, "
                "and Create will still inspect the saved .ptx during transfer.",
            )
            return
        self._override_timer.stop()
        self.cancel_preview(wait=False, restore_ui=False)
        self._cancelled_ids.discard(dp.id)
        dp_state = self._dp_state()
        snapshot = _snapshot_session_for_preview(session, self._state.flat_config())
        worker = SignalFlowWorker(
            ptxt_path, snapshot, dp_state.get("assignments") or {}, dp_state.get("folders") or [],
            self._get_pt_options(), dp_state.get("bus_fader_overrides") or {},
            session_ref=session, processor_id=dp.id, silent_refresh=silent, parent=self,
        )
        worker.progress.connect(self._on_worker_progress)
        worker.completed.connect(self._on_worker_completed)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        if not silent:
            self._panel.show_busy(0, 0, "Summing template signal flow…")
        worker.start()

    @Slot()
    def cancel_preview(
        self, *, wait: bool = False, restore_ui: bool = True, user_initiated: bool = False,
    ) -> None:
        self._override_timer.stop()
        worker = self._worker
        self._worker = None
        if worker is not None and isValid(worker):
            _disconnect_worker(worker)
            if worker.isRunning():
                worker.requestInterruption()
                if wait:
                    worker.wait()
        if user_initiated:
            dp = self.active_protools_processor()
            if dp is not None:
                self._cancelled_ids.add(dp.id)
        if restore_ui:
            self.sync_from_session()
            if user_initiated:
                self._panel.restore_focus_after_cancel()

    def _matches_active_worker(self, sender: Any) -> SignalFlowWorker | None:
        if sender is not self._worker or not isinstance(sender, SignalFlowWorker):
            return None
        dp = self.active_protools_processor()
        if dp is None or dp.id != sender.processor_id or self._state.session is not sender.session_ref:
            return None
        return sender

    @Slot(int, int, str)
    def _on_worker_progress(self, current: int, total: int, message: str) -> None:
        worker = self._matches_active_worker(self.sender())
        if worker is None or worker.silent_refresh:
            return
        self._panel.show_busy(current, total, message)

    @Slot(object)
    def _on_worker_completed(self, graph_dict: dict[str, Any]) -> None:
        worker = self._matches_active_worker(self.sender())
        if worker is None:
            return
        self._worker = None
        self._stale_ids.discard(worker.processor_id)
        dp_state = self._dp_state()
        dp_state["signal_flow"] = graph_dict
        pruned = _reconcile_overrides(dp_state.get("bus_fader_overrides") or {}, graph_dict)
        dp_state["bus_fader_overrides"] = pruned
        opts = self._get_pt_options()
        staging_on = bool(opts.get("stage_bus_faders", True))
        focus_bus = self._pending_focus_bus
        self._pending_focus_bus = None
        self._panel.show_graph(graph_dict, staging_enabled=staging_on, focus_bus=focus_bus)
        self._panel.set_stale(False)
        self._update_tab_chrome(graph_dict, pruned if staging_on else {}, staging_enabled=staging_on)

    @Slot(str, str)
    def _on_worker_failed(self, title: str, detail: str) -> None:
        if self._matches_active_worker(self.sender()) is None:
            return
        self._worker = None
        self._pending_focus_bus = None
        self._panel.show_error(title, detail)

    @Slot()
    def _on_worker_cancelled(self) -> None:
        if self._matches_active_worker(self.sender()) is None:
            return
        self._worker = None
        self.sync_from_session()

    @Slot(str, float)
    def set_bus_override(self, bus_name: str, value_db: float) -> None:
        dp_state = self._dp_state()
        overrides = dp_state.setdefault("bus_fader_overrides", {})
        val = round(float(value_db), 1)
        graph = dp_state.get("signal_flow")
        node = ((graph or {}).get("nodes") or {}).get(bus_name) if isinstance(graph, dict) else None
        if isinstance(node, dict) and abs(val - float(node.get("auto_fader_db", 0.0))) < 0.05:
            overrides.pop(bus_name, None)
        else:
            overrides[bus_name] = val
        self._pending_focus_bus = bus_name
        self._update_tab_chrome(graph if isinstance(graph, dict) else None, overrides)
        self._override_timer.start()

    @Slot(str)
    def reset_bus_override(self, bus_name: str) -> None:
        dp_state = self._dp_state()
        overrides = dp_state.get("bus_fader_overrides") or {}
        overrides.pop(bus_name, None)
        self._pending_focus_bus = bus_name
        self._update_tab_chrome(dp_state.get("signal_flow"), overrides)
        self._override_timer.stop()
        self.compute_preview(silent=True)

    @Slot()
    def reset_all_overrides(self) -> None:
        dp_state = self._dp_state()
        if not dp_state.get("bus_fader_overrides"):
            return
        dp_state["bus_fader_overrides"] = {}
        self._pending_focus_bus = None
        self._update_tab_chrome(dp_state.get("signal_flow"), {})
        self._override_timer.stop()
        self.compute_preview(silent=True)

    def _update_tab_chrome(
        self,
        graph_dict: dict[str, Any] | None,
        overrides: dict[str, float],
        *,
        staging_enabled: bool = True,
    ) -> None:
        if self._right_tabs.count() < 2:
            return
        if not staging_enabled:
            self._right_tabs.setTabText(1, "Signal Flow")
            set_tab_marker(self._right_tabs, 1, False, "")
            return
        rec = (graph_dict or {}).get("recommended_bus_faders") or {}
        trims = (graph_dict or {}).get("recommended_slot_a_trims") or {}
        count = len(set(rec) | set(trims))
        if count == 1:
            self._right_tabs.setTabText(1, "Signal Flow (1 bus staged)")
        elif count > 1:
            self._right_tabs.setTabText(1, f"Signal Flow ({count} busses staged)")
        else:
            self._right_tabs.setTabText(1, "Signal Flow")
        set_tab_marker(self._right_tabs, 1, bool(overrides), "Manual bus fader overrides active")
