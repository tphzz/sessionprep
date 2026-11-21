from __future__ import annotations

import logging
import os
from typing import Any
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot
from sessionpreplib.models import SessionContext
from sessionpreplib.daw_processor import DawProcessor
from sessionpreplib.daw_processors import create_runtime_daw_processors
from sessionprepgui.analysis.worker import DawCheckWorker, DawTransferWorker
from sessionprepgui.session.io import restore_layout_sources

from .panel import BatchItem

log = logging.getLogger(__name__)

class BatchManager(QObject):
    """Orchestrates DAW check and sequential transfer workers for batch processing."""

    # Emitted when all jobs are complete (or single job finishes)
    finished = Signal()
    # Emitted when a single item finishes (item_id, status, result_text)
    item_finished = Signal(str, str, str)
    # Emitted when a batch (or single item) starts
    started = Signal()
    # Emitted to update overall progress (current, total)
    batch_progress_value = Signal(int, int)
    # Emitted to update status bar message
    batch_progress_message = Signal(str)

    def __init__(self, main_window: Any):
        super().__init__(main_window if isinstance(main_window, QObject) else None)
        self._main_window = main_window
        self._queue: list[BatchItem] = []
        self._current_index: int = 0
        self._running: bool = False

        self._worker: DawCheckWorker | DawTransferWorker | None = None
        self._worker_result: tuple[bool, str] | None = None
        self._current_dp: DawProcessor | None = None
        self._current_session: SessionContext | None = None
        self._is_single_job: bool = False
        self._stop_requested = False

    @property
    def is_running(self) -> bool:
        return self._running

    def stop_after_current(self) -> None:
        """Finish the active transfer before yielding the queue to shutdown."""
        if self._running:
            self._stop_requested = True

    def start_batch(self, items: list[BatchItem]):
        if self._running or not items:
            return
        log.info("Batch start: %d items", len(items))

        self._queue = items
        self._current_index = 0
        self._running = True
        self._is_single_job = False

        self.started.emit()
        self._process_next()

    def start_single(self, item: BatchItem):
        if self._running:
            return

        self._queue = [item]
        self._current_index = 0
        self._running = True
        self._is_single_job = True

        self.started.emit()
        self._process_next()

    def _process_next(self):
        while not self._stop_requested and self._current_index < len(self._queue):
            self._on_transfer_progress_value(0, 1)

            item = self._queue[self._current_index]
            item.status = "Running"
            item.result_text = "Checking DAW..."
            self.item_finished.emit(item.id, item.status, item.result_text)
            self.batch_progress_message.emit(f"[{item.project_name}] Checking DAW...")

            try:
                source_dir = item.session_state.get("source_dir")
                if source_dir and not os.path.isdir(source_dir):
                    raise ValueError(f"Source directory is unavailable: {source_dir}")
                if item.output_path and os.path.lexists(item.output_path):
                    raise ValueError(f"Project already exists: {item.output_path}. Choose a new destination before retrying.")
                self._current_session = self._rehydrate_session(item.session_state)
                self._current_session.config["_batch_transfer"] = True
                self._current_dp = self._get_daw_processor(
                    item.daw_processor_id, self._current_session.config)
            except Exception as exc:
                self._complete_item("Failed", f"Failed to prepare job: {exc}")
                continue

            if not self._current_dp:
                self._complete_item("Failed", f"DAW Processor '{item.daw_processor_id}' not found.")
                continue

            worker = DawCheckWorker(self._current_dp, parent=self)
            self._start_worker(worker, self._on_check_finished)
            return

        self._finish_batch()

    def _start_worker(self, worker: DawCheckWorker | DawTransferWorker, on_finished: Callable[[], None]):
        self._worker = worker
        worker.result.connect(self._record_worker_result)
        worker.finished.connect(on_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @Slot(bool, str)
    def _record_worker_result(self, ok: bool, message: str):
        self._worker_result = (ok, message)

    def _take_worker_result(self) -> tuple[bool, str]:
        # A worker can emit its result before run() unwinds.
        result = self._worker_result
        self._worker_result = None
        self._worker = None
        return result if result is not None else (False, "DAW worker finished without a result.")

    @Slot()
    def _on_check_finished(self):
        ok, message = self._take_worker_result()
        if not ok:
            self._complete_item("Failed", f"DAW Check Failed: {message}")
            self._process_next()
            return

        item = self._queue[self._current_index]
        item.result_text = "Transferring..."
        self.item_finished.emit(item.id, item.status, item.result_text)
        self.batch_progress_message.emit(f"[{item.project_name}] Transferring...")

        worker = DawTransferWorker(
            self._current_dp, self._current_session, item.output_path, parent=self)
        worker.progress.connect(self._on_transfer_progress)
        worker.progress_value.connect(self._on_transfer_progress_value)
        self._start_worker(worker, self._on_transfer_finished)

    @Slot(str)
    def _on_transfer_progress(self, message: str):
        if self._current_index < len(self._queue):
            item = self._queue[self._current_index]
            self.batch_progress_message.emit(f"[{item.project_name}] {message}")

    @Slot(int, int)
    def _on_transfer_progress_value(self, current: int, total: int):
        fraction = current / total if total > 0 else 0
        if not self._is_single_job:
            overall_total = len(self._queue) * 100
            overall_current = int((self._current_index * 100) + (fraction * 100))
        else:
            overall_total = 100
            overall_current = int(fraction * 100)

        self.batch_progress_value.emit(overall_current, overall_total)

    @Slot()
    def _on_transfer_finished(self):
        ok, message = self._take_worker_result()
        self._complete_item("Success" if ok else "Failed", message)
        self._process_next()

    def _complete_item(self, status: str, message: str):
        item = self._queue[self._current_index]
        item.status = status
        item.result_text = message
        self.item_finished.emit(item.id, item.status, item.result_text)
        self._current_index += 1

    def _finish_batch(self):
        completed = self._queue[:self._current_index]
        if completed:
            failures = sum(item.status == "Failed" for item in completed)
            if self._is_single_job:
                message = completed[0].result_text
            elif failures:
                message = f"Batch finished with {failures} failed item(s)."
            else:
                message = "Batch processing complete."
            self.batch_progress_message.emit(message)

        self._running = False
        self._stop_requested = False
        self._current_index = 0
        self._queue = []
        self._current_dp = None
        self._current_session = None
        self.finished.emit()

    def _main_config(self) -> dict[str, Any]:
        mw = self._main_window
        if mw is None:
            return {}
        state = getattr(mw, "state", None)
        if state is not None and isinstance(getattr(state, "config", None), dict):
            return state.config
        cfg = getattr(mw, "_config", None)
        return cfg if isinstance(cfg, dict) else {}

    def _rehydrate_session(self, state_dict: dict[str, Any]) -> SessionContext:
        """Create a SessionContext instance from the captured dictionary state."""
        from sessionpreplib.models import SessionContext
        from sessionpreplib.detectors import default_detectors
        from sessionpreplib.processors import default_processors
        from sessionpreplib.daw_processors import default_daw_processors
        from sessionpreplib.config import default_config

        tracks = state_dict.get("tracks", [])
        source_dir = state_dict.get("source_dir", "")
        main_config = self._main_config()
        app_config = main_config.get("app", {})
        restore_layout_sources(state_dict, app_config.get("phase1_output_folder", "sp_01_tracklayout"))

        flat_config = dict(default_config())

        # Inject defaults for all components so that toggles like protools_enabled exist
        for det in default_detectors():
            for param in getattr(det.__class__, "config_params", lambda: [])():
                flat_config[param.key] = param.default
        for proc in default_processors():
            for param in getattr(proc.__class__, "config_params", lambda: [])():
                flat_config[param.key] = param.default
        for dp in default_daw_processors():
            for param in getattr(dp.__class__, "config_params", lambda: [])():
                flat_config[param.key] = param.default

        if state_dict.get("session_config"):
            from sessionpreplib.config import flatten_structured_config
            flat_config.update(flatten_structured_config(state_dict["session_config"]))

        # Re-inject the saved groups and colors for the DAW processor to use
        # (This matches what _do_daw_transfer does before calling the worker)
        flat_config.setdefault("gui", {})["groups"] = state_dict.get("session_groups", [])

        # Colors come from the active global config via main_window.state.config.
        from sessionprepgui.theme import PT_DEFAULT_COLORS
        flat_config["gui"]["colors"] = main_config.get("colors", PT_DEFAULT_COLORS)

        flat_config["_source_dir"] = source_dir
        flat_config["_use_processed"] = state_dict.get("use_processed", False)
        flat_config["_prepare_errors"] = state_dict.get("prepare_errors", [])

        all_detectors = default_detectors()
        for d in all_detectors:
            d.configure(flat_config)

        all_processors = []
        for proc in default_processors():
            proc.configure(flat_config)
            if proc.enabled:
                all_processors.append(proc)
        all_processors.sort(key=lambda p: p.priority)

        session = SessionContext(
            tracks=tracks,
            config=flat_config,
            groups={},
            detectors=all_detectors,
            processors=all_processors,
            daw_state=state_dict.get("daw_state", {}),
            prepare_state=state_dict.get("prepare_state", "none"),
            transfer_manifest=state_dict.get("transfer_manifest", []),
            base_transfer_manifest=state_dict.get("base_transfer_manifest", []),
            project_name=state_dict.get("project_name", ""),
        )
        # Assuming topology and topology_applied are needed we could restore them too,
        # but for transfer, `transfer_manifest` and `output_tracks` (which we rebuilt during load) are key.
        # Restore output_tracks directly from the state dict (added in v6 format)
        # Rebuilding from topology would lose processor_results (e.g. fader_offset)
        session.output_tracks = state_dict.get("output_tracks", [])

        return session

    def _get_daw_processor(self, dp_id: str, flat_config: dict[str, Any]) -> DawProcessor | None:
        processors = create_runtime_daw_processors(flat_config)
        for dp in processors:
            if dp.id == dp_id:
                return dp
        return None
