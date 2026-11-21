"""Phase 3 (DAW Transfer) controller: processors, fetch, transfer, folder tree."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
import logging
import os
import re
from typing import Any
import uuid

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QTreeWidgetItem

from sessionpreplib.config import flatten_structured_config
from sessionpreplib.daw_processors import create_runtime_daw_processors
from sessionpreplib.models import SessionContext, TransferEntry

from ..analysis.worker import DawCheckWorker, DawFetchWorker, DawTransferWorker
from ..controls import set_action_busy
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import COLORS, PT_DEFAULT_COLORS
from ..tracks.table_widgets import _PHASE_SETUP, _SETUP_RIGHT_PLACEHOLDER, _SETUP_RIGHT_TREE
from .page import DawSetupPage

log = logging.getLogger(__name__)


class DawTransferController(QObject):
    """Owns Phase 3 (DAW Transfer) page, workers, folder tree, and assignments."""

    transfer_finished = Signal(bool, str)

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        on_populate_setup_table: Callable[[], None],
        on_group_color_map: Callable[[], dict[str, str]],
        on_tint_group_color: Callable[[str | None, dict[str, str] | None], Any],
        on_normalize_groups: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        on_capture_session_state: Callable[[], dict[str, Any]],
        on_enqueue_batch_item: Callable[[Any], bool],
        on_clear_workspace: Callable[[], None],
        on_is_prepare_running: Callable[[], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._on_populate_setup_table = on_populate_setup_table
        self._on_group_color_map = on_group_color_map
        self._on_tint_group_color = on_tint_group_color
        self._on_normalize_groups = on_normalize_groups
        self._on_capture_session_state = on_capture_session_state
        self._on_enqueue_batch_item = on_enqueue_batch_item
        self._on_clear_workspace = on_clear_workspace
        self._on_is_prepare_running = on_is_prepare_running

        self.check_worker: DawCheckWorker | None = None
        self.pending_after_check: Callable[[], None] | None = None
        self.fetch_worker: DawFetchWorker | None = None
        self.transfer_worker: DawTransferWorker | None = None
        self.setup_table_populating: bool = False

        self.page = DawSetupPage()
        self._shell.register_phase_progress_layout(_PHASE_SETUP, self.page.main_layout)
        self._connect_page_signals()
        self.configure_daw_processors()
        self.populate_daw_combo()
        self.page.daw_combo.currentIndexChanged.connect(self._on_daw_combo_changed)

    def _connect_page_signals(self) -> None:
        p = self.page
        p.reset_manifest_action.triggered.connect(self.reset_manifest)
        p.use_processed_cb.toggled.connect(self._on_use_processed_toggled)
        p.fetch_action.clicked.connect(lambda _c=False: self.fetch(ignore_cache=False))
        p.fetch_cached_action.triggered.connect(lambda _c=False: self.fetch(ignore_cache=False))
        p.fetch_ignore_cache_action.triggered.connect(lambda _c=False: self.fetch(ignore_cache=True))
        p.auto_assign_action.triggered.connect(self.auto_assign)
        p.unassign_all_action.triggered.connect(self.unassign_all)
        p.transfer_action.triggered.connect(self.transfer)
        p.setup_table.customContextMenuRequested.connect(self._on_setup_table_context_menu)
        p.setup_table.itemChanged.connect(self._on_setup_table_item_changed)
        p.project_name_edit.textChanged.connect(self._on_project_name_changed)
        p.open_project_btn.clicked.connect(self._on_open_active_project_folder)
        p.folder_tree.tracks_dropped.connect(self.assign_tracks_to_folder)
        p.folder_tree.tracks_unassigned.connect(self.unassign_tracks)

    # ── Lifecycle & workspace clearing ───────────────────────────────────

    def clear_workspace(self) -> None:
        set_action_busy(self.page.transfer_action, False)
        self.page.use_processed_cb.setChecked(False)
        self.update_use_processed_action()
        self.page.setup_table.setRowCount(0)
        self.page.folder_tree.clear()
        self.page.right_stack.setCurrentIndex(_SETUP_RIGHT_PLACEHOLDER)
        self.page.project_name_edit.clear()

    def has_running_workers(self) -> bool:
        for kind in (DawCheckWorker, DawFetchWorker, DawTransferWorker):
            if any(w.isRunning() for w in self.findChildren(kind)):
                return True
        return False

    def dispose(self) -> None:
        self.pending_after_check = None
        for attr in ("check_worker", "fetch_worker", "transfer_worker"):
            worker = getattr(self, attr, None)
            if isinstance(worker, QThread) and worker.isRunning():
                worker.requestInterruption()
                worker.wait()
            setattr(self, attr, None)

    # ── DAW processor configuration & template bindings ──────────────────

    def configure_daw_processors(self) -> None:
        """Rebuild DAW processor list from the current flat config."""
        processors = create_runtime_daw_processors(self._state.flat_config(), include_disabled=True)
        previous = self._state.daw_template_bindings
        self._state.daw_template_bindings = {dp.id: dp.template_identity for dp in processors}
        self._state.daw_processors = [dp for dp in processors if dp.enabled]
        if self._state.session:
            self._state.session.daw_state = self.rebind_daw_state(
                self._state.session.daw_state, previous
            )

    def matching_daw_id(self, previous_id: str | None, previous_bindings: dict[str, Any]) -> str | None:
        """Find the same template, including when its index changes."""
        if previous_id is None:
            return None
        identity = previous_bindings.get(previous_id)
        current = self._state.daw_template_bindings
        if identity is None:
            return None
        if current.get(previous_id) == identity:
            return previous_id
        matching = [dp_id for dp_id, key in current.items() if key == identity]
        if len(matching) == 1 and list(previous_bindings.values()).count(identity) == 1:
            return matching[0]
        return None

    def rebind_daw_state(self, daw_state: dict[str, Any], previous_bindings: dict[str, Any]) -> dict[str, Any]:
        if previous_bindings == self._state.daw_template_bindings:
            return daw_state
        rebound = {
            dp_id: value
            for dp_id, value in daw_state.items()
            if dp_id not in previous_bindings and dp_id not in self._state.daw_template_bindings
        }
        for previous_id in daw_state.keys() & previous_bindings.keys():
            new_id = self.matching_daw_id(previous_id, previous_bindings)
            if new_id is not None:
                rebound[new_id] = daw_state[previous_id]
        return rebound

    def restore_daw_bindings(self, data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        daw_state = deepcopy(data.get("daw_state", {}))
        active_id = data.get("active_daw_processor_id")
        saved_config = data.get("session_config")
        if not isinstance(saved_config, dict):
            return daw_state, active_id
        previous = {
            dp.id: dp.template_identity
            for dp in create_runtime_daw_processors(
                flatten_structured_config(saved_config), include_disabled=True
            )
        }
        return self.rebind_daw_state(daw_state, previous), self.matching_daw_id(active_id, previous)

    def populate_daw_combo(self, preferred_id: str | None = None) -> None:
        selected = -1
        combo = self.page.daw_combo
        combo.blockSignals(True)
        combo.clear()
        for i, dp in enumerate(self._state.daw_processors):
            if dp.enabled:
                combo.addItem(dp.name, i)
                if selected < 0 or dp.id == preferred_id:
                    selected = combo.count() - 1
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)
        if combo.count() > 0:
            self._on_daw_combo_changed(selected)
        else:
            self._state.active_daw_processor = None

    def refresh_from_config(self) -> None:
        """Rebuild DAW processors and related Phase 3 UI from current config."""
        active_id = getattr(self._state.active_daw_processor, "id", None)
        previous = self._state.daw_template_bindings
        self.configure_daw_processors()
        self.populate_daw_combo(self.matching_daw_id(active_id, previous))
        if self._state.session and (
            previous != self._state.daw_template_bindings
            or active_id != getattr(self._state.active_daw_processor, "id", None)
        ):
            self.populate_folder_tree()
            self._on_populate_setup_table()
        self.page.daw_check_label.setText("")
        self.update_lifecycle_buttons()

    def daw_operation_running(self) -> bool:
        return any(
            w is not None for w in (self.check_worker, self.fetch_worker, self.transfer_worker)
        )

    def update_lifecycle_buttons(self) -> None:
        """Enable/disable Fetch/Transfer/Assign buttons based on active processor state."""
        p = self.page
        set_action_busy(p.transfer_action, self.transfer_worker is not None, "Creating\u2026")
        if self.daw_operation_running():
            for widget in (
                p.fetch_action,
                p.fetch_menu_button,
                p.auto_assign_action,
                p.unassign_all_action,
                p.transfer_action,
                p.reset_manifest_action,
            ):
                widget.setEnabled(False)
            return

        has_processor = self._state.active_daw_processor is not None
        p.fetch_action.setEnabled(has_processor)
        p.fetch_menu_button.setEnabled(has_processor)
        dp_id = self._state.active_daw_processor.id if has_processor else None
        dp_state = (
            self._state.session.daw_state.get(dp_id, {})
            if self._state.session and dp_id
            else {}
        )
        has_folders = bool(dp_state.get("folders"))
        has_assignments = bool(dp_state.get("assignments"))
        p.auto_assign_action.setEnabled(has_folders)
        p.unassign_all_action.setEnabled(has_assignments)
        p.transfer_action.setEnabled(
            has_processor and has_assignments and not self._on_is_prepare_running()
        )
        p.transfer_action.setText("Enqueue >>" if self._shell.is_batch_mode() else "Create")
        p.reset_manifest_action.setEnabled(
            bool(self._state.session and self._state.session.transfer_manifest)
        )

    @Slot(int)
    def _on_daw_combo_changed(self, index: int) -> None:
        if index < 0 or index >= self.page.daw_combo.count():
            self._state.active_daw_processor = None
        else:
            proc_idx = self.page.daw_combo.itemData(index)
            self._state.active_daw_processor = self._state.daw_processors[proc_idx]
        self.page.daw_check_label.setText("")
        self.update_lifecycle_buttons()

    # ── Connectivity check & Fetch ───────────────────────────────────────

    def run_daw_check_then(self, processor: Any, on_success: Callable[[], None]) -> None:
        self.pending_after_check = on_success
        self.page.daw_check_label.setText("Connecting\u2026")
        self.page.daw_check_label.setStyleSheet(f"color: {COLORS['dim']};")
        worker = DawCheckWorker(processor, parent=self)
        self.check_worker = worker
        worker.result.connect(self._on_daw_check_result)
        worker.finished.connect(self._shell.resume_close_after_daw)
        worker.finished.connect(worker.deleteLater)
        self.update_lifecycle_buttons()
        worker.start()

    @Slot(bool, str)
    def _on_daw_check_result(self, ok: bool, message: str) -> None:
        worker = self.check_worker
        if worker is None:
            return
        self.check_worker = None
        if self._shell.is_closing_after_daw():
            return
        if ok:
            self.page.daw_check_label.setText(message)
            self.page.daw_check_label.setStyleSheet(f"color: {COLORS['clean']};")
            cb = self.pending_after_check
            self.pending_after_check = None
            if cb:
                cb()
        else:
            self.page.daw_check_label.setText("Connection failed")
            self.page.daw_check_label.setStyleSheet(f"color: {COLORS['problems']};")
            self.pending_after_check = None
            QMessageBox.warning(
                self._shell.dialog_parent(),
                "Connection Failed",
                f"{worker._processor.name} connection could not be established.\n\n{message}",
            )
        self.update_lifecycle_buttons()

    @Slot()
    def fetch(self, ignore_cache: bool = False) -> None:
        if (
            not self._state.active_daw_processor
            or not self._state.session
            or self.daw_operation_running()
        ):
            return
        log.info("DAW fetch: %s ignore_cache=%s", self._state.active_daw_processor.name, ignore_cache)
        self.page.fetch_action.setEnabled(False)
        self.page.fetch_menu_button.setEnabled(False)
        self._start_fetch(ignore_cache=ignore_cache)

    def _start_fetch(self, ignore_cache: bool = False) -> None:
        message = (
            "Fetching folder structure without cache..."
            if ignore_cache
            else "Fetching folder structure..."
        )
        self.page.right_stack.setCurrentIndex(_SETUP_RIGHT_TREE)
        self.page.folder_tree.clear()
        self._shell.progress_start(message)

        worker = DawFetchWorker(
            self._state.active_daw_processor,
            self._state.session,
            ignore_cache=ignore_cache,
            parent=self,
        )
        self.fetch_worker = worker
        worker.progress.connect(self._on_transfer_progress)
        worker.progress_value.connect(self._on_transfer_progress_value)
        worker.result.connect(self._on_daw_fetch_result)
        worker.finished.connect(self._shell.resume_close_after_daw)
        worker.finished.connect(worker.deleteLater)
        self.update_lifecycle_buttons()
        worker.start()

    @Slot(bool, str, object)
    def _on_daw_fetch_result(self, ok: bool, message: str, session: Any) -> None:
        worker = self.fetch_worker
        if worker is None:
            return
        self.fetch_worker = None
        if self._shell.is_closing_after_daw():
            return
        if worker._session is not self._state.session:
            self.update_lifecycle_buttons()
            return
        processor = worker._processor
        if self._state.daw_template_bindings.get(processor.id) != processor.template_identity:
            self._shell.progress_fail("DAW template changed. Fetch again.")
            self.update_lifecycle_buttons()
            return

        self.page.fetch_action.setEnabled(True)
        self.page.fetch_menu_button.setEnabled(True)
        if "PRO_TOOLS_SESSION_OPEN" in message:
            self._shell.progress_fail("Fetch aborted: Pro Tools session is open.")
            QMessageBox.warning(
                self._shell.dialog_parent(),
                "Pro Tools Session Open",
                "A Pro Tools session is currently open.\n\n"
                "Please save and close the open session in Pro Tools, then try again.",
            )
            self._shell.show_status("Fetch aborted: Pro Tools session is open.")
            self.update_lifecycle_buttons()
            return

        if ok and session is not None:
            assert self._state.session is not None
            if processor.id in session.daw_state:
                self._state.session.daw_state[processor.id] = session.daw_state[processor.id]
            self.populate_folder_tree()
            self.page.right_stack.setCurrentIndex(_SETUP_RIGHT_TREE)
            self._on_populate_setup_table()
            self._shell.progress_finish(message)
            self._shell.show_status(message)
        else:
            self._shell.progress_fail(message)
            QMessageBox.critical(
                self._shell.dialog_parent(),
                "Fetch Failed",
                f"Could not fetch folder structure from {processor.name}.\n\n{message}",
            )
            self._shell.show_status(f"Fetch failed: {message}")
        self.update_lifecycle_buttons()

    def _on_project_name_changed(self, text: str) -> None:
        sanitized = "".join(c for c in text if c not in '<>:"/\\|?*').strip()
        if sanitized != text:
            edit = self.page.project_name_edit
            edit.blockSignals(True)
            pos = edit.cursorPosition()
            edit.setText(sanitized)
            edit.setCursorPosition(max(0, pos - 1))
            edit.blockSignals(False)
        if self._state.session:
            self._state.session.project_name = sanitized

    @Slot(bool)
    def _on_use_processed_toggled(self, checked: bool) -> None:
        if self._state.session:
            self._state.session.config["_use_processed"] = checked
        self.update_use_processed_action()

    def update_use_processed_action(self) -> None:
        cb = self.page.use_processed_cb
        if not self._state.session:
            cb.setEnabled(False)
            cb.setText("Use Processed")
            return
        state = self._state.session.prepare_state
        cb.setEnabled(state in ("ready", "stale"))
        cb.setText("Use Processed (!)" if state == "stale" and cb.isChecked() else "Use Processed")

    # ── Transfer & validation ────────────────────────────────────────────

    def _validate_transfer_target(self, processor: Any, project_name: str) -> bool:
        parent = self._shell.dialog_parent()
        if not project_name:
            QMessageBox.warning(
                parent, "Project Name Required", "Please enter a Project Name before clicking Create."
            )
            return False
        if not hasattr(processor, "project_dir"):
            return True
        project_dir = processor.project_dir.strip()
        if not project_dir:
            QMessageBox.critical(
                parent,
                "Project Directory Not Set",
                f"A 'Project directory' must be configured in {processor.name} preferences before creating a project.",
            )
            return False
        if not os.path.isdir(project_dir):
            QMessageBox.critical(
                parent,
                "Project Directory Not Found",
                f"The configured project directory does not exist:\n\n{project_dir}\n\n"
                "Please create it or specify a different directory in Preferences.",
            )
            return False
        if processor.id.startswith("protools") and os.path.exists(os.path.join(project_dir, project_name)):
            QMessageBox.warning(
                parent,
                "Project Already Exists",
                f"A folder named '{project_name}' already exists in the project directory.\n\n"
                "Please choose a different project name.",
            )
            return False
        return True

    @Slot()
    def transfer(self) -> None:
        processor, source = self._state.active_daw_processor, self._state.session
        if (
            not processor
            or not source
            or self.daw_operation_running()
            or self._on_is_prepare_running()
        ):
            return
        log.info("DAW transfer: %s (%d tracks)", processor.name, len(source.transfer_manifest))
        project_name = self.page.project_name_edit.text().strip()
        if not self._validate_transfer_target(processor, project_name):
            return

        snapshot = replace(
            source,
            config=self._transfer_config(source),
            project_name=project_name,
            daw_state=deepcopy(source.daw_state),
            transfer_manifest=deepcopy(source.transfer_manifest),
            output_tracks=list(source.output_tracks),
        )
        if self._shell.is_batch_mode():
            self._enqueue_batch_transfer(processor, snapshot, project_name)
            return

        self.run_daw_check_then(processor, lambda: self._do_transfer(processor, source, snapshot))

    def _enqueue_batch_transfer(self, processor: Any, snapshot: SessionContext, project_name: str) -> None:
        from ..batch import BatchItem

        output_path = self.resolve_output_path(processor, snapshot, batch_mode=True)
        if output_path is None:
            return
        item = BatchItem(
            id=uuid.uuid4().hex,
            project_name=project_name,
            daw_processor_id=processor.id,
            daw_processor_name=processor.name,
            output_path=output_path,
            session_state=self._on_capture_session_state(),
        )
        if self._on_enqueue_batch_item(item):
            self._on_clear_workspace()
            self._shell.show_status("Session added to batch queue.")

    def _transfer_config(self, session: SessionContext) -> dict[str, Any]:
        config = {**session.config, **self._state.flat_config()}
        config["gui"] = {
            **config.get("gui", {}),
            "groups": self._on_normalize_groups(self._state.session_groups),
            "colors": deepcopy(self._state.config.get("colors", PT_DEFAULT_COLORS)),
        }
        config["_source_dir"] = self._state.source_dir
        config["_use_processed"] = self.page.use_processed_cb.isChecked()
        config["_output_folder"] = self._state.config.get("app", {}).get(
            "phase2_output_folder", "sp_02_prepared"
        )
        return config

    def resolve_output_path(
        self, processor: Any, snapshot: SessionContext, *, batch_mode: bool
    ) -> str | None:
        """Compute a processor's output path, prompting interactively unless in batch mode."""
        default_path: str | None = processor.default_output_path(snapshot)
        if not default_path or batch_mode or not processor.output_file_filter:
            return default_path
        path, _ = QFileDialog.getSaveFileName(
            self._shell.dialog_parent(),
            processor.output_dialog_title or "Save Output",
            default_path,
            processor.output_file_filter,
        )
        return path if path else None

    def _do_transfer(self, processor: Any, source: SessionContext, snapshot: SessionContext) -> None:
        if self._state.session is not source:
            return
        output_path = self.resolve_output_path(processor, snapshot, batch_mode=False)
        if output_path is None:
            self.update_lifecycle_buttons()
            return

        self._shell.progress_start(f"Transferring to {processor.name}\u2026")
        worker = DawTransferWorker(
            processor, snapshot, output_path, parent=self, close_session=False
        )
        self.transfer_worker = worker
        worker.progress.connect(self._on_transfer_progress)
        worker.progress_value.connect(self._on_transfer_progress_value)
        worker.result.connect(self._on_daw_transfer_result)
        worker.finished.connect(self._shell.resume_close_after_daw)
        worker.finished.connect(worker.deleteLater)
        self.update_lifecycle_buttons()
        worker.start()

    @Slot(str)
    def _on_transfer_progress(self, message: str) -> None:
        self._shell.progress_message(message)

    @Slot(int, int)
    def _on_transfer_progress_value(self, current: int, total: int) -> None:
        self._shell.progress_value(current, total)

    @Slot(bool, str, object)
    def _on_daw_transfer_result(self, ok: bool, message: str, _results: Any) -> None:
        if self.transfer_worker is None:
            return
        set_action_busy(self.page.transfer_action, False)
        self.transfer_worker = None
        if self._shell.is_closing_after_daw():
            return
        self.update_lifecycle_buttons()
        if ok:
            log.info("DAW transfer complete")
            self._shell.progress_finish(message)
        else:
            log.error("DAW transfer failed: %s", message)
            self._shell.progress_fail(message)
        self._shell.show_status(message)
        self.transfer_finished.emit(ok, message)

    # ── Folder tree & assignments ────────────────────────────────────────

    def populate_folder_tree(self) -> None:
        """Build the folder tree from the active DAW processor's daw_state."""
        tree = self.page.folder_tree
        tree.clear()
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.get(processor.id, {})
        children_map = _build_folder_children_map(dp_state.get("folders", []))
        folder_tracks = _build_folder_tracks_map(
            dp_state.get("assignments", {}), dp_state.get("track_order", {})
        )
        gcm = self._on_group_color_map()
        track_map = {t.filename: t for t in session.output_tracks}
        entry_map = {e.entry_id: e for e in session.transfer_manifest}
        routing_icon = _folder_icon(COLORS["information"])
        basic_icon = _folder_icon(COLORS["dim"])

        for folder in children_map.get(None, []):
            self._add_folder_node(
                tree, folder, children_map, folder_tracks, track_map, entry_map,
                gcm, routing_icon, basic_icon,
            )
        tree.expandAll()

    def _add_folder_node(
        self,
        parent_widget: Any,
        folder: dict[str, Any],
        children_map: dict[str | None, list[dict[str, Any]]],
        folder_tracks: dict[str, list[str]],
        track_map: dict[str, Any],
        entry_map: dict[str, TransferEntry],
        gcm: dict[str, str],
        routing_icon: QIcon,
        basic_icon: QIcon,
    ) -> None:
        item = QTreeWidgetItem(parent_widget)
        item.setText(0, folder["name"])
        item.setData(0, Qt.ItemDataRole.UserRole, folder["id"])
        item.setData(0, Qt.ItemDataRole.UserRole + 1, "folder")
        item.setIcon(0, routing_icon if folder.get("folder_type") == "routing" else basic_icon)
        item.setFlags(
            (item.flags() | Qt.ItemFlag.ItemIsDropEnabled) & ~Qt.ItemFlag.ItemIsDragEnabled
        )
        for fname in folder_tracks.get(folder["id"], []):
            self._add_folder_track_node(item, fname, track_map, entry_map, gcm)
        for child in children_map.get(folder["id"], []):
            self._add_folder_node(
                item, child, children_map, folder_tracks, track_map, entry_map,
                gcm, routing_icon, basic_icon,
            )
        item.setExpanded(True)

    def _add_folder_track_node(
        self,
        parent_item: QTreeWidgetItem,
        fname: str,
        track_map: dict[str, Any],
        entry_map: dict[str, TransferEntry],
        gcm: dict[str, str],
    ) -> None:
        track_item = QTreeWidgetItem(parent_item)
        te = entry_map.get(fname)
        if te:
            default_stem = os.path.splitext(te.output_filename)[0]
            label = (
                f"{te.daw_track_name}  \u2190 {te.output_filename}"
                if te.daw_track_name != default_stem
                else fname
            )
            track_item.setText(0, label)
        else:
            track_item.setText(0, fname)
        track_item.setData(0, Qt.ItemDataRole.UserRole, fname)
        track_item.setData(0, Qt.ItemDataRole.UserRole + 1, "track")
        track_item.setFlags(
            (track_item.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsDropEnabled
        )
        out_fn = te.output_filename if te else fname
        tc = track_map.get(out_fn)
        if tc and tc.group:
            tint = self._on_tint_group_color(tc.group, gcm)
            if tint:
                track_item.setBackground(0, tint)

    @Slot(list, str, int)
    def assign_tracks_to_folder(
        self, filenames: list[str], folder_id: str, insert_index: int = -1
    ) -> None:
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.setdefault(processor.id, {})
        assignments = dp_state.setdefault("assignments", {})
        track_order = dp_state.setdefault("track_order", {})

        for fname in filenames:
            old_fid = assignments.get(fname)
            if old_fid and old_fid in track_order:
                try:
                    track_order[old_fid].remove(fname)
                except ValueError:
                    pass
            assignments[fname] = folder_id

        order = track_order.setdefault(folder_id, [])
        for fname in filenames:
            try:
                order.remove(fname)
            except ValueError:
                pass
        if insert_index < 0 or insert_index >= len(order):
            order.extend(filenames)
        else:
            for i, fname in enumerate(filenames):
                order.insert(insert_index + i, fname)

        self.populate_folder_tree()
        self._on_populate_setup_table()
        self.update_lifecycle_buttons()

    @Slot(list)
    def unassign_tracks(self, filenames: list[str]) -> None:
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.get(processor.id)
        if not dp_state:
            return
        assignments = dp_state.get("assignments", {})
        track_order = dp_state.get("track_order", {})
        for fname in filenames:
            fid = assignments.pop(fname, None)
            if fid and fid in track_order:
                try:
                    track_order[fid].remove(fname)
                except ValueError:
                    pass
        self.populate_folder_tree()
        self._on_populate_setup_table()
        self.update_lifecycle_buttons()

    @Slot()
    def unassign_all(self) -> None:
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.get(processor.id)
        if not dp_state:
            return
        assignments = dp_state.get("assignments", {})
        track_order = dp_state.get("track_order", {})
        if not assignments and not track_order:
            return

        reply = QMessageBox.question(
            self._shell.dialog_parent(),
            "Unassign All",
            "Clear all DAW folder assignments?\n\n"
            "This only changes SessionPrep's local assignment state. "
            "It does not change the DAW project until you create again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        assignments.clear()
        track_order.clear()
        self.populate_folder_tree()
        self._on_populate_setup_table()
        self.update_lifecycle_buttons()
        self._shell.show_status("All DAW folder assignments cleared.")

    @Slot()
    def auto_assign(self) -> None:
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.get(processor.id, {})
        folders = dp_state.get("folders", [])
        if not folders:
            return

        group_target = {
            g["name"]: g["daw_target"].strip().lower()
            for g in self._state.session_groups
            if g.get("daw_target", "").strip()
        }
        if not group_target:
            QMessageBox.information(
                self._shell.dialog_parent(),
                "Auto-Assign",
                "No DAW targets are configured.\n\n"
                "Open the Groups tab and set a DAW Target for each "
                "group that should be mapped to a DAW folder.",
            )
            return

        batch, reasons = _compute_auto_assign_batch(
            session.transfer_manifest, folders, dp_state.get("assignments", {}), group_target
        )
        if not batch:
            detail = "\n".join(reasons) if reasons else "No unassigned tracks found."
            QMessageBox.information(
                self._shell.dialog_parent(), "Auto-Assign", f"Nothing to assign.\n\n{detail}"
            )
            return

        total = 0
        for folder_id, fnames in batch.items():
            self.assign_tracks_to_folder(fnames, folder_id)
            total += len(fnames)
        self._shell.show_status(
            f"Auto-Assign: assigned {total} track(s) to {len(batch)} folder(s)."
        )

    # ── Setup table context menu & manifest edits ────────────────────────

    @Slot(object)
    def _on_setup_table_context_menu(self, pos: Any) -> None:
        session = self._state.session
        if not session:
            return
        row = self.page.setup_table.rowAt(pos.y())
        if row < 0:
            return
        fname_item = self.page.setup_table.item(row, 0)
        entry_id = fname_item.data(Qt.ItemDataRole.UserRole) if fname_item else None
        if not entry_id:
            return
        entry = next((e for e in session.transfer_manifest if e.entry_id == entry_id), None)
        if entry is None:
            return

        menu = QMenu(self._shell.dialog_parent())
        dup_act = menu.addAction("Duplicate")
        dup_act.triggered.connect(lambda _c=False, eid=entry_id: self.duplicate_transfer_entry(eid))
        if entry.entry_id != entry.output_filename:
            menu.addSeparator()
            remove_act = menu.addAction("Remove Duplicate")
            remove_act.triggered.connect(
                lambda _c=False, eid=entry_id: self.remove_transfer_entry(eid)
            )
        menu.exec(self.page.setup_table.viewport().mapToGlobal(pos))

    def duplicate_transfer_entry(self, source_entry_id: str) -> None:
        session = self._state.session
        if not session:
            return
        manifest = session.transfer_manifest
        source_pair = next(
            ((i, e) for i, e in enumerate(manifest) if e.entry_id == source_entry_id), None
        )
        if source_pair is None:
            return
        source_idx, source = source_pair
        siblings = [e for e in manifest if e.output_filename == source.output_filename]
        suffix_re = re.compile(r"-\[(\d+)\]$")
        max_n = max(
            (int(m.group(1)) for sib in siblings if (m := suffix_re.search(sib.daw_track_name))),
            default=0,
        )
        stem = source.daw_track_name
        if stem == source.output_filename:
            stem = os.path.splitext(stem)[0]
        stem = suffix_re.sub("", stem)
        if max_n == 0:
            source.daw_track_name = f"{stem}-[1]"
            new_n = 2
        else:
            new_n = max_n + 1

        new_entry = TransferEntry(
            entry_id=f"{source.output_filename}:{uuid.uuid4().hex[:8]}",
            output_filename=source.output_filename,
            daw_track_name=f"{stem}-[{new_n}]",
            group=source.group,
        )
        manifest.insert(source_idx + 1, new_entry)
        self._on_populate_setup_table()
        self.populate_folder_tree()
        self._shell.show_status(f"Duplicated → '{new_entry.daw_track_name}'")

    def _on_setup_table_item_changed(self, item: Any) -> None:
        if self.setup_table_populating or item.column() != 0 or not self._state.session:
            return
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        if not entry_id:
            return
        new_name = item.text().strip()
        entry = next(
            (e for e in self._state.session.transfer_manifest if e.entry_id == entry_id), None
        )
        if entry is None:
            return
        if not new_name:
            self.page.setup_table.blockSignals(True)
            item.setText(entry.daw_track_name)
            self.page.setup_table.blockSignals(False)
            return
        if entry.daw_track_name != new_name:
            entry.daw_track_name = new_name
            self.populate_folder_tree()

    def reset_manifest(self) -> None:
        session = self._state.session
        if not session:
            return
        if session.base_transfer_manifest:
            session.transfer_manifest = deepcopy(session.base_transfer_manifest)
        elif session.topology:
            from sessionpreplib.topology import build_transfer_manifest

            session.transfer_manifest = build_transfer_manifest(
                session.topology, session.tracks, existing_manifest=None
            )
        else:
            return
        self.prune_assignments_to_manifest()
        self._on_populate_setup_table()
        self.populate_folder_tree()
        self.update_lifecycle_buttons()
        self._shell.show_status("DAW track list reset.")

    def prune_assignments_to_manifest(self) -> None:
        session, processor = self._state.session, self._state.active_daw_processor
        if not session or not processor:
            return
        dp_state = session.daw_state.get(processor.id)
        if not dp_state:
            return
        valid_ids = {entry.entry_id for entry in session.transfer_manifest}
        assignments = dp_state.get("assignments", {})
        for entry_id in list(assignments):
            if entry_id not in valid_ids:
                assignments.pop(entry_id, None)
        track_order = dp_state.get("track_order", {})
        for folder_id in list(track_order):
            kept = [eid for eid in track_order[folder_id] if eid in valid_ids]
            if kept:
                track_order[folder_id] = kept
            else:
                track_order.pop(folder_id, None)

    def remove_transfer_entry(self, entry_id: str) -> None:
        if not self._state.session:
            return
        manifest = self._state.session.transfer_manifest
        for i, entry in enumerate(manifest):
            if entry.entry_id == entry_id:
                if entry.entry_id != entry.output_filename:
                    del manifest[i]
                    self._on_populate_setup_table()
                    self._shell.show_status(f"Removed duplicate '{entry.daw_track_name}'")
                break

    @Slot()
    def _on_open_active_project_folder(self) -> None:
        processor = self._state.active_daw_processor
        if not processor:
            return
        project_name = self.page.project_name_edit.text().strip()
        project_dir = getattr(processor, "project_dir", "").strip()
        parent = self._shell.dialog_parent()
        if not project_dir:
            QMessageBox.information(
                parent, "Open Project Folder", f"No project directory configured for {processor.name}."
            )
            return
        target_path = os.path.join(project_dir, project_name) if project_name else project_dir
        if os.path.isdir(target_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_path))
        elif os.path.isdir(project_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(project_dir))
        else:
            QMessageBox.warning(
                parent, "Open Project Folder", f"The directory does not exist:\n\n{project_dir}"
            )


def _folder_icon(color_hex: str) -> QIcon:
    sz = 14
    pix = QPixmap(sz, sz)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color_hex))
    p.setPen(QPen(QColor(color_hex).darker(130), 1))
    p.drawRoundedRect(1, 1, sz - 2, sz - 2, 3, 3)
    p.end()
    return QIcon(pix)


def _build_folder_children_map(folders: list[dict[str, Any]]) -> dict[str | None, list[dict[str, Any]]]:
    children_map: dict[str | None, list[dict[str, Any]]] = {}
    for folder in folders:
        children_map.setdefault(folder.get("parent_id"), []).append(folder)
    for children in children_map.values():
        children.sort(key=lambda f: f.get("index", 0))
    return children_map


def _build_folder_tracks_map(
    assignments: dict[str, str], track_order: dict[str, list[str]]
) -> dict[str, list[str]]:
    folder_tracks: dict[str, list[str]] = {}
    for fname, fid in assignments.items():
        folder_tracks.setdefault(fid, []).append(fname)
    for fid, fnames in folder_tracks.items():
        order = track_order.get(fid, [])
        order_map = {name: i for i, name in enumerate(order)}
        length = len(order)
        fnames.sort(key=lambda name: (order_map.get(name, length), name))
    return folder_tracks


def _compute_auto_assign_batch(
    manifest: list[TransferEntry],
    folders: list[dict[str, Any]],
    assignments: dict[str, str],
    group_target: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    folder_by_name: dict[str, str] = {}
    for folder in folders:
        key = folder["name"].strip().lower()
        if key and key not in folder_by_name:
            folder_by_name[key] = folder["id"]

    batch: dict[str, list[str]] = {}
    no_group = no_target = no_folder = already_assigned = 0
    for entry in manifest:
        if entry.entry_id in assignments:
            already_assigned += 1
        elif not entry.group:
            no_group += 1
        elif (target_key := group_target.get(entry.group)) is None:
            no_target += 1
        elif (folder_id := folder_by_name.get(target_key)) is None:
            no_folder += 1
        else:
            batch.setdefault(folder_id, []).append(entry.entry_id)

    reasons: list[str] = []
    if no_group:
        reasons.append(f"\u2022 {no_group} track(s) have no group assigned.")
    if no_target:
        reasons.append(f"\u2022 {no_target} track(s) belong to groups without a DAW target.")
    if no_folder:
        reasons.append(f"\u2022 {no_folder} track(s) have DAW targets that don\u2019t match any fetched folder name.")
    if already_assigned:
        reasons.append(f"\u2022 {already_assigned} track(s) are already assigned.")
    return batch, reasons
