"""Batch queue UI controller: owns BatchManager, BatchQueueDock, and queue file I/O."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from sessionpreplib.file_io import atomic_writer

from ..shell import MainWindowShell
from ..state import WorkspaceState
from .manager import BatchManager
from .panel import BatchItem, BatchQueueDock


class BatchQueueController(QObject):
    """Coordinates BatchQueueDock, BatchManager, queue persistence, and shutdown deferral."""

    batch_mode_changed = Signal(bool)
    clear_workspace_requested = Signal()
    close_ready = Signal()

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        main_window: QMainWindow,
        *,
        on_restore_session: Callable[[dict[str, Any]], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._main_window = main_window
        self._on_restore_session = on_restore_session
        self._batch_mode_action: QAction | None = None
        self._close_after_batch = False

        self.manager = BatchManager(main_window)
        self.manager.started.connect(self._on_batch_started)
        self.manager.finished.connect(self._on_batch_finished)
        self.manager.item_finished.connect(self._on_batch_item_finished)

        self.dock = BatchQueueDock(main_window)
        self.dock.load_requested.connect(self.load_batch_item)
        self.dock.run_batch_requested.connect(self.manager.start_batch)
        self.dock.run_single_requested.connect(self.manager.start_single)
        self.dock.open_project_requested.connect(self.open_project_folder)

        self.manager.batch_progress_value.connect(self.dock.update_progress)
        self.manager.batch_progress_message.connect(self.dock.update_progress_message)
        self.manager.batch_progress_message.connect(self._shell.show_status)

        main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()

    def bind_batch_mode_action(self, action: QAction) -> None:
        self._batch_mode_action = action
        action.toggled.connect(self.on_batch_mode_toggled)

    @property
    def is_batch_mode(self) -> bool:
        return bool(self._batch_mode_action is not None and self._batch_mode_action.isChecked())

    @property
    def is_running(self) -> bool:
        return self.manager.is_running

    @property
    def close_after_batch(self) -> bool:
        return self._close_after_batch

    def reset_close_flag(self) -> None:
        self._close_after_batch = False

    def enqueue_item(self, item: BatchItem) -> bool:
        return bool(self.dock.add_item(item))

    def confirm_or_defer_close(self, *, closing_after_daw: bool) -> bool:
        """Return True if window close must be ignored/deferred by batch queue state."""
        if self.dock.has_items and not (self._close_after_batch or closing_after_daw):
            reply = QMessageBox.warning(
                self._shell.dialog_parent(),
                "Pending Batch Items",
                "You have pending sessions in the batch queue. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return True
        if self.manager.is_running:
            self._close_after_batch = True
            self.manager.stop_after_current()
            return True
        return False

    @Slot()
    def save_queue(self) -> None:
        parent = self._shell.dialog_parent()
        if not self.dock.has_items:
            QMessageBox.information(parent, "Save Batch Queue", "The batch queue is empty.")
            return

        start_dir = self._state.config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save Batch Queue",
            start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)",
        )
        if not path:
            return

        try:
            state = self.dock.get_state()
            with atomic_writer(path) as handle:
                json.dump(state, handle, indent=2)
            self._shell.show_status(f"Batch queue saved to {path}")
        except Exception as exc:
            QMessageBox.critical(
                parent, "Save Batch Queue Failed", f"Could not save batch queue:\n\n{exc}"
            )

    @Slot()
    def load_queue(self) -> None:
        parent = self._shell.dialog_parent()
        start_dir = self._state.config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load Batch Queue",
            start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception as exc:
            QMessageBox.critical(
                parent, "Load Batch Queue Failed", f"Could not load batch queue:\n\n{exc}"
            )
            return

        if not isinstance(state, list):
            QMessageBox.critical(
                parent, "Invalid File", "The selected file is not a valid batch queue format."
            )
            return

        append = self._ask_append_or_replace(parent)
        if append is None:
            return

        try:
            self.dock.load_state(state, append=append)
        except ValueError as exc:
            QMessageBox.critical(parent, "Load Batch Queue Failed", str(exc))
            return

        if self._batch_mode_action is not None:
            self._batch_mode_action.setChecked(True)
        self.clear_workspace_requested.emit()
        self._shell.show_status(f"Batch queue loaded from {path}")

    def _ask_append_or_replace(self, parent: Any) -> bool | None:
        if not self.dock.has_items:
            return False
        msg = QMessageBox(parent)
        msg.setWindowTitle("Load Batch Queue")
        msg.setText("How would you like to load the batch queue?")
        replace_btn = msg.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        append_btn = msg.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        choice = msg.clickedButton()
        if choice == append_btn:
            return True
        if choice == replace_btn:
            return False
        return None

    @Slot(bool)
    def on_batch_mode_toggled(self, checked: bool) -> None:
        self.dock.setVisible(checked)
        self.batch_mode_changed.emit(checked)

    @Slot(object)
    def load_batch_item(self, item: BatchItem) -> None:
        if self._state.session:
            reply = QMessageBox.question(
                self._shell.dialog_parent(),
                "Load Session",
                "You have an active session in the workspace. Discard current workspace and load from queue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if self._on_restore_session(item.session_state):
            self.dock.remove_item(item.id)

    @Slot(object)
    def open_project_folder(self, item: BatchItem) -> None:
        parent = self._shell.dialog_parent()
        daw_processor = next(
            (dp for dp in self._state.daw_processors if dp.id == item.daw_processor_id),
            None,
        )
        if not daw_processor:
            QMessageBox.warning(
                parent,
                "Open Project Folder",
                f"The configured DAW processor '{item.daw_processor_name}' is not available.",
            )
            return

        project_dir = getattr(daw_processor, "project_dir", "").strip()
        if not project_dir:
            QMessageBox.information(
                parent,
                "Open Project Folder",
                f"No project directory configured for {daw_processor.name}.",
            )
            return

        if item.output_path and os.path.exists(os.path.dirname(item.output_path)):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(item.output_path)))
            return

        target_path = (
            os.path.join(project_dir, item.project_name) if item.project_name else project_dir
        )
        if os.path.isdir(target_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_path))
        elif os.path.isdir(project_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(project_dir))
        else:
            QMessageBox.warning(
                parent,
                "Open Project Folder",
                f"The directory does not exist:\n\n{project_dir}",
            )

    @Slot()
    def _on_batch_started(self) -> None:
        self.dock.set_running_state(True)

    @Slot()
    def _on_batch_finished(self) -> None:
        if self._close_after_batch:
            QTimer.singleShot(0, self.close_ready.emit)
        else:
            self.dock.set_running_state(False)

    @Slot(str, str, str)
    def _on_batch_item_finished(self, item_id: str, status: str, result_text: str) -> None:
        self.dock.update_item(item_id, status, result_text)
