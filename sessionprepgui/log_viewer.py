"""Detached log viewer for SessionPrep."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.logging_setup import (
    ensure_file_logging,
    get_current_log_level,
    get_log_dir,
    get_log_path,
    set_runtime_log_level,
)


_LOG_LINE_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[(?P<level>[A-Z ]{5})\] "
    r"(?P<logger>[^:]+): "
    r"(?P<message>.*)$"
)


@dataclass(frozen=True)
class LogRow:
    """Parsed display data for one log line."""

    time: str
    level: str
    logger: str
    message: str


def parse_log_line(line: str) -> LogRow:
    """Parse a SessionPrep log line into table columns."""
    text = line.rstrip("\r\n")
    match = _LOG_LINE_RE.match(text)
    if not match:
        return LogRow("", "", "", text)
    return LogRow(
        match.group("time"),
        match.group("level").strip(),
        match.group("logger"),
        match.group("message"),
    )


def read_last_lines(path: str, line_count: int) -> list[str]:
    """Read at most ``line_count`` trailing lines without loading large files."""
    if line_count <= 0 or not os.path.exists(path):
        return []

    chunk_size = 8192
    chunks: list[bytes] = []
    newline_count = 0
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newline_count <= line_count:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    data = b"".join(reversed(chunks))
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-line_count:]


class LogViewerWindow(QDialog):
    """Detached window for viewing and tailing the SessionPrep log."""

    _LEVELS = [
        ("Debug", logging.DEBUG),
        ("Info", logging.INFO),
        ("Warning", logging.WARNING),
        ("Error", logging.ERROR),
        ("Critical", logging.CRITICAL),
        ("Off", None),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Viewer")
        self.setWindowFlag(Qt.Window, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setMinimumSize(700, 360)
        self.resize(1000, 650)

        self._offset = 0
        self._partial_line = ""
        self._timer = QTimer(self)
        self._timer.setInterval(750)
        self._timer.timeout.connect(self._poll_log_file)

        self._init_ui()
        self._refresh_level_combo()
        self._reload_from_tail()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("Log level:"))
        self._level_combo = QComboBox(self)
        for label, level in self._LEVELS:
            self._level_combo.addItem(label, level)
        self._level_combo.currentIndexChanged.connect(self._on_level_changed)
        top.addWidget(self._level_combo)
        top.addStretch()
        layout.addLayout(top)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Time", "Level", "Logger", "Message"])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.setStyleSheet(
            "QTableWidget::item { padding-top: 0px; padding-bottom: 0px; }"
        )
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.verticalHeader().setMinimumSectionSize(18)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._update_copy_enabled)
        layout.addWidget(self._table, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("Lines:"))
        self._line_count = QSpinBox(self)
        self._line_count.setRange(0, 50000)
        self._line_count.setValue(5000)
        self._line_count.setSingleStep(100)
        self._line_count.valueChanged.connect(self._reload_from_tail)
        bottom.addWidget(self._line_count)

        self._follow_cb = QCheckBox("Follow", self)
        self._follow_cb.setChecked(True)
        bottom.addWidget(self._follow_cb)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom.addWidget(spacer)

        self._copy_all_btn = QPushButton("Copy All Rows", self)
        self._copy_all_btn.clicked.connect(self._copy_all_rows)
        bottom.addWidget(self._copy_all_btn)

        self._copy_btn = QPushButton("Copy Selected Rows", self)
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._copy_selection)
        bottom.addWidget(self._copy_btn)

        self._clear_btn = QPushButton("Clear Rows", self)
        self._clear_btn.clicked.connect(self._clear_list)
        bottom.addWidget(self._clear_btn)

        self._open_folder_btn = QPushButton("Open Log Folder", self)
        self._open_folder_btn.clicked.connect(self._open_log_folder)
        bottom.addWidget(self._open_folder_btn)
        layout.addLayout(bottom)

    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        self._refresh_level_combo()
        self._reload_from_tail()
        self._timer.start()

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self._timer.stop()
        super().closeEvent(event)

    def _refresh_level_combo(self) -> None:
        current_level = get_current_log_level()
        self._level_combo.blockSignals(True)
        for index, (_label, level) in enumerate(self._LEVELS):
            if level == current_level:
                self._level_combo.setCurrentIndex(index)
                break
        else:
            self._level_combo.setCurrentIndex(1)
        self._level_combo.blockSignals(False)

    def _on_level_changed(self, index: int) -> None:
        set_runtime_log_level(self._level_combo.itemData(index))
        self._reload_from_tail()

    def _reload_from_tail(self) -> None:
        ensure_file_logging()
        path = get_log_path()
        self._partial_line = ""
        self._table.setRowCount(0)
        for line in read_last_lines(path, self._line_count.value()):
            self._append_log_line(line)
        self._offset = os.path.getsize(path) if os.path.exists(path) else 0
        if self._follow_cb.isChecked():
            self._scroll_to_bottom()

    def _poll_log_file(self) -> None:
        if not self._follow_cb.isChecked():
            return

        path = get_log_path()
        if not os.path.exists(path):
            return

        size = os.path.getsize(path)
        if size < self._offset:
            self._reload_from_tail()
            return
        if size == self._offset:
            return

        with open(path, "rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
            self._offset = handle.tell()

        text = self._partial_line + data.decode("utf-8", errors="replace")
        if text.endswith(("\n", "\r")):
            lines = text.splitlines()
            self._partial_line = ""
        else:
            lines = text.splitlines()
            self._partial_line = lines.pop() if lines else text

        if not lines:
            return
        for line in lines:
            self._append_log_line(line)
        self._trim_rows()
        self._scroll_to_bottom()

    def _append_log_line(self, line: str) -> None:
        row_data = parse_log_line(line)
        row = self._table.rowCount()
        self._table.insertRow(row)
        for column, value in enumerate(
            [row_data.time, row_data.level, row_data.logger, row_data.message]
        ):
            item = QTableWidgetItem(value)
            if column in (0, 1):
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._table.setItem(row, column, item)

    def _trim_rows(self) -> None:
        max_rows = self._line_count.value() or self._line_count.maximum()
        while self._table.rowCount() > max_rows:
            self._table.removeRow(0)

    def _scroll_to_bottom(self) -> None:
        if self._table.rowCount() > 0:
            self._table.scrollToBottom()

    def _update_copy_enabled(self) -> None:
        self._copy_btn.setEnabled(bool(self._selected_rows()))

    def _selected_rows(self) -> list[int]:
        rows = {index.row() for index in self._table.selectionModel().selectedRows()}
        return sorted(rows)

    def _copy_selection(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        QApplication.clipboard().setText(self._table_rows_text(rows))

    def _copy_all_rows(self) -> None:
        rows = list(range(self._table.rowCount()))
        if not rows:
            return
        QApplication.clipboard().setText(self._table_rows_text(rows))

    def _table_rows_text(self, rows: list[int]) -> str:
        lines = []
        for row in rows:
            values = []
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                values.append(item.text() if item else "")
            lines.append("\t".join(values))
        return "\n".join(lines)

    def _clear_list(self) -> None:
        self._table.setRowCount(0)
        self._partial_line = ""
        path = get_log_path()
        self._offset = os.path.getsize(path) if os.path.exists(path) else 0
        self._update_copy_enabled()

    def _open_log_folder(self) -> None:
        folder = get_log_dir()
        os.makedirs(folder, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
