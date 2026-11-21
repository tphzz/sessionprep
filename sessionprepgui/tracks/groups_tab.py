"""Phase 2 Groups tab UI container."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..controls import FormActionButton
from .table_widgets import DETAIL_TAB_INSET


class GroupsTabWidget(QWidget):
    """Pure UI container for the session-local Groups editor tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET
        )
        layout.setSpacing(6)

        desc = QLabel(
            "Session-local track groups. Changes here apply only to the current session."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(desc)

        layout.addLayout(self._build_header())
        self.table = self._build_table()
        layout.addWidget(self.table, 1)
        layout.addLayout(self._build_buttons())

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(QLabel("Group Preset:"))
        self.group_preset_combo = QComboBox()
        self.group_preset_combo.setMinimumWidth(140)
        header.addWidget(self.group_preset_combo)
        header.addStretch()
        self.revert_btn = FormActionButton("Revert to Preset")
        self.revert_btn.setToolTip(
            "Replace session-local groups with the selected group preset"
        )
        header.addWidget(self.revert_btn)
        return header

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["Name", "Color", "Gain-Linked", "DAW Target", "Match", "Match Pattern"]
        )
        table.verticalHeader().setSectionsMovable(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        gh = table.horizontalHeader()
        gh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        gh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        gh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        gh.resizeSection(1, 160)
        gh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        gh.resizeSection(2, 80)
        gh.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        gh.resizeSection(3, 140)
        gh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        gh.resizeSection(4, 90)
        gh.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        gh.resizeSection(5, 200)
        return table

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.add_btn = QPushButton("Add")
        row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove")
        row.addWidget(self.remove_btn)
        row.addStretch()
        self.sort_az_btn = QPushButton("Sort A→Z")
        row.addWidget(self.sort_az_btn)
        return row
