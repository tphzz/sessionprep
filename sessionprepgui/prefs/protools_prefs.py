"""Pro Tools Preferences widgets: 3-column templates table and Transfer Mode PTSL wiring."""

from __future__ import annotations

import os
import platform
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import COLORS
from .param_form import PathPicker

PTSL_ONLY_PARAM_KEYS: frozenset[str] = frozenset({
    "protools_temp_dir",
    "protools_company_name",
    "protools_application_name",
    "protools_host",
    "protools_port",
    "protools_command_delay",
    "protools_template_create_timeout",
})

_PATH_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def _auto_ptxt_placeholder(group: str, name: str) -> str:
    if platform.system() not in ("Windows", "Darwin"):
        return "(Required on Linux \u2014 click Browse .ptxt\u2026)"
    g = group.strip() or "<group>"
    n = name.strip() or "<name>"
    return f"(Auto \u2014 ~/Documents/Pro Tools/Session Templates/{g}/{n}.ptxt)"


class ProToolsTemplatesWidget(QWidget):
    """Editable 3-column table of Pro Tools mix templates with optional ``.ptxt`` path."""

    templates_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        top_divider = QFrame()
        top_divider.setFrameShape(QFrame.Shape.HLine)
        top_divider.setStyleSheet(f"color: {COLORS['accent']};")
        layout.addWidget(top_divider)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)
        title_box.addWidget(QLabel("<b>Pro Tools Templates</b>"))
        sub = QLabel(
            "Group + Name resolves from ~/Documents/Pro Tools/Session Templates on Windows/macOS. "
            "Select a row to Browse .ptxt\u2026 or Clear Path (explicit path required on Linux)."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {COLORS['metadata']}; font-size: 9pt;")
        title_box.addWidget(sub)
        header_row.addLayout(title_box, 1)

        self._add_btn = QPushButton("+ Add Template")
        self._browse_btn = QPushButton("Browse .ptxt\u2026")
        self._clear_btn = QPushButton("Clear Path")
        self._remove_btn = QPushButton("Remove")
        for btn in (self._add_btn, self._browse_btn, self._clear_btn, self._remove_btn):
            header_row.addWidget(btn)
        self._add_btn.clicked.connect(self._on_add)
        self._browse_btn.clicked.connect(self._on_browse_ptxt)
        self._clear_btn.clicked.connect(self._on_clear_path)
        self._remove_btn.clicked.connect(self._on_remove)
        layout.addLayout(header_row)

        self._table = self._build_table()
        layout.addWidget(self._table, 1)
        self._sync_selection_buttons()

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(
            ["Template Group", "Template Name", "Optional .ptxt File Path"]
        )
        gh = table.horizontalHeader()
        gh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        gh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        gh.resizeSection(0, 145)
        gh.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        gh.resizeSection(1, 215)
        gh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(130)
        table.itemSelectionChanged.connect(self._sync_selection_buttons)
        table.cellChanged.connect(self._on_cell_changed)
        return table

    def _selected_row(self) -> int:
        sel = self._table.selectionModel()
        if sel is None:
            return -1
        rows = sel.selectedRows()
        return rows[0].row() if rows else -1

    def _sync_selection_buttons(self) -> None:
        row = self._selected_row()
        has_sel = row >= 0
        self._browse_btn.setEnabled(has_sel)
        self._remove_btn.setEnabled(has_sel)
        has_path = bool(self._raw_ptxt_path(row)) if has_sel else False
        self._clear_btn.setEnabled(has_path)

    def _raw_ptxt_path(self, row: int) -> str:
        item = self._table.item(row, 2)
        if item is None:
            return ""
        raw = item.data(_PATH_ROLE)
        return str(raw).strip() if raw else ""

    def _apply_path_item(self, row: int, group: str, name: str, raw_path: str) -> None:
        cleaned = raw_path.strip()
        item = self._table.item(row, 2)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, 2, item)
        item.setData(_PATH_ROLE, cleaned)
        if cleaned:
            item.setText(cleaned)
            item.setData(Qt.ItemDataRole.ForegroundRole, None)
        else:
            item.setText(_auto_ptxt_placeholder(group, name))
            item.setForeground(QColor(COLORS["metadata"]))

    def set_templates(self, templates: list[dict[str, Any]]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._table.setRowCount(len(templates))
        for row, tpl in enumerate(templates):
            group = str(tpl.get("group", "") or "").strip()
            name = str(tpl.get("name", "") or "").strip()
            ptxt_path = str(tpl.get("ptxt_path") or tpl.get("path") or "").strip()
            self._table.setItem(row, 0, QTableWidgetItem(group))
            self._table.setItem(row, 1, QTableWidgetItem(name))
            self._apply_path_item(row, group, name, ptxt_path)
        self._table.blockSignals(False)
        self._sync_selection_buttons()

    def get_templates(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            g_item = self._table.item(row, 0)
            n_item = self._table.item(row, 1)
            group = g_item.text().strip() if g_item else ""
            name = n_item.text().strip() if n_item else ""
            ptxt_path = self._raw_ptxt_path(row)
            if name or group or ptxt_path:
                entry: dict[str, Any] = {"group": group, "name": name}
                if ptxt_path:
                    entry["ptxt_path"] = ptxt_path
                templates.append(entry)
        return templates

    def _on_cell_changed(self, row: int, col: int) -> None:
        self._table.blockSignals(True)
        g_item = self._table.item(row, 0)
        n_item = self._table.item(row, 1)
        group = g_item.text().strip() if g_item else ""
        name = n_item.text().strip() if n_item else ""
        if col == 2:
            p_item = self._table.item(row, 2)
            entered = p_item.text().strip() if p_item else ""
            placeholder = _auto_ptxt_placeholder(group, name)
            raw = "" if (not entered or entered == placeholder or entered.startswith("(Auto \u2014 ") or entered.startswith("(Required on Linux")) else entered
            self._apply_path_item(row, group, name, raw)
        elif not self._raw_ptxt_path(row):
            self._apply_path_item(row, group, name, "")
        self._table.blockSignals(False)
        self._sync_selection_buttons()
        self.templates_changed.emit()

    def _on_add(self) -> None:
        row = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.setRowCount(row + 1)
        self._table.setItem(row, 0, QTableWidgetItem(""))
        self._table.setItem(row, 1, QTableWidgetItem(""))
        self._apply_path_item(row, "", "", "")
        self._table.blockSignals(False)
        self._table.setCurrentCell(row, 0)
        first_item = self._table.item(row, 0)
        if first_item is not None:
            self._table.editItem(first_item)
        self._sync_selection_buttons()
        self.templates_changed.emit()

    def _on_browse_ptxt(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        start = self._raw_ptxt_path(row)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pro Tools Template (.ptxt)",
            start,
            "Pro Tools Templates (*.ptxt *.ptx);;All Files (*)",
        )
        if not path:
            return
        self._table.blockSignals(True)
        g_item = self._table.item(row, 0)
        n_item = self._table.item(row, 1)
        group = g_item.text().strip() if g_item else ""
        name = n_item.text().strip() if n_item else ""
        if not name:
            name = os.path.splitext(os.path.basename(path))[0]
            self._table.setItem(row, 1, QTableWidgetItem(name))
        self._apply_path_item(row, group, name, path)
        self._table.blockSignals(False)
        self._sync_selection_buttons()
        self.templates_changed.emit()

    def _on_clear_path(self) -> None:
        row = self._selected_row()
        if row < 0 or not self._raw_ptxt_path(row):
            return
        self._table.blockSignals(True)
        g_item = self._table.item(row, 0)
        n_item = self._table.item(row, 1)
        group = g_item.text().strip() if g_item else ""
        name = n_item.text().strip() if n_item else ""
        self._apply_path_item(row, group, name, "")
        self._table.blockSignals(False)
        self._sync_selection_buttons()
        self.templates_changed.emit()

    def _on_remove(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        self._table.removeRow(row)
        self._sync_selection_buttons()
        self.templates_changed.emit()


def _build_ptsl_section_banner() -> tuple[QWidget, QLabel]:
    wrapper = QWidget()
    w_lay = QVBoxLayout(wrapper)
    w_lay.setContentsMargins(0, 4, 0, 2)
    w_lay.setSpacing(6)

    divider = QFrame()
    divider.setFrameShape(QFrame.Shape.HLine)
    divider.setStyleSheet(f"color: {COLORS['accent']};")
    w_lay.addWidget(divider)

    banner = QWidget()
    banner.setObjectName("ptslModeBanner")
    banner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    b_lay = QHBoxLayout(banner)
    b_lay.setContentsMargins(10, 6, 10, 6)
    label = QLabel("")
    label.setWordWrap(True)
    label.setStyleSheet("border: none; background: transparent;")
    b_lay.addWidget(label, 1)
    w_lay.addWidget(banner)
    wrapper._banner_frame = banner  # type: ignore[attr-defined]
    return wrapper, label


def _set_param_row_enabled(widget: QWidget, enabled: bool) -> None:
    widget.setEnabled(enabled)
    if isinstance(widget, PathPicker):
        return
    for companion in getattr(widget, "_param_row_widgets", ()):
        companion.setEnabled(enabled)


def install_protools_mode_wiring(
    page: QWidget,
    widgets: list[tuple[str, QWidget]],
    templates_widget: ProToolsTemplatesWidget,
) -> None:
    """Mount ProToolsTemplatesWidget + PTSL banner and bind Transfer Mode disabling."""
    layout = page.layout()
    if not isinstance(layout, QVBoxLayout):
        return
    by_key = dict(widgets)
    temp_widget = by_key.get("protools_temp_dir")
    insert_idx = layout.indexOf(temp_widget) if temp_widget is not None else -1
    if insert_idx < 0:
        insert_idx = min(3, layout.count())
    layout.insertWidget(insert_idx, templates_widget)
    banner_wrapper, banner_label = _build_ptsl_section_banner()
    layout.insertWidget(insert_idx + 1, banner_wrapper)
    banner_frame: QWidget = banner_wrapper._banner_frame  # type: ignore[attr-defined]

    mode_combo = by_key.get("protools_transfer_backend")
    ptsl_widgets = [by_key[k] for k in PTSL_ONLY_PARAM_KEYS if k in by_key]

    def apply_mode() -> None:
        backend = mode_combo.currentData() if isinstance(mode_combo, QComboBox) else "offline_ptx"
        is_offline = str(backend or "offline_ptx") == "offline_ptx"
        for w in ptsl_widgets:
            _set_param_row_enabled(w, not is_offline)
        if is_offline:
            banner_frame.setStyleSheet(
                "QWidget#ptslModeBanner { background-color: #2d2d2d; "
                "border: 1px solid #3a3a3a; border-left: 3px solid #6b6b6b; border-radius: 3px; }"
            )
            banner_label.setText(
                "PTSL Connection Settings (7 parameters) \u2014 Inactive in Offline .ptx Writer mode. "
                "No running Pro Tools instance, gRPC port, or temporary folder required."
            )
            banner_label.setStyleSheet(f"border: none; background: transparent; color: {COLORS['metadata']};")
        else:
            banner_frame.setStyleSheet(
                "QWidget#ptslModeBanner { background-color: #282f38; "
                f"border: 1px solid #3a3a3a; border-left: 3px solid {COLORS['information']}; border-radius: 3px; }}"
            )
            banner_label.setText(
                "PTSL Connection Settings (7 parameters) \u2014 Active in Live PTSL + .ptx Hybrid mode. "
                "Connects to a running Pro Tools instance via PTSL gRPC."
            )
            banner_label.setStyleSheet(f"border: none; background: transparent; color: {COLORS['text']};")

    if isinstance(mode_combo, QComboBox):
        mode_combo.currentIndexChanged.connect(lambda _idx: apply_mode())
    apply_mode()
