"""Phase 2 Groups controller: group editor tab, colors, auto-group, linked levels."""

from __future__ import annotations

from collections.abc import Callable
import copy
import os
import re
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from sessionpreplib.gain_staging import apply_group_gains

from ..settings import build_defaults
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import COLORS, PT_DEFAULT_COLORS
from ..widgets import BatchComboBox, ColorPickerButton, TableCellComboBox
from . import group_utils
from .groups_tab import GroupsTabWidget
from .table_widgets import TrackCol, _SortableItem, _TAB_GROUPS

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTabWidget
    from .table_widgets import _DraggableTrackTable

_GROUP_DIRTY_BG = QColor("#3a3121")
_GROUP_NONE_LABEL = group_utils.GROUP_NONE_LABEL


def match_track_to_group(filename: str, groups: list[dict[str, Any]]) -> str | None:
    """Match *filename* against ordered group rules, picking the longest match."""
    stem = os.path.splitext(filename)[0].lower()
    matched_group: str | None = None
    best_len = 0
    for group in groups:
        pattern = group.get("match_pattern", "").strip()
        if not pattern:
            continue
        method = group.get("match_method", "contains")
        if method == "regex":
            try:
                match = re.search(pattern, stem, re.IGNORECASE)
            except re.error:
                continue
            if match and (match.end() - match.start()) > best_len:
                best_len = match.end() - match.start()
                matched_group = group["name"]
        else:
            for tok in (t.strip().lower() for t in pattern.split(",") if t.strip()):
                if tok in stem and len(tok) > best_len:
                    best_len = len(tok)
                    matched_group = group["name"]
    return matched_group


class GroupsController(QObject):
    """Owns the session-local Groups tab, track group combos, auto-group, and linked gains."""

    groups_modified = Signal()
    setup_table_refresh_requested = Signal()
    file_tab_refresh_requested = Signal(object)
    prepare_stale_requested = Signal()
    detail_tab_dirty_changed = Signal(int, str, bool)
    auto_fit_requested = Signal()

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        track_table: _DraggableTrackTable,
        detail_tabs: QTabWidget | None = None,
        on_find_table_row: Callable[[str], int] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._track_table = track_table
        self._detail_tabs = detail_tabs
        self._on_find_table_row = on_find_table_row

        self.page = GroupsTabWidget()
        self.populate_group_preset_combo()
        self._connect_page_signals()

    def _connect_page_signals(self) -> None:
        p = self.page
        p.group_preset_combo.currentTextChanged.connect(self.on_group_preset_changed)
        p.revert_btn.clicked.connect(self.merge_groups_from_preset)
        p.table.verticalHeader().sectionMoved.connect(self._on_row_moved)
        p.table.cellChanged.connect(self._on_cell_changed)
        p.add_btn.clicked.connect(self.add_group_row)
        p.remove_btn.clicked.connect(self.remove_selected_group_row)
        p.sort_az_btn.clicked.connect(self.sort_groups_az)

    def clear_workspace(self) -> None:
        self.page.table.setRowCount(0)

    # ── Preset combo & colors ────────────────────────────────────────────

    def populate_group_preset_combo(self) -> None:
        combo = self.page.group_preset_combo
        presets = self._state.config.get(
            "group_presets", build_defaults().get("group_presets", {})
        )
        active = self._state.active_session_preset
        combo.blockSignals(True)
        combo.clear()
        for name in presets:
            combo.addItem(name)
        idx = combo.findText(active)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def color_names_from_config(self) -> list[str]:
        return group_utils.color_names_from_config(self._state.config)

    def color_argb_by_name(self, name: str) -> str | None:
        return group_utils.color_argb_by_name(self._state.config, name)

    @staticmethod
    def color_swatch_icon(argb: str, size: int = 16) -> QIcon:
        return group_utils.color_swatch_icon(argb, size)

    def tint_group_color(
        self, group_name: str | None, gcm: dict[str, str] | None = None
    ) -> QColor | None:
        return group_utils.tint_group_color(self._state, group_name, gcm)

    def apply_row_group_color(
        self,
        row: int,
        group_name: str | None,
        gcm: dict[str, str] | None = None,
        table: Any = None,
    ) -> None:
        target = self._track_table if table is None else table
        group_utils.apply_row_group_color(target, self._state, row, group_name, gcm)

    def normalize_session_groups(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return group_utils.normalize_session_groups(self._state.config, groups)

    # ── Group lookup maps ────────────────────────────────────────────────

    def group_combo_items(self) -> list[str]:
        return group_utils.group_combo_items(self._state)

    def gain_linked_map(self) -> dict[str, bool]:
        return group_utils.gain_linked_map(self._state)

    def group_display_name(self, name: str, glm: dict[str, bool] | None = None) -> str:
        return group_utils.group_display_name(self._state, name, glm)

    def group_rank_map(self) -> dict[str, int]:
        return group_utils.group_rank_map(self._state)

    def group_color_map(self) -> dict[str, str]:
        return group_utils.group_color_map(self._state)

    # ── Groups tab row operations ────────────────────────────────────────

    def _set_tab_row(
        self,
        row: int,
        name: str,
        color: str,
        gain_linked: bool,
        daw_target: str = "",
        match_method: str = "contains",
        match_pattern: str = "",
    ) -> None:
        table = self.page.table
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, name)
        table.setItem(row, 0, name_item)

        colors = self._state.config.get("colors", PT_DEFAULT_COLORS)
        color_picker = ColorPickerButton(colors, table)
        color_picker.setCurrentColor(color)
        color_picker.colorChanged.connect(lambda *_: self.sync_session_groups())
        table.setCellWidget(row, 1, color_picker)

        chk = QCheckBox()
        chk.setChecked(gain_linked)
        chk.toggled.connect(lambda *_: self.sync_session_groups())
        chk_container = QWidget()
        chk_layout = QHBoxLayout(chk_container)
        chk_layout.setContentsMargins(0, 0, 0, 0)
        chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chk_layout.addWidget(chk)
        table.setCellWidget(row, 2, chk_container)

        table.setItem(row, 3, QTableWidgetItem(daw_target))

        match_combo = TableCellComboBox()
        match_combo.addItems(["contains", "regex"])
        mi = match_combo.findText(match_method)
        if mi >= 0:
            match_combo.setCurrentIndex(mi)
        match_combo.currentTextChanged.connect(
            lambda _t, combo=match_combo: self._validate_pattern_for_combo(combo)
        )
        match_combo.currentTextChanged.connect(lambda *_: self.sync_session_groups())
        table.setCellWidget(row, 4, match_combo)

        table.setItem(row, 5, QTableWidgetItem(match_pattern))
        self._validate_pattern(row)

    def populate_groups_tab(self) -> None:
        self._state.session_groups = self.normalize_session_groups(self._state.session_groups)
        table = self.page.table
        table.blockSignals(True)
        table.setRowCount(0)
        table.setRowCount(len(self._state.session_groups))
        for row, g in enumerate(self._state.session_groups):
            self._set_tab_row(
                row,
                g["name"],
                g.get("color", ""),
                g.get("gain_linked", False),
                g.get("daw_target", ""),
                g.get("match_method", "contains"),
                g.get("match_pattern", ""),
            )
        table.blockSignals(False)
        self.refresh_dirty_indicators()

    def read_session_groups(self) -> list[dict[str, Any]]:
        return _read_groups_table(self.page.table, range(self.page.table.rowCount()))

    @staticmethod
    def _group_names_in_table(table: QTableWidget, exclude_row: int = -1) -> set[str]:
        names: set[str] = set()
        for r in range(table.rowCount()):
            if r == exclude_row:
                continue
            item = table.item(r, 0)
            if item and (text := item.text().strip()):
                names.add(text)
        return names

    def _unique_group_name(self, base: str = "New Group") -> str:
        existing = self._group_names_in_table(self.page.table)
        if base not in existing:
            return base
        n = 2
        while f"{base} {n}" in existing:
            n += 1
        return f"{base} {n}"

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col == 3:
            self.sync_session_groups()
            return
        if col == 5:
            self._validate_pattern(row)
            self.sync_session_groups()
            return
        if col != 0:
            return
        table = self.page.table
        item = table.item(row, 0)
        if not item:
            return
        name = item.text().strip()
        if name in self._group_names_in_table(table, exclude_row=row):
            table.blockSignals(True)
            item.setText(self._unique_group_name(name))
            table.blockSignals(False)
        old_name = item.data(Qt.ItemDataRole.UserRole)
        new_name = item.text().strip()
        if old_name != new_name:
            table.blockSignals(True)
            item.setData(Qt.ItemDataRole.UserRole, new_name)
            table.blockSignals(False)
            session = self._state.session
            if old_name and session:
                for members in (
                    session.tracks,
                    session.output_tracks,
                    session.transfer_manifest,
                    session.base_transfer_manifest,
                ):
                    for member in members:
                        if member.group == old_name:
                            member.group = new_name or None
        self.sync_session_groups()

    def _validate_pattern_for_combo(self, combo: QWidget) -> None:
        table = self.page.table
        for row in range(table.rowCount()):
            if table.cellWidget(row, 4) is combo:
                self._validate_pattern(row)
                return

    def _validate_pattern(self, row: int) -> None:
        table = self.page.table
        match_combo: Any = table.cellWidget(row, 4)
        pattern_item = table.item(row, 5)
        if not pattern_item:
            return
        method = match_combo.currentText() if match_combo else "contains"
        pattern = pattern_item.text().strip()
        if method == "regex" and pattern:
            try:
                re.compile(pattern)
                pattern_item.setForeground(QColor("#4ec94e"))
                pattern_item.setToolTip("")
            except re.error as exc:
                pattern_item.setForeground(QColor("#e05050"))
                pattern_item.setToolTip(f"Invalid regex: {exc}")
        else:
            pattern_item.setForeground(QColor("#cccccc"))
            pattern_item.setToolTip("")

    def sync_session_groups(self) -> None:
        self._state.session_groups = self.read_session_groups()
        self.refresh_group_combos()
        self.refresh_dirty_indicators()
        self.groups_modified.emit()

    @Slot()
    def add_group_row(self) -> None:
        table = self.page.table
        row = table.rowCount()
        table.insertRow(row)
        color_names = self.color_names_from_config()
        default_color = color_names[0] if color_names else ""
        self._set_tab_row(row, self._unique_group_name(), default_color, False)
        table.scrollToBottom()
        item = table.item(row, 0)
        if item is not None:
            table.editItem(item)
        self.sync_session_groups()

    @Slot()
    def remove_selected_group_row(self) -> None:
        row = self.page.table.currentRow()
        if row >= 0:
            self.page.table.removeRow(row)
            self.sync_session_groups()

    def _on_row_moved(self, _logical: int, _old_visual: int, _new_visual: int) -> None:
        table = self.page.table
        vh = table.verticalHeader()
        n = table.rowCount()
        visual_to_logical = sorted(range(n), key=vh.visualIndex)
        ordered = _read_groups_table(table, visual_to_logical)

        vh.blockSignals(True)
        table.blockSignals(True)
        for i in range(n):
            vh.moveSection(vh.visualIndex(i), i)
        table.setRowCount(0)
        table.setRowCount(len(ordered))
        for row, entry in enumerate(ordered):
            self._set_tab_row(
                row,
                entry["name"],
                entry["color"],
                entry["gain_linked"],
                entry.get("daw_target", ""),
                entry.get("match_method", "contains"),
                entry.get("match_pattern", ""),
            )
        table.blockSignals(False)
        vh.blockSignals(False)
        self._state.session_groups = ordered
        self.refresh_group_combos()
        self.refresh_dirty_indicators()

    @Slot()
    def sort_groups_az(self) -> None:
        groups = self.read_session_groups()
        groups.sort(key=lambda g: g["name"].lower())
        self._state.session_groups = groups
        self.populate_groups_tab()
        self.refresh_group_combos()
        self.refresh_dirty_indicators()

    @Slot()
    def merge_groups_from_preset(self) -> None:
        presets = self._state.config.get(
            "group_presets", build_defaults().get("group_presets", {})
        )
        preset = presets.get(self._state.active_session_preset, presets.get("Default", []))
        new_groups = self.normalize_session_groups(copy.deepcopy(preset))
        new_names = {g["name"].strip().lower() for g in new_groups}
        if self._state.session:
            for track in self._state.session.tracks:
                if track.group is not None and track.group.strip().lower() not in new_names:
                    track.group = None
        self._state.session_groups = new_groups
        self.populate_groups_tab()
        self.refresh_group_combos()
        self.refresh_dirty_indicators()

    @Slot(str)
    def on_group_preset_changed(self, preset_name: str) -> None:
        presets = self._state.config.get(
            "group_presets", build_defaults().get("group_presets", {})
        )
        if preset_name not in presets:
            return
        self._state.active_session_preset = preset_name
        self.merge_groups_from_preset()

    # ── Dirty indicators ─────────────────────────────────────────────────

    def selected_group_preset_groups(self) -> list[dict[str, Any]]:
        presets = self._state.config.get(
            "group_presets", build_defaults().get("group_presets", {})
        )
        preset = presets.get(self._state.active_session_preset, presets.get("Default", []))
        return self.normalize_session_groups(preset)

    def groups_dirty_cells(self) -> dict[int, set[int]]:
        if not self._state.session:
            return {}
        current = self.normalize_session_groups(self.read_session_groups())
        preset = self.selected_group_preset_groups()
        dirty: dict[int, set[int]] = {}
        preset_by_name = {group["name"]: group for group in preset}
        keys_by_col = {
            0: "name",
            1: "color",
            2: "gain_linked",
            3: "daw_target",
            4: "match_method",
            5: "match_pattern",
        }
        for row, group in enumerate(current):
            if row < len(preset) and group == preset[row]:
                continue
            baseline = preset_by_name.get(group["name"])
            if baseline is None:
                dirty[row] = set(keys_by_col)
                continue
            changed = {col for col, key in keys_by_col.items() if group.get(key) != baseline.get(key)}
            if changed:
                dirty[row] = changed
        return dirty

    def groups_tab_is_dirty(self) -> bool:
        if not self._state.session:
            return False
        if (
            self._detail_tabs is not None
            and not self._detail_tabs.isTabEnabled(_TAB_GROUPS)
            and not self._state.session_groups
        ):
            return False
        current = self.normalize_session_groups(self.read_session_groups())
        return current != self.selected_group_preset_groups()

    def refresh_dirty_indicators(self) -> None:
        dirty_cells = self.groups_dirty_cells()
        table = self.page.table
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                for col in range(table.columnCount()):
                    self._set_cell_dirty(row, col, col in dirty_cells.get(row, set()))
        finally:
            table.blockSignals(False)
        self.detail_tab_dirty_changed.emit(_TAB_GROUPS, "Groups", self.groups_tab_is_dirty())

    def _set_cell_dirty(self, row: int, col: int, dirty: bool) -> None:
        table = self.page.table
        item = table.item(row, col)
        brush = QBrush(_GROUP_DIRTY_BG) if dirty else QBrush()
        if item is not None:
            item.setBackground(brush)
        widget = table.cellWidget(row, col)
        if widget is None:
            return
        widget.setProperty("sessionDirty", dirty)
        if isinstance(widget, ColorPickerButton):
            widget.setDirtyIndicator(dirty)
        else:
            widget.setStyleSheet(f"background-color: {_GROUP_DIRTY_BG.name()};" if dirty else "")
        widget.setToolTip("Changed from selected preset" if dirty else "")

    # ── Auto-group ───────────────────────────────────────────────────────

    @Slot()
    def auto_group(self) -> None:
        session = self._state.session
        if not session:
            return
        ok_tracks = [t for t in session.tracks if t.status == "OK"]
        if not ok_tracks:
            return
        reply = QMessageBox.question(
            self._shell.dialog_parent(),
            "Auto-Group",
            f"Auto-Group will reassign all {len(ok_tracks)} tracks based on matching rules.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        assigned = self._apply_auto_group_matches(ok_tracks)
        self.auto_fit_group_column()
        self.apply_linked_group_levels()
        self._shell.show_status(f"Auto-Group: assigned {assigned} of {len(ok_tracks)} tracks")

    def _apply_auto_group_matches(self, ok_tracks: list[Any]) -> int:
        glm = self.gain_linked_map()
        gcm = self.group_color_map()
        grm = self.group_rank_map()
        assigned = 0
        self._track_table.setSortingEnabled(False)
        for track in ok_tracks:
            matched = match_track_to_group(track.filename, self._state.session_groups)
            track.group = matched
            if matched:
                assigned += 1
            row = self._on_find_table_row(track.filename) if self._on_find_table_row else -1
            if row >= 0:
                self._update_track_group_cell(row, matched, glm, grm, gcm)
        self._track_table.setSortingEnabled(True)
        return assigned

    def _update_track_group_cell(
        self,
        row: int,
        group_name: str | None,
        glm: dict[str, bool],
        grm: dict[str, int],
        gcm: dict[str, str],
    ) -> None:
        w = self._track_table.cellWidget(row, TrackCol.GROUP)
        if isinstance(w, BatchComboBox):
            w.blockSignals(True)
            _select_combo_group(w, group_name)
            w.blockSignals(False)
        display = self.group_display_name(group_name, glm) if group_name else _GROUP_NONE_LABEL
        rank = grm.get(group_name, len(grm)) if group_name else len(grm)
        sort_item = self._track_table.item(row, TrackCol.GROUP)
        if isinstance(sort_item, _SortableItem):
            sort_item.setText(display)
            sort_item._sort_key = rank
        self.apply_row_group_color(row, group_name, gcm)

    # ── Track table Group column (col 6) ─────────────────────────────────

    def _populate_group_combo_widget(
        self, combo: BatchComboBox, glm: dict[str, bool], gcm: dict[str, str]
    ) -> None:
        combo.setIconSize(QSize(16, 16))
        combo.addItem(_GROUP_NONE_LABEL)
        combo.setItemData(0, None, Qt.ItemDataRole.UserRole)
        for i, gname in enumerate(g["name"] for g in self._state.session_groups):
            disp = self.group_display_name(gname, glm)
            argb = gcm.get(gname)
            if argb:
                combo.addItem(self.color_swatch_icon(argb), disp)
            else:
                combo.addItem(disp)
            combo.setItemData(i + 1, gname, Qt.ItemDataRole.UserRole)

    def create_group_combo(self, row: int, track: Any) -> None:
        glm = self.gain_linked_map()
        display = self.group_display_name(track.group, glm) if track.group else _GROUP_NONE_LABEL
        grm = self.group_rank_map()
        rank = grm.get(track.group, len(grm)) if track.group else len(grm)
        self._track_table.setItem(row, TrackCol.GROUP, _SortableItem(display, rank))

        combo = BatchComboBox()
        self._populate_group_combo_widget(combo, glm, self.group_color_map())
        combo.blockSignals(True)
        _select_combo_group(combo, track.group)
        combo.blockSignals(False)
        combo.setProperty("track_filename", track.filename)
        combo.setStyleSheet(f"QComboBox {{ color: {COLORS['text']}; }}")
        combo.textActivated.connect(lambda text, c=combo: self.on_group_changed(text, c))
        self._track_table.setCellWidget(row, TrackCol.GROUP, combo)

    def on_group_changed(self, text: str, combo: Any = None) -> None:
        if combo is None:
            combo = self.sender()
        session = self._state.session
        if not combo or not session:
            return
        fname = combo.property("track_filename")
        track = next((t for t in session.tracks if t.filename == fname), None) if fname else None
        if not track:
            return
        new_group = combo.currentData(Qt.ItemDataRole.UserRole)

        if getattr(combo, "batch_mode", False) or combo.property("_batch_mode"):
            combo.setProperty("_batch_mode", False)
            combo.batch_mode = False
            self._apply_batch_group_change(track, new_group, text)
        else:
            if track.group == new_group:
                return
            track.group = new_group
            grm = self.group_rank_map()
            rank = grm.get(new_group, len(grm)) if new_group else len(grm)
            row = self._on_find_table_row(fname) if self._on_find_table_row else -1
            if row >= 0:
                sort_item = self._track_table.item(row, TrackCol.GROUP)
                if isinstance(sort_item, _SortableItem):
                    sort_item.setText(text)
                    sort_item._sort_key = rank
                self.apply_row_group_color(row, new_group)
        self.auto_fit_group_column()
        self.apply_linked_group_levels()

    def _apply_batch_group_change(self, track: Any, new_group: str | None, display: str) -> None:
        session = self._state.session
        assert session is not None
        track.group = new_group
        batch_keys = self._track_table.batch_selected_keys()
        track_map = {t.filename: t for t in session.tracks}
        gcm = self.group_color_map()
        grm = self.group_rank_map()
        rank = grm.get(new_group, len(grm)) if new_group else len(grm)
        self._track_table.setSortingEnabled(False)
        for bfname in batch_keys:
            bt = track_map.get(bfname)
            if not bt or bt.status != "OK":
                continue
            bt.group = new_group
            row = self._on_find_table_row(bfname) if self._on_find_table_row else -1
            if row >= 0:
                w = self._track_table.cellWidget(row, TrackCol.GROUP)
                if isinstance(w, BatchComboBox):
                    w.blockSignals(True)
                    _select_combo_group(w, new_group)
                    w.blockSignals(False)
                sort_item = self._track_table.item(row, TrackCol.GROUP)
                if isinstance(sort_item, _SortableItem):
                    sort_item.setText(display)
                    sort_item._sort_key = rank
                self.apply_row_group_color(row, new_group, gcm)
        self._track_table.setSortingEnabled(True)
        self._track_table.restore_selection(batch_keys)

    def refresh_group_combos(self) -> None:
        gcm = self.group_color_map()
        grm = self.group_rank_map()
        glm = self.gain_linked_map()
        tracks = {t.filename: t for t in self._state.session.tracks} if self._state.session else {}
        sorting = self._track_table.isSortingEnabled()
        self._track_table.setSortingEnabled(False)
        for row in range(self._track_table.rowCount()):
            w = self._track_table.cellWidget(row, TrackCol.GROUP)
            if not isinstance(w, BatchComboBox):
                continue
            track = tracks.get(w.property("track_filename"))
            old_group = track.group if track else None
            w.blockSignals(True)
            w.clear()
            self._populate_group_combo_widget(w, glm, gcm)
            if not _select_combo_group(w, old_group):
                w.setCurrentIndex(0)
                if track:
                    track.group = None
            w.blockSignals(False)
            gname = w.currentData(Qt.ItemDataRole.UserRole)
            sort_item = self._track_table.item(row, TrackCol.GROUP)
            if isinstance(sort_item, _SortableItem):
                sort_item._sort_key = grm.get(gname, len(grm)) if gname else len(grm)
                sort_item.setText(w.currentText())
            self.apply_row_group_color(row, gname, gcm)
        self._track_table.setSortingEnabled(sorting)
        self.auto_fit_group_column()
        self.apply_linked_group_levels()

    def auto_fit_group_column(self) -> None:
        self.auto_fit_requested.emit()

    # ── Linked group levels ──────────────────────────────────────────────

    @Slot()
    def apply_linked_group_levels(self) -> None:
        session = self._state.session
        if not session:
            return
        if session.processors:
            previous = [
                (pr, pr.gain_db, pr.data.get("fader_offset", 0.0))
                for t in session.tracks
                for pr in t.processor_results.values()
            ]
            apply_group_gains(
                session,
                linked_groups={name for name, linked in self.gain_linked_map().items() if linked},
                headroom_db=session.config.get("fader_headroom_db", 8.0),
                anchor=session.config.get("anchor"),
            )
            if any(
                pr.gain_db != gain or pr.data.get("fader_offset", 0.0) != fader
                for pr, gain, fader in previous
            ):
                self.prepare_stale_requested.emit()

            self._track_table.setSortingEnabled(False)
            for row in range(self._track_table.rowCount()):
                fname_item = self._track_table.item(row, TrackCol.FILE)
                if not fname_item:
                    continue
                track = next((t for t in session.tracks if t.filename == fname_item.text()), None)
                if not track or track.status != "OK":
                    continue
                pr = next(iter(track.processor_results.values()), None)
                if not pr:
                    continue
                spin = self._track_table.cellWidget(row, TrackCol.GAIN)
                if isinstance(spin, QDoubleSpinBox):
                    spin.blockSignals(True)
                    spin.setValue(pr.gain_db)
                    spin.blockSignals(False)
                gain_sort = self._track_table.item(row, TrackCol.GAIN)
                if isinstance(gain_sort, _SortableItem):
                    gain_sort.setText(f"{pr.gain_db:+.1f}")
                    gain_sort._sort_key = pr.gain_db
            self._track_table.setSortingEnabled(True)
        self.setup_table_refresh_requested.emit()
        if self._state.current_track and self._state.current_track.status == "OK":
            self.file_tab_refresh_requested.emit(self._state.current_track)


def _select_combo_group(combo: BatchComboBox, group_name: str | None) -> bool:
    if group_name is None:
        combo.setCurrentIndex(0)
        return True
    for idx in range(combo.count()):
        if combo.itemData(idx, Qt.ItemDataRole.UserRole) == group_name:
            combo.setCurrentIndex(idx)
            return True
    return False


def _read_groups_table(table: QTableWidget, row_indices: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for row in row_indices:
        name_item = table.item(row, 0)
        if not name_item or not (name := name_item.text().strip()):
            continue
        color_picker: Any = table.cellWidget(row, 1)
        color = color_picker.currentColor() if color_picker else ""
        chk_container = table.cellWidget(row, 2)
        chk = chk_container.findChild(QCheckBox) if chk_container else None
        daw_item = table.item(row, 3)
        match_combo: Any = table.cellWidget(row, 4)
        pattern_item = table.item(row, 5)
        groups.append({
            "name": name,
            "color": color,
            "gain_linked": chk.isChecked() if chk else False,
            "daw_target": daw_item.text().strip() if daw_item else "",
            "match_method": match_combo.currentText() if match_combo else "contains",
            "match_pattern": pattern_item.text().strip() if pattern_item else "",
        })
    return groups
