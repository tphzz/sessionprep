"""Phase 2 track table & Phase 3 setup table controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QMenu,
    QToolButton,
)

from sessionpreplib.utils import protools_sort_key

from ..detail.repair_options import REPAIR_NAMES, track_repair_options
from ..helpers import track_analysis_label
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import (
    COLORS,
    FILE_COLOR_ERROR,
    FILE_COLOR_OK,
    FILE_COLOR_SILENT,
    FILE_COLOR_SUSTAINED,
    FILE_COLOR_TRANSIENT,
)
from ..widgets import BatchComboBox, BatchToolButton, TableCellDoubleSpinBox
from . import group_utils
from .table_layout import TRACK_TABLE_RIGHT_MIN_WIDTH, apply_track_table_layout
from .table_widgets import TrackCol, _PAGE_TABS, _SortableItem, _make_analysis_cell

if TYPE_CHECKING:
    from PySide6.QtWidgets import QSplitter, QStackedWidget, QTabWidget
    from .table_widgets import _DraggableTrackTable, _SetupDragTable

_ANCHOR_LABELS = ["Default", "Max", "P99", "P95", "P90", "P85"]
_ANCHOR_TO_OVERRIDE: dict[str, str | None] = {
    "Default": None,
    "Max": "max",
    "P99": "p99",
    "P95": "p95",
    "P90": "p90",
    "P85": "p85",
}
_OVERRIDE_TO_LABEL: dict[str | None, str] = {v: k for k, v in _ANCHOR_TO_OVERRIDE.items()}


def _base_classification(cls_text: str | None) -> str:
    text = cls_text or "Unknown"
    if "Transient" in text:
        return "Transient"
    if text == "Skip":
        return "Skip"
    return "Sustained"


class TrackTableController(QObject):
    """Owns Phase 2 track table & Phase 3 setup table population, cell widgets, and batch edits."""

    track_selected = Signal(object)                    # TrackContext
    file_report_refresh_requested = Signal(object)     # TrackContext | None
    waveform_issues_refresh_requested = Signal()
    track_repairs_requested = Signal(object)           # TrackContext
    linked_group_levels_requested = Signal()
    setup_column_modes_requested = Signal()
    setup_splitter_fit_requested = Signal()
    setup_table_populating_changed = Signal(bool)
    prepare_stale_requested = Signal()
    repair_plan_requested = Signal(object, bool)       # (tracks, refresh_anchors)
    analyze_enabled_requested = Signal(bool)

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        track_table: _DraggableTrackTable,
        setup_table: _SetupDragTable | None = None,
        main_splitter: QSplitter | None = None,
        right_stack: QStackedWidget | None = None,
        detail_tabs: QTabWidget | None = None,
        on_create_group_combo: Callable[[int, Any], None] | None = None,
        on_is_analysis_running: Callable[[], bool] | None = None,
        on_is_repair_running: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._track_table = track_table
        self._setup_table = setup_table
        self._main_splitter = main_splitter
        self._right_stack = right_stack
        self._detail_tabs = detail_tabs
        self._on_create_group_combo = on_create_group_combo
        self._on_is_analysis_running = on_is_analysis_running
        self._on_is_repair_running = on_is_repair_running

        self.batch_filenames: set[str] = set()

    # ── Track selection & row lookup ─────────────────────────────────────

    @Slot(int, int)
    def on_row_clicked(self, row: int, _column: int) -> None:
        self.select_row(row)

    @Slot(int, int, int, int)
    def on_current_cell_changed(self, row: int, _col: int, prev_row: int, prev_col: int) -> None:
        tabs = self._detail_tabs
        if tabs is not None and tabs.currentWidget() is not tabs.widget(tabs.currentIndex()):
            table = self._track_table
            if 0 <= prev_row < table.rowCount() and 0 <= prev_col < table.columnCount():
                with QSignalBlocker(table):
                    table.setCurrentCell(prev_row, prev_col)
            return
        self.select_row(row)

    def select_row(self, row: int) -> None:
        session = self._state.session
        if not session or row < 0:
            return
        fname_item = self._track_table.item(row, TrackCol.FILE)
        if not fname_item:
            return
        track = next((t for t in session.tracks if t.filename == fname_item.text()), None)
        if track:
            self.track_selected.emit(track)

    def find_table_row(self, filename: str) -> int:
        for row in range(self._track_table.rowCount()):
            item = self._track_table.item(row, TrackCol.FILE)
            if item and item.text() == filename:
                return row
        return -1

    # ── Phase 2 table population ─────────────────────────────────────────

    def rebuild_track_table_rows(self, tracks: list[Any]) -> None:
        """Reset row skeleton for *tracks* before populating per-row widgets."""
        self._track_table.setSortingEnabled(False)
        self._track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            item = _SortableItem(track.filename, protools_sort_key(track.filename))
            item.setForeground(FILE_COLOR_OK if track.status == "OK" else FILE_COLOR_ERROR)
            self._track_table.setItem(row, TrackCol.FILE, item)
            for col in range(1, self._track_table.columnCount()):
                cell = _SortableItem("", "")
                cell.setForeground(QColor(COLORS["dim"]))
                self._track_table.setItem(row, col, cell)
        self._track_table.setSortingEnabled(True)

    def populate_table(self, session: Any) -> None:
        """Update the Phase 2 track table with analysis results."""
        self._track_table.setSortingEnabled(False)
        track_map = {t.filename: t for t in session.tracks}
        dets = session.detectors if hasattr(session, "detectors") else None
        for row in range(self._track_table.rowCount()):
            for col in (
                TrackCol.CLASSIFICATION,
                TrackCol.GAIN,
                TrackCol.ANCHOR,
                TrackCol.GROUP,
                TrackCol.PROCESSING,
            ):
                self._track_table.removeCellWidget(row, col)
            fname_item = self._track_table.item(row, TrackCol.FILE)
            track = track_map.get(fname_item.text()) if fname_item else None
            if track is not None:
                self.populate_track_row(row, track, dets)
        self._track_table.setSortingEnabled(True)
        self.auto_fit_track_table()

    def populate_track_row(self, row: int, track: Any, detectors: Any) -> None:
        ch_item = _SortableItem(str(track.channels), track.channels)
        ch_item.setForeground(QColor(COLORS["metadata"]))
        self._track_table.setItem(row, TrackCol.CHANNELS, ch_item)

        _plain, html, _color, sort_key = track_analysis_label(track, detectors)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._track_table.setItem(row, TrackCol.ANALYSIS, item)
        self._track_table.setCellWidget(row, TrackCol.ANALYSIS, lbl)

        pr = next(iter(track.processor_results.values()), None) if track.processor_results else None
        if track.status != "OK":
            self._set_static_cls_gain(row, "Error", "error", FILE_COLOR_ERROR, "", 0.0)
        elif pr and pr.classification == "Silent":
            self._set_static_cls_gain(row, "Silent", "silent", FILE_COLOR_SILENT, "0.0 dB", 0.0)
        elif pr:
            self._install_classification_and_gain_widgets(row, track, pr)
            self.create_anchor_combo(row, track)
        else:
            gain = track.manual_gain_db
            self._track_table.setItem(row, TrackCol.CLASSIFICATION, _SortableItem("", "zzz"))
            self._track_table.setItem(
                row,
                TrackCol.GAIN,
                _SortableItem(f"{gain:+.1f} dB" if gain is not None else "", gain or 0.0),
            )

        if track.status == "OK":
            if self._on_create_group_combo is not None:
                self._on_create_group_combo(row, track)
            self.create_processing_button(row, track)
            group_utils.apply_row_group_color(self._track_table, self._state, row, track.group)

    def _set_static_cls_gain(
        self, row: int, cls_text: str, cls_sort: str, cls_color: QColor, gain_text: str, gain_val: float
    ) -> None:
        cls_item = _SortableItem(cls_text, cls_sort)
        cls_item.setForeground(cls_color)
        self._track_table.setItem(row, TrackCol.CLASSIFICATION, cls_item)
        gain_item = _SortableItem(gain_text, gain_val)
        gain_item.setForeground(QColor(COLORS["metadata"]))
        self._track_table.setItem(row, TrackCol.GAIN, gain_item)

    def _install_classification_and_gain_widgets(self, row: int, track: Any, pr: Any) -> None:
        base_cls = _base_classification(pr.classification)
        self._track_table.setItem(row, TrackCol.CLASSIFICATION, _SortableItem(base_cls, base_cls.lower()))
        self.replace_classification_combo(row, base_cls, track.filename)

        gain_db = pr.gain_db
        self._track_table.setItem(row, TrackCol.GAIN, _SortableItem(f"{gain_db:+.1f}", gain_db))
        spin = TableCellDoubleSpinBox()
        spin.setRange(-60.0, 60.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setSuffix(" dB")
        spin.blockSignals(True)
        spin.setValue(gain_db)
        spin.blockSignals(False)
        spin.setProperty("track_filename", track.filename)
        spin.setEnabled(base_cls != "Skip")
        spin.valueChanged.connect(lambda value, s=spin: self.on_gain_changed(value, s))
        self._track_table.setCellWidget(row, TrackCol.GAIN, spin)

    # ── Phase 3 setup table population ───────────────────────────────────

    @Slot()
    def populate_setup_table(self) -> None:
        """Refresh the Session Setup track table from the transfer manifest."""
        session = self._state.session
        table = self._setup_table
        if not session or table is None:
            return
        self.setup_table_populating_changed.emit(True)
        table.setSortingEnabled(False)
        try:
            table.setRowCount(0)
            manifest = session.transfer_manifest
            if not manifest:
                return

            out_map = {t.filename: t for t in session.output_tracks}
            table.setRowCount(len(manifest))
            gcm = group_utils.group_color_map(self._state)
            gcm_rank = group_utils.group_rank_map(self._state)
            glm = group_utils.gain_linked_map(self._state)

            from sessionpreplib.daw_processors.protools import ProToolsDawProcessor

            assignments: dict[str, str] = {}
            if session.daw_state and self._state.active_daw_processor:
                dp_state = session.daw_state.get(self._state.active_daw_processor.id, {})
                assignments = dp_state.get("assignments", {})
            pt_gain_mode = "clip_gain_and_fader"
            if isinstance(self._state.active_daw_processor, ProToolsDawProcessor):
                pt_gain_mode = str(self._state.flat_config().get("protools_gain_mode", "clip_gain_and_fader"))

            for row, entry in enumerate(manifest):
                self._populate_setup_row(row, entry, out_map, assignments, gcm, gcm_rank, glm, pt_gain_mode)
        finally:
            table.setSortingEnabled(True)
            self.setup_table_populating_changed.emit(False)

        if not manifest:
            return
        sh = table.horizontalHeader()
        for col in range(table.columnCount()):
            sh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        table.resizeColumnsToContents()
        self.setup_column_modes_requested.emit()
        self.setup_splitter_fit_requested.emit()

    def _populate_setup_row(
        self,
        row: int,
        entry: Any,
        out_map: dict[str, Any],
        assignments: dict[str, str],
        gcm: dict[str, str],
        gcm_rank: dict[str, int],
        glm: dict[str, bool],
        pt_gain_mode: str = "clip_gain_and_fader",
    ) -> None:
        table = self._setup_table
        assert table is not None
        track = out_map.get(entry.output_filename)
        pr = next(iter(track.processor_results.values()), None) if track and track.processor_results else None

        tn_item = _SortableItem(entry.daw_track_name, protools_sort_key(entry.daw_track_name))
        tn_item.setForeground(QColor(COLORS["text"]))
        tn_item.setFlags(tn_item.flags() | Qt.ItemFlag.ItemIsEditable)
        tn_item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
        table.setItem(row, 0, tn_item)

        assigned = entry.entry_id in assignments
        chk_item = _SortableItem("✓" if assigned else "", int(not assigned))
        chk_item.setFlags(chk_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if assigned:
            chk_item.setForeground(QColor(COLORS["clean"]))
        table.setItem(row, 1, chk_item)

        fname_item = _SortableItem(entry.output_filename, protools_sort_key(entry.output_filename))
        fname_item.setForeground(FILE_COLOR_OK)
        fname_item.setFlags(fname_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        fname_item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
        table.setItem(row, 2, fname_item)

        channels = track.channels if track else 0
        ch_item = _SortableItem(str(channels), channels)
        ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        ch_item.setForeground(QColor(COLORS["metadata"]))
        table.setItem(row, 3, ch_item)

        clip_text, clip_val, fader_text, fader_val = self._setup_gain_display(track, pr, pt_gain_mode)
        cg_item = _SortableItem(clip_text, clip_val)
        cg_item.setFlags(cg_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        cg_item.setForeground(QColor(COLORS["metadata"] if "(baked)" in clip_text else COLORS["text"]))
        table.setItem(row, 4, cg_item)

        fg_item = _SortableItem(fader_text, fader_val)
        fg_item.setFlags(fg_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        fg_item.setForeground(QColor(COLORS["text"]))
        table.setItem(row, 5, fg_item)

        grp = entry.group
        grp_label = group_utils.group_display_name(self._state, grp, glm) if grp else ""
        grp_rank = gcm_rank.get(grp, len(gcm_rank)) if grp else len(gcm_rank)
        grp_item = _SortableItem(grp_label, grp_rank)
        grp_item.setFlags(grp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        grp_item.setForeground(QColor(COLORS["text"]))
        table.setItem(row, 6, grp_item)

        group_utils.apply_row_group_color(table, self._state, row, grp, gcm)

    @staticmethod
    def _fmt_signed_db(val: float) -> str:
        return "0.0 dB" if abs(val) < 0.05 else f"{val:+.1f} dB"

    def _setup_gain_display(self, track: Any, pr: Any, pt_gain_mode: str) -> tuple[str, float, str, float]:
        from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
        from sessionpreplib.daw_processors.protools_ptx import (
            GAIN_MODE_CLIP_UNITY,
            GAIN_MODE_FADER_ONLY,
            GAIN_MODE_NONE,
            resolve_protools_track_staging,
        )

        dp = self._state.active_daw_processor
        session = self._state.session
        clip_gain = pr.gain_db if pr else 0.0
        fader_gain = pr.data.get("fader_offset", 0.0) if pr else 0.0
        if isinstance(dp, ProToolsDawProcessor) and track is not None and session is not None:
            try:
                staging = resolve_protools_track_staging(session, track, gain_mode=pt_gain_mode)
            except ValueError:
                staging = None
            if staging is not None:
                if abs(staging.baked_gain_db) >= 0.05 and abs(staging.clip_gain_db) < 0.05:
                    clip_text = "0.0 dB (baked)"
                else:
                    clip_text = self._fmt_signed_db(staging.clip_gain_db)
                return clip_text, staging.clip_gain_db, self._fmt_signed_db(staging.fader_db), staging.fader_db
            if pt_gain_mode == GAIN_MODE_NONE:
                clip_gain, fader_gain = 0.0, 0.0
            elif pt_gain_mode == GAIN_MODE_CLIP_UNITY:
                fader_gain = 0.0
            elif pt_gain_mode == GAIN_MODE_FADER_ONLY:
                clip_gain, fader_gain = 0.0, 0.0
        return self._fmt_signed_db(clip_gain), clip_gain, self._fmt_signed_db(fader_gain), fader_gain

    # ── Classification & gain overrides ──────────────────────────────────

    @staticmethod
    def style_classification_combo(combo: QComboBox, cls_text: str) -> None:
        if cls_text == "Transient":
            color = FILE_COLOR_TRANSIENT.name()
        elif cls_text == "Sustained":
            color = FILE_COLOR_SUSTAINED.name()
        else:
            color = FILE_COLOR_SILENT.name()
        combo.setStyleSheet(f"QComboBox {{ color: {color}; font-weight: bold; }}")

    def replace_classification_combo(self, row: int, cls_text: str, fname: str) -> BatchComboBox:
        combo = BatchComboBox()
        combo.addItems(["Transient", "Sustained", "Skip"])
        combo.blockSignals(True)
        combo.setCurrentText(cls_text)
        combo.blockSignals(False)
        combo.setProperty("track_filename", fname)
        self.style_classification_combo(combo, cls_text)
        combo.textActivated.connect(lambda text, c=combo: self.on_classification_changed(text, c))
        self._track_table.setCellWidget(row, TrackCol.CLASSIFICATION, combo)
        return combo

    def on_classification_changed(self, text: str, combo: Any = None) -> None:
        if combo is None:
            combo = self.sender()
        session = self._state.session
        if not combo or not session:
            return
        fname = combo.property("track_filename")
        track = next((t for t in session.tracks if t.filename == fname), None) if fname else None
        if not track:
            return

        if getattr(combo, "batch_mode", False) or combo.property("_batch_mode"):
            combo.setProperty("_batch_mode", False)
            combo.batch_mode = False
            self.batch_apply_combo(
                combo, TrackCol.CLASSIFICATION, text, lambda t: setattr(t, "classification_override", text), run_detectors=False
            )
        else:
            if track.classification_override == text:
                return
            track.classification_override = text
            self.recalculate_processor(track)
            row = self.find_table_row(fname)
            if row >= 0:
                self.replace_classification_combo(row, text, fname)
            self.update_track_row(fname)
            self.refresh_file_tab(track)
        self.prepare_stale_requested.emit()

    def on_gain_changed(self, value: float, spin: Any = None) -> None:
        if spin is None:
            spin = self.sender()
        session = self._state.session
        if not spin or not session:
            return
        fname = spin.property("track_filename")
        track = next((t for t in session.tracks if t.filename == fname), None) if fname else None
        if not track:
            return

        track.manual_gain_db = float(value)
        pr = track.processor_results.get("bimodal_normalize")
        if pr:
            pr.gain_db = value
            pr.data["original_gain_db"] = value
        self.linked_group_levels_requested.emit()
        if self._on_is_repair_running is not None and self._on_is_repair_running():
            self.repair_plan_requested.emit([track], False)
        self.prepare_stale_requested.emit()

        row = self.find_table_row(fname)
        if row >= 0:
            gain_sort = self._track_table.item(row, TrackCol.GAIN)
            if isinstance(gain_sort, _SortableItem):
                gain_sort.setText(f"{value:+.1f}")
                gain_sort._sort_key = value
        if self._state.current_track and self._state.current_track.filename == fname:
            self.file_report_refresh_requested.emit(track)

    # ── RMS Anchor column ────────────────────────────────────────────────

    def create_anchor_combo(self, row: int, track: Any) -> None:
        self._track_table.setItem(row, TrackCol.ANCHOR, _SortableItem("Default", "default"))
        combo = BatchComboBox()
        combo.addItems(_ANCHOR_LABELS)
        combo.blockSignals(True)
        combo.setCurrentText(_OVERRIDE_TO_LABEL.get(track.rms_anchor_override, "Default"))
        combo.blockSignals(False)
        combo.setProperty("track_filename", track.filename)
        combo.setStyleSheet(f"QComboBox {{ color: {COLORS['text']}; }}")
        combo.textActivated.connect(lambda text, c=combo: self.on_rms_anchor_changed(text, c))
        self._track_table.setCellWidget(row, TrackCol.ANCHOR, combo)

    def on_rms_anchor_changed(self, text: str, combo: Any = None) -> None:
        if combo is None:
            combo = self.sender()
        session = self._state.session
        if not combo or not session:
            return
        fname = combo.property("track_filename")
        track = next((t for t in session.tracks if t.filename == fname), None) if fname else None
        if not track:
            return
        new_override = _ANCHOR_TO_OVERRIDE.get(text)

        if combo.property("_batch_mode"):
            combo.setProperty("_batch_mode", False)
            combo.batch_mode = False
            self.batch_apply_combo(
                combo, TrackCol.ANCHOR, text, lambda t: setattr(t, "rms_anchor_override", new_override), run_detectors=True
            )
        else:
            if track.rms_anchor_override == new_override:
                return
            track.rms_anchor_override = new_override
            self.repair_plan_requested.emit([track], True)
        self.prepare_stale_requested.emit()

    # ── Processing column ────────────────────────────────────────────────

    def create_processing_button(self, row: int, track: Any) -> None:
        if track.status != "OK":
            self._track_table.setItem(row, TrackCol.PROCESSING, _SortableItem("", "zzz"))
            return
        processors = self._state.session.processors if self._state.session else []
        btn: Any = BatchToolButton()
        btn.setProperty("track_filename", track.filename)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)
        for proc in processors:
            action = menu.addAction(proc.name)
            action.setCheckable(True)
            action.setChecked(proc.id not in track.processor_skip)
            action.setData(proc.id)
            action.toggled.connect(lambda checked, a=action: self.on_processing_toggled(checked, a))
        btn.repair_separator = menu.addSeparator() if processors else None
        btn.repair_action = menu.addAction("Track repairs…")
        btn.repair_action.triggered.connect(lambda _c=False, t=track: self.track_repairs_requested.emit(t))
        btn.setMenu(menu)
        menu.aboutToShow.connect(lambda: self.update_processing_button_label(btn, track, processors))
        self.update_processing_button_label(btn, track, processors)
        self._track_table.setItem(row, TrackCol.PROCESSING, _SortableItem("", len(track.processor_skip)))
        self._track_table.setCellWidget(row, TrackCol.PROCESSING, btn)

    def update_processing_button_label(self, btn: Any, track: Any, processors: list[Any]) -> None:
        available = track_repair_options(track, self._state.session).available
        btn.repair_action.setVisible(available)
        if btn.repair_separator is not None:
            btn.repair_separator.setVisible(available)
        btn.setEnabled(bool(processors) or available)
        if track.repair_recipe.active:
            btn.setText("Repairs")
            btn.setToolTip(
                "Before normalization: "
                + ", ".join(REPAIR_NAMES[key] for key in track.repair_recipe.applied_ids())
            )
            return
        if not processors:
            btn.setText("None")
            btn.setToolTip(
                "Open Track repairs to review corrective processing."
                if available
                else "No audio processors enabled."
            )
            return
        active = [p for p in processors if p.id not in track.processor_skip]
        is_default = all((p.id not in track.processor_skip) == p.default for p in processors)
        if is_default:
            default_names = [p.name for p in processors if p.default]
            btn.setText("Default")
            btn.setToolTip(
                "Default selection: " + ", ".join(default_names)
                if default_names
                else "Default: all processors deselected"
            )
        elif not active:
            btn.setText("None")
            btn.setToolTip("All processors skipped for this track")
        else:
            btn.setText(", ".join(p.shorthand if p.shorthand else p.name for p in active))
            btn.setToolTip("Active processors: " + ", ".join(p.name for p in active))

    def on_processing_toggled(self, checked: bool, action: Any = None) -> None:
        if action is None:
            action = self.sender()
        menu = action.parent() if action else None
        btn = menu.parent() if menu else None
        session = self._state.session
        fname = btn.property("track_filename") if btn else None
        if not btn or not fname or not session:
            return
        track = next((t for t in session.tracks if t.filename == fname), None)
        if not track:
            return
        proc_id = action.data()
        processors = session.processors

        if btn.property("_batch_mode"):
            btn.setProperty("_batch_mode", False)
            btn.batch_mode = False
            batch_keys = self._track_table.batch_selected_keys()
            track_map = {t.filename: t for t in session.tracks}
            for bfname in batch_keys:
                t = track_map.get(bfname)
                if not t or t.status != "OK":
                    continue
                t.processor_skip.discard(proc_id) if checked else t.processor_skip.add(proc_id)
                row = self.find_table_row(bfname)
                if row >= 0 and (b := self._track_table.cellWidget(row, TrackCol.PROCESSING)):
                    self.update_processing_button_label(b, t, processors)
            self._track_table.restore_selection(batch_keys)
        else:
            track.processor_skip.discard(proc_id) if checked else track.processor_skip.add(proc_id)
            self.update_processing_button_label(btn, track, processors)

        self.linked_group_levels_requested.emit()
        self.prepare_stale_requested.emit()

    # ── Batch edits, recalculation & row updates ─────────────────────────

    def batch_apply_combo(
        self,
        source_combo: Any,
        column: int,
        value: str,
        prepare_fn: Callable[[Any], None],
        run_detectors: bool = True,
    ) -> None:
        session = self._state.session
        if not session or (self._on_is_analysis_running is not None and self._on_is_analysis_running()):
            return
        track_map = {t.filename: t for t in session.tracks}
        batch_keys = self._track_table.batch_selected_keys()
        tracks_to_reanalyze: list[Any] = []
        self._track_table.setSortingEnabled(False)
        for fname in batch_keys:
            track = track_map.get(fname)
            if not track or track.status != "OK":
                continue
            prepare_fn(track)
            tracks_to_reanalyze.append(track)
            row = self.find_table_row(fname)
            if row >= 0 and isinstance(w := self._track_table.cellWidget(row, column), BatchComboBox):
                w.blockSignals(True)
                w.setCurrentText(value)
                w.blockSignals(False)
        if not tracks_to_reanalyze:
            self._track_table.setSortingEnabled(True)
            return

        self.batch_filenames = set(batch_keys)
        self._shell.progress_start("Re-analyzing\u2026")
        self._shell.progress_value(0, len(tracks_to_reanalyze))
        self.analyze_enabled_requested.emit(False)
        self.repair_plan_requested.emit(tracks_to_reanalyze, run_detectors)

    @Slot()
    def on_batch_done(self) -> None:
        self.analyze_enabled_requested.emit(True)
        if self._right_stack is not None:
            self._right_stack.setCurrentIndex(_PAGE_TABS)
        session = self._state.session
        failures = [
            t.filename
            for t in (session.tracks if session else [])
            if t.filename in self.batch_filenames and t._cache.get("repair_error")
        ]
        if failures:
            self._shell.progress_fail("Re-analysis failed for " + ", ".join(failures))
        else:
            self._shell.progress_finish("Re-analysis complete")

        self._track_table.setSortingEnabled(True)
        self._track_table.restore_selection(self.batch_filenames)
        self.batch_filenames = set()
        self.populate_setup_table()
        if self._state.current_track:
            self.refresh_file_tab(self._state.current_track)

    def recalculate_processor(self, track: Any) -> None:
        session = self._state.session
        if not session or not session.processors:
            return
        if (
            track.repair_recipe.active
            or track.repair_overrides
            or track.manual_gain_db is not None
            or (self._on_is_repair_running is not None and self._on_is_repair_running())
        ):
            self.repair_plan_requested.emit([track], False)
            return
        for proc in session.processors:
            result = proc.process(track)
            result.data["original_gain_db"] = result.gain_db
            track.processor_results[proc.id] = result
        self.linked_group_levels_requested.emit()

    def update_track_row(self, filename: str) -> None:
        session = self._state.session
        if not session:
            return
        track = next((t for t in session.tracks if t.filename == filename), None)
        row = self.find_table_row(filename) if track else -1
        if not track or row < 0:
            return

        _plain, html, _color, sort_key = track_analysis_label(track, session.detectors)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._track_table.setItem(row, TrackCol.ANALYSIS, item)
        self._track_table.setCellWidget(row, TrackCol.ANALYSIS, lbl)

        pr = next(iter(track.processor_results.values()), None)
        new_gain = pr.gain_db if pr else (track.manual_gain_db or 0.0)
        base_cls = (
            track.classification_override
            if track.classification_override
            else (_base_classification(pr.classification) if pr else None)
        )

        spin = self._track_table.cellWidget(row, TrackCol.GAIN)
        if isinstance(spin, QDoubleSpinBox):
            spin.blockSignals(True)
            spin.setValue(new_gain)
            if base_cls is not None:
                spin.setEnabled(base_cls != "Skip")
            spin.blockSignals(False)
        gain_sort = self._track_table.item(row, TrackCol.GAIN)
        if isinstance(gain_sort, _SortableItem):
            gain_sort.setText(
                f"{new_gain:+.1f}"
                if isinstance(spin, QDoubleSpinBox)
                else (f"{new_gain:+.1f} dB" if pr or track.manual_gain_db is not None else "")
            )
            gain_sort._sort_key = new_gain

        if base_cls is not None:
            self.replace_classification_combo(row, base_cls, track.filename)
            sort_item = self._track_table.item(row, TrackCol.CLASSIFICATION)
            if isinstance(sort_item, _SortableItem):
                sort_item.setText(base_cls)
                sort_item._sort_key = base_cls.lower()

        group_utils.apply_row_group_color(self._track_table, self._state, row, track.group)
        self.populate_setup_table()

    @Slot(object)
    def refresh_file_tab(self, track: Any) -> None:
        if not self._state.current_track or self._state.current_track.filename != track.filename:
            return
        self.file_report_refresh_requested.emit(track)
        self.waveform_issues_refresh_requested.emit()

    @Slot()
    def refresh_analysis_column(self) -> None:
        session = self._state.session
        if not session:
            return
        track_map = {t.filename: t for t in session.tracks}
        dets = session.detectors if hasattr(session, "detectors") else None
        self._track_table.setSortingEnabled(False)
        for row in range(self._track_table.rowCount()):
            fname_item = self._track_table.item(row, TrackCol.FILE)
            track = track_map.get(fname_item.text()) if fname_item else None
            if not track:
                continue
            _plain, html, _color, sort_key = track_analysis_label(track, dets)
            lbl, item = _make_analysis_cell(html, sort_key)
            self._track_table.setItem(row, TrackCol.ANALYSIS, item)
            self._track_table.setCellWidget(row, TrackCol.ANALYSIS, lbl)
        self._track_table.setSortingEnabled(True)

    @Slot()
    def refresh_processing_column(self) -> None:
        session = self._state.session
        if not session:
            return
        for row in range(self._track_table.rowCount()):
            fname_item = self._track_table.item(row, TrackCol.FILE)
            if not fname_item:
                continue
            track = next((t for t in session.tracks if t.filename == fname_item.text()), None)
            if not track or track.status != "OK":
                continue
            self._track_table.removeCellWidget(row, TrackCol.PROCESSING)
            self.create_processing_button(row, track)

    @Slot()
    def auto_fit_track_table(self) -> None:
        layout = apply_track_table_layout(self._track_table)
        left_min = self._track_table.minimumWidth()
        needed = max(layout.table_width, left_min)
        if self._main_splitter is None:
            return
        splitter_total = self._main_splitter.width() - self._main_splitter.handleWidth()
        if splitter_total > 0:
            right_min = TRACK_TABLE_RIGHT_MIN_WIDTH
            if self._right_stack is not None:
                right_min = max(right_min, self._right_stack.minimumWidth())
            max_left = max(splitter_total - right_min, left_min)
            left_w = min(needed, max_left)
            right_w = max(splitter_total - left_w, right_min)
            self._main_splitter.setSizes([left_w, right_w])
        self._track_table.fit_file_column()
