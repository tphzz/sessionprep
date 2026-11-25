"""Phase 3 Signal Flow inspector panel and bus-fader override controls."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.audio import dbfs_offset

from ..controls import FormActionButton
from ..icons import icon, icon_path
from ..prefs.param_form import ConfigResetButton
from ..theme import COLORS, FILE_COLOR_SUSTAINED, FILE_COLOR_TRANSIENT

_STACK_EMPTY = 0
_STACK_BUSY = 1
_STACK_ERROR = 2
_STACK_TABLE = 3

_AMBER_BG = "#2a251b"
_AMBER_BORDER = "#5c4a26"
_INFO_BG = "#1c2634"
_ERR_BG = "#2e1a1a"
_ERR_BORDER = "#6e3030"
_OVERRIDE_TINT = "#4a3b1f"
_TIER_BG = "#202226"
_MASTER_BG = "#222a30"

_FOCUS_BTN_QSS = f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
_FLAT_LINK_QSS = (
    f"QPushButton {{ border: 2px solid transparent; background: transparent; "
    f"color: {COLORS['information']}; font-weight: bold; padding: 0px 2px; border-radius: 2px; }}"
    f"QPushButton:hover {{ text-decoration: underline; }}"
    f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
    f"QPushButton:disabled {{ color: {COLORS['dim']}; }}"
)


class BusFaderSpinBox(QDoubleSpinBox):
    """StrongFocus 0.1 dB bus-fader editor that ignores unfocused wheel events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("tableNumericEditor", True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setKeyboardTracking(False)
        self.setRange(-144.0, 12.0)
        self.setSingleStep(0.1)
        self.setDecimals(1)
        self.setSuffix(" dB")
        self.setMinimumWidth(88)
        up_url = icon_path("spin-up").as_posix()
        dn_url = icon_path("spin-down").as_posix()
        self.setStyleSheet(
            f"QDoubleSpinBox {{ background-color: {COLORS['bg_alt']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 1px 4px; }}"
            f"QDoubleSpinBox:focus {{ border: 2px solid {COLORS['information']}; }}"
            f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ border: none; background: transparent; }}"
            f"QDoubleSpinBox::up-button:hover:enabled, QDoubleSpinBox::down-button:hover:enabled {{ background: #4a4a4a; }}"
            f"QDoubleSpinBox::up-arrow {{ image: url(\"{up_url}\"); width: 10px; height: 10px; }}"
            f"QDoubleSpinBox::down-arrow {{ image: url(\"{dn_url}\"); width: 10px; height: 10px; }}"
        )

    def wheelEvent(self, event: Any) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_PageUp:
            self.setValue(min(self.maximum(), round(self.value() + 1.0, 1)))
            return
        if event.key() == Qt.Key.Key_PageDown:
            self.setValue(max(self.minimum(), round(self.value() - 1.0, 1)))
            return
        super().keyPressEvent(event)


def _make_banner(
    bg: str, accent: str, border: str, *, action_label: str = "", flat_link: bool = False,
) -> tuple[QWidget, QLabel, QPushButton | None]:
    box = QWidget()
    box.setObjectName("sfBanner")
    box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    box.setStyleSheet(
        f"QWidget#sfBanner {{ background-color: {bg}; border: 1px solid {border}; "
        f"border-left: 3px solid {accent}; border-radius: 2px; }}"
    )
    lay = QHBoxLayout(box)
    lay.setContentsMargins(8, 5, 8, 5)
    lay.setSpacing(8)
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"border: none; background: transparent; color: {COLORS['text']};")
    lay.addWidget(lbl, 1)
    btn: QPushButton | None = None
    if action_label:
        if flat_link:
            btn = QPushButton(action_label)
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            btn.setStyleSheet(_FLAT_LINK_QSS)
        else:
            btn = FormActionButton(action_label)
            btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            btn.setStyleSheet(_FOCUS_BTN_QSS)
        lay.addWidget(btn)
    box.hide()
    return box, lbl, btn


def _classify_tier(node: dict[str, Any]) -> int:
    ttype = str(node.get("track_type", ""))
    name_up = str(node.get("track_name", "")).upper()
    if ttype == "RoutingFolder":
        return 1
    if ttype == "Master" or name_up.startswith("MIX"):
        return 3
    return 2


def _tier_heading(tier: int) -> str:
    if tier == 1:
        return "TIER 1 · ROUTING FOLDERS (ELIGIBLE FOR BUS STAGING)"
    if tier == 2:
        return "TIER 2 · INSERT & SUBGROUP / STEM AUXES (SINGLE-INPUT FIXED, MULTI-INPUT ELIGIBLE)"
    return "TIER 3 · MIX BUS CHAIN & MASTER (MASTER ANNOTATED ONLY)"


def _format_db(val: float | None, offset: float = 0.0, *, signed: bool = False) -> str:
    if val is None:
        return "\u2014"
    try:
        num = float(val) + float(offset)
    except (TypeError, ValueError):
        return "\u2014"
    if not math.isfinite(num):
        return "\u2014"
    if abs(num) < 0.05:
        return "0.0 dB"
    return f"{num:+.1f} dB" if signed else f"{num:.1f} dB"


def _node_type_label(node: dict[str, Any]) -> str:
    ttype = str(node.get("track_type", ""))
    if ttype == "RoutingFolder":
        return "Routing Folder"
    if ttype == "Master":
        return "Master"
    if node.get("is_parallel_parked"):
        return "Parallel Aux"
    if ttype == "Aux":
        ups = node.get("upstream_tracks") or []
        return "Stem Aux" if len(ups) >= 2 else "Aux (1-in)"
    return ttype


def _is_bus_staging_eligible(node: dict[str, Any]) -> bool:
    ttype = str(node.get("track_type", ""))
    if ttype not in ("RoutingFolder", "Aux") or node.get("is_parallel_parked"):
        return False
    if node.get("is_muted") or node.get("is_inactive"):
        return False
    ups = len(node.get("upstream_tracks") or [])
    if node.get("slot_a_trim_plugin") and not node.get("slot_a_trim_bypassed") and ups >= 1:
        return True
    return ups >= 1 if ttype == "RoutingFolder" else ups >= 2


def _format_non_staged_fader(node: dict[str, Any], staging_enabled: bool) -> str:
    if node.get("is_parallel_parked"):
        return "-\u221e dB (parked)"
    rec = node.get("recommended_fader_db")
    if rec is None:
        rec = node.get("template_fader_db", 0.0)
    db_txt = _format_db(rec, signed=True)
    if not staging_enabled and _is_bus_staging_eligible(node):
        return f"{db_txt} (staging off)"
    return f"{db_txt} (fixed)"


def _node_notes(node: dict[str, Any], *, staging_enabled: bool = True) -> str:
    parts: list[str] = []
    ups = list(node.get("upstream_tracks") or [])
    if ups:
        summary = " + ".join(ups[:3]) + (f" (+{len(ups) - 3})" if len(ups) > 3 else "")
        parts.append(summary)
    inserts = list(node.get("active_inserts") or [])
    if inserts:
        ins_s = ", ".join(inserts[:2]) + (f" +{len(inserts) - 2}" if len(inserts) > 2 else "")
        parts.append(f"insert: {ins_s}")
    if node.get("track_type") == "Master":
        out_b = node.get("output_bus") or ""
        parts.append(f"Annotated only{f' · {out_b}' if out_b else ''}")
    elif node.get("is_parallel_parked"):
        parts.append("Untouched (-144.0 dB template default)")
    elif staging_enabled:
        strat = str(node.get("staging_strategy") or "")
        trim_plug = node.get("slot_a_trim_plugin")
        if node.get("staged") and trim_plug and strat in ("trim_and_fader", "trim_only"):
            trim_db = float(node.get("recommended_trim_db", 0.0))
            parts.append(f"Slot A {trim_plug}: {_format_db(trim_db, signed=True)}")
        elif node.get("staged") and not node.get("is_overridden") and abs(float(node.get("recommended_fader_db", 0.0))) >= 0.05:
            parts.append("auto-staged")
    return " · ".join(parts)


class SignalFlowPanel(QWidget):
    """Pure UI view for the Phase 3 Signal Flow bus-staging inspector tab."""

    compute_requested = Signal()
    cancel_requested = Signal()
    reset_all_overrides_requested = Signal()
    bus_override_changed = Signal(str, float)
    bus_override_reset = Signal(str)
    active_only_toggled = Signal(bool)
    edit_config_requested = Signal()
    back_to_routing_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._populating = False
        self._last_graph: dict[str, Any] | None = None
        self._staging_enabled = True
        self._spin_by_bus: dict[str, BusFaderSpinBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 6, 0, 0)
        root.setSpacing(6)

        self.controls_bar = self._build_controls_bar()
        root.addWidget(self.controls_bar)

        self.stale_banner, self._stale_lbl, self._stale_btn = _make_banner(
            _INFO_BG, COLORS["information"], COLORS["accent"], action_label="Refresh Preview",
        )
        assert self._stale_btn is not None
        self._stale_btn.clicked.connect(self.compute_requested.emit)
        root.addWidget(self.stale_banner)

        self.disabled_banner, self._disabled_lbl, self._disabled_btn = _make_banner(
            _INFO_BG, COLORS["information"], COLORS["accent"],
            action_label="Edit in Session Config", flat_link=True,
        )
        assert self._disabled_btn is not None
        self._disabled_btn.clicked.connect(self.edit_config_requested.emit)
        root.addWidget(self.disabled_banner)

        self.inserts_banner, self._inserts_lbl, _ = _make_banner(_AMBER_BG, COLORS["attention"], _AMBER_BORDER)
        root.addWidget(self.inserts_banner)

        self.cycle_banner, self._cycle_lbl, _ = _make_banner(_AMBER_BG, COLORS["attention"], _AMBER_BORDER)
        root.addWidget(self.cycle_banner)

        self.excluded_banner, self._excluded_lbl, _ = _make_banner(_AMBER_BG, COLORS["attention"], _AMBER_BORDER)
        root.addWidget(self.excluded_banner)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_empty_page())
        self.stack.addWidget(self._build_busy_page())
        self.stack.addWidget(self._build_error_page())
        self.stack.addWidget(self._build_table_page())
        root.addWidget(self.stack, 1)

        self.footer_label = QLabel("")
        self.footer_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; border: 1px solid {COLORS['accent']}; "
            f"color: {COLORS['metadata']}; padding: 4px 8px;"
        )
        self.footer_label.setWordWrap(True)
        self.footer_label.hide()
        root.addWidget(self.footer_label)

        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._esc_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._esc_shortcut.activated.connect(self._on_escape_pressed)

    def _build_controls_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("sfControlsBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet(
            f"QWidget#sfControlsBar {{ background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['accent']}; border-radius: 2px; }}"
        )
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(6, 4, 8, 4)
        outer.setSpacing(4)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)
        self.refresh_btn = FormActionButton("Refresh Preview")
        self.refresh_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.refresh_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.refresh_btn.clicked.connect(self.compute_requested.emit)
        row1.addWidget(self.refresh_btn)

        self.reset_all_btn = FormActionButton("Reset Overrides")
        self.reset_all_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.reset_all_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.reset_all_btn.setEnabled(False)
        self.reset_all_btn.clicked.connect(self.reset_all_overrides_requested.emit)
        row1.addWidget(self.reset_all_btn)
        row1.addStretch(1)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        self.active_only_cb = QCheckBox("Active path only")
        self.active_only_cb.setChecked(True)
        self.active_only_cb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.active_only_cb.setStyleSheet("border: none; background: transparent;")
        self.active_only_cb.setToolTip("Hide silent or -∞ dB parked parallel aux tracks")
        self.active_only_cb.toggled.connect(self._on_active_only_changed)
        row2.addWidget(self.active_only_cb)

        self.target_label = QLabel("Target: -6.0 dBFS pk / -18.0 dBFS RMS · Pan law: -3.0 dB")
        self.target_label.setStyleSheet(f"border: none; background: transparent; color: {COLORS['metadata']};")
        self.target_label.setWordWrap(True)
        self.target_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row2.addWidget(self.target_label, 1)
        outer.addLayout(row2)
        return bar

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 24, 16, 24)
        lay.setSpacing(8)
        lay.addStretch(2)
        self.empty_title = QLabel("Assign at least one track to a Pro Tools template folder")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_title.setStyleSheet(f"color: {COLORS['heading']}; font-weight: bold;")
        self.empty_title.setWordWrap(True)
        lay.addWidget(self.empty_title)

        self.empty_detail = QLabel(
            "Drag tracks into Routing Folders or click Auto-Assign to enable offline Signal Flow preview."
        )
        self.empty_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_detail.setStyleSheet(f"color: {COLORS['metadata']};")
        self.empty_detail.setWordWrap(True)
        lay.addWidget(self.empty_detail)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.compute_empty_btn = FormActionButton("Compute Signal Flow Preview")
        self.compute_empty_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.compute_empty_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.compute_empty_btn.setEnabled(False)
        self.compute_empty_btn.clicked.connect(self.compute_requested.emit)
        btn_row.addWidget(self.compute_empty_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addStretch(3)
        return page

    def _build_busy_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 24, 16, 24)
        lay.setSpacing(10)
        lay.addStretch(1)

        self.busy_label = QLabel("Summing template signal flow…")
        self.busy_label.setStyleSheet(f"color: {COLORS['heading']}; font-weight: bold;")
        self.busy_label.setWordWrap(True)
        lay.addWidget(self.busy_label)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 100)
        self.busy_bar.setFixedHeight(14)
        self.busy_bar.setTextVisible(False)
        lay.addWidget(self.busy_bar)

        foot = QHBoxLayout()
        hint = QLabel("Toolbar Create stays enabled (cancels preview cleanly) · Press Escape to cancel")
        hint.setStyleSheet(f"color: {COLORS['metadata']};")
        hint.setWordWrap(True)
        foot.addWidget(hint, 1)
        self.cancel_busy_btn = FormActionButton("Cancel")
        self.cancel_busy_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cancel_busy_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.cancel_busy_btn.setAccessibleName("Cancel Signal Flow preview")
        self.cancel_busy_btn.clicked.connect(self.cancel_requested.emit)
        foot.addWidget(self.cancel_busy_btn)
        lay.addLayout(foot)
        lay.addStretch(2)
        return page

    def _build_error_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(10)

        card = QWidget()
        card.setObjectName("sfErrorCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"QWidget#sfErrorCard {{ background-color: {_ERR_BG}; border: 1px solid {_ERR_BORDER}; "
            f"border-left: 3px solid {COLORS['problems']}; border-radius: 2px; }}"
        )
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(12, 10, 12, 10)
        c_lay.setSpacing(4)
        self.error_title = QLabel("Offline Signal Flow preview unavailable")
        self.error_title.setStyleSheet(f"border: none; background: transparent; color: {COLORS['heading']}; font-weight: bold;")
        self.error_title.setWordWrap(True)
        c_lay.addWidget(self.error_title)
        self.error_detail = QLabel("")
        self.error_detail.setStyleSheet(f"border: none; background: transparent; color: {COLORS['metadata']};")
        self.error_detail.setWordWrap(True)
        c_lay.addWidget(self.error_detail)
        lay.addWidget(card)

        btn_row = QHBoxLayout()
        self.retry_btn = FormActionButton("Retry Template Scan")
        self.retry_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.retry_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.retry_btn.clicked.connect(self.compute_requested.emit)
        btn_row.addWidget(self.retry_btn)
        self.back_routing_btn = FormActionButton("Back to Routing && Folders")
        self.back_routing_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.back_routing_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.back_routing_btn.clicked.connect(self.back_to_routing_requested.emit)
        btn_row.addWidget(self.back_routing_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addStretch(1)
        return page

    def _build_table_page(self) -> QWidget:
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Signal Path / Node", "Type", "Class", "Pre Peak", "Pre RMS",
            "Bus Fader", "Post Peak", "Post RMS", "Feeders / Inserts",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        hdr = self.table.horizontalHeader()
        hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        widths = [150, 98, 78, 76, 76, 152, 76, 76, 260]
        for col, w in enumerate(widths):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(col, w)
        hdr.setStretchLastSection(True)
        return self.table

    def _on_escape_pressed(self) -> None:
        if self.stack.currentIndex() == _STACK_BUSY:
            self.cancel_requested.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self.stack.currentIndex() == _STACK_BUSY:
            self.cancel_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_active_only_changed(self, checked: bool) -> None:
        self.active_only_toggled.emit(checked)
        if self._last_graph is not None:
            self.show_graph(self._last_graph, staging_enabled=self._staging_enabled)

    def focus_bus_editor(self, bus_name: str) -> None:
        spin = self._spin_by_bus.get(bus_name)
        if spin is not None and spin.isEnabled():
            spin.setFocus(Qt.FocusReason.OtherFocusReason)

    def restore_focus_after_cancel(self) -> None:
        if self.refresh_btn.isEnabled():
            self.refresh_btn.setFocus(Qt.FocusReason.OtherFocusReason)
        elif self.compute_empty_btn.isEnabled():
            self.compute_empty_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    # ── State renderers ──────────────────────────────────────────────────

    def show_empty(self, *, assigned_count: int, folder_count: int, template_name: str = "") -> None:
        self._last_graph = None
        self._spin_by_bus.clear()
        self._hide_all_banners()
        self.footer_label.hide()
        self.refresh_btn.setEnabled(assigned_count > 0)
        self.reset_all_btn.setEnabled(False)
        if assigned_count <= 0:
            self.empty_title.setText("Assign at least one track to a Pro Tools template folder")
            self.empty_detail.setText(
                "Drag tracks into Routing Folders or click Auto-Assign to enable offline Signal Flow preview."
            )
            self.compute_empty_btn.setEnabled(False)
        else:
            tmpl = f" ({template_name})" if template_name else ""
            self.empty_title.setText(f"{assigned_count} track(s) assigned across {folder_count} template folder(s)")
            self.empty_detail.setText(f"Computes time-aligned float64 bus sums offline{tmpl}.")
            self.compute_empty_btn.setEnabled(True)
        self.stack.setCurrentIndex(_STACK_EMPTY)

    def show_busy(self, current: int, total: int, message: str) -> None:
        was_busy = self.stack.currentIndex() == _STACK_BUSY
        self._hide_all_banners()
        self.footer_label.hide()
        self.refresh_btn.setEnabled(False)
        self.reset_all_btn.setEnabled(False)
        self.busy_label.setText(message)
        if total > 0:
            self.busy_bar.setRange(0, total)
            self.busy_bar.setValue(min(current, total))
        else:
            self.busy_bar.setRange(0, 0)
        self.stack.setCurrentIndex(_STACK_BUSY)
        if not was_busy:
            self.cancel_busy_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def show_error(self, title: str, detail: str) -> None:
        self._last_graph = None
        self._spin_by_bus.clear()
        self._hide_all_banners()
        self.footer_label.hide()
        self.refresh_btn.setEnabled(True)
        self.reset_all_btn.setEnabled(False)
        self.error_title.setText(f"\u2715  {title}")
        self.error_detail.setText(detail)
        self.stack.setCurrentIndex(_STACK_ERROR)

    def set_stale(self, stale: bool, message: str = "") -> None:
        if stale and self._last_graph is not None:
            self._stale_lbl.setText(
                message
                or "\u21bb  Signal Flow preview is out of date — click Refresh Preview to recalculate (manual overrides preserved)."
            )
            self.stale_banner.show()
        else:
            self.stale_banner.hide()

    def _hide_all_banners(self) -> None:
        for b in (self.stale_banner, self.disabled_banner, self.inserts_banner, self.cycle_banner, self.excluded_banner):
            b.hide()

    def show_graph(
        self,
        graph_dict: dict[str, Any],
        *,
        staging_enabled: bool = True,
        focus_bus: str | None = None,
    ) -> None:
        self._last_graph = graph_dict
        self._staging_enabled = staging_enabled
        self.refresh_btn.setEnabled(True)

        target_pk = float(graph_dict.get("target_peak_db", -6.0))
        target_rms = float(graph_dict.get("target_rms_db", -18.0))
        pan_law = float(graph_dict.get("pan_law_db", -3.0))
        conv = str(graph_dict.get("dbfs_convention", "standard"))
        rms_off = dbfs_offset({"dbfs_convention": conv})
        self.target_label.setText(
            f"Target: {target_pk:.1f} dBFS pk / {target_rms + rms_off:.1f} dBFS RMS · Pan law: {pan_law:.1f} dB"
        )
        self._update_graph_banners(graph_dict, staging_enabled)
        self._populate_table(graph_dict, target_pk, rms_off, staging_enabled)
        self._update_footer(graph_dict, conv, staging_enabled)
        self.stack.setCurrentIndex(_STACK_TABLE)
        if focus_bus:
            self.focus_bus_editor(focus_bus)

    def _update_graph_banners(self, graph_dict: dict[str, Any], staging_enabled: bool) -> None:
        self.disabled_banner.setVisible(not staging_enabled)
        if not staging_enabled:
            self._disabled_lbl.setText(
                "\u2139  Bus fader staging is disabled — showing read-only summed levels (no bus faders modified on Create)."
            )

        nodes = graph_dict.get("nodes") or {}
        ins_items: list[str] = []
        for name, n in nodes.items():
            if n.get("track_type") == "Audio" or not n.get("active_inserts"):
                continue
            unsim = list(n["active_inserts"])
            trim_name = n.get("slot_a_trim_plugin")
            if trim_name and unsim and unsim[0] == trim_name:
                unsim = unsim[1:]
            if unsim:
                ins_items.append(f"{name} ({unsim[0]})")
        if ins_items:
            shown = ", ".join(ins_items[:3])
            extra = f" — and {len(ins_items) - 3} more" if len(ins_items) > 3 else ""
            self._inserts_lbl.setText(
                f"\u26a0  Linear pre-insert sum estimate: active plugins on {shown}{extra} are not simulated."
            )
            self.inserts_banner.setToolTip("\n".join(ins_items))
            self.inserts_banner.show()
        else:
            self.inserts_banner.hide()

        cycles = graph_dict.get("cycles") or []
        if cycles:
            c_txt = "; ".join(" \u2192 ".join(c) for c in cycles[:2])
            self._cycle_lbl.setText(f"\u26a0  Routing cycle detected: {c_txt}. Cycle nodes skipped from bus staging.")
            self.cycle_banner.show()
        else:
            self.cycle_banner.hide()

        excluded = graph_dict.get("excluded_tracks") or {}
        if excluded:
            names = ", ".join(list(excluded.keys())[:3])
            self._excluded_lbl.setText(
                f"\u26a0  {len(excluded)} track(s) excluded from Signal Flow sum (unreadable audio): {names}"
            )
            self.excluded_banner.setToolTip("\n".join(f"{k}: {v}" for k, v in excluded.items()))
            self.excluded_banner.show()
        else:
            self.excluded_banner.hide()

    def _populate_table(
        self, graph_dict: dict[str, Any], target_pk: float, rms_off: float, staging_enabled: bool,
    ) -> None:
        self._populating = True
        self._spin_by_bus.clear()
        try:
            self.table.setRowCount(0)
            nodes_map: dict[str, dict[str, Any]] = graph_dict.get("nodes") or {}
            order: list[str] = list(graph_dict.get("topological_order") or nodes_map.keys())
            active_only = self.active_only_cb.isChecked()
            tiers: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
            override_count = 0
            for name in order:
                node = nodes_map.get(name)
                if not node or node.get("track_type") == "Audio":
                    continue
                if active_only and node.get("classification") == "Silent" and not node.get("staged"):
                    continue
                if node.get("is_overridden"):
                    override_count += 1
                tiers[_classify_tier(node)].append(node)
            self.reset_all_btn.setEnabled(staging_enabled and override_count > 0)
            for tier_id in (1, 2, 3):
                rows = tiers[tier_id]
                if not rows:
                    continue
                self._append_tier_row(_tier_heading(tier_id))
                for node in rows:
                    self._append_bus_row(node, target_pk, rms_off, staging_enabled)
        finally:
            self._populating = False

    def _append_tier_row(self, title: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        tier_h = max(22, self.fontMetrics().height() + 8)
        self.table.setRowHeight(r, tier_h)
        self.table.setSpan(r, 0, 1, 9)
        item = QTableWidgetItem(title)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        item.setForeground(QColor(COLORS["metadata"]))
        item.setBackground(QColor(_TIER_BG))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.table.setItem(r, 0, item)

    def _append_bus_row(
        self, node: dict[str, Any], target_pk: float, rms_off: float, staging_enabled: bool,
    ) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        base_h = max(28, self.fontMetrics().height() + 14)
        self.table.setRowHeight(r, base_h)
        name = str(node.get("track_name", ""))
        ttype = str(node.get("track_type", ""))
        cls = str(node.get("classification", "Silent"))
        staged = bool(node.get("staged"))
        overridden = bool(node.get("is_overridden"))
        pre_pk = node.get("pre_fader_peak_db")
        post_pk = node.get("post_fader_peak_db")

        def _mk(text: str, color: str | QColor, *, bold: bool = False) -> QTableWidgetItem:
            it = QTableWidgetItem(text)
            it.setForeground(QColor(color))
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            if ttype == "Master":
                it.setBackground(QColor(_MASTER_BG))
            return it

        dim_row = cls == "Silent" and not staged
        name_col = COLORS["dim"] if dim_row else (COLORS["heading"] if (staged or ttype == "Master") else COLORS["text"])
        meta_col = COLORS["dim"] if dim_row else COLORS["metadata"]
        cls_col = FILE_COLOR_TRANSIENT if cls == "Transient" else (FILE_COLOR_SUSTAINED if cls == "Sustained" else COLORS["dim"])
        pre_hot = pre_pk is not None and math.isfinite(float(pre_pk)) and float(pre_pk) > target_pk + 0.05
        post_hot = post_pk is not None and math.isfinite(float(post_pk)) and float(post_pk) > target_pk + 0.05
        post_clean = post_pk is not None and math.isfinite(float(post_pk)) and not post_hot and (staged or ttype == "Master")

        self.table.setItem(r, 0, _mk(name, name_col, bold=(staged or ttype == "Master")))
        self.table.setItem(r, 1, _mk(_node_type_label(node), meta_col))
        self.table.setItem(r, 2, _mk(cls, cls_col, bold=(cls != "Silent")))
        self.table.setItem(r, 3, _mk(_format_db(pre_pk), COLORS["attention"] if pre_hot else name_col, bold=pre_hot))
        self.table.setItem(r, 4, _mk(_format_db(node.get("pre_fader_rms_db"), rms_off), meta_col))

        if staged and staging_enabled:
            self._set_fader_editor_cell(
                r, name, float(node.get("recommended_fader_db", 0.0)),
                float(node.get("auto_fader_db", 0.0)), overridden,
            )
        else:
            self.table.setItem(r, 5, _mk(_format_non_staged_fader(node, staging_enabled), meta_col))

        post_col = COLORS["attention"] if post_hot else (COLORS["clean"] if post_clean else name_col)
        self.table.setItem(r, 6, _mk(_format_db(post_pk), post_col, bold=(post_hot or post_clean)))
        self.table.setItem(r, 7, _mk(_format_db(node.get("post_fader_rms_db"), rms_off), meta_col))
        self.table.setItem(r, 8, _mk(_node_notes(node, staging_enabled=staging_enabled), meta_col))

    def _set_fader_editor_cell(
        self, row: int, bus_name: str, value_db: float, auto_db: float, overridden: bool,
    ) -> None:
        cell = QWidget()
        if overridden:
            cell.setObjectName("overrideCell")
            cell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            cell.setStyleSheet(f"QWidget#overrideCell {{ background-color: {_OVERRIDE_TINT}; }}")
        lay = QHBoxLayout(cell)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        spin = BusFaderSpinBox(cell)
        spin.setAccessibleName(f"{bus_name} target fader (dB)")
        spin.setValue(value_db)
        spin.valueChanged.connect(lambda val, bn=bus_name: self._emit_override(bn, val))
        editor_h = max(22, spin.sizeHint().height())
        spin.setMinimumHeight(editor_h)
        lay.addWidget(spin)
        self._spin_by_bus[bus_name] = spin
        if overridden:
            rst = ConfigResetButton()
            rst.setIcon(icon("reset"))
            rst.setIconSize(QSize(14, 14))
            rst.setFixedSize(max(24, editor_h + 2), editor_h)
            rst.setAccessibleName(f"Reset {bus_name} fader override")
            rst.setToolTip(f"Reset {bus_name} to auto-calculated {_format_db(auto_db, signed=True)}")
            rst.setStyleSheet(
                f"QPushButton {{ background-color: #3a3a3a; border: 1px solid {COLORS['border']}; border-radius: 2px; }}"
                f"QPushButton:hover {{ background-color: #4a4a4a; }}"
                f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
            )
            rst.clicked.connect(lambda _c=False, bn=bus_name: self.bus_override_reset.emit(bn))
            lay.addWidget(rst)
        else:
            spacer = QWidget()
            spacer.setFixedSize(max(24, editor_h + 2), editor_h)
            lay.addWidget(spacer)
        lay.addStretch(1)
        self.table.setRowHeight(row, max(self.table.rowHeight(row), editor_h + 6))
        self.table.setCellWidget(row, 5, cell)

    def _emit_override(self, bus_name: str, val: float) -> None:
        if not self._populating:
            self.bus_override_changed.emit(bus_name, round(val, 1))

    def _update_footer(self, graph_dict: dict[str, Any], conv: str, staging_enabled: bool = True) -> None:
        rec = (graph_dict.get("recommended_bus_faders") or {}) if staging_enabled else {}
        trims = (graph_dict.get("recommended_slot_a_trims") or {}) if staging_enabled else {}
        staged_names = list(dict.fromkeys([*trims.keys(), *rec.keys()]))
        parts: list[str] = []
        for k in staged_names:
            t_v, f_v = trims.get(k), rec.get(k)
            if t_v is not None and f_v is not None:
                parts.append(f"{k} (Trim {_format_db(t_v, signed=True)}, Fader {_format_db(f_v, signed=True)})")
            elif t_v is not None:
                parts.append(f"{k} (Trim {_format_db(t_v, signed=True)})")
            elif f_v is not None:
                parts.append(f"{k} ({_format_db(f_v, signed=True)})")
        staged_txt = ", ".join(parts) if parts else "none"
        nodes = graph_dict.get("nodes") or {}
        master = next((n for n in nodes.values() if n.get("track_type") == "Master"), None)
        mpk_val = None
        if master is not None:
            mpk_val = master.get("post_fader_peak_db")
            if mpk_val is None:
                mpk_val = master.get("pre_fader_peak_db")
        mpk = _format_db(mpk_val)
        scale = "AES-17 dBFS" if conv == "aes17" else "Standard dBFS"
        self.footer_label.setText(
            f"Staged on Create: {staged_txt}  ·  Master Peak: {mpk}  ·  Scale: {scale}  ·  Atomic post-close .ptx pass with rollback"
        )
        self.footer_label.show()
