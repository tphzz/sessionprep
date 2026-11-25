"""Phase 3 Signal Flow inspector panel, corrected calculation table, and DAG graph view."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.audio import dbfs_offset
from sessionpreplib.ptx.signal_flow_analysis import trace_signal_flow_path

from ..controls import FormActionButton
from ..icons import icon, icon_path
from ..prefs.param_form import ConfigResetButton
from ..theme import COLORS, FILE_COLOR_SUSTAINED, FILE_COLOR_TRANSIENT
from .signal_flow_graph_view import SignalFlowGraphView

_STACK_EMPTY = 0
_STACK_BUSY = 1
_STACK_ERROR = 2
_STACK_CONTENT = 3

_VIEW_TABLE = 0
_VIEW_GRAPH = 1

_AMBER_BG = "#2a251b"
_AMBER_BORDER = "#5c4a26"
_INFO_BG = "#1c2634"
_INFO_BORDER = "#2e5078"
_GREEN_BG = "#1b2920"
_GREEN_BORDER = "#2e5a3b"
_ERR_BG = "#2e1a1a"
_ERR_BORDER = "#6e3030"
_OVERRIDE_TINT = "#4a3b1f"
_TIER_BG = "#202226"
_MASTER_BG = "#222a30"
_BLOCKED_ROW_BG = "#2e1a1a"

_FOCUS_BTN_QSS = f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
_FLAT_LINK_QSS = (
    f"QPushButton {{ border: 2px solid transparent; background: transparent; "
    f"color: {COLORS['information']}; font-weight: bold; padding: 0px 2px; border-radius: 2px; }}"
    f"QPushButton:hover {{ text-decoration: underline; }}"
    f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
    f"QPushButton:disabled {{ color: {COLORS['dim']}; }}"
)
_TRIM_SHORT_TAG = {"TrackControl": "TC", "Protility": "Pr", "Trim": "Tr"}


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


class _SelectedWhiteTextDelegate(QStyledItemDelegate):
    """Guarantee pure #ffffff text across all cells when a table row is selected on #2a6db5."""

    def initStyleOption(self, option: Any, index: Any) -> None:
        super().initStyleOption(option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            white = QColor("#ffffff")
            option.palette.setColor(QPalette.ColorRole.Text, white)
            option.palette.setColor(QPalette.ColorRole.HighlightedText, white)


class _ShortcutSafeButton(QPushButton):
    """PushButton that claims Space via ShortcutOverride so global playback does not steal it."""

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.ShortcutOverride and isinstance(ev, QKeyEvent):
            if ev.key() == Qt.Key.Key_Space:
                ev.accept()
                return True
        return super().event(ev)


class _ViewModeToggle(QWidget):
    """Accessible [Table | Graph] segmented toggle with Left/Right/Space navigation and #4499ff focus."""

    mode_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = _VIEW_TABLE
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.table_btn = _ShortcutSafeButton("Table")
        self.graph_btn = _ShortcutSafeButton("Graph")
        for btn, mode, acc in (
            (self.table_btn, _VIEW_TABLE, "Table view"),
            (self.graph_btn, _VIEW_GRAPH, "Auto-routed graph view"),
        ):
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            btn.setAccessibleName(acc)
            btn.setMinimumHeight(24)
            btn.clicked.connect(lambda _c=False, m=mode: self.set_mode(m))
            lay.addWidget(btn)
        self._apply_styles()

    @property
    def mode(self) -> int:
        return self._mode

    def set_mode(self, mode: int, *, emit: bool = True) -> None:
        target = _VIEW_GRAPH if mode == _VIEW_GRAPH else _VIEW_TABLE
        changed = target != self._mode
        self._mode = target
        self._apply_styles()
        if emit and changed:
            self.mode_changed.emit(self._mode)

    def _apply_styles(self) -> None:
        self.table_btn.setChecked(self._mode == _VIEW_TABLE)
        self.graph_btn.setChecked(self._mode == _VIEW_GRAPH)
        base = (
            "QPushButton { padding: 1px 10px; border: 1px solid #444444; background-color: #2c2c2c; color: #cccccc; }"
            "QPushButton:hover { background-color: #3a3a3a; color: #ffffff; }"
            "QPushButton:checked { background-color: #2a6db5; border-color: #2a6db5; color: #ffffff; font-weight: bold; }"
            f"QPushButton:focus {{ padding: 0px 9px; border: 2px solid {COLORS['information']}; }}"
        )
        self.table_btn.setStyleSheet(
            base + "QPushButton { border-right: none; border-top-left-radius: 3px; border-bottom-left-radius: 3px; }"
            f"QPushButton:focus {{ border-right: 2px solid {COLORS['information']}; }}"
        )
        self.graph_btn.setStyleSheet(
            base + "QPushButton { border-top-right-radius: 3px; border-bottom-right-radius: 3px; }"
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Left:
            self.set_mode(_VIEW_TABLE)
            self.table_btn.setFocus(Qt.FocusReason.TabFocusReason)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.set_mode(_VIEW_GRAPH)
            self.graph_btn.setFocus(Qt.FocusReason.TabFocusReason)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self.set_mode(_VIEW_GRAPH if self._mode == _VIEW_TABLE else _VIEW_TABLE)
            event.accept()
            return
        super().keyPressEvent(event)


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
        self.setMinimumWidth(86)
        up_url = icon_path("spin-up").as_posix()
        dn_url = icon_path("spin-down").as_posix()
        self.setStyleSheet(
            f"QDoubleSpinBox {{ background-color: #3a3a3a; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 1px 4px; }}"
            f"QDoubleSpinBox:focus {{ border: 2px solid {COLORS['information']}; }}"
            f"QDoubleSpinBox:disabled {{ background-color: #2b2b2b; color: {COLORS['dim']}; border-color: #3a3a3a; }}"
            f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ border: none; background: transparent; }}"
            f"QDoubleSpinBox::up-button:hover:enabled, QDoubleSpinBox::down-button:hover:enabled {{ background: #4a4a4a; }}"
            f"QDoubleSpinBox::up-button:pressed:enabled, QDoubleSpinBox::down-button:pressed:enabled {{ background: #2a6db5; }}"
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
    if ttype == "Master" or node.get("is_terminal") or node.get("is_post_terminal") or name_up.startswith("MIX"):
        return 3
    return 2


def _tier_heading(tier: int, *, has_trims: bool) -> str:
    if tier == 1:
        return (
            "TIER 1 \u00b7 ROUTING FOLDERS (PASS-THROUGH \u00b7 STAGED DOWNSTREAM)"
            if has_trims
            else "TIER 1 \u00b7 ROUTING FOLDERS (ELIGIBLE FOR BUS STAGING)"
        )
    if tier == 2:
        return (
            "TIER 2 \u00b7 SUBGROUP & STEM AUXES (SLOT A TRIM + INVERSE FADER \u00b7 NET \u0394 = 0.0 dB)"
            if has_trims
            else "TIER 2 \u00b7 INSERT & SUBGROUP / STEM AUXES (SINGLE-INPUT FIXED, MULTI-INPUT ELIGIBLE)"
        )
    return (
        "TIER 3 \u00b7 MIX BUS CHAIN & MASTER (TERMINAL TRIM ONLY \u00b7 MASTER ANNOTATED)"
        if has_trims
        else "TIER 3 \u00b7 MIX BUS CHAIN & MASTER (MASTER ANNOTATED ONLY)"
    )


def _format_db(val: Any, offset: float = 0.0, *, signed: bool = False) -> str:
    if val is None or isinstance(val, bool):
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


def _format_pk_rms_pair(pk: Any, rms: Any, rms_off: float) -> str:
    if pk is None and rms is None:
        return "\u2014"
    try:
        p_f = None if pk is None or isinstance(pk, bool) else float(pk)
    except (TypeError, ValueError):
        p_f = None
    try:
        r_f = None if rms is None or isinstance(rms, bool) else float(rms)
    except (TypeError, ValueError):
        r_f = None
    p_s = "\u2014" if (p_f is None or not math.isfinite(p_f)) else f"{p_f:.1f}"
    r_s = "\u2014" if (r_f is None or not math.isfinite(r_f)) else f"{r_f + rms_off:.1f}"
    return f"{p_s} / {r_s}"


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


def _format_slot_a_cell(node: dict[str, Any], staging_enabled: bool) -> tuple[str, str, bool]:
    if node.get("slot_a_blocked_error"):
        return "SLOT A BLOCKED", COLORS["problems"], True
    if node.get("is_post_terminal"):
        return "\u2014 (exempt)", COLORS["metadata"], False
    if not staging_enabled:
        return "\u2014", COLORS["metadata"], False
    strat = str(node.get("staging_strategy") or "")
    plug = node.get("slot_a_trim_plugin")
    if node.get("staged") and plug and strat in ("trim_and_fader", "trim_only"):
        trim_db = _safe_float(node.get("recommended_trim_db"))
        tag = "auto" if node.get("slot_a_trim_auto_insert") else _TRIM_SHORT_TAG.get(str(plug), str(plug)[:3])
        return f"{_format_db(trim_db, signed=True)} ({tag})", FILE_COLOR_TRANSIENT.name(), True
    return "\u2014", COLORS["metadata"], False


def _node_notes(node: dict[str, Any], *, staging_enabled: bool = True) -> str:
    if node.get("slot_a_blocked_error"):
        return str(node["slot_a_blocked_error"])
    parts: list[str] = []
    ups = list(node.get("upstream_tracks") or [])
    if ups:
        parts.append(" + ".join(ups[:3]) + (f" (+{len(ups) - 3})" if len(ups) > 3 else ""))
    trim_plug = node.get("slot_a_trim_plugin")
    inserts = [p for p in (node.get("active_inserts") or ()) if p != trim_plug]
    if inserts:
        parts.append("insert: " + ", ".join(inserts[:2]) + (f" +{len(inserts) - 2}" if len(inserts) > 2 else ""))
    if node.get("track_type") == "Master":
        out_b = node.get("output_bus") or ""
        parts.append(f"Annotated only{f' \u00b7 {out_b}' if out_b else ''}")
    elif node.get("is_parallel_parked"):
        parts.append("Untouched (-144.0 dB template default)")
    elif staging_enabled:
        strat = str(node.get("staging_strategy") or "")
        if node.get("staged") and trim_plug and strat in ("trim_and_fader", "trim_only"):
            tag = " (auto-insert)" if node.get("slot_a_trim_auto_insert") else ""
            parts.append(f"Slot A {trim_plug}{tag}: {_format_db(_safe_float(node.get('recommended_trim_db')), signed=True)}")
        elif node.get("staged") and not node.get("is_overridden") and abs(_safe_float(node.get("recommended_fader_db"))) >= 0.05:
            parts.append("auto-staged")
    return " \u00b7 ".join(parts)


def _collect_unsimulated_inserts(graph_dict: dict[str, Any]) -> list[str]:
    nodes = graph_dict.get("nodes") or {}
    items: list[str] = []
    for name, n in nodes.items():
        if not isinstance(n, dict) or n.get("track_type") in ("Audio", "Instrument"):
            continue
        if n.get("is_inactive") or n.get("is_muted") or n.get("is_parallel_parked") or n.get("pre_fader_peak_db") is None:
            continue
        unsim = list(n.get("active_inserts") or ())
        trim_name = n.get("slot_a_trim_plugin")
        if trim_name and unsim and unsim[0] == trim_name:
            unsim = unsim[1:]
        if unsim:
            items.append(f"{name} ({', '.join(str(p) for p in unsim)})")
    return items


class SignalFlowPanel(QWidget):
    """Pure UI view for Phase 3 Signal Flow Table, Graph View, Validation Strip, and Path Bar."""

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
        self._selected_node: str | None = None
        self._spin_by_bus: dict[str, BusFaderSpinBox] = {}
        self._row_by_node: dict[str, int] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 6, 0, 0)
        root.setSpacing(6)

        self.controls_bar = self._build_controls_bar()
        root.addWidget(self.controls_bar)

        self.balance_banner, self._balance_lbl, self._balance_btn = _make_banner(
            _GREEN_BG, COLORS["clean"], _GREEN_BORDER, action_label="Action", flat_link=False,
        )
        assert self._balance_btn is not None
        self._balance_btn.hide()
        self._balance_btn.clicked.connect(self._on_balance_action_clicked)
        self._balance_action_kind = ""
        root.addWidget(self.balance_banner)

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

        self.cycle_banner, self._cycle_lbl, _ = _make_banner(_AMBER_BG, COLORS["attention"], _AMBER_BORDER)
        root.addWidget(self.cycle_banner)

        self.excluded_banner, self._excluded_lbl, _ = _make_banner(_AMBER_BG, COLORS["attention"], _AMBER_BORDER)
        root.addWidget(self.excluded_banner)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_empty_page())
        self.stack.addWidget(self._build_busy_page())
        self.stack.addWidget(self._build_error_page())
        self.stack.addWidget(self._build_content_page())
        root.addWidget(self.stack, 1)

        self.path_bar = QLabel("")
        self.path_bar.setObjectName("sfPathBar")
        self.path_bar.setStyleSheet(
            f"QLabel#sfPathBar {{ background-color: #1c2430; border: 1px solid {_INFO_BORDER}; "
            f"border-left: 3px solid {COLORS['information']}; color: {COLORS['text']}; padding: 4px 8px; }}"
        )
        self.path_bar.setWordWrap(True)
        self.path_bar.hide()
        root.addWidget(self.path_bar)

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
        self._esc_shortcut.setEnabled(False)
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
        self.view_toggle = _ViewModeToggle()
        self.view_toggle.mode_changed.connect(self._on_view_mode_changed)
        row1.addWidget(self.view_toggle)

        self.refresh_btn = FormActionButton("Refresh Preview")
        self.refresh_btn.setMinimumHeight(24)
        self.refresh_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.refresh_btn.setStyleSheet(_FOCUS_BTN_QSS)
        self.refresh_btn.clicked.connect(self.compute_requested.emit)
        row1.addWidget(self.refresh_btn)

        self.reset_all_btn = FormActionButton("Reset Overrides")
        self.reset_all_btn.setMinimumHeight(24)
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
        self.active_only_cb.setToolTip("Hide silent or -\u221e dB parked parallel aux tracks")
        self.active_only_cb.toggled.connect(self._on_active_only_changed)
        row2.addWidget(self.active_only_cb)

        self.target_label = QLabel("Target: -6.0 dBFS pk / -18.0 dBFS RMS \u00b7 Pan law: -3.0 dB")
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

        self.busy_label = QLabel("Summing template signal flow\u2026")
        self.busy_label.setStyleSheet(f"color: {COLORS['heading']}; font-weight: bold;")
        self.busy_label.setWordWrap(True)
        lay.addWidget(self.busy_label)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 100)
        self.busy_bar.setFixedHeight(14)
        self.busy_bar.setTextVisible(False)
        lay.addWidget(self.busy_bar)

        foot = QHBoxLayout()
        hint = QLabel("Toolbar Create stays enabled (cancels preview cleanly) \u00b7 Press Escape to cancel")
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

    def _build_content_page(self) -> QWidget:
        self.view_stack = QStackedWidget()
        self.table = self._build_table()
        self.graph_view = SignalFlowGraphView()
        self.graph_view.node_selected.connect(self._on_graph_node_selected)
        self.graph_view.selection_cleared.connect(self._on_graph_selection_cleared)
        self.graph_view.node_edit_requested.connect(self._open_bus_override_dialog)
        self.view_stack.addWidget(self.table)
        self.view_stack.addWidget(self.graph_view)
        return self.view_stack

    def _build_table(self) -> QTableWidget:
        tbl = QTableWidget(0, 9)
        tbl.setHorizontalHeaderLabels([
            "Node", "Class", "Bus In (Pk / RMS)", "Slot A Trim", "To Inserts",
            "Bus Fader", "Bus Out (Pk / RMS)", "Net \u0394", "Feeders / Notes",
        ])
        tbl.setItemDelegate(_SelectedWhiteTextDelegate(tbl))
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        tbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tbl.itemSelectionChanged.connect(self._on_table_selection_changed)

        hdr = tbl.horizontalHeader()
        hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        widths = [110, 76, 114, 118, 108, 148, 114, 78, 220]
        for col, w in enumerate(widths):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            tbl.setColumnWidth(col, w)
        hdr.setStretchLastSection(True)
        return tbl

    def _on_view_mode_changed(self, mode: int) -> None:
        self.view_stack.setCurrentIndex(mode)
        if mode == _VIEW_GRAPH and self._last_graph is not None:
            self.graph_view.select_node(self._selected_node, emit=False)

    def _on_balance_action_clicked(self) -> None:
        if self._balance_action_kind == "reset":
            self.reset_all_overrides_requested.emit()
        elif self._balance_action_kind == "config":
            self.edit_config_requested.emit()

    def _on_table_selection_changed(self) -> None:
        if self._populating or self._last_graph is None:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        it = self.table.item(row, 0)
        if it is None or not (it.flags() & Qt.ItemFlag.ItemIsSelectable):
            return
        name = it.text()
        self._selected_node = name
        self.graph_view.select_node(name, emit=False)
        self._update_path_bar(name)

    def _on_graph_node_selected(self, name: str) -> None:
        self._selected_node = name
        self._update_path_bar(name)
        self._sync_table_selection(name)

    def _on_graph_selection_cleared(self) -> None:
        self._selected_node = None
        self._sync_table_selection(None)
        self.path_bar.hide()

    def _sync_table_selection(self, name: str | None) -> None:
        self._populating = True
        try:
            row = self._row_by_node.get(name) if name else None
            if row is not None:
                self.table.selectRow(row)
            else:
                self.table.clearSelection()
        finally:
            self._populating = False

    def _open_bus_override_dialog(self, bus_name: str) -> None:
        if self._last_graph is None or not self._staging_enabled:
            return
        node = (self._last_graph.get("nodes") or {}).get(bus_name)
        if not isinstance(node, dict) or not node.get("staged") or node.get("slot_a_blocked_error"):
            return
        dlg = QDialog(self)
        try:
            dlg.setWindowTitle(f"Bus Fader Override \u2014 {bus_name}")
            lay = QVBoxLayout(dlg)
            auto_db = _safe_float(node.get("auto_fader_db"))
            cur_db = _safe_float(node.get("recommended_fader_db"), auto_db)
            lay.addWidget(QLabel(f"Target bus fader for {bus_name} (auto: {_format_db(auto_db, signed=True)}):"))
            spin = BusFaderSpinBox(dlg)
            spin.setValue(cur_db)
            lay.addWidget(spin)
            btns = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Reset
                | QDialogButtonBox.StandardButton.Cancel
            )
            reset_clicked = [False]

            def _on_reset() -> None:
                reset_clicked[0] = True
                dlg.accept()

            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            rst_btn = btns.button(QDialogButtonBox.StandardButton.Reset)
            if rst_btn is not None:
                rst_btn.clicked.connect(_on_reset)
            lay.addWidget(btns)
            accepted = dlg.exec() == QDialog.DialogCode.Accepted
            val = round(spin.value(), 1)
        finally:
            dlg.deleteLater()
        if accepted and self._last_graph is not None and self._staging_enabled:
            if reset_clicked[0]:
                self.bus_override_reset.emit(bus_name)
            else:
                self.bus_override_changed.emit(bus_name, val)

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
        if self.view_stack.currentIndex() == _VIEW_GRAPH:
            self.graph_view.view.setFocus(Qt.FocusReason.OtherFocusReason)
            return
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
        self._selected_node = None
        self._spin_by_bus.clear()
        self._row_by_node.clear()
        self._esc_shortcut.setEnabled(False)
        self._hide_all_banners()
        self.path_bar.hide()
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
        self._esc_shortcut.setEnabled(True)
        self._hide_all_banners()
        self.path_bar.hide()
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
        self._selected_node = None
        self._spin_by_bus.clear()
        self._row_by_node.clear()
        self._esc_shortcut.setEnabled(False)
        self._hide_all_banners()
        self.path_bar.hide()
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
                or "\u21bb  Signal Flow preview is out of date \u2014 click Refresh Preview to recalculate (manual overrides preserved)."
            )
            self.stale_banner.show()
        else:
            self.stale_banner.hide()

    def _hide_all_banners(self) -> None:
        for b in (self.balance_banner, self.stale_banner, self.disabled_banner, self.cycle_banner, self.excluded_banner):
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
        self._esc_shortcut.setEnabled(False)
        self.refresh_btn.setEnabled(True)

        target_pk = _safe_float(graph_dict.get("target_peak_db", -6.0), -6.0)
        target_rms = _safe_float(graph_dict.get("target_rms_db", -18.0), -18.0)
        pan_law = _safe_float(graph_dict.get("pan_law_db", -3.0), -3.0)
        conv = str(graph_dict.get("dbfs_convention", "standard"))
        rms_off = dbfs_offset({"dbfs_convention": conv})
        self.target_label.setText(
            f"Target: {target_pk:.1f} dBFS pk / {target_rms + rms_off:.1f} dBFS RMS \u00b7 Pan law: {pan_law:.1f} dB"
        )
        self._update_balance_strip(graph_dict, staging_enabled)
        self._update_graph_banners(graph_dict, staging_enabled)
        self._populate_table(graph_dict, target_pk, rms_off, staging_enabled)

        nodes_map = graph_dict.get("nodes") or {}
        if self._selected_node not in nodes_map:
            self._selected_node = self._pick_default_selected_node(graph_dict)
        self._sync_table_selection(self._selected_node)
        self.graph_view.set_graph(
            graph_dict,
            active_only=self.active_only_cb.isChecked(),
            staging_enabled=staging_enabled,
            selected_node=self._selected_node,
        )
        self._update_path_bar(self._selected_node)
        self._update_footer(graph_dict, conv, staging_enabled)
        self.stack.setCurrentIndex(_STACK_CONTENT)
        if focus_bus:
            self.focus_bus_editor(focus_bus)

    def _pick_default_selected_node(self, graph_dict: dict[str, Any]) -> str | None:
        nodes = graph_dict.get("nodes") or {}
        for name in graph_dict.get("topological_order") or nodes.keys():
            n = nodes.get(name)
            if isinstance(n, dict) and n.get("staged"):
                return str(name)
        return next(iter(self._row_by_node.keys()), None)

    def _update_balance_strip(self, graph_dict: dict[str, Any], staging_enabled: bool) -> None:
        assert self._balance_btn is not None
        val = graph_dict.get("balance_validation") or {}
        status = str(val.get("status", "verified"))
        summary = str(val.get("summary") or "")
        unsim = _collect_unsimulated_inserts(graph_dict)
        if not staging_enabled and status != "blocked":
            self.balance_banner.hide()
            return
        if not summary and not unsim:
            self.balance_banner.hide()
            return
        palette_map = {
            "verified": (_GREEN_BG, COLORS["clean"], _GREEN_BORDER, "\u2713", "", ""),
            "converged_shifted": (_INFO_BG, COLORS["information"], _INFO_BORDER, "\u2713", "", ""),
            "override_drift": (_AMBER_BG, COLORS["attention"], _AMBER_BORDER, "\u26a0", "Reset Overrides", "reset"),
            "caveat": (_AMBER_BG, COLORS["attention"], _AMBER_BORDER, "\u26a0", "", ""),
            "unconverged": (_AMBER_BG, COLORS["attention"], _AMBER_BORDER, "\u26a0", "Edit in Session Config", "config"),
            "blocked": (_ERR_BG, COLORS["problems"], _ERR_BORDER, "\u2715", "Edit in Session Config", "config"),
        }
        bg, acc, bdr, glyph, act_lbl, act_kind = palette_map.get(status, palette_map["verified"])
        self.balance_banner.setStyleSheet(
            f"QWidget#sfBanner {{ background-color: {bg}; border: 1px solid {bdr}; "
            f"border-left: 3px solid {acc}; border-radius: 2px; }}"
        )
        ins_cue = f" \u00b7 {len(unsim)} active insert{'s' if len(unsim) != 1 else ''} unsimulated \u24d8" if unsim else ""
        self._balance_lbl.setText(f"{glyph}  {summary or 'Balance summary unavailable'}{ins_cue}")
        tips: list[str] = []
        blocked_map = val.get("blocked_busses") or {}
        if isinstance(blocked_map, dict) and blocked_map:
            tips.append("Blocked Slot A busses:\n" + "\n".join(f"{k}: {v}" for k, v in blocked_map.items()))
        if unsim:
            tips.append("Unsimulated active inserts (linear estimate):\n" + "\n".join(unsim))
        for cav in val.get("pre_fader_send_caveats") or ():
            tips.append(f"Pre-fader send: {cav.get('source')} -> {cav.get('destination')} ({_safe_float(cav.get('shift_db')):+.1f} dB)")
        self.balance_banner.setToolTip("\n\n".join(tips))
        self._balance_action_kind = act_kind
        if act_lbl:
            self._balance_btn.setText(act_lbl)
            self._balance_btn.show()
        else:
            self._balance_btn.hide()
        self.balance_banner.show()

    def _update_graph_banners(self, graph_dict: dict[str, Any], staging_enabled: bool) -> None:
        self.disabled_banner.setVisible(not staging_enabled)
        if not staging_enabled:
            self._disabled_lbl.setText(
                "\u2139  Bus fader staging is disabled \u2014 showing read-only summed levels (no bus faders modified on Create)."
            )
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
        self._row_by_node.clear()
        try:
            self.table.setRowCount(0)
            nodes_map: dict[str, dict[str, Any]] = graph_dict.get("nodes") or {}
            order: list[str] = list(graph_dict.get("topological_order") or nodes_map.keys())
            active_only = self.active_only_cb.isChecked()
            has_trims = staging_enabled and any(
                n.get("slot_a_trim_plugin") or n.get("slot_a_blocked_error") for n in nodes_map.values()
            )
            tiers: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
            override_count = 0
            for name in order:
                node = nodes_map.get(name)
                if not node or node.get("track_type") in ("Audio", "Instrument"):
                    continue
                if active_only and node.get("classification") == "Silent" and not node.get("staged") and not node.get("slot_a_blocked_error"):
                    continue
                if node.get("is_overridden"):
                    override_count += 1
                tiers[_classify_tier(node)].append(node)
            self.reset_all_btn.setEnabled(staging_enabled and override_count > 0)
            for tier_id in (1, 2, 3):
                rows = tiers[tier_id]
                if not rows:
                    continue
                self._append_tier_row(_tier_heading(tier_id, has_trims=has_trims))
                for node in rows:
                    self._append_bus_row(node, target_pk, rms_off, staging_enabled)
        finally:
            self._populating = False

    def _append_tier_row(self, title: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setRowHeight(r, max(22, self.fontMetrics().height() + 8))
        self.table.setSpan(r, 0, 1, 9)
        item = QTableWidgetItem(title)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        item.setForeground(QColor(COLORS["metadata"]))
        item.setBackground(QColor(_TIER_BG))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.table.setItem(r, 0, item)

    def _format_net_delta_cell(self, node: dict[str, Any], *, blocked: bool, staged: bool, overridden: bool, meta_col: str) -> tuple[str, str, bool]:
        if blocked:
            return "BLOCKED", COLORS["problems"], True
        net_val = _safe_float(node.get("net_gain_delta_db"))
        base = _format_db(net_val, signed=True)
        if node.get("is_terminal") and abs(net_val) >= 0.05:
            return f"{base} (T)", FILE_COLOR_TRANSIENT.name(), True
        if overridden or (staged and not node.get("is_terminal") and abs(net_val) > 0.15):
            return f"{base} \u26a0", COLORS["attention"], True
        if staged:
            return f"{base} \u2713", COLORS["clean"], True
        return base, meta_col, False

    def _append_bus_row(
        self, node: dict[str, Any], target_pk: float, rms_off: float, staging_enabled: bool,
    ) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setRowHeight(r, max(28, self.fontMetrics().height() + 14))
        name = str(node.get("track_name", ""))
        self._row_by_node[name] = r
        ttype = str(node.get("track_type", ""))
        cls = str(node.get("classification", "Silent"))
        staged, overridden, blocked = bool(node.get("staged")), bool(node.get("is_overridden")), bool(node.get("slot_a_blocked_error"))

        def _mk(text: str, color: str | QColor, *, bold: bool = False) -> QTableWidgetItem:
            it = QTableWidgetItem(text)
            it.setForeground(QColor(color))
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            if blocked:
                it.setBackground(QColor(_BLOCKED_ROW_BG))
            elif ttype == "Master":
                it.setBackground(QColor(_MASTER_BG))
            return it

        dim_row = cls == "Silent" and not staged and not blocked
        name_col = COLORS["dim"] if dim_row else (COLORS["heading"] if (staged or ttype == "Master" or blocked) else COLORS["text"])
        meta_col = COLORS["dim"] if dim_row else COLORS["metadata"]
        cls_col = FILE_COLOR_TRANSIENT if cls == "Transient" else (FILE_COLOR_SUSTAINED if cls == "Sustained" else COLORS["dim"])

        in_pk = node.get("pre_insert_peak_db") if node.get("pre_insert_peak_db") is not None else node.get("pre_fader_peak_db")
        in_rms = node.get("pre_insert_rms_db") if node.get("pre_insert_rms_db") is not None else node.get("pre_fader_rms_db")
        ins_pk, ins_rms = node.get("pre_fader_peak_db"), node.get("pre_fader_rms_db")
        out_pk, out_rms = node.get("post_fader_peak_db"), node.get("post_fader_rms_db")
        in_val = _safe_float(in_pk, -999.0)
        in_clip = in_pk is not None and in_val >= -0.05
        in_hot = in_pk is not None and in_val > target_pk + 0.05
        in_col = COLORS["problems"] if in_clip else (COLORS["attention"] if in_hot else meta_col)
        ins_hot = ins_pk is not None and _safe_float(ins_pk, -999.0) > target_pk + 0.05
        out_hot = out_pk is not None and _safe_float(out_pk, -999.0) > target_pk + 0.05

        self.table.setItem(r, 0, _mk(name, name_col, bold=(staged or ttype == "Master" or blocked)))
        self.table.setItem(r, 1, _mk(cls, cls_col, bold=(cls != "Silent")))
        self.table.setItem(r, 2, _mk(_format_pk_rms_pair(in_pk, in_rms, rms_off), in_col, bold=(in_clip or in_hot)))

        trim_txt, trim_col, trim_bold = _format_slot_a_cell(node, staging_enabled)
        self.table.setItem(r, 3, _mk(trim_txt, trim_col, bold=trim_bold))

        ins_col = COLORS["attention"] if ins_hot else (COLORS["clean"] if (staged and ins_pk is not None) else name_col)
        self.table.setItem(r, 4, _mk(_format_pk_rms_pair(ins_pk, ins_rms, rms_off), ins_col, bold=(staged or ins_hot)))

        if staged and staging_enabled and not blocked:
            self._set_fader_editor_cell(
                r, name, _safe_float(node.get("recommended_fader_db")),
                _safe_float(node.get("auto_fader_db")), overridden,
            )
        else:
            self.table.setItem(r, 5, _mk(_format_non_staged_fader(node, staging_enabled), meta_col))

        out_col = COLORS["attention"] if out_hot else (COLORS["clean"] if (ttype == "Master" and out_pk is not None) else name_col)
        self.table.setItem(r, 6, _mk(_format_pk_rms_pair(out_pk, out_rms, rms_off), out_col, bold=(out_hot or ttype == "Master")))

        net_txt, net_col, net_bold = self._format_net_delta_cell(node, blocked=blocked, staged=staged, overridden=overridden, meta_col=meta_col)
        self.table.setItem(r, 7, _mk(net_txt, net_col, bold=net_bold))
        self.table.setItem(r, 8, _mk(_node_notes(node, staging_enabled=staging_enabled), COLORS["problems"] if blocked else meta_col))

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

    def _update_path_bar(self, node_name: str | None) -> None:
        if self._last_graph is None or not node_name:
            self.path_bar.hide()
            return
        trace = trace_signal_flow_path(self._last_graph, node_name)
        if not trace.equation_text:
            self.path_bar.hide()
            return
        if trace.total_delta_db is not None:
            sig_s = _format_db(trace.total_delta_db, signed=True)
            val = self._last_graph.get("balance_validation") or {}
            mark = "\u2713" if val.get("status") in ("verified", "converged_shifted", "caveat") else "\u26a0"
            self.path_bar.setText(f"Path: {trace.equation_text}   \u03a3 = {sig_s} {mark}")
        else:
            self.path_bar.setText(f"Path: {trace.equation_text}")
        self.path_bar.show()

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
        master = next((n for n in nodes.values() if isinstance(n, dict) and n.get("track_type") == "Master"), None)
        master_pk = None
        if master is not None:
            master_pk = master.get("post_fader_peak_db")
            if master_pk is None:
                master_pk = master.get("pre_fader_peak_db")
        master_txt = f" \u00b7 Master Peak: {_format_db(master_pk)}" if master_pk is not None else ""
        scale_txt = "AES17 (+3.01 dB)" if conv == "aes17" else "Standard dBFS"
        self.footer_label.setText(
            f"Staged on Create: {staged_txt}{master_txt} \u00b7 Scale: {scale_txt} \u00b7 Atomic post-close .ptx pass with rollback"
        )
        self.footer_label.show()
