"""Auto-routed semantic-tier Signal Flow DAG QGraphicsView for Phase 3."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFocusEvent,
    QFont,
    QFontMetrics,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.audio import dbfs_offset
from sessionpreplib.ptx.signal_flow_analysis import (
    GraphLayoutEdge,
    GraphLayoutNode,
    PathTraceResult,
    SignalFlowGraphLayout,
    compute_signal_flow_graph_layout,
    trace_signal_flow_path,
)

from ..theme import COLORS, FILE_COLOR_SUSTAINED, FILE_COLOR_TRANSIENT

_ZOOM_MIN = 0.45
_ZOOM_MAX = 1.85
_ZOOM_STEP = 1.15
_OVERRIDE_BG = "#4a3b1f"
_OVERRIDE_BORDER = "#ffaa00"
_BLOCKED_BORDER = "#ff4444"

_ROLE_BG: dict[str, str] = {
    "leaf": "#23262b",
    "folder": "#23262b",
    "bus": "#242a32",
    "terminal": "#1f2c38",
    "exempt": "#23262b",
    "master": "#222a30",
    "blocked": "#2e1a1a",
}
_ROLE_BORDER: dict[str, str] = {
    "leaf": "#3a414c",
    "folder": "#3a414c",
    "bus": "#3e4c5e",
    "terminal": "#2a6db5",
    "exempt": "#3a414c",
    "master": "#3e5260",
    "blocked": _BLOCKED_BORDER,
}
_ZOOM_BTN_QSS = (
    f"QPushButton {{ background-color: {COLORS['bg_alt']}; color: {COLORS['text']}; "
    f"border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 1px 6px; min-height: 22px; }}"
    f"QPushButton:hover {{ background-color: #3a3a3a; }}"
    f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


def _fmt_num(val: Any, offset: float = 0.0, *, signed: bool = False) -> str:
    if val is None or isinstance(val, bool):
        return "\u2014"
    try:
        num = float(val) + float(offset)
    except (TypeError, ValueError):
        return "\u2014"
    if not math.isfinite(num):
        return "\u2014"
    if abs(num) < 0.05:
        return "0.0"
    return f"{num:+.1f}" if signed else f"{num:.1f}"


def _fmt_pair(pk: Any, rms: Any, rms_off: float) -> str:
    if pk is None and rms is None:
        return "\u2014"
    return f"{_fmt_num(pk)} / {_fmt_num(rms, rms_off)}"


def _role_badge(role: str, node: dict[str, Any]) -> tuple[str, str]:
    if role == "blocked":
        return "SLOT A BLOCKED", COLORS["problems"]
    if node.get("is_overridden"):
        diff = _safe_float(node.get("recommended_fader_db")) - _safe_float(node.get("auto_fader_db"))
        return f"{_fmt_num(diff, signed=True)} \u26a0", _OVERRIDE_BORDER
    if role == "terminal":
        return "TERMINAL", COLORS["information"]
    if role == "exempt":
        return "POST-TERM", COLORS["metadata"]
    if role == "master":
        return "MASTER", COLORS["metadata"]
    cls = str(node.get("classification", ""))
    if cls == "Transient":
        return "Trans", FILE_COLOR_TRANSIENT.name()
    if cls == "Sustained":
        return "Sust", FILE_COLOR_SUSTAINED.name()
    return "", COLORS["metadata"]


def _card_metric_lines(
    role: str,
    node: dict[str, Any],
    rms_off: float,
    target_pk: float,
    staging_enabled: bool,
) -> list[tuple[str, str, str]]:
    """Return ``(label, value, color_hex)`` lines for one DAG node card."""
    if role == "blocked":
        in_txt = _fmt_pair(node.get("pre_insert_peak_db"), node.get("pre_insert_rms_db"), rms_off)
        return [
            ("In", in_txt, COLORS["text"]),
            ("Slot A", "Occupied / Blocked", COLORS["problems"]),
            ("Action", "Move plugin to Slot B", COLORS["attention"]),
        ]
    if role == "leaf":
        baked = _safe_float(node.get("baked_gain_db"))
        cg = _safe_float(node.get("clip_gain_db"))
        fdr = _fmt_num(node.get("recommended_fader_db", 0.0), signed=True)
        net = _fmt_num(node.get("net_gain_delta_db", 0.0), signed=True)
        lbl = "Bk / Fdr" if (abs(baked) >= 0.05 and abs(cg) < 0.05) else "CG / Fdr"
        src_v = _fmt_num(baked if (abs(baked) >= 0.05 and abs(cg) < 0.05) else cg, signed=True)
        return [
            (lbl, f"{src_v} / {fdr} dB", COLORS["text"]),
            ("Out / \u0394", f"{_fmt_num(node.get('post_fader_peak_db'))} pk \u00b7 {net}", COLORS["metadata"]),
        ]
    in_pk = node.get("pre_insert_peak_db") if node.get("pre_insert_peak_db") is not None else node.get("pre_fader_peak_db")
    in_rms = node.get("pre_insert_rms_db") if node.get("pre_insert_rms_db") is not None else node.get("pre_fader_rms_db")
    out_pk, out_rms = node.get("post_fader_peak_db"), node.get("post_fader_rms_db")
    net_val = _safe_float(node.get("net_gain_delta_db"))
    if role in ("folder", "exempt"):
        return [
            ("In", _fmt_pair(in_pk, in_rms, rms_off), COLORS["text"]),
            ("Out", f"{_fmt_pair(out_pk, out_rms, rms_off)} ({_fmt_num(net_val, signed=True)})", COLORS["metadata"]),
        ]
    if role == "master":
        hot = out_pk is not None and _safe_float(out_pk, -999.0) > target_pk + 0.05
        return [
            ("In", _fmt_pair(in_pk, in_rms, rms_off), COLORS["metadata"]),
            ("Post", _fmt_pair(out_pk, out_rms, rms_off), COLORS["attention"] if hot else COLORS["clean"]),
            ("Mode", "Annotated only", COLORS["metadata"]),
        ]
    ins_pk, ins_rms = node.get("pre_fader_peak_db"), node.get("pre_fader_rms_db")
    ins_hot = ins_pk is not None and _safe_float(ins_pk, -999.0) > target_pk + 0.05
    strat = str(node.get("staging_strategy", "none"))
    if staging_enabled and node.get("staged") and strat in ("trim_and_fader", "trim_only"):
        auto_tag = " (auto)" if node.get("slot_a_trim_auto_insert") else ""
        trim_txt = f"{_fmt_num(node.get('recommended_trim_db', 0.0), signed=True)} dB{auto_tag}"
        trim_col = FILE_COLOR_TRANSIENT.name()
    else:
        trim_txt, trim_col = "\u2014", COLORS["metadata"]
    fdr_txt = f"{_fmt_num(node.get('recommended_fader_db', 0.0), signed=True)} dB" + (" *" if node.get("is_overridden") else "")
    fdr_col = _OVERRIDE_BORDER if node.get("is_overridden") else COLORS["text"]
    net_col = (
        FILE_COLOR_TRANSIENT.name() if role == "terminal"
        else (_OVERRIDE_BORDER if (node.get("is_overridden") or abs(net_val) > 0.15) else COLORS["clean"])
    )
    return [
        ("In", _fmt_pair(in_pk, in_rms, rms_off), COLORS["metadata"]),
        ("Slot A", trim_txt, trim_col),
        ("Ins", _fmt_pair(ins_pk, ins_rms, rms_off), COLORS["attention"] if ins_hot else COLORS["clean"]),
        ("Fdr", fdr_txt, fdr_col),
        ("Out", f"{_fmt_num(out_pk)}/{_fmt_num(out_rms, rms_off)} \u0394{_fmt_num(net_val, signed=True)}", net_col),
    ]


class _NodeCardItem(QGraphicsRectItem):
    """Interactive semantic-tier node card rendered inside SignalFlowGraphView."""

    def __init__(
        self,
        layout_node: GraphLayoutNode,
        node_dict: dict[str, Any],
        *,
        rms_off: float,
        target_pk: float,
        staging_enabled: bool,
        on_click: Any,
        on_double_click: Any,
        on_hover: Any,
    ) -> None:
        super().__init__(0.0, 0.0, layout_node.width, layout_node.height)
        self.setPos(layout_node.x, layout_node.y)
        self.layout_node = layout_node
        self.node_dict = node_dict
        self.node_name = layout_node.name
        self._lines = _card_metric_lines(layout_node.role, node_dict, rms_off, target_pk, staging_enabled)
        self._badge_txt, self._badge_col = _role_badge(layout_node.role, node_dict)
        self._on_click = on_click
        self._on_double_click = on_double_click
        self._on_hover = on_hover
        self._is_selected = False
        self._is_highlighted = False
        self._is_dimmed = False
        self._is_focused = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_tooltip()

    def boundingRect(self) -> QRectF:
        return self.rect().adjusted(-4.0, -4.0, 4.0, 4.0)

    def _init_tooltip(self) -> None:
        n = self.node_dict
        if n.get("slot_a_blocked_error"):
            self.setToolTip(str(n["slot_a_blocked_error"]))
            return
        lines = [f"{self.node_name} ({n.get('track_type', '')})"]
        if n.get("slot_a_trim_plugin"):
            lines.append(f"Slot A: {n['slot_a_trim_plugin']} ({_fmt_num(n.get('recommended_trim_db', 0.0), signed=True)} dB)")
        if n.get("active_inserts"):
            lines.append("Inserts: " + ", ".join(str(x) for x in n["active_inserts"]))
        if n.get("staged"):
            lines.append("Double-click to edit bus fader override")
        self.setToolTip("\n".join(lines))

    def set_visual_state(self, *, selected: bool, highlighted: bool, dimmed: bool, focused: bool) -> None:
        self._is_selected = selected
        self._is_highlighted = highlighted
        self._is_dimmed = dimmed
        self._is_focused = focused
        self.setOpacity(0.42 if dimmed else 1.0)
        self.update()

    def hoverEnterEvent(self, event: Any) -> None:
        self._on_hover(self.node_name)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: Any) -> None:
        self._on_hover(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self.node_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_double_click(self.node_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.save()
        rect = self.rect()
        role = self.layout_node.role
        overridden = bool(self.node_dict.get("is_overridden")) and role != "blocked"
        base_bg = _OVERRIDE_BG if overridden else _ROLE_BG.get(role, "#23262b")
        bg = QColor("#1f2e42" if (self._is_selected and not overridden and role != "blocked") else base_bg)
        if self._is_selected or self._is_highlighted:
            border_col, border_w = QColor(COLORS["information"]), (2.0 if self._is_selected else 1.5)
        elif role == "blocked":
            border_col, border_w = QColor(_BLOCKED_BORDER), 2.0
        elif overridden:
            border_col, border_w = QColor(_OVERRIDE_BORDER), 1.5
        else:
            border_col, border_w = QColor(_ROLE_BORDER.get(role, "#3a414c")), 1.0

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border_col, border_w))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4.0, 4.0)

        if self._is_focused:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(COLORS["information"]), 2.0))
            painter.drawRoundedRect(rect.adjusted(-2.5, -2.5, 2.5, 2.5), 5.0, 5.0)

        badge_w = 0.0
        if self._badge_txt:
            b_font = QFont(painter.font())
            b_font.setPointSizeF(7.0)
            b_font.setBold(True)
            painter.setFont(b_font)
            badge_w = float(QFontMetrics(b_font).horizontalAdvance(self._badge_txt)) + 4.0
            painter.setPen(QColor(self._badge_col))
            painter.drawText(
                QRectF(rect.width() - badge_w - 6.0, 4.0, badge_w, 15.0),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._badge_txt,
            )

        hdr_font = QFont(painter.font())
        hdr_font.setPointSizeF(8.5)
        hdr_font.setBold(True)
        painter.setFont(hdr_font)
        painter.setPen(QColor(COLORS["heading"]))
        title_w = max(24.0, rect.width() - 14.0 - (badge_w + 6.0 if badge_w > 0 else 0.0))
        elided = painter.fontMetrics().elidedText(self.node_name, Qt.TextElideMode.ElideRight, int(title_w))
        painter.drawText(QRectF(7.0, 4.0, title_w, 15.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)

        painter.setPen(QPen(QColor("#333840"), 1.0))
        painter.drawLine(QPointF(6.0, 21.0), QPointF(rect.width() - 6.0, 21.0))

        body_font = QFont(painter.font())
        body_font.setPointSizeF(7.5)
        body_font.setBold(False)
        painter.setFont(body_font)
        y = 24.0
        step_y = 15.5 if len(self._lines) >= 4 else 16.5
        for lbl, val, col_hex in self._lines:
            painter.setPen(QColor(COLORS["metadata"]))
            painter.drawText(QRectF(7.0, y, 42.0, 14.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, lbl)
            painter.setPen(QColor(col_hex))
            val_elided = painter.fontMetrics().elidedText(val, Qt.TextElideMode.ElideRight, int(rect.width() - 54.0))
            painter.drawText(QRectF(48.0, y, rect.width() - 54.0, 14.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, val_elided)
            y += step_y
        painter.restore()


class _DagGraphicsView(QGraphicsView):
    """Pannable, zoomable, keyboard-accessible DAG viewport."""

    zoom_changed = Signal(int)
    node_focus_changed = Signal(str)
    node_activated = Signal(str)
    selection_cleared = Signal()
    focus_state_changed = Signal()

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self._zoom = 1.0
        self._order: list[str] = []
        self._nodes: dict[str, GraphLayoutNode] = {}
        self._focused_node: str | None = None
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(
            f"QGraphicsView {{ background-color: {COLORS['bg']}; border: 1px solid {COLORS['accent']}; border-radius: 2px; }}"
            f"QGraphicsView:focus {{ border: 2px solid {COLORS['information']}; }}"
        )

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.ShortcutOverride and isinstance(ev, QKeyEvent):
            if ev.key() in (Qt.Key.Key_Space, Qt.Key.Key_Escape):
                ev.accept()
                return True
        return super().event(ev)

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self.focus_state_changed.emit()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.focus_state_changed.emit()

    def set_navigation_graph(self, order: list[str], nodes: dict[str, GraphLayoutNode]) -> None:
        self._order = [n for n in order if n in nodes]
        self._nodes = nodes
        if self._focused_node not in nodes:
            self._focused_node = self._order[0] if self._order else None

    @property
    def focused_node(self) -> str | None:
        return self._focused_node

    def set_focused_node(self, name: str | None, *, emit: bool = True) -> None:
        if name is None or name in self._nodes:
            self._focused_node = name
            if emit and name is not None:
                self.node_focus_changed.emit(name)

    def set_zoom(self, scale: float) -> None:
        clamped = max(_ZOOM_MIN, min(_ZOOM_MAX, float(scale)))
        self._zoom = clamped
        self.resetTransform()
        self.scale(clamped, clamped)
        self.zoom_changed.emit(int(round(clamped * 100)))

    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def fit_contents(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        if bounds.isEmpty():
            self.set_zoom(1.0)
            return
        self.fitInView(bounds.adjusted(-16, -16, 16, 16), Qt.AspectRatioMode.KeepAspectRatio)
        eff = max(_ZOOM_MIN, min(_ZOOM_MAX, self.transform().m11()))
        self._zoom = eff
        self.resetTransform()
        self.scale(eff, eff)
        self.zoom_changed.emit(int(round(eff * 100)))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            self.set_zoom(self._zoom * (_ZOOM_STEP if delta > 0 else 1.0 / _ZOOM_STEP))
            event.accept()
            return
        super().wheelEvent(event)

    def _move_focus_direction(self, key: int) -> None:
        if not self._order:
            return
        if self._focused_node is None or self._focused_node not in self._nodes:
            self.set_focused_node(self._order[0])
            return
        cur = self._nodes[self._focused_node]
        cx, cy = cur.x + cur.width * 0.5, cur.y + cur.height * 0.5
        best_name, best_score = None, float("inf")
        for nm, cand in self._nodes.items():
            if nm == cur.name:
                continue
            dx = (cand.x + cand.width * 0.5) - cx
            dy = (cand.y + cand.height * 0.5) - cy
            if key == Qt.Key.Key_Right and dx <= 10.0:
                continue
            if key == Qt.Key.Key_Left and dx >= -10.0:
                continue
            if key == Qt.Key.Key_Down and (cand.column != cur.column or dy <= 4.0):
                continue
            if key == Qt.Key.Key_Up and (cand.column != cur.column or dy >= -4.0):
                continue
            score = abs(dx) + abs(dy) * (2.5 if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) else 0.5)
            if score < best_score:
                best_score, best_name = score, nm
        if best_name is not None:
            self.set_focused_node(best_name)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mods = event.modifiers()
        if (mods & Qt.KeyboardModifier.ControlModifier) and key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.set_zoom(self._zoom * _ZOOM_STEP)
            event.accept()
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Minus:
            self.set_zoom(self._zoom / _ZOOM_STEP)
            event.accept()
            return
        if key == Qt.Key.Key_0 and (mods & Qt.KeyboardModifier.ControlModifier or mods == Qt.KeyboardModifier.NoModifier):
            self.set_zoom(1.0)
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._move_focus_direction(key)
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self._focused_node:
            self.node_activated.emit(self._focused_node)
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.selection_cleared.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _build_edge_path(src: GraphLayoutNode, dst: GraphLayoutNode, edge: GraphLayoutEdge) -> tuple[QPainterPath, QPolygonF]:
    path = QPainterPath()
    if edge.is_intra_column:
        p1 = QPointF(src.x + src.width * 0.5, src.y + src.height)
        p2 = QPointF(dst.x + dst.width * 0.5, dst.y)
        path.moveTo(p1)
        path.lineTo(p2)
        arrow = QPolygonF([p2, QPointF(p2.x() - 4.0, p2.y() - 7.0), QPointF(p2.x() + 4.0, p2.y() - 7.0)])
        return path, arrow
    p1 = QPointF(src.x + src.width, src.y + src.height * 0.5)
    p2 = QPointF(dst.x, dst.y + dst.height * 0.5)
    dx = max(20.0, abs(p2.x() - p1.x()) * 0.45)
    path.moveTo(p1)
    path.cubicTo(QPointF(p1.x() + dx, p1.y()), QPointF(p2.x() - dx, p2.y()), p2)
    arrow = QPolygonF([p2, QPointF(p2.x() - 7.0, p2.y() - 4.0), QPointF(p2.x() - 7.0, p2.y() + 4.0)])
    return path, arrow


class SignalFlowGraphView(QWidget):
    """Auto-routed Signal Flow DAG widget with zoom toolbar, path highlight, and node editing."""

    node_selected = Signal(str)
    selection_cleared = Signal()
    node_edit_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._graph: dict[str, Any] | None = None
        self._layout: SignalFlowGraphLayout | None = None
        self._staging_enabled = True
        self._selected_node: str | None = None
        self._hovered_node: str | None = None
        self._card_items: dict[str, _NodeCardItem] = {}
        self._edge_items: list[tuple[tuple[str, str], bool, QGraphicsPathItem, QGraphicsItem]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.toolbar = self._build_zoom_bar()
        root.addWidget(self.toolbar)

        self.scene = QGraphicsScene(self)
        self.view = _DagGraphicsView(self.scene, self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.zoom_changed.connect(lambda pct: self.zoom_reset_btn.setText(f"{pct}%"))
        self.view.node_focus_changed.connect(self._on_focus_changed)
        self.view.node_activated.connect(self._on_node_activated)
        self.view.selection_cleared.connect(lambda: self.select_node(None, emit=True))
        self.view.focus_state_changed.connect(lambda: self._apply_highlight(self._active_highlight_node()))
        root.addWidget(self.view, 1)

    def _build_zoom_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(4)

        self.zoom_out_btn = QPushButton("\u2212")
        self.zoom_out_btn.setAccessibleName("Zoom out signal flow graph")
        self.zoom_out_btn.setFixedWidth(26)
        self.zoom_out_btn.setStyleSheet(_ZOOM_BTN_QSS)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        lay.addWidget(self.zoom_out_btn)

        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_reset_btn.setAccessibleName("Reset signal flow graph zoom to 100%")
        self.zoom_reset_btn.setFixedWidth(50)
        self.zoom_reset_btn.setStyleSheet(_ZOOM_BTN_QSS)
        self.zoom_reset_btn.clicked.connect(self.zoom_reset)
        lay.addWidget(self.zoom_reset_btn)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setAccessibleName("Zoom in signal flow graph")
        self.zoom_in_btn.setFixedWidth(26)
        self.zoom_in_btn.setStyleSheet(_ZOOM_BTN_QSS)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        lay.addWidget(self.zoom_in_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setAccessibleName("Fit signal flow graph to view")
        self.fit_btn.setStyleSheet(_ZOOM_BTN_QSS)
        self.fit_btn.clicked.connect(self.fit_to_view)
        lay.addWidget(self.fit_btn)

        self.hint_label = QLabel("Click node to trace path \u00b7 Double-click staged bus to edit fader")
        self.hint_label.setStyleSheet(f"color: {COLORS['metadata']}; border: none; background: transparent;")
        self.hint_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.hint_label, 1)
        return bar

    def zoom_in(self) -> None:
        self.view.set_zoom(self.view._zoom * _ZOOM_STEP)

    def zoom_out(self) -> None:
        self.view.set_zoom(self.view._zoom / _ZOOM_STEP)

    def zoom_reset(self) -> None:
        self.view.set_zoom(1.0)

    def fit_to_view(self) -> None:
        self.view.fit_contents()

    def set_graph(
        self,
        graph_dict: dict[str, Any],
        *,
        active_only: bool = True,
        staging_enabled: bool = True,
        selected_node: str | None = None,
    ) -> None:
        self._graph = graph_dict
        self._staging_enabled = staging_enabled
        self._hovered_node = None
        self._layout = compute_signal_flow_graph_layout(graph_dict, active_only=active_only)
        self._rebuild_scene(graph_dict, self._layout, staging_enabled)
        order = list(graph_dict.get("topological_order") or self._layout.nodes.keys())
        self.view.set_navigation_graph(order, self._layout.nodes)
        target_sel = selected_node if selected_node in self._card_items else None
        self.select_node(target_sel, emit=False)

    def _rebuild_scene(
        self,
        graph_dict: dict[str, Any],
        layout: SignalFlowGraphLayout,
        staging_enabled: bool,
    ) -> None:
        self.scene.clear()
        self._card_items.clear()
        self._edge_items.clear()
        conv = str(graph_dict.get("dbfs_convention", "standard"))
        rms_off = dbfs_offset({"dbfs_convention": conv})
        target_pk = _safe_float(graph_dict.get("target_peak_db", -6.0), -6.0)
        nodes_map: dict[str, dict[str, Any]] = graph_dict.get("nodes") or {}

        hdr_font = QFont()
        hdr_font.setPointSizeF(7.5)
        hdr_font.setBold(True)
        for _col, title, cx, _cw in layout.columns:
            lbl = QGraphicsSimpleTextItem(title)
            lbl.setFont(hdr_font)
            lbl.setBrush(QBrush(QColor(COLORS["metadata"])))
            lbl.setPos(cx, 12.0)
            self.scene.addItem(lbl)

        for edge in layout.edges:
            src_n, dst_n = layout.nodes.get(edge.source), layout.nodes.get(edge.target)
            if src_n is None or dst_n is None:
                continue
            path, arrow_poly = _build_edge_path(src_n, dst_n, edge)
            base_col = QColor(_OVERRIDE_BORDER if edge.is_pre_fader else "#565e68")
            pen = QPen(base_col, 1.4, Qt.PenStyle.DashLine if edge.is_pre_fader else Qt.PenStyle.SolidLine)
            e_item = self.scene.addPath(path, pen)
            e_item.setZValue(1.0)
            a_item = self.scene.addPolygon(arrow_poly, QPen(Qt.PenStyle.NoPen), QBrush(base_col))
            a_item.setZValue(1.5)
            self._edge_items.append(((edge.source, edge.target), edge.is_pre_fader, e_item, a_item))

        for name, lnode in layout.nodes.items():
            ndict = nodes_map.get(name) or {"track_name": name}
            card = _NodeCardItem(
                lnode, ndict, rms_off=rms_off, target_pk=target_pk, staging_enabled=staging_enabled,
                on_click=lambda nm=name: self.select_node(nm, emit=True),
                on_double_click=self._on_card_double_clicked,
                on_hover=self._on_card_hover,
            )
            card.setZValue(2.0)
            self.scene.addItem(card)
            self._card_items[name] = card

        self.scene.setSceneRect(0.0, 0.0, layout.canvas_width, layout.canvas_height)

    def _active_highlight_node(self) -> str | None:
        return self._hovered_node if self._hovered_node is not None else self._selected_node

    def _on_card_hover(self, name: str | None) -> None:
        if name == self._hovered_node:
            return
        self._hovered_node = name
        self._apply_highlight(self._active_highlight_node())

    def _on_card_double_clicked(self, name: str) -> None:
        self.select_node(name, emit=True)
        if self._staging_enabled and self._graph is not None:
            node = (self._graph.get("nodes") or {}).get(name)
            if isinstance(node, dict) and node.get("staged") and not node.get("slot_a_blocked_error"):
                self.node_edit_requested.emit(name)

    def _on_focus_changed(self, name: str) -> None:
        self._apply_highlight(self._active_highlight_node())
        card = self._card_items.get(name)
        if card is not None:
            self.view.ensureVisible(card, 24, 24)

    def _on_node_activated(self, name: str) -> None:
        if self._selected_node == name:
            self._on_card_double_clicked(name)
        else:
            self.select_node(name, emit=True)

    def select_node(self, name: str | None, *, emit: bool = True) -> PathTraceResult | None:
        self._selected_node = name if (name and name in self._card_items) else None
        if self._selected_node is not None:
            self.view.set_focused_node(self._selected_node, emit=False)
        trace = self._apply_highlight(self._active_highlight_node())
        if emit:
            if self._selected_node is not None:
                self.node_selected.emit(self._selected_node)
            else:
                self.selection_cleared.emit()
        return trace

    def _apply_highlight(self, active: str | None) -> PathTraceResult | None:
        trace: PathTraceResult | None = None
        if active and self._graph is not None:
            trace = trace_signal_flow_path(self._graph, active)
        hi_nodes = trace.highlighted_nodes if trace is not None else frozenset()
        hi_edges = trace.highlighted_edges if trace is not None else frozenset()
        focused = self.view.focused_node if self.view.hasFocus() else None

        for nm, card in self._card_items.items():
            card.set_visual_state(
                selected=(nm == self._selected_node),
                highlighted=(nm in hi_nodes),
                dimmed=(active is not None and nm not in hi_nodes),
                focused=(nm == focused),
            )
        for key, is_pre, e_item, a_item in self._edge_items:
            is_hi = key in hi_edges
            if is_pre:
                col = QColor(_OVERRIDE_BORDER)
            else:
                col = QColor(COLORS["information"] if is_hi else ("#3a414b" if active is not None else "#565e68"))
            pen = QPen(col, 2.0 if is_hi else 1.3, Qt.PenStyle.DashLine if is_pre else Qt.PenStyle.SolidLine)
            e_item.setPen(pen)
            e_item.setZValue(1.8 if is_hi else 1.0)
            if isinstance(a_item, QGraphicsItem):
                a_item.setZValue(1.9 if is_hi else 1.2)
            if hasattr(a_item, "setBrush"):
                a_item.setBrush(QBrush(col))
        return trace
