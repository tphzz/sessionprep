"""Shared vector change marker for modified pages, tabs and commands.

One information-blue dot (7 logical px in a 16 px slot) marks a page, tab or
command whose values differ from their saved or preset state. It is painted
from geometry at the target device-pixel ratio, so it stays crisp at every
application scale, and turns white on selected or pressed backgrounds. Tabs
and buttons always reserve the slot so nothing reflows when it toggles.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPixmap
from PySide6.QtWidgets import (
    QPushButton, QStyle, QStyleOptionTab, QTabBar, QTabWidget, QTreeWidgetItem, QWidget,
)

from .theme import COLORS

MARKER_SLOT = 16
MARKER_DOT = 7
MARKER_GAP = 4
MARKER_COLUMN_WIDTH = 24
PARENT_TOOLTIP = "Contains changes"

_BUTTON_PADDING = 12  # matches the shared QPushButton horizontal padding


def _paint_dot(painter: QPainter, slot: QRectF, color: str) -> None:
    side = min(slot.width(), slot.height())
    diameter = side * MARKER_DOT / MARKER_SLOT
    dot = QRectF(slot.center().x() - diameter / 2, slot.center().y() - diameter / 2,
                 diameter, diameter)
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(dot)
    painter.restore()


# Qt owns engine copies by pointer; keep their Python wrappers alive so a
# modified copy of the icon never refers to a freed engine.
_ENGINES: list = []


class _ChangeMarkerEngine(QIconEngine):
    """Paint the marker at the requested size and device-pixel ratio."""

    def paint(self, painter, rect, mode, state):
        if mode == QIcon.Selected:
            color = COLORS["heading"]
        elif mode == QIcon.Disabled:
            color = COLORS["dim"]
        else:
            color = COLORS["information"]
        _paint_dot(painter, QRectF(rect), color)

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        scale = max(float(scale), 1.0)
        pixmap = QPixmap(QSize(max(1, round(size.width() * scale)),
                               max(1, round(size.height() * scale))))
        pixmap.fill(Qt.transparent)
        pixmap.setDevicePixelRatio(scale)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return pixmap

    def clone(self):
        engine = _ChangeMarkerEngine()
        _ENGINES.append(engine)
        return engine


_ICON: QIcon | None = None


def change_marker_icon() -> QIcon:
    """Return the shared vector marker icon (blue; white in Selected mode)."""
    global _ICON
    if _ICON is None:
        engine = _ChangeMarkerEngine()
        _ENGINES.append(engine)
        _ICON = QIcon(engine)
    return QIcon(_ICON)


def set_item_marker(item: QTreeWidgetItem, column: int, marked: bool,
                    tooltip: str) -> None:
    """Show or clear the marker in a tree item's marker column.

    Parents (items with children) explain that they contain changes.
    """
    text = (PARENT_TOOLTIP if item.childCount() else tooltip) if marked else ""
    item.setIcon(column, change_marker_icon() if marked else QIcon())
    item.setToolTip(column, text)
    item.setData(column, Qt.AccessibleDescriptionRole, text or None)


class ChangeMarkerSlot(QWidget):
    """A reserved slot that paints the marker when marked.

    A ``MARKER_GAP`` precedes the 16 px slot.
    """

    def __init__(self, parent=None, *, on_selection=None):
        super().__init__(parent)
        self._marked = False
        self._on_selection = on_selection
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(MARKER_GAP + MARKER_SLOT, MARKER_SLOT)

    def is_marked(self) -> bool:
        return self._marked

    def set_marked(self, marked: bool) -> None:
        marked = bool(marked)
        if marked != self._marked:
            self._marked = marked
            self.update()

    def paintEvent(self, event):
        if not self._marked or not self.isEnabled():
            return
        selected = bool(self._on_selection and self._on_selection())
        painter = QPainter(self)
        _paint_dot(painter, QRectF(MARKER_GAP, 0, MARKER_SLOT, MARKER_SLOT),
                   COLORS["heading"] if selected else COLORS["information"])
        painter.end()


class _TabSpacer(QWidget):
    """Invisible right-side tab button that reserves the marker slot width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.marked = False
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(MARKER_GAP + MARKER_SLOT, MARKER_SLOT)


class MarkerTabBar(QTabBar):
    """Tab bar that draws change markers directly after a tab's label.

    Qt lays a themed tab out as padding, label, spacing, padding, side button.
    A reserved tab gets an invisible spacer button sized so the tab is exactly
    one marker gap + slot wider than the style's tab for its label alone; the
    bar paints the dot right after the style's label rect, so geometry follows
    the theme and the tab never reflows when the marker toggles.
    """

    def reserve_marker(self, index: int) -> None:
        self.setTabButton(index, QTabBar.RightSide, _TabSpacer(self))
        self._sync_spacers()

    def is_reserved(self, index: int) -> bool:
        return isinstance(self.tabButton(index, QTabBar.RightSide), _TabSpacer)

    def set_marked(self, index: int, marked: bool) -> None:
        spacer = self.tabButton(index, QTabBar.RightSide)
        if isinstance(spacer, _TabSpacer):
            spacer.marked = bool(marked)
            self.update()

    def is_marked(self, index: int) -> bool:
        spacer = self.tabButton(index, QTabBar.RightSide)
        return isinstance(spacer, _TabSpacer) and spacer.marked

    def _label_width(self, index: int) -> int:
        return self.fontMetrics().size(Qt.TextShowMnemonic, self.tabText(index)).width()

    def _option(self, index: int) -> QStyleOptionTab:
        option = QStyleOptionTab()
        self.initStyleOption(option, index)
        return option

    def _sync_spacers(self) -> None:
        for index in range(self.count()):
            spacer = self.tabButton(index, QTabBar.RightSide)
            if not isinstance(spacer, _TabSpacer):
                continue
            # The tab the style would draw for the label alone (its own padding
            # and margin), plus exactly one marker gap and slot.
            plain = self.style().sizeFromContents(
                QStyle.CT_TabBarTab, self._option(index),
                QSize(self._label_width(index), self.fontMetrics().height()), self).width()
            without_spacer = super().tabSizeHint(index).width() - spacer.width()
            width = max(1, plain + MARKER_GAP + MARKER_SLOT - without_spacer)
            if width != spacer.width():
                spacer.setFixedSize(width, MARKER_SLOT)

    def event(self, event):
        if event.type() in (QEvent.Polish, QEvent.StyleChange, QEvent.FontChange, QEvent.Show):
            result = super().event(event)
            self._sync_spacers()
            return result
        return super().event(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for index in range(self.count()):
            if not self.is_marked(index) or not self.isTabEnabled(index):
                continue
            rect = self.tabRect(index)
            text = self.style().subElementRect(QStyle.SE_TabBarTabText, self._option(index), self)
            left = min(text.left() + min(self._label_width(index), text.width()) + MARKER_GAP,
                       rect.right() + 1 - MARKER_SLOT)
            slot = QRectF(left, rect.center().y() + 0.5 - MARKER_SLOT / 2, MARKER_SLOT, MARKER_SLOT)
            _paint_dot(painter, slot, COLORS["information"])
        painter.end()


def use_marker_tab_bar(tabs: QTabWidget) -> None:
    """Give an empty tab widget the marker-aware tab bar."""
    if tabs.count():
        raise RuntimeError("install the marker tab bar before adding tabs")
    tabs.setTabBar(MarkerTabBar())


def _marker_bar(tabs: QTabWidget) -> MarkerTabBar | None:
    bar = tabs.tabBar()
    return bar if isinstance(bar, MarkerTabBar) else None


def install_tab_marker(tabs: QTabWidget, index: int) -> None:
    """Reserve a marker slot after a tab's label (requires use_marker_tab_bar)."""
    bar = _marker_bar(tabs)
    if bar is None:
        raise RuntimeError("tab markers need use_marker_tab_bar()")
    bar.reserve_marker(index)


def set_tab_marker(tabs: QTabWidget, index: int, marked: bool, tooltip: str) -> None:
    """Show or clear a tab's marker; disabled tabs never show it."""
    bar = _marker_bar(tabs)
    if bar is None or not bar.is_reserved(index):
        return
    marked = bool(marked) and tabs.isTabEnabled(index)
    bar.set_marked(index, marked)
    tabs.setTabToolTip(index, tooltip if marked else "")
    if hasattr(bar, "setAccessibleTabName"):
        label = tabs.tabText(index)
        bar.setAccessibleTabName(index, f"{label}, {tooltip}" if marked else label)


def is_tab_marked(tabs: QTabWidget, index: int) -> bool:
    bar = _marker_bar(tabs)
    return bool(bar and bar.is_marked(index) and tabs.isTabEnabled(index))


class _ButtonMarkerFilter(QObject):
    def __init__(self, button: QPushButton, slot: ChangeMarkerSlot):
        super().__init__(button)
        self._button = button
        self._slot = slot

    def eventFilter(self, watched, event):
        if watched is self._button and event.type() in (QEvent.Resize, QEvent.Show):
            self.place()
        return False

    def place(self):
        button = self._button
        self._slot.move(button.width() - _BUTTON_PADDING - self._slot.width(),
                        (button.height() - self._slot.height()) // 2)


def install_button_marker(button: QPushButton) -> ChangeMarkerSlot:
    """Reserve a marker slot after a push button's label.

    The slot sits in the button's right padding (left-to-right layouts) and
    replaces any widget-level stylesheet on the button.
    """
    button.setStyleSheet(
        f"QPushButton {{ padding-right: {_BUTTON_PADDING + MARKER_GAP + MARKER_SLOT}px; }}")
    slot = ChangeMarkerSlot(button, on_selection=button.isDown)
    button.pressed.connect(slot.update)
    button.released.connect(slot.update)
    placer = _ButtonMarkerFilter(button, slot)
    button.installEventFilter(placer)
    placer.place()
    return slot


def _button_slot(button: QPushButton) -> ChangeMarkerSlot | None:
    return button.findChild(ChangeMarkerSlot, options=Qt.FindDirectChildrenOnly)


def set_button_marker(button: QPushButton, marked: bool, tooltip: str) -> None:
    slot = _button_slot(button)
    if slot is None:
        return
    slot.set_marked(marked)
    button.setToolTip(tooltip if marked else "")
    button.setAccessibleDescription(tooltip if marked else "")


def is_button_marked(button: QPushButton) -> bool:
    slot = _button_slot(button)
    return bool(slot and slot.is_marked())
