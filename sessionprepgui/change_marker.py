"""Shared vector change marker for modified pages, tabs and commands.

One information-blue dot (7 logical px in a 16 px slot) marks a page, tab or
command whose values differ from their saved or preset state. It is painted
from geometry at the target device-pixel ratio, so it stays crisp at every
application scale, and turns white on selected or pressed backgrounds. Tabs
and buttons keep their normal width and widen by a 4 px gap plus the dot
only while the marker is shown.
"""
from __future__ import annotations

import shiboken6
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
        _ENGINES[:] = [e for e in _ENGINES if shiboken6.isValid(e)]
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
    """A gap plus a 7 px dot after a button label; painted only when marked."""

    def __init__(self, parent=None, *, on_selection=None):
        super().__init__(parent)
        self._marked = False
        self._on_selection = on_selection
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(MARKER_GAP + MARKER_DOT, MARKER_SLOT)

    def is_marked(self) -> bool:
        return self._marked

    def set_marked(self, marked: bool) -> None:
        marked = bool(marked)
        if marked != self._marked:
            self._marked = marked
            self.setVisible(marked)
            self.update()

    def paintEvent(self, event):
        if not self._marked or not self.isEnabled():
            return
        selected = bool(self._on_selection and self._on_selection())
        painter = QPainter(self)
        _paint_dot(painter, _dot_slot(MARKER_GAP + MARKER_DOT / 2, self.height() / 2),
                   COLORS["heading"] if selected else COLORS["information"])
        painter.end()


def _dot_slot(center_x: float, center_y: float) -> QRectF:
    """The 16 px slot whose centred 7 px dot sits at the given centre."""
    return QRectF(center_x - MARKER_SLOT / 2, center_y - MARKER_SLOT / 2, MARKER_SLOT, MARKER_SLOT)


_MARKABLE = "sessionprep-change-marker"


class _TabSpacer(QWidget):
    """Invisible right-side tab button that widens a marked tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(MARKER_GAP + MARKER_DOT, MARKER_SLOT)


class MarkerTabBar(QTabBar):
    """Tab bar that draws change markers directly after a tab's label.

    Unmarked tabs keep their normal width. A marked tab gets an invisible
    spacer button sized so the tab is exactly one marker gap + dot wider than
    the style's tab for its label alone; the bar paints the dot right after
    the style's label rect, so geometry follows the theme. Neighbouring tabs
    shift while a marker is shown. Markability travels with the tab (tab data).
    """

    def make_markable(self, index: int) -> None:
        self.setTabData(index, _MARKABLE)

    def is_markable(self, index: int) -> bool:
        return self.tabData(index) == _MARKABLE

    def set_marked(self, index: int, marked: bool) -> None:
        current = self.tabButton(index, QTabBar.RightSide)
        if marked and not isinstance(current, _TabSpacer):
            self.setTabButton(index, QTabBar.RightSide, _TabSpacer(self))
            self._sync_spacers()
        elif not marked and isinstance(current, _TabSpacer):
            self.setTabButton(index, QTabBar.RightSide, None)
            current.hide()
            current.deleteLater()
        self.update()

    def is_marked(self, index: int) -> bool:
        return isinstance(self.tabButton(index, QTabBar.RightSide), _TabSpacer)

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
            # and margin), plus exactly one marker gap and dot.
            option = self._option(index)
            style = self.style()
            content = QSize(
                self._label_width(index) + style.pixelMetric(QStyle.PM_TabBarTabHSpace, option, self),
                self.fontMetrics().height() + style.pixelMetric(QStyle.PM_TabBarTabVSpace, option, self))
            plain = style.sizeFromContents(QStyle.CT_TabBarTab, option, content, self).width()
            without_spacer = super().tabSizeHint(index).width() - spacer.width()
            width = max(1, plain + MARKER_GAP + MARKER_DOT - without_spacer)
            if width != spacer.width():
                spacer.setFixedSize(width, MARKER_SLOT)
                # Resizing a side button does not re-lay out the tabs; attaching
                # it again does (detach first: re-setting the same widget hides it).
                self.setTabButton(index, QTabBar.RightSide, None)
                self.setTabButton(index, QTabBar.RightSide, spacer)

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
                       rect.right() + 1 - MARKER_DOT)
            _paint_dot(painter, _dot_slot(left + MARKER_DOT / 2, rect.center().y() + 0.5),
                       COLORS["information"])
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
    """Allow a change marker after a tab's label (requires use_marker_tab_bar)."""
    bar = _marker_bar(tabs)
    if bar is None:
        raise RuntimeError("tab markers need use_marker_tab_bar()")
    bar.make_markable(index)


def set_tab_marker(tabs: QTabWidget, index: int, marked: bool, tooltip: str) -> None:
    """Show or clear a tab's marker; disabled tabs never show it.

    Call it again after enabling or disabling the tab: a marked tab that is
    disabled later keeps its widened width until the marker is refreshed.
    """
    bar = _marker_bar(tabs)
    if bar is None or not bar.is_markable(index):
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
    """Allow a change marker after a push button's label.

    While marked, the button's right padding grows by the gap + dot (the
    button widens; in a right-aligned button box its right edge stays put).
    The button's own stylesheet is kept: marking appends the padding rule and
    unmarking removes exactly that rule from the button's current sheet.
    """
    slot = ChangeMarkerSlot(button, on_selection=button.isDown)
    slot.hide()
    button.pressed.connect(slot.update)
    button.released.connect(slot.update)
    placer = _ButtonMarkerFilter(button, slot)
    button.installEventFilter(placer)
    placer.place()
    return slot


_MARKED_BUTTON_RULE = (
    f"QPushButton {{ padding-right: {_BUTTON_PADDING + MARKER_GAP + MARKER_DOT}px; }}")


def _button_slot(button: QPushButton) -> ChangeMarkerSlot | None:
    return button.findChild(ChangeMarkerSlot, options=Qt.FindDirectChildrenOnly)


def set_button_marker(button: QPushButton, marked: bool, tooltip: str) -> None:
    slot = _button_slot(button)
    if slot is None:
        return
    marked = bool(marked)
    if marked != slot.is_marked():
        sheet = button.styleSheet()
        if marked:
            sheet = f"{sheet}\n{_MARKED_BUTTON_RULE}" if sheet else _MARKED_BUTTON_RULE
        else:
            sheet = sheet.replace(f"\n{_MARKED_BUTTON_RULE}", "").replace(_MARKED_BUTTON_RULE, "")
        button.setStyleSheet(sheet)
        slot.set_marked(marked)
        button.updateGeometry()  # the resize that follows re-places the dot
        placer = button.findChild(_ButtonMarkerFilter, options=Qt.FindDirectChildrenOnly)
        if placer is not None:
            placer.place()
    button.setToolTip(tooltip if marked else "")
    button.setAccessibleDescription(tooltip if marked else "")


def is_button_marked(button: QPushButton) -> bool:
    slot = _button_slot(button)
    return bool(slot and slot.is_marked())
