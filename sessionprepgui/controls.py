"""Shared desktop controls and action behavior."""
from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QDoubleSpinBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from .theme import CONTENT_INSET


class FormSpinBox(QSpinBox):
    """Integer input sized to match the other controls in a form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("formNumericEditor", True)


class FormDoubleSpinBox(QDoubleSpinBox):
    """Decimal input with the same sizing as the integer form input."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("formNumericEditor", True)


class FormActionButton(QPushButton):
    """An action that shares a row with a form input."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setProperty("formRowAction", True)


class FormBrowseButton(FormActionButton):
    """A path picker action with room for larger label fonts."""

    def __init__(self, parent=None):
        super().__init__("Browse\u2026", parent)

    def sizeHint(self):
        return super().sizeHint().expandedTo(QSize(80, 0))


def content_splitter(orientation=Qt.Horizontal) -> QSplitter:
    """Use one painted logical pixel while Qt keeps its larger drag region."""
    splitter = QSplitter(orientation)
    splitter.setHandleWidth(1)
    return splitter


def inset_pane(
    content: QWidget, margins: tuple[int, int, int, int] = (CONTENT_INSET,) * 4,
) -> QWidget:
    """Give one pane ownership of the gutters surrounding its content."""
    pane = QWidget()
    layout = QVBoxLayout(pane)
    layout.setContentsMargins(*margins)
    layout.setSpacing(0)
    layout.addWidget(content)
    max_width = content.maximumWidth()
    if max_width < pane.maximumWidth():
        pane.setMaximumWidth(max_width + margins[0] + margins[2])
    return pane


def settings_scroll_area(page: QWidget) -> QScrollArea:
    """Inset a settings form from the divider and from a visible scroll rail."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    layout = page.layout()

    def fit_right_gutter(low: int, high: int) -> None:
        right = CONTENT_INSET if high > low else 0
        layout.setContentsMargins(CONTENT_INSET, CONTENT_INSET, right, 0)

    fit_right_gutter(0, 0)
    scroll.verticalScrollBar().rangeChanged.connect(fit_right_gutter)
    scroll.setWidget(page)
    return scroll


class _PrimaryActionBinding(QObject):
    def __init__(self, button: QToolButton, action: QAction):
        super().__init__(button)
        self.button = button
        self.action = action
        self._refreshing = False
        button.setProperty("primaryAction", True)
        button.setFocusPolicy(Qt.StrongFocus)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setFixedHeight(32)
        button.setIconSize(QSize(16, 16))
        button.installEventFilter(self)
        action.changed.connect(self.refresh)
        self.refresh()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.FontChange:
            self.refresh()
        return super().eventFilter(watched, event)

    def refresh(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            busy = bool(self.action.property("primaryBusy"))
            if busy and self.action.isEnabled():
                self.action.setEnabled(False)
            self.button.setProperty("primaryBusy", busy)
            label = self.action.text().replace("&", "")
            width = self.button.fontMetrics().horizontalAdvance(label) + 16
            if not self.action.icon().isNull():
                width += self.button.iconSize().width() + 8
            self.button.setFixedWidth(width)
            self.button.setAccessibleName(label)
            self.button.setAccessibleDescription(self.action.toolTip())
            self.button.style().unpolish(self.button)
            self.button.style().polish(self.button)
            self.button.update()
        finally:
            self._refreshing = False


def bind_primary_action(toolbar: QToolBar, action: QAction) -> QToolButton:
    """Keep the existing QAction while giving its tool button a shared role."""
    button = toolbar.widgetForAction(action)
    if not isinstance(button, QToolButton):
        raise TypeError("Primary actions must already belong to a toolbar")
    if not hasattr(button, "_primary_action_binding"):
        button._primary_action_binding = _PrimaryActionBinding(button, action)
    return button


def set_action_busy(action: QAction, busy: bool, text: str = "") -> None:
    """Show running work without changing ownership of eligibility on exit."""
    was_busy = bool(action.property("primaryBusy"))
    if busy and not was_busy:
        action.setProperty("primaryIdleText", action.text())
        action.setProperty("primaryIdleIcon", action.icon())
    action.setProperty("primaryBusy", busy)
    if busy:
        action.setEnabled(False)
        if text:
            action.setText(text)
        action.setIcon(QIcon())
    elif was_busy:
        action.setText(action.property("primaryIdleText") or action.text())
        action.setIcon(action.property("primaryIdleIcon") or QIcon())
    if busy != was_busy:
        # Dynamic properties alone do not notify bindings when text/icon are unchanged.
        action.changed.emit()


def fit_navigation_pane(splitter: QSplitter) -> None:
    """Start a navigation-tree | page splitter with the tree at its maximum width.

    Without explicit sizes QSplitter may hand the width-capped tree more space
    than its maximum; the surplus is never painted (a dark strip beside the
    handle) until the user drags the handle. Call after adding both panes.
    """
    pane = splitter.widget(0)
    if pane is None or splitter.count() < 2:
        return
    if pane.maximumWidth() >= 16777215:  # QWIDGETSIZE_MAX: no cap, no gap to fix
        return
    width = max(pane.minimumWidth(), pane.maximumWidth())
    splitter.setSizes([width] + [1 << 20] * (splitter.count() - 1))
