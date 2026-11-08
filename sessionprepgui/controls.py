"""Shared desktop action behavior, independent of phase eligibility."""
from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QSplitter, QToolBar, QToolButton


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
