"""Non-modal inline confirmation prompt hosted inside the main window status bar."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent, QResizeEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QWidget,
)
import shiboken6

from .icons import icon


class _PromptButton(QPushButton):
    """Status-bar prompt button that accepts Space and Enter when focused."""

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.ShortcutOverride and self.hasFocus():
            if isinstance(ev, QKeyEvent) and (
                (ev.key() == Qt.Key.Key_Space and ev.modifiers() == Qt.KeyboardModifier.NoModifier)
                or ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            ):
                ev.accept()
                return True
        return super().event(ev)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)


class _ElidingPromptLabel(QLabel):
    """Single-line label that middle-elides a highlighted filename without widening the window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._prefix = ""
        self._highlight = ""
        self._suffix = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_prompt_parts(self, prefix: str, highlight: str, suffix: str) -> None:
        self._prefix = prefix
        self._highlight = highlight
        self._suffix = suffix
        self._update_elided_text()

    def full_text(self) -> str:
        return f"{self._prefix}{self._highlight}{self._suffix}"

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_elided_text()

    def _update_elided_text(self) -> None:
        full = self.full_text()
        avail = max(self.contentsRect().width(), 0)
        if avail <= 0:
            super().setText(full)
            return
        fm = self.fontMetrics()
        if fm.horizontalAdvance(full) <= avail:
            super().setText(full)
            return
        if self._highlight:
            fixed_w = fm.horizontalAdvance(f"{self._prefix}{self._suffix}")
            budget = avail - fixed_w
            if budget >= fm.horizontalAdvance("a…a"):
                elided_name = fm.elidedText(
                    self._highlight, Qt.TextElideMode.ElideMiddle, budget
                )
                super().setText(f"{self._prefix}{elided_name}{self._suffix}")
                return
        super().setText(fm.elidedText(full, Qt.TextElideMode.ElideRight, avail))


class StatusBarPromptWidget(QWidget):
    """Atomic amber-accented inline prompt widget for ``QStatusBar``."""

    accepted = Signal()
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBarPrompt")
        self.setProperty("statusPrompt", "warning")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setSpacing(8)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("statusBarPromptIcon")
        self.icon_label.setPixmap(icon("warning").pixmap(16, 16))
        self.icon_label.setFixedSize(16, 16)
        self.icon_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.icon_label.setAccessibleName("")
        layout.addWidget(self.icon_label)

        self.message_label = _ElidingPromptLabel(self)
        self.message_label.setObjectName("statusBarPromptMessage")
        layout.addWidget(self.message_label, 1)

        self.accept_btn = _PromptButton("Overwrite", self)
        self.accept_btn.setObjectName("statusBarPromptAccept")
        self.accept_btn.setProperty("promptAction", "caution")
        self.accept_btn.clicked.connect(self.accepted)
        layout.addWidget(self.accept_btn)

        self.cancel_btn = _PromptButton("Cancel", self)
        self.cancel_btn.setObjectName("statusBarPromptCancel")
        self.cancel_btn.clicked.connect(self.cancelled)
        layout.addWidget(self.cancel_btn)

        self.setVisible(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def show_prompt(
        self,
        prefix: str,
        highlight: str,
        suffix: str,
        *,
        tooltip: str,
        accept_label: str = "Overwrite",
        cancel_label: str = "Cancel",
        accept_description: str = "",
        cancel_description: str = "",
    ) -> None:
        self.message_label.set_prompt_parts(prefix, highlight, suffix)
        self.message_label.setToolTip(tooltip)
        self.message_label.setAccessibleName(self.message_label.full_text())
        self.message_label.setAccessibleDescription(tooltip)
        self.setToolTip(tooltip)

        self.accept_btn.setText(accept_label)
        self.accept_btn.setAccessibleName(accept_label)
        self.accept_btn.setAccessibleDescription(accept_description or tooltip)
        self.accept_btn.setToolTip(accept_description or tooltip)

        self.cancel_btn.setText(cancel_label)
        self.cancel_btn.setAccessibleName(cancel_label)
        self.cancel_btn.setAccessibleDescription(cancel_description)
        self.cancel_btn.setToolTip(cancel_description)

        self.setVisible(True)
        self.cancel_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def dismiss(self) -> None:
        self.setVisible(False)


class StatusBarPromptCoordinator:
    """Owns ``QStatusBar`` message vs. inline-prompt exclusivity and focus return."""

    def __init__(self, status_bar: QStatusBar) -> None:
        self._status_bar = status_bar
        self.widget = StatusBarPromptWidget(status_bar)
        self._status_bar.addPermanentWidget(self.widget, 1)
        self._last_status = ""
        self._active = False
        self._on_accept: Callable[[], None] | None = None
        self._on_cancel: Callable[[], None] | None = None
        self._return_focus_widget: QWidget | None = None

        self.widget.accepted.connect(self._handle_accept)
        self.widget.cancelled.connect(self._handle_cancel)

    @property
    def is_active(self) -> bool:
        return self._active

    def show_status(self, message: str) -> None:
        self._last_status = message
        if not self._active and shiboken6.isValid(self._status_bar):
            self._status_bar.showMessage(message)

    def show_prompt(
        self,
        prefix: str,
        highlight: str,
        suffix: str,
        *,
        tooltip: str,
        on_accept: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        accept_label: str = "Overwrite",
        cancel_label: str = "Cancel",
        accept_description: str = "",
        cancel_description: str = "",
        return_focus_widget: QWidget | None = None,
    ) -> None:
        if not shiboken6.isValid(self._status_bar) or not shiboken6.isValid(self.widget):
            return
        self._active = True
        self._on_accept = on_accept
        self._on_cancel = on_cancel
        self._return_focus_widget = return_focus_widget
        self._status_bar.clearMessage()
        self.widget.show_prompt(
            prefix,
            highlight,
            suffix,
            tooltip=tooltip,
            accept_label=accept_label,
            cancel_label=cancel_label,
            accept_description=accept_description,
            cancel_description=cancel_description,
        )

    def dismiss(
        self,
        *,
        status_message: str | None = None,
        restore_focus: bool = False,
        restore_last_status: bool = True,
    ) -> None:
        was_active = self._active
        focus_target = self._return_focus_widget
        self._active = False
        self._on_accept = None
        self._on_cancel = None
        self._return_focus_widget = None
        if was_active and shiboken6.isValid(self.widget):
            self.widget.dismiss()
        if shiboken6.isValid(self._status_bar):
            if status_message is not None:
                self._last_status = status_message
                self._status_bar.showMessage(status_message)
            elif was_active and restore_last_status and self._last_status:
                self._status_bar.showMessage(self._last_status)
        if (
            was_active
            and restore_focus
            and focus_target is not None
            and shiboken6.isValid(focus_target)
            and focus_target.isVisible()
            and focus_target.isEnabled()
        ):
            focus_target.setFocus(Qt.FocusReason.OtherFocusReason)

    def _handle_accept(self) -> None:
        callback = self._on_accept
        self.dismiss(restore_focus=False, restore_last_status=callback is None)
        if callback is not None:
            callback()

    def _handle_cancel(self) -> None:
        callback = self._on_cancel
        self.dismiss(restore_focus=True, restore_last_status=callback is None)
        if callback is not None:
            callback()
