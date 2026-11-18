"""Joined source selection used by the preview toolbar and its overflow."""
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QSizePolicy, QToolButton, QWidget

from ..theme import COLORS


class _FocusRing(QWidget):
    """Paint over the joined faces without changing their layout on focus."""
    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setPen(QPen(QColor(COLORS["information"]), 2))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 2, 2)


class AuditionSourceSelector(QFrame):
    selected = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("auditionSource")
        self.setAccessibleName("Audition source")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame#auditionSource { border: 1px solid #555; border-radius: 3px; background: #555; }
            QToolButton#auditionSegment { border: none; background: #3a3a3a; color: #ddd;
                                         padding: 1px 8px; }
            QToolButton#auditionSegment:checked { background: #2a6db5; color: white; font-weight: 600; }
            QToolButton#auditionSegment:disabled { background: #303030; color: #888; }
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        self.setMinimumWidth(174)
        group = QButtonGroup(self)
        self.buttons = []
        for index, label in enumerate(("Original", "Corrected")):
            button = QToolButton(self)
            button.setObjectName("auditionSegment")
            button.setText(label)
            button.setCheckable(True)
            button.setMinimumSize(85, 24)
            button.setAccessibleName(f"Audition {label.lower()}")
            button.installEventFilter(self)
            group.addButton(button, index)
            self.buttons.append(button)
            row.addWidget(button)
        group.idClicked.connect(lambda index: self.selected.emit(bool(index)))
        self._ring = _FocusRing(self)
        self._ring.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._ring.hide()
        self.set_corrected(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ring.setGeometry(self.rect())

    def set_corrected(self, corrected: bool):
        button = self.buttons[int(corrected)]
        button.setChecked(True)
        self.setFocusProxy(button)
        for part in self.buttons:
            part.setFocusPolicy(Qt.StrongFocus if part is button else Qt.ClickFocus)

    def eventFilter(self, watched, event):
        kind = event.type()
        keyboard = None
        if kind == QEvent.FocusIn:
            keyboard = event.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
        elif kind in (QEvent.FocusOut, QEvent.MouseButtonPress):
            keyboard = False
        elif kind == QEvent.KeyPress:
            keyboard = True
        if keyboard is not None and self.property("keyboardFocus") != keyboard:
            self.setProperty("keyboardFocus", keyboard)
            self._ring.setVisible(keyboard)
        if kind == QEvent.KeyPress and event.key() in (Qt.Key_Left, Qt.Key_Right):
            target = self.buttons[int(event.key() == Qt.Key_Right)]
            if target.isEnabled():
                target.setFocus(Qt.ShortcutFocusReason)
                target.click()
            return True
        if kind == QEvent.KeyPress and event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            watched.click()
            return True
        return super().eventFilter(watched, event)
