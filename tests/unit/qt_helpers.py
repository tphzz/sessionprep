"""Lifecycle helpers for test-owned Qt windows."""

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid


def dispose_window(window: QWidget) -> None:
    """Run the window's close handler and release its native widget tree."""
    if not window.close():
        raise AssertionError(f"{type(window).__name__} rejected close during teardown")
    # Deliver queued callbacks while the window and its children still exist.
    QApplication.processEvents()
    if isValid(window):
        window.deleteLater()
        QApplication.sendPostedEvents(window, QEvent.DeferredDelete)
