"""Helpers for choosing sane top-level window startup geometry."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import QCursor, QGuiApplication, QScreen

STARTUP_SCREEN_WIDTH_FRACTION = 0.82
STARTUP_SCREEN_HEIGHT_FRACTION = 5 / 6
STARTUP_SCREEN_MARGIN = 24
FALLBACK_STARTUP_SIZE = QSize(1280, 800)


def screen_for_startup() -> QScreen | None:
    """Return the best screen for a newly opened top-level window."""
    screen = QGuiApplication.screenAt(QCursor.pos())
    if screen is not None:
        return screen
    return QGuiApplication.primaryScreen()


def calculate_startup_geometry(
    available: QRect,
    minimum_size: QSize | None = None,
    *,
    width_fraction: float = STARTUP_SCREEN_WIDTH_FRACTION,
    height_fraction: float = STARTUP_SCREEN_HEIGHT_FRACTION,
    margin: int = STARTUP_SCREEN_MARGIN,
) -> QRect:
    """Calculate centered startup geometry inside a screen's available area."""
    if available.width() <= 0 or available.height() <= 0:
        return QRect(
            0, 0, FALLBACK_STARTUP_SIZE.width(), FALLBACK_STARTUP_SIZE.height()
        )

    edge_margin = max(0, int(margin))
    max_width = max(1, available.width() - edge_margin * 2)
    max_height = max(1, available.height() - edge_margin * 2)

    target_width = int(round(available.width() * width_fraction))
    target_height = int(round(available.height() * height_fraction))

    min_width = 0
    if minimum_size is not None and minimum_size.isValid():
        min_width = max(0, minimum_size.width())

    width = min(max(target_width, min_width), max_width)
    height = min(target_height, max_height)

    x = available.x() + (available.width() - width) // 2
    y = available.y() + (available.height() - height) // 2
    return QRect(x, y, width, height)
