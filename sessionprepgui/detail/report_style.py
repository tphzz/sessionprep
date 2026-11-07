"""Shared styling helpers for GUI report browsers."""

from __future__ import annotations

from typing import Protocol

from PySide6.QtGui import QFont, QGuiApplication, QScreen
from PySide6.QtWidgets import QApplication, QTextBrowser

from ..theme import COLORS


# Qt rich text ignores em sizes but honours CSS size keywords relative to the
# document font, so the title role scales with the report font (x-large = 1.5x).
REPORT_TITLE_SIZE = "x-large"
REPORT_SECTION_SIZE = "1.15em"
# Shared File report section role ("Bimodal Normalization", "Detector
# findings"): report body size, semibold, 20 px above and 8 px to its content.
# The explicit "medium" keyword keeps the heading out of the finding body role.
REPORT_SECTION_HEADING = "font-size:medium; font-weight:600; margin-top:20px; margin-bottom:8px;"
REPORT_BADGE_SIZE = "0.82em"
REPORT_CONTENT_SCALE = 1.15


class _ScreenLike(Protocol):
    def devicePixelRatio(self) -> float: ...
    def logicalDotsPerInch(self) -> float: ...


def report_font_scale(screen: _ScreenLike | None) -> float:
    """Return a small report-only font boost for high-DPI screens."""
    if screen is None:
        return 1.0

    try:
        dpr = float(screen.devicePixelRatio())
    except (TypeError, ValueError, AttributeError):
        dpr = 1.0

    try:
        logical_scale = float(screen.logicalDotsPerInch()) / 96.0
    except (TypeError, ValueError, AttributeError):
        logical_scale = 1.0

    effective = max(dpr, logical_scale)
    if effective >= 1.75:
        return 1.15
    if effective >= 1.25:
        return 1.08
    return 1.0


def configure_report_browser(
    browser: QTextBrowser,
    screen: QScreen | _ScreenLike | None = None,
    *, content_scale: float = REPORT_CONTENT_SCALE,
) -> None:
    """Make report panes follow the platform application font and DPI."""
    app = QApplication.instance()
    font = QFont(app.font() if app is not None else browser.font())
    if screen is None:
        screen = browser.screen() or QGuiApplication.primaryScreen()
    scale = content_scale * report_font_scale(screen)
    size = font.pointSizeF()
    if size > 0:
        font.setPointSizeF(size * scale)
    browser.setFont(font)
    browser.document().setDefaultFont(font)


def wrap_report_html(body: str) -> str:
    """Wrap report HTML without overriding the platform font face or size."""
    return (
        f'<body style="background-color:{COLORS["bg"]}; color:{COLORS["text"]};'
        f' padding:12px;">'
        f'{body}</body>'
    )
