"""Shared styling helpers for GUI report browsers."""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QTextBrowser

from ..theme import COLORS


REPORT_TITLE_SIZE = "1.3em"
REPORT_SECTION_SIZE = "1.15em"
REPORT_BADGE_SIZE = "0.82em"


def configure_report_browser(browser: QTextBrowser) -> None:
    """Make report panes follow the platform application font and DPI."""
    app = QApplication.instance()
    font = QFont(app.font() if app is not None else browser.font())
    browser.setFont(font)
    browser.document().setDefaultFont(font)


def wrap_report_html(body: str) -> str:
    """Wrap report HTML without overriding the platform font face or size."""
    return (
        f'<body style="background-color:{COLORS["bg"]}; color:{COLORS["text"]};'
        f' padding:12px;">'
        f'{body}</body>'
    )
