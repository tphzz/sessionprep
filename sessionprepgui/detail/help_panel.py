"""The File report's on-demand, independently scrolling detector Help pane."""
from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextOption
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from ..icons import icon
from ..theme import COLORS
from ..tracks.table_widgets import DETAIL_TAB_INSET
from .report_style import configure_report_browser

# Report-to-dock gap; the report's own text inset sits inside the report.
_HELP_GAP = 8
# Below this Help body width, two-column Help tables stack (term, then meaning).
_HELP_STACK_BELOW = 240

# One table cell that holds no nested cell, row or table.
_CELL = r"<td\b[^>]*>(?:(?!<td\b|</?tr\b|<table\b|</td>).)*</td>"
_TWO_CELL_ROW = re.compile(rf"<tr>\s*({_CELL})\s*({_CELL})\s*</tr>", re.S | re.I)
_LEFT_PADDING = re.compile(r"padding-left\s*:\s*[^;'\"]*;?", re.I)
_SEPARATOR = re.compile(r"border-bottom\s*:\s*[^;'\"]*;?", re.I)
_BREAKS_BEFORE_TABLE = re.compile(r"(?:<br\s*/?>\s*)+(?=<table\b)", re.I)
_BREAKS_AFTER_TABLE = re.compile(r"(?<=</table>)(?:\s*<br\s*/?>)+", re.I)
_TABLE_TAG = re.compile(r"<table\b[^>]*>", re.I)
_CELL_TAG = re.compile(r"<(t[dh])\b([^>]*)>", re.I)
_STYLE_ATTR = re.compile(r"""\sstyle\s*=\s*(['"])(.*?)\1""", re.I | re.S)


def normalize_help_tables(html: str) -> str:
    """Give detector Help tables the approved Help-body rhythm.

    No blank line before or after a table, 8 px above and below it, body-size
    text and a 1 px separator under every row. Detector HTML stays untouched.
    """
    html = _BREAKS_BEFORE_TABLE.sub("", html)
    html = _BREAKS_AFTER_TABLE.sub("", html)
    html = _TABLE_TAG.sub(
        '<table width="100%" cellspacing="0" cellpadding="0"'
        ' style="margin-top:8px; margin-bottom:8px; border-collapse:collapse;">', html)

    def cell(match):
        tag, attrs = match.group(1), match.group(2)
        style = _STYLE_ATTR.search(attrs)
        # An existing separator is replaced rather than duplicated.
        existing = _SEPARATOR.sub("", style.group(2)).strip().rstrip(";") if style else ""
        existing = f"{existing}; " if existing else ""
        attrs = _STYLE_ATTR.sub("", attrs)
        separator = f"border-bottom:1px solid {COLORS['accent']}; padding-top:4px; padding-bottom:4px;"
        quote = "'" if '"' in existing else '"'   # keep quoted values intact
        return f"<{tag}{attrs} style={quote}{existing}{separator}{quote}>"
    return _CELL_TAG.sub(cell, html)


def stack_two_column_rows(html: str) -> str:
    """Put the second cell of each simple two-cell table row on its own row.

    Rows with nested tables or another cell count are left unchanged; the
    stacked meaning loses its side padding so it aligns with the term.
    """
    def stacked(match):
        term, meaning = match.group(1), match.group(2)
        opening = meaning[:meaning.index(">") + 1]
        meaning = _LEFT_PADDING.sub("", opening) + meaning[len(opening):]
        # One separator per stacked pair: under the meaning, not the term.
        term_open = term[:term.index(">") + 1]
        term = _SEPARATOR.sub("", term_open) + term[len(term_open):]
        return f"<tr>{term}</tr><tr>{meaning}</tr>"
    return _TWO_CELL_ROW.sub(stacked, html)


class _HelpCloseButton(QToolButton):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.click()
            event.accept()
        else:
            super().keyPressEvent(event)


class FileReportPane(QWidget):
    """Keep the original report browser beside one contextual Help panel."""

    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.report = report
        self._active_detector_id = None
        self._help_generation = None
        self._content_signature = None
        self._help_stacked = False
        self._help_anchor = None  # (block text, occurrence, top) across a width change
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_HELP_GAP)
        layout.addWidget(report, 1)

        self._help_panel = QFrame()
        self._help_panel.setObjectName("detectorHelpPanel")
        self._help_panel.setAccessibleName("Detector help")
        self._help_panel.setStyleSheet(f"""
            QFrame#detectorHelpPanel {{
                background:{COLORS["surface"]}; border:1px solid {COLORS["border"]};
            }}
            QFrame#detectorHelpHeader {{
                background:{COLORS["surface"]}; border:0;
                border-bottom:1px solid {COLORS["border"]};
            }}
            QTextBrowser#detectorHelpBody {{
                background:{COLORS["surface"]}; color:{COLORS["text"]}; border:0;
            }}
            QToolButton#detectorHelpClose {{ background:transparent; border:1px solid transparent; padding:0; }}
            QToolButton#detectorHelpClose:hover {{ background:{COLORS["accent"]}; }}
            QToolButton#detectorHelpClose:focus {{ border-color:{COLORS["information"]}; }}
        """)
        panel_layout = QVBoxLayout(self._help_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("detectorHelpHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        context = QLabel("Detector help")
        context.setObjectName("detectorHelpContext")
        context.setWordWrap(True)  # wraps beside Close in a narrow dock (approved design)
        context.setStyleSheet(f"color:{COLORS['metadata']};")
        self.help_title = QLabel()
        self.help_title.setTextFormat(Qt.PlainText)
        self.help_title.setWordWrap(True)
        title_font = QFont(self.help_title.font())
        title_font.setWeight(QFont.DemiBold)
        self.help_title.setFont(title_font)
        titles.addWidget(context)
        titles.addWidget(self.help_title)
        header_layout.addLayout(titles, 1)
        self.close_button = _HelpCloseButton()
        self.close_button.setObjectName("detectorHelpClose")
        self.close_button.setAutoRaise(True)
        self.close_button.setIcon(icon("close"))
        self.close_button.setIconSize(QSize(16, 16))
        self.close_button.setFixedSize(32, 32)
        self.close_button.setToolTip("Close detector help (Esc)")
        self.close_button.setAccessibleName("Close detector help")
        self.close_button.setFocusPolicy(Qt.StrongFocus)
        self.close_button.clicked.connect(lambda: self.close_help())
        header_layout.addWidget(self.close_button, 0, Qt.AlignTop)
        panel_layout.addWidget(header)

        self.help_browser = QTextBrowser()
        self.help_browser.setObjectName("detectorHelpBody")
        self.help_browser.setAccessibleName("Detector explanation")
        self.help_browser.setOpenLinks(False)
        self.help_browser.setOpenExternalLinks(False)
        self.help_browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.help_browser.document().setDocumentMargin(16)
        panel_layout.addWidget(self.help_browser, 1)
        self.help_browser.installEventFilter(self)
        # The dock keeps the detail-tab padding from the pane's top, right and
        # bottom edges; the report keeps the full height beside it.
        self._help_dock = QWidget()
        dock_layout = QVBoxLayout(self._help_dock)
        dock_layout.setContentsMargins(0, DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET)
        dock_layout.setSpacing(0)
        dock_layout.addWidget(self._help_panel)
        layout.addWidget(self._help_dock)
        self._help_dock.hide()
        self._escape = QShortcut(QKeySequence(Qt.Key_Escape), self._help_panel)
        self._escape.setContext(Qt.WidgetWithChildrenShortcut)
        self._escape.activated.connect(self.close_help)

        report.helpRequested.connect(self._show_help)
        report.reportChanged.connect(self._report_changed)
        self.setFocusProxy(report)
        self.refresh_fonts()
        self._fit_help_width()

    @property
    def active_detector_id(self):
        return self._active_detector_id

    def refresh_fonts(self, screen=None):
        cursor = self.help_browser.cursorForPosition(QPoint(0, 0))
        top = self.help_browser.cursorRect(cursor).top()
        configure_report_browser(self.help_browser, screen, content_scale=1.0)
        self._fit_help_width()
        self.layout().activate()
        self.help_browser.document().documentLayout().documentSize()
        bar = self.help_browser.verticalScrollBar()
        bar.setValue(bar.value() + self.help_browser.cursorRect(cursor).top() - top)

    def _fit_help_width(self):
        # Preserve the approved 360 px dock at desktop widths, retaining report
        # space when the user makes the File region narrower.
        preferred = min(360, max(140, round(self.width() * 0.4)))
        if self._active_detector_id is not None:
            available = (self.width() - _HELP_GAP - DETAIL_TAB_INSET
                         - self.report.minimum_header_width())
            preferred = min(preferred, max(140, available))
        # The panel's 1 px border on each side surrounds the Help body.
        stacked = preferred - 2 < _HELP_STACK_BELOW
        restack = stacked != self._help_stacked and self._content_signature is not None
        if restack:
            # The body only reaches its new width later; keep the passage the
            # reader was on and re-apply it once the body has been resized.
            self._help_anchor = self._capture_help_anchor()
        self._help_panel.setFixedWidth(preferred)
        if stacked != self._help_stacked:
            self._help_stacked = stacked
            if restack:
                self._render_help(keep_scroll=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_help_width()

    def _set_content(self, row):
        signature = (row["title"], row["help"])
        if signature != self._content_signature:
            self.help_title.setText(row["title"])
            self.help_browser.setAccessibleName(f'Help for {row["title"]}')
            self._content_signature = signature
            self._help_anchor = None
            self._render_help(keep_scroll=False)

    def _render_help(self, *, keep_scroll):
        help_html = normalize_help_tables(self._content_signature[1])
        if self._help_stacked:
            help_html = stack_two_column_rows(help_html)
        self.help_browser.setHtml(
            f'<body style="background-color:{COLORS["surface"]};'
            f' color:{COLORS["text"]};">{help_html}</body>')
        self.help_browser.document().setDocumentMargin(16)
        self.help_browser.document().documentLayout().documentSize()
        if keep_scroll:
            self._apply_help_anchor(self._help_anchor)
        else:
            self.help_browser.verticalScrollBar().setValue(0)

    def _capture_help_anchor(self):
        """The first visible text block and its offset from the body top."""
        browser = self.help_browser
        block = browser.cursorForPosition(QPoint(0, 0)).block()
        while block.isValid() and not block.text().strip():
            block = block.next()
        if not block.isValid():
            return None
        text = block.text()
        occurrence, previous = 0, block.previous()
        while previous.isValid():
            occurrence += previous.text() == text
            previous = previous.previous()
        cursor = browser.textCursor()
        cursor.setPosition(block.position())
        return text, occurrence, browser.cursorRect(cursor).top()

    def _apply_help_anchor(self, anchor):
        if anchor is None:
            return
        browser = self.help_browser
        text, occurrence, top = anchor
        document = browser.document()
        document.documentLayout().documentSize()
        block = document.begin()
        while block.isValid():
            if block.text() == text:
                if not occurrence:
                    break
                occurrence -= 1
            block = block.next()
        if not block.isValid():
            return  # the passage is gone; keep the current position
        cursor = browser.textCursor()
        cursor.setPosition(block.position())
        bar = browser.verticalScrollBar()
        bar.setValue(bar.value() + browser.cursorRect(cursor).top() - top)

    def _apply_pending_help_anchor(self):
        anchor, self._help_anchor = self._help_anchor, None
        if self._content_signature is not None:
            self._apply_help_anchor(anchor)

    def eventFilter(self, watched, event):
        if (watched is self.help_browser and event.type() == QEvent.Resize
                and self._help_anchor is not None):
            # Re-apply after the browser has laid the document out at its new width.
            QTimer.singleShot(0, self._apply_pending_help_anchor)
        return super().eventFilter(watched, event)

    def _show_help(self, key):
        row = self.report.help_row(key)
        if row is None:
            return
        anchor = self.report.capture_reading_anchor()
        same = (self._help_generation == self.report.context_generation
                and self._active_detector_id == key)
        self._active_detector_id = key
        self._help_generation = self.report.context_generation
        if not same:
            self._content_signature = None
        self._set_content(row)
        self._fit_help_width()
        self._help_dock.show()
        self.layout().activate()
        self.report.set_active_help(key)
        self.report.restore_reading_anchor(anchor)
        help_scroll = self.help_browser.verticalScrollBar().value()
        self.help_browser.setFocus(Qt.OtherFocusReason)
        self.help_browser.verticalScrollBar().setValue(help_scroll)

    def _report_changed(self):
        if self._active_detector_id is None:
            return
        row = self.report.help_row(self._active_detector_id)
        if self._help_generation != self.report.context_generation or row is None:
            self.close_help(restore_focus=False)
        else:
            self._set_content(row)
            self._fit_help_width()
            self.layout().activate()

    def close_help(self, *, restore_focus=True):
        if self._active_detector_id is None:
            return
        key = self._active_detector_id
        same_context = self._help_generation == self.report.context_generation
        anchor = self.report.capture_reading_anchor()
        self._active_detector_id = None
        self._help_generation = None
        self._content_signature = None
        self._help_anchor = None
        self._help_dock.hide()
        self.help_browser.clear()
        self.help_title.clear()
        self.layout().activate()
        self.report.set_active_help(None)
        self.report.restore_reading_anchor(anchor)
        if restore_focus and same_context:
            self.report.focus_report_link("report-help", key)
