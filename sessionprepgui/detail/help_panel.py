"""The File report's on-demand, independently scrolling detector Help pane."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextOption
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from ..icons import icon
from ..theme import COLORS
from .report_style import configure_report_browser


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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
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
        layout.addWidget(self._help_panel)
        self._help_panel.hide()
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
            available = self.width() - 16 - self.report.minimum_header_width()
            preferred = min(preferred, max(140, available))
        self._help_panel.setFixedWidth(preferred)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_help_width()

    def _set_content(self, row):
        signature = (row["title"], row["help"])
        if signature != self._content_signature:
            self.help_title.setText(row["title"])
            self.help_browser.setAccessibleName(f'Help for {row["title"]}')
            self.help_browser.setHtml(
                f'<body style="background-color:{COLORS["surface"]};'
                f' color:{COLORS["text"]};">{row["help"]}</body>')
            self.help_browser.document().setDocumentMargin(16)
            self.help_browser.verticalScrollBar().setValue(0)
            self._content_signature = signature

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
        self._help_panel.show()
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
        self._help_panel.hide()
        self.help_browser.clear()
        self.help_title.clear()
        self.layout().activate()
        self.report.set_active_help(None)
        self.report.restore_reading_anchor(anchor)
        if restore_focus and same_context:
            self.report.focus_report_link("report-help", key)
