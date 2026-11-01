"""Interactive compact reports with one scrollable document."""
from __future__ import annotations
from urllib.parse import quote, unquote
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QTextCursor
from ..tracks.table_widgets import _HelpBrowser
from .report import prepare_track_report, render_prepared_track_report, render_summary_html
from .report_style import wrap_report_html


class ReportBrowser(_HelpBrowser):
    """Render cached data; disclosure state belongs to a session, file and detector."""

    reviewRequested = Signal(str)

    def __init__(self, help_map, parent=None):
        super().__init__(help_map, parent)
        self.report_session = None
        self._mode = None
        self._track_key = None
        self._prepared = None
        self._states = {}
        self._review_files = set()
        self._help_origins = {}
        self.anchorClicked.connect(self.activate_report_link)
        self.setAccessibleName("Analysis report")

    def _use_session(self, session):
        if session is not self.report_session:
            self._states.clear()
            self._help_origins.clear()
        self.report_session = session

    def clear(self):
        self.report_session = None
        self._track_key = None
        self._states.clear()
        self._help_origins.clear()
        self._mode = None
        self._prepared = None
        self._review_files.clear()
        super().clear()

    def setHtml(self, html):
        """External loading/error messages invalidate interactive content."""
        self.clear()
        super().setHtml(html)

    def present_track(self, track, session, *, show_clean=True, verbose=False):
        previous = self._track_key if session is self.report_session else None
        self._use_session(session)
        self._mode = "file"
        self._track_key = track.filename
        self._review_files.clear()
        self._prepared = prepare_track_report(
            track, session, show_clean=show_clean, verbose=verbose)
        self._states.setdefault(self._track_key, {"expanded": set(), "help": set()})
        scroll = self.verticalScrollBar().value() if previous == self._track_key else 0
        self._render_track()
        self.verticalScrollBar().setValue(scroll)

    def present_summary(self, summary, session, *, show_clean=True):
        same_session = session is self.report_session
        self._use_session(session)
        self._mode = "summary"
        self._prepared = None
        current_files = {track.filename for track in session.tracks}
        self._review_files = {
            ref["filename"]
            for category in ("problems", "attention", "information", "clean")
            if category != "clean" or show_clean
            for group in summary.get(category) or []
            for ref in group.get("item_refs") or []
            if ref.get("filename") in current_files
        }
        scroll = self.verticalScrollBar().value() if same_session else 0
        self._set_document(render_summary_html(
            summary, show_faders=False, show_clean=show_clean, session=session))
        self.verticalScrollBar().setValue(scroll)

    def _set_document(self, html):
        super().setHtml(wrap_report_html(html))
        root = self.document().rootFrame()
        fmt = root.frameFormat()
        fmt.setLeftMargin(24)
        fmt.setRightMargin(24)
        fmt.setTopMargin(16 if self._mode == "summary" else 4)
        fmt.setBottomMargin(16)
        root.setFrameFormat(fmt)
        self.document().documentLayout().documentSize()

    def _render_track(self):
        state = self._states[self._track_key]
        self._set_document(render_prepared_track_report(
            self._prepared, expanded=state["expanded"], help_open=state["help"]))

    def _anchor_cursor(self, *, name=None, href=None):
        block = self.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    fmt = fragment.charFormat()
                    if ((name is not None and name in fmt.anchorNames())
                            or (href is not None and fmt.anchorHref() == href)):
                        cursor = QTextCursor(self.document())
                        cursor.setPosition(fragment.position())
                        return cursor
                iterator += 1
            block = block.next()
        return None

    def _row_cursor(self, key):
        return self._anchor_cursor(href="report-details:" + quote(key, safe=""))

    def keyPressEvent(self, event):
        # setHtml resets QTextBrowser's internal anchor focus. The restored text
        # cursor still identifies the link, allowing repeated keyboard activation.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            cursor = self.textCursor()
            href = cursor.charFormat().anchorHref()
            if not href:
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
                href = cursor.charFormat().anchorHref()
            if QUrl(href).scheme() in {
                    "report-review", "report-details", "report-help", "report-close-help"}:
                self.activate_report_link(QUrl(href))
                event.accept()
                return
        super().keyPressEvent(event)

    def activate_report_link(self, url):
        scheme = url.scheme()
        if scheme not in {"report-review", "report-details", "report-help", "report-close-help"}:
            return
        key = unquote(url.path(QUrl.ComponentFormattingOption.FullyEncoded))
        if scheme == "report-review":
            if self._mode == "summary" and key in self._review_files:
                self.reviewRequested.emit(key)
            return
        if self._mode != "file" or not self._prepared:
            return
        row = next((row for row in self._prepared["rows"] if row["id"] == key), None)
        if row is None:
            return
        state = self._states[self._track_key]
        cursor = self._row_cursor(key)
        old_top = self.cursorRect(cursor).top() if cursor is not None else None
        old_scroll = self.verticalScrollBar().value()
        if scheme == "report-details":
            state["expanded"].symmetric_difference_update({key})
            state["help"].discard(key)
            focus_scheme = "report-details"
        elif scheme == "report-help":
            if not row["help"]:
                return
            if key in state["help"]:
                state["help"].remove(key)
                old_top = self._help_origins.pop((self._track_key, key), old_top)
            else:
                self._help_origins[(self._track_key, key)] = old_top
                state["help"].add(key)
            focus_scheme = "report-close-help" if key in state["help"] else "report-help"
        else:
            state["help"].discard(key)
            old_top = self._help_origins.pop((self._track_key, key), old_top)
            focus_scheme = "report-help"
        self._render_track()
        self.verticalScrollBar().setValue(old_scroll)
        cursor = self._row_cursor(key)
        if cursor is not None and old_top is not None:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + self.cursorRect(cursor).top() - old_top)
        if scheme == "report-help" and key in state["help"]:
            help_cursor = self._anchor_cursor(name="help-" + quote(key, safe=""))
            table = help_cursor.currentTable() if help_cursor is not None else None
            if table is not None:
                rect = self.document().documentLayout().frameBoundingRect(table)
                bar = self.verticalScrollBar()
                height = self.viewport().height()
                if rect.height() > height - 24:
                    bar.setValue(round(rect.top()) - 12)
                elif rect.bottom() - bar.value() > height - 12:
                    bar.setValue(round(rect.bottom()) - height + 12)
                elif rect.top() - bar.value() < 12:
                    bar.setValue(round(rect.top()) - 12)
        focus = self._anchor_cursor(href=focus_scheme + ":" + quote(key, safe=""))
        if focus is not None:
            self.setTextCursor(focus)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
