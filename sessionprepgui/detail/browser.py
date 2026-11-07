"""Interactive cached reports; detector Help is presented separately."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from urllib.parse import quote, unquote

from PySide6.QtCore import QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QTextCharFormat, QTextCursor, QTextDocument, QTextFormat

from ..tracks.table_widgets import DETAIL_TAB_INSET, _HelpBrowser
from .report import prepare_track_report, render_prepared_track_report, render_summary_html
from .report_style import REPORT_CONTENT_SCALE, configure_report_browser, wrap_report_html


def _apply_report_inset(document):
    """Inset report text by the shared detail-tab inset on every side."""
    root = document.rootFrame()
    fmt = root.frameFormat()
    fmt.setLeftMargin(DETAIL_TAB_INSET)
    fmt.setRightMargin(DETAIL_TAB_INSET)
    fmt.setTopMargin(DETAIL_TAB_INSET)
    fmt.setBottomMargin(DETAIL_TAB_INSET)
    root.setFrameFormat(fmt)


@dataclass(frozen=True)
class _ReadingAnchor:
    generation: int
    detector_id: str | None
    position: int
    top: int
    scroll: int


class ReportBrowser(_HelpBrowser):
    """Render prepared content; retain disclosure by session, file and detector."""

    reviewRequested = Signal(str)
    helpRequested = Signal(str)
    visualizeRequested = Signal(str)
    reportChanged = Signal()

    def __init__(self, help_map, parent=None):
        super().__init__(help_map, parent)
        self.report_session = None
        self._mode = None
        self._track_key = None
        self._prepared = None
        self._states = {}
        self._review_files = set()
        self._anchor_positions = {}
        self._named_positions = {}
        self._finding_font_ranges = []
        self._minimum_header_cache = None
        self._generation = 0
        self._compact = False
        self._active_help = None
        self.anchorClicked.connect(self.activate_report_link)
        self.setAccessibleName("Analysis report")

    @property
    def context_generation(self):
        return self._generation

    @property
    def track_filename(self):
        return self._track_key if self._mode == "file" else None

    def visualization_labels(self, key):
        if self._mode == "file" and self._prepared:
            return next((r.get("visualize_labels", ()) for r in self._prepared["rows"]
                         if r["id"] == key), ())
        return ()

    def help_row(self, key):
        if self._mode == "file" and self._prepared:
            return next((r for r in self._prepared["rows"]
                         if r["id"] == key and r["help"]), None)
        return None

    def _use_session(self, session):
        if session is not self.report_session:
            self._states.clear()
        self.report_session = session

    def clear(self):
        self._generation += 1
        self.report_session = None
        self._track_key = None
        self._states.clear()
        self._mode = None
        self._prepared = None
        self._active_help = None
        self._review_files.clear()
        self._anchor_positions.clear()
        self._named_positions.clear()
        self._finding_font_ranges.clear()
        self._minimum_header_cache = None
        super().clear()
        _apply_report_inset(self.document())
        self.reportChanged.emit()

    def setHtml(self, html):
        """External loading/error messages invalidate interactive content."""
        self.clear()
        super().setHtml(html)
        _apply_report_inset(self.document())

    def present_track(self, track, session, *, show_clean=True, verbose=False):
        same = (session is self.report_session and self._mode == "file"
                and self._track_key == track.filename)
        anchor = self.capture_reading_anchor() if same else None
        if not same:
            self._generation += 1
        self._use_session(session)
        self._mode = "file"
        self._track_key = track.filename
        self._review_files.clear()
        self._prepared = prepare_track_report(
            track, session, show_clean=show_clean, verbose=verbose)
        expanded = self._states.setdefault(self._track_key, set())
        expanded.intersection_update(r["id"] for r in self._prepared["rows"])
        self.reportChanged.emit()
        self._compact = self._wants_compact()
        self._render_track()
        if anchor is not None:
            self.restore_reading_anchor(anchor)
        else:
            self.verticalScrollBar().setValue(0)

    def present_summary(self, summary, session, *, show_clean=True):
        same_session = session is self.report_session
        self._generation += 1
        self._use_session(session)
        self._mode = "summary"
        self._prepared = None
        self._active_help = None
        self.reportChanged.emit()
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
        _apply_report_inset(self.document())
        self.document().documentLayout().documentSize()
        self._anchor_positions.clear()
        self._named_positions.clear()
        inherited_fonts = []
        block = self.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    char = fragment.charFormat()
                    # Keyword-sized text (title/section roles) keeps its size.
                    if (char.fontPointSize() <= 0 and char.font().pixelSize() < 0
                            and not char.hasProperty(QTextFormat.FontSizeAdjustment)):
                        inherited_fonts.append((fragment.position(), fragment.length()))
                    if char.anchorHref():
                        self._anchor_positions.setdefault(char.anchorHref(), fragment.position())
                    for name in char.anchorNames():
                        self._named_positions.setdefault(name, fragment.position())
                iterator += 1
            block = block.next()
        # Processor bodies (normalization) and findings use the finding body role.
        starts = [self._named_positions[name] for name in ("report-processors", "report-findings")
                  if name in self._named_positions]
        start = min(starts) if starts else None
        self._finding_font_ranges = [r for r in inherited_fonts if start is not None and r[0] >= start]
        self._apply_finding_font()

    def _apply_finding_font(self):
        # Qt's HTML subset ignores em sizes. Use the native UI role without
        # changing faces, emphasis, or explicit custom-renderer font sizes.
        size = self.font().pointSizeF() / REPORT_CONTENT_SCALE
        if size <= 0 or not self._finding_font_ranges:
            return
        fmt = QTextCharFormat()
        fmt.setFontPointSize(size)
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        for position, length in self._finding_font_ranges:
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)
        cursor.endEditBlock()

    def _render_track(self):
        if self._mode == "file" and self._prepared is not None:
            self._set_document(render_prepared_track_report(
                self._prepared, expanded=self._states[self._track_key],
                compact=self._compact, active_help=self._active_help))

    def _anchor_cursor(self, *, name=None, href=None):
        position = (self._named_positions.get(name) if name is not None
                    else self._anchor_positions.get(href))
        if position is not None:
            cursor = QTextCursor(self.document())
            cursor.setPosition(position)
            return cursor
        return None

    def _row_cursor(self, key):
        return self._anchor_cursor(href="report-details:" + quote(key, safe=""))

    def _reading_cursor(self, key):
        return (self._anchor_cursor(href="report-help:" + quote(key, safe=""))
                or self._anchor_cursor(href="report-visualize:" + quote(key, safe=""))
                or self._row_cursor(key))

    def capture_reading_anchor(self):
        """Pin visible content, rather than pixels that change during reflow."""
        if not getattr(self, "_prepared", None):
            return None
        cursor = self.cursorForPosition(QPoint(1, 1))
        detector_id = None
        for row in self._prepared["rows"]:
            candidate = self._reading_cursor(row["id"])
            if candidate is not None and 0 <= self.cursorRect(candidate).top() < self.viewport().height():
                detector_id, cursor = row["id"], candidate
                break
        return _ReadingAnchor(self._generation, detector_id, cursor.position(),
                              self.cursorRect(cursor).top(), self.verticalScrollBar().value())

    def restore_reading_anchor(self, anchor):
        if anchor is None or anchor.generation != self._generation or self._prepared is None:
            return
        cursor = self._reading_cursor(anchor.detector_id) if anchor.detector_id is not None else None
        if cursor is None:
            cursor = QTextCursor(self.document())
            cursor.setPosition(min(anchor.position, self.document().characterCount() - 1))
        bar = self.verticalScrollBar()
        bar.setValue(anchor.scroll)
        bar.setValue(bar.value() + self.cursorRect(cursor).top() - anchor.top)

    def minimum_header_width(self):
        """Measure cached metadata only; custom detector details may still scroll."""
        if self._mode != "file" or self._prepared is None:
            return 0
        # The compact header is the narrowest layout the report renders.
        return self._header_width(self._prepared.get("header_compact", self._prepared["header"]))

    def _header_width(self, header):
        """Widget width the cached header needs without horizontal scrolling."""
        key = (header, self.font().key())
        cache = self._minimum_header_cache if isinstance(self._minimum_header_cache, dict) else {}
        if key not in cache:
            if len(cache) >= 8:  # current wide/compact headers at a few fonts
                cache.clear()
            document = QTextDocument()
            document.setDefaultFont(self.font())
            document.setDefaultTextOption(self.document().defaultTextOption())
            document.setHtml(wrap_report_html(header))
            _apply_report_inset(document)
            document.setTextWidth(1)
            minimum = ceil(document.documentLayout().documentSize().width())
            cache[key] = minimum + 2 * self.frameWidth() + self.verticalScrollBar().sizeHint().width()
        self._minimum_header_cache = cache
        return cache[key]

    def _wants_compact(self):
        """Compact below 850 px, or when the wide header would need to scroll."""
        if self.viewport().width() < 850:
            return True
        if self._mode != "file" or self._prepared is None:
            return False
        return self.width() < self._header_width(self._prepared["header"])

    def refresh_font(self, screen=None):
        anchor = self.capture_reading_anchor()
        configure_report_browser(self, screen)
        self._apply_finding_font()
        self.document().documentLayout().documentSize()
        self.restore_reading_anchor(anchor)

    def set_active_help(self, key):
        if self._active_help != key:
            anchor = self.capture_reading_anchor()
            self._active_help = key
            self._render_track()
            self.restore_reading_anchor(anchor)

    def focus_report_link(self, scheme, key):
        scroll = self.verticalScrollBar().value()
        cursor = self._anchor_cursor(href=scheme + ":" + quote(key, safe=""))
        if cursor is not None:
            self.setTextCursor(cursor)
        self.setFocus(Qt.OtherFocusReason)
        self.verticalScrollBar().setValue(scroll)

    def resizeEvent(self, event):
        anchor = self.capture_reading_anchor()
        super().resizeEvent(event)
        compact = self._wants_compact()
        if compact != self._compact:
            self._compact = compact
            self._render_track()
        self.restore_reading_anchor(anchor)

    def keyPressEvent(self, event):
        # setHtml resets Qt's anchor focus; retain repeated keyboard activation.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            cursor = self.textCursor()
            href = cursor.charFormat().anchorHref()
            if not href:
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
                href = cursor.charFormat().anchorHref()
            if QUrl(href).scheme() in {"report-review", "report-details", "report-help", "report-visualize"}:
                self.activate_report_link(QUrl(href))
                event.accept()
                return
        super().keyPressEvent(event)

    def activate_report_link(self, url):
        scheme = url.scheme()
        key = unquote(url.path(QUrl.ComponentFormattingOption.FullyEncoded))
        if scheme == "report-review":
            if self._mode == "summary" and key in self._review_files:
                self.reviewRequested.emit(key)
            return
        if self._mode != "file" or not self._prepared:
            return
        if scheme == "report-help":
            if self.help_row(key) is not None:
                self.helpRequested.emit(key)
        elif scheme == "report-visualize":
            if self.visualization_labels(key):
                self.visualizeRequested.emit(key)
        elif scheme == "report-details" and any(r["id"] == key for r in self._prepared["rows"]):
            anchor = self.capture_reading_anchor()
            self._states[self._track_key].symmetric_difference_update({key})
            self._render_track()
            self.restore_reading_anchor(anchor)
            self.focus_report_link(scheme, key)
