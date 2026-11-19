"""Navigation between File findings and repairs above the shared waveform."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFrame, QScrollArea, QStackedWidget

from .repair_editor import TrackRepairEditor
from .repair_options import track_repair_options


class FileReviewPane(QStackedWidget):
    processingFocusRequested = Signal(str)

    def __init__(self, findings, parent=None):
        super().__init__(parent)
        self.findings = findings
        self.report = findings.report
        self.editor = TrackRepairEditor()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidget(self.editor)
        self.scroll.viewport().installEventFilter(self)
        self.addWidget(findings)
        self.addWidget(self.scroll)
        self._opener = None
        self._reading_anchor = self._help_position = None
        self._horizontal_scroll = 0
        self.editor.back_button.clicked.connect(self.back_to_findings)
        self.report.reportChanged.connect(self._check_context)
        # Alt+Left also works while auditioning in the File waveform.
        self._back = QShortcut(QKeySequence("Alt+Left"), parent if parent is not None else self)
        self._escape = QShortcut(QKeySequence(Qt.Key_Escape), self.scroll)
        for shortcut in (self._back, self._escape):
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._keyboard_back)
        self._back.setEnabled(False)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Resize and watched is self.scroll.viewport():
            self.editor.set_viewport_width(event.size().width())
        return super().eventFilter(watched, event)

    def open_track(self, track, session, detector=None):
        if (track is None or track.status != "OK" or track is not self.report.report_track
                or session is not self.report.report_session):
            return
        options = track_repair_options(track, session)
        if not options.available or (detector and detector not in options.targets):
            return
        # None identifies the track's Processing menu. An empty detector
        # identifies the track-wide action in the findings heading.
        if self.currentWidget() is self.findings:
            self._reading_anchor = self.report.capture_reading_anchor()
            self._horizontal_scroll = self.report.horizontalScrollBar().value()
            self._help_position = self.findings.capture_help_position()
        self._opener = detector
        self.editor.show_track(track, session, detector, options)
        self.setCurrentWidget(self.scroll)
        self.editor.set_viewport_width(self.scroll.viewport().width())
        self.scroll.verticalScrollBar().setValue(0)
        self.editor.back_button.setFocus(Qt.OtherFocusReason)
        self._back.setEnabled(True)

    def _check_context(self):
        if self.currentWidget() is self.scroll and (
                self.editor.track is not self.report.report_track or
                self.editor.session is not self.report.report_session or self.editor.track.status != "OK"):
            self._show_findings()

    def _show_findings(self):
        self._back.setEnabled(False)
        self.editor.track = self.editor.session = None
        self._reading_anchor = self._help_position = self._opener = None
        self.setCurrentWidget(self.findings)

    def _keyboard_back(self):
        if QApplication.activePopupWidget() is None:
            self.back_to_findings()

    def back_to_findings(self):
        if self.currentWidget() is not self.scroll:
            return
        filename = self.editor.track.filename
        opener, anchor, help_position = self._opener, self._reading_anchor, self._help_position
        self._show_findings()
        self.findings.layout().activate()
        if opener is None:
            self.processingFocusRequested.emit(filename)
        elif not opener or not (self.report.focus_report_link("report-repair", opener)
                                or self.report.focus_report_link("report-details", opener)):
            self.report.focus_report_link("report-repairs", "track")
        self.report.restore_reading_anchor(anchor)
        self.report.horizontalScrollBar().setValue(self._horizontal_scroll)
        self.findings.restore_help_position(help_position)
