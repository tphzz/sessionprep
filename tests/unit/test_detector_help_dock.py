"""Behavior of the file report's independent detector Help dock."""

import gc
import weakref
from urllib.parse import quote

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sessionpreplib.models import DetectorResult, SessionContext, Severity, TrackContext
from sessionprepgui.detail import report
from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane


class _Detector:
    def __init__(self, key, *, long_help=False):
        self.id = key
        self.name = "Detector " + key
        self.render_calls = 0
        self.help_calls = 0
        paragraphs = 60 if long_help else 2
        self.help_html = (
            "<h3>Description</h3>"
            + "".join(f"<p>Explanation {key}, paragraph {i}: review the original "
                      "audio and confirm that the finding is intentional.</p>"
                      for i in range(paragraphs))
            + f"<h3>Interpretation</h3><p>End of help for {key}</p>"
        )

    def effective_severity(self, result):
        return result.severity

    def render_html(self, result, current):
        self.render_calls += 1
        return f"<tr><td>Rich result {self.id}</td></tr>"

    def html_help(self):
        self.help_calls += 1
        return self.help_html


def _session(count=2, *, filename="Lead Vocal.wav", long_help=False):
    current = TrackContext(filename, "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    detectors = [_Detector(f"d{i:02}", long_help=long_help) for i in range(count)]
    for detector in detectors:
        current.detector_results[detector.id] = DetectorResult(
            detector.id, Severity.PROBLEM, "Review the source recording", {},
            hint="Review the original audio at this location and confirm whether "
                 "the finding is intentional before preparing the session.")
    return SessionContext([current], {}, detectors=detectors), current


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _settle(qapp, pane):
    if pane.layout() is not None:
        pane.layout().activate()
    qapp.processEvents()
    qapp.processEvents()


@pytest.fixture
def pane(qapp, monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    browser = ReportBrowser({})
    view = FileReportPane(browser)
    view.resize(1069, 620)
    view.show()
    view.activateWindow()
    _settle(qapp, view)
    yield view
    view.close()


def _activate(pane, detector, *, scheme="report-help"):
    pane.report.activate_report_link(QUrl(scheme + ":" + quote(detector, safe="")))


def _link_cursor(browser, detector, *, scheme="report-help"):
    cursor = browser._anchor_cursor(href=scheme + ":" + quote(detector, safe=""))
    assert cursor is not None, f"Missing {scheme} link for {detector}"
    return cursor


def _link_top(browser, detector):
    return browser.cursorRect(_link_cursor(browser, detector)).top()


def _read_at(browser, detector, top=40):
    bar = browser.verticalScrollBar()
    bar.setValue(bar.value() + _link_top(browser, detector) - top)


def _focused_href(browser):
    cursor = browser.textCursor()
    href = cursor.charFormat().anchorHref()
    if not href:
        cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
        href = cursor.charFormat().anchorHref()
    return href


def test_help_uses_one_right_panel_and_keeps_report_content_separate(qapp, pane):
    browser = pane.report
    session, current = _session()
    browser.present_track(current, session)
    _settle(qapp, pane)
    assert not pane.help_browser.isVisible()
    assert pane.active_detector_id is None

    _activate(pane, "d00")
    _settle(qapp, pane)
    assert pane.report is browser
    assert pane.active_detector_id == "d00"
    assert pane.help_browser.isVisible()
    assert pane.help_title.text() == session.detectors[0].name
    assert "Explanation d00" in pane.help_browser.toPlainText()
    assert "Explanation d00" not in browser.toPlainText()
    assert pane.help_browser.mapTo(pane, pane.help_browser.rect().topLeft()).x() >= (
        browser.mapTo(pane, browser.rect().topRight()).x())
    help_browser = pane.help_browser

    _activate(pane, "d01")
    _settle(qapp, pane)
    assert pane.help_browser is help_browser
    assert pane.active_detector_id == "d01"
    assert pane.help_title.text() == session.detectors[1].name
    assert "Explanation d01" in help_browser.toPlainText()
    assert "Explanation d00" not in help_browser.toPlainText()


def test_repeating_help_retains_help_scroll_switching_resets_only_help(qapp, pane):
    session, current = _session(50, long_help=True)
    pane.report.present_track(current, session)
    _settle(qapp, pane)
    _read_at(pane.report, "d25")
    _activate(pane, "d25")
    _settle(qapp, pane)
    help_bar = pane.help_browser.verticalScrollBar()
    assert help_bar.maximum() > 0
    help_bar.setValue(help_bar.maximum() // 2)
    help_position = help_bar.value()
    report_position = pane.report.verticalScrollBar().value()

    _activate(pane, "d25")
    _settle(qapp, pane)
    assert pane.active_detector_id == "d25"
    assert help_bar.value() == help_position
    assert pane.report.verticalScrollBar().value() == report_position

    _activate(pane, "d26")
    _settle(qapp, pane)
    assert pane.active_detector_id == "d26"
    assert help_bar.value() == 0
    assert pane.report.verticalScrollBar().value() == report_position


def test_help_and_details_reuse_prepared_metadata_and_detector_content(qapp, pane, monkeypatch):
    reads = []
    monkeypatch.setattr(report, "read_chunks", lambda path: (reads.append(path) or ("WAVE", [])))
    session, current = _session(long_help=True)
    pane.report.present_track(current, session)
    for detector in ("d00", "d01", "d01"):
        _activate(pane, detector)
    _activate(pane, "d01", scheme="report-details")
    pane.close_help()
    _settle(qapp, pane)
    assert reads == [current.filepath]
    assert [detector.render_calls for detector in session.detectors] == [1, 1]
    assert [detector.help_calls for detector in session.detectors] == [1, 1]


def test_details_and_help_remain_independent(qapp, pane):
    session, current = _session(long_help=True)
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    _settle(qapp, pane)
    bar = pane.help_browser.verticalScrollBar()
    assert "Rich result d00" not in pane.report.toPlainText()
    bar.setValue(bar.maximum() // 2)
    position = bar.value()

    for _ in range(2):
        _activate(pane, "d00", scheme="report-details")
        _settle(qapp, pane)
        assert pane.active_detector_id == "d00"
        assert pane.help_browser.isVisible()
        assert bar.value() == position
    assert "Rich result d00" not in pane.report.toPlainText()

    _activate(pane, "d00", scheme="report-details")
    _activate(pane, "d01")
    pane.close_help()
    _settle(qapp, pane)
    assert "Rich result d00" in pane.report.toPlainText()


def test_fiftieth_row_help_has_independent_scroll_and_complete_content(qapp, pane):
    session, current = _session(50, long_help=True)
    pane.report.present_track(current, session)
    _settle(qapp, pane)
    _read_at(pane.report, "d49", top=100)
    _activate(pane, "d49")
    _settle(qapp, pane)
    bar = pane.help_browser.verticalScrollBar()
    assert bar.maximum() > 0
    report_scroll = pane.report.verticalScrollBar().value()
    bar.setValue(bar.maximum())
    cursor = pane.help_browser.document().find("End of help for d49")
    assert not cursor.isNull()
    assert pane.help_browser.cursorRect(cursor).bottom() <= pane.help_browser.viewport().height()
    assert pane.report.verticalScrollBar().value() == report_scroll


def test_report_reading_anchor_survives_help_width_reflow_and_close(qapp, pane):
    session, current = _session(50)
    pane.report.present_track(current, session)
    _settle(qapp, pane)
    _read_at(pane.report, "d25")
    before = _link_top(pane.report, "d25")
    width = pane.report.viewport().width()
    _activate(pane, "d25")
    _settle(qapp, pane)
    assert pane.report.viewport().width() < width
    assert abs(_link_top(pane.report, "d25") - before) <= 3
    pane.close_help()
    _settle(qapp, pane)
    assert pane.report.viewport().width() == width
    assert abs(_link_top(pane.report, "d25") - before) <= 3


def test_close_returns_help_link_focus_without_scrolling_away_from_reading_position(qapp, pane):
    session, current = _session(50)
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    _settle(qapp, pane)
    _read_at(pane.report, "d25")
    before = _link_top(pane.report, "d25")
    QTest.mouseClick(pane.close_button, Qt.LeftButton)
    _settle(qapp, pane)
    assert pane.active_detector_id is None
    assert not pane.help_browser.isVisible()
    assert pane.report.hasFocus()
    assert _focused_href(pane.report) == "report-help:d00"
    assert abs(_link_top(pane.report, "d25") - before) <= 3


def test_keyboard_open_escape_close_and_repeat_return(qapp, pane):
    session, current = _session()
    pane.report.present_track(current, session)
    cursor = pane.report.textCursor()
    cursor.movePosition(QTextCursor.Start)
    pane.report.setTextCursor(cursor)
    pane.report.setFocus(Qt.TabFocusReason)
    for _ in range(20):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
        _settle(qapp, pane)
        if pane.report.hasFocus() and _focused_href(pane.report) == "report-help:d00":
            break
    else:
        pytest.fail("Detector Help link could not be reached with Tab")
    QTest.keyClick(pane.report, Qt.Key_Return)
    _settle(qapp, pane)
    assert pane.active_detector_id == "d00"
    pane.help_browser.setFocus(Qt.TabFocusReason)
    QTest.keyClick(pane.help_browser, Qt.Key_Escape)
    _settle(qapp, pane)
    assert pane.active_detector_id is None
    assert pane.report.hasFocus()
    assert _focused_href(pane.report) == "report-help:d00"
    QTest.keyClick(pane.report, Qt.Key_Return)
    _settle(qapp, pane)
    assert pane.active_detector_id == "d00"


@pytest.mark.parametrize("replacement", ["file", "session", "clear", "error", "summary"])
def test_changing_report_context_closes_help(qapp, pane, replacement):
    session, current = _session()
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    _settle(qapp, pane)
    assert pane.active_detector_id == "d00"
    if replacement == "file":
        _, other = _session(filename="Another file.wav")
        session.tracks.append(other)
        pane.report.present_track(other, session)
    elif replacement == "session":
        pane.report.present_track(current, SessionContext([current], {}, detectors=session.detectors))
    elif replacement == "clear":
        pane.report.clear()
    elif replacement == "error":
        pane.report.setHtml("<p>Could not load this file.</p>")
    else:
        pane.report.present_summary({"problems": []}, session)
    _settle(qapp, pane)
    assert pane.active_detector_id is None
    assert not pane.help_browser.isVisible()


@pytest.mark.parametrize("suppression", ["removed", "skipped", "renderer", "help", "clean"])
def test_refresh_closes_help_when_detector_is_no_longer_available(qapp, pane, suppression):
    session, current = _session()
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    if suppression == "removed":
        current.detector_results.pop("d00")
    elif suppression == "skipped":
        session.detectors[0].effective_severity = lambda result: None
    elif suppression == "renderer":
        session.detectors[0].render_html = lambda *args: ""
    elif suppression == "help":
        session.detectors[0].help_html = ""
    else:
        current.detector_results["d00"].severity = Severity.CLEAN
    pane.report.present_track(current, session, show_clean=False)
    _settle(qapp, pane)
    assert pane.active_detector_id is None
    assert not pane.help_browser.isVisible()


def test_unknown_links_and_previous_inline_close_do_not_replace_current_help(qapp, pane):
    session, current = _session()
    pane.report.present_track(current, session)
    _activate(pane, "d01")
    for href in ("report-help:missing", "report-help:", "report-help:%zz",
                 "report-details:missing", "report-close-help:d00", "unknown:d00"):
        pane.report.activate_report_link(QUrl(href))
    _settle(qapp, pane)
    assert pane.active_detector_id == "d01"
    assert "Explanation d01" in pane.help_browser.toPlainText()


def test_clearing_help_drops_session_and_queued_old_actions_cannot_reopen_it(qapp, pane):
    session, current = _session()
    reference = weakref.ref(session)
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    QTimer.singleShot(0, lambda: _activate(pane, "d00"))
    QTimer.singleShot(0, lambda: _activate(pane, "d00", scheme="report-close-help"))
    pane.report.clear()
    del session
    gc.collect()
    _settle(qapp, pane)
    assert reference() is None
    assert pane.active_detector_id is None
    assert not pane.help_browser.isVisible()


def test_monitor_font_refresh_preserves_report_and_help_reading_positions(qapp, pane):
    from types import SimpleNamespace
    from sessionprepgui.mainwindow import SessionPrepWindow

    class Screen:
        def __init__(self, scale):
            self.scale = scale

        def devicePixelRatio(self):
            return self.scale

        def logicalDotsPerInch(self):
            return 96.0

    session, current = _session(50, long_help=True)
    pane.report.present_track(current, session)
    owner = SimpleNamespace(_summary_view=None, _file_report=pane.report,
                            _file_report_pane=pane)
    SessionPrepWindow._refresh_report_browser_fonts(owner, Screen(1.0))
    _settle(qapp, pane)
    _read_at(pane.report, "d25")
    _activate(pane, "d25")
    _settle(qapp, pane)
    help_bar = pane.help_browser.verticalScrollBar()
    help_bar.setValue(help_bar.maximum() // 2)
    help_cursor = pane.help_browser.cursorForPosition(pane.help_browser.viewport().rect().topLeft())
    help_top = pane.help_browser.cursorRect(help_cursor).top()
    report_top = _link_top(pane.report, "d25")

    for scale in (2.0, 1.0):
        SessionPrepWindow._refresh_report_browser_fonts(owner, Screen(scale))
        _settle(qapp, pane)
        assert abs(_link_top(pane.report, "d25") - report_top) <= 3
        assert abs(pane.help_browser.cursorRect(help_cursor).top() - help_top) <= 3
        assert pane.active_detector_id == "d25"


def test_stock_normalization_report_stays_readable_with_help_at_large_scale(qapp, pane):
    from sessionpreplib.models import ProcessorResult
    from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor

    class Screen:
        def devicePixelRatio(self):
            return 1.5
        def logicalDotsPerInch(self):
            return 96.0

    session, current = _session()
    processor = BimodalNormalizeProcessor()
    processor.configure({})
    session.processors = [processor]
    current.processor_results[processor.id] = ProcessorResult(
        processor.id, 5.0, "Sustained", "RMS",
        {"detected_peak_db": -15, "detected_rms_db": -21, "rms_anchor_label": "p95",
         "target_peak": -6.0, "target_rms": -16.0, "gain_for_peak": 9.0,
         "gain_for_rms": 5.0, "fader_offset": -5.0})
    pane.resize(713, 412)
    pane.report.refresh_font(Screen())
    from PySide6.QtGui import QFontInfo
    if not QFontInfo(pane.report.font()).family():
        pytest.skip("Geometry requires a resolved platform font; Qt exposes no font face")
    pane.refresh_fonts(Screen())
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    _settle(qapp, pane)
    assert pane.help_browser.isVisible()
    assert pane.report.horizontalScrollBar().maximum() == 0
    chosen = pane.report.document().find("Chosen")
    assert not chosen.isNull()
    assert pane.report.cursorRect(chosen).right() <= pane.report.viewport().width()

def test_help_opens_on_reading_area_and_close_remains_keyboard_reachable(qapp, pane):
    session, current = _session()
    pane.report.present_track(current, session)
    _activate(pane, "d00")
    _settle(qapp, pane)
    assert pane.help_browser.hasFocus()
    assert not pane.close_button.hasFocus()
    QTest.keyClick(pane.help_browser, Qt.Key_Backtab)
    _settle(qapp, pane)
    assert pane.close_button.hasFocus()
    QTest.keyClick(pane.close_button, Qt.Key_Escape)
    _settle(qapp, pane)
    assert pane.active_detector_id is None
    assert pane.report.hasFocus()


def test_help_close_artwork_is_centered_in_its_icon_canvas(qapp, pane):
    image = pane.close_button.icon().pixmap(16, 16).toImage()
    points = [(x, y) for x in range(image.width()) for y in range(image.height())
              if image.pixelColor(x, y).alpha() > 0]
    assert points
    xs, ys = zip(*points)
    assert abs((min(xs) + max(xs)) / 2 - (image.width() - 1) / 2) <= 0.5
    assert abs((min(ys) + max(ys)) / 2 - (image.height() - 1) / 2) <= 0.5
