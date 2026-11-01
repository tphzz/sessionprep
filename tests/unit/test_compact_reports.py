from types import SimpleNamespace
from urllib.parse import quote

import pytest
pytest.importorskip("PySide6")
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QAbstractScrollArea

from sessionpreplib.models import DetectorResult, SessionContext, Severity, TrackContext
from sessionpreplib.rendering import build_diagnostic_summary
from sessionprepgui.detail import report
from sessionprepgui.detail.browser import ReportBrowser


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def track(filename="Lead Vocal.wav"):
    return TrackContext(filename, "missing.wav", None, 48000, 1, 96000,
                        "24-bit", "PCM_24", 2.0)


class Detector:
    def __init__(self, key):
        self.id = key
        self.name = "Finding " + key
        self.calls = 0
        self.html_help = "<p>Explanation</p><p>End of help for " + key + "</p>"
    def effective_severity(self, result):
        return result.severity
    def render_html(self, result, current):
        self.calls += 1
        return "<tr><td><b>Rich result " + self.id + "</b></td></tr>"


def session_with_findings(count=1):
    current = track()
    detectors = [Detector(f"d{i:02}") for i in range(count)]
    for detector in detectors:
        current.detector_results[detector.id] = DetectorResult(
            detector.id, Severity.PROBLEM, "Result summary", {},
            hint="Review the source.")
    return SessionContext([current], {}, detectors=detectors), current


def test_custom_fragments_remain_available_and_suppression_is_respected(monkeypatch):
    session, current = session_with_findings()
    data = report.prepare_track_report(current, session, show_clean=False)
    assert len(data["rows"]) == 1
    fragment = "<tr><td><b>Rich result d00</b></td></tr>"
    assert data["rows"][0]["fragment"] == fragment
    assert fragment not in report.render_prepared_track_report(data)
    assert fragment in report.render_prepared_track_report(data, expanded={"d00"})
    session.detectors[0].render_html = lambda *args: ""
    assert report.prepare_track_report(current, session)["rows"] == []
    session.detectors[0].effective_severity = lambda result: None
    assert report.prepare_track_report(current, session)["rows"] == []


def test_interaction_reuses_metadata_and_custom_renderer(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(report, "read_chunks", lambda path: (calls.append(path) or ("WAVE", [])))
    session, current = session_with_findings()
    browser = ReportBrowser({})
    browser.present_track(current, session)
    for url in ("report-details:d00", "report-help:d00", "report-close-help:d00"):
        browser.activate_report_link(QUrl(url))
    assert len(calls) == 1
    assert session.detectors[0].calls == 1
    assert "Rich result d00" in browser.toPlainText()
    browser.activate_report_link(QUrl("report-details:d00"))
    assert "Rich result d00" not in browser.toPlainText()


def test_fiftieth_row_help_grows_the_single_scrollable_document(qapp):
    session, current = session_with_findings(50)
    browser = ReportBrowser({})
    browser.resize(1069, 620)
    browser.show()
    browser.present_track(current, session)
    qapp.processEvents()
    before = browser.verticalScrollBar().maximum()
    assert before > 0
    browser.activate_report_link(QUrl("report-help:d49"))
    qapp.processEvents()
    assert browser.verticalScrollBar().maximum() > before
    browser.verticalScrollBar().setValue(browser.verticalScrollBar().maximum())
    cursor = browser.document().find("End of help for d49")
    assert not cursor.isNull()
    assert browser.cursorRect(cursor).bottom() <= browser.viewport().height()
    assert browser.findChildren(QAbstractScrollArea) == []
    browser.close()


def test_summary_review_uses_structured_identity(qapp):
    filename = "Vocal: alternate & <take>.wav"
    current = track(filename)
    current.detector_results["clipping"] = DetectorResult(
        "clipping", Severity.PROBLEM, "clipped", {"is_clipped": True, "runs": 1})
    session = SessionContext([current], {})
    browser = ReportBrowser({})
    seen = []
    browser.reviewRequested.connect(seen.append)
    browser.present_summary(build_diagnostic_summary(session), session, show_clean=False)
    browser.activate_report_link(QUrl("report-review:" + quote(filename, safe="")))
    assert seen == [filename]
    browser.activate_report_link(QUrl("report-review:unknown.wav"))
    assert seen == [filename]


def test_standalone_and_missing_locations_are_not_dropped(qapp):
    summary = {"clean_count": 0, "total_ok": 1,
               "problems": [{"title": "Session warning", "hint": None,
                              "items": [], "standalone": True}],
               "attention": [], "information": [], "clean": []}
    html = report.render_summary_html(summary, show_clean=False)
    assert "Session warning" in html
    session, current = session_with_findings()
    row = report.prepare_track_report(current, session)["rows"][0]
    assert row["location"] == "\u2014"


def test_real_detector_callable_help_is_prepared_once(qapp):
    from sessionpreplib.detectors.clipping import ClippingDetector
    detector = ClippingDetector()
    detector.configure({})
    current = track()
    current.detector_results["clipping"] = DetectorResult(
        "clipping", Severity.PROBLEM, "Clipped", {"is_clipped": True, "runs": 1})
    session = SessionContext([current], {}, detectors=[detector])
    browser = ReportBrowser({})
    browser.present_track(current, session)
    browser.activate_report_link(QUrl("report-help:clipping"))
    assert "Digital Clipping" in browser.toPlainText()


def test_state_is_local_to_session_file_and_help_does_not_expand(qapp):
    session, current = session_with_findings()
    browser = ReportBrowser({})
    browser.present_track(current, session)
    browser.activate_report_link(QUrl("report-help:d00"))
    assert "Explanation" in browser.toPlainText()
    assert "Rich result" not in browser.toPlainText()
    browser.activate_report_link(QUrl("report-close-help:d00"))
    assert "Explanation" not in browser.toPlainText()
    browser.activate_report_link(QUrl("report-details:d00"))
    other = track("Other.wav")
    session.tracks.append(other)
    browser.present_track(other, session)
    assert "Rich result" not in browser.toPlainText()
    browser.present_track(current, session)
    assert "Rich result" in browser.toPlainText()
    new_session = SessionContext([current], {}, detectors=session.detectors)
    browser.present_track(current, new_session)
    assert "Rich result" not in browser.toPlainText()


def test_literal_percent_escapes_in_filename_are_decoded_once(qapp):
    session, current = session_with_findings()
    current.filename = "Vocal%20literal: & <take>.wav"
    browser = ReportBrowser({})
    seen = []
    browser.reviewRequested.connect(seen.append)
    summary = {"problems": [{"title": "Test", "items": ["legacy"],
                             "item_refs": [{"filename": current.filename, "detail": "Issue"}]}]}
    browser.present_summary(summary, session)
    browser.activate_report_link(QUrl("report-review:" + quote(current.filename, safe="")))
    assert seen == [current.filename]


def test_external_clear_releases_session_and_disclosure_state(qapp):
    import gc
    import weakref
    session, current = session_with_findings()
    reference = weakref.ref(session)
    browser = ReportBrowser({})
    browser.present_track(current, session)
    browser.activate_report_link(QUrl("report-help:d00"))
    browser.clear()
    del session
    gc.collect()
    assert reference() is None
    assert browser.report_session is None
    assert not browser._states
    assert not browser._help_origins


def test_review_selects_identity_after_sort_and_refreshes_selected_widgets(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidgetItem, QTabWidget
    from sessionprepgui.detail.mixin import DetailMixin
    from sessionprepgui.widgets import BatchEditTableWidget
    from sessionprepgui.tracks.table_widgets import _make_analysis_cell
    session, current = session_with_findings()
    other = track("A different file.wav")
    session.tracks.append(other)
    table = BatchEditTableWidget()
    table.setColumnCount(2)
    table.setRowCount(2)
    for row, entry in enumerate(session.tracks):
        table.setItem(row, 0, QTableWidgetItem(entry.filename))
        label, item = _make_analysis_cell('<span style="color:#ffaa00">1A</span>', 1)
        table.setItem(row, 1, item)
        table.setCellWidget(row, 1, label)
    table.setSortingEnabled(True)
    table.sortItems(0, Qt.AscendingOrder)
    table.selectRow(0)
    summary = ReportBrowser({})
    summary.present_summary({"problems": []}, session)
    called = []
    owner = SimpleNamespace(_session=session, _summary_view=summary,
                            _track_table=table, _file_report=ReportBrowser({}),
                            _show_track_detail=called.append)
    DetailMixin._review_summary_file(owner, current.filename)
    assert called == [current]
    assert table.currentItem().text() == current.filename
    assert table.cellWidget(0, 1).text().startswith("<span")
    assert table.cellWidget(1, 1).text() == "1A"
    owner._session = SessionContext([other], {})
    DetailMixin._review_summary_file(owner, current.filename)
    assert called == [current]


def test_repeated_keyboard_activation_preserves_report_link_focus(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    session, current = session_with_findings()
    browser = ReportBrowser({})
    browser.resize(1069, 620)
    browser.show()
    browser.present_track(current, session)
    browser.setFocus()
    QTest.keyClick(browser, Qt.Key_Tab)
    QTest.keyClick(browser, Qt.Key_Return)
    assert "Rich result d00" in browser.toPlainText()
    QTest.keyClick(browser, Qt.Key_Return)
    assert "Rich result d00" not in browser.toPlainText()
    cursor = browser._anchor_cursor(href="report-help:d00")
    browser.setTextCursor(cursor)
    QTest.keyClick(browser, Qt.Key_Return)
    assert "Explanation" in browser.toPlainText()
    cursor = browser._anchor_cursor(href="report-close-help:d00")
    browser.setTextCursor(cursor)
    QTest.keyClick(browser, Qt.Key_Return)
    assert "Explanation" not in browser.toPlainText()
    QTest.keyClick(browser, Qt.Key_Return)
    assert "Explanation" in browser.toPlainText()
    browser.close()


def test_help_reveal_shows_the_complete_card_and_focuses_close(qapp):
    session, current = session_with_findings(50)
    browser = ReportBrowser({})
    browser.resize(1069, 620)
    browser.show()
    browser.present_track(current, session)
    browser.activate_report_link(QUrl("report-help:d49"))
    qapp.processEvents()
    last = browser.document().find("End of help for d49")
    assert browser.cursorRect(last).bottom() < browser.viewport().height()
    cursor = browser.textCursor()
    from PySide6.QtGui import QTextCursor
    cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
    assert cursor.charFormat().anchorHref() == "report-close-help:d49"
    browser.close()
