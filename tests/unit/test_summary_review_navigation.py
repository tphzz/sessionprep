"""Summary review must synchronize the selected tab, visible pane and track."""
from urllib.parse import quote

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, QPoint, QSignalBlocker, Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from sessionprepgui import mainwindow, settings
from sessionprepgui.detail import report
from sessionpreplib.detectors.clipping import ClippingDetector
from sessionpreplib.models import DetectorResult, ProcessorResult, SessionContext, Severity, TrackContext
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *args: None)
    monkeypatch.setattr(mainwindow.SessionPrepWindow, "_configure_daw_processors",
                        lambda self: setattr(self, "_daw_processors", []))
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    owner = mainwindow.SessionPrepWindow()
    # No audio jobs, devices or DAW services are involved in this navigation test.
    monkeypatch.setattr(owner, "_load_waveform", lambda track: None)
    tracks = []
    for filename in ["A file.wav", "B file.wav"]:
        current = TrackContext(filename, "missing-" + filename, None, 48000, 1,
                               96000, "24-bit", "PCM_24", 2.0)
        current.processor_results["bimodal_normalize"] = ProcessorResult(
            "bimodal_normalize", 0.0, "Transient", "peak",
            {"detected_peak_db": -6.0, "detected_rms_db": -18.0,
             "rms_anchor_label": "p95", "target_peak": -6.0, "target_rms": -16.0,
             "gain_for_peak": 0.0, "gain_for_rms": 0.0, "fader_offset": 0.0})
        current.detector_results["clipping"] = DetectorResult(
            "clipping", Severity.PROBLEM, "clipping detected (1 clipped range)",
            {"is_clipped": True, "runs": 1, "ranges": [(1000, 1004, 0)]},
            hint="request reprint / check limiting")
        tracks.append(current)
    detector = ClippingDetector()
    detector.configure({})
    processors = default_processors()
    for processor in processors:
        processor.configure({})
    session = SessionContext(tracks, {}, detectors=[detector], processors=processors)
    owner._session = session
    owner._summary = build_diagnostic_summary(session)
    with QSignalBlocker(owner._track_table):
        owner._track_table.setRowCount(len(tracks))
        for row, current in enumerate(tracks):
            owner._track_table.setItem(row, 0, QTableWidgetItem(current.filename))
        owner._populate_table(session)
    owner._render_summary()
    for index in range(owner._detail_tabs.count()):
        owner._detail_tabs.setTabEnabled(index, True)
    owner._phase_tabs.setTabEnabled(1, True)
    owner._phase_tabs.setCurrentIndex(1)
    owner.resize(1600, 900)
    owner.show()
    owner.activateWindow()
    qapp.processEvents()
    yield owner, tracks
    owner.close()
    qapp.processEvents()


def review(owner, target, method):
    href = "report-review:" + quote(target.filename, safe="")
    if method == "direct":
        owner._summary_view.activate_report_link(QUrl(href))
        return
    browser = owner._summary_view
    cursor = browser._anchor_cursor(href=href)
    assert cursor is not None
    browser.setTextCursor(cursor)
    browser.ensureCursorVisible()
    browser.setFocus()
    if method == "keyboard":
        QTest.keyClick(browser, Qt.Key_Return)
    else:
        rect = browser.cursorRect(cursor)
        point = QPoint(rect.x() + 4, rect.center().y())
        assert QUrl(browser.anchorAt(point)) == QUrl(href)
        QTest.mouseClick(browser.viewport(), Qt.LeftButton, Qt.NoModifier, point)


def assert_file_visible(owner, target):
    assert owner._detail_tabs.currentIndex() == 1
    assert owner._detail_tabs.currentWidget() is owner._file_splitter
    assert owner._file_splitter.isVisibleTo(owner)
    assert owner._file_report.isVisibleTo(owner)
    assert not owner._summary_view.isVisibleTo(owner)
    assert owner._current_track is target
    assert owner._file_report.track_filename == target.filename
    assert owner._file_report.toPlainText().startswith(target.filename)
    assert owner._track_table.currentItem().row() == owner._find_table_row(target.filename)


@pytest.mark.parametrize("method", ["direct", "mouse", "keyboard"])
def test_repeated_summary_review_switches_the_visible_pane(qapp, window, method):
    owner, tracks = window
    for cycle in range(8):
        owner._detail_tabs.setCurrentIndex(0)
        owner._track_table.sortItems(0, Qt.DescendingOrder if cycle % 2 else Qt.AscendingOrder)
        qapp.processEvents()
        target = tracks[1 if cycle % 2 == 0 else 0]
        review(owner, target, method)
        assert_file_visible(owner, target)
        qapp.processEvents()
        assert_file_visible(owner, target)


@pytest.mark.parametrize("destination", [0, 2, 3])
@pytest.mark.parametrize("order", [Qt.AscendingOrder, Qt.DescendingOrder])
def test_leaving_file_does_not_reenter_it_on_focus_restoration(qapp, window, destination, order):
    owner, tracks = window
    owner._track_table.sortItems(0, order)
    review(owner, tracks[1], "direct")
    qapp.processEvents()
    assert_file_visible(owner, tracks[1])
    owner._file_report.setFocus()
    owner._detail_tabs.setCurrentIndex(destination)
    qapp.processEvents()
    assert owner._detail_tabs.currentIndex() == destination
    assert owner._detail_tabs.currentWidget() is owner._detail_tabs.widget(destination)
    assert owner._detail_tabs.widget(destination).isVisibleTo(owner)
    assert not owner._file_splitter.isVisibleTo(owner)
    current_row = owner._track_table.currentRow()
    assert owner._track_table.item(current_row, 0).text() == tracks[1].filename
    assert owner._track_table.batch_selected_keys() == {tracks[1].filename}
    owner._detail_tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert_file_visible(owner, tracks[1])


def test_table_keyboard_row_navigation_still_opens_file(qapp, window):
    owner, tracks = window
    owner._detail_tabs.setCurrentIndex(0)
    owner._track_table.sortItems(0, Qt.AscendingOrder)
    with QSignalBlocker(owner._track_table):
        owner._track_table.setCurrentCell(0, 0)
    owner._track_table.setFocus()
    QTest.keyClick(owner._track_table, Qt.Key_Down)
    qapp.processEvents()
    assert_file_visible(owner, tracks[1])



def test_tab_departure_preserves_multiple_selected_rows_and_current_track(qapp, window):
    owner, tracks = window
    review(owner, tracks[1], "direct")
    table = owner._track_table
    with QSignalBlocker(table):
        table.selectionModel().select(
            table.model().index(owner._find_table_row(tracks[0].filename), 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows)
    owner._file_report.setFocus()
    owner._detail_tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert table.batch_selected_keys() == {current.filename for current in tracks}
    assert table.item(table.currentRow(), 0).text() == tracks[1].filename
    owner._detail_tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert_file_visible(owner, tracks[1])


def test_review_enables_file_before_switching_to_its_content(qapp, window):
    owner, tracks = window
    owner._detail_tabs.setCurrentIndex(0)
    owner._detail_tabs.setTabEnabled(1, False)
    review(owner, tracks[1], "mouse")
    qapp.processEvents()
    assert owner._detail_tabs.isTabEnabled(1)
    assert_file_visible(owner, tracks[1])
