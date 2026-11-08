"""Phase 2 detail-tab content shares a 4 px padding: report text, Groups/Config pages, Help dock."""
from urllib.parse import quote

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QSignalBlocker, Qt, QUrl
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTableWidgetItem, QWidget

from sessionprepgui import mainwindow, settings
from sessionprepgui.detail import report
from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane
from sessionprepgui.tracks.table_widgets import (
    DETAIL_TAB_INSET, _TAB_FILE, _TAB_GROUPS, _TAB_SESSION, _TAB_SUMMARY)
from sessionpreplib.detectors.clipping import ClippingDetector
from sessionpreplib.models import DetectorResult, ProcessorResult, SessionContext, Severity, TrackContext
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary

INSET = 4
GAP = 8  # report-to-Help-dock gap


def test_detail_tab_inset_is_four_logical_pixels():
    assert DETAIL_TAB_INSET == INSET


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def _settle(qapp, widget):
    if widget.layout() is not None:
        widget.layout().activate()
    qapp.processEvents()
    qapp.processEvents()


def _track(filename):
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
    return current


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *args: None)
    monkeypatch.setattr(mainwindow.SessionPrepWindow, "_configure_daw_processors",
                        lambda self: setattr(self, "_daw_processors", []))
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    owner = mainwindow.SessionPrepWindow()
    # No audio, devices or DAW services are involved in this layout test.
    monkeypatch.setattr(owner, "_load_waveform", lambda track: None)
    tracks = [_track("A file.wav"), _track("B file.wav")]
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
    qapp.processEvents()
    yield owner, tracks
    owner.close()
    qapp.processEvents()


def _show_tab(qapp, owner, index):
    owner._detail_tabs.setCurrentIndex(index)
    _settle(qapp, owner)
    return owner._detail_tabs.currentWidget()


def _root_margins(browser):
    fmt = browser.document().rootFrame().frameFormat()
    return (fmt.leftMargin(), fmt.topMargin(), fmt.rightMargin(), fmt.bottomMargin())


def _first_text_origin(browser, page):
    """Top-left of the first laid-out text block, in tab-page coordinates."""
    document = browser.document()
    rect = document.documentLayout().blockBoundingRect(document.begin())
    origin = browser.viewport().mapTo(page, QPoint(0, 0))
    scroll = browser.verticalScrollBar().value()
    return origin.x() + round(rect.left()), origin.y() + round(rect.top()) - scroll


def test_summary_and_file_reports_inset_text_on_every_side(qapp, window):
    owner, tracks = window
    summary_page = _show_tab(qapp, owner, _TAB_SUMMARY)
    assert _root_margins(owner._summary_view) == (INSET,) * 4
    summary_origin = _first_text_origin(owner._summary_view, summary_page)

    owner._detail_tabs.setTabEnabled(_TAB_FILE, True)
    owner._file_report.present_track(tracks[0], owner._session)
    file_page = _show_tab(qapp, owner, _TAB_FILE)
    assert _root_margins(owner._file_report) == (INSET,) * 4
    file_origin = _first_text_origin(owner._file_report, file_page)

    # The report widgets are flush with the tab page; only their text is inset.
    assert owner._summary_view is summary_page
    assert owner._file_report.mapTo(file_page, QPoint(0, 0)) == QPoint(0, 0)
    for browser, page in ((owner._summary_view, summary_page),
                          (owner._file_report, file_page)):
        frame = browser.frameWidth()
        assert browser.viewport().mapTo(page, QPoint(0, 0)) == QPoint(frame, frame)
    frame = owner._summary_view.frameWidth()
    assert summary_origin == (frame + INSET, frame + INSET)
    assert file_origin == (frame + INSET, frame + INSET)


def test_report_message_path_keeps_the_same_inset(qapp, window):
    owner, _tracks = window
    owner._summary_view.setHtml(owner._wrap_html(
        '<div style="font-weight:bold;">Analysis Error</div><div>Message</div>'))
    page = _show_tab(qapp, owner, _TAB_SUMMARY)
    frame = owner._summary_view.frameWidth()
    assert _root_margins(owner._summary_view) == (INSET,) * 4
    assert _first_text_origin(owner._summary_view, page) == (frame + INSET, frame + INSET)


def test_report_header_minimum_width_follows_the_shared_inset(qapp, window, monkeypatch):
    import sessionprepgui.detail.browser as browser_module

    owner, tracks = window
    browser = owner._file_report
    browser.present_track(tracks[0], owner._session)
    minimum = browser.minimum_header_width()
    browser._minimum_header_cache = None
    monkeypatch.setattr(browser_module, "DETAIL_TAB_INSET", INSET + 10)
    # Help-width give-back measures the header with the same left/right inset.
    assert browser.minimum_header_width() == minimum + 20


@pytest.mark.parametrize("tab", [_TAB_GROUPS, _TAB_SESSION])
def test_groups_and_config_pages_inset_every_edge(qapp, window, tab):
    owner, _tracks = window
    page = _show_tab(qapp, owner, tab)
    margins = page.layout().contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (INSET,) * 4

    description = next(w for w in page.findChildren(QLabel)
                       if w.text().startswith("Session-local") and w.parentWidget() is page)
    assert description.mapTo(page, QPoint(0, 0)) == QPoint(INSET, INSET)
    revert = next(b for b in page.findChildren(QPushButton)
                  if b.text() == "Revert to Preset")
    right = revert.mapTo(page, QPoint(revert.width(), 0)).x()
    assert right == page.width() - INSET
    bottom = max(w.mapTo(page, QPoint(0, w.height())).y()
                 for w in page.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
                 if w.isVisible())
    assert bottom == page.height() - INSET


def test_waveform_preview_stays_full_width_in_file_tab(qapp, window):
    owner, tracks = window
    owner._file_report.present_track(tracks[0], owner._session)
    page = _show_tab(qapp, owner, _TAB_FILE)
    panel = owner._wf_panel
    assert panel.mapTo(page, QPoint(0, 0)).x() == 0
    assert panel.width() == page.width()


# ── Help dock ────────────────────────────────────────────────────────────


class _Detector:
    id = "d00"
    name = "Detector d00"

    def effective_severity(self, result):
        return result.severity

    def render_html(self, result, current):
        return "<tr><td>Rich result</td></tr>"

    def html_help(self):
        return "<h3>Description</h3><p>Help body.</p>"


@pytest.fixture
def pane(qapp, monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    view = FileReportPane(ReportBrowser({}))
    view.resize(1069, 620)
    view.show()
    _settle(qapp, view)
    current = TrackContext("Lead Vocal.wav", "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    detector = _Detector()
    current.detector_results[detector.id] = DetectorResult(
        detector.id, Severity.PROBLEM, "Review the source recording", {},
        hint="Review the original audio.")
    view.report.present_track(current, SessionContext([current], {}, detectors=[detector]))
    _settle(qapp, view)
    yield view
    view.close()


def _open_help(qapp, pane):
    pane.report.activate_report_link(QUrl("report-help:" + quote("d00", safe="")))
    _settle(qapp, pane)


def _panel_rect(pane):
    panel = pane._help_panel
    top_left = panel.mapTo(pane, QPoint(0, 0))
    return top_left.x(), top_left.y(), top_left.x() + panel.width(), top_left.y() + panel.height()


def test_help_dock_keeps_the_inset_from_top_right_and_bottom(qapp, pane):
    _open_help(qapp, pane)
    assert pane.help_browser.isVisible()
    left, top, right, bottom = _panel_rect(pane)
    assert (top, pane.width() - right, pane.height() - bottom) == (INSET, INSET, INSET)
    assert pane._help_panel.width() == 360
    report_right = pane.report.mapTo(pane, QPoint(pane.report.width(), 0)).x()
    assert left - report_right == GAP
    # The report keeps the full height and its own text inset.
    assert pane.report.mapTo(pane, QPoint(0, 0)) == QPoint(0, 0)
    assert pane.report.height() == pane.height()


def test_closing_help_returns_the_dock_inset_to_the_report(qapp, pane):
    width = pane.report.width()
    _open_help(qapp, pane)
    assert pane.report.width() < width
    pane.close_help()
    _settle(qapp, pane)
    assert not pane._help_dock.isVisible()
    assert pane.report.width() == width == pane.width()


def test_help_width_gives_back_gap_and_header_minimum(qapp, pane, monkeypatch):
    monkeypatch.setattr(pane.report, "minimum_header_width", lambda: 500)
    _open_help(qapp, pane)
    pane.resize(800, 620)
    _settle(qapp, pane)
    # preferred min(360, 0.4 * 800) = 320; available = 800 - 8 gap - 4 inset - 500.
    assert pane._help_panel.width() == 800 - GAP - INSET - 500


def test_constrained_help_dock_keeps_its_inset_and_minimum(qapp, pane, monkeypatch):
    monkeypatch.setattr(pane.report, "minimum_header_width", lambda: 5000)
    _open_help(qapp, pane)
    _settle(qapp, pane)
    left, top, right, bottom = _panel_rect(pane)
    assert pane._help_panel.width() == 140
    assert pane._help_dock.width() == 140 + INSET
    assert (top, pane.width() - right, pane.height() - bottom) == (INSET, INSET, INSET)
    report_right = pane.report.mapTo(pane, QPoint(pane.report.width(), 0)).x()
    assert left - report_right == GAP


def test_scrolling_report_keeps_text_inset_inside_the_viewport(qapp, monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    current = TrackContext("Long.wav", "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    detectors = []
    for index in range(30):
        detector = _Detector()
        detector.id = f"d{index:02}"
        detectors.append(detector)
        current.detector_results[detector.id] = DetectorResult(
            detector.id, Severity.PROBLEM, "Review the source recording", {},
            hint="Review the original audio.")
    browser = ReportBrowser({})
    browser.resize(900, 300)
    browser.show()
    browser.present_track(current, SessionContext([current], {}, detectors=detectors))
    _settle(qapp, browser)
    bar = browser.verticalScrollBar()
    assert bar.isVisible()
    viewport = browser.viewport()
    frame = browser.frameWidth()
    assert viewport.mapTo(browser, QPoint(0, 0)) == QPoint(frame, frame)
    # The bar sits outside the viewport at the pane edge; text keeps 16 inside it.
    assert bar.mapTo(browser, QPoint(0, 0)).x() >= viewport.width() + frame
    document = browser.document()
    rect = document.documentLayout().blockBoundingRect(document.begin())
    assert round(rect.left()) == INSET
    assert round(document.textWidth()) == viewport.width()
    assert _root_margins(browser) == (INSET,) * 4
    bar.setValue(bar.maximum())
    _settle(qapp, browser)
    assert _root_margins(browser) == (INSET,) * 4
    browser.close()


def test_cleared_report_keeps_the_inset(qapp):
    browser = ReportBrowser({})
    browser.clear()
    assert _root_margins(browser) == (INSET,) * 4


def test_narrow_help_header_wraps_instead_of_clipping(qapp, pane, monkeypatch):
    from PySide6.QtWidgets import QLabel

    monkeypatch.setattr(pane.report, "minimum_header_width", lambda: 5000)
    context = pane._help_panel.findChild(QLabel, "detectorHelpContext")
    # Pin a small pixel font: the offscreen platform draws every glyph as a
    # 12 px box, so real-font word widths cannot be assumed here.
    font = context.font()
    font.setPixelSize(7)
    context.setFont(font)
    _open_help(qapp, pane)
    _settle(qapp, pane)
    assert pane._help_panel.width() == 140
    assert context.wordWrap() and pane.help_title.wordWrap()
    # Every word of the kicker fits inside the label's width (no clipping).
    widest = max(context.fontMetrics().horizontalAdvance(w) for w in context.text().split())
    assert widest <= context.width()
    assert context.heightForWidth(context.width()) <= context.height()


def test_phase2_config_tree_has_no_gap_beside_the_splitter(qapp, window):
    from PySide6.QtWidgets import QSplitter

    owner, _tracks = window
    page = _show_tab(qapp, owner, _TAB_SESSION)
    splitter = owner._session_tree.parentWidget()
    assert isinstance(splitter, QSplitter)
    assert splitter.sizes()[0] == owner._session_tree.width()
    assert owner._session_tree.width() == owner._session_tree.maximumWidth()
