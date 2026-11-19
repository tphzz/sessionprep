import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QHBoxLayout, QTableWidget, QTableWidgetItem, QTabWidget, QWidget

from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane
from sessionprepgui.detail.mixin import DetailMixin
from sessionprepgui.detail.repair_editor import TrackRepairEditor
from sessionprepgui.tracks.columns_mixin import TrackColumnsMixin
from sessionpreplib.detectors.subsonic import SubsonicDetector
from sessionpreplib.models import DetectorResult, SessionContext, Severity, TrackContext
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor_pane(app):
    track = TrackContext("Synth bus.wav", "missing.wav", None, 44100, 2, 88200, "16-bit", "PCM_16", 2)
    detector = SubsonicDetector()
    detector.configure({})
    track.detector_results[detector.id] = DetectorResult(detector.id, Severity.INFO, "rumble",
        {"subsonic_warn": True, "subsonic_ratio_db": -10, "per_channel": {}})
    track.repair_measurements = {
        "suggested": {"dc_remove": True, "low_cut": True},
        "inherited": {"dc_remove": False, "low_cut": False, "ultrasonic": False},
        "original": {"signed_dc": [-0.013, 0.001], "peak_dbfs": -4.2, "subsonic_ratio_db": -10.0},
        "corrected": {"signed_dc": [0, 0], "peak_dbfs": -3.8, "subsonic_ratio_db": -28.9},
    }
    track.repair_overrides = {"low_cut": True, "low_cut_hz": 14}
    track.repair_recipe = RepairRecipe(low_cut_hz=14)
    session = SessionContext([track], {"repair_low_cut_hz": 10.0}, detectors=[detector])
    browser = ReportBrowser({})
    editor = TrackRepairEditor(browser)
    browser.repairRequested.connect(lambda det: editor.open_track(track, session, det))
    pane = FileReportPane(browser)
    pane.resize(1010, 520)
    pane.show()
    browser.present_track(track, session)
    app.processEvents()
    yield pane, editor, track, session
    pane.close()


def test_repair_link_choices_reset_and_source_evidence(editor_pane, app):
    pane, editor, track, session = editor_pane
    report = pane.report
    changes = []
    editor.choiceChanged.connect(lambda t, key, value: changes.append((t, key, value)))
    assert "Review repair" in report.toPlainText()
    report.activate_report_link(QUrl("report-repair:clipping"))
    assert editor.isHidden()
    report.activate_report_link(QUrl("report-repair:subsonic"))
    app.processEvents()
    assert editor.isVisible()
    row = editor.rows["low_cut"]
    assert [row.choice.itemText(i) for i in range(3)] == ["Use preset (Off)", "On", "Off"]
    assert row.choice.currentData() is True
    assert editor.cutoff.value() == 14
    assert "−10.0 → −28.9 dB" in row.evidence.text()
    assert "−4.2 → −3.8 dBFS" in editor.peak.text()
    assert "44.1 kHz" in editor.rows["ultrasonic"].description.text()
    assert editor.rows["ultrasonic"].choice.isHidden()
    row.choice.setCurrentIndex(2)
    row.reset.click()
    editor.cutoff.setValue(12)
    editor.cutoff_reset.click()
    assert [(key, value) for _, key, value in changes] == [
        ("low_cut", False), ("low_cut", None), ("low_cut_hz", 12), ("low_cut_hz", None)]
    assert all(t is track for t, _, _ in changes)
    track.repair_measurements["inherited"]["low_cut"] = True
    track.repair_overrides.clear()
    session.config["repair_low_cut_hz"] = 10.125
    editor.refresh(track, session)
    assert row.choice.currentText() == "Use preset (On)" and editor.cutoff.value() == 10.125
    assert len(changes) == 4
    QTest.keyClick(row.choice, Qt.Key_Escape)
    app.processEvents()
    assert editor.isHidden() and report.hasFocus()
    assert report.textCursor().charFormat().anchorHref() == "report-repair:subsonic"


def test_narrow_editor_and_help_share_width_and_scroll_independently(editor_pane, app):
    pane, editor, track, session = editor_pane
    pane.resize(400, 475)
    report = pane.report
    report.activate_report_link(QUrl("report-help:subsonic"))
    report.activate_report_link(QUrl("report-repair:subsonic"))
    app.processEvents()
    assert pane.active_detector_id == "subsonic" and editor.isVisible()
    assert editor.width() <= report.viewport().width()
    assert report.horizontalScrollBar().maximum() == 0
    row = editor.rows["low_cut"]
    assert row.choice.isVisible() and row.reset.geometry().right() < row.width()
    assert not editor.rows["dc_remove"].isVisible()
    bar = report.verticalScrollBar()
    bar.setValue(bar.maximum())
    app.processEvents()
    assert editor.geometry().bottom() <= report.viewport().height()
    assert pane.help_browser.verticalScrollBar().value() == 0
    report.present_track(track, SessionContext([track], {}))
    assert editor.isHidden()


def test_settling_in_a_previously_silent_gap_is_visible_in_the_editor(editor_pane):
    pane, editor, track, session = editor_pane
    track.repair_measurements["boundary_settling"] = [
        {"channel": 1, "start_sample": 48000, "end_sample": 70000, "peak_dbfs": -19.97}]
    pane.report.activate_report_link(QUrl("report-repair:subsonic"))
    assert editor.error.isVisible()
    message = editor.error.text()
    assert "Boundary settling" in message and "Ch 2" in message
    assert "48000" in message and "70000" in message and "20.0 dBFS" in message
    track.repair_measurements.pop("boundary_settling")
    editor.refresh(track, session)
    assert editor.error.isHidden()


def test_peak_layout_releases_moved_native_items_on_reflow(editor_pane, app):
    from shiboken6 import isValid

    pane, editor, track, session = editor_pane
    grid = editor.peak_layout
    label = grid.itemAtPosition(0, 0)
    editor.open_track(track, session)
    app.processEvents()
    for width, position in [(450, (1, 0)), (1010, (0, 1))]:
        old_item = grid.itemAt(grid.indexOf(editor.peak))
        pane.resize(width, 520)
        app.processEvents()
        editor.reflow()
        # An implicit Qt layout move can leave a Python wrapper registered
        # for deleted native memory, which a newly created worker can reuse.
        assert not isValid(old_item)
        assert grid.itemAtPosition(0, 0) is label
        assert grid.itemAtPosition(*position).widget() is editor.peak


def test_processing_menu_opens_same_editor_and_restores_focus_without_normalization(app, editor_pane):
    pane, editor, track, session = editor_pane
    class Table(TrackColumnsMixin, DetailMixin):
        pass
    owner = Table()
    owner._repair_editor = editor
    owner._current_track = track
    owner._session = session
    owner._detail_tabs = QTabWidget()
    owner._detail_tabs.addTab(QWidget(), "Summary")
    owner._detail_tabs.addTab(pane, "File")
    owner._detail_tabs.setCurrentIndex(1)
    owner._track_table = QTableWidget(1, 8)
    owner._track_table.setItem(0, 0, QTableWidgetItem(track.filename))
    owner._create_processing_button(0, track)
    editor.processingFocusRequested.connect(owner._focus_processing_menu)
    window = QWidget()
    layout = QHBoxLayout(window)
    layout.addWidget(owner._track_table)
    layout.addWidget(owner._detail_tabs)
    window.show()
    window.activateWindow()
    app.processEvents()
    button = owner._track_table.cellWidget(0, 7)
    action, = [a for a in button.menu().actions() if a.text() == "Track repairs…"]
    action.trigger()
    assert editor.track is track and editor.isVisible()
    QTest.mouseClick(editor.close_button, Qt.LeftButton)
    app.processEvents()
    assert editor.isHidden() and button.hasFocus()
    pane.setParent(None)
    window.close()


def test_local_only_suggestion_can_be_reviewed_without_changing_the_detector(editor_pane, app):
    pane, editor, track, session = editor_pane
    detector = session.detectors[0]
    raw = DetectorResult("subsonic", Severity.CLEAN, "whole-track ratio below threshold",
                         {"subsonic_ratio_db": -27.8, "subsonic_warn": False})
    track.detector_results["subsonic"] = raw
    local = {"band_dbfs": -44.95, "ratio_db": -11.78, "regions": [[0, 44099, 1]]}
    track.repair_measurements["low_band"] = track.repair_measurements["original"]["low_band"] = local
    track.repair_measurements["corrected"]["low_band"] = {"band_dbfs": -50.32, "ratio_db": -17, "regions": []}
    pane.report.present_track(track, session, show_clean=False)
    assert "local energy" in pane.report.toPlainText() and "Review repair" in pane.report.toPlainText()
    pane.report.activate_report_link(QUrl("report-repair:subsonic-local"))
    assert editor.isVisible()
    assert "−45.0 → −50.3 dBFS RMS" in editor.rows["low_cut"].evidence.text()
    assert "1 region → 0 regions" in editor.rows["low_cut"].evidence.text()
    assert "Right" in editor.rows["low_cut"].evidence.text()
    pane.report.present_track(track, session, show_clean=True)
    assert "Clean" in pane.report.toPlainText() and "Information" in pane.report.toPlainText()
    assert track.detector_results["subsonic"] is raw and raw.severity is Severity.CLEAN
    detector.configure({"subsonic_report_as": "skip"})
    pane.report.present_track(track, session, show_clean=False)
    assert "Review repair" not in pane.report.toPlainText()


def test_high_rate_evidence_and_headroom_recovery_use_the_selected_track(editor_pane):
    pane, editor, track, session = editor_pane
    track.samplerate = 96000
    track.repair_measurements["suggested"]["ultrasonic"] = True
    track.repair_measurements["original"]["ultrasonic"] = {
        "regions": [[0, 62399, 0], [0, 62399, 1]], "band_dbfs": -33.5, "eligible": True}
    track.repair_measurements["corrected"]["ultrasonic"] = {"regions": [], "band_dbfs": -121}
    session.config["_prepare_errors"] = [(track.filename, "Output peak +0.58 dBFS exceeds PCM_16.")]
    session.config["_output_gain_adjustments"] = {track.filename: {"peak_db": 0.58, "gain_db": 0, "reduction_db": 0.6}}
    editor.open_track(track, session)
    high = editor.rows["ultrasonic"]
    assert high.choice.isVisible() and "Qualifying windows span 0.65 s" in high.description.text()
    assert "−33.5" in high.evidence.text() and "dBFS RMS" in high.evidence.text()
    requested = []
    editor.adjustGainRequested.connect(requested.append)
    editor.adjust_gain.click()
    assert requested == [track] and "Output peak +0.58" in editor.error.text()
    session.config["_output_gain_adjustments"].clear()
    editor.refresh(track, session)
    assert not editor.adjust_gain.isVisible()
