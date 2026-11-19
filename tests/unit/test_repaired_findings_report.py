from dataclasses import replace

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.report import prepare_track_report, render_prepared_track_report, select_report_source
from sessionprepgui.waveform.overlay import issue_y_bounds
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import CorrectedAnalysis, DetectorResult, IssueLocation, SessionContext, Severity, TrackContext
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pair():
    track = TrackContext("Synth.wav", "missing.wav", None, 44100, 2, 44100, "24-bit", "PCM_24", 1,
                         repair_recipe=RepairRecipe(dc_remove=True))
    issues = [IssueLocation(400, 1400, ch, Severity.INFO, "subsonic", "Energy below 30 Hz", 0, 30)
              for ch in (0, 1)]
    low = DetectorResult("subsonic", Severity.INFO, "Subsonic content", {
        "subsonic_ratio_db": -10.6, "subsonic_warn": True,
        "per_channel": {0: {"warn": True}, 1: {"warn": True}}}, issues=issues)
    clean = DetectorResult("subsonic", Severity.CLEAN, "Below threshold", {
        "subsonic_ratio_db": -29.7, "subsonic_warn": False,
        "per_channel": {0: {"warn": False}, 1: {"warn": False}}})
    clip = DetectorResult("clipping", Severity.CLEAN, "No clipping", {"runs": 0})
    new = replace(clip, severity=Severity.PROBLEM, data={"runs": 1, "is_clipped": True},
                  issues=[IssueLocation(800, 804, 0, Severity.PROBLEM, "clipping", "Clipping")])
    track.detector_results = {"subsonic": low, "clipping": clip}
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": clean, "clipping": new})
    detectors = default_detectors()
    for d in detectors:
        d.configure({})
    return track, SessionContext([track], {}, detectors=detectors)


def test_cached_report_switches_location_actions_and_retains_open_rows(pair, monkeypatch, qapp):
    track, session = pair
    view = ReportBrowser({})
    view.present_track(track, session, show_clean=False)
    view.activate_report_link(QUrl("report-details:subsonic"))
    generation = view.context_generation
    requests = []
    view.sourceLocationsRequested.connect(lambda key, corrected: requests.append((key, corrected)))
    def unexpected(*args, **kwargs):
        pytest.fail("Switching source must use the cached report")
    monkeypatch.setattr("sessionprepgui.detail.browser.prepare_track_report", unexpected)
    assert "✓ Correction applied" in view.toPlainText() and "New after repair" in view.toPlainText()
    assert view.visualization_labels("subsonic") == ("subsonic",)
    view.activate_report_link(QUrl("report-corrected:clipping"))
    view.set_report_source(True)
    assert view.context_generation == generation and "subsonic" in view._states[track.filename]
    assert view.visualization_labels("subsonic") == () and view.visualization_labels("clipping") == ("clipping",)
    assert "−29.7 dB" in view.toPlainText() and "−10.6 dB" in view.toPlainText()
    view.activate_report_link(QUrl("report-original:subsonic"))
    view.activate_report_link(QUrl("report-corrected:clipping"))
    assert requests == [("clipping", True), ("subsonic", False)]
    view.set_report_source(False)
    assert view.visualization_labels("subsonic") == ("subsonic",)
    view.close()
    qapp.processEvents()


def test_partial_channels_are_identified_in_both_layouts(pair):
    track, session = pair
    original = track.detector_results["subsonic"]
    track.corrected_analysis.results["subsonic"] = replace(original, issues=original.issues[1:], data={
        "subsonic_ratio_db": -16.4, "subsonic_warn": True,
        "per_channel": {0: {"warn": False}, 1: {"warn": True}}})
    data = prepare_track_report(track, session, show_clean=False)
    for narrow in (False, True):
        html = render_prepared_track_report(data, expanded={"subsonic"}, narrow=narrow)
        assert "2 affected channels → 1" in html
        assert "Still detected on the right channel" in html and "−16.4 dB" in html
        assert "✓ Correction applied" not in html


def test_expanded_file_comparison_aligns_readings_and_action_at_desktop_width(pair, qapp, form_font):
    track, session = pair
    original = track.detector_results["subsonic"]
    original.issues = original.issues * 5
    view = ReportBrowser({})
    try:
        view.resize(1069, 550)
        view.show()
        view.present_track(track, session, show_clean=False)
        view.set_report_source(True)
        view.activate_report_link(QUrl("report-details:subsonic"))
        qapp.processEvents()
        document = view.document()
        table = document.find("Measurement").currentTable()
        def position(row, column):
            return view.cursorRect(table.cellAt(row, column).firstCursorPosition()).topLeft()

        assert table.rows() == 3
        assert [table.cellAt(1, c).firstCursorPosition().block().text() for c in range(3)] == [
            "Energy at or below 30 Hz", "−10.6 dB", "−29.7 dB"]
        assert "10 → 0 locations" in view.toPlainText() and "Locations" not in view.toPlainText().splitlines()
        label, original_column, corrected_column = (position(0, c) for c in range(3))
        assert original_column.x() - label.x() == pytest.approx(261, abs=22)
        assert corrected_column.x() - label.x() == pytest.approx(451, abs=26)
        assert position(1, 0).x() == label.x() and position(1, 1).x() == original_column.x()
        assert table.cellAt(0, 0).format().toTableCellFormat().bottomBorder() == 1
        assert position(2, 0).y() - position(1, 0).y() >= view.fontMetrics().height() + 6
        action = view._repair_buttons["report-repair:subsonic"].geometry()
        last_line = view.cursorRect(table.cellAt(2, 0).firstCursorPosition())
        assert action.left() > corrected_column.x() and abs(action.center().y() - last_line.center().y()) <= 8
        assert view.horizontalScrollBar().maximum() == 0
        for width in (1092, 1064, 1004, 965, 924, 884, 850, 650, 476, 400, 210):
            view.resize(width, 550)
            qapp.processEvents()
            assert view.horizontalScrollBar().maximum() == 0
            assert "−29.7 dB" in view.toPlainText() and "Review repair" in view.toPlainText()
            if view._layout == "wide":
                count = document.find("10 → 0 locations").block()
                assert count.layout().lineCount() == 1
        view.resize(1069, 550)
        view.activate_report_link(QUrl("report-details:clipping"))
        qapp.processEvents()
        clipping_table = document.find("Clipped ranges").currentTable()
        assert clipping_table.columns() == 3
    finally:
        view.close()


def test_pending_and_failed_checks_preserve_channel_specific_original_locations(pair):
    track, session = pair
    track._cache["repair_verification_pending"] = True
    data = prepare_track_report(track, session, show_clean=False)
    select_report_source(data, True)
    row, = data["rows"]
    assert row["location"] == "2 locations" and row["visualize_labels"] == ()
    assert "Checking…" in render_prepared_track_report(data)
    track._cache.clear()
    track._cache["repair_error"] = "Source changed"
    data = prepare_track_report(track, session, show_clean=False)
    html = render_prepared_track_report(data)
    assert len(data["rows"]) == 1 and html.count("report-retry:track") == 1
    assert "Unable to verify" in html and "✓ Correction applied" not in html


def test_spectrogram_hover_and_paint_share_the_visible_frequency_band(pair):
    track, _ = pair
    issue = track.detector_results["subsonic"].issues[0]
    assert issue_y_bounds(issue, 100, "spectrogram", 2, 0, 100, True) == (70, 100)
    assert issue_y_bounds(issue, 100, "spectrogram", 2, 40, 100, True) == (100, 100)
    assert issue_y_bounds(issue, 100, "waveform", 2, 0, 100, True) == (0, 50)
    assert issue_y_bounds(replace(issue, channel=1), 100, "waveform", 2, 0, 100, True) == (50, 100)
