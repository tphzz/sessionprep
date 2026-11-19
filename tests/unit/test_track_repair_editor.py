from dataclasses import replace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextDocument, QTextTable
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QHBoxLayout, QTableWidget, QTableWidgetItem, QTabWidget, QWidget

from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane
from sessionprepgui.detail.mixin import DetailMixin
from sessionprepgui.detail.file_review import FileReviewPane
from sessionprepgui.tracks.columns_mixin import TrackColumnsMixin
from sessionpreplib.detectors.subsonic import SubsonicDetector
from sessionpreplib.models import CorrectedAnalysis, DetectorResult, IssueLocation, SessionContext, Severity, TrackContext
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionpreplib.repair_verification import local_rumble
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def evidence_tables(label):
    document = QTextDocument()
    document.setHtml(label.text())
    return [
        [[table.cellAt(r, c).firstCursorPosition().block().text() for c in range(table.columns())]
         for r in range(table.rows())]
        for table in document.rootFrame().childFrames() if isinstance(table, QTextTable)
    ]


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
    pane = FileReviewPane(FileReportPane(browser))
    editor = pane.editor
    browser.repairRequested.connect(lambda det: pane.open_track(track, session, det))
    pane.resize(1010, 520)
    pane.show()
    pane.activateWindow()
    browser.present_track(track, session)
    app.processEvents()
    yield pane, editor, track, session
    pane.close()


def test_repair_link_choices_reset_and_source_evidence(editor_pane, app):
    pane, editor, track, session = editor_pane
    report = pane.report
    changes = []
    editor.choiceChanged.connect(lambda t, key, value: changes.append((t, key, value)))
    assert "Track repairs…" in report.toPlainText() and "Review repair" not in report.toPlainText()
    report.activate_report_link(QUrl("report-repair:subsonic"))
    assert not editor.isVisible()
    report.activate_report_link(QUrl("report-details:subsonic"))
    assert "Review repair" in report.toPlainText()
    report.activate_report_link(QUrl("report-repair:clipping"))
    assert not editor.isVisible()
    report.activate_report_link(QUrl("report-repair:subsonic"))
    app.processEvents()
    assert editor.isVisible()
    row = editor.rows["low_cut"]
    assert [row.choice.itemText(i) for i in range(3)] == ["Use preset (Off)", "On", "Off"]
    assert row.choice.currentData() is True
    assert editor.cutoff.value() == 14
    whole, = evidence_tables(row.evidence)
    assert whole == [["Measurement", "Original", "Corrected"],
                     ["Energy at or below 30 Hz", "−10.0 dB", "−28.9 dB"]]
    assert "−4.2 → −3.8 dBFS" in editor.peak.text()
    assert not editor.rows["ultrasonic"].isVisible()
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
    assert not editor.isVisible() and report.hasFocus()
    assert report.textCursor().charFormat().anchorHref() == "report-repair:subsonic"


@pytest.mark.parametrize("return_by", ["back_button", "alt_left"])
def test_back_restores_both_scrollers_disclosure_and_opener_after_refresh(editor_pane, app, return_by):
    pane, editor, track, session = editor_pane
    pane.resize(540, 235)
    report = pane.report
    # Put the opener below the current reading position. Restoring focus must
    # not scroll to the link if the user has since read another passage.
    for i in range(12):
        key = f"extra-{i:02}"
        track.detector_results[key] = DetectorResult(key, Severity.INFO, f"Finding number {i}", {})
    report.present_track(track, session)
    report.activate_report_link(QUrl("report-details:extra-01"))
    report.activate_report_link(QUrl("report-details:subsonic"))
    report.activate_report_link(QUrl("report-help:subsonic"))
    app.processEvents()
    help_browser = pane.findings.help_browser
    for browser in (report, help_browser):
        bar = browser.verticalScrollBar()
        assert bar.maximum() > 50
        bar.setValue(bar.maximum() // 3)
    offsets = [b.verticalScrollBar().value() for b in (report, help_browser)]
    report.activate_report_link(QUrl("report-repair:subsonic"))
    assert pane.currentWidget() is pane.scroll and not pane.findings.isVisible()
    assert editor.rows["low_cut"].toggle.isChecked()
    report.present_track(track, session)
    editor.refresh(track, session)
    app.processEvents()
    assert pane.currentWidget() is pane.scroll and editor.rows["low_cut"].toggle.isChecked()
    if return_by == "alt_left":
        QTest.keyClick(editor.back_button, Qt.Key_Left, Qt.AltModifier)
    else:
        QTest.mouseClick(editor.back_button, Qt.LeftButton)
    app.processEvents()
    assert pane.currentWidget() is pane.findings and pane.findings.active_detector_id == "subsonic"
    assert offsets == [b.verticalScrollBar().value() for b in (report, help_browser)]
    assert report._states[track.filename] == {"extra-01", "subsonic"}
    assert report.hasFocus() and report.textCursor().charFormat().anchorHref() == "report-repair:subsonic"
    QTest.keyClick(report, Qt.Key_Return)
    assert editor.isVisible()


@pytest.mark.parametrize("larger_font", [False, True])
def test_400px_editor_uses_whole_file_pane_and_detected_repairs_remain_reachable(editor_pane, app, larger_font):
    pane, editor, track, session = editor_pane
    if larger_font:
        font = QFont(editor.font())
        font.setPointSize(max(15, font.pointSize() + 2))
        editor.setFont(font)
    pane.resize(400, 475)
    report = pane.report
    report.activate_report_link(QUrl("report-help:subsonic"))
    report.activate_report_link(QUrl("report-details:subsonic"))
    report.activate_report_link(QUrl("report-repair:subsonic"))
    app.processEvents()
    assert editor.width() == pane.scroll.viewport().width()
    assert editor.width() > report.width() and pane.scroll.horizontalScrollBar().maximum() == 0
    assert editor.title.text() == track.filename
    assert {key for key, row in editor.rows.items() if row.toggle.isVisible()} == {"dc_remove", "low_cut"}
    low = editor.rows["low_cut"]
    assert low.choice.isVisible() and low.reset.geometry().right() < low.width()
    dc = editor.rows["dc_remove"]
    QTest.mouseClick(dc.toggle, Qt.LeftButton)
    assert dc.choice.isVisible() and not low.choice.isVisible()
    assert "Original" in dc.evidence.text() and "Left" in dc.evidence.text()
    assert "Right" in dc.evidence.text() and "Corrected" in dc.evidence.text()
    app.processEvents()
    assert pane.scroll.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("keep_finding", [True, False])
def test_late_refresh_after_back_keeps_a_live_keyboard_target(editor_pane, keep_finding):
    pane, editor, track, session = editor_pane
    report = pane.report
    report.activate_report_link(QUrl("report-details:subsonic"))
    report.activate_report_link(QUrl("report-repair:subsonic"))
    editor.back_button.click()
    if not keep_finding:
        track.detector_results.clear()
    report.present_track(track, session)
    cursor = report.textCursor()
    target = "report-repair:subsonic" if keep_finding else "report-repairs:track"
    assert cursor.hasSelection()
    assert cursor.selectionStart() == report._anchor_cursor(href=target).position()
    assert cursor.charFormat().anchorHref() == target
    QTest.keyClick(report, Qt.Key_Return)
    assert editor.isVisible()


def test_long_filename_keeps_full_identity_and_fits_the_compact_editor(editor_pane, app):
    pane, editor, track, session = editor_pane
    track.filename = "Lead_synth_" * 15 + "outro.wav"
    pane.report.present_track(track, session)
    pane.resize(400, 475)
    pane.report.activate_report_link(QUrl("report-repairs:track"))
    app.processEvents()
    assert pane.scroll.horizontalScrollBar().maximum() == 0
    assert editor.title.accessibleName() == editor.title.toolTip() == track.filename
    assert editor.title.text().startswith("Lead") and editor.title.text().endswith(".wav")
    assert editor.title.text() != track.filename


def test_long_render_error_keeps_recovery_within_the_compact_pane(editor_pane, app):
    pane, editor, track, session = editor_pane
    message = "Missing audio file: " + "session_take_02_" * 12 + "synth.wav"
    session.config["_prepare_errors"] = [(track.filename, message)]
    pane.resize(400, 475)
    pane.report.activate_report_link(QUrl("report-repairs:track"))
    app.processEvents()
    assert pane.scroll.horizontalScrollBar().maximum() == 0
    assert editor.error.isVisible() and editor.error.height() >= 3 * editor.error.fontMetrics().height()
    assert editor.error.toPlainText() == message and editor.error.verticalScrollBar().maximum() == 0
    QTest.mouseClick(editor.error.viewport(), Qt.LeftButton, pos=QPoint(3, 4))
    assert editor.error.hasFocus()


def test_repair_buttons_accept_clicks_on_their_padded_edges(editor_pane, app):
    pane, editor, track, session = editor_pane
    report = pane.report
    playback = []
    shortcut = QShortcut(QKeySequence(Qt.Key_Space), pane)
    shortcut.activated.connect(lambda: playback.append(True))
    for href in ("report-repairs:track", "report-repair:subsonic"):
        if href.startswith("report-repair:"):
            report.activate_report_link(QUrl("report-details:subsonic"))
        report.setTextCursor(report._anchor_cursor(href=href))
        report.ensureCursorVisible()
        app.processEvents()
        button = report._repair_buttons[href]
        edge = QPoint(2, button.height() // 2)
        assert report.viewport().childAt(button.pos() + edge) is button
        QTest.mouseClick(button, Qt.LeftButton, pos=edge)
        assert editor.isVisible()
        editor.back_button.click()
        assert report.textCursor().selectedText().replace("\xa0", " ") == button.text()
        QTest.keyClick(report, Qt.Key_Space)
        assert editor.isVisible() and not playback
        editor.back_button.click()


def test_unreadable_track_rejects_stale_repair_actions(editor_pane):
    pane, editor, track, session = editor_pane
    report = pane.report
    report.activate_report_link(QUrl("report-details:subsonic"))
    track.status = "Read error"
    report.activate_report_link(QUrl("report-repairs:track"))
    report.activate_report_link(QUrl("report-repair:subsonic"))
    assert not editor.isVisible()
    report.present_track(track, session)
    assert "Track repairs…" not in report.toPlainText()
    assert not report.repairs_button.isVisible()


@pytest.mark.parametrize("replacement", ["session", "track", "loading", "clear"])
def test_replaced_context_invalidates_return_and_ignores_old_refresh(editor_pane, replacement):
    pane, editor, track, session = editor_pane
    pane.report.activate_report_link(QUrl("report-repairs:track"))
    assert editor.isVisible()
    if replacement == "session":
        pane.report.present_track(track, SessionContext([track], {}))
    elif replacement == "track":
        other = replace(track)
        session.tracks[:] = [other]
        pane.report.present_track(other, session)
    elif replacement == "loading":
        pane.report.setHtml("<p>Loading audio…</p>")
    else:
        pane.report.clear()
    pane.back_to_findings()
    editor.refresh(track, session)
    assert pane.currentWidget() is pane.findings and editor.track is None


def test_track_with_existing_repair_keeps_entry_when_contextual_opener_disappears(editor_pane):
    pane, editor, track, session = editor_pane
    report = pane.report
    report.activate_report_link(QUrl("report-details:subsonic"))
    report.activate_report_link(QUrl("report-repair:subsonic"))
    track.detector_results.clear()
    report.present_track(track, session, show_clean=False)
    editor.back_button.click()
    assert report.textCursor().charFormat().anchorHref() == "report-repairs:track"
    QTest.keyClick(report, Qt.Key_Return)
    assert editor.isVisible() and editor.rows["dc_remove"].toggle.isChecked()
    editor.back_button.click()
    assert report.hasFocus() and "No findings to show." in report.toPlainText()


def test_settling_in_a_previously_silent_gap_is_visible_in_the_editor(editor_pane):
    pane, editor, track, session = editor_pane
    track.repair_measurements["boundary_settling"] = [
        {"channel": 1, "start_sample": 48000, "end_sample": 70000, "peak_dbfs": -19.97}]
    pane.report.activate_report_link(QUrl("report-repairs:track"))
    assert editor.boundary.isVisible() and not editor.boundary_details.isVisible()
    editor.boundary.click()
    assert editor.boundary_details.isVisible()
    message = editor.boundary_details.text()
    assert "Boundary settling" in message and "Ch 2" in message
    assert "48000" in message and "70000" in message and "20.0 dBFS" in message
    track.repair_measurements.pop("boundary_settling")
    editor.refresh(track, session)
    assert not editor.boundary.isVisible() and not editor.boundary_details.isVisible()


def test_peak_layout_releases_moved_native_items_on_reflow(editor_pane, app):
    from shiboken6 import isValid

    pane, editor, track, session = editor_pane
    grid = editor.peak_layout
    label = grid.itemAtPosition(0, 0)
    pane.open_track(track, session)
    app.processEvents()
    for width, position in [(450, (1, 0)), (1010, (0, 1))]:
        old_item = grid.itemAt(grid.indexOf(editor.peak))
        pane.resize(width, 520)
        app.processEvents()
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
    owner._file_review = pane
    owner._file_report = pane.report
    owner._current_track = track
    owner._session = session
    owner._detail_tabs = QTabWidget()
    owner._detail_tabs.addTab(QWidget(), "Summary")
    owner._detail_tabs.addTab(pane, "File")
    owner._detail_tabs.setCurrentIndex(1)
    owner._track_table = QTableWidget(1, 8)
    owner._track_table.setItem(0, 0, QTableWidgetItem(track.filename))
    owner._create_processing_button(0, track)
    pane.processingFocusRequested.connect(owner._focus_processing_menu)
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
    QTest.mouseClick(editor.back_button, Qt.LeftButton)
    app.processEvents()
    assert not editor.isVisible() and button.hasFocus()
    pane.setParent(None)
    window.close()


def test_local_only_suggestion_can_be_reviewed_without_changing_the_detector(editor_pane, app):
    pane, editor, track, session = editor_pane
    detector = session.detectors[0]
    raw = DetectorResult("subsonic", Severity.CLEAN, "whole-track ratio below threshold",
                         {"subsonic_ratio_db": -27.8, "subsonic_warn": False})
    track.detector_results["subsonic"] = raw
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": replace(
        raw, data={"subsonic_ratio_db": -30.1, "subsonic_warn": False})})
    local = {"band_dbfs": -44.95, "ratio_db": -11.78, "regions": [[0, 44099, 1]]}
    track.repair_measurements["low_band"] = track.repair_measurements["original"]["low_band"] = local
    track.repair_measurements["corrected"]["low_band"] = {"band_dbfs": -50.32, "ratio_db": -17, "regions": []}
    pane.report.present_track(track, session, show_clean=False)
    assert "local energy" in pane.report.toPlainText() and "Review repair" not in pane.report.toPlainText()
    pane.report.activate_report_link(QUrl("report-details:subsonic-local"))
    assert "Review repair" in pane.report.toPlainText()
    pane.report.activate_report_link(QUrl("report-repair:subsonic-local"))
    assert editor.isVisible()
    whole, local = evidence_tables(editor.rows["low_cut"].evidence)
    assert whole[1] == ["Energy at or below 30 Hz", "−27.8 dB", "−30.1 dB"]
    assert local == [["Local energy ≤20 Hz", "Original", "Corrected"],
                     ["Band level", "−45.0 dBFS RMS", "−50.3 dBFS RMS"],
                     ["Regions", "1", "0"], ["Affected channels", "Right", "None"]]
    pane.report.present_track(track, session, show_clean=True)
    assert "Clean" in pane.report.toPlainText() and "✓ Correction applied" in pane.report.toPlainText()
    assert track.detector_results["subsonic"] is raw and raw.severity is Severity.CLEAN
    detector.configure({"subsonic_report_as": "skip"})
    pane.report.present_track(track, session, show_clean=False)
    assert "Review repair" not in pane.report.toPlainText()


def test_whole_track_verdict_and_remaining_local_regions_have_separate_source_columns(editor_pane, app):
    pane, editor, track, session = editor_pane
    original = track.detector_results["subsonic"]
    original.data.update(subsonic_ratio_db=-19.1, per_channel={0: {"warn": True}, 1: {"warn": True}})
    original.issues = [IssueLocation(i * 10, i * 10 + 9, i % 2, Severity.INFO, "subsonic", "Low energy")
                       for i in range(10)]
    clean = replace(original, severity=Severity.CLEAN, issues=[], data={"subsonic_ratio_db": -23.1})
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": clean})
    track.repair_measurements["original"]["low_band"] = {
        "regions": [[20*i, 20*i + 9, i % 2] for i in range(12)], "band_dbfs": -25.4,
    }
    track.repair_measurements["corrected"]["low_band"] = {
        "regions": [[30, 39, 1], [60, 69, 1]], "band_dbfs": -30.1,
    }
    pane.open_track(track, session, "subsonic")
    app.processEvents()
    row = editor.rows["low_cut"]
    whole, local = evidence_tables(row.evidence)
    assert whole == [["Measurement", "Original", "Corrected"],
                     ["Energy at or below 30 Hz", "−19.1 dB", "−23.1 dB"],
                     ["Affected channels", "Left and right", "None"]]
    assert local[1:] == [["Band level", "−25.4 dBFS RMS", "−30.1 dBFS RMS"],
                         ["Regions", "12", "2"], ["Affected channels", "Left and right", "Right"]]
    assert row.verdict.text() == "✓ Correction applied"
    for widget in (row.choice, row.toggle):
        center = widget.mapTo(row, widget.rect().center())
        assert abs(center.y() - row.verdict.geometry().center().y()) <= 2
    assert row.toggle.geometry().right() < row.verdict.x()
    assert row.verdict.geometry().right() < row.choice.mapTo(row, QPoint()).x()
    assert row.evidence.mapTo(row, QPoint()).x() == row.description.indent()
    assert row.evidence.width() >= 600

    pane.resize(400, 520)
    app.processEvents()
    compact, local_compact = evidence_tables(row.evidence)
    assert "Original" in compact[1][0] and "−19.1 dB" in compact[1][0]
    assert "Corrected" in compact[1][1] and "−23.1 dB" in compact[1][1]
    assert local_compact == local
    assert pane.scroll.horizontalScrollBar().maximum() == 0
    assert editor.cutoff.isVisible() and row.choice.isVisible()

    del track.repair_measurements["corrected"]["low_band"]
    editor.refresh(track, session)
    _, unknown = evidence_tables(row.evidence)
    assert [line[2] for line in unknown[1:]] == ["—", "—", "—"]
    track._cache["repair_verification_pending"] = True
    editor.refresh(track, session)
    assert row.verdict.text() == "Checking…"
    assert "−23.1 dB" not in row.evidence.text()


@pytest.mark.parametrize("region", [[0, 80, 2], [0, 80, None], [0, 80, -1], [0, 80], [90, 80, 1]])
def test_unusable_saved_local_regions_do_not_claim_a_count_or_a_channel(editor_pane, region):
    pane, editor, track, session = editor_pane
    measured = {"band_dbfs": -25.4, "regions": [[0, 80, 1]]}
    track.repair_measurements["original"]["low_band"] = measured
    track.repair_measurements["corrected"]["low_band"] = {"band_dbfs": -30.1, "regions": [region]}
    track.corrected_analysis = CorrectedAnalysis("verified", {})
    pane.open_track(track, session, "subsonic")
    _, local = evidence_tables(editor.rows["low_cut"].evidence)
    assert local[1:] == [["Band level", "−25.4 dBFS RMS", "−30.1 dBFS RMS"],
                         ["Regions", "1", "—"], ["Affected channels", "Right", "—"]]
    assert local_rumble(track, session.config, corrected=True) is None


def test_legacy_low_cut_input_can_identify_original_locations_only_when_dc_was_bypassed(editor_pane):
    pane, editor, track, session = editor_pane
    track.repair_measurements["low_band"] = {"regions": [[0, 80, 1]], "band_dbfs": -25.4}
    assert len(local_rumble(track, session.config).issues) == 1
    pane.open_track(track, session, "subsonic-local")
    _, local = evidence_tables(editor.rows["low_cut"].evidence)
    assert local[1:] == [["Band level", "−25.4 dBFS RMS", "—"], ["Regions", "1", "—"],
                         ["Affected channels", "Right", "—"]]
    track.repair_recipe = RepairRecipe(dc_remove=True, low_cut_hz=10)
    assert local_rumble(track, session.config) is None
    editor.refresh(track, session)
    assert "Local energy" not in editor.rows["low_cut"].evidence.text()


def test_high_rate_evidence_and_headroom_recovery_use_the_selected_track(editor_pane):
    pane, editor, track, session = editor_pane
    track.samplerate = 96000
    track.repair_measurements["suggested"]["ultrasonic"] = True
    track.repair_measurements["original"]["ultrasonic"] = {
        "regions": [[0, 62399, 0], [0, 62399, 1]], "band_dbfs": -33.5, "eligible": True}
    track.repair_measurements["corrected"]["ultrasonic"] = {"regions": [], "band_dbfs": -121}
    session.config["_prepare_errors"] = [(track.filename, "Output peak +0.58 dBFS exceeds PCM_16.")]
    session.config["_output_gain_adjustments"] = {track.filename: {"peak_db": 0.58, "gain_db": 0, "reduction_db": 0.6}}
    pane.open_track(track, session)
    high = editor.rows["ultrasonic"]
    QTest.mouseClick(high.toggle, Qt.LeftButton)
    assert high.choice.isVisible() and "Qualifying windows span 0.65 s" in high.description.text()
    assert "−33.5" in high.evidence.text() and "dBFS RMS" in high.evidence.text()
    requested = []
    editor.adjustGainRequested.connect(requested.append)
    editor.adjust_gain.click()
    assert requested == [track] and "Output peak +0.58" in editor.error.toPlainText()
    session.config["_output_gain_adjustments"].clear()
    editor.refresh(track, session)
    assert not editor.adjust_gain.isVisible()


def test_resolved_subsonic_link_opens_dc_and_back_returns_to_its_original_finding(editor_pane, app):
    pane, editor, track, session = editor_pane
    track.repair_recipe = RepairRecipe(dc_remove=True)
    track.repair_overrides = {"dc_remove": True}
    original = track.detector_results["subsonic"]
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": replace(
        original, severity=Severity.CLEAN, data={"subsonic_warn": False, "subsonic_ratio_db": -29.7})})
    report = pane.report
    report.present_track(track, session, show_clean=False)
    report.activate_report_link(QUrl("report-details:subsonic"))
    assert "Resolved by DC removal" in report.toPlainText()
    report.activate_report_link(QUrl("report-repair:subsonic"))
    assert editor.rows["dc_remove"].toggle.isChecked()
    assert {key for key, row in editor.rows.items() if row.isVisible()} == {"dc_remove"}
    editor.back_button.click()
    assert report.hasFocus() and report.textCursor().charFormat().anchorHref() == "report-repair:subsonic"


def test_whole_track_clear_with_local_rumble_leaves_local_filter_actionable(editor_pane):
    pane, editor, track, session = editor_pane
    orig = track.detector_results["subsonic"]
    track.corrected_analysis = CorrectedAnalysis("verified", {"subsonic": replace(
        orig, severity=Severity.CLEAN, data={"subsonic_ratio_db": -29.7, "subsonic_warn": False})})
    track.repair_measurements["original"]["low_band"] = {"regions": [[20, 40, 0], [60, 80, 1]]}
    track.repair_measurements["corrected"]["low_band"] = {"regions": [[60, 80, 1]]}
    track.repair_overrides.clear()
    track.repair_recipe = RepairRecipe(dc_remove=True)
    report = pane.report
    report.present_track(track, session, show_clean=False)
    report.activate_report_link(QUrl("report-details:subsonic-local"))
    report.activate_report_link(QUrl("report-repair:subsonic-local"))
    low = editor.rows["low_cut"]
    assert low.toggle.isChecked() and low.verdict.text() == "Still detected"
    assert "Resolved" not in low.description.text()
    editor.back_button.click()
    assert report.textCursor().charFormat().anchorHref() == "report-repair:subsonic-local"
    track.repair_recipe = RepairRecipe(low_cut_hz=10)
    pane.open_track(track, session)
    assert low.verdict.text() == "Still detected"
    pane.open_track(track, session, "subsonic")
    assert low.verdict.text() == "✓ Correction applied"
    track.corrected_analysis.results["subsonic"] = orig
    track.repair_measurements["corrected"]["low_band"]["regions"] = []
    pane.open_track(track, session)
    assert low.verdict.text() == "Still detected"


def test_untreated_track_does_not_show_a_corrected_measurement_or_peak_comparison(editor_pane):
    pane, editor, track, session = editor_pane
    track.repair_recipe = RepairRecipe()
    track.repair_overrides.clear()
    track.repair_measurements["original"]["low_band"] = {"regions": [[0, 80, 1]], "band_dbfs": -22}
    # The planner records equal buffers and scalar measurements for an identity recipe.
    track.repair_measurements["corrected"] = dict(track.repair_measurements["original"])
    pane.open_track(track, session)
    dc, = evidence_tables(editor.rows["dc_remove"].evidence)
    assert dc[1:] == [["Left", "−1.30%", "—"], ["Right", "0.10%", "—"]]
    whole, local = evidence_tables(editor.rows["low_cut"].evidence)
    assert all(row[2] == "—" for row in whole[1:] + local[1:])
    assert not editor.peak_box.isVisible()


def test_last_saved_cutoff_can_be_reset_and_back_releases_space_to_transport(editor_pane, app):
    pane, editor, track, session = editor_pane
    track.repair_recipe = RepairRecipe()
    track.detector_results.clear()
    track.repair_measurements.clear()
    track.repair_overrides = {"low_cut_hz": 18}
    report = pane.report
    report.present_track(track, session)
    report.activate_report_link(QUrl("report-repairs:track"))
    assert {k for k, row in editor.rows.items() if row.isVisible()} == {"low_cut"}
    assert editor.cutoff.isVisible() and editor.cutoff_reset.isEnabled()

    def reset(t, key, value):
        assert t is track and value is None
        del t.repair_overrides[key]
        report.present_track(track, session)
        editor.refresh(track, session)

    editor.choiceChanged.connect(reset)
    editor.cutoff_reset.setFocus()
    QTest.mouseClick(editor.cutoff_reset, Qt.LeftButton)
    assert editor.empty.isVisible() and editor.back_button.hasFocus()
    QTest.keyClick(editor.back_button, Qt.Key_Left, Qt.AltModifier)
    app.processEvents()
    assert report.hasFocus() and not report.repairs_button.isVisible()
    assert not report._focused_repair and not report.textCursor().charFormat().anchorHref()
    report.activate_report_link(QUrl("report-repairs:track"))
    assert not editor.isVisible()
    played = []
    transport = QShortcut(QKeySequence(Qt.Key_Space), pane)
    transport.activated.connect(lambda: played.append(True))
    QTest.keyClick(report, Qt.Key_Space)
    assert played == [True] and not editor.isVisible()


def test_processing_menu_tracks_findings_without_hiding_normalization(editor_pane):
    _, _, track, session = editor_pane
    session.processors = [BimodalNormalizeProcessor()]
    session.processors[0].configure({})
    track.repair_recipe = RepairRecipe()
    track.repair_measurements.clear()
    track.repair_overrides.clear()
    owner = TrackColumnsMixin()
    owner._session = session
    owner._track_table = QTableWidget(1, 8)
    owner._create_processing_button(0, track)
    button = owner._track_table.cellWidget(0, 7)
    normalization, separator, repairs = button.menu().actions()
    assert repairs.isVisible() and separator.isVisible()
    session.config["subsonic_enabled"] = False
    button.menu().aboutToShow.emit()
    assert button.isEnabled() and normalization.isVisible() and normalization.isChecked()
    assert not repairs.isVisible() and not separator.isVisible()
    track.repair_overrides["low_cut"] = False
    button.menu().aboutToShow.emit()
    assert repairs.isVisible() and separator.isVisible()
