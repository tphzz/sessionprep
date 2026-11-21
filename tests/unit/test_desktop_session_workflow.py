"""Drive Track Layout, File repair review, grouping and recall on generated WAVs."""
from dataclasses import replace
import json
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionpreplib.detectors.dc_offset import DCOffsetDetector
from sessionpreplib.repairs import render_corrected
from sessionpreplib.reports import save_json


def wait_until(predicate):
    deadline = monotonic() + 30
    while not predicate():
        assert monotonic() < deadline, "The desktop did not finish the operation"
        QApplication.processEvents()
        # Give Python analysis workers time to run between GUI events.
        sleep(0.005)


@pytest.fixture
def analyzed_window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "prefs"))
    folder = tmp_path / "audio"
    folder.mkdir()
    t = np.arange(24000) / 8000
    for name, bias in [("Kit Kick", .09), ("Kit Snare", .02), ("Room", 0)]:
        sample = .13 * np.sin(2 * np.pi * 300 * t)
        sf.write(folder / f"{name}.wav", np.column_stack((bias + sample, .5 * sample)), 8000, subtype="PCM_24")
    path = str(tmp_path / "review.spsession")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (path, ""))
    window = SessionPrepWindow()
    window.resize(1300, 650)
    window.show()
    try:
        window.lifecycle_controller.load_directory(str(folder))
        wait_until(window.topology_controller.page.apply_action.isEnabled)
        window.topology_controller.page.apply_action.trigger()
        wait_until(lambda: window.lifecycle_controller.analyze_worker is None and window.state.topology_dir
                   and window.state.session is not None and window.state.session.tracks
                   and "dc_offset" in window.state.session.tracks[0].detector_results)
        yield window
    finally:
        dispose_window(window)
        app.processEvents()


def test_file_repair_audition_and_save_reopen_preserve_evidence(analyzed_window, tmp_path, monkeypatch):
    window = analyzed_window
    device_streams = []

    class Sink:
        def __init__(self, *, callback, channels, samplerate, **kwargs):
            assert (channels, samplerate) == (2, 8000)
            self.callback = callback
            self.closed = 0
            device_streams.append(self)

        def start(self):
            self.active = True

        def close(self):
            self.closed += 1
            self.active = False

    monkeypatch.setattr("sessionprepgui.detail.playback.sd.OutputStream", Sink)
    name = "Kit Kick.wav"
    track = next(t for t in window.state.session.tracks if t.filename == name)
    samples, rate = sf.read(tmp_path / "audio" / name)
    routed, routed_rate = sf.read(track.filepath)
    assert routed_rate == rate
    np.testing.assert_array_equal(routed, samples)
    detector = DCOffsetDetector()
    detector.configure(window.state.flat_config())
    original = detector.analyze(replace(track, audio_data=samples))
    assert original.issues and original.data["signed_dc"][0] == pytest.approx(.09, abs=1e-6)
    window._track_table.selectRow(window.track_table_controller.find_table_row(name))
    window.detail_controller.show_track_detail(track)
    report = window._file_report
    wait_until(lambda: window._repair_preview.worker is None and "DC Offset" in report.toPlainText())
    report.activate_report_link(QUrl("report-details:dc_offset"))
    report.setTextCursor(report._anchor_cursor(href="report-repair:dc_offset"))
    report.ensureCursorVisible()
    anchor = report.capture_reading_anchor()
    assert anchor is not None and anchor.detector_id
    report.activate_report_link(QUrl("report-repair:dc_offset"))
    choice = window._repair_editor.rows["dc_remove"].choice
    choice.setCurrentIndex(choice.findData(True))
    wait_until(lambda: track.corrected_analysis and window._repair_preview.worker is None
               and window.lifecycle_controller.repair_plan_worker is None and window._repair_preview.audition)
    corrected = render_corrected(samples, rate, track.repair_recipe)
    expected = detector.analyze(replace(track, audio_data=corrected))
    assert not expected.issues
    panel = window._wf_panel
    panel.set_expanded(True)
    wave = panel.waveform
    wave.set_cursor(10000)
    wave.zoom_in()
    view = (wave._view_start, wave._view_end)
    panel.source_selector.buttons[1].click()
    wait_until(lambda: wave._corrected and window._repair_preview.worker is None)
    assert (wave._cursor_sample, wave._view_start, wave._view_end) == (10000, *view)
    panel.play_btn.click()
    wait_until(lambda: device_streams)
    stream, = device_streams
    out = np.empty((512, 2), dtype="float32")
    stream.callback(out, len(out), None, None)
    np.testing.assert_allclose(out[256:], corrected[10256:10512], atol=2e-7)
    panel.source_selector.buttons[0].click()
    wait_until(lambda: not wave._corrected and window._repair_preview.worker is None)
    stream.callback(out, len(out), None, None)
    np.testing.assert_allclose(out[256:], samples[10768:11024], atol=2e-7)
    # Simulate an idle device after its callback block.
    stream.active = False
    panel.stop_btn.click()
    assert stream.closed == 1 and len(device_streams) == 1
    window._repair_editor.back_button.click()
    assert "Correction applied" in report.toPlainText()
    assert report.cursorRect(report._reading_cursor(anchor.detector_id)).top() == anchor.top
    assert report.textCursor().charFormat().anchorHref() == "report-repair:dc_offset"
    window.lifecycle_controller.save_session_dialog()
    window.lifecycle_controller.load_session_dialog()
    restored = next(t for t in window.state.session.tracks if t.filename == name)
    assert restored is not track and restored.repair_overrides["dc_remove"]
    assert restored.detector_results["dc_offset"].issues == original.issues
    verified = restored.corrected_analysis.results["dc_offset"]
    assert verified.issues == expected.issues
    assert verified.data["dc_db"] == pytest.approx(expected.data["dc_db"])
    np.testing.assert_array_equal(sf.read(restored.filepath)[0], samples)


def test_sorted_auto_group_cancel_reassign_rename_and_export(analyzed_window, tmp_path, monkeypatch):
    window = analyzed_window
    window.state.session_groups = [
        {"name": "Kit", "match_pattern": "kit"},
        {"name": "Kick", "match_pattern": "kit kick"},
    ]
    window.groups_controller.populate_groups_tab()
    window.groups_controller.refresh_group_combos()
    table = window._track_table
    table.sortItems(0, Qt.DescendingOrder)

    def groups():
        return {t.filename: t.group for t in window.state.session.tracks}

    before = groups()
    answers = iter([QMessageBox.No, QMessageBox.Yes])
    monkeypatch.setattr(QMessageBox, "question", lambda *args: next(answers))
    window._auto_group_action.trigger()
    assert groups() == before
    window._auto_group_action.trigger()
    assert groups() == {"Kit Kick.wav": "Kick", "Kit Snare.wav": "Kit", "Room.wav": None}
    combo = table.cellWidget(window.track_table_controller.find_table_row("Room.wav"), 6)
    combo.setCurrentIndex(combo.findData("Kit"))
    combo.textActivated.emit(combo.currentText())
    window._prepare_action.trigger()
    wait_until(lambda: window.lifecycle_controller.prepare_worker is None and window.state.session.prepare_state == "ready")
    window.groups_controller.page.table.item(0, 0).setText("Drums")
    expected = {"Kit Kick.wav": "Kick", "Kit Snare.wav": "Drums", "Room.wav": "Drums"}
    assert groups() == expected
    for name, group in expected.items():
        assert table.cellWidget(window.track_table_controller.find_table_row(name), 6).currentData() == group
    window.lifecycle_controller.save_session_dialog()
    window.lifecycle_controller.load_session_dialog()
    assert groups() == expected
    for manifest in (window.state.session.transfer_manifest, window.state.session.base_transfer_manifest):
        assert {e.output_filename: e.group for e in manifest} == expected
    path = tmp_path / "report.json"
    save_json(window.state.session, window.state.flat_config(), str(path), [])
    exported = json.loads(path.read_text(encoding="utf8"))["tracks"]
    assert {row["filename"]: row["processing"]["group"] for row in exported} == expected
