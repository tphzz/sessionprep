"""Isolated unit tests for TrackDetailController (Phase 2 detail, waveform & playback)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QTabWidget, QTableWidget, QWidget

from sessionpreplib.models import DetectorResult, IssueLocation, SessionContext, Severity, TrackContext
from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.controller import TrackDetailController
from sessionprepgui.detail.file_review import FileReviewPane
from sessionprepgui.detail.help_panel import FileReportPane
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.waveform.panel import WaveformPanel


class _StubShell:
    def __init__(self) -> None:
        self.parent_widget = QWidget()
        self.status_messages: list[str] = []

    def show_status(self, message: str) -> None:
        self.status_messages.append(message)

    def progress_start(self, text: str, *, owner=None) -> None: pass
    def progress_message(self, text: str) -> None: pass
    def progress_value(self, current: int, total: int) -> None: pass
    def progress_finish(self, text: str) -> None: pass
    def progress_fail(self, text: str) -> None: pass
    def clear_progress(self) -> None: pass
    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None: pass
    def switch_to_phase(self, phase_index: int) -> None: pass
    def current_phase(self) -> int: return 1
    def is_phase_enabled(self, phase_index: int) -> bool: return True
    def register_phase_progress_layout(self, phase_index: int, layout) -> None: pass
    def dialog_parent(self) -> QWidget: return self.parent_widget
    def is_batch_mode(self) -> bool: return False
    def is_closing_after_daw(self) -> bool: return False
    def resume_close_after_daw(self) -> None: pass
    def set_window_title(self, title: str) -> None: pass


class _StubPlayback:
    def __init__(self) -> None:
        self.is_playing = False
        self.play_start_sample = 0

    def play(self, *args, **kwargs) -> None:
        self.is_playing = True

    def stop(self) -> None:
        self.is_playing = False


def _make_controller(state: WorkspaceState, shell: _StubShell):
    summary = ReportBrowser({})
    report = ReportBrowser({})
    pane = FileReportPane(report)
    review = FileReviewPane(pane)
    tabs = QTabWidget()
    tabs.addTab(summary, "Summary")
    tabs.addTab(review, "File")
    table = QTableWidget(0, 8)
    wf_panel = WaveformPanel(analysis_mode=True)
    ctrl = TrackDetailController(
        state,
        shell,
        summary_view=summary,
        file_report=report,
        file_review=review,
        repair_editor=review.editor,
        detail_tabs=tabs,
        track_table=table,
        wf_panel=wf_panel,
        playback=_StubPlayback(),
    )
    return ctrl, [tabs, table, wf_panel, shell.parent_widget]


def test_detail_controller_ignores_stale_worker_and_preserves_overlay_memory(_font_app):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    ctrl, widgets = _make_controller(state, shell)
    try:
        track = TrackContext("vox.wav", "missing.wav", None, 48000, 1, 96000, "24-bit", "PCM_24", 2.0)
        track.detector_results["clipping"] = DetectorResult(
            "clipping", Severity.PROBLEM, "clipped", {},
            issues=[IssueLocation(10, 20, None, Severity.PROBLEM, "clipping", "clip")],
        )
        session = SessionContext([track], {})
        state.session = session
        state.current_track = track
        ctrl._file_report.present_track(track, session)

        old_worker, current_worker = object(), object()
        ctrl.wf_worker = current_worker
        ctrl.on_waveform_loaded({}, track, old_worker)
        assert ctrl.wf_worker is current_worker

        ctrl.update_overlay_menu([SimpleNamespace(label="clipping")])
        action = ctrl._wf_panel.overlay_menu.actions()[0]
        action.setChecked(True)
        assert state.selected_detector_overlays == {"clipping"}

        ctrl.update_overlay_menu([])
        assert state.selected_detector_overlays == {"clipping"}
        assert ctrl._wf_panel.overlay_btn.text() == "Detector Overlays"
    finally:
        ctrl.dispose()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()


def test_detail_controller_audio_load_error_marks_generation_unavailable(_font_app):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    ctrl, widgets = _make_controller(state, shell)
    try:
        track = TrackContext("bass.wav", "missing.wav", None, 48000, 1, 96000, "24-bit", "PCM_24", 2.0)
        session = SessionContext([track], {})
        state.session = session
        state.current_track = track
        ctrl._file_report.present_track(track, session)

        worker = object()
        ctrl.audio_load_worker = worker
        ctrl.on_audio_load_error("read failed", track, worker)
        assert ctrl.audio_load_worker is None
        assert ctrl.waveform_unavailable_generation == ctrl._file_report.context_generation
        assert shell.status_messages == ["Could not load audio: read failed"]

        # Valid load afterwards clears unavailable generation
        samples = np.zeros(4800, dtype=np.float32)
        wf_worker = object()
        ctrl.wf_worker = wf_worker
        ctrl.on_waveform_loaded(
            {
                "channels": [samples], "total_samples": 4800, "samplerate": 48000,
                "rms_window_samples": 480, "peak_sample": 0, "peak_channel": 0,
                "peak_db": -60.0, "peak_amplitude": 0.001, "rms_max_sample": 0,
                "rms_max_db": -60.0, "rms_max_amplitude": 0.001,
            },
            track,
            wf_worker,
        )
        assert ctrl.waveform_unavailable_generation is None
        assert ctrl.waveform_context_generation == ctrl._file_report.context_generation
    finally:
        ctrl.dispose()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()


def test_toggle_play_ignores_phase3_and_allows_phase2(_font_app):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    phase = [2]  # _PHASE_SETUP (Phase 3)
    shell.current_phase = lambda: phase[0]
    ctrl, widgets = _make_controller(state, shell)
    try:
        samples = np.zeros(4800, dtype=np.float32)
        track = TrackContext("vox.wav", "vox.wav", samples, 48000, 1, 4800, "24-bit", "PCM_24", 0.1)
        state.current_track = track

        # Space on Phase 3 must NOT start Phase 2 playback
        ctrl.toggle_play()
        assert not ctrl._playback.is_playing

        # Space on Phase 2 toggles playback on and off
        phase[0] = 1  # _PHASE_ANALYSIS (Phase 2)
        ctrl.toggle_play()
        assert ctrl._playback.is_playing
        ctrl.toggle_play()
        assert not ctrl._playback.is_playing

        # Defensive stop: if playback is somehow active on Phase 3, toggle_play stops it
        ctrl.toggle_play()
        assert ctrl._playback.is_playing
        phase[0] = 2
        ctrl.toggle_play()
        assert not ctrl._playback.is_playing
    finally:
        ctrl.dispose()
        for w in widgets:
            w.deleteLater()
        _font_app.processEvents()
