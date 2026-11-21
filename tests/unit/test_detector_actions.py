"""Direct detector visualization from cached findings, with context safety."""
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sessionpreplib.models import (
    DetectorResult, IssueLocation, SessionContext, Severity, TrackContext,
)
from sessionprepgui.detail import report
from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane


class Detector:
    def __init__(self, key):
        self.id = key
        self.name = key.title()
        self.render_calls = 0
        self.help_calls = 0
        self.relevant = True
        self.suppressed = False

    def effective_severity(self, result):
        return None if self.suppressed else result.severity

    def is_relevant(self, result, current):
        return self.relevant

    def render_html(self, result, current):
        self.render_calls += 1
        return f"<tr><td>Full {self.id} details</td></tr>" if self.relevant else ""

    def html_help(self):
        self.help_calls += 1
        return f"<p>Explanation for {self.id}</p>"


def session_with_issues():
    current = TrackContext("Lead.wav", "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    detectors = [Detector("tail"), Detector("subsonic"), Detector("format")]
    for detector in detectors:
        issues = [] if detector.id == "format" else [
            IssueLocation(1000, 2000, None, Severity.ATTENTION,
                          detector.id + "-region", "Reported region")]
        current.detector_results[detector.id] = DetectorResult(
            detector.id, Severity.ATTENTION, "Review this finding", {},
            hint="Keep this next-step guidance.", issues=issues)
    return SessionContext([current], {}, detectors=detectors), current


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pane(qapp, monkeypatch):
    monkeypatch.setattr(report, "chunk_sizes", lambda _: [])
    view = FileReportPane(ReportBrowser({}))
    view.resize(1069, 620)
    view.show()
    view.activateWindow()
    qapp.processEvents()
    yield view
    view.close()


@pytest.mark.parametrize("compact", [False, True])
def test_actions_offer_visualize_only_for_waveform_locations(pane, compact):
    session, current = session_with_issues()
    prepared = report.prepare_track_report(current, session)
    rows = {row["id"]: row for row in prepared["rows"]}
    assert rows["tail"]["visualize_labels"] == ("tail-region",)
    assert rows["format"]["visualize_labels"] == ()
    html = report.render_prepared_track_report(prepared, compact=compact)
    assert "Actions" in html
    assert "report-visualize:tail" in html and "report-help:tail" in html
    assert "report-visualize:format" not in html
    assert "report-help:format" in html
    assert "Keep this next-step guidance." in html


@pytest.mark.parametrize("start,end,channel", [
    (-1, 100, None), (96000, None, None), (10, 9, None),
    (10, 96000, None), ("10", 20, None), (10, 20, -1), (10, 20, 1),
])
def test_invalid_locations_do_not_offer_visualize(pane, start, end, channel):
    session, current = session_with_issues()
    current.detector_results["tail"].issues = [
        IssueLocation(start, end, channel, Severity.ATTENTION, "tail-region", "Invalid")]
    row = next(row for row in report.prepare_track_report(current, session)["rows"]
               if row["id"] == "tail")
    assert row["visualize_labels"] == ()


def test_visualize_keyboard_dispatch_preserves_help_details_and_cached_content(qapp, pane, monkeypatch):
    reads = []
    monkeypatch.setattr(report, "chunk_sizes", lambda path: (reads.append(path) or []))
    session, current = session_with_issues()
    pane.report.present_track(current, session)
    pane.report.activate_report_link(QUrl("report-help:tail"))
    requests = []
    pane.report.visualizeRequested.connect(requests.append)
    pane.report.focus_report_link("report-visualize", "tail")
    QTest.keyClick(pane.report, Qt.Key_Return)
    QTest.keyClick(pane.report, Qt.Key_Return)
    qapp.processEvents()
    assert requests == ["tail", "tail"]
    assert pane.active_detector_id == "tail"
    assert "Full tail details" not in pane.report.toPlainText()
    assert reads == ["missing.wav"]
    assert all(detector.render_calls == 1 and detector.help_calls == 1
               for detector in session.detectors)
    for href in ("report-visualize:format", "report-visualize:missing"):
        pane.report.activate_report_link(QUrl(href))
    assert requests == ["tail", "tail"]
    pane.report.clear()
    pane.report.activate_report_link(QUrl("report-visualize:tail"))
    assert requests == ["tail", "tail"]

@pytest.fixture
def host(pane, monkeypatch):
    import numpy as np
    from sessionprepgui.detail.controller import TrackDetailController
    from sessionprepgui.state import WorkspaceState
    from sessionprepgui.waveform.panel import WaveformPanel

    session, track = session_with_issues()
    state = WorkspaceState(config={})
    state.session = session
    state.current_track = track
    state.selected_detector_overlays = {"tail-region", "unavailable"}
    wf_panel = WaveformPanel()
    status_bar = SimpleNamespace(showMessage=lambda message: None)
    shell = SimpleNamespace(show_status=lambda message: status_bar.showMessage(message))
    detail_tabs = SimpleNamespace(currentIndex=lambda: 1)

    ctrl = TrackDetailController(
        state,
        shell,
        summary_view=None,
        file_report=pane.report,
        file_review=None,
        repair_editor=None,
        detail_tabs=detail_tabs,
        track_table=None,
        wf_panel=wf_panel,
        playback=None,
    )
    ctrl.update_time_label = lambda value=0: None

    class Host:
        def __init__(self):
            self._state = state
            self._ctrl = ctrl
            self._file_report = pane.report
            self._wf_panel = wf_panel
            self._wf_container = wf_panel
            self._waveform = wf_panel.waveform
            self._overlay_menu = wf_panel.overlay_menu
            self._overlay_btn = wf_panel.overlay_btn
            self._cmap_group = wf_panel.cmap_group
            self._play_btn = wf_panel.play_btn
            self._status_bar = status_bar
            self._detail_tabs = detail_tabs

        @property
        def _session(self): return self._state.session
        @_session.setter
        def _session(self, v): self._state.session = v
        @property
        def _current_track(self): return self._state.current_track
        @_current_track.setter
        def _current_track(self, v): self._state.current_track = v
        @property
        def _config(self): return self._state.config
        @property
        def _selected_detector_overlays(self): return self._state.selected_detector_overlays
        @property
        def _pending_detector_visualization(self): return self._ctrl.pending_detector_visualization
        @_pending_detector_visualization.setter
        def _pending_detector_visualization(self, v): self._ctrl.pending_detector_visualization = v
        @property
        def _waveform_context_generation(self): return self._ctrl.waveform_context_generation
        @_waveform_context_generation.setter
        def _waveform_context_generation(self, v): self._ctrl.waveform_context_generation = v
        @property
        def _wf_worker(self): return self._ctrl.wf_worker
        @_wf_worker.setter
        def _wf_worker(self, v): self._ctrl.wf_worker = v
        @property
        def _audio_load_worker(self): return self._ctrl.audio_load_worker
        @_audio_load_worker.setter
        def _audio_load_worker(self, v): self._ctrl.audio_load_worker = v

        def _visualize_detector(self, key): self._ctrl.visualize_detector(key)
        def _on_file_report_context_changed(self): self._ctrl.on_file_report_context_changed()
        def _on_waveform_loaded(self, *a, **kw): self._ctrl.on_waveform_loaded(*a, **kw)
        def _on_audio_loaded(self, *a, **kw): self._ctrl.on_audio_loaded(*a, **kw)
        def _on_audio_load_error(self, *a, **kw): self._ctrl.on_audio_load_error(*a, **kw)
        def _load_waveform(self, track): self._ctrl.load_waveform(track)
        def _refresh_waveform_issues(self): self._ctrl.refresh_waveform_issues()
        def _start_wf_worker(self, track): self._ctrl.start_wf_worker(track)

    owner = Host()
    owner._file_report.visualizeRequested.connect(ctrl.visualize_detector)
    owner._file_report.reportChanged.connect(ctrl.on_file_report_context_changed)
    owner._file_report.present_track(track, session)
    ctrl.waveform_context_generation = owner._file_report.context_generation
    samples = np.zeros(96000, dtype=np.float32)
    owner._waveform.set_audio(samples, 48000)
    owner._waveform._view_start = 20000
    owner._waveform._view_end = 30000
    owner._wf_panel.rms_lr_toggle.setChecked(True)
    owner._wf_panel.rms_avg_toggle.setChecked(True)
    owner._wf_panel.markers_toggle.setChecked(True)
    owner._wf_panel.set_expanded(False)
    owner.fit_calls = 0
    fit = owner._waveform.zoom_fit

    def count_fit():
        owner.fit_calls += 1
        fit()

    monkeypatch.setattr(owner._waveform, "zoom_fit", count_fit)
    owner.waveform_result = {
        "channels": [samples], "total_samples": 96000, "samplerate": 48000,
        "rms_window_samples": 19200, "peak_sample": 1000, "peak_channel": 0,
        "peak_db": -6.0, "peak_amplitude": 0.5, "rms_max_sample": 20000,
        "rms_max_db": -18.0, "rms_max_amplitude": 0.125,
    }
    yield owner
    owner._wf_panel.close()


def activate(host, key):
    host._file_report.activate_report_link(QUrl("report-visualize:" + quote(key, safe="")))


def checked_labels(host):
    return {action.data() for action in host._overlay_menu.actions() if action.isChecked()}


def test_visualize_isolates_detectors_opens_preview_and_retains_other_overlays(host, pane):
    host._file_report.activate_report_link(QUrl("report-help:tail"))
    activate(host, "subsonic")
    assert host._selected_detector_overlays == {"subsonic-region"}
    assert host._waveform._enabled_overlays == {"subsonic-region"}
    assert checked_labels(host) == {"subsonic-region"}
    assert host._overlay_btn.text() == "Detector Overlays (1)"
    assert host._wf_panel.is_expanded()
    assert (host._waveform._view_start, host._waveform._view_end) == (0, 96000)
    assert host._waveform._show_rms_lr and host._waveform._show_rms_avg
    assert host._waveform._show_markers
    assert pane.active_detector_id == "tail"
    assert host._pending_detector_visualization is None
    activate(host, "tail")
    assert checked_labels(host) == {"tail-region"}
    assert host.fit_calls == 2


def test_only_latest_visualization_waits_for_waveform_readiness(host):
    host._waveform_context_generation = None
    activate(host, "tail")
    activate(host, "subsonic")
    assert host.fit_calls == 0
    worker = object()
    host._wf_worker = worker
    host._on_waveform_loaded(host.waveform_result, host._current_track, worker)
    assert host.fit_calls == 1
    assert checked_labels(host) == {"subsonic-region"}
    assert host._waveform._enabled_overlays == {"subsonic-region"}
    assert host._pending_detector_visualization is None


def test_manual_overlay_selection_cancels_a_pending_visualize_fit(host):
    host._waveform_context_generation = None
    activate(host, "tail")
    next(action for action in host._overlay_menu.actions()
         if action.data() == "subsonic-region").setChecked(True)
    worker = object()
    host._wf_worker = worker
    host._on_waveform_loaded(host.waveform_result, host._current_track, worker)
    assert host.fit_calls == 0
    assert checked_labels(host) == {"tail-region", "subsonic-region"}


@pytest.mark.parametrize("change", ["suppressed", "irrelevant", "locations", "removed"])
def test_stale_cached_action_does_not_visualize_unavailable_current_data(host, change):
    detector = host._session.detectors[0]
    if change == "suppressed":
        detector.suppressed = True
    elif change == "irrelevant":
        detector.relevant = False
    elif change == "locations":
        host._current_track.detector_results["tail"].issues.clear()
    else:
        host._session.tracks.clear()
    activate(host, "tail")
    assert host.fit_calls == 0
    assert not host._wf_panel.is_expanded()


def test_clearing_report_cancels_pending_visualization(host):
    host._waveform_context_generation = None
    activate(host, "tail")
    host._file_report.clear()
    assert host._pending_detector_visualization is None
    assert host._waveform_context_generation is None


def test_stale_worker_for_same_track_does_not_clear_or_replace_new_worker(host):
    old, current = object(), object()
    host._wf_worker = current
    host._on_waveform_loaded({}, host._current_track, old)
    assert host._wf_worker is current
    assert host.fit_calls == 0


def test_stale_audio_completion_and_error_do_not_clear_current_load(host, monkeypatch):
    old, current = object(), object()
    host._audio_load_worker = current
    calls = []
    monkeypatch.setattr(host._ctrl, "load_waveform", calls.append)
    host._on_audio_loaded(host._current_track, host._current_track, old)
    host._on_audio_load_error("old failure", host._current_track, old)
    assert host._audio_load_worker is current
    assert calls == []

@pytest.mark.parametrize("detector_type", ["format", "length"])
@pytest.mark.parametrize("severity", [Severity.CLEAN, Severity.PROBLEM])
def test_real_session_detector_results_without_locations_do_not_break_waveform(
        host, detector_type, severity):
    from sessionpreplib.detectors.format_consistency import FormatConsistencyDetector
    from sessionpreplib.detectors.length_consistency import LengthConsistencyDetector

    detector = FormatConsistencyDetector() if detector_type == "format" else LengthConsistencyDetector()
    detector.configure({})
    host._session.detectors.append(detector)
    host._current_track.detector_results[detector.id] = DetectorResult(
        detector.id, severity, "Session result", {"filename": host._current_track.filename})
    host._refresh_waveform_issues()
    activate(host, "subsonic")
    assert host._waveform._enabled_overlays == {"subsonic-region"}
    assert checked_labels(host) == {"subsonic-region"}


@pytest.mark.parametrize("refresh", ["file", "presentation"])
def test_real_refresh_callers_keep_suppressed_custom_labels_out(host, monkeypatch, refresh):
    from sessionprepgui.analysis.config_controller import SessionConfigController
    from sessionprepgui.tracks.table_controller import TrackTableController
    import sessionprepgui.analysis.config_controller as config_ctrl_mod

    host._session.detectors[0].suppressed = True
    host._refresh_waveform_issues()
    host._refresh_file_report = lambda track: host._file_report.present_track(track, host._session)
    if refresh == "file":
        table_ctrl = TrackTableController(
            host._state,
            SimpleNamespace(),
            track_table=None,
        )
        table_ctrl.file_report_refresh_requested.connect(host._refresh_file_report)
        table_ctrl.waveform_issues_refresh_requested.connect(host._refresh_waveform_issues)
        table_ctrl.refresh_file_tab(host._current_track)
    else:
        for detector in host._session.detectors:
            detector.configure = lambda config: None
        host._state.flat_config = lambda: {}
        monkeypatch.setattr(config_ctrl_mod, "build_diagnostic_summary", lambda session, *, remaining: None)
        config_ctrl = SessionConfigController(
            host._state,
            SimpleNamespace(),
        )
        config_ctrl.file_report_refresh_requested.connect(host._refresh_file_report)
        config_ctrl.waveform_issues_refresh_requested.connect(host._refresh_waveform_issues)
        config_ctrl.refresh_presentation()
    assert [action.data() for action in host._overlay_menu.actions()] == ["subsonic-region"]
    assert {issue.label for issue in host._waveform._issues} == {"subsonic-region"}


@pytest.mark.parametrize("failure", ["error", "missing"])
def test_terminal_unavailable_preview_does_not_queue_forever_and_reload_recovers(host, monkeypatch, failure):
    from sessionprepgui.detail import controller

    messages = []
    host._status_bar.showMessage = messages.append
    if failure == "error":
        worker = object()
        host._audio_load_worker = worker
        host._on_audio_load_error("cannot read file", host._current_track, worker)
    else:
        monkeypatch.setattr(controller.os.path, "isfile", lambda path: False)
        host._load_waveform(host._current_track)
    activate(host, "tail")
    assert not host._waveform._loading
    assert host._pending_detector_visualization is None
    assert host._audio_load_worker is None and host._wf_worker is None
    assert "unavailable" in messages[-1].lower()

    # Selecting/reloading the file again clears the terminal state. No real worker starts.
    host._current_track.audio_data = host.waveform_result["channels"][0]
    worker = object()
    monkeypatch.setattr(host._ctrl, "start_wf_worker", lambda track: setattr(host._ctrl, "wf_worker", worker))
    host._load_waveform(host._current_track)
    activate(host, "subsonic")
    assert host._pending_detector_visualization is not None
    host._on_waveform_loaded(host.waveform_result, host._current_track, worker)
    assert host._pending_detector_visualization is None
    assert checked_labels(host) == {"subsonic-region"}
    assert host.fit_calls == 1
