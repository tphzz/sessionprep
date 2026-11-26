"""Phase 2 track detail, waveform, overlay, and playback controller."""

from __future__ import annotations

from collections.abc import Callable
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QSignalBlocker, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction
from shiboken6 import isValid

from sessionpreplib.audio import get_window_samples
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.repair_verification import (
    audition_results as _core_audition_results,
    original_report_track,
    separate_rumble_finding,
)

from ..analysis.worker import AudioLoadWorker
from ..helpers import fmt_time
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..tracks.table_widgets import TrackCol, _PHASE_ANALYSIS, _PHASE_TOPOLOGY, _TAB_FILE
from ..waveform.compute import WaveformLoadWorker
from ..waveform.peakcache import peak_source_key
from .report import visualizable_issues

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTabWidget
    from ..topology.controller import TopologyController
    from ..tracks.table_widgets import _DraggableTrackTable
    from ..waveform.panel import WaveformPanel
    from .browser import ReportBrowser
    from .file_review import FileReviewPane
    from .playback import PlaybackController
    from .repair_editor import TrackRepairEditor
    from .repair_preview import RepairPreview

log = logging.getLogger(__name__)


def resolve_audition_results(track: Any, session: Any, *, corrected: bool) -> dict[str, Any]:
    """Return the detector results dict for *track* in Original or Corrected audition mode."""
    pending = bool(track._cache.get("repair_verification_pending"))
    selected = _core_audition_results(track, corrected=corrected, pending=pending)
    if corrected and (pending or track._cache.get("repair_error")):
        return selected
    local = separate_rumble_finding(track, session.config)
    if local:
        evidence = local.corrected if corrected and track.repair_recipe.active else local.original
        if evidence and evidence.issues:
            return {**selected, "subsonic-local": evidence}
    return selected


class TrackDetailController(QObject):
    """Owns Phase 2 File detail view, summary rendering, waveform loading, overlays, and playback."""

    repair_plan_requested = Signal(object, bool)  # (tracks, revalidate_source)
    peak_prioritize_requested = Signal(str)       # source_key

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        summary_view: ReportBrowser,
        file_report: ReportBrowser,
        file_review: FileReviewPane,
        repair_editor: TrackRepairEditor,
        detail_tabs: QTabWidget,
        track_table: _DraggableTrackTable,
        wf_panel: WaveformPanel,
        playback: PlaybackController,
        repair_preview: RepairPreview | None = None,
        topology_ctrl: TopologyController | None = None,
        on_find_table_row: Callable[[str], int] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._summary_view = summary_view
        self._file_report = file_report
        self._file_review = file_review
        self._repair_editor = repair_editor
        self._detail_tabs = detail_tabs
        self._track_table = track_table
        self._wf_panel = wf_panel
        self._playback = playback
        self._repair_preview = repair_preview
        self._topology_ctrl = topology_ctrl
        self._on_find_table_row = on_find_table_row

        # Owned workers & generation state
        self.wf_worker: Any = None
        self.audio_load_worker: Any = None
        self.pending_detector_visualization: tuple[int, str] | None = None
        self.waveform_context_generation: int | None = None
        self.waveform_unavailable_generation: int | None = None
        self._corrected_overlay_track: Any = None
        self._corrected_overlay_labels: set[str] = set()

    def dispose(self) -> None:
        """Stop all detail waveform/audio load threads cleanly."""
        for attr in ("wf_worker", "audio_load_worker"):
            worker = getattr(self, attr, None)
            if isinstance(worker, QThread) and isValid(worker) and worker.isRunning():
                cancel_fn = getattr(worker, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    worker.requestInterruption()
                worker.wait()
            setattr(self, attr, None)
        for worker in self.findChildren(QThread):
            if isValid(worker) and worker.isRunning():
                cancel_fn = getattr(worker, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    worker.requestInterruption()
                worker.wait()

    def clear_workspace(self) -> None:
        self.stop()
        self._cancel_load_workers()
        self.pending_detector_visualization = None
        self.waveform_context_generation = None
        self.waveform_unavailable_generation = None
        self._wf_panel.waveform.set_audio(None, 44100)
        self._summary_view.clear()
        self._file_report.clear()
        self._wf_panel.set_content_available(False)
        self._wf_panel.play_btn.setEnabled(False)
        self._wf_panel.stop_btn.setEnabled(False)

    # ── Display settings ─────────────────────────────────────────────────

    @property
    def show_clean(self) -> bool:
        config = self._state.effective_session_config()
        return bool(config.get("presentation", {}).get("show_clean_detectors", False))

    @property
    def verbose(self) -> bool:
        return self._state.config.get("app", {}).get("report_verbosity", "normal") == "verbose"

    # ── Report & summary rendering ───────────────────────────────────────

    def render_summary(self) -> None:
        if not self._state.summary or not self._state.session:
            return
        self._summary_view.present_summary(
            self._state.summary, self._state.session, show_clean=self.show_clean
        )

    def refresh_verified_summary(self) -> None:
        if self._state.session is not None:
            self._state.summary = build_diagnostic_summary(self._state.session, remaining=True)
            self.render_summary()

    def refresh_file_report(self, track: Any = None) -> None:
        target = track if track is not None else self._state.current_track
        if target is not None and self._state.session is not None:
            self._file_report.present_track(
                target,
                self._state.session,
                show_clean=self.show_clean,
                verbose=self.verbose,
                corrected=self.audition_is_corrected(),
            )

    def open_track_repairs(self, track: Any, detector_id: str | None = None) -> None:
        session = self._state.session
        if (
            session is None
            or track is None
            or track.status != "OK"
            or not any(current is track for current in session.tracks)
        ):
            return
        if (
            track is not self._state.current_track
            or self._file_report.report_track is not track
            or self._file_report.report_session is not session
        ):
            row = self._on_find_table_row(track.filename) if self._on_find_table_row else -1
            if row < 0:
                return
            with QSignalBlocker(self._track_table):
                self._track_table.setCurrentCell(row, TrackCol.PROCESSING)
            self.show_track_detail(track)
        elif self._detail_tabs.currentIndex() != _TAB_FILE:
            self._detail_tabs.setCurrentIndex(_TAB_FILE)
        self._file_review.open_track(track, session, detector_id)

    def refresh_track_repair_editor(self, track: Any = None) -> None:
        target = self._state.current_track if track is None else track
        if target is not None and self._state.session is not None:
            self._repair_editor.refresh(target, self._state.session)

    def focus_processing_menu(self, filename: str) -> None:
        row = self._on_find_table_row(filename) if self._on_find_table_row else -1
        button = self._track_table.cellWidget(row, TrackCol.PROCESSING) if row >= 0 else None
        if button is not None and button.isEnabled():
            file_item = self._track_table.item(row, TrackCol.FILE)
            if file_item is not None:
                self._track_table.scrollToItem(file_item)
            button.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self._file_report.setFocus(Qt.FocusReason.OtherFocusReason)

    def review_summary_file(self, filename: str) -> None:
        """Select the current file by identity, including after table sorting."""
        session = self._state.session
        if session is None or self._summary_view.report_session is not session:
            return
        track = next((t for t in session.tracks if t.filename == filename), None)
        if track is None:
            return
        for row in range(self._track_table.rowCount()):
            item = self._track_table.item(row, TrackCol.FILE)
            if item is not None and item.text() == filename:
                with QSignalBlocker(self._track_table):
                    self._track_table.clearSelection()
                    self._track_table.setCurrentCell(row, TrackCol.FILE)
                    self._track_table.selectRow(row)
                self._track_table.scrollToItem(item)
                self.show_track_detail(track)
                self._file_review.back_to_findings()
                self._file_report.setFocus()
                return

    def show_track_detail(self, track: Any) -> None:
        """Populate the File tab with per-track detail and defer waveform loading."""
        if (
            track is self._state.current_track
            and self._file_report.report_track is track
            and self._file_report.report_session is self._state.session
        ):
            self._detail_tabs.setCurrentIndex(_TAB_FILE)
            return
        self.stop()
        self.pending_detector_visualization = None
        self.waveform_context_generation = None
        self.waveform_unavailable_generation = None
        self._state.current_track = track

        self.refresh_file_report(track)
        self._detail_tabs.setTabEnabled(_TAB_FILE, True)
        self._detail_tabs.setCurrentIndex(_TAB_FILE)
        QTimer.singleShot(0, lambda: self.load_waveform(track))

    # ── Waveform loading ─────────────────────────────────────────────────

    def _cancel_load_workers(self) -> None:
        for attr in ("wf_worker", "audio_load_worker"):
            worker = getattr(self, attr, None)
            setattr(self, attr, None)
            if worker is None or (isinstance(worker, QObject) and not isValid(worker)):
                continue
            cancel_fn = getattr(worker, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
            for sig_name in ("completed", "error", "failed"):
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except RuntimeError:
                        pass

    def _setup_waveform_preview_state(self, track: Any) -> None:
        key = peak_source_key(getattr(track, "filepath", None))
        self.waveform_unavailable_generation = None
        self.waveform_context_generation = None
        if key in self._state.peak_cache:
            ready = self._wf_panel.waveform.set_preview_mode(
                track.channels, track.total_samples, track.samplerate, self._state.peak_cache[key]
            )
            if ready:
                self.waveform_context_generation = self._file_report.context_generation
        else:
            self._wf_panel.waveform.set_loading(True)
            if key:
                self.peak_prioritize_requested.emit(key)

        self.refresh_waveform_issues()
        self.apply_pending_detector_visualization()
        if self._detail_tabs.currentIndex() == _TAB_FILE:
            self._wf_panel.set_content_available(True)
        self._wf_panel.play_btn.setEnabled(False)
        self.update_time_label(0)

    def _set_waveform_unavailable(self) -> None:
        self.pending_detector_visualization = None
        self.waveform_context_generation = None
        self.waveform_unavailable_generation = self._file_report.context_generation
        self._wf_panel.waveform.set_loading(False)
        self._wf_panel.waveform.set_audio(None, 44100)
        self.update_overlay_menu([])
        if self._detail_tabs.currentIndex() == _TAB_FILE:
            self._wf_panel.set_content_available(False)
        self._wf_panel.play_btn.setEnabled(False)
        self.update_time_label(0)

    def load_waveform(self, track: Any) -> None:
        """Start background waveform loading for *track*."""
        t0 = time.perf_counter()
        if self._state.current_track is not track or track is None:
            return
        if self._repair_preview is not None and hasattr(track, "repair_recipe"):
            self._repair_preview.select(track)
            return

        self._cancel_load_workers()
        self._setup_waveform_preview_state(track)

        fn = getattr(track, "filename", None)
        if fn:
            log.debug("[Trace] load_waveform UI setup for '%s': %.2f ms", fn, (time.perf_counter() - t0) * 1000)

        if (
            (track.audio_data is None or track.audio_data.size == 0)
            and track.status == "OK"
            and os.path.isfile(track.filepath)
        ):
            worker = AudioLoadWorker(track, parent=self)
            self.audio_load_worker = worker
            worker.completed.connect(lambda t, orig=track, w=worker: self.on_audio_loaded(t, orig, w))
            worker.error.connect(lambda msg, orig=track, w=worker: self.on_audio_load_error(msg, orig, w))
            worker.finished.connect(worker.deleteLater)
            worker.start()
            return

        if track.audio_data is not None and track.audio_data.size > 0:
            nch = track.audio_data.shape[1] if track.audio_data.ndim > 1 else 1
            self._wf_panel.update_play_mode_channels(nch)
            self._wf_panel.play_btn.setEnabled(True)
            self.start_wf_worker(track)
        else:
            self._set_waveform_unavailable()

    def start_wf_worker(self, track: Any) -> None:
        prev = self.wf_worker
        self.wf_worker = None
        if prev is not None and (not isinstance(prev, QObject) or isValid(prev)):
            cancel_fn = getattr(prev, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
            for sig_name in ("completed", "failed"):
                sig = getattr(prev, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except RuntimeError:
                        pass
        win_ms = self._state.flat_config().get("window", 400)
        ws = get_window_samples(track, win_ms)
        wf = self._wf_panel.waveform
        worker = WaveformLoadWorker(
            track.audio_data,
            track.samplerate,
            ws,
            spec_n_fft=wf.spec_n_fft,
            spec_window=wf.spec_window,
            compute_spectrogram=(wf._display_mode == "spectrogram"),
            parent=self,
        )
        self.wf_worker = worker
        worker.completed.connect(lambda result, t=track, w=worker: self.on_waveform_loaded(result, t, w))
        worker.failed.connect(lambda msg, t=track, w=worker: self._on_wf_error(msg, t, w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_wf_error(self, message: str, track: Any, worker: Any = None) -> None:
        if worker is not None and worker is not self.wf_worker:
            return
        self.wf_worker = None
        if self._state.current_track is not track:
            return
        self._set_waveform_unavailable()
        self._shell.show_status(f"Waveform error: {message}")

    @Slot(str)
    def on_display_mode_changed(self, mode: str) -> None:
        if self._repair_preview is not None and self._repair_preview.track is not None:
            self._repair_preview.request()
            return
        track = self._state.current_track
        if mode == "spectrogram" and track is not None:
            if track.audio_data is not None and track.audio_data.size > 0:
                if (
                    getattr(self._wf_panel.waveform._spec_renderer, "_spec_data", None) is None
                    and self.wf_worker is None
                ):
                    self.start_wf_worker(track)

    @Slot(object, object)
    def on_waveform_loaded(
        self, result: dict[str, Any], track: Any, worker: Any = None, *, preserve_view: bool = False
    ) -> None:
        if worker is not None and worker is not self.wf_worker:
            return
        self.wf_worker = None
        if self._state.current_track is not track or not self.visualization_context_is_current():
            return

        wf = self._wf_panel.waveform
        wf.set_precomputed(result, preserve_view=True) if preserve_view else wf.set_precomputed(result)
        self.waveform_unavailable_generation = None
        self.waveform_context_generation = self._file_report.context_generation

        key = peak_source_key(getattr(track, "filepath", None))
        if key and not result.get("corrected"):
            if key in self._state.peak_cache:
                wf.set_peak_data(self._state.peak_cache[key])
            else:
                self.peak_prioritize_requested.emit(key)

        if not preserve_view:
            cmap = self._state.config.get("app", {}).get("spectrogram_colormap", "magma")
            wf.set_colormap(cmap)
            for act in self._wf_panel.cmap_group.actions():
                if act.data() == cmap:
                    act.setChecked(True)
                    break

        self.refresh_waveform_issues()
        self.apply_pending_detector_visualization()
        self.update_time_label(wf._cursor_sample if preserve_view else 0)

    def on_peak_file_done(self, source_key: str, peak_data: Any) -> None:
        """Push newly built peak data into the Phase 2 waveform if matching."""
        cur_track = self._state.current_track
        if not cur_track or peak_source_key(cur_track.filepath) != source_key:
            return
        preview = self._repair_preview
        if preview and preview.track is cur_track and (
            preview.panel.corrected or not preview.displayed or preview.worker is not None
        ):
            return
        wf = self._wf_panel.waveform
        wf.set_peak_data(peak_data)
        if getattr(wf, "_loading", False):
            wf.set_preview_mode(cur_track.channels, cur_track.total_samples, cur_track.samplerate, peak_data)

    def refresh_repair_preview(self, track: Any) -> None:
        if track is self._state.current_track and track is not None and self._repair_preview is not None:
            sample = self._playback.current_sample() if self._playback.is_playing else None
            self.stop()
            if sample is not None:
                self._wf_panel.waveform.set_cursor(sample)
            self._repair_preview.select(track, refresh=True)

    def on_audio_loaded(self, track: Any, orig_track: Any, worker: Any = None) -> None:
        if worker is not None and worker is not self.audio_load_worker:
            return
        self.audio_load_worker = None
        if self._state.current_track is not orig_track:
            return
        self.load_waveform(track)

    def on_audio_load_error(self, message: str, track: Any, worker: Any = None) -> None:
        if worker is not None and worker is not self.audio_load_worker:
            return
        self.audio_load_worker = None
        if self._state.current_track is not track:
            return
        self.pending_detector_visualization = None
        self.waveform_context_generation = None
        self.waveform_unavailable_generation = self._file_report.context_generation
        self._wf_panel.waveform.set_loading(False)
        self._wf_panel.waveform.set_audio(None, 44100)
        self.update_overlay_menu([])
        self._wf_panel.set_content_available(False)
        self._wf_panel.play_btn.setEnabled(False)
        self._shell.show_status(f"Could not load audio: {message}")

    # ── Visualization context & overlays ─────────────────────────────────

    def visualization_context_is_current(self) -> bool:
        track = self._state.current_track
        session = self._state.session
        return (
            session is not None
            and track is not None
            and any(t is track for t in session.tracks)
            and self._file_report.report_session is session
            and self._file_report.track_filename == track.filename
        )

    def audition_is_corrected(self) -> bool:
        preview = self._repair_preview
        return bool(preview and preview.track is self._state.current_track and preview.panel.corrected)

    def audition_results(self) -> dict[str, Any]:
        return resolve_audition_results(
            self._state.current_track, self._state.session, corrected=self.audition_is_corrected()
        )

    def detector_visualization_issues(self, key: str, results: dict[str, Any] | None = None) -> list[Any]:
        if not self.visualization_context_is_current():
            return []
        session = self._state.session
        assert session is not None
        detector_id = "subsonic" if key == "subsonic-local" else key
        detector = next((d for d in session.detectors if d.id == detector_id), None)
        result = (self.audition_results() if results is None else results).get(key)
        issues = visualizable_issues(result, self._state.current_track)
        if not issues:
            return []
        if detector is not None:
            severity = getattr(detector, "effective_severity", None)
            relevant = getattr(detector, "is_relevant", None)
            if callable(severity) and severity(result) is None:
                return []
            context = self._state.current_track
            if key == "tail_exceedance" and not self.audition_is_corrected():
                context = original_report_track(context, session)
            if callable(relevant) and not relevant(result, context):
                return []
        return list(issues)

    def refresh_waveform_issues(self) -> None:
        issues: list[Any] = []
        gov_sample: int | None = None
        gov_peak_db: float | None = None
        if self.visualization_context_is_current():
            results = self.audition_results()
            for key in results:
                issues.extend(self.detector_visualization_issues(key, results))
            track = self._state.current_track
            pr = track.processor_results.get("bimodal_normalize") if track is not None else None
            show_norm_peak = (
                track is not None
                and pr is not None
                and pr.classification not in ("Silent", "Error")
                and pr.data.get("governing_peak_sample") is not None
                and (self.audition_is_corrected() or not track.repair_recipe.active)
            )
            if show_norm_peak and pr is not None:
                gov_sample = int(pr.data["governing_peak_sample"])
                gov_peak_db = pr.data.get("detected_peak_db")
        self._wf_panel.waveform.set_issues(issues)
        self._wf_panel.waveform.set_governing_peak(gov_sample, gov_peak_db)
        self.update_overlay_menu(issues)

    @Slot(int)
    def jump_to_peak(self, sample_index: int) -> None:
        """Expand the waveform panel and move the cursor to *sample_index*."""
        if not self.visualization_context_is_current() or sample_index < 0:
            return
        self._wf_panel.set_expanded(True)
        wf = self._wf_panel.waveform
        wf.center_on_sample(sample_index)
        wf.set_cursor(sample_index)
        self.update_time_label(wf._cursor_sample)

    def on_file_report_context_changed(self) -> None:
        if not self.visualization_context_is_current():
            self.pending_detector_visualization = None
            self.waveform_context_generation = None
            self.waveform_unavailable_generation = None
            preview = self._repair_preview
            if preview and preview.track is not None:
                preview.clear_material()
                preview.track = None
                preview.displayed = False
                preview.panel.set_corrected(False)
                preview.panel.set_audition_available(False, show=False)
                preview.panel.set_preview_notice()
            return
        pending = self.pending_detector_visualization
        if pending is not None and (
            pending[0] != self._file_report.context_generation
            or not self.detector_visualization_issues(pending[1])
        ):
            self.pending_detector_visualization = None
        if self.waveform_context_generation == self._file_report.context_generation:
            self.refresh_waveform_issues()

    @Slot(str)
    def visualize_detector(self, key: str) -> None:
        if not self.visualization_context_is_current():
            return
        if not self._file_report.visualization_labels(key):
            return
        issues = self.detector_visualization_issues(key)
        if not issues:
            return
        generation = self._file_report.context_generation
        if self.waveform_unavailable_generation == generation:
            self._shell.show_status(
                "Waveform unavailable for this file. Select the file again to retry loading."
            )
            return
        anchor = self._file_report.capture_reading_anchor()
        self.pending_detector_visualization = (generation, key)
        selected = self._remembered_detector_overlays()
        selected.clear()
        selected.update(issue.label for issue in issues)
        if self.waveform_context_generation != generation:
            self._wf_panel.waveform.set_loading(True)
        self.refresh_waveform_issues()
        self._wf_panel.set_expanded(True)
        self.apply_pending_detector_visualization()
        self._file_report.restore_reading_anchor(anchor)
        self._file_report.focus_report_link("report-visualize", key)

    def apply_pending_detector_visualization(self) -> None:
        pending = self.pending_detector_visualization
        if pending is None:
            return
        generation, key = pending
        if (
            not self.visualization_context_is_current()
            or generation != self._file_report.context_generation
            or not self._file_report.visualization_labels(key)
            or not self.detector_visualization_issues(key)
        ):
            self.pending_detector_visualization = None
            return
        if self.waveform_context_generation == generation:
            self._wf_panel.waveform.zoom_fit()
            self.pending_detector_visualization = None

    @Slot()
    def retry_detector_verification(self) -> None:
        if self.visualization_context_is_current():
            track = self._state.current_track
            assert track is not None
            self.repair_plan_requested.emit(
                [track], bool(track._cache.get("repair_error"))
            )

    @Slot(str, bool)
    def show_source_locations(self, key: str, corrected: bool) -> None:
        track = self._state.current_track
        if not self.visualization_context_is_current() or track is None or not track.repair_recipe.active:
            return
        self._wf_panel._choose_source(corrected)
        issues = self.detector_visualization_issues(key)
        if issues:
            selected = self._remembered_detector_overlays()
            selected.clear()
            selected.update(i.label for i in issues)
            self.refresh_waveform_issues()
            self._wf_panel.set_expanded(True)

    def _set_overlay_button_count(self, count: int) -> None:
        self._wf_panel.overlay_btn.setText(
            f"Detector Overlays ({count})" if count else "Detector Overlays"
        )

    def _remembered_detector_overlays(self) -> set[str]:
        if self.audition_is_corrected():
            if self._corrected_overlay_track is not self._state.current_track:
                self._corrected_overlay_track = self._state.current_track
                self._corrected_overlay_labels = self._state.selected_detector_overlays.copy()
            return self._corrected_overlay_labels
        return self._state.selected_detector_overlays

    def _apply_active_detector_overlays(self, available_labels: set[str]) -> None:
        active = self._remembered_detector_overlays() & available_labels
        self._wf_panel.waveform.set_enabled_overlays(active)
        self._set_overlay_button_count(len(active))

    def update_overlay_menu(self, issues: list[Any]) -> None:
        menu = self._wf_panel.overlay_menu
        menu.clear()
        self._remembered_detector_overlays()
        if not issues:
            self._apply_active_detector_overlays(set())
            return

        det_names: dict[str, str] = {"subsonic-local": "Subsonic Content · local energy"}
        if self._state.session and hasattr(self._state.session, "detectors"):
            for det in self._state.session.detectors:
                det_names[det.id] = det.name

        label_counts: dict[str, int] = {}
        for issue in issues:
            label_counts[issue.label] = label_counts.get(issue.label, 0) + 1

        selected = self._remembered_detector_overlays()
        for label in sorted(label_counts, key=lambda lb: (det_names.get(lb) or lb).lower()):
            action = menu.addAction(f"{det_names.get(label, label)} ({label_counts[label]})")
            action.setCheckable(True)
            action.setData(label)
            action.setChecked(label in selected)
            action.toggled.connect(self.on_overlay_toggled)
        self._apply_active_detector_overlays(set(label_counts))

    @Slot()
    def on_overlay_toggled(self) -> None:
        self.pending_detector_visualization = None
        available: set[str] = set()
        checked: set[str] = set()
        for action in self._wf_panel.overlay_menu.actions():
            if not action.isCheckable():
                continue
            label = action.data()
            available.add(label)
            if action.isChecked():
                checked.add(label)
        selected = self._remembered_detector_overlays()
        selected.difference_update(available)
        selected.update(checked)
        self._wf_panel.waveform.set_enabled_overlays(checked)
        self._set_overlay_button_count(len(checked))

    # ── Spectrogram action handlers ──────────────────────────────────────

    @Slot(QAction)
    def on_spec_fft_changed(self, action: QAction) -> None:
        if self._repair_preview and self._repair_preview.track:
            self._wf_panel.waveform._spec_renderer.set_n_fft(int(action.data()))
            self._repair_preview.request()
            return
        self._wf_panel.waveform.set_spec_fft(int(action.data()))

    @Slot(QAction)
    def on_spec_window_changed(self, action: QAction) -> None:
        if self._repair_preview and self._repair_preview.track:
            self._wf_panel.waveform._spec_renderer.set_window(action.data())
            self._repair_preview.request()
            return
        self._wf_panel.waveform.set_spec_window(action.data())

    @Slot(QAction)
    def on_spec_cmap_changed(self, action: QAction) -> None:
        self._wf_panel.waveform.set_colormap(action.data())

    @Slot(QAction)
    def on_spec_floor_changed(self, action: QAction) -> None:
        self._wf_panel.waveform.set_spec_db_floor(float(action.data()))

    @Slot(QAction)
    def on_spec_ceil_changed(self, action: QAction) -> None:
        self._wf_panel.waveform.set_spec_db_ceil(float(action.data()))

    # ── Playback coordination ────────────────────────────────────────────

    def is_phase_active(self) -> bool:
        """Return True when Phase 2 (Analysis & Preparation) is the active tab."""
        return self._shell.current_phase() == _PHASE_ANALYSIS

    @Slot()
    def toggle_play(self) -> None:
        preview = self._repair_preview
        if self._playback.is_playing or (preview and preview.pending_play is not None):
            if preview:
                preview.pending_play = None
            self._playback.stop()
            self._wf_panel.stop_btn.setEnabled(False)
            if self._state.current_track is not None:
                self._wf_panel.play_btn.setEnabled(True)
            if self._topology_ctrl is not None:
                self._topology_ctrl.page.wf_panel.stop_btn.setEnabled(False)
                self._topology_ctrl.page.wf_panel.play_btn.setEnabled(
                    self._topology_ctrl.cached_audio is not None
                )
            return

        if self._shell.current_phase() == _PHASE_TOPOLOGY:
            if self._topology_ctrl is not None and self._topology_ctrl.cached_audio is not None:
                self._topology_ctrl.play_preview()
        elif self._shell.current_phase() == _PHASE_ANALYSIS and self._state.current_track is not None:
            self.play()

    @Slot()
    def play(self) -> None:
        track = self._state.current_track
        if track is None:
            return
        preview = self._repair_preview
        audition = preview.audition if preview and preview.track is track else None
        if (preview and audition is None) or (audition is None and track.audio_data is None):
            return
        self.stop()
        start = self._wf_panel.waveform._cursor_sample
        mode, channel = self._wf_panel.play_mode
        if audition is None or preview is None:
            self._playback.play(track.audio_data, track.samplerate, start, mode=mode, channel=channel)
        else:
            preview.play(start, mode, channel)
        if self._playback.is_playing or (preview and preview.pending_play is not None):
            self._wf_panel.play_btn.setEnabled(False)
            self._wf_panel.stop_btn.setEnabled(True)

    @Slot()
    def stop(self) -> None:
        preview = self._repair_preview
        if preview:
            preview.pending_play = None
        was_playing = self._playback.is_playing
        start_sample = self._playback.play_start_sample
        self._playback.stop()
        self._wf_panel.stop_btn.setEnabled(False)
        if self._state.current_track is not None and (not preview or preview.audition is not None):
            self._wf_panel.play_btn.setEnabled(True)
        if was_playing:
            self._wf_panel.waveform.set_cursor(start_sample)
            self.update_time_label(start_sample)

    @Slot(int)
    def on_cursor_updated(self, sample_pos: int) -> None:
        if self._shell.current_phase() == _PHASE_TOPOLOGY and self._topology_ctrl is not None:
            self._topology_ctrl.page.wf_panel.waveform.set_cursor(sample_pos)
            self._topology_ctrl.update_time_label(sample_pos)
            return
        self._wf_panel.waveform.set_cursor(sample_pos)
        self.update_time_label(sample_pos)

    @Slot()
    def on_playback_finished(self) -> None:
        if self._shell.current_phase() == _PHASE_TOPOLOGY and self._topology_ctrl is not None:
            self._topology_ctrl.stop_preview()
            return
        self._wf_panel.stop_btn.setEnabled(False)
        if self._state.current_track is not None:
            self._wf_panel.play_btn.setEnabled(True)
        self._wf_panel.waveform.set_cursor(0)
        self.update_time_label(0)

    @Slot(str)
    def on_playback_error(self, message: str) -> None:
        self._shell.show_status(f"Playback error: {message}")

    @Slot(int)
    def on_waveform_seek(self, sample_index: int) -> None:
        preview = self._repair_preview
        if preview and preview.pending_play is not None:
            _, mode, channel = preview.pending_play
            preview.pending_play = (sample_index, mode, channel)
        elif self._playback.is_playing:
            self._playback.stop()
            self._wf_panel.waveform.set_cursor(sample_index)
            self.play()
        self.update_time_label(sample_index)

    def update_time_label(self, sample_pos: int = 0) -> None:
        track = self._state.current_track
        if track is None or track.samplerate <= 0:
            self._wf_panel.time_label.setText("00:00 / 00:00")
            return
        sr = track.samplerate
        self._wf_panel.time_label.setText(
            f"{fmt_time(sample_pos / sr)} / {fmt_time(track.total_samples / sr)}  \u2022  {sample_pos:,}"
        )
