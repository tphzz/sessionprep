"""Detail view mixin: file detail, waveform, overlays, and playback."""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import Qt, Slot, QTimer, QSignalBlocker
from PySide6.QtGui import QAction

from sessionpreplib.audio import get_window_samples

from ..helpers import fmt_time
from ..tracks.table_widgets import _TAB_FILE, _TAB_SUMMARY, _PHASE_TOPOLOGY
from ..theme import COLORS
from ..analysis.worker import AudioLoadWorker
from ..waveform.compute import WaveformLoadWorker
from .report import visualizable_issues


class DetailMixin:  # pylint: disable=too-few-public-methods
    """File detail view, waveform display, overlays, and playback.

    Mixed into ``SessionPrepWindow`` — not meant to be used standalone.
    """

    # ── Report rendering ──────────────────────────────────────────────────

    @property
    def _show_clean(self) -> bool:
        if self._session_config is not None:
            cfg = self._read_session_config()
            return cfg.get("presentation", {}).get(
                "show_clean_detectors", False)
        preset = self._active_preset()
        return preset.get("presentation", {}).get("show_clean_detectors", False)

    @property
    def _verbose(self) -> bool:
        return self._config.get("app", {}).get("report_verbosity", "normal") == "verbose"

    def _render_summary(self):
        """Render the diagnostic summary into the Summary tab."""
        if not self._summary or not self._session:
            return
        self._summary_view.present_summary(
            self._summary, self._session, show_clean=self._show_clean)

    def _refresh_file_report(self, track=None):
        track = track if track is not None else self._current_track
        if track is not None and self._session is not None:
            self._file_report.present_track(
                track, self._session, show_clean=self._show_clean,
                verbose=self._verbose)

    def _review_summary_file(self, filename):
        """Select the current file by identity, including after table sorting."""
        if self._session is None or self._summary_view.report_session is not self._session:
            return
        track = next((track for track in self._session.tracks
                      if track.filename == filename), None)
        if track is None:
            return
        for row in range(self._track_table.rowCount()):
            item = self._track_table.item(row, 0)
            if item is not None and item.text() == filename:
                with QSignalBlocker(self._track_table):
                    self._track_table.clearSelection()
                    self._track_table.setCurrentCell(row, 0)
                    self._track_table.selectRow(row)
                self._track_table.scrollToItem(item)
                self._show_track_detail(track)
                self._file_report.setFocus()
                return

    def _show_track_detail(self, track):
        """Populate the File tab with per-track detail + waveform.

        The HTML report is rendered and displayed immediately so the UI
        feels responsive.  Waveform loading (dtype conversion, peak
        finding, RMS setup) is deferred to the next event-loop iteration
        via ``QTimer.singleShot`` so the tab switch paints first.
        """
        self._on_stop()
        self._pending_detector_visualization = None
        self._waveform_context_generation = None
        self._waveform_unavailable_generation = None
        self._current_track = track

        # Show HTML report immediately
        self._refresh_file_report(track)

        # Enable and switch to File tab before heavy work
        self._detail_tabs.setTabEnabled(_TAB_FILE, True)
        self._detail_tabs.setCurrentIndex(_TAB_FILE)

        # Defer waveform loading so the UI repaints first
        QTimer.singleShot(0, lambda: self._load_waveform(track))

    def _load_waveform(self, track):
        """Start background waveform loading for *track*."""
        import time, logging
        t0 = time.perf_counter()
        log = logging.getLogger(__name__)

        # Guard: user may have clicked a different track while we were queued
        if self._current_track is not track:
            return

        # Cancel any in-flight workers
        if self._wf_worker is not None:
            self._wf_worker.cancel()
            self._wf_worker.finished.disconnect()
            self._wf_worker = None
        if self._audio_load_worker is not None:
            self._audio_load_worker.cancel()
            self._audio_load_worker.finished.disconnect()
            self._audio_load_worker = None

        # 1. Handle instant preview or loading state
        peak_cache = getattr(self, '_peak_cache', {})
        fn = getattr(track, 'filename', None)
        self._waveform_unavailable_generation = None
        self._waveform_context_generation = None
        if fn and fn in peak_cache:
            ready = self._waveform.set_preview_mode(
                track.channels, track.total_samples, track.samplerate, peak_cache[fn]
            )
            if ready:
                self._waveform_context_generation = self._file_report.context_generation
        else:
            self._waveform.set_loading(True)
            if fn and hasattr(self, '_prioritize_peak'):
                self._prioritize_peak(fn)

        self._refresh_waveform_issues()
        self._apply_pending_detector_visualization()
        if self._detail_tabs.currentIndex() == _TAB_FILE:
            self._wf_container.set_content_available(True)
        self._play_btn.setEnabled(False)
        self._update_time_label(0)

        if fn:
            log.debug("[Trace] _load_waveform UI setup for '%s': %.2f ms", fn, (time.perf_counter() - t0) * 1000)

        # 2. If audio_data is absent but the file exists, load it from disk first
        if (track.audio_data is None or track.audio_data.size == 0) and \
                track.status == "OK" and os.path.isfile(track.filepath):
            worker = AudioLoadWorker(track, parent=self)
            self._audio_load_worker = worker
            worker.finished.connect(
                lambda t, orig=track, w=worker: self._on_audio_loaded(t, orig, w))
            worker.error.connect(
                lambda msg, orig=track, w=worker: self._on_audio_load_error(msg, orig, w))
            worker.start()
            return

        # 3. If audio is available in memory, run WaveformLoadWorker (for RMS/Spectrogram)
        has_audio = track.audio_data is not None and track.audio_data.size > 0
        if has_audio:
            nch = track.audio_data.shape[1] if track.audio_data.ndim > 1 else 1
            self._wf_panel.update_play_mode_channels(nch)
            self._play_btn.setEnabled(True)
            self._start_wf_worker(track)
        else:
            self._pending_detector_visualization = None
            self._waveform_context_generation = None
            self._waveform_unavailable_generation = self._file_report.context_generation
            self._waveform.set_loading(False)
            self._waveform.set_audio(None, 44100)
            self._update_overlay_menu([])
            if self._detail_tabs.currentIndex() == _TAB_FILE:
                self._wf_container.set_content_available(False)
            self._play_btn.setEnabled(False)
            self._update_time_label(0)

    def _start_wf_worker(self, track):
        flat_cfg = self._flat_config()
        win_ms = flat_cfg.get("window", 400)
        ws = get_window_samples(track, win_ms)

        self._wf_worker = WaveformLoadWorker(
            track.audio_data, track.samplerate, ws,
            spec_n_fft=self._waveform.spec_n_fft,
            spec_window=self._waveform.spec_window,
            compute_spectrogram=(self._waveform._display_mode == "spectrogram"),
            parent=self)
        self._wf_worker.finished.connect(
            lambda result, t=track, w=self._wf_worker: self._on_waveform_loaded(result, t, w))
        self._wf_worker.start()

    @Slot(str)
    def _on_display_mode_changed(self, mode: str):
        if mode == "spectrogram" and self._current_track:
            track = self._current_track
            if track.audio_data is not None and track.audio_data.size > 0:
                if getattr(self._waveform._spec_renderer, '_spec_data', None) is None:
                    if self._wf_worker is None:
                        self._start_wf_worker(track)

    @Slot(object, object)
    def _on_waveform_loaded(self, result: dict, track, worker=None):
        """Receive waveform data only from the currently owned worker."""
        if worker is not None and worker is not self._wf_worker:
            return
        self._wf_worker = None

        # Discard if user switched to a different track
        if self._current_track is not track:
            return

        if not self._visualization_context_is_current():
            return
        self._waveform.set_precomputed(result)
        self._waveform_unavailable_generation = None
        self._waveform_context_generation = self._file_report.context_generation
        # Apply cached peak data for mip-level rendering
        peak_cache = getattr(self, '_peak_cache', {})
        fn = getattr(track, 'filename', None)
        if fn and fn in peak_cache:
            self._waveform.set_peak_data(peak_cache[fn])
        elif fn and hasattr(self, '_prioritize_peak'):
            self._prioritize_peak(fn)

        cmap = self._config.get("app", {}).get("spectrogram_colormap", "magma")
        self._waveform.set_colormap(cmap)
        # Sync colormap dropdown with preference
        for act in self._cmap_group.actions():
            if act.data() == cmap:
                act.setChecked(True)
                break

        self._refresh_waveform_issues()
        self._apply_pending_detector_visualization()
        self._update_time_label(0)

    def _on_audio_loaded(self, track, orig_track, worker=None):
        """Audio data loaded from disk; proceed to waveform rendering."""
        if worker is not None and worker is not self._audio_load_worker:
            return
        self._audio_load_worker = None
        # Discard if user switched tracks while we were loading
        if self._current_track is not orig_track:
            return
        # Now kick off the normal waveform worker path
        self._load_waveform(track)

    def _on_audio_load_error(self, message: str, track, worker=None):
        """Audio file could not be read from disk."""
        if worker is not None and worker is not self._audio_load_worker:
            return
        self._audio_load_worker = None
        if self._current_track is not track:
            return
        self._pending_detector_visualization = None
        self._waveform_context_generation = None
        self._waveform_unavailable_generation = self._file_report.context_generation
        self._waveform.set_loading(False)
        self._waveform.set_audio(None, 44100)
        self._update_overlay_menu([])
        self._wf_container.set_content_available(False)
        self._play_btn.setEnabled(False)
        self._status_bar.showMessage(f"Could not load audio: {message}")

    def _visualization_context_is_current(self):
        track = self._current_track
        return (self._session is not None and track is not None
                and any(t is track for t in self._session.tracks)
                and self._file_report.report_session is self._session
                and self._file_report.track_filename == track.filename)

    def _detector_visualization_issues(self, key):
        if not self._visualization_context_is_current():
            return []
        detector = next((d for d in self._session.detectors if d.id == key), None)
        result = self._current_track.detector_results.get(key)
        issues = visualizable_issues(result, self._current_track)
        if not issues:
            return []
        if detector is not None:
            severity = getattr(detector, "effective_severity", None)
            relevant = getattr(detector, "is_relevant", None)
            if callable(severity) and severity(result) is None:
                return []
            if callable(relevant) and not relevant(result, self._current_track):
                return []
        return issues

    def _refresh_waveform_issues(self):
        issues = []
        if self._visualization_context_is_current():
            for key in self._current_track.detector_results:
                issues.extend(self._detector_visualization_issues(key))
        self._waveform.set_issues(issues)
        self._update_overlay_menu(issues)

    def _on_file_report_context_changed(self):
        if not self._visualization_context_is_current():
            self._pending_detector_visualization = None
            self._waveform_context_generation = None
            self._waveform_unavailable_generation = None
            return
        pending = self._pending_detector_visualization
        if pending is not None and (
                pending[0] != self._file_report.context_generation
                or not self._detector_visualization_issues(pending[1])):
            self._pending_detector_visualization = None
        if self._waveform_context_generation == self._file_report.context_generation:
            self._refresh_waveform_issues()

    @Slot(str)
    def _visualize_detector(self, key):
        if not self._visualization_context_is_current():
            return
        if not self._file_report.visualization_labels(key):
            return
        issues = self._detector_visualization_issues(key)
        if not issues:
            return
        generation = self._file_report.context_generation
        if getattr(self, "_waveform_unavailable_generation", None) == generation:
            self._status_bar.showMessage(
                "Waveform unavailable for this file. Select the file again to retry loading.")
            return
        anchor = self._file_report.capture_reading_anchor()
        self._pending_detector_visualization = (generation, key)
        selected = self._remembered_detector_overlays()
        selected.clear()
        selected.update(issue.label for issue in issues)
        if self._waveform_context_generation != generation:
            self._waveform.set_loading(True)
        self._refresh_waveform_issues()
        self._wf_panel.set_expanded(True)
        self._apply_pending_detector_visualization()
        self._file_report.restore_reading_anchor(anchor)
        self._file_report.focus_report_link("report-visualize", key)

    def _apply_pending_detector_visualization(self):
        pending = self._pending_detector_visualization
        if pending is None:
            return
        generation, key = pending
        if (not self._visualization_context_is_current()
                or generation != self._file_report.context_generation
                or not self._file_report.visualization_labels(key)
                or not self._detector_visualization_issues(key)):
            self._pending_detector_visualization = None
            return
        if self._waveform_context_generation == generation:
            self._waveform.zoom_fit()
            self._pending_detector_visualization = None

    # ── Overlay dropdown ────────────────────────────────────────────────

    def _set_overlay_button_count(self, count: int) -> None:
        self._overlay_btn.setText(
            f"Detector Overlays ({count})" if count else "Detector Overlays")

    def _remembered_detector_overlays(self) -> set[str]:
        if not hasattr(self, "_selected_detector_overlays"):
            self._selected_detector_overlays = set()
        return self._selected_detector_overlays

    def _apply_active_detector_overlays(self, available_labels: set[str]) -> None:
        selected = self._remembered_detector_overlays()
        active = selected & available_labels
        self._waveform.set_enabled_overlays(active)
        self._set_overlay_button_count(len(active))

    def _update_overlay_menu(self, issues: list):
        """Rebuild the overlay dropdown menu based on current track issues."""
        self._overlay_menu.clear()
        self._remembered_detector_overlays()

        if not issues:
            self._apply_active_detector_overlays(set())
            return

        # Build detector instance map from session
        det_map: dict[str, object] = {}
        det_names: dict[str, str] = {}
        if self._session and hasattr(self._session, "detectors"):
            for d in self._session.detectors:
                det_map[d.id] = d
                det_names[d.id] = d.name

        # Filter out issues from detectors that suppress themselves or are skipped
        track = self._current_track
        filtered_issues = []
        for issue in issues:
            det = det_map.get(issue.label)
            if det and track:
                result = track.detector_results.get(issue.label)
                if result:
                    if hasattr(det, 'effective_severity') and det.effective_severity(result) is None:
                        continue
                    relevant = getattr(det, "is_relevant", None)
                    if callable(relevant) and not relevant(result, track):
                        continue
            filtered_issues.append(issue)

        if not filtered_issues:
            self._apply_active_detector_overlays(set())
            return

        # Build {label: count} from filtered issue list
        label_counts: dict[str, int] = {}
        for issue in filtered_issues:
            label_counts[issue.label] = label_counts.get(issue.label, 0) + 1

        # Add a checkable action per detector that has issues
        selected = self._remembered_detector_overlays()
        for label in sorted(label_counts, key=lambda lb: det_names.get(lb, lb).lower()):
            name = det_names.get(label, label)
            count = label_counts[label]
            action = self._overlay_menu.addAction(f"{name} ({count})")
            action.setCheckable(True)
            action.setData(label)
            action.setChecked(label in selected)
            action.toggled.connect(self._on_overlay_toggled)

        self._apply_active_detector_overlays(set(label_counts))

    @Slot()
    def _on_overlay_toggled(self):
        """Collect checked overlay labels and update the waveform."""
        self._pending_detector_visualization = None
        available = set()
        checked = set()
        for action in self._overlay_menu.actions():
            if not action.isCheckable():
                continue
            label = action.data()
            available.add(label)
            if action.isChecked():
                checked.add(label)
        selected = self._remembered_detector_overlays()
        selected.difference_update(available)
        selected.update(checked)
        self._waveform.set_enabled_overlays(checked)
        self._set_overlay_button_count(len(checked))

    @Slot(QAction)
    def _on_spec_fft_changed(self, action):
        self._waveform.set_spec_fft(int(action.data()))

    @Slot(QAction)
    def _on_spec_window_changed(self, action):
        self._waveform.set_spec_window(action.data())

    @Slot(QAction)
    def _on_spec_cmap_changed(self, action):
        self._waveform.set_colormap(action.data())

    @Slot(QAction)
    def _on_spec_floor_changed(self, action):
        self._waveform.set_spec_db_floor(float(action.data()))

    @Slot(QAction)
    def _on_spec_ceil_changed(self, action):
        self._waveform.set_spec_db_ceil(float(action.data()))

    # ── Playback ──────────────────────────────────────────────────────────

    @Slot()
    def _on_toggle_play(self):
        # Stop always works regardless of which phase started playback
        if self._playback.is_playing:
            self._playback.stop()
            # Reset both phases' transport UI
            self._stop_btn.setEnabled(False)
            if self._current_track is not None:
                self._play_btn.setEnabled(True)
            self._topo_wf_panel.stop_btn.setEnabled(False)
            self._topo_wf_panel.play_btn.setEnabled(
                self._topo_cached_audio is not None)
            return
        # Start based on current tab
        if self._phase_tabs.currentIndex() == _PHASE_TOPOLOGY:
            if self._topo_cached_audio is not None:
                self._on_topo_play()
        elif self._current_track is not None:
            self._on_play()

    @Slot()
    def _on_play(self):
        track = self._current_track
        if track is None or track.audio_data is None:
            return
        self._on_stop()
        start = self._waveform._cursor_sample
        mode, channel = self._wf_panel.play_mode
        self._playback.play(track.audio_data, track.samplerate, start,
                            mode=mode, channel=channel)
        if self._playback.is_playing:
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)

    @Slot()
    def _on_stop(self):
        was_playing = self._playback.is_playing
        start_sample = self._playback.play_start_sample
        self._playback.stop()
        self._stop_btn.setEnabled(False)
        if self._current_track is not None:
            self._play_btn.setEnabled(True)
        if was_playing:
            self._waveform.set_cursor(start_sample)
            self._update_time_label(start_sample)

    @Slot(int)
    def _on_cursor_updated(self, sample_pos: int):
        if self._phase_tabs.currentIndex() == _PHASE_TOPOLOGY:
            self._topo_wf_panel.waveform.set_cursor(sample_pos)
            self._topo_update_time_label(sample_pos)
            return
        self._waveform.set_cursor(sample_pos)
        self._update_time_label(sample_pos)

    @Slot()
    def _on_playback_finished(self):
        if self._phase_tabs.currentIndex() == _PHASE_TOPOLOGY:
            self._on_topo_stop()
            return
        self._stop_btn.setEnabled(False)
        if self._current_track is not None:
            self._play_btn.setEnabled(True)
        self._waveform.set_cursor(0)
        self._update_time_label(0)

    @Slot(str)
    def _on_playback_error(self, message: str):
        self._status_bar.showMessage(f"Playback error: {message}")

    @Slot(int)
    def _on_waveform_seek(self, sample_index: int):
        if self._playback.is_playing:
            self._on_stop()
            self._waveform.set_cursor(sample_index)
            self._on_play()
        else:
            self._update_time_label(sample_index)

    def _update_time_label(self, sample_pos: int = 0):
        track = self._current_track
        if track is None or track.samplerate <= 0:
            self._time_label.setText("00:00 / 00:00")
            return
        sr = track.samplerate
        self._time_label.setText(
            f"{fmt_time(sample_pos / sr)} / {fmt_time(track.total_samples / sr)}"
            f"  \u2022  {sample_pos:,}"
        )
