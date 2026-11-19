"""Own the selected File preview and discard obsolete background results."""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Slot

from sessionpreplib.audio import get_window_samples
from ..waveform.compute import WaveformLoadWorker


class RepairPreview(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.panel = window._wf_panel
        self.track = None
        self.recipe = None
        self.audition = None
        self.signature = None
        self.pending_play = None
        self.worker = None
        self.generation = 0
        self.pending = False
        self.preserve_view = False
        self.displayed = False
        self.cache = {}
        self.panel.audition_source_changed.connect(self._source_changed)
        self.panel.inspection_band_changed.connect(self.request)
        self.panel.preview_retry.connect(self.retry)

    def select(self, track, *, refresh=False):
        window = self.window
        changed = self.track is not track
        if changed or refresh or self.recipe != track.repair_recipe:
            self.clear_material()
            self.track = track
            self.recipe = track.repair_recipe
            if changed:
                self.displayed = False
                self.panel.set_corrected(False)
        self.panel.set_audition_available(
            self.recipe.active, "Enable a repair to audition the corrected source.")
        window._file_report.set_report_source(self.panel.corrected)
        self.panel.set_inspection_rate(track.samplerate)
        self.panel.update_play_mode_channels(track.channels)
        window._wf_container.set_content_available(True)
        if not self.displayed:
            window._waveform_context_generation = None
            window._update_time_label(0)
        window._waveform_unavailable_generation = None
        window._refresh_waveform_issues()
        self.request()

    def clear_material(self):
        self.generation += 1
        self.pending = False
        if self.worker is not None:
            self.worker.cancel()
        self.audition = None
        self.signature = None
        self.pending_play = None
        self.cache.clear()

    @Slot(bool)
    def _source_changed(self, corrected):
        if self.track is not None and self.track is self.window._current_track:
            self.window._pending_detector_visualization = None
            self.window._file_report.set_report_source(corrected)
            self.window._refresh_waveform_issues()
            self.request()

    def planning(self):
        if self.track is None or self.track is not self.window._current_track:
            return
        self.clear_material()
        if not (self.track.repair_recipe.active or self.track._cache.get("repair_verification_pending")):
            self.panel.set_audition_available(False, "Enable a repair to audition the corrected source.")
            self.window._file_report.set_report_source(False)
        if self.window._playback.is_playing:
            self.window._waveform.set_cursor(self.window._playback.current_sample())
            self.window._playback.stop()
        self.panel.play_btn.setEnabled(False)
        self.panel.stop_btn.setEnabled(False)
        self.window._waveform.set_loading(True, preserve_view=self.displayed)
        self.panel.set_preview_notice("Updating repair preview…")

    @Slot()
    def retry(self):
        if self.track is self.window._current_track and self.track is not None:
            if self.track._cache.get("repair_error"):
                self.window._queue_repair_plan([self.track], revalidate_source=True)
            else:
                self.select(self.track, refresh=True)

    def play(self, sample, mode, channel):
        if self.audition is None:
            return
        self.pending_play = (sample, mode, channel)
        self.request(update_plot=False)

    @Slot()
    def request(self, *, update_plot=True):
        window = self.window
        if self.track is None or self.track is not window._current_track:
            return
        plan = window._repair_plan_worker
        if self.track.filename in window._repair_plan_pending or (
                plan and any(t.filename == self.track.filename for t in plan.tracks)):
            self.planning()
            return
        self.generation += 1
        self.pending = True
        self.preserve_view = self.displayed
        if update_plot:
            window._waveform.set_loading(True, preserve_view=self.preserve_view)
        if self.audition is None:
            self.panel.set_preview_notice()
            self.panel.play_btn.setEnabled(False)
        if self.worker is not None:
            self.worker.cancel()
        else:
            self._start()

    def _start(self):
        window = self.window
        waveform = window._waveform
        self.pending = False
        spectrogram = waveform._display_mode == "spectrogram"
        band = self.panel.inspection_band if spectrogram else "full"
        win = get_window_samples(self.track, window._flat_config().get("window", 400))
        key = (self.panel.corrected, spectrogram, band, waveform.spec_n_fft, waveform.spec_window, win)
        cached = self.cache.get(self.panel.corrected)
        worker = WaveformLoadWorker(
            None, self.track.samplerate, win, spec_n_fft=waveform.spec_n_fft,
            spec_window=waveform.spec_window, compute_spectrogram=spectrogram,
            preview_track=self.track, audition=self.audition, corrected=self.panel.corrected,
            band=band, cached=cached[1] if cached and cached[0] == key else None,
            source_signature=self.signature, parent=self)
        self.worker = worker
        worker.generation = self.generation
        worker.key = key
        worker.source_ready.connect(self._source_ready, Qt.QueuedConnection)
        worker.finished.connect(self._ready, Qt.QueuedConnection)
        worker.failed.connect(self._failed, Qt.QueuedConnection)
        worker.settled.connect(self._settled, Qt.QueuedConnection)
        worker.start()

    def _current(self, worker):
        return (worker is self.worker and worker.generation == self.generation
                and worker.track is self.window._current_track
                and worker.recipe == worker.track.repair_recipe
                and self.window._visualization_context_is_current())

    @Slot()
    def _source_ready(self):
        """Audio can switch as soon as validated, even if the plot is still rendering."""
        worker = self.sender()
        if not self._current(worker):
            return
        self.audition = worker.audition
        self.signature = worker.source_signature
        self.window._playback.set_audition_source(worker.corrected)
        if self.pending_play is not None:
            sample, mode, channel = self.pending_play
            self.pending_play = None
            self.window._playback.play(
                self.audition.original, self.audition.sample_rate, sample, mode=mode,
                channel=channel, audition=self.audition, corrected=worker.corrected)
        playing = self.window._playback.is_playing
        self.panel.set_content_available(True)
        self.panel.play_btn.setEnabled(not playing)
        self.panel.stop_btn.setEnabled(playing)
        pad = self.audition.monitor_gain_db
        self.panel.set_preview_notice(f"Monitor pad {pad:.1f} dB · both sources" if pad else "")

    @Slot(object)
    def _ready(self, result):
        worker = self.sender()
        if not self._current(worker):
            return
        cached = self.cache.get(worker.corrected)
        if cached and cached[1] is result and not self.window._waveform.is_loading:
            # Play and seek validate the file even when this plot is already visible.
            return
        self.cache[worker.corrected] = (worker.key, result)
        self.window._on_waveform_loaded(result, self.track, preserve_view=self.preserve_view)
        self.displayed = True

    @Slot(str)
    def _failed(self, detail):
        worker = self.sender()
        if not self._current(worker):
            return
        self.audition = None
        self.pending_play = None
        self.cache.clear()
        self.window._playback.stop()
        self.window._waveform.set_unavailable()
        self.window._pending_detector_visualization = None
        self.window._waveform_context_generation = None
        self.window._waveform_unavailable_generation = self.window._file_report.context_generation
        self.panel.set_content_available(False)
        self.panel.set_preview_notice(f"Preview unavailable · {detail}", retry=True)
        if worker.invalid_source and (self.track.repair_recipe.active or self.track.corrected_analysis):
            self.track.corrected_analysis = None
            self.track._cache["repair_error"] = detail
            self.window._mark_prepare_stale()
            self.window._refresh_file_tab(self.track)
            self.window._refresh_track_repair_editor(self.track)
            self.window._update_track_row(self.track.filename)
            self.window._refresh_verified_summary()
            self.window._update_prepare_button()

    @Slot()
    def _settled(self):
        worker = self.sender()
        if worker is not self.worker:
            return
        worker.wait()
        self.worker = None
        worker.deleteLater()
        if self.pending:
            self._start()

    def close(self):
        self.clear_material()
        if self.worker is not None:
            self.worker.wait()
