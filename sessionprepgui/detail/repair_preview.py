"""Own the selected File preview and discard obsolete background results."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Qt, Signal, Slot
from shiboken6 import isValid

from sessionpreplib.audio import get_window_samples
from ..state import WorkspaceState
from ..waveform.compute import WaveformLoadWorker

if TYPE_CHECKING:
    from ..waveform.panel import WaveformPanel
    from .browser import ReportBrowser
    from .controller import TrackDetailController
    from .playback import PlaybackController


class RepairPreview(QObject):
    revalidate_repair_requested = Signal(object)  # TrackContext
    invalid_source_detected = Signal(object)      # TrackContext

    def __init__(
        self,
        state: WorkspaceState,
        panel: WaveformPanel,
        file_report: ReportBrowser,
        playback: PlaybackController,
        *,
        get_detail_ctrl: Callable[[], TrackDetailController],
        is_track_repair_pending: Callable[[str], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self.panel = panel
        self._file_report = file_report
        self._playback = playback
        self._get_detail_ctrl = get_detail_ctrl
        self._is_track_repair_pending = is_track_repair_pending
        self.track: Any = None
        self.recipe: Any = None
        self.audition: Any = None
        self.signature: Any = None
        self.pending_play: tuple[int, str, int | None] | None = None
        self.worker: Any = None
        self.generation = 0
        self.pending = False
        self.preserve_view = False
        self.displayed = False
        self.cache: dict[bool, tuple[Any, Any]] = {}
        self.panel.audition_source_changed.connect(self._source_changed)
        self.panel.inspection_band_changed.connect(self.request)
        self.panel.preview_retry.connect(self.retry)

    def select(self, track: Any, *, refresh: bool = False) -> None:
        detail = self._get_detail_ctrl()
        changed = self.track is not track
        if changed or refresh or self.recipe != track.repair_recipe:
            self.clear_material()
            self.track = track
            self.recipe = track.repair_recipe
            if changed:
                self.displayed = False
                self.panel.set_corrected(False)
        self.panel.set_audition_available(
            self.recipe.active, "Enable a repair to audition the corrected source."
        )
        self._file_report.set_report_source(self.panel.corrected)
        self.panel.set_inspection_rate(track.samplerate)
        self.panel.update_play_mode_channels(track.channels)
        self.panel.set_content_available(True)
        if not self.displayed:
            detail.waveform_context_generation = None
            detail.update_time_label(0)
        detail.waveform_unavailable_generation = None
        detail.refresh_waveform_issues()
        self.request()

    def clear_material(self) -> None:
        self.generation += 1
        self.pending = False
        if self.worker is not None:
            self.worker.cancel()
        self.audition = None
        self.signature = None
        self.pending_play = None
        self.cache.clear()

    @Slot(bool)
    def _source_changed(self, corrected: bool) -> None:
        if self.track is not None and self.track is self._state.current_track:
            detail = self._get_detail_ctrl()
            detail.pending_detector_visualization = None
            self._file_report.set_report_source(corrected)
            detail.refresh_waveform_issues()
            self.request()

    def planning(self) -> None:
        track = self.track
        current = self._state.current_track
        if track is None or current is None or track is not current:
            return
        self.clear_material()
        if not (current.repair_recipe.active or current._cache.get("repair_verification_pending")):
            self.panel.set_audition_available(False, "Enable a repair to audition the corrected source.")
            self._file_report.set_report_source(False)
        if self._playback.is_playing:
            self.panel.waveform.set_cursor(self._playback.current_sample())
            self._playback.stop()
        self.panel.play_btn.setEnabled(False)
        self.panel.stop_btn.setEnabled(False)
        self.panel.waveform.set_loading(True, preserve_view=self.displayed)
        self.panel.set_preview_notice("Updating repair preview…")

    @Slot()
    def retry(self) -> None:
        if self.track is self._state.current_track and self.track is not None:
            if self.track._cache.get("repair_error"):
                self.revalidate_repair_requested.emit(self.track)
            else:
                self.select(self.track, refresh=True)

    def play(self, sample: int, mode: str, channel: int | None) -> None:
        if self.audition is None:
            return
        self.pending_play = (sample, mode, channel)
        self.request(update_plot=False)

    @Slot()
    def request(self, *, update_plot: bool = True) -> None:
        track = self.track
        current = self._state.current_track
        if track is None or current is None or track is not current:
            return
        if self._is_track_repair_pending(current.filename):
            self.planning()
            return
        self.generation += 1
        self.pending = True
        self.preserve_view = self.displayed
        if update_plot:
            self.panel.waveform.set_loading(True, preserve_view=self.preserve_view)
        if self.audition is None:
            self.panel.set_preview_notice()
            self.panel.play_btn.setEnabled(False)
        if self.worker is not None:
            self.worker.cancel()
        else:
            self._start()

    def _start(self) -> None:
        waveform = self.panel.waveform
        self.pending = False
        spectrogram = waveform._display_mode == "spectrogram"
        band = self.panel.inspection_band if spectrogram else "full"
        win = get_window_samples(self.track, self._state.flat_config().get("window", 400))
        key = (self.panel.corrected, spectrogram, band, waveform.spec_n_fft, waveform.spec_window, win)
        cached = self.cache.get(self.panel.corrected)
        worker_cls: Any = WaveformLoadWorker
        worker: Any = worker_cls(
            None,
            self.track.samplerate,
            win,
            spec_n_fft=waveform.spec_n_fft,
            spec_window=waveform.spec_window,
            compute_spectrogram=spectrogram,
            preview_track=self.track,
            audition=self.audition,
            corrected=self.panel.corrected,
            band=band,
            cached=cached[1] if cached and cached[0] == key else None,
            source_signature=self.signature,
            parent=self,
        )
        self.worker = worker
        worker.generation = self.generation
        worker.key = key
        worker.source_ready.connect(self._source_ready, Qt.ConnectionType.QueuedConnection)
        worker.completed.connect(self._ready, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)
        worker.settled.connect(self._settled, Qt.ConnectionType.QueuedConnection)
        worker.start()

    def _current(self, worker: Any) -> bool:
        track = self._state.current_track
        return (
            worker is self.worker
            and worker.generation == self.generation
            and track is not None
            and worker.track is track
            and worker.recipe == track.repair_recipe
            and self._get_detail_ctrl().visualization_context_is_current()
        )

    @Slot()
    def _source_ready(self) -> None:
        """Audio can switch as soon as validated, even if the plot is still rendering."""
        worker: Any = self.sender()
        if not self._current(worker):
            return
        self.audition = worker.audition
        self.signature = worker.source_signature
        self._playback.set_audition_source(worker.corrected)
        if self.pending_play is not None:
            sample, mode, channel = self.pending_play
            self.pending_play = None
            self._playback.play(
                self.audition.original,
                self.audition.sample_rate,
                sample,
                mode=mode,
                channel=channel,
                audition=self.audition,
                corrected=worker.corrected,
            )
        detail_ctrl = self._get_detail_ctrl()
        playing = self._playback.is_playing and detail_ctrl.is_phase_active()
        self.panel.set_content_available(True)
        self.panel.play_btn.setEnabled(not playing)
        self.panel.stop_btn.setEnabled(playing)
        pad = self.audition.monitor_gain_db
        self.panel.set_preview_notice(f"Monitor pad {pad:.1f} dB · both sources" if pad else "")

    @Slot(object)
    def _ready(self, result: Any) -> None:
        worker: Any = self.sender()
        if not self._current(worker):
            return
        cached = self.cache.get(worker.corrected)
        if cached and cached[1] is result and not self.panel.waveform.is_loading:
            # Play and seek validate the file even when this plot is already visible.
            return
        self.cache[worker.corrected] = (worker.key, result)
        self._get_detail_ctrl().on_waveform_loaded(result, self.track, preserve_view=self.preserve_view)
        self.displayed = True

    @Slot(str)
    def _failed(self, detail: str) -> None:
        worker: Any = self.sender()
        if not self._current(worker):
            return
        detail_ctrl = self._get_detail_ctrl()
        self.audition = None
        self.pending_play = None
        self.cache.clear()
        if detail_ctrl.is_phase_active():
            self._playback.stop()
        self.panel.waveform.set_unavailable()
        detail_ctrl.pending_detector_visualization = None
        detail_ctrl.waveform_context_generation = None
        detail_ctrl.waveform_unavailable_generation = self._file_report.context_generation
        self.panel.set_content_available(False)
        self.panel.set_preview_notice(f"Preview unavailable · {detail}", retry=True)
        if worker.invalid_source and (self.track.repair_recipe.active or self.track.corrected_analysis):
            self.track.corrected_analysis = None
            self.track._cache["repair_error"] = detail
            self.invalid_source_detected.emit(self.track)

    @Slot()
    def _settled(self) -> None:
        worker: Any = self.sender()
        if worker is not self.worker:
            return
        worker.wait()
        self.worker = None
        worker.deleteLater()
        if self.pending:
            self._start()

    def close(self) -> None:
        self.clear_material()
        worker = self.worker
        self.worker = None
        if worker is not None and (not isinstance(worker, QObject) or isValid(worker)):
            worker.wait()
