"""Waveform background computation: colormaps, mel spectrogram, load workers."""

from __future__ import annotations

import collections
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
import logging
import threading

import numpy as np

from PySide6.QtCore import QThread, Signal

from sessionpreplib.audio import max_momentary_rms, squared_sample_prefix, validate_analysis_level
from sessionpreplib.audition import RepairAudition, inspection_spectrogram
from sessionpreplib.repair_plan import source_for_render
from sessionpreplib.spectrum import full_spectrogram, hz_to_mel as _hz_to_mel, mel_to_hz as _mel_to_hz
from .peakcache import peak_source_key


# ---------------------------------------------------------------------------
# Spectrogram colormaps
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

SPECTROGRAM_COLORMAPS: dict[str, np.ndarray] = {}  # name → (256, 4) uint8 RGBA


def _register_colormap(name: str,
                       controls: list[tuple[float, tuple[int, int, int]]]):
    """Build a 256-entry RGBA LUT from control points via linear interpolation."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:, 3] = 255  # fully opaque
    positions = np.array([c[0] for c in controls])
    for ch in range(3):
        values = np.array([c[1][ch] for c in controls], dtype=np.float64)
        lut[:, ch] = np.clip(
            np.interp(np.linspace(0, 1, 256), positions, values), 0, 255
        ).astype(np.uint8)
    SPECTROGRAM_COLORMAPS[name] = lut


_register_colormap("magma", [
    (0.0, (0, 0, 4)),
    (0.25, (81, 18, 124)),
    (0.5, (183, 55, 121)),
    (0.75, (254, 159, 109)),
    (1.0, (252, 253, 191)),
])

_register_colormap("viridis", [
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
])

_register_colormap("grayscale", [
    (0.0, (0, 0, 0)),
    (1.0, (255, 255, 255)),
])


# ---------------------------------------------------------------------------
# Spectrogram computation (used by background workers)
# ---------------------------------------------------------------------------

_SPEC_N_FFT = 2048
_SPEC_HOP = 512
_SPEC_N_MELS = 256
_SPEC_F_MIN = 20.0
_SPEC_F_MAX = 22050.0
_SPEC_DB_FLOOR = -80.0  # dB floor for normalization


def compute_mel_spectrogram(channels: list[np.ndarray], sr: int, *,
                            n_fft: int = _SPEC_N_FFT,
                            hop: int | None = None,
                            window: str = "hann",
                            compare_channels: bool = False,
                            cancelled=None,
                            ) -> np.ndarray | None:
    """Return the full-band preview's calibrated peak dBFS on its mel axis."""
    if not channels or not len(channels[0]):
        return None
    if not compare_channels and len(channels) > 1:
        mono = channels[0].astype(np.float64, copy=True)
        for channel in channels[1:]:
            mono += channel
        channels = [mono / len(channels)]
    f_max = sr / 2 if compare_channels else min(_SPEC_F_MAX, sr / 2)
    return full_spectrogram(channels, sr, width=n_fft, window=window, frequency_max=f_max,
                            rows=_SPEC_N_MELS, hop=hop, cancelled=cancelled)[2]


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class WaveformLoadWorker(QThread):
    """Background thread for heavy waveform preparation work.

    Splits channels, finds peak position, and computes RMS-max position
    so the main thread stays responsive.
    """

    finished = Signal(object)  # emits a dict with all computed results
    source_ready = Signal()
    failed = Signal(str)
    settled = Signal()

    def __init__(self, audio_data: np.ndarray, samplerate: int,
                 rms_window_samples: int, *,
                 spec_n_fft: int = _SPEC_N_FFT,
                 spec_window: str = "hann",
                 compute_spectrogram: bool = True,
                 preview_track=None, audition=None, corrected=False, band="full", cached=None,
                 source_signature=None,
                 parent=None):
        super().__init__(parent)
        self._audio_data = audio_data
        self._samplerate = samplerate
        self._rms_win = rms_window_samples
        self._spec_n_fft = spec_n_fft
        self._spec_window = spec_window
        self._compute_spectrogram = compute_spectrogram
        self.track = preview_track
        self.recipe = getattr(preview_track, "repair_recipe", None)
        self.audition = audition
        self.corrected = corrected
        self.band = band
        self._cached = cached
        self.source_signature = source_signature
        self._cancelled = threading.Event()

    def cancel(self):
        """Request early termination of the computation."""
        self._cancelled.set()

    def run(self):
        try:
            if self.track is not None and not self._cancelled.is_set():
                if self.source_signature is None:
                    self.source_signature = self.track.source_file_signature
                    if self.source_signature is None and self.track.audio_data is None:
                        stat = Path(self.track.filepath).stat()
                        self.source_signature = (stat.st_size, stat.st_mtime_ns)
                source_track = replace(
                    self.track, source_file_signature=self.source_signature,
                    audio_data=self.track.audio_data if self.audition is None else self.audition.original)
                source = source_for_render(source_track)
                if self.audition is None:
                    if source.audio_data is None or not source.audio_data.size:
                        raise ValueError("Source audio is empty.")
                    self.audition = RepairAudition(source.audio_data, source.samplerate, self.recipe,
                                                  source_subtype=source.subtype)
                validate_analysis_level(self.audition.monitor_pad.peak, len(self.audition.original))
                if not self._cancelled.is_set():
                    self.source_ready.emit()
                if self._cached is not None:
                    self.finished.emit(self._cached)
                    return
                self._audio_data = self.audition.corrected if self.corrected else self.audition.original
            if not self._cancelled.is_set():
                self._compute()
        except CancelledError:
            pass
        except FileNotFoundError:
            if not self._cancelled.is_set():
                self.failed.emit("Source file is missing. Restore the source and retry.")
        except Exception as exc:
            if not self._cancelled.is_set():
                log.exception("Waveform preview failed")
                self.failed.emit(str(exc))
        finally:
            self.settled.emit()

    def _compute(self):
        import time, logging
        t_start = time.perf_counter()
        log = logging.getLogger(__name__)

        data = self._audio_data
        sr = self._samplerate
        win = self._rms_win

        # --- Channel splitting ---
        if isinstance(data, list):
            channels = data
        else:
            if data is None or data.size == 0:
                return
            if data.ndim == 1:
                channels = [data]
            else:
                channels = [data[:, ch] for ch in range(data.shape[1])]
        if not channels:
            return
        nch = len(channels)
        total = len(channels[0])

        if self._cancelled.is_set():
            return

        # --- Peak finding (Zero allocation) ---
        t0 = time.perf_counter()

        def _find_peak(ch: np.ndarray) -> int:
            p_max = int(np.argmax(ch))
            p_min = int(np.argmin(ch))
            return p_min if abs(float(ch[p_min])) > abs(float(ch[p_max])) else p_max

        if nch == 1:
            peak_sample = _find_peak(channels[0])
            peak_channel = 0
        else:
            peaks_per_ch = [_find_peak(ch) for ch in channels]
            max_vals = [abs(float(channels[i][p])) for i, p in enumerate(peaks_per_ch)]
            peak_channel = int(np.argmax(max_vals))
            peak_sample = peaks_per_ch[peak_channel]
        peak_lin = abs(float(channels[peak_channel][peak_sample]))
        validate_analysis_level(peak_lin, total)
        peak_db = 20.0 * np.log10(peak_lin) if peak_lin > 0 else float('-inf')
        peak_amplitude = float(channels[peak_channel][peak_sample])

        log.debug("[Trace] WaveformLoadWorker Peak finding: %.2f ms", (time.perf_counter() - t0) * 1000)

        if self._cancelled.is_set():
            return

        # --- RMS cumsum (computed once, reused for envelope drawing) ---
        t0 = time.perf_counter()
        win = min(win, total)
        rms_cumsums = []
        if win > 0:
            for channel in channels:
                if self._cancelled.is_set():
                    return
                rms_cumsums.append(squared_sample_prefix(channel))
        rms_max_sample, rms_max_db, rms_max_amplitude = max_momentary_rms(rms_cumsums, win)

        log.debug("[Trace] WaveformLoadWorker RMS: %.2f ms", (time.perf_counter() - t0) * 1000)

        if self._cancelled.is_set():
            return

        # --- Spectrogram ---
        spec_db = None
        spec_freq = None
        spec_times = None
        if self._compute_spectrogram:
            t0 = time.perf_counter()
            source_steps = self.audition.source_steps if self.audition is not None else None
            if self.band == "full" and self.track is not None:
                spec_times, spec_freq, spec_db = full_spectrogram(
                    channels, sr, width=self._spec_n_fft, window=self._spec_window,
                    frequency_max=sr / 2, rows=_SPEC_N_MELS, cancelled=self._cancelled.is_set,
                    source_steps=source_steps)
            elif self.band == "full":
                spec_db = compute_mel_spectrogram(
                    channels, sr, n_fft=self._spec_n_fft, window=self._spec_window,
                    cancelled=self._cancelled.is_set)
            else:
                spec_times, spec_freq, spec_db = inspection_spectrogram(
                    data, sr, self.band, cancelled=self._cancelled.is_set, source_steps=source_steps)
            log.debug("[Trace] WaveformLoadWorker STFT: %.2f ms", (time.perf_counter() - t0) * 1000)

            if self._cancelled.is_set():
                return

        log.debug("[Trace] WaveformLoadWorker TOTAL: %.2f ms", (time.perf_counter() - t_start) * 1000)

        self.finished.emit({
            "channels": channels,
            "samplerate": sr,
            "total_samples": total,
            "peak_sample": peak_sample,
            "peak_channel": peak_channel,
            "peak_db": peak_db,
            "peak_amplitude": peak_amplitude,
            "rms_window_samples": win,
            "rms_max_sample": rms_max_sample,
            "rms_max_db": rms_max_db,
            "rms_max_amplitude": rms_max_amplitude,
            "rms_cumsums": rms_cumsums,
            "spec_db": spec_db,
            "spec_freq": spec_freq,
            "spec_times": spec_times,
            "inspection_band": self.band,
            "audition": self.audition,
            "corrected": self.corrected,
        })


class SpectrogramRecomputeWorker(QThread):
    """Lightweight background thread to recompute the mel spectrogram."""

    finished = Signal(object)  # emits np.ndarray | None

    def __init__(self, channels: list[np.ndarray], sr: int, *,
                 n_fft: int = _SPEC_N_FFT, window: str = "hann",
                 parent=None):
        super().__init__(parent)
        self._channels = channels
        self._sr = sr
        self._n_fft = n_fft
        self._window = window
        self._cancelled = threading.Event()

    def cancel(self):
        """Request early termination."""
        self._cancelled.set()

    def run(self):
        result = compute_mel_spectrogram(
            self._channels, self._sr,
            n_fft=self._n_fft, window=self._window,
        )
        if self._cancelled.is_set():
            return
        self.finished.emit(result)


class PeakBuildWorker(QThread):
    """Eagerly build and save ``.peaks`` files for a batch of audio files.

    Runs in the background with a thread pool — does not block the UI.
    Each queued delivery carries its job. Successful files also carry their
    source path so projects and routed outputs can reuse the same track name.

    The work queue is mutable: call ``prioritize(filepath)`` to move a file
    to the front of the pending queue so it is processed next.
    """

    progress = Signal(object, str)
    progress_value = Signal(object, int, int)
    file_done = Signal(object, str, object)
    all_done = Signal(object, int)

    def __init__(self, items: list[tuple[str, str, str]],
                 parent=None):
        """
        Parameters
        ----------
        items : list of (filepath, filename, peaks_path)
            *filepath*: absolute path to the audio file on disk.
            *filename*: user-facing track name.
            *peaks_path*: absolute path for the ``.peaks`` output file.
        """
        super().__init__(parent)
        self._items_map = {peak_source_key(item[0]): item for item in items}
        self._queue = collections.deque(self._items_map)
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._total = len(self._items_map)

    def cancel(self):
        self._cancelled.set()

    def prioritize(self, source_key: str):
        """Move this physical source to the front if it is still pending."""
        with self._lock:
            try:
                self._queue.remove(source_key)
            except ValueError:
                return  # already processed or not in queue
            self._queue.appendleft(source_key)
            log.debug("Prioritized peak build for '%s'", source_key)

    def _next_item(self) -> tuple[str, str, str] | None:
        """Pop the next item from the queue under lock."""
        with self._lock:
            while self._queue:
                fn = self._queue.popleft()
                item = self._items_map.get(fn)
                if item:
                    return item
        return None

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import soundfile as sf
        from .peakcache import (
            build_peaks, save_peaks, load_peaks, get_source_mtime,
        )

        def _process(filepath, peaks_path):
            if self._cancelled.is_set():
                return None
            mtime = get_source_mtime(filepath)
            # Check if existing peaks are still valid
            existing = load_peaks(peaks_path, expected_mtime=mtime)
            if existing is not None:
                return existing
            # Build from audio
            import time as _time
            _t0 = _time.perf_counter()
            try:
                data, sr = sf.read(filepath, dtype="float64")
            except Exception as e:
                log.debug("Failed to read '%s' for peak cache: %s", filepath, e)
                return None
            if self._cancelled.is_set():
                return None
            peak_data = build_peaks(data, sr, source_mtime=mtime)
            try:
                save_peaks(peak_data, peaks_path)
                _elapsed = (_time.perf_counter() - _t0) * 1000
                log.debug("Built peak cache for '%s' -> %s (%d levels, %.1f ms)", filepath, peaks_path, len(peak_data.levels), _elapsed)
            except OSError as e:
                log.debug("Failed to save peak cache for '%s' -> %s: %s", filepath, peaks_path, e)
            return peak_data

        max_workers = min(os.cpu_count() or 4, 6)
        completed = 0
        built = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Process items in small batches to keep the queue reorderable
            while not self._cancelled.is_set():
                # grab a batch of up to max_workers items
                batch = []
                for _ in range(max_workers):
                    item = self._next_item()
                    if item is None:
                        break
                    batch.append(item)
                if not batch:
                    break  # queue exhausted

                futures = {
                    pool.submit(_process, fp, pp): (fp, fn)
                    for fp, fn, pp in batch
                }
                for future in as_completed(futures):
                    if self._cancelled.is_set():
                        return
                    filepath, filename = futures[future]
                    peak_data = future.result()
                    completed += 1
                    if peak_data is not None:
                        built += 1
                        self.progress.emit(self,
                            f"Building peak cache: {filename}"
                            f"  ({completed}/{self._total})")
                        self.file_done.emit(self, filepath, peak_data)
                    self.progress_value.emit(self, completed, self._total)

        if not self._cancelled.is_set():
            self.progress.emit(self, "Peak cache creation finished.")
            log.info("Peak cache background batch finished (%d of %d files available)", built, self._total)
            self.all_done.emit(self, built)
