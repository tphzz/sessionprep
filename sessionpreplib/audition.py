"""Sample-aligned, unity-gain audition of whole-track repairs."""
from __future__ import annotations

import math
from concurrent.futures import CancelledError

import numpy as np
from scipy.signal.windows import hann

from .repairs import RepairRecipe, render_corrected


class RepairAudition:
    """The original and corrected share one time base and one monitor gain."""

    def __init__(self, original: np.ndarray, sample_rate: int, recipe: RepairRecipe):
        self.original = np.asarray(original)
        self.corrected = render_corrected(self.original, sample_rate, recipe)
        self.sample_rate = sample_rate
        peak = max(np.max(np.abs(side), initial=0.) for side in (self.original, self.corrected))
        self.monitor_gain_db = -math.ceil(200 * math.log10(peak)) / 10 if peak > 1 else 0.0
        self._scale = 10 ** (self.monitor_gain_db / 20)

    def playback_samples(self, corrected: bool) -> np.ndarray:
        audio = self.corrected if corrected else self.original
        return audio if self._scale == 1 else audio * self._scale

    def switch_samples(self, sample: int, from_corrected: bool, to_corrected: bool,
                       frames: int, *, fade_offset: int = 0) -> np.ndarray:
        """Render a callback at the source cursor. Carry elapsed fade frames between callbacks."""
        if min(sample, frames, fade_offset) < 0:
            raise ValueError("Playback position, frame count and fade offset must be nonnegative")
        dest = self.corrected if to_corrected else self.original
        result = np.array(dest[sample:sample + frames], dtype=np.float64, copy=True)
        if from_corrected != to_corrected:
            fade = max(2, round(0.01 * self.sample_rate))
            count = min(max(0, fade - fade_offset), len(result))
            wet = (fade_offset + np.arange(count, dtype=np.float64)) / (fade - 1)
            if result.ndim == 2:
                wet = wet[:, None]
            start = self.corrected if from_corrected else self.original
            result[:count] = start[sample:sample + count] * (1 - wet) + result[:count] * wet
        return result * self._scale


def inspection_spectrogram(audio: np.ndarray, sample_rate: int, band: str, *,
                           max_columns: int = 512, cancelled=None):
    """Return time, native frequency bins, and maximum per-channel peak dBFS.

    Low-band windows span one second (1 Hz bins). High-band windows span
    20 ms (50 Hz bins). Max-hold aggregation retains events between display
    columns. The returned levels use the coherent gain of the Hann window.
    """
    data = np.asarray(audio)
    if sample_rate <= 0 or max_columns < 1 or band not in ("low", "high"):
        raise ValueError("Invalid inspection band, sample rate or column count")
    if band == "high" and sample_rate <= 40000:
        raise ValueError("High-band inspection needs a Nyquist frequency above 20 kHz")
    if data.ndim not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Expected finite mono or multichannel audio")
    width = max(4, round(sample_rate * (1 if band == "low" else .02)))
    bins = np.fft.rfftfreq(width, 1 / sample_rate)
    region = bins <= 100 if band == "low" else bins >= 20000
    freq = bins[region]
    if not len(data):
        return np.empty(0), freq, np.empty((len(freq), 0))
    channels = data.reshape(len(data), -1)
    hop = max(1, width // 2)
    centers = list(range(0, len(data), hop))
    if centers[-1] != len(data) - 1:
        centers.append(len(data) - 1)
    columns = min(max_columns, len(centers))
    magnitudes = np.zeros((len(freq), columns), dtype=np.float64)
    times = np.zeros(columns)
    counts = np.zeros(columns, dtype=int)
    window = hann(width, sym=False)
    scale = np.full(len(bins), 2.)
    scale[0] = 1.
    if width % 2 == 0:
        scale[-1] = 1.
    for i, center in enumerate(centers):
        if cancelled is not None and cancelled():
            raise CancelledError()
        start = center - width // 2
        left, right = max(start, 0), min(start + width, len(data))
        block = channels[left:right]
        # Center a window on each file edge. Only observed samples contribute
        # to coherent gain, including when the whole clip fits inside a window.
        weights = window[left - start:right - start]
        bins_ch = np.fft.rfft(block * weights[:, None], n=width, axis=0)
        peak = np.max(np.abs(bins_ch[region]), axis=1) * scale[region] / weights.sum()
        col = i * columns // len(centers)
        np.maximum(magnitudes[:, col], peak, out=magnitudes[:, col])
        times[col] += center / sample_rate
        counts[col] += 1
    return times / counts, freq, 20 * np.log10(np.maximum(magnitudes, 1e-10))
