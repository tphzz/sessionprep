"""Whole-track corrective processing at unity gain, before normalization."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class RepairRecipe:
    """The resolved corrections for one routed source, in signal order."""

    dc_remove: bool = False
    low_cut_hz: float | None = None
    ultrasonic: bool = False

    def __post_init__(self):
        if not isinstance(self.dc_remove, bool) or not isinstance(self.ultrasonic, bool):
            raise ValueError("DC and ultrasonic selections must be true or false")
        if self.low_cut_hz is not None:
            if not isinstance(self.low_cut_hz, Real) or not 5 <= self.low_cut_hz <= 40:
                raise ValueError("Low cut must be between 5 and 40 Hz")

    @property
    def active(self) -> bool:
        return self.dc_remove or self.low_cut_hz is not None or self.ultrasonic

    def applied_ids(self) -> list[str]:
        return [name for name, selected in (
            ("dc_remove", self.dc_remove), ("low_cut", self.low_cut_hz is not None),
            ("ultrasonic", self.ultrasonic)) if selected]


@lru_cache(maxsize=16)
def _ultrasonic_taps(sample_rate: int) -> np.ndarray:
    # 20–24 kHz transition. The extra design margin keeps the whole stopband
    # below -80 dB, including its first sidelobe.
    count, beta = signal.kaiserord(86, 4000 / (sample_rate / 2))
    return signal.firwin(count | 1, 22000, fs=sample_rate, window=("kaiser", beta))


def render_corrected(audio: np.ndarray, sample_rate: int, recipe: RepairRecipe) -> np.ndarray:
    """Return corrected samples at their original time origin and length.

    Bypass may share the source storage. Callers must treat both as read-only.
    The causal high-pass starts with zero history. Centering the odd FIR
    compensates its delay and assumes silence outside the file.
    """
    data = np.asarray(audio)
    if sample_rate <= 0 or not math.isfinite(sample_rate):
        raise ValueError("Sample rate must be positive and finite")
    if data.ndim not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Expected finite mono or multichannel audio")
    if recipe.ultrasonic and sample_rate < 50000:
        raise ValueError("Ultrasonic filtering requires a sample rate of at least 50 kHz")
    cutoff = recipe.low_cut_hz or 2.0
    if (recipe.low_cut_hz is not None or recipe.dc_remove) and cutoff >= sample_rate / 2:
        raise ValueError("High-pass cutoff must be below Nyquist")
    if not recipe.active or not data.size:
        return data

    result = np.asarray(data, dtype=np.float64)
    if recipe.low_cut_hz is not None or recipe.dc_remove:
        order = 2 if recipe.low_cut_hz is not None else 1
        sos = signal.butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
        result = signal.sosfilt(sos, result, axis=0)
    if recipe.ultrasonic:
        taps = _ultrasonic_taps(sample_rate)
        delay = (len(taps) - 1) // 2
        full = signal.upfirdn(taps, result, axis=0)
        result = full[delay:delay + len(result)]
    return result


class OutputHeadroomError(ValueError):
    def __init__(self, peak_db: float, reduction_db: float, subtype: str):
        self.peak_db = peak_db
        self.reduction_db = reduction_db
        super().__init__(f"Output peak {peak_db:+.2f} dBFS exceeds {subtype}. "
                         f"Reduce the output gain by at least {reduction_db:.1f} dB and retry.")


def check_output_headroom(audio: np.ndarray, subtype: str) -> None:
    """Check finite floating-point ranges and normalized integer encoder inputs."""
    if not np.isfinite(audio).all():
        raise ValueError("Output contains non-finite samples")
    if not audio.size or subtype == "DOUBLE":
        return
    if subtype == "FLOAT":
        negative_range = positive_range = float(np.finfo(np.float32).max)
    else:
        # Companded WAV/AIFF codecs such as G.711 and ADPCM encode normalized
        # samples through signed integer intermediates. Overload can wrap.
        bits = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
                "DWVW_12": 12, "DWVW_16": 16, "DWVW_24": 24}
        negative_range = 1.0
        positive_range = 1.0 - 2.0 ** (1 - bits.get(subtype, 16))
    low, high = float(np.min(audio)), float(np.max(audio))
    if low >= -negative_range and high <= positive_range:
        return
    # Subtract logs to keep a finite gain recommendation even at float64 max.
    positive_over = math.log10(high) - math.log10(positive_range) if high > positive_range else 0.0
    negative_over = math.log10(-low) - math.log10(negative_range) if low < -negative_range else 0.0
    reduction = max(0.1, math.ceil(200 * max(positive_over, negative_over)) / 10)
    peak_db = 20 * math.log10(max(-low, high))
    raise OutputHeadroomError(peak_db, reduction, subtype)


def band_evidence(audio, sample_rate, *, low_hz, high_hz, window_sec,
                  level_db, ratio_db, consecutive):
    """Measure sustained AC band energy per channel, with absolute and relative gates.

    Report inclusive frame spans. A run belongs to one channel; alternating
    bursts on different channels cannot satisfy the persistence requirement.
    """
    data = np.asarray(audio)
    channels = data.reshape(len(data), -1) if data.size else None
    width = max(2, round(window_sec * sample_rate))
    hop = max(1, width // 2)
    freqs = np.fft.rfftfreq(width, 1 / sample_rate)
    band = (freqs > 0) & (freqs >= low_hz) & (freqs <= high_hz)
    spans = []
    peak_level = peak_ratio = -math.inf
    if channels is None or len(data) < width or not band.any():
        return {"regions": spans, "band_dbfs": peak_level, "ratio_db": peak_ratio}
    window = signal.windows.hann(width, sym=False)
    weights = np.full(len(freqs), 2.0 / (width * np.dot(window, window)))
    weights[0] = 0
    if width % 2 == 0:
        weights[-1] *= 0.5
    for ch, channel in enumerate(channels.T):
        count = 0
        for start in range(0, len(channel) - width + 1, hop):
            block = channel[start:start + width]
            spectrum = np.fft.rfft((block - np.mean(block, dtype=np.float64)) * window)
            powers = np.abs(spectrum) ** 2 * weights
            energy, total = float(powers[band].sum()), float(powers.sum())
            level = 10 * math.log10(energy) if energy > 0 else -math.inf
            ratio = 10 * math.log10(energy / total) if energy > 0 and total > 0 else -math.inf
            if level > peak_level:
                peak_level, peak_ratio = level, ratio
            if level < level_db or ratio < ratio_db:
                count = 0
                continue
            count += 1
            end = start + width - 1
            if count == consecutive:
                spans.append([start - (count - 1) * hop, end, ch])
            elif count > consecutive:
                spans[-1][1] = end
    return {"regions": spans, "band_dbfs": peak_level, "ratio_db": peak_ratio}
