"""Windowed peak-amplitude spectra for the offline waveform preview."""
from __future__ import annotations

from concurrent.futures import CancelledError

import numpy as np
from scipy import fft, signal

from .audio import validate_analysis_level


def hz_to_mel(frequency):
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _peak_spectrogram(channels, rate, width, hop, window, frequencies, max_columns, cancelled):
    """Measure each window, holding channel, frequency-cell and time-column maxima."""
    frames = len(channels[0])
    if not frames:
        return np.empty(0), frequencies, np.empty((len(frequencies), 0))
    for channel in channels:
        validate_analysis_level(max(-float(channel.min()), float(channel.max())), frames)
    centers = np.arange(0, frames, hop)
    if centers[-1] != frames - 1:
        centers = np.append(centers, frames - 1)
    if len(centers) <= max_columns:
        ncols = len(centers)
        column = np.arange(ncols)
        times = centers / rate
    else:
        # Each returned timestamp is the middle of its exact time cell, also
        # when the last analysis window is a partial hop from the previous one.
        ncols = min(max_columns, max(1, frames // hop))
        column = centers * ncols // frames
        times = (np.arange(ncols) + 0.5) * (frames / (ncols * rate))
    levels = np.full((len(frequencies), ncols), -200., dtype=np.float32)
    taper = signal.get_window(window, width)

    # Four samples across each native FFT interval limit worst-case scalloping
    # to 0.11 dB for every selectable window. Resolution still comes from width.
    n_fft = 4 * width
    hz_per_bin = rate / n_fft
    midpoints = (frequencies[:-1] + frequencies[1:]) / 2
    first = int(np.ceil(frequencies[0] / hz_per_bin))
    last = min(n_fft // 2, int(np.floor(frequencies[-1] / hz_per_bin)))
    edges = np.clip(np.ceil(np.r_[frequencies[0], midpoints] / hz_per_bin).astype(int), first, last)
    # Obtain quadrature from the surrounding audio. Doing the Hilbert transform
    # on individual short FFT windows biases tones near DC and Nyquist whenever
    # a window spans a fractional cycle. Tiles bound working memory on long stems.
    tile_frames = max(1, 4 * rate // hop)
    guard = max(width, rate)
    for offset in range(0, len(centers), tile_frames):
        tile = centers[offset:offset + tile_frames]
        left, right = max(0, tile[0] - guard), min(frames, tile[-1] + guard + 1)
        for channel in channels:
            if cancelled is not None and cancelled():
                raise CancelledError()
            segment = np.asarray(channel[left:right], dtype=np.float64)
            if not segment.any():
                continue
            analytic = signal.hilbert(segment, N=fft.next_fast_len(len(segment)))
            for i, center in enumerate(tile):
                if cancelled is not None and cancelled():
                    raise CancelledError()
                start = center - width // 2
                lo, hi = max(start, 0), min(start + width, frames)
                weights = taper[lo - start:hi - start]
                weighted = analytic[lo - left:hi - left] * weights
                spectrum = fft.fft(weighted, n=n_fft)
                amplitude = np.abs(spectrum[first:last + 1])
                # The self-conjugate real-input endpoints have no quadrature
                # component. Keep other frequencies sharing their display cell.
                if frequencies[0] == 0:
                    amplitude[0] = abs(weighted.real.sum())
                if frequencies[-1] == rate / 2:
                    amplitude[-1] = abs(weighted.real[::2].sum() - weighted.real[1::2].sum())
                cells = np.maximum.reduceat(amplitude, edges - first)
                db = 20 * np.log10(np.maximum(cells / weights.sum(), 1e-10))
                dest = levels[:, column[offset + i]]
                np.maximum(dest, db, out=dest)
    return times, frequencies, levels


def full_spectrogram(channels, rate, *, width, window, frequency_max, rows,
                     hop=None, cancelled=None):
    """Full overview on the mel axis, in the same peak dBFS as focused inspection."""
    if rate <= 0 or width < 4 or rows < 2 or (hop is not None and hop < 1):
        raise ValueError("Invalid full spectrogram dimensions")
    frequencies = mel_to_hz(np.linspace(hz_to_mel(20.), hz_to_mel(frequency_max), rows))
    frequencies[0], frequencies[-1] = 20., frequency_max
    # Include both edges and overlap the windows even at the coarsest request.
    # Hold their peaks in the original overview's time-column budget.
    requested_hop = width if hop is None else hop
    max_columns = 1 + (len(channels[0]) + requested_hop - 1) // requested_hop
    return _peak_spectrogram(channels, rate, width, min(requested_hop, width // 2),
                             window, frequencies, max_columns, cancelled)


def inspection_spectrogram(audio: np.ndarray, sample_rate: int, band: str, *,
                           max_columns: int = 512, cancelled=None):
    """Time, native frequency labels and maximum per-channel sine-peak dBFS.

    Low windows span one second (1 Hz bins). High windows span 20 ms
    (50 Hz bins). Every window contributes, including the file boundaries.
    """
    data = np.asarray(audio)
    if sample_rate <= 0 or max_columns < 1 or band not in ("low", "high"):
        raise ValueError("Invalid inspection band, sample rate or column count")
    if band == "high" and sample_rate <= 40000:
        raise ValueError("High-band inspection needs a Nyquist frequency above 20 kHz")
    if data.ndim not in (1, 2) or (data.ndim == 2 and not data.shape[1]):
        raise ValueError("Expected finite mono or multichannel audio")
    width = max(4, round(sample_rate * (1 if band == "low" else .02)))
    bins = fft.rfftfreq(width, 1 / sample_rate)
    freq = bins[bins <= 100] if band == "low" else bins[bins >= 20000]
    channels = [data] if data.ndim == 1 else data.T
    return _peak_spectrogram(channels, sample_rate, width, width // 2,
                             "hann", freq, max_columns, cancelled)
