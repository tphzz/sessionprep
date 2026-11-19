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


def _edge_frequency(samples):
    """Angular frequency of a coherent real oscillator, in radians per sample."""
    crossings = np.count_nonzero(np.diff(np.signbit(samples)))
    lag = max(1, (len(samples) - 1) // max(3, crossings + 2))
    middle = samples[lag:-lag]
    curvature = 2 * middle - samples[:-2 * lag] - samples[2 * lag:]
    denominator = middle @ middle
    coefficient = (curvature @ middle) / denominator if denominator else 0.
    # 2y(t)-y(t-L)-y(t+L) = 4 sin²(omega L/2) y(t). Zero crossings
    # keep L below half a period. A long lag reduces quantization bias.
    if not 1e-14 < coefficient <= 4.:
        return None
    return 2 * np.arcsin(.5 * np.sqrt(coefficient)) / lag


def _fit_sinusoid(values, omega):
    phase = omega * (np.arange(len(values)) - (len(values) - 1) / 2)
    cs, sn = np.cos(phase), np.sin(phase)
    # Centering makes the even cosine and odd sine bases orthogonal.
    a = np.einsum("i,i->", values, cs) / np.einsum("i,i->", cs, cs)
    b = np.einsum("i,i->", values, sn) / np.einsum("i,i->", sn, sn)
    residual = values - (a * cs + b * sn)
    if np.einsum("i,i->", residual, residual) > 1e-8 * np.einsum("i,i->", values, values):
        return None
    return np.hypot(a, b), a * sn - b * cs, residual


def _corroborated_edge(channel, start, stop, high, omega, rate):
    """Confirm a sub-cycle frequency on adjacent audio that exhibits its swing.

    A short ramp can fit the tip of an arbitrarily large hypothetical sine.
    Require measured support for the line, in either time direction so that
    an edit on one side doesn't invalidate the other. Only frequency and a
    consistency bound cross the window: no outside samples supply its level.
    """
    n = stop - start
    span = max(4 * n, int(np.ceil(np.pi / omega)) if omega else 4 * rate)
    span = min(len(channel), span)
    center = (start + stop) // 2
    # Limit work to 96 observations even when the beat period is many seconds.
    step = max(1, span // 96)
    count = min(96, span // step)
    if count < 8:
        return None
    tried = set()
    for origin in (center - span // 2, stop - span, start):
        origin = min(len(channel) - span, max(0, origin))
        if origin in tried:
            continue
        tried.add(origin)
        indices = origin + np.arange(count) * step
        values = np.asarray(channel[indices], dtype=np.float64)
        if high:
            values[(indices - start) % 2 != 0] *= -1
        peak = np.max(np.abs(values))
        if not peak:
            continue
        values /= peak
        angular = _edge_frequency(values)
        if angular is None:
            continue
        fit = _fit_sinusoid(values, angular)
        # A quarter-cycle observation reaches at least 1/sqrt(2) of a
        # sinusoid's peak at some point, including around a beat null.
        if fit is not None and fit[0] <= 1.43:
            return angular / step, fit[0] * peak
    return None


def _endpoint_model(channel, start, stop, rate):
    """Return a supported near-endpoint analytic line and its real local residual."""
    frame = np.asarray(channel[start:stop], dtype=np.float64)
    n = len(frame)
    scale = np.max(np.abs(frame))
    x = frame / scale
    high = np.einsum("i,i->", x[1:], x[:-1]) < 0
    if high:
        x[1::2] *= -1
    if np.ptp(x) < 1e-10:
        return frame, np.zeros_like(frame)
    if n < 12:
        return None
    variation = np.diff(x)
    energy = np.einsum("i,i->", x, x)
    # Only unresolved positive/negative lobes at DC or Nyquist need a fit.
    if np.einsum("i,i->", variation, variation) > energy * (8 * np.pi / n) ** 2:
        return None
    # Equal block averages share the oscillator frequency and suppress
    # rounding noise in curvature estimates for FLOAT and integer PCM.
    stride = max(1, n // 48)
    count = n // stride
    means = x[:count * stride].reshape(count, stride).mean(axis=1)
    angular = _edge_frequency(means)
    omega = angular / stride if angular is not None else None
    reference_peak = None
    if omega is None or omega * n < np.pi / 2:
        line = _corroborated_edge(channel, start, stop, high, omega, rate)
        if line is not None:
            omega, reference_peak = line
    if omega is None:
        return None
    fit = _fit_sinusoid(x, omega)
    if fit is None:
        return None
    amplitude, quadrature, residual = fit
    if reference_peak is not None:
        if abs(amplitude * scale - reference_peak) > .03 * reference_peak:
            return None
    elif omega * n < np.pi / 2 and amplitude > 1.035:
        # An uncorroborated sub-cycle fit must not invent an unseen peak.
        return None
    if high:
        # Mirroring the baseband frequency at Nyquist conjugates quadrature.
        residual[1::2] *= -1
        quadrature[::2] *= -1
    residual *= scale
    return (frame - residual) + 1j * (scale * quadrature), residual


def _window_spectrum(channel, start, stop, rate, weights, n_fft):
    model = _endpoint_model(channel, start, stop, rate)
    residual = channel[start:stop] if model is None else model[1]
    # A real windowed Fourier transform keeps edits, noise and multiple
    # components inside the nominated time window. DC and Nyquist have no
    # negative-frequency counterpart. Resolved endpoint tones contribute
    # their isolated positive-frequency lobe, without doubling its image.
    spectrum = fft.rfft(residual * weights, n=n_fft)
    spectrum[1:-1] *= 2
    if model is not None:
        spectrum += fft.fft(model[0] * weights, n=n_fft)[:len(spectrum)]
    return spectrum


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
    for i, center in enumerate(centers):
        if cancelled is not None and cancelled():
            raise CancelledError()
        start = center - width // 2
        lo, hi = max(start, 0), min(start + width, frames)
        weights = taper[lo - start:hi - start]
        for channel in channels:
            frame = np.asarray(channel[lo:hi], dtype=np.float64)
            if not frame.any():
                continue
            spectrum = _window_spectrum(channel, lo, hi, rate, weights, n_fft)
            amplitude = np.abs(spectrum[first:last + 1])
            cells = np.maximum.reduceat(amplitude, edges - first)
            db = 20 * np.log10(np.maximum(cells / weights.sum(), 1e-10))
            dest = levels[:, column[i]]
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
