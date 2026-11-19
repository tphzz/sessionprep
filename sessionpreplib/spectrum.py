"""Windowed peak-amplitude spectra for the offline waveform preview."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass

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


def _pcm_step(channel):
    """Measure integer sample spacing, including after a floating-point gain."""
    distributed = channel[np.linspace(0, len(channel) - 1, min(len(channel), 513), dtype=int)]
    # Adjacent samples help reveal a single code step in short, quiet material.
    observations = np.r_[distributed, channel[len(channel) // 2:len(channel) // 2 + 256]]
    peak = np.max(np.abs(observations))
    if not peak or np.all(np.abs(observations) == abs(observations[0])):
        return 0.
    if peak <= 1.:
        for bits in (7, 15, 23):
            codes = np.ldexp(observations, bits)
            if np.array_equal(codes, np.rint(codes)):
                return 2. ** -bits
    values = np.unique(np.abs(observations / peak))
    if len(values) < 8:
        return 0.
    spacing = min(np.diff(values))
    # The smallest observed difference can span several actual codes. Accept
    # a candidate only when every observed value fits the same lattice.
    for divisor in range(1, 9):
        step = spacing / divisor
        if step < 2. ** -24:
            break
        codes = values / step
        if np.max(np.abs(codes - np.rint(codes))) <= 1e-5:
            return step * peak
    return 0.


def source_sample_steps(audio, subtype=None):
    """Per-channel rounding steps inherited by a floating-point repair render.

    A linear PCM header gives the exact digital step. For floating-point or
    in-memory originals, recognize an existing sample grid when available.
    The spectrum still measures frequency, phase and level on the selected audio.
    """
    data = np.asarray(audio)
    count = 1 if data.ndim == 1 else data.shape[1]
    bits = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(subtype)
    if bits is not None:
        return (2. ** (1 - bits),) * count
    if not data.size:
        return (0.,) * count
    return tuple(_pcm_step(channel) for channel in data.reshape(len(data), count).T)


@dataclass
class _EdgeLine:
    """A resolved oscillator on a measured time interval, in absolute sample phase."""

    start: int
    end: int
    origin: float
    omega: float
    a: float
    b: float
    high: bool
    residual_limit: float
    rms_limit: float = np.inf

    def local_components(self, frame, start):
        phase = self.omega * (np.arange(len(frame)) + (start - self.origin))
        cs, sn = np.cos(phase), np.sin(phase)
        real = self.a * cs + self.b * sn
        if self.high:
            real[1 - start % 2::2] *= -1
        residual = frame - real
        # The reference supplies quadrature only when its real samples agree
        # with this entire window, allowing for integer-code rounding.
        if np.max(np.abs(residual)) > self.residual_limit:
            return None
        if np.isfinite(self.rms_limit) and np.mean(residual ** 2) > self.rms_limit ** 2:
            return None
        quadrature = self.a * sn - self.b * cs
        if self.high:
            quadrature[start % 2::2] *= -1
        return real + 1j * quadrature, residual


def _measure_edge_line(channel, start, stop, high, quantum):
    """Fit frequency, phase and amplitude to observations spanning a real swing."""
    # Average adjacent samples after folding. Paired samples cancel the DC
    # bias of integer PCM near Nyquist, even when the observation stride is even.
    stride = max(1, (stop - start - 1) // 192)
    indices = np.arange(start, stop - 1, stride)
    if len(indices) < 8:
        return None
    first = np.asarray(channel[indices], dtype=np.float64)
    second = np.asarray(channel[indices + 1], dtype=np.float64)
    if high:
        y = (first - second) * .5
        y[indices % 2 != 0] *= -1
    else:
        y = (first + second) * .5
    scale = np.max(np.abs(y))
    if not scale or scale < 4 * quantum:
        return None
    y /= scale
    angular = _edge_frequency(y)
    if angular is None:
        return None
    # Fit over a meaningful fraction of a cycle. A short linear fade alone
    # cannot establish a sinusoid's amplitude on the other side of a zero.
    angle = angular * (len(y) - 1)
    # Coarse samples need at least half a cycle to distinguish the peak
    # amplitude from quantization bias on the curved flank of a wave.
    tolerance = quantum / scale
    required_angle = np.pi if tolerance > .005 else np.pi / 2
    if angle < required_angle:
        return None
    t = np.linspace(-.5, .5, len(y))
    for _ in range(5):
        cs, sn = np.cos(angle * t), np.sin(angle * t)
        basis = np.column_stack((cs, sn, np.ones(len(y))))
        a, b, dc = np.linalg.lstsq(basis, y, rcond=None)[0]
        residual = y - (a * cs + b * sn + dc)
        # Refine the recurrence estimate by projecting the phase derivative
        # alongside the linear coefficients (frequency is the only nonlinear parameter).
        derivative = (b * cs - a * sn) * t
        delta = np.linalg.lstsq(np.column_stack((basis, derivative)), residual, rcond=None)[0][-1]
        if abs(delta) < 1e-9:
            break
        if abs(delta) > angle / 3:
            return None
        angle += delta
    else:
        return None
    amplitude = np.hypot(a, b)
    # Rounding, triangular dither or an earlier gain change can leave a
    # small residual around a coherent line. Require a low RMS error and
    # independently bound the largest deviation before supplying quadrature.
    rms_error = np.sqrt(np.mean(residual ** 2))
    if angle < required_angle or amplitude > 1.45:
        return None
    peak_error = np.max(np.abs(residual))
    rms_limit = np.inf
    residual_limit = max(2.5 * quantum, 2e-4 * amplitude * scale, 5 * rms_error * scale)
    if (abs(dc) > tolerance + .001 * amplitude or rms_error > .75 * tolerance + .0005 * amplitude or
            peak_error > 2 * tolerance + .002 * amplitude):
        # A float recording may have no grid and still contain a coherent
        # oscillator plus light noise. Before using its measured residual,
        # require a whole cycle with the same amplitude and phase on both halves.
        if (quantum or angle < 2 * np.pi or rms_error > .012 * amplitude or
                peak_error > 6 * rms_error or abs(dc) > 4 * rms_error):
            return None
        for part in (slice(None, len(y) // 2), slice(len(y) // 2, None)):
            aa, bb, _ = np.linalg.lstsq(basis[part], y[part], rcond=None)[0]
            if np.hypot(aa - a, bb - b) > .02 * amplitude:
                return None
        # The paired reference samples average two noise contributions. The
        # selected raw window must agree within both peak and RMS bounds.
        rms_limit = scale * (abs(dc) + 2.5 * rms_error)
        residual_limit = scale * (abs(dc) + 9 * rms_error)
    omega = angle / (indices[-1] - indices[0])
    # Undo the adjacent-pair average's cosine response. Its timestamp is
    # halfway between the two samples.
    gain = scale / np.cos(omega / 2)
    origin = (indices[0] + indices[-1] + 1) / 2
    return _EdgeLine(start, stop, origin, omega, a * gain, b * gain, high, residual_limit, rms_limit)


def _endpoint_model(channel, start, stop, rate, quantum, previous):
    frame = np.asarray(channel[start:stop], dtype=np.float64)
    n = len(frame)
    if previous is not None and previous.start <= start and stop <= previous.end:
        components = previous.local_components(frame, start)
        if components is not None:
            return components, previous
    scale = np.max(np.abs(frame))
    x = frame / max(scale, quantum)
    high = np.einsum("i,i->", x[1:], x[:-1]) < 0
    if high:
        x[1::2] *= -1
    constant = np.ptp(x) < 1e-10
    if constant and (not quantum or n < 12):
        return (frame, np.zeros_like(frame)), None
    if n < 12:
        return None, None
    variation = np.diff(x)
    energy = np.einsum("i,i->", x, x)
    # Only unresolved real-spectrum images at DC or Nyquist need separation.
    # Finite precision contributes to the adjacent difference on both samples.
    noise = quantum / max(quantum, scale)
    expected_variation = energy * (8 * np.pi / n) ** 2 + n * (2 * noise) ** 2
    if scale > 4 * quantum and np.einsum("i,i->", variation, variation) > expected_variation:
        # Near a beat null, broadband noise can dominate single-sample
        # differences. Test its block-averaged trend before rejecting the fit.
        stride = max(1, n // 48)
        if quantum or stride == 1:
            return None, None
        groups = x[:(n // stride) * stride].reshape(-1, stride).mean(axis=1)
        if np.mean(np.diff(groups) ** 2) > np.mean(groups ** 2) * (8 * np.pi / len(groups)) ** 2:
            return None, None
    # Very low-level quantized samples can alternate between zero and a
    # single code; adjacent products then cannot distinguish the two endpoints.
    orientations = (high, not high) if scale <= 4 * quantum else (high,)
    # Begin with the selected window. A filter transient elsewhere in a
    # short recording must not invalidate an already settled local oscillator.
    span = n
    limit = min(len(channel), max(n * 4, rate * 8))
    while True:
        origins = ((start + stop - span) // 2, start, stop - span)
        origins = dict.fromkeys(max(0, min(len(channel) - span, lo)) for lo in origins)
        for lo in origins:
            for orientation in orientations:
                line = _measure_edge_line(channel, lo, lo + span, orientation, quantum)
                if line is None:
                    continue
                components = line.local_components(frame, start)
                if components is not None:
                    return components, line
        if span == limit:
            break
        span = min(span * 2, limit)
    if constant:
        return (frame, np.zeros_like(frame)), None
    return None, None


def _window_amplitudes(channel, start, stop, rate, weights, n_fft, quantum, previous, cutoffs):
    model, line = _endpoint_model(channel, start, stop, rate, quantum, previous)
    residual = channel[start:stop] if model is None else model[1]
    # A real windowed Fourier transform keeps edits, noise and multiple
    # components inside the nominated time window. DC and Nyquist have no
    # negative-frequency counterpart. Resolved endpoint tones contribute
    # their isolated positive-frequency lobe, without doubling its image.
    weighted = residual * weights
    amplitudes = np.abs(fft.rfft(weighted, n=n_fft))
    amplitudes[1:-1] *= 2
    # An off-grid cutoff is always between DC and Nyquist. Measure its actual
    # Fourier coefficient with the same window and one-sided normalization.
    edge_levels = [2 * abs(np.einsum("i,i->", kernel[:len(weights)], weighted)) for kernel in cutoffs]
    if model is not None:
        # Report the strongest measured component. Adding the quantizer's
        # DC bias coherently to the fitted line would modulate its readout.
        weighted_line = model[0] * weights
        carrier = fft.fft(weighted_line, n=n_fft)[:len(amplitudes)]
        np.maximum(amplitudes, np.abs(carrier), out=amplitudes)
        for i, kernel in enumerate(cutoffs):
            edge_levels[i] = max(edge_levels[i], abs(np.einsum("i,i->", kernel[:len(weights)], weighted_line)))
    return amplitudes, edge_levels, line


def _peak_spectrogram(channels, rate, width, hop, window, frequencies, max_columns, cancelled, source_steps):
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
    # A physical band edge can fall between FFT samples, particularly at
    # pulled sample rates or with a short Full window at a high sample rate.
    limits = frequencies[[0, -1]] * n_fft / rate
    nearest = np.rint(limits)
    off_grid = np.abs(limits - nearest) > 8 * np.spacing(np.maximum(1., limits))
    lower, upper = np.where(off_grid, limits, nearest)
    first, last = int(np.ceil(lower)), int(np.floor(upper))
    extra_bins = limits[off_grid]
    lower_count = int(off_grid[0])
    sample_bins = np.concatenate((extra_bins[:lower_count], np.arange(first, last + 1),
                                  extra_bins[lower_count:]))
    cutoffs = [np.exp(-2j * np.pi * pos * np.arange(width) / n_fft) for pos in extra_bins]
    row_bins = frequencies * n_fft / rate
    midpoints = (row_bins[:-1] + row_bins[1:]) / 2
    edges = np.r_[0, np.searchsorted(sample_bins, midpoints), len(sample_bins)]
    # More display rows than Fourier samples is common at the low end of
    # Full. Extend the nearest measured level into otherwise empty cells.
    empty_rows = np.flatnonzero(np.diff(edges) == 0)
    right = np.clip(edges[empty_rows], 1, len(sample_bins) - 1)
    nearest_empty = np.where(row_bins[empty_rows] * 2 <= sample_bins[right - 1] + sample_bins[right],
                             right - 1, right)
    quanta = [_pcm_step(channel) for channel in channels] if source_steps is None else source_steps
    if len(quanta) != len(channels) or not all(np.isfinite(q) and q >= 0 for q in quanta):
        raise ValueError("Expected one finite, nonnegative sample step per channel")
    previous = [None] * len(channels)
    for i, center in enumerate(centers):
        if cancelled is not None and cancelled():
            raise CancelledError()
        start = center - width // 2
        lo, hi = max(start, 0), min(start + width, frames)
        weights = taper[lo - start:hi - start]
        for ch, channel in enumerate(channels):
            frame = np.asarray(channel[lo:hi], dtype=np.float64)
            if not frame.any():
                continue
            spectrum, boundaries, previous[ch] = _window_amplitudes(
                channel, lo, hi, rate, weights, n_fft, quanta[ch], previous[ch], cutoffs)
            amplitude = spectrum[first:last + 1]
            if cutoffs:
                amplitude = np.concatenate((boundaries[:lower_count], amplitude, boundaries[lower_count:]))
            cells = np.maximum.reduceat(amplitude, edges[:-1])
            cells[empty_rows] = amplitude[nearest_empty]
            db = 20 * np.log10(np.maximum(cells / weights.sum(), 1e-10))
            dest = levels[:, column[i]]
            np.maximum(dest, db, out=dest)
    return times, frequencies, levels


def full_spectrogram(channels, rate, *, width, window, frequency_max, rows,
                     hop=None, cancelled=None, source_steps=None):
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
                             window, frequencies, max_columns, cancelled, source_steps)


def inspection_spectrogram(audio: np.ndarray, sample_rate: int, band: str, *,
                           max_columns: int = 512, cancelled=None, source_steps=None):
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
    bins = np.arange(width // 2 + 1) * float(sample_rate) / width
    lower, upper = (0., min(100., sample_rate / 2)) if band == "low" else (20000., sample_rate / 2)
    freq = np.r_[lower, bins[(bins > lower) & (bins < upper)], upper]
    channels = [data] if data.ndim == 1 else data.T
    return _peak_spectrogram(channels, sample_rate, width, width // 2,
                             "hann", freq, max_columns, cancelled, source_steps)
