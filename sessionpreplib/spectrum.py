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
    if (angle < required_angle or amplitude > 1.45 or abs(dc) > tolerance + .001 * amplitude or
            rms_error > .75 * tolerance + .0005 * amplitude or
            np.max(np.abs(residual)) > 2 * tolerance + .002 * amplitude):
        return None
    omega = angle / (indices[-1] - indices[0])
    # Undo the adjacent-pair average's cosine response. Its timestamp is
    # halfway between the two samples.
    gain = scale / np.cos(omega / 2)
    origin = (indices[0] + indices[-1] + 1) / 2
    residual_limit = max(2.5 * quantum, 2e-4 * amplitude * scale, 5 * rms_error * scale)
    return _EdgeLine(start, stop, origin, omega, a * gain, b * gain, high, residual_limit)


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
        return None, None
    # Very low-level quantized samples can alternate between zero and a
    # single code; adjacent products then cannot distinguish the two endpoints.
    orientations = (high, not high) if scale <= 4 * quantum else (high,)
    span = min(len(channel), n * 4)
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


def _window_amplitudes(channel, start, stop, rate, weights, n_fft, quantum, previous):
    model, line = _endpoint_model(channel, start, stop, rate, quantum, previous)
    residual = channel[start:stop] if model is None else model[1]
    # A real windowed Fourier transform keeps edits, noise and multiple
    # components inside the nominated time window. DC and Nyquist have no
    # negative-frequency counterpart. Resolved endpoint tones contribute
    # their isolated positive-frequency lobe, without doubling its image.
    amplitudes = np.abs(fft.rfft(residual * weights, n=n_fft))
    amplitudes[1:-1] *= 2
    if model is not None:
        # Report the strongest measured component. Adding the quantizer's
        # DC bias coherently to the fitted line would modulate its readout.
        carrier = fft.fft(model[0] * weights, n=n_fft)[:len(amplitudes)]
        np.maximum(amplitudes, np.abs(carrier), out=amplitudes)
    return amplitudes, line


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
    quanta = [_pcm_step(channel) for channel in channels]
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
            spectrum, previous[ch] = _window_amplitudes(
                channel, lo, hi, rate, weights, n_fft, quanta[ch], previous[ch])
            amplitude = spectrum[first:last + 1]
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
