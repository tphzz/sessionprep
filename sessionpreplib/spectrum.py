"""Windowed peak-amplitude spectra for the offline waveform preview."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
from functools import lru_cache

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
    # A stationary line satisfies this recurrence throughout the interval.
    # Reject a poor initial model before iterating on ordinary program audio.
    innovation = curvature - coefficient * middle
    if np.var(innovation) > .025 * np.mean(samples ** 2):
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


class _EndpointBand:
    """A decimated baseband measurement at DC or Nyquist, with a known FIR support."""

    def __init__(self, channel, width, high):
        self.channel = channel
        self.high = high
        self.width = width
        self.step = max(1, min(width, len(channel)) // 64)
        self.radius = 8 * self.step
        # Keep the first several native FFT bins flat. The strong stopband
        # rejects unrelated program audio before the endpoint is decimated.
        self.taps = signal.firwin(2 * self.radius + 1, .55 / self.step,
                                  window=("kaiser", 14.))
        kernel = self.taps.copy()
        if high:
            kernel[1::2] *= -1
        filtered = signal.upfirdn(kernel, channel, down=self.step)
        self.samples = filtered[8:8 + (len(channel) + self.step - 1) // self.step]
        if high and self.step % 2:
            self.samples[1::2] *= -1

    def observations(self, start, stop, max_count=None):
        # Use only FIR observations supported wholly by the requested samples.
        # An edit outside this window cannot supply evidence inside it.
        lo = (start + self.radius + self.step - 1) // self.step
        hi = max(lo, (stop - 1 - self.radius) // self.step + 1)
        count = hi - lo
        step = max(1, count // max_count) if max_count else 1
        return self.samples[lo:hi:step], lo * self.step, step * self.step

    def response(self, omega):
        phase = omega * (np.arange(len(self.taps)) - self.radius)
        return np.einsum("i,i->", self.taps, np.cos(phase))

    def local_bias(self, frame, start):
        # Use the whole local support: the FIR observations lose one radius
        # on each side, widening their taper enough to admit a resolved line.
        if self.high:
            pairs = len(frame) // 2
            difference = np.subtract(frame[:2 * pairs:2], frame[1:2 * pairs:2], dtype=np.float64)
            return (.5 if start % 2 == 0 else -.5) * _bias_level(difference)
        return _bias_level(frame)


@dataclass
class _EdgeLine:
    """A fitted line whose actual band-limited samples agree with the current window."""

    start: int
    end: int
    origin: float
    omega: float
    a: float
    b: float
    dc: float
    band: _EndpointBand
    response: float
    residual_limit: float
    rms_limit: float

    def local_components(self, frame, start):
        """Return the real line and its quadrature as a single complex signal."""
        samples, first, step = self.band.observations(start, start + len(frame))
        if len(samples) < 6:
            return None
        phase = self.omega * (np.arange(len(samples)) * step + (first - self.origin))
        expected = self.response * (self.a * np.cos(phase) + self.b * np.sin(phase))
        residual = samples - expected - self.dc
        # Never supply a remote quadrature peak to a locally empty band.
        if (np.max(np.abs(samples)) < .1 * np.max(np.abs(expected)) or
                np.max(np.abs(residual)) > self.residual_limit or
                np.einsum("i,i->", residual, residual) > len(residual) * self.rms_limit ** 2):
            return None
        phase = self.omega * (np.arange(len(frame)) + (start - self.origin))
        cs, sn = np.cos(phase), np.sin(phase)
        real = self.a * cs + self.b * sn
        quadrature = self.a * sn - self.b * cs
        if self.band.high:
            real[1 - start % 2::2] *= -1
            quadrature[start % 2::2] *= -1
        return real + 1j * quadrature


@lru_cache(maxsize=64)
def _bias_weights(count):
    """Unit DC response with strong rejection of other spectral lines."""
    taper = signal.windows.blackmanharris(count)
    return taper / taper.sum()


def _bias_level(samples):
    # Reduce long contexts by integration so their window storage is bounded.
    # Plain subsampling could alias a distant tone to the endpoint.
    stride = max(1, len(samples) // 128)
    if stride > 1:
        count = len(samples) // stride
        samples = samples[:count * stride].reshape(count, stride).mean(axis=1, dtype=np.float64)
    return np.einsum("i,i->", samples, _bias_weights(len(samples)))


def _consistent_bias(samples, level):
    # A 0.5% amplitude difference is about 0.043 dB: only a small part
    # of the display's 0.3 dB calibration allowance. Independent means test
    # the endpoint itself without requiring all other frequencies to be quiet.
    if len(samples) < 12 or level == 0.:
        return False
    half = len(samples) // 2
    return all(abs(_bias_level(part) - level) <= .005 * abs(level)
               for part in (samples[:half], samples[half:]))


def _context_starts(start, stop, span, length):
    origins = ((start + stop - span) // 2, start, stop - span)
    return dict.fromkeys(max(0, min(length - span, lo)) for lo in origins)


@dataclass
class _EndpointBias:
    start: int
    end: int
    level: float
    band: _EndpointBand

    def local_components(self, frame, start):
        local = self.band.local_bias(frame, start)
        # One short window is noisier than the supporting interval. Require
        # its measured level within 0.13 dB and leave the rest in the real FFT.
        if abs(local - self.level) > .015 * abs(self.level):
            return None
        # Report the signed endpoint present in this window. Its DC or
        # alternating component has no separate negative-frequency image.
        real = np.full(len(frame), local)
        if self.band.high:
            real[1 - start % 2::2] *= -1
        return real


def _stationary_bias(band, start, stop, limit):
    frame = band.channel[start:stop]
    samples, _, _ = band.observations(start, stop)
    level = band.local_bias(frame, start)
    if level == 0.:
        return None, None
    # Prefer evidence on both sides, then allow a steady section next to an
    # edit. Never import its amplitude into a window that fails the local test.
    n = stop - start
    span = min(2 * n, limit)
    # Start with neighboring evidence. A stationary endpoint must survive
    # the improved resolution here before testing much longer intervals.
    while span > n:
        agrees = False
        for lo in _context_starts(start, stop, span, len(band.channel)):
            context, _, _ = band.observations(lo, lo + span)
            reference = _bias_level(context)
            if abs(level - reference) > .015 * abs(reference):
                continue
            agrees = True
            if all(_consistent_bias(part, reference) for part in np.array_split(context, 2)):
                bias = _EndpointBias(lo, lo + span, reference, band)
                return bias.local_components(frame, start), bias
        if not agrees or span == limit:
            break
        span = min(span * 2, limit)
    # An isolated section can be shorter than two windows. Its samples must
    # themselves be flat to the same tolerance before it qualifies locally.
    if np.max(np.abs(samples - level)) <= .005 * abs(level):
        bias = _EndpointBias(start, stop, level, band)
        return bias.local_components(frame, start), bias
    return None, None


def _measure_edge_line(band, start, stop, quantum):
    """Resolve a coherent oscillator's swing and complex phase in the selected band."""
    samples, first, step = band.observations(start, stop, max_count=192)
    if len(samples) < 8:
        return None
    scale = np.max(np.abs(samples))
    if scale <= 1e-15:
        return None
    y = samples / scale
    angular = _edge_frequency(y)
    if angular is None:
        return None
    angle = angular * (len(y) - 1)
    # Select precision against the measured band. In a filter stopband a
    # coherent floating-point residual can be far below one input PCM code.
    tolerance = min(quantum, .04 * scale) / scale
    required_angle = np.pi if tolerance > .005 else np.pi / 2
    if angle < required_angle:
        return None
    t = np.linspace(-.5, .5, len(y))
    for _ in range(5):
        cs, sn = np.cos(angle * t), np.sin(angle * t)
        basis = np.column_stack((cs, sn, np.ones(len(y))))
        a, b, dc = np.linalg.lstsq(basis, y, rcond=None)[0]
        residual = y - (a * cs + b * sn + dc)
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
    omega = angle / ((len(y) - 1) * step)
    if (angle < required_angle or amplitude > 1.45 or amplitude < .1 or
            omega * band.width > 8 * np.pi):
        return None
    rms_error = np.sqrt(np.mean(residual ** 2))
    peak_error = np.max(np.abs(residual))
    if (abs(dc) > tolerance + .001 * amplitude or rms_error > .75 * tolerance + .0005 * amplitude or
            peak_error > 2 * tolerance + .002 * amplitude):
        # For a noisier fit, independently verify its amplitude and phase in
        # both halves of a complete oscillation. A fade alone cannot satisfy it.
        if (angle < 2 * np.pi or rms_error > .012 * amplitude or
                peak_error > 6 * rms_error or abs(dc) > 4 * rms_error):
            return None
        for part in (slice(None, len(y) // 2), slice(len(y) // 2, None)):
            aa, bb, _ = np.linalg.lstsq(basis[part], y[part], rcond=None)[0]
            if np.hypot(aa - a, bb - b) > .02 * amplitude:
                return None
    # Reference and local validation use the same band and FIR response.
    # Neither comparisons against raw noise peaks nor a source code step
    # larger than the entire residual determines the displayed level.
    rms_limit = scale * max(1.5 * tolerance, .0001 * amplitude, 3 * rms_error)
    peak_limit = scale * max(2.5 * tolerance, .0005 * amplitude, 8 * rms_error)
    response = band.response(omega)
    origin = first + (len(y) - 1) * step / 2
    return _EdgeLine(start, stop, origin, omega, a * scale / response, b * scale / response,
                     dc * scale, band, response, peak_limit, rms_limit)


def _endpoint_model(band, start, stop, rate, quantum, previous):
    frame = band.channel[start:stop]
    n = len(frame)
    if previous is not None and previous.start <= start and stop <= previous.end:
        components = previous.local_components(frame, start)
        if components is not None:
            return components, previous
    samples, _, _ = band.observations(start, stop)
    if len(samples) < 6:
        # Even a very short exact DC or Nyquist record has a known peak.
        values = frame
        if band.high:
            values = frame.copy()
            values[1 - start % 2::2] *= -1
        if np.all(values == values[0]):
            return frame, None
        return None, None
    peak = np.max(np.abs(samples))
    if peak < 1e-15:
        return None, None
    x = samples / peak
    # Only positive/negative lobes overlapping at a real-spectrum endpoint
    # need a quadrature estimate. Resolved frequencies use the ordinary STFT.
    span = n
    limit = min(len(band.channel), max(n * 4, rate * 8))
    if np.mean(np.diff(x) ** 2) > np.mean(x ** 2) * (8 * np.pi / len(x)) ** 2 + .01:
        return _stationary_bias(band, start, stop, limit)
    while True:
        for lo in _context_starts(start, stop, span, len(band.channel)):
            line = _measure_edge_line(band, lo, lo + span, quantum)
            if line is not None:
                components = line.local_components(frame, start)
                if components is not None:
                    return components, line
        if span == limit:
            return _stationary_bias(band, start, stop, limit)
        span = min(span * 2, limit)


def _window_amplitudes(channel, start, stop, rate, weights, n_fft, quantum, bands, previous, cutoffs):
    residual = channel[start:stop]
    if len(residual) == 1:
        level = abs(residual[0] * weights[0])
        return np.full(n_fft // 2 + 1, level), [level] * len(cutoffs)
    lines = []
    for i, band in enumerate(bands):
        carrier, previous[i] = _endpoint_model(band, start, stop, rate, quantum, previous[i])
        if carrier is not None:
            residual = residual - carrier.real
            lines.append(carrier)
    # Preserve everything other than the fitted lines in the ordinary, real
    # local STFT. Hold the strongest actual component in each frequency cell.
    weighted = residual * weights
    amplitudes = np.abs(fft.rfft(weighted, n=n_fft))
    amplitudes[1:-1] *= 2
    edge_levels = [2 * abs(np.einsum("i,i->", kernel[:len(weights)], weighted)) for kernel in cutoffs]
    for line in lines:
        weighted_line = line * weights
        carrier = fft.fft(weighted_line, n=n_fft)[:len(amplitudes)]
        np.maximum(amplitudes, np.abs(carrier), out=amplitudes)
        for i, kernel in enumerate(cutoffs):
            edge_levels[i] = max(edge_levels[i], abs(np.einsum("i,i->", kernel[:len(weights)], weighted_line)))
    return amplitudes, edge_levels


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
    orientations = []
    if frequencies[0] < 4 * rate / width:
        orientations.append(False)
    if frequencies[-1] > rate / 2 - 4 * rate / width:
        orientations.append(True)
    bands = []
    for ch in channels:
        channel_bands = []
        for high in orientations:
            if cancelled is not None and cancelled():
                raise CancelledError()
            channel_bands.append(_EndpointBand(ch, width, high))
        bands.append(channel_bands)
    previous = [[None] * len(orientations) for _ in channels]
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
            spectrum, boundaries = _window_amplitudes(
                channel, lo, hi, rate, weights, n_fft, quanta[ch], bands[ch], previous[ch], cutoffs)
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
