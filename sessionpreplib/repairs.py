"""Whole-track corrective processing at unity gain, before normalization."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache, partial
from hashlib import sha256
import math
from numbers import Real
from threading import Lock

import numpy as np
from scipy import optimize, signal


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


def _apply_highpass(data, sos):
    """Carry the IIR history across blocks and retire inaudibly small delay states."""
    state = np.zeros((len(sos), 2, *data.shape[1:]), dtype=np.float64)
    output = np.empty_like(data)
    for start in range(0, len(data), 4096):
        block = slice(start, start + 4096)
        output[block], state = signal.sosfilt(sos, data[block], axis=0, zi=state)
        # -2000 dBFS leaves the representable tail well clear of the subnormal
        # range between checks at audio sample rates, independently per channel.
        state[np.abs(state) < 1e-100] = 0.0
    return output


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
        result = _apply_highpass(result, sos)
    if recipe.ultrasonic:
        taps = _ultrasonic_taps(sample_rate)
        delay = (len(taps) - 1) // 2
        full = signal.upfirdn(taps, result, axis=0)
        result = full[delay:delay + len(result)]
    if not np.isfinite(result).all():
        raise ValueError("Corrected audio exceeds the finite sample range. Lower the source level.")
    result[np.abs(result) < np.finfo(result.dtype).tiny] = 0.0
    return result


class OutputHeadroomError(ValueError):
    def __init__(self, peak_db: float, reduction_db: float, subtype: str, *,
                 excess_db: float | None = None, adjustment: dict[str, float] | None = None):
        self.peak_db = peak_db
        self.reduction_db = reduction_db
        self.excess_db = reduction_db if excess_db is None else excess_db
        self.adjustment = adjustment
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
    # Check the nearest tenth in sample units. A log round-trip can turn
    # exactly 0.1 dB into 0.1000000000000001 and recommend an extra step.
    exact_tenths = 200 * max(positive_over, negative_over)
    tenths = round(exact_tenths)
    candidate_db = tenths / 10
    if tenths < 60000:
        # Use the same dB-to-linear conversion as the actual gain stage.
        factor = 10 ** (candidate_db / 20)
        fits = high <= positive_range * factor and -low <= negative_range * factor
    else:
        attenuation = 10 ** (-candidate_db / 20)
        fits = high * attenuation <= positive_range and -low * attenuation <= negative_range
    reduction = (tenths if fits else tenths + 1) / 10
    peak_db = 20 * math.log10(max(-low, high))
    raise OutputHeadroomError(peak_db, reduction, subtype, excess_db=exact_tenths / 10)


class _BoundaryPartial:
    """Resolve one stationary partial in the Hann main lobe at a decision edge."""

    def __init__(self, cycles, width, window_dc):
        self.width = width
        self.edge = float(cycles)
        self.bins = np.arange(max(0, math.ceil(cycles - 3)),
                              min(len(window_dc), math.floor(cycles + 3) + 1))
        self.extended_bins = np.arange(self.bins[0] - 1, self.bins[-1] + 2)
        self.window_dc = window_dc[self.bins]
        self.dc_axis = None
        if self.bins[0] <= 1:
            # A separate non-coherent program tone can contribute to the
            # frame mean. Fit the bass partial independently of that constant.
            dc = np.concatenate((self.window_dc.real, self.window_dc.imag))
            self.dc_axis = dc / np.linalg.norm(dc)
        self.background = None
        if self.bins[0] > 1 and self.bins[-1] < len(window_dc) - 1:
            # Distant Hann sidelobes vary smoothly across this narrow lobe.
            # Fit the same quadratic background to the measured coefficients
            # and sinusoidal basis so several unrelated partials can coexist.
            positions = np.linspace(-1, 1, len(self.bins))
            columns = np.polynomial.polynomial.polyvander(positions, 2)
            self.background = np.eye(len(self.bins)) - columns @ np.linalg.pinv(columns)
        self.grid = np.unique(np.clip(self.edge + np.arange(-2, 2.25, 0.5),
                                      1e-8, self.width / 2))
        self.bases = self._basis(self.grid)
        self.inverses = np.linalg.pinv(self.bases)
        self._trial_bases = {}

    def _rect(self, offsets):
        # Exact normalized finite geometric sum, periodic in DFT-bin units.
        # Wrapping also covers the negative-frequency image near Nyquist.
        d = np.remainder(offsets + self.width / 2, self.width) - self.width / 2
        return np.exp(1j * np.pi * d * (1 - 1 / self.width)) * np.sinc(d) / np.sinc(d / self.width)

    def _basis(self, f):
        # The measured frame has its arithmetic mean removed before windowing.
        # Fit both images of a real sinusoid, including the windowed mean.
        cycles = np.stack((f, -f), axis=-1)[..., None]
        rect = self._rect(cycles - self.extended_bins)
        tones = 0.5 * rect[..., 1:-1] - 0.25 * (rect[..., :-2] + rect[..., 2:])
        tones -= self._rect(cycles) * self.window_dc
        cosine = (tones[..., 0, :] + tones[..., 1, :]) / 2
        sine = (tones[..., 0, :] - tones[..., 1, :]) / (2j)
        basis = np.stack((cosine, sine), axis=-1)
        if self.background is not None:
            basis = self.background @ basis
        basis = np.concatenate((basis.real, basis.imag), axis=-2)
        if self.dc_axis is not None:
            basis -= self.dc_axis[:, None] * (self.dc_axis @ basis)[..., None, :]
        return basis

    def observe(self, spectrum):
        local = spectrum[self.bins] / self.width
        if self.background is not None:
            local = self.background @ local
        vector = np.concatenate((local.real, local.imag))
        if self.dc_axis is not None:
            vector -= self.dc_axis * (self.dc_axis @ vector)
        return vector

    def fit(self, vector, noise_power):
        power = vector @ vector
        if not power or len(self.bins) < 3:
            return None
        coeffs = self.inverses @ vector
        residuals = (self.bases @ coeffs[..., None])[..., 0] - vector
        errors = np.sum(residuals ** 2, axis=1)
        best = int(errors.argmin())
        if errors[best] > 0.25 * power:
            return None
        centre = self.grid[best]
        # The residual of a real line includes the source's spectral noise
        # floor. Cap its allowance so a noisy or blended lobe cannot qualify.
        tolerance = max(1e-12 * power, min(1e-5 * power, 12 * len(self.bins) * noise_power))
        if errors[best] <= 1e-12 * power or (centre == self.edge and errors[best] <= tolerance):
            return centre, coeffs[best], True

        def fit_at(offset):
            frequency = centre + offset
            basis = self._trial_bases.get(frequency)
            if basis is None:
                basis = self._basis(frequency)
                # The bounded solver revisits the same trial frequencies in
                # nearby audio frames. Keep a small, window-local working set.
                if len(self._trial_bases) >= 96:
                    self._trial_bases.clear()
                self._trial_bases[frequency] = basis
            cs = np.linalg.lstsq(basis, vector, rcond=None)[0]
            residual = vector - basis @ cs
            return residual @ residual, cs

        lower = self.grid[max(0, best - 1)] - centre
        upper = self.grid[min(len(self.grid) - 1, best + 1)] - centre
        optimum = optimize.minimize_scalar(lambda offset: fit_at(offset)[0], method="bounded",
                                          bounds=(lower, upper), options={"xatol": 1e-10})
        error, cs = fit_at(optimum.x)
        if error > 0.02 * power:
            return None
        return centre + optimum.x, cs, error <= tolerance


class _BandWindow:
    """One calibrated Hann window reused for every channel and hop."""

    def __init__(self, width, rate, low, high):
        self.low, self.high = low, high
        self.bin_hz = rate / width
        self.nyquist = rate / 2
        self.freqs = np.fft.rfftfreq(width, 1 / rate)
        self.phase = 2 * np.pi * np.arange(width) / width
        self.window = signal.windows.hann(width, sym=False)
        self.window_energy = np.sum(self.window ** 2)
        window_dc = np.fft.rfft(self.window) / width
        # Adjacent hops and channels often resolve the same neighboring FFT bins.
        # Only the basis depends on the bin. Its fit still uses the current audio.
        self._partial = lru_cache(maxsize=64)(partial(_BoundaryPartial, width=width, window_dc=window_dc))
        self.weights = np.full(len(self.freqs), 2.0 / (width * self.window_energy))
        # The signal's DC was removed before windowing. Tapering a real AC
        # component can put legitimate energy in the DC bin (and at Nyquist).
        self.weights[0] *= 0.5
        if width % 2 == 0:
            self.weights[-1] *= 0.5
        self.edges = [(hz, side, self._partial(hz / self.bin_hz))
                      for hz, side in ((low, 1), (high, -1)) if 0 < hz < self.nyquist]
        self.partials = [partial for _, _, partial in self.edges]
        if low == 0 and high > 2 * self.bin_hz:
            # The real signal's positive- and negative-frequency lobes meet
            # here even when the upper decision edge is many bins away.
            self.partials.append(self._partial(0))

    def _resolve(self, partial, spectrum, powers, noise):
        vector = partial.observe(spectrum)
        if not np.any(vector):
            return None
        fit = partial.fit(vector, noise)
        cycles, cs = partial.edge, None
        if fit is not None:
            cycles, cs, resolved = fit
            if resolved:
                return cycles, cs

        # A separated program partial can leak into a quiet band-edge lobe.
        # Identify it at its own peak, subtract its predicted leakage only
        # from the coefficients being fitted, then check the edge fit again.
        local_peak = powers[partial.bins].max()
        peaks = signal.find_peaks(powers)[0]
        distance = np.abs(peaks - partial.edge)
        eligible = (distance >= 4) & (powers[peaks] >= 0.01 * local_peak)
        peaks, distance = peaks[eligible], distance[eligible]
        # The Hann sidelobe amplitude decays as inverse frequency cubed.
        influence = powers[peaks] / (distance - 3) ** 6
        order = np.argsort(influence)[-3:][::-1]
        for index in order:
            if fit is None and influence[index] < 1e-5 * local_peak:
                break
            k = peaks[index]
            remote = self._partial(k)
            measured = remote.observe(spectrum)
            corrected = vector
            # Each initially fitted tone contains a little of the other's
            # sidelobe. Refine both estimates before trying another neighbor.
            for _ in range(3):
                remote_vector = measured if cs is None else measured - remote._basis(cycles) @ cs
                background = remote.fit(remote_vector, noise)
                if background is None:
                    break
                f, gains, _ = background
                if abs(f - cycles) < 4:
                    break  # overlapping Hann main lobes
                corrected = vector - partial._basis(f) @ gains
                candidate = partial.fit(corrected, noise)
                if candidate is None:
                    break
                cycles, cs, resolved = candidate
                if resolved:
                    return cycles, cs
            vector = corrected
        return None

    def _noise_power(self, powers):
        # Median power across broad frequency bands estimates the recording
        # noise even for quantized periodic signals with a sparse noise floor.
        bands = min(32, len(powers) // 16)
        noise = 0.0
        if bands:
            bins = (len(powers) - 1) // bands
            noise = np.median(powers[1:1 + bins * bands].reshape(bands, bins).mean(axis=1))
            noise /= len(self.window) ** 2
        return noise

    def _spectrum(self, ac):
        raw = np.fft.rfft(ac)
        # Multiplication by a periodic Hann or its derivative is a three-bin
        # convolution. Reflect the one-sided real FFT at DC and Nyquist.
        last = -2 if len(ac) % 2 == 0 else -1
        padded = np.concatenate((raw[1:2].conj(), raw, [raw[last].conj()]))
        left, right = padded[:-2], padded[2:]
        sides = (left + right) * 0.25
        return 0.5 * raw - sides, self.bin_hz / (4j) * (left - right), sides

    def levels(self, block):
        scale = np.max(np.abs(block))
        if not scale:
            return -math.inf, -math.inf, -math.inf
        ac = np.asarray(block, dtype=np.float64) / scale
        ac -= ac.mean()
        spectrum, derivative, sides = self._spectrum(ac)
        powers = np.abs(spectrum) ** 2
        total = np.sum(powers * self.weights)
        if not total:
            return -math.inf, -math.inf, -math.inf
        tol = 1e-7 * self.bin_hz
        partial_power = 0.0
        noise = self._noise_power(powers)
        for boundary in self.partials:
            fit = self._resolve(boundary, spectrum, powers, noise)
            if fit is None:
                continue
            cycles, (cosine, sine) = fit
            phase = cycles * self.phase
            line = cosine * np.cos(phase) + sine * np.sin(phase)
            line -= line.mean()
            if self.low - tol <= cycles * self.bin_hz <= self.high + tol:
                partial_power += np.sum((line * self.window) ** 2) / self.window_energy
            ac -= line
            spectrum, derivative, sides = self._spectrum(ac)
            powers = np.abs(spectrum) ** 2
            noise = self._noise_power(powers)

        # Diffuse energy and overlapping lines use the local phase derivative
        # (Flandrin et al., Time-frequency reassignment, eq. 5.20).
        powers *= self.weights
        shift = np.divide(derivative, spectrum, out=derivative, where=spectrum != 0)
        frequency = np.clip(self.freqs - shift.imag, 0, self.nyquist)
        inside = (frequency > 0) & (frequency >= self.low - tol) & (frequency <= self.high + tol)
        confirmed = inside.copy()
        for hz, direction, edge_partial in self.edges:
            bins = edge_partial.bins
            local = spectrum[bins]
            # For two unresolved lines, the magnitude of the second central
            # frequency moment is the product of their distances from the mean.
            second = self.bin_hz ** 2 * sides[bins]
            np.divide(second, local, out=second, where=local != 0)
            spread = np.sqrt(np.abs(shift[bins] ** 2 - second))
            confirmed[bins] &= direction * (frequency[bins] - hz) >= spread - tol

        power = powers[inside].sum() + partial_power
        if not power:
            return -math.inf, -math.inf, -math.inf
        confident = powers[confirmed].sum() + partial_power
        level = 10 * math.log10(power) + 20 * math.log10(scale)
        ratio = min(0.0, 10 * math.log10(power / total))
        confidence_db = 10 * math.log10(confident / power) if confident else -math.inf
        return level, ratio, confidence_db


# Cache window readings only. Each entry holds at most 32 triples of numbers.
_band_readings: OrderedDict[tuple[object, ...], list[tuple[int, float, float, float]]] = OrderedDict()
_band_readings_lock = Lock()


def _channel_readings(channel, meter, sample_rate, width, hop):
    """Measure overlapping windows in bounded chunks, in their original order."""
    parameters = (sample_rate, meter.low, meter.high)
    if channel.dtype.hasobject or not all(isinstance(value, Hashable) for value in parameters):
        # Object buffers and mutable numeric arguments such as 0-D arrays
        # can be measured directly, but cannot identify immutable readings.
        for start in range(0, len(channel) - width + 1, hop):
            yield start, *meter.levels(channel[start:start + width])
        return
    # Limit the private sample copy to 2 MiB or one analysis window, whichever
    # is larger. Hash and measure this same copy even if the caller edits its
    # input buffer or the track came from a strided routing view.
    chunk_samples = (2 * 1024 * 1024) // channel.itemsize
    chunk_windows = max(1, min(32, 1 + (chunk_samples - width) // hop))
    # NumPy scalar arithmetic can calibrate a boundary at a different
    # precision even when its value compares equal to a Python number.
    geometry = tuple((type(value), value) for value in parameters)
    for start in range(0, len(channel) - width + 1, chunk_windows * hop):
        count = min(chunk_windows, 1 + (len(channel) - start - width) // hop)
        samples = channel[start:start + (count - 1) * hop + width].copy(order="C")
        key = (geometry, width, samples.dtype.str, sha256(samples).digest())
        with _band_readings_lock:
            readings = _band_readings.get(key)
            if readings is not None:
                _band_readings.move_to_end(key)
        if readings is None:
            readings = tuple(meter.levels(samples[i * hop:i * hop + width]) for i in range(count))
            with _band_readings_lock:
                _band_readings[key] = readings
                _band_readings.move_to_end(key)
                if len(_band_readings) > 2048:
                    _band_readings.popitem(last=False)
        for i, (level, ratio, confidence) in enumerate(readings):
            yield start + i * hop, level, ratio, confidence


def band_evidence(audio, sample_rate, *, low_hz, high_hz, window_sec,
                  level_db, ratio_db, consecutive):
    """Measure sustained AC band energy per channel, with absolute and relative gates.

    Spans are inclusive and refer to full windows on a single channel.
    Short clips supply a meter reading but cannot qualify a repair.
    """
    data = np.asarray(audio)
    full_width = max(2, round(window_sec * sample_rate))
    width = min(len(data), full_width)
    if width < 2 or not data.size:
        return {"regions": [], "band_dbfs": None, "ratio_db": None}
    spans = []
    peak_level = peak_ratio = -math.inf
    channels = data.reshape(len(data), -1)
    hop = max(1, width // 2)
    meter = _BandWindow(width, sample_rate, low_hz, high_hz)
    for ch, channel in enumerate(channels.T):
        count = 0
        for start, level, ratio, confidence_db in _channel_readings(channel, meter, sample_rate, width, hop):
            if level > peak_level:
                peak_level, peak_ratio = level, ratio
            if width < full_width or level + confidence_db < level_db or ratio + confidence_db < ratio_db:
                count = 0
                continue
            count += 1
            end = start + width - 1
            if count == consecutive:
                spans.append([start - (count - 1) * hop, end, ch])
            elif count > consecutive:
                spans[-1][1] = end
    return {"regions": spans, "band_dbfs": peak_level, "ratio_db": peak_ratio}
