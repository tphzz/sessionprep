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
    # Check the nearest tenth in sample units. A log round-trip can turn
    # exactly 0.1 dB into 0.1000000000000001 and recommend an extra step.
    tenths = round(200 * max(positive_over, negative_over))
    if tenths < 60000:
        factor = 10 ** (tenths / 200)
        fits = high <= positive_range * factor and -low <= negative_range * factor
    else:
        attenuation = 10 ** (-tenths / 200)
        fits = high * attenuation <= positive_range and -low * attenuation <= negative_range
    reduction = (tenths if fits else tenths + 1) / 10
    peak_db = 20 * math.log10(max(-low, high))
    raise OutputHeadroomError(peak_db, reduction, subtype)


class _BandWindow:
    """One calibrated Hann window reused for every channel and hop."""

    def __init__(self, width, rate, low, high):
        self.low, self.high = low, high
        self.bin_hz = rate / width
        self.nyquist = rate / 2
        self.freqs = np.fft.rfftfreq(width, 1 / rate)
        window = signal.windows.hann(width, sym=False)
        self.weights = np.full(len(self.freqs), 2.0 / (width * np.sum(window ** 2)))
        self.weights[0] = 0
        if width % 2 == 0:
            self.weights[-1] *= 0.5

        self.edges = []
        for hz, side in ((low, 1), (high, -1)):
            if not 0 < hz < self.nyquist:
                continue
            # Two bins on either side span the Hann main lobe. The measured
            # sine/cosine basis also represents an edge between native bins.
            bins = np.flatnonzero(np.abs(self.freqs - hz) <= 2 * self.bin_hz)
            phase = 2 * np.pi * hz * np.arange(width) / rate
            basis = np.column_stack((np.cos(phase), np.sin(phase)))
            basis -= basis.mean(axis=0)
            tones = np.fft.rfft(window[:, None] * basis, axis=0)[bins]
            components = np.concatenate((tones.real, tones.imag))
            projection = components @ np.linalg.pinv(components)
            self.edges.append((hz, side, bins, projection))

    def levels(self, block):
        scale = np.max(np.abs(block))
        if not scale:
            return -math.inf, -math.inf, -math.inf
        ac = np.asarray(block, dtype=np.float64) / scale
        ac -= ac.mean()
        raw = np.fft.rfft(ac)
        # Multiplication by the periodic Hann (and its two time derivatives)
        # is a three-bin convolution. Reflect the real FFT at DC and Nyquist.
        last = -2 if len(block) % 2 == 0 else -1
        padded = np.concatenate((raw[1:2].conj(), raw, [raw[last].conj()]))
        left, right = padded[:-2], padded[2:]
        sides = (left + right) * 0.25
        spectrum = 0.5 * raw - sides
        powers = np.abs(spectrum) ** 2 * self.weights
        total = powers.sum()
        if not total:
            return -math.inf, -math.inf, -math.inf

        # Local frequency from the phase derivative (Flandrin et al.,
        # Time-frequency reassignment, eq. 5.20). A single tone carries its
        # main-lobe energy to its own frequency, also at the inclusive edge.
        derivative = self.bin_hz / (4j) * (left - right)
        shift = np.divide(derivative, spectrum, out=derivative, where=spectrum != 0)
        frequency = np.clip(self.freqs - shift.imag, 0, self.nyquist)
        tolerance = 1e-7 * self.bin_hz
        inside = (frequency > 0) & (frequency >= self.low - tolerance) & (frequency <= self.high + tolerance)
        confirmed = inside.copy()
        for hz, direction, bins, projection in self.edges:
            local = spectrum[bins]
            vector = np.concatenate((local.real, local.imag))
            residual = vector - projection @ vector
            local_energy = vector @ vector
            if local_energy and residual @ residual <= 1e-12 * local_energy:
                # One edge tone fits the measured complex coefficients to
                # within one part per million in amplitude, at any phase.
                inside[bins] = confirmed[bins] = True
                continue
            # Two unresolved tones can beat at an apparent frequency outside
            # both. For two lines the magnitude of the second central moment
            # is the product of their distances from the complex mean.
            # Keep the reading, but require its uncertainty to clear the edge
            # before using those bins to select a whole-track filter.
            second = self.bin_hz ** 2 * sides[bins]
            np.divide(second, local, out=second, where=local != 0)
            spread = np.sqrt(np.abs(shift[bins] ** 2 - second))
            confirmed[bins] &= direction * (frequency[bins] - hz) >= spread - tolerance

        power = powers[inside].sum()
        if not power:
            return -math.inf, -math.inf, -math.inf
        confident = powers[confirmed].sum()
        level = 10 * math.log10(power) + 20 * math.log10(scale)
        ratio = min(0.0, 10 * math.log10(power / total))
        confidence_db = 10 * math.log10(confident / power) if confident else -math.inf
        return level, ratio, confidence_db


def band_evidence(audio, sample_rate, *, low_hz, high_hz, window_sec,
                  level_db, ratio_db, consecutive):
    """Measure sustained AC band energy per channel, with absolute and relative gates.

    Spans are inclusive and refer to full windows on a single channel.
    Short clips supply a meter reading but cannot qualify a repair.
    """
    data = np.asarray(audio)
    full_width = max(2, round(window_sec * sample_rate))
    width = min(len(data), full_width)
    spans = []
    peak_level = peak_ratio = -math.inf
    if width < 2 or not data.size:
        return {"regions": spans, "band_dbfs": None, "ratio_db": None}
    channels = data.reshape(len(data), -1)
    hop = max(1, width // 2)
    meter = _BandWindow(width, sample_rate, low_hz, high_hz)
    for ch, channel in enumerate(channels.T):
        count = 0
        for start in range(0, len(channel) - width + 1, hop):
            block = channel[start:start + width]
            level, ratio, confidence_db = meter.levels(block)
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
