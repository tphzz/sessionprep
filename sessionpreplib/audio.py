from __future__ import annotations

import math
import os
import sys
from bisect import bisect_right
from hashlib import file_digest, sha256
from typing import Any

import numpy as np
import soundfile as sf
from scipy import fft
from scipy.signal import ZoomFFT, stft as scipy_stft

from .models import TrackContext
from .paths import DEFAULT_STEREO_WINDOW_MS
from .chunks import chunk_ids as _chunk_ids


# Supported audio file extensions (lowercase, with leading dot)
AUDIO_EXTENSIONS = ('.wav', '.aif', '.aiff')


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

AES17_OFFSET = 3.0103  # dB offset: 20 * log10(sqrt(2))


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    if linear <= 0:
        return float(-np.inf)
    return float(20 * np.log10(linear))


def dbfs_offset(config: dict) -> float:
    """Return the dBFS offset for the configured convention.

    Standard → 0.0; AES17 → +3.0103 dB.
    """
    return AES17_OFFSET if config.get("dbfs_convention") == "aes17" else 0.0


def format_duration(samples: int, samplerate: int) -> str:
    if samplerate <= 0:
        return "00:00.000"
    seconds = samples / samplerate
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_SUBTYPE_MAP = {
    'PCM_16': '16-bit',
    'PCM_24': '24-bit',
    'PCM_32': '32-bit',
    'FLOAT': '32-bit Float',
    'DOUBLE': '64-bit Float',
}


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = [(key, ctypes.c_int64) for key in
                    ("creation", "access", "write", "change")] + [("attributes", wintypes.DWORD)]


class SourceChangedError(ValueError):
    """A source no longer matches the audio being read or its saved analysis."""


def file_revision(path: str, stat: os.stat_result, *, allow_digest: bool = True) -> str | None:
    """Detect replacements and rewrites that preserve the WAV size and mtime."""
    if sys.platform == "win32":
        with open(path, "rb") as stream:
            info = _FileBasicInfo()
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
            if ctypes.windll.kernel32.GetFileInformationByHandleEx(
                    handle, 0, ctypes.byref(info), ctypes.sizeof(info)) and info.change:
                changed = info.change
            else:
                # Some remote filesystems provide no change timestamp.
                return file_digest(stream, "sha256").hexdigest() if allow_digest else None
    else:
        changed = stat.st_ctime_ns
    return f"{stat.st_dev}:{stat.st_ino}:{changed}"


def sample_digest(data: np.ndarray) -> str:
    """Identify both the sample values and their channel interpretation."""
    pcm = np.ascontiguousarray(data)
    return f"{pcm.dtype.str}:{pcm.shape}:{sha256(pcm).hexdigest()}"


def discover_track(filepath: str) -> TrackContext:
    """Read audio file metadata without loading audio data.

    Returns a TrackContext with filename, filepath, channels, samplerate,
    total_samples, bitdepth, subtype, duration_sec populated.
    ``audio_data`` is left as ``None``.
    """
    stat = os.stat(filepath)
    info = sf.info(filepath)
    return TrackContext(
        filename=os.path.basename(filepath),
        filepath=filepath,
        audio_data=None,
        samplerate=info.samplerate,
        channels=info.channels,
        total_samples=info.frames,
        bitdepth=_SUBTYPE_MAP.get(info.subtype, info.subtype),
        subtype=info.subtype,
        duration_sec=info.duration,
        source_file_signature=(stat.st_size, stat.st_mtime_ns),
        source_file_revision=file_revision(filepath, stat, allow_digest=False),
    )


def load_track(filepath: str) -> TrackContext:
    """Read an audio file (WAV/AIFF) and return a fully populated TrackContext."""
    stat = os.stat(filepath)
    revision = file_revision(filepath, stat)
    signature = (stat.st_size, stat.st_mtime_ns)
    try:
        with sf.SoundFile(filepath) as stream:
            samplerate, subtype = stream.samplerate, stream.subtype
            data = stream.read(dtype='float64')
    finally:
        # A replacement can make the decoder fail before it returns any samples.
        latest = os.stat(filepath)
        if signature != (latest.st_size, latest.st_mtime_ns) or revision != file_revision(filepath, latest):
            raise SourceChangedError("Source changed while it was being read. Reanalyze this source.")
    channels = 1 if data.ndim == 1 else data.shape[1]
    try:
        cids = _chunk_ids(filepath)
    except (ValueError, OSError):
        cids = []
    return TrackContext(
        filename=os.path.basename(filepath),
        filepath=filepath,
        audio_data=data,
        samplerate=samplerate,
        channels=channels,
        total_samples=len(data),
        bitdepth=_SUBTYPE_MAP.get(subtype, subtype),
        subtype=subtype,
        duration_sec=len(data) / samplerate,
        chunk_ids=cids,
        source_file_signature=signature,
        source_file_revision=revision,
        source_sample_digest=sample_digest(data),
    )


def discover_audio_files(
    root_dir: str,
    recursive: bool = False,
    skip_folders: set[str] | None = None,
) -> list[str]:
    """Return a sorted list of audio file paths relative to *root_dir*.

    When *recursive* is ``False``, returns bare filenames (flat listing).
    When ``True``, walks subdirectories (symlinks are **not** followed)
    and returns forward-slash–separated relative paths such as
    ``"drums/01_Kick.wav"``.

    Directories whose name appears in *skip_folders* are pruned from the
    walk (e.g. ``{"sp_01_tracklayout", "sp_02_prepared"}``).
    """
    from .utils import protools_sort_key

    skip = skip_folders or set()
    result: list[str] = []

    if not recursive:
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(AUDIO_EXTENSIONS):
                result.append(fname)
    else:
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
            # Prune skipped directories in-place so os.walk won't descend
            dirnames[:] = [
                d for d in dirnames if d not in skip
            ]
            for fname in filenames:
                if fname.lower().endswith(AUDIO_EXTENSIONS):
                    rel = os.path.relpath(
                        os.path.join(dirpath, fname), root_dir)
                    # Normalise to forward slashes for cross-platform keys
                    result.append(rel.replace("\\", "/"))

    result.sort(key=protools_sort_key)
    return result


def write_track(track: TrackContext, output_path: str) -> None:
    """Write track audio_data to WAV, preserving original subtype."""
    sf.write(output_path, track.audio_data, track.samplerate, subtype=track.subtype)


# ---------------------------------------------------------------------------
# Cached helpers — expensive computations stored on TrackContext._cache
# ---------------------------------------------------------------------------

def get_peak(track: TrackContext) -> float:
    """Peak linear amplitude. Cached."""
    if "peak_linear" not in track._cache:
        if track.audio_data is not None and track.audio_data.size > 0:
            track._cache["peak_linear"] = float(np.max(np.abs(track.audio_data)))
        else:
            track._cache["peak_linear"] = 0.0
    return track._cache["peak_linear"]


def get_peak_db(track: TrackContext) -> float:
    """Peak in dBFS. Cached."""
    if "peak_db" not in track._cache:
        track._cache["peak_db"] = linear_to_db(get_peak(track))
    return track._cache["peak_db"]


def is_silent(track: TrackContext) -> bool:
    """True if the file is absolute silence (peak == 0)."""
    peak = get_peak(track)
    validate_analysis_level(peak, track.total_samples)
    return peak == 0.0


def validate_analysis_level(peak: float, samples: int) -> None:
    """Require finite samples and room to square windowed spectral sums in DOUBLE."""
    if not math.isfinite(peak):
        raise ValueError("Source contains non-finite samples")
    if samples and peak > math.sqrt(np.finfo(np.float64).max) / samples:
        raise ValueError("Source level exceeds the finite range for analysis. Lower the source level.")


def squared_sample_prefix(samples: np.ndarray) -> np.ndarray:
    """Cumulative channel energy with a zero prefix for arbitrary RMS windows."""
    prefix = np.empty(len(samples) + 1, dtype=np.float64)
    prefix[0] = 0
    np.square(samples, dtype=np.float64, out=prefix[1:])
    np.cumsum(prefix[1:], out=prefix[1:])
    return prefix


def max_momentary_rms(prefixes: list[np.ndarray], window: int) -> tuple[int, float, float]:
    """Center sample, dBFS and linear maximum of the channel-averaged RMS."""
    if not prefixes or window <= 0 or len(prefixes[0]) < 2:
        return -1, float("-inf"), 0.0
    window = min(window, len(prefixes[0]) - 1)
    energy = (prefixes[0][window:] - prefixes[0][:-window]) / len(prefixes)
    for prefix in prefixes[1:]:
        energy += (prefix[window:] - prefix[:-window]) / len(prefixes)
    start = int(np.argmax(energy))
    rms = float(np.sqrt(energy[start] / window))
    return start + window // 2, 20 * np.log10(rms) if rms else float("-inf"), rms


def get_rms_window_means(
    track: TrackContext,
    window_ms: int,
    stereo_mode: str,
) -> np.ndarray:
    """
    Momentary RMS window means via sliding-window cumsum.
    Cached on track._cache.
    """
    key = f"rms_window_means_{window_ms}_{stereo_mode}"
    if key not in track._cache:
        data = track.audio_data
        assert data is not None
        samplerate = track.samplerate

        if data.ndim > 1 and data.shape[1] > 1:
            if stereo_mode == "sum":
                squared = np.sum(data.astype(np.float64) ** 2, axis=1)
            else:  # avg
                squared = np.mean(data.astype(np.float64) ** 2, axis=1)
        else:
            flat = data.flatten() if data.ndim > 1 else data
            squared = flat.astype(np.float64) ** 2

        window_samples = max(1, int((window_ms / 1000) * samplerate))

        if len(squared) <= window_samples:
            window_means = np.array([np.mean(squared)], dtype=np.float64)
        else:
            cumsum = np.cumsum(squared, dtype=np.float64)
            cumsum = np.concatenate(([0.0], cumsum))
            window_sums = cumsum[window_samples:] - cumsum[:-window_samples]
            window_means = window_sums / window_samples

        track._cache[key] = window_means
        track._cache[f"window_samples_{window_ms}"] = window_samples
    return track._cache[key]


def get_window_samples(track: TrackContext, window_ms: int) -> int:
    """Number of samples per RMS window. Cached as side-effect of get_rms_window_means."""
    key = f"window_samples_{window_ms}"
    if key not in track._cache:
        track._cache[key] = max(1, int((window_ms / 1000) * track.samplerate))
    return track._cache[key]


def get_gated_rms_data(
    track: TrackContext,
    window_ms: int,
    stereo_mode: str,
    gate_relative_db: float,
) -> dict[str, Any]:
    """
    Gated RMS analysis data. Returns dict with:
        active_means, active_mask, max_window_db, window_means
    Cached on track._cache.
    """
    key = f"gated_rms_{window_ms}_{stereo_mode}_{gate_relative_db}"
    if key not in track._cache:
        window_means = get_rms_window_means(track, window_ms, stereo_mode)
        floor = np.finfo(np.float64).tiny
        window_rms_db = 10.0 * np.log10(np.maximum(window_means, floor))
        max_window_db = float(np.max(window_rms_db)) if window_rms_db.size else float(-np.inf)
        gate_threshold_db = max_window_db - float(gate_relative_db)
        active_mask = window_rms_db >= gate_threshold_db
        active_means = window_means[active_mask] if active_mask.size else window_means
        if active_means.size == 0:
            active_means = window_means
            active_mask = np.ones(window_means.shape, dtype=bool)

        track._cache[key] = {
            "active_means": active_means,
            "active_mask": active_mask,
            "max_window_db": max_window_db,
            "window_means": window_means,
        }
    return track._cache[key]


# ---------------------------------------------------------------------------
# Stereo helpers (cached)
# ---------------------------------------------------------------------------

def _channel_ac_rms(channel: np.ndarray, block_size: int) -> float:
    low, high = float(channel.min()), float(channel.max())
    if low == high:
        return 0.0
    # Work relative to the signal's range so a small ripple around a large
    # DC offset survives centering. A square wave at the threshold has
    # unit mean square at every file length.
    midpoint = low + (high - low) * 0.5
    scale = max(high - midpoint, midpoint - low)
    mean = 0.0
    sum_squares = 0.0
    for start in range(0, len(channel), block_size):
        block = np.subtract(channel[start:start + block_size], midpoint, dtype=np.float64)
        block /= scale
        block_mean = float(block.mean())
        block -= block_mean
        # Pool the two centered sums using their difference in means.
        weight = len(block) / (start + len(block))
        delta = block_mean - mean
        sum_squares += float(np.sum(np.square(block, out=block))) + delta**2 * start * weight
        mean += delta * weight
    return scale * math.sqrt(sum_squares / len(channel))


def stereo_layout_levels(track: TrackContext) -> tuple[float, float, float] | None:
    """Peak absolute L/R difference and DC-removed RMS of each complete channel.

    Both layout detectors can recommend discarding a channel. Share exact
    measurements and keep only the three scalar results in the track cache.
    """
    key = "stereo_layout_levels"
    if key not in track._cache:
        data = track.audio_data
        if data is None or data.ndim != 2 or data.shape[1] != 2 or not len(data):
            return None
        left, right = data.T
        max_difference = 0.0
        block_size = 65_536
        for start in range(0, len(data), block_size):
            block = data[start:start + block_size]
            difference = np.subtract(block[:, 0], block[:, 1], dtype=np.float64)
            max_difference = np.maximum(max_difference, np.max(np.abs(difference)))
        left_rms = _channel_ac_rms(left, block_size)
        right_rms = left_rms if max_difference == 0 else _channel_ac_rms(right, block_size)
        track._cache[key] = float(max_difference), left_rms, right_rms
    return track._cache[key]


def get_stereo_channels_subsampled(
    track: TrackContext,
    max_samples: int = 200000,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """
    Subsampled raw L/R channels (NOT DC-removed).
    Returns (l_raw, r_raw, step) or None for mono/invalid.
    """
    key = f"stereo_channels_subsampled_{max_samples}"
    if key not in track._cache:
        data = track.audio_data
        if data is not None and data.ndim > 1 and data.shape[1] == 2 and data.shape[0] > 1:
            l_full = data[:, 0].astype(np.float64)
            r_full = data[:, 1].astype(np.float64)
            step = max(1, int(l_full.size // max_samples))
            l_raw = l_full[::step]
            r_raw = r_full[::step]
            track._cache[key] = (l_raw, r_raw, step)
        else:
            track._cache[key] = None
    return track._cache[key]


def get_stereo_channels_dc_removed(
    track: TrackContext,
    max_samples: int = 200000,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """
    Subsampled DC-removed L/R channels.
    Returns (l, r, step) or None for mono/invalid.
    """
    key = f"stereo_channels_dc_removed_{max_samples}"
    if key not in track._cache:
        raw = get_stereo_channels_subsampled(track, max_samples)
        if raw is None:
            track._cache[key] = None
        else:
            l_raw, r_raw, step = raw
            left_ac = l_raw - np.mean(l_raw)
            right_ac = r_raw - np.mean(r_raw)
            track._cache[key] = (left_ac, right_ac, step)
    return track._cache[key]


def get_stereo_rms(
    track: TrackContext,
    max_samples: int = 200000,
) -> tuple[float, float, float, float] | None:
    """
    Per-channel RMS from DC-removed subsampled stereo.
    Returns (l_rms_lin, r_rms_lin, l_rms_db, r_rms_db) or None.
    """
    key = f"stereo_rms_{max_samples}"
    if key not in track._cache:
        dc_removed = get_stereo_channels_dc_removed(track, max_samples)
        if dc_removed is None:
            track._cache[key] = None
        else:
            left_ac, right_ac, _step = dc_removed
            l_rms_lin = float(np.sqrt(np.mean(left_ac ** 2)))
            r_rms_lin = float(np.sqrt(np.mean(right_ac ** 2)))
            l_rms_db = linear_to_db(l_rms_lin)
            r_rms_db = linear_to_db(r_rms_lin)
            track._cache[key] = (l_rms_lin, r_rms_lin, l_rms_db, r_rms_db)
    return track._cache[key]


# ---------------------------------------------------------------------------
# Stateless DSP functions
# ---------------------------------------------------------------------------

def detect_clipping_ranges(
    data: np.ndarray,
    threshold_count: int,
    max_ranges: int = 10,
    *,
    subtype: str | None = None,
) -> tuple[int, list[tuple[int, int, int | None]]]:
    """Return (count, ranges) where ranges are (start_sample, end_sample, channel).

    ``channel`` is an int (0-based) for multi-channel data or ``None`` for mono.
    """
    total, ranges, _channels = clipping_evidence(data, threshold_count, max_ranges, subtype=subtype)
    return total, ranges


def clipping_evidence(data: np.ndarray, threshold_count: int, max_ranges: int, *, subtype: str | None = None):
    """Measure all channels but materialize only the requested number of overlay locations."""
    channels = data.T if data.ndim == 2 else (data,)
    total = 0
    affected: list[int] = []
    ranges: list[tuple[int, int, int | None]] = []
    # The largest 8-bit PCM code decodes below the usual near-full-scale threshold.
    positive_limit = 127 / 128 if subtype in {"PCM_U8", "PCM_S8"} else 0.9999
    for ch, samples in enumerate(channels):
        idx = np.flatnonzero((samples >= positive_limit) | (samples <= -0.9999))
        if not idx.size:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = idx[np.r_[0, splits + 1]]
        ends = idx[np.r_[splits, idx.size - 1]]
        qualifying = np.flatnonzero(ends - starts + 1 >= threshold_count)
        if qualifying.size:
            total += int(qualifying.size)
            affected.append(ch)
            label = ch if len(channels) > 1 else None
            for i in qualifying[:max(0, int(max_ranges) - len(ranges))]:
                ranges.append((int(starts[i]), int(ends[i]), label))
    return total, ranges, affected


def _band_stop(length: int, samplerate: int, cutoff_hz: float) -> int:
    """First bin above the inclusive cutoff, compared in the user's Hz units."""
    return bisect_right(range(length // 2 + 1), cutoff_hz, key=lambda k: k * samplerate / length)


def _whole_subsonic_ratio(ac: np.ndarray, samplerate: int, cutoff_hz: float, silence_rms: float) -> float:
    """Power in whole-record DFT bins through the cutoff, relative to all AC power."""
    n = len(ac)
    last = _band_stop(n, samplerate, cutoff_hz) - 1
    total = np.einsum("i,i->", ac, ac) / n
    if last < 1 or total == 0 or total < silence_rms**2:
        return -math.inf
    if last == n // 2:
        return 0.0

    if n <= 262144 or fft.next_fast_len(n, real=True) == n or last > n // 16:
        bins = fft.rfft(ac, norm="forward")[1:last + 1]
    else:
        # An exact-length FFT of a multi-minute, prime-length stem can need
        # several times the source's memory. Evaluate only the low-frequency
        # DFT bins in bounded blocks on the same whole-file frequency grid.
        width = 65536
        transform = ZoomFFT(width, [1 / n, (last + 1) / n], m=last, fs=1)
        radians = -2j * np.pi * np.arange(1, last + 1) / n
        bins = np.zeros(last, dtype=np.complex128)
        for start in range(0, n, width):
            block = ac[start:start + width]
            if block.size < width:
                block = np.pad(block, (0, width - block.size))
            bins += transform(block) * np.exp(start * radians)
        bins /= n
    # The real signal's negative-frequency bins have the same energy. The
    # all-bin shortcut above already handles Nyquist and the excluded DC.
    band = 2 * np.vdot(bins, bins).real
    return min(0.0, 10 * (math.log10(band) - math.log10(total))) if band > 0 else -math.inf


def _windowed_subsonic_ratios(
    padded: np.ndarray,
    ac: np.ndarray,
    original_size: int,
    samplerate: int,
    cutoff_hz: float,
    nperseg: int,
    hop_samples: int,
    n_frames: int,
    abs_gate_db: float,
    silence_rms: float,
) -> list[tuple[int, int, float]]:
    """Compute per-window STFT subsonic ratios with silence and absolute-power gates."""
    noverlap = nperseg - hop_samples
    nfft = 1 << (nperseg - 1).bit_length()

    # Gate on the real, DC-free samples contributing to each STFT frame.
    rms_arr = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop_samples
        e = min(s + nperseg, original_size)
        rms_arr[i] = np.sqrt(np.mean(ac[s:e] ** 2))

    _f, _t, Zxx = scipy_stft(
        padded, fs=samplerate, nperseg=nperseg, nfft=nfft,
        noverlap=noverlap, window='hann', boundary=None,
    )
    power = np.abs(Zxx) ** 2
    power[[0, -1]] *= 0.5
    cutoff_bin = _band_stop(nfft, samplerate, cutoff_hz)

    band = np.sum(power[:cutoff_bin], axis=0)
    total = np.sum(power, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(
            (band > 0) & (total > 0),
            10.0 * np.log10(band / total),
            -np.inf,
        )

    ratios[rms_arr < silence_rms] = -np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        rms_db = np.where(rms_arr > 0, 20.0 * np.log10(rms_arr), -200.0)
    abs_mask = np.isfinite(ratios) & ((rms_db + ratios) < abs_gate_db)
    ratios[abs_mask] = -np.inf

    return [
        (i * hop_samples, min(i * hop_samples + nperseg, original_size) - 1, float(ratios[i]))
        for i in range(n_frames)
    ]


def subsonic_stft_analysis(
    signal: np.ndarray,
    samplerate: int,
    cutoff_hz: float,
    *,
    window_ms: int = 500,
    hop_ms: int | None = None,
    abs_gate_db: float = -40.0,
    silence_rms: float = 1e-7,
    include_windows: bool = True,
    minimum_whole_ratio_db: float = -math.inf,
) -> tuple[float, list[tuple[int, int, float]]]:
    """Whole-record AC band energy and local Hann-window spectral evidence.

    Returns ``(whole_file_ratio_db, per_window_ratios)`` where each per-window
    tuple is ``(sample_start, sample_end, ratio_db)``.
    """
    _NEG_INF = float(-np.inf)

    if signal.ndim != 1 or signal.size < 8 or samplerate <= 0:
        return _NEG_INF, []

    original_size = signal.size
    nperseg = max(8, int(samplerate * window_ms / 1000))
    hop_samples = min(nperseg, max(1, int(samplerate * (hop_ms or window_ms) / 1000)))

    # Pad signal so scipy.signal.stft (boundary=None) includes the tail.
    # Remove the channel's stationary bias before tapering or zero padding.
    n_frames = 1 + (max(0, original_size - nperseg) + hop_samples - 1) // hop_samples
    padded = np.zeros(nperseg + (n_frames - 1) * hop_samples, dtype=np.float64)
    ac = padded[:original_size]
    np.subtract(signal, signal[0], dtype=np.float64, out=ac)
    ac -= ac.mean()
    whole_ratio = _whole_subsonic_ratio(ac, samplerate, cutoff_hz, silence_rms)
    if not include_windows or whole_ratio < minimum_whole_ratio_db:
        return whole_ratio, []

    results = _windowed_subsonic_ratios(
        padded, ac, original_size, samplerate, cutoff_hz,
        nperseg, hop_samples, n_frames, abs_gate_db, silence_rms,
    )
    return whole_ratio, results


# ---------------------------------------------------------------------------
# Windowed stereo correlation + mono folddown
# ---------------------------------------------------------------------------

def _centered_energy(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale owned rows in place. Return their scaled norm² and log energy."""
    # Keep constant rows exactly silent and tiny fold residuals above underflow.
    rows -= rows[:, :1].copy()
    scale = np.maximum(rows.max(axis=1), -rows.min(axis=1))
    scale[scale == 0] = 1.0
    rows /= scale[:, None]
    rows -= rows.mean(axis=1, keepdims=True)
    power = np.einsum("ij,ij->i", rows, rows)
    with np.errstate(divide="ignore"):
        return power, np.log(power) + 2 * np.log(scale)


def _stereo_readings(left: np.ndarray, right: np.ndarray):
    """Measure each row independently over its real samples (no padding)."""
    # Fold before centering the channels. Separate centering can lose a real
    # residual, and forming the average here can underflow at a deep null.
    folded = np.add(left, right, dtype=np.float64)
    reference = folded[:, :1].copy()
    # TwoSum recovers the rounding residual of L+R. Remove the large
    # common DC term before adding the residual back. This preserves small
    # AC alongside a DC offset, as well as a deep null in the stereo fold.
    for start in range(0, folded.shape[1], 65536):
        sl = slice(start, start + 65536)
        high = folded[:, sl]
        left_blk, right_blk = left[:, sl], right[:, sl]
        virtual_r = high - left_blk
        low = (left_blk - (high - virtual_r)) + (right_blk - virtual_r)
        high -= reference
        high += low
    _, log_fold = _centered_energy(folded)
    del folded, high
    l_ac = left.astype(np.float64)
    r_ac = right.astype(np.float64)
    norm_l, log_l = _centered_energy(l_ac)
    norm_r, log_r = _centered_energy(r_ac)
    denom = np.sqrt(norm_l) * np.sqrt(norm_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.clip(np.einsum("ij,ij->i", l_ac, r_ac) / denom, -1, 1)
        log_stereo = np.logaddexp(log_l, log_r)
        loss = (10 / math.log(10)) * (log_stereo + math.log(2) - log_fold)
    loss = np.where(np.isneginf(log_stereo), 0.0, np.maximum(loss, 0.0))
    return corr, loss, np.maximum(log_l, log_r)


def windowed_stereo_correlation(
    left: np.ndarray,
    right: np.ndarray,
    samplerate: int,
    window_ms: int = DEFAULT_STEREO_WINDOW_MS,
    silence_rms: float = 1e-7,
    *,
    include_windows: bool = True,
) -> tuple[float, float, list[tuple[int, int, float, float]]]:
    """Whole-file AC Pearson correlation and mono fold-down loss, plus local readings.

    Local windows have inclusive sample bounds and their own silence gate.
    Correlation is undefined when either channel has zero variance. Loss can
    still be measured on a hard-panned signal and is infinite at a full null.
    """
    if (left.ndim != 1 or right.ndim != 1 or not left.size
            or left.size != right.size or samplerate <= 0):
        return math.nan, 0.0, []

    whole = _stereo_readings(left[None, :], right[None, :])
    results: list[tuple[int, int, float, float]] = []
    whole_corr, whole_loss = float(whole[0][0]), float(whole[1][0])
    if not include_windows:
        return whole_corr, whole_loss, results

    width = max(8, int(samplerate * window_ms / 1000))
    start = 0
    while start < left.size:
        # Group equal-length windows. The final partial window uses only its real frames.
        size = min(width, left.size - start)
        count = min(max(1, 65536 // size), (left.size - start) // size)
        end = start + count * size
        if size == left.size:
            readings = whole
        else:
            readings = _stereo_readings(left[start:end].reshape(count, size),
                                        right[start:end].reshape(count, size))
        corr, loss, log_energy = readings
        gate = 2 * math.log(silence_rms) + math.log(size) if silence_rms > 0 else -math.inf
        for pos, c, db, energy in zip(range(start, end, size), corr, loss, log_energy):
            if energy < gate:
                c = db = math.nan
            results.append((pos, pos + size - 1, float(c), float(db)))
        start = end
    return whole_corr, whole_loss, results
