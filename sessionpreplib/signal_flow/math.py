"""DAW-agnostic gain-staging math, pan law, buffer normalization, measurement, and track staging resolver."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..audio import (
    INACTIVE_NORMALIZATION_CLASSIFICATIONS,
    NOISE_FLOOR_PEAK_DB,
    compute_peak_capped_gain,
    get_peak_db,
    resolve_rms_anchor,
)
from ..daw_processor import resolve_transfer_audio
from ..detectors.silence import SilenceDetector
from ..models import SessionContext, TrackContext
from .models import (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    MIN_SIGNAL_LIN,
    PROTOOLS_GAIN_LIMITS,
    VALID_GAIN_MODES,
    HostGainLimits,
    MixerTrack,
    TrackStagingPlan,
)


def pan_gains(pan: float | None, pan_law_db: float = -3.0) -> tuple[float, float]:
    """Compute ``(left_gain, right_gain)`` for a pan value in ``-100..+100``.

    Uses a 0 dB balance law at ``pan_law_db >= 0.0`` and a continuous raised-trigonometric
    law for negative pan depths (exact center gain ``10**(pan_law_db/20)`` at ``pan=0``,
    constant power at ``-3.01 dB``, constant voltage at ``-6.02 dB``, zero opposite bleed
    at ``-100`` / ``+100``).
    """
    p = 0.0 if pan is None else max(-100.0, min(100.0, float(pan)))
    u = (p + 100.0) / 200.0
    if u <= 0.0:
        return 1.0, 0.0
    if u >= 1.0:
        return 0.0, 1.0
    depth = float(pan_law_db)
    if depth >= 0.0:
        return min(1.0, 2.0 * (1.0 - u)), min(1.0, 2.0 * u)
    exponent = -depth / (10.0 * math.log10(2.0))
    angle = 0.5 * math.pi * u
    return math.cos(angle) ** exponent, math.sin(angle) ** exponent


def _clamp_finite(db: float, lo: float, hi: float) -> float:
    try:
        val = float(db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(lo, min(hi, val))


def clamp_clip_gain(db: float, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> float:
    return _clamp_finite(db, limits.clip_gain_min_db, limits.clip_gain_max_db)


def clamp_baked_gain(db: float) -> float:
    return _clamp_finite(db, -144.0, 144.0)


def clamp_leaf_trim_gain(db: float, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> float:
    return _clamp_finite(db, limits.leaf_trim_min_db, limits.leaf_trim_max_db)


def clamp_fader_db(db: float, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> float:
    return round(_clamp_finite(db, limits.fader_min_db, limits.fader_max_db), 1)


def db_to_lin(db: float, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> float:
    try:
        val = float(db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val) or val <= limits.fader_min_db:
        return 0.0
    return float(10.0 ** (min(200.0, val) / 20.0))


def round_opt(val: float | None) -> float | None:
    return round(float(val), 2) if (val is not None and math.isfinite(float(val))) else None


def empty_like_format(is_mono: bool) -> np.ndarray:
    return np.zeros(0, dtype=np.float64) if is_mono else np.zeros((0, 2), dtype=np.float64)


def has_audible_energy(buf: np.ndarray) -> bool:
    return buf.size > 0 and float(np.max(np.abs(buf))) > MIN_SIGNAL_LIN


def normalize_track_buffer(samples: np.ndarray, is_mono: bool) -> np.ndarray | None:
    """Validate finite samples and shape to 1-D mono ``(N,)`` or 2-D stereo ``(N, 2)``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return empty_like_format(is_mono)
    if not np.all(np.isfinite(arr)):
        return None
    if is_mono:
        if arr.ndim == 1:
            return arr.copy()
        if arr.shape[1] == 1:
            return arr[:, 0].copy()
        return 0.5 * (arr[:, 0] + arr[:, 1])
    if arr.ndim == 1:
        return np.column_stack((arr, arr))
    if arr.shape[1] == 1:
        col = arr[:, 0]
        return np.column_stack((col, col))
    return arr[:, :2].copy()


def pan_stereo(stereo: np.ndarray, pan_left: float | None, pan_right: float | None, pan_law_db: float) -> np.ndarray:
    pl = -100.0 if pan_left is None else float(pan_left)
    pr = 100.0 if pan_right is None else float(pan_right)
    if pl == -100.0 and pr == 100.0:
        return stereo
    ll, lr = pan_gains(pl, pan_law_db)
    rl, rr = pan_gains(pr, pan_law_db)
    out = np.empty_like(stereo)
    out[:, 0] = stereo[:, 0] * ll + stereo[:, 1] * rl
    out[:, 1] = stereo[:, 0] * lr + stereo[:, 1] * rr
    return out


def adapt_for_target(
    buf: np.ndarray,
    src_trk: MixerTrack,
    dst_trk: MixerTrack,
    pan_law_db: float,
) -> np.ndarray:
    """Apply output panning and channel adaptation from *src_trk* into *dst_trk*."""
    src_mono = buf.ndim == 1
    dst_mono = dst_trk.format == "Mono"
    if src_mono and dst_mono:
        return buf
    if src_mono and not dst_mono:
        g_l, g_r = pan_gains(src_trk.pan_left, pan_law_db)
        out = np.empty((buf.shape[0], 2), dtype=np.float64)
        out[:, 0] = buf * g_l
        out[:, 1] = buf * g_r
        return out
    panned = pan_stereo(buf, src_trk.pan_left, src_trk.pan_right, pan_law_db)
    if dst_mono:
        return 0.5 * (panned[:, 0] + panned[:, 1])
    return panned


def resample_if_needed(buf: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr or buf.shape[0] == 0:
        return buf
    gcd = math.gcd(src_sr, dst_sr)
    return np.asarray(resample_poly(buf, dst_sr // gcd, src_sr // gcd, axis=0), dtype=np.float64)


def accumulate_into(acc: np.ndarray | None, contrib: np.ndarray) -> np.ndarray:
    if contrib.shape[0] == 0:
        return acc if acc is not None else contrib
    if acc is None or acc.shape[0] == 0:
        return contrib.copy()
    if contrib.shape[0] == acc.shape[0]:
        acc += contrib
        return acc
    target_len = max(acc.shape[0], contrib.shape[0])
    if acc.shape[0] < target_len:
        shape = (target_len,) if acc.ndim == 1 else (target_len, acc.shape[1])
        grown = np.zeros(shape, dtype=np.float64)
        grown[: acc.shape[0]] = acc
        acc = grown
    acc[: contrib.shape[0]] += contrib
    return acc


def measure_buffer(
    buf: np.ndarray,
    samplerate: int,
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    target_peak_db: float,
    target_rms_db: float,
) -> tuple[float | None, float | None, str, str | None, str | None]:
    """Measure unrounded float64 peak/RMS and governing constraint without masking NaN/Inf."""
    if buf.size == 0:
        return None, None, "Silent", None, None
    if not np.all(np.isfinite(buf)):
        return None, None, "Error", None, "Non-finite samples in signal-flow buffer"
    peak = float(np.max(np.abs(buf)))
    if peak <= 0.0:
        return None, None, "Silent", None, None
    channels = 1 if buf.ndim == 1 else buf.shape[1]
    temp = TrackContext(
        filename="",
        filepath="",
        audio_data=buf,
        samplerate=samplerate,
        channels=channels,
        total_samples=buf.shape[0],
        bitdepth="64-bit Float",
        subtype="DOUBLE",
        duration_sec=buf.shape[0] / float(samplerate),
    )
    sil = silence_det.analyze(temp)
    if sil.data.get("is_silent"):
        return float(20.0 * math.log10(peak)), None, "Silent", None, None
    peak_db = get_peak_db(temp)
    anchor = resolve_rms_anchor(
        temp,
        int(cfg.get("window", 400)),
        str(cfg.get("stereo_mode", "avg")),
        float(cfg.get("gate_relative_db", 40.0)),
        str(cfg.get("rms_anchor", "percentile")),
        float(cfg.get("rms_percentile", 95.0)),
    )
    rms_db = float(anchor["rms_anchor_db"])
    governing, _gain, _clamped = compute_peak_capped_gain(
        peak_db, rms_db, target_peak_db, target_rms_db,
    )
    if governing == "Silent":
        return None, None, "Silent", None, None
    if governing == "NoiseFloor":
        return peak_db, (rms_db if math.isfinite(rms_db) else None), "Noise Floor", None, None
    return peak_db, rms_db, governing, governing, None


def compute_unconstrained_bus_gain(
    pre_peak_db: float | None,
    pre_rms_db: float | None,
    target_peak_db: float,
    target_rms_db: float,
    *,
    max_gain_db: float = 24.0,
) -> float:
    if pre_peak_db is None or not math.isfinite(pre_peak_db) or pre_peak_db <= NOISE_FLOOR_PEAK_DB:
        return 0.0
    mode = "Peak" if (pre_rms_db is None or not math.isfinite(pre_rms_db)) else "Auto"
    _gov, gain, _clamped = compute_peak_capped_gain(
        pre_peak_db,
        pre_rms_db if pre_rms_db is not None else float("-inf"),
        target_peak_db,
        target_rms_db,
        max_gain_db=max_gain_db,
        mode=mode,
    )
    return float(gain) if math.isfinite(gain) else 0.0


def compute_bus_target_fader(
    pre_peak_db: float | None,
    pre_rms_db: float | None,
    target_peak_db: float,
    target_rms_db: float,
    *,
    attenuate_only: bool,
    max_fader_db: float,
    base_fader_db: float = 0.0,
    max_gain_db: float = 24.0,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> float:
    raw = compute_unconstrained_bus_gain(
        pre_peak_db, pre_rms_db, target_peak_db, target_rms_db, max_gain_db=max_gain_db,
    )
    upper = base_fader_db if attenuate_only else max(base_fader_db, min(limits.fader_max_db, max_fader_db))
    return round(max(limits.fader_min_db, min(upper, raw)), 1)


def planned_normalization_gains(session: SessionContext, track: TrackContext) -> tuple[float, float]:
    """Return ``(forward_gain_db, inverse_fader_db)`` planned for *track*."""
    proc_id = "bimodal_normalize"
    source = next((t for t in session.tracks if t.filename == track.filename), track)
    enabled = bool(session.config.get(f"{proc_id}_enabled", True))
    skipped = proc_id in track.processor_skip or proc_id in source.processor_skip
    result = track.processor_results.get(proc_id) or source.processor_results.get(proc_id)
    if result is not None:
        if enabled and not skipped and not result.error and result.classification not in INACTIVE_NORMALIZATION_CLASSIFICATIONS:
            gain = float(result.gain_db)
            fader = float(result.data.get("fader_offset", -gain))
            if math.isfinite(gain) and math.isfinite(fader):
                return gain, fader
        return 0.0, 0.0
    manual = track.manual_gain_db if track.manual_gain_db is not None else source.manual_gain_db
    if manual is not None and math.isfinite(float(manual)):
        return float(manual), -float(manual)
    return 0.0, 0.0


def resolve_track_staging(
    session: SessionContext,
    track: TrackContext,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
    *,
    limits: HostGainLimits | None = None,
) -> TrackStagingPlan:
    """Resolve transfer media plus pre-insert (Clip Gain or Trim) and Track Fader staging.

    Guarantees normalization gain is never double-applied or cancelled pre-insert:
      - Un-normalized audio:
        * ``clip_gain_and_fader`` / ``clip_gain_unity``: Clip Gain ``+gain_db`` (+ optional inverse fader).
        * ``trim_and_fader`` / ``trim_unity``: Leaf Trim ``+gain_db`` (+ optional inverse fader).
        * ``fader_only`` / ``none``: ``0.0 dB`` pre-insert and ``0.0 dB`` fader.
      - Prepared (already gain-baked) WAV:
        * Clip Gain and Leaf Trim stay ``0.0 dB``; Track Fader carries ``prepared_fader_db``
          in ``clip_gain_and_fader``, ``trim_and_fader``, and ``fader_only`` modes.
      - When *limits* is supplied on raw audio, pre-insert gain is clamped to the host's
        clip-gain or leaf-trim range and the inverse fader is symmetrically adjusted so
        rough-mix balance remains exact.
    """
    mode = gain_mode if gain_mode in VALID_GAIN_MODES else GAIN_MODE_CLIP_AND_FADER
    forward_gain, planned_fader = planned_normalization_gains(session, track)
    media = resolve_transfer_audio(session, track)
    proc_path = track.processed_filepath
    is_prepared = proc_path is not None and bool(proc_path) and (
        os.path.normcase(os.path.realpath(media.filepath))
        == os.path.normcase(os.path.realpath(proc_path))
    )
    baked = forward_gain if is_prepared else 0.0
    if mode == GAIN_MODE_NONE:
        return TrackStagingPlan(media.filepath, baked_gain_db=baked, prepared_fader_db=media.fader_db)
    if mode == GAIN_MODE_FADER_ONLY:
        return TrackStagingPlan(
            media.filepath,
            fader_db=media.fader_db if is_prepared else 0.0,
            baked_gain_db=baked,
            prepared_fader_db=media.fader_db,
        )
    use_trim = mode in (GAIN_MODE_TRIM_AND_FADER, GAIN_MODE_TRIM_UNITY)
    preserve_balance = mode in (GAIN_MODE_CLIP_AND_FADER, GAIN_MODE_TRIM_AND_FADER)
    pre_db = 0.0 if is_prepared else forward_gain
    eff_planned_fader = planned_fader
    if not is_prepared and limits is not None:
        clamped_pre = clamp_leaf_trim_gain(pre_db, limits) if use_trim else clamp_clip_gain(pre_db, limits)
        if abs(clamped_pre - pre_db) > 1e-9:
            eff_planned_fader = eff_planned_fader + (pre_db - clamped_pre)
            pre_db = clamped_pre
    fader_db = (media.fader_db if is_prepared else eff_planned_fader) if preserve_balance else 0.0
    return TrackStagingPlan(
        filepath=media.filepath,
        clip_gain_db=0.0 if use_trim else pre_db,
        trim_gain_db=pre_db if use_trim else 0.0,
        fader_db=fader_db,
        baked_gain_db=baked,
        prepared_fader_db=media.fader_db,
    )
