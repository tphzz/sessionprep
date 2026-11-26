from __future__ import annotations

import html
import math
import re
from typing import Any

import numpy as np

from ..audio import (
    INACTIVE_NORMALIZATION_CLASSIFICATIONS,
    NOISE_FLOOR_PEAK_DB,
    compute_peak_capped_gain,
    db_to_linear,
    dbfs_offset,
    find_governing_and_next_peak,
    is_silent,
    legacy_patterns_to_regex,
    load_track,
    measure_true_peak_db,
    migrate_legacy_override,
    parse_rms_anchor_override,
    resolve_rms_anchor,
)
from ..models import ParamSpec, ProcessorResult, TrackContext
from ..processor import AudioProcessor, PRIORITY_NORMALIZE
from ..repairs import render_corrected

NEXT_PEAK_EXCLUSION_MS = 100.0
# Covers +3.01 dB fs/4 sine ISPs and +4.46 dB 4-sample transient ISPs against -1.0 dBTP Amber
# while keeping default -6.0 dBFS peak-ceiling tracks lazy-skipped.
TRUE_PEAK_CHECK_THRESHOLD_DB = -5.5

_PRESERVED_SOURCE_KEYS = (
    "detected_peak_db",
    "detected_rms_db",
    "rms_max_db",
    "rms_anchor_label",
    "effective_window_ms",
    "governing_peak_sample",
    "governing_peak_time_s",
    "next_peak_db",
    "next_peak_sample",
    "next_peak_time_s",
    "next_peak_exclusion_ms",
    "source_true_peak_db",
    "_measured_window_ms",
    "_measured_stereo_mode",
    "_measured_gate_relative_db",
    "_measured_anchor_spec",
)


def _safe_db(value: Any, default: float = float("-inf")) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _matches_force_peak(filename: str, pattern: str | list[str]) -> bool:
    expr = legacy_patterns_to_regex(pattern)
    if not expr:
        return False
    try:
        return bool(re.search(expr, filename, re.IGNORECASE))
    except re.error:
        return expr.lower() in filename.lower()


def _format_peak_timestamp(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "00:00.00"
    total_cs = round(seconds * 100.0)
    mins, rem_cs = divmod(total_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"{mins:02d}:{secs:02d}.{cs:02d}"


def compute_peak_warning_level(post_peak_db: float, post_true_peak_db: float | None) -> str:
    """Classify post-gain headroom risk from sample peak and optional lazy true peak."""
    peaks = [p for p in (post_peak_db, post_true_peak_db) if p is not None and math.isfinite(p)]
    if not peaks:
        return "none"
    worst = max(peaks)
    if worst > 0.0:
        return "red"
    if worst > -1.0:
        return "amber"
    return "none"


def _resolve_source_audio_for_true_peak(
    track: TrackContext | None,
    source_audio: np.ndarray | None,
    *,
    allow_disk_reload: bool,
) -> tuple[np.ndarray | None, int]:
    if source_audio is not None and track is not None:
        return source_audio, track.samplerate
    if track is None:
        return None, 0
    recipe = getattr(track, "repair_recipe", None)
    repair_active = bool(recipe and getattr(recipe, "active", False))
    if track.audio_data is not None:
        if repair_active:
            return render_corrected(track.audio_data, track.samplerate, recipe), track.samplerate
        return track.audio_data, track.samplerate
    if allow_disk_reload and track.filepath:
        try:
            loaded = load_track(track.filepath)
        except Exception:
            return None, 0
        if loaded.audio_data is None:
            return None, 0
        audio = (
            render_corrected(loaded.audio_data, loaded.samplerate, recipe)
            if repair_active
            else loaded.audio_data
        )
        return audio, loaded.samplerate
    return None, 0


def update_post_gain_metrics(
    result: ProcessorResult,
    track: TrackContext | None,
    effective_gain_db: float,
    *,
    source_audio: np.ndarray | None = None,
    allow_disk_reload: bool = True,
) -> None:
    """Recompute post-gain sample peak, RMS, lazy BS.1770 true peak, and warning state."""
    d = result.data
    peak_db = _safe_db(d.get("detected_peak_db"))
    rms_db = _safe_db(d.get("detected_rms_db"))
    if not math.isfinite(effective_gain_db):
        d["post_peak_db"] = float("nan")
        d["post_rms_db"] = float("nan")
        d["post_true_peak_db"] = None
        d["true_peak_unverified"] = False
        d["peak_warning_level"] = "red"
        return

    post_peak = peak_db + effective_gain_db if math.isfinite(peak_db) else float("-inf")
    post_rms = rms_db + effective_gain_db if math.isfinite(rms_db) else float("-inf")
    post_tp: float | None = None
    tp_unverified = False
    if math.isfinite(post_peak) and post_peak > TRUE_PEAK_CHECK_THRESHOLD_DB:
        src_tp = d.get("source_true_peak_db")
        if src_tp is None:
            pcm, sr = _resolve_source_audio_for_true_peak(
                track, source_audio, allow_disk_reload=allow_disk_reload,
            )
            if pcm is not None and sr > 0:
                src_tp = measure_true_peak_db(pcm, sr)
                d["source_true_peak_db"] = src_tp
            else:
                tp_unverified = True
        if src_tp is not None and math.isfinite(float(src_tp)):
            post_tp = float(src_tp) + effective_gain_db

    d["post_peak_db"] = float(post_peak)
    d["post_rms_db"] = float(post_rms)
    d["post_true_peak_db"] = post_tp
    d["true_peak_unverified"] = tp_unverified
    d["peak_warning_level"] = compute_peak_warning_level(post_peak, post_tp)


class BimodalNormalizeProcessor(AudioProcessor):
    """Peak-Capped RMS track normalizer (pure broadband scalar gain).

    Latency: 0 samples (no filtering, look-ahead, or delay).
    Worst-case output for full-scale (0 dBFS) input:
      - ``Auto`` / ``Peak`` modes: held ``<= target_peak`` (``<= 0.0 dBFS``).
      - ``RMS`` override: up to ``+max_gain_db`` (max ``+36.0 dB``) sustained;
        guarded by lazy oversampled true-peak warnings (``> -1.0`` / ``> 0.0 dBTP``)
        and Prepare output-headroom checks.
      - Noise Floor Hold: tracks at or below ``-70.0 dBFS`` peak hold at ``0.0 dB``.
    """

    id = "bimodal_normalize"
    name = "Peak-Capped RMS"
    shorthand = "BN"
    priority = PRIORITY_NORMALIZE

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="target_rms", type=(int, float), default=-18.0,
                min=-80.0, max=0.0,
                label="Target RMS (dBFS)",
                description=(
                    "Target operating RMS level (anchored to gated loud sections). "
                    "Dense tracks and busses hit this level."
                ),
            ),
            ParamSpec(
                key="target_peak", type=(int, float), default=-6.0,
                min=-80.0, max=0.0,
                label="Peak ceiling (dBFS)",
                description=(
                    "Maximum allowed sample peak level. "
                    "High-crest tracks stop here before reaching target RMS."
                ),
            ),
            ParamSpec(
                key="max_gain_db", type=(int, float), default=24.0,
                min=0.0, max=36.0,
                label="Max boost (dB)",
                description=(
                    "Maximum automatic boost applied to quiet stems. "
                    "Tracks at or below -70.0 dBFS peak hold at 0.0 dB."
                ),
            ),
            ParamSpec(
                key="force_peak", type=str, default="",
                label="Force peak pattern",
                description=(
                    "Optional regex for track names that always target the peak ceiling "
                    "(e.g. kick|snare|tom)."
                ),
            ),
        ]

    def configure(self, config: dict[str, Any]) -> None:
        super().configure(config)
        self.target_rms = float(config.get("target_rms", -18.0))
        self.target_peak = float(config.get("target_peak", -6.0))
        self.max_gain_db = float(config.get("max_gain_db", 24.0))
        self.force_peak = config.get("force_peak", "")
        if not self.force_peak and config.get("force_transient"):
            self.force_peak = config["force_transient"]
        self.window_ms = int(config.get("window", 400))
        self.stereo_mode = str(config.get("stereo_mode", "avg"))
        self.rms_anchor_mode = str(config.get("rms_anchor", "percentile"))
        self.rms_percentile = float(config.get("rms_percentile", 95.0))
        self.gate_relative_db = float(config.get("gate_relative_db", 40.0))
        self._db_offset = dbfs_offset(config)
        self._rms_anchor_label = (
            "max" if self.rms_anchor_mode == "max" else f"p{self.rms_percentile:g}"
        )

    def _no_gain_result(
        self,
        classification: str,
        *,
        mode: str = "Auto",
        governing: str = "None",
        method: str = "None",
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        data: dict[str, Any] = {
            "gain_db_individual": 0.0,
            "mode": mode,
            "governing": governing,
            "is_manual_override": mode in ("Peak", "RMS", "Skip"),
            "forced_by_pattern": False,
            "rms_anchor_active": False,
            "boost_clamped": False,
            "noise_floor_held": governing == "NoiseFloor",
            "target_peak": self.target_peak,
            "target_rms": self.target_rms,
            "max_gain_db": self.max_gain_db,
            "post_peak_db": float("-inf"),
            "post_rms_db": float("-inf"),
            "post_true_peak_db": None,
            "peak_warning_level": "none",
            "stereo_mode": self.stereo_mode,
            "rms_weighting": "unweighted",
        }
        if extra:
            data.update(extra)
        return ProcessorResult(
            processor_id=self.id,
            gain_db=0.0,
            classification=classification,
            method=method,
            data=data,
            error=error,
        )

    def _effective_anchor_spec(self, track: TrackContext) -> str:
        mode, pct = parse_rms_anchor_override(
            getattr(track, "rms_anchor_override", None), self.rms_anchor_mode, self.rms_percentile,
        )
        return "max" if mode == "max" else f"p{pct:g}"

    def _resolve_mode_and_gain(
        self,
        filename: str,
        override: str | None,
        peak_db: float,
        rms_db: float,
    ) -> tuple[str, str, float, bool, bool, bool, str, str]:
        forced_by_pattern = override is None and _matches_force_peak(filename, self.force_peak)
        if override == "Peak" or forced_by_pattern:
            mode, is_manual = "Peak", override == "Peak"
        elif override == "RMS":
            mode, is_manual = "RMS", True
        else:
            mode, is_manual = "Auto", False
        governing, gain, clamped = compute_peak_capped_gain(
            peak_db, rms_db, self.target_peak, self.target_rms,
            max_gain_db=self.max_gain_db, mode=mode,
        )
        if mode == "Peak":
            classification, method = "Peak", f"Peak → {self.target_peak:.1f} dBFS"
        elif mode == "RMS":
            classification, method = "RMS", f"RMS (override) → {self.target_rms:.1f} dBFS"
        elif governing == "Peak":
            classification, method = "Auto (Peak)", f"Peak (ceiling) → {self.target_peak:.1f} dBFS"
        else:
            classification, method = "Auto (RMS)", f"RMS → {self.target_rms:.1f} dBFS"
        return mode, governing, gain, clamped, is_manual, forced_by_pattern, classification, method

    def _prior_source_snapshot(self, track: TrackContext) -> dict[str, Any]:
        prior = track.processor_results.get(self.id)
        if prior is None:
            return {}
        return {k: prior.data[k] for k in _PRESERVED_SOURCE_KEYS if k in prior.data}

    def _measure_inputs(self, track: TrackContext) -> tuple[dict[str, Any], dict[str, Any], float | None]:
        if track.audio_data is not None:
            peak_info = find_governing_and_next_peak(
                track.audio_data, track.samplerate, NEXT_PEAK_EXCLUSION_MS,
            )
            anchor = resolve_rms_anchor(
                track, self.window_ms, self.stereo_mode,
                self.gate_relative_db, self.rms_anchor_mode, self.rms_percentile,
            )
            return peak_info, anchor, None

        prior = track.processor_results.get(self.id)
        if prior and "detected_peak_db" in prior.data and "detected_rms_db" in prior.data:
            d = prior.data
            if (
                ("_measured_window_ms" in d and d["_measured_window_ms"] != self.window_ms)
                or ("_measured_stereo_mode" in d and d["_measured_stereo_mode"] != self.stereo_mode)
                or ("_measured_gate_relative_db" in d and d["_measured_gate_relative_db"] != self.gate_relative_db)
                or ("_measured_anchor_spec" in d and d["_measured_anchor_spec"] != self._effective_anchor_spec(track))
            ):
                raise ValueError("Cached Peak-Capped RMS metrics do not match current analysis settings")
            peak_info = {
                "peak_db": _safe_db(d["detected_peak_db"]),
                "peak_sample": d.get("governing_peak_sample"),
                "peak_time_s": d.get("governing_peak_time_s"),
                "next_peak_db": d.get("next_peak_db"),
                "next_peak_sample": d.get("next_peak_sample"),
                "next_peak_time_s": d.get("next_peak_time_s"),
                "exclusion_ms": d.get("next_peak_exclusion_ms", NEXT_PEAK_EXCLUSION_MS),
            }
            anchor = {
                "rms_anchor_db": _safe_db(d["detected_rms_db"]),
                "rms_max_db": _safe_db(d.get("rms_max_db", d["detected_rms_db"])),
                "rms_anchor_label": d.get("rms_anchor_label", self._rms_anchor_label),
                "effective_window_ms": float(d.get("effective_window_ms", self.window_ms)),
            }
            return peak_info, anchor, d.get("source_true_peak_db")
        raise ValueError("Audio samples unavailable for Peak-Capped RMS measurement")

    def _build_noise_floor_result(
        self,
        peak_info: dict[str, Any],
        anchor: dict[str, Any],
        peak_db: float,
        rms_anchor_db: float,
        anchor_label: str,
        anchor_spec: str,
    ) -> ProcessorResult:
        return self._no_gain_result(
            "Noise Floor",
            mode="Auto",
            governing="NoiseFloor",
            method="Noise Floor Hold",
            extra={
                "detected_peak_db": peak_db,
                "detected_rms_db": rms_anchor_db,
                "rms_anchor_label": anchor_label,
                "governing_peak_sample": peak_info["peak_sample"],
                "governing_peak_time_s": peak_info["peak_time_s"],
                "next_peak_db": peak_info["next_peak_db"],
                "next_peak_sample": peak_info["next_peak_sample"],
                "next_peak_time_s": peak_info["next_peak_time_s"],
                "next_peak_exclusion_ms": peak_info["exclusion_ms"],
                "effective_window_ms": anchor["effective_window_ms"],
                "post_peak_db": peak_db,
                "post_rms_db": rms_anchor_db,
                "_measured_window_ms": self.window_ms,
                "_measured_stereo_mode": self.stereo_mode,
                "_measured_gate_relative_db": self.gate_relative_db,
                "_measured_anchor_spec": anchor_spec,
            },
        )

    def process(self, track: TrackContext) -> ProcessorResult:
        silence_result = track.detector_results.get("silence")
        if silence_result and silence_result.data.get("is_silent"):
            return self._no_gain_result("Silent", governing="Silent")
        if track.audio_data is not None:
            if not np.all(np.isfinite(track.audio_data)):
                return self._no_gain_result("Error", governing="Error", error="Source contains non-finite samples")
            if is_silent(track):
                return self._no_gain_result("Silent", governing="Silent")

        override = migrate_legacy_override(track.classification_override)
        if override == "Skip":
            return self._no_gain_result("Skip", mode="Skip", governing="Skip", extra=self._prior_source_snapshot(track))

        peak_info, anchor, cached_src_tp = self._measure_inputs(track)
        peak_db = float(peak_info["peak_db"])
        rms_anchor_db = float(anchor["rms_anchor_db"])
        anchor_label = str(anchor.get("rms_anchor_label", self._rms_anchor_label))
        anchor_spec = self._effective_anchor_spec(track)
        if math.isnan(peak_db) or peak_db == float("inf") or math.isnan(rms_anchor_db) or rms_anchor_db == float("inf"):
            return self._no_gain_result("Error", governing="Error", error="Non-finite measured peak or RMS")
        if not math.isfinite(peak_db) or peak_db <= NOISE_FLOOR_PEAK_DB or not math.isfinite(rms_anchor_db):
            return self._build_noise_floor_result(peak_info, anchor, peak_db, rms_anchor_db, anchor_label, anchor_spec)

        gain_for_peak, gain_for_rms = self.target_peak - peak_db, self.target_rms - rms_anchor_db
        mode, governing, gain, boost_clamped, is_manual, forced_pat, classification, method = (
            self._resolve_mode_and_gain(track.filename, override, peak_db, rms_anchor_db)
        )
        if not math.isfinite(gain):
            return self._no_gain_result("Error", governing="Error", error="Computed normalization gain is non-finite")

        data: dict[str, Any] = {
            "gain_db_individual": float(gain), "mode": mode, "governing": governing,
            "is_manual_override": is_manual, "forced_by_pattern": forced_pat,
            "rms_anchor_active": governing == "RMS", "boost_clamped": boost_clamped,
            "noise_floor_held": False,
            "target_peak": self.target_peak, "target_rms": self.target_rms, "max_gain_db": self.max_gain_db,
            "detected_peak_db": peak_db, "detected_rms_db": rms_anchor_db,
            "rms_max_db": float(anchor.get("rms_max_db", rms_anchor_db)),
            "rms_anchor_label": anchor_label,
            "effective_window_ms": float(anchor.get("effective_window_ms", self.window_ms)),
            "governing_peak_sample": peak_info["peak_sample"], "governing_peak_time_s": peak_info["peak_time_s"],
            "next_peak_db": peak_info["next_peak_db"], "next_peak_sample": peak_info["next_peak_sample"],
            "next_peak_time_s": peak_info["next_peak_time_s"],
            "next_peak_exclusion_ms": float(peak_info["exclusion_ms"]),
            "gain_for_peak": float(gain_for_peak), "gain_for_rms": float(gain_for_rms),
            "effective_crest_db": float(peak_db - rms_anchor_db),
            "stereo_mode": self.stereo_mode, "rms_weighting": "unweighted",
            "_measured_window_ms": self.window_ms, "_measured_stereo_mode": self.stereo_mode,
            "_measured_gate_relative_db": self.gate_relative_db, "_measured_anchor_spec": anchor_spec,
        }
        if cached_src_tp is not None:
            data["source_true_peak_db"] = cached_src_tp
        result = ProcessorResult(
            processor_id=self.id, gain_db=float(gain), classification=classification, method=method, data=data,
        )
        # Pass track.audio_data directly as source_audio because Pipeline._plan_track invokes
        # process(gain_track) where gain_track.audio_data is already the post-repair PCM.
        update_post_gain_metrics(result, track, float(gain), source_audio=track.audio_data)
        return result

    def render_html(
        self,
        result: ProcessorResult,
        track: Any = None,
        *,
        verbose: bool = False,
        compact: bool = False,
        narrow: bool = False,
    ) -> str:
        """Render Peak-Capped RMS summary, 5-column table, governing peak, and warnings."""
        d = result.data
        cls_text = result.classification or "Unknown"
        metadata, muted = "#aaaaaa", "#888888"
        if cls_text in INACTIVE_NORMALIZATION_CLASSIFICATIONS:
            return _render_inactive_html(cls_text, d, result.error, metadata, muted)

        rms_off = self._db_offset
        mode = str(d.get("mode", "Auto"))
        governing = str(d.get("governing", "Peak" if "Peak" in cls_text else "RMS"))
        type_color = "#cc77ff" if governing == "Peak" else "#44cccc"

        header_label = _summary_mode_label(mode, governing, d)
        post_pk = _safe_db(d.get("post_peak_db"), _safe_db(d.get("detected_peak_db")) + result.gain_db)
        post_rms = _safe_db(d.get("post_rms_db"), _safe_db(d.get("detected_rms_db")) + result.gain_db)
        summary_line = (
            f'<div><span style="color:{type_color}; font-weight:bold;">{header_label}</span>'
            f' &nbsp;&middot;&nbsp; <b>{_fmt_rel_db(result.gain_db)} dB</b>'
            f' &nbsp;&middot;&nbsp; <span style="color:{metadata};">'
            f'Result: peak {_fmt_abs_db(post_pk, 0.0)} dBFS &middot; '
            f'RMS {_fmt_abs_db(post_rms, rms_off)} dBFS</span></div>'
        )
        table_html = _build_result_table_html(
            result, d, rms_off, mode, governing, type_color, metadata, compact=compact, narrow=narrow,
        )
        extras = _build_extras_html(result, d, track, rms_off, metadata, verbose=verbose)
        return summary_line + table_html + extras

    def apply(self, track: TrackContext, result: ProcessorResult) -> np.ndarray:
        assert track.audio_data is not None
        if not math.isfinite(result.gain_db):
            raise ValueError(f"Non-finite normalization gain: {result.gain_db}")
        if result.classification in INACTIVE_NORMALIZATION_CLASSIFICATIONS or result.gain_db == 0.0:
            return track.audio_data
        try:
            linear_gain = db_to_linear(float(result.gain_db))
        except OverflowError as exc:
            raise ValueError(f"Normalization gain {result.gain_db} dB overflows linear scale") from exc
        if not math.isfinite(linear_gain):
            raise ValueError(f"Normalization gain {result.gain_db} dB overflows linear scale")
        out = np.asarray(track.audio_data, dtype=np.float64) * linear_gain
        tiny = np.finfo(np.float64).tiny
        out = np.where(np.abs(out) < tiny, 0.0, out)
        if not np.all(np.isfinite(out)):
            raise ValueError("Normalization produced non-finite output samples")
        return out


def _fmt_inf(val: float) -> str:
    return "+&infin;" if val > 0 else "&minus;&infin;"


def _html_minus(text: str) -> str:
    return "&minus;" + text[1:] if text.startswith("-") else text


def _fmt_abs_db(val: float, offset: float, *, plus_sign: bool = False) -> str:
    if not math.isfinite(val):
        return _fmt_inf(val)
    v = val + offset
    fmt = f"{v:+.1f}" if (plus_sign and v > 0.0) else f"{v:.1f}"
    return _html_minus(fmt)


def _fmt_rel_db(val: float) -> str:
    if not math.isfinite(val):
        return _fmt_inf(val)
    return _html_minus(f"{val:+.1f}")


def _render_inactive_html(
    cls_text: str,
    d: dict[str, Any],
    error: str | None,
    metadata: str,
    muted: str,
) -> str:
    if cls_text == "Noise Floor":
        pk = _safe_db(d.get("detected_peak_db"))
        floor_txt = _html_minus(f"{NOISE_FLOOR_PEAK_DB:.1f}")
        return (
            f'<div><span style="color:#ffaa00; font-weight:bold;">Noise Floor Hold</span>'
            f' &nbsp;&middot;&nbsp; <span style="color:{metadata};">'
            f'Peak {_fmt_abs_db(pk, 0.0)} dBFS (&le; {floor_txt} dBFS) &mdash; held at 0.0 dB</span></div>'
        )
    if cls_text == "Error":
        err_html = html.escape(str(error or "Failed"))
        return f'<div><span style="color:#ff4444; font-weight:bold;">Error</span> &middot; {err_html}</div>'
    return (
        f'<div><span style="color:{muted}; font-weight:bold;">{html.escape(cls_text)}</span>'
        f' &nbsp;&middot;&nbsp; <span style="color:{metadata};">No normalization</span></div>'
    )


def _summary_mode_label(mode: str, governing: str, d: dict[str, Any]) -> str:
    if d.get("group_limited"):
        src = html.escape(str(d.get("group_governing_track", "group")))
        gc = html.escape(str(d.get("group_governing_constraint", governing)).lower())
        prefix = f"{mode} (Grp)" if mode in ("Peak", "RMS") else "Auto (Grp)"
        return f"{prefix} &mdash; {src} ({gc})"
    if mode == "RMS":
        return "RMS (override)"
    if mode == "Peak":
        return "Peak (pattern)" if d.get("forced_by_pattern") else "Peak (override)"
    return f"Auto ({html.escape(governing)})"


def _fmt_peak_warn_cell(post_pk: float, post_tp: float | None, warn: str) -> str:
    color = "#ff4444" if warn == "red" else "#ffaa00"
    limit_val = 0.0 if warn == "red" else -1.0
    limit_str = "0" if warn == "red" else "&minus;1"
    if math.isfinite(post_pk) and post_pk > limit_val:
        txt = f"{_fmt_abs_db(post_pk, 0.0, plus_sign=True)} dBFS (&gt;{limit_str})"
    elif post_tp is not None and math.isfinite(post_tp):
        txt = f"{_fmt_abs_db(post_pk, 0.0)} dBFS ({_fmt_abs_db(post_tp, 0.0, plus_sign=True)} dBTP &gt;{limit_str})"
    else:
        txt = f"{_fmt_abs_db(post_pk, 0.0)} dBFS"
    return f'<span style="color:{color}; font-weight:600;">{txt}</span>'


def _fmt_delta_note(target: float, post: float, unit_noun: str, *, verb: str = "") -> str:
    delta = target - post
    if not math.isfinite(delta) or abs(delta) <= 0.05:
        return f"At {unit_noun}"
    prefix = f"{verb} " if verb else ""
    if delta > 0.05:
        return f"{prefix}{delta:.1f} dB below {unit_noun}"
    return f"{prefix}{-delta:.1f} dB above {unit_noun}"


def _result_cell_notes(
    result: ProcessorResult,
    d: dict[str, Any],
    mode: str,
    governing: str,
    type_color: str,
    metadata: str,
) -> tuple[str, str]:
    tgt_peak = _safe_db(d.get("target_peak"), -6.0)
    tgt_rms = _safe_db(d.get("target_rms"), -18.0)
    det_pk = _safe_db(d.get("detected_peak_db"))
    det_rms = _safe_db(d.get("detected_rms_db"))
    post_pk = _safe_db(d.get("post_peak_db"), det_pk + result.gain_db)
    post_rms = _safe_db(d.get("post_rms_db"), det_rms + result.gain_db)
    post_tp = _safe_db(d.get("post_true_peak_db"), float("nan"))
    post_tp_opt = post_tp if math.isfinite(post_tp) else None
    warn = str(d.get("peak_warning_level", "none"))
    clamped = bool(d.get("boost_clamped", False))
    max_g = _safe_db(d.get("max_gain_db"), 24.0)
    group_limited = bool(d.get("group_limited", False))

    if group_limited:
        pk_note = (
            _fmt_peak_warn_cell(post_pk, post_tp_opt, warn)
            if warn in ("red", "amber")
            else f'<span style="color:{metadata};">{_fmt_delta_note(tgt_peak, post_pk, "ceiling")}</span>'
        )
        rms_note = f'<span style="color:{metadata};">Group-limited &middot; {_fmt_delta_note(tgt_rms, post_rms, "target")}</span>'
        return pk_note, rms_note

    if mode == "RMS":
        pk_note = (
            _fmt_peak_warn_cell(post_pk, post_tp_opt, warn)
            if warn in ("red", "amber")
            else f'<span style="color:{metadata};">Ceiling bypassed</span>'
        )
        rms_txt = f"Clamped (+{max_g:.1f} dB max)" if clamped else "Governs"
        return pk_note, f'<span style="color:{type_color}; font-weight:600;">{rms_txt}</span>'

    if governing == "Peak":
        pk_note = (
            _fmt_peak_warn_cell(post_pk, post_tp_opt, warn)
            if warn in ("red", "amber")
            else f'<span style="color:{type_color}; font-weight:600;">'
                 f'{f"Clamped (+{max_g:.1f} dB max)" if clamped else "Governs"}</span>'
        )
        rms_note = f'<span style="color:{metadata};">{_fmt_delta_note(tgt_rms, post_rms, "target", verb="Lands")}</span>'
        return pk_note, rms_note

    pk_note = (
        _fmt_peak_warn_cell(post_pk, post_tp_opt, warn)
        if warn in ("red", "amber")
        else f'<span style="color:{metadata};">{_fmt_delta_note(tgt_peak, post_pk, "ceiling")}</span>'
    )
    rms_txt = f"Clamped (+{max_g:.1f} dB max)" if clamped else "Governs"
    return pk_note, f'<span style="color:{type_color}; font-weight:600;">{rms_txt}</span>'


def _build_result_table_html(
    result: ProcessorResult,
    d: dict[str, Any],
    rms_off: float,
    mode: str,
    governing: str,
    type_color: str,
    metadata: str,
    *,
    compact: bool,
    narrow: bool,
) -> str:
    tgt_peak = _safe_db(d.get("target_peak"), -6.0)
    tgt_rms = _safe_db(d.get("target_rms"), -18.0)
    anchor_label = str(d.get("rms_anchor_label", "p95"))
    rms_metric = f"RMS ({anchor_label})" if anchor_label else "RMS"
    peak_active = governing == "Peak" and not d.get("group_limited")
    rms_active = governing == "RMS" and not d.get("group_limited")
    pk_note, rms_note = _result_cell_notes(result, d, mode, governing, type_color, metadata)
    # AES17 offset applies ONLY to RMS readouts/targets; sample peaks stay unshifted.
    peak_values = (
        f"{_fmt_abs_db(_safe_db(d.get('detected_peak_db')), 0.0)} dBFS",
        f"{_fmt_abs_db(tgt_peak, 0.0)} dBFS",
        f"{_fmt_rel_db(_safe_db(d.get('gain_for_peak'), 0.0))} dB",
    )
    rms_values = (
        f"{_fmt_abs_db(_safe_db(d.get('detected_rms_db')), rms_off)} dBFS",
        f"{_fmt_abs_db(tgt_rms, rms_off)} dBFS",
        f"{_fmt_rel_db(_safe_db(d.get('gain_for_rms'), 0.0))} dB",
    )
    return _render_comparison_table(
        peak_values, rms_values, peak_active, rms_active,
        pk_note, rms_note, rms_metric, compact=compact, narrow=narrow,
    )


def _build_extras_html(
    result: ProcessorResult,
    d: dict[str, Any],
    track: Any,
    rms_off: float,
    metadata: str,
    *,
    verbose: bool,
) -> str:
    parts: list[str] = []
    warn = str(d.get("peak_warning_level", "none"))
    post_pk = _safe_db(d.get("post_peak_db"))
    post_tp_raw = d.get("post_true_peak_db")
    post_tp = _safe_db(post_tp_raw, float("nan"))
    tp_frag = f" ({_fmt_abs_db(post_tp, 0.0, plus_sign=True)} dBTP)" if math.isfinite(post_tp) else ""

    if warn == "red":
        if math.isfinite(post_pk) and post_pk > 0.0:
            parts.append(
                f'<div style="margin-top:8px; color:#ff4444; font-weight:600;">'
                f'&#9888; Peak reaches {_fmt_abs_db(post_pk, 0.0, plus_sign=True)} dBFS{tp_frag} '
                f'&mdash; over 0 dBFS into first insert; Prepare to fixed-point PCM would clip.</div>'
            )
        else:
            parts.append(
                f'<div style="margin-top:8px; color:#ff4444; font-weight:600;">'
                f'&#9888; True peak reaches {_fmt_abs_db(post_tp, 0.0, plus_sign=True)} dBTP '
                f'(sample peak {_fmt_abs_db(post_pk, 0.0)} dBFS) '
                f'&mdash; inter-sample over (&gt; 0 dBTP) into oversampled inserts or DAC reconstruction.</div>'
            )
    elif warn == "amber":
        parts.append(
            f'<div style="margin-top:8px; color:#ffaa00;">'
            f'&#9888; Peak reaches {_fmt_abs_db(post_pk, 0.0)} dBFS{tp_frag} '
            f'(&gt; &minus;1.0 dBFS/dBTP headroom notice).</div>'
        )
    elif d.get("true_peak_unverified"):
        parts.append(
            f'<div style="margin-top:8px; color:{metadata};">'
            f'True peak check pending audio load (sample peak {_fmt_abs_db(post_pk, 0.0)} dBFS).</div>'
        )

    pk_sample = d.get("governing_peak_sample")
    pk_time = _format_peak_timestamp(d.get("governing_peak_time_s"))
    next_pk = _safe_db(d.get("next_peak_db"), float("nan"))
    next_str = f"{_fmt_abs_db(next_pk, 0.0)} dBFS" if math.isfinite(next_pk) else "&mdash;"
    if pk_sample is not None:
        parts.append(
            f'<div style="margin-top:6px; color:{metadata};">'
            f'Peak at <a href="peak:{int(pk_sample)}" style="color:#4499ff;">{pk_time}</a>'
            f' &middot; next {next_str} (&plusmn;{int(d.get("next_peak_exclusion_ms", NEXT_PEAK_EXCLUSION_MS))} ms)</div>'
        )

    rebalance = _safe_db(d.get("fader_rebalance_shift"), 0.0)
    if rebalance != 0.0:
        fader_off = _safe_db(d.get("fader_offset"), 0.0)
        parts.append(
            f'<div style="margin-top:6px; color:{metadata};">'
            f'Fader rebalanced: {_fmt_rel_db(-rebalance)} dB shift'
            f' &rarr; fader offset {_fmt_rel_db(fader_off)} dB</div>'
        )

    if verbose:
        parts.append(_render_verbose_metrics(d, rms_off, metadata))
    return "".join(parts)


def _render_narrow_rows(
    rows: list[tuple[str, tuple[str, str, str], bool, str]],
    head: str,
    cell: str,
    text: str,
    metadata: str,
) -> tuple[str, str, str]:
    header = f'<td style="{head}">Measure<br>Detected &rarr; Target &middot; Gain</td>'
    body = ""
    for metric, values, active, row_note in rows:
        weight = " font-weight:bold;" if active else ""
        det, tgt, gain = (f"<nobr>{value}</nobr>" for value in values)
        body += (
            f'<tr><td style="{cell}"><span style="color:{metadata};">{metric}</span>'
            + (f"<br>{row_note}" if row_note else "")
            + f'<br><span style="color:{text};{weight}">'
            f'{det} &rarr; {tgt} &middot; {gain}</span></td></tr>'
        )
    return 'width="100%" ', header, body


def _render_comparison_table(
    peak_values: tuple[str, str, str],
    rms_values: tuple[str, str, str],
    peak_active: bool,
    rms_active: bool,
    pk_note: str,
    rms_note: str,
    rms_metric: str,
    *,
    compact: bool,
    narrow: bool,
) -> str:
    surface, separator = "#2d2d2d", "#3a3a3a"
    text, metadata = "#dddddd", "#aaaaaa"
    head = (
        f'padding:4px 12px; background-color:{surface}; color:{text};'
        f' border-bottom:1px solid {separator};'
    )
    cell = f'padding:4px 12px; border-bottom:1px solid {separator};'
    rows = [("Peak", peak_values, peak_active, pk_note), (rms_metric, rms_values, rms_active, rms_note)]

    def value_cells(values: tuple[str, str, str], active: bool, width_attrs: tuple[str, ...]) -> str:
        weight = " font-weight:bold;" if active else ""
        return "".join(
            f'<td{w} style="{cell} white-space:nowrap; color:{text};{weight}">{v}</td>'
            for v, w in zip(values, width_attrs)
        )

    if narrow:
        table_attrs, header, body = _render_narrow_rows(rows, head, cell, text, metadata)
    elif compact:
        c_widths = (' width="30%"', ' width="24%"', ' width="24%"', ' width="22%"')
        header = "".join(
            f'<td{w} style="{head}">{label}</td>'
            for label, w in zip(("Measure", "Detected", "Target", "Gain"), c_widths)
        )
        body = "".join(
            f'<tr><td{c_widths[0]} style="{cell} color:{metadata};">'
            f'{metric}{f"<br>{row_note}" if row_note else ""}</td>'
            + value_cells(values, active, c_widths[1:]) + '</tr>'
            for metric, values, active, row_note in rows
        )
        table_attrs = 'width="100%" '
    else:
        w_widths = (' width="112"', ' width="104"', ' width="104"', ' width="88"', ' width="160"')
        header = "".join(
            f'<td{w} style="{head}">{label}</td>'
            for label, w in zip(("Measure", "Detected", "Target", "Gain", "Result"), w_widths)
        )
        body = "".join(
            f'<tr><td{w_widths[0]} style="{cell} color:{metadata};">{metric}</td>'
            + value_cells(values, active, w_widths[1:4])
            + f'<td{w_widths[4]} style="{cell} white-space:nowrap;">{row_note or "&nbsp;"}</td></tr>'
            for metric, values, active, row_note in rows
        )
        table_attrs = ""

    return (
        f'<table {table_attrs}cellpadding="0" cellspacing="0"'
        f' style="margin-top:8px; border-collapse:collapse;">'
        f'<tr>{header}</tr>{body}</table>'
    )


def _render_verbose_metrics(d: dict[str, Any], rms_off: float, metadata: str) -> str:
    items: list[str] = []
    crest = _safe_db(d.get("effective_crest_db"), float("nan"))
    if math.isfinite(crest):
        items.append(f"Effective crest {crest:.1f} dB")
    eff_win = d.get("effective_window_ms")
    if eff_win is not None:
        ew = float(eff_win)
        ew_txt = f"{ew:.1f}" if ew < 10.0 else f"{ew:.0f}"
        items.append(f"Window {ew_txt} ms unweighted ({html.escape(str(d.get('stereo_mode', 'avg')))})")
    if rms_off != 0.0:
        items.append("AES17 RMS (+3.01 dB, peaks unshifted)")
    if not items:
        return ""
    return f'<div style="margin-top:6px; color:{metadata};">{" &middot; ".join(items)}</div>'
