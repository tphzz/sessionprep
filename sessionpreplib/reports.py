from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from .audio import INACTIVE_NORMALIZATION_CLASSIFICATIONS, format_duration
from .file_io import atomic_writer
from .models import SessionContext
from .utils import protools_sort_key


def _report_db(value) -> float | None:
    return round(value, 2) if isinstance(value, (int, float)) and math.isfinite(value) else None


def format_boundary_settling(regions):
    """Present each affected channel and inclusive sample range at unity gain."""
    return [
        f"Boundary settling in previously silent audio · Ch {r['channel'] + 1} · "
        f"samples {r['start_sample']}–{r['end_sample']} · peak {r['peak_dbfs']:.1f} dBFS before gain"
        for r in regions
    ]


def _processing_outcomes(session: SessionContext) -> dict[str, dict[str, Any]]:
    """Report the last execution receipt separately from the current plan."""
    receipts = session.execution_receipts.executed_files or session.config.get("_executed_files", {})
    failures = dict(session.execution_receipts.execute_errors or session.config.get("_execute_errors", []))
    outcomes = {}
    for track in session.tracks:
        result = track.processor_results.get("bimodal_normalize")
        normalizing = (session.config.get("bimodal_normalize_enabled", True)
                       and result is not None and "bimodal_normalize" not in track.processor_skip
                       and result.classification not in INACTIVE_NORMALIZATION_CLASSIFICATIONS)
        error = failures.get(track.filename) or track._cache.get("repair_error")
        if track.status != "OK":
            error = error or track.status
        if normalizing and result is not None and result.error:
            error = error or result.error
        receipt = None if error else receipts.get(track.filename)
        evidence = receipt if receipt is not None else track.repair_measurements
        if result is not None:
            planned_gain = result.gain_db if normalizing else 0.0
        else:
            planned_gain = track.manual_gain_db or 0.0
        outcomes[track.filename] = {
            "status": "failed" if error else "written" if receipt is not None else "planned",
            "error": error,
            "classification": result.classification if result else "Unknown",
            "mode": result.data.get("mode", "Auto") if result else "Unknown",
            "governing": result.data.get("governing") if result else None,
            "post_peak_db": _report_db(result.data.get("post_peak_db")) if result else None,
            "post_true_peak_db": _report_db(result.data.get("post_true_peak_db")) if result else None,
            "peak_warning_level": result.data.get("peak_warning_level", "none") if result else "none",
            "planned_repairs": track.repair_recipe.applied_ids(),
            "boundary_settling": evidence.get("boundary_settling", []),
            "applied_repairs": [p for p in receipt["applied_processors"]
                                if p in ("dc_remove", "low_cut", "ultrasonic")] if receipt else [],
            "planned_gain_db": _report_db(planned_gain),
            "gain_applied_db": _report_db(receipt["gain_db"]) if receipt else None,
            "fader_db": _report_db(receipt["fader_db"]) if receipt else None,
            "gain_individual_db": _report_db(result.data.get("gain_db_individual", planned_gain))
                                  if result else _report_db(planned_gain),
            "group": track.group,
        }
    return outcomes


def _report_group_lines(session: SessionContext, outcomes: dict[str, dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[tuple[str, float | None]]] = {}
    for t in session.tracks:
        if t.group is not None:
            grouped.setdefault(t.group, []).append((t.filename, outcomes[t.filename]["gain_applied_db"]))
    if not grouped:
        return []
    lines = ["-" * 80, "TRACK GROUPS", "-" * 80, ""]
    for gid in sorted(grouped):
        members = grouped[gid]
        gains = {gain for _, gain in members}
        gain = next(iter(gains)) if len(gains) == 1 else None
        lines.append(f"{gid} | Group level {gain:+.2f} dB" if gain is not None else gid)
        lines.extend(f"  - {name}" for name in sorted((name for name, _ in members), key=protools_sort_key))
        lines.append("")
    return lines


def _report_fader_lines(session: SessionContext, outcomes: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "-" * 80,
        "FADER POSITIONS (Set these in your DAW to restore original balance)",
        "-" * 80, "",
        f"{'TRACK':<32} {'STATUS':<8} {'APPLIED GAIN':>12} {'FADER':>12}  TYPE / REPAIRS",
        "-" * 80,
    ]
    for t in session.tracks:
        outcome = outcomes[t.filename]
        gain, fader = outcome["gain_applied_db"], outcome["fader_db"]
        gain_str = f"{gain:+.2f} dB" if gain is not None else "—"
        fader_str = f"{fader:+.2f} dB" if fader is not None else "—"
        repairs = outcome["applied_repairs"] if outcome["status"] == "written" else outcome["planned_repairs"]
        detail = ", ".join([outcome["classification"], *repairs])
        lines.append(f"{t.filename:<32} {outcome['status']:<8} {gain_str:>12} {fader_str:>12}  {detail}")
        lines.extend(f"  {warning}" for warning in format_boundary_settling(outcome["boundary_settling"]))
    return lines


def _report_tail_lines(session: SessionContext, config: dict[str, Any]) -> list[str]:
    rms_anchor = config.get("rms_anchor", "percentile")
    window = config.get("window", 400)
    rms_percentile = config.get("rms_percentile", 95.0)
    lines = [
        "", "-" * 80,
        "MOMENTARY RMS UPPER TAIL (Expected Windows Above Anchor)", "-" * 80,
        f"Default Anchor Mode: {rms_anchor} | Window: {window} ms"
        + (f" | Percentile: P{rms_percentile:g}" if rms_anchor == "percentile" else ""),
        f"Report Limit: {config.get('tail_max_regions', 20)} regions per file | "
        f"Min Exceed: {config.get('tail_min_exceed_db', 3.0)} dB | Hop: {config.get('tail_hop_ms', 10)} ms", "",
    ]
    measured = [
        (t, tail) for t in session.tracks
        if t.status == "OK" and (tail := t.detector_results.get("tail_exceedance")) and tail.applicable and not tail.error
    ]
    if not measured:
        return [*lines, "No applicable tail measurements. Tail reporting requires percentile anchoring.", ""]

    any_reported = False
    for t, tail_r in measured:
        regions = tail_r.data.get("tail_regions") or []
        if not regions:
            continue
        summary = tail_r.data.get("tail_summary") or {}
        norm = t.processor_results.get("bimodal_normalize")
        anchor_label = (
            tail_r.data.get("rms_anchor_label")
            or summary.get("anchor_label")
            or (norm.data.get("rms_anchor_label") if norm else None)
        )
        anchor_note = f" ({anchor_label})" if anchor_label else ""
        lines.extend([
            f"{t.filename}",
            f"  Anchor RMS: {summary.get('anchor_db', float('-inf')):.2f} dBFS{anchor_note}",
            f"  Tail Regions Reported: {int(summary.get('regions', 0))} | "
            f"Total Tail Duration: {summary.get('total_duration_sec', 0.0):.3f}s | "
            f"Max Exceed: +{summary.get('max_exceed_db', 0.0):.2f} dB",
        ])
        any_reported = True
        for i, reg in enumerate(regions, start=1):
            lines.append(
                f"  {i:>2}. {reg['start_time']} - {reg['end_time']} | "
                f"samples {reg['start_sample']}-{reg['end_sample'] - 1} | "
                f"max +{reg['max_exceed_db']:.2f} dB (RMS {reg['max_rms_db']:.2f} dBFS)"
            )
        lines.append("")
    if not any_reported:
        lines.extend(["No significant tail exceedances above threshold were found.", ""])
    return lines


def _report_overview_lines(session: SessionContext, outcomes: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "", "-" * 80, "FILE OVERVIEW", "-" * 80, "",
        f"{'TRACK':<25} {'SR(kHz)':>8} {'BIT':>8} {'DUR':>10}", "-" * 80,
    ]
    errors = []
    for t in session.tracks:
        if outcomes[t.filename]["error"]:
            errors.append((t.filename, outcomes[t.filename]["error"]))
        if t.status != "OK":
            continue
        lines.append(
            f"{t.filename[:23]:<25} {t.samplerate / 1000:>8.1f} {t.bitdepth:>8} "
            f"{format_duration(t.total_samples, t.samplerate):>10}"
        )
    if errors:
        lines.extend(["", "-" * 80, "FILES WITH ERRORS", "-" * 80])
        lines.extend(f"{name}: {status}" for name, status in errors)
    return lines


def generate_report(
    session: SessionContext,
    config: dict[str, Any],
    output_path: str,
    warnings: list[str],
    diagnostic_summary_text: str | None = None,
) -> str:
    """Generate a text report with fader positions and diagnostics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outcomes = _processing_outcomes(session)
    lines = [
        "=" * 80, "SessionPrep Report", "=" * 80,
        f"Generated: {timestamp}",
        f"Source Directory: {os.path.abspath(config.get('_source_dir', ''))}",
        f"Output Directory: {os.path.dirname(output_path)}",
        f"Target RMS: {config.get('target_rms', -18.0)} dBFS (unweighted) | "
        f"Peak Ceiling: {config.get('target_peak', -6.0)} dBFS | "
        f"Max Boost: +{config.get('max_gain_db', 24.0)} dB",
        f"Clipping Detection Threshold: {config.get('clip_consecutive', 3)} consecutive near-full-scale samples",
        "",
    ]
    if diagnostic_summary_text:
        lines.extend([diagnostic_summary_text, ""])
    lines.extend(_report_group_lines(session, outcomes))
    lines.extend(_report_fader_lines(session, outcomes))
    lines.extend(_report_tail_lines(session, config))
    lines.extend(_report_overview_lines(session, outcomes))
    lines.extend([
        "", "=" * 80, "HOW TO USE:",
        "1. Import the files marked written into your DAW",
        "2. Apply the fader offsets (manually or via automation from the JSON export)",
        "3. These offsets compensate the normalization gain applied to the prepared audio",
        "=" * 80,
    ])
    report_text = "\n".join(lines)
    with atomic_writer(output_path) as f:
        f.write(report_text)
    return report_text


def save_json(
    session: SessionContext,
    config: dict[str, Any],
    output_path: str,
    warnings: list[str],
) -> None:
    """Generate JSON output for automation tools."""
    source_dir = config.get("_source_dir", "")
    outcomes = _processing_outcomes(session)

    tracks_data: list[dict[str, Any]] = []
    data: dict[str, Any] = {
        "schema_version": "3.0",
        "session": {
            "timestamp": datetime.now().isoformat(),
            "source_directory": os.path.abspath(source_dir),
            "warnings": warnings,
            "config": {
                "target_rms": config.get("target_rms", -18.0),
                "target_peak": config.get("target_peak", -6.0),
                "max_gain_db": config.get("max_gain_db", 24.0),
                "force_peak": config.get("force_peak", ""),
                "dbfs_convention": config.get("dbfs_convention", "standard"),
                "stereo_mode": config.get("stereo_mode", "avg"),
                "window_ms": config.get("window", 400),
                "rms_anchor": config.get("rms_anchor", "percentile"),
                "rms_percentile": config.get("rms_percentile", 95.0),
                "tail_hop_ms": config.get("tail_hop_ms", 10),
            }
        },
        "tracks": tracks_data,
    }

    sort_index = 0
    for t in session.tracks:
        clip_r = t.detector_results.get("clipping")
        sil_r = t.detector_results.get("silence")
        sc_r = t.detector_results.get("stereo_compat")

        is_clipped = bool(clip_r.data.get("is_clipped")) if clip_r else False
        is_silent = bool(sil_r.data.get("is_silent")) if sil_r else False
        mono_loss_db = sc_r.data.get("mono_loss_db") if sc_r else None
        mono_warn = bool(sc_r.data.get("mono_warn")) if sc_r else False

        tracks_data.append({
            "sort_index": sort_index,
            "filename": t.filename,
            "diagnostics": {
                "samplerate": int(t.samplerate),
                "bitdepth": t.bitdepth,
                "duration_sec": float(t.duration_sec),
                "samples": int(t.total_samples),
                "is_clipped": bool(is_clipped),
                "is_silent": bool(is_silent),
                "mono_loss_db": (float(mono_loss_db) if mono_loss_db is not None
                                 and math.isfinite(mono_loss_db) else None),
                "mono_fully_cancelled": bool(mono_loss_db == math.inf),
                "mono_loss_warn": bool(mono_warn),
            },
            "processing": outcomes[t.filename],
        })
        sort_index += 1

    signal_flows = {
        dp_id: _sanitize_json_value(dp_state["signal_flow"])
        for dp_id, dp_state in (session.daw_state or {}).items()
        if isinstance(dp_state, dict) and isinstance(dp_state.get("signal_flow"), dict)
    }
    if signal_flows:
        data["daw_signal_flow"] = signal_flows

    with atomic_writer(output_path) as f:
        json.dump(data, f, indent=4, ensure_ascii=False, allow_nan=False)


def _sanitize_json_value(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for RFC 8259 JSON compliance."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return value


def build_warnings(
    session: SessionContext,
    config: dict[str, Any],
) -> list[str]:
    """Build the warnings list from detector results (for JSON/report compat)."""
    warnings = list(session.warnings)

    ok_tracks = [t for t in session.tracks if t.status == "OK"]

    # SR mismatch warnings
    sr_counter = Counter(t.samplerate for t in ok_tracks)
    if len(sr_counter) > 1:
        most_common_sr = sr_counter.most_common(1)[0][0]
        warnings.append(f"Samplerate mismatch detected! Most common: {most_common_sr} Hz")
        for t in ok_tracks:
            if t.samplerate != most_common_sr:
                warnings.append(f"  - {t.filename} is {t.samplerate} Hz")

    # BD mismatch warnings
    bd_counter = Counter(t.bitdepth for t in ok_tracks)
    if len(bd_counter) > 1:
        most_common_bd = bd_counter.most_common(1)[0][0]
        warnings.append(f"Bit-depth mismatch detected! Most common: {most_common_bd}")
        for t in ok_tracks:
            if t.bitdepth != most_common_bd:
                warnings.append(f"  - {t.filename} is {t.bitdepth}")

    # Clipping warnings
    clipped = [t for t in ok_tracks
               if (t.detector_results.get("clipping") or
                   type('', (), {"data": {}})()).data.get("is_clipped")]
    if clipped:
        clip_consec = config.get("clip_consecutive", 3)
        warnings.append(f"Digital Clipping detected in source ({clip_consec} or more consecutive near-full-scale samples):")
        for t in clipped:
            warnings.append(f"  - {t.filename}")

    # DC offset warnings
    dc_warn_tracks = [t for t in ok_tracks
                      if (t.detector_results.get("dc_offset") or
                          type('', (), {"data": {}})()).data.get("dc_warn")]
    if dc_warn_tracks:
        dc_thresh = config.get("dc_offset_warn_db", -40.0)
        warnings.append(f"DC offset detected (>{dc_thresh} dBFS):")
        for t in dc_warn_tracks:
            dc_r = t.detector_results.get("dc_offset")
            dc_db = dc_r.data.get("dc_db", float('-inf')) if dc_r else float('-inf')
            if np.isfinite(dc_db):
                warnings.append(f"  - {t.filename} is {dc_db:.1f} dBFS")
            else:
                warnings.append(f"  - {t.filename}")

    # Silent file warnings
    silent = [t for t in ok_tracks
              if (t.detector_results.get("silence") or
                  type('', (), {"data": {}})()).data.get("is_silent")]
    if silent:
        warnings.append("Silent files detected:")
        for t in silent:
            warnings.append(f"  - {t.filename}")

    return warnings
