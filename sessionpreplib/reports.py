from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from .audio import dbfs_offset, format_duration
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
    receipts = session.config.get("_executed_files", {})
    failures = dict(session.config.get("_execute_errors", []))
    outcomes = {}
    for track in session.tracks:
        result = track.processor_results.get("bimodal_normalize")
        normalizing = (session.config.get("bimodal_normalize_enabled", True)
                       and result is not None and "bimodal_normalize" not in track.processor_skip
                       and result.classification not in ("Skip", "Silent"))
        error = failures.get(track.filename) or track._cache.get("repair_error")
        if track.status != "OK":
            error = error or track.status
        if normalizing and result.error:
            error = error or result.error
        receipt = None if error else receipts.get(track.filename)
        evidence = receipt if receipt is not None else track.repair_measurements
        planned_gain = result.gain_db if normalizing else (track.manual_gain_db or 0.0)
        outcomes[track.filename] = {
            "status": "failed" if error else "written" if receipt is not None else "planned",
            "error": error,
            "classification": result.classification if result else "Unknown",
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


def generate_report(
    session: SessionContext,
    config: dict[str, Any],
    output_path: str,
    warnings: list[str],
    diagnostic_summary_text: str | None = None,
) -> str:
    """Generate a text report with fader positions and diagnostics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_dir = config.get("_source_dir", "")
    outcomes = _processing_outcomes(session)

    lines = [
        "=" * 80,
        "SessionPrep Report",
        "=" * 80,
        f"Generated: {timestamp}",
        f"Source Directory: {os.path.abspath(source_dir)}",
        f"Output Directory: {os.path.dirname(output_path)}",
        f"Target RMS: {config.get('target_rms', -18.0)} dBFS | Target Peak: {config.get('target_peak', -6.0)} dBFS",
        f"Clipping Detection Threshold: {config.get('clip_consecutive', 3)} consecutive samples @ \u00b11.0",
        "",
    ]

    if diagnostic_summary_text:
        lines.extend([
            diagnostic_summary_text,
            "",
        ])

    # Track groups section
    grouped = {}
    for t in session.tracks:
        outcome = outcomes[t.filename]
        if t.group is None:
            continue
        gain = outcome["gain_applied_db"]
        grouped.setdefault(t.group, []).append((t.filename, gain))

    if grouped:
        lines.extend([
            "-" * 80,
            "TRACK GROUPS",
            "-" * 80,
            "",
        ])
        for gid in sorted(grouped.keys()):
            members = grouped[gid]
            gains = {gain for _, gain in members}
            gain = next(iter(gains)) if len(gains) == 1 else None
            lines.append(f"{gid} | Group level {gain:+.2f} dB" if gain is not None else gid)
            for name in sorted((name for name, _ in members), key=protools_sort_key):
                lines.append(f"  - {name}")
            lines.append("")

    # Fader positions
    lines.extend([
        "-" * 80,
        "FADER POSITIONS (Set these in your DAW to restore original balance)",
        "-" * 80,
        "",
        f"{'TRACK':<32} {'STATUS':<8} {'APPLIED GAIN':>12} {'FADER':>12}  TYPE / REPAIRS",
        "-" * 80,
    ])

    for t in session.tracks:
        outcome = outcomes[t.filename]
        gain = outcome["gain_applied_db"]
        fader = outcome["fader_db"]
        gain_str = f"{gain:+.2f} dB" if gain is not None else "—"
        fader_str = f"{fader:+.2f} dB" if fader is not None else "—"
        repairs = outcome["applied_repairs"] if outcome["status"] == "written" else outcome["planned_repairs"]
        detail = ", ".join([outcome["classification"], *repairs])
        lines.append(f"{t.filename:<32} {outcome['status']:<8} {gain_str:>12} {fader_str:>12}  {detail}")
        lines.extend(f"  {warning}" for warning in format_boundary_settling(outcome["boundary_settling"]))

    # Tail report
    rms_anchor = config.get("rms_anchor", "percentile")
    window = config.get("window", 400)
    rms_percentile = config.get("rms_percentile", 95.0)
    tail_max_regions = config.get("tail_max_regions", 20)
    tail_min_exceed_db = config.get("tail_min_exceed_db", 3.0)
    tail_hop_ms = config.get("tail_hop_ms", 10)

    lines.extend([
        "",
        "-" * 80,
        "MOMENTARY RMS UPPER TAIL (Expected Windows Above Anchor)",
        "-" * 80,
        f"Default Anchor Mode: {rms_anchor} | Window: {window} ms" + (
            f" | Percentile: P{rms_percentile:g}" if rms_anchor == "percentile" else ""
        ),
        f"Report Limit: {tail_max_regions} regions per file | Min Exceed: {tail_min_exceed_db} dB | Hop: {tail_hop_ms} ms",
        "",
    ])

    measured_tails = []
    for track in session.tracks:
        tail = track.detector_results.get("tail_exceedance")
        if track.status == "OK" and tail and tail.applicable and not tail.error:
            measured_tails.append((track, tail))
    if not measured_tails:
        lines.append("No applicable tail measurements. Tail reporting requires percentile anchoring.")
        lines.append("")
    else:
        any_tail_reported = False
        for t, tail_r in measured_tails:
            regions = tail_r.data.get("tail_regions") or []
            if not regions:
                continue

            summary = tail_r.data.get("tail_summary") or {}
            anchor_db = summary.get("anchor_db", float('-inf'))
            total_dur = summary.get("total_duration_sec", 0.0)
            max_exceed = summary.get("max_exceed_db", 0.0)

            lines.append(f"{t.filename}")
            classifier = t.detector_results.get("audio_classifier")
            anchor_label = classifier.data.get("rms_anchor_label") if classifier else None
            anchor_note = f" ({anchor_label})" if anchor_label else ""
            lines.append(f"  Anchor RMS: {anchor_db:.2f} dBFS{anchor_note}")
            lines.append(f"  Tail Regions Reported: {int(summary.get('regions', 0))} | Total Tail Duration: {total_dur:.3f}s | Max Exceed: +{max_exceed:.2f} dB")

            any_tail_reported = True
            for i, reg in enumerate(regions, start=1):
                lines.append(
                    f"  {i:>2}. {reg['start_time']} - {reg['end_time']} | "
                    f"samples {reg['start_sample']}-{reg['end_sample'] - 1} | "
                    f"max +{reg['max_exceed_db']:.2f} dB (RMS {reg['max_rms_db']:.2f} dBFS)"
                )
            lines.append("")

        if not any_tail_reported:
            lines.append("No significant tail exceedances above threshold were found.")
            lines.append("")

    # File overview
    lines.extend([
        "",
        "-" * 80,
        "FILE OVERVIEW",
        "-" * 80,
        "",
        f"{'TRACK':<25} {'SR(kHz)':>8} {'BIT':>8} {'DUR':>10}",
        "-" * 80,
    ])

    errors = []
    for t in session.tracks:
        if outcomes[t.filename]["error"]:
            errors.append((t.filename, outcomes[t.filename]["error"]))
        if t.status != "OK":
            continue

        sr_khz = f"{t.samplerate/1000:.1f}"
        dur_fmt = format_duration(t.total_samples, t.samplerate)
        lines.append(f"{t.filename[:23]:<25} {sr_khz:>8} {t.bitdepth:>8} {dur_fmt:>10}")

    if errors:
        lines.extend([
            "",
            "-" * 80,
            "FILES WITH ERRORS",
            "-" * 80,
        ])
        for name, status in errors:
            lines.append(f"{name}: {status}")

    lines.extend([
        "",
        "=" * 80,
        "HOW TO USE:",
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

    data = {
        "schema_version": "2.0",
        "session": {
            "timestamp": datetime.now().isoformat(),
            "source_directory": os.path.abspath(source_dir),
            "warnings": warnings,
            "config": {
                "target_rms": config.get("target_rms", -18.0),
                "target_peak": config.get("target_peak", -6.0),
                "stereo_mode": config.get("stereo_mode", "avg"),
                "window_ms": config.get("window", 400),
                "rms_anchor": config.get("rms_anchor", "percentile"),
                "rms_percentile": config.get("rms_percentile", 95.0),
                "tail_hop_ms": config.get("tail_hop_ms", 10),
            }
        },
        "tracks": []
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

        data["tracks"].append({
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

    with atomic_writer(output_path) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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
        warnings.append(f"Digital Clipping detected in source (>{clip_consec} samples @ \u00b11.0):")
        for t in clipped:
            warnings.append(f"  - {t.filename}")

    # DC offset warnings
    dc_warn_tracks = [t for t in ok_tracks
                      if (t.detector_results.get("dc_offset") or
                          type('', (), {"data": {}})()).data.get("dc_warn")]
    if dc_warn_tracks:
        dc_thresh = config.get("dc_offset_warn_db", -40.0)
        _off = dbfs_offset(config)
        warnings.append(f"DC offset detected (>{dc_thresh} dBFS):")
        for t in dc_warn_tracks:
            dc_r = t.detector_results.get("dc_offset")
            dc_db = dc_r.data.get("dc_db", float('-inf')) if dc_r else float('-inf')
            if np.isfinite(dc_db):
                warnings.append(f"  - {t.filename} is {dc_db + _off:.1f} dBFS")
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
