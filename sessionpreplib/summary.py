"""Diagnostic summary aggregation (data transformation separated from rendering)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .audio import dbfs_offset
from .detectors.stereo_compat import format_mono_loss
from .models import DetectorResult, SessionContext, Severity, TrackContext
from .repair_verification import remaining_results
from .utils import GROUP_OVERLAP_PREFIX


class _SummaryAccumulator:
    """Collect categorized diagnostic items while keeping stable item references."""

    def __init__(self, det_map: dict[str, Any]) -> None:
        self.det_map = det_map
        self.problems: list[dict[str, Any]] = []
        self.attention: list[dict[str, Any]] = []
        self.information: list[dict[str, Any]] = []
        self.clean: list[dict[str, Any]] = []
        self.buckets = {
            "problem": self.problems,
            "attention": self.attention,
            "information": self.information,
            "info": self.information,
        }
        self._refs: dict[int, list[dict[str, str]]] = {}

    def is_skipped(self, det_id: str) -> bool:
        det = self.det_map.get(det_id)
        return bool(det and getattr(det, "_report_as", "default") == "skip")

    def routed_bucket(self, det_id: str, actual_severity: Severity) -> list[dict[str, Any]] | None:
        det = self.det_map.get(det_id)
        if det and getattr(det, "_report_as", "default") == "skip":
            return None
        if det and hasattr(det, "effective_severity"):
            eff = det.effective_severity(DetectorResult(det_id, actual_severity, "", {}))
        else:
            eff = actual_severity
        if eff is None:
            return None
        sev_val = eff.value if hasattr(eff, "value") else str(eff)
        return self.buckets.get(sev_val, self.information)

    def add_item(self, items: list[str], filename: str, detail: str) -> None:
        items.append(f"{filename}: {detail}")
        self._refs.setdefault(id(items), []).append(
            {"filename": str(filename), "detail": str(detail)}
        )

    def refs_for(self, items: list[str]) -> list[dict[str, str]]:
        return self._refs.get(id(items), [])

    def add_group(
        self,
        dst: list[dict[str, Any]],
        title: str,
        hint: str | None,
        items: list[str],
        *,
        standalone: bool = False,
        detector_id: str | None = None,
    ) -> None:
        if not items and not standalone:
            return
        dst.append({
            "title": title,
            "hint": hint,
            "items": items or [],
            "item_refs": self.refs_for(items),
            "detector_id": detector_id,
            "standalone": bool(standalone),
        })


def _prepare_summary_tracks(
    session: SessionContext,
    track_detectors: list | None,
    *,
    remaining: bool,
) -> tuple[list[TrackContext], set[str], set[str], list]:
    ok_tracks = [t for t in session.tracks if t.status == "OK"]
    pending_tracks: set[str] = set()
    failed_tracks: set[str] = set()
    detectors = track_detectors if track_detectors is not None else session.detectors
    if remaining:
        for t in ok_tracks:
            snapshot = t.corrected_analysis
            if t._cache.get("repair_error") or (snapshot and any(r.error for r in snapshot.results.values())):
                failed_tracks.add(t.filename)
            elif t._cache.get("repair_verification_pending") or (t.repair_recipe.active and snapshot is None):
                pending_tracks.add(t.filename)
        ok_tracks = [
            replace(t, detector_results=remaining_results(t, detectors, pending=t.filename in pending_tracks))
            for t in ok_tracks
        ]
    return ok_tracks, pending_tracks, failed_tracks, detectors


def _session_det_results(session: SessionContext, det_id: str) -> list[DetectorResult]:
    if det_id in session.session_detector_results:
        return session.session_detector_results[det_id]
    return session.config.get(f"_session_det_{det_id}", [])


def _consensus_values(session: SessionContext) -> tuple[int | None, str | None, int | None, str | None]:
    fc = session.format_consensus
    sr = fc.most_common_sr if fc.most_common_sr is not None else session.config.get("_most_common_sr")
    bd = fc.most_common_bd if fc.most_common_bd is not None else session.config.get("_most_common_bd")
    length = fc.most_common_len if fc.most_common_len is not None else session.config.get("_most_common_len")
    length_fmt = (
        fc.most_common_len_fmt if fc.most_common_len_fmt is not None else session.config.get("_most_common_len_fmt")
    )
    return sr, bd, length, length_fmt


def _collect_format_consistency(
    session: SessionContext, acc: _SummaryAccumulator, total_ok: int,
    most_common_sr: int | None, most_common_bd: str | None,
) -> set[str]:
    mismatch_names: set[str] = set()
    if acc.is_skipped("format_consistency"):
        return mismatch_names

    items: list[str] = []
    actual_sev = Severity.PROBLEM
    for r in _session_det_results(session, "format_consistency"):
        if r.severity == Severity.CLEAN:
            continue
        actual_sev = r.severity
        fname = r.data.get("filename", "")
        reasons = r.data.get("mismatch_reasons", [])
        details = ", ".join(reasons) if reasons else "mismatch"
        acc.add_item(items, fname, f"format mismatch ({details})")
        mismatch_names.add(fname)

    common_fmt = f"{most_common_sr} Hz / {most_common_bd}" if (most_common_sr is not None and most_common_bd is not None) else None
    title = "Format mismatches"
    if total_ok:
        matches = total_ok - len(items)
        summary = f"{matches}/{total_ok} file(s) match the most common session format"
        if common_fmt:
            summary += f" ({common_fmt})"
        title = f"Format mismatches. Deviations from {summary}"

    bucket = acc.routed_bucket("format_consistency", actual_sev)
    if bucket is not None:
        acc.add_group(bucket, title, "request corrected exports", items)
    if total_ok and common_fmt and not items:
        acc.add_group(acc.clean, f"No inconsistent session formats ({common_fmt})", None, [], standalone=True)
    return mismatch_names


def _collect_length_consistency(
    session: SessionContext, acc: _SummaryAccumulator, total_ok: int,
    most_common_len: int | None, most_common_len_fmt: str | None,
) -> set[str]:
    mismatch_names: set[str] = set()
    if acc.is_skipped("length_consistency"):
        return mismatch_names

    items: list[str] = []
    actual_sev = Severity.PROBLEM
    for r in _session_det_results(session, "length_consistency"):
        if r.severity == Severity.CLEAN:
            continue
        actual_sev = r.severity
        fname = r.data.get("filename", "")
        actual_samples = r.data.get("actual_samples")
        actual_fmt = r.data.get("actual_duration_fmt", "")
        detail = f"length mismatch ({int(actual_samples)} samples / {actual_fmt})" if actual_samples is not None else "length mismatch"
        acc.add_item(items, fname, detail)
        mismatch_names.add(fname)

    title = "Length mismatches"
    if total_ok and most_common_len is not None:
        matches = total_ok - len(items)
        fmt = f"{int(most_common_len)} samples / {most_common_len_fmt}" if most_common_len_fmt else f"{int(most_common_len)} samples"
        title = f"Length mismatches. Deviations from {matches}/{total_ok} file(s) match the most common length ({fmt})"

    bucket = acc.routed_bucket("length_consistency", actual_sev)
    if bucket is not None:
        acc.add_group(bucket, title, "request aligned exports", items)
    if total_ok and most_common_len is not None and not items:
        fmt = f"{int(most_common_len)} samples" + (f" / {most_common_len_fmt}" if most_common_len_fmt else "")
        acc.add_group(acc.clean, f"No inconsistent file lengths ({fmt})", None, [], standalone=True)
    return mismatch_names


def _stereo_compat_detail(sc_r: DetectorResult, config: dict[str, Any], *, remaining: bool) -> str | None:
    if not (sc_r.data.get("corr_warn") or sc_r.data.get("mono_warn") or sc_r.data.get("windowed_regions")):
        return None
    parts: list[str] = []
    lr_corr = sc_r.data.get("lr_corr")
    if sc_r.data.get("corr_warn"):
        if lr_corr is None:
            parts.append("corr < threshold")
        else:
            parts.append(f"corr {float(lr_corr):.2f} (< {float(config.get('corr_warn', -0.3)):g})")
    mono_loss_db = sc_r.data.get("mono_loss_db")
    if sc_r.data.get("mono_warn"):
        if mono_loss_db is None or np.isnan(mono_loss_db):
            parts.append("mono loss > threshold")
        else:
            parts.append(
                f"mono loss {format_mono_loss(mono_loss_db)} (> {float(config.get('mono_loss_warn_db', 6.0)):g} dB)"
            )
    region_count = len(sc_r.data.get("windowed_regions", []))
    if region_count:
        minimum = sc_r.data.get("min_window_corr")
        if remaining and minimum is not None:
            parts.append(f"minimum correlation {minimum:.2f}")
        parts.append(f"{region_count} marked {'location' if region_count == 1 else 'locations'}")
    return ", ".join(parts) if parts else None


def _one_sided_detail(oss_r: DetectorResult, offset: float) -> str:
    side = oss_r.data.get("one_sided_silence_side")
    l_db = oss_r.data.get("l_rms_db", float(-np.inf))
    r_db = oss_r.data.get("r_rms_db", float(-np.inf))
    l_str = f"{float(l_db) + offset:.1f}" if np.isfinite(l_db) else "-inf"
    r_str = f"{float(r_db) + offset:.1f}" if np.isfinite(r_db) else "-inf"
    prefix = f"one-sided silence ({side})" if side else "one-sided silence"
    return f"{prefix} (L {l_str} dBFS, R {r_str} dBFS)"


def _collect_track_basic_findings(
    t: TrackContext, session: SessionContext, acc: _SummaryAccumulator,
    lists: dict[str, list[str]], issue_names: set[str], offset: float, *, remaining: bool,
) -> None:
    if not acc.is_skipped("clipping"):
        clip_r = t.detector_results.get("clipping")
        if clip_r and clip_r.data.get("is_clipped"):
            acc.add_item(lists["clipping"], t.filename, f"clipping detected ({int(clip_r.data.get('runs', 0))} clipped ranges)")
            issue_names.add(t.filename)
    if not acc.is_skipped("dc_offset"):
        dc_r = t.detector_results.get("dc_offset")
        if dc_r and dc_r.data.get("dc_warn"):
            dc_db = dc_r.data.get("dc_db", float(-np.inf))
            # AES17 offset applies to RMS only; DC offset stays unshifted dBFS.
            detail = (
                f"DC offset {float(dc_db):.1f} dBFS"
                if isinstance(dc_db, (int, float)) and np.isfinite(dc_db)
                else "DC offset issue"
            )
            acc.add_item(lists["dc_offset"], t.filename, detail)
            issue_names.add(t.filename)
    if not acc.is_skipped("stereo_compat"):
        sc_r = t.detector_results.get("stereo_compat")
        sc_detail = _stereo_compat_detail(sc_r, session.config, remaining=remaining) if sc_r else None
        if sc_detail:
            acc.add_item(lists["stereo_compat"], t.filename, sc_detail)
            issue_names.add(t.filename)
    if not acc.is_skipped("dual_mono"):
        dm_r = t.detector_results.get("dual_mono")
        if dm_r and dm_r.data.get("dual_mono"):
            acc.add_item(lists["dual_mono"], t.filename, "dual-mono (identical L/R)")
            if remaining:
                issue_names.add(t.filename)


def _collect_track_spectral_and_tail_findings(
    t: TrackContext, session: SessionContext, acc: _SummaryAccumulator,
    lists: dict[str, list[str]], issue_names: set[str], offset: float,
) -> None:
    if not acc.is_skipped("silence"):
        sil_r = t.detector_results.get("silence")
        if sil_r and sil_r.data.get("is_silent"):
            acc.add_item(lists["silence"], t.filename, "silent")
            issue_names.add(t.filename)
    if not acc.is_skipped("one_sided_silence"):
        oss_r = t.detector_results.get("one_sided_silence")
        if oss_r and oss_r.data.get("one_sided_silence"):
            acc.add_item(lists["one_sided_silence"], t.filename, _one_sided_detail(oss_r, offset))
            issue_names.add(t.filename)
    if not acc.is_skipped("subsonic"):
        sub_r = t.detector_results.get("subsonic")
        if sub_r and sub_r.data.get("subsonic_warn"):
            ratio_db = sub_r.data.get("subsonic_ratio_db", float(-np.inf))
            cutoff_hz = session.config.get("subsonic_hz", 30.0)
            detail = f"subsonic energy {float(ratio_db):.1f} dB (<= {float(cutoff_hz):g} Hz)" if np.isfinite(ratio_db) else "subsonic content detected"
            acc.add_item(lists["subsonic"], t.filename, detail)
            issue_names.add(t.filename)
    ultrasonic = t.detector_results.get("ultrasonic")
    if ultrasonic and ultrasonic.data.get("regions") and not acc.is_skipped("ultrasonic"):
        acc.add_item(lists["ultrasonic"], t.filename, ultrasonic.summary)
        issue_names.add(t.filename)
    if not acc.is_skipped("tail_exceedance"):
        tail_r = t.detector_results.get("tail_exceedance")
        tail_det = acc.det_map.get("tail_exceedance")
        if tail_r and not (tail_det and not tail_det.is_relevant(tail_r, t)):
            tail_summary = tail_r.data.get("tail_summary", {})
            regions = int(tail_summary.get("regions", 0))
            if regions > 0:
                max_exceed = float(tail_summary.get("max_exceed_db", 0.0))
                tail_min = float(session.config.get("tail_min_exceed_db", 3.0))
                acc.add_item(lists["tail_exceedance"], t.filename, f"{regions} tail region(s) exceed anchor by >{tail_min:g} dB (max +{max_exceed:.1f} dB)")
                issue_names.add(t.filename)


def _route_detector_groups(
    ok_tracks: list[TrackContext], session: SessionContext, acc: _SummaryAccumulator,
    lists: dict[str, list[str]], detectors: list, issue_names: set[str], *, remaining: bool,
) -> None:
    sub_hz = float(session.config.get("subsonic_hz", 30.0))
    det_groups: list[tuple[str, str, str | None, list[str]]] = [
        ("clipping", "Digital clipping", "request reprint / check limiting", lists["clipping"]),
        ("dc_offset", "DC offset", "consider DC removal", lists["dc_offset"]),
        ("stereo_compat", "Stereo compatibility", None, lists["stereo_compat"]),
        ("dual_mono", "Dual-mono (identical L/R)", None, lists["dual_mono"]),
        ("silence", "Silent files", "confirm intentional", lists["silence"]),
        ("one_sided_silence", "One-sided silence", "check stereo export / channel routing", lists["one_sided_silence"]),
        ("subsonic", "Subsonic content", f"consider HPF ~{sub_hz:g} Hz", lists["subsonic"]),
        ("ultrasonic", "Ultrasonic content", "review ultrasonic filtering", lists["ultrasonic"]),
        ("tail_exceedance", "Tail regions exceeded anchor", "check for section-based riding", lists["tail_exceedance"]),
    ]
    if remaining:
        standard_ids = {group[0] for group in det_groups}
        for det in detectors:
            if det.id in standard_ids or not hasattr(det, "is_relevant") or acc.is_skipped(det.id):
                continue
            extra: list[str] = []
            for t in ok_tracks:
                res = t.detector_results.get(det.id)
                if res and res.applicable and not res.error and res.severity != Severity.CLEAN and det.is_relevant(res, t):
                    acc.add_item(extra, t.filename, res.summary)
                    issue_names.add(t.filename)
            det_groups.append((det.id, det.name, None, extra))

    tracks_by_name = {t.filename: t for t in ok_tracks}
    for det_id, title, hint, items in det_groups:
        if not items:
            continue
        severity_items: dict[Severity, list[str]] = {}
        for ref in acc.refs_for(items):
            res = tracks_by_name[ref["filename"]].detector_results[det_id]
            sev = acc.det_map[det_id].effective_severity(res) if det_id in acc.det_map else res.severity
            if sev is not None and sev != Severity.CLEAN:
                acc.add_item(severity_items.setdefault(sev, []), ref["filename"], ref["detail"])
        for sev, listed in severity_items.items():
            acc.add_group(acc.buckets[sev.value], title, hint, listed, detector_id=det_id)


def _add_clean_groups(total_ok: int, lists: dict[str, list[str]], acc: _SummaryAccumulator) -> None:
    if not total_ok:
        return
    clean_specs = (
        ("clipping", "No digital clipping detected"),
        ("dc_offset", "No DC offset issues detected"),
        ("silence", "No silent files detected"),
        ("one_sided_silence", "No one-sided silent stereo files detected"),
        ("subsonic", "No significant subsonic content detected"),
    )
    for det_id, message in clean_specs:
        if not lists[det_id] and not acc.is_skipped(det_id):
            acc.add_group(acc.clean, message, None, [], standalone=True)


def _build_overview(
    ok_tracks: list[TrackContext], offset: float,
    most_common_sr: int | None, most_common_bd: str | None,
    most_common_len: int | None, most_common_len_fmt: str | None,
) -> dict[str, Any]:
    non_silent = [
        t for t in ok_tracks
        if not (t.detector_results.get("silence") or DetectorResult("", Severity.CLEAN, "", {})).data.get("is_silent", False)
    ]

    def _metric(track: TrackContext, key: str, apply_offset: bool) -> float:
        pr = track.processor_results.get("bimodal_normalize")
        val = pr.data.get(key) if pr else None
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val):
            return float(-np.inf)
        return float(val) + offset if apply_offset else float(val)

    overview: dict[str, Any] = {
        "total_ok": len(ok_tracks),
        "most_common_sr": most_common_sr,
        "most_common_bd": most_common_bd,
        "most_common_len": most_common_len,
        "most_common_len_fmt": most_common_len_fmt,
    }
    for key, prefix, apply_off in (
        ("detected_peak_db", "peak", False),
        ("detected_rms_db", "rms", True),
    ):
        scored = [(t, _metric(t, key, apply_off)) for t in non_silent]
        scored = [(t, v) for t, v in scored if np.isfinite(v)]
        if scored:
            vals = [v for _, v in scored]
            overview[f"loudest_{prefix}"] = max(scored, key=lambda pair: pair[1])[0].filename
            overview[f"loudest_{prefix}_db"] = max(vals)
            overview[f"quietest_{prefix}"] = min(scored, key=lambda pair: pair[1])[0].filename
            overview[f"quietest_{prefix}_db"] = min(vals)
            overview[f"median_{prefix}_db"] = float(np.median(vals))
    return overview


def build_diagnostic_summary(
    session: SessionContext,
    track_detectors: list | None = None,
    session_detectors: list | None = None,
    *,
    remaining: bool = False,
) -> dict[str, Any]:
    """Aggregate detector results into the four-category diagnostic summary."""
    ok_tracks, pending_tracks, failed_tracks, detectors = _prepare_summary_tracks(
        session, track_detectors, remaining=remaining,
    )
    total_ok = len(ok_tracks)
    offset = dbfs_offset(session.config)
    acc = _SummaryAccumulator({d.id: d for d in getattr(session, "detectors", None) or []})

    file_errors: list[str] = []
    for t in session.tracks:
        if t.status != "OK":
            acc.add_item(file_errors, t.filename, f"{t.status}")
    acc.add_group(acc.problems, "File errors", None, file_errors)

    sr, bd, length, length_fmt = _consensus_values(session)
    issue_names = (
        _collect_format_consistency(session, acc, total_ok, sr, bd)
        | _collect_length_consistency(session, acc, total_ok, length, length_fmt)
        | pending_tracks
        | failed_tracks
    )

    lists: dict[str, list[str]] = {
        k: [] for k in (
            "clipping", "dc_offset", "stereo_compat", "dual_mono",
            "silence", "one_sided_silence", "subsonic", "ultrasonic", "tail_exceedance",
        )
    }
    for t in ok_tracks:
        _collect_track_basic_findings(t, session, acc, lists, issue_names, offset, remaining=remaining)
        _collect_track_spectral_and_tail_findings(t, session, acc, lists, issue_names, offset)

    _route_detector_groups(ok_tracks, session, acc, lists, detectors, issue_names, remaining=remaining)
    grouping_items = [str(w) for w in session.warnings if str(w).startswith(GROUP_OVERLAP_PREFIX)]
    acc.add_group(acc.attention, "Grouping overlaps", "review group patterns", grouping_items)
    _add_clean_groups(total_ok, lists, acc)

    clean_count = sum(1 for t in ok_tracks if t.filename not in issue_names)
    overview = _build_overview(ok_tracks, offset, sr, bd, length, length_fmt)
    return {
        "problems": acc.problems,
        "attention": acc.attention,
        "information": acc.information,
        "clean": acc.clean,
        "clean_count": int(clean_count),
        "total_ok": int(total_ok),
        "overview": overview,
        **({"verification_pending": sorted(pending_tracks), "verification_failed": sorted(failed_tracks)} if remaining else {}),
    }
