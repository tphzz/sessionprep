"""Session-wide gain linking, DAW headroom and named fader anchors."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Collection

from .audio import INACTIVE_NORMALIZATION_CLASSIFICATIONS
from .models import GainStagingSummary, ProcessorResult, SessionContext, TrackContext
from .processors.bimodal_normalize import update_post_gain_metrics

log = logging.getLogger(__name__)


def _individual_gain_db(track: TrackContext, proc_id: str, result: ProcessorResult) -> float:
    if proc_id == "bimodal_normalize" and track.manual_gain_db is not None:
        return float(track.manual_gain_db)
    raw = result.data.get("gain_db_individual")
    if raw is not None:
        return float(raw)
    return float(result.gain_db)


def _apply_linked_group_gains(
    proc_id: str,
    groups: dict[str, list[tuple[TrackContext, ProcessorResult]]],
) -> None:
    for members in groups.values():
        if not members:
            continue
        gov_track, gov_result = min(members, key=lambda pair: pair[1].gain_db)
        group_gain = float(gov_result.gain_db)
        gov_constraint = str(gov_result.data.get("governing", "Peak"))
        for track, result in members:
            indiv = float(result.gain_db)
            limited_by_sibling = (track.filename != gov_track.filename) and (group_gain < indiv - 1e-9)
            result.gain_db = group_gain
            result.data["group_limited"] = limited_by_sibling
            result.data["group_governing_track"] = gov_track.filename if limited_by_sibling else None
            result.data["group_governing_constraint"] = gov_constraint if limited_by_sibling else None
            if proc_id == "bimodal_normalize":
                result.data["rms_anchor_active"] = (
                    (not limited_by_sibling) and str(result.data.get("governing", "")) == "RMS"
                )
                update_post_gain_metrics(result, track, group_gain)


def _rebalance_and_anchor(
    eligible: list[tuple[str, ProcessorResult]],
    *,
    proc_id: str,
    ceiling: float,
    headroom_db: float,
    anchor: str | None,
) -> tuple[float, float]:
    shift = 0.0
    if headroom_db > 0 and eligible:
        highest_fader = max(-result.gain_db for _, result in eligible)
        shift = max(0.0, highest_fader - (ceiling - headroom_db))
        if shift:
            log.debug(
                "Fader rebalance (%s): shifted %.1f dB down (ceiling %.1f dB, headroom %.1f dB)",
                proc_id, shift, ceiling, headroom_db,
            )
    for _, result in eligible:
        result.data["fader_offset"] = -result.gain_db - shift
        if shift:
            result.data["fader_rebalance_shift"] = shift

    anchor_offset = 0.0
    if anchor:
        for filename, result in eligible:
            if anchor.lower() in filename.lower():
                anchor_offset = float(result.data["fader_offset"])
                break
        for _, result in eligible:
            result.data["fader_offset"] -= anchor_offset
    return shift, anchor_offset


def apply_group_gains(
    session: SessionContext,
    *,
    linked_groups: Collection[str] | None,
    headroom_db: float,
    anchor: str | None,
) -> None:
    """Recompute linked group gains, DAW headroom rebalance, and named fader anchor."""
    ceiling = float(session.config.get("_fader_ceiling_db", 12.0))
    for proc in session.processors:
        eligible: list[tuple[str, ProcessorResult]] = []
        groups: dict[str, list[tuple[TrackContext, ProcessorResult]]] = defaultdict(list)
        for track in session.tracks:
            result = track.processor_results.get(proc.id)
            if result is None or track.status != "OK":
                continue
            result.data.pop("fader_rebalance_shift", None)
            result.data["group_limited"] = False
            result.data["group_governing_track"] = None
            result.data["group_governing_constraint"] = None
            if (
                result.error
                or result.classification in INACTIVE_NORMALIZATION_CLASSIFICATIONS
                or proc.id in track.processor_skip
            ):
                result.data["fader_offset"] = 0.0
                if proc.id == "bimodal_normalize":
                    result.data["rms_anchor_active"] = False
                continue
            indiv = _individual_gain_db(track, proc.id, result)
            if not math.isfinite(indiv):
                result.error = "Non-finite normalization gain quarantined"
                result.gain_db = 0.0
                result.data["fader_offset"] = 0.0
                continue
            result.gain_db = indiv
            result.data["gain_db_individual"] = indiv
            if proc.id == "bimodal_normalize":
                result.data["rms_anchor_active"] = str(result.data.get("governing", "")) == "RMS"
                update_post_gain_metrics(result, track, indiv)
            eligible.append((track.filename, result))
            if track.group is not None and (linked_groups is None or track.group in linked_groups):
                groups[track.group].append((track, result))

        _apply_linked_group_gains(proc.id, groups)
        shift, anchor_offset = _rebalance_and_anchor(
            eligible, proc_id=proc.id, ceiling=ceiling, headroom_db=headroom_db, anchor=anchor,
        )
        session.gain_staging[proc.id] = GainStagingSummary(
            fader_rebalance_shift=shift, anchor_offset=anchor_offset,
        )
        session.config[f"_fader_rebalance_{proc.id}"] = shift
        session.config[f"_anchor_offset_{proc.id}"] = anchor_offset
