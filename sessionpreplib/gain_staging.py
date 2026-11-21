"""Session-wide gain linking, DAW headroom and named fader anchors."""

import logging
from collections import defaultdict
from collections.abc import Collection

from .models import GainStagingSummary, ProcessorResult, SessionContext

log = logging.getLogger(__name__)


def apply_group_gains(
    session: SessionContext, *, linked_groups: Collection[str] | None,
    headroom_db: float, anchor: str | None,
) -> None:
    """Recompute from individual gains. A missing group filter links all groups."""
    ceiling = session.config.get("_fader_ceiling_db", 12.0)
    for proc in session.processors:
        eligible: list[tuple[str, ProcessorResult]] = []
        groups: dict[str, list[ProcessorResult]] = defaultdict(list)
        for track in session.tracks:
            result = track.processor_results.get(proc.id)
            if result is None or track.status != "OK":
                continue
            result.data.pop("fader_rebalance_shift", None)
            if result.error or result.classification in ("Skip", "Silent") or proc.id in track.processor_skip:
                result.data["fader_offset"] = 0.0
                continue
            result.gain_db = result.data.setdefault("original_gain_db", result.gain_db)
            eligible.append((track.filename, result))
            if track.group is not None and (linked_groups is None or track.group in linked_groups):
                groups[track.group].append(result)

        for members in groups.values():
            group_gain = min(result.gain_db for result in members)
            for result in members:
                result.gain_db = group_gain

        shift = 0.0
        if headroom_db > 0 and eligible:
            highest_fader = max(-result.gain_db for _, result in eligible)
            shift = max(0.0, highest_fader - (ceiling - headroom_db))
            if shift:
                log.debug("Fader rebalance (%s): shifted %.1f dB down (ceiling %.1f dB, headroom %.1f dB)",
                          proc.id, shift, ceiling, headroom_db)
        for _, result in eligible:
            result.data["fader_offset"] = -result.gain_db - shift
            if shift:
                result.data["fader_rebalance_shift"] = shift

        anchor_offset = 0.0
        if anchor:
            for filename, result in eligible:
                if anchor.lower() in filename.lower():
                    anchor_offset = result.data["fader_offset"]
                    break
            for _, result in eligible:
                result.data["fader_offset"] -= anchor_offset
        session.gain_staging[proc.id] = GainStagingSummary(
            fader_rebalance_shift=shift, anchor_offset=anchor_offset,
        )
        session.config[f"_fader_rebalance_{proc.id}"] = shift
        session.config[f"_anchor_offset_{proc.id}"] = anchor_offset
