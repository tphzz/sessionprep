"""Resolve the repair controls and links for one track from stored evidence."""
from __future__ import annotations

from dataclasses import dataclass

from sessionpreplib.repair_verification import (
    CheckOutcome, detector_check, local_band_regions, source_low_band,
)

REPAIR_NAMES = {"dc_remove": "DC removal", "low_cut": "Subsonic filter", "ultrasonic": "Ultrasonic filter"}
DETECTOR_REPAIRS = {"dc_offset": "dc_remove", "subsonic": "low_cut", "subsonic-local": "low_cut",
                    "ultrasonic": "ultrasonic"}


@dataclass
class RepairOptions:
    rows: set[str]
    targets: dict[str, str]
    errors: list[str]
    gain_adjustment: dict | None
    has_stale_output: bool = False

    @property
    def available(self):
        return bool(self.rows or self.errors or self.gain_adjustment)

    def attribution(self, finding):
        target = self.targets.get(finding)
        if target and target != DETECTOR_REPAIRS[finding]:
            return f"Resolved by {REPAIR_NAMES[target]}"
        return ""


def track_repair_options(track, session=None):
    if track.status != "OK":
        return RepairOptions(set(), {}, [], None)
    config = session.config if session else {}
    recipe = track.repair_recipe
    applied = recipe.applied_ids()
    rows = set(applied)
    overrides = track.repair_overrides
    rows.update(REPAIR_NAMES.keys() & overrides.keys())
    if "low_cut_hz" in overrides:
        rows.add("low_cut")
    targets = {}

    def add_finding(finding, repair, resolved):
        if resolved and repair not in applied:
            # With several active processors the isolated cause is unknown.
            if len(applied) == 1:
                targets[finding] = applied[0]
        else:
            targets[finding] = repair

    suggested = track.repair_measurements.get("suggested", {})
    pending = bool(track._cache.get("repair_verification_pending"))
    for detector, flag in (("dc_offset", "dc_warn"), ("subsonic", "subsonic_warn"), ("ultrasonic", "regions")):
        repair = DETECTOR_REPAIRS[detector]
        if (not config.get(f"{detector}_enabled", True) or config.get(f"{detector}_report_as") == "skip"
                or (repair == "ultrasonic" and track.samplerate < 50000)):
            continue
        check = detector_check(track, detector, pending=pending)
        measured = any(r and r.applicable and not r.error and r.data.get(flag)
                       for r in (check.original, check.corrected))
        # Older plans can store a detector-backed suggestion without its result.
        if measured or (check.original is None and repair != "low_cut" and suggested.get(repair)):
            add_finding(detector, repair, check.outcome == CheckOutcome.RESOLVED)
        if detector == "subsonic":
            before = local_band_regions(source_low_band(track), track)
            after = local_band_regions(source_low_band(track, corrected=True), track)
            if before or after:
                cleared = (bool(before) and after == [] and check.corrected is not None
                           and check.corrected.applicable and not check.corrected.error)
                add_finding("subsonic-local", repair, cleared)
    rows.update(targets.values())

    errors = []
    adjustment = config.get("_output_gain_adjustments", {}).get(track.filename)
    stale = False
    if config.get("_prepare_errors") or adjustment:
        names = {track.filename}
        if session.topology:
            names.update(entry.output_filename for entry in session.topology.entries
                         if any(src.input_filename == track.filename for src in entry.sources))
        errors = [str(message) for name, message in config.get("_prepare_errors", ()) if name in names]
        stale = session.prepare_state == "stale" and any(
            t.filename in names and t.processed_filepath for t in session.output_tracks)
    failure = track._cache.get("repair_error")
    if failure and failure not in errors:
        errors.append(failure)
    return RepairOptions(rows, targets, errors, adjustment, stale)
