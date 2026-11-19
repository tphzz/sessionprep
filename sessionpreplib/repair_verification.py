"""Measure each audition source with the same configured signal detectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import os

from .audio import sample_digest
from .config import ANALYSIS_PARAMS
from .detector import TrackDetector
from .detectors.audio_classifier import AudioClassifierDetector
from .detectors.silence import SilenceDetector
from .models import CorrectedAnalysis, DetectorResult, IssueLocation, Severity


class CheckOutcome(Enum):
    ORIGINAL = "original"
    PENDING = "pending"
    FAILED = "failed"
    INAPPLICABLE = "inapplicable"
    RESOLVED = "resolved"
    REMAINING = "remaining"
    NEW = "new"
    CLEAN = "clean"


@dataclass(frozen=True)
class DetectorCheck:
    original: DetectorResult | None
    corrected: DetectorResult | None
    outcome: CheckOutcome

    @property
    def remaining_result(self) -> DetectorResult | None:
        """Carry source findings forward while the corrected check is unavailable."""
        if self.corrected is not None and self.corrected.applicable and not self.corrected.error:
            return self.corrected
        return self.original


def detector_check(track, detector_id, *, pending=False) -> DetectorCheck:
    """Compare the two actual measurements independent of the audition source."""
    before = track.detector_results.get(detector_id)
    failure = track._cache.get("repair_error")
    if pending:
        return DetectorCheck(before, None, CheckOutcome.PENDING)
    if failure:
        return DetectorCheck(before, None, CheckOutcome.FAILED)
    if not track.repair_recipe.active:
        return DetectorCheck(before, None, CheckOutcome.ORIGINAL)
    checked = track.corrected_analysis
    after = checked.results.get(detector_id) if checked is not None else None
    return _compare_results(before, after)


def _compare_results(before, after):
    if after is None:
        status = CheckOutcome.PENDING
    elif after.error:
        status = CheckOutcome.FAILED
    elif not after.applicable:
        status = CheckOutcome.INAPPLICABLE
    elif before is None or before.error or not before.applicable:
        # An unavailable original measurement cannot establish that a finding
        # is new or that a repair resolved it.
        status = CheckOutcome.REMAINING if after.severity != Severity.CLEAN else CheckOutcome.INAPPLICABLE
    elif after.severity == Severity.CLEAN:
        status = CheckOutcome.CLEAN if before.severity == Severity.CLEAN else CheckOutcome.RESOLVED
    else:
        status = CheckOutcome.NEW if before.severity == Severity.CLEAN else CheckOutcome.REMAINING
    return DetectorCheck(before, after, status)


def audition_results(track, *, corrected: bool, pending=False):
    """Return only evidence recorded on the selected signal, including its overlay bounds."""
    if not corrected:
        return track.detector_results
    if pending or track._cache.get("repair_error"):
        return {}
    if not track.repair_recipe.active:
        return track.detector_results
    return track.corrected_analysis.results if track.corrected_analysis is not None else {}


def local_band_regions(band, track):
    """Return the measured regions, or None for absent or invalid saved evidence."""
    regions = band.get("regions") if isinstance(band, dict) else None
    if not isinstance(regions, (list, tuple)):
        return None
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 3:
            return None
        start, end, channel = region
        if not all(type(value) is int for value in region):
            return None
        if not (0 <= start <= end < track.total_samples and 0 <= channel < track.channels):
            return None
    return regions


def source_low_band(track, *, corrected=False):
    """Read the stored local measurement for one audition source."""
    if corrected and (track._cache.get("repair_verification_pending") or track._cache.get("repair_error")
                      or (track.repair_recipe.active and not track.corrected_analysis)):
        return None
    measurements = track.repair_measurements
    if corrected and track.repair_recipe.active:
        return measurements.get("corrected", {}).get("low_band")
    band = measurements.get("original", {}).get("low_band")
    if band is None and not track.repair_recipe.dc_remove:
        # Older plans saved their low-cut input. It equals Original only
        # when an earlier DC filter did not change that signal.
        band = measurements.get("low_band")
    return band


def local_rumble(track, config, *, corrected=False):
    """Expose the planner's sustained local band observation separately from the whole-track detector."""
    band = source_low_band(track, corrected=corrected)
    regions = local_band_regions(band, track)
    if regions is None:
        return None
    cutoff = min(config.get("subsonic_hz", 30), 20)
    issues = [IssueLocation(start, end, channel, Severity.INFO, "subsonic-local",
                            f"Sustained local energy at or below {cutoff:g} Hz", 0, cutoff)
              for start, end, channel in regions]
    count = len(issues)
    description = f"Sustained infrasonic energy in {count} local {'region' if count == 1 else 'regions'}"
    return DetectorResult("subsonic-local", Severity.INFO if count else Severity.CLEAN, description,
                          band, hint="review rumble" if count else None, issues=issues)


def local_rumble_check(track, config, whole_check):
    """Use local regions for the local verdict once the corrected check completes."""
    before = local_rumble(track, config)
    if not track.repair_recipe.active:
        return DetectorCheck(before, None, whole_check.outcome)
    after = local_rumble(track, config, corrected=True)
    if after and after.issues:
        return _compare_results(before, after)
    if whole_check.corrected is None or whole_check.corrected.error or not whole_check.corrected.applicable:
        return DetectorCheck(before, None, whole_check.outcome)
    return _compare_results(before, after)


def separate_rumble_finding(track, config):
    """Expose local evidence when the global finding does not cover it."""
    if not config.get("subsonic_enabled", True) or config.get("subsonic_report_as") == "skip":
        return None
    whole = detector_check(track, "subsonic", pending=bool(track._cache.get("repair_verification_pending")))
    local = local_rumble_check(track, config, whole)
    if not any(result and result.issues for result in (local.original, local.corrected)):
        return None
    if (whole.original and not whole.original.error and whole.original.data.get("subsonic_warn")
            and not (local.corrected and local.corrected.issues and
                     (not whole.corrected or not whole.corrected.data.get("subsonic_warn")))):
        return None
    return local


def original_report_track(track, session):
    """Restore the original normalization context for detector relevance using stored analysis."""
    if not track.repair_recipe.active or session is None or "tail_exceedance" not in track.detector_results:
        return track
    norm = next((p for p in session.processors if p.id == "bimodal_normalize"), None)
    classifier = track.detector_results.get("audio_classifier")
    if norm is None or classifier is None or classifier.error:
        return track
    # Bimodal gain is calculated from the stored classifier scalars. No audio is read.
    try:
        original = norm.process(track)
    except (KeyError, TypeError, ValueError):
        # Legacy reports can lack classifier scalars until background verification completes.
        return track
    return replace(track, processor_results={**track.processor_results, norm.id: original})


def remaining_results(track, detectors, *, pending=False):
    """Unify signal verdicts with session-wide format and length observations."""
    if not (track.repair_recipe.active or pending or track._cache.get("repair_error")):
        return track.detector_results
    results = dict(track.detector_results)
    ids = [d.id for d in detectors if isinstance(d, TrackDetector)]
    if not detectors and track.corrected_analysis is not None:
        ids = track.corrected_analysis.results
    for detector_id in ids:
        result = detector_check(track, detector_id, pending=pending).remaining_result
        if result is not None:
            results[detector_id] = result
    return results


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def measure_detector(detector, track, *, require_dependencies=True, allowed=None):
    """A skipped prerequisite cannot establish a clean measurement downstream."""
    try:
        for dep in detector.depends_on:
            if allowed is not None and dep not in allowed:
                raise ValueError(f"Required detector {dep} is disabled")
            result = track.detector_results.get(dep)
            if result is None and not require_dependencies:
                continue
            if result is None or result.error:
                raise ValueError(f"Required detector {dep} is unavailable")
            if not result.applicable:
                return DetectorResult(detector.id, Severity.CLEAN, f"Requires applicable {dep}",
                                      {}, applicable=False)
        return detector.analyze(track)
    except Exception as exc:  # A detector failure must not discard its siblings' measurements.
        return DetectorResult(detector.id, Severity.PROBLEM, f"Detector error: {exc}", {},
                              error=str(exc) or type(exc).__name__, applicable=False)


class RepairVerification:
    """An immutable configuration shared by the worker's per-track checks."""

    def __init__(self, detectors, config):
        self.detectors = [d for d in detectors if config.get(f"{d.id}_enabled", True)]
        self.ids = {d.id for d in self.detectors}
        # These two measurements are also inputs to gain planning. Some library
        # callers configure a processor without registering its diagnostic rows.
        self.support = []
        for kind in (SilenceDetector, AudioClassifierDetector):
            if kind.id not in self.ids:
                d = kind()
                d.configure(config)
                self.support.append(d)
        self.required_ids = self.ids | {d.id for d in self.support}
        self.disabled_ids = {d.id for d in detectors} - self.required_ids
        registered = [*self.support, *self.detectors]
        silence = next(d for d in registered if d.id == "silence")
        self.ordered = [silence, *(d for d in registered if d is not silence)]
        settings = {s.key: config[s.key] for s in ANALYSIS_PARAMS
                    if s.key in config and not s.presentation_only and s.key != "fader_headroom_db"}
        settings.update({key: config.get(key, []) for key in ("force_transient", "force_sustained")})
        for detector in (*self.support, *self.detectors):
            for spec in detector.config_params():
                if not spec.presentation_only and spec.key in config:
                    # A detector's legacy configure({}) fallback can differ
                    # from the value its ParamSpec inserts in a new preset.
                    settings[spec.key] = config[spec.key]
        implementations = [(d.id, type(d).__module__, type(d).__qualname__, d.depends_on)
                           for d in self.detectors]
        # Bump the evidence version when detector or render semantics change.
        self.settings_key = _digest([4, implementations, settings])

    def original_key(self, track):
        samples = track.detector_sample_digest
        if track.audio_data is not None:
            samples = sample_digest(track.audio_data)
            if track.detector_sample_digest is not None and samples != track.detector_sample_digest:
                # Cached peak, RMS envelopes and stereo measurements all refer
                # to the previous PCM, including when the same array was edited.
                track._cache = {}
                track.detector_analysis_key = None
                track.corrected_analysis = None
            track.detector_sample_digest = samples
        return self._original_key(track)

    def _original_key(self, track):
        if track.detector_sample_digest is None:
            return None
        return _digest([self.settings_key, os.path.normcase(os.path.abspath(track.filepath)),
                        track.filename, track.source_file_signature, track.detector_sample_digest,
                        track.samplerate, track.channels,
                        track.total_samples, track.subtype, track.rms_anchor_override])

    def corrected_key(self, track, recipe):
        return _digest([self.original_key(track), asdict(recipe)])

    def has_current_evidence(self, track):
        """Cheap restore check after validating the source file's revision."""
        snapshot = track.corrected_analysis
        if snapshot is None or not track.repair_recipe.active:
            return False
        key = self._original_key(track)
        return (key is not None and track.detector_analysis_key == key
                and snapshot.key == _digest([key, asdict(track.repair_recipe)])
                and self.ids == snapshot.results.keys()
                and self.required_ids <= track.detector_results.keys())

    def _original_ids(self, track, key):
        if key is not None and key == track.detector_analysis_key:
            return set(track.detector_results)
        phase_key, ids = track._cache.get("detectors_measured", (None, ()))
        return set(ids) if key == phase_key else set()

    def prepare_phase(self, track, detectors):
        """Measure prerequisites of a standalone phase once on the original PCM."""
        key = self.original_key(track)
        measured = self._original_ids(track, key)
        visiting = {d.id for d in detectors}
        required = set()
        for detector in reversed(self.ordered):
            if detector.id in visiting:
                required.update(detector.depends_on)
                visiting.update(detector.depends_on)
        required.difference_update(d.id for d in detectors)
        for detector in self.ordered:
            if detector.id in required and detector.id not in measured:
                track.detector_results[detector.id] = measure_detector(detector, track, allowed=self.required_ids)
                measured.add(detector.id)
        track._cache["detectors_measured"] = key, measured
        return key

    def record_phase(self, track, detectors, key):
        """Only vouch for results measured with current prerequisites in either phase."""
        measured = self._original_ids(track, key)
        for detector in detectors:
            if measured.issuperset(detector.depends_on):
                measured.add(detector.id)
            else:
                measured.discard(detector.id)
        track._cache["detectors_measured"] = (key, measured)
        track.detector_analysis_key = key if self.ids <= measured else None
        track.corrected_analysis = None

    def complete_original(self, track):
        """Fill missing analysis phases and refresh only stale or failed evidence."""
        key = self.original_key(track)
        measured = self._original_ids(track, key)
        # A source without matching provenance can also have obsolete signal
        # caches. Keep caller-owned planning and UI state on the original track.
        source = track if measured else replace(track, _cache={})
        self._run(source, measured, preserve_equal=True)
        track.detector_analysis_key = key

    def _run(self, track, measured, *, preserve_equal=False):
        refreshed = set()
        for detector in self.ordered:
            prior = track.detector_results.get(detector.id)
            if (detector.id not in measured or prior is None or prior.error
                    or refreshed.intersection(detector.depends_on)):
                result = measure_detector(detector, track, allowed=self.required_ids)
                if preserve_equal and prior is not None:
                    try:
                        if result == prior:
                            result = prior
                    except (TypeError, ValueError):
                        # Extension detectors may return arrays without scalar equality.
                        pass
                track.detector_results[detector.id] = result
                refreshed.add(detector.id)

    def corrected(self, track, gain_track, recipe):
        # complete_original fingerprints the precise input to this render.
        key = _digest([track.detector_analysis_key, asdict(recipe)])
        previous = track.corrected_analysis
        fresh = previous is not None and previous.key == key
        gain_track.detector_results = {**previous.support_results, **previous.results} if fresh else {}
        self._run(gain_track, set(gain_track.detector_results))
        return CorrectedAnalysis(key, {k: v for k, v in gain_track.detector_results.items() if k in self.ids},
                                 {d.id: gain_track.detector_results[d.id] for d in self.support})
