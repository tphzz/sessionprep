"""Resolve detector evidence and sparse track choices into a reproducible render."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from .audio import SourceChangedError, file_revision, load_track, validate_analysis_level
from .detectors.dc_offset import DCOffsetDetector
from .detectors.subsonic import SubsonicDetector
from .detectors.ultrasonic import UltrasonicDetector
from .repairs import RepairRecipe, band_evidence, render_corrected


def _analyze(kind, track, config):
    detector = kind()
    detector.configure(config)
    return detector.analyze(track)


def repair_levels(data):
    """Per-channel signed DC and overall sample peak at unity gain."""
    if not data.size:
        return {"signed_dc": [], "peak_dbfs": -math.inf}
    peak = float(np.max(np.abs(data)))
    return {"signed_dc": np.atleast_1d(np.mean(data, axis=0, dtype=np.float64)).tolist(),
            "peak_dbfs": 20 * math.log10(peak) if peak else -math.inf}


def boundary_settling(source, corrected, rate):
    """Locate audible filter tails in channel-local runs of digital silence."""
    regions = []
    if not source.size:
        return regions
    minimum = math.ceil(rate / 10)  # 100 ms excludes ordinary zero crossings.
    for channel, (before, after) in enumerate(zip(source.reshape(len(source), -1).T,
                                                  corrected.reshape(len(corrected), -1).T)):
        edges = np.flatnonzero(np.diff(before == 0, prepend=False, append=False))
        runs = edges.reshape(-1, 2)
        for start, stop in runs[runs[:, 1] - runs[:, 0] >= minimum]:
            magnitude = np.abs(after[start:stop])
            peak = float(np.max(magnitude))
            if peak > 1e-6:  # -120 dBFS sample peak, before any gain.
                above = magnitude > 1e-6
                regions.append({"channel": channel, "start_sample": int(start + above.argmax()),
                                "end_sample": int(stop - 1 - above[::-1].argmax()),
                                "peak_dbfs": 20 * math.log10(peak)})
    return regions


def _persistent_bias(data, rate, threshold):
    width = max(1, rate // 2)
    n = len(data) // width
    if n < 3:
        return False
    means = data[:n * width].reshape(n, width, -1).mean(axis=1, dtype=np.float64)
    for mask in (means > threshold, means < -threshold):
        if (mask[:-2] & mask[1:-1] & mask[2:]).any():
            return True
    return False


def _low_cut_evidence(data, rate, config):
    return band_evidence(
        data, rate, low_hz=0, high_hz=min(20, config.get("subsonic_hz", 30)),
        window_sec=1.0, level_db=-50, ratio_db=config.get("subsonic_warn_ratio_db", -20),
        consecutive=3,
    )


def check_source_signature(track, *, allow_digest=True):
    if track.source_file_signature is not None:
        stat = Path(track.filepath).stat()
        changed = (stat.st_size, stat.st_mtime_ns) != track.source_file_signature
        if not changed and track.source_file_revision is not None:
            revision = file_revision(track.filepath, stat, allow_digest=allow_digest)
            changed = revision is not None and revision != track.source_file_revision
        if changed:
            raise SourceChangedError(f"{track.filename} has changed since analysis. Reanalyze this source.")


def revalidate_restored_source(track):
    """On explicit retry, recognize restored PCM even if the filesystem revision changed."""
    try:
        check_source_signature(track)
    except SourceChangedError:
        if not track.source_sample_digest:
            raise
        restored = load_track(track.filepath)
        if (restored.source_sample_digest != track.source_sample_digest
                or (restored.total_samples, restored.channels, restored.samplerate, restored.subtype)
                != (track.total_samples, track.channels, track.samplerate, track.subtype)):
            raise
        track.source_file_signature = restored.source_file_signature
        track.source_file_revision = restored.source_file_revision
        if track.audio_data is None:
            track.audio_data = restored.audio_data


def source_for_render(track):
    """Guard a saved plan against a replaced file before any audition or write."""
    if "repair_error" in track._cache:
        raise ValueError(track._cache["repair_error"])
    check_source_signature(track)
    if track.audio_data is not None:
        return track
    loaded = load_track(track.filepath)
    if ((track.source_file_signature is not None and loaded.source_file_signature != track.source_file_signature)
            or (track.source_file_revision is not None and loaded.source_file_revision != track.source_file_revision)):
        raise SourceChangedError(f"{track.filename} has changed since analysis. Reanalyze this source.")
    if (loaded.samplerate, loaded.channels, loaded.total_samples, loaded.subtype) != (
            track.samplerate, track.channels, track.total_samples, track.subtype):
        raise SourceChangedError(f"{track.filename} no longer matches its saved analysis. Reanalyze this source.")
    return replace(track, audio_data=loaded.audio_data, _cache={},
                   source_file_signature=loaded.source_file_signature, source_file_revision=loaded.source_file_revision,
                   source_sample_digest=loaded.source_sample_digest)


def plan_repairs(track, config, verification=None):
    """Keep source findings intact. Return a temporary track for gain analysis."""
    choices = track.repair_overrides
    dc = track.detector_results.get("dc_offset")
    sub = track.detector_results.get("subsonic")
    high_rate = track.samplerate >= 50000
    consider_dc = config.get("dc_offset_enabled", True) and (
        dc is not None and dc.data.get("dc_warn") or config.get("repair_auto_dc"))
    consider_low = config.get("subsonic_enabled", True) and (
        sub is not None or config.get("repair_auto_low_cut"))
    if not (choices or consider_dc or consider_low or high_rate):
        track.repair_recipe = RepairRecipe()
        track.repair_measurements = {}
        track.corrected_analysis = None
        return track
    source = source_for_render(track)
    cached = track._cache.pop("unvalidated_corrected_analysis", None)
    if cached is not None:
        # source_for_render just checked the complete on-disk revision.
        source.corrected_analysis = cached
    data = source.audio_data
    validate_analysis_level(np.max(np.abs(data), initial=0.), len(data))
    rate = source.samplerate
    if (any(choices.get(key) for key in ("dc_remove", "low_cut", "ultrasonic"))
            or any(config.get(key) for key in ("repair_auto_dc", "repair_auto_low_cut", "repair_auto_ultrasonic"))):
        if verification is None:
            from .detectors import default_detectors
            from .pipeline import Pipeline
            verification = Pipeline(default_detectors(), config=config).repair_verification
        verification.complete_original(source)
        dc = source.detector_results.get("dc_offset")
        sub = source.detector_results.get("subsonic")
        consider_dc = config.get("dc_offset_enabled", True) and (dc is not None or config.get("repair_auto_dc"))
        consider_low = config.get("subsonic_enabled", True) and (sub is not None or config.get("repair_auto_low_cut"))
    if consider_dc and dc is None:
        dc = _analyze(DCOffsetDetector, source, config)
    dc_suggested = bool(dc and dc.data.get("dc_warn") and config.get("dc_offset_enabled", True))
    dc_auto = bool(config.get("repair_auto_dc") and dc_suggested and
                   _persistent_bias(data, rate, 10 ** (config.get("dc_offset_warn_db", -40) / 20)))
    dc_on = choices.get("dc_remove", dc_auto)
    if consider_low and sub is None:
        sub = _analyze(SubsonicDetector, source, config)

    measure_low = consider_low or choices.get("low_cut", False)
    original_low = _low_cut_evidence(data, rate, config) if measure_low else {}
    low = original_low
    dc_audio = None
    if measure_low and dc_on:
        dc_audio = render_corrected(data, rate, RepairRecipe(dc_remove=True))
        low = _low_cut_evidence(dc_audio, rate, config)
    low_match = bool(consider_low and low.get("regions"))
    low_on = choices.get("low_cut", bool(config.get("repair_auto_low_cut") and low_match))
    high = source.detector_results.get("ultrasonic")
    high_enabled = high_rate and config.get("ultrasonic_enabled", True)
    if high is None and high_rate and (high_enabled or choices.get("ultrasonic")):
        high = _analyze(UltrasonicDetector, source, config)
        if high_enabled:
            track.detector_results["ultrasonic"] = high
    high_match = bool(high_enabled and high and high.data.get("regions"))
    high_on = choices.get("ultrasonic", bool(config.get("repair_auto_ultrasonic") and high_match))
    recipe = RepairRecipe(dc_on, choices.get("low_cut_hz", config.get("repair_low_cut_hz", 10))
                          if low_on else None, high_on)
    measurements = {
        "suggested": {"dc_remove": dc_suggested, "low_cut": low_match, "ultrasonic": high_match},
        "inherited": {"dc_remove": dc_auto, "low_cut": bool(config.get("repair_auto_low_cut") and low_match),
                      "ultrasonic": bool(config.get("repair_auto_ultrasonic") and high_match)},
        "original": repair_levels(data), "low_band": low,
    }
    if measure_low:
        measurements["original"]["low_band"] = original_low
    if sub:
        measurements["original"]["subsonic_ratio_db"] = sub.data.get("subsonic_ratio_db")
    if high:
        measurements["original"]["ultrasonic"] = high.data
    gain_track = source
    checked = None
    if recipe.active:
        audio = dc_audio if dc_audio is not None and not low_on and not high_on else render_corrected(data, rate, recipe)
        gain_track = replace(source, audio_data=audio, _cache={}, detector_results={}, processor_results={})
        checked = verification.corrected(source, gain_track, recipe)
        most_recent = repair_levels(audio)
        measurements["boundary_settling"] = boundary_settling(data, audio, rate)
        if measure_low:
            corrected_low = low if audio is dc_audio else _low_cut_evidence(audio, rate, config)
            most_recent["low_band"] = corrected_low
            if low_on and low.get("regions"):
                measurements["low_cut_resolved"] = not corrected_low["regions"]
        result = gain_track.detector_results.get("subsonic")
        if result and not result.error and result.applicable:
            most_recent["subsonic_ratio_db"] = result.data["subsonic_ratio_db"]
            if sub and sub.data.get("subsonic_warn"):
                measurements["subsonic_resolved"] = not result.data["subsonic_warn"]
        result = gain_track.detector_results.get("ultrasonic")
        if result is None and high_rate and high_on:
            # A manually selected filter still has before/after meters when
            # the diagnostic is disabled. These readings are not findings.
            result = _analyze(UltrasonicDetector, gain_track, config)
        if high_rate and result and not result.error and result.applicable:
            most_recent["ultrasonic"] = result.data
        measurements["corrected"] = most_recent
        check_source_signature(source)
    else:
        measurements["corrected"] = measurements["original"].copy()
    track.repair_measurements = measurements
    track.repair_recipe = recipe
    track.detector_analysis_key = source.detector_analysis_key
    track.detector_sample_digest = source.detector_sample_digest
    track.source_file_signature = source.source_file_signature
    track.source_file_revision = source.source_file_revision
    track.source_sample_digest = source.source_sample_digest
    track.corrected_analysis = checked
    return gain_track
