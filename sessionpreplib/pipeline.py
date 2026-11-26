from __future__ import annotations

import os
import time
import math
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from collections.abc import Callable

import logging
from collections import Counter
from collections.abc import Collection

import numpy as np

from .gain_staging import apply_group_gains
from .processors.bimodal_normalize import update_post_gain_metrics
from .models import (
    ConfigError,
    DetectorResult,
    ProcessorResult,
    Severity,
    TrackContext,
    SessionContext,
    LifecyclePhase,
)
from .detector import TrackDetector, SessionDetector, topo_sort_detectors as _topo_sort_detectors
from .processor import AudioProcessor
from .repair_plan import boundary_settling, plan_repairs, repair_levels, revalidate_restored_source, source_for_render
from .repair_verification import RepairVerification, measure_detector
from .repairs import OutputHeadroomError, check_output_headroom, render_corrected
from .events import EventBus
from .file_io import ensure_audio_backup
from .audio import INACTIVE_NORMALIZATION_CLASSIFICATIONS, db_to_linear, file_revision, load_track, linear_to_db
from .utils import (
    parse_group_specs,
    assign_groups,
)

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        detectors: list,
        audio_processors: list[AudioProcessor] | None = None,
        config: dict[str, Any] | None = None,
        event_bus: EventBus | None = None,
        max_workers: int | None = None,
    ):
        self.config = config or {}
        self.event_bus = event_bus
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

        # Separate track vs session detectors
        self.track_detectors: list[TrackDetector] = [
            d for d in detectors if isinstance(d, TrackDetector)
        ]
        self.session_detectors: list[SessionDetector] = [
            d for d in detectors if isinstance(d, SessionDetector)
        ]

        # Topologically sort track detectors by depends_on
        self.track_detectors = _topo_sort_detectors(self.track_detectors)

        # Configure all components
        for d in self.track_detectors:
            d.configure(self.config)
        for sd in self.session_detectors:
            sd.configure(self.config)
        self.repair_verification = RepairVerification(self.track_detectors, self.config)

        # Configure audio processors first, then filter to enabled only
        all_procs = audio_processors or []
        for p in all_procs:
            p.configure(self.config)
        self.audio_processors: list[AudioProcessor] = sorted(
            [p for p in all_procs if p.enabled],
            key=lambda p: (p.phase, p.priority),
        )

        # Validate
        self._validate()

    def _validate(self):
        """Validate pipeline configuration at construction time."""
        all_ids = {d.id for d in self.track_detectors}
        for d in self.session_detectors:
            if d.id in all_ids:
                raise ConfigError(f"Duplicate detector ID: {d.id}")
            all_ids.add(d.id)
        for p in self.audio_processors:
            if p.id in all_ids:
                raise ConfigError(f"Duplicate processor ID: {p.id}")
            all_ids.add(p.id)

    def _emit(self, event_type: str, **data):
        if self.event_bus:
            self.event_bus.emit(event_type, **data)

    # ------------------------------------------------------------------
    # Phase 1: Analyze (run all detectors)
    # ------------------------------------------------------------------

    def _analyze_track(self, track: TrackContext, idx: int, total: int, detectors: list[TrackDetector]):
        """Run all track-level detectors for a single track (thread-safe)."""
        if self.event_bus and self.event_bus.is_cancelled.is_set():
            return

        self._emit("track.analyze_start", filename=track.filename,
                   index=idx, total=total)
        t_track_start = time.perf_counter()
        original_key = self.repair_verification.prepare_phase(track, detectors)
        for det in detectors:
            try:
                self._emit("detector.start", detector_id=det.id,
                           filename=track.filename)
                t0 = time.perf_counter()
                result = measure_detector(det, track, require_dependencies=False,
                                          allowed=self.repair_verification.required_ids)
                dt = (time.perf_counter() - t0) * 1000
                log.debug(f"detector {det.id} on {track.filename}: {dt:.1f} ms")
                track.detector_results[det.id] = result
                self._emit("detector.complete", detector_id=det.id,
                           filename=track.filename,
                           severity=result.severity)
            except Exception as e:
                track.detector_results[det.id] = DetectorResult(
                    detector_id=det.id,
                    severity=Severity.PROBLEM,
                    summary=f"detector error: {e}",
                    data={},
                    error=str(e),
                )
        self.repair_verification.record_phase(track, detectors, original_key)
        dt_track = (time.perf_counter() - t_track_start) * 1000
        log.debug(f"all detectors on {track.filename}: {dt_track:.1f} ms")
        self._emit("track.analyze_complete", filename=track.filename,
                   index=idx, total=total)

    def _run_analysis_phase(self, session: SessionContext, phase: LifecyclePhase) -> SessionContext:
        """Run track-level and session-level detectors matching the specified phase."""
        phase_detectors = [d for d in self.track_detectors if d.phase == phase]
        track_dets = [d for d in phase_detectors if d.id in self.repair_verification.required_ids]
        disabled = {d.id for d in phase_detectors} - self.repair_verification.required_ids
        if disabled:
            for track in session.tracks:
                for detector_id in disabled:
                    track.detector_results.pop(detector_id, None)

        total = len(session.tracks)
        ok_items = [
            (idx, track)
            for idx, track in enumerate(session.tracks)
            if track.status == "OK"
        ]

        n = len(ok_items)
        t_phase = time.perf_counter()
        workers = min(self.max_workers, n) if ok_items else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._analyze_track, track, idx, total, track_dets): (idx, track)
                for idx, track in ok_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    idx, track = futures[future]
                    track.status = f"Error: {exc}"
                    self._emit("track.analyze_complete",
                               filename=track.filename,
                               index=idx, total=total)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        if n:
            log.debug(f"analyze {phase.value} (track detectors): {n} tracks in "
                f"{dt_phase:.1f} ms ({dt_phase / n:.1f} ms/track avg)")

        self.analyze_session_detectors(session, phase)
        # Keep the complete detector set available for GUI overlay generation.
        session.detectors = self.repair_verification.detectors + self.session_detectors
        return session

    def analyze_session_detectors(self, session: SessionContext, phase: LifecyclePhase) -> None:
        """Update findings that depend on the complete set of tracks in this phase."""
        track_map = {t.filename: t for t in session.tracks}
        for det in self.session_detectors:
            if det.phase != phase:
                continue
            if self.event_bus and self.event_bus.is_cancelled.is_set():
                break

            for track in session.tracks:
                track.detector_results.pop(det.id, None)
            try:
                self._emit("session_detector.start", detector_id=det.id)
                t0 = time.perf_counter()
                results = det.analyze(session)
                dt = (time.perf_counter() - t0) * 1000
                log.debug(f"session detector {det.id}: {dt:.1f} ms")
                session.session_detector_results[det.id] = results
                session.config[f"_session_det_{det.id}"] = results
                # Distribute per-track results back into each track
                for result in results:
                    fname = result.data.get("filename")
                    if fname and fname in track_map:
                        track_map[fname].detector_results[det.id] = result
                self._emit("session_detector.complete", detector_id=det.id)
            except Exception as e:
                err_results = [
                    DetectorResult(
                        detector_id=det.id,
                        severity=Severity.PROBLEM,
                        summary=f"session detector error: {e}",
                        data={},
                        error=str(e),
                    )
                ]
                session.session_detector_results[det.id] = err_results
                session.config[f"_session_det_{det.id}"] = err_results

    def analyze_phase1(self, session: SessionContext) -> SessionContext:
        """Run structural/formatting detectors for track layout and validation."""
        log.info("Pipeline: analyze phase1 (%d tracks)", len(session.tracks))
        return self._run_analysis_phase(session, LifecyclePhase.PHASE1)

    def analyze_phase2(self, session: SessionContext) -> SessionContext:
        """Run acoustic and DSP-based detectors."""
        log.info("Pipeline: analyze phase2 (%d tracks)", len(session.tracks))
        return self._run_analysis_phase(session, LifecyclePhase.PHASE2)

    # ------------------------------------------------------------------
    # Phase 2: Plan (run audio processors, compute gains)
    # ------------------------------------------------------------------

    def _plan_track(self, track: TrackContext, idx: int, total: int, anchor_detectors=(), revalidate_source=False):
        """Run all audio processors for a single track (thread-safe)."""
        if self.event_bus and self.event_bus.is_cancelled.is_set():
            return

        self._emit("track.plan_start", filename=track.filename,
                   index=idx, total=total)
        t_track_start = time.perf_counter()
        track._cache.pop("repair_error", None)
        track._cache.pop("_repair_emitted", None)
        for detector_id in self.repair_verification.disabled_ids:
            track.detector_results.pop(detector_id, None)
        try:
            if revalidate_source:
                revalidate_restored_source(track)
            source = track
            if anchor_detectors:
                source = source_for_render(track)
                self._analyze_track(source, idx, total, anchor_detectors)
            gain_track = plan_repairs(source, self.config, self.repair_verification)
            track.repair_recipe = source.repair_recipe
            track.repair_measurements = source.repair_measurements
            track.detector_analysis_key = source.detector_analysis_key
            track.detector_sample_digest = source.detector_sample_digest
            track.corrected_analysis = source.corrected_analysis
            track.source_file_signature = source.source_file_signature
            track.source_file_revision = source.source_file_revision
            track.source_sample_digest = source.source_sample_digest
        except Exception as exc:
            track.corrected_analysis = None
            track._cache["repair_error"] = str(exc)
            track.processor_results.clear()
            track._cache["_repair_emitted"] = True
            self._emit("track.repair_complete", filename=track.filename,
                       index=idx, total=total)
            self._emit("track.plan_complete", filename=track.filename, index=idx, total=total)
            return
        track._cache["_repair_emitted"] = True
        self._emit("track.repair_complete", filename=track.filename,
                   index=idx, total=total)
        for proc in self.audio_processors:
            try:
                self._emit("processor.start", processor_id=proc.id,
                           filename=track.filename)
                t0 = time.perf_counter()
                if gain_track.audio_data is None and track.filepath:
                    gain_track = source_for_render(gain_track)
                result = proc.process(gain_track)
                if (
                    proc.id == "bimodal_normalize"
                    and track.manual_gain_db is not None
                    and not result.error
                    and proc.id not in track.processor_skip
                    and result.classification not in INACTIVE_NORMALIZATION_CLASSIFICATIONS
                ):
                    result.gain_db = float(track.manual_gain_db)
                    result.data["gain_db_individual"] = float(track.manual_gain_db)
                    update_post_gain_metrics(
                        result, gain_track, float(track.manual_gain_db), source_audio=gain_track.audio_data,
                    )
                dt = (time.perf_counter() - t0) * 1000
                log.debug(f"processor {proc.id} on {track.filename}: {dt:.1f} ms")
                track.processor_results[proc.id] = result
                self._emit("processor.complete", processor_id=proc.id,
                           filename=track.filename)
            except Exception as e:
                track.processor_results[proc.id] = ProcessorResult(
                    processor_id=proc.id,
                    gain_db=0.0,
                    classification="Error",
                    method="None",
                    error=str(e),
                )
        dt_track = (time.perf_counter() - t_track_start) * 1000
        log.debug(f"all processors on {track.filename}: {dt_track:.1f} ms")
        self._emit("track.plan_complete", filename=track.filename,
                   index=idx, total=total)

    def plan(self, session: SessionContext, *, preserve_selections: Collection[str] = (),
             refresh_anchors: Collection[str] = (), revalidate_sources: Collection[str] = ()) -> SessionContext:
        """
        Run all audio processors in priority order.
        Computes gains and classifications without modifying audio.
        Also handles group gain levelling and fader offsets.

        For changed RMS anchors, refresh the original classification and its
        dependent findings before calculating processed gain. Per-track processors run in parallel.
        Group levelling and fader offsets run after all tracks complete.
        An explicit retry may revalidate a restored file against its stored PCM and audio format.
        """
        log.info("Pipeline: plan (%d tracks)", len(session.tracks))
        session.execution_receipts.execute_errors.clear()
        session.execution_receipts.executed_files.clear()
        session.config.pop("_execute_errors", None)
        session.config.pop("_executed_files", None)
        total = len(session.tracks)
        ok_items = [
            (idx, track)
            for idx, track in enumerate(session.tracks)
            if track.status == "OK"
        ]

        n = len(ok_items)
        t_phase = time.perf_counter()
        workers = min(self.max_workers, n) if ok_items else 1
        anchor_detectors = []
        if refresh_anchors:
            affected = {"tail_exceedance"}
            for detector in self.track_detectors:
                if detector.id in affected or affected.intersection(detector.depends_on):
                    anchor_detectors.append(detector)
                    affected.add(detector.id)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._plan_track, track, idx, total,
                            anchor_detectors if track.filename in refresh_anchors else (),
                            track.filename in revalidate_sources): track
                for idx, track in ok_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    track = futures[future]
                    if not track._cache.pop("_repair_emitted", False):
                        self._emit("track.repair_complete",
                                   filename=track.filename,
                                   index=0, total=total)
                    self._emit("track.plan_complete",
                               filename=track.filename,
                               index=0, total=total)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        if n:
            log.debug(f"plan phase (processors): {n} tracks in "
                f"{dt_phase:.1f} ms ({dt_phase / n:.1f} ms/track avg)")

        # Skipped processors do not contribute to gain linking or DAW offsets.
        for proc in self.audio_processors:
            if not proc.default:
                for track in session.tracks:
                    if track.status == "OK" and track.filename not in preserve_selections:
                        track.processor_skip.add(proc.id)

        session.processors = list(self.audio_processors)
        self._emit("session.group_gains_start")
        apply_group_gains(
            session, linked_groups=session.config.get("_gain_linked_groups"),
            headroom_db=self.config.get("fader_headroom_db", 8.0),
            anchor=self.config.get("anchor"),
        )
        self._emit("session.group_gains_complete")
        return session

    # ------------------------------------------------------------------
    # Prepare (apply processors, write processed files)
    # ------------------------------------------------------------------

    def _render_track(self, track: TrackContext) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Render a verified source, retaining the signal just before output gain."""
        assert track.audio_data is not None
        corrected = render_corrected(track.audio_data, track.samplerate, track.repair_recipe)
        audio = corrected
        applied = track.repair_recipe.applied_ids()
        norm_handled_inactive = False
        for proc in self.audio_processors:
            pr = track.processor_results.get(proc.id)
            if pr is None or proc.id in track.processor_skip:
                continue
            if pr.error:
                raise ValueError(f"{proc.name}: {pr.error}")
            if proc.id == "bimodal_normalize" and pr.classification in INACTIVE_NORMALIZATION_CLASSIFICATIONS:
                norm_handled_inactive = True
                continue
            temporary = replace(track, audio_data=audio.copy(), _cache={})
            audio = proc.apply(temporary, pr)
            if pr.classification not in INACTIVE_NORMALIZATION_CLASSIFICATIONS:
                applied.append(proc.id)
        if track.manual_gain_db is not None and "bimodal_normalize" not in applied and not norm_handled_inactive:
            gain_db = float(track.manual_gain_db)
            if not np.isfinite(gain_db):
                raise ValueError(f"Non-finite manual gain: {gain_db}")
            if gain_db != 0.0:
                try:
                    lin = db_to_linear(gain_db)
                except OverflowError as exc:
                    raise ValueError(f"Manual gain {gain_db} dB overflows linear scale") from exc
                if not np.isfinite(lin):
                    raise ValueError(f"Manual gain {gain_db} dB overflows linear scale")
                audio = np.asarray(audio, dtype=np.float64) * lin
                tiny = np.finfo(np.float64).tiny
                audio = np.where(np.abs(audio) < tiny, 0.0, audio)
                if not np.all(np.isfinite(audio)):
                    raise ValueError("Manual gain produced non-finite output samples")
        return corrected, audio, applied

    def _render_prepare_sources(self, needed, sources, routed_names, direct_names, expected, progress_cb):
        rendered, original, applied, source_errors = {}, {}, {}, {}
        unity, source_gaps = {}, {}
        for step, name in enumerate(needed):
            if progress_cb:
                progress_cb(step, expected, f"Processing {name}")
            try:
                track = source_for_render(sources[name])
                corrected, audio, applied[name] = self._render_track(track)
                original[name] = (track.audio_data, track.samplerate)
                rendered[name] = (audio, track.samplerate)
                if name in routed_names:
                    unity[name] = (corrected, track.samplerate)
                if name in direct_names and track.repair_recipe.active:
                    source_gaps[name] = boundary_settling(track.audio_data, corrected, track.samplerate)
            except Exception as exc:
                source_errors[name] = str(exc)
        return rendered, original, applied, source_errors, unity, source_gaps

    def _prepare_routed_dry_source(self, entry, names, src, root, prior, sources, original, unity):
        import hashlib
        from .topology import resolve_entry_audio

        dry_audio = resolve_entry_audio(entry, original)
        try:
            check_output_headroom(dry_audio, "FLOAT")
        except ValueError as exc:
            raise ValueError(
                "The routed original cannot be encoded as finite FLOAT audio. "
                "Check the source levels and routing gains."
            ) from exc
        measurements = {}
        if any(sources[n].repair_recipe.active for n in names):
            corrected_route = resolve_entry_audio(entry, unity)
            measurements = {
                "original": repair_levels(dry_audio),
                "corrected": repair_levels(corrected_route),
                "boundary_settling": boundary_settling(dry_audio, corrected_route, src.samplerate),
            }
        digest = hashlib.blake2b(dry_audio, digest_size=16).hexdigest()
        dry = root / ".routed" / str(src.samplerate) / digest / (Path(entry.output_filename).name + ".wav")
        old = prior.get(entry.output_filename)
        current_signature, dry_revision = None, None
        if dry.is_file():
            stat = dry.stat()
            current_signature = (stat.st_size, stat.st_mtime_ns)
            dry_revision = file_revision(dry, stat)
        if (current_signature is None or old is None or Path(old.filepath) != dry
                or old.source_file_signature != current_signature or old.source_file_revision != dry_revision):
            current_signature, dry_revision = _write_atomic(dry, dry_audio, src.samplerate, "FLOAT")
        return str(dry), current_signature, dry_revision, measurements

    def _prepare_single_entry(
        self, index, entry, root, destinations, input_paths, sources, prior, rerouted,
        rendered, original, applied, source_errors, unity, source_gaps, gain_adjustments,
    ):
        from .topology import resolve_entry_audio

        name = entry.output_filename
        names = [s.input_filename for s in entry.sources]
        for source in names:
            if source in source_errors:
                raise ValueError(f"{source}: {source_errors[source]}")
        src = sources[names[0]]
        if any(sources[n].samplerate != src.samplerate for n in names):
            raise ValueError("Routed sources have different sample rates")
        dest = (root / name).resolve()
        if destinations[dest] != 1:
            raise ValueError("Multiple output tracks use this filename")
        if not dest.is_relative_to(root) or dest in input_paths or dest.is_relative_to(root / ".routed"):
            raise ValueError("Prepared output must be separate from its source")
        audio = resolve_entry_audio(entry, rendered)
        try:
            if len(set(names)) == 1:
                self._check_source_headroom(audio, src, original[src.filename][0], applied[src.filename], entry)
            else:
                check_output_headroom(audio, src.subtype)
        except OutputHeadroomError as overload:
            advice = overload.adjustment
            if advice is not None:
                prev = gain_adjustments.get(src.filename)
                if prev is None or (advice["reduction_db"], advice["peak_db"]) > (prev["reduction_db"], prev["peak_db"]):
                    gain_adjustments[src.filename] = advice
            raise

        fader = _routed_fader(entry, sources, rendered, applied)
        identity = index not in rerouted
        if identity:
            measurements = deepcopy(src.repair_measurements)
            if src.filename in source_gaps:
                measurements["boundary_settling"] = source_gaps[src.filename]
            elif "boundary_settling" in measurements:
                for key in ("boundary_settling", "low_cut_resolved", "subsonic_resolved"):
                    measurements.pop(key, None)
                measurements["corrected"] = deepcopy(measurements.get("original", {}))
            dry_path, dry_signature, dry_revision = src.filepath, src.source_file_signature, src.source_file_revision
        else:
            dry_path, dry_signature, dry_revision, measurements = self._prepare_routed_dry_source(
                entry, names, src, root, prior, sources, original, unity,
            )

        ids = list(dict.fromkeys(pid for n in names for pid in applied[n]))
        out = replace(
            src, filename=name, filepath=dry_path, processed_filepath=str(dest),
            audio_data=None, total_samples=len(audio), channels=entry.output_channels,
            duration_sec=len(audio) / src.samplerate, applied_processors=ids, prepared_fader_db=fader,
            source_file_signature=dry_signature, source_file_revision=dry_revision,
            source_sample_digest=src.source_sample_digest if identity else None,
            detector_results=dict(src.detector_results) if identity else {},
            detector_analysis_key=src.detector_analysis_key if identity else None,
            detector_sample_digest=src.detector_sample_digest if identity else None,
            corrected_analysis=src.corrected_analysis if identity and src.repair_recipe.active else None,
            processor_results=deepcopy(src.processor_results), _cache={},
            repair_overrides=dict(src.repair_overrides), repair_measurements=measurements,
        )
        out.prepared_file_signature, out.prepared_file_revision = _write_atomic(dest, audio, src.samplerate, src.subtype)
        if identity:
            src.processed_filepath = str(dest)
            src.applied_processors = list(ids)
            src.prepared_file_signature = out.prepared_file_signature
            src.prepared_file_revision = out.prepared_file_revision
            src.prepared_fader_db = out.prepared_fader_db
        return out

    @staticmethod
    def _cleanup_obsolete_prepared_files(root: Path, input_paths: set[Path], output_tracks, prior) -> None:
        protected = input_paths | {
            Path(path).resolve() for track in output_tracks
            for path in (track.filepath, track.processed_filepath) if path
        }
        for obsolete in prior.values():
            if not obsolete.processed_filepath or obsolete.prepared_file_signature is None:
                continue
            path = Path(obsolete.processed_filepath)
            try:
                resolved = path.resolve()
                if (resolved in protected or not resolved.is_relative_to(root)
                        or resolved.is_relative_to(root / ".routed") or path.is_symlink()):
                    continue
                stat = path.stat()
                if ((stat.st_size, stat.st_mtime_ns) == obsolete.prepared_file_signature
                        and obsolete.prepared_file_revision is not None
                        and file_revision(str(path), stat) == obsolete.prepared_file_revision):
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("Could not remove obsolete prepared file %s: %s", path, exc)

    def prepare(
        self,
        session: SessionContext,
        output_dir: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> SessionContext:
        """Render each output atomically. A failed retry retains its last successful render."""
        from .topology import build_default_topology, build_transfer_manifest

        session.prepare_state = "stale" if session.output_tracks else "none"
        written: list[str] = []
        session.execution_receipts.prepared_files = written
        session.config["_prepared_files"] = written
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        topology = session.topology or build_default_topology(session.tracks)
        session.topology = topology
        sources = {t.filename: t for t in session.tracks if t.status == "OK"}
        needed = dict.fromkeys(s.input_filename for e in topology.entries for s in e.sources)
        rerouted = {i for i, entry in enumerate(topology.entries) if not _passthrough_entry(entry, sources)}
        routed_names = {s.input_filename for i in rerouted for s in topology.entries[i].sources}
        direct_names = {e.sources[0].input_filename for i, e in enumerate(topology.entries) if i not in rerouted}
        expected = len(needed) + len(topology.entries)
        destinations = Counter((root / e.output_filename).resolve() for e in topology.entries)
        rendered, original, applied, source_errors, unity, source_gaps = self._render_prepare_sources(
            needed, sources, routed_names, direct_names, expected, progress_cb,
        )

        prior = {t.filename: t for t in session.output_tracks}
        output_tracks, errors = [], []
        gain_adjustments: dict[str, dict[str, float]] = {}
        input_paths = {Path(t.filepath).resolve() for t in session.tracks if t.filepath}
        for index, entry in enumerate(topology.entries):
            name = entry.output_filename
            step = len(needed) + index
            if progress_cb:
                progress_cb(step, expected, f"Writing {name}")
            try:
                out = self._prepare_single_entry(
                    index, entry, root, destinations, input_paths, sources, prior, rerouted,
                    rendered, original, applied, source_errors, unity, source_gaps, gain_adjustments,
                )
                output_tracks.append(out)
                written.append(name)
                self._emit("track.prepared", filename=name, index=index, total=len(topology.entries))
            except Exception as exc:
                errors.append((name, str(exc)))
                if name in prior:
                    output_tracks.append(prior[name])
                self._emit("prepare.error", filename=name, error=str(exc))
            if progress_cb:
                progress_cb(step + 1, expected, name)

        if not errors:
            self._cleanup_obsolete_prepared_files(root, input_paths, output_tracks, prior)
        session.config["_prepare_errors"] = errors
        session.config["_output_gain_adjustments"] = gain_adjustments
        session.output_tracks = output_tracks
        manifest = build_transfer_manifest(topology, session.tracks, session.transfer_manifest or None)
        session.transfer_manifest = manifest
        session.base_transfer_manifest = deepcopy(manifest)
        session.prepare_state = ("stale" if output_tracks else "none") if errors else "ready"
        return session

    def _check_source_headroom(self, audio, track, source_audio, applied, entry=None):
        """Validate encoding and offer a trim in the units the caller can apply."""
        from .topology import resolve_entry_audio

        try:
            check_output_headroom(audio, track.subtype)
            return
        except OutputHeadroomError as exc:
            overload = exc

        norm = track.processor_results.get("bimodal_normalize")
        normalizing = norm is not None and "bimodal_normalize" in applied
        gain = norm.gain_db if normalizing else (track.manual_gain_db or 0.0)
        # Adjust Gain selects an absolute value on a decimal grid. The CLI
        # gives a reduction in tenths from the user's full-precision setting.
        limit = gain - overload.excess_db if entry is not None else -overload.excess_db
        step = round(limit * 10)
        if abs(step / 10 - limit) > 1e-9:
            step = math.floor(limit * 10)
        else:
            target = step / 10
            if entry is None:
                decimal_target = float(Decimal(str(gain)) + Decimal(str(target)))
                target = max(gain + target, decimal_target)
            retry = replace(track, audio_data=source_audio, manual_gain_db=target,
                            processor_results=deepcopy(track.processor_results), _cache={})
            if normalizing:
                retry.processor_results["bimodal_normalize"].gain_db = target
            _, trial, _ = self._render_track(retry)
            if entry is not None:
                trial = resolve_entry_audio(entry, {track.filename: (trial, track.samplerate)})
            try:
                check_output_headroom(trial, track.subtype)
            except OutputHeadroomError:
                step -= 1
        if entry is None:
            reduction = -step / 10
            target = gain - reduction
        else:
            target = step / 10
            difference = Decimal(str(gain)) - Decimal(str(target))
            reduction = float(difference.quantize(Decimal("0.1"), rounding=ROUND_CEILING))
        advice = {"peak_db": overload.peak_db, "reduction_db": reduction,
                  "gain_db": gain, "target_gain_db": target}
        raise OutputHeadroomError(overload.peak_db, reduction, track.subtype,
                                  excess_db=overload.excess_db, adjustment=advice) from overload

    # ------------------------------------------------------------------
    # Phase 3: Execute (apply gains, write files)
    # ------------------------------------------------------------------

    def execute(
        self,
        session: SessionContext,
        output_dir: str,
        backup_dir: str | None = None,
        is_overwriting: bool = False,
    ) -> SessionContext:
        """Apply audio processor gains and write files."""
        os.makedirs(output_dir, exist_ok=True)
        if backup_dir and is_overwriting:
            os.makedirs(backup_dir, exist_ok=True)

        root = Path(output_dir).resolve()
        inputs = {Path(t.filepath).resolve() for t in session.tracks}
        destinations = Counter((root / t.filename).resolve() for t in session.tracks)
        errors = []
        receipts = {}
        total = len(session.tracks)
        for idx, track in enumerate(session.tracks):
            if track.status != "OK":
                errors.append((track.filename, track.status))
                continue

            self._emit("track.write_start", filename=track.filename,
                       index=idx, total=total)

            try:
                src_filepath = track.filepath
                record_backup_render = None
                dst_filepath = (root / track.filename).resolve()
                if (not dst_filepath.is_relative_to(root) or destinations[dst_filepath] != 1 or
                        dst_filepath in inputs and
                        (not is_overwriting or dst_filepath != Path(src_filepath).resolve())):
                    raise ValueError("Output filename conflicts with another track or its source")

                if is_overwriting and backup_dir:
                    backup_root = Path(backup_dir).resolve()
                    backup_path = backup_root / track.filename
                    if not backup_path.resolve().is_relative_to(backup_root) or backup_path.resolve() in inputs:
                        raise ValueError("Backup filename conflicts with its source or backup directory")
                    record_backup_render = ensure_audio_backup(src_filepath, backup_path)

                source = source_for_render(track)
                corrected, audio, applied = self._render_track(source)
                self._check_source_headroom(audio, source, source.audio_data, applied)
                settling = (boundary_settling(source.audio_data, corrected, source.samplerate)
                            if source.repair_recipe.active else [])
                gain, fader = _resolved_output_gain_and_fader(track, applied)
                if not np.isfinite([gain, fader]).all():
                    raise ValueError("Output gain and fader offset must be finite")
                peak = linear_to_db(float(np.max(np.abs(audio)))) if audio.size else -np.inf
                signature, revision = _write_atomic(dst_filepath, audio, track.samplerate, track.subtype,
                                                    before_publish=record_backup_render)
                for result in track.processor_results.values():
                    result.data["out_peak"] = peak
                receipt = {"gain_db": gain, "fader_db": fader, "applied_processors": applied}
                if settling:
                    receipt["boundary_settling"] = settling
                receipts[track.filename] = receipt
                track.processed_filepath = str(dst_filepath)
                track.applied_processors = applied
                track.prepared_file_signature = signature
                track.prepared_file_revision = revision
                track.prepared_fader_db = fader
            except Exception as exc:
                errors.append((track.filename, str(exc)))
                self._emit("write.error", filename=track.filename, error=str(exc))

            self._emit("track.write_complete", filename=track.filename,
                       index=idx, total=total)

        session.execution_receipts.execute_errors = errors
        session.execution_receipts.executed_files = receipts
        session.config["_execute_errors"] = errors
        session.config["_executed_files"] = receipts
        return session


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolved_output_gain_and_fader(track: TrackContext, applied: list[str]) -> tuple[float, float]:
    """Return executed (gain_db, fader_db) consistent with applied processors."""
    norm = track.processor_results.get("bimodal_normalize")
    if norm is not None and "bimodal_normalize" in applied:
        gain = float(norm.gain_db)
        return gain, float(norm.data.get("fader_offset", -gain))
    if norm is not None and "bimodal_normalize" not in track.processor_skip and norm.classification in INACTIVE_NORMALIZATION_CLASSIFICATIONS:
        return 0.0, 0.0
    return float(track.manual_gain_db or 0.0), 0.0


def _passthrough_entry(entry, sources) -> bool:
    from .topology import passthrough_routes

    if len(entry.sources) != 1:
        return False
    source = entry.sources[0]
    track = sources.get(source.input_filename)
    return (track is not None and track.channels == entry.output_channels
            and source.routes == passthrough_routes(track.channels))


def _routed_fader(entry, sources, rendered, applied) -> float:
    """Return the single fader that can compensate all contributing channels."""
    by_offset: dict[float, set[tuple[str, int]]] = {}
    for source in entry.sources:
        name = source.input_filename
        track = sources[name]
        norm = track.processor_results.get("bimodal_normalize")
        offset = (float(norm.data.get("fader_offset", -norm.gain_db))
                  if norm and "bimodal_normalize" in applied[name] else 0.0)
        for route in source.routes:
            if route.gain and 0 <= route.source_channel < track.channels:
                by_offset.setdefault(offset, set()).add((name, route.source_channel))
    if not all(np.isfinite(offset) for offset in by_offset):
        raise ValueError("Prepared fader offsets must be finite")
    if len(by_offset) <= 1:
        return next(iter(by_offset), 0.0)

    # Distinct offsets only conflict when both routes carry signal. Inspect
    # precisely the selected channels, including a silent side of a stereo file.
    audible = []
    for offset, channels in by_offset.items():
        for name, channel in channels:
            samples = rendered[name][0]
            if np.any(samples if samples.ndim == 1 else samples[:, channel]):
                audible.append(offset)
                break
    if len(audible) > 1:
        raise ValueError("One output fader cannot restore these sources' different normalization "
                         "gains. Link their gains or route them to separate outputs, then Prepare again.")
    return audible[0] if audible else 0.0


def _write_atomic(destination: Path, audio: np.ndarray, sample_rate: int,
                  subtype: str, *, before_publish=None) -> tuple[tuple[int, int], str | None]:
    """Publish only a fully closed audio file, including on Windows."""
    import soundfile as sf

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".sp-", suffix=destination.suffix,
                            delete=False) as temp:
        path = Path(temp.name)
    try:
        sf.write(path, audio, sample_rate, subtype=subtype)
        if before_publish is not None:
            before_publish(path)
        os.replace(path, destination)
        stat = destination.stat()
        return (stat.st_size, stat.st_mtime_ns), file_revision(str(destination), stat)
    finally:
        path.unlink(missing_ok=True)


def _load_one_track(
    source_dir: str,
    filename: str,
    idx: int,
    total: int,
    event_bus: EventBus | None,
) -> TrackContext:
    """Load a single WAV file (used by thread pool in load_session)."""
    if event_bus and event_bus.is_cancelled.is_set():
        return TrackContext(
            filename=filename,
            filepath=os.path.join(source_dir, filename),
            audio_data=None,
            samplerate=0,
            channels=0,
            total_samples=0,
            bitdepth="",
            subtype="",
            duration_sec=0.0,
            status="Cancelled",
        )

    filepath = os.path.join(source_dir, filename)
    if event_bus:
        event_bus.emit("track.load", filename=filename,
                       index=idx, total=total)
    try:
        tc = load_track(filepath)
        tc.filename = filename  # preserve relative path from discover_audio_files
        return tc
    except Exception as e:
        return TrackContext(
            filename=filename,
            filepath=filepath,
            audio_data=None,
            samplerate=0,
            channels=0,
            total_samples=0,
            bitdepth="",
            subtype="",
            duration_sec=0.0,
            status=f"Error: {e}",
        )


def load_session(
    source_dir: str,
    config: dict[str, Any],
    event_bus: EventBus | None = None,
    recursive: bool = False,
    *, filenames: Collection[str] | None = None,
) -> SessionContext:
    """
    Load all audio files from source_dir into a SessionContext.
    Files are loaded in parallel using a thread pool.
    Handles group assignment.

    When *recursive* is True, subdirectories are scanned and filenames
    become relative paths (e.g. ``"drums/01_Kick.wav"``).
    """
    from .audio import discover_audio_files
    if filenames is None:
        files = discover_audio_files(source_dir, recursive=recursive)
    else:
        # A committed Track Layout can share its directory with other user media.
        root = Path(source_dir).resolve()
        files = list(filenames)
        if any(not (root / name).resolve().is_relative_to(root) for name in files):
            raise ValueError("A layout input is outside its source directory")

    total = len(files)
    workers = min(os.cpu_count() or 4, 8, total) if total else 1
    tracks: list[TrackContext] = [None] * total  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _load_one_track, source_dir, filename, idx, total, event_bus
            ): idx
            for idx, filename in enumerate(files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            tracks[idx] = future.result()

    session = SessionContext(tracks=tracks, config=dict(config))

    # Group assignment
    group_args = config.get("group", [])
    group_specs = parse_group_specs(group_args)

    if group_specs:
        filenames = [t.filename for t in tracks]
        assignments, warnings = assign_groups(filenames, group_specs)
        session.groups = assignments

        for track in tracks:
            track.group = assignments.get(track.filename)

        session.warnings.extend(warnings)

    return session
