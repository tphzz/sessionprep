from __future__ import annotations

import os
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import logging
from collections import Counter
from collections.abc import Collection

log = logging.getLogger(__name__)

import numpy as np

from .models import (
    DetectorResult,
    ProcessorResult,
    Severity,
    TrackContext,
    SessionContext,
    LifecyclePhase,
)
from .detector import TrackDetector, SessionDetector
from .processor import AudioProcessor
from .repair_plan import plan_repairs, source_for_render
from .repairs import OutputHeadroomError, check_output_headroom, render_corrected
from .events import EventBus
from .audio import load_track, format_duration, linear_to_db, AUDIO_EXTENSIONS
from .config import ConfigError, validate_config
from .utils import (
    protools_sort_key,
    parse_group_specs,
    assign_groups,
)


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
        for d in self.session_detectors:
            d.configure(self.config)

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
        all_ids = set()
        for d in self.track_detectors:
            if d.id in all_ids:
                raise ConfigError(f"Duplicate detector ID: {d.id}")
            all_ids.add(d.id)
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
        for det in detectors:
            try:
                self._emit("detector.start", detector_id=det.id,
                           filename=track.filename)
                t0 = time.perf_counter()
                result = det.analyze(track)
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
        dt_track = (time.perf_counter() - t_track_start) * 1000
        log.debug(f"all detectors on {track.filename}: {dt_track:.1f} ms")
        self._emit("track.analyze_complete", filename=track.filename,
                   index=idx, total=total)

    def _run_analysis_phase(self, session: SessionContext, phase: LifecyclePhase) -> SessionContext:
        """Run track-level and session-level detectors matching the specified phase."""
        track_dets = [d for d in self.track_detectors if getattr(d, 'phase', LifecyclePhase.PHASE2) == phase]
        session_dets = [d for d in self.session_detectors if getattr(d, 'phase', LifecyclePhase.PHASE2) == phase]

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
                pool.submit(self._analyze_track, track, idx, total, track_dets): track
                for idx, track in ok_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    track = futures[future]
                    self._emit("track.analyze_complete",
                               filename=track.filename,
                               index=0, total=total)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        if n:
            log.debug(f"analyze {phase.value} (track detectors): {n} tracks in "
                f"{dt_phase:.1f} ms ({dt_phase / n:.1f} ms/track avg)")

        # Session-level detectors
        track_map = {t.filename: t for t in session.tracks}
        for det in session_dets:
            if self.event_bus and self.event_bus.is_cancelled.is_set():
                break

            try:
                self._emit("session_detector.start", detector_id=det.id)
                t0 = time.perf_counter()
                results = det.analyze(session)
                dt = (time.perf_counter() - t0) * 1000
                log.debug(f"session detector {det.id}: {dt:.1f} ms")
                session.config[f"_session_det_{det.id}"] = results
                # Distribute per-track results back into each track
                for result in results:
                    fname = result.data.get("filename")
                    if fname and fname in track_map:
                        track_map[fname].detector_results[det.id] = result
                self._emit("session_detector.complete", detector_id=det.id)
            except Exception as e:
                session.config[f"_session_det_{det.id}"] = [
                    DetectorResult(
                        detector_id=det.id,
                        severity=Severity.PROBLEM,
                        summary=f"session detector error: {e}",
                        data={},
                        error=str(e),
                    )
                ]

        # Ensure the overall session.detectors always has all detectors
        # required for GUI overlay generation, independent of which phase ran
        session.detectors = self.track_detectors + self.session_detectors
        return session

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

    def _plan_track(self, track: TrackContext, idx: int, total: int, anchor_detectors=()):
        """Run all audio processors for a single track (thread-safe)."""
        if self.event_bus and self.event_bus.is_cancelled.is_set():
            return

        self._emit("track.plan_start", filename=track.filename,
                   index=idx, total=total)
        t_track_start = time.perf_counter()
        track._cache.pop("repair_error", None)
        try:
            source = track
            if anchor_detectors:
                source = source_for_render(track)
                self._analyze_track(source, idx, total, anchor_detectors)
            gain_track = plan_repairs(source, self.config)
            track.repair_recipe = source.repair_recipe
            track.repair_measurements = source.repair_measurements
        except Exception as exc:
            track._cache["repair_error"] = str(exc)
            track.processor_results.clear()
            self._emit("track.plan_complete", filename=track.filename, index=idx, total=total)
            return
        for proc in self.audio_processors:
            try:
                self._emit("processor.start", processor_id=proc.id,
                           filename=track.filename)
                t0 = time.perf_counter()
                result = proc.process(gain_track)
                if proc.id == "bimodal_normalize" and track.manual_gain_db is not None:
                    result.gain_db = track.manual_gain_db
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
             refresh_anchors: Collection[str] = ()) -> SessionContext:
        """
        Run all audio processors in priority order.
        Computes gains and classifications without modifying audio.
        Also handles group gain levelling and fader offsets.

        For changed RMS anchors, refresh the original classification and its
        dependent findings before calculating processed gain. Per-track processors run in parallel.
        Group levelling and fader offsets run after all tracks complete.
        """
        log.info("Pipeline: plan (%d tracks)", len(session.tracks))
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
            affected = {"audio_classifier"}
            for detector in self.track_detectors:
                if detector.id in affected or affected.intersection(detector.depends_on):
                    anchor_detectors.append(detector)
                    affected.add(detector.id)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._plan_track, track, idx, total,
                            anchor_detectors if track.filename in refresh_anchors else ()): track
                for idx, track in ok_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    track = futures[future]
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

        # --- Group gain levelling ---
        t0 = time.perf_counter()
        self._apply_group_levels(session)
        dt = (time.perf_counter() - t0) * 1000
        log.debug(f"group levelling: {dt:.1f} ms")

        # --- Fader offsets ---
        t0 = time.perf_counter()
        self._compute_fader_offsets(session)
        dt = (time.perf_counter() - t0) * 1000
        log.debug(f"fader offsets: {dt:.1f} ms")

        # Store configured processor instances on the session for render-time access
        session.processors = list(self.audio_processors)

        return session

    def _apply_group_levels(self, session: SessionContext):
        """Apply group levels for gain-linked groups (minimum gain of the group).

        Reads ``_gain_linked_groups`` from *session.config* — a set of group
        names whose members should share the same gain.  If the key is absent
        **all** groups are levelled (backward-compatible CLI behaviour).
        """
        linked: set[str] | None = session.config.get("_gain_linked_groups")

        for proc in self.audio_processors:
            by_gid: dict[str, list[TrackContext]] = {}
            for track in session.tracks:
                if track.status != "OK" or track.group is None:
                    continue
                pr = track.processor_results.get(proc.id)
                if (pr is None or pr.error or pr.classification in ("Silent", "Skip")
                        or proc.id in track.processor_skip):
                    continue
                # Preserve the per-track gain before any group adjustment
                if "original_gain_db" not in pr.data:
                    pr.data["original_gain_db"] = pr.gain_db
                by_gid.setdefault(track.group, []).append(track)

            for gid, members in by_gid.items():
                if linked is not None and gid not in linked:
                    continue
                orig = [m.processor_results[proc.id].data["original_gain_db"]
                        for m in members]
                group_gain = min(orig) if orig else 0.0
                for m in members:
                    m.processor_results[proc.id].gain_db = float(group_gain)

    def _compute_fader_offsets(self, session: SessionContext):
        """Compute fader offsets (inverse of gain) with headroom rebalancing."""
        ceiling = session.config.get("_fader_ceiling_db", 12.0)
        headroom = self.config.get("fader_headroom_db", 8.0)
        max_allowed = ceiling - headroom

        for proc in self.audio_processors:
            valid = []
            for track in session.tracks:
                if track.status != "OK":
                    continue
                pr = track.processor_results.get(proc.id)
                if pr is None:
                    continue
                pr.data.pop("fader_rebalance_shift", None)
                if pr.error or pr.classification in ("Silent", "Skip") or proc.id in track.processor_skip:
                    pr.data["fader_offset"] = 0.0
                else:
                    pr.data["fader_offset"] = -float(pr.gain_db)
                    valid.append(track)

            # Headroom rebalancing: shift faders so max ≤ ceiling - headroom
            rebalance_shift = 0.0
            if headroom > 0.0 and valid:
                fader_offsets = [
                    t.processor_results[proc.id].data.get("fader_offset", 0.0)
                    for t in valid
                ]
                max_fader = max(fader_offsets)
                if max_fader > max_allowed:
                    rebalance_shift = max_fader - max_allowed
                    for track in valid:
                        pr = track.processor_results.get(proc.id)
                        if pr:
                            pr.data["fader_offset"] -= rebalance_shift
                            pr.data["fader_rebalance_shift"] = rebalance_shift
            session.config[f"_fader_rebalance_{proc.id}"] = rebalance_shift
            if rebalance_shift != 0.0:
                log.debug(f"fader rebalance ({proc.id}): shifted {rebalance_shift:+.1f} dB "
                    f"(ceiling {ceiling}, headroom {headroom})")

            # Anchor-track adjustment (user-selected anchor)
            anchor_name = self.config.get("anchor")
            anchor_offset = 0.0
            if anchor_name and valid:
                anchor_track = next(
                    (t for t in valid
                     if anchor_name.lower() in t.filename.lower()),
                    None,
                )
                if anchor_track:
                    pr = anchor_track.processor_results.get(proc.id)
                    if pr:
                        anchor_offset = pr.data.get("fader_offset", 0.0)

            # Store anchor offset for GUI-side recomputation
            session.config[f"_anchor_offset_{proc.id}"] = anchor_offset

            if anchor_offset != 0.0:
                for track in valid:
                    pr = track.processor_results.get(proc.id)
                    if pr:
                        pr.data["fader_offset"] = pr.data.get("fader_offset", 0.0) - anchor_offset

    # ------------------------------------------------------------------
    # Prepare (apply processors, write processed files)
    # ------------------------------------------------------------------

    def _render_track(self, track: TrackContext) -> tuple[np.ndarray, list[str]]:
        """Render from the routed original without mutating its samples or analysis cache."""
        if "repair_error" in track._cache:
            raise ValueError(track._cache["repair_error"])
        source = source_for_render(track)
        audio = render_corrected(source.audio_data, source.samplerate, track.repair_recipe)
        applied = track.repair_recipe.applied_ids()
        for proc in self.audio_processors:
            pr = track.processor_results.get(proc.id)
            if pr is None or proc.id in track.processor_skip:
                continue
            if pr.error:
                raise ValueError(f"{proc.name}: {pr.error}")
            if proc.id == "bimodal_normalize" and pr.classification in ("Silent", "Skip"):
                continue
            temporary = replace(track, audio_data=audio.copy(), _cache={})
            audio = proc.apply(temporary, pr)
            if pr.classification not in ("Silent", "Skip"):
                applied.append(proc.id)
        if track.manual_gain_db is not None and "bimodal_normalize" not in applied:
            audio = audio * (10.0 ** (track.manual_gain_db / 20))
        return audio, applied

    def prepare(
        self,
        session: SessionContext,
        output_dir: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> SessionContext:
        """Render each output atomically. A failed retry retains its last successful render."""
        from .topology import build_default_topology, build_transfer_manifest, passthrough_routes, resolve_entry_audio

        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        topology = session.topology or build_default_topology(session.tracks)
        session.topology = topology
        sources = {t.filename: t for t in session.tracks if t.status == "OK"}
        needed = dict.fromkeys(s.input_filename for e in topology.entries for s in e.sources)
        rendered, original, applied, source_errors = {}, {}, {}, {}
        expected = len(needed) + len(topology.entries)
        destinations = Counter((root / e.output_filename).resolve() for e in topology.entries)
        for step, name in enumerate(needed):
            if progress_cb:
                progress_cb(step, expected, f"Processing {name}")
            try:
                track = source_for_render(sources[name])
                audio, applied[name] = self._render_track(track)
                original[name] = (track.audio_data, track.samplerate)
                rendered[name] = (audio, track.samplerate)
            except Exception as exc:
                source_errors[name] = str(exc)

        prior = {t.filename: t for t in session.output_tracks}
        output_tracks, errors, written = [], [], []
        gain_adjustments = {}
        input_paths = {Path(t.filepath).resolve() for t in session.tracks if t.filepath}
        for index, entry in enumerate(topology.entries):
            name = entry.output_filename
            step = len(needed) + index
            if progress_cb:
                progress_cb(step, expected, f"Writing {name}")
            try:
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
                check_output_headroom(audio, src.subtype)
                identity = (len(names) == 1 and entry.output_channels == src.channels and
                            entry.sources[0].routes == passthrough_routes(src.channels))
                dry_path = src.filepath
                dry_signature = src.source_file_signature
                if not identity:
                    # A gain mix or channel mapping has no single dry source file.
                    # Give each routed revision its own path so a failed retry
                    # cannot invalidate an older output's dry counterpart.
                    import hashlib
                    dry_audio = resolve_entry_audio(entry, original)
                    try:
                        check_output_headroom(dry_audio, "FLOAT")
                    except ValueError as exc:
                        raise ValueError(
                            "The routed original cannot be encoded as finite FLOAT audio. "
                            "Check the source levels and routing gains."
                        ) from exc
                    digest = hashlib.blake2b(dry_audio, digest_size=16).hexdigest()
                    dry = root / ".routed" / str(src.samplerate) / digest / (Path(name).name + ".wav")
                    old = prior.get(name)
                    current_signature = None
                    if dry.is_file():
                        stat = dry.stat()
                        current_signature = (stat.st_size, stat.st_mtime_ns)
                    if (current_signature is None or old is None or Path(old.filepath) != dry
                            or old.source_file_signature != current_signature):
                        current_signature = _write_atomic(dry, dry_audio, src.samplerate, "FLOAT")
                    dry_path = str(dry)
                    dry_signature = current_signature
                faders = set()
                for n in names:
                    norm = sources[n].processor_results.get("bimodal_normalize")
                    faders.add(float(norm.data.get("fader_offset", -norm.gain_db))
                               if norm and "bimodal_normalize" in applied[n] else 0.0)
                if not all(np.isfinite(fader) for fader in faders):
                    raise ValueError("Prepared fader offsets must be finite")
                ids = list(dict.fromkeys(pid for n in names for pid in applied[n]))
                out = replace(
                    src, filename=name, filepath=dry_path, processed_filepath=str(dest),
                    audio_data=None, total_samples=len(audio), channels=entry.output_channels,
                    duration_sec=len(audio) / src.samplerate, applied_processors=ids,
                    prepared_fader_db=faders.pop() if len(faders) == 1 else 0.0,
                    source_file_signature=dry_signature,
                    processor_results=deepcopy(src.processor_results), _cache={},
                    repair_overrides=dict(src.repair_overrides),
                    repair_measurements=deepcopy(src.repair_measurements),
                )
                out.prepared_file_signature = _write_atomic(dest, audio, src.samplerate, src.subtype)
                output_tracks.append(out)
                written.append(name)
                if identity:
                    src.processed_filepath = str(dest)
                    src.applied_processors = list(ids)
                    src.prepared_file_signature = out.prepared_file_signature
                    src.prepared_fader_db = out.prepared_fader_db
                self._emit("track.prepared", filename=name, index=index, total=len(topology.entries))
            except Exception as exc:
                errors.append((name, str(exc)))
                if isinstance(exc, OutputHeadroomError) and len(set(names)) == 1 and names[0] in sources:
                    failed = sources[names[0]]
                    norm = failed.processor_results.get("bimodal_normalize")
                    gain = norm.gain_db if norm and "bimodal_normalize" in applied[names[0]] else failed.manual_gain_db
                    previous = gain_adjustments.get(failed.filename)
                    if previous is None or exc.peak_db > previous["peak_db"]:
                        gain_adjustments[failed.filename] = {
                            "peak_db": exc.peak_db, "reduction_db": exc.reduction_db, "gain_db": gain or 0.0,
                        }
                if name in prior:
                    output_tracks.append(prior[name])
                self._emit("prepare.error", filename=name, error=str(exc))
            if progress_cb:
                progress_cb(step + 1, expected, name)

        if not errors:
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
                    if (stat.st_size, stat.st_mtime_ns) == obsolete.prepared_file_signature:
                        path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    log.warning("Could not remove obsolete prepared file %s: %s", path, exc)
        session.config["_prepare_errors"] = errors
        session.config["_output_gain_adjustments"] = gain_adjustments
        session.config["_prepared_files"] = written
        session.output_tracks = output_tracks
        manifest = build_transfer_manifest(topology, session.tracks, session.transfer_manifest or None)
        session.transfer_manifest = manifest
        session.base_transfer_manifest = deepcopy(manifest)
        session.prepare_state = ("stale" if output_tracks else "none") if errors else "ready"
        return session

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
        import shutil

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
                dst_filepath = (root / track.filename).resolve()
                if (not dst_filepath.is_relative_to(root) or destinations[dst_filepath] != 1 or
                        dst_filepath in inputs and
                        (not is_overwriting or dst_filepath != Path(src_filepath).resolve())):
                    raise ValueError("Output filename conflicts with another track or its source")

                if is_overwriting and backup_dir:
                    backup_path = os.path.join(backup_dir, track.filename)
                    if not os.path.exists(backup_path):
                        shutil.copy2(src_filepath, backup_path)

                audio, applied = self._render_track(track)
                check_output_headroom(audio, track.subtype)
                norm = track.processor_results.get("bimodal_normalize")
                gain, fader = float(track.manual_gain_db or 0), 0.0
                if norm and "bimodal_normalize" in applied:
                    gain = float(norm.gain_db)
                    fader = float(norm.data.get("fader_offset", -gain))
                if not np.isfinite([gain, fader]).all():
                    raise ValueError("Output gain and fader offset must be finite")
                peak = linear_to_db(float(np.max(np.abs(audio)))) if audio.size else -np.inf
                signature = _write_atomic(dst_filepath, audio, track.samplerate, track.subtype)
                for result in track.processor_results.values():
                    result.data["out_peak"] = peak
                receipts[track.filename] = {"gain_db": gain, "fader_db": fader, "applied_processors": applied}
                track.processed_filepath = str(dst_filepath)
                track.applied_processors = applied
                track.prepared_file_signature = signature
                track.prepared_fader_db = fader
            except Exception as exc:
                errors.append((track.filename, str(exc)))
                self._emit("write.error", filename=track.filename, error=str(exc))

            self._emit("track.write_complete", filename=track.filename,
                       index=idx, total=total)

        session.config["_execute_errors"] = errors
        session.config["_executed_files"] = receipts
        return session


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _write_atomic(destination: Path, audio: np.ndarray, sample_rate: int,
                  subtype: str) -> tuple[int, int]:
    """Publish only a fully closed audio file, including on Windows."""
    import soundfile as sf

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".sp-", suffix=destination.suffix,
                            delete=False) as temp:
        path = Path(temp.name)
    try:
        sf.write(path, audio, sample_rate, subtype=subtype)
        stat = path.stat()
        os.replace(path, destination)
        return stat.st_size, stat.st_mtime_ns
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
) -> SessionContext:
    """
    Load all audio files from source_dir into a SessionContext.
    Files are loaded in parallel using a thread pool.
    Handles group assignment.

    When *recursive* is True, subdirectories are scanned and filenames
    become relative paths (e.g. ``"drums/01_Kick.wav"``).
    """
    from .audio import discover_audio_files
    files = discover_audio_files(source_dir, recursive=recursive)

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


def _topo_sort_detectors(detectors: list[TrackDetector]) -> list[TrackDetector]:
    """
    Topological sort by depends_on.
    Raises ConfigError if a cycle is detected or a dependency is missing.
    """
    by_id = {d.id: d for d in detectors}
    all_ids = set(by_id.keys())

    # Validate dependencies exist
    for d in detectors:
        for dep in (d.depends_on or []):
            if dep not in all_ids:
                raise ConfigError(
                    f"Detector '{d.id}' depends on '{dep}' which does not exist"
                )

    # Kahn's algorithm
    in_degree = {d.id: 0 for d in detectors}
    adjacency: dict[str, list[str]] = {d.id: [] for d in detectors}
    for d in detectors:
        for dep in (d.depends_on or []):
            adjacency[dep].append(d.id)
            in_degree[d.id] += 1

    queue = [did for did, deg in in_degree.items() if deg == 0]
    sorted_ids = []
    while queue:
        node = queue.pop(0)
        sorted_ids.append(node)
        for neighbor in adjacency.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_ids) != len(detectors):
        raise ConfigError("Cyclic dependency detected among track detectors")

    return [by_id[did] for did in sorted_ids]
