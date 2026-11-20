"""Background worker threads for pipeline analysis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

from PySide6.QtCore import QThread, Signal

from sessionpreplib.audio import SourceChangedError
from sessionpreplib.daw_processor import DawProcessor
from sessionpreplib.detector import TrackDetector
from sessionpreplib.pipeline import Pipeline, load_session
from sessionpreplib.detectors import default_detectors
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary
from sessionpreplib.events import EventBus
from sessionpreplib.models import LifecyclePhase
from sessionpreplib.repair_plan import check_source_signature


def _gui_source_samples(samples):
    """Share read-only source PCM with Qt workers; loads and edits replace the array."""
    owner = samples
    while isinstance(owner, np.ndarray):
        owner.setflags(write=False)
        owner = owner.base
    return samples


def _adopt_decoded_source(track, decoded):
    """Keep the project-relative name and publish the PCM with its decoder metadata."""
    track.channels = decoded.channels
    track.samplerate = decoded.samplerate
    track.total_samples = decoded.total_samples
    track.duration_sec = decoded.duration_sec
    track.bitdepth = decoded.bitdepth
    track.subtype = decoded.subtype
    track.chunk_ids = decoded.chunk_ids
    track.source_file_signature = decoded.source_file_signature
    track.source_file_revision = decoded.source_file_revision
    track.source_sample_digest = decoded.source_sample_digest
    track.audio_data = _gui_source_samples(decoded.audio_data)


def _needs_source_decode(track):
    if track.audio_data is None:
        return True
    try:
        check_source_signature(track)
    except (OSError, SourceChangedError):
        return True
    return False


class DawCheckWorker(QThread):
    """Runs DawProcessor.check_connectivity() off the main thread."""

    result = Signal(bool, str)  # (ok, message)

    def __init__(self, processor: DawProcessor, parent=None):
        super().__init__(parent)
        self._processor = processor

    def run(self):
        try:
            ok, msg = self._processor.check_connectivity()
            self.result.emit(ok, msg)
        except Exception as e:
            self.result.emit(False, str(e))


class DawFetchWorker(QThread):
    """Runs DawProcessor.fetch() off the main thread."""

    progress = Signal(str)              # status text
    progress_value = Signal(int, int)   # (current, total)
    result = Signal(bool, str, object)  # (ok, message, session_or_none)

    def __init__(
        self,
        processor: DawProcessor,
        session,
        *,
        ignore_cache: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._processor = processor
        self._session = session
        self._ignore_cache = ignore_cache

    def _on_progress(self, current: int, total: int, message: str):
        self.progress.emit(message)
        self.progress_value.emit(current, total)

    def run(self):
        try:
            import inspect

            params = inspect.signature(self._processor.fetch).parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            kwargs = {}
            if accepts_kwargs or "progress_cb" in params:
                kwargs["progress_cb"] = self._on_progress
            if accepts_kwargs or "ignore_cache" in params:
                kwargs["ignore_cache"] = self._ignore_cache
            snapshot = replace(self._session, daw_state=deepcopy(self._session.daw_state))
            session = self._processor.fetch(snapshot, **kwargs)
            self.result.emit(True, "Fetch complete", session)
        except Exception as e:
            self.result.emit(False, str(e), None)


class DawTransferWorker(QThread):
    """Runs DawProcessor.transfer() off the main thread with progress."""

    progress = Signal(str)              # status text
    progress_value = Signal(int, int)   # (current, total)
    result = Signal(bool, str, object)  # (ok, message, results_list)

    def __init__(self, processor: DawProcessor, session, output_path: str, parent=None, close_session: bool = True):
        super().__init__(parent)
        self._processor = processor
        self._session = session
        self._output_path = output_path
        self._close_session = close_session

    def _on_progress(self, current: int, total: int, message: str):
        self.progress.emit(message)
        self.progress_value.emit(current, total)

    def run(self):
        try:
            results = self._processor.transfer(
                self._session, self._output_path, progress_cb=self._on_progress, close_when_done=self._close_session)
        except Exception as exc:
            self.result.emit(False, f"Transfer failed. {str(exc) or type(exc).__name__}", None)
            return

        if not results:
            self.result.emit(False, "No transfer operations ran. Check the track assignments.", results)
            return

        failures = [result for result in results if not result.success]
        if not failures:
            self.result.emit(True, f"Transfer complete ({len(results)} operations)", results)
            return

        for failure in failures:
            log.warning("DAW operation %s failed for %s: %s",
                        failure.command.command_type, failure.command.target, failure.error)

        first = failures[0]
        target = first.command.target
        message = first.error or f"{first.command.command_type.replace('_', ' ').capitalize()} failed."
        if target:
            message = f"{message} ({target})"
        completed = len(results) - len(failures)
        summary = (
            f"Transfer incomplete ({completed}/{len(results)} operations)."
            if completed else "Transfer failed."
        )
        message = f"{summary} {message}"
        if len(failures) > 1:
            message = f"{message} See the log for the other failed operations."
        self.result.emit(False, message, results)


class Phase1AnalyzeWorker(QThread):
    """Runs Phase 1 (Structural & Format) pipeline analysis in a background thread."""

    progress = Signal(str)                # descriptive text
    progress_value = Signal(int, int)     # (current_step, total_steps)
    track_analyzed = Signal(str, object)  # (filename, track) after detectors
    finished = Signal(object)             # (session)
    error = Signal(str)

    def __init__(self, session_context: object, config: dict):
        super().__init__()
        self.session_context = session_context
        self.config = config
        self._event_bus = EventBus()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self._event_bus.cancel()

    def run(self):
        try:
            event_bus = self._event_bus

            # Use the already loaded session
            session = self.session_context

            if not session.tracks:
                self.error.emit("No audio files found in session.")
                return

            self.progress.emit("Building pipeline\u2026")
            detectors = default_detectors()
            pipeline = Pipeline(
                detectors=detectors,
                config=self.config,
                event_bus=event_bus,
            )

            p1_track_dets = [d for d in pipeline.track_detectors if d.phase == LifecyclePhase.PHASE1
                             and d.id in pipeline.repair_verification.required_ids]
            p1_sess_dets = [d for d in pipeline.session_detectors if getattr(d, 'phase', LifecyclePhase.PHASE2) == LifecyclePhase.PHASE1]

            num_track_det = len(p1_track_dets)
            num_session_det = len(p1_sess_dets)

            # Preview can already have decoded individual sources.
            missing = [track for track in session.tracks
                       if track.status == "OK" and _needs_source_decode(track)]
            if missing:
                from sessionpreplib.audio import load_track
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import os

                self.progress.emit("Loading audio data\u2026")
                self.progress_value.emit(0, len(missing))

                with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as pool:
                    futures = {
                        pool.submit(load_track, t.filepath): t
                        for t in missing
                    }
                    for loaded, future in enumerate(as_completed(futures), 1):
                        if self._is_cancelled:
                            break
                        t = futures[future]
                        try:
                            _adopt_decoded_source(t, future.result())
                        except Exception as e:
                            t.status = "ERROR"
                            t.error = str(e)
                            t.audio_data = None
                        self.progress_value.emit(loaded, len(missing))

            # Discovery-time measurements (such as the peak of an unloaded source)
            # are stale even when the preview worker supplied the samples.
            ok_tracks = [t for t in session.tracks if t.status == "OK"]
            for track in ok_tracks:
                track._cache.clear()
            num_ok = len(ok_tracks)

            total_steps = (
                num_ok * num_track_det
                + num_session_det
            )
            self._step = 0
            self._total = max(total_steps, 1)
            self._step_lock = threading.Lock()

            # Track map for emitting track objects
            track_map = {t.filename: t for t in session.tracks}

            # Subscribe to pipeline events
            def on_detector_complete(detector_id, filename, **_kw):
                with self._step_lock:
                    self._step += 1
                    step = self._step
                self.progress.emit(f"Checking {filename} \u2014 {detector_id}")
                self.progress_value.emit(step, self._total)

            def on_session_detector_complete(detector_id, **_kw):
                with self._step_lock:
                    self._step += 1
                    step = self._step
                self.progress.emit(f"Session check \u2014 {detector_id}")
                self.progress_value.emit(step, self._total)

            def on_track_analyze_complete(filename, **_kw):
                track = track_map.get(filename)
                if track:
                    self.track_analyzed.emit(filename, track)

            event_bus.subscribe("detector.complete", on_detector_complete)
            event_bus.subscribe("session_detector.complete", on_session_detector_complete)
            event_bus.subscribe("track.analyze_complete", on_track_analyze_complete)

            # Phase 1: Analyze
            self.progress.emit("Validating Layout\u2026")
            self.progress_value.emit(0, self._total)
            session = pipeline.analyze_phase1(session)

            self.finished.emit(session)
        except Exception as e:
            self.error.emit(str(e))


class AnalyzeWorker(QThread):
    """Runs pipeline analysis in a background thread."""

    progress = Signal(str)                # descriptive text
    progress_value = Signal(int, int)     # (current_step, total_steps)
    track_analyzed = Signal(str, object)  # (filename, track) after detectors
    track_planned = Signal(str, object)   # (filename, track) after processors
    finished = Signal(object, object)     # (session, diagnostic_summary)
    error = Signal(str)

    def __init__(self, source_dir: str, config: dict, recursive: bool = False, track_settings=None):
        super().__init__()
        self.source_dir = source_dir
        self.config = config
        self.recursive = recursive
        self.track_settings = track_settings or {}
        self._event_bus = EventBus()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self._event_bus.cancel()

    def run(self):
        try:
            event_bus = self._event_bus

            self.progress.emit("Loading session\u2026")
            session = load_session(self.source_dir, self.config, event_bus=event_bus,
                                  recursive=self.recursive)
            for track in session.tracks:
                _gui_source_samples(track.audio_data)
                for key, value in self.track_settings.get(track.filename, {}).items():
                    setattr(track, key, value)

            if not session.tracks:
                self.error.emit("No audio files found in directory.")
                return

            self.progress.emit("Building pipeline\u2026")
            detectors = default_detectors()
            processors = default_processors()
            pipeline = Pipeline(
                detectors=detectors,
                audio_processors=processors,
                config=self.config,
                event_bus=event_bus,
            )

            # Calculate total progress steps
            ok_tracks = [t for t in session.tracks if t.status == "OK"]
            num_ok = len(ok_tracks)
            num_track_det = sum(d.phase == LifecyclePhase.PHASE2 and d.id in pipeline.repair_verification.required_ids
                                for d in pipeline.track_detectors)
            num_session_det = len([d for d in pipeline.session_detectors if getattr(d, 'phase', LifecyclePhase.PHASE2) == LifecyclePhase.PHASE2])
            total_steps = (
                num_ok * num_track_det      # analyze: per-track detectors
                + num_session_det           # analyze: session detectors
                + num_ok                    # plan: repairs and gain for each track
                + 1                         # build summary
            )
            self._step = 0
            self._total = max(total_steps, 1)
            self._step_lock = threading.Lock()

            # Track map for emitting track objects
            track_map = {t.filename: t for t in session.tracks}

            # Subscribe to pipeline events (called from pool threads)
            def on_detector_complete(detector_id, filename, **_kw):
                with self._step_lock:
                    self._step += 1
                    step = self._step
                self.progress.emit(f"Analyzing {filename} \u2014 {detector_id}")
                self.progress_value.emit(step, self._total)

            def on_session_detector_complete(detector_id, **_kw):
                with self._step_lock:
                    self._step += 1
                    step = self._step
                self.progress.emit(f"Session detector \u2014 {detector_id}")
                self.progress_value.emit(step, self._total)

            def on_track_analyze_complete(filename, **_kw):
                track = track_map.get(filename)
                if track:
                    self.track_analyzed.emit(filename, track)

            def on_track_plan_complete(filename, **_kw):
                with self._step_lock:
                    self._step += 1
                    step = self._step
                self.progress_value.emit(step, self._total)
                track = track_map.get(filename)
                if track:
                    self.track_planned.emit(filename, track)

            event_bus.subscribe("detector.complete", on_detector_complete)
            event_bus.subscribe("session_detector.complete", on_session_detector_complete)
            event_bus.subscribe("track.analyze_complete", on_track_analyze_complete)
            event_bus.subscribe("track.plan_complete", on_track_plan_complete)

            # Phase 2: Analyze (per-track detectors run in parallel)
            self.progress.emit("Analyzing Content\u2026")
            self.progress_value.emit(0, self._total)
            session = pipeline.analyze_phase2(session)

            # Phase 2: Plan (per-track processors run in parallel)
            self.progress.emit("Checking track repairs and planning gains\u2026")
            session = pipeline.plan(session, preserve_selections={
                name for name, settings in self.track_settings.items() if "processor_skip" in settings})

            # Build summary
            with self._step_lock:
                self._step += 1
                step = self._step
            self.progress.emit("Building summary\u2026")
            self.progress_value.emit(step, self._total)
            summary = build_diagnostic_summary(session, remaining=True)

            self.finished.emit(session, summary)
        except Exception as e:
            self.error.emit(str(e))


class AudioLoadWorker(QThread):
    """Load audio data from disk for a single track (no analysis).

    Used when a session is loaded from file and ``track.audio_data`` is
    ``None`` but the source file still exists on disk.  Emits ``finished``
    with the populated track on success, or ``error`` with a message.
    """

    finished = Signal(object)   # track with audio_data populated
    error = Signal(str)

    def __init__(self, track, parent=None):
        super().__init__(parent)
        self._track = track
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import time
            t0 = time.perf_counter()

            from sessionpreplib.audio import load_track
            track = self._track
            decoded = load_track(track.filepath)

            elapsed = (time.perf_counter() - t0) * 1000
            log.debug("AudioLoadWorker decoded '%s' in %.2f ms", track.filename, elapsed)

            if self._cancelled:
                return
            if ((track.source_file_revision is not None and track.source_file_revision != decoded.source_file_revision)
                    or (track.detector_results and track.source_file_signature is not None
                        and track.source_file_signature != decoded.source_file_signature)):
                raise SourceChangedError(f"{track.filename} has changed since analysis. Reanalyze this source.")
            _adopt_decoded_source(track, decoded)
            self.finished.emit(track)
        except Exception as exc:
            self.error.emit(str(exc))


class TopoAudioResolveWorker(QThread):
    """Load source audio and resolve a TopologyEntry off the UI thread.

    Emits ``finished`` with ``(audio_ndarray, samplerate)`` on success.
    """

    finished = Signal(object, int)  # (audio_data, samplerate)
    error = Signal(str)

    def __init__(self, entry, source_dir: str, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._source_dir = source_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import os
            import soundfile as sf
            from sessionpreplib.topology import resolve_entry_audio

            track_audio: dict[str, tuple] = {}
            sr = 44100

            for src in self._entry.sources:
                if self._cancelled:
                    return
                path = os.path.join(self._source_dir, src.input_filename)
                data, file_sr = sf.read(path, dtype='float64')
                track_audio[src.input_filename] = (data, file_sr)
                sr = file_sr  # use last samplerate (all should match)

            if self._cancelled:
                return

            resolved = resolve_entry_audio(self._entry, track_audio)
            self.finished.emit(resolved, sr)
        except Exception as exc:
            self.error.emit(str(exc))


class TopoMultiAudioWorker(QThread):
    """Load multiple tracks and produce stacked display + downmixed playback.

    Emits ``finished(display_audio, playback_audio, samplerate, labels)``
    where *display_audio* has all tracks' channels concatenated and
    *playback_audio* is summed by channel position.
    """

    finished = Signal(object, object, int, list)  # display, playback, sr, labels
    error = Signal(str)

    def __init__(self, items: list, side: str, source_dir: str, parent=None):
        """
        Parameters
        ----------
        items : list
            For ``side="input"``: list of ``(filepath, display_name, n_channels)``.
            For ``side="output"``: list of ``(TopologyEntry, display_name)``.
        side : str
            ``"input"`` or ``"output"``.
        source_dir : str
            Root directory for source audio files.
        """
        super().__init__(parent)
        self._items = items
        self._side = side
        self._source_dir = source_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import logging, time
        t0 = time.perf_counter()
        log = logging.getLogger(__name__)
        log.debug("[Trace] TopoMultiAudioWorker loading %d items", len(self._items))

        try:
            import os
            import numpy as np
            import soundfile as sf

            track_arrays = []   # list of 2-D arrays (samples, ch)
            track_ch_counts = []
            track_labels_list = []
            sr = 44100

            from sessionprepgui.waveform.panel import WaveformPanel

            from concurrent.futures import ThreadPoolExecutor, as_completed

            # --- 1. Pre-collect all unique audio file paths ---
            required_files = set()
            if self._side == "input":
                required_files.update(item[0] for item in self._items)
            else:
                for item in self._items:
                    for src in item[0].sources:
                        required_files.add(os.path.join(self._source_dir, src.input_filename))

            # --- 2. Parallel I/O Load ---
            audio_cache: dict[str, tuple[np.ndarray, int]] = {}
            # Limit workers to 4 to prevent SSD thrashing and massive memory spike
            max_workers = min(4, (os.cpu_count() or 1))

            log.debug("[Trace] TopoMultiAudioWorker loading %d unique files with %d workers", len(required_files), max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_to_path = {
                    pool.submit(sf.read, path, dtype='float64'): path
                    for path in required_files
                }
                for fut in as_completed(fut_to_path):
                    if self._cancelled:
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    path = fut_to_path[fut]
                    t_load = time.perf_counter()
                    data, file_sr = fut.result()
                    log.debug("[Trace] TopoMultiAudioWorker loaded '%s'. Shape: %s, %.1f MB",
                              os.path.basename(path), data.shape, data.nbytes / (1024 * 1024))
                    audio_cache[path] = (data, file_sr)

            if self._cancelled:
                return

            log.debug("[Trace] TopoMultiAudioWorker completed parallel I/O in %.2f ms", (time.perf_counter() - t0) * 1000)
            t_process = time.perf_counter()

            # --- 3. Process loaded audio natively ---
            if self._side == "input":
                for item in self._items:
                    if self._cancelled:
                        return
                    filepath = item[0]
                    name = item[1]
                    channels_to_keep = item[2] if len(item) > 2 else None

                    # Retrieve directly from in-memory parallel cache
                    data, file_sr = audio_cache[filepath]
                    sr = file_sr
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)

                    if channels_to_keep is not None:
                        data = data[:, channels_to_keep]
                        ch_labels = [f"{name} Ch{c}" for c in channels_to_keep]
                    else:
                        n_ch = data.shape[1]
                        ch_labels = []
                        names = WaveformPanel._CHANNEL_LABELS.get(n_ch)
                        for c in range(n_ch):
                            if names and c < len(names):
                                ch_labels.append(f"{name} {names[c]}")
                            else:
                                ch_labels.append(f"{name} Ch{c}")

                    track_arrays.append(data)
                    track_ch_counts.append(data.shape[1])
                    track_labels_list.append(ch_labels)
            else:  # output
                from sessionpreplib.topology import resolve_entry_audio
                for item in self._items:
                    if self._cancelled:
                        return
                    entry = item[0]
                    name = item[1]
                    channels_to_keep = item[2] if len(item) > 2 else None

                    # Load source audio for this entry
                    track_audio: dict[str, tuple] = {}
                    for src in entry.sources:
                        if self._cancelled:
                            return
                        path = os.path.join(self._source_dir, src.input_filename)
                        data, file_sr = audio_cache[path]
                        track_audio[src.input_filename] = (data, file_sr)
                        sr = file_sr

                    resolved = resolve_entry_audio(entry, track_audio)
                    if resolved.ndim == 1:
                        resolved = resolved.reshape(-1, 1)

                    if channels_to_keep is not None:
                        resolved = resolved[:, channels_to_keep]
                        ch_labels = [f"{name} Ch{c}" for c in channels_to_keep]
                    else:
                        n_ch = resolved.shape[1]
                        ch_labels = []
                        names = WaveformPanel._CHANNEL_LABELS.get(n_ch)
                        for c in range(n_ch):
                            if names and c < len(names):
                                ch_labels.append(f"{name} {names[c]}")
                            else:
                                ch_labels.append(f"{name} Ch{c}")

                    track_arrays.append(resolved)
                    track_ch_counts.append(resolved.shape[1])
                    track_labels_list.append(ch_labels)

            if self._cancelled or not track_arrays:
                return

            # --- Build display audio (list of contiguous 1D channels) ---
            log.debug("[Trace] TopoMultiAudioWorker building display & playback arrays...")
            t_stack = time.perf_counter()
            display_audio = []
            max_samples = max(a.shape[0] for a in track_arrays)
            for a in track_arrays:
                if a.shape[0] < max_samples:
                    pad = np.zeros((max_samples - a.shape[0], a.shape[1]),
                                   dtype=np.float64)
                    a = np.vstack([a, pad])
                # Each channel becomes its own perfectly contiguous slice
                for c in range(a.shape[1]):
                    display_audio.append(np.ascontiguousarray(a[:, c]))

            # --- Build playback audio (summed by channel position) ---
            max_ch = max(track_ch_counts)
            n_tracks = len(track_arrays)
            playback = np.zeros((max_samples, max_ch), dtype=np.float64)
            for a in track_arrays:
                playback[:a.shape[0], :a.shape[1]] += a
            playback /= n_tracks

            # Squeeze mono playback
            if playback.shape[1] == 1:
                playback = playback[:, 0]

            # --- Channel labels ---
            labels = []
            for lst in track_labels_list:
                labels.extend(lst)

            log.debug("[Trace] TopoMultiAudioWorker finished %d items (Total: %.2f ms, Process/Stack: %.2f ms)",
                      len(self._items), (time.perf_counter() - t0) * 1000, (time.perf_counter() - t_process) * 1000)
            self.finished.emit(display_audio, playback, sr, labels)
        except Exception as exc:
            import traceback, logging
            logging.getLogger(__name__).error("TopoMultiAudioWorker error: %s", traceback.format_exc())
            self.error.emit(str(exc))


class RepairPlanWorker(QThread):
    """Plan edited repairs from immutable choices without changing a live session."""

    completed = Signal(object)
    progress_value = Signal(int, int)

    def __init__(self, session, tracks, config, parent=None, *, refresh_anchors=(), revalidate_sources=()):
        super().__init__(parent)
        self.session = session
        self.config = dict(config)
        self.refresh_anchors = frozenset(refresh_anchors)
        self.revalidate_sources = frozenset(revalidate_sources)
        self.targets = {track.filename: (track, _gui_source_samples(track.audio_data)) for track in tracks}
        self.tracks = [
            replace(t, audio_data=audio, repair_overrides=dict(t.repair_overrides), processor_skip=set(t.processor_skip),
                    _cache=dict(t._cache), detector_results=dict(t.detector_results), processor_results={})
            for t, audio in self.targets.values()
        ]
        self.failure = ""
        self._events = EventBus()
        self._progress_lock = threading.Lock()
        self._done = 0
        self._events.subscribe("track.plan_complete", self._on_planned)

    def _on_planned(self, **_):
        with self._progress_lock:
            self._done += 1
            self.progress_value.emit(self._done, len(self.tracks))

    def cancel(self):
        self._events.cancel()

    def run(self):
        try:
            from sessionpreplib.models import SessionContext
            pipeline = Pipeline(default_detectors(), default_processors(), self.config, self._events)
            snapshot = SessionContext(self.tracks, self.config)
            pipeline.plan(snapshot, preserve_selections={t.filename for t in self.tracks},
                          refresh_anchors=self.refresh_anchors, revalidate_sources=self.revalidate_sources)
        except Exception as exc:
            self.failure = str(exc) or type(exc).__name__
        self.completed.emit(self)


class PrepareWorker(QThread):
    """Runs Pipeline.prepare() off the main thread with progress."""

    progress = Signal(str)              # status text
    progress_value = Signal(int, int)   # (current, total)
    track_prepared = Signal(str)        # filename
    prepare_finished = Signal()         # renamed: avoid shadowing QThread.finished
    error = Signal(str)
    completed = Signal(object)

    def __init__(self, session, processors, output_dir: str):
        super().__init__()
        self.owner = session
        self.outdated = False
        self.failure = ""
        # Share read-only sample buffers. Freeze all choices used for the bytes
        # and fader receipt together before the GUI accepts another edit.
        tracks = [replace(t, processor_results=deepcopy(t.processor_results),
                          repair_overrides=dict(t.repair_overrides),
                          processor_skip=set(t.processor_skip), _cache=dict(t._cache))
                  for t in session.tracks]
        self._session = replace(session, tracks=tracks, config=dict(session.config),
                                topology=deepcopy(session.topology),
                                transfer_manifest=deepcopy(session.transfer_manifest),
                                output_tracks=list(session.output_tracks))
        self._processors = [deepcopy(proc) for proc in processors]
        self._output_dir = output_dir

    def publish(self):
        """Publish committed output receipts without replacing later edits to the tracks."""
        result, session = self._session, self.owner
        if session.topology is None:
            session.topology = result.topology
        session.output_tracks = result.output_tracks
        session.transfer_manifest = result.transfer_manifest
        session.base_transfer_manifest = result.base_transfer_manifest
        session.prepare_state = "stale" if self.outdated else result.prepare_state
        for key in ("_prepared_files", "_prepare_errors", "_output_gain_adjustments"):
            session.config[key] = result.config[key]
        if self.outdated:
            session.config["_output_gain_adjustments"] = {}
        for current, frozen in zip(session.tracks, result.tracks, strict=True):
            current.processed_filepath = frozen.processed_filepath
            current.applied_processors = frozen.applied_processors
            current.prepared_file_signature = frozen.prepared_file_signature
            current.prepared_fader_db = frozen.prepared_fader_db

    def _on_progress(self, current: int, total: int, message: str):
        if self.isInterruptionRequested():
            raise InterruptedError("Prepare cancelled")
        self.progress.emit(message)
        self.progress_value.emit(current, total)

    def run(self):
        try:
            from sessionpreplib.pipeline import Pipeline

            # Build a lightweight pipeline with just the processors
            pipeline = Pipeline(
                detectors=[],
                audio_processors=self._processors,
                config=self._session.config,
            )
            pipeline.prepare(
                self._session,
                self._output_dir,
                progress_cb=self._on_progress,
            )
            self.prepare_finished.emit()
        except Exception as e:
            self.failure = str(e)
            self.error.emit(self.failure)
        finally:
            self.completed.emit(self)


class TopologyApplyWorker(QThread):
    """Resolve channel topology and write rerouted files to output dir.

    This worker loads source audio on demand, applies the channel routing
    defined in the session topology, and writes the resulting files.
    No audio processors are applied — that is Phase 2's job.
    """

    progress = Signal(str)
    progress_value = Signal(int, int)
    apply_finished = Signal()
    error = Signal(str)

    def __init__(self, session, output_dir: str, source_dir: str | None = None,
                 peaks_dir: str | None = None, *, source_tracks=None):
        super().__init__()
        self._session = session
        self._output_dir = output_dir
        self._source_dir = source_dir
        self._source_tracks = source_tracks
        self._peaks_dir = peaks_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            import os
            from pathlib import Path
            import soundfile as sf
            from sessionpreplib.audio import load_track, AUDIO_EXTENSIONS
            from sessionpreplib.repair_plan import check_source_signature
            from sessionpreplib.topology import resolve_entry_audio
            from sessionpreplib.models import TrackContext

            session = self._session
            topology = session.topology
            if topology is None:
                self.error.emit("No topology defined.")
                return

            output_dir = self._output_dir
            output_path = Path(output_dir).resolve()
            source_root = Path(self._source_dir).resolve() if self._source_dir else None
            if source_root is not None and source_root.is_relative_to(output_path):
                raise ValueError("The layout output folder must be separate from the source folder.")

            # Collect source filenames referenced by topology
            tracks = self._source_tracks if self._source_tracks is not None else session.tracks
            track_map = {t.filename: t for t in tracks if t.status == "OK"}

            needed_sources = {src.input_filename for entry in topology.entries
                              if entry.output_channels > 0 for src in entry.sources}

            n_sources = len(needed_sources)
            n_entries = len(topology.entries)
            # Total includes peak building phase when peaks_dir is set
            total = n_sources + n_entries + (n_entries if self._peaks_dir else 0)

            # Phase 2 tracks may be renders with the same relative names. Keep
            # the PCM and format together, anchored to the original source path.
            source_audio: dict[str, tuple] = {}
            sources: dict[str, TrackContext] = {}
            for step, filename in enumerate(sorted(needed_sources)):
                if self._is_cancelled:
                    return

                log.debug("Apply topology: loading source '%s' (%d/%d)", filename, step + 1, total)
                self.progress.emit(f"Loading {filename}")
                self.progress_value.emit(step, total)
                track = track_map.get(filename)
                if source_root is None and track is None:
                    raise ValueError(f"Missing source: {filename}")
                path = (source_root / filename if source_root is not None else Path(track.filepath)).resolve()
                if path.is_relative_to(output_path):
                    raise ValueError(f"{filename}: the original source is inside the layout output folder.")
                if source_root is not None and track is not None and Path(track.filepath).resolve() != path:
                    track = None
                if (track is not None and track.audio_data is not None and
                        (source_root is None or (track.source_file_signature and track.source_file_revision))):
                    source = track
                else:
                    source = load_track(str(path))
                    if track is not None:
                        source.group = track.group
                if source_root is not None and track is not None:
                    check_source_signature(track)
                sources[filename] = source
                source_audio[filename] = (source.audio_data, source.samplerate)

            # Validate every route before replacing any previous layout audio.
            for entry in topology.entries:
                if entry.output_channels < 1:
                    continue
                for src in entry.sources:
                    samples = sources[src.input_filename].audio_data
                    channels = 1 if samples.ndim == 1 else samples.shape[1]
                    for route in src.routes:
                        if not 0 <= route.source_channel < channels:
                            raise ValueError(f"{src.input_filename}: source channel {route.source_channel + 1} "
                                             f"is unavailable ({channels} channels).")
                        if not 0 <= route.target_channel < entry.output_channels:
                            raise ValueError(f"{entry.output_filename}: output channel {route.target_channel + 1} "
                                             f"is unavailable ({entry.output_channels} channels).")

            if self._is_cancelled:
                return
            if os.path.isdir(output_dir):
                for dirpath, _dirnames, filenames in os.walk(output_dir, topdown=False):
                    for fname in filenames:
                        if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                            try:
                                os.unlink(os.path.join(dirpath, fname))
                            except OSError:
                                pass
                    if dirpath != output_dir:
                        try:
                            os.rmdir(dirpath)
                        except OSError:
                            pass
            os.makedirs(output_dir, exist_ok=True)

            # Phase B: Resolve topology + write output files (no peak building)
            output_tracks = []
            errors = []
            written_files: list[tuple[str, str, int]] = []  # (output_filename, dst_path, sr)
            base_step = n_sources
            for idx, entry in enumerate(topology.entries):
                if self._is_cancelled:
                    return

                step = base_step + idx
                log.debug("Apply topology: writing '%s' (%d/%d)", entry.output_filename, step + 1, total)
                self.progress.emit(f"Writing {entry.output_filename}")
                self.progress_value.emit(step, total)

                # Skip entries with no channels (e.g. all channels moved elsewhere)
                if entry.output_channels < 1 or not entry.sources:
                    continue

                try:
                    resolved = resolve_entry_audio(entry, source_audio)
                    src_track = sources[entry.sources[0].input_filename]
                    sr = src_track.samplerate

                    dst = os.path.join(output_dir, entry.output_filename)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    sf.write(dst, resolved, sr, subtype=src_track.subtype)
                    written_files.append((entry.output_filename, dst, sr))

                    n_samples = resolved.shape[0]
                    out_tc = TrackContext(
                        filename=entry.output_filename,
                        filepath=dst,
                        audio_data=None,
                        samplerate=sr,
                        channels=entry.output_channels,
                        total_samples=n_samples,
                        bitdepth=src_track.bitdepth,
                        subtype=src_track.subtype,
                        duration_sec=(n_samples / sr) if sr > 0 else 0.0,
                        group=src_track.group,
                    )
                    output_tracks.append(out_tc)
                except Exception as e:
                    errors.append((entry.output_filename, str(e)))

            # Free source audio memory before peak building
            source_audio.clear()
            sources.clear()

            # Phase C: Build peak caches (re-reading from just-written files)
            if self._peaks_dir:
                from ..waveform.peakcache import (
                    build_peaks, save_peaks,
                    peaks_path_for, get_source_mtime,
                )
                peak_base_step = n_sources + n_entries
                for idx, (out_fn, dst, sr) in enumerate(written_files):
                    if self._is_cancelled:
                        return

                    step = peak_base_step + idx
                    log.debug("Apply topology: building peaks '%s' (%d/%d)", out_fn, step + 1, total)
                    self.progress.emit(f"Building peaks for {out_fn}")
                    self.progress_value.emit(step, total)

                    try:
                        pp = peaks_path_for(self._peaks_dir, os.path.abspath(dst))
                        mtime = get_source_mtime(dst)
                        import time as _time
                        _t0 = _time.perf_counter()
                        data, file_sr = sf.read(dst, dtype="float64")
                        _t_read = _time.perf_counter()
                        pd = build_peaks(data, file_sr, source_mtime=mtime)
                        _t_build = _time.perf_counter()
                        save_peaks(pd, pp)
                        _t_save = _time.perf_counter()
                        log.debug(
                            "Built peak cache for '%s' -> %s "
                            "(%d ch, %d levels, read=%.1f ms, build=%.1f ms, save=%.1f ms, total=%.1f ms)",
                            out_fn, pp, pd.channels, len(pd.levels),
                            (_t_read - _t0) * 1000,
                            (_t_build - _t_read) * 1000,
                            (_t_save - _t_build) * 1000,
                            (_t_save - _t0) * 1000,
                        )
                    except Exception as e:
                        log.debug("Failed to build/save peak cache for '%s' -> %s: %s", out_fn, pp, e)

            self.progress_value.emit(total, total)

            session.output_tracks = output_tracks
            session.config["_topology_apply_errors"] = errors
            self.apply_finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class BatchReanalyzeWorker(QThread):
    """Re-run detectors and/or processors for a subset of tracks.

    Used by the batch-edit workflow: after the user applies a change
    (e.g. RMS anchor override) to multiple tracks at once, this worker
    performs the heavy re-analysis in the background so the GUI stays
    responsive.

    Parameters
    ----------
    tracks:
        The tracks to re-analyze.
    detectors:
        Configured detector instances (from ``session.detectors``).
    processors:
        Configured processor instances (from ``session.processors``).
    run_detectors:
        If ``True`` (default), re-run track-level detectors before
        processors.  Set to ``False`` for lightweight changes that
        only need processor re-calculation (e.g. classification override).
    """

    progress = Signal(str)
    progress_value = Signal(int, int)     # (current, total)
    track_done = Signal(str)              # filename
    batch_finished = Signal()             # renamed to avoid QThread.finished collision
    error = Signal(str)

    def __init__(self, tracks, detectors, processors,
                 run_detectors: bool = True):
        super().__init__()
        self._tracks = list(tracks)
        self._detectors = detectors
        self._processors = processors
        self._run_detectors = run_detectors

    def run(self):
        try:
            total = len(self._tracks)
            for i, track in enumerate(self._tracks):
                self.progress.emit(f"Re-analyzing {track.filename}\u2026")
                self.progress_value.emit(i, total)

                if self._run_detectors:
                    for det in self._detectors:
                        if isinstance(det, TrackDetector):
                            try:
                                result = det.analyze(track)
                                track.detector_results[det.id] = result
                            except Exception:
                                pass

                for proc in self._processors:
                    try:
                        result = proc.process(track)
                        track.processor_results[proc.id] = result
                    except Exception:
                        pass

                self.track_done.emit(track.filename)

            self.progress_value.emit(total, total)
            self.batch_finished.emit()
        except Exception as e:
            self.error.emit(str(e))
