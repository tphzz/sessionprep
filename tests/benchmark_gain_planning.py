"""Repeatable gain-planning benchmark. Run explicitly outside the regression gate."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.config import default_config
from sessionpreplib.detectors import default_detectors
from sessionpreplib.gain_staging import apply_group_gains
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repair_verification import detector_check, remaining_results


def run_case(files):
    cfg = default_config()
    pipeline = Pipeline(default_detectors(), default_processors(), cfg, max_workers=1)
    session = SessionContext([], cfg)
    calls = Counter()
    readings = {}

    def measure(label, operation):
        calls.clear()
        tracemalloc.start()
        start = perf_counter()
        try:
            operation()
            elapsed = perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        readings[label] = dict(seconds=round(elapsed, 5), peak_mib=round(peak / 2**20, 2),
                               decoder_calls=calls["decode"], detector_calls=calls["detect"])

    def counted(function, key):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return function(*args, **kwargs)
        return wrapper

    def analyze():
        session.tracks[:] = [load_track(str(path)) for path in files]
        pipeline.analyze_phase1(session)
        pipeline.analyze_phase2(session)

    with ExitStack() as patches:
        patches.enter_context(patch.object(sf.SoundFile, "read", counted(sf.SoundFile.read, "decode")))
        for kind in {type(d) for d in pipeline.track_detectors}:
            patches.enter_context(patch.object(kind, "analyze", counted(kind.analyze, "detect")))
        measure("source_analysis", analyze)
        measure("bypassed", lambda: pipeline.plan(session))
        track = session.tracks[0]
        track.repair_overrides["dc_remove"] = True
        measure("one_repair_cold", lambda: pipeline.plan(session))
        if track.corrected_analysis is None or not track.repair_recipe.active:
            raise RuntimeError("The benchmark fixture did not activate DC removal")
        measure("one_repair_warm", lambda: pipeline.plan(session))
        track.classification_override = "Transient"
        measure("classification_edit", lambda: pipeline.plan(SessionContext([track], cfg)))
        for member in session.tracks:
            member.group = "Band"
        measure("group_edit", lambda: apply_group_gains(session, linked_groups={"Band"},
                headroom_db=8.0, anchor=None))

        def compare_sources():
            for member in session.tracks:
                remaining_results(member, pipeline.track_detectors)
                for detector in pipeline.track_detectors:
                    detector_check(member, detector.id)
        measure("findings_comparison", compare_sources)
    if readings["source_analysis"]["decoder_calls"] != len(files) or not readings["one_repair_cold"]["detector_calls"]:
        raise RuntimeError(f"Missing cold decoder/detector work: {readings}")
    for label in ("bypassed", "one_repair_warm", "classification_edit", "group_edit", "findings_comparison"):
        if readings[label]["detector_calls"] or readings[label]["decoder_calls"]:
            raise RuntimeError(f"Unexpected work in {label}: {readings[label]}")
    return readings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.seconds, args.tracks, args.repeats) < 1:
        parser.error("seconds, tracks and repeats must be positive")
    rate = 48000
    seconds = np.arange(rate * args.seconds) / rate
    with TemporaryDirectory(prefix="sessionprep-gain-benchmark-") as directory:
        files = []
        for index in range(args.tracks):
            tone = 0.14 * np.sin(2 * np.pi * (220 + 55 * index) * seconds)
            path = Path(directory) / f"track-{index}.wav"
            sf.write(path, np.column_stack((tone + 0.02, tone * 0.8)), rate, subtype="PCM_24")
            files.append(path)
        results = [run_case(files) for _ in range(args.repeats)]
    print(json.dumps(dict(platform=platform.platform(), processor=platform.processor(),
                          logical_cpus=os.cpu_count(), workers=1, rate=rate, channels=2,
                          seconds_per_track=args.seconds, tracks=args.tracks, runs=results), indent=2))


if __name__ == "__main__":
    main()
