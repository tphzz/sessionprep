"""Public analysis entry points with independent, deterministic extension detectors."""

from threading import Barrier

import numpy as np
import pytest

from sessionpreplib.config import ConfigError
from sessionpreplib.detector import SessionDetector, TrackDetector
from sessionpreplib.events import EventBus
from sessionpreplib.models import DetectorResult, LifecyclePhase, SessionContext, Severity, TrackContext
from sessionpreplib.pipeline import Pipeline


class Probe(TrackDetector):
    def __init__(self, name, read, *, depends_on=(), phase=LifecyclePhase.PHASE2):
        self.id = name
        self.read = read
        self.depends_on = list(depends_on)
        self.phase = phase

    @classmethod
    def html_help(cls):
        return ""

    def analyze(self, track):
        return DetectorResult(self.id, Severity.CLEAN, "", {"value": self.read(track)})


class SessionProbe(SessionDetector):
    def __init__(self, name, read):
        self.id = name
        self.read = read

    @classmethod
    def html_help(cls):
        return ""

    def analyze(self, session):
        return [DetectorResult(self.id, Severity.CLEAN, "", {"filename": t.filename, "value": self.read(t)})
                for t in session.tracks if t.status == "OK"]


def track_pair():
    return [TrackContext(name, "", np.full(16, value), 8000, 1, 16, "64-bit float", "DOUBLE", 16/8000)
            for name, value in (("left.wav", .125), ("right.wav", -.25))]


def unavailable(_):
    raise RuntimeError("measurement unavailable")


def test_extensions_follow_dependencies_and_isolate_failures_across_two_workers():
    barrier = Barrier(2, timeout=5)

    def first_sample(track):
        barrier.wait()
        return track.audio_data[0]

    input_probe = Probe("input", first_sample, phase=LifecyclePhase.PHASE1)
    doubled = Probe("double", lambda t: 2 * t.detector_results["input"].data["value"], depends_on=["input"])
    failed = Probe("broken", unavailable)
    dependent = Probe("needs_broken", lambda _: 100, depends_on=["broken"])
    echo = SessionProbe("echo", lambda t: t.detector_results["double"].data["value"])
    pipeline = Pipeline([dependent, doubled, failed, echo, SessionProbe("session_fail", unavailable), input_probe],
                        max_workers=2)
    session = SessionContext(track_pair(), {})
    pipeline.analyze_phase1(session)
    assert [set(t.detector_results) for t in session.tracks] == [{"input"}, {"input"}]
    pipeline.analyze_phase2(session)
    for t, expected in zip(session.tracks, [.25, -.5], strict=True):
        assert t.detector_results["double"].data["value"] == expected
        assert t.detector_results["echo"].data == {"filename": t.filename, "value": expected}
        assert t.detector_results["broken"].error == "measurement unavailable"
        assert "broken" in t.detector_results["needs_broken"].error
    assert session.config["_session_det_session_fail"][0].error == "measurement unavailable"

    # A rerun may fail after the previous session detector completed successfully.
    echo.read = unavailable
    pipeline.analyze_session_detectors(session, LifecyclePhase.PHASE2)
    assert session.config["_session_det_echo"][0].error == "measurement unavailable"
    assert all("echo" not in t.detector_results for t in session.tracks)
    assert [t.detector_results["double"].data["value"] for t in session.tracks] == [.25, -.5]


@pytest.mark.parametrize(("specs", "message"), [
    ([("first", ()), ("first", ())], "Duplicate detector ID: first"),
    ([("first", ("lost",))], "first.*lost.*does not exist"),
    ([("first", ("second",)), ("second", ("first",))], "[Cc]ycl"),
])
def test_bad_detector_graph_fails_at_construction(specs, message):
    with pytest.raises(ConfigError, match=message):
        Pipeline([Probe(name, unavailable, depends_on=deps) for name, deps in specs])


def test_worker_error_marks_the_correct_track_and_keeps_healthy_sibling(monkeypatch):
    bus = EventBus()
    complete = []
    bus.subscribe("track.analyze_complete", lambda **event: complete.append((event["filename"], event["index"])))
    session = SessionContext(track_pair(), {})
    pipeline = Pipeline([Probe("sample", lambda track: track.audio_data[0])], event_bus=bus, max_workers=2)
    prepare = pipeline.repair_verification.prepare_phase

    def fail_right(track, detectors):
        if track.filename == "right.wav":
            raise OSError("source became unreadable")
        return prepare(track, detectors)

    monkeypatch.setattr(pipeline.repair_verification, "prepare_phase", fail_right)
    pipeline.analyze_phase2(session)
    left, right = session.tracks
    assert left.status == "OK" and left.detector_results["sample"].data["value"] == .125
    assert "source became unreadable" in right.status
    assert sorted(complete) == [("left.wav", 0), ("right.wav", 1)]
