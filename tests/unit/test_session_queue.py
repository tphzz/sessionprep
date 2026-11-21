"""Exercise the public session queue with real decoder, analysis and rendering."""

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.detectors import AudioClassifierDetector, DCOffsetDetector, SilenceDetector
from sessionpreplib.events import EventBus
from sessionpreplib.models import JobStatus
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.queue import SessionQueue


def pipeline(config):
    return Pipeline([SilenceDetector(), AudioClassifierDetector(), DCOffsetDetector()],
                    default_processors(), config)


@pytest.mark.parametrize("execute,overwrite", [(False, False), (True, False), (True, True)])
def test_real_pipeline_measures_both_phases_and_renders_only_on_execute(tmp_path, execute, overwrite):
    samples = np.tile([0.25, 0.0], 12000)
    source = tmp_path / "Keys.wav"
    sf.write(source, samples, 48000, subtype="PCM_16")
    original = source.read_bytes()
    queue = SessionQueue({"force_transient": ["Keys"], "target_peak": 20 * np.log10(0.5)})
    queue.add(str(tmp_path), {"execute": execute, "overwrite": overwrite})
    job = queue.run_next(pipeline)
    assert job.status == JobStatus.COMPLETED, job.error
    track, = job.result.session.tracks
    assert track.detector_results["silence"].data["is_silent"] is False
    assert track.detector_results["dc_offset"].data["signed_dc"] == [0.125]
    assert track.processor_results["bimodal_normalize"].gain_db == pytest.approx(20 * np.log10(2))
    if execute:
        result, rate = sf.read(source if overwrite else tmp_path / "processed" / source.name)
        assert rate == 48000
        np.testing.assert_array_equal(result, 2 * samples)
    else:
        assert list(tmp_path.iterdir()) == [source]
    assert (tmp_path / "_originals" / source.name if overwrite else source).read_bytes() == original


def test_failed_write_keeps_per_track_results_and_queue_continues_in_priority_order(tmp_path):
    bad_dir = tmp_path / "bad"
    good_dir = tmp_path / "good"
    bad_dir.mkdir()
    good_dir.mkdir()
    for directory, names in [(bad_dir, ("A.wav", "B.wav")), (good_dir, ("Good.wav",))]:
        for name in names:
            sf.write(directory / name, np.tile([-0.25, 0.25], 1200), 48000)
    (bad_dir / "processed" / "A.wav").mkdir(parents=True)
    queue = SessionQueue({"execute": True})
    bad = queue.add(str(bad_dir), label="bad", priority=3)
    good = queue.add(str(good_dir), label="good", priority=1)
    second = queue.add(str(good_dir), {"execute": False}, label="second", priority=1)
    cancelled = queue.add(str(good_dir), label="cancelled", priority=0)
    queue.cancel(cancelled.job_id)
    queue.reorder(bad.job_id, 0)
    events = []
    bus = EventBus()
    bus.subscribe("job.start", lambda job_id: events.append((job_id, "start")))
    bus.subscribe("job.complete", lambda job_id, status: events.append((job_id, status)))
    assert queue.run_all(pipeline, bus) == [bad, good, second]
    assert events == [("bad", "start"), ("bad", "failed"), ("good", "start"),
                      ("good", "completed"), ("second", "start"), ("second", "completed")]
    assert "A.wav" in bad.error
    assert list(bad.result.session.config["_executed_files"]) == ["B.wav"]
    assert (bad_dir / "processed" / "B.wav").is_file()
    assert cancelled.status == JobStatus.CANCELLED
    assert queue.run_next(pipeline) is None


@pytest.mark.parametrize("broken", ["absent", "empty", "unreadable"])
def test_unusable_input_job_cannot_report_success(tmp_path, broken):
    directory = tmp_path / broken
    if broken != "absent":
        directory.mkdir()
    if broken == "unreadable":
        (directory / "Invalid.wav").write_bytes(b"unreadable wave")
    queue = SessionQueue()
    queue.add(str(directory))
    job = queue.run_next(pipeline)
    assert job.status == JobStatus.FAILED
    assert ("Invalid.wav" if broken == "unreadable" else str(directory)) in job.error


def test_jobs_own_captured_settings_and_have_unambiguous_ids(tmp_path):
    defaults = {"group": ["Kit:Kick"]}
    individual = {"force_transient": ["Kick"]}
    queue = SessionQueue(defaults)
    first = queue.add(str(tmp_path), individual, label="one")
    defaults["group"].clear()
    individual["force_transient"].append("Bass")
    second = queue.add(str(tmp_path))
    first.config["group"].append("Bass:DI")
    assert second.config["group"] == ["Kit:Kick"]
    assert first.config["force_transient"] == ["Kick"]
    with pytest.raises(ValueError, match="one"):
        queue.add(str(tmp_path), label="one")


def test_library_overwrite_requires_backup_distinct_from_source(tmp_path):
    source = tmp_path / "Keys.wav"
    sf.write(source, np.tile([0.125, -0.125], 1200), 48000)
    original = source.read_bytes()
    queue = SessionQueue()
    queue.add(str(tmp_path), {"execute": True, "overwrite": True, "backup": "."})
    job = queue.run_next(pipeline)
    assert job.status == JobStatus.FAILED
    assert source.name in job.error
    assert source.read_bytes() == original
