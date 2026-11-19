import json

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.repair_plan import plan_repairs
from sessionpreplib.repairs import RepairRecipe
from sessionpreplib.reports import generate_report, save_json


def test_reports_distinguish_planned_failed_and_delivered_gain(tmp_path):
    def track(name):
        item = TrackContext(name, name, None, 48000, 1, 48000, "24", "PCM_24", 1)
        item.repair_recipe = RepairRecipe(dc_remove=True)
        item.processor_results["bimodal_normalize"] = ProcessorResult(
            "bimodal_normalize", 9, "Transient", "Peak", {"fader_offset": -9})
        return item

    planned, failed, written = (track(f"{name}.wav") for name in ("Planned", "Failed", "Written"))
    failed.processed_filepath = "an older render"
    session = SessionContext([planned, failed, written], {
        "_execute_errors": [(failed.filename, "Output exceeds PCM headroom")],
        "_executed_files": {written.filename: {"gain_db": -1.75, "fader_db": 0,
                                               "applied_processors": ["dc_remove"]}},
    })
    text = generate_report(session, {}, str(tmp_path / "report.txt"), [])
    assert "Output exceeds PCM headroom" in text
    assert "dc_remove" in text
    assert "-1.75" in text
    path = tmp_path / "report.json"
    save_json(session, {}, str(path), [])
    rows = {entry["filename"]: entry["processing"] for entry in json.loads(path.read_text())["tracks"]}
    assert [rows[t.filename]["status"] for t in session.tracks] == ["planned", "failed", "written"]
    for t in (planned, failed):
        row = rows[t.filename]
        assert row["gain_applied_db"] is None and row["fader_db"] is None
        assert row["applied_repairs"] == [] and row["planned_repairs"] == ["dc_remove"]
    assert rows[failed.filename]["error"] == "Output exceeds PCM headroom"
    assert rows[written.filename]["gain_applied_db"] == -1.75
    assert rows[written.filename]["fader_db"] == 0
    assert rows[written.filename]["applied_repairs"] == ["dc_remove"]


def test_edit_gap_settling_is_reported_by_channel_through_delivery(tmp_path):
    rate = 48000
    tone = 0.1 + 0.2 * np.cos(2 * np.pi * 1000 * np.arange(2 * rate) / rate + np.pi / 2)
    source = np.zeros((6 * rate, 2))
    source[2 * rate:4 * rate, 0] = tone
    source[rate:3 * rate, 1] = -tone
    source[4 * rate:, 1] = -tone
    path = tmp_path / "Edited.wav"
    sf.write(path, source, rate, subtype="DOUBLE")
    track = load_track(str(path))
    track.repair_overrides = {"dc_remove": True}
    corrected = plan_repairs(track, {"subsonic_enabled": False}).audio_data
    gaps = track.repair_measurements["boundary_settling"]
    assert len(gaps) == 2
    for gap, (channel, start, stop) in zip(gaps, [(0, 4 * rate, 6 * rate), (1, 3 * rate, 4 * rate)]):
        tail = np.abs(corrected[start:stop, channel])
        above = np.flatnonzero(tail > 1e-6)
        assert (gap["channel"], gap["start_sample"], gap["end_sample"]) == (
            channel, start + above[0], start + above[-1])
        assert gap["peak_dbfs"] == pytest.approx(20 * np.log10(tail.max()))
    assert gaps[0]["peak_dbfs"] == pytest.approx(-19.967, abs=0.01)
    assert (gaps[0]["end_sample"] - gaps[0]["start_sample"] + 1) / rate == pytest.approx(0.91648, abs=1/rate)
    assert not corrected[:rate].any()
    np.testing.assert_array_equal(track.audio_data, source)

    session = SessionContext([track], {})
    output = tmp_path / "prepared"
    Pipeline([]).execute(session, str(output))
    assert not session.config["_execute_errors"]
    np.testing.assert_array_equal(sf.read(output / track.filename)[0], corrected)
    # The delivered result stays attached to its receipt after a new plan.
    track.repair_overrides = {"dc_remove": False}
    plan_repairs(track, {})
    assert not track.repair_measurements.get("boundary_settling")
    report = generate_report(session, {}, str(tmp_path / "report.txt"), [])
    assert "Boundary settling" in report and str(4 * rate) in report
    target = tmp_path / "report.json"
    save_json(session, {}, str(target), [])
    assert json.loads(target.read_text())["tracks"][0]["processing"]["boundary_settling"] == gaps
