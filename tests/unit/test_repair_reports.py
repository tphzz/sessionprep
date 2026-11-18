import json

from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext
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
