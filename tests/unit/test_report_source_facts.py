"""Read the real producer's evidence at the public report boundaries."""

import json

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.detectors import AudioClassifierDetector, SilenceDetector, TailExceedanceDetector
from sessionpreplib.pipeline import Pipeline, load_session
from sessionpreplib.rendering import build_diagnostic_summary, render_diagnostic_summary_text
from sessionpreplib.reports import build_warnings, generate_report, save_json


@pytest.mark.parametrize(("rules", "winner", "loser"), [
    (["Drums:kick", "Inside:Kick In.wav$"], "Drums", "Inside"),
    (["Inside:Kick In.wav$", "Drums:kick"], "Inside", "Drums"),
])
def test_group_overlap_visible_once_per_surface(tmp_path, rules, winner, loser):
    for name in ("Kick In.wav", "Keys.wav"):
        sf.write(tmp_path / name, [.125, -.125], 8000)
    config = {"group": rules}
    session = load_session(str(tmp_path), config)
    warning = f"Group overlap: Kick In.wav matches '{loser}' but is already assigned to '{winner}'"
    assert session.groups == {"Kick In.wav": winner}
    assert session.warnings == [warning]
    warnings = build_warnings(session, config)
    summary = build_diagnostic_summary(session)
    overlaps = [g["items"] for g in summary["attention"] if g["title"] == "Grouping overlaps"]
    assert overlaps == [[warning]]
    text = generate_report(session, config, str(tmp_path / "out.txt"), warnings, render_diagnostic_summary_text(summary))
    assert text.count(warning) == 1
    save_json(session, config, str(tmp_path / "out.json"), warnings)
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf8"))
    assert data["session"]["warnings"].count(warning) == 1
    assert {t["filename"]: t["processing"]["group"] for t in data["tracks"]} == {"Kick In.wav": winner, "Keys.wav": None}


@pytest.mark.parametrize(("default", "overrides"), [("max", ("p80", None)), ("percentile", (None, "max"))])
def test_report_uses_measured_tail_with_mixed_anchors(tmp_path, default, overrides):
    samples = np.tile([.125, -.125], 32000)
    samples[20000:22000] *= 4
    for name in ("A.wav", "B.wav"):
        sf.write(tmp_path / name, samples, 8000)
    sf.write(tmp_path / "Clean.wav", np.full_like(samples, .125), 8000)
    config = {"rms_anchor": default, "window": 50, "rms_percentile": 80}
    session = load_session(str(tmp_path), config)
    tracks = {t.filename: t for t in session.tracks}
    tracks["A.wav"].rms_anchor_override, tracks["B.wav"].rms_anchor_override = overrides
    tracks["Clean.wav"].rms_anchor_override = "p80"
    pipeline = Pipeline([TailExceedanceDetector(), AudioClassifierDetector(), SilenceDetector()], config=config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    result = tracks["A.wav"].detector_results["tail_exceedance"]
    assert [(i.sample_start, i.sample_end) for i in result.issues] == [(19680, 22319)]
    assert result.data["tail_summary"]["max_exceed_db"] == pytest.approx(20 * np.log10(4))
    assert not tracks["B.wav"].detector_results["tail_exceedance"].applicable
    assert not tracks["Clean.wav"].detector_results["tail_exceedance"].issues
    text = generate_report(session, config, str(tmp_path / "report.txt"), [])
    tail = text.split("MOMENTARY RMS UPPER TAIL")[1].split("FILE OVERVIEW")[0]
    assert "A.wav" in tail and "B.wav" not in tail and "Clean.wav" not in tail
    assert "samples 19680-22319" in tail and "+12.04 dB" in tail and "0.330s" in tail
    assert "Anchor RMS: -18.06 dBFS (p80)" in tail
    summary = build_diagnostic_summary(session)
    ref, = next(g["item_refs"] for g in summary["attention"] if g["detector_id"] == "tail_exceedance")
    assert ref["filename"] == "A.wav"


def test_report_distinguishes_no_measurement_from_a_clean_measured_track(tmp_path):
    sf.write(tmp_path / "Steady.wav", np.tile([.125, -.125], 4000), 8000)
    config = {"rms_anchor": "percentile", "window": 50}
    session = load_session(str(tmp_path), config)
    report = tmp_path / "out.txt"
    assert "No applicable tail measurements" in generate_report(session, config, str(report), [])
    pipeline = Pipeline([TailExceedanceDetector(), SilenceDetector(), AudioClassifierDetector()], config=config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    assert "No significant tail exceedances" in generate_report(session, config, str(report), [])
