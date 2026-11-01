from sessionpreplib.models import DetectorResult, SessionContext, Severity, TrackContext
from sessionpreplib.rendering import build_diagnostic_summary


def test_summary_preserves_strings_and_adds_unambiguous_track_references():
    filename = "Vocal: alternate & <take>.wav"
    track = TrackContext(filename, "missing.wav", None, 48000, 1, 96000,
                         "24-bit", "PCM_24", 2.0)
    track.detector_results["clipping"] = DetectorResult(
        "clipping", Severity.PROBLEM, "clipped", {"is_clipped": True, "runs": 1})
    session = SessionContext([track], {})
    summary = build_diagnostic_summary(session)
    group = next(g for g in summary["problems"] if g["title"] == "Digital clipping")
    assert group["items"] == [filename + ": clipping detected (1 clipped ranges)"]
    assert group["item_refs"] == [{"filename": filename,
                                  "detail": "clipping detected (1 clipped ranges)"}]
    assert group["detector_id"] == "clipping"
    assert summary["clean_count"] == 0
    assert summary["total_ok"] == 1
