"""The same calibrated band evidence drives suggestions and automatic low cuts."""
import numpy as np
import pytest

from sessionpreplib.detectors.subsonic import SubsonicDetector
from sessionpreplib.models import TrackContext
from sessionpreplib.repair_plan import plan_repairs
from sessionpreplib.repairs import RepairRecipe


@pytest.mark.parametrize("automatic", [False, True])
def test_quiet_local_rumble_is_suggested_before_auto_selection(automatic):
    rate = 8000
    t = np.arange(40 * rate) / rate
    program = 0.03 * np.sin(2 * np.pi * 440 * t)
    rumble = 0.008 * np.sin(2 * np.pi * 8 * t) * (t < 3)
    original = np.column_stack((program, -program - rumble))
    track = TrackContext("local.wav", "local.wav", original, rate, 2, len(t), "64-bit Float", "DOUBLE", 40)
    sub = SubsonicDetector()
    sub.configure({})
    raw = sub.analyze(track)
    track.detector_results[sub.id] = raw
    assert not raw.data["subsonic_warn"]

    plan_repairs(track, {"repair_auto_low_cut": automatic})

    assert track.detector_results[sub.id] is raw
    assert track.repair_measurements["suggested"]["low_cut"]
    assert track.repair_recipe == (RepairRecipe(low_cut_hz=10) if automatic else RepairRecipe())
    before = track.repair_measurements["original"]["low_band"]
    after = track.repair_measurements["corrected"]["low_band"]
    assert before["band_dbfs"] == pytest.approx(-44.95, abs=0.02)
    assert before["ratio_db"] == pytest.approx(-11.78, abs=0.02)
    assert before["regions"] and all(ch == 1 for _, _, ch in before["regions"])
    if automatic:
        assert after["band_dbfs"] == pytest.approx(before["band_dbfs"] - 5.37, abs=0.05)
        assert not after["regions"]
        assert track.repair_measurements["low_cut_resolved"]
        assert track.repair_measurements["corrected"]["subsonic_ratio_db"] < raw.data["subsonic_ratio_db"]
        assert "subsonic_resolved" not in track.repair_measurements
    else:
        assert after == before
