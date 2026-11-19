"""Band decisions through the track planner, calibrated against known source RMS."""
import numpy as np
import pytest

from sessionpreplib.models import TrackContext
from sessionpreplib.repair_plan import plan_repairs
from sessionpreplib.repairs import band_evidence


def tone(rate, hz, rms_db, seconds):
    time = np.arange(round(seconds * rate)) / rate
    return np.sqrt(2.0) * 10 ** (rms_db / 20) * np.cos(2 * np.pi * hz * time + 0.17)


def track_for(samples, rate):
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    return TrackContext("edge.wav", "edge.wav", samples, rate, channels, len(samples),
                        "64-bit Float", "DOUBLE", len(samples) / rate)


@pytest.mark.parametrize(("hz", "inside"), [
    (15, True), (19.4, True), (20, True), (20.6, False), (21, False), (22, False),
])
def test_infrasonic_selection_protects_bass_above_the_inclusive_edge(hz, inside):
    rate = 48000
    track = track_for(tone(rate, hz, -9.03, 2), rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "repair_low_cut_hz": 40})
    evidence = track.repair_measurements["original"]["low_band"]
    assert track.repair_measurements["suggested"]["low_cut"] is inside
    assert track.repair_recipe.low_cut_hz == (40 if inside else None)
    if inside:
        assert evidence["band_dbfs"] == pytest.approx(-9.03, abs=0.3)
        assert evidence["regions"] == [[0, len(track.audio_data) - 1, 0]]
    else:
        assert evidence["band_dbfs"] < -50


@pytest.mark.parametrize(("hz", "rms_db", "seconds", "selected"), [
    (24090, -9.03, 0.30, False), (24096, -9.03, 0.30, False),
    (24100, -59.9, 0.30, True), (24100, -60.1, 0.30, False),
    (24104, -59.9, 0.30, True), (24110, -59.9, 0.30, True),
    (30000, -59.9, 0.299, False),
])
def test_ultrasonic_cutoff_level_and_persistence_are_independent(hz, rms_db, seconds, selected):
    rate = 96000
    track = track_for(tone(rate, hz, rms_db, seconds), rate)
    plan_repairs(track, {"repair_auto_ultrasonic": True})
    evidence = track.detector_results["ultrasonic"].data
    assert track.repair_recipe.ultrasonic is selected
    if selected:
        assert evidence["regions"] == [[0, len(track.audio_data) - 1, 0]]
    else:
        assert evidence["regions"] == []
    if hz >= 24100:
        assert evidence["band_dbfs"] == pytest.approx(rms_db, abs=0.1)
    else:
        assert evidence["band_dbfs"] < -60


@pytest.mark.parametrize("enable_detector", [True, False])
def test_short_manual_ultrasonic_repair_measures_both_sides_without_selection(enable_detector):
    rate = 96000
    source = tone(rate, 30000, -9.0309, 0.05)
    track = track_for(source, rate)
    track.repair_overrides = {"ultrasonic": True}
    plan_repairs(track, {"ultrasonic_enabled": enable_detector})
    measures = track.repair_measurements
    before = measures["original"]["ultrasonic"]
    after = measures["corrected"]["ultrasonic"]
    assert before["regions"] == after["regions"] == []
    assert not measures["suggested"]["ultrasonic"]
    assert before["band_dbfs"] == pytest.approx(-9.0309, abs=0.02)
    assert np.isfinite(after["band_dbfs"]) and after["band_dbfs"] < before["band_dbfs"] - 80
    assert track.repair_recipe.ultrasonic


def test_each_channel_measures_only_its_part_of_a_resolvable_musical_mix():
    rate = 48000
    program = tone(rate, 20.6, -9, 2) + tone(rate, 82.41, -12, 2)
    rumble = tone(rate, 13.4, -21, 2)
    track = track_for(np.column_stack((program, -program + rumble)), rate)
    plan_repairs(track, {"repair_auto_low_cut": True})
    evidence = track.repair_measurements["original"]["low_band"]
    assert evidence["regions"] == [[0, len(program) - 1, 1]]
    assert evidence["band_dbfs"] == pytest.approx(-21, abs=0.3)
    expected_ratio = -21 - 10 * np.log10(sum(10 ** (db / 10) for db in (-9, -12, -21)))
    assert evidence["ratio_db"] == pytest.approx(expected_ratio, abs=0.3)


def test_unresolved_beating_just_above_the_low_edge_remains_advisory():
    rate = 48000
    t = np.arange(3 * rate) / rate
    source = (0.5 * np.cos(2 * np.pi * 20.001 * t + 0.17) +
              0.25 * np.cos(2 * np.pi * 20.06 * t + 2.45))
    track = track_for(source, rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "repair_low_cut_hz": 40})
    assert not track.repair_measurements["suggested"]["low_cut"]
    assert track.repair_recipe.low_cut_hz is None


@pytest.mark.parametrize(("rate", "low", "high", "hz", "seconds", "consecutive"), [
    (48000, 0, 19.3, 19.3, 1.0, 3),
    (50001, 24100, 25000.5, 24100, 0.1, 5),
])
def test_an_inclusive_edge_can_fall_between_native_bins(rate, low, high, hz, seconds, consecutive):
    source = tone(rate, hz, -59.9, seconds * (1 + consecutive))
    measured = band_evidence(source, rate, low_hz=low, high_hz=high, window_sec=seconds,
                             level_db=-60, ratio_db=-35, consecutive=consecutive)
    assert measured["band_dbfs"] == pytest.approx(-59.9, abs=0.1)
    assert len(measured["regions"]) == 1 and measured["regions"][0][0] == 0


@pytest.mark.parametrize(("rate", "seconds", "low", "high", "noise_band", "pink"), [
    (48000, 1.0, 0, 20, (5, 14), True),
    (96000, 0.1, 24100, 48000, (26000, 37000), False),
])
def test_band_noise_levels_match_isolated_reference_in_a_mixture(rate, seconds, low, high, noise_band, pink):
    n = round(rate * seconds)
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    rng = np.random.default_rng(130198)

    def noise(bounds, amplitude):
        bins = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
        coeffs = np.zeros(len(frequencies), complex)
        coeffs[bins] = np.exp(1j * rng.uniform(0, 2 * np.pi, bins.sum()))
        if pink:
            coeffs[bins] /= np.sqrt(frequencies[bins])
        out = np.fft.irfft(coeffs, n)
        return amplitude * out / np.sqrt(np.mean(out * out))

    band = noise(noise_band, 0.05)
    program = noise((250, 1100), 0.15)
    window = np.sin(np.pi * np.arange(n) / n) ** 2
    expected = 10 * np.log10(np.sum((band * window) ** 2) / np.sum(window ** 2))
    measured = band_evidence(program + band, rate, low_hz=low, high_hz=high, window_sec=seconds,
                             level_db=-50, ratio_db=-20, consecutive=1)
    assert measured["regions"] == [[0, n - 1, 0]]
    assert measured["band_dbfs"] == pytest.approx(expected, abs=0.3)


def test_unmeasurable_ac_and_measured_silence_have_distinct_readings():
    kwargs = dict(sample_rate=96000, low_hz=24100, high_hz=48000, window_sec=0.1,
                  level_db=-60, ratio_db=-35, consecutive=5)
    for samples in (np.empty(0), np.array([0.8])):
        assert band_evidence(samples, **kwargs)["band_dbfs"] is None
    assert band_evidence(np.zeros(4800), **kwargs)["band_dbfs"] == -np.inf
