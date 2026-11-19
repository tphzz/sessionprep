"""Planning reuses calibrated models and measurements without reusing stale audio."""
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sessionpreplib import repairs


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(repairs, "_band_readings", OrderedDict())


@pytest.fixture
def scans(monkeypatch):
    started = []
    levels = repairs._BandWindow.levels

    def counted(self, block):
        started.append(len(block))
        return levels(self, block)

    monkeypatch.setattr(repairs._BandWindow, "levels", counted)
    return started


def measure(samples, rate=1024, **kwargs):
    options = dict(low_hz=0, high_hz=20, window_sec=1., level_db=-50., ratio_db=-20., consecutive=3)
    return repairs.band_evidence(samples, rate, **(options | kwargs))


def rumble():
    t = np.arange(4 * 1024) / 1024
    return 0.1 * np.cos(2 * np.pi * 8 * t)


def test_reloaded_samples_hit_and_each_caller_owns_its_regions(scans):
    samples = rumble()
    reference = measure(samples)
    first_scan = len(scans)
    assert first_scan == 7
    assert reference["regions"] == [[0, len(samples) - 1, 0]]
    assert reference["band_dbfs"] == pytest.approx(-23.0103, abs=0.01)
    reference["regions"][0][0] = 77
    repeated = measure(samples.copy())
    repeated["regions"].clear()
    assert measure(samples)["regions"] == [[0, len(samples) - 1, 0]]
    assert len(scans) == first_scan

    # Same array identity and dimensions, new audio.
    samples[:] = 0
    assert measure(samples)["regions"] == []
    assert len(scans) == first_scan * 2


@pytest.mark.parametrize("options", [
    {"rate": 2048}, {"low_hz": 12}, {"high_hz": 4}, {"window_sec": 0.5},
])
def test_changed_meter_geometry_measures_again_and_strided_copies_can_reuse_it(scans, options):
    samples = rumble()
    assert measure(samples)["regions"]
    count = len(scans)
    changed = measure(samples, **options)
    assert len(scans) > count
    count = len(scans)
    storage = np.empty(len(samples) * 2)
    storage[::2] = samples
    assert measure(storage[::2], **options) == changed
    assert len(scans) == count


@pytest.mark.parametrize("setting", [{"level_db": -10}, {"ratio_db": 1}, {"consecutive": 10}])
def test_gates_and_persistence_reapply_to_saved_readings(scans, setting):
    samples = rumble()
    assert measure(samples)["regions"]
    count = len(scans)
    assert measure(samples, **setting)["regions"] == []
    assert measure(samples)["regions"]
    assert len(scans) == count


def test_warming_a_lower_precision_band_edge_cannot_change_an_ultrasonic_decision(monkeypatch):
    rate = 95904
    t = np.arange(28770) / rate
    source = 0.1 * np.sin(2 * np.pi * 24100 * t + 0.41)

    def ultrasonic(edge):
        return measure(source, rate, low_hz=edge, high_hz=rate / 2, window_sec=0.1,
                       level_db=-60, ratio_db=-35, consecutive=5)

    def cold(edge):
        with monkeypatch.context() as m:
            m.setattr(repairs, "_band_readings", OrderedDict())
            return ultrasonic(edge)

    expected = cold(24100)
    assert expected["regions"] == [[0, len(t) - 1, 0]]
    narrow = cold(np.float32(24100))
    # Zero-dimensional arrays are also accepted numeric arguments and can
    # change value between calls.
    edge_array = np.array(24100, dtype=np.int64)
    assert ultrasonic(edge_array) == expected
    edge_array[...] = 24500
    assert ultrasonic(edge_array) == cold(24500)
    assert ultrasonic(np.array(24100, dtype=np.float32)) == narrow
    for first, second in ((np.float32(24100), 24100), (24100, np.float32(24100))):
        with monkeypatch.context() as m:
            m.setattr(repairs, "_band_readings", OrderedDict())
            for edge in (first, second, first):
                assert ultrasonic(edge) == (narrow if isinstance(edge, np.float32) else expected)


def test_identical_raw_bytes_with_different_format_or_routing_get_a_new_scan(scans):
    samples = np.zeros((2048, 2), dtype=np.float64)
    for view in (samples, samples.reshape(-1), samples.view(np.int64)):
        count = len(scans)
        assert measure(view)["regions"] == []
        assert len(scans) > count
    samples[:, 1] = rumble()[:2048]
    assert measure(samples)["regions"] == [[0, 2047, 1]]
    assert measure(samples[:, ::-1].copy())["regions"] == [[0, 2047, 0]]


def test_mutating_then_restoring_input_during_measurement_cannot_poison_the_cache(monkeypatch):
    samples = rumble()
    original = samples.copy()
    levels = repairs._BandWindow.levels

    def edit_during_scan(self, block):
        samples[:] = 0
        try:
            return levels(self, block)
        finally:
            samples[:] = original

    with monkeypatch.context() as m:
        m.setattr(repairs._BandWindow, "levels", edit_during_scan)
        assert measure(samples)["regions"] == [[0, len(samples) - 1, 0]]
    assert measure(original)["regions"] == [[0, len(original) - 1, 0]]


def test_concurrent_planners_get_independent_measurements():
    samples = rumble()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(measure, [samples] * 8))
    assert all(result["regions"] == [[0, len(samples) - 1, 0]] for result in results)
    results[0]["regions"][0].clear()
    assert all(result["regions"][0][0] == 0 for result in results[1:])
    assert measure(samples)["regions"][0][0] == 0


def test_persistence_crosses_cached_chunks_and_an_edit_rescans_only_its_own_audio(scans, monkeypatch):
    rate = 1024
    t = np.arange(40 * rate) / rate
    source = np.column_stack((np.zeros(len(t)), 0.1 * np.cos(2 * np.pi * 8 * t)))
    result = measure(source, rate, consecutive=35)
    assert result["regions"] == [[0, len(t) - 1, 1]]
    count = len(scans)
    assert measure(source.copy(), rate, consecutive=35) == result
    assert len(scans) == count

    source[-2 * rate:, 1] = 0
    changed = measure(source, rate, consecutive=35)
    assert 0 < len(scans) - count < 32
    assert changed["regions"][0][1] < len(t) - 1
    with monkeypatch.context() as m:
        m.setattr(repairs, "_band_readings", OrderedDict())
        assert measure(source, rate, consecutive=35) == changed


def test_repeated_neighboring_tones_share_calibration_across_frames_and_channels(monkeypatch):
    constructed = Counter()
    init = repairs._BoundaryPartial.__init__

    def counted(self, cycles, width, window_dc):
        constructed[cycles] += 1
        init(self, cycles, width, window_dc)

    monkeypatch.setattr(repairs._BoundaryPartial, "__init__", counted)
    rate = 8000
    t = np.arange(4 * rate) / rate
    source = 0.4 * np.cos(2 * np.pi * 26.37 * t + 0.3) + 0.08 * np.cos(2 * np.pi * 19.94 * t + 0.17)
    result = measure(np.column_stack((source, -source)), rate)
    assert result["regions"] == [[0, len(t) - 1, 0], [0, len(t) - 1, 1]]
    assert constructed[26] == 1
    assert constructed[20] == 1
