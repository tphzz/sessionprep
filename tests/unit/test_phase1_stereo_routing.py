"""The default layout must be backed by measurements of the entire source."""

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.detectors import DualMonoDetector, OneSidedSilenceDetector, SilenceDetector
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.topology import build_default_topology


def _scan(path, samples, *, subtype="FLOAT", **config):
    sf.write(path, samples, 48_000, subtype=subtype)
    track = load_track(str(path))
    pipeline = Pipeline([SilenceDetector(), DualMonoDetector(), OneSidedSilenceDetector()],
                        config=config)
    pipeline.analyze_phase1(SessionContext(tracks=[track], config=config))
    return track.detector_results, build_default_topology([track]).entries


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
def test_exact_duplicate_routes_left_across_file_formats(tmp_path, subtype):
    signal = np.tile([0.75, -0.25, 0.125, 0.0], 100_001)
    results, (entry,) = _scan(tmp_path / "dual.wav", np.column_stack((signal, signal)),
                              subtype=subtype)
    assert results["dual_mono"].data["dual_mono"] is True
    assert entry.output_channels == 1
    assert [(r.source_channel, r.target_channel) for r in entry.sources[0].routes] == [(0, 0)]


def test_single_nonzero_stereo_frame_is_comparable(tmp_path):
    results, (entry,) = _scan(tmp_path / "one.wav", [[0.25, 0.25]])
    assert results["dual_mono"].applicable
    assert results["dual_mono"].data["dual_mono"] is True
    assert entry.output_channels == 1


@pytest.mark.parametrize("index", [1, 131_071, 399_999])
def test_isolated_stereo_difference_preserves_both_channels(tmp_path, index):
    signal = np.tile([0.25, -0.25, -0.125, 0.125], 100_000)
    stereo = np.column_stack((signal, signal))
    stereo[index, 1] += 0.125
    results, (entry,) = _scan(tmp_path / "edit.wav", stereo)
    assert results["dual_mono"].data["dual_mono"] is False
    assert entry.output_channels == 2


@pytest.mark.parametrize("quiet_side", [0, 1])
def test_activity_between_sampled_frames_is_preserved(tmp_path, quiet_side):
    # Every even frame on one side is silent. Both channels have substantial AC energy.
    steady = np.tile([0.5, -0.5, -0.5, 0.5], 100_000)
    sparse = np.tile([0.0, 0.5, 0.0, -0.5], 100_000)
    stereo = np.column_stack((sparse, steady) if quiet_side == 0 else (steady, sparse))
    results, (entry,) = _scan(tmp_path / "stereo.wav", stereo)
    measured = results["one_sided_silence"].data
    assert measured["one_sided_silence"] is False
    assert measured["l_rms_db" if quiet_side == 0 else "r_rms_db"] == pytest.approx(
        20 * np.log10(0.5 / np.sqrt(2)))
    assert entry.output_channels == 2


@pytest.mark.parametrize(("delta", "expected"), [
    (2**-15, True),
    (np.nextafter(2**-15, np.inf), False),
])
def test_epsilon_is_an_inclusive_absolute_sample_difference(tmp_path, delta, expected):
    stereo = [[0.0, delta], [0.5, 0.5], [-0.5, -0.5]]
    results, (entry,) = _scan(tmp_path / "edge.wav", stereo, subtype="DOUBLE",
                              dual_mono_eps=2**-15)
    assert results["dual_mono"].data["dual_mono"] is expected
    assert entry.output_channels == (1 if expected else 2)


@pytest.mark.parametrize("frames", [22, 998, 65538])
@pytest.mark.parametrize("above_threshold", [False, True])
def test_one_sided_rms_threshold_includes_equality_at_any_file_length(tmp_path, frames, above_threshold):
    threshold = 10**(-80 / 20)
    amplitude = np.nextafter(threshold, np.inf) if above_threshold else threshold
    alternating = np.tile([1.0, -1.0], frames // 2)
    stereo = np.column_stack((amplitude * alternating, 0.125 * alternating))
    results, (entry,) = _scan(tmp_path / "level.wav", stereo, subtype="DOUBLE",
                              one_sided_silence_db=-80)
    assert results["one_sided_silence"].data["one_sided_silence"] is (not above_threshold)
    assert entry.output_channels == (2 if above_threshold else 1)


@pytest.mark.parametrize("ripple", [False, True])
def test_one_sided_rms_removes_dc_without_creating_or_losing_ac(tmp_path, ripple):
    dc = np.float64(0.1)
    high = np.nextafter(dc, np.inf) if ripple else dc
    left = np.tile([dc, high], 300_000)
    right = 0.2 * np.sin(2 * np.pi * 997 * np.arange(left.size) / 48000)
    results, (entry,) = _scan(tmp_path / "dc.wav", np.column_stack((left, right)),
                              subtype="DOUBLE", one_sided_silence_db=-350)
    levels = results["one_sided_silence"].data
    expected_rms_db = 20 * np.log10((high - dc) / 2) if ripple else -np.inf
    assert levels["l_rms_db"] == pytest.approx(expected_rms_db)
    assert entry.output_channels == (2 if ripple else 1)


def test_separate_mono_files_are_independent_layout_inputs(tmp_path):
    tracks = []
    for side in ("L", "R"):
        path = tmp_path / f"keys.{side}.wav"
        sf.write(path, [0.25, -0.25], 48_000)
        tracks.append(load_track(str(path)))
    Pipeline([SilenceDetector(), DualMonoDetector()]).analyze_phase1(SessionContext(tracks, {}))
    assert all(not t.detector_results["dual_mono"].applicable for t in tracks)
    assert [(e.output_filename, e.output_channels) for e in build_default_topology(tracks).entries] == [
        ("keys.L.wav", 1), ("keys.R.wav", 1),
    ]
