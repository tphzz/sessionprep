from __future__ import annotations

import numpy as np

from sessionprepgui.waveform.peakcache import (
    build_peaks,
    load_peaks,
    peaks_path_for,
    query_peaks_fast,
    save_peaks,
)


def test_peakcache_round_trip(tmp_path):
    samples = np.linspace(-1.0, 1.0, 4096, dtype=np.float64)
    audio = np.column_stack([samples, -samples])
    peak_data = build_peaks(audio, 48_000, source_mtime=123)
    path = tmp_path / "track.peaks"

    save_peaks(peak_data, str(path))
    loaded = load_peaks(str(path), expected_mtime=123)

    assert loaded is not None
    assert loaded.channels == 2
    assert loaded.samplerate == 48_000
    assert loaded.total_samples == 4096
    assert loaded.source_mtime == 123
    assert [level.samples_per_bin for level in loaded.levels] == [256, 1024, 4096]


def test_peakcache_stale_mtime_returns_none(tmp_path):
    audio = np.zeros((4096, 1), dtype=np.float64)
    peak_data = build_peaks(audio, 44_100, source_mtime=123)
    path = tmp_path / "track.peaks"

    save_peaks(peak_data, str(path))

    assert load_peaks(str(path), expected_mtime=456) is None


def test_peakcache_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "track.peaks"
    path.write_bytes(b"not a valid peaks file")

    assert load_peaks(str(path)) is None


def test_query_peaks_fast_returns_channel_min_max_arrays():
    samples = np.sin(np.linspace(0.0, 8.0 * np.pi, 8192, dtype=np.float64))
    audio = np.column_stack([samples, samples * 0.5])
    peak_data = build_peaks(audio, 48_000, source_mtime=123)

    result = query_peaks_fast(peak_data, 0, 8192, 64)

    assert len(result) == 2
    for mins, maxs in result:
        assert mins.shape == (64,)
        assert maxs.shape == (64,)
        assert np.all(mins <= maxs)


def test_peaks_path_uses_basename_stem(tmp_path):
    path = peaks_path_for(str(tmp_path), "nested/source file.wav")

    assert path == str(tmp_path / "source file.peaks")
