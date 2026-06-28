from __future__ import annotations

import numpy as np

from sessionprepgui.waveform.renderer import WaveformRenderer, WaveformRenderCtx
from sessionprepgui.waveform.peakcache import (
    build_peaks,
    load_peaks,
    MipLevel,
    peaks_path_for,
    PeakData,
    query_peaks,
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


def _peak_data_with_identifiable_levels() -> PeakData:
    total_samples = 1_048_576
    levels = []
    for spb in (256, 1024, 4096, 16384):
        n_bins = total_samples // spb
        marker = float(spb)
        data = np.full((n_bins, 1, 2), marker, dtype=np.float32)
        levels.append(MipLevel(samples_per_bin=spb, data=data))
    return PeakData(
        channels=1,
        samplerate=48_000,
        total_samples=total_samples,
        source_mtime=123,
        levels=levels,
    )


def test_query_peaks_uses_finest_level_when_view_is_more_detailed_than_cache():
    peak_data = _peak_data_with_identifiable_levels()

    result = query_peaks(peak_data, 0, peak_data.total_samples, 8192)

    mins, maxs = result[0]
    assert np.all(mins == 256.0)
    assert np.all(maxs == 256.0)


def test_query_peaks_fast_uses_finest_level_when_view_is_more_detailed_than_cache():
    peak_data = _peak_data_with_identifiable_levels()

    result = query_peaks_fast(peak_data, 0, peak_data.total_samples, 8192)

    mins, maxs = result[0]
    assert np.all(mins == 256.0)
    assert np.all(maxs == 256.0)


def test_query_peaks_fast_selects_expected_cache_resolution_by_view_density():
    peak_data = _peak_data_with_identifiable_levels()
    cases = [
        (4096, 256.0),    # 256 samples/pixel
        (2048, 256.0),    # 512 samples/pixel
        (768, 1024.0),    # 1365 samples/pixel
        (192, 4096.0),    # 5461 samples/pixel
        (32, 16384.0),    # 32768 samples/pixel
    ]

    for width, expected_marker in cases:
        result = query_peaks_fast(peak_data, 0, peak_data.total_samples, width)
        mins, maxs = result[0]
        assert np.all(mins == expected_marker)
        assert np.all(maxs == expected_marker)


def test_waveform_renderer_prefers_raw_audio_when_view_is_finer_than_peak_cache():
    peak_data = _peak_data_with_identifiable_levels()
    raw = np.linspace(-1.0, 1.0, peak_data.total_samples, dtype=np.float64)
    renderer = WaveformRenderer()
    renderer.set_track_data([raw])
    renderer.set_peak_data(peak_data)
    ctx = WaveformRenderCtx(
        x0=0,
        draw_w=8192,
        draw_h=100,
        margin_right=0,
        view_start=0,
        view_end=peak_data.total_samples,
        vscale=1.0,
        channels=[raw],
        num_channels=1,
        show_rms_lr=False,
        show_rms_avg=False,
        show_markers=False,
        wf_antialias=False,
        wf_line_width=1,
    )

    renderer._build_peaks(ctx)

    mins, maxs = renderer._peaks_cache[0]
    assert np.all(mins < 2.0)
    assert np.all(maxs < 2.0)


def test_peaks_path_uses_basename_stem(tmp_path):
    path = peaks_path_for(str(tmp_path), "nested/source file.wav")

    assert path == str(tmp_path / "source file.peaks")


def test_waveform_renderer_ignores_stale_peak_cache_channel_count(caplog):
    peak_data = _peak_data_with_identifiable_levels()
    raw_channels = [
        np.linspace(-1.0, 1.0, peak_data.total_samples, dtype=np.float64)
        for _ in range(4)
    ]
    renderer = WaveformRenderer()
    renderer.set_track_data(raw_channels)
    renderer.set_peak_data(peak_data)
    ctx = WaveformRenderCtx(
        x0=0,
        draw_w=128,
        draw_h=100,
        margin_right=0,
        view_start=0,
        view_end=peak_data.total_samples,
        vscale=1.0,
        channels=raw_channels,
        num_channels=4,
        show_rms_lr=False,
        show_rms_avg=False,
        show_markers=False,
        wf_antialias=False,
        wf_line_width=1,
    )

    renderer._build_peaks(ctx)

    assert len(renderer._peaks_cache) == 4
    assert "Ignoring waveform peak cache with 1 channels for 4-channel view" in caplog.text
