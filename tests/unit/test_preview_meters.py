"""Calibration at the waveform preview's numerical and painted outputs."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from sessionprepgui.waveform.compute import WaveformLoadWorker, compute_mel_spectrogram
from sessionprepgui.waveform.renderer import WaveformRenderer, WaveformRenderCtx
from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx
from sessionpreplib.audition import inspection_spectrogram
from sessionpreplib.models import TrackContext


@pytest.mark.parametrize("rate", [44100, 48000, 96000, 192000])
@pytest.mark.parametrize("band,tones", [("low", (37.5, 42.37)), ("high", (20725, 21518.75))])
def test_inspection_tone_pairs_hold_their_peak_levels_between_bins(rate, band, tones):
    time = np.arange(3 * rate) / rate
    signal = sum(amp * np.cos(2 * np.pi * f * time + 0.27)
                 for amp, f in zip((0.5, 0.25), tones))
    times, bins, levels = inspection_spectrogram(np.column_stack([signal, -signal]), rate, band)
    for f, amplitude in zip(tones, (0.5, 0.25)):
        near = np.abs(bins - f) <= (1 if band == "low" else 50)
        interior = (times >= 1) & (times <= 2)
        np.testing.assert_allclose(levels[near][:, interior].max(axis=0),
                                   20 * np.log10(amplitude), atol=0.3)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackmanharris"])
@pytest.mark.parametrize("fft", [512, 1024, 2048, 4096, 8192])
def test_full_band_uses_the_same_peak_dbfs_at_every_selectable_resolution(fft, window):
    rate = 48000
    time = np.arange(max(rate, 16 * fft)) / rate
    for frequency in (20, 1125, 1234.56, 23980, 24000):
        signal = 0.25 * np.cos(2 * np.pi * frequency * time)
        result = compute_mel_spectrogram([signal, -signal], rate, n_fft=fft,
                                         window=window, compare_channels=True)
        interior = result[:, result.shape[1] // 4:3 * result.shape[1] // 4]
        np.testing.assert_allclose(interior.max(axis=0), 20 * np.log10(0.25), atol=0.3)


@pytest.mark.parametrize("edge", [0, 2048, 3072, -1])
def test_full_band_covers_window_boundaries_and_file_edges(edge):
    audio = np.zeros(8192)
    audio[edge] = 1
    levels = compute_mel_spectrogram([audio], 48000, n_fft=2048, compare_channels=True)
    # A one-sample impulse at the center of a 2048-point Hann window
    # contributes 2 / (2048 / 2) in the calibrated one-sided spectrum.
    assert levels.max() >= 20 * np.log10(4 / 2048) - 0.3


def test_short_ultrasonic_burst_survives_time_and_frequency_reduction_in_the_painted_image():
    rate = 96000
    samples = 25 * rate
    signal = np.zeros(samples)
    burst = np.arange(-480, 480)
    signal[round(rate * 12.285) + burst] = (
        0.5 * np.cos(2 * np.pi * 30000 * burst / rate) * np.hanning(len(burst)))
    times, freq, levels = inspection_spectrogram(signal, rate, "high")
    renderer = SpectrogramRenderer()
    renderer.set_colormap("grayscale")
    renderer.set_inspection(dict(spec_times=times, spec_freq=freq, spec_db=levels,
                                 inspection_band="high", samplerate=rate))
    for width, height in [(512, len(freq)), (400, 80), (300, 40), (800, 50)]:
        ctx = SpecRenderCtx(0, width, height, 0, samples, samples, rate)
        renderer._build_spec_image(ctx)
        image = renderer._spec_image
        x = int(12.285 / 25 * width)
        y = int((freq[-1] - 30000) / (freq[-1] - freq[0]) * height)
        bright = max(image.pixelColor(i, j).red() for i in range(x - 2, x + 3)
                     for j in range(y - 2, y + 3))
        # -19 dBFS on the configured -80..0 dBFS gray ramp exceeds 194.
        assert bright >= 194, (width, height, bright)
        assert image.pixelColor(x - 10, y).red() == 0


@pytest.mark.parametrize("frames", [1, 960, 38400, 48000])
def test_waveform_worker_marker_and_rendered_envelope_use_the_available_rms_window(frames):
    rate, win = 96000, 38400
    signal = 0.5 * np.cos(2 * np.pi * 1000 * np.arange(frames) / rate)
    stereo = np.column_stack([signal, np.zeros(frames)])
    worker = WaveformLoadWorker(stereo, rate, win, compute_spectrogram=False)
    ready = []
    worker.finished.connect(ready.append)
    worker.run()
    result, = ready
    expected = np.sqrt(np.mean(signal ** 2) / 2)
    assert result["rms_max_db"] == pytest.approx(20 * np.log10(expected), abs=0.01)
    assert 0 <= result["rms_max_sample"] < frames
    assert result["rms_cumsums"][0][-1] == pytest.approx((signal ** 2).sum())

    renderer = WaveformRenderer()
    renderer.set_track_data(result["channels"], rms_cumsums=result["rms_cumsums"])
    renderer.set_rms_window(win)
    renderer._ensure_rms_max_computed()
    assert renderer._rms_max_db == pytest.approx(20 * np.log10(expected), abs=0.01)
    assert 0 <= renderer._rms_max_sample < frames
    ctx = WaveformRenderCtx(0, 123, 80, 0, 0, frames, 1., result["channels"], 2,
                            True, True, True, False, 1)
    renderer._build_rms_envelope(ctx)
    np.testing.assert_allclose(renderer._rms_combined, expected, atol=1e-6)


@pytest.mark.parametrize("compare", [False, True])
def test_unmeasurable_double_reaches_neither_preview_ready_nor_waveform_meters(compare):
    source = np.tile(np.finfo(np.float64).max * np.array([1., -1., -1., 1.]), 16)
    track = TrackContext("recording.wav", "", source, 48000, 1, len(source),
                         "64-bit float", "DOUBLE", len(source) / 48000)
    worker = WaveformLoadWorker(source, 48000, 128, preview_track=track if compare else None)
    ready, results, errors = [], [], []
    worker.source_ready.connect(lambda: ready.append(True))
    worker.finished.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert errors and "finite range" in errors[0]
    assert not ready and not results


@pytest.mark.parametrize(("start", "end", "width"), [(0, 48000, 600), (12000, 45600, 607)])
def test_rms_curve_and_marker_share_the_same_sample_axis(start, end, width):
    rate, window = 48000, 19200
    samples = np.zeros(rate)
    samples[24000:43200] = 0.5
    renderer = WaveformRenderer()
    renderer.set_track_data([samples], rms_window=window, rms_max_dirty=True)
    renderer._ensure_rms_max_computed()
    assert renderer._rms_max_sample == 33600
    ctx = WaveformRenderCtx(0, width, 400, 0, start, end, 1., [samples], 1,
                            True, False, False, False, 1)
    renderer._build_rms_envelope(ctx)
    marker_x = (33600 - start) * width // (end - start)
    assert np.argmax(renderer._rms_envelope[0]) == marker_x
    assert renderer._rms_combined[marker_x] == pytest.approx(0.5)
