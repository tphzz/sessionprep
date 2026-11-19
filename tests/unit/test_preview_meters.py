"""Calibration at the waveform preview's numerical and painted outputs."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from sessionprepgui.waveform.compute import WaveformLoadWorker, compute_mel_spectrogram
from sessionprepgui.waveform.renderer import WaveformRenderer, WaveformRenderCtx
from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx
from sessionpreplib.audition import inspection_spectrogram
from sessionpreplib.models import TrackContext
from sessionpreplib.spectrum import full_spectrogram


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


@pytest.mark.parametrize("rate", [40001, 40026, 40049, 47952, 48048, 96096])
def test_high_worker_measures_both_physical_band_edges_at_pulled_rates(rate):
    t = np.arange(2 * rate) / rate
    for frequency in (20000., rate / 2 - min(4., (rate / 2 - 20000) / 2)):
        samples = .25 * np.cos(2 * np.pi * frequency * t + .33)
        worker = WaveformLoadWorker(samples, rate, 256, band="high")
        results, errors = [], []
        worker.finished.connect(results.append)
        worker.failed.connect(errors.append)
        worker.run()
        assert not errors
        measured, = results
        frequency_rows = measured["spec_freq"]
        assert frequency_rows[0] == 20000. and frequency_rows[-1] == rate / 2
        assert np.all(np.diff(frequency_rows) > 0)
        interior = (measured["spec_times"] > .4) & (measured["spec_times"] < 1.6)
        np.testing.assert_allclose(measured["spec_db"][:, interior].max(axis=0),
                                   20 * np.log10(.25), atol=.3)


def test_full_measures_the_lower_cutoff_when_it_lies_below_the_first_fft_bin():
    rate = 384000
    t = np.arange(rate) / rate
    samples = .25 * np.cos(2 * np.pi * 20 * t + .42)
    times, _, levels = full_spectrogram([samples], rate, width=512, window="hamming",
                                        frequency_max=rate / 2, rows=256)
    interior = (times > .1) & (times < .9)
    np.testing.assert_allclose(levels[0, interior], 20 * np.log10(.25), atol=.3)


def test_high_painter_uses_the_actual_frequency_of_the_partial_edge_rows():
    renderer = SpectrogramRenderer()
    renderer.set_colormap("grayscale")
    renderer.set_inspection(dict(spec_times=np.array([.5]), samplerate=40200,
                                 spec_freq=np.array([20000., 20001., 20051., 20100.]),
                                 spec_db=np.array([[-200.], [0.], [-200.], [-200.]]),
                                 inspection_band="high"))
    renderer._build_spec_image(SpecRenderCtx(0, 8, 100, 0, 40200, 40200, 40200))
    # The second cell extends to the midpoint at 20,026 Hz.
    assert renderer._spec_image.pixelColor(4, 89).red() == 255  # 20,010 Hz
    assert renderer._spec_image.pixelColor(4, 54).red() == 0    # 20,045 Hz


@pytest.mark.parametrize("band", ["low", "high"])
@pytest.mark.parametrize(("companion", "bins"), [
    ("distant tone", None), ("resolved tone", 4), ("resolved tone", 8), ("noise", None),
])
def test_exact_dc_and_nyquist_keep_their_one_sided_level_with_other_audio(band, companion, bins):
    rate = 48000
    index = np.arange(8 * rate)
    endpoint = np.ones(len(index))
    if band == "high":
        endpoint[1::2] = -1
    source = .05 * endpoint
    if companion == "distant tone":
        source += .24 * np.sin(2 * np.pi * (1000 / rate) * index)
    elif companion == "resolved tone":
        frequency = bins if band == "low" else rate / 2 - 50 * bins
        source += .24 * np.sin(2 * np.pi * (frequency / rate) * index + .48)
    else:
        source += np.random.default_rng(1107).normal(0., 2e-5, len(index))
    source = source.astype(np.float32)
    result, errors = [], []
    worker = WaveformLoadWorker(source, rate, 256, band=band)
    worker.finished.connect(result.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    view, = result
    # A mean measures exactly DC. Alternating the observation signs measures
    # exactly Nyquist. Neither endpoint has a separate negative-frequency image.
    measured = 20 * np.log10(abs(np.mean((source * endpoint)[2 * rate:6 * rate])))
    columns = np.flatnonzero((view["spec_times"] >= 2.) & (view["spec_times"] < 6.))
    edge_row = 0 if band == "low" else -1
    levels = view["spec_db"][edge_row, columns]
    np.testing.assert_allclose(levels, measured, atol=.3)
    expected_peak = 20 * np.log10(.24) if companion == "resolved tone" else measured
    np.testing.assert_allclose(view["spec_db"][:, columns].max(axis=0), expected_peak, atol=.3)
    renderer = SpectrogramRenderer()
    renderer.set_colormap("grayscale")
    renderer.set_inspection(view)
    center = round(view["spec_times"][columns[np.argmax(levels)]] * rate)
    renderer._build_spec_image(SpecRenderCtx(0, 6, 128, center - 1, center + 1, len(index), rate))
    pixel = renderer._spec_image.pixelColor(3, 127 if band == "low" else 0).red()
    assert abs(pixel - 255 * (measured + 80.) / 80.) <= 2
