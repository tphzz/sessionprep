"""Calibrate the real-input spectral endpoints and the support of each readout."""
import numpy as np
import pytest

from sessionpreplib.spectrum import full_spectrogram, inspection_spectrogram


@pytest.mark.parametrize("phase", [0.12, 1.43, 2.81])
@pytest.mark.parametrize(("band", "offset"), [("low", .125), ("low", .375),
                                               ("high", .125), ("high", 2.5)])
def test_edge_tone_peak_is_stationary_at_fractional_cycles_and_record_lengths(band, offset, phase):
    rate = 48000
    time = np.arange(6 * rate + 10193) / rate
    frequency = offset if band == "low" else rate / 2 - offset
    source = .5 * np.cos(2 * np.pi * frequency * time + phase)
    times, _, levels = inspection_spectrogram(source, rate, band, max_columns=2000)
    interior = (times >= .6) & (times <= time[-1] - .6)
    np.testing.assert_allclose(levels[:, interior].max(axis=0), 20 * np.log10(.5), atol=.3)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackmanharris"])
@pytest.mark.parametrize("width", [512, 1024, 2048, 4096, 8192])
def test_full_edge_tones_across_fft_and_window_choices(width, window):
    rate = 192000
    time = np.arange(309731) / rate
    for f, phase in [(20.125, .75), (rate / 2 - .375, 1.47)]:
        source = np.cos(2 * np.pi * f * time + phase)
        times, _, levels = full_spectrogram([source], rate, width=width, window=window,
                                            frequency_max=rate / 2, rows=256)
        interior = (times >= .2) & (times <= time[-1] - .2)
        np.testing.assert_allclose(levels[:, interior].max(axis=0), 0., atol=.3)


@pytest.mark.parametrize("band", ["low", "high", "full"])
@pytest.mark.parametrize("background", [0., 1e-5])
def test_quiet_windows_ignore_the_signal_in_an_unrelated_edit(band, background):
    rate = 48000
    t = np.arange(5 * rate + 1417) / rate
    frequency = 31.125 if band != "high" else 23572.
    source = background * np.cos(2 * np.pi * frequency * t + .23)
    def spec(x):
        if band == "full":
            return full_spectrogram([x], rate, width=2048, window="hann",
                                    frequency_max=rate / 2, rows=256)
        return inspection_spectrogram(x, rate, band)
    times, _, quiet = spec(source)
    burst = np.arange(rate // 4)
    source[3 * rate + burst] += .8 * np.sin(2 * np.pi * frequency * burst / rate)
    _, frequencies, edited = spec(source)
    outside = (times < 2.3) | (times > 4.05)
    np.testing.assert_array_equal(quiet[:, outside], edited[:, outside])
    if not background:
        assert (edited[:, outside] == -200).all()
        pytest.importorskip("PySide6")
        from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx
        renderer = SpectrogramRenderer()
        renderer.set_colormap("grayscale")
        renderer.set_inspection(dict(spec_times=times, spec_freq=frequencies, spec_db=edited,
                                     inspection_band=band, samplerate=rate))
        renderer._build_spec_image(SpecRenderCtx(0, 100, 40, rate, 2 * rate, len(source), rate))
        image = renderer._spec_image
        assert all(image.pixelColor(x, y).red() == 0 for x in range(100) for y in range(40))


@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
@pytest.mark.parametrize("rate,offset,band", [(48000, .125, "high"), (192000, .375, "full")])
def test_decoded_stationary_edge_tone_has_one_level_through_beat_nulls(tmp_path, subtype, rate, offset, band):
    import soundfile as sf
    time = np.arange(3 * rate + 479) / rate
    wave = .71 * np.cos(2 * np.pi * (rate / 2 - offset) * time + 1.0)
    path = tmp_path / "edge.wav"
    sf.write(path, wave, rate, subtype=subtype)
    decoded, _ = sf.read(path)
    if band == "high":
        times, _, db = inspection_spectrogram(decoded, rate, band, max_columns=2000)
    else:
        times, _, db = full_spectrogram([decoded], rate, width=512, window="hamming",
                                        frequency_max=rate / 2, rows=256)
    interior = (times > .1) & (times < time[-1] - .1)
    np.testing.assert_allclose(db[:, interior].max(axis=0), 20 * np.log10(.71), atol=.3)


@pytest.mark.parametrize("high", [False, True])
def test_gently_curved_edit_ramp_cannot_extrapolate_a_huge_tone(high):
    rate = 48000
    t = np.linspace(-1., 1., 2 * rate)
    # A cubic ramp can agree locally with a tiny fragment of a huge sine.
    # It never produces that extrapolated level anywhere in this recording.
    source = .5 * (t - 1e-6 * t ** 3)
    if high:
        source[1::2] *= -1
    _, _, levels = inspection_spectrogram(source, rate, "high" if high else "low")
    assert levels.max() <= 20 * np.log10(1.)


def test_midrange_tones_do_not_create_a_false_ultrasonic_line_at_frame_seams():
    rate = 48000
    t = np.arange(2 * rate + 41) / rate
    source = .61 * np.cos(2 * np.pi * 533.25 * t) + .20 * np.sin(2 * np.pi * 3984.37 * t)
    times, _, levels = inspection_spectrogram(source, rate, "high")
    interior = (times > .05) & (times < 1.95)
    # The real, local 20 ms Hann spectrum's sidelobes in 20–24 kHz are
    # below -145 dBFS. A periodic analytic extension adds spurious energy.
    assert levels[:, interior].max() < -140


@pytest.mark.parametrize("codec,db", [("PCM_U8", -12), ("PCM_16", -6), ("PCM_16", -18),
                                       ("PCM_16", -24), ("PCM_24", -20), ("PCM_24", -24),
                                       ("FLOAT", -24)])
@pytest.mark.parametrize("rate,frequency,band,seconds", [
    (48000, 41.2034446, "full", 1.6), (192000, 41.2034446, "full", 1.6),
    (48000, .25, "low", 8), (48000, 23999.875, "high", 8.37),
    (96000, 47999.75, "high", 6.15),
])
def test_decoded_edge_levels_match_measured_line_through_nulls(
        tmp_path, codec, db, rate, frequency, band, seconds):
    import soundfile as sf
    phase = 2 * np.pi * frequency * (np.arange(round(seconds * rate)) / rate) + .923
    path = tmp_path / "stationary.wav"
    sf.write(path, 10 ** (db / 20) * np.cos(phase), rate, subtype=codec)
    decoded, _ = sf.read(path)
    # Measure the recorded line independently; eight-bit files can carry
    # a measurable DC bias and amplitude quantization error.
    basis = np.column_stack((np.cos(phase), np.sin(phase), np.ones(len(phase))))
    a, b, _ = np.linalg.lstsq(basis, decoded, rcond=None)[0]
    expected = 20 * np.log10(np.hypot(a, b))
    if band == "full":
        times, _, levels = full_spectrogram([decoded], rate, width=512, window="hann",
                                            frequency_max=rate / 2, rows=128)
    else:
        times, _, levels = inspection_spectrogram(decoded, rate, band, max_columns=4096)
    settled = (times >= .2 * seconds) & (times <= .8 * seconds)
    np.testing.assert_allclose(levels[:, settled].max(axis=0), expected, atol=.3)


@pytest.mark.parametrize("codec", ["PCM_U8", "PCM_16", "PCM_24", "FLOAT"])
def test_decoded_stationary_sections_remeasure_the_local_level_across_edits(tmp_path, codec):
    import soundfile as sf
    rate = 48000
    t = np.arange(12 * rate) / rate
    carrier = np.cos(2 * np.pi * 23999.75 * t + .13)
    level = np.select([(t >= 1) & (t < 6), (t >= 6) & (t < 11)], [.6, .15])
    path = tmp_path / "edited.wav"
    sf.write(path, level * carrier, rate, subtype=codec)
    decoded, _ = sf.read(path)
    times, _, spectrum = inspection_spectrogram(decoded, rate, "high", max_columns=2000)
    peaks = spectrum.max(axis=0)
    assert (peaks[(times < .98) | (times > 11.02)] == -200).all()
    for start, end in [(1, 6), (6, 11)]:
        samples = (t >= start) & (t < end)
        amplitude = np.dot(carrier[samples], decoded[samples]) / np.dot(carrier[samples], carrier[samples])
        interior = (times > start + .03) & (times < end - .03)
        np.testing.assert_allclose(peaks[interior], 20 * np.log10(amplitude), atol=.3)


@pytest.mark.parametrize("codec,amplitude,step", [("PCM_U8", .3, 2. ** -7),
                                                ("PCM_16", .063, 2. ** -15)])
@pytest.mark.parametrize("frequency", [41.2, 23999.75])
def test_dithered_pcm_line_survives_a_common_floating_point_gain(
        tmp_path, codec, amplitude, step, frequency):
    import soundfile as sf
    rate = 48000
    t = np.arange(4 * rate) / rate
    rng = np.random.default_rng(420)
    dither = step * (rng.random(len(t)) - rng.random(len(t)))
    carrier = np.cos(2 * np.pi * frequency * t + .456)
    path = tmp_path / "dithered.wav"
    sf.write(path, amplitude * carrier + dither, rate, subtype=codec)
    decoded, _ = sf.read(path)
    for gain in (1., .37):
        source = decoded * gain
        expected = 20 * np.log10(np.dot(source, carrier) / np.dot(carrier, carrier))
        if frequency < 50:
            times, _, levels = full_spectrogram([source], rate, width=512, window="hann",
                                                frequency_max=rate / 2, rows=128)
        else:
            times, _, levels = inspection_spectrogram(source, rate, "high")
        settled = (times > .5) & (times < 3.5)
        np.testing.assert_allclose(levels[:, settled].max(axis=0), expected, atol=.3)
