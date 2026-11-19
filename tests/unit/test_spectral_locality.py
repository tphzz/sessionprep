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
