"""Signal response and boundary contracts for rendering selected audio repairs at unity gain."""
import numpy as np
import pytest

from sessionpreplib.repairs import RepairRecipe, render_corrected


def tone_peak(samples, frequency, sample_rate):
    """Read a bin-centred sinusoid's peak amplitude independently of its phase."""
    index = round(frequency * len(samples) / sample_rate)
    return 2 * abs(np.fft.rfft(samples)[index]) / len(samples)


def test_default_recipe_preserves_asymmetric_music_and_over_range_samples():
    sample_rate = 48000
    time = np.arange(7183) / sample_rate
    kick = np.exp(-time * 18) * np.cos(2 * np.pi * (42 * time + 8 * (1 - np.exp(-time * 12))))
    bass = np.sin(2 * np.pi * 55 * time) + 0.4 * np.cos(2 * np.pi * 110 * time)
    source = np.column_stack([1.4 * kick + 0.03, 0.24 * bass - 0.05])
    original = source.copy()
    source.setflags(write=False)

    corrected = render_corrected(source, sample_rate, RepairRecipe())

    assert isinstance(corrected, np.ndarray)
    assert np.max(np.abs(original)) > 1.0
    np.testing.assert_array_equal(corrected, original)
    np.testing.assert_array_equal(source, original)


def test_dc_blocker_rejects_signed_bias_without_cutting_musical_bass_or_limiting_peaks():
    sample_rate = 48000
    time = np.arange(3 * sample_rate) / sample_rate
    program = (0.12 * np.sin(2 * np.pi * 2 * time)
               + 0.25 * np.sin(2 * np.pi * 20 * time)
               + 1.11 * np.sin(2 * np.pi * 1000 * time))
    source = np.column_stack([program + 0.06, -program - 0.09])
    original = source.copy()

    corrected = render_corrected(source, sample_rate, RepairRecipe(dc_remove=True))
    settled = corrected[-sample_rate:]

    np.testing.assert_allclose(settled.mean(axis=0), 0.0, atol=1e-7)
    for channel in settled.T:
        for frequency, amplitude, gain_db in [(2, 0.12, -3.0103), (20, 0.25, -0.0432), (1000, 1.11, 0.0)]:
            actual_db = 20 * np.log10(tone_peak(channel, frequency, sample_rate) / amplitude)
            assert actual_db == pytest.approx(gain_db, abs=0.01)
    assert np.max(np.abs(settled)) > 1.1
    np.testing.assert_array_equal(source, original)


def test_dc_blocker_keeps_leading_silence_and_uses_zero_prior_history():
    sample_rate = 48000
    start = 7500
    source = np.zeros((sample_rate, 2))
    source[start:] = [0.05, -0.075]

    corrected = render_corrected(source, sample_rate, RepairRecipe(dc_remove=True))
    starting_on_bias = render_corrected(source[start:], sample_rate, RepairRecipe(dc_remove=True))

    np.testing.assert_array_equal(corrected[:start], 0.0)
    np.testing.assert_allclose(starting_on_bias, corrected[start:], atol=1e-10)
    np.testing.assert_allclose(corrected[start], source[start], rtol=2e-4)
    assert np.max(np.abs(corrected[-1])) < 2e-6


@pytest.mark.parametrize("cutoff", [5.0, 20.0, 40.0])
def test_low_cut_has_second_order_butterworth_response(cutoff):
    sample_rate = 48000
    time = np.arange(3 * sample_rate) / sample_rate
    frequencies = [cutoff / 2, cutoff, cutoff * 4]
    source = sum(0.2 * np.sin(2 * np.pi * freq * time) for freq in frequencies)
    corrected = render_corrected(source, sample_rate, RepairRecipe(low_cut_hz=cutoff))
    settled = corrected[-2 * sample_rate:]

    for frequency, gain_db in zip(frequencies, [-12.3045, -3.0103, -0.0169]):
        actual_db = 20 * np.log10(tone_peak(settled, frequency, sample_rate) / 0.2)
        assert actual_db == pytest.approx(gain_db, abs=0.03)


def test_dc_and_low_cut_share_one_filter_and_each_channel_has_independent_history():
    sample_rate = 48000
    time = np.arange(sample_rate) / sample_rate
    bass = 0.4 * np.sin(2 * np.pi * 31 * time) * np.exp(-3 * time)
    source = np.column_stack([bass + 0.03, -0.6 * bass, np.zeros_like(bass)])
    low_cut = RepairRecipe(low_cut_hz=12.5)
    combined = RepairRecipe(dc_remove=True, low_cut_hz=12.5)

    corrected = render_corrected(source, sample_rate, combined)

    for index, channel in enumerate(source.T):
        expected = render_corrected(channel, sample_rate, low_cut)
        np.testing.assert_allclose(corrected[:, index], expected, atol=1e-9, rtol=1e-7)
    np.testing.assert_array_equal(corrected[:, 2], 0.0)


@pytest.mark.parametrize("sample_rate", [50000, 88200, 96000, 192000])
def test_ultrasonic_filter_meets_native_rate_passband_and_stopband(sample_rate):
    # A half-second FFT puts both specified band edges exactly on measured bins.
    source = np.zeros(sample_rate // 2)
    source[len(source) // 2] = 1.0
    corrected = render_corrected(source, sample_rate, RepairRecipe(ultrasonic=True))

    assert corrected.shape == source.shape
    frequencies = np.fft.rfftfreq(len(source), d=1 / sample_rate)
    magnitude = np.abs(np.fft.rfft(corrected))
    passband_db = 20 * np.log10(magnitude[frequencies <= 20000])
    np.testing.assert_allclose(passband_db, 0.0, atol=0.01)
    assert np.max(magnitude[frequencies >= 24000]) <= 1e-4


def test_ultrasonic_filter_centres_each_channels_impulse_on_its_original_sample():
    sample_rate = 96000
    source = np.zeros((6144, 2))
    positions = [1949, 3473]
    source[positions[0], 0] = 1.0
    source[positions[1], 1] = -0.5

    corrected = render_corrected(source, sample_rate, RepairRecipe(ultrasonic=True))

    np.testing.assert_array_equal(np.argmax(np.abs(corrected), axis=0), positions)
    for channel, position in enumerate(positions):
        radius = min(position, len(source) - position - 1)
        impulse = corrected[position - radius:position + radius + 1, channel]
        np.testing.assert_allclose(impulse, impulse[::-1], atol=1e-8)


def test_all_repairs_preserve_a_bass_note_with_opposite_polarity_rumble_and_ultrasound():
    sample_rate = 96000
    time = np.arange(3 * sample_rate) / sample_rate
    envelope = np.clip(np.minimum(time, 3.0 - time) / 0.04, 0.0, 1.0)
    bass = envelope * (0.43 * np.sin(2 * np.pi * 55 * time)
                       + 0.14 * np.cos(2 * np.pi * 110 * time))
    noise = 0.21 * np.sin(2 * np.pi * 8 * time) + 0.18 * np.sin(2 * np.pi * 30000 * time)
    source = np.column_stack([bass + noise + 0.06, bass - noise - 0.04])
    source.setflags(write=False)

    corrected = render_corrected(source, sample_rate, RepairRecipe(
        dc_remove=True, low_cut_hz=20, ultrasonic=True))

    for channel in corrected[sample_rate:2 * sample_rate].T:
        assert abs(channel.mean()) < 1e-7
        assert tone_peak(channel, 8, sample_rate) < 0.035
        assert tone_peak(channel, 55, sample_rate) == pytest.approx(0.43, rel=0.02)
        assert tone_peak(channel, 30000, sample_rate) <= 0.18e-4


@pytest.mark.parametrize("length", [1, 11])
def test_short_ultrasonic_render_matches_an_excerpt_surrounded_by_digital_silence(length):
    sample_rate = 96000
    source = np.linspace(-0.7, 0.2, length)
    margin = 4096
    recipe = RepairRecipe(ultrasonic=True)
    expected = render_corrected(np.pad(source, margin), sample_rate, recipe)

    corrected = render_corrected(source, sample_rate, recipe)

    assert corrected.shape == (length,)
    np.testing.assert_allclose(corrected, expected[margin:margin + length], atol=1e-8)


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000, 49999])
def test_unavailable_ultrasonic_band_rejects_explicit_repair(sample_rate):
    with pytest.raises(ValueError):
        render_corrected(np.zeros(64), sample_rate, RepairRecipe(ultrasonic=True))


@pytest.mark.parametrize("shape", [(0,), (0, 2), (1,), (1, 1), (15, 4)])
@pytest.mark.parametrize("recipe", [
    RepairRecipe(), RepairRecipe(dc_remove=True),
    RepairRecipe(dc_remove=True, low_cut_hz=10, ultrasonic=True),
])
def test_silent_short_files_keep_their_native_frame_and_channel_shape(shape, recipe):
    source = np.zeros(shape, dtype=np.float32)

    corrected = render_corrected(source, 96000, recipe)

    assert isinstance(corrected, np.ndarray)
    np.testing.assert_array_equal(corrected, source)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("recipe", [RepairRecipe(), RepairRecipe(dc_remove=True)])
def test_nonfinite_channel_samples_fail_even_when_others_are_valid(invalid, recipe):
    source = np.zeros((12, 2))
    source[-1, 1] = invalid
    with pytest.raises(ValueError):
        render_corrected(source, 48000, recipe)


@pytest.mark.parametrize("sample_rate", [0, -48000, np.nan, np.inf])
def test_invalid_sample_rate_fails_even_in_bypass(sample_rate):
    with pytest.raises(ValueError):
        render_corrected(np.zeros(8), sample_rate, RepairRecipe())


@pytest.mark.parametrize("cutoff", [0.0, 4.99, 40.01, np.nan, np.inf])
def test_invalid_low_cut_cannot_be_applied(cutoff):
    with pytest.raises(ValueError):
        render_corrected(np.zeros(8), 48000, RepairRecipe(low_cut_hz=cutoff))


@pytest.mark.parametrize(("recipe", "seconds"), [
    (RepairRecipe(dc_remove=True), 75), (RepairRecipe(low_cut_hz=40), 8),
])
def test_highpass_settles_to_zero_without_subnormal_samples(recipe, seconds):
    rate = 48000
    source = np.zeros((seconds * rate, 2))
    source[0, 0] = 1.0
    source[-2:, 1] = [1e-12, -1e-12]
    output = render_corrected(source, rate, recipe)
    assert output[0, 0] > 0.9
    assert not ((np.abs(output) < np.finfo(output.dtype).tiny) & (output != 0)).any()
    assert not output[-rate:, 0].any()
    assert not output[:-2, 1].any()
    assert abs(output[-2, 1]) > 0.9e-12


def test_corrector_surfaces_nonfinite_filter_output_from_finite_double():
    maximum = np.finfo(np.float64).max
    source = np.tile([maximum, -maximum, -maximum, maximum], 16)
    with pytest.raises(ValueError, match="finite|range"):
        render_corrected(source, 48000, RepairRecipe(dc_remove=True))
