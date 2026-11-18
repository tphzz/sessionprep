"""Sample-aligned A/B playback and calibrated frequency inspection."""
import numpy as np
import pytest

from sessionpreplib.audition import RepairAudition, inspection_spectrogram
from sessionpreplib.repairs import RepairRecipe, render_corrected


def test_audition_keeps_unclipped_unity_sources_and_applies_one_common_monitor_gain():
    rate = 48000
    time = np.arange(rate) / rate
    step = np.where(time < 0.5, -0.82, 0.82) + 0.07 * np.sin(2 * np.pi * 997 * time)
    source = np.column_stack([step, -0.7 * step])
    original = source.copy()
    source.setflags(write=False)
    recipe = RepairRecipe(dc_remove=True)
    expected = render_corrected(original, rate, recipe)
    assert np.max(np.abs(original)) < 1.0 < np.max(np.abs(expected))

    audition = RepairAudition(source, rate, recipe)

    np.testing.assert_array_equal(audition.original, original)
    np.testing.assert_allclose(audition.corrected, expected, atol=1e-9)
    assert audition.monitor_gain_db < 0.0
    gain = 10 ** (audition.monitor_gain_db / 20)
    for corrected, unity in [(False, original), (True, expected)]:
        playback = audition.playback_samples(corrected)
        np.testing.assert_allclose(playback, unity * gain, atol=1e-7)
        assert np.max(np.abs(playback)) <= 1.0 + 1e-7
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("amplitude", [1.0, 0.25])
@pytest.mark.parametrize("band, target", [("low", 12), ("high", 30000)])
def test_bin_centred_tone_reports_peak_dbfs_even_in_opposite_polarity_channels(amplitude, band, target):
    rate = 96000
    time = np.arange(3 * rate) / rate
    _, bins, _ = inspection_spectrogram(np.zeros_like(time), rate, band, max_columns=11)
    index = np.argmin(np.abs(bins - target))
    tone = amplitude * np.sin(2 * np.pi * bins[index] * time)
    stereo = np.column_stack([tone, -tone])

    times, frequencies, levels = inspection_spectrogram(stereo, rate, band, max_columns=11)

    np.testing.assert_array_equal(frequencies, bins)
    assert levels.shape == (len(frequencies), len(times))
    # Inspect the middle of the uninterrupted tone, away from file boundaries.
    column = np.argmin(np.abs(times - 1.5))
    assert levels[index, column] == pytest.approx(20 * np.log10(amplitude), abs=0.1)


@pytest.mark.parametrize("frames", [3 * 48000, 24000, 2400, 48, 1])
def test_dc_bin_reports_the_actual_bias_even_when_a_short_source_partly_fills_the_window(frames):
    rate = 48000
    source = np.tile([0.05, -0.05], (frames, 1))
    times, frequencies, levels = inspection_spectrogram(source, rate, "low", max_columns=9)

    assert frequencies[0] == 0
    middle = np.argmin(np.abs(times - frames / (2 * rate)))
    assert levels[0, middle] == pytest.approx(20 * np.log10(0.05), abs=0.05)


def test_overrange_original_also_sets_the_common_pad_when_corrected_peak_is_smaller():
    rate = 96000
    time = np.arange(rate) / rate
    fade = np.clip(np.minimum(time, 1.0 - time) / 0.03, 0, 1)
    source = fade * (1.3 * np.sin(2 * np.pi * 30000 * time) + 0.1 * np.sin(2 * np.pi * 1000 * time))
    audition = RepairAudition(source, rate, RepairRecipe(ultrasonic=True))

    assert np.max(np.abs(audition.corrected)) < 1.0 < np.max(np.abs(audition.original))
    gain = 10 ** (audition.monitor_gain_db / 20)
    assert np.max(np.abs(audition.original)) * gain <= 1.0 + 1e-7
    np.testing.assert_allclose(audition.playback_samples(False), audition.original * gain, atol=1e-7)
    np.testing.assert_allclose(audition.playback_samples(True), audition.corrected * gain, atol=1e-7)


@pytest.mark.parametrize("from_corrected", [False, True])
def test_switch_crossfades_ten_ms_at_the_same_nonzero_cursor_and_keeps_playback_timing(from_corrected):
    rate = 48000
    time = np.arange(rate) / rate
    step = np.where(time < 0.5, -0.8, 0.8)
    source = np.column_stack([step + 0.1 * np.sin(2 * np.pi * 997 * time),
                              -step + 0.04 * np.cos(2 * np.pi * 2000 * time)])
    audition = RepairAudition(source, rate, RepairRecipe(dc_remove=True))
    cursor = 3 * rate // 4 + 29
    length = round(rate * 0.010)
    frames = length + 61
    old = audition.playback_samples(from_corrected)[cursor:cursor + frames]
    new = audition.playback_samples(not from_corrected)[cursor:cursor + frames]
    assert np.max(np.abs(old - new)) > 0.25

    actual = audition.switch_samples(cursor, from_corrected, not from_corrected, frames)

    assert actual.shape == old.shape
    weight = np.linspace(0.0, 1.0, length)[:, None]
    expected = old[:length] * (1 - weight) + new[:length] * weight
    # Inclusive and exclusive ramp endpoints differ by one blend-weight increment.
    tolerance = np.max(np.abs(old - new)) / length + 1e-7
    np.testing.assert_allclose(actual[:length], expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(actual[length:], new[length:], atol=1e-7)


def test_switch_near_end_returns_the_remaining_frames_of_the_original_ten_ms_ramp():
    rate = 48000
    source = np.full(rate, 0.5)
    pair = RepairAudition(source, rate, RepairRecipe(dc_remove=True))
    cursor = len(source) - 64
    old = pair.playback_samples(False)[cursor:]
    new = pair.playback_samples(True)[cursor:]
    fade_frames = round(0.010 * rate)
    weight = np.arange(len(old)) / (fade_frames - 1)

    switched = pair.switch_samples(cursor, False, True, 1024)

    assert switched.shape == (64,)
    np.testing.assert_allclose(switched, old * (1 - weight) + new * weight,
                               atol=np.max(np.abs(old - new)) / fade_frames + 1e-7)


def test_switch_spans_multiple_audio_callbacks_without_restarting_the_ramp():
    source = np.full(48000, 0.25)
    pair = RepairAudition(source, 48000, RepairRecipe(dc_remove=True))
    cursor = 16000
    expected = pair.switch_samples(cursor, False, True, 640)
    blocks = [pair.switch_samples(cursor + offset, False, True, 128, fade_offset=offset)
              for offset in range(0, 640, 128)]
    np.testing.assert_array_equal(np.concatenate(blocks), expected)


@pytest.mark.parametrize("edge", [0, -1])
def test_file_edge_impulses_are_visible_in_frequency_inspection(edge):
    source = np.zeros(48001)
    source[edge] = 0.5
    _, _, levels = inspection_spectrogram(source, 48000, "low")
    assert levels.max() > -110


@pytest.mark.parametrize("rate, band, minimum, maximum", [
    (48000, "low", 0, 100), (44100, "high", 20000, 22050), (96000, "high", 20000, 48000),
])
def test_frequency_axes_cover_the_native_inspection_range(rate, band, minimum, maximum):
    times, frequencies, levels = inspection_spectrogram(np.zeros(rate), rate, band)

    assert levels.shape == (len(frequencies), len(times))
    spacing = np.diff(frequencies)
    assert len(spacing) and np.all(spacing > 0)
    assert minimum <= frequencies[0] <= minimum + spacing[0] + 1e-8
    assert maximum - spacing[-1] - 1e-8 <= frequencies[-1] <= maximum + 1e-8


@pytest.mark.parametrize("rate", [32000, 40000])
def test_high_frequency_inspection_rejects_an_empty_frequency_range(rate):
    with pytest.raises(ValueError):
        inspection_spectrogram(np.zeros(rate), rate, "high")


def test_capped_time_columns_cover_early_and_late_audio_and_repeat_deterministically():
    rate = 96000
    time = np.arange(6 * rate) / rate
    early = (0.5 < time) & (time < 2.0)
    late = (4.0 < time) & (time < 5.5)
    signal = 0.5 * (early * np.sin(2 * np.pi * 27000 * time) + late * np.sin(2 * np.pi * 36000 * time))
    times, frequencies, levels = inspection_spectrogram(signal, rate, "high", max_columns=12)

    assert 2 <= len(times) <= 12
    assert np.all(np.diff(times) > 0)
    first = levels[np.argmin(np.abs(frequencies - 27000))]
    last = levels[np.argmin(np.abs(frequencies - 36000))]
    assert 0.3 < times[np.argmax(first)] < 2.2
    assert 3.8 < times[np.argmax(last)] < 5.7
    repeated = inspection_spectrogram(signal, rate, "high", max_columns=12)
    for original, again in zip((times, frequencies, levels), repeated):
        np.testing.assert_array_equal(original, again)


def test_capped_columns_retain_a_short_carrier_present_on_only_one_channel():
    rate = 96000
    time = np.arange(4 * rate) / rate
    source = np.zeros((len(time), 2))
    burst = (2.16 < time) & (time < 2.23)
    source[burst, 1] = 0.5 * np.sin(2 * np.pi * 30000 * time[burst])

    times, frequencies, levels = inspection_spectrogram(source, rate, "high", max_columns=4)

    carrier = levels[np.argmin(np.abs(frequencies - 30000))]
    assert len(times) <= 4
    assert np.max(carrier) == pytest.approx(20 * np.log10(0.5), abs=0.1)
    assert abs(times[np.argmax(carrier)] - 2.2) < 0.6


@pytest.mark.parametrize("band, frequency, minimum_reduction", [("low", 8, 14), ("high", 30000, 70)])
def test_spectrogram_pair_shares_axes_and_shows_measured_reduction(band, frequency, minimum_reduction):
    rate = 96000
    time = np.arange(3 * rate) / rate
    signal = (0.2 * np.sin(2 * np.pi * 8 * time) + 0.3 * np.sin(2 * np.pi * 440 * time)
              + 0.25 * np.sin(2 * np.pi * 30000 * time))
    pair = RepairAudition(signal, rate, RepairRecipe(low_cut_hz=20, ultrasonic=True))
    times, frequencies, before = inspection_spectrogram(pair.original, rate, band, max_columns=12)
    after_times, after_freqs, after = inspection_spectrogram(pair.corrected, rate, band, max_columns=12)

    np.testing.assert_array_equal(times, after_times)
    np.testing.assert_array_equal(frequencies, after_freqs)
    column = np.argmin(np.abs(times - 1.5))
    row = np.argmin(np.abs(frequencies - frequency))
    assert before[row, column] - after[row, column] >= minimum_reduction


@pytest.mark.parametrize("frames", [0, 1, 17])
@pytest.mark.parametrize("band", ["low", "high"])
def test_short_and_empty_silence_have_valid_spectrum_dimensions_and_unity_samples(frames, band):
    source = np.zeros((frames, 2))
    pair = RepairAudition(source, 96000, RepairRecipe())
    times, frequencies, levels = inspection_spectrogram(source, 96000, band, max_columns=2)

    assert len(times) <= 2
    assert levels.shape == (len(frequencies), len(times))
    assert np.isfinite(times).all() and np.isfinite(frequencies).all()
    assert not np.isnan(levels).any()
    assert np.all(levels < -60)
    assert np.isfinite(pair.monitor_gain_db) and pair.monitor_gain_db <= 0
    for corrected in [False, True]:
        np.testing.assert_array_equal(pair.playback_samples(corrected), source)
