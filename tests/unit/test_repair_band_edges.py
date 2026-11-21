"""Band decisions through the track planner, calibrated against known source RMS."""
import io

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.models import TrackContext
from sessionpreplib.repair_plan import plan_repairs
from sessionpreplib.repairs import band_evidence


def tone(rate, hz, rms_db, seconds):
    time = np.arange(round(seconds * rate)) / rate
    return np.sqrt(2.0) * 10 ** (rms_db / 20) * np.cos(2 * np.pi * hz * time + 0.17)


def track_for(samples, rate):
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    return TrackContext("edge.wav", "edge.wav", samples, rate, channels, len(samples),
                        "64-bit Float", "DOUBLE", len(samples) / rate)


@pytest.mark.parametrize(("hz", "inside"), [
    (15, True), (19.4, True), (20, True), (20.6, False), (21, False), (22, False),
])
def test_infrasonic_selection_protects_bass_above_the_inclusive_edge(hz, inside):
    rate = 48000
    track = track_for(tone(rate, hz, -9.03, 2), rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "repair_low_cut_hz": 40})
    evidence = track.repair_measurements["original"]["low_band"]
    assert track.repair_measurements["suggested"]["low_cut"] is inside
    assert track.repair_recipe.low_cut_hz == (40 if inside else None)
    if inside:
        assert evidence["band_dbfs"] == pytest.approx(-9.03, abs=0.3)
        assert evidence["regions"] == [[0, len(track.audio_data) - 1, 0]]
    else:
        assert evidence["band_dbfs"] < -50


@pytest.mark.parametrize(("hz", "rms_db", "seconds", "selected"), [
    (24090, -9.03, 0.30, False), (24096, -9.03, 0.30, False),
    (24100, -59.9, 0.30, True), (24100, -60.1, 0.30, False),
    (24104, -59.9, 0.30, True), (24110, -59.9, 0.30, True),
    (30000, -59.9, 0.299, False),
])
def test_ultrasonic_cutoff_level_and_persistence_are_independent(hz, rms_db, seconds, selected):
    rate = 96000
    track = track_for(tone(rate, hz, rms_db, seconds), rate)
    plan_repairs(track, {"repair_auto_ultrasonic": True})
    evidence = track.detector_results["ultrasonic"].data
    assert track.repair_recipe.ultrasonic is selected
    if selected:
        assert evidence["regions"] == [[0, len(track.audio_data) - 1, 0]]
    else:
        assert evidence["regions"] == []
    if hz >= 24100:
        assert evidence["band_dbfs"] == pytest.approx(rms_db, abs=0.1)
    else:
        assert evidence["band_dbfs"] < -60


@pytest.mark.parametrize("enable_detector", [True, False])
def test_short_manual_ultrasonic_repair_measures_both_sides_without_selection(enable_detector):
    rate = 96000
    source = tone(rate, 30000, -9.0309, 0.05)
    track = track_for(source, rate)
    track.repair_overrides = {"ultrasonic": True}
    plan_repairs(track, {"ultrasonic_enabled": enable_detector})
    measures = track.repair_measurements
    before = measures["original"]["ultrasonic"]
    after = measures["corrected"]["ultrasonic"]
    assert before["regions"] == after["regions"] == []
    assert not measures["suggested"]["ultrasonic"]
    assert before["band_dbfs"] == pytest.approx(-9.0309, abs=0.02)
    assert np.isfinite(after["band_dbfs"]) and after["band_dbfs"] < before["band_dbfs"] - 80
    assert track.repair_recipe.ultrasonic


def test_each_channel_measures_only_its_part_of_a_resolvable_musical_mix():
    rate = 48000
    program = tone(rate, 20.6, -9, 2) + tone(rate, 82.41, -12, 2)
    rumble = tone(rate, 13.4, -21, 2)
    track = track_for(np.column_stack((program, -program + rumble)), rate)
    plan_repairs(track, {"repair_auto_low_cut": True})
    evidence = track.repair_measurements["original"]["low_band"]
    assert evidence["regions"] == [[0, len(program) - 1, 1]]
    assert evidence["band_dbfs"] == pytest.approx(-21, abs=0.3)
    expected_ratio = -21 - 10 * np.log10(sum(10 ** (db / 10) for db in (-9, -12, -21)))
    assert evidence["ratio_db"] == pytest.approx(expected_ratio, abs=0.3)


def test_unresolved_beating_just_above_the_low_edge_remains_advisory():
    rate = 48000
    t = np.arange(3 * rate) / rate
    source = (0.5 * np.cos(2 * np.pi * 20.001 * t + 0.17) +
              0.25 * np.cos(2 * np.pi * 20.06 * t + 2.45))
    track = track_for(source, rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "repair_low_cut_hz": 40})
    assert not track.repair_measurements["suggested"]["low_cut"]
    assert track.repair_recipe.low_cut_hz is None


@pytest.mark.parametrize(("rate", "low", "high", "hz", "seconds", "consecutive"), [
    (48000, 0, 19.3, 19.3, 1.0, 3),
    (50001, 24100, 25000.5, 24100, 0.1, 5),
])
def test_an_inclusive_edge_can_fall_between_native_bins(rate, low, high, hz, seconds, consecutive):
    source = tone(rate, hz, -59.9, seconds * (1 + consecutive))
    measured = band_evidence(source, rate, low_hz=low, high_hz=high, window_sec=seconds,
                             level_db=-60, ratio_db=-35, consecutive=consecutive)
    assert measured["band_dbfs"] == pytest.approx(-59.9, abs=0.1)
    assert len(measured["regions"]) == 1 and measured["regions"][0][0] == 0


@pytest.mark.parametrize(("rate", "seconds", "low", "high", "noise_band", "pink"), [
    (48000, 1.0, 0, 20, (5, 14), True),
    (96000, 0.1, 24100, 48000, (26000, 37000), False),
])
def test_band_noise_levels_match_isolated_reference_in_a_mixture(rate, seconds, low, high, noise_band, pink):
    n = round(rate * seconds)
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    rng = np.random.default_rng(130198)

    def noise(bounds, amplitude):
        bins = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
        coeffs = np.zeros(len(frequencies), complex)
        coeffs[bins] = np.exp(1j * rng.uniform(0, 2 * np.pi, bins.sum()))
        if pink:
            coeffs[bins] /= np.sqrt(frequencies[bins])
        out = np.fft.irfft(coeffs, n)
        return amplitude * out / np.sqrt(np.mean(out * out))

    band = noise(noise_band, 0.05)
    program = noise((250, 1100), 0.15)
    window = np.sin(np.pi * np.arange(n) / n) ** 2
    expected = 10 * np.log10(np.sum((band * window) ** 2) / np.sum(window ** 2))
    measured = band_evidence(program + band, rate, low_hz=low, high_hz=high, window_sec=seconds,
                             level_db=-50, ratio_db=-20, consecutive=1)
    assert measured["regions"] == [[0, n - 1, 0]]
    assert measured["band_dbfs"] == pytest.approx(expected, abs=0.3)


def test_unmeasurable_ac_and_measured_silence_have_distinct_readings():
    kwargs = dict(sample_rate=96000, low_hz=24100, high_hz=48000, window_sec=0.1,
                  level_db=-60, ratio_db=-35, consecutive=5)
    for samples in (np.empty(0), np.array([0.8])):
        assert band_evidence(samples, **kwargs)["band_dbfs"] is None
    assert band_evidence(np.zeros(4800), **kwargs)["band_dbfs"] == -np.inf


@pytest.mark.parametrize(("hz", "ratio", "selected"), [
    (24099.4, -28.0, False), (24100, -34.9, True), (24100.01, -34.9, True),
    (24100.6, -35.1, False), (24100.6, -34.9, True), (24100.6, -28.0, True),
])
def test_widely_separated_off_bin_program_and_carrier_use_the_relative_gate(hz, ratio, selected):
    rate = 96000
    t = np.arange(round(0.4 * rate)) / rate
    fraction = 10 ** (ratio / 10)
    carrier_peak = 0.4 * np.sqrt(fraction / (1 - fraction))
    source = (0.4 * np.cos(2 * np.pi * 997 * t + 0.3) +
              carrier_peak * np.cos(2 * np.pi * hz * t + 0.17))
    track = track_for(source, rate)
    plan_repairs(track, {"repair_auto_ultrasonic": True})
    measured = track.repair_measurements["original"]["ultrasonic"]
    assert track.repair_recipe.ultrasonic is selected
    assert measured["regions"] == ([[0, len(t) - 1, 0]] if selected else [])
    if hz >= 24100:
        assert measured["ratio_db"] == pytest.approx(ratio, abs=0.1)
        assert measured["band_dbfs"] == pytest.approx(20 * np.log10(carrier_peak / np.sqrt(2)), abs=0.1)


@pytest.mark.parametrize(("edge", "hz", "phase", "selected"), [
    (1, 1, 0, True), (1, 1, 0.73, True), (1, 1, np.pi / 2, True),
    (1, 0.98, 0.37, True), (1, 1.02, 0.37, False), (0.7, 0.7, 0.37, True),
    (2.5, 2.48, 0.37, True), (2.5, 2.52, 0.37, False), (20, 19.98, 0.37, True),
])
def test_infrasonic_edge_preserves_the_full_windowed_ac_at_each_phase(edge, hz, phase, selected):
    rate = 8000
    t = np.arange(4 * rate) / rate
    samples = 0.5 * np.cos(2 * np.pi * hz * t + phase) + 0.07
    track = track_for(samples, rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "subsonic_hz": edge})
    measured = track.repair_measurements["original"]["low_band"]
    assert bool(track.repair_recipe.low_cut_hz) is selected
    if selected:
        window = np.sin(np.pi * np.arange(rate) / rate) ** 2
        frames = np.lib.stride_tricks.sliding_window_view(samples, rate)[::rate // 2]
        ac = frames - frames.mean(axis=1, keepdims=True)
        direct = 10 * np.log10(np.sum((window * ac) ** 2, axis=1) / np.sum(window ** 2))
        assert measured["band_dbfs"] == pytest.approx(max(direct), abs=0.1)
        assert measured["ratio_db"] == pytest.approx(0, abs=0.1)
    else:
        assert measured["band_dbfs"] < -50


def test_infrasonic_windowed_level_reaches_the_planner_at_the_absolute_gate():
    rate = 48000
    t = np.arange(4 * rate) / rate
    track = track_for(0.004720304381429618 * np.cos(2 * np.pi * t), rate)
    plan_repairs(track, {"repair_auto_low_cut": True, "subsonic_hz": 1})
    evidence = track.repair_measurements["original"]["low_band"]
    assert evidence["band_dbfs"] == pytest.approx(-48.861432, abs=0.01)
    assert evidence["regions"] == [[0, len(t) - 1, 0]] and track.repair_recipe.low_cut_hz == 10


@pytest.mark.parametrize(("hz", "program_peak"), [(0.8, 0), (1, 0), (1, 0.01), (1.5, 0.02)])
def test_default_band_keeps_tapered_ac_at_dc_in_a_separated_mixture(hz, program_peak):
    rate = 8000
    t = np.arange(4 * rate) / rate
    infrasonic = 0.009 * np.cos(2 * np.pi * hz * t + 0.37)
    source = infrasonic + program_peak * np.cos(2 * np.pi * 997.3 * t)
    track = track_for(source, rate)
    plan_repairs(track, {"repair_auto_low_cut": True})
    evidence = track.repair_measurements["original"]["low_band"]
    window = np.sin(np.pi * np.arange(rate) / rate) ** 2
    frames = np.lib.stride_tricks.sliding_window_view(infrasonic, rate)[::rate // 2]
    frame_ac = frames - frames.mean(axis=1, keepdims=True)
    expected = np.max(10 * np.log10(np.sum((frame_ac * window) ** 2, axis=1) / np.sum(window ** 2)))
    assert evidence["band_dbfs"] == pytest.approx(expected, abs=0.1)
    assert track.repair_recipe.low_cut_hz == 10


def decoded(samples, rate, subtype):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype=subtype)
    stream.seek(0)
    return sf.read(stream)[0]


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
@pytest.mark.parametrize(("cutoff", "hz", "inside"), [
    (1, 1, True), (1, 1.1, False), (20, 19.98, True), (20, 20.02, False),
])
def test_decoded_infrasonic_gate_and_windowed_level(subtype, cutoff, hz, inside):
    rate = 48000
    t = np.arange(4 * rate) / rate
    phase = 0.0 if hz == 1 else 0.37
    source = decoded(0.004720304381429618 * np.cos(2 * np.pi * hz * t + phase), rate, subtype)
    track = track_for(source, rate)
    plan_repairs(track, {"subsonic_hz": cutoff, "repair_auto_low_cut": True})
    evidence = track.repair_measurements["original"]["low_band"]
    assert bool(track.repair_recipe.low_cut_hz) is inside
    assert bool(evidence["regions"]) is inside
    if inside:
        window = np.sin(np.pi * np.arange(rate) / rate) ** 2
        frames = np.lib.stride_tricks.sliding_window_view(source, rate)[::rate // 2]
        ac = frames - frames.mean(axis=1, keepdims=True)
        expected = np.max(10 * np.log10(np.sum((window * ac) ** 2, axis=1) / np.sum(window ** 2)))
        assert evidence["band_dbfs"] == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize(("rate", "hz", "program_hz", "relative"), [
    (48000, 19.94, 26.37, -15), (48000, 19.94, 31.113, -15),
    (48000, 19.94, 82.413, -15), (48000, 19.94, 187.37, -15),
    (96000, 24100.6, 22003.4, -28), (96000, 24100.6, 22897.31, -28),
    (96000, 24100.6, 24001.17, -28),
])
@pytest.mark.parametrize("subtype", ["PCM_16", "DOUBLE"])
def test_separated_musical_program_cannot_veto_a_sustained_carrier(rate, hz, program_hz, relative, subtype):
    high = rate == 96000
    t = np.arange(round((0.45 if high else 3.5) * rate)) / rate
    q = 10 ** (relative / 10)
    carrier = 0.4 * np.sqrt(q / (1 - q)) * np.cos(2 * np.pi * hz * t + 0.17)
    source = decoded(0.4 * np.cos(2 * np.pi * program_hz * t + 0.3) + carrier, rate, subtype)
    track = track_for(source, rate)
    plan_repairs(track, {"repair_auto_ultrasonic": high, "repair_auto_low_cut": not high})
    evidence = track.repair_measurements["original"]["ultrasonic" if high else "low_band"]
    assert evidence["regions"] == [[0, len(t) - 1, 0]]
    assert (track.repair_recipe.ultrasonic if high else bool(track.repair_recipe.low_cut_hz))
    assert evidence["ratio_db"] == pytest.approx(relative, abs=0.1)


@pytest.mark.parametrize(("rate", "hz", "relative", "subtype"), [
    (88200, 24100.2, -33., "PCM_16"), (192000, 24100.1, -34.7, "PCM_16"),
    (96000, 24100.6, -34.7, "DOUBLE"),
])
def test_quiet_band_edge_still_resolves_six_bins_from_a_louder_neighbor(rate, hz, relative, subtype):
    t = np.arange(round(0.45 * rate)) / rate
    q = 10 ** (relative / 10)
    source = 0.3 * (np.cos(2 * np.pi * 24037 * t + 0.3) +
                    np.sqrt(q / (1 - q)) * np.cos(2 * np.pi * hz * t + 0.15))
    track = track_for(decoded(source, rate, subtype), rate)
    plan_repairs(track, {"repair_auto_ultrasonic": True})
    assert track.repair_recipe.ultrasonic
    assert track.detector_results["ultrasonic"].data["regions"] == [[0, len(t) - 1, 0]]


@pytest.mark.parametrize("high", [False, True])
def test_several_separated_partials_in_the_program_preserve_an_edge_decision(high):
    rate, edge, spacing = (96000, 24100, 10) if high else (48000, 20, 1)
    t = np.arange(round(rate * (0.45 if high else 3.5))) / rate
    direction = -1 if high else 1
    offsets = [12.73, 21.02, 28.71, 39.34, 47.1, 56.7]
    program = sum(0.09 * np.cos(2 * np.pi * (edge + direction * spacing * delta) * t + 0.7 * i)
                  for i, delta in enumerate(offsets))
    carrier = (0.008 if high else 0.035) * np.cos(2 * np.pi * (edge - direction * .06 * spacing) * t)
    track = track_for(program + carrier, rate)
    plan_repairs(track, {"repair_auto_ultrasonic": high, "repair_auto_low_cut": not high})
    evidence = track.repair_measurements["original"]["ultrasonic" if high else "low_band"]
    assert evidence["regions"] == [[0, len(t) - 1, 0]]


@pytest.mark.parametrize("high", [False, True])
def test_out_of_band_modulation_and_low_noise_do_not_trigger_an_automatic_repair(high):
    rate, edge, spacing = (96000, 24100, 10) if high else (48000, 20, 1)
    t = np.arange(round(rate * (0.45 if high else 3.5))) / rate
    direction = -1 if high else 1
    # Modulation creates two sidebands, each still outside the repair band.
    a = 0.32 + 0.18 * np.cos(2 * np.pi * 0.14 * spacing * t)
    near = a * np.cos(2 * np.pi * (edge + direction * 0.2 * spacing) * t + 0.31)
    program = 0.12 * np.cos(2 * np.pi * (edge + direction * 6.3 * spacing) * t)
    quiet_noise = np.random.default_rng(4619).normal(scale=3e-5, size=len(t))
    track = track_for(decoded(near + program + quiet_noise, rate, "PCM_16"), rate)
    plan_repairs(track, {"repair_auto_ultrasonic": high, "repair_auto_low_cut": not high})
    evidence = track.repair_measurements["original"]["ultrasonic" if high else "low_band"]
    assert evidence["regions"] == [] and not track.repair_recipe.active


@pytest.mark.parametrize("width", [48000, 50001, 96000, 192000])
def test_boundary_partial_dirichlet_kernel_matches_exact_reference_across_bins_and_nyquist(width):
    from sessionpreplib.repairs import _BoundaryPartial, _rect_scalar

    window = np.sin(np.pi * np.arange(width) / width) ** 2
    window_dc = np.fft.rfft(window) / width
    nyq = width // 2
    centres = sorted({20, nyq - 1, nyq, min(nyq, 2410), min(nyq, 4820), min(nyq, 24121)})
    for centre in centres:
        bp = _BoundaryPartial(centre, width, window_dc)
        for delta in (0.0, 2e-12, 1e-10, 1e-5, 0.37, -1e-10, -2e-12):
            f = float(centre) + delta
            if f < 0 or f > width / 2:
                continue
            for signed_f in (f, -f):
                f_int = round(signed_f)
                df = signed_f - f_int
                k = (f_int - bp.extended_bins) % width
                k = np.where(k > width / 2, k - width, k)
                d = k + df
                ref = np.exp(1j * np.pi * d * (1 - 1 / width)) * np.sinc(d) / np.sinc(d / width)
                actual = bp._rect_ext(signed_f)
                assert np.max(np.abs(actual - ref)) < 1e-12

                k0 = f_int % width
                if k0 > width // 2:
                    k0 -= width
                d0 = k0 + df
                ref0 = np.exp(1j * np.pi * d0 * (1 - 1 / width)) * np.sinc(d0) / np.sinc(d0 / width)
                assert abs(_rect_scalar(signed_f, width) - ref0) < 1e-12


def test_inband_tone_near_nyquist_resolves_accurately():
    rate = 48000
    window = np.sin(np.pi * np.arange(rate) / rate) ** 2
    in_band = tone(rate, 23999.0, -14.55, 2.0)
    frame = in_band[:rate] - np.mean(in_band[:rate])
    expected_db = 10.0 * np.log10(np.sum((window * frame) ** 2) / np.sum(window ** 2))
    measured = band_evidence(
        in_band, rate, low_hz=0, high_hz=23999.3, window_sec=1.0,
        level_db=-50, ratio_db=-20, consecutive=1,
    )
    assert measured["band_dbfs"] == pytest.approx(expected_db, abs=1e-4)
    assert measured["ratio_db"] == pytest.approx(0.0, abs=1e-4)

    out_of_band = tone(rate, 23999.7, -14.55, 2.0)
    excluded = band_evidence(
        out_of_band, rate, low_hz=0, high_hz=23999.3, window_sec=1.0,
        level_db=-50, ratio_db=-20, consecutive=1,
    )
    assert excluded["band_dbfs"] < -50.0
    assert excluded["regions"] == []
