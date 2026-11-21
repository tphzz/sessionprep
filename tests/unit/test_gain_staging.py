"""Analytic envelope, PCM and fader oracles for the normalization plan."""

import html
import json
import math
import re

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.daw_processor import resolve_transfer_audio
from sessionpreplib.detectors import AudioClassifierDetector, SilenceDetector
from sessionpreplib.models import SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionpreplib.reports import save_json


def classify(samples, rate, **settings):
    track = TrackContext("Bass_01.wav", "", audio_data=samples, samplerate=rate,
                         channels=1, total_samples=len(samples), duration_sec=len(samples) / rate,
                         bitdepth="64-bit float", subtype="DOUBLE")
    detector = AudioClassifierDetector()
    detector.configure(settings)
    return detector.analyze(track).data


@pytest.mark.parametrize("shape,padding,expected", [
    ("attack", 0, "Sustained"), ("attack", 4, "Transient"),
    ("flat", 4, "Sustained"), ("burst", 0, "Transient"),
])
def test_classifier_votes_on_measured_envelopes(shape, padding, expected):
    rate = 2000
    signal = np.full(rate, 1 / 16)
    if shape == "attack":
        signal[rate // 4] = 0.5
    elif shape == "burst":
        signal[:rate // 10] = 0.5
    signal[::2] *= -1
    signal = np.pad(signal, padding * rate)
    result = classify(signal, rate)

    # Direct window sums use exact binary samples; no detector/cache helpers.
    power = signal ** 2
    momentary = np.convolve(power, np.ones(800), mode="valid") / 800
    env = np.convolve(power, np.ones(20), mode="valid") / 20
    peak_frame = int(env.argmax())
    tail = env[peak_frame + 301:peak_frame + 401]
    expected_decay = 10 * math.log10(env.max() / np.median(tail))
    expected_crest = 10 * math.log10(power.max() / momentary.max())
    density = np.mean(momentary >= momentary.max() * 1e-4)
    assert result["crest"] == pytest.approx(expected_crest, abs=1e-10)
    assert result["decay_db"] == pytest.approx(expected_decay, abs=1e-10)
    assert result["density"] == pytest.approx(density, abs=1e-10)
    assert result["classification"] == expected
    if shape == "attack" and padding:
        # Density must be strictly below its configured threshold.
        assert classify(signal, rate, sparse_density_threshold=density)["classification"] == "Sustained"


@pytest.mark.parametrize("force_transient,force_sustained,expected", [
    (["bass"], ["Bass_01.wav$"], "Transient (Forced)"),
    ([], ["BASS_??.wav"], "Sustained (Forced)"),
    (["bass_01$"], [], "Transient (Forced)"),
    (["bass_0$"], ["snare"], "Sustained"),
])
def test_classifier_filename_override_precedence(force_transient, force_sustained, expected):
    samples = np.tile([0.125, -0.125], 1000)
    result = classify(samples, 2000, force_transient=force_transient, force_sustained=force_sustained)
    assert result["classification"] == expected


def test_rms_window_at_the_relative_gate_is_included():
    samples = np.tile([0.5, -0.25, 0.125], 100)
    width_db = 10 * np.log10(0.5**2) - 10 * np.log10(0.25**2)
    equal = classify(samples, 1000, window=1, gate_relative_db=width_db, rms_percentile=25)
    above = classify(samples, 1000, window=1, gate_relative_db=width_db - 1e-8, rms_percentile=25)
    assert equal["density"] == pytest.approx(2 / 3)
    assert above["density"] == pytest.approx(1 / 3)
    assert equal["rms_anchor_db"] == pytest.approx(20 * np.log10(0.25))
    assert above["rms_anchor_db"] == pytest.approx(20 * np.log10(0.5))


@pytest.mark.parametrize("metric", ["crest", "decay"])
def test_dynamic_vote_requires_strictly_exceeding_the_threshold(metric):
    rate = 2000
    signal = np.full(rate, 1 / 16)
    if metric == "crest":
        signal[rate // 4] = 0.5
        signal = np.pad(signal, 4 * rate)
        window_power = (0.5**2 + 799 * (1 / 16)**2) / 800
        threshold = 20 * np.log10(0.5) - 20 * np.log10(math.sqrt(window_power))
        key = "crest_threshold"
    else:
        signal[:rate // 10] = 0.5
        threshold = 10 * np.log10(0.5**2) - 10 * np.log10((1 / 16)**2)
        key = "decay_db_threshold"
    signal[::2] *= -1
    equal = classify(signal, rate, **{key: threshold})
    below = classify(signal, rate, **{key: np.nextafter(threshold, -math.inf)})
    assert equal["classification"] == "Sustained"
    assert below["classification"] == "Transient"


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_percussion_backbeat_and_sustained_bass_phrase(rate):
    # Original four-note bass line and two-hit snare backbeat at 120 BPM.
    beat = np.arange(rate // 2) / rate
    smooth_gate = np.minimum(1.0, np.minimum(beat, 0.5 - beat) / 0.015)
    bass = np.concatenate([
        smooth_gate * (0.2 * np.sin(2 * np.pi * hz * beat) + 0.04 * np.sin(4 * np.pi * hz * beat))
        for hz in (110.0, 130.8128, 146.8324, 110.0)
    ])
    snare = np.zeros_like(bass)
    noise = np.random.default_rng(42).uniform(-0.35, 0.35, size=len(beat))
    hit = np.exp(-beat / 0.029) * (noise + 0.25 * np.sin(2 * np.pi * 185 * beat))
    for tick, accent in ((1, 0.8), (3, 1.0)):
        snare[tick * len(beat):(tick + 1) * len(beat)] = hit * accent
    assert classify(snare, rate)["classification"] == "Transient"
    assert classify(bass, rate)["classification"] == "Sustained"


@pytest.mark.parametrize("stereo_mode,anchor,classification,high,convention", [
    ("avg", "percentile", "Sustained", 0.25, "standard"),
    ("sum", "percentile", "Sustained", 0.25, "aes17"),
    ("avg", "max", "Sustained", 0.25, "aes17"),
    ("sum", "max", "Transient", 0.25, "standard"),
    ("avg", "percentile", "Sustained", 0.875, "aes17"),
])
def test_reported_target_gain_and_decoded_stereo_pcm(tmp_path, stereo_mode, anchor, classification,
                                                    high, convention):
    envelope = np.repeat([0.125, high], [900, 100])
    envelope[::2] *= -1
    original = np.column_stack((envelope, -envelope))
    source = tmp_path / "Keys.wav"
    sf.write(source, original, 1000, subtype="PCM_24")
    track = load_track(str(source))
    track.classification_override = classification
    config = dict(window=1, rms_anchor=anchor, rms_percentile=75, stereo_mode=stereo_mode,
                  dbfs_convention=convention, target_rms=-18.0, target_peak=-6.0, fader_headroom_db=0)
    processor = BimodalNormalizeProcessor()
    pipeline = Pipeline([SilenceDetector(), AudioClassifierDetector()], [processor], config, max_workers=1)
    session = SessionContext([track], config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    result = track.processor_results[processor.id]
    amplitude = high if anchor == "max" else 0.125
    rms = 20 * math.log10(amplitude) + (10 * math.log10(2) if stereo_mode == "sum" else 0)
    peak_gain = -6.0 - 20 * math.log10(high)
    rms_gain = -18.0 - rms
    use_peak = classification == "Transient" or peak_gain < rms_gain
    gain = peak_gain if use_peak else rms_gain
    assert result.data["detected_rms_db"] == pytest.approx(rms)
    assert result.gain_db == pytest.approx(gain)
    pipeline.execute(session, str(tmp_path / "printed"))
    decoded, rate = sf.read(tmp_path / "printed" / source.name, always_2d=True)
    assert rate == 1000
    np.testing.assert_allclose(decoded, original * 10 ** (gain / 20), rtol=0, atol=2**-23)
    report = tmp_path / "readings.json"
    save_json(session, config, str(report), [])
    receipt = json.loads(report.read_text(encoding="utf8"))["tracks"][0]["processing"]
    assert receipt["gain_applied_db"] == pytest.approx(gain, abs=0.005)
    assert receipt["fader_db"] == pytest.approx(-gain, abs=0.005)

    display_offset = 10 * math.log10(2) if convention == "aes17" else 0.0
    expected_target = (-6 if use_peak else -18) + display_offset
    summary = html.unescape(re.sub("<[^>]*>", " ", processor.render_html(result))).replace("−", "-")
    assert f"→ {expected_target:.1f} dBFS" in summary


def test_linked_and_bypassed_tracks_reach_receipts_and_manifest_with_one_headroom_shift(tmp_path):
    config = {"_fader_ceiling_db": 6, "fader_headroom_db": 3}
    tracks = []
    for name, manual, group in [
        ("Room L", 6, "Pair"), ("Room R", -2, "Pair"), ("Snare", -9, None),
        ("Bypassed", -20, "Pair"), ("Silent", -25, "Pair"),
    ]:
        path = tmp_path / f"{name}.wav"
        samples = np.zeros(4800) if name == "Silent" else np.tile([-0.0625, 0.0625], 2400)
        sf.write(path, samples, 48000, subtype="PCM_24")
        track = load_track(str(path))
        track.group = group
        track.manual_gain_db = manual
        tracks.append(track)
    tracks[3].processor_skip.add("bimodal_normalize")
    pipeline = Pipeline([SilenceDetector(), AudioClassifierDetector()], [BimodalNormalizeProcessor()],
                        config, max_workers=1)
    session = SessionContext(tracks, config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    # Eligible gains are -2, -2 and -9. A common 6 dB fader shift puts +9 at the +3 ceiling.
    expected = {"Room L.wav": -4, "Room R.wav": -4, "Snare.wav": 3, "Bypassed.wav": 0}
    for track in tracks[:3]:
        result = track.processor_results["bimodal_normalize"]
        assert result.gain_db + expected[track.filename] == pytest.approx(-6)
        for narrow in (True, False):
            rendered = html.unescape(pipeline.audio_processors[0].render_html(result, track, narrow=narrow))
            assert "Fader rebalanced: −6.0 dB shift" in rendered
    assert tracks[4].processor_results["bimodal_normalize"].data["fader_offset"] == 0
    pipeline.prepare(session, str(tmp_path / "prepared"))
    outputs = {t.filename: t for t in session.output_tracks}
    assert {entry.output_filename for entry in session.transfer_manifest} == set(expected)
    for entry in session.transfer_manifest:
        output = outputs[entry.output_filename]
        media = resolve_transfer_audio(session, output)
        assert media.fader_db == pytest.approx(expected[entry.output_filename])
        assert entry.group == ("Pair" if entry.output_filename != "Snare.wav" else None)
    pipeline.execute(session, str(tmp_path / "printed"))
    report_path = tmp_path / "results.json"
    save_json(session, config, str(report_path), [])
    records = json.loads(report_path.read_text(encoding="utf8"))["tracks"]
    assert {r["filename"]: r["processing"]["fader_db"] for r in records} == {**expected, "Silent.wav": 0}

    # Raising the -4 dB Room anchor to unity raises Snare from +3 to +7,
    # above the DAW's +6 dB maximum. The planner preserves relative balance.
    anchor_pipeline = Pipeline([SilenceDetector(), AudioClassifierDetector()], [BimodalNormalizeProcessor()],
                               dict(config, anchor="room l"), max_workers=1)
    anchor_pipeline.plan(session, preserve_selections={track.filename for track in tracks})
    faders = [t.processor_results["bimodal_normalize"].data["fader_offset"] for t in tracks]
    assert faders == pytest.approx([0, 0, 7, 0, 0])
