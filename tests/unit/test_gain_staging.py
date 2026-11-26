"""Analytic envelope, PCM and fader oracles for the Peak-Capped RMS normalization plan."""

import html
import json
import math
import re

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track, resolve_rms_anchor
from sessionpreplib.daw_processor import resolve_transfer_audio
from sessionpreplib.detectors import SilenceDetector
from sessionpreplib.models import SessionContext, TrackContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionpreplib.reports import save_json


def _evaluate(samples, rate, filename="Bass_01.wav", **settings):
    track = TrackContext(
        filename, "", audio_data=samples, samplerate=rate,
        channels=1, total_samples=len(samples), duration_sec=len(samples) / rate,
        bitdepth="64-bit float", subtype="DOUBLE",
    )
    proc = BimodalNormalizeProcessor()
    proc.configure(settings)
    return proc.process(track)


@pytest.mark.parametrize("trim_db", [-6, -7])
def test_held_pcm_note_keeps_rms_normalization_across_input_trims(tmp_path, trim_db):
    rate = 48000
    frames = np.arange(62400)
    t = frames / rate
    cycle = np.sin(2 * np.pi * (frames % 480) / 480)
    held = cycle * np.minimum(1, t / .010) * np.clip((1 - t) / .040, 0, 1)
    path = tmp_path / "Note.wav"
    sf.write(path, held * 10**(trim_db / 20), rate, subtype="PCM_24")
    source, _ = sf.read(path)
    rms = np.sqrt(np.mean(source[480:960]**2))
    gain = 10**(-18 / 20) / rms
    track = load_track(str(path))
    pipeline = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()])
    session = SessionContext([track], {})
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    result = track.processor_results["bimodal_normalize"]
    assert result.classification == "Auto (RMS)"
    assert result.data["governing"] == "RMS"
    pipeline.execute(session, str(tmp_path / "printed"))
    printed, _ = sf.read(tmp_path / "printed" / path.name)
    np.testing.assert_allclose(printed, source * gain, atol=2**-23, rtol=0)
    assert 20 * np.log10(max(abs(printed))) == pytest.approx(-18 + 10 * np.log10(2), abs=.0001)


def test_post_repair_signal_determines_peak_capped_rms_gain_and_report(tmp_path):
    from sessionpreplib.repairs import render_corrected

    rate = 48000
    samples = np.zeros(rate)
    samples[2400:8160] = 0.25  # unipolar pulse with significant DC mean
    path = tmp_path / "Pulse.wav"
    sf.write(path, samples, rate, subtype="PCM_24")
    track = load_track(str(path))
    track.repair_overrides["dc_remove"] = True
    proc = BimodalNormalizeProcessor()
    pipeline = Pipeline([SilenceDetector()], [proc], {})
    session = SessionContext([track], {})
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    result = track.processor_results[proc.id]
    corrected = render_corrected(track.audio_data, rate, track.repair_recipe)
    expected_peak_db = 20 * math.log10(float(np.max(np.abs(corrected))))
    raw_rms_db = 20 * math.log10(0.25)
    assert result.data["detected_peak_db"] == pytest.approx(expected_peak_db, abs=0.01)
    # DC removal alters the pulse body RMS substantially relative to the unipolar raw pulse
    assert result.data["detected_rms_db"] < raw_rms_db - 1.0
    rendered = html.unescape(re.sub("<[^>]*>", " ", proc.render_html(result, track, verbose=True))).replace("−", "-")
    assert f"{result.data['detected_rms_db']:.1f} dBFS" in rendered


@pytest.mark.parametrize("force_peak,filename,expected_mode", [
    ("kick|snare", "Kick_01.wav", "Peak"),
    ("kick|snare", "SNARE_TOP.wav", "Peak"),
    ("kick|snare", "Bass_01.wav", "Auto"),
    ("", "Kick_01.wav", "Auto"),
])
def test_force_peak_regex_override(force_peak, filename, expected_mode):
    samples = np.tile([0.125, -0.125], 1000)
    result = _evaluate(samples, 2000, filename=filename, force_peak=force_peak)
    assert result.data["mode"] == expected_mode
    if expected_mode == "Peak":
        assert result.data["governing"] == "Peak"
        assert result.classification == "Peak"
        assert result.gain_db == pytest.approx(-6.0 - 20 * math.log10(0.125), abs=0.01)
    else:
        assert result.data["governing"] == "RMS"
        assert result.classification == "Auto (RMS)"


def test_rms_window_at_the_relative_gate_is_included():
    samples = np.tile([0.5, -0.25, 0.125], 100)
    width_db = 10 * np.log10(0.5**2) - 10 * np.log10(0.25**2)
    t_eq = TrackContext("eq.wav", "", audio_data=samples, samplerate=1000, channels=1,
                        total_samples=len(samples), bitdepth="64-bit float", subtype="DOUBLE",
                        duration_sec=len(samples) / 1000)
    t_ab = TrackContext("ab.wav", "", audio_data=samples, samplerate=1000, channels=1,
                        total_samples=len(samples), bitdepth="64-bit float", subtype="DOUBLE",
                        duration_sec=len(samples) / 1000)
    eq = resolve_rms_anchor(t_eq, window_ms=1, stereo_mode="avg", gate_relative_db=width_db, rms_anchor_mode="percentile", rms_percentile=25.0)
    ab = resolve_rms_anchor(t_ab, window_ms=1, stereo_mode="avg", gate_relative_db=width_db - 1e-8, rms_anchor_mode="percentile", rms_percentile=25.0)
    assert eq["rms_anchor_db"] == pytest.approx(20 * np.log10(0.25))
    assert ab["rms_anchor_db"] == pytest.approx(20 * np.log10(0.5))


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
    assert _evaluate(snare, rate).data["governing"] == "Peak"
    assert _evaluate(bass, rate).data["governing"] == "RMS"


@pytest.mark.parametrize("stereo_mode,anchor,override,high,convention", [
    ("avg", "percentile", None, 0.25, "standard"),
    ("sum", "percentile", None, 0.25, "aes17"),
    ("avg", "max", None, 0.25, "aes17"),
    ("sum", "max", "Peak", 0.25, "standard"),
    ("avg", "percentile", None, 0.875, "aes17"),
])
def test_reported_target_gain_and_decoded_stereo_pcm(tmp_path, stereo_mode, anchor, override,
                                                    high, convention):
    envelope = np.repeat([0.125, high], [900, 100])
    envelope[::2] *= -1
    original = np.column_stack((envelope, -envelope))
    source = tmp_path / "Keys.wav"
    sf.write(source, original, 1000, subtype="PCM_24")
    track = load_track(str(source))
    track.classification_override = override
    config = dict(window=1, rms_anchor=anchor, rms_percentile=75, stereo_mode=stereo_mode,
                  dbfs_convention=convention, target_rms=-18.0, target_peak=-6.0, fader_headroom_db=0)
    processor = BimodalNormalizeProcessor()
    pipeline = Pipeline([SilenceDetector()], [processor], config, max_workers=1)
    session = SessionContext([track], config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    result = track.processor_results[processor.id]
    amplitude = high if anchor == "max" else 0.125
    rms = 20 * math.log10(amplitude) + (10 * math.log10(2) if stereo_mode == "sum" else 0)
    peak_gain = -6.0 - 20 * math.log10(high)
    rms_gain = -18.0 - rms
    use_peak = override == "Peak" or (override is None and peak_gain < rms_gain)
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

    # Under AES17, ONLY RMS shifts by +3.01 dB; sample peak stays unshifted.
    rms_offset = 10 * math.log10(2) if convention == "aes17" else 0.0
    summary = html.unescape(re.sub("<[^>]*>", " ", processor.render_html(result))).replace("−", "-")
    expected_peak = 20 * math.log10(high) + gain
    expected_rms = rms + gain + rms_offset
    assert f"peak {expected_peak:.1f} dBFS" in summary
    assert f"RMS {expected_rms:.1f} dBFS" in summary


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
    pipeline = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()],
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
    anchor_pipeline = Pipeline([SilenceDetector()], [BimodalNormalizeProcessor()],
                               dict(config, anchor="room l"), max_workers=1)
    anchor_pipeline.plan(session, preserve_selections={track.filename for track in tracks})
    faders = [t.processor_results["bimodal_normalize"].data["fader_offset"] for t in tracks]
    assert faders == pytest.approx([0, 0, 7, 0, 0])
