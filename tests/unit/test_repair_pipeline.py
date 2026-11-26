"""Audio repairs through session analysis, normalization and the two delivery paths."""
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.config import default_config
from sessionpreplib.daw_processor import resolve_transfer_audio
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import RepairRecipe, render_corrected
from sessionpreplib.topology import ChannelRoute, TopologyEntry, TopologyMapping, TopologySource


def source_track(directory, name, audio, rate=48000, subtype="FLOAT"):
    path = directory / name
    sf.write(path, audio, rate, subtype=subtype)
    return load_track(str(path))


def analyze(tracks, **settings):
    config = {**default_config(), **settings}
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    session = SessionContext(tracks, config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    return pipeline, session


def run_cli(directory, *args):
    script = Path(__file__).resolve().parents[2] / "sessionprep.py"
    env = {**os.environ, "SP_LOG_LEVEL": "NONE", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(script), str(directory), *args], cwd=script.parent,
                          env=env, capture_output=True, encoding="utf-8", timeout=60)


@pytest.mark.parametrize("interruption", ["encoder", "cancel"])
def test_partial_prepare_rejects_transfer_until_retry(tmp_path, monkeypatch, interruption):
    import sessionpreplib.pipeline as pipeline_module

    signal = np.array([0.25, -0.125, 0.5])
    tracks = [source_track(tmp_path, name, signal * level) for name, level in [("A.wav", 1), ("B.wav", 0.5)]]
    config = {"_use_processed": True}
    pipeline = Pipeline([], [], config)
    session = SessionContext(tracks, config)
    dest = tmp_path / "prepared"
    pipeline.prepare(session, str(dest))
    assert session.prepare_state == "ready"
    originals = {Path(t.filepath): Path(t.filepath).read_bytes() for t in tracks}
    prior = {t.filename: (dest / t.filename).read_bytes() for t in tracks}
    sentinel = dest / "unowned.wav"
    sentinel.write_bytes(b"keep this")
    for track in tracks:
        track.manual_gain_db = -6.020599913279624
    write = pipeline_module._write_atomic

    def fail_encoder(path, *args, **kwargs):
        if path.name == "B.wav":
            raise OSError("encoder stopped")
        return write(path, *args, **kwargs)

    def cancel_before_second(_step, _total, label):
        if label == "Writing B.wav":
            raise InterruptedError("cancelled between outputs")

    if interruption == "encoder":
        with monkeypatch.context() as fault:
            fault.setattr(pipeline_module, "_write_atomic", fail_encoder)
            pipeline.prepare(session, str(dest))
        assert session.config["_prepared_files"] == ["A.wav"]
        assert session.config["_prepare_errors"] == [("B.wav", "encoder stopped")]
    else:
        with pytest.raises(InterruptedError):
            pipeline.prepare(session, str(dest), progress_cb=cancel_before_second)
    assert session.prepare_state == "stale"
    for output in session.output_tracks:
        with pytest.raises(ValueError, match="out of date"):
            resolve_transfer_audio(session, output)
    assert (dest / "A.wav").read_bytes() != prior["A.wav"]
    assert (dest / "B.wav").read_bytes() == prior["B.wav"]
    pipeline.prepare(session, str(dest))
    assert session.prepare_state == "ready"
    for track, level in zip(session.output_tracks, [1, 0.5], strict=True):
        published = resolve_transfer_audio(session, track)
        np.testing.assert_array_equal(sf.read(published.filepath)[0], signal * level * 0.5)
    assert sentinel.read_bytes() == b"keep this"
    assert {path: path.read_bytes() for path in originals} == originals


def test_default_plan_uses_existing_gain_and_audio_even_for_flagged_material(tmp_path):
    rate = 96000
    time = np.arange(2 * rate) / rate
    signal = 0.07 + 0.18 * np.sin(2 * np.pi * 8 * time) + 0.27 * np.sin(2 * np.pi * 30000 * time)
    track = source_track(tmp_path, "Track.wav", signal, rate)
    track.classification_override = "Transient"
    original = track.audio_data.copy()
    pipeline, session = analyze([track])
    pipeline.plan(session)

    assert track.repair_recipe == RepairRecipe()
    assert track.processor_results["bimodal_normalize"].gain_db == pytest.approx(
        -6 - 20 * np.log10(np.max(np.abs(original))))
    track.processor_skip.add("bimodal_normalize")
    pipeline.prepare(session, str(tmp_path / "prepared"))
    prepared, saved_rate = sf.read(tmp_path / "prepared" / track.filename)
    assert saved_rate == rate
    np.testing.assert_array_equal(prepared, original)


def test_automatic_modes_require_evidence_on_the_individual_track_and_band(tmp_path):
    rate = 96000
    time = np.arange(3 * rate) / rate
    program = 0.12 * np.sin(2 * np.pi * 440 * time)
    program[:rate // 2] = 0
    bias = source_track(tmp_path, "Bias.wav", program + 0.05, rate)
    rumble = source_track(tmp_path, "Rumble.wav", program + 0.2 * np.sin(2 * np.pi * 8 * time), rate)
    carrier = 0.16 * np.sin(2 * np.pi * 30000 * time)
    stereo_carrier = np.column_stack([program + carrier, program - carrier])
    ultrasonic = source_track(tmp_path, "Carrier.wav", stereo_carrier, rate)
    clean = source_track(tmp_path, "Clean.wav", program, rate)
    quiet = source_track(tmp_path, "Quiet.wav", 0.0001 * np.sin(2 * np.pi * 8 * time), rate)
    burst = source_track(tmp_path, "Burst.wav", program + carrier * (time < 0.12), rate)
    low_rate = source_track(tmp_path, "LowRate.wav", program[::2], rate // 2)
    musical_bass = source_track(tmp_path, "Bass.wav", program + 0.18 * np.sin(2 * np.pi * 25 * time), rate)
    tracks = [bias, rumble, ultrasonic, clean, quiet, burst, low_rate, musical_bass]

    pipeline, session = analyze(tracks, repair_auto_dc=True, repair_auto_low_cut=True,
                                repair_auto_ultrasonic=True, subsonic_report_as="skip")
    pipeline.plan(session)

    assert [track.repair_recipe for track in tracks] == [
        RepairRecipe(dc_remove=True), RepairRecipe(low_cut_hz=10.0), RepairRecipe(ultrasonic=True),
        RepairRecipe(), RepairRecipe(), RepairRecipe(), RepairRecipe(), RepairRecipe(),
    ]
    # The 30 Hz advisory still sees this bass note outside the 20 Hz automatic band.
    assert musical_bass.detector_results["subsonic"].data["subsonic_warn"]

    # One explicit opt-out must retain inherited choices for the other bands.
    ultrasonic.repair_overrides = {"ultrasonic": False, "low_cut": True, "low_cut_hz": 13.5}
    rumble.repair_overrides = {"ultrasonic": True}
    musical_bass.repair_overrides = {"low_cut": True}
    pipeline.plan(session)
    assert ultrasonic.repair_recipe == RepairRecipe(low_cut_hz=13.5)
    assert rumble.repair_recipe == RepairRecipe(low_cut_hz=10.0, ultrasonic=True)
    assert musical_bass.repair_recipe == RepairRecipe(low_cut_hz=10.0)


@pytest.mark.parametrize("writer", ["prepare", "execute"])
def test_normalization_uses_corrected_peak_and_each_render_starts_from_original(tmp_path, writer):
    rate = 96000
    time = np.arange(2 * rate) / rate
    fade = np.clip(np.minimum(time, 2.0 - time) / 0.04, 0, 1)
    signal = fade * (0.07 * np.sin(2 * np.pi * 1000 * time) + 0.41 * np.sin(2 * np.pi * 30000 * time))
    track = source_track(tmp_path, "Peak.wav", signal, rate)
    track.classification_override = "Transient"
    original = track.audio_data.copy()
    source_bytes = Path(track.filepath).read_bytes()
    original_peak = 20 * np.log10(np.max(np.abs(original)))
    corrected = render_corrected(original, rate, RepairRecipe(ultrasonic=True))
    corrected_peak = 20 * np.log10(np.max(np.abs(corrected)))
    assert original_peak - corrected_peak > 6
    pipeline, session = analyze([track], repair_auto_ultrasonic=True)
    pipeline.plan(session)

    normalization = track.processor_results["bimodal_normalize"]
    gain = normalization.gain_db
    assert normalization.data["detected_peak_db"] == pytest.approx(corrected_peak, abs=1e-5)
    assert gain == pytest.approx(-6 - corrected_peak, abs=1e-5)
    assert normalization.data["fader_offset"] == pytest.approx(-gain)
    output = tmp_path / "output"
    render = getattr(pipeline, writer)
    for _ in range(2):
        render(session, str(output))
        prepared, saved_rate = sf.read(output / track.filename)
        assert saved_rate == rate
        np.testing.assert_allclose(prepared, corrected * 10 ** (gain / 20), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(track.audio_data, original)

    track.processor_skip.add("bimodal_normalize")
    render(session, str(output))
    skipped, _ = sf.read(output / track.filename)
    np.testing.assert_allclose(skipped, corrected, rtol=1e-6, atol=1e-7)

    track.processor_skip.discard("bimodal_normalize")
    track.repair_overrides = {"ultrasonic": False}
    pipeline.plan(session)
    restored_gain = track.processor_results["bimodal_normalize"].gain_db
    assert restored_gain == pytest.approx(-6 - original_peak)
    render(session, str(output))
    restored, _ = sf.read(output / track.filename)
    np.testing.assert_allclose(restored, original * 10 ** (restored_gain / 20), rtol=1e-6, atol=1e-7)
    assert Path(track.filepath).read_bytes() == source_bytes


def test_gain_linked_group_uses_the_corrected_sustained_rms_before_rebalancing(tmp_path):
    rate = 96000
    time = np.arange(3 * rate) / rate
    note = np.sin(2 * np.pi * 1000 * time)
    first = source_track(tmp_path, "First.wav", 0.07 * note + 0.4 * np.sin(2 * np.pi * 30000 * time), rate)
    second = source_track(tmp_path, "Second.wav", 0.25 * note, rate)
    for track in [first, second]:
        track.group = "Pair"
        track.classification_override = "Auto"
    pipeline, session = analyze([first, second], repair_auto_ultrasonic=True)
    pipeline.plan(session)

    first_result = first.processor_results["bimodal_normalize"]
    second_result = second.processor_results["bimodal_normalize"]
    first_gain = -18 - 20 * math.log10(0.07 / math.sqrt(2))
    second_gain = -18 - 20 * math.log10(0.25 / math.sqrt(2))
    assert first_result.data["gain_db_individual"] == pytest.approx(first_gain, abs=0.03)
    for result in [first_result, second_result]:
        assert result.gain_db == pytest.approx(second_gain, abs=0.03)
        assert result.data["fader_offset"] == pytest.approx(-second_gain, abs=0.03)


def test_local_rumble_is_eligible_when_the_whole_track_average_is_clean(tmp_path):
    rate = 12000
    time = np.arange(24 * rate) / rate
    audio = 0.2 * np.sin(2 * np.pi * 440 * time)
    audio += (time < 3) * 0.04 * np.sin(2 * np.pi * 8 * time)
    track = source_track(tmp_path, "Local.wav", audio, rate)
    pipeline, session = analyze([track], repair_auto_low_cut=True)
    pipeline.plan(session)
    assert not track.detector_results["subsonic"].data["subsonic_warn"]
    assert track.repair_recipe.low_cut_hz == 10


def test_ultrasonic_only_finding_is_visible_without_enabling_a_repair(tmp_path):
    from sessionpreplib.rendering import build_diagnostic_summary

    rate = 96000
    time = np.arange(rate) / rate
    track = source_track(tmp_path, "Carrier.wav", 0.05 * np.sin(2 * np.pi * 30000 * time), rate)
    pipeline, session = analyze([track])
    pipeline.plan(session)
    assert not track.repair_recipe.active
    assert any(group.get("detector_id") == "ultrasonic"
               for group in build_diagnostic_summary(session)["information"])


def test_execution_receipts_track_only_files_written_in_this_attempt(tmp_path):
    audio = np.tile([0.25, -0.25], 24000)
    track = source_track(tmp_path, "Trim.wav", audio, subtype="PCM_16")
    pipeline, session = analyze([track], bimodal_normalize_enabled=False)
    track.manual_gain_db = -1.0
    pipeline.plan(session)
    output = tmp_path / "render"
    pipeline.execute(session, str(output))
    assert session.config["_executed_files"] == {
        "Trim.wav": {"gain_db": -1.0, "fader_db": 0.0, "applied_processors": []}}
    good = (output / track.filename).read_bytes()
    track.manual_gain_db = 20.0
    pipeline.execute(session, str(output))
    assert not session.config["_executed_files"]
    assert session.config["_execute_errors"] and (output / track.filename).read_bytes() == good
    pipeline.plan(session)
    assert "_executed_files" not in session.config and "_execute_errors" not in session.config


def test_duplicate_destination_preserves_previous_file_and_reports_both_outputs(tmp_path):
    track = source_track(tmp_path, "Input.wav", np.full(200, 0.125))
    source = TopologySource(track.filename, [ChannelRoute(0, 0)])
    entry = TopologyEntry("Mix.wav", 1, [source])
    pipeline, session = analyze([track])
    session.topology = TopologyMapping([entry, entry])
    output = tmp_path / "render"
    output.mkdir()
    old = output / "Mix.wav"
    old.write_bytes(b"previous")
    pipeline.prepare(session, str(output))
    assert not session.config["_prepared_files"]
    assert len(session.config["_prepare_errors"]) == 2
    assert old.read_bytes() == b"previous"


@pytest.mark.parametrize("writer", ["prepare", "execute"])
def test_overrange_pcm_reports_minimum_reduction_retains_prior_output_and_can_retry(tmp_path, writer):
    rate = 48000
    step = np.full(rate, -0.875)
    step[rate // 2:] = 0.75
    affected = source_track(tmp_path, "Affected.wav", step, rate, subtype="PCM_16")
    affected.repair_overrides = {"dc_remove": True}
    affected.classification_override = "Transient"
    affected.processor_skip.add("bimodal_normalize")
    good = source_track(tmp_path, "Good.wav", 0.125 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate), rate)
    good.processor_skip.add("bimodal_normalize")
    original = affected.audio_data.copy()
    corrected = render_corrected(original, rate, RepairRecipe(dc_remove=True))
    minimum_db = 20 * math.log10(np.max(corrected) / (1 - 2 ** -15))
    assert minimum_db > 4.0
    pipeline, session = analyze([affected, good])
    pipeline.plan(session)
    output = tmp_path / "render"
    output.mkdir()
    previous = output / affected.filename
    sf.write(previous, [0.125, -0.0625], rate, subtype="PCM_16")
    previous_bytes = previous.read_bytes()

    render = getattr(pipeline, writer)
    render(session, str(output))

    assert previous.read_bytes() == previous_bytes
    np.testing.assert_array_equal(sf.read(output / good.filename)[0], good.audio_data)
    np.testing.assert_array_equal(affected.audio_data, original)
    errors = session.config.get(f"_{writer}_errors", [])
    messages = " ".join([affected.status] + [str(error) for name, error in errors if name == affected.filename])
    db_values = [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*dB\b", messages, re.I)]
    candidates = [value for value in db_values if minimum_db - 1e-7 <= value <= minimum_db + 0.101]
    assert candidates, messages

    affected.manual_gain_db = -candidates[0]
    pipeline.plan(session)
    render(session, str(output))
    expected = corrected * 10 ** (affected.manual_gain_db / 20)
    np.testing.assert_allclose(sf.read(previous)[0], expected, atol=2 ** -15)
    if writer == "prepare":
        session.config["_use_processed"] = True
        published = next(t for t in session.output_tracks if t.filename == affected.filename)
        media = resolve_transfer_audio(session, published)
        assert Path(media.filepath) == previous
        assert media.fader_db == 0.0

    # The explicit output gain is applied once when normalization is selected, too.
    affected.processor_skip.discard("bimodal_normalize")
    pipeline.plan(session)
    render(session, str(output))
    assert affected.processor_results["bimodal_normalize"].gain_db == pytest.approx(affected.manual_gain_db)
    np.testing.assert_allclose(sf.read(previous)[0], expected, atol=2 ** -15)
    assert sf.info(previous).subtype == "PCM_16"


def test_prepare_checks_each_routed_output_after_summing_sources(tmp_path):
    rate = 48000
    program = 0.7 * np.sin(2 * np.pi * 400 * np.arange(rate) / rate)
    tracks = [source_track(tmp_path, name, program, rate, subtype="PCM_16") for name in ["Left.wav", "Right.wav"]]
    for track in tracks:
        track.repair_overrides = {"dc_remove": True}
        track.processor_skip.add("bimodal_normalize")
    pipeline, session = analyze(tracks)
    pipeline.plan(session)
    session.topology = TopologyMapping([
        TopologyEntry(name, 1, [TopologySource(t.filename, [ChannelRoute(0, 0, gain)]) for t in tracks])
        for name, gain in [("Average.wav", 0.5), ("Sum.wav", 1.0)]
    ])
    output = tmp_path / "prepared"
    output.mkdir()
    old_sum = output / "Sum.wav"
    sf.write(old_sum, [-0.25, 0.25], rate, subtype="PCM_16")
    old_bytes = old_sum.read_bytes()

    pipeline.prepare(session, str(output))

    assert old_sum.read_bytes() == old_bytes
    assert any(name == "Sum.wav" for name, _ in session.config["_prepare_errors"])
    expected = render_corrected(tracks[0].audio_data, rate, RepairRecipe(dc_remove=True))
    np.testing.assert_allclose(sf.read(output / "Average.wav")[0], expected, atol=2 ** -15)


def test_repeat_prepare_restores_a_replaced_routed_original(tmp_path):
    rate = 48000
    audio = 0.18 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate)
    track = source_track(tmp_path, "Audio.wav", audio, rate, subtype="PCM_16")
    pipeline, session = analyze([track])
    pipeline.plan(session)
    session.topology = TopologyMapping([
        TopologyEntry("Half.wav", 1, [TopologySource(track.filename, [ChannelRoute(0, 0, 0.5)])])
    ])
    output = tmp_path / "prepared"
    pipeline.prepare(session, str(output))
    session.config["_use_processed"] = False
    dry = Path(resolve_transfer_audio(session, session.output_tracks[0]).filepath)
    original = sf.read(dry)[0]
    np.testing.assert_allclose(original, track.audio_data * 0.5)
    sf.write(dry, np.full(len(original), 0.8), rate, subtype="FLOAT")
    stamp = session.output_tracks[0].source_file_signature[1] + 2_000_000_000
    os.utime(dry, ns=(stamp, stamp))
    with pytest.raises(ValueError, match="changed"):
        resolve_transfer_audio(session, session.output_tracks[0])

    pipeline.prepare(session, str(output))
    current = resolve_transfer_audio(session, session.output_tracks[0])
    assert session.prepare_state == "ready" and current.fader_db == 0
    np.testing.assert_array_equal(sf.read(current.filepath)[0], original)


def test_prepare_freezes_settling_on_the_actual_routed_channels_before_gain(tmp_path):
    from sessionpreplib.repair_plan import plan_repairs

    rate = 48000
    silence = np.zeros(rate)
    pulse = silence.copy()
    pulse[23998:24000] = 0.5
    a = source_track(tmp_path, "Silence.wav", silence, rate)
    b = source_track(tmp_path, "Pulse.wav", pulse, rate)
    stereo = source_track(tmp_path, "Stereo.wav", np.column_stack((silence, pulse)), rate)
    for track in (b, stereo):
        track.repair_overrides = {"low_cut": True, "low_cut_hz": 40}
        track.manual_gain_db = -6.0
        plan_repairs(track, {})
    routes = TopologyMapping([
        TopologyEntry("Merged.wav", 2, [TopologySource(a.filename, [ChannelRoute(0, 0)]),
                                        TopologySource(b.filename, [ChannelRoute(0, 1)])]),
        TopologyEntry("Right.wav", 1, [TopologySource(stereo.filename, [ChannelRoute(1, 0)])]),
        TopologyEntry("Left.wav", 1, [TopologySource(stereo.filename, [ChannelRoute(0, 0)])]),
        TopologyEntry("Swapped.wav", 2, [TopologySource(stereo.filename,
                                                       [ChannelRoute(1, 0), ChannelRoute(0, 1)])]),
        TopologyEntry("Cancel.wav", 1, [TopologySource(stereo.filename, [ChannelRoute(1, 0)]),
                                        TopologySource(b.filename, [ChannelRoute(0, 0, -1.0)])]),
    ])
    session = SessionContext([a, b, stereo], {}, topology=routes)
    Pipeline([]).prepare(session, str(tmp_path / "out"))
    assert session.prepare_state == "ready", session.config["_prepare_errors"]
    for track, ch in zip(session.output_tracks, [1, 0, None, 0, None], strict=True):
        gaps = track.repair_measurements.get("boundary_settling", [])
        if ch is None:
            assert not gaps
            continue
        gap, = gaps
        samples, _ = sf.read(track.processed_filepath, always_2d=True)
        untrimmed = samples[24000:, ch] * 10 ** (6 / 20)
        audible = np.flatnonzero(np.abs(untrimmed) > 1e-6)
        assert (gap["channel"], gap["start_sample"], gap["end_sample"]) == (
            ch, 24000 + audible[0], 24000 + audible[-1])
        assert gap["peak_dbfs"] == pytest.approx(20 * np.log10(np.abs(untrimmed).max()), abs=1e-5)
    assert stereo.repair_measurements["boundary_settling"][0]["channel"] == 1

    # A library caller can bypass a repair while retaining its earlier analysis.
    # The success receipt must describe the new write for a direct output too.
    stereo.repair_recipe = RepairRecipe()
    session.topology = TopologyMapping([TopologyEntry("Direct.wav", 2, [TopologySource(
        stereo.filename, [ChannelRoute(0, 0), ChannelRoute(1, 1)])])])
    Pipeline([]).prepare(session, str(tmp_path / "direct"))
    assert session.prepare_state == "ready"
    direct = session.output_tracks[0].repair_measurements
    assert not direct.get("boundary_settling")
    assert direct["corrected"] == direct["original"]


def test_multi_source_prepare_only_publishes_a_representable_fader(tmp_path):
    rate = 48000
    time = np.arange(rate) / rate
    tracks = [source_track(tmp_path, f"Stem{i}.wav", 0.05 * np.sin(2 * np.pi * hz * time), rate)
              for i, hz in enumerate([330, 550])]
    pipeline, session = analyze(tracks, fader_headroom_db=0)
    routes = [TopologySource(track.filename, [ChannelRoute(0, 0)]) for track in tracks]
    session.topology = TopologyMapping([TopologyEntry("Mix.wav", 1, routes)])
    target = tmp_path / "prepared"
    for track, gain in zip(tracks, [6.0, 12.0]):
        track.manual_gain_db = gain
    pipeline.plan(session)
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "none" and not session.output_tracks
    assert "fader" in dict(session.config["_prepare_errors"])["Mix.wav"].lower()
    assert not list(target.rglob("*.wav"))

    tracks[1].manual_gain_db = 6.0
    pipeline.plan(session)
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "ready"
    previous = session.output_tracks[0]
    transfer = resolve_transfer_audio(session, previous)
    assert transfer.fader_db == -6.0
    recovered = sf.read(transfer.filepath)[0] * 10 ** (transfer.fader_db / 20)
    np.testing.assert_allclose(recovered, tracks[0].audio_data + tracks[1].audio_data, atol=1e-7)
    saved = Path(transfer.filepath).read_bytes()

    tracks[1].manual_gain_db = 12.0
    pipeline.plan(session)
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "stale"
    assert session.output_tracks == [previous] and not session.config["_prepared_files"]
    assert Path(transfer.filepath).read_bytes() == saved
    with pytest.raises(ValueError, match="out of date"):
        resolve_transfer_audio(session, previous)

    routes[1].routes[0].gain = 0.0
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "ready"
    assert resolve_transfer_audio(session, session.output_tracks[0]).fader_db == -6.0

    # The silence may be confined to the particular channel that is routed.
    partial = source_track(tmp_path, "Stereo.wav", np.column_stack([time / 8, np.zeros(rate)]), rate)
    partial.manual_gain_db = 12.0
    session.tracks.append(partial)
    routes[1] = TopologySource(partial.filename, [ChannelRoute(1, 0)])
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "ready", session.config["_prepare_errors"]
    assert resolve_transfer_audio(session, session.output_tracks[0]).fader_db == -6.0


@pytest.mark.parametrize("bad_fader", [float("nan"), float("inf")])
def test_prepare_rejects_nonfinite_fader_before_replacing_media(tmp_path, bad_fader):
    rate = 48000
    track = source_track(tmp_path, "Audio.wav", 0.18 * np.sin(np.arange(rate) * 0.05), rate)
    pipeline, session = analyze([track])
    pipeline.plan(session)
    output = tmp_path / "prepared"
    pipeline.prepare(session, str(output))
    previous = session.output_tracks[0]
    before = Path(previous.processed_filepath).read_bytes()
    track.processor_results["bimodal_normalize"].data["fader_offset"] = bad_fader

    pipeline.prepare(session, str(output))
    assert session.prepare_state == "stale"
    assert session.config["_prepared_files"] == []
    assert "finite" in session.config["_prepare_errors"][0][1]
    assert session.output_tracks[0] is previous
    assert Path(previous.processed_filepath).read_bytes() == before


@pytest.mark.parametrize("writer", ["prepare", "execute"])
def test_mid_write_error_never_replaces_the_last_valid_output(tmp_path, monkeypatch, writer):
    rate = 48000
    time = np.arange(rate) / rate
    track = source_track(tmp_path, "Signal.wav", np.sin(2 * np.pi * 440 * time) * 0.25, rate)
    track.repair_overrides = {"dc_remove": True}
    track.processor_skip.add("bimodal_normalize")
    pipeline, session = analyze([track])
    pipeline.plan(session)
    output = tmp_path / "render"
    output.mkdir()
    destination = output / track.filename
    sf.write(destination, [0.25, 0.5], rate, subtype="FLOAT")
    previous_bytes = destination.read_bytes()
    native_write = sf.SoundFile.write

    def fail_after_prefix(file, data):
        native_write(file, data[:13])
        raise OSError("injected write failure after thirteen frames")

    render = getattr(pipeline, writer)
    with monkeypatch.context() as scoped:
        scoped.setattr(sf.SoundFile, "write", fail_after_prefix)
        render(session, str(output))
    assert destination.read_bytes() == previous_bytes

    render(session, str(output))
    expected = render_corrected(track.audio_data, rate, RepairRecipe(dc_remove=True))
    np.testing.assert_allclose(sf.read(destination)[0], expected, atol=1e-7)


def test_cli_repeated_fixes_select_the_corresponding_tracks_and_respect_dry_run(tmp_path):
    rate = 96000
    time = np.arange(3 * rate) / rate
    program = 0.15 * np.sin(2 * np.pi * 440 * time)
    program[:rate // 2] = 0
    tracks_and_recipes = [
        (source_track(tmp_path, "Bias.wav", program + 0.05, rate), RepairRecipe(dc_remove=True)),
        (source_track(tmp_path, "Rumble.wav", program + 0.21 * np.sin(2 * np.pi * 8 * time), rate),
         RepairRecipe(low_cut_hz=16.5)),
        (source_track(tmp_path, "Carrier.wav", program + 0.16 * np.sin(2 * np.pi * 30000 * time), rate),
         RepairRecipe(ultrasonic=True)),
        (source_track(tmp_path, "Clean.wav", program, rate), RepairRecipe()),
    ]
    options = ["--fix", "dc", "--fix", "subsonic", "--fix", "ultrasonic",
               "--low-cut-hz", "16.5", "--force_transient", "*"]
    output = tmp_path / "processed"
    for execute in [False, True]:
        result = run_cli(tmp_path, *options, *(["--execute"] if execute else []))
        assert result.returncode == 0, result.stderr + result.stdout
        assert output.exists() == execute

    for track, recipe in tracks_and_recipes:
        corrected = render_corrected(track.audio_data, rate, recipe)
        expected = corrected * (10 ** (-6 / 20) / np.max(np.abs(corrected)))
        prepared, saved_rate = sf.read(output / track.filename)
        assert saved_rate == rate
        np.testing.assert_allclose(prepared, expected, rtol=1e-6, atol=1e-7, err_msg=track.filename)


def test_cli_reports_pcm_overflow_as_failure_and_preserves_previous_file_for_retry(tmp_path):
    rate = 48000
    step = np.full(3 * rate, -0.875)
    step[7 * rate // 4:] = 0.75
    track = source_track(tmp_path, "Hot.wav", step, rate, subtype="PCM_16")
    source_bytes = Path(track.filepath).read_bytes()
    output = tmp_path / "processed"
    output.mkdir()
    previous = output / track.filename
    sf.write(previous, [0.125, -0.0625], rate, subtype="PCM_16")
    old_bytes = previous.read_bytes()
    options = ["--fix", "dc", "--force_transient", "*", "--execute", "--target_peak"]

    # DC removal makes the highest peak positive. +1.0 cannot be encoded by signed PCM.
    failed = run_cli(tmp_path, *options, "0")

    report = failed.stdout + failed.stderr
    assert failed.returncode != 0
    assert "Hot.wav" in report and "PCM_16" in report and re.search(r"0\.1\s*dB\b", report), report
    assert previous.read_bytes() == old_bytes

    retry = run_cli(tmp_path, *options, "-0.2")
    assert retry.returncode == 0, retry.stderr + retry.stdout
    corrected = render_corrected(track.audio_data, rate, RepairRecipe(dc_remove=True))
    expected = corrected * (10 ** (-0.2 / 20) / np.max(np.abs(corrected)))
    np.testing.assert_allclose(sf.read(previous)[0], expected, atol=2 ** -15)
    assert Path(track.filepath).read_bytes() == source_bytes
