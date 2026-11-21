"""Regression coverage for repair source identity, encoding and output retirement."""
import math
import os
import time
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib import repair_plan
from sessionpreplib.audio import file_revision, load_track
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.repairs import OutputHeadroomError, RepairRecipe, check_output_headroom
from sessionpreplib.topology import ChannelRoute, TopologyEntry, TopologySource, build_default_topology


def _rewrite_same_stamp(path, write_fn):
    """Rewrite *path* in place with identical mtime after its filesystem revision tick advances."""
    stamp = path.stat()
    before_rev = file_revision(str(path), stamp)
    deadline = time.monotonic() + 1.0
    while True:
        write_fn()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        after = path.stat()
        if file_revision(str(path), after) != before_rev:
            assert after.st_size == stamp.st_size
            return
        assert time.monotonic() < deadline, f"file_revision did not advance for {path}"
        time.sleep(0.005)


def source_session(tmp_path, subtype="PCM_16"):
    path = tmp_path / "Source.wav"
    data = np.full(16000, 0.875)
    data[8000:] *= -1
    sf.write(path, data, 8000, subtype=subtype)
    track = load_track(str(path))
    return track, SessionContext([track], {}, topology=build_default_topology([track]))


def test_prepare_reopening_failed_repair_never_publishes_dry_audio(tmp_path):
    track, session = source_session(tmp_path)
    pipeline = Pipeline([])
    output = tmp_path / "prepared"
    pipeline.prepare(session, str(output))
    destination = output / track.filename
    previous = destination.read_bytes()
    track.audio_data = None
    track.repair_overrides = {"ultrasonic": True}
    track._cache["repair_error"] = "Ultrasonic filtering requires a higher sample rate"

    pipeline.prepare(session, str(output))

    assert session.prepare_state == "stale"
    assert track._cache["repair_error"] in session.config["_prepare_errors"][0][1]
    assert session.config["_prepared_files"] == []
    assert destination.read_bytes() == previous


def test_rehydration_rejects_same_geometry_revision_changed_before_read(tmp_path, monkeypatch):
    track, _ = source_session(tmp_path)
    track.audio_data = None
    previous = track.source_file_signature

    def load_replacement(path):
        sf.write(path, np.full(track.total_samples, 0.125), track.samplerate, subtype=track.subtype)
        os.utime(path, ns=(previous[1], previous[1] + 2_000_000_000))
        return load_track(path)

    monkeypatch.setattr(repair_plan, "load_track", load_replacement)
    with pytest.raises(ValueError, match="(?i)reanalyze"):
        repair_plan.source_for_render(track)


@pytest.mark.parametrize("subtype", ["ULAW", "ALAW", "IMA_ADPCM", "MS_ADPCM"])
def test_encoded_wave_overload_preserves_previous_file(tmp_path, subtype):
    track, session = source_session(tmp_path, subtype)
    output = tmp_path / "prepared"
    pipeline = Pipeline([])
    pipeline.prepare(session, str(output))
    destination = output / track.filename
    previous = destination.read_bytes()
    track.repair_recipe = RepairRecipe(dc_remove=True)

    pipeline.prepare(session, str(output))

    assert session.prepare_state == "stale"
    assert subtype in session.config["_prepare_errors"][0][1]
    assert "Reduce the output gain" in session.config["_prepare_errors"][0][1]
    assert destination.read_bytes() == previous


def test_float_subtypes_accept_overrange_only_within_their_own_finite_range():
    largest_float = float(np.finfo(np.float32).max)
    check_output_headroom(np.array([-largest_float, 1.5, largest_float]), "FLOAT")
    check_output_headroom(np.array([-2 * largest_float, 2 * largest_float]), "DOUBLE")
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="FLOAT"):
            check_output_headroom(np.array([sign * 2 * largest_float]), "FLOAT")


@pytest.mark.parametrize("limit", [-1.0, 1 - 2 ** -15])
def test_minimum_trim_at_and_one_ulp_above_a_tenth_db(limit):
    boundary = limit * 10 ** (0.1 / 20)
    for sample, expected in [(boundary, 0.1), (np.nextafter(boundary, 2 * limit), 0.2)]:
        with pytest.raises(OutputHeadroomError) as error:
            check_output_headroom(np.array([sample]), "PCM_16")
        assert error.value.reduction_db == expected


def test_peak_at_negative_full_scale_recovers_with_one_tenth_db_on_disk(tmp_path):
    path = tmp_path / "Negative.wav"
    sf.write(path, [-1.0, 0.0], 48000, subtype="PCM_16")
    track = load_track(str(path))
    track.manual_gain_db = 0.1
    session = SessionContext([track], {})
    pipeline = Pipeline([])
    target = tmp_path / "prepared"
    pipeline.prepare(session, str(target))
    adjustment = session.config["_output_gain_adjustments"][track.filename]
    assert adjustment["reduction_db"] == 0.1
    track.manual_gain_db -= adjustment["reduction_db"]
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "ready"
    np.testing.assert_array_equal(sf.read(target / track.filename)[0], [-1.0, 0.0])


def test_minimum_trim_matches_the_rendered_gain_at_every_tenth():
    for step in range(1, 241):
        gain = step / 10
        for limit in (-1.0, 1 - 2 ** -15):
            sample = limit * 10 ** (gain / 20)
            with pytest.raises(OutputHeadroomError) as exact:
                check_output_headroom(np.array([sample]), "PCM_16")
            assert exact.value.reduction_db == gain
            with pytest.raises(OutputHeadroomError) as above:
                check_output_headroom(np.array([np.nextafter(sample, 2 * sample)]), "PCM_16")
            assert above.value.reduction_db == (step + 1) / 10


@pytest.mark.parametrize(("subtype", "bits"), [("PCM_16", 16), ("PCM_24", 24)])
def test_adjust_gain_recovers_positive_full_scale_through_a_routing_gain(tmp_path, subtype, bits):
    path = tmp_path / "Positive.wav"
    limit = 1 - 2 ** (1 - bits)
    sf.write(path, np.full(64, limit), 48000, subtype=subtype)
    track = load_track(str(path))
    topology = build_default_topology([track])
    topology.entries[0].sources[0].routes[0].gain = 10 ** (0.1 / 20)
    session = SessionContext([track], {}, topology=topology)
    pipeline = Pipeline([])
    target = tmp_path / "prepared"
    pipeline.prepare(session, str(target))
    adjustment = session.config["_output_gain_adjustments"][track.filename]
    # Applying gain before route summing rounds a -0.1 dB retry one ULP above
    # positive full scale. The actual encoder must accept the offered retry.
    assert adjustment["reduction_db"] == 0.2
    assert "0.2 dB" in session.config["_prepare_errors"][0][1]
    track.manual_gain_db = adjustment["gain_db"] - adjustment["reduction_db"]
    pipeline.prepare(session, str(target))
    assert session.prepare_state == "ready"
    assert sf.read(target / path.name)[0].max() <= limit


@pytest.mark.parametrize(("subtype", "bits", "code"), [
    ("PCM_U8", 8, 87), ("PCM_16", 16, 16404),
    ("PCM_24", 24, 4194316), ("PCM_32", 32, 1073741851),
])
def test_execute_reports_the_first_gain_step_that_can_actually_be_written(tmp_path, subtype, bits, code):
    full_scale = 2 ** (bits - 1)
    source = code / full_scale
    limit = 1 - 1 / full_scale
    path = tmp_path / "Input.wav"
    sf.write(path, np.full(128, source), 48000, subtype=subtype)
    track = load_track(str(path))
    gain = 20 * math.log10(limit / source) + 0.1
    # log10 and pow can round to opposite sides of the encoder ceiling on
    # different platforms. Exercise the side where a 0.1 dB retry still clips.
    while source * 10 ** ((gain - 0.1) / 20) <= limit:
        gain = math.nextafter(gain, math.inf)
    track.manual_gain_db = gain
    session = SessionContext([track], {})
    pipeline = Pipeline([])
    output = tmp_path / "out"
    pipeline.execute(session, str(output))
    assert not session.config["_executed_files"]
    assert "at least 0.2 dB" in session.config["_execute_errors"][0][1]
    track.manual_gain_db = gain - 0.1
    pipeline.execute(session, str(output))
    assert not session.config["_executed_files"]
    assert session.config["_execute_errors"]
    track.manual_gain_db = gain - 0.2
    pipeline.execute(session, str(output))
    assert not session.config["_execute_errors"]
    assert path.name in session.config["_executed_files"]
    assert sf.read(output / path.name)[0].max() <= limit


@pytest.mark.parametrize(("starting_gain", "routing_gain", "trim", "target_gain"), [
    (4.1, 0.954992586021436, 3.8, 0.3),
    (8.3, 0.9772372209558107, 8.1, 0.2),
    (4.79, 10 ** (-4.62 / 20), 0.2, 4.6),
])
def test_routed_retry_checks_the_exact_decimal_gain_the_user_can_select(
        tmp_path, starting_gain, routing_gain, trim, target_gain):
    path = tmp_path / "Floor.wav"
    sf.write(path, [-1.0, 0.0], 48000, subtype="PCM_16")
    track = load_track(str(path))
    track.manual_gain_db = starting_gain
    topology = build_default_topology([track])
    topology.entries[0].sources[0].routes[0].gain = routing_gain
    session = SessionContext([track], {}, topology=topology)
    output = tmp_path / "out"
    pipeline = Pipeline([])
    pipeline.prepare(session, str(output))
    advice = session.config["_output_gain_adjustments"][track.filename]
    assert advice["reduction_db"] == trim
    assert advice["target_gain_db"] == target_gain
    track.manual_gain_db = round(target_gain + 0.1, 1)
    pipeline.prepare(session, str(output))
    assert not session.config["_prepared_files"]
    track.manual_gain_db = advice["target_gain_db"]
    pipeline.prepare(session, str(output))
    assert session.prepare_state == "ready"
    assert sf.read(output / path.name)[0].min() >= -1


def test_routed_dry_float_overload_is_rejected_before_publishing(tmp_path):
    path = tmp_path / "Source.wav"
    largest_float = float(np.finfo(np.float32).max)
    sf.write(path, np.full(100, largest_float), 8000, subtype="DOUBLE")
    track = load_track(str(path))
    track.manual_gain_db = -12.0
    session = SessionContext([track], {}, topology=build_default_topology([track]))
    entry = session.topology.entries[0]
    entry.output_filename = "Routed.wav"
    entry.sources[0].routes = [ChannelRoute(0, 0, 2)]
    output = tmp_path / "prepared"
    output.mkdir()
    destination = output / entry.output_filename
    sf.write(destination, [0.25, -0.25], 8000, subtype="DOUBLE")
    previous = destination.read_bytes()

    Pipeline([]).prepare(session, str(output))

    assert session.config["_prepared_files"] == []
    assert "FLOAT" in session.config["_prepare_errors"][0][1]
    assert "routing gain" in session.config["_prepare_errors"][0][1]
    assert destination.read_bytes() == previous
    assert not list((output / ".routed").rglob("*.wav"))


@pytest.mark.parametrize("routing_gains", [(4, 2), (2, 4)])
def test_fanout_offers_one_source_gain_that_recovers_all_its_outputs(tmp_path, routing_gains):
    track, session = source_session(tmp_path)
    output = tmp_path / "prepared"
    session.topology.entries = [
        TopologyEntry(f"Output{gain}.wav", 1, [TopologySource(track.filename, [ChannelRoute(0, 0, gain)])])
        for gain in routing_gains
    ]
    pipeline = Pipeline([])
    pipeline.prepare(session, str(output))
    assert len(session.config["_prepare_errors"]) == 2
    adjustment = session.config["_output_gain_adjustments"][track.filename]
    peak = 20 * np.log10(0.875 * max(routing_gains))
    assert adjustment["peak_db"] == pytest.approx(peak)
    assert adjustment["reduction_db"] == pytest.approx(10.9)

    track.manual_gain_db = adjustment["gain_db"] - adjustment["reduction_db"]
    pipeline.prepare(session, str(output))
    assert session.prepare_state == "ready" and not session.config["_prepare_errors"]
    assert set(session.config["_prepared_files"]) == {entry.output_filename for entry in session.topology.entries}
    for gain in routing_gains:
        expected = track.audio_data * gain * 10 ** (track.manual_gain_db / 20)
        np.testing.assert_allclose(sf.read(output / f"Output{gain}.wav")[0], expected, atol=2 ** -15)


def test_retirement_cannot_unlink_the_new_revision_at_the_same_path(tmp_path):
    track, session = source_session(tmp_path)
    entry = session.topology.entries[0]
    entry.output_filename = "Kick.wav"
    output = tmp_path / "prepared"
    pipeline = Pipeline([])
    pipeline.prepare(session, str(output))
    entry.output_filename = "kick.wav" if os.name == "nt" else "./Kick.wav"

    pipeline.prepare(session, str(output))

    destination = output / entry.output_filename
    assert session.prepare_state == "ready"
    np.testing.assert_array_equal(sf.read(destination)[0], track.audio_data)


@pytest.mark.parametrize("state", ["generated", "replaced", "same_stamp", "unverified", "input"])
def test_retire_only_an_owned_unchanged_output(tmp_path, state):
    track, session = source_session(tmp_path)
    entry = session.topology.entries[0]
    entry.output_filename = "Retired.wav"
    output = tmp_path / "prepared"
    pipeline = Pipeline([])
    pipeline.prepare(session, str(output))
    retired = output / entry.output_filename
    if state == "replaced":
        sf.write(retired, [0.25, -0.125], 48000, subtype="PCM_16")
    elif state == "same_stamp":
        _rewrite_same_stamp(
            retired,
            lambda: sf.write(retired, -track.audio_data, track.samplerate, subtype=track.subtype),
        )
    elif state == "unverified":
        session.output_tracks[0].prepared_file_signature = None
    elif state == "input":
        session.tracks.append(replace(track, filename="Recovered.wav", filepath=str(retired), status="Error"))
    preserved = retired.read_bytes()
    entry.output_filename = "Current.wav"

    pipeline.prepare(session, str(output))

    assert session.prepare_state == "ready"
    assert (output / "Current.wav").is_file()
    if state == "generated":
        assert not retired.exists()
    else:
        assert retired.read_bytes() == preserved


@pytest.mark.parametrize("prepared", [True, False])
def test_transfer_rejects_same_length_substitution_after_reopen(tmp_path, prepared):
    from pathlib import Path
    from sessionprepgui.session.io import save_session, load_session
    from sessionpreplib.daw_processor import resolve_transfer_audio

    track, session = source_session(tmp_path)
    # A scaled route has a separate dry render as well as a prepared render.
    session.topology.entries[0].sources[0].routes[0].gain = 0.5
    pipeline = Pipeline([])
    output = tmp_path / "prepared"
    pipeline.prepare(session, str(output))
    project = tmp_path / "mix.spsession"
    save_session(str(project), {"source_dir": str(tmp_path), "output_tracks": session.output_tracks})
    media = load_session(str(project))["output_tracks"][0]
    session.config["_use_processed"] = prepared
    path = Path(resolve_transfer_audio(session, media).filepath)
    samples, sr = sf.read(path)
    subtype = sf.info(path).subtype
    _rewrite_same_stamp(path, lambda: sf.write(path, -samples, sr, subtype=subtype))
    with pytest.raises(ValueError, match="changed"):
        resolve_transfer_audio(session, media)
    pipeline.prepare(session, str(output))
    replacement = resolve_transfer_audio(session, session.output_tracks[0]).filepath
    np.testing.assert_array_equal(sf.read(replacement)[0], samples)
