"""Phase 1 must analyze each source's decoded samples before suggesting a layout."""

import time

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from sessionprepgui.analysis.worker import AudioLoadWorker, Phase1AnalyzeWorker, TopologyApplyWorker
from sessionprepgui.session.io import capture_phase1_sources
from sessionpreplib.audio import discover_track, file_revision, get_peak, load_track
from sessionpreplib.models import SessionContext
from sessionpreplib.topology import (
    ChannelRoute, TopologyEntry, TopologyMapping, TopologySource, build_default_topology,
    extract_channel, passthrough_routes,
)


def _write_stereo(path, *, identical=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    left = np.tile([0.25, -0.125], 256)
    sf.write(path, np.column_stack((left, left if identical else -left)), 48_000,
             subtype="PCM_24")


def _rewrite_past_revision_tick(path, write_fn):
    """Rewrite *path* in place until its filesystem revision advances past a coarse clock tick."""
    before_rev = file_revision(str(path), path.stat())
    deadline = time.monotonic() + 1.0
    while True:
        write_fn()
        if file_revision(str(path), path.stat()) != before_rev:
            return
        assert time.monotonic() < deadline, f"file_revision did not advance for {path}"
        time.sleep(0.005)


def _analyze(tracks):
    session = SessionContext(tracks=tracks, config={})
    worker = Phase1AnalyzeWorker(session, {})
    errors = []
    completed = []
    worker.error.connect(errors.append)
    worker.completed.connect(completed.append)
    worker.run()
    assert errors == []
    assert completed == [session]
    return build_default_topology(tracks).entries


def test_fresh_directory_decodes_identical_stereo_and_routes_one_channel(tmp_path):
    path = tmp_path / "Keys.wav"
    _write_stereo(path)
    source = discover_track(str(path))

    (output,) = _analyze([source])

    assert source.detector_results["dual_mono"].data["dual_mono"] is True
    assert output.output_channels == 1
    assert output.sources[0].input_filename == source.filename
    assert output.sources[0].routes == extract_channel(0)


def test_preview_loaded_track_forgets_measurements_taken_before_decode(tmp_path):
    path = tmp_path / "Keys.wav"
    _write_stereo(path)
    track = discover_track(str(path))
    assert get_peak(track) == 0.0
    preview = AudioLoadWorker(track)
    preview.run()
    preview_samples = track.audio_data
    assert preview_samples is not None

    (output,) = _analyze([track])

    assert track.audio_data is preview_samples
    assert get_peak(track) == 0.25
    assert output.sources[0].routes == extract_channel(0)


@pytest.mark.parametrize("preview_first", [False, True])
def test_decoded_metadata_and_source_identity_match_replaced_file(tmp_path, preview_first):
    path = tmp_path / "Take.wav"
    sf.write(path, np.ones(20) * 0.2, 44_100, subtype="PCM_16")
    discovered = discover_track(str(path))
    _write_stereo(path, identical=False)
    decoded = load_track(str(path))
    if preview_first:
        AudioLoadWorker(discovered).run()

    (output,) = _analyze([discovered])

    assert discovered.channels == decoded.channels == 2
    assert discovered.samplerate == decoded.samplerate
    assert discovered.total_samples == decoded.total_samples
    assert discovered.duration_sec == decoded.duration_sec
    assert discovered.subtype == decoded.subtype and discovered.bitdepth == decoded.bitdepth
    assert discovered.chunk_ids == decoded.chunk_ids
    assert discovered.source_file_signature == decoded.source_file_signature
    assert discovered.source_file_revision == decoded.source_file_revision
    assert discovered.source_sample_digest == decoded.source_sample_digest
    assert output.sources[0].routes == passthrough_routes(2)
    assert discovered.filename in capture_phase1_sources([discovered])


def test_phase1_rechecks_a_known_source_replaced_after_preview(tmp_path):
    path = tmp_path / "Take.wav"
    _write_stereo(path)
    track = discover_track(str(path))
    AudioLoadWorker(track).run()
    _rewrite_past_revision_tick(path, lambda: _write_stereo(path, identical=False))
    (output,) = _analyze([track])
    assert output.sources[0].routes == passthrough_routes(2)
    assert track.source_sample_digest == load_track(str(path)).source_sample_digest


def test_preview_cannot_certify_old_findings_against_a_replaced_source(tmp_path):
    path = tmp_path / "Take.wav"
    _write_stereo(path)
    track = discover_track(str(path))
    _analyze([track])
    old_revision = track.source_file_revision
    track.audio_data = None
    sf.write(path, np.linspace(-0.2, 0.2, 101), 44100, subtype="PCM_16")
    preview = AudioLoadWorker(track)
    errors = []
    preview.error.connect(errors.append)
    preview.run()
    assert any("changed" in msg for msg in errors)
    assert track.audio_data is None and track.source_file_revision == old_revision


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_float_source_cannot_certify_a_layout_or_hide_a_good_source(tmp_path, invalid):
    bad_path, good_path = tmp_path / "Bad.wav", tmp_path / "Good.wav"
    stereo = np.tile([[0.125, -0.25], [-0.125, 0.25]], (1000, 1))
    stereo[601, 0] = invalid
    sf.write(bad_path, stereo, 48000, subtype="FLOAT")
    _write_stereo(good_path)
    bad, good = (discover_track(str(path)) for path in (bad_path, good_path))
    outputs = {entry.output_filename: entry for entry in _analyze([bad, good])}
    assert "non-finite" in bad.detector_results["silence"].error
    assert all(bad.detector_results[key].error for key in ("dual_mono", "one_sided_silence"))
    assert outputs[bad.filename].sources[0].routes == passthrough_routes(2)
    assert outputs[good.filename].sources[0].routes == extract_channel(0)
    assert set(capture_phase1_sources([bad, good])) == {good.filename}


@pytest.mark.parametrize("loaded_first", [True, False], ids=["loaded-first", "unloaded-first"])
def test_partial_loading_keeps_each_source_and_its_relative_name(tmp_path, loaded_first):
    stereo_path = tmp_path / "band" / "Keys.wav"
    duplicate_path = tmp_path / "overdubs" / "Keys.wav"
    _write_stereo(stereo_path, identical=False)
    _write_stereo(duplicate_path)
    stereo = load_track(str(stereo_path))
    duplicate = discover_track(str(duplicate_path))
    stereo.filename = str(stereo_path.relative_to(tmp_path))
    duplicate.filename = str(duplicate_path.relative_to(tmp_path))
    decoded_stereo = stereo.audio_data
    assert get_peak(duplicate) == 0.0

    tracks = [stereo, duplicate] if loaded_first else [duplicate, stereo]
    outputs = {entry.output_filename: entry for entry in _analyze(tracks)}

    assert duplicate.detector_results["dual_mono"].data["dual_mono"] is True
    assert duplicate.detector_results["silence"].data["is_silent"] is False
    assert get_peak(duplicate) == 0.25
    assert stereo.detector_results["dual_mono"].data["dual_mono"] is False
    assert stereo.audio_data is decoded_stereo
    assert set(outputs) == {stereo.filename, duplicate.filename}
    assert outputs[stereo.filename].sources[0].input_filename == stereo.filename
    assert outputs[stereo.filename].sources[0].routes == passthrough_routes(2)
    assert outputs[duplicate.filename].sources[0].input_filename == duplicate.filename
    assert outputs[duplicate.filename].sources[0].routes == extract_channel(0)


@pytest.mark.parametrize("damage", ["deleted", "corrupt"])
def test_unreadable_source_is_isolated_and_recovers_on_reanalysis(tmp_path, damage):
    ready_path = tmp_path / "Drums.wav"
    damaged_path = tmp_path / "Strings.wav"
    good_path = tmp_path / "Keys.wav"
    for path in (ready_path, damaged_path, good_path):
        _write_stereo(path)
    ready = load_track(str(ready_path))
    damaged = discover_track(str(damaged_path))
    good = discover_track(str(good_path))
    if damage == "deleted":
        damaged_path.unlink()
    else:
        damaged_path.write_bytes(b"invalid audio")

    outputs = _analyze([ready, damaged, good])

    assert damaged.status.startswith("Error: ") and len(damaged.status) > len("Error: ")
    assert not hasattr(damaged, "error")
    assert not damaged.detector_results
    assert [output.output_filename for output in outputs] == [ready.filename, good.filename]
    assert all(output.output_channels == 1 for output in outputs)

    _write_stereo(damaged_path)
    rediscovered = discover_track(str(damaged_path))
    (recovered,) = _analyze([rediscovered])
    assert rediscovered.status == "OK"
    assert rediscovered.detector_results["dual_mono"].data["dual_mono"] is True
    assert recovered.sources[0].routes == extract_channel(0)


def _apply(session, output_dir, source_dir=None, **kwargs):
    worker = TopologyApplyWorker(session, str(output_dir),
                                 str(source_dir) if source_dir else None, **kwargs)
    errors = []
    completed = []
    worker.error.connect(lambda worker, message: errors.append(message))
    worker.apply_finished.connect(lambda worker: completed.append(True))
    worker.run()
    if completed:
        assert completed == [True]
        errors.extend(f"{name}: {message}" for name, message in
                      session.config["_topology_apply_errors"])
    else:
        assert errors
    return errors


def _route_one_channel(filename, channel):
    return TopologyMapping([TopologyEntry(filename, 1, [
        TopologySource(filename, extract_channel(channel)),
    ])])


@pytest.mark.parametrize("failure", ["encoder", "cancel", "publish"])
def test_layout_retry_preserves_previous_files_and_unrelated_media(tmp_path, monkeypatch, failure):
    from pathlib import Path
    import os

    layout = tmp_path / "layout"
    for name in ["A.wav", "B.wav"]:
        _write_stereo(tmp_path / name, identical=False)
    tracks = [load_track(str(tmp_path / name)) for name in ["A.wav", "B.wav"]]
    session = SessionContext(tracks, {}, topology=TopologyMapping([
        _route_one_channel(t.filename, 0).entries[0] for t in tracks]))
    assert _apply(session, layout, tmp_path) == []
    sentinel = layout / "Notes.wav"
    _write_stereo(sentinel)
    files = [layout / t.filename for t in tracks] + [sentinel] + [Path(t.filepath) for t in tracks]
    previous = {path: path.read_bytes() for path in files}
    for entry in session.topology.entries:
        entry.sources[0].routes = extract_channel(1)
    worker = TopologyApplyWorker(session, str(layout), str(tmp_path))
    errors, finished = [], []
    worker.error.connect(lambda worker, message: errors.append(message))
    worker.apply_finished.connect(lambda worker: finished.append(True))
    write, publish = sf.write, os.replace

    def fail_second_encoder(path, *args, **kwargs):
        if Path(path).name == "B.wav":
            raise OSError("encoder failed")
        write(path, *args, **kwargs)
        if failure == "cancel":
            worker.cancel()

    def fail_second_publication(source, dest):
        if Path(dest) == layout / "B.wav" and Path(source).name == "B.wav":
            raise OSError("locked destination")
        publish(source, dest)

    with monkeypatch.context() as fault:
        if failure == "publish":
            fault.setattr(os, "replace", fail_second_publication)
        else:
            fault.setattr(sf, "write", fail_second_encoder)
        worker.run()
    assert errors and not finished
    assert {path: path.read_bytes() for path in files} == previous
    assert _apply(session, layout, tmp_path) == []
    for track in tracks:
        np.testing.assert_array_equal(sf.read(layout / track.filename)[0], track.audio_data[:, 1])
    assert sentinel.read_bytes() == previous[sentinel]


def test_apply_then_reapply_right_channel_of_detected_duplicate(tmp_path):
    source = tmp_path / "Keys.wav"
    layout = tmp_path / "layout"
    _write_stereo(source)
    original = discover_track(str(source))
    entries = _analyze([original])
    session = SessionContext([original], {}, topology=TopologyMapping(entries))
    assert _apply(session, layout, tmp_path) == []
    np.testing.assert_array_equal(sf.read(layout / source.name)[0], original.audio_data[:, 0])

    # After Apply the normal Phase 2 analysis populates the session with rendered outputs.
    session.tracks = [load_track(str(layout / source.name))]
    session.topology = _route_one_channel(source.name, 1)
    assert _apply(session, layout, tmp_path) == []
    np.testing.assert_array_equal(sf.read(layout / source.name)[0], original.audio_data[:, 1])


@pytest.mark.parametrize("replacement", ["unrecognized", "modified", "unchanged"])
def test_apply_from_a_new_session_verifies_existing_destinations(tmp_path, replacement):
    layout = tmp_path / "layout"
    path = tmp_path / "Keys.wav"
    _write_stereo(path, identical=False)
    target = layout / "Keys.wav"
    track = load_track(str(path))
    if replacement != "unrecognized":
        first = SessionContext([track], {}, topology=_route_one_channel(path.name, 0))
        assert _apply(first, layout, tmp_path) == []
    if replacement != "unchanged":
        layout.mkdir(exist_ok=True)
        sf.write(target, np.full(512, 0.1875), 48000, subtype="PCM_24")
    before = target.read_bytes()
    next_session = SessionContext([track], {}, topology=_route_one_channel(path.name, 1))
    errors = _apply(next_session, layout, tmp_path)
    if replacement == "unchanged":
        assert not errors
        np.testing.assert_array_equal(sf.read(target)[0], track.audio_data[:, 1])
    else:
        assert errors
        assert target.read_bytes() == before


@pytest.mark.parametrize("absolute", [False, True])
def test_apply_rejects_input_references_outside_restored_source_directory(tmp_path, absolute):
    folder = tmp_path / "project"
    folder.mkdir()
    foreign = tmp_path / "External.wav"
    _write_stereo(foreign)
    topology = _route_one_channel(foreign.name, 0)
    topology.entries[0].sources[0].input_filename = str(foreign) if absolute else "../External.wav"
    session = SessionContext([], {}, topology=topology)
    assert _apply(session, folder / "layout", folder)
    assert not (folder / "layout" / foreign.name).exists()


def test_cancelling_peak_cache_after_commit_keeps_the_complete_layout(tmp_path, monkeypatch):
    from sessionprepgui.waveform import peakcache

    for name in ("Kick.wav", "Snare.wav"):
        _write_stereo(tmp_path / name, identical=False)
    tracks = [load_track(str(tmp_path / name)) for name in ("Kick.wav", "Snare.wav")]
    session = SessionContext(tracks, {}, topology=TopologyMapping([
        _route_one_channel(track.filename, 1).entries[0] for track in tracks]), prepare_state="ready")
    output = tmp_path / "layout"
    worker = TopologyApplyWorker(session, str(output), str(tmp_path), str(output / "peaks"))
    built = []
    build_peaks = peakcache.build_peaks

    def cancel_after_first(*args, **kwargs):
        built.append(True)
        worker.cancel()
        return build_peaks(*args, **kwargs)

    monkeypatch.setattr(peakcache, "build_peaks", cancel_after_first)
    finished, errors = [], []
    worker.apply_finished.connect(finished.append)
    worker.error.connect(lambda _, error: errors.append(error))
    worker.run()
    assert finished == [worker] and not errors and built == [True]
    assert session.prepare_state == "none"
    assert [t.filename for t in session.output_tracks] == [t.filename for t in tracks]
    for track in tracks:
        np.testing.assert_array_equal(sf.read(output / track.filename)[0], track.audio_data[:, 1])


def test_concurrent_layout_publish_cannot_interleave_or_remove_other_generations(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    import os
    from pathlib import Path
    from threading import Event
    from sessionpreplib.layout_files import publish_layout

    root = tmp_path / "layout"
    root.mkdir()
    stages = [tmp_path / label for label in ("first", "second")]
    target = root / "Take.wav"
    for stage, level in zip(stages, (0.125, 0.25), strict=True):
        stage.mkdir()
        sf.write(stage / target.name, np.full(16, level), 8000)
    track = replace(discover_track(str(stages[0] / target.name)), filepath=str(target))
    publishing, proceed = Event(), Event()
    rename = os.replace

    def hold_first(source, dest):
        if Path(source) == stages[0] / target.name:
            publishing.set()
            assert proceed.wait(5)
        rename(source, dest)

    monkeypatch.setattr(os, "replace", hold_first)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(publish_layout, root, stages[0], [track], [])
        try:
            assert publishing.wait(5)
            with pytest.raises(OSError, match="lock"):
                publish_layout(root, stages[1], [replace(track)], [])
        finally:
            proceed.set()
        first.result(timeout=5)
    np.testing.assert_array_equal(sf.read(target)[0], np.full(16, 0.125))
    publish_layout(root, stages[1], [replace(track)], [])
    np.testing.assert_array_equal(sf.read(target)[0], np.full(16, 0.25))


@pytest.mark.parametrize("hydrated_output", [True, False])
def test_apply_uses_original_format_and_pcm_for_same_named_output(tmp_path, hydrated_output):
    source = tmp_path / "Bass.wav"
    layout = tmp_path / "layout"
    layout.mkdir()
    left = np.array([0.005, 0.003, -0.002, -0.005])
    sf.write(source, np.column_stack((left, -left)), 96_000, subtype="DOUBLE")
    render = layout / source.name
    sf.write(render, np.zeros(4), 44_100, subtype="PCM_16")
    previous = (load_track if hydrated_output else discover_track)(str(render))
    prior_samples = previous.audio_data
    session = SessionContext([previous], {}, topology=_route_one_channel(source.name, 1))

    assert _apply(session, layout, tmp_path) == []

    actual, rate = sf.read(render)
    assert rate == 96_000
    assert sf.info(render).subtype == "DOUBLE"
    np.testing.assert_array_equal(actual, -left)
    assert previous.audio_data is prior_samples
    assert previous.samplerate == 44_100
    (output,) = session.output_tracks
    assert (output.subtype, output.bitdepth, output.samplerate) == ("DOUBLE", "64-bit Float", 96_000)


@pytest.mark.parametrize("failure", ["missing-original", "unavailable-channel", "replaced-original"])
def test_apply_rejects_unusable_source_before_replacing_the_last_layout(tmp_path, failure):
    source = tmp_path / "Keys.wav"
    layout = tmp_path / "layout"
    _write_stereo(source)
    original = load_track(str(source))
    session = SessionContext([original], {}, topology=_route_one_channel(source.name, 0))
    assert _apply(session, layout, tmp_path) == []
    previous = (layout / source.name).read_bytes()
    session.topology = _route_one_channel(source.name, 1)
    if failure == "missing-original":
        session.tracks = [load_track(str(layout / source.name))]
        source.unlink()
    elif failure == "unavailable-channel":
        session.topology = _route_one_channel(source.name, 2)
    else:
        sf.write(source, np.zeros(4096), 48_000, subtype="PCM_16")

    errors = _apply(session, layout, tmp_path)

    assert any(source.name in message for message in errors)
    assert (layout / source.name).read_bytes() == previous


def test_reapply_reuses_verified_phase1_source_even_after_phase2_replaced_the_track_list(
        tmp_path, monkeypatch):
    source = tmp_path / "Keys.wav"
    layout = tmp_path / "layout"
    layout.mkdir()
    _write_stereo(source, identical=False)
    original = load_track(str(source))
    previous = layout / source.name
    sf.write(previous, original.audio_data[:, 0], 48_000, subtype="PCM_24")
    session = SessionContext([load_track(str(previous))], {}, topology=_route_one_channel(source.name, 1))

    def unexpected_decode(path):
        pytest.fail(f"Redecoded unchanged original: {path}")

    monkeypatch.setattr("sessionpreplib.audio.load_track", unexpected_decode)
    assert _apply(session, layout, tmp_path, source_tracks=[original]) == []
    np.testing.assert_array_equal(sf.read(previous)[0], original.audio_data[:, 1])


def test_in_memory_sources_can_be_combined_without_a_source_directory(tmp_path):
    paths = [tmp_path / "Keys.wav", tmp_path / "Rim.wav"]
    for path in paths:
        _write_stereo(path, identical=False)
    keys, rim = [load_track(str(path)) for path in paths]
    for path in paths:
        path.unlink()
    topology = TopologyMapping([TopologyEntry("combined.wav", 2, [
        TopologySource(keys.filename, [ChannelRoute(1, 0)]),
        TopologySource(rim.filename, [ChannelRoute(0, 1)]),
    ])])
    session = SessionContext([keys, rim], {}, topology=topology)

    assert _apply(session, tmp_path / "layout") == []

    actual, rate = sf.read(tmp_path / "layout" / "combined.wav")
    assert rate == keys.samplerate
    np.testing.assert_array_equal(actual, np.column_stack((keys.audio_data[:, 1], rim.audio_data[:, 0])))


def test_apply_does_not_clean_an_output_folder_that_contains_the_source(tmp_path):
    path = tmp_path / "Keys.wav"
    _write_stereo(path)
    original = load_track(str(path))
    before = path.read_bytes()
    session = SessionContext([original], {}, topology=_route_one_channel(path.name, 0))

    assert _apply(session, tmp_path, tmp_path)
    assert path.read_bytes() == before
