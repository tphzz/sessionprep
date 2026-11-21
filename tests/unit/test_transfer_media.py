"""Delivery uses the selected audio and the offset rendered into that revision."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
import numpy as np
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.config import default_config
from sessionpreplib.daw_processors.dawproject import DawProjectDawProcessor
from sessionpreplib.daw_processor import resolve_transfer_audio
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import RepairRecipe, render_corrected
from sessionprepgui.session.io import load_session, save_session, serialize_session_state


@pytest.fixture
def media(tmp_path):
    name = "Guitar.wav"
    source = tmp_path / "layout" / name
    prepared = tmp_path / "prepared" / name
    for path in (source, prepared):
        path.parent.mkdir()
        path.touch()
    track = TrackContext(name, str(source), None, 48000, 2, 48000, "24", "PCM_24", 1.0)
    track.processed_filepath = str(prepared)
    track.prepared_fader_db = -2.5
    track.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", 9, "Transient", "Peak", {"fader_offset": -9.0})
    track.applied_processors = ["bimodal_normalize"]
    return track


@pytest.fixture
def export_project(media, tmp_path):
    template = tmp_path / "Template.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", '<Project version="1.0"><Application name="Fixture" version="1"/>'
                         '<Structure><Track id="folder" name="Band" contentType="tracks"/></Structure></Project>')
    processor = DawProjectDawProcessor(template_path=str(template))
    session = SessionContext(
        [media], {}, output_tracks=[media], prepare_state="ready",
        transfer_manifest=[TransferEntry("guitar", media.filename, "Guitar")],
        daw_state={"dawproject": {"folders": [{"id": "folder", "name": "Band"}],
                                  "assignments": {"guitar": "folder"}}},
    )
    output = tmp_path / "Mix.dawproject"
    return output, session, lambda: processor.transfer(session, str(output))


@pytest.mark.parametrize("processed", [False, True])
def test_dawproject_references_selected_file_and_only_compensates_rendered_gain(media, export_project, processed):
    output, session, export = export_project
    session.config["_use_processed"] = processed
    results = export()
    assert all(result.success for result in results)
    expected = media.processed_filepath if processed else media.filepath
    with ZipFile(output) as archive:
        project = ElementTree.fromstring(archive.read("project.xml"))
    assert Path(project.find(".//Audio/File").get("path")) == Path(expected)
    assert float(project.find(".//Channel/Volume").get("value")) == pytest.approx(10 ** (-2.5 / 20) if processed else 1.0)


@pytest.mark.parametrize("invalid", ["missing", "stale", "replaced"])
def test_unavailable_prepared_media_never_creates_a_clip(media, export_project, invalid):
    output, session, export = export_project
    session.config["_use_processed"] = True
    path = Path(media.processed_filepath)
    stamp = path.stat()
    media.prepared_file_signature = (stamp.st_size, stamp.st_mtime_ns)
    if invalid == "missing":
        path.unlink()
    elif invalid == "stale":
        session.prepare_state = "stale"
    else:
        path.write_bytes(b"a later render")
    errors = [result for result in export() if not result.success]
    assert len(errors) == 1
    assert errors[0].command.target == "guitar"
    assert not output.exists()


@pytest.mark.parametrize("processed", [False, True])
def test_reopen_and_batch_keep_source_prepared_revision_and_repair_choices(media, tmp_path, processed):
    pytest.importorskip("PySide6")
    from sessionprepgui.batch.manager import BatchManager

    source = deepcopy(media)
    source.processed_filepath = None
    source.repair_overrides = {"dc_remove": False, "low_cut": True, "low_cut_hz": 13.5}
    source.repair_recipe = RepairRecipe(low_cut_hz=13.5)
    source.repair_measurements = {"peak_db": -8.5}
    source.manual_gain_db = -3.0
    dry_stat = Path(source.filepath).stat()
    media.source_file_signature = source.source_file_signature = (dry_stat.st_size, dry_stat.st_mtime_ns)
    media.repair_recipe = source.repair_recipe
    stat = Path(media.processed_filepath).stat()
    media.prepared_file_signature = (stat.st_size, stat.st_mtime_ns)
    path = tmp_path / "session.spsession"
    save_session(str(path), {
        "tracks": [source], "output_tracks": [media], "source_dir": str(tmp_path),
        "topology_dir": str(Path(source.filepath).parent), "topology_applied": True,
        "prepare_state": "ready", "use_processed": processed,
    })
    restored = load_session(str(path))
    assert restored["tracks"][0].filepath == source.filepath
    assert restored["tracks"][0].repair_overrides == source.repair_overrides
    assert restored["tracks"][0].manual_gain_db == -3.0
    assert restored["tracks"][0].source_file_signature == source.source_file_signature
    assert restored["tracks"][0].repair_measurements == source.repair_measurements
    assert restored["output_tracks"][0].repair_recipe == media.repair_recipe
    assert restored["output_tracks"][0].applied_processors == media.applied_processors
    assert restored["output_tracks"][0].prepared_file_signature == media.prepared_file_signature
    assert restored["output_tracks"][0].source_file_signature == media.source_file_signature
    assert "repair_overrides" not in json.loads(path.read_text())["output_tracks"][media.filename]
    queued = BatchManager(None)._rehydrate_session(restored)
    assert queued.config["_use_processed"] is processed
    transfer = resolve_transfer_audio(queued, queued.output_tracks[0])
    assert transfer.filepath == (media.processed_filepath if processed else source.filepath)
    assert transfer.fader_db == (-2.5 if processed else 0.0)
    Path(media.filepath).write_bytes(b"newly routed source")
    if processed:
        assert resolve_transfer_audio(queued, queued.output_tracks[0]) == transfer
    else:
        with pytest.raises(ValueError, match="source"):
            resolve_transfer_audio(queued, queued.output_tracks[0])


def test_v6_batch_resolves_legacy_prepared_alias_against_the_routed_layout(media, tmp_path):
    pytest.importorskip("PySide6")
    from sessionprepgui.batch.manager import BatchManager
    from sessionprepgui.session.io import deserialize_session_state

    layout = Path(media.filepath).parent
    # A same-named input is a different signal when Track Layout routes or sums channels.
    (tmp_path / media.filename).touch()
    old = serialize_session_state({"source_dir": str(tmp_path), "tracks": [media],
                                   "output_tracks": [media], "topology_applied": True,
                                   "prepare_state": "ready", "use_processed": False})
    old["version"] = 6
    old.pop("topology_dir", None)
    entry = old["output_tracks"][media.filename]
    entry["filepath"] = entry["processed_filepath"]
    for key in ("prepared_file_signature", "prepared_fader_db", "applied_processors"):
        entry.pop(key, None)
    owner = SimpleNamespace(_config={"app": {"phase1_output_folder": layout.name}})
    queue = BatchManager(owner)._rehydrate_session(deserialize_session_state(old))
    selected = resolve_transfer_audio(queue, queue.output_tracks[0])
    assert selected.filepath == media.filepath
    assert selected.fader_db == 0

    # Version 6 output snapshots omitted the source's per-track skip list.
    old["use_processed"] = True
    old["tracks"][media.filename]["processor_skip"] = ["bimodal_normalize"]
    queue = BatchManager(owner)._rehydrate_session(deserialize_session_state(old))
    selected = resolve_transfer_audio(queue, queue.output_tracks[0])
    assert selected.filepath == media.processed_filepath
    assert selected.fader_db == 0


def test_reopening_a_failed_repair_plan_keeps_its_error(media, tmp_path):
    media.repair_overrides = {"ultrasonic": True}
    media._cache["repair_error"] = "Ultrasonic filtering requires a higher sample rate"
    path = tmp_path / "session.spsession"
    save_session(str(path), {"tracks": [media], "source_dir": str(Path(media.filepath).parent),
                             "prepare_state": "stale", "prepare_errors": [[media.filename, "render failed"]]})
    saved = load_session(str(path))
    assert saved["prepare_state"] == "stale"
    assert saved["prepare_errors"] == [[media.filename, "render failed"]]
    assert saved["tracks"][0]._cache == {"repair_error": media._cache["repair_error"]}


def test_rendered_and_original_samples_survive_reopening_with_exactly_one_gain(tmp_path):
    layout = tmp_path / "Custom Layout"
    layout.mkdir()
    source_path = layout / "Voice.wav"
    rate = 48000
    time = np.arange(rate) / rate
    sf.write(source_path, 0.08 + 0.13 * np.sin(2 * np.pi * 500 * time), rate, subtype="FLOAT")
    track = load_track(str(source_path))
    track.repair_overrides = {"dc_remove": True}
    track.classification_override = "Transient"
    session = SessionContext([track], default_config())
    pipeline = Pipeline(default_detectors(), default_processors(), session.config)
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    pipeline.prepare(session, str(tmp_path / "prepared"))
    assert session.prepare_state == "ready"
    corrected = render_corrected(track.audio_data, rate, RepairRecipe(dc_remove=True))
    gain = track.processor_results["bimodal_normalize"].gain_db
    assert gain > 1.0

    path = tmp_path / "voice.spsession"
    save_session(str(path), {"source_dir": str(tmp_path), "topology_applied": True,
                             "topology_dir": str(layout), "tracks": session.tracks,
                             "output_tracks": session.output_tracks, "prepare_state": "ready"})
    saved = load_session(str(path))
    reopened = SessionContext(saved["tracks"], {"_use_processed": False}, output_tracks=saved["output_tracks"],
                              prepare_state=saved["prepare_state"])
    output, = reopened.output_tracks
    dry = resolve_transfer_audio(reopened, output)
    np.testing.assert_array_equal(sf.read(dry.filepath)[0], track.audio_data)
    assert dry.fader_db == 0.0
    reopened.config["_use_processed"] = True
    wet = resolve_transfer_audio(reopened, output)
    assert wet.fader_db == pytest.approx(-gain)
    np.testing.assert_allclose(sf.read(wet.filepath)[0] * 10 ** (wet.fader_db / 20), corrected, atol=1e-7)
