from pathlib import Path
from types import SimpleNamespace

import pytest

from sessionpreplib.daw_processors import protools
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    pytest.importorskip("ptsl")
    project_dir = tmp_path / "projects"
    project_dir.mkdir()
    processor = protools.ProToolsDawProcessor(
        instance_index=0, instance_group="Mix Templates", instance_name="Studio",
    )
    processor.configure({"protools_project_dir": str(project_dir), "protools_command_delay": 0})
    closes = []
    faders = []
    spotted = []
    imports = []
    engine = SimpleNamespace(session_path=lambda: "", close=lambda: None)

    def create_session(_engine, name, location, *_args, **_kwargs):
        session_path = Path(location) / name / f"{name}.ptx"
        session_path.parent.mkdir()
        session_path.touch()
        engine.session_path = lambda: str(session_path)

    def import_audio(_engine, paths, **_kwargs):
        imports.extend(paths)
        return {
            "file_list": [
                {
                    "original_input_path": path,
                    "destination_file_list": [{"clip_id_list": [f"clip-{Path(path).stem}"]}],
                }
                for path in paths
            ]
        }

    monkeypatch.setattr(processor, "_open_engine", lambda: engine)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda _engine: False)
    monkeypatch.setattr(protools.ptslh, "wait_for_host_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(protools.ptslh, "create_session_from_template", create_session)
    monkeypatch.setattr(protools.ptslh, "get_color_palette", lambda _engine: [])
    monkeypatch.setattr(protools.ptslh, "get_session_audio_dir", lambda _engine: str(project_dir))
    monkeypatch.setattr(protools.ptslh, "create_batch_job", lambda *_args: "pt-batch")
    monkeypatch.setattr(protools.ptslh, "complete_batch_job", lambda *_args: None)
    monkeypatch.setattr(protools.ptslh, "cancel_batch_job", lambda *_args: None)
    monkeypatch.setattr(protools.ptslh, "batch_import_audio", import_audio)
    monkeypatch.setattr(protools.ptslh, "create_track_with_name",
                        lambda _engine, name, *_args, **_kwargs: (f"pt-{name}", name))
    monkeypatch.setattr(protools.ptslh, "spot_clips", lambda _engine, _clips, track_id, **_kwargs: spotted.append(track_id))
    monkeypatch.setattr(protools.ptslh, "set_track_volume",
                        lambda _engine, track_id, db, **kwargs: faders.append((track_id, db, kwargs["batch_job_id"])))
    monkeypatch.setattr(protools.ptslh, "close_session", lambda _engine, **kwargs: closes.append(kwargs["save_on_close"]))

    tracks = []
    for name in ("Kick", "Snare"):
        filename = f"{name}.wav"
        path = tmp_path / filename
        path.write_bytes(b"audio fixture")
        tracks.append(TrackContext(filename, str(path), None, 48000, 1, 48000, "24", "PCM_24", 1.0))

    session = SessionContext(
        tracks=tracks,
        output_tracks=tracks,
        config={},
        project_name="Mix",
        transfer_manifest=[TransferEntry(track.filename, track.filename, Path(track.filename).stem) for track in tracks],
        daw_state={processor.id: {
            "assignments": {track.filename: "folder-id" for track in tracks},
            "folders": [{"id": "folder-id", "name": "Drums"}],
        }},
    )
    return SimpleNamespace(processor=processor, session=session, engine=engine, closes=closes,
                           spotted=spotted, faders=faders, imports=imports)


@pytest.mark.parametrize("use_processed", [False, True])
def test_selected_media_and_faders_travel_together(transfer, tmp_path, monkeypatch, use_processed):
    monkeypatch.setattr(protools.ptslh, "VOLUME_AUTOMATION_ENABLED", True)
    session = transfer.session
    session.prepare_state = "ready"
    session.config["_use_processed"] = use_processed
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for track in session.output_tracks:
        prepared = processed_dir / track.filename
        prepared.write_bytes(b"rendered fixture")
        track.processed_filepath = str(prepared)
        track.prepared_fader_db = -2.5 if track.filename == "Kick.wav" else 0.0
        track.processor_results["bimodal_normalize"] = ProcessorResult(
            "bimodal_normalize", 8.0, "Sustained", "Bimodal", {"fader_offset": -8.0})

    results = transfer.processor.transfer(session, "")

    assert all(result.success for result in results)
    assert set(transfer.imports) == {
        t.processed_filepath if use_processed else t.filepath for t in session.output_tracks}
    assert transfer.faders == ([("pt-Kick", -2.5, "pt-batch")] if use_processed else [])


def test_pro_tools_reports_a_rejected_file_and_spots_the_remaining_track(transfer, monkeypatch):
    def partial_import(_engine, paths, **_kwargs):
        paths = {Path(path).stem: path for path in paths}
        return {
            "file_list": [{"original_input_path": paths["Kick"],
                           "destination_file_list": [{"clip_id_list": ["kick-clip"]}]}],
            "failure_list": [{"file_path": paths["Snare"], "failure_message": "Unsupported media"}],
        }

    monkeypatch.setattr(protools.ptslh, "batch_import_audio", partial_import)
    results = transfer.processor.transfer(transfer.session, "")

    assert transfer.spotted == ["pt-Kick"]
    assert [(r.command.target, r.error) for r in results if not r.success] == [("Snare", "Unsupported media")]
    assert transfer.closes == [True]


def test_pro_tools_preserves_the_created_session_result_and_closes_after_import_error(transfer, monkeypatch):
    def fail_import(*_args, **_kwargs):
        raise RuntimeError("Copy failed")

    monkeypatch.setattr(protools.ptslh, "batch_import_audio", fail_import)
    results = transfer.processor.transfer(transfer.session, "")

    assert any(r.command.command_type == "create_session" and r.success for r in results)
    assert any(not r.success and r.error == "Copy failed" for r in results)
    assert transfer.closes == [True]


def test_pro_tools_does_not_close_another_session_after_transfer_failure(transfer, monkeypatch, tmp_path):
    def fail_import(*_args, **_kwargs):
        transfer.engine.session_path = lambda: str(tmp_path / "other.ptx")
        raise RuntimeError("Copy failed")

    monkeypatch.setattr(protools.ptslh, "batch_import_audio", fail_import)
    results = transfer.processor.transfer(transfer.session, "")

    assert any(not r.success and r.error == "Copy failed" for r in results)
    assert any(r.command.command_type == "close_session" and not r.success for r in results)
    assert transfer.closes == []


def test_pro_tools_transfers_audio_without_sending_volume_automation_by_default(transfer):
    proc_id = "bimodal_normalize"
    for track in transfer.session.output_tracks:
        track.processor_results[proc_id] = ProcessorResult(
            proc_id, 6.0, "Sustained", "Bimodal", {"fader_offset": -6.0},
        )

    results = transfer.processor.transfer(transfer.session, "")

    assert transfer.faders == []
    assert transfer.spotted == ["pt-Snare", "pt-Kick"]
    assert transfer.closes == [True]
    assert results and all(result.success for result in results)
    assert all(result.command.command_type != "set_fader" for result in results)


@pytest.mark.parametrize("write_fails", [False, True])
def test_pro_tools_records_the_sample_zero_volume_write_result(transfer, monkeypatch, write_fails):
    monkeypatch.setattr(protools.ptslh, "VOLUME_AUTOMATION_ENABLED", True)
    processor_id = "bimodal_normalize"
    transfer.session.output_tracks[0].processor_results[processor_id] = ProcessorResult(
        processor_id, 3.75, "Sustained", "Bimodal", {"fader_offset": -3.75},
    )
    if write_fails:
        def refuse_volume(*_args, **_kwargs):
            raise RuntimeError("Private API has not been enabled in this session.")

        monkeypatch.setattr(protools.ptslh, "set_track_volume", refuse_volume)

    results = transfer.processor.transfer(transfer.session, "")
    fader = next(result for result in results if result.command.command_type == "set_fader")

    assert fader.command.target == "Kick"
    assert fader.command.params["fader_db"] == -3.75
    assert fader.success is not write_fails
    if write_fails:
        assert "Private API has not been enabled" in fader.error
    else:
        assert transfer.faders == [("pt-Kick", -3.75, "pt-batch")]
    assert transfer.closes == [True]


@pytest.mark.parametrize(
    ("error", "first_transfer_writes"),
    [
        ("ErrType 135: PT_CannotBeDone (Private API has not been enabled in this session.)", ["pt-Snare"]),
        ("PT_UnsupportedCommand", ["pt-Snare"]),
        ("PT_CannotBeDone (Pro Tools could not find this track.)", ["pt-Snare", "pt-Kick"]),
    ],
    ids=["private-api-denied", "unsupported-command", "track-specific-error"],
)
def test_pro_tools_stops_volume_writes_only_on_a_capability_rejection(
    transfer, monkeypatch, error, first_transfer_writes,
):
    monkeypatch.setattr(protools.ptslh, "VOLUME_AUTOMATION_ENABLED", True)
    proc_id = "bimodal_normalize"
    for track in transfer.session.output_tracks:
        track.processor_results[proc_id] = ProcessorResult(
            proc_id, 6.0, "Sustained", "Bimodal", {"fader_offset": -6.0},
        )
    attempted = []

    def write_volume(_engine, track_id, _db, **_kwargs):
        attempted.append(track_id)
        if len(attempted) == 1:
            raise RuntimeError(error)

    monkeypatch.setattr(protools.ptslh, "set_track_volume", write_volume)
    first_results = transfer.processor.transfer(transfer.session, "")

    assert attempted == first_transfer_writes
    assert transfer.spotted == ["pt-Snare", "pt-Kick"]
    assert transfer.closes == [True]
    failed, = [result for result in first_results if not result.success]
    assert failed.command.command_type == "set_fader"
    assert failed.command.target == "Snare"
    assert error in failed.error

    transfer.session.project_name = "Retry"
    next_results = transfer.processor.transfer(transfer.session, "")

    assert attempted == [*first_transfer_writes, "pt-Snare", "pt-Kick"]
    assert all(result.success for result in next_results)
    assert transfer.closes == [True, True]
