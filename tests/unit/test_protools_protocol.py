"""Record the protobuf requests from a full Fetch and Create against a stateful transport."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sessionpreplib.daw_processors import ptsl_helpers
from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.models import SessionContext, TrackContext, TransferEntry

pt = pytest.importorskip("ptsl.PTSL_pb2")
from ptsl.errors import CommandError


class Host:
    def __init__(self):
        self.client = SimpleNamespace(raw_client=self, session_id="connection", run=self.host_ready)
        self.open_path = None
        self.name_error = None
        self.reject = None
        self.calls = []
        self.closed_paths = []
        self.new_tracks = []
        self.reject_file = None

    def host_ready(self, operation):
        operation.response = SimpleNamespace(is_host_ready=True)

    def close(self):
        pass

    def session_name(self):
        if self.name_error:
            raise self.name_error
        if not self.open_path:
            raise CommandError([pt.CommandError(command_error_type=pt.PT_NoOpenedSession)])
        return self.open_path.stem

    def session_path(self):
        return str(self.open_path) if self.open_path else ""

    def track_list(self):
        return [SimpleNamespace(id="child", name="Kits", type=pt.TrackType.RoutingFolder, index=7, parent_folder_id="parent"),
                SimpleNamespace(id="parent", name="Music", type=pt.TrackType.BasicFolder, index=2, parent_folder_id="")]

    def SendGrpcRequest(self, request):
        name = pt.CommandId.Name(request.header.command).removeprefix("CId_")
        data = json.loads(request.request_body_json)
        self.calls.append((name, data))
        if name == self.reject:
            return pt.Response(header=pt.ResponseHeader(status=pt.Failed), response_error_json=f"{name} rejected")
        result = {}
        if name == "CreateSession":
            title = data["session_name"]
            self.open_path = Path(data["session_location"]) / title / (title + ".ptx")
            self.open_path.parent.mkdir()
            self.open_path.touch()
        elif name == "CreateBatchJob":
            result = {"id": "batch"}
        elif name == "ImportAudioToClipList":
            result = {"file_list": [
                {"original_input_path": p, "destination_file_list": [{"clip_id_list": [Path(p).stem]}]}
                for p in data["file_list"] if p != self.reject_file
            ], "failure_list": []}
            if self.reject_file:
                result["failure_list"].append({"file_path": self.reject_file, "failure_message": "unreadable media"})
        elif name == "CreateNewTracks":
            track_name = data["track_name"] + " (host)"
            track_id = f"track-{len(self.new_tracks)}"
            self.new_tracks.insert(0, (track_id, track_name))
            result = {"created_track_ids": [track_id], "created_track_names": [track_name]}
        elif name == "CloseSession":
            self.closed_paths.append(self.open_path)
            self.open_path = None
        return pt.Response(header=pt.ResponseHeader(status=pt.Completed), response_body_json=json.dumps(result))


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    from sessionpreplib import config, paths
    monkeypatch.setattr(config, "get_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(paths, "get_app_dir", lambda: str(tmp_path))
    host = Host()
    processor = ProToolsDawProcessor(instance_group="Example", instance_name="Nested")
    processor.configure({"protools_transfer_backend": "ptsl_hybrid", "protools_project_dir": str(tmp_path), "protools_temp_dir": str(tmp_path), "protools_command_delay": 0})
    monkeypatch.setattr(processor, "_open_engine", lambda: host)
    tracks = []
    for title in ("A.wav", "B.wav"):
        p = tmp_path / title
        p.touch()
        tracks.append(TrackContext(title, str(p), None, 48000, 1, 48, "24", "PCM_24", .001))
    session = SessionContext(tracks, {"_use_processed": False}, output_tracks=tracks, project_name="Disposable",
                             transfer_manifest=[TransferEntry("a", "A.wav", "First"), TransferEntry("b", "B.wav", "Second"),
                                                TransferEntry("repeat", "A.wav", "Again")])
    processor.fetch(session, ignore_cache=True)
    state = session.daw_state[processor.id]
    assert [(f["id"], f["parent_id"]) for f in state["folders"]] == [("child", "parent"), ("parent", None)]
    assert len(host.closed_paths) == 1 and not host.closed_paths[0].exists()
    state["assignments"] = dict.fromkeys(("a", "repeat", "b"), "child")
    state["track_order"] = {"child": ["repeat", "b", "a"]}
    host.calls.clear()
    return host, processor, session


def test_reordered_tracks_use_host_ids_and_repeat_media_once(transfer):
    host, processor, session = transfer
    host.reject_file = session.tracks[1].filepath
    results = processor.transfer(session, "")
    failures = [(r.command.command_type, r.command.target, r.error) for r in results if not r.success]
    assert failures == [("import_audio", "Second", "unreadable media")]
    imported, = [p for name, p in host.calls if name == "ImportAudioToClipList"]
    assert imported["file_list"] == [t.filepath for t in session.tracks]
    creates = [d for op, d in host.calls if op == "CreateNewTracks"]
    assert [d["track_name"] for d in creates] == ["First", "Again"]
    assert all(d["insertion_point_track_name"] == "Kits" and d["insertion_point_position"] == "TIPoint_First" for d in creates)
    assert [name for _, name in host.new_tracks] == ["Again (host)", "First (host)"]
    spots = [d for op, d in host.calls if op == "SpotClipsByID"]
    assert [(d["src_clips"], d["dst_track_id"]) for d in spots] == [(["A"], "track-0"), (["A"], "track-1")]
    assert all(d["dst_location_data"]["location"] == {"time_type": "TLType_Samples", "location": "0"} for d in spots)
    assert not any("Breakpoint" in op for op, _ in host.calls)
    assert host.closed_paths[-1].stem == "Disposable"


@pytest.mark.parametrize("operation", ["CompleteBatchJob", "SaveSession"])
def test_completion_and_save_failure_are_reported(transfer, operation):
    host, processor, session = transfer
    host.reject = operation
    results = processor.transfer(session, "", close_when_done=False)
    failures = [(r.command.command_type, r.error) for r in results if not r.success]
    expected = "complete_batch" if operation == "CompleteBatchJob" else "save_session"
    assert failures == [(expected, f"{operation} rejected")]
    assert host.open_path.stem == "Disposable"
    assert len(host.closed_paths) == 1
    assert sum(op == "CancelBatchJob" for op, _ in host.calls) == int(operation == "CompleteBatchJob")


def test_connection_lost_after_fetch_never_starts_create(transfer):
    host, processor, session = transfer
    host.name_error = ConnectionError("connection lost")
    results = processor.transfer(session, "")
    assert len(results) == 1 and not results[0].success and "connection lost" in results[0].error
    assert host.calls == []
    with pytest.raises(ConnectionError, match="connection lost"):
        processor.fetch(session, ignore_cache=True)
    assert host.calls == []


def test_only_definite_no_open_session_permits_creation():
    host = Host()
    assert not ptsl_helpers.is_session_open(host)
    host.name_error = CommandError([pt.CommandError(command_error_type=pt.PT_UnknownError)])
    with pytest.raises(CommandError):
        ptsl_helpers.is_session_open(host)
    host.name_error = CommandError([pt.CommandError(command_error_type=pt.PT_NoOpenedSession),
                                    pt.CommandError(command_error_type=pt.PT_UnknownError)])
    with pytest.raises(CommandError):
        ptsl_helpers.is_session_open(host)


def test_fetch_cannot_delete_a_temporary_session_the_host_failed_to_close(transfer):
    host, processor, session = transfer
    state = session.daw_state[processor.id]
    host.reject = "CloseSession"
    progress = []
    with pytest.raises(RuntimeError, match="CloseSession rejected"):
        processor.fetch(session, progress_cb=lambda value, *_: progress.append(value), ignore_cache=True)
    assert host.open_path.is_file()
    assert session.daw_state[processor.id] is state
    assert 100 not in progress


@pytest.mark.parametrize("post_close", ["still open", "connection lost"])
def test_fetch_verifies_close_before_removing_the_project(transfer, monkeypatch, post_close):
    host, processor, session = transfer
    state = session.daw_state[processor.id]
    progress = []
    send = host.SendGrpcRequest

    def acknowledge_without_closing(request):
        if request.header.command == pt.CId_CloseSession:
            if post_close == "connection lost":
                host.name_error = ConnectionError("connection lost")
            return pt.Response(header=pt.ResponseHeader(status=pt.Completed))
        return send(request)

    monkeypatch.setattr(host, "SendGrpcRequest", acknowledge_without_closing)
    with pytest.raises(RuntimeError, match="close the temporary Pro Tools session"):
        processor.fetch(session, progress_cb=lambda value, *_: progress.append(value), ignore_cache=True)
    assert host.open_path.is_file()
    assert session.daw_state[processor.id] is state
    assert 100 not in progress


def test_rejected_create_never_claims_an_open_project_at_the_requested_path(transfer, monkeypatch):
    host, processor, session = transfer
    previous_closes = host.closed_paths.copy()

    def concurrent_open_then_failed_create(engine, title, directory, *_args, **_kwargs):
        engine.open_path = Path(directory) / title / (title + ".ptx")
        engine.open_path.parent.mkdir()
        engine.open_path.write_bytes(b"other session")
        raise RuntimeError("CreateSession rejected")

    monkeypatch.setattr(ptsl_helpers, "create_session_from_template", concurrent_open_then_failed_create)
    with pytest.raises(RuntimeError, match="CreateSession rejected"):
        processor.fetch(session, ignore_cache=True)
    assert host.closed_paths == previous_closes
    assert host.open_path.read_bytes() == b"other session"


def test_fetch_refuses_preexisting_project_directory_before_creation(transfer, tmp_path, monkeypatch):
    import uuid

    host, processor, session = transfer
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="1234abcd"))
    existing = tmp_path / "SessionPrep_Temp_1234abcd"
    existing.mkdir()
    marker = existing / "existing media"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="already exists"):
        processor.fetch(session, ignore_cache=True)
    assert host.calls == []
    assert marker.read_bytes() == b"keep"
