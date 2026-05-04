import json
import pytest

try:
    from ptsl import PTSL_pb2 as pt
except ImportError:
    pt = None

from sessionpreplib.daw_processors import ptsl_helpers

pytestmark = pytest.mark.skipif(
    pt is None, 
    reason="py-ptsl not installed"
)

def test_command_id_uses_fallback_for_newer_ptsl_commands():
    assert ptsl_helpers.command_id("CId_GetTrackMainOutputAssignments") == 165
    assert ptsl_helpers.command_id(165) == 165


def test_command_id_rejects_unknown_command_name():
    with pytest.raises(ValueError):
        ptsl_helpers.command_id("CId_DoesNotExist")


def test_run_command_accepts_fallback_command_name(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok(
        {"signalpath_ids": ["0x30000245"]}
    )

    resp = ptsl_helpers.run_command(
        mock_engine,
        "CId_GetTrackMainOutputAssignments",
        {"track_ids": ["track-123"]},
    )

    assert resp == {"signalpath_ids": ["0x30000245"]}
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    assert req.header.command == 165
    assert json.loads(req.request_body_json) == {"track_ids": ["track-123"]}


def test_run_command_builds_correct_header(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok({"dummy": "value"})

    resp = ptsl_helpers.run_command(
        mock_engine, 
        pt.CommandId.CId_GetSessionName, 
        {"body_key": "body_val"}
    )
    
    # Assert return value equals parsed json
    assert resp == {"dummy": "value"}
    
    # Assert header construction
    mock_engine.client.raw_client.SendGrpcRequest.assert_called_once()
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    
    assert req.header.session_id == "test-session-123"
    assert req.header.command == pt.CommandId.CId_GetSessionName
    assert req.header.version == 2025
    assert req.header.version_minor == 10
    
    # Assert body construction
    assert json.loads(req.request_body_json) == {"body_key": "body_val"}

def test_run_command_with_batch_job_header(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok()
    
    ptsl_helpers.run_command(
        mock_engine,
        pt.CommandId.CId_GetSessionName,
        {},
        batch_job_id="test-batch-uuid",
        progress=75
    )
    
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    
    # Versioned JSON field must contain the batch job header
    vheader = json.loads(req.header.versioned_request_header_json)
    assert "batch_job_header" in vheader
    assert vheader["batch_job_header"]["id"] == "test-batch-uuid"
    assert vheader["batch_job_header"]["progress"] == 75

def test_run_command_raises_on_failure(mock_engine, ptsl_factory):
    # Construct a Failed status response with an error message
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.fail(
        error_msg="No session is currently open",
        error_type=pt.PT_NoOpenedSession
    )
    
    with pytest.raises(RuntimeError) as exc:
        ptsl_helpers.run_command(mock_engine, pt.CommandId.CId_GetSessionName, {})
        
    assert "No session is currently open" in str(exc.value)
    assert "PT_NoOpenedSession" in str(exc.value)

def test_extract_clip_ids_happy_path():
    resp = {
        "file_list": [{
            "destination_file_list": [{
                "clip_id_list": ["clip-123", "clip-456"]
            }]
        }]
    }
    assert ptsl_helpers.extract_clip_ids(resp) == ["clip-123", "clip-456"]

def test_extract_clip_ids_malformed():
    with pytest.raises(RuntimeError):
        ptsl_helpers.extract_clip_ids({"file_list": []})
        
    with pytest.raises(RuntimeError):
        ptsl_helpers.extract_clip_ids({})

def test_extract_track_id():
    resp = {"created_track_ids": ["new-track-uuid"]}
    assert ptsl_helpers.extract_track_id(resp) == "new-track-uuid"
    
    with pytest.raises(RuntimeError):
        ptsl_helpers.extract_track_id({"created_track_ids": []})


def test_extract_track_name_falls_back():
    assert (
        ptsl_helpers.extract_track_name(
            {"created_track_names": ["Audio 1.01"]}, fallback="Audio 1"
        )
        == "Audio 1.01"
    )
    assert ptsl_helpers.extract_track_name({}, fallback="Audio 1") == "Audio 1"

def test_set_track_volume_body_construction(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok()
    
    ptsl_helpers.set_track_volume(mock_engine, "track-123", -6.5, batch_job_id="batch1")
    
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)
    
    assert body["track_id"] == "track-123"
    assert body["control_id"]["section"] == "TSId_MainOut"
    assert body["control_id"]["control_type"] == "TCType_Volume"
    
    bp = body["breakpoints"][0]
    assert bp["time"]["location"] == "0"
    assert bp["time"]["time_type"] == "TLType_Samples"
    
    # Verify bare float precision without truncation
    assert bp["value"] == -6.5

def test_set_track_volume_boundary_values(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok()
    
    # Legal boundaries
    ptsl_helpers.set_track_volume(mock_engine, "t1", 12.0)
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    assert json.loads(req.request_body_json)["breakpoints"][0]["value"] == 12.0
    
    ptsl_helpers.set_track_volume(mock_engine, "t1", -144.0)
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    assert json.loads(req.request_body_json)["breakpoints"][0]["value"] == -144.0

def test_create_track_with_folder(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok({
        "created_track_ids": ["t-uuid-001"]
    })
    
    track_id = ptsl_helpers.create_track(
        mock_engine, "Bass", "TF_Stereo", folder_name="Drums Folder"
    )
    
    assert track_id == "t-uuid-001"
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)
    
    assert body["track_name"] == "Bass"
    assert body["track_format"] == "TF_Stereo"
    assert body["insertion_point_track_name"] == "Drums Folder"
    assert body["insertion_point_position"] == "TIPoint_Last"


def test_create_track_with_explicit_insertion_point(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok({
        "created_track_ids": ["t-uuid-003"],
        "created_track_names": ["Bass.01"],
    })

    track_id, track_name = ptsl_helpers.create_track_with_name(
        mock_engine,
        "Bass",
        "TF_Mono",
        insertion_point_track_name="Routing Folder",
        insertion_point_position="TIPoint_First",
    )

    assert track_id == "t-uuid-003"
    assert track_name == "Bass.01"
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)

    assert body["track_name"] == "Bass"
    assert body["track_format"] == "TF_Mono"
    assert body["insertion_point_track_name"] == "Routing Folder"
    assert body["insertion_point_position"] == "TIPoint_First"

def test_create_track_without_folder(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok({
        "created_track_ids": ["t-uuid-002"]
    })
    
    ptsl_helpers.create_track(mock_engine, "Guitars", "TF_Stereo", folder_name=None)
    
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)
    
    assert "insertion_point_track_name" not in body
    assert "insertion_point_position" not in body


def test_set_track_main_output_assignments(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok()

    ptsl_helpers.set_track_main_output_assignments(
        mock_engine,
        ["0x30000245"],
        track_ids=["track-123"],
    )

    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)
    assert req.header.command == 147
    assert body == {
        "track_ids": ["track-123"],
        "signalpath_ids": ["0x30000245"],
    }


def test_set_track_height(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok(
        {"success_count": 1}
    )

    resp = ptsl_helpers.set_track_height(
        mock_engine,
        "THeight_Large",
        track_ids=["track-123"],
    )

    assert resp == {"success_count": 1}
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    body = json.loads(req.request_body_json)
    assert req.header.command == 160
    assert body == {
        "track_ids": ["track-123"],
        "height": "THeight_Large",
    }


def test_get_track_main_output_assignments(mock_engine, ptsl_factory):
    mock_engine.client.raw_client.SendGrpcRequest.return_value = ptsl_factory.ok(
        {"signalpath_ids": ["0x30000245"]}
    )

    output_ids = ptsl_helpers.get_track_main_output_assignments(
        mock_engine,
        ["track-123"],
    )

    assert output_ids == ["0x30000245"]
    req = mock_engine.client.raw_client.SendGrpcRequest.call_args[0][0]
    assert req.header.command == 165


def test_create_session_from_template_waits_until_session_file_exists(
    mock_engine, monkeypatch, tmp_path
):
    monkeypatch.setattr(ptsl_helpers, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    attempts = iter([False, False, True])
    checked_paths = []

    def fake_isfile(path):
        checked_paths.append(path)
        return next(attempts)

    monkeypatch.setattr(ptsl_helpers.os.path, "isfile", fake_isfile)

    ptsl_helpers.create_session_from_template(
        mock_engine,
        "TempSession",
        str(tmp_path),
        "SessionPrep",
        "Template",
        timeout=60.0,
    )

    assert checked_paths[-1].endswith("TempSession.ptx")


def test_create_session_from_template_honors_timeout(
    mock_engine, monkeypatch, tmp_path
):
    monkeypatch.setattr(ptsl_helpers, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(ptsl_helpers.os.path, "isfile", lambda _path: False)

    with pytest.raises(RuntimeError, match="failed to create the session"):
        ptsl_helpers.create_session_from_template(
            mock_engine,
            "TempSession",
            str(tmp_path),
            "SessionPrep",
            "Template",
            timeout=0.01,
        )
