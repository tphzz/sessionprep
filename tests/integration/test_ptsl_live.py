from pathlib import Path
import time
import os
import re
from uuid import uuid4

import pytest

from sessionpreplib.daw_processors import ptsl_helpers
from sessionpreplib.daw_processors.ptsl_connection import create_ptsl_engine

# Marks all tests in this file as requiring a live PT connection
pytestmark = pytest.mark.ptsl_live


def test_duplicate_media_reopens_on_two_tracks_at_sample_zero(tmp_path, request, monkeypatch):
    """Use a template containing at least one folder, in a disposable empty host."""
    if not request.config.getoption("--ptsl-live"):
        pytest.skip("Use --ptsl-live to create a disposable Pro Tools session.")
    group = os.getenv("SESSIONPREP_TEST_TEMPLATE_GROUP")
    template = os.getenv("SESSIONPREP_TEST_TEMPLATE_NAME")
    if not group or not template:
        pytest.skip("Set SESSIONPREP_TEST_TEMPLATE_GROUP and SESSIONPREP_TEST_TEMPLATE_NAME.")
    pytest.importorskip("ptsl")
    import numpy as np
    import soundfile as sf
    from ptsl import PTSL_pb2 as pt
    from ptsl.errors import CommandError
    from sessionpreplib import config
    from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
    from sessionpreplib.models import TransferEntry
    from sessionpreplib.pipeline import load_session

    suffix = uuid4().hex[:10]
    name = f"SessionPrep Repeat {suffix}"
    media = f"SP Source {suffix}"
    track_names = [f"SP One {suffix}", f"SP Two {suffix}"]
    path = tmp_path / name / f"{name}.ptx"
    engine = create_ptsl_engine()
    try:
        try:
            engine.session_path()
        except CommandError as exc:
            if exc.error_name != "PT_NoOpenedSession":
                raise
        else:
            pytest.skip("Close the open Pro Tools session before running this test.")
        from sessionpreplib import paths
        monkeypatch.setattr(config, "get_app_dir", lambda: str(tmp_path))
        monkeypatch.setattr(paths, "get_app_dir", lambda: str(tmp_path))
        assert not ptsl_helpers.VOLUME_AUTOMATION_ENABLED
        sent_commands = []
        send_command = ptsl_helpers.run_command

        def record_command(engine, command, *args, **kwargs):
            command_id = ptsl_helpers.command_id(command)
            sent_commands.append(command_id)
            assert command_id != pt.CommandId.CId_SetTrackControlBreakpoints
            return send_command(engine, command, *args, **kwargs)

        monkeypatch.setattr(ptsl_helpers, "run_command", record_command)
        assert not path.parent.exists()
        source = tmp_path / "inputs"
        source.mkdir()
        samples = np.tile([.25, -.125, 0, 0], 12000)
        sf.write(source / f"{media}.wav", samples, 48000, subtype="PCM_24")
        session = load_session(str(source), {"_use_processed": False})
        track, = session.tracks
        session.output_tracks = [track]
        session.project_name = name
        session.transfer_manifest = [TransferEntry(str(i), track.filename, title) for i, title in enumerate(track_names)]
        processor = ProToolsDawProcessor(instance_group=group, instance_name=template)
        processor.configure({"protools_project_dir": str(tmp_path), "protools_temp_dir": str(tmp_path)})
        processor.fetch(session, ignore_cache=True)
        state = session.daw_state[processor.id]
        folder = next(f for f in state["folders"] if f["folder_type"] == "routing")
        state["assignments"] = {"0": folder["id"], "1": folder["id"]}
        state["track_order"] = {folder["id"]: ["0", "1"]}
        results = processor.transfer(session, "", close_when_done=True)
        assert results and all(r.success for r in results), [(r.command, r.error) for r in results if not r.success]
        assert not ptsl_helpers.is_session_open(engine)
        engine.open_session(str(path))
        assert ptsl_helpers.wait_for_host_ready(engine, timeout=15)
        assert ptsl_helpers.session_matches_path(engine, str(path))
        tracks = engine.track_list()
        found = [t for t in tracks if t.name in track_names]
        assert [t.name for t in found] == track_names
        assert len({t.id for t in found}) == 2
        folder_id = next(t.id for t in tracks if t.name == folder["name"])
        assert all(t.parent_folder_id == folder_id for t in found)
        text = ptsl_helpers.run_command(engine, "CId_ExportSessionInfoAsText", {
            "include_file_list": True, "include_track_edls": True, "track_list_type": "AllTracks",
            "track_offset_options": "Samples", "text_as_file_format": "UTF8", "output_type": "ESI_String",
            "fade_handling_type": "ShowCrossfades",
        })["session_info"]
        for title in track_names:
            edl = text.split(title, 1)[1].split("TRACK NAME:", 1)[0]
            assert re.search(re.escape(media) + r"[^\r\n]*?\s+0\s+48000\s+48000", edl), edl[:1500]
        copied, rate = sf.read(path.parent / "Audio Files" / f"{media}.wav")
        assert rate == 48000
        np.testing.assert_array_equal(copied, samples)
        assert pt.CommandId.CId_SetTrackControlBreakpoints not in sent_commands
    finally:
        try:
            if path.is_file() and ptsl_helpers.session_matches_path(engine, str(path)):
                ptsl_helpers.close_session(engine)
        finally:
            engine.close()

def test_read_all_tracks_and_folders(live_engine, capsys):
    """
    Reads the track list from the live session and prints basic properties.
    """
    with capsys.disabled():
        print("\n--- Live Tracks ---")
    
    # TrackListInSession is a direct engine operation in py-ptsl
    # For ptsl_helpers, we can call the SDK method. 
    # Usually it returns a pt.TrackListInSessionResponseBody structure
    try:
        track_list = live_engine.track_list()
        
        with capsys.disabled():
            for t in track_list:
                print(f"Track: '{t.name}' | ID: {t.id} | Type: {t.type} | Folder?: {t.is_folder}")
        
        assert len(track_list) > 0, "Session must contain at least one track for this test."
        
    except Exception as e:
        pytest.fail(f"Failed to read track list: {e}")


def test_sample_zero_volume_automation_survives_reopen(tmp_path, request):
    if not request.config.getoption("--ptsl-live"):
        pytest.skip("Use --ptsl-live to create a disposable Pro Tools session.")
    if not ptsl_helpers.VOLUME_AUTOMATION_ENABLED:
        pytest.skip("Pro Tools volume automation is currently disabled.")
    pytest.importorskip("ptsl")
    from ptsl import PTSL_pb2 as pt
    from ptsl.errors import CommandError

    name = "SessionPrep Volume Automation Test"
    session_path = str(Path(tmp_path) / name / f"{name}.ptx")
    track_name = "Sample Zero"
    target_db = -6.0
    engine = create_ptsl_engine()
    try:
        try:
            open_path = engine.session_path()
        except CommandError as exc:
            if exc.error_name != "PT_NoOpenedSession":
                raise
            open_path = ""
        if open_path:
            pytest.skip("Close the open Pro Tools session before running this test.")
        assert ptsl_helpers.wait_for_host_ready(engine, timeout=10)

        builder = engine.create_session(name, str(tmp_path))
        builder.wave_format()
        builder.sample_rate(48000)
        builder.bit_depth(24)
        builder.create()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if Path(session_path).is_file() and ptsl_helpers.session_matches_path(engine, session_path):
                break
            time.sleep(0.25)
        assert ptsl_helpers.session_matches_path(engine, session_path)
        assert ptsl_helpers.wait_for_host_ready(engine, timeout=10)

        track_id, _ = ptsl_helpers.create_track_with_name(engine, track_name, "TF_Mono")
        try:
            ptsl_helpers.set_track_volume(engine, track_id, target_db)
        except RuntimeError as exc:
            if "Private API has not been enabled" in str(exc):
                pytest.skip(f"Pro Tools refused the volume automation write. {exc}")
            raise
        ptsl_helpers.close_session(engine, save_on_close=True)
        engine.open_session(session_path)
        assert ptsl_helpers.wait_for_host_ready(engine, timeout=10)
        assert ptsl_helpers.session_matches_path(engine, session_path)
        track_id = next(track.id for track in engine.track_list() if track.name == track_name)
        try:
            data = ptsl_helpers.run_command(
                engine, pt.CommandId.CId_GetTrackControlBreakpoints,
                {"track_id": track_id, "control_id": {
                    "section": "TSId_MainOut", "control_type": "TCType_Volume",
                }},
            )
        except RuntimeError as exc:
            if "Private API has not been enabled" in str(exc) or "unsupported" in str(exc).lower():
                pytest.skip(f"Pro Tools wrote the automation. Saved breakpoint readback is unavailable. {exc}")
            raise

        points = (data or {}).get("breakpoints", [])
        assert [point["value"] for point in points if str(point["time"]["location"]) == "0"] == [pytest.approx(target_db)]
        ptsl_helpers.run_command(engine, pt.CommandId.CId_SetTimelineSelection, {
            "in_time": "0", "out_time": "0", "play_start_marker_time": "0", "location_type": "TLType_Samples",
        })
        engine.toggle_play_state()
        deadline = time.monotonic() + 5
        state = engine.transport_state()
        while not state.endswith("Playing") and time.monotonic() < deadline:
            time.sleep(0.1)
            state = engine.transport_state()
        assert state.endswith("Playing"), state
        engine.toggle_play_state()
    finally:
        try:
            if ptsl_helpers.session_matches_path(engine, session_path):
                if engine.transport_state().endswith("Playing"):
                    engine.toggle_play_state()
                ptsl_helpers.close_session(engine)
        finally:
            engine.close()
