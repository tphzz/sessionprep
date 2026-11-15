from pathlib import Path
import time

import pytest

from sessionpreplib.daw_processors import ptsl_helpers
from sessionpreplib.daw_processors.ptsl_connection import create_ptsl_engine

# Marks all tests in this file as requiring a live PT connection
pytestmark = pytest.mark.ptsl_live

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
