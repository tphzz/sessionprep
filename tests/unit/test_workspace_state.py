"""Unit tests for WorkspaceState signals, config resolution, and reset invariants."""

from __future__ import annotations

from types import SimpleNamespace

from sessionpreplib.models import SessionContext, TrackContext
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState


def test_workspace_state_emits_signals_on_transitions(_font_app):
    cfg = build_defaults()
    state = WorkspaceState(config=cfg)

    events: list[tuple[str, object]] = []
    state.session_changed.connect(lambda s: events.append(("session", s)))
    state.current_track_changed.connect(lambda t: events.append(("track", t)))
    state.summary_changed.connect(lambda d: events.append(("summary", d)))
    state.topology_changed.connect(lambda: events.append(("topology", True)))
    state.groups_changed.connect(lambda: events.append(("groups", True)))
    state.daw_processor_changed.connect(lambda dp: events.append(("daw", dp)))
    state.overlays_changed.connect(lambda ov: events.append(("overlays", set(ov))))

    track = TrackContext(
        filename="kick.wav",
        filepath="/tmp/kick.wav",
        audio_data=None,
        samplerate=48000,
        channels=1,
        total_samples=48000,
        bitdepth="24-bit",
        subtype="PCM_24",
        duration_sec=1.0,
    )
    session = SessionContext(tracks=[track], config={})
    dp = SimpleNamespace(id="dawproject", fader_ceiling_db=12.0)

    state.session = session
    state.session = session  # identity repeat does not re-emit
    state.current_track = track
    state.summary = {"total_files": 1}
    state.topo_topology = SimpleNamespace(entries=[])
    state.session_groups = [{"name": "Drums"}]
    state.active_daw_processor = dp
    state.selected_detector_overlays = {"clipping"}

    assert ("session", session) in events
    assert events.count(("session", session)) == 1
    assert ("track", track) in events
    assert ("summary", {"total_files": 1}) in events
    assert ("topology", True) in events
    assert ("groups", True) in events
    assert ("daw", dp) in events
    assert ("overlays", {"clipping"}) in events


def test_workspace_state_flat_config_includes_overrides_and_fader_ceiling(_font_app):
    cfg = build_defaults()
    state = WorkspaceState(config=cfg)
    state.session_overrides["analysis.window"] = 250
    state.active_daw_processor = SimpleNamespace(id="pt", fader_ceiling_db=12.0)

    flat = state.flat_config()
    assert flat["window"] == 250
    assert flat["_fader_ceiling_db"] == 12.0


def test_workspace_state_reset_preserves_or_clears_session_config(_font_app):
    cfg = build_defaults()
    cfg["config_presets"]["Custom"] = dict(cfg["config_presets"]["Default"])
    state = WorkspaceState(config=cfg)
    state.source_dir = "/tmp/song"
    state.topology_dir = "/tmp/song/sp_01_tracklayout"
    state.active_config_preset_name = "Custom"
    state.session_overrides["analysis.window"] = 300
    state.session_groups = [{"name": "Vocals"}]
    state.selected_detector_overlays = {"dc_offset"}

    state.reset_workspace(keep_session_config=True)
    assert state.source_dir is None
    assert state.topology_dir is None
    assert state.session is None
    assert state.session_groups == []
    assert state.selected_detector_overlays == set()
    assert state.active_config_preset_name == "Custom"
    assert state.session_overrides["analysis.window"] == 300

    state.reset_workspace(keep_session_config=False)
    assert state.active_config_preset_name == "Default"
    assert state.session_overrides == {}
