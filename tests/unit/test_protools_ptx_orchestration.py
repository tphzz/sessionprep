"""Unit tests for Pro Tools .ptx routing, two-level clip-gain staging, and VCA grouping."""

from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.daw_processors import protools, protools_ptx
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionpreplib.ptx import PtxSession, build_annotated_signal_flow, patch_track_volumes
from sessionpreplib.ptx.codec import encrypt_ptx
from sessionpreplib.ptx.models import PTX_MAGIC
from tests.unit.test_ptx_core import (
    _build_synthetic_ptx,
    _clip_2629,
    _io_1021,
    _lp,
    _mixer_audio_261c,
    _mixer_track_261e,
    _mixer_vca_261f,
    _sp_entry,
    _track_251a,
    _uuid,
    _zchunk,
)


def _build_nested_folder_ptx(*, folder_has_input: bool = True) -> bytes:
    """Synthetic .ptx with RoutingFolder KIK -> BasicFolder Sub -> Audio Kick, plus unassigned KIK-VCA."""
    bus_out, bus_kik, hw_in = _uuid(1), _uuid(2), _uuid(3)
    rf_uid, bf_uid, kick_uid, vca_uid = _uuid(10), _uuid(11), _uuid(12), _uuid(13)
    c0003 = _zchunk(0x0003, b"Pro Tools Studio\x00\x002026.11.1\x00\x00Release\x00\x00Session\x00\x00Windows\x00")
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c1022 = _zchunk(0x1022, struct.pack("<I", 1) + _io_1021("AIO Analog 1-2", 0, False, hw_in), version=1)
    c2603 = _zchunk(
        0x2603,
        struct.pack("<I", 2) + _sp_entry("Main Out", 1, True, bus_out) + _sp_entry("KIK", 2, True, bus_kik),
        version=2,
    )
    c2519 = _zchunk(
        0x2519,
        _track_251a("KIK", 2, rf_uid, is_folder=True)
        + _track_251a("Sub", 11, bf_uid, is_folder=True, is_stereo=False)
        + _track_251a("Kick", 0, kick_uid, is_stereo=False)
        + _track_251a("KIK-VCA", 9, vca_uid, is_stereo=False),
        version=9,
    )
    c210c = _zchunk(
        0x210C,
        struct.pack("<I", 2)
        + b"\x00" + rf_uid + struct.pack("<I", 2) + b"\x00" + bf_uid + b"\x00" + vca_uid
        + b"\x00" + bf_uid + struct.pack("<I", 1) + b"\x00" + kick_uid,
    )
    c2624 = _zchunk(
        0x2624,
        struct.pack("<I", 4)
        + _mixer_track_261e(rf_uid, bus_kik if folder_has_input else None, bus_out, 0.0)
        + _mixer_track_261e(bf_uid, None, bus_out, 0.0)
        + _mixer_audio_261c(kick_uid, "AIO Analog 1-2", hw_in, bus_out, 0.0)
        + _mixer_vca_261f(vca_uid, -3.0),
    )
    c202b = _zchunk(0x202B, struct.pack("<II", 0, 0) + b"\x00\x00", version=4)
    c2637 = _zchunk(0x2637, struct.pack("<I", 0), version=1)
    c262a = _zchunk(0x262A, _clip_2629("Kick_01", 0, 0, 1.0, _uuid(100)))
    lane0 = _zchunk(0x1052, _lp("Kick") + _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 0) + b"\x00" + struct.pack("<H", 0)))
    c1054 = _zchunk(0x1054, lane0)
    c1015 = _zchunk(0x1015, _zchunk(0x1014, _lp("Kick") + b"\x00" + struct.pack("<IH", 1, 0) + _uuid(200)))
    pad = _zchunk(0x7FFE, b"\x00" * 9000)

    out = bytearray(PTX_MAGIC + b"\x00\x05\x90")
    out.extend(struct.pack("<BHII", 0x5A, 1, 4, 0))
    offsets = []
    for ch in (c0003, c1028, c1022, c2603, c2519, c210c, c2624, c202b, c2637, c262a, c1054, c1015, pad):
        offsets.append(len(out))
        out.extend(ch)
    struct.pack_into("<I", out, 0x1B, len(out))
    idx_body = struct.pack("<III", 1, 0, 1) + struct.pack("<BHH", 0, 4, 1) + struct.pack("<I", offsets[-1]) + b"\x00" * 6
    out.extend(_zchunk(0x0002, idx_body, version=2))
    return encrypt_ptx(bytes(out), 0x40)


def test_folder_output_routing_walks_through_basic_folder_and_handles_missing_bus_and_cycles():
    ptx = PtxSession.from_bytes(_build_nested_folder_ptx(folder_has_input=True))
    rf, bf, kick = ptx.track_by_name("KIK"), ptx.track_by_name("Sub"), ptx.track_by_name("Kick")
    live = [
        {"id": rf.uuid_braced, "name": "KIK", "type": "TType_RoutingFolder", "parent_folder_id": ""},
        {"id": bf.uuid_braced, "name": "Sub", "type": "TType_BasicFolder", "parent_folder_id": rf.uuid_braced},
        {"id": kick.uuid_braced, "name": "Kick", "type": "TType_Audio", "parent_folder_id": bf.uuid_braced},
    ]
    plan, warnings = protools_ptx.plan_folder_output_routing(ptx, live, base=0x30000100, created_track_ids={kick.uuid_braced})
    assert warnings == []
    assert len(plan) == 1
    assert plan[0].target_bus_name == "KIK" and plan[0].target_signalpath_id == hex(0x30000101)

    # Missing bus input on RoutingFolder produces clean warning instead of crashing
    ptx_nobus = PtxSession.from_bytes(_build_nested_folder_ptx(folder_has_input=False))
    plan2, warns2 = protools_ptx.plan_folder_output_routing(ptx_nobus, live, base=0x30000100, created_track_ids={kick.uuid_braced})
    assert plan2 == [] and len(warns2) == 1

    # Cyclic parent_folder_id terminates cleanly
    cyclic = [
        {"id": bf.uuid_braced, "name": "Sub", "type": "TType_BasicFolder", "parent_folder_id": kick.uuid_braced},
        {"id": kick.uuid_braced, "name": "Kick", "type": "TType_Audio", "parent_folder_id": bf.uuid_braced},
    ]
    plan3, warns3 = protools_ptx.plan_folder_output_routing(ptx, cyclic, base=0x30000100, created_track_ids={kick.uuid_braced})
    assert plan3 == [] and warns3 == []


@pytest.mark.parametrize(
    ("mode", "use_processed", "exp_clip", "exp_fader"),
    [
        ("clip_gain_and_fader", False, 8.0, -8.0),
        ("clip_gain_unity", False, 8.0, 0.0),
        ("fader_only", False, 0.0, 0.0),  # Never apply inverse fader on un-normalized WAV
        ("none", False, 0.0, 0.0),
        ("clip_gain_and_fader", True, 0.0, -5.5),
        ("fader_only", True, 0.0, -5.5),
        ("clip_gain_unity", True, 0.0, 0.0),
    ],
)
def test_resolve_protools_track_staging_prevents_double_gain_and_balance_inversion(tmp_path, mode, use_processed, exp_clip, exp_fader):
    raw_wav = tmp_path / "Kick.wav"
    prep_wav = tmp_path / "Kick_prep.wav"
    sf.write(raw_wav, np.zeros(480), 48000, subtype="PCM_24")
    sf.write(prep_wav, np.zeros(480), 48000, subtype="PCM_24")
    track = TrackContext("Kick.wav", str(raw_wav), None, 48000, 1, 480, "24", "PCM_24", 0.01)
    track.processed_filepath = str(prep_wav)
    track.prepared_fader_db = -5.5
    track.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", 8.0, "Transient", "Peak", {"fader_offset": -8.0},
    )
    session = SessionContext(
        tracks=[track],
        output_tracks=[track],
        config={"_use_processed": use_processed},
        prepare_state="ready",
    )
    staging = protools_ptx.resolve_protools_track_staging(session, track, mode)
    assert staging.clip_gain_db == pytest.approx(exp_clip)
    assert staging.fader_db == pytest.approx(exp_fader)
    assert staging.filepath == str(prep_wav if use_processed else raw_wav)


def test_folder_groups_and_vca_planning_links_vca_and_avoids_cross_folder_collision():
    ptx = PtxSession.from_bytes(_build_synthetic_ptx())
    actions = protools_ptx.plan_folder_groups_and_vcas(ptx, ["Kick"], link_vcas=True)
    assert len(actions) == 1
    assert actions[0].op == "add" and actions[0].group_name == "KIK" and actions[0].vca_track == "KIK-VCA"

    unlinked = protools_ptx.plan_folder_groups_and_vcas(ptx, ["Kick"], link_vcas=False)
    assert len(unlinked) == 1 and unlinked[0].vca_track is None

    # Edit-only groups never claim a VCA
    edit_only = protools_ptx.plan_folder_groups_and_vcas(ptx, ["Kick"], link_vcas=True, group_type="Edit")
    assert len(edit_only) == 1 and edit_only[0].vca_track is None

    # Cross-folder VCA collision: two distinct folders resolving to the same VCA -> second falls back to unlinked
    ptx_col = PtxSession.from_bytes(_build_synthetic_ptx())
    ptx_col.track_by_name("KIK-VCA").parent_folder_name = "Alias"
    ptx_col.track_by_name("Kick").parent_folder_name = "Alias"
    ptx_col.track_by_name("Snare").parent_folder_name = "KIK"
    # Alias owns KIK-VCA by folder membership; KIK also matches KIK-VCA by naming convention
    col_actions = protools_ptx.plan_folder_groups_and_vcas(ptx_col, ["Kick", "Snare"], link_vcas=True)
    by_folder = {a.group_name: a for a in col_actions}
    assert by_folder["Alias"].vca_track == "KIK-VCA"
    assert by_folder["KIK"].vca_track is None

    # In-memory group/VCA simulation exposes newly wired VCA offset to DAG evaluation
    sim_raw, _ = patch_track_volumes(_build_synthetic_ptx(), {"KIK-VCA": -3.0})
    sim_ptx = PtxSession.from_bytes(sim_raw)
    sim_actions = protools_ptx.plan_folder_groups_and_vcas(sim_ptx, ["Kick"], link_vcas=True)
    updated_ptx = protools_ptx._apply_group_actions_in_memory(sim_raw, sim_actions)
    graph = build_annotated_signal_flow(updated_ptx, {}, stage_bus_faders=False)
    assert graph.nodes["Kick"].vca_offset_db == pytest.approx(-3.0, abs=0.1)


def _setup_real_ptx_transfer(tmp_path, monkeypatch, *, raw_ptx: bytes | None = None):
    pytest.importorskip("ptsl")
    project_dir = tmp_path / "projects"
    project_dir.mkdir(parents=True, exist_ok=True)
    ptx_bytes = raw_ptx if raw_ptx is not None else _build_synthetic_ptx()
    ptx_ref = PtxSession.from_bytes(ptx_bytes)
    kick_uuid = ptx_ref.track_by_name("Kick").uuid_braced
    rf_uuid = ptx_ref.track_by_name("KIK").uuid_braced

    kick_wav = tmp_path / "Kick.wav"
    sr = 48000
    t = np.arange(sr // 2) / sr
    sf.write(kick_wav, 0.25 * np.sin(2 * np.pi * 100 * t), sr, subtype="PCM_24")
    track = TrackContext("Kick.wav", str(kick_wav), None, sr, 1, len(t), "24", "PCM_24", 0.5)
    track.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", 6.0, "Transient", "Peak", {"fader_offset": -6.0},
    )

    processor = protools.ProToolsDawProcessor(instance_index=0, instance_group="Mix", instance_name="Studio")
    processor.configure({"protools_project_dir": str(project_dir), "protools_command_delay": 0})
    state = {"session_path": "", "open": False, "routed": [], "reopened": [], "closed": False}

    live_tracks = [
        SimpleNamespace(id=rf_uuid, name="KIK", type="TType_RoutingFolder", parent_folder_id=""),
        SimpleNamespace(id=kick_uuid, name="Kick", type="TType_Audio", parent_folder_id=rf_uuid),
    ]
    engine = SimpleNamespace(
        session_path=lambda: state["session_path"],
        track_list=lambda: live_tracks,
        close=lambda: state.__setitem__("closed", True),
    )

    def create_session(_eng, name, location, *_a, **_kw):
        sp = Path(location) / name / f"{name}.ptx"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(ptx_bytes)
        state["session_path"] = str(sp)
        state["open"] = True

    monkeypatch.setattr(processor, "_open_engine", lambda: engine)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda _e: state["open"])
    monkeypatch.setattr(protools.ptslh, "wait_for_host_ready", lambda *_a, **_kw: True)
    monkeypatch.setattr(protools.ptslh, "create_session_from_template", create_session)
    monkeypatch.setattr(protools.ptslh, "get_color_palette", lambda _e: [])
    monkeypatch.setattr(protools.ptslh, "create_batch_job", lambda *_a: "b1")
    monkeypatch.setattr(protools.ptslh, "complete_batch_job", lambda *_a: None)
    monkeypatch.setattr(protools.ptslh, "cancel_batch_job", lambda *_a: None)
    monkeypatch.setattr(
        protools.ptslh, "batch_import_audio",
        lambda _e, paths, **_kw: {"file_list": [{"original_input_path": paths[0], "destination_file_list": [{"clip_id_list": ["c1"]}]}]},
    )
    monkeypatch.setattr(protools.ptslh, "create_track_with_name", lambda _e, name, *_a, **_kw: (kick_uuid, "Kick"))
    monkeypatch.setattr(protools.ptslh, "spot_clips", lambda *_a, **_kw: None)
    monkeypatch.setattr(protools.ptslh, "save_session", lambda _e: None)
    monkeypatch.setattr(protools.ptslh, "close_session", lambda _e, **_kw: state.__setitem__("open", False))
    monkeypatch.setattr(protools.ptslh, "open_session", lambda _e, p: (state["reopened"].append(p), state.__setitem__("open", True)))
    monkeypatch.setattr(protools.ptslh, "get_track_main_output_assignments", lambda _e, tids: [hex(0x30000100)])
    monkeypatch.setattr(
        protools.ptslh, "set_track_main_output_assignments",
        lambda _e, sp_ids, track_ids=None, **_kw: state["routed"].append((list(track_ids or []), list(sp_ids))),
    )

    session = SessionContext(
        tracks=[track], output_tracks=[track], config={"_use_processed": False}, project_name="Song",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
        daw_state={processor.id: {"assignments": {"Kick.wav": rf_uuid}, "folders": [{"id": rf_uuid, "name": "KIK"}]}},
    )
    return processor, session, state, ptx_bytes, kick_uuid


def test_transfer_end_to_end_routes_stages_and_groups(tmp_path, monkeypatch):
    processor, session, state, _raw, kick_uuid = _setup_real_ptx_transfer(tmp_path, monkeypatch)
    results = processor.transfer(session, "", close_when_done=False)

    assert all(r.success for r in results)
    assert state["routed"] == [([kick_uuid], [hex(0x30000101)])]
    assert len(state["reopened"]) == 1
    assert state["closed"] is True
    assert "signal_flow" in session.daw_state[processor.id]
    assert any(r.command.command_type == "clear_input" for r in results)

    patched = PtxSession.from_file(state["reopened"][0])
    assert patched.track_by_name("Kick").input_path is None
    assert patched.clips[0].gain_db == pytest.approx(6.0, abs=0.02)
    assert patched.track_by_name("Kick").volume_db == -6.0
    assert patched.group_by_name("KIK").vca_track_name == "KIK-VCA"
    assert patched.track_by_name("KIK-VCA").vca_group_name == "KIK"


def test_transfer_rolls_back_ptx_atomically_when_reopen_fails(tmp_path, monkeypatch):
    processor, session, state, raw_ptx, _kick_uuid = _setup_real_ptx_transfer(tmp_path, monkeypatch)
    attempts = {"n": 0}

    def fail_first_open(_eng, path):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Pro Tools refused patched session")
        state["reopened"].append(path)

    monkeypatch.setattr(protools.ptslh, "open_session", fail_first_open)
    results = processor.transfer(session, "", close_when_done=False)

    assert state["closed"] is True
    assert Path(state["session_path"]).read_bytes() == raw_ptx
    assert "signal_flow" not in session.daw_state[processor.id]
    assert not any(r.command.command_type in ("set_clip_gain", "set_fader_ptx", "set_group") for r in results)
    reopen_err = next(r for r in results if r.command.command_type == "reopen_session")
    assert not reopen_err.success and "rolled back" in reopen_err.error


def test_transfer_reopens_untouched_session_on_patch_failure_and_guards_session_switch(tmp_path, monkeypatch):
    processor, session, state, raw_ptx, _kick_uuid = _setup_real_ptx_transfer(tmp_path, monkeypatch)
    monkeypatch.setattr(protools, "run_post_close_ptx_enhancement", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    results = processor.transfer(session, "", close_when_done=False)

    assert any(r.command.command_type == "patch_ptx" and not r.success for r in results)
    assert len(state["reopened"]) == 1
    assert Path(state["session_path"]).read_bytes() == raw_ptx

    # Session switch mid-transfer prevents routing the wrong session
    processor2, session2, state2, _raw2, _ku2 = _setup_real_ptx_transfer(tmp_path / "s2", monkeypatch)
    orig_complete = protools.ptslh.complete_batch_job

    def switch_session(*_a, **_kw):
        orig_complete(*_a, **_kw)
        state2["session_path"] = str(tmp_path / "other.ptx")

    monkeypatch.setattr(protools.ptslh, "complete_batch_job", switch_session)
    res2 = processor2.transfer(session2, "", close_when_done=True)
    assert state2["routed"] == []
    assert any(r.command.command_type == "route_output" and not r.success and "switched" in r.error for r in res2)


def test_ptsl_delete_tracks_helper(monkeypatch):
    from sessionpreplib.daw_processors import ptsl_helpers as ptslh

    sent: list[tuple[str, dict, str | None, int]] = []

    def fake_run(engine, cmd, body, batch_job_id=None, progress=0):
        sent.append((cmd, body, batch_job_id, progress))
        return {"success_count": len(body.get("track_ids") or body.get("track_names") or ())}

    monkeypatch.setattr(ptslh, "run_command", fake_run)
    assert ptslh.delete_tracks(object()) == 0
    assert sent == []

    count = ptslh.delete_tracks(
        object(), track_ids=["id1", "id2"], track_names=["A"], keep_folder_members=True,
        keep_alternate_playlists=False, coalesce_automation=False, batch_job_id="bj1", progress=94,
    )
    assert count == 2
    cmd, body, bjid, prog = sent[0]
    assert cmd == "CId_DeleteTracks"
    assert body["track_ids"] == ["id1", "id2"]
    assert body["track_names"] == ["A"]
    assert body["delete_track_behavior_options"] == {
        "keep_folder_members": True,
        "keep_alternate_playlists": False,
        "coalesce_automation": False,
    }
    assert bjid == "bj1" and prog == 94


def test_empty_folder_and_cascading_vca_cleanup_plan_and_execution(monkeypatch):
    from sessionpreplib.daw_processors import ptsl_helpers as ptslh
    from sessionpreplib.daw_processors.protools_ptx import (
        execute_empty_folder_cleanup,
        plan_empty_folders_and_vcas,
    )
    from sessionpreplib.ptx.models import PtxGroup, PtxSend, PtxSignalPathRef, PtxTrack

    def _trk(
        idx: int, name: str, ttype: str, *, parent: str = "",
        in_bus: str = "", out_bus: str = "", sends: list[str] | None = None,
    ) -> PtxTrack:
        return PtxTrack(
            index=idx, name=name, uuid_hex=f"{idx:032x}", uuid_braced=f"{{{idx:032x}}}",
            track_type=ttype, format="Stereo", is_folder=ttype in ("RoutingFolder", "BasicFolder"),
            parent_folder_name=parent,
            input_path=PtxSignalPathRef(in_bus, "bus", f"in{idx}") if in_bus else None,
            main_output=PtxSignalPathRef(out_bus, "bus", f"out{idx}") if out_bus else None,
            sends=[PtxSend("A", 0, PtxSignalPathRef(sb, "bus", f"s{i}")) for i, sb in enumerate(sends or ())],
        )

    tracks = [
        # Populated folder + empty nested basic subfolder
        _trk(0, "KIK", "RoutingFolder", in_bus="KIK", out_bus="SUB_DRM"),
        _trk(1, "Kick_In", "Audio", parent="KIK", out_bus="KIK"),
        _trk(2, "KIK-EmptySub", "BasicFolder", parent="KIK"),
        # Empty routing folders fed by surviving main_output / send -> protected!
        _trk(3, "SUB_DRM", "RoutingFolder", in_bus="SUB_DRM", out_bus="MIX"),
        _trk(4, "SEND_BUS", "RoutingFolder", in_bus="SEND_BUS", out_bus="MIX"),
        # Empty folders (one with nested empty basic folder, one with single contained VCA)
        _trk(5, "HIT", "RoutingFolder", in_bus="HIT", out_bus="MIX"),
        _trk(6, "HIT-Sub", "BasicFolder", parent="HIT"),
        _trk(7, "RIS", "RoutingFolder", in_bus="RIS", out_bus="MIX"),
        _trk(8, "RIS-VCA", "Vca", parent="RIS"),
        # Dedicated VCA-organizer folder holding surviving VCAs, standalone unlinked PRINT-VCA, and unused HIT-VCA
        _trk(9, "VCAs", "BasicFolder"),
        _trk(10, "KIK-VCA", "Vca", parent="VCAs"),
        _trk(11, "HIT-VCA", "Vca", parent="VCAs"),
        _trk(12, "PRINT-VCA", "Vca", parent="VCAs"),
        _trk(13, "ALL-VCA", "Vca", parent="VCAs"),
        # 3-tier cascading super-VCAs: HIT-VCA/RIS-VCA -> SFX-VCA -> ALL-SFX-VCA
        _trk(14, "SFX-VCA", "Vca"),
        _trk(15, "ALL-SFX-VCA", "Vca"),
        # FX folder with pre-existing Aux return + VCA -> stays!
        _trk(16, "FXRV", "RoutingFolder", in_bus="FXRV", out_bus="MIX"),
        _trk(17, "RV-CLS", "Aux", parent="FXRV", in_bus="RV-CLS", out_bus="FXRV", sends=["SEND_BUS"]),
        _trk(18, "RV-VCA", "Vca", parent="FXRV"),
        # Folder containing a VCA controlling surviving Kick_In -> both VCA and folder stay!
        _trk(19, "ORPHAN", "BasicFolder"),
        _trk(20, "GLOBAL-VCA", "Vca", parent="ORPHAN"),
        # Standalone Aux downstream bus -> stays!
        _trk(21, "cHIT", "Aux", in_bus="cHIT", out_bus="MIX"),
    ]
    groups = [
        PtxGroup(0, "<ALL>", "edit", "Mix/Edit", -1, "!", member_names=[t.name for t in tracks]),
        PtxGroup(1, "SFX-VCA", "mix", "Mix/Edit", 1, "a", vca_track_name="SFX-VCA", member_names=["HIT-VCA", "RIS-VCA"]),
        PtxGroup(2, "ALL-SFX-VCA", "mix", "Mix/Edit", 2, "b", vca_track_name="ALL-SFX-VCA", member_names=["SFX-VCA"]),
        PtxGroup(3, "ALL-VCA", "mix", "Mix/Edit", 3, "c", vca_track_name="ALL-VCA", member_names=["KIK-VCA", "ALL-SFX-VCA"]),
        PtxGroup(4, "RV-VCA", "mix", "Mix/Edit", 4, "d", vca_track_name="RV-VCA", member_names=["RV-CLS"]),
        PtxGroup(5, "GLOBAL-VCA", "mix", "Mix/Edit", 5, "e", vca_track_name="GLOBAL-VCA", member_names=["Kick_In"]),
    ]
    ptx = SimpleNamespace(tracks=tracks, groups=groups)

    plan = plan_empty_folders_and_vcas(ptx)
    folder_names = [t.name for t in plan.empty_folders]
    vca_names = [t.name for t in plan.unused_vcas]
    assert set(folder_names) == {"KIK-EmptySub", "HIT-Sub", "HIT", "RIS"}
    assert folder_names.index("HIT-Sub") < folder_names.index("HIT")
    assert set(vca_names) == {"HIT-VCA", "RIS-VCA", "SFX-VCA", "ALL-SFX-VCA"}
    assert "VCAs" not in folder_names
    assert "ORPHAN" not in folder_names
    assert "PRINT-VCA" not in vca_names
    assert "GLOBAL-VCA" not in vca_names
    assert "ALL-VCA" not in vca_names
    assert "KIK-VCA" not in vca_names
    assert "RV-VCA" not in vca_names

    # Verify PTSL execution: VCAs first, then folders deepest-first, keep_folder_members=True
    calls: list[tuple[list[str], bool]] = []

    def fake_delete_tracks(engine, *, track_ids=None, keep_folder_members=True, **_kw):
        calls.append((list(track_ids or ()), keep_folder_members))
        return len(track_ids or ())

    monkeypatch.setattr(ptslh, "delete_tracks", fake_delete_tracks)
    live = [{"id": t.uuid_braced, "name": t.name} for t in tracks]
    del_folders, del_vcas, del_errors = execute_empty_folder_cleanup(object(), ptx, live)
    assert del_errors == []
    assert set(del_vcas) == {"HIT-VCA", "RIS-VCA", "SFX-VCA", "ALL-SFX-VCA"}
    assert set(del_folders) == {"KIK-EmptySub", "HIT-Sub", "HIT", "RIS"}
    assert all(keep for _, keep in calls)
    id_to_name = {t.uuid_braced: t.name for t in tracks}
    batches = [{id_to_name[i] for i in batch} for batch, _ in calls]
    assert batches[0] == {"HIT-VCA", "RIS-VCA", "SFX-VCA", "ALL-SFX-VCA"}
    assert batches[1] == {"KIK-EmptySub", "HIT-Sub"}
    assert batches[2] == {"HIT", "RIS"}

    # Partial failure (success_count < len) reconciles against live track_list and skips parent of failed child
    hit_sub_id = next(t.uuid_braced for t in tracks if t.name == "HIT-Sub")
    remaining_ids = {t.uuid_braced for t in tracks}

    def fake_partial_delete(engine, *, track_ids=None, **_kw):
        deleted = 0
        for tid in track_ids or ():
            if tid == hit_sub_id:
                continue  # Simulate Pro Tools refusing to delete HIT-Sub
            remaining_ids.discard(tid)
            deleted += 1
        return deleted

    fake_engine = SimpleNamespace(track_list=lambda: [{"id": i, "name": id_to_name[i]} for i in remaining_ids])
    monkeypatch.setattr(ptslh, "delete_tracks", fake_partial_delete)
    p_folders, p_vcas, p_errors = execute_empty_folder_cleanup(fake_engine, ptx, live)
    assert "KIK-EmptySub" in p_folders and "RIS" in p_folders
    assert "HIT-Sub" not in p_folders and "HIT" not in p_folders  # Parent HIT skipped after HIT-Sub failed
    assert any(op == "remove_empty_folder" and "HIT-Sub" in tgt for op, tgt, _ in p_errors)

    # Exception in delete_tracks is isolated into errors without raising
    monkeypatch.setattr(ptslh, "delete_tracks", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("PTSL boom")))
    e_folders, e_vcas, e_errors = execute_empty_folder_cleanup(object(), ptx, live)
    assert e_folders == [] and e_vcas == []
    assert any("PTSL boom" in msg for _, _, msg in e_errors)


def test_protools_transfer_runs_empty_folder_cleanup_and_guards_session_switch(tmp_path, monkeypatch):
    import os
    import sys
    import types
    from sessionpreplib.daw_processors import ptsl_helpers as ptslh
    from sessionpreplib.daw_processors.protools import ProToolsDawProcessor

    wav = tmp_path / "Kick.wav"
    sf.write(wav, np.full(9600, 0.2, dtype=np.float32), 48000, subtype="PCM_24")
    ptx_bytes = _build_synthetic_ptx()
    ptx_ref = PtxSession.from_bytes(ptx_bytes)
    rf_kik = ptx_ref.track_by_name("KIK")
    session_ptx = tmp_path / "CleanupSong.ptx"
    session_ptx.write_bytes(ptx_bytes)

    deleted_ops: list[str] = []

    def _run_transfer(*, session_switched: bool):
        fake_ptsl = types.ModuleType("ptsl")
        fake_ptsl.PTSL_pb2 = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "ptsl", fake_ptsl)

        class FakeEngine:
            def close(self):
                pass
            def track_list(self):
                return [{"id": rf_kik.uuid_braced, "name": "KIK", "type": "TType_RoutingFolder"}]

        proc = ProToolsDawProcessor()
        proc.configure({"protools_project_dir": str(tmp_path), "protools_auto_route_folders": False})
        monkeypatch.setattr(proc, "_open_engine", lambda: FakeEngine())
        monkeypatch.setattr(ptslh, "wait_for_host_ready", lambda *a, **kw: True)
        monkeypatch.setattr(proc, "_create_transfer_session", lambda *a, **kw: str(session_ptx))
        monkeypatch.setattr(proc, "_build_group_palette", lambda *a, **kw: {})
        monkeypatch.setattr(ptslh, "create_batch_job", lambda *a, **kw: "bj1")
        monkeypatch.setattr(ptslh, "complete_batch_job", lambda *a, **kw: None)
        monkeypatch.setattr(ptslh, "cancel_batch_job", lambda *a, **kw: None)
        monkeypatch.setattr(proc, "_import_transfer_audio", lambda *a, **kw: ({os.path.normcase(str(wav)): ["c1"]}, {}))
        monkeypatch.setattr(ptslh, "create_track_with_name", lambda *a, **kw: ("t1", "Kick"))
        monkeypatch.setattr(ptslh, "spot_clips", lambda *a, **kw: None)
        monkeypatch.setattr(ptslh, "save_session", lambda *a, **kw: None)
        monkeypatch.setattr(ptslh, "close_session", lambda *a, **kw: None)
        monkeypatch.setattr(ptslh, "is_session_open", lambda *a, **kw: False)
        monkeypatch.setattr(ptslh, "session_matches_path", lambda *a, **kw: not session_switched)
        monkeypatch.setattr(
            "sessionpreplib.daw_processors.protools.execute_empty_folder_cleanup",
            lambda *a, **kw: (deleted_ops.append("called") or ["RIM"], ["RIM-VCA"], []),
        )
        monkeypatch.setattr(
            "sessionpreplib.daw_processors.protools.run_post_close_ptx_enhancement",
            lambda *a, **kw: (ptx_bytes, {"cleared_inputs": {}, "clip_gains": {}, "volumes": {}, "groups": []}, None),
        )
        tc = TrackContext("Kick.wav", str(wav), None, 48000, 1, 9600, "24", "PCM_24", 0.2)
        sess = SessionContext(
            tracks=[tc], output_tracks=[tc], config={"_use_processed": False}, project_name="CleanupSong",
            transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
            daw_state={proc.id: {
                "folders": [{"id": rf_kik.uuid_braced, "name": "KIK", "folder_type": "routing"}],
                "assignments": {"Kick.wav": rf_kik.uuid_braced},
            }},
        )
        return proc.transfer(sess, "", close_when_done=True)

    res_ok = _run_transfer(session_switched=False)
    assert deleted_ops == ["called"]
    assert any(r.command.command_type == "remove_empty_folder" and r.command.target == "RIM" and r.success for r in res_ok)
    assert any(r.command.command_type == "remove_unused_vca" and r.command.target == "RIM-VCA" and r.success for r in res_ok)

    deleted_ops.clear()
    res_sw = _run_transfer(session_switched=True)
    assert deleted_ops == []
    assert any(r.command.command_type == "remove_empty_folder" and not r.success and "switched" in r.error for r in res_sw)


def test_apply_post_close_ptx_patches_writes_slot_a_trims_and_records_results(tmp_path):
    from tests.unit.test_ptx_plugin_trim import _build_plugin_trim_ptx

    raw = _build_plugin_trim_ptx()
    ptx = PtxSession.from_bytes(raw)
    t_tc = ptx.tracks[0].name
    t_prot = ptx.tracks[4].name
    ptx_path = tmp_path / "SlotATrim.ptx"
    ptx_path.write_bytes(raw)

    orig_bytes, summary = protools_ptx.apply_post_close_ptx_patches(
        ptx_path,
        slot_a_trims_db={t_tc: 5.5, t_prot: -7.5},
        track_volumes_db={t_tc: -5.5},
    )
    assert orig_bytes == raw
    assert set(summary["slot_a_trims"].keys()) == {t_tc, t_prot}
    assert summary["slot_a_trims"][t_tc]["activated"] is True

    re_ptx = PtxSession.from_file(ptx_path)
    assert re_ptx.track_by_name(t_tc).slot_a_trim.gain_db == pytest.approx(5.5, abs=0.05)
    assert re_ptx.track_by_name(t_tc).slot_a_trim.is_active is True
    assert re_ptx.track_by_name(t_tc).volume_db == pytest.approx(-5.5, abs=0.1)
    assert re_ptx.track_by_name(t_prot).slot_a_trim.gain_db == pytest.approx(-7.5, abs=0.05)


def test_protools_transfer_records_set_slot_a_trim_result(tmp_path, monkeypatch):
    from tests.unit.test_ptx_plugin_trim import _build_plugin_trim_ptx

    raw = _build_plugin_trim_ptx()
    ptx_ref = PtxSession.from_bytes(raw)
    t_tc = ptx_ref.tracks[0].name
    session_ptx = tmp_path / "TrimSong.ptx"
    session_ptx.write_bytes(raw)

    proc = protools.ProToolsDawProcessor()
    proc.configure({"protools_project_dir": str(tmp_path)})
    monkeypatch.setattr(protools.ptslh, "close_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda *_a, **_kw: False)
    monkeypatch.setattr(protools.ptslh, "open_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        protools,
        "run_post_close_ptx_enhancement",
        lambda *_a, **_kw: (
            raw,
            {
                "cleared_inputs": {},
                "clip_gains": {},
                "slot_a_trims": {t_tc: {"plugin": "TrackControl", "target_db": 4.5, "activated": True}},
                "volumes": {t_tc: {"target_db": -4.5}},
                "groups": [],
            },
            None,
        ),
    )
    sess = SessionContext(tracks=[], output_tracks=[], config={}, project_name="TrimSong")
    recorded: list[tuple[str, str, dict]] = []

    def _record(cmd_type: str, target: str, error: str | None = None, **params) -> None:
        recorded.append((cmd_type, target, {"error": error, **params}))

    proc._apply_post_close_ptx_pass(
        SimpleNamespace(), sess, str(session_ptx), [], True, None, _record,
    )
    trim_res = [r for r in recorded if r[0] == "set_slot_a_trim"]
    assert len(trim_res) == 1
    assert trim_res[0][1] == t_tc
    assert trim_res[0][2]["error"] is None
    assert trim_res[0][2]["trim_db"] == 4.5
    assert trim_res[0][2]["plugin"] == "TrackControl"
    assert trim_res[0][2]["activated"] is True


def test_gain_mode_none_preserves_rough_mix_balance_in_preview_and_post_close(tmp_path):
    from sessionpreplib.daw_processors.protools_ptx_preview import preview_protools_signal_flow
    from tests.unit.test_ptx_signal_flow import _build_console_ptx, _sustained_tone

    ptxt = tmp_path / "Console.ptxt"
    ptxt.write_bytes(_build_console_ptx())
    ptx_ref = PtxSession.from_bytes(ptxt.read_bytes())
    sr = 48000
    quiet_wav = tmp_path / "QuietKick.wav"
    sf.write(quiet_wav, _sustained_tone(sr, 0.25, 10.0 ** (-28.0 / 20.0)), sr, subtype="PCM_24")
    tc = TrackContext("QuietKick.wav", str(quiet_wav), None, sr, 1, sr // 4, "24", "PCM_24", 0.25)
    kik_id = ptx_ref.track_by_name("KIK").uuid_braced
    sess = SessionContext(
        tracks=[tc], output_tracks=[tc], config={"target_peak": -6.0, "target_rms": -18.0},
        transfer_manifest=[TransferEntry("QuietKick.wav", "QuietKick.wav", "QuietKick")],
    )
    g_none = preview_protools_signal_flow(
        ptxt, sess, {"QuietKick.wav": kik_id},
        [{"id": kik_id, "name": "KIK", "folder_type": "routing", "parent_id": None}],
        gain_mode="none", stage_bus_faders=True,
    )
    # gain_mode="none" uses attenuate_only=True so quiet busses are NEVER boosted
    assert g_none.nodes["KIK"].recommended_fader_db <= 0.0
