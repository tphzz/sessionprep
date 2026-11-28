"""Unit tests for ProToolsCreationPlan, optional explicit ptxt_path, and offline check_connectivity."""

from __future__ import annotations

from pathlib import Path
import platform

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.daw_processors.protools_ptx_plan import build_protools_creation_plan
from sessionpreplib.daw_processors.protools_ptx_preview import read_template_folders_from_ptxt
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionpreplib.ptx import decrypt_ptx, encrypt_ptx
from sessionpreplib.ptx.lowlevel import build_zchunk
from tests.unit.test_protools_ptx_orchestration import _build_nested_folder_ptx
from tests.unit.test_ptx_core import _build_synthetic_ptx


def _write_wav(path: Path, sr: int = 48000, channels: int = 1) -> Path:
    t = np.arange(sr // 10, dtype=np.float64) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, sig if channels == 1 else np.column_stack([sig, sig]), sr, subtype="PCM_24")
    return path


def test_build_protools_creation_plan_resolves_tracks_routing_colors_groups_and_signal_flow(tmp_path: Path):
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=True))
    folders = read_template_folders_from_ptxt(ptxt)
    sub_id = next(f["id"] for f in folders if f["name"] == "Sub")

    w1 = _write_wav(tmp_path / "Kick_In.wav", 48000, 1)
    w2 = _write_wav(tmp_path / "Kick_OH.wav", 48000, 2)
    w4 = tmp_path / "Quad.wav"
    sf.write(w4, np.zeros((480, 4), dtype=np.float32), 48000, subtype="PCM_24")

    t1 = TrackContext("Kick_In.wav", str(w1), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
    t1.processor_results["bimodal_normalize"] = ProcessorResult("bimodal_normalize", 6.0, "Peak", "Peak", {"fader_offset": -6.0})
    t2 = TrackContext("Kick_OH.wav", str(w2), None, 48000, 2, 4800, "24", "PCM_24", 0.1, group="Drums")
    t4 = TrackContext("Quad.wav", str(w4), None, 48000, 4, 480, "24", "PCM_24", 0.01)

    sess = SessionContext(
        tracks=[t1, t2, t4],
        output_tracks=[t1, t2, t4],
        project_name="  DemoSong  ",
        config={
            "gui": {
                "groups": [{"name": "Drums", "color": "Blue"}, "not-a-dict"],
                "colors": [{"name": "Blue", "argb": "#FF2C00FC"}],
            }
        },
        transfer_manifest=[
            TransferEntry("Kick_In.wav", "Kick_In.wav", "Kick_In", group="Drums"),
            TransferEntry("Kick_OH.wav", "Kick_OH.wav", "Kick_OH", group=""),  # falls back to track.group
            TransferEntry("Quad.wav", "Quad.wav", "Quad"),
        ],
    )
    plan = build_protools_creation_plan(
        ptxt, sess,
        assignments={"Kick_In.wav": sub_id, "Kick_OH.wav": sub_id, "Quad.wav": sub_id},
        folders=folders,
        track_order={sub_id: ["Kick_OH.wav", "Kick_In.wav"]},
        bus_fader_overrides={"KIK": -3.5},
    )

    assert plan.session_name == "DemoSong"
    assert plan.audio_spec.sample_rate == 48000 and plan.audio_spec.bit_depth == 24
    assert [t.track_name for t in plan.tracks] == ["Kick_OH", "Kick_In"]  # Honored track_order!
    assert plan.tracks[0].track_format == "Stereo" and plan.tracks[1].track_format == "Mono"
    assert all(t.output_bus_name == "KIK" and t.output_bus_uuid_hex for t in plan.tracks)
    assert all(t.color_index == 25 for t in plan.tracks)
    assert "Quad.wav" in plan.excluded_tracks
    assert len(plan.group_actions) >= 1 and plan.group_actions[0].folder_name == "Sub"
    assert plan.group_actions[0].member_tracks == ("Kick_OH", "Kick_In")
    assert plan.cleanup_plan is not None
    assert plan.signal_flow is not None and "Kick_In" in plan.signal_flow.nodes
    assert plan.recommended_faders_db.get("KIK") == -3.5
    assert "Kick_In" in plan.recommended_faders_db
    assert isinstance(plan.recommended_slot_a_trims_db, dict)
    assert plan.template_audit is not None


def test_build_protools_creation_plan_edge_cases_and_guards(tmp_path: Path):
    ptxt = tmp_path / "NoInput.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=False))
    folders = read_template_folders_from_ptxt(ptxt)
    sub_id = next(f["id"] for f in folders if f["name"] == "Sub")

    w1 = _write_wav(tmp_path / "KIK.wav", 48000, 1)
    t1 = TrackContext("KIK.wav", str(w1), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
    t_miss = TrackContext("Missing.wav", str(tmp_path / "gone.wav"), None, 48000, 1, 4800, "24", "PCM_24", 0.1)

    # Stale prepared state raises ValueError
    stale = SessionContext(
        tracks=[t1], output_tracks=[t1], config={"_use_processed": True}, prepare_state="stale",
        transfer_manifest=[TransferEntry("KIK.wav", "KIK.wav", "KIK")],
    )
    with pytest.raises(ValueError, match="out of date"):
        build_protools_creation_plan(ptxt, stale, assignments={"KIK.wav": sub_id}, folders=folders)

    # Template missing 0x2519 raises ValueError
    dec0, delta0 = decrypt_ptx(_build_synthetic_ptx())
    bad_ptxt = tmp_path / "No2519.ptxt"
    bad_ptxt.write_bytes(encrypt_ptx(dec0[:0x1F] + build_zchunk(0x1029, b"\x00" * 8, version=1), delta0))
    with pytest.raises(ValueError, match="0x2519"):
        build_protools_creation_plan(bad_ptxt, SessionContext([], [], {}), assignments={}, folders=[])

    # Collision with existing template track ("KIK"), missing manifest entry, non-folder id, unrouted warning,
    # missing source file exclusion, null gui lists, blank project name, and compute_signal_flow=False
    sess = SessionContext(
        tracks=[t1, t_miss],
        output_tracks=[t1, t_miss],
        project_name="   ",
        config={"gui": {"groups": None, "colors": None}},
        transfer_manifest=[
            TransferEntry("e1", "KIK.wav", "KIK"),
            TransferEntry("e2", "KIK.wav", "   "),
            TransferEntry("e3", "Missing.wav", "Missing"),
        ],
    )
    plan = build_protools_creation_plan(
        ptxt, sess,
        assignments={"e1": sub_id, "e2": sub_id, "e3": sub_id, "ghost": sub_id},
        folders=folders,
        compute_signal_flow=False,
    )
    assert plan.session_name == "Session"
    assert plan.signal_flow is None
    assert [t.track_name for t in plan.tracks] == ["KIK_2", "KIK_3"]
    assert all(t.output_bus_name is None for t in plan.tracks)
    assert "Missing.wav" in plan.excluded_tracks
    assert any("no ancestor Routing Folder bus" in w for w in plan.warnings)
    assert any("Skipped assignment 'ghost'" in w for w in plan.warnings)

    # auto_route=False skips routing cleanly without unrouted warnings
    plan_no_route = build_protools_creation_plan(
        ptxt, sess, assignments={"e1": sub_id}, folders=folders, auto_route=False, compute_signal_flow=False,
    )
    assert plan_no_route.tracks[0].output_bus_name is None
    assert not any("no ancestor Routing Folder bus" in w for w in plan_no_route.warnings)


def test_explicit_ptxt_path_cross_platform_offline_connectivity_and_fetch(tmp_path: Path, monkeypatch):
    dir_a = tmp_path / "projA"
    dir_b = tmp_path / "projB"
    dir_a.mkdir()
    dir_b.mkdir()
    ptxt_a = dir_a / "Mix.ptxt"
    ptxt_b = dir_b / "Mix.ptxt"
    ptxt_a.write_bytes(_build_nested_folder_ptx(folder_has_input=True))
    ptxt_b.write_bytes(_build_nested_folder_ptx(folder_has_input=False))
    proj_dir = tmp_path / "Projects"
    proj_dir.mkdir()

    # Backward-compatible group + name entry alongside two same-basename explicit ptxt_path entries on Linux
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    instances = ProToolsDawProcessor.create_instances({
        "protools_templates": [
            {"group": "bnz", "name": "LegacyTpl"},
            {"ptxt_path": str(ptxt_a)},
            {"path": str(ptxt_b)},
        ]
    })
    assert [p.name for p in instances] == [
        "Pro Tools \u2013 bnz / LegacyTpl",
        "Pro Tools \u2013 Mix",
        "Pro Tools \u2013 Mix",
    ]
    proc_a, proc_b = instances[1], instances[2]
    assert proc_a.template_identity != proc_b.template_identity

    # Non-localhost protools_host must NOT block offline_ptx connectivity or offline fetch!
    cache_store: dict[str, dict] = {}
    for p in (proc_a, proc_b):
        p.configure({
            "protools_transfer_backend": "offline_ptx",
            "protools_project_dir": str(proj_dir),
            "protools_host": "192.168.1.99",
        })
        monkeypatch.setattr(p, "_load_template_cache", lambda: (tmp_path / "cache.json", cache_store))
        monkeypatch.setattr(p, "_save_template_cache", lambda _cf, cd, ck, mt, fl: cd.__setitem__(ck, {"mtime": mt, "folders": fl}))

    ok, msg = proc_a.check_connectivity()
    assert ok is True and "Mix.ptxt" in msg

    sess = SessionContext(tracks=[], output_tracks=[], config={})
    proc_a.fetch(sess, ignore_cache=False)
    proc_b.fetch(sess, ignore_cache=False)
    assert len(cache_store) == 2  # Distinct cache keys for same-basename explicit paths!
    assert [f["name"] for f in sess.daw_state[proc_a.id]["folders"]] == ["KIK", "Sub"]

    # Missing project_dir, directory-as-ptxt_path, corrupt file, invalid path chars, and unknown backend
    proc_a.configure({"protools_transfer_backend": "offline_ptx", "protools_project_dir": str(tmp_path / "nope")})
    assert proc_a.check_connectivity()[0] is False

    corrupt = tmp_path / "Bad.ptxt"
    corrupt.write_bytes(b"not a ptx file")
    for bad_path in (str(tmp_path / "Missing.ptxt"), str(proj_dir), str(corrupt), "bad|path*?.ptxt"):
        bad_proc = ProToolsDawProcessor(instance_index=9, instance_ptxt_path=bad_path)
        bad_proc.configure({"protools_transfer_backend": "offline_ptx", "protools_project_dir": str(proj_dir)})
        assert bad_proc.check_connectivity()[0] is False
        monkeypatch.setattr(bad_proc, "_load_template_cache", lambda: (tmp_path / "cache.json", {}))
        with pytest.raises(RuntimeError, match="offline fetch"):
            bad_proc.fetch(sess, ignore_cache=True)

    proc_a.configure({"protools_transfer_backend": "bogus_mode", "protools_project_dir": str(proj_dir)})
    ok_bad, msg_bad = proc_a.check_connectivity()
    assert ok_bad is False and "Unknown Pro Tools transfer backend" in msg_bad


def test_strip_known_audio_ext_preserves_dots_in_track_names():
    from sessionpreplib.daw_processors.protools_ptx_plan import _strip_known_audio_ext

    assert _strip_known_audio_ext("Kick.wav") == "Kick"
    assert _strip_known_audio_ext("Snare.AIFF") == "Snare"
    assert _strip_known_audio_ext("Vox.flac") == "Vox"
    assert _strip_known_audio_ext("Dr. Kick") == "Dr. Kick"
    assert _strip_known_audio_ext("Gtr v1.2") == "Gtr v1.2"
    assert _strip_known_audio_ext("Synth.L") == "Synth.L"
    assert _strip_known_audio_ext("Dr. Kick.wav") == "Dr. Kick"
