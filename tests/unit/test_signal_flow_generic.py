"""Unit tests for DAW-agnostic signal-flow DAG engine, host gain limits, leaf trim, and Trim codecs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext
from sessionpreplib.signal_flow import (
    DAWPROJECT_GAIN_LIMITS,
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    PROTOOLS_GAIN_LIMITS,
    LeafAudioInput,
    MixerInsert,
    MixerSend,
    MixerSession,
    MixerTrack,
    TrimPluginInfo,
    build_signal_flow,
    compute_signal_flow_graph_layout,
    decode_trackcontrol_dgct_zip,
    decompress_bounded,
    extract_trackcontrol_gain,
    patch_protility_vc2_xml,
    patch_trackcontrol_dgct_zip,
    read_protility_vc2_xml,
    resolve_track_staging,
    select_preferred_trim_donor,
    trace_signal_flow_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sine_stereo(db_peak: float, sr: int = 48000, dur_sec: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * dur_sec), dtype=np.float64) / float(sr)
    amp = 10.0 ** (db_peak / 20.0)
    sig = amp * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack((sig, sig))


def _make_trackcontrol_trim(gain_db: float = 0.0, *, is_active: bool = True, is_clean: bool = True) -> TrimPluginInfo:
    return TrimPluginInfo(
        plugin_id="trackcontrol",
        plugin_name="TrackControl",
        gain_db=gain_db,
        min_db=-200.0,
        max_db=18.0,
        is_active=is_active,
        is_bypassed=False,
        is_clean_donor=is_clean,
        precision_decimals=1,
    )


def test_signal_flow_package_has_no_ptx_imports():
    """sessionpreplib/signal_flow must remain strictly independent of sessionpreplib/ptx."""
    sf_dir = REPO_ROOT / "sessionpreplib" / "signal_flow"
    violations: list[str] = []
    for path in sorted(sf_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "ptx" in alias.name.split("."):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if "ptx" in node.module.split("."):
                    violations.append(f"{path.name}:{node.lineno} imports {node.module}")
    assert not violations, "signal_flow must not import ptx:\n" + "\n".join(violations)


def test_dawproject_6db_fader_ceiling_and_iterative_leaf_convergence():
    """DAWPROJECT_GAIN_LIMITS (+6 dB fader max) clamps intermediate trim compensation and converges via leaf shift."""
    # Two loud sustained tracks (+12 dB combined on subgroup bus -> needs ~-9 to -12 dB trim).
    # At +6 dB max fader, intermediate bus can only compensate +6.0 dB without leaf shift.
    trim = _make_trackcontrol_trim(0.0)
    mixer = MixerSession(
        sample_rate=48000,
        tracks=(
            MixerTrack("Vox 1", "Audio", "Stereo", index=0, output_bus="VOX BUS"),
            MixerTrack("Vox 2", "Audio", "Stereo", index=1, output_bus="VOX BUS"),
            MixerTrack("Pad", "Audio", "Stereo", index=2, output_bus="MIX BUS"),
            MixerTrack(
                "VOX BUS", "Aux", "Stereo", index=3, input_bus="VOX BUS", output_bus="MIX BUS",
                inserts=(MixerInsert(0, "TrackControl"), MixerInsert(1, "Comp")),
                slot_a_trim=trim,
            ),
            MixerTrack(
                "MIX BUS", "Aux", "Stereo", index=4, input_bus="MIX BUS", output_bus="Master",
                inserts=(MixerInsert(0, "TrackControl"),),
                slot_a_trim=trim,
            ),
            MixerTrack("Master", "Master", "Stereo", index=5, output_bus="Master"),
        ),
    )
    # 2 identical -12 dBFS RMS sines -> summed pre-insert RMS = -6 dBFS (needs -12 dB to reach -18 dBFS)
    sig = _sine_stereo(-9.0)  # RMS ~ -12.01 dBFS; 2 coherent tracks -> -6.0 dBFS RMS
    pad_sig = _sine_stereo(-24.0, freq=220.0)
    leaves = {
        "Vox 1": LeafAudioInput(sig, 48000, trim_gain_db=0.0, fader_db=0.0),
        "Vox 2": LeafAudioInput(sig, 48000, trim_gain_db=0.0, fader_db=0.0),
        "Pad": LeafAudioInput(pad_sig, 48000, trim_gain_db=0.0, fader_db=0.0),
    }
    cfg = {"target_peak": -6.0, "target_rms": -18.0}

    # Without leaf convergence, +6 dB fader ceiling leaves excess and marks unconverged while preserving balance
    g_unc = build_signal_flow(
        mixer, leaves, cfg,
        host_limits=DAWPROJECT_GAIN_LIMITS,
        stage_bus_faders=True,
        attenuate_only=True,
        allow_leaf_convergence=False,
    )
    vox = g_unc.nodes["VOX BUS"]
    assert vox.recommended_fader_db == pytest.approx(6.0, abs=0.05)
    assert vox.recommended_trim_db == pytest.approx(-6.0, abs=0.05)
    assert vox.trim_clamped_excess_db > 5.0
    assert g_unc.balance_validation["status"] == "unconverged"
    assert g_unc.balance_validation["max_balance_spread_db"] <= 0.15

    # With leaf convergence, global negative leaf shift relieves the +6 dB ceiling cleanly
    g_conv = build_signal_flow(
        mixer, leaves, cfg,
        host_limits=DAWPROJECT_GAIN_LIMITS,
        stage_bus_faders=True,
        attenuate_only=True,
        allow_leaf_convergence=True,
    )
    assert g_conv.leaf_fader_shift_db < -4.0
    assert g_conv.nodes["VOX BUS"].recommended_fader_db <= 6.0
    assert g_conv.nodes["VOX BUS"].trim_clamped_excess_db <= 0.15
    assert g_conv.balance_validation["status"] == "converged_shifted"
    assert g_conv.balance_validation["converged"] is True
    assert g_conv.balance_validation["max_balance_spread_db"] <= 0.15

    # Compare against Pro Tools +12 dB ceiling: needs no leaf shift for the same -12 dB trim!
    g_pt = build_signal_flow(
        mixer, leaves, cfg,
        host_limits=PROTOOLS_GAIN_LIMITS,
        stage_bus_faders=True,
        attenuate_only=True,
        allow_leaf_convergence=True,
    )
    assert g_pt.leaf_fader_shift_db == 0.0
    assert g_pt.nodes["VOX BUS"].recommended_fader_db == pytest.approx(12.0, abs=0.1)
    assert g_pt.balance_validation["status"] == "verified"


def test_leaf_trim_gain_staging_balance_caveat_and_equation():
    """Leaf Slot 1 Trim (e.g. DAWproject TrackControl) integrates with summing, balance, caveats, and equations."""
    mixer = MixerSession(
        sample_rate=48000,
        tracks=(
            MixerTrack(
                "Kick", "Audio", "Stereo", index=0, output_bus="DRUMS",
                sends=(MixerSend("Verb", slot="Send 1", volume_db=-6.0, is_pre_fader=True),),
            ),
            MixerTrack("Snare", "Audio", "Stereo", index=1, output_bus="DRUMS"),
            MixerTrack("DRUMS", "Aux", "Stereo", index=2, input_bus="DRUMS", output_bus="Master"),
            MixerTrack("Verb", "Aux", "Stereo", index=3, input_bus="Verb", output_bus="Master"),
            MixerTrack("Master", "Master", "Stereo", index=4, output_bus="Master"),
        ),
    )
    sig = _sine_stereo(-23.0)  # RMS ~ -26 dBFS; +8 dB Trim hits -18 dBFS RMS
    leaves = {
        "Kick": LeafAudioInput(sig, 48000, clip_gain_db=0.0, trim_gain_db=8.0, fader_db=-8.0),
        "Snare": LeafAudioInput(sig, 48000, clip_gain_db=0.0, trim_gain_db=8.0, fader_db=-8.0),
    }
    g = build_signal_flow(
        mixer, leaves, {"target_peak": -6.0, "target_rms": -18.0},
        host_limits=DAWPROJECT_GAIN_LIMITS,
        stage_bus_faders=False,
        attenuate_only=True,
    )
    kick = g.nodes["Kick"]
    assert kick.leaf_trim_db == pytest.approx(8.0, abs=0.01)
    assert kick.clip_gain_db == 0.0
    assert kick.pre_fader_peak_db == pytest.approx(-15.0, abs=0.1)
    assert kick.post_fader_peak_db == pytest.approx(-23.0, abs=0.1)
    assert kick.net_gain_delta_db == pytest.approx(0.0, abs=0.01)

    # Pre-fader send caveat detects +8.0 dB shift relative to post-fader main
    assert g.balance_validation["status"] == "caveat"
    assert g.balance_validation["global_path_delta_db"] == pytest.approx(0.0, abs=0.05)
    caveats = g.balance_validation["pre_fader_send_caveats"]
    assert len(caveats) == 1
    assert caveats[0]["source"] == "Kick"
    assert caveats[0]["shift_db"] == pytest.approx(8.0, abs=0.1)

    # Path trace equation explicitly shows Trim instead of CG
    tr = trace_signal_flow_path(g.to_dict(), "Kick")
    assert tr.total_delta_db == pytest.approx(0.0, abs=0.05)
    assert "+8.0 Trim -8.0 Fdr = 0.0" in tr.equation_text

    # Layout succeeds cleanly on DAW-agnostic graph
    layout = compute_signal_flow_graph_layout(g, active_only=True)
    assert "Kick" in layout.nodes and "DRUMS" in layout.nodes

    # Lazy leaf_loader + 4-tuple leaf_gains=(clip, fader, baked, trim) parity
    g_lazy = build_signal_flow(
        mixer, None, {"target_peak": -6.0, "target_rms": -18.0},
        host_limits=DAWPROJECT_GAIN_LIMITS,
        stage_bus_faders=False,
        attenuate_only=True,
        leaf_gains={"Kick": (0.0, -8.0, 0.0, 8.0), "Snare": (0.0, -8.0, 0.0, 8.0)},
        leaf_loader=lambda name: leaves.get(name),
    )
    assert g_lazy.nodes["Kick"].leaf_trim_db == pytest.approx(8.0, abs=0.01)
    assert g_lazy.nodes["Kick"].post_fader_peak_db == pytest.approx(kick.post_fader_peak_db, abs=0.01)


def test_custom_quantizer_and_synthetic_donor_trim_by_format():
    """Custom TrimPluginInfo.quantize_db and MixerSession.donor_trim_by_format work without template donors."""
    half_db_trim = TrimPluginInfo(
        plugin_id="custom_step",
        plugin_name="StepTrim",
        gain_db=0.0,
        min_db=-60.0,
        max_db=12.0,
        is_active=True,
        is_bypassed=False,
        is_clean_donor=True,
        precision_decimals=2,
        quantize_db=lambda db: round(db * 2.0) / 2.0,
    )
    mixer = MixerSession(
        sample_rate=48000,
        tracks=(
            MixerTrack("Gtr L", "Audio", "Stereo", index=0, output_bus="GTRS"),
            MixerTrack("Gtr R", "Audio", "Stereo", index=1, output_bus="GTRS"),
            MixerTrack("Bass", "Audio", "Stereo", index=2, output_bus="MIX"),
            MixerTrack(
                "GTRS", "Aux", "Stereo", index=3, input_bus="GTRS", output_bus="MIX",
                inserts=(MixerInsert(0, "StepTrim"),),
                slot_a_trim=half_db_trim,
            ),
            # Empty Slot A on terminal MIX bus -> auto-inserted from donor_trim_by_format!
            MixerTrack("MIX", "Aux", "Stereo", index=4, input_bus="MIX", output_bus="Master"),
            MixerTrack("Master", "Master", "Stereo", index=5, output_bus="Master"),
        ),
        donor_trim_by_format={"Stereo": half_db_trim},
    )
    sig = _sine_stereo(-13.3)  # non-half-dB level to exercise 0.5 dB step quantizer
    leaves = {
        "Gtr L": LeafAudioInput(sig, 48000),
        "Gtr R": LeafAudioInput(sig, 48000),
        "Bass": LeafAudioInput(_sine_stereo(-14.0, freq=110.0), 48000),
    }
    g = build_signal_flow(
        mixer, leaves, {"target_peak": -6.0, "target_rms": -18.0},
        stage_bus_faders=True,
        attenuate_only=True,
    )
    gtrs = g.nodes["GTRS"]
    mix = g.nodes["MIX"]
    assert gtrs.staged is True
    # Quantized strictly to 0.5 dB increments
    assert (gtrs.recommended_trim_db * 2.0) == pytest.approx(round(gtrs.recommended_trim_db * 2.0), abs=1e-6)
    assert gtrs.slot_a_trim_precision == 2
    assert "GTRS" in g.recommended_slot_a_trims()
    assert mix.is_terminal is True
    assert mix.slot_a_trim_auto_insert is True
    assert mix.slot_a_trim_plugin == "StepTrim"
    assert "MIX" in g.recommended_slot_a_trims()


def test_resolve_track_staging_all_modes_and_prepared_audio_guard(tmp_path):
    """resolve_track_staging supports clip, trim, fader-only, and none modes without double-applying gain."""
    orig = tmp_path / "kick.wav"
    proc = tmp_path / "kick_prep.wav"
    orig.write_bytes(b"raw")
    proc.write_bytes(b"prep")

    res = ProcessorResult(
        processor_id="bimodal_normalize",
        classification="Transient",
        method="Peak",
        gain_db=7.5,
        data={"fader_offset": -7.5},
    )
    track = TrackContext(
        filename="kick.wav",
        filepath=str(orig),
        audio_data=None,
        samplerate=48000,
        channels=2,
        total_samples=48000,
        bitdepth="24-bit",
        subtype="PCM_24",
        duration_sec=1.0,
        processed_filepath=str(proc),
        prepared_fader_db=-7.5,
        processor_results={"bimodal_normalize": res},
    )
    raw_session = SessionContext(tracks=[track], config={"_use_processed": False})
    prep_session = SessionContext(tracks=[track], config={"_use_processed": True}, prepare_state="ready")

    # Raw audio: Trim + Fader puts +7.5 on trim_gain_db and -7.5 on fader_db, 0 on clip_gain_db
    p_tf = resolve_track_staging(raw_session, track, GAIN_MODE_TRIM_AND_FADER)
    assert (p_tf.clip_gain_db, p_tf.trim_gain_db, p_tf.fader_db, p_tf.baked_gain_db) == (0.0, 7.5, -7.5, 0.0)

    # Raw audio: Trim Unity puts +7.5 on trim_gain_db and 0.0 on fader_db
    p_tu = resolve_track_staging(raw_session, track, GAIN_MODE_TRIM_UNITY)
    assert (p_tu.clip_gain_db, p_tu.trim_gain_db, p_tu.fader_db) == (0.0, 7.5, 0.0)

    # Raw audio: Clip + Fader vs Clip Unity vs Fader Only vs None
    p_cf = resolve_track_staging(raw_session, track, GAIN_MODE_CLIP_AND_FADER)
    assert (p_cf.clip_gain_db, p_cf.trim_gain_db, p_cf.fader_db) == (7.5, 0.0, -7.5)
    p_cu = resolve_track_staging(raw_session, track, GAIN_MODE_CLIP_UNITY)
    assert (p_cu.clip_gain_db, p_cu.trim_gain_db, p_cu.fader_db) == (7.5, 0.0, 0.0)
    p_fo = resolve_track_staging(raw_session, track, GAIN_MODE_FADER_ONLY)
    assert (p_fo.clip_gain_db, p_fo.trim_gain_db, p_fo.fader_db) == (0.0, 0.0, 0.0)
    p_no = resolve_track_staging(raw_session, track, GAIN_MODE_NONE)
    assert (p_no.clip_gain_db, p_no.trim_gain_db, p_no.fader_db) == (0.0, 0.0, 0.0)

    # Prepared (gain-baked) WAV: never double-applies Trim or Clip Gain!
    p_prep = resolve_track_staging(prep_session, track, GAIN_MODE_TRIM_AND_FADER)
    assert p_prep.filepath == str(proc)
    assert p_prep.clip_gain_db == 0.0
    assert p_prep.trim_gain_db == 0.0
    assert p_prep.baked_gain_db == pytest.approx(7.5)
    assert p_prep.fader_db == pytest.approx(-7.5)


def test_resolve_track_staging_clamps_leaf_trim_symmetrically(tmp_path):
    """Host limits symmetrically clamp TrackControl +18 dB leaf trim ceiling & inverse fader."""
    orig = tmp_path / "quiet.wav"
    orig.write_bytes(b"raw")
    quiet_res = ProcessorResult(
        processor_id="bimodal_normalize",
        classification="Sustained",
        method="RMS",
        gain_db=22.0,
        data={"fader_offset": -22.0},
    )
    quiet_trk = TrackContext(
        filename="quiet.wav",
        filepath=str(orig),
        audio_data=None,
        samplerate=48000,
        channels=2,
        total_samples=48000,
        bitdepth="24-bit",
        subtype="PCM_24",
        duration_sec=1.0,
        processor_results={"bimodal_normalize": quiet_res},
    )
    quiet_session = SessionContext(tracks=[quiet_trk], config={"_use_processed": False})
    p_clamped = resolve_track_staging(
        quiet_session, quiet_trk, GAIN_MODE_TRIM_AND_FADER, limits=DAWPROJECT_GAIN_LIMITS,
    )
    assert p_clamped.trim_gain_db == pytest.approx(18.0)
    assert p_clamped.fader_db == pytest.approx(-18.0)
    assert p_clamped.trim_gain_db + p_clamped.fader_db == pytest.approx(0.0)


def test_synthetic_donor_trim_without_preexisting_template_trims():
    """MixerSession.donor_trim_by_format enables bus Slot A auto-insert even with zero template trims."""
    donor = _make_trackcontrol_trim(0.0)
    # Converging multi-bus DAG with empty Slot A on every bus (typical DAWproject/LUNA scenario)
    mixer = MixerSession(
        sample_rate=48000,
        tracks=(
            MixerTrack("Vox 1", "Audio", "Stereo", index=0, output_bus="VOX"),
            MixerTrack("Vox 2", "Audio", "Stereo", index=1, output_bus="VOX"),
            MixerTrack("Inst", "Audio", "Stereo", index=2, output_bus="MIX"),
            MixerTrack(
                "VOX", "Aux", "Stereo", index=3, input_bus="VOX", output_bus="MIX",
                inserts=(MixerInsert(1, "Comp"),),  # Slot 0 is empty!
            ),
            MixerTrack("MIX", "Aux", "Stereo", index=4, input_bus="MIX", output_bus="Master"),
            MixerTrack("Master", "Master", "Stereo", index=5, output_bus="Master"),
        ),
        donor_trim_by_format={"Stereo": donor},
    )
    sig = _sine_stereo(-12.0)
    leaves = {
        "Vox 1": LeafAudioInput(sig, 48000, trim_gain_db=0.0, fader_db=0.0),
        "Vox 2": LeafAudioInput(sig, 48000, trim_gain_db=0.0, fader_db=0.0),
        "Inst": LeafAudioInput(_sine_stereo(-18.0, freq=220.0), 48000),
    }
    g = build_signal_flow(
        mixer, None, {"target_peak": -6.0, "target_rms": -18.0},
        host_limits=DAWPROJECT_GAIN_LIMITS,
        stage_bus_faders=True,
        attenuate_only=True,
        allow_leaf_convergence=True,
        leaf_gains={"Vox 1": (0.0, 0.0, 0.0, 0.0), "Vox 2": (0.0, 0.0, 0.0, 0.0), "Inst": (0.0,)},
        leaf_loader=lambda name: leaves.get(name),
    )
    assert g.nodes["VOX"].slot_a_trim_auto_insert is True
    assert g.nodes["VOX"].staging_strategy == "trim_and_fader"
    assert g.nodes["MIX"].is_terminal is True
    assert g.nodes["MIX"].slot_a_trim_auto_insert is True
    assert g.nodes["MIX"].staging_strategy == "trim_only"
    assert "VOX" in g.recommended_slot_a_trims() and "MIX" in g.recommended_slot_a_trims()
    assert g.balance_validation["converged"] is True

    # Single-input bus chain with zero pre-existing trims still elects terminal sink
    single_mixer = MixerSession(
        sample_rate=48000,
        tracks=(
            MixerTrack("Lead", "Audio", "Stereo", index=0, output_bus="STEM"),
            MixerTrack("STEM", "Aux", "Stereo", index=1, input_bus="STEM", output_bus="Master"),
            MixerTrack("Master", "Master", "Stereo", index=2, output_bus="Master"),
        ),
        donor_trim_by_format={"Stereo": donor},
    )
    g_single = build_signal_flow(
        single_mixer, {"Lead": LeafAudioInput(_sine_stereo(-6.0), 48000)},
        {"target_peak": -6.0, "target_rms": -18.0},
        stage_bus_faders=True,
        attenuate_only=True,
    )
    assert g_single.nodes["STEM"].is_terminal is True
    assert g_single.nodes["STEM"].slot_a_trim_auto_insert is True
    assert g_single.nodes["STEM"].staging_strategy == "trim_only"


def test_non_finite_and_out_of_range_hardening():
    """Public trim_codecs and generic solver safely reject or clamp non-finite and extreme inputs."""
    tc_zip = zlib.compress(json.dumps({"settings": {"currentAB": 0}, "ab": [{"parameters": "default"}]}).encode())
    with pytest.raises(ValueError, match="Non-finite"):
        patch_trackcontrol_dgct_zip(tc_zip, float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        patch_trackcontrol_dgct_zip(tc_zip, float("inf"))
    # Out-of-range +30 dB clamps cleanly to TrackControl +18 dB ceiling (norm == 1.0)
    clamped_zip, clamped_norm = patch_trackcontrol_dgct_zip(tc_zip, 30.0)
    assert clamped_norm == pytest.approx(1.0)
    js_c, _ = decode_trackcontrol_dgct_zip(clamped_zip)
    assert extract_trackcontrol_gain(js_c) == (18.0, True)

    xml = b'<PROTILITY activeSlot="A"><slotA PRESET_MODIFIED="0"><PARAM id="GAIN" value="0.0"/></slotA></PROTILITY>'
    with pytest.raises(ValueError, match="Non-finite"):
        patch_protility_vc2_xml(xml, float("nan"))
    assert read_protility_vc2_xml(patch_protility_vc2_xml(xml, 50.0)) == (24.0, True)
    assert read_protility_vc2_xml(patch_protility_vc2_xml(xml, -99.0)) == (-60.0, True)

    # Non-finite track/send values and negative sample rate do not crash or poison solver
    bad_quant = TrimPluginInfo("bad", "BadTrim", float("nan"), -60.0, 12.0, True, False, quantize_db=lambda _: float("nan"))
    mixer = MixerSession(
        sample_rate=-1,
        tracks=(
            MixerTrack(
                "A", "Audio", "Stereo", index=0, output_bus="BUS",
                volume_db=float("nan"), vca_offset_db=float("inf"),
                sends=(MixerSend("BUS", "S1", volume_db=1e9, is_pre_fader=False),),
            ),
            MixerTrack(
                "BUS", "Aux", "Stereo", index=1, input_bus="BUS", output_bus="Master",
                inserts=(MixerInsert(0, "BadTrim"),), slot_a_trim=bad_quant,
            ),
            MixerTrack("Master", "Master", "Stereo", index=2, output_bus="Master"),
        ),
    )
    g = build_signal_flow(
        mixer, {"A": LeafAudioInput(_sine_stereo(-12.0), 48000)},
        {"target_peak": -6.0, "target_rms": -18.0},
        include_sends=True,
        stage_bus_faders=True,
        attenuate_only=True,
    )
    assert g.samplerate == 48000
    assert g.nodes["A"].vca_offset_db == 0.0
    assert g.nodes["BUS"].recommended_trim_db <= 12.0 and g.nodes["BUS"].recommended_trim_db >= -60.0


def test_trim_codecs_trackcontrol_protility_and_donor_selection():
    """TrackControl DGCT JSON, Protility VC2! XML, decompression limit, and donor selection work without AAX."""
    # TrackControl DGCT zlib JSON round-trip
    tc_obj = {"settings": {"currentAB": 0}, "ab": [{"parameters": "default"}], "presets": [{"parameters": {}}]}
    tc_zip = zlib.compress(json.dumps(tc_obj).encode("utf-8") + b"\x00")
    js, trailing = decode_trackcontrol_dgct_zip(tc_zip)
    assert trailing == b"\x00"
    assert extract_trackcontrol_gain(js) == (0.0, True)

    patched_zip, norm = patch_trackcontrol_dgct_zip(tc_zip, -6.5)
    js2, trailing2 = decode_trackcontrol_dgct_zip(patched_zip)
    assert trailing2 == b"\x00"
    gain_db, is_clean = extract_trackcontrol_gain(js2) or (None, False)
    assert gain_db == pytest.approx(-6.5, abs=0.02)
    assert is_clean is True
    assert 0.0 < norm < 1.0

    # Non-finite kOpGain rejected explicitly
    with pytest.raises(ValueError, match="Non-finite"):
        extract_trackcontrol_gain({"settings": {"currentAB": 0}, "ab": [{"parameters": {"kOpGain": float("nan")}}]})

    # Decompression bomb protection
    bomb = zlib.compress(b"A" * 4096)
    with pytest.raises(ValueError, match="safety limit"):
        decompress_bounded(bomb, max_bytes=1024)

    # Protility VC2! XML round-trip
    xml = (
        b'<PROTILITY activeSlot="A"><slotA PRESET_MODIFIED="0">'
        b'<PARAM id="GAIN" value="0.0"/><PARAM id="PAN" value="0.0"/>'
        b"</slotA></PROTILITY>\x00"
    )
    assert read_protility_vc2_xml(xml) == (0.0, True)
    patched_xml = patch_protility_vc2_xml(xml, -4.5)
    assert patched_xml.endswith(b"\x00")
    assert b'PRESET_MODIFIED="1"' in patched_xml
    assert read_protility_vc2_xml(patched_xml) == (-4.5, True)

    # Donor selection on MixerTrack objects prefers most-used clean unbypassed unity donor
    t_dirty = MixerTrack("Dirty", "Aux", "Stereo", index=0, slot_a_trim=_make_trackcontrol_trim(-3.0, is_clean=False))
    t_clean_nz = MixerTrack("CleanNZ", "Aux", "Stereo", index=1, slot_a_trim=_make_trackcontrol_trim(-2.0, is_clean=True))
    t_clean_z = MixerTrack("CleanZ", "Aux", "Stereo", index=2, slot_a_trim=_make_trackcontrol_trim(0.0, is_clean=True))
    t_mono = MixerTrack("Mono", "Aux", "Mono", index=3, slot_a_trim=_make_trackcontrol_trim(0.0, is_clean=True))
    chosen = select_preferred_trim_donor((t_dirty, t_clean_nz, t_clean_z, t_mono), "Stereo")
    assert chosen is t_clean_z
