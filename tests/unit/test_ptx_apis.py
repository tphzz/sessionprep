"""Unit tests for the 5-layer PTX Python API, stock plugin decoders, and SessionPrep integration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sessionpreplib.daw_processors import protools_ptx
from sessionpreplib.daw_processors.protools_ptx_preview import inspect_protools_template_topology
from sessionpreplib.ptx import (
    TICKS_PER_QUARTER,
    PtxAraPlugin,
    PtxAudioFile,
    PtxAutomationLane,
    PtxAutomationPoint,
    PtxBarBeat,
    PtxClip,
    PtxClipGainBreakpoint,
    PtxClipGainCurve,
    PtxClipPlacement,
    PtxCompoundClip,
    PtxCompoundClipMember,
    PtxFade,
    PtxFadePlacement,
    PtxGroup,
    PtxInsert,
    PtxMarker,
    PtxMeterEvent,
    PtxMidiChunk,
    PtxMidiClip,
    PtxMidiControlEvent,
    PtxMidiControlLane,
    PtxMidiNote,
    PtxMidiPlacement,
    PtxPatchBuilder,
    PtxPlaylist,
    PtxPluginAutomationLane,
    PtxPluginState,
    PtxSend,
    PtxSession,
    PtxSignalPathRef,
    PtxTempoEvent,
    PtxTrack,
    PtxVideoClip,
    PtxVideoPlacement,
    PtxWarpMarker,
    decode_insert_parameters,
    decode_plugin_automation_value,
)
from tests.unit.test_ptx_core import _build_synthetic_ptx


def _make_session() -> PtxSession:
    """Create a synthetic in-memory PtxSession populated for API tests."""
    ptx = PtxSession.from_bytes(_build_synthetic_ptx())
    ptx.info.sample_rate = 48000
    ptx.info.tempo_bpm = 120.0
    ptx.info.fps_code = 2  # 25 fps
    ptx.info.frame_rate = 25.0
    ptx.info.start_frames = 25 * 3600  # 01:00:00:00
    ptx.tempo_events = [
        PtxTempoEvent(tick=0, bpm=120.0, seconds=0.0),
        PtxTempoEvent(tick=4 * TICKS_PER_QUARTER, bpm=60.0, seconds=2.0),
    ]
    ptx.meter_events = [
        PtxMeterEvent(tick=0, numerator=4, denominator=4),
        PtxMeterEvent(tick=4 * TICKS_PER_QUARTER, numerator=7, denominator=8),
    ]
    ptx.markers = [
        PtxMarker(1, "Intro", 0, 0, start_seconds=0.0),
        PtxMarker(2, "Chorus", 48000 * 4, 48000 * 8, time_properties="Selection", start_seconds=4.0, end_seconds=8.0),
    ]
    return ptx


def test_layer1_timeline_conversions_tempo_meter_timecode_and_markers():
    ptx = _make_session()
    tl = ptx.timeline

    assert tl.samples_to_seconds(48000) == pytest.approx(1.0)
    assert tl.seconds_to_samples(1.5) == 72000
    assert tl.tempo_at_tick(0) == 120.0
    assert tl.tempo_at_tick(4 * TICKS_PER_QUARTER) == 60.0
    assert tl.meter_at_tick(0) == (4, 4)
    assert tl.meter_at_tick(4 * TICKS_PER_QUARTER) == (7, 8)

    # 4 beats at 120 BPM = 2.0 s; +2 beats at 60 BPM = 4.0 s
    t_6beats = 6 * TICKS_PER_QUARTER
    assert tl.ticks_to_seconds(t_6beats) == pytest.approx(4.0)
    assert ptx.ticks_to_seconds(t_6beats) == pytest.approx(4.0)
    assert tl.seconds_to_ticks(4.0) == t_6beats
    assert tl.ticks_to_samples(t_6beats) == 48000 * 4
    assert tl.samples_to_ticks(48000 * 4) == t_6beats

    # Bar|Beat|Ticks across 4/4 (Bar 1) -> 7/8 (Bar 2)
    assert tl.ticks_to_bar_beat(0) == PtxBarBeat(1, 1, 0)
    assert tl.ticks_to_bar_beat(4 * TICKS_PER_QUARTER) == PtxBarBeat(2, 1, 0)
    b2_e3 = 4 * TICKS_PER_QUARTER + 2 * (TICKS_PER_QUARTER // 2) + 120
    assert tl.ticks_to_bar_beat(b2_e3) == PtxBarBeat(2, 3, 120)
    assert tl.bar_beat_to_ticks(2, 3, 120) == b2_e3
    with pytest.raises(ValueError):
        tl.bar_beat_to_ticks(0, 1, 0)

    # Timecode with and without session start offset (01:00:00:00 @ 25 fps)
    assert tl.seconds_to_timecode(2.0, include_session_start=True) == "01:00:02:00"
    assert tl.samples_to_timecode(96000, include_session_start=False) == "00:00:02:00"

    # Marker queries & duration
    assert [m.name for m in tl.markers_in_range(1.0, 5.0)] == ["Chorus"]
    assert tl.marker_by_number(1).name == "Intro"
    assert tl.marker_by_name("chorus").number == 2
    assert tl.session_duration_seconds() >= 8.0


def test_layer2_automation_and_clip_gain_curves():
    ptx = _make_session()
    tl = ptx.timeline

    lane_s = PtxAutomationLane(
        "volume",
        "Samples",
        [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, -12.0)],
    )
    assert lane_s.is_automated is True
    assert lane_s.initial_value == 0.0
    assert lane_s.value_at(24000) == pytest.approx(-6.0)
    assert lane_s.value_at(24000, stepped=True) == pytest.approx(0.0)
    assert lane_s.value_at_seconds(0.5, tl) == pytest.approx(-6.0)
    assert lane_s.points_in_seconds(tl) == [(0.0, 0.0), (1.0, -12.0)]

    lane_t = PtxAutomationLane(
        "volume",
        "Ticks",
        [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(4 * TICKS_PER_QUARTER, -8.0)],
    )
    assert lane_t.value_at_seconds(1.0, tl) == pytest.approx(-4.0)

    flat_lane = PtxAutomationLane("volume", "Samples", [PtxAutomationPoint(0, -3.0), PtxAutomationPoint(1000, -3.0)])
    assert flat_lane.is_automated is False

    curve = PtxClipGainCurve(
        0, 0, "Samples",
        [PtxClipGainBreakpoint(0, 0.0, 0.0), PtxClipGainBreakpoint(48000, 6.0, 1.0)],
    )
    assert curve.is_dynamic is True
    assert curve.initial_gain_db == 0.0
    assert curve.gain_db_at(24000) == pytest.approx(3.0)


def test_layer2_track_helpers_and_effective_mix_state():
    ptx = _make_session()
    tl = ptx.timeline
    kick = ptx.track_by_name("Kick")
    vca = ptx.track_by_name("KIK-VCA")
    assert kick is not None and vca is not None
    kick.volume_automation = PtxAutomationLane(
        "volume", "Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, -12.0)],
    )
    kick.groups = ["KIK_GRP"]
    vca.volume_db = -3.0
    ptx.groups.append(PtxGroup(0, "KIK_GRP", "mix", "Mix/Edit", 1, "a", vca_track_name="KIK-VCA", member_names=["Kick"]))
    kick.sends = [
        PtxSend("A", 0, PtxSignalPathRef("FX1", "bus", "00"), volume_db=-6.0,
                volume_automation=PtxAutomationLane("send_volume", "Samples", [PtxAutomationPoint(0, -10.0), PtxAutomationPoint(48000, -2.0)]))
    ]
    plug_lane = PtxPluginAutomationLane("gai2", "LF Gain", "Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, 0.25)])
    kick.inserts = [
        PtxInsert("A", 0, 0, "EQ3 7-Band", "com.avid.eq3", "Digi/EQ7B/EQ7B",
                  state=PtxPluginState(controls={"gai2": 3.5}), automation=[plug_lane])
    ]

    assert kick.insert_by_slot("a").plugin_name == "EQ3 7-Band"
    assert kick.insert_by_slot("A").control_value("gai2") == 3.5
    assert kick.insert_by_slot("A").automation_by_tag("gai2") is plug_lane
    assert kick.insert_by_slot("A").automation_by_name("lf gain") is plug_lane
    assert plug_lane.is_automated is True
    assert plug_lane.value_at_seconds(0.5, tl) == pytest.approx(0.125)
    assert kick.send_by_slot("a").destination.name == "FX1"

    assert ptx.effective_track_volume_db("Kick", 0.5) == pytest.approx(-9.0)
    assert ptx.effective_track_volume_db("Kick", 0.5, include_vca=False) == pytest.approx(-6.0)
    assert ptx.is_track_audible("Kick", 0.5) is True
    assert ptx.effective_send_volume_db("Kick", "A", 0.5) == pytest.approx(-6.0)

    vca.is_muted = True
    assert ptx.effective_track_volume_db("Kick", 0.5) == -144.0
    assert ptx.is_track_audible("Kick", 0.5) is False


def test_layer3_resolve_audio_events_fades_compounds_loops_sync_and_overlaps():
    ptx = _make_session()
    ptx.audio_files = [
        PtxAudioFile(0, 1, "kick.wav", "Audio Files", 48000, 1, 24, 96000, 2.0, 0, "C:\\S\\Audio Files\\kick.wav", "Audio Files/kick.wav"),
        PtxAudioFile(1, 2, "ext.wav", "External", 48000, 1, 24, 48000, 1.0, 0, "D:\\Lib\\ext.wav", None),
    ]
    ptx.clip_gain_curves = [PtxClipGainCurve(0, 0, "Samples", [PtxClipGainBreakpoint(0, 2.5, 0.0)])]
    ptx.clips = [
        PtxClip(index=0, name="C1", clip_id="c1", start_sample=1000, length_samples=48000, clip_gain_curve_index=0, gain_db=2.5, audio_file_index=0, timeline_length_samples=48000, sync_point_samples=500),
        PtxClip(index=1, name="C2", clip_id="c2", start_sample=0, length_samples=48000, clip_gain_curve_index=0, gain_db=0.0, audio_file_index=0, timeline_length_samples=48000),
        PtxClip(index=2, name="CMem", clip_id="c3", start_sample=0, length_samples=24000, clip_gain_curve_index=0, gain_db=1.0, audio_file_index=0, timeline_length_samples=24000),
    ]
    ptx.fades = [
        PtxFade(0, "in", 0, 1000),
        PtxFade(1, "out", 1000, 1000),
        PtxFade(2, "crossfade", 500, 1000),
    ]
    ptx.compound_clips = [
        PtxCompoundClip(0, "Grp1", 1000, 24000, [PtxCompoundClipMember(2, 1000, 0)]),
    ]
    kick = ptx.track_by_name("Kick")
    assert kick is not None
    kick.clips = [
        PtxClipPlacement(0, 0, "C1", "kick.wav", 1000, 48000, 49000, is_covered=True),  # adj_start=500, end=48500
        PtxClipPlacement(0, 1, "C2", "kick.wav", 30000, 48000, 78000, loop_count=2),    # len=96000, overlaps C1!
        PtxClipPlacement(0, 0, "Grp1", "", 200000, 24000, 224000, is_compound=True),
        PtxClipPlacement(0, 1, "C2_Muted", "kick.wav", 300000, 48000, 348000, is_muted=True),
    ]
    kick.fade_placements = [
        PtxFadePlacement(0, 0, 500, "in", 0, 1000),
        PtxFadePlacement(0, 1, 224000, "out", 1000, 1000),
    ]
    kick.alternate_playlists = [
        PtxPlaylist(9, "Kick.01", "pl9", 1, [0], placements=[PtxClipPlacement(0, 1, "C2", "kick.wav", 0, 48000, 48000)])
    ]

    evs = ptx.resolve_audio_events("Kick", include_muted=False, trim_overlaps=True)
    assert len(evs) == 3
    assert evs[0].clip_name == "C1" and evs[0].start_sample == 500 and evs[0].end_sample == 30000
    assert evs[0].fade_in is ptx.fades[0] and evs[0].audio_file is ptx.audio_files[0]
    assert evs[1].clip_name == "C2" and evs[1].length_samples == 96000
    assert evs[2].clip_name == "CMem" and evs[2].compound_group_name == "Grp1" and evs[2].fade_out is ptx.fades[1]

    # Alternate playlist resolution & unexpanded compound option
    alt_evs = ptx.resolve_audio_events("Kick", playlist="Kick.01")
    assert len(alt_evs) == 1 and alt_evs[0].playlist_name == "Kick.01"
    unexp = ptx.resolve_audio_events("Kick", expand_compounds=False, include_muted=False)
    assert any(e.clip_name == "Grp1" and e.compound_group_name == "Grp1" for e in unexp)


def test_layer3_resolve_midi_and_video_events():
    ptx = _make_session()
    ptx.midi_chunks = [
        PtxMidiChunk(
            index=0,
            notes=[PtxMidiNote(100, 60, 500, 100), PtxMidiNote(900, 64, 400, 90), PtxMidiNote(2000, 67, 100, 80)],
            control_lanes=[PtxMidiControlLane(1, "cc", 64, [PtxMidiControlEvent(500, 127), PtxMidiControlEvent(1500, 0)])],
        )
    ]
    ptx.midi_clips = [PtxMidiClip(index=0, name="Lead", start_tick=200, length_ticks=1000, chunk_index=0)]
    ptx.video_clips = [PtxVideoClip(index=0, name="Reel1", offset_frames=10, length_frames=50)]

    midi_trk = PtxTrack(
        90, "Synth", "aa" * 16, "{" + "aa" * 16 + "}", "Instrument", "Stereo",
        midi_clips=[PtxMidiPlacement(0, "Lead", start_tick=4 * TICKS_PER_QUARTER, length_ticks=1000)],
    )
    vid_trk = PtxTrack(
        91, "Movie", "bb" * 16, "{" + "bb" * 16 + "}", "Video", "None",
        video_clips=[PtxVideoPlacement(0, "Reel1", start_sample=48000)],
    )
    ptx.tracks.extend([midi_trk, vid_trk])

    mevs = ptx.resolve_midi_events("Synth", absolute_ticks=True)
    assert len(mevs) == 1
    assert len(mevs[0].notes) == 1
    assert mevs[0].notes[0].pitch == 64
    assert mevs[0].notes[0].tick == 4 * TICKS_PER_QUARTER + (900 - 200)
    assert mevs[0].notes[0].duration_ticks == 300  # Clipped to rend=1200
    assert mevs[0].control_lanes[0].events[0].tick == 4 * TICKS_PER_QUARTER + (500 - 200)

    mevs_rel = ptx.resolve_midi_events("Synth", absolute_ticks=False)
    assert mevs_rel[0].notes[0].tick == 700

    vevs = ptx.resolve_video_events("Movie")
    assert len(vevs) == 1
    assert vevs[0].start_seconds == pytest.approx(1.0)
    assert vevs[0].duration_seconds == pytest.approx(2.0)  # 50 frames @ 25 fps


def test_layer4_queries_folder_tree_cycle_safety_and_audit():
    ptx = _make_session()
    ptx.audio_files = [
        PtxAudioFile(0, 1, "used.wav", "Audio Files", 48000, 1, 24, 48000, 1.0, 0, "C:\\S\\Audio Files\\used.wav", "Audio Files/used.wav"),
        PtxAudioFile(1, 2, "unused.wav", "Ext", 48000, 1, 24, 48000, 1.0, 0, "D:\\Ext\\unused.wav", None),
    ]
    ptx.clip_gain_curves = [
        PtxClipGainCurve(0, 0, "Samples", [PtxClipGainBreakpoint(0, 0.0), PtxClipGainBreakpoint(1000, 3.0)]),
    ]
    ptx.clips = [
        PtxClip(
            index=0, name="C0", clip_id="c0", start_sample=0, length_samples=48000,
            clip_gain_curve_index=0, gain_db=0.0, audio_file_index=0,
            warp_markers=[PtxWarpMarker(0, 0), PtxWarpMarker(48000, 40000)], ara_plugin_name="Melodyne",
        )
    ]
    ptx.playlists = [PtxPlaylist(0, "Kick", "pl0", 1, [0], placements=[PtxClipPlacement(0, 0, "C0", "used.wav", 0, 48000, 48000, is_muted=True)])]
    kick = ptx.track_by_name("Kick")
    assert kick is not None
    kick.clips = list(ptx.playlists[0].placements)
    kick.volume_automation = PtxAutomationLane("volume", "Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(1000, -6.0)])
    kick.ara_plugin = PtxAraPlugin(0, "Melodyne")
    kick.inserts = [PtxInsert("A", 0, 0, "Trim", "com.avid.trim", "Digi/Trim/Trim", is_active=True, is_bypassed=False)]
    ptx.has_unparsed_tempo_ramps = True

    assert [t.name for t in ptx.iter_tracks(folder="KIK", recursive=True, track_types={"Audio"})] == ["Kick"]
    assert len(ptx.find_inserts(plugin_name="trim", manufacturer_4cc="Digi", active_only=True)) == 1
    assert [f.filename for f in ptx.used_audio_files()] == ["used.wav"]
    assert [f.filename for f in ptx.unused_audio_files()] == ["unused.wav"]
    assert [f.filename for f in ptx.external_audio_files()] == ["unused.wav"]

    tree = ptx.folder_tree()
    assert any(n.track.name == "KIK" and any(c.track.name == "Kick" for c in n.children) for n in tree)

    # Cycle in parent_folder_name still produces a complete cycle-safe tree
    a = PtxTrack(100, "CycA", "cc" * 16, "{" + "cc" * 16 + "}", "BasicFolder", "None", is_folder=True, parent_folder_name="CycB")
    b = PtxTrack(101, "CycB", "dd" * 16, "{" + "dd" * 16 + "}", "BasicFolder", "None", is_folder=True, parent_folder_name="CycA")
    ptx.tracks.extend([a, b])
    cyc_tree = ptx.folder_tree()
    names_in_tree = {n.track.name for n in cyc_tree} | {c.track.name for n in cyc_tree for c in n.children}
    assert {"CycA", "CycB"} <= names_in_tree

    audit = ptx.audit()
    assert "Kick" in audit.automated_volume_tracks
    assert "Kick" in audit.dynamic_clip_gain_tracks
    assert "Kick" in audit.warped_tracks
    assert "Kick" in audit.ara_tracks
    assert audit.muted_clip_count == 1
    assert audit.unused_audio_files == ("unused.wav",)
    assert audit.external_audio_files == ("unused.wav",)
    assert audit.has_tempo_changes is True
    assert audit.has_unparsed_tempo_ramps is True
    assert audit.has_meter_changes is True


def test_layer4_stock_plugin_parameter_decoders_and_automation_tapers():
    eq7 = PtxInsert("A", 0, 0, "EQ3 7-Band", "", "Digi/EQ7B/EQ7B", state=PtxPluginState(controls={
        "in 1": 1, "hprg": 1, "frq1": 80.0, "hpfs": 2,
        "in 2": 1, "lfrg": 1, "frq2": 120.0, "gai2": 2.5, "bnd2": 0.8,
        "in 4": 1, "frq4": 1000.0, "gai4": -3.0, "bnd4": 1.4,
        "in 7": 1, "lprg": 0, "frq7": 12000.0, "lpfq": 2.5,
        "inpt": -1.0, "outp": 0.5, "Phse": 1,
    }))
    d7 = decode_insert_parameters(eq7)
    assert d7 is not None
    assert d7.parameters["bands"][0]["type"] == "high_pass" and d7.parameters["bands"][0]["slope_db_per_oct"] == 18
    assert d7.parameters["bands"][1]["type"] == "low_shelf" and d7.parameters["bands"][1]["gain_db"] == 2.5
    assert d7.parameters["bands"][6]["type"] == "notch" and d7.parameters["bands"][6]["q"] == 2.5
    assert d7.parameters["phase_invert"] is True

    eq1 = PtxInsert("B", 1, 1, "EQ3 1-Band", "", "Digi/EQ1B/EQ1B", state=PtxPluginState(controls={
        "type": 0, "ctrf": 95.5, "slop": 3, "trim": -1.5,
    }))
    d1 = decode_insert_parameters(eq1)
    assert d1 is not None and d1.parameters["type"] == "high_pass" and d1.parameters["slope_db_per_oct"] == 24

    comp = PtxInsert("C", 2, 2, "Dyn3 Compressor/Limiter", "", "Digi/CpLm/CpLm", state=PtxPluginState(controls={
        "sldC": -18.0, "sld2": 4.0, "sld1": 0.01, "sld3": 0.12, "sldG": 3.0, "sldD": 2.5, "LoFE": 1, "LoFF": 100.0,
    }))
    dc = decode_insert_parameters(comp)
    assert dc is not None and dc.parameters["attack_ms"] == 10.0 and dc.parameters["key_hpf_hz"] == 100.0

    deess = PtxInsert("D", 3, 3, "Dyn3 De-Esser", "", "Digi/DEss/DEss", state=PtxPluginState(controls={
        "DFrq": 6500.0, "DRng": -8.0, "HFOn": 1,
    }))
    dd = decode_insert_parameters(deess)
    assert dd is not None and dd.parameters["freq_hz"] == 6500.0 and dd.parameters["hf_only"] is True

    gate = PtxInsert("E", 4, 4, "Dyn3 Expander/Gate", "", "Digi/ExpG/ExpG", state=PtxPluginState(controls={
        "sldC": -35.0, "sld2": 5.0, "sldG": -40.0, "sld1": 0.002, "sldD": 0.05, "sld3": 0.2, "LkAh": 1,
    }))
    dg = decode_insert_parameters(gate)
    assert dg is not None and dg.parameters["hold_ms"] == 50.0 and dg.parameters["lookahead"] is True

    assert decode_insert_parameters(PtxInsert("F", 5, 5, "UnknownPlugin", "", "")) is None
    assert decode_plugin_automation_value("EQ3 7-Band", "gai2", 0.0) == pytest.approx(0.0)
    assert decode_plugin_automation_value("EQ3 7-Band", "gai2", 0.5) == pytest.approx(18.0)
    assert decode_plugin_automation_value("EQ3 7-Band", "byp", 1.0, parameter_name="Master Bypass") == 1.0
    assert decode_plugin_automation_value("Unknown", "foo", 0.0) is None


def test_layer5_patch_builder_and_sessionprep_wiring(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    raw = _build_synthetic_ptx()

    # In-memory build + commit guard
    pb_mem = PtxPatchBuilder.from_bytes(raw)
    assert pb_mem.has_pending_changes is False
    pb_mem.clear_track_inputs(["Kick"]).set_clip_gains({"Kick": 4.0}).set_track_volumes({"Kick": -4.0}).add_group(
        "KIK", ["Kick"], vca_track="KIK-VCA",
    )
    assert pb_mem.has_pending_changes is True
    patched, summary = pb_mem.build()
    assert summary["volumes"]["Kick"]["target_db"] == -4.0
    re_mem = PtxSession.from_bytes(patched)
    assert re_mem.track_by_name("Kick").input_path is None
    assert re_mem.clips[0].gain_db == pytest.approx(4.0, abs=0.02)
    assert re_mem.group_by_name("KIK").vca_track_name == "KIK-VCA"
    with pytest.raises(ValueError):
        pb_mem.commit()

    # File-backed transactional commit + no-op commit
    ptx_file = tmp_path / "BuilderSong.ptx"
    ptx_file.write_bytes(raw)
    orig_noop, noop_sum = PtxPatchBuilder.from_file(ptx_file).commit()
    assert orig_noop == raw and noop_sum["volumes"] == {}

    with PtxPatchBuilder.from_file(ptx_file) as pb_file:
        pb_file.set_track_volumes({"Kick": -5.5})
        orig_bytes, file_sum = pb_file.commit()
    assert orig_bytes == raw
    assert file_sum["volumes"]["Kick"]["target_db"] == -5.5
    assert PtxSession.from_file(ptx_file).track_by_name("Kick").volume_db == -5.5

    # SessionPrep wiring: inspect_protools_template_topology populates .audit & log_ptx_audit_caveats logs warnings
    topo = inspect_protools_template_topology(ptx_file)
    assert topo.audit is not None
    assert topo.audit.has_unparsed_tempo_ramps is False

    ptx_warn = _make_session()
    ptx_warn.has_unparsed_tempo_ramps = True
    with caplog.at_level(logging.WARNING):
        protools_ptx.log_ptx_audit_caveats(ptx_warn.audit(), "TestTemplate.ptxt")
    assert "unparsed tempo ramps" in caplog.text


def test_edge_stepped_breakpoints_duplicate_timestamps_and_unsorted_initial():
    ptx = _make_session()
    mute_lane = PtxAutomationLane(
        "mute", "Samples",
        [PtxAutomationPoint(96000, 0.0), PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, 1.0)],
    )
    assert mute_lane.initial_value == 0.0
    assert mute_lane.value_at(48000, stepped=True) == 1.0
    assert mute_lane.value_at(96000, stepped=True) == 0.0

    # Duplicate timestamps (vertical jump) are right-continuous at the jump point
    step_lane = PtxAutomationLane(
        "volume", "Samples",
        [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, 0.0), PtxAutomationPoint(48000, -12.0), PtxAutomationPoint(96000, -12.0)],
    )
    assert step_lane.value_at(47999) == pytest.approx(0.0, abs=1e-4)
    assert step_lane.value_at(48000) == pytest.approx(-12.0)

    cg = PtxClipGainCurve(
        0, 0, "Samples",
        [PtxClipGainBreakpoint(48000, 6.0), PtxClipGainBreakpoint(0, -2.0), PtxClipGainBreakpoint(48000, -4.0), PtxClipGainBreakpoint(96000, -4.0)],
    )
    assert cg.initial_gain_db == -2.0
    assert cg.gain_db_at(48000) == pytest.approx(-4.0)

    kick = ptx.track_by_name("Kick")
    assert kick is not None
    kick.mute_automation = mute_lane
    assert ptx.is_track_audible("Kick", 0.5) is True
    assert ptx.is_track_audible("Kick", 1.0) is False
    assert ptx.is_track_audible("Kick", 2.0) is True


def test_edge_timeline_drop_frame_unsorted_meters_late_tempo_and_duration():
    ptx = _make_session()
    ptx.info.fps_code = 5  # 29.97 drop-frame
    ptx.info.frame_rate = 30000.0 / 1001.0
    ptx.info.start_frames = 0
    assert ";" in ptx.timeline.seconds_to_timecode(60.0, include_session_start=False)
    assert ptx.timeline.seconds_to_timecode(-5.0, include_session_start=False) == ""
    assert ptx.timeline.seconds_to_timecode(float("nan")) == ""
    with pytest.raises(ValueError):
        ptx.timeline.seconds_to_samples(float("inf"))

    # Unsorted meter events and first tempo event at tick > 0 remain monotonic & consistent
    ptx.meter_events = [
        PtxMeterEvent(tick=8 * TICKS_PER_QUARTER, numerator=3, denominator=4),
        PtxMeterEvent(tick=4 * TICKS_PER_QUARTER, numerator=5, denominator=4),
    ]
    assert ptx.timeline.meter_at_tick(0) == (4, 4)
    assert ptx.timeline.meter_at_tick(4 * TICKS_PER_QUARTER) == (5, 4)
    assert ptx.timeline.meter_at_tick(8 * TICKS_PER_QUARTER) == (3, 4)

    ptx.tempo_events = [PtxTempoEvent(tick=4 * TICKS_PER_QUARTER, bpm=120.0, seconds=0.0)]
    assert ptx.timeline.ticks_to_seconds(4 * TICKS_PER_QUARTER) == pytest.approx(2.0)
    assert ptx.timeline.seconds_to_ticks(1.0) == 2 * TICKS_PER_QUARTER

    # Duration includes looped audio clips and video clip length
    ptx.playlists = [PtxPlaylist(0, "Kick", "pl0", 1, [0], placements=[PtxClipPlacement(0, 0, "C", "a.wav", 0, 48000, 48000, loop_count=3)])]
    ptx.video_clips = [PtxVideoClip(index=0, name="V0", offset_frames=0, length_frames=300)]
    ptx.video_playlists = {"Movie": [PtxVideoPlacement(0, "V0", start_sample=48000 * 5)]}
    assert ptx.timeline.session_duration_seconds() >= 15.0

    ptx.info.sample_rate = 0
    assert ptx.timeline.samples_to_seconds(48000) == 0.0
    assert ptx.timeline.seconds_to_samples(1.0) == 0


def test_edge_arrangement_overlaps_crossfade_sync_clamp_loops_and_playlist_guard():
    ptx = _make_session()
    ptx.audio_files = [PtxAudioFile(10, 1, "a.wav", "Audio Files", 48000, 1, 24, 200000, 4.0, 0)]
    ptx.clips = [
        PtxClip(index=0, name="Base", clip_id="c0", start_sample=0, length_samples=100000, audio_file_index=10, timeline_length_samples=100000),
        PtxClip(index=1, name="Mid", clip_id="c1", start_sample=0, length_samples=20000, audio_file_index=10, timeline_length_samples=20000),
        PtxClip(index=2, name="Top", clip_id="c2", start_sample=0, length_samples=20000, audio_file_index=10, timeline_length_samples=20000),
        PtxClip(index=3, name="SyncClip", clip_id="c3", start_sample=100, length_samples=10000, audio_file_index=10, timeline_length_samples=10000, sync_point_samples=500),
        PtxClip(index=4, name="XfL", clip_id="c4", start_sample=0, length_samples=10500, audio_file_index=10, timeline_length_samples=10500),
        PtxClip(index=5, name="XfR", clip_id="c5", start_sample=0, length_samples=10500, audio_file_index=10, timeline_length_samples=10500),
    ]
    ptx.fades = [PtxFade(0, "crossfade", pre_samples=500, total_samples=1000)]
    ptx.compound_clips = [PtxCompoundClip(0, "LoopGrp", 0, 10000, [PtxCompoundClipMember(1, 0, 0)])]
    kick = ptx.track_by_name("Kick")
    assert kick is not None
    # 3-way overlap (Base split by Mid, then Top overlaps Mid + Base's tail)
    kick.clips = [
        PtxClipPlacement(0, 0, "Base", "a.wav", 0, 100000, 100000),
        PtxClipPlacement(0, 1, "Mid", "a.wav", 20000, 20000, 40000),
        PtxClipPlacement(0, 2, "Top", "a.wav", 30000, 20000, 50000),
        PtxClipPlacement(1, 3, "SyncClip", "a.wav", 200, 10000, 10200),  # start < sync -> clamps & trims 300
        PtxClipPlacement(2, 4, "XfL", "a.wav", 0, 10500, 10500),
        PtxClipPlacement(2, 5, "XfR", "a.wav", 9500, 10500, 20000),      # overlaps by 1000 across crossfade
        PtxClipPlacement(3, 0, "LoopGrp", "", 0, 10000, 10000, is_compound=True, loop_count=2),
    ]
    kick.fade_placements = [PtxFadePlacement(2, 0, 10000, "crossfade", 500, 1000)]
    ptx.playlists = [PtxPlaylist(1, "OtherPl", "p1", 1, [0], placements=[PtxClipPlacement(0, 0, "Base", "a.wav", 0, 1000, 1000)], track_name="Snare")]

    evs = ptx.resolve_audio_events("Kick", trim_overlaps=True)
    ch0 = [(e.clip_name, e.start_sample, e.end_sample) for e in evs if e.channel == 0]
    assert ch0 == [("Base", 0, 20000), ("Mid", 20000, 30000), ("Top", 30000, 50000), ("Base", 50000, 100000)]
    ch1 = [e for e in evs if e.channel == 1][0]
    assert ch1.start_sample == 0 and ch1.length_samples == 9700 and ch1.source_offset_samples == 400
    ch2 = [e for e in evs if e.channel == 2]
    assert ch2[0].fade_out is ptx.fades[0] and ch2[1].fade_in is ptx.fades[0] and ch2[0].end_sample == 10500
    ch3 = [e for e in evs if e.channel == 3]
    assert [(e.start_sample, e.compound_group_name) for e in ch3] == [(0, "LoopGrp"), (10000, "LoopGrp")]
    # Cross-track playlist isolation
    assert ptx.resolve_audio_events("Kick", playlist="OtherPl") == []
    assert len(ptx.resolve_audio_events("Kick", playlist="Kick")) > 0


def test_edge_vca_cycles_folder_recursion_compound_audit_and_send_mute():
    ptx = _make_session()
    v1 = PtxTrack(50, "VCA1", "11" * 16, "{" + "11" * 16 + "}", "Vca", "None", volume_db=-3.0, groups=["G2"])
    v2 = PtxTrack(51, "VCA2", "22" * 16, "{" + "22" * 16 + "}", "Vca", "None", volume_db=-2.0, groups=["G1"])
    sub = PtxTrack(52, "OldFolder", "33" * 16, "{" + "33" * 16 + "}", "BasicFolder", "None", is_folder=False, parent_folder_name="KIK")
    leaf = PtxTrack(53, "DeepAudio", "44" * 16, "{" + "44" * 16 + "}", "Audio", "Mono", parent_folder_name="OldFolder", groups=["G1"], volume_db=0.0)
    leaf.sends = [
        PtxSend("A", 0, PtxSignalPathRef("FX", "bus", "00"), volume_db=-6.0, is_muted=True),
        PtxSend("B", 1, PtxSignalPathRef("FX", "bus", "00"), volume_db=-6.0, is_active=False),
    ]
    ptx.tracks.extend([v1, v2, sub, leaf])
    ptx.groups.extend([
        PtxGroup(10, "G1", "mix", "Mix/Edit", 1, "a", vca_track_name="VCA1", member_names=["DeepAudio", "VCA2"]),
        PtxGroup(11, "G2", "mix", "Mix/Edit", 2, "b", vca_track_name="VCA2", member_names=["VCA1"]),
    ])
    assert [v.name for v in ptx.vcas_for_track("DeepAudio")] == ["VCA1", "VCA2"]
    assert [v.name for v in ptx.vcas_for_track("VCA1")] == ["VCA2"]  # Never self-counts VCA1!
    assert ptx.effective_track_volume_db("VCA1") == pytest.approx(-5.0)
    assert ptx.effective_send_volume_db("DeepAudio", "A") == -144.0
    assert ptx.effective_send_volume_db("DeepAudio", "B") == -144.0
    assert "DeepAudio" in [t.name for t in ptx.iter_tracks(folder="KIK", recursive=True)]

    # Compound clip audit + dangling compound index must not mark unrelated clip as used
    ptx.audio_files = [PtxAudioFile(0, 1, "f0.wav", "Audio Files", 48000, 1, 24, 48000, 1.0, 0)]
    ptx.clips = [PtxClip(index=0, name="C0", clip_id="c0", audio_file_index=0, clip_gain_chunk_size=64, ara_plugin_name="Melodyne")]
    ptx.compound_clips = [PtxCompoundClip(0, "G", 0, 48000, [PtxCompoundClipMember(0, 0, 0)])]
    ptx.playlists = [PtxPlaylist(0, "DeepAudio", "p0", 1, [0], placements=[PtxClipPlacement(0, 999, "Missing", "", 0, 1000, 1000, is_compound=True)])]
    assert ptx.used_audio_files() == []
    leaf.clips = [PtxClipPlacement(0, 0, "G", "", 0, 48000, 48000, is_compound=True)]
    aud = ptx.audit()
    assert "DeepAudio" in aud.dynamic_clip_gain_tracks and "DeepAudio" in aud.ara_tracks


def test_edge_plugin_decoders_non_finite_clamping_4cc_and_patch_builder_safety(tmp_path: Path):
    # Non-finite values in plugin state & automation tapers + 4CC fallback
    ins_4cc = PtxInsert("A", 0, 0, "Custom Label", "", "Digi/EQ1B/EQ1B", state=PtxPluginState(controls={
        "type": 3, "ctrf": 1000.0, "gain": float("nan"), "bndw": float("inf"), "trim": 1.5,
    }))
    dec = decode_insert_parameters(ins_4cc)
    assert dec is not None and dec.plugin_name == "EQ3 1-Band" and dec.parameters["gain_db"] == 0.0 and dec.parameters["q"] == 1.0

    assert decode_plugin_automation_value("EQ3 7-Band", "byp", float("nan"), parameter_name="Master Bypass") is None
    assert decode_plugin_automation_value("EQ3 7-Band", "frq2", float("inf")) is None
    assert decode_plugin_automation_value("EQ3 7-Band", "frq2", 50.0) == pytest.approx(500.0)
    assert decode_plugin_automation_value("Dyn3 Expander/Gate", "sld2", 0.5) == pytest.approx(100.0)

    # PatchBuilder safety: ambiguous init, rollback on mid-batch failure, double-commit no-op, summary preservation
    raw = _build_synthetic_ptx()
    ptx_path = tmp_path / "Safety.ptx"
    ptx_path.write_bytes(raw)
    with pytest.raises(ValueError):
        PtxPatchBuilder(raw_bytes=raw, file_path=ptx_path)

    with pytest.raises(KeyError):
        PtxPatchBuilder.from_file(ptx_path).set_track_volumes({"Kick": -3.0}).delete_group("NoSuchGroup").commit()
    assert ptx_path.read_bytes() == raw  # Disk file untouched after mid-batch failure

    pb = PtxPatchBuilder.from_file(ptx_path).set_track_volumes({"Kick": -2.0}).set_track_volumes({"Kick": -6.0}).add_group("KIK", ["Kick"])
    orig, sum1 = pb.commit()
    assert orig == raw and sum1["volumes"]["Kick"]["previous_db"] == 0.0 and sum1["volumes"]["Kick"]["target_db"] == -6.0
    assert pb.has_pending_changes is False
    _, sum2 = pb.commit()  # Second commit is a safe no-op (does not replay add_group!)
    assert sum2["groups"] == [] and sum2["volumes"] == {}

    # Modify + delete group roundtrip via builder
    pb2 = PtxPatchBuilder.from_file(ptx_path).modify_group("KIK", new_name="DRUMS").delete_group("DRUMS")
    pb2.commit()
    assert PtxSession.from_file(ptx_path).group_by_name("DRUMS") is None
