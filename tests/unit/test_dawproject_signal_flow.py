"""Unit tests for DAWproject VST3 Slot 1 Trim codecs/cloning and Signal Flow DAG preview/patching."""

from __future__ import annotations

import json
import struct
from xml.etree import ElementTree as ET
import zipfile
import zlib

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.daw_processors.dawproject import (
    DawProjectDawProcessor,
    _apply_signal_flow_and_trims,
)
from sessionpreplib.daw_processors.dawproject_signal_flow import (
    dawproject_config_params,
    dawproject_xml_to_mixer_session,
    inspect_dawproject_template_topology,
    patch_dawproject_channel_volumes_xml,
    preview_dawproject_signal_flow,
    prune_orphaned_submix_channels_xml,
    resolve_dawproject_track_staging,
)
from sessionpreplib.daw_processors.dawproject_vst3 import (
    VstPresetChunk,
    VstPresetData,
    build_vstpreset,
    extract_vstpreset_chunk,
    has_calibrated_vst3_trim_donor,
    insert_vst3_trim_slot1_xml,
    inspect_vst3_slot1_trim,
    parse_vstpreset,
    patch_vst3_trim_preset,
    select_vst3_trim_donor,
)
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionpreplib.signal_flow import (
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
)


def _sine_stereo(db_peak: float, sr: int = 48000, dur_sec: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * dur_sec), dtype=np.float64) / float(sr)
    sig = (10.0 ** (db_peak / 20.0)) * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack((sig, sig))


def _make_trackcontrol_vstpreset(gain_db: float = 0.0, *, clean: bool = True) -> bytes:
    params: dict[str, float] | str = "default" if (abs(gain_db) < 1e-6 and clean) else {"kOpGain": (gain_db + 200.0) / 218.0}
    if not clean and isinstance(params, dict):
        params["kHPF"] = 0.5
    payload = json.dumps({"settings": {"currentAB": 0}, "ab": [{"parameters": params}]}).encode()
    comp = b"DMGAHDR\x00DGCT\x00\x00\x00\x00ZIP " + zlib.compress(payload)
    preset = VstPresetData(
        version=1,
        class_id="565354444D4741547243740000000001",
        chunks=(VstPresetChunk(b"Comp", comp), VstPresetChunk(b"Info", b"<MetaInfo/>")),
    )
    return build_vstpreset(preset)


def _make_protility_vstpreset(gain_db: float = 0.0) -> bytes:
    xml = (
        f'<PROTILITY activeSlot="A"><slotA PRESET_MODIFIED="0">'
        f'<PARAM id="GAIN" value="{gain_db:.1f}"/><PARAM id="PAN" value="0.0"/>'
        f"</slotA></PROTILITY>\x00"
    ).encode()
    comp = b"JUCEHDR\x00VC2!" + struct.pack("<I", len(xml)) + xml
    preset = VstPresetData(
        version=1,
        class_id="56535450724175507254790000000001",
        chunks=(VstPresetChunk(b"Comp", comp),),
    )
    return build_vstpreset(preset)


CUBASE_MIX_TEMPLATE_XML = """<Project version="1.0">
  <Application name="Cubase" version="14.0.20"/>
  <Structure>
    <Track id="f_vox" name="Vocals" contentType="tracks" loaded="true">
      <Channel id="ch_vox_f" role="master" audioChannels="2" name="Vocals" destination="ch_vox_bus">
        <Volume id="v_vf" value="1.0" unit="linear" min="0.0" max="2.0"/>
      </Channel>
    </Track>
    <Track id="f_inst" name="Instruments" contentType="tracks" loaded="true">
      <Channel id="ch_inst_f" role="master" audioChannels="2" name="Instruments" destination="ch_mix"/>
    </Track>
    <Track id="f_empty" name="Unused" contentType="tracks" loaded="true">
      <Channel id="ch_empty_f" role="master" audioChannels="2" name="Unused" destination="ch_orphan_bus"/>
    </Track>
    <Channel id="ch_vox_bus" role="submix" audioChannels="2" name="VOX BUS" destination="ch_mix">
      <Devices>
        <Vst3Plugin id="d_tc_vox" name="TrackControl" deviceID="56535444-4D47-4154-7243-740000000001"
                    deviceName="TrackControl" deviceVendor="DMG Audio" deviceRole="audioFX" loaded="true" pluginVersion="1.2">
          <Enabled id="d_tc_vox_en" value="true"/>
          <State path="plugins/tc_vox.vstpreset"/>
        </Vst3Plugin>
        <Compressor id="d_comp" deviceName="Comp" deviceRole="audioFX" loaded="true"/>
      </Devices>
      <Volume id="v_vb" value="1.0" unit="linear" min="0.0" max="2.0"/>
    </Channel>
    <Channel id="ch_orphan_bus" role="submix" audioChannels="2" name="ORPHAN BUS" destination="ch_mix"/>
    <Channel id="ch_verb" role="effect" audioChannels="2" name="Reverb" destination="ch_mix"/>
    <Channel id="ch_mix" role="submix" audioChannels="2" name="MIX BUS" destination="ch_master">
      <Devices>
        <Vst3Plugin id="d_tc_mix" name="TrackControl" deviceID="56535444-4D47-4154-7243-740000000001"
                    deviceName="TrackControl" deviceVendor="DMG Audio" deviceRole="audioFX" loaded="true">
          <Enabled id="d_tc_mix_en" value="true"/>
          <State path="plugins/tc_mix.vstpreset"/>
        </Vst3Plugin>
      </Devices>
      <Volume id="v_mix" value="1.0" unit="linear" min="0.0" max="2.0"/>
    </Channel>
    <Channel id="ch_hw" role="master" audioChannels="2" name="AIO 5/6 In"/>
    <Channel id="ch_master" role="master" audioChannels="2" name="Stereo Out"/>
  </Structure>
</Project>"""


def test_vstpreset_binary_roundtrip_patching_and_validation():
    """Steinberg .vstpreset parser/writer round-trips TrackControl & Protility and fails safely on uncalibrated blobs."""
    tc_raw = _make_trackcontrol_vstpreset(0.0)
    parsed = parse_vstpreset(tc_raw)
    assert parsed.version == 1
    assert parsed.class_id == "565354444D4741547243740000000001"
    assert extract_vstpreset_chunk(parsed, b"Comp") is not None
    assert extract_vstpreset_chunk(parsed, b"Info") == b"<MetaInfo/>"

    tc_patched = patch_vst3_trim_preset(tc_raw, "trackcontrol", -6.5)
    assert tc_patched is not None
    tc_parsed2 = parse_vstpreset(tc_patched)
    assert extract_vstpreset_chunk(tc_parsed2, b"Info") == b"<MetaInfo/>"
    comp2 = extract_vstpreset_chunk(tc_parsed2, b"Comp") or b""
    assert b"DGCT" in comp2 and b"ZIP " in comp2

    pr_raw = _make_protility_vstpreset(0.0)
    pr_patched = patch_vst3_trim_preset(pr_raw, "protility", -4.5)
    assert pr_patched is not None
    pr_comp = extract_vstpreset_chunk(parse_vstpreset(pr_patched), b"Comp") or b""
    assert b'value="-4.5"' in pr_comp

    # Uncalibrated binary Comp chunk returns None without corrupting or crashing
    opaque = build_vstpreset(VstPresetData(1, "00112233445566778899AABBCCDDEEFF", (VstPresetChunk(b"Comp", b"OPAQUE"),)))
    assert patch_vst3_trim_preset(opaque, "trackcontrol", -3.0) is None

    with pytest.raises(ValueError, match="VST3"):
        parse_vstpreset(b"NOPE" + b"\x00" * 60)
    with pytest.raises(ValueError, match="Non-finite"):
        patch_vst3_trim_preset(tc_raw, "trackcontrol", float("nan"))


def test_vst3_only_matching_donor_selection_and_xsd_slot1_insertion():
    """VST2 plugins are ignored; VST3 TrackControl donor clones into Slot 1 at <Channel> index 0 (unity by default)."""
    xml = """<Project version="1.0"><Structure>
      <Channel id="c_v2" role="submix" audioChannels="2" name="LegacyBus" destination="c_m">
        <Devices>
          <Vst2Plugin id="d_v2" deviceID="1234" deviceName="TrackControl" deviceVendor="DMG Audio" deviceRole="audioFX"/>
        </Devices>
      </Channel>
      <Channel id="c_v3" role="submix" audioChannels="2" name="Vst3Bus" destination="c_m">
        <Devices>
          <Vst3Plugin id="d_v3" deviceID="56535444-4D47-4154-7243-740000000001"
                      deviceName="TrackControl" deviceVendor="DMG Audio" deviceRole="audioFX" loaded="true" pluginVersion="1.2">
            <Enabled id="en_1" value="true"/>
            <State path="plugins/tc.vstpreset"/>
          </Vst3Plugin>
        </Devices>
      </Channel>
      <Channel id="c_m" role="master" audioChannels="2" name="Master"/>
    </Structure></Project>"""
    tc_blob = _make_trackcontrol_vstpreset(0.0)
    root = ET.fromstring(xml)
    mmap = dawproject_xml_to_mixer_session(
        root, lambda p: tc_blob if p == "plugins/tc.vstpreset" else None, apply_trim_gain=True,
    )
    # Only the VST3 TrackControl is recognized as a donor; VST2 is ignored
    assert len(mmap.donors) == 1
    assert mmap.donors[0].source_channel_id == "c_v3"
    # Stereo VST3 donor cleanly serves Mono track via channel-agnostic fallback
    donor = select_vst3_trim_donor(mmap.donors, "Mono")
    assert donor is not None and donor.plugin_id == "trackcontrol"

    # Insert into a freshly created Audio track Channel (with existing Mute/Pan/Volume)
    ch_el = ET.fromstring('<Channel id="c_leaf" role="regular"><Mute value="false"/><Volume value="1.0" unit="linear"/></Channel>')
    ids = iter(["id10", "id11"])
    blobs: dict[str, bytes] = {}
    used: set[str] = {"plugins/tc.vstpreset"}
    meta = insert_vst3_trim_slot1_xml(
        ch_el, donor,
        make_element=lambda tag, attrib=None: ET.Element(tag, attrib or {}),
        next_xml_id=lambda: next(ids),
        used_members=used,
        state_counter=[0],
        extra_archive_blobs=blobs,
        apply_gain_db=None,  # Phase 1: unity insertion without gain adjustment
    )
    assert meta["plugin"] == "TrackControl"
    assert meta["gain_applied"] is False
    assert meta["target_db"] == 0.0
    # XSD sequence order: <Devices> is inserted at index 0 before <Mute> and <Volume>
    assert [child.tag for child in list(ch_el)] == ["Devices", "Mute", "Volume"]
    vst3_el = ch_el.find("Devices/Vst3Plugin")
    assert vst3_el is not None
    assert vst3_el.get("deviceID") == donor.device_id
    state_path = vst3_el.find("State").get("path")
    assert state_path in blobs
    assert blobs[state_path] == tc_blob  # pristine unity preset copied verbatim


def test_dawproject_topology_preview_unity_vs_calibrated_and_volume_patching(tmp_path):
    """Offline topology & signal-flow preview handles Cubase multi-stage DAG in both unity and calibrated modes."""
    tpl = tmp_path / "CubaseTemplate.dawproject"
    tc_blob = _make_trackcontrol_vstpreset(0.0)
    with zipfile.ZipFile(tpl, "w") as zf:
        zf.writestr("project.xml", CUBASE_MIX_TEMPLATE_XML)
        zf.writestr("plugins/tc_vox.vstpreset", tc_blob)
        zf.writestr("plugins/tc_mix.vstpreset", tc_blob)

    topo = inspect_dawproject_template_topology(tpl)
    assert topo.donor_plugin_name == "TrackControl"
    vox_meta = topo.lookup("f_vox", "Vocals")
    assert vox_meta is not None
    assert vox_meta.folder_type == "routing"
    assert "VOX BUS" in vox_meta.output_chain and "MIX BUS" in vox_meta.output_chain and "Stereo Out" in vox_meta.output_chain

    # Build 3 audio tracks: two loud vocals into Vocals -> VOX BUS -> MIX BUS, one pad into Instruments -> MIX BUS
    sig_loud = _sine_stereo(-9.0)  # RMS ~ -12 dBFS; 2 coherent tracks -> -6 dBFS RMS on VOX BUS
    sig_pad = _sine_stereo(-24.0, freq=220.0)
    wav1, wav2, wav3 = tmp_path / "Vox1.wav", tmp_path / "Vox2.wav", tmp_path / "Pad.wav"
    sf.write(wav1, sig_loud, 48000, subtype="PCM_24")
    sf.write(wav2, sig_loud, 48000, subtype="PCM_24")
    sf.write(wav3, sig_pad, 48000, subtype="PCM_24")

    def _mk_track(p, gain_db):
        res = ProcessorResult(
            processor_id="bimodal_normalize", classification="Sustained", method="RMS",
            gain_db=gain_db, data={"fader_offset": -gain_db},
        )
        return TrackContext(
            filename=p.name, filepath=str(p), audio_data=None, samplerate=48000, channels=2,
            total_samples=24000, bitdepth="24-bit", subtype="PCM_24", duration_sec=0.5,
            processor_results={"bimodal_normalize": res},
        )

    t1, t2, t3 = _mk_track(wav1, -6.0), _mk_track(wav2, -6.0), _mk_track(wav3, 9.0)
    entries = [TransferEntry("1", "Vox1.wav", "Vox 1"), TransferEntry("2", "Vox2.wav", "Vox 2"), TransferEntry("3", "Pad.wav", "Pad")]
    folders = [{"id": "f_vox", "name": "Vocals"}, {"id": "f_inst", "name": "Instruments"}]
    assignments = {"1": "f_vox", "2": "f_vox", "3": "f_inst"}
    session = SessionContext(
        [t1, t2, t3], {"_use_processed": False, "target_peak": -6.0, "target_rms": -18.0},
        output_tracks=[t1, t2, t3], transfer_manifest=entries,
    )

    # Phase 1 (apply_trim_gain=False): leaves stay 0.0 dB Trim & 0.0 dB Fader (no balance inversion); buses stage fader_only
    g_unity = preview_dawproject_signal_flow(
        tpl, session, assignments, folders,
        gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=False, fader_ceiling_db=6.0,
    )
    assert g_unity.nodes["Vox 1"].leaf_trim_db == 0.0
    assert g_unity.nodes["Vox 1"].recommended_fader_db == 0.0
    assert g_unity.nodes["VOX BUS"].staging_strategy in ("fader_only", "none")

    # Calibrated mode (apply_trim_gain=True): leaves get Trim + inverse Fader; VOX BUS uses trim_and_fader capped at +6 dB
    g_cal = preview_dawproject_signal_flow(
        tpl, session, assignments, folders,
        gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=True, fader_ceiling_db=6.0,
    )
    assert g_cal.nodes["Pad"].leaf_trim_db == pytest.approx(9.0)
    assert g_cal.nodes["VOX BUS"].staging_strategy == "trim_and_fader"
    assert g_cal.nodes["VOX BUS"].recommended_fader_db <= 6.0
    assert g_cal.nodes["MIX BUS"].is_terminal is True
    assert g_cal.nodes["MIX BUS"].staging_strategy == "trim_only"
    assert g_cal.balance_validation["converged"] is True

    # XML volume patching & orphaned submix pruning
    root = ET.fromstring(CUBASE_MIX_TEMPLATE_XML)
    # Remove f_empty and prune orphaned ch_orphan_bus while protecting ch_verb (effect) & ch_mix (still fed)
    struct_el = root.find("Structure")
    struct_el.remove(root.find(".//Track[@id='f_empty']"))
    removed_buses = prune_orphaned_submix_channels_xml(root, {"ch_orphan_bus"})
    assert removed_buses == ["ORPHAN BUS"]
    assert root.find(".//Channel[@id='ch_orphan_bus']") is None
    assert root.find(".//Channel[@id='ch_verb']") is not None
    assert root.find(".//Channel[@id='ch_mix']") is not None

    patched = patch_dawproject_channel_volumes_xml(
        root, {"ch_vox_bus": 6.0, "ch_mix": -3.0},
        make_element=lambda tag, attrib=None: ET.Element(tag, attrib or {}),
        next_xml_id=lambda: "id999",
    )
    assert patched == {"ch_vox_bus": 6.0, "ch_mix": -3.0}
    vb_vol = root.find(".//Channel[@id='ch_vox_bus']/Volume")
    assert float(vb_vol.get("value")) == pytest.approx(10 ** (6.0 / 20.0), rel=1e-4)
    assert float(vb_vol.get("max")) >= float(vb_vol.get("value"))


def test_resolve_dawproject_track_staging_and_config_params(tmp_path):
    """resolve_dawproject_track_staging guards unity-insert balance and DAWproject config_params exposes all 6 specs."""
    orig = tmp_path / "kick.wav"
    proc = tmp_path / "kick_prep.wav"
    orig.write_bytes(b"raw")
    proc.write_bytes(b"prep")
    res = ProcessorResult(
        processor_id="bimodal_normalize", classification="Transient", method="Peak",
        gain_db=22.0, data={"fader_offset": -22.0},
    )
    trk = TrackContext(
        filename="kick.wav", filepath=str(orig), audio_data=None, samplerate=48000, channels=2,
        total_samples=48000, bitdepth="24-bit", subtype="PCM_24", duration_sec=1.0,
        processed_filepath=str(proc), prepared_fader_db=-10.0,
        processor_results={"bimodal_normalize": res},
    )
    raw_sess = SessionContext([trk], {"_use_processed": False})
    prep_sess = SessionContext([trk], {"_use_processed": True}, prepare_state="ready")

    # Unity insert (apply_trim_gain=False) on raw audio keeps Trim & Fader at 0.0 dB
    p_unity = resolve_dawproject_track_staging(raw_sess, trk, gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=False)
    assert (p_unity.trim_gain_db, p_unity.fader_db) == (0.0, 0.0)

    # Calibrated (apply_trim_gain=True) clamps +22 dB to TrackControl +18 dB and symmetrically sets fader to -18 dB
    p_cal = resolve_dawproject_track_staging(raw_sess, trk, gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=True)
    assert (p_cal.trim_gain_db, p_cal.fader_db) == (18.0, -18.0)

    # Prepared WAV carries prepared_fader_db even when apply_trim_gain=False
    p_prep = resolve_dawproject_track_staging(prep_sess, trk, gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=False)
    assert (p_prep.trim_gain_db, p_prep.fader_db, p_prep.baked_gain_db) == (0.0, -10.0, 22.0)

    # Trim unity, fader_only, none modes
    assert resolve_dawproject_track_staging(raw_sess, trk, gain_mode=GAIN_MODE_TRIM_UNITY, apply_trim_gain=True).fader_db == 0.0
    assert resolve_dawproject_track_staging(raw_sess, trk, gain_mode=GAIN_MODE_FADER_ONLY, apply_trim_gain=True).fader_db == 0.0
    assert resolve_dawproject_track_staging(prep_sess, trk, gain_mode=GAIN_MODE_NONE, apply_trim_gain=True).fader_db == 0.0

    # Config params exposed on DawProjectDawProcessor
    keys = [p.key for p in DawProjectDawProcessor.config_params()]
    for expected in (p.key for p in dawproject_config_params()):
        assert expected in keys


def test_vst3_hardening_comments_trailing_bytes_cumulative_cap_and_device_name():
    """Verify VST3 parser/patcher hardening: cumulative chunk cap, trailing zlib bytes, uncalibrated donors, comments, deviceName."""
    # Cumulative chunk size bomb rejected
    big_chunk = b"\x00" * (2 * 1024 * 1024)
    header = b"VST3" + struct.pack("<i", 1) + (b"A" * 32) + struct.pack("<q", 48 + len(big_chunk))
    # 3 overlapping entries pointing to the same 2 MiB slice -> 6 MiB cumulative > 4 MiB cap
    entries = b"".join(b"Comp" + struct.pack("<qq", 48, len(big_chunk)) for _ in range(3))
    bomb = header + big_chunk + b"List" + struct.pack("<i", 3) + entries
    with pytest.raises(ValueError, match="cumulative chunk size"):
        parse_vstpreset(bomb)

    # Trailing bytes after TrackControl DGCT zlib stream fail closed
    payload = json.dumps({"settings": {"currentAB": 0}, "ab": [{"parameters": "default"}]}).encode()
    trailing_comp = b"DMGAHDR\x00DGCT\x00\x00\x00\x00ZIP " + zlib.compress(payload) + b"TRAILING"
    trailing_preset = build_vstpreset(
        VstPresetData(1, "565354444D4741547243740000000001", (VstPresetChunk(b"Comp", trailing_comp),))
    )
    assert patch_vst3_trim_preset(trailing_preset, "trackcontrol", -6.0) is None

    # Uncalibrated / external / missing state donors are never marked is_calibrated=True
    opaque_preset = build_vstpreset(
        VstPresetData(1, "565354444D4741547243740000000001", (VstPresetChunk(b"Comp", b"OPAQUE_VENDOR_BLOB"),))
    )
    plug_opaque = ET.fromstring(
        '<Vst3Plugin id="p1" name="TrackControl" deviceName="DMGAudio TrackControl" '
        'deviceID="11111111-2222-3333-4444-555555555555" deviceVendor="DMG Audio" loaded="true">'
        '<Enabled value="true"/><State path="plugins/opaque.vstpreset"/></Vst3Plugin>'
    )
    donor_opaque = inspect_vst3_slot1_trim(plug_opaque, lambda _p: opaque_preset)
    assert donor_opaque is not None
    assert donor_opaque.is_calibrated is False
    assert donor_opaque.has_valid_state is True
    assert donor_opaque.device_name == "DMGAudio TrackControl"
    assert not has_calibrated_vst3_trim_donor([donor_opaque])
    assert select_vst3_trim_donor([donor_opaque], "Stereo", require_calibrated=True) is None
    # Phase 1 unity insert still permits valid-state uncalibrated template donors
    assert select_vst3_trim_donor([donor_opaque], "Stereo", require_calibrated=False) is donor_opaque

    # External state or missing blob is rejected for BOTH calibrated and unity cloning
    plug_ext = ET.fromstring(
        '<Vst3Plugin id="p2" name="TrackControl" deviceID="11111111-2222-3333-4444-555555555555">'
        '<State path="ext.vstpreset" external="true"/></Vst3Plugin>'
    )
    donor_ext = inspect_vst3_slot1_trim(plug_ext, lambda _p: _make_trackcontrol_vstpreset(0.0))
    assert donor_ext is not None and donor_ext.has_valid_state is False
    assert select_vst3_trim_donor([donor_ext], "Stereo", require_calibrated=False) is None

    # Preserves original host-prefixed deviceName when cloning into Slot 1
    ch = ET.fromstring('<Channel id="c1"><Volume value="1.0" unit="linear"/></Channel>')
    extra: dict[str, bytes] = {}
    insert_vst3_trim_slot1_xml(
        ch, donor_opaque,
        make_element=lambda tag, attrib=None: ET.Element(tag, attrib or {}),
        next_xml_id=lambda: "id10",
        used_members=set(), state_counter=[0], extra_archive_blobs=extra, apply_gain_db=None,
    )
    cloned = ch.find("Devices/Vst3Plugin")
    assert cloned is not None and cloned.get("deviceName") == "DMGAudio TrackControl"

    # XML comment (callable .tag, matching lxml.etree._Comment) inside <Devices> must not shift Slot 1 indexing
    xml_with_comment = ET.fromstring(
        '<Project version="1.0"><Structure>'
        '<Channel id="m" role="master" audioChannels="2" name="Master"/>'
        '<Channel id="b" role="submix" audioChannels="2" name="Bus" destination="m">'
        '<Devices><Vst3Plugin id="p1" name="TrackControl" deviceID="11111111-2222-3333-4444-555555555555">'
        '<Enabled value="true"/><State path="plugins/tc.vstpreset"/></Vst3Plugin></Devices>'
        '</Channel></Structure></Project>'
    )
    xml_with_comment.find(".//Channel[@id='b']/Devices").insert(0, ET.Comment("slot 1 comment"))
    mmap = dawproject_xml_to_mixer_session(
        xml_with_comment, lambda _p: _make_trackcontrol_vstpreset(0.0), apply_trim_gain=True,
    )
    bus_trk = next(t for t in mmap.mixer.tracks if t.name == "Bus")
    assert bus_trk.slot_a_trim is not None and bus_trk.slot_a_trim.plugin_id == "trackcontrol"
    assert len(bus_trk.inserts) == 1 and bus_trk.inserts[0].slot_index == 0


def test_prune_orphaned_submix_dangling_idrefs_and_decibel_bounds():
    """Orphaned submix pruning removes dependent automation/scenes cleanly, respects @reference, and decibel volumes expand bounds."""
    root = ET.fromstring(
        """<Project version="1.0">
  <Structure>
    <Channel id="ch_sub_dead" role="submix" name="DeadSub" destination="ch_sub_chain">
      <Volume id="p_dead_vol" value="0.0" unit="decibel" min="-60.0" max="6.0"/>
    </Channel>
    <Channel id="ch_sub_chain" role="submix" name="ChainSub" destination="ch_master"/>
    <Channel id="ch_sub_ref" role="submix" name="RefSub" destination="ch_master">
      <Volume id="p_ref_vol" value="0.0" unit="decibel" min="-60.0" max="6.0"/>
    </Channel>
    <Channel id="ch_master" role="master" name="Master"/>
  </Structure>
  <Arrangement>
    <Lanes timeUnit="seconds">
      <Lanes track="t_surviving">
        <Points><Target parameter="p_dead_vol"/></Points>
        <Clips><Clip time="0" duration="1" reference="ch_sub_ref"/></Clips>
      </Lanes>
      <Lanes track="ch_sub_dead"/>
    </Lanes>
  </Arrangement>
  <Scenes>
    <Scene id="sc_dead"><ClipSlot track="ch_sub_dead"/></Scene>
    <Scene id="sc_keep"><ClipSlot track="t_surviving"/></Scene>
  </Scenes>
</Project>"""
    )
    removed = prune_orphaned_submix_channels_xml(root, {"ch_sub_dead", "ch_sub_ref"})
    assert removed == ["DeadSub", "ChainSub"]
    assert root.find(".//Channel[@id='ch_sub_dead']") is None
    assert root.find(".//Channel[@id='ch_sub_chain']") is None
    # Externally referenced submix survives
    assert root.find(".//Channel[@id='ch_sub_ref']") is not None
    # Dependent automation target, track lane, and emptied scene were pruned (zero dangling IDREFs)
    assert root.find(".//Target[@parameter='p_dead_vol']") is None
    assert root.find(".//Lanes[@track='ch_sub_dead']") is None
    assert root.find(".//Scene[@id='sc_dead']") is None
    assert root.find(".//Scene[@id='sc_keep']") is not None

    # Decibel volume patching expands min/max bounds
    patched = patch_dawproject_channel_volumes_xml(
        root, {"ch_sub_ref": 9.5, "__dp_ch_99": -3.0},
        make_element=lambda tag, attrib=None: ET.Element(tag, attrib or {}),
        next_xml_id=lambda: "id900",
    )
    assert patched == {"ch_sub_ref": 9.5}
    vol = root.find(".//Channel[@id='ch_sub_ref']/Volume")
    assert float(vol.get("value")) == pytest.approx(9.5)
    assert float(vol.get("max")) == pytest.approx(9.5)


def test_bus_slot1_trim_patching_auto_insert_and_uncalibrated_leaf_fallback(tmp_path, monkeypatch):
    """Calibrated bus Slot 1 trims patch existing & auto-insert terminal plugins; missing donor falls back safely."""
    import sessionpreplib.daw_processors.dawproject as dp_mod

    monkeypatch.setattr(dp_mod, "_make_lxml_element", lambda tag, attrib=None: ET.Element(tag, attrib or {}))
    monkeypatch.setattr(
        dp_mod,
        "_next_xml_id_factory",
        lambda root: (lambda c=[100]: (c.__setitem__(0, c[0] + 1) or f"id{c[0]}")),
    )

    # Build a Cubase-style template ZIP where VOX BUS has Slot 1 TrackControl and MIX BUS has empty Devices
    tpl_path = tmp_path / "calibrated.dawproject"
    tc_clean = _make_trackcontrol_vstpreset(0.0, clean=True)
    with zipfile.ZipFile(tpl_path, "w") as zf:
        zf.writestr("project.xml", CUBASE_MIX_TEMPLATE_XML)
        zf.writestr("metadata.xml", "<MetaData/>")
        zf.writestr("plugins/tc_vox.vstpreset", tc_clean)
        zf.writestr("plugins/tc_dirty.vstpreset", _make_trackcontrol_vstpreset(-4.0, clean=False))

    # Hot stereo leaf tracks that drive VOX BUS (existing TC) and MIX BUS (auto-insert TC)
    wav1 = tmp_path / "vox1.wav"
    wav2 = tmp_path / "vox2.wav"
    sf.write(wav1, _sine_stereo(-12.0, freq=330.0), 48000, subtype="PCM_24")
    sf.write(wav2, _sine_stereo(-12.0, freq=550.0), 48000, subtype="PCM_24")
    t1 = TrackContext(
        filename="vox1.wav", filepath=str(wav1), audio_data=None, samplerate=48000, channels=2,
        total_samples=24000, bitdepth="24-bit", subtype="PCM_24", duration_sec=0.5,
        processor_results={
            "bimodal_normalize": ProcessorResult(
                processor_id="bimodal_normalize", classification="Sustained", method="RMS",
                gain_db=6.0, data={"fader_offset": -6.0},
            )
        },
    )
    t2 = TrackContext(
        filename="vox2.wav", filepath=str(wav2), audio_data=None, samplerate=48000, channels=2,
        total_samples=24000, bitdepth="24-bit", subtype="PCM_24", duration_sec=0.5,
        processor_results={
            "bimodal_normalize": ProcessorResult(
                processor_id="bimodal_normalize", classification="Sustained", method="RMS",
                gain_db=6.0, data={"fader_offset": -6.0},
            )
        },
    )
    session = SessionContext([t1, t2], {"_use_processed": False, "target_rms": -18.0, "target_peak": -6.0})

    root = ET.fromstring(CUBASE_MIX_TEMPLATE_XML)
    # Remove <Devices> from ch_mix so MIX BUS genuinely exercises auto-insertion on an empty Slot 1
    mix_ch = root.find(".//Channel[@id='ch_mix']")
    mix_devs = mix_ch.find("Devices")
    if mix_devs is not None:
        mix_ch.remove(mix_devs)
    vox_folder = root.find(".//Track[@id='f_vox']")
    p1 = resolve_dawproject_track_staging(session, t1, gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=True)
    p2 = resolve_dawproject_track_staging(session, t2, gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=True)
    trk1_el = ET.fromstring(
        f'<Track id="t_v1" name="Vox 1" contentType="audio"><Channel id="ch_v1" role="regular" audioChannels="2" destination="ch_vox_f">'
        f'<Volume id="v_v1" value="{10 ** (p1.fader_db / 20.0):.6f}" unit="linear" min="0.0" max="2.0"/></Channel></Track>'
    )
    trk2_el = ET.fromstring(
        f'<Track id="t_v2" name="Vox 2" contentType="audio"><Channel id="ch_v2" role="regular" audioChannels="2" destination="ch_vox_f">'
        f'<Volume id="v_v2" value="{10 ** (p2.fader_db / 20.0):.6f}" unit="linear" min="0.0" max="2.0"/></Channel></Track>'
    )
    vox_folder.append(trk1_el)
    vox_folder.append(trk2_el)

    extra_blobs: dict[str, bytes] = {}
    used_members = {"project.xml", "metadata.xml", "plugins/tc_vox.vstpreset", "plugins/tc_dirty.vstpreset"}
    with zipfile.ZipFile(tpl_path, "r") as zf:
        graph = _apply_signal_flow_and_trims(
            zf, root, session,
            [("Vox 1", trk1_el, t1, p1, "Stereo"), ("Vox 2", trk2_el, t2, p2, "Stereo")],
            gain_mode=GAIN_MODE_TRIM_AND_FADER, insert_track_trim=True, apply_trim_gain=True,
            stage_bus_faders=True, pan_law_db=-3.0, fader_ceiling_db=6.0,
            bus_fader_overrides=None, used_members=used_members, extra_blobs=extra_blobs,
        )

    assert graph is not None
    rec_trims = graph.recommended_slot_a_trims()
    assert "VOX BUS" in rec_trims
    # Existing VOX BUS Slot 1 TrackControl got patched to a fresh archive member (no duplicate ZIP overwrite)
    vox_bus_state = root.find(".//Channel[@id='ch_vox_bus']/Devices/Vst3Plugin/State")
    assert vox_bus_state is not None
    new_vox_path = vox_bus_state.get("path")
    assert new_vox_path != "plugins/tc_vox.vstpreset" and new_vox_path in extra_blobs
    patched_vox_preset = parse_vstpreset(extra_blobs[new_vox_path])
    patched_vox_comp = extract_vstpreset_chunk(patched_vox_preset, b"Comp")
    assert patched_vox_comp is not None
    # Terminal MIX BUS (empty Devices) unconditionally received auto-inserted calibrated TrackControl
    assert "MIX BUS" in rec_trims
    mix_plug = root.find(".//Channel[@id='ch_mix']/Devices/Vst3Plugin")
    assert mix_plug is not None and mix_plug.get("name") == "TrackControl"
    assert mix_plug.find("State").get("path") in extra_blobs

    # Preview with apply_trim_gain=True on a template with NO calibrated donor falls back cleanly (no leaf balance inversion)
    no_donor_tpl = tmp_path / "no_donor.dawproject"
    no_donor_xml = CUBASE_MIX_TEMPLATE_XML.replace("plugins/tc_vox.vstpreset", "plugins/missing.vstpreset")
    with zipfile.ZipFile(no_donor_tpl, "w") as zf:
        zf.writestr("project.xml", no_donor_xml)
        zf.writestr("metadata.xml", "<MetaData/>")
    sess_prev = SessionContext(
        [t1], {"_use_processed": False}, output_tracks=[t1],
        transfer_manifest=[TransferEntry("e1", t1.filename, "Vox 1")],
    )
    g_fallback = preview_dawproject_signal_flow(
        no_donor_tpl, sess_prev, {"e1": "f_vox"}, [{"id": "f_vox", "name": "VOX"}],
        gain_mode=GAIN_MODE_TRIM_AND_FADER, apply_trim_gain=True,
    )
    assert g_fallback.nodes["Vox 1"].leaf_trim_db == 0.0
    assert g_fallback.nodes["Vox 1"].recommended_fader_db == 0.0


def test_dawproject_transfer_end_to_end_trims_and_buses(tmp_path):
    """Full DawProjectDawProcessor.transfer() roundtrip when dawproject-py is installed (CI)."""
    pytest.importorskip("dawproject")
    tpl_path = tmp_path / "full_cubase.dawproject"
    with zipfile.ZipFile(tpl_path, "w") as zf:
        zf.writestr("project.xml", CUBASE_MIX_TEMPLATE_XML)
        zf.writestr("metadata.xml", "<MetaData/>")
        zf.writestr("plugins/tc_vox.vstpreset", _make_trackcontrol_vstpreset(0.0, clean=True))
        zf.writestr("plugins/tc_dirty.vstpreset", _make_trackcontrol_vstpreset(-4.0, clean=False))

    wav = tmp_path / "lead.wav"
    sf.write(wav, _sine_stereo(-16.0, freq=440.0), 48000, subtype="PCM_24")
    tc = TrackContext(
        filename="lead.wav", filepath=str(wav), audio_data=None, samplerate=48000, channels=2,
        total_samples=24000, bitdepth="24-bit", subtype="PCM_24", duration_sec=0.5,
        processor_results={
            "bimodal_normalize": ProcessorResult(
                processor_id="bimodal_normalize", classification="Sustained", method="RMS",
                gain_db=6.0, data={"fader_offset": -6.0},
            )
        },
    )
    session = SessionContext(
        [tc],
        {"_use_processed": False, "dawproject_apply_trim_gain": True, "dawproject_remove_unused_buses": True},
        output_tracks=[tc], prepare_state="ready",
        transfer_manifest=[TransferEntry("e1", tc.filename, "Lead")],
    )
    adapter = DawProjectDawProcessor(template_path=str(tpl_path), template_fader_ceiling_db=6.0)
    adapter.fetch(session)
    session.daw_state[adapter.id]["assignments"] = {"e1": "f_vox"}
    out = tmp_path / "exported.dawproject"
    results = adapter.transfer(session, str(out))
    assert all(r.success for r in results), [r.error for r in results]
    assert "signal_flow" in session.daw_state[adapter.id]
    with zipfile.ZipFile(out, "r") as zf:
        # Zero duplicate ZIP entries
        names = zf.namelist()
        assert len(names) == len(set(names))
        out_root = ET.fromstring(zf.read("project.xml"))
        # Orphaned submix bus pruned, effect return preserved
        assert out_root.find(".//Channel[@id='ch_orphan_bus']") is None
        assert out_root.find(".//Channel[@id='ch_verb']") is not None
        # Leaf received Slot 1 TrackControl VST3
        lead_plug = out_root.find(".//Track[@name='Lead']/Channel/Devices/Vst3Plugin")
        assert lead_plug is not None and lead_plug.get("name") == "TrackControl"
