"""Read exported DAWproject archives independently of the DAWproject writer."""
from io import BytesIO
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track
from sessionpreplib.daw_processors.dawproject import DawProjectDawProcessor
from sessionpreplib.models import SessionContext, TransferEntry


TEMPLATE_XML = """<Project version="1.0">
  <Application name="Fixture" version="1"/>
  <Structure><Channel id="mixOut" role="master"/><Track id="band" name="Band" contentType="tracks">
    <Channel id="bus" role="submix"/>
    <Track id="keys" name="Keys" contentType="tracks"><Channel id="keysBus" role="submix"/></Track>
    <Track id="guide" name="Guide" contentType="audio"><Channel id="guideCh" role="regular" destination="laterBus"/></Track>
    <Track id="busTrack" name="Mix return" contentType="audio"><Channel id="laterBus" role="effect"/></Track>
  </Track></Structure>
  <Arrangement><Lanes timeUnit="seconds"><Lanes track="guide"><Clips>
    <Clip time="0" duration="0.001"><Audio duration="0.001" channels="1" sampleRate="8000">
      <File path="audio/guide.wav" external="false"/>
    </Audio></Clip><Clip time="1" duration="0.001"><Audio duration="0.001" channels="1" sampleRate="8000">
      <File path="Reference files/cue.wav" external="true"/>
    </Audio></Clip></Clips></Lanes></Lanes></Arrangement>
</Project>"""


@pytest.fixture
def archive_session(tmp_path):
    template = tmp_path / "Band.dawproject"
    embedded = BytesIO()
    sf.write(embedded, np.arange(8) / 16, 8000, format="WAV", subtype="PCM_16")
    cue = tmp_path / "Reference files" / "cue.wav"
    cue.parent.mkdir()
    cue.write_bytes(embedded.getvalue())
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", TEMPLATE_XML)
        archive.writestr("metadata.xml", "<MetaData><Title>Guide session</Title></MetaData>")
        archive.writestr("audio/guide.wav", embedded.getvalue())
    tracks = []
    entries = []
    for i, name in enumerate(["Clav été.wav", "Synth.wav"], 1):
        path = tmp_path / name
        pcm = np.array([0.125, -0.25, 0.375]) * i
        sf.write(path, pcm, 48000)
        track = load_track(str(path))
        prepared = tmp_path / ("Prepared " + name)
        sf.write(prepared, pcm * 0.5, 48000)
        track.processed_filepath = str(prepared)
        track.prepared_fader_db = 6.020599913279624
        tracks.append(track)
        entries.append(TransferEntry(str(i), name, "Keys part"))
    adapter = DawProjectDawProcessor(template_path=str(template))
    session = SessionContext(tracks, {}, output_tracks=tracks, prepare_state="ready", transfer_manifest=entries)
    adapter.fetch(session)
    state = session.daw_state[adapter.id]
    assert [(f["id"], f["parent_id"]) for f in state["folders"]] == [("band", None), ("keys", "band")]
    state.update(assignments={"1": "keys", "2": "keys"}, track_order={"keys": ["2", "1"]})
    return adapter, session


@pytest.mark.parametrize("processed", [False, True])
def test_reopen_template_and_measure_exported_media(archive_session, tmp_path, processed):
    adapter, session = archive_session
    session.config["_use_processed"] = processed
    output = tmp_path / "Mix" / "Song.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]
    with ZipFile(output) as archive:
        root = ET.fromstring(archive.read("project.xml"))
        assert ET.fromstring(archive.read("metadata.xml")).findtext("Title") == "Guide session"
        guide = root.find(".//Lanes[@track='guide']//File")
        np.testing.assert_array_equal(sf.read(BytesIO(archive.read(guide.attrib["path"])))[0], np.arange(8) / 16)
        external = root.find(".//Lanes[@track='guide']//File[@external='true']")
        assert (output.parent / external.get("path")).samefile(tmp_path / "Reference files" / "cue.wav")
        assert root.find(".//Channel[@id='guideCh']").get("destination") == "laterBus"
        tracks = root.findall(".//Track[@id='keys']/Track")
        assert [t.get("name") for t in tracks] == ["Keys part", "Keys part"]
        ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
        assert len(set(ids)) == len(ids)
        for number, track in zip([2, 1], tracks, strict=True):
            channel = track.find("Channel")
            assert channel.get("destination") == "keysBus"
            assert channel.get("audioChannels") == "1"
            assert channel.get("name") == "Keys part"
            assert channel.find("Mute").get("value") == "false"
            assert float(channel.find("Pan").get("min")) == 0.0
            assert float(channel.find("Pan").get("max")) == 1.0
            vol = channel.find("Volume")
            assert float(vol.get("value")) == pytest.approx(2 if processed else 1)
            assert float(vol.get("min")) == 0.0
            assert float(vol.get("max")) >= float(vol.get("value"))
            clip = root.find(f".//Lanes[@track='{track.get('id')}']/Clips/Clip")
            assert clip.get("name") == "Keys part"
            assert clip.get("contentTimeUnit") == "seconds"
            assert float(clip.get("time")) == 0
            assert float(clip.get("duration")) == pytest.approx(3 / 48000)
            file = clip.find("Audio/File")
            assert file.get("external") == "false"
            arc_path = file.get("path")
            assert arc_path.startswith("audio/")
            data, rate = sf.read(BytesIO(archive.read(arc_path)))
            assert rate == 48000
            np.testing.assert_array_equal(data, np.array([0.125, -0.25, 0.375]) * number * (0.5 if processed else 1))


def test_locked_export_preserves_previous_archive_and_reports_failure(archive_session, tmp_path, monkeypatch):
    adapter, session = archive_session
    target = tmp_path / "Saved.dawproject"
    prior = (tmp_path / "Band.dawproject").read_bytes()
    target.write_bytes(prior)
    publish = os.replace

    def locked(source, destination):
        if Path(destination) == target:
            raise PermissionError("export is locked")
        publish(source, destination)

    monkeypatch.setattr(os, "replace", locked)
    results = adapter.transfer(session, str(target))
    assert not any(result.success for result in results)
    assert any("locked" in (r.error or "") for r in results)
    assert target.read_bytes() == prior


def test_missing_original_excludes_only_its_track(archive_session, tmp_path):
    adapter, session = archive_session
    session.config["_use_processed"] = False
    Path(session.tracks[1].filepath).unlink()
    output = tmp_path / "Partial.dawproject"
    results = adapter.transfer(session, str(output))
    assert [r.command.target for r in results if not r.success] == ["2"]
    with ZipFile(output) as archive:
        root = ET.fromstring(archive.read("project.xml"))
        assert len(root.findall(".//Track[@id='keys']/Track")) == 1


def test_batch_destination_created_during_transfer_is_preserved(archive_session, tmp_path):
    adapter, session = archive_session
    session.config["_batch_transfer"] = True
    target = tmp_path / "Race.dawproject"

    def another_export(_current, _total, _message):
        if not target.exists():
            target.write_bytes(b"project saved by another window")

    results = adapter.transfer(session, str(target), progress_cb=another_export)
    assert results and all(not r.success for r in results)
    assert target.read_bytes() == b"project saved by another window"


CUBASE_ROUTING_TEMPLATE = """<Project version="1.0">
  <Application name="Cubase" version="16.0.5"/>
  <Structure>
    <Track id="id325" name="Kick" contentType="tracks" loaded="true">
      <Channel id="id0" role="master" audioChannels="2" name="Kick" destination="id75"/>
      <Track id="id326" name="Kick Sub" contentType="tracks" loaded="true"/>
    </Track>
    <Track id="id427" name="Input/Output" contentType="tracks" loaded="true"/>
    <Channel id="id75" role="submix" audioChannels="2" name="cKick" destination="id150"/>
    <Channel id="id150" role="submix" audioChannels="2" name="Mix 01" destination="id175"/>
    <Channel id="id200" role="master" audioChannels="2" name="AIO 5/6 (E5K In)"/>
    <Channel id="id175" role="master" audioChannels="2" name="Stereo Out"/>
  </Structure>
  <Arrangement><Lanes timeUnit="seconds"><Lanes track="id325"/></Lanes></Arrangement>
</Project>"""


def test_cubase_routing_channels_colors_and_basic_folder_fallback(tmp_path):
    template = tmp_path / "Cubase.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", CUBASE_ROUTING_TEMPLATE)
        archive.writestr("metadata.xml", "<MetaData/>")

    mono_wav = tmp_path / "Kick_In.wav"
    stereo_wav = tmp_path / "Room.wav"
    sf.write(mono_wav, np.full(4800, 0.25), 48000, subtype="PCM_24")
    sf.write(stereo_wav, np.full((4800, 2), 0.125), 48000, subtype="PCM_24")
    t_mono = load_track(str(mono_wav))
    t_stereo = load_track(str(stereo_wav))
    t_mono.processed_filepath, t_mono.prepared_fader_db = str(mono_wav), 9.0
    t_stereo.processed_filepath, t_stereo.prepared_fader_db = str(stereo_wav), -3.0

    entries = [
        TransferEntry("e_kick", t_mono.filename, "Kick In.wav", group="Kick"),
        TransferEntry("e_sub", t_mono.filename, "Kick Sub.wav", group="Kick"),
        TransferEntry("e_io", t_stereo.filename, "Room.wav", group="Room"),
    ]
    cfg = {
        "_use_processed": True,
        "gui": {
            "groups": [{"name": "Kick", "color": "Lime"}, {"name": "Room", "color": "Amber"}],
            "colors": [{"name": "Lime", "argb": "#FF90FF20"}, {"name": "Amber", "argb": "#FFFFAA00"}],
        },
    }
    session = SessionContext([t_mono, t_stereo], cfg, output_tracks=[t_mono, t_stereo],
                             prepare_state="ready", transfer_manifest=entries)
    adapter = DawProjectDawProcessor(template_path=str(template))
    adapter.fetch(session)
    session.daw_state[adapter.id]["assignments"] = {"e_kick": "id325", "e_sub": "id326", "e_io": "id427"}

    output = tmp_path / "CubaseOut.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]

    with ZipFile(output) as archive:
        root = ET.fromstring(archive.read("project.xml"))
        names = archive.namelist()
        # Same source WAV referenced by two entries is embedded only once
        assert names.count("audio/Kick_In.wav") == 1
        assert "audio/Room.wav" in names

        kick_tr = root.find(".//Track[@id='id325']/Track[@name='Kick In']")
        sub_tr = root.find(".//Track[@id='id326']/Track[@name='Kick Sub']")
        io_tr = root.find(".//Track[@id='id427']/Track[@name='Room']")
        assert kick_tr.get("color") == "#90FF20"
        assert kick_tr.find("Channel").get("color") == "#90FF20"
        assert kick_tr.find("Channel").get("audioChannels") == "1"
        assert kick_tr.find("Channel").get("destination") == "id0"
        # +9 dB linear gain (~2.818) expands Volume max so value stays in-range
        vol = kick_tr.find("Channel/Volume")
        assert float(vol.get("value")) == pytest.approx(10 ** (9.0 / 20))
        assert float(vol.get("max")) == pytest.approx(float(vol.get("value")))
        # Nested basic folder inherits ancestor routing folder channel id0
        assert sub_tr.find("Channel").get("destination") == "id0"
        # Top-level basic folder falls back to referenced Stereo Out (id175), NOT AIO 5/6 (id200)
        assert io_tr.find("Channel").get("audioChannels") == "2"
        assert io_tr.find("Channel").get("destination") == "id175"
        assert io_tr.find("Channel").get("color") == "#FFAA00"
        room_clip = root.find(f".//Lanes[@track='{io_tr.get('id')}']/Clips/Clip")
        assert room_clip.get("color") == "#FFAA00"
        assert room_clip.get("name") == "Room"


def test_embedded_audio_disambiguates_template_and_duplicate_basenames(tmp_path):
    template = tmp_path / "Collide.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", TEMPLATE_XML)
        archive.writestr("audio/guide.wav", b"original-template-guide")

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    wav_a = dir_a / "guide.wav"
    wav_b = dir_b / "guide.wav"
    sf.write(wav_a, np.full(16, 0.25), 48000, subtype="PCM_16")
    sf.write(wav_b, np.full(16, -0.5), 48000, subtype="PCM_16")
    ta = load_track(str(wav_a))
    tb = load_track(str(wav_b))
    ta.filename, tb.filename = "a/guide.wav", "b/guide.wav"

    entries = [TransferEntry("1", "a/guide.wav", "Guide A"), TransferEntry("2", "b/guide.wav", "Guide B")]
    session = SessionContext([ta, tb], {"_use_processed": False}, output_tracks=[ta, tb],
                             prepare_state="ready", transfer_manifest=entries)
    adapter = DawProjectDawProcessor(template_path=str(template))
    adapter.fetch(session)
    session.daw_state[adapter.id].update(assignments={"1": "keys", "2": "keys"}, track_order={"keys": ["1", "2"]})

    output = tmp_path / "Disambiguated.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]

    with ZipFile(output) as archive:
        assert archive.read("audio/guide.wav") == b"original-template-guide"
        assert archive.read("audio/guide_2.wav") == wav_a.read_bytes()
        assert archive.read("audio/guide_3.wav") == wav_b.read_bytes()
        root = ET.fromstring(archive.read("project.xml"))
        paths = [
            root.find(f".//Lanes[@track='{t.get('id')}']/Clips/Clip/Audio/File").get("path")
            for t in root.findall(".//Track[@id='keys']/Track")
        ]
        assert paths == ["audio/guide_2.wav", "audio/guide_3.wav"]


BITWIG_IDLESS_TEMPLATE = """<Project version="1.0">
  <Application name="Bitwig Studio" version="5.2"/>
  <Structure>
    <!-- comment inside Structure must not break element indexing -->
    <Track name="Drums" contentType="tracks" loaded="true">
      <Channel role="submix" audioChannels="2" name="Drums"/>
    </Track>
    <Track id="id0" name="Basic" contentType="tracks" loaded="true"/>
    <Track name="Master" contentType="audio notes" loaded="true">
      <Channel role="master" audioChannels="2" name="Master"/>
    </Track>
  </Structure>
</Project>"""


def test_bitwig_master_track_idless_channels_and_structure_comments(tmp_path):
    template = tmp_path / "Bitwig.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", BITWIG_IDLESS_TEMPLATE)

    wav = tmp_path / "Snare.wav"
    sf.write(wav, np.full(480, 0.2), 48000, subtype="PCM_16")
    track = load_track(str(wav))
    entries = [TransferEntry("1", track.filename, "Snare A"), TransferEntry("2", track.filename, "Snare B")]
    session = SessionContext([track], {"_use_processed": False}, output_tracks=[track],
                             prepare_state="ready", transfer_manifest=entries)
    adapter = DawProjectDawProcessor(template_path=str(template))
    adapter.fetch(session)
    fmap = {f["name"]: f["id"] for f in session.daw_state[adapter.id]["folders"]}
    session.daw_state[adapter.id]["assignments"] = {"1": fmap["Drums"], "2": fmap["Basic"]}

    output = tmp_path / "BitwigOut.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]

    with ZipFile(output) as archive:
        root = ET.fromstring(archive.read("project.xml"))
        ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

        drums_ch_id = root.find(".//Track[@name='Drums']/Channel").get("id")
        master_ch_id = root.find(".//Track[@name='Master']/Channel").get("id")
        assert drums_ch_id and master_ch_id
        assert root.find(".//Track[@name='Snare A']/Channel").get("destination") == drums_ch_id
        assert root.find(".//Track[@name='Snare B']/Channel").get("destination") == master_ch_id


def test_pre_1980_mtime_case_collision_and_compression_types(tmp_path):
    import zipfile

    template = tmp_path / "Case.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", TEMPLATE_XML)
        archive.writestr("audio/Guide.WAV", b"template-upper")

    wav = tmp_path / "guide.wav"
    sf.write(wav, np.full(16, 0.1), 48000, subtype="PCM_16")
    os.utime(wav, (0, 0))  # 1970-01-01 pre-1980 timestamp
    track = load_track(str(wav))
    session = SessionContext([track], {"_use_processed": False}, output_tracks=[track],
                             prepare_state="ready", transfer_manifest=[TransferEntry("1", track.filename, "G")])
    adapter = DawProjectDawProcessor(template_path=str(template))
    adapter.fetch(session)
    session.daw_state[adapter.id]["assignments"] = {"1": "keys"}

    output = tmp_path / "Pre1980.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]

    with ZipFile(output) as archive:
        assert archive.read("audio/Guide.WAV") == b"template-upper"
        assert archive.read("audio/guide_2.wav") == wav.read_bytes()
        assert archive.getinfo("project.xml").compress_type == zipfile.ZIP_DEFLATED
        assert archive.getinfo("audio/guide_2.wav").compress_type == zipfile.ZIP_STORED


def test_embedded_audio_io_failure_rolls_back_atomically(archive_session, tmp_path):
    adapter, session = archive_session
    session.config["_use_processed"] = False
    target = tmp_path / "Atomic.dawproject"
    prior = b"previous-good-archive"
    target.write_bytes(prior)
    wav_path = Path(session.tracks[0].filepath)

    def remove_audio_before_save(current, total, _msg):
        if current == total and wav_path.exists():
            wav_path.unlink()

    results = adapter.transfer(session, str(target), progress_cb=remove_audio_before_save)
    assert results and all(not r.success for r in results)
    assert target.read_bytes() == prior


def test_prunes_empty_folders_vcas_and_dangling_lanes_while_protecting_routed_buses(tmp_path):
    xml = """<Project version="1.0">
  <Application name="Cubase" version="14"/>
  <Structure>
    <Track id="f_kick" name="Kick" contentType="tracks" loaded="true">
      <Channel id="ch_kick" role="master" audioChannels="2" name="Kick" destination="ch_drum_bus"/>
    </Track>
    <Track id="f_drum_bus" name="DrumBus" contentType="tracks" loaded="true">
      <Channel id="ch_drum_bus" role="submix" audioChannels="2" name="DrumBus" destination="ch_master"/>
    </Track>
    <Track id="f_snare" name="Snare" contentType="tracks" loaded="true">
      <Channel id="ch_snare" role="master" audioChannels="2" name="Snare" destination="ch_sub_snare">
        <Volume id="p_snare_vol" value="1.0" unit="linear"/>
      </Channel>
      <Track id="t_snare_vca" name="Snare VCA" contentType="audio" loaded="true">
        <Channel id="ch_snare_vca" role="vca" name="Snare VCA"/>
      </Track>
    </Track>
    <Track id="f_chain_a" name="ChainA" contentType="tracks" loaded="true">
      <Channel id="ch_chain_a" role="submix" audioChannels="2" name="ChainA" destination="ch_chain_b"/>
    </Track>
    <Track id="f_chain_b" name="ChainB" contentType="tracks" loaded="true">
      <Channel id="ch_chain_b" role="submix" audioChannels="2" name="ChainB" destination="ch_master"/>
    </Track>
    <Track id="f_ref_protected" name="RefProtected" contentType="tracks" loaded="true"/>
    <Track id="f_vcas" name="VCAs" contentType="tracks" loaded="true">
      <Track id="t_vca_kick" name="Kick VCA" contentType="audio" loaded="true">
        <Channel id="ch_vca_kick" role="vca" name="Kick VCA"/>
      </Track>
      <Track id="t_vca_other" name="Print VCA" contentType="audio" loaded="true">
        <Channel id="ch_vca_other" role="vca" name="Print VCA"/>
      </Track>
    </Track>
    <Track id="f_outer_empty" name="OuterEmpty" contentType="tracks" loaded="true">
      <Track id="f_inner_empty" name="InnerEmpty" contentType="tracks" loaded="true"/>
    </Track>
    <Track name="BitwigUnused" contentType="tracks" loaded="true"/>
    <Track id="f_fx" name="FX" contentType="tracks" loaded="true">
      <Track id="t_reverb" name="Reverb" contentType="audio" loaded="true">
        <Channel id="ch_reverb" role="effect" audioChannels="2" name="Reverb" destination="ch_master"/>
      </Track>
    </Track>
    <Channel id="ch_sub_snare" role="submix" audioChannels="2" name="cSnare" destination="ch_master"/>
    <Channel id="ch_master" role="master" audioChannels="2" name="Stereo Out"/>
  </Structure>
  <Arrangement>
    <Lanes timeUnit="seconds">
      <Lanes track="f_kick">
        <Points><Target parameter="p_snare_vol"/></Points>
        <Clips><Clip time="0" duration="1" reference="lane_ref_protected"/></Clips>
      </Lanes>
      <Lanes id="lane_ref_protected" track="f_ref_protected"/>
      <Lanes track="f_snare"/>
      <Lanes track="f_inner_empty"/>
      <Lanes track="t_snare_vca"/>
      <Lanes track="f_fx"/>
    </Lanes>
  </Arrangement>
  <Scenes>
    <Scene id="sc_empty"><ClipSlot track="f_snare"/></Scene>
    <Scene id="sc_keep"><ClipSlot track="f_kick"/></Scene>
  </Scenes>
</Project>"""
    template = tmp_path / "EmptyPrune.dawproject"
    with ZipFile(template, "w") as archive:
        archive.writestr("project.xml", xml)
        archive.writestr("metadata.xml", "<MetaData/>")
    template_bytes_before = template.read_bytes()

    wav = tmp_path / "Kick.wav"
    sf.write(wav, np.full(480, 0.25), 48000, subtype="PCM_16")
    track = load_track(str(wav))
    session = SessionContext(
        [track], {"_use_processed": False}, output_tracks=[track],
        prepare_state="ready", transfer_manifest=[TransferEntry("e1", track.filename, "Kick")],
    )
    adapter = DawProjectDawProcessor(template_path=str(template))
    adapter.fetch(session)
    session.daw_state[adapter.id]["assignments"] = {"e1": "f_kick"}

    output = tmp_path / "Pruned.dawproject"
    results = adapter.transfer(session, str(output))
    assert all(r.success for r in results), [r.error for r in results]
    assert template.read_bytes() == template_bytes_before
    removed_targets = {r.command.target for r in results if r.command.command_type == "remove_empty_folder"}
    assert removed_targets == {"Snare", "ChainA", "ChainB", "InnerEmpty", "OuterEmpty", "BitwigUnused"}
    removed_buses = {r.command.target for r in results if r.command.command_type == "remove_unused_bus"}
    assert removed_buses == {"cSnare"}

    with ZipFile(output) as archive:
        root = ET.fromstring(archive.read("project.xml"))
        # Populated Kick, externally-fed DrumBus, externally-referenced RefProtected, VCAs organizer, FX survive
        assert root.find(".//Track[@id='f_kick']") is not None
        assert root.find(".//Track[@id='f_drum_bus']") is not None
        assert root.find(".//Track[@id='f_ref_protected']") is not None
        assert root.find(".//Track[@id='f_vcas']") is not None
        assert root.find(".//Track[@id='f_fx']") is not None
        # Orphaned standalone submix cSnare (previously fed only by removed f_snare) is pruned by default
        assert root.find(".//Channel[@id='ch_sub_snare']") is None
        # Empty Snare (VCA only), chained ChainA->ChainB, nested OuterEmpty/InnerEmpty, ID-less BitwigUnused removed
        for gone_id in ("f_snare", "t_snare_vca", "f_chain_a", "f_chain_b", "f_outer_empty", "f_inner_empty"):
            assert root.find(f".//Track[@id='{gone_id}']") is None
        assert root.find(".//Track[@name='BitwigUnused']") is None
        # Standalone automation Points targeting deleted p_snare_vol inside surviving f_kick lane was pruned
        assert root.find(".//Target[@parameter='p_snare_vol']") is None
        # Empty Scene sc_empty removed (no invalid empty <Scene/>); sc_keep preserved
        assert root.find(".//Scene[@id='sc_empty']") is None
        assert root.find(".//Scene[@id='sc_keep']") is not None
        # Zero dangling IDREFs anywhere
        all_ids = {el.get("id") for el in root.iter() if el.get("id")}
        for attr in ("track", "destination", "reference", "parameter"):
            for el in root.iter():
                ref = el.get(attr)
                if ref is not None:
                    assert ref in all_ids, f"Dangling {attr}={ref!r} on <{el.tag}>"

    # When dawproject_remove_unused_buses=False, standalone submix cSnare is preserved
    keep_session = SessionContext(
        [track], {"_use_processed": False, "dawproject_remove_unused_buses": False}, output_tracks=[track],
        prepare_state="ready", transfer_manifest=[TransferEntry("e1", track.filename, "Kick")],
    )
    adapter.fetch(keep_session)
    keep_session.daw_state[adapter.id]["assignments"] = {"e1": "f_kick"}
    keep_out = tmp_path / "KeptSubmix.dawproject"
    keep_results = adapter.transfer(keep_session, str(keep_out))
    assert all(r.success for r in keep_results), [r.error for r in keep_results]
    assert not any(r.command.command_type == "remove_unused_bus" for r in keep_results)
    with ZipFile(keep_out) as archive:
        keep_root = ET.fromstring(archive.read("project.xml"))
        assert keep_root.find(".//Channel[@id='ch_sub_snare']") is not None
