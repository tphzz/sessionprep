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
            assert float(channel.find("Volume").get("value")) == pytest.approx(2 if processed else 1)
            clip = root.find(f".//Lanes[@track='{track.get('id')}']/Clips/Clip")
            assert float(clip.get("time")) == 0
            assert float(clip.get("duration")) == pytest.approx(3 / 48000)
            file = clip.find("Audio/File")
            path = output.parent / file.get("path")
            assert file.get("external") == "true"
            data, rate = sf.read(path)
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
