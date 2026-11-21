"""Verify Track Layout edits against independently specified PCM samples."""

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from sessionprepgui.analysis.worker import TopologyApplyWorker
from sessionprepgui.session import io
from sessionprepgui.topology import operations as edit
from sessionprepgui.topology.output_tree import OutputTree
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import load_session
from sessionpreplib.topology import ChannelRoute, TopologyEntry, TopologySource, build_default_topology, resolve_entry_audio


def apply_layout(session, source, destination):
    errors = []
    worker = TopologyApplyWorker(session, str(destination), str(source))
    worker.error.connect(lambda _, error: errors.append(error))
    worker.run()
    return errors + [message for _, message in session.config.get("_topology_apply_errors", [])]


@pytest.mark.parametrize("fold", [True, False], ids=["summed", "unity"])
def test_move_channel_through_output_tree_and_reopen(tmp_path, _font_app, fold):
    # Different lengths and two files with the same basename expose identity,
    # channel swaps, missing weights, and incorrect zero padding.
    src = tmp_path / "src"
    for folder, pcm in [
        ("stereo", [[0.25, 0.5], [-0.125, 0.25], [0, -0.5]]),
        ("mono", [0.0625, -0.25]),
    ]:
        (src / folder).mkdir(parents=True)
        sf.write(src / folder / "Take.wav", pcm, 48000, subtype="PCM_24")
    session = load_session(str(src), {}, recursive=True)
    tracks = {t.filename: t for t in session.tracks}
    topo = build_default_topology(session.tracks)
    donor = "stereo/Take.wav"
    target = "mono/Take.wav"
    expected = [0.25, -0.125, 0.]
    if fold:
        edit.sum_to_mono(topo, tracks, donor)
        donor = "stereo/Take_mono.wav"
        expected = [0.375, 0.0625, -0.25]

    tree = OutputTree()
    tree.resize(800, 400)
    try:
        tree.populate(topo, tracks)
        tree.expandAll()
        tree.show()
        _font_app.processEvents()
        donor_item, = tree.findItems(donor, Qt.MatchExactly)
        target_item, = tree.findItems(target, Qt.MatchExactly)
        mime = tree.mimeData([donor_item.child(0)])
        position = QPointF(tree.visualItemRect(target_item).center())
        event = QDropEvent(position, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
        tree.dropEvent(event)
        assert event.isAccepted()
    finally:
        tree.close()

    saved = tmp_path / "edits.spsession"
    io.save_session(str(saved), {"source_dir": str(src), "tracks": session.tracks, "topology": topo})
    state = io.load_session(str(saved))
    restored = SessionContext(state["tracks"], {}, topology=state["topology"])
    by_name = {e.output_filename: e for e in restored.topology.entries}
    track_audio = {name: (t.audio_data, t.samplerate) for name, t in tracks.items()}
    reference = np.column_stack(([0.0625, -0.25, 0.], expected))
    np.testing.assert_array_equal(resolve_entry_audio(by_name[target], track_audio), reference)
    out_dir = tmp_path / "layout"
    assert apply_layout(restored, src, out_dir) == []
    audio, sr = sf.read(out_dir / target)
    assert sr == 48000
    np.testing.assert_array_equal(audio, reference)
    if not fold:
        np.testing.assert_array_equal(sf.read(out_dir / donor)[0], [0.5, 0.25, -0.5])


def test_split_extract_clear_rewire_and_reorder_maintain_channel_identity(tmp_path):
    left, right = [0.25, 0, 0, 0], [0, -0.5, 0, 0]
    bass = [0, 0, 0.125]
    sf.write(tmp_path / "Keys.wav", np.column_stack((left, right)), 44100)
    sf.write(tmp_path / "Bass.wav", bass, 44100)
    session = load_session(str(tmp_path), {})
    tracks = {t.filename: t for t in session.tracks}
    inputs = {name: (t.audio_data, t.samplerate) for name, t in tracks.items()}
    topo = build_default_topology(session.tracks)

    def samples(name):
        return resolve_entry_audio(next(e for e in topo.entries if e.output_filename == name), inputs)

    edit.split_stereo(topo, tracks, "Keys.wav")
    np.testing.assert_array_equal(samples("Keys_L.wav"), left)
    np.testing.assert_array_equal(samples("Keys_R.wav"), right)
    edit.extract_channel(topo, tracks, "Keys.wav", 1)
    edit.new_output_file(topo, "Mix.wav", 3)
    edit.wire_channel(topo, "Mix.wav", 0, "Keys.wav", 0)
    edit.sum_channel(topo, "Mix.wav", 0, "Bass.wav", 0)
    edit.wire_channel(topo, "Mix.wav", 1, "Keys.wav", 0)
    edit.clear_channel(topo, "Mix.wav", 1)
    np.testing.assert_array_equal(samples("Mix.wav")[:, 1], np.zeros(4))
    edit.wire_channel(topo, "Mix.wav", 1, "Keys.wav", 1)
    edit.reorder_channel(topo, "Mix.wav", 1, 0)
    np.testing.assert_array_equal(samples("Mix.wav"),
                                  np.column_stack((right, [0.25, 0, 0.125, 0], np.zeros(4))))
    np.testing.assert_array_equal(samples("Keys_R.wav"), right)
    np.testing.assert_array_equal(samples("Bass.wav"), bass)


@pytest.mark.parametrize(("source_ch", "target_ch"), [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_malformed_saved_route_cannot_select_a_different_channel_or_render_silence(source_ch, target_ch):
    entry = TopologyEntry("Mix.wav", 1, [
        TopologySource("Input.wav", [ChannelRoute(source_ch, target_ch)]),
    ])
    with pytest.raises(ValueError, match="channel"):
        resolve_entry_audio(entry, {"Input.wav": (np.array([0.5, -0.5]), 48000)})


@pytest.mark.parametrize(("rate", "subtype", "amplitude", "accepted"), [
    (44100, "PCM_16", 0.25, False),
    (48000, "PCM_16", 0.75, False),
    (48000, "PCM_16", 0.25, True),
    (48000, "FLOAT", 0.75, True),
])
def test_apply_validates_rate_and_encoding_before_publication(tmp_path, rate, subtype, amplitude, accepted):
    src, dest = tmp_path / "src", tmp_path / "layout"
    src.mkdir()
    pulse = np.array([amplitude, -amplitude])
    for name, sr in (("A.wav", 48000), ("B.wav", rate)):
        sf.write(src / name, pulse, sr, subtype=subtype)
    session = load_session(str(src), {})
    session.topology = build_default_topology([])
    edit.new_output_file(session.topology, "Mix.wav", 1)
    for name in ("A.wav", "B.wav"):
        edit.sum_channel(session.topology, "Mix.wav", 0, name, 0)
    errors = apply_layout(session, src, dest)
    if accepted:
        assert errors == []
        pcm, sr = sf.read(dest / "Mix.wav")
        assert sr == 48000
        np.testing.assert_array_equal(pcm, 2 * pulse)
    else:
        assert errors
        assert not (dest / "Mix.wav").exists()
        assert not session.output_tracks
