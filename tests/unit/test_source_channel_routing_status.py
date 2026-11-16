"""Track Layout cues for source channels with no output routes."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sessionprepgui import mainwindow, settings
from sessionpreplib.models import DetectorResult, SessionContext, Severity, TrackContext
from sessionpreplib.topology import build_default_topology


PIANO = "07_Piano with a long take name.wav"
BASS = "08_Bass.wav"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def layout_window(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *_: None)
    window = mainwindow.SessionPrepWindow()
    tracks = [
        TrackContext(name, str(tmp_path / name), None, 44100, channels,
                     88200, "24-bit", "PCM_24", 2.0)
        for name, channels in [(PIANO, 2), (BASS, 1)]
    ]
    tracks[0].detector_results["clipping"] = DetectorResult(
        "clipping", Severity.PROBLEM, "clipping detected (1 clipped range)", {})
    window._topo_source_tracks = tracks
    window._topo_topology = build_default_topology(tracks)
    window._session = SessionContext(tracks, window._flat_config())
    window._populate_topology_tab()
    window.resize(1320, 680)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    try:
        yield window
    finally:
        window.close()


def _output_item(window, filename, channel=None):
    tree = window._topo_output_tree
    item = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                if tree.topLevelItem(i).data(0, Qt.UserRole) == ("file", filename))
    return item if channel is None else item.child(channel)


def _delete_output(window, filename, channel=None):
    tree = window._topo_output_tree
    tree.setCurrentItem(_output_item(window, filename, channel))
    tree.setFocus()
    QTest.keyClick(tree, Qt.Key_Delete)
    QApplication.processEvents()


def _drop_piano_right(window, target):
    inputs = window._topo_input_tree
    outputs = window._topo_output_tree
    source = inputs.find_item(PIANO).child(1)
    mime = inputs.mimeData([source])
    pos = outputs.visualItemRect(_output_item(window, target)).center()
    drop = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    outputs.dropEvent(drop)
    assert drop.isAccepted()
    QApplication.processEvents()


def _cell_colors(tree, item, column=0):
    viewport = tree.viewport()
    rect = tree.visualRect(tree.indexFromItem(item, column)).intersected(viewport.rect())
    if rect.isEmpty():
        return set()
    image = viewport.grab(rect).toImage()
    return {image.pixelColor(x, y).getRgb()[:3]
            for y in range(image.height()) for x in range(image.width())}


def _has_green(colors):
    return any(g > r + 32 and g > b + 32 for r, g, b in colors)


def test_release_and_reassignment_follow_the_last_route(layout_window):
    window = layout_window
    inputs = window._topo_input_tree
    assert inputs.find_item(PIANO).child(1).foreground(0).color().name() == "#aaaaaa"

    _drop_piano_right(window, BASS)
    _delete_output(window, PIANO, 1)
    still_assigned = inputs.find_item(PIANO).child(1)
    assert still_assigned.foreground(0).color().name() == "#aaaaaa"
    assert not still_assigned.data(0, Qt.AccessibleTextRole)

    _delete_output(window, BASS, 1)
    piano = inputs.find_item(PIANO)
    available = piano.child(1)
    assert available.foreground(0).color().name() == "#e0e0e0"
    assert available.data(0, Qt.AccessibleTextRole).endswith("Unassigned")
    assert available.data(0, Qt.UserRole) == ("channel", PIANO, 1)
    assert piano.foreground(0).color().name() == "#ffaa00"
    assert piano.child(0).foreground(0).color().name() == "#aaaaaa"
    assert inputs.find_item(BASS).child(0).foreground(0).color().name() == "#aaaaaa"
    assert _output_item(window, PIANO).foreground(0).color().name() == "#cccccc"
    assert _output_item(window, PIANO, 0).foreground(0).color().name() == "#44cc44"

    _drop_piano_right(window, PIANO)
    routed = inputs.find_item(PIANO).child(1)
    assert routed.foreground(0).color().name() == "#aaaaaa"
    assert not routed.data(0, Qt.AccessibleTextRole)
    assert "Unassigned" not in routed.toolTip(0)


def test_collapsed_count_and_warning_paint_together(form_font, layout_window):
    inputs = layout_window._topo_input_tree
    inputs.collapseAll()
    _delete_output(layout_window, PIANO)
    piano = inputs.find_item(PIANO)
    assert not piano.isExpanded()
    assert piano.data(0, Qt.AccessibleTextRole).endswith("2 unassigned")
    assert "clipping detected" in piano.toolTip(0)
    assert piano.foreground(0).color().name() == "#ffaa00"
    assert _has_green(_cell_colors(inputs, piano))

    piano.setExpanded(True)
    QApplication.processEvents()
    assert not _has_green(_cell_colors(inputs, piano))
    assert _has_green(_cell_colors(inputs, piano.child(0)))
    assert _has_green(_cell_colors(inputs, piano.child(1)))


def test_selected_and_narrow_rows_keep_the_unassigned_status(form_font, layout_window):
    _delete_output(layout_window, PIANO, 1)
    inputs = layout_window._topo_input_tree
    channel = inputs.find_item(PIANO).child(1)
    inputs.setColumnWidth(0, 190)
    QApplication.processEvents()
    colors = _cell_colors(inputs, channel)
    assert _has_green(colors)
    assert any(r == g == b and r > 214 for r, g, b in colors)
    assert not _has_green(_cell_colors(inputs, channel, 1))

    inputs.setCurrentItem(channel)
    inputs.setFocus()
    QApplication.processEvents()
    selected = _cell_colors(inputs, channel)
    assert (42, 109, 181) in selected
    assert any(r == g == b and r > 247 for r, g, b in selected)
    assert not _has_green(selected)

    inputs.clearSelection()
    inputs.setColumnWidth(0, 52)
    QApplication.processEvents()
    assert not _has_green(_cell_colors(inputs, channel))
    assert not _has_green(_cell_colors(inputs, channel, 1))
    assert PIANO in channel.toolTip(1)
    assert "Unassigned" in channel.toolTip(1)
    assert channel.data(0, Qt.AccessibleTextRole).endswith("Unassigned")
