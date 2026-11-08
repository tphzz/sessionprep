"""Tool windows share a 4 px window margin; tool pages pad 4 px; empty status lines hide."""
import copy

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QTableWidget

from sessionprepgui import settings
from sessionprepgui.daw_tools.protools.window import ProToolsUtilsWindow
from sessionprepgui.log_viewer import LogViewerWindow
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.prefs.page_colors import ColorsPage
from sessionprepgui.theme import WINDOW_MARGIN


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def _margins(widget):
    m = widget.layout().contentsMargins()
    return m.left(), m.top(), m.right(), m.bottom()


def test_window_margin_is_four_logical_pixels():
    assert WINDOW_MARGIN == 4


def test_log_viewer_content_sits_four_pixels_from_every_edge(qapp):
    viewer = LogViewerWindow()
    viewer.resize(1000, 650)
    viewer.show()
    qapp.processEvents()
    assert _margins(viewer) == (WINDOW_MARGIN,) * 4
    table = viewer.findChild(QTableWidget)
    top_left = table.mapTo(viewer, QPoint(0, 0))
    assert top_left.x() == WINDOW_MARGIN
    assert top_left.x() + table.width() == viewer.width() - WINDOW_MARGIN
    viewer.close()


def test_preferences_window_margin(qapp):
    dialog = PreferencesDialog(copy.deepcopy(settings.build_defaults()))
    assert _margins(dialog) == (WINDOW_MARGIN,) * 4
    dialog.close()


@pytest.fixture
def protools(qapp):
    window = ProToolsUtilsWindow(settings.build_defaults())
    window.resize(1600, 720)
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def test_protools_window_and_tool_pages_pad_four_pixels(protools):
    assert _margins(protools) == (WINDOW_MARGIN,) * 4
    for tool in (protools._color_tool, protools._track_height_tool):
        assert _margins(tool) == (WINDOW_MARGIN,) * 4  # not 8 + 8
    # The colour grid adds no margin of its own inside the padded page...
    assert _margins(protools._color_tool._grid) == (0, 0, 0, 0)


def test_preferences_colour_preview_grid_keeps_its_own_margin(qapp):
    page = ColorsPage()
    assert _margins(page._grid_preview) == (4, 4, 4, 4)  # only the Color Picker changed


@pytest.mark.parametrize("tool_name", ["_color_tool", "_track_height_tool"])
def test_status_line_shows_only_while_it_has_a_message(protools, tool_name):
    tool = getattr(protools, tool_name)
    assert tool._status.isHidden()
    tool._set_status("Loading...", "#aaa")
    assert not tool._status.isHidden() and tool._status.text() == "Loading..."
    tool._set_status("", "#aaa")
    assert tool._status.isHidden()


def test_status_messages_never_change_the_preferred_window_height(protools):
    color, heights = protools._color_tool, protools._track_height_tool
    width = 1500
    color_empty = color.preferred_compact_height_for_width(width)
    heights_empty = heights.preferred_expanded_height_for_width(width)
    color._set_status("Not connected to Pro Tools", "#f44336")
    heights._set_status("Loading preview...", "#aaa")
    assert color.preferred_compact_height_for_width(width) == color_empty
    assert heights.preferred_expanded_height_for_width(width) == heights_empty
    color._set_status("", "#aaa")
    heights._set_status("", "#aaa")
    assert color.preferred_compact_height_for_width(width) == color_empty
    assert heights.preferred_expanded_height_for_width(width) == heights_empty
