"""Tool-window and Preferences content insets."""
import copy

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QTableWidget

from qt_helpers import dispose_window
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


def test_window_margin_is_six_logical_pixels():
    assert WINDOW_MARGIN == 6


def test_log_viewer_content_sits_six_pixels_from_every_edge(qapp):
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


def test_detached_log_viewer_uses_shared_form_control_sizes(qapp, form_font):
    viewer = LogViewerWindow()
    viewer.show()
    try:
        qapp.processEvents()
        height = viewer._level_combo.height()
        assert viewer._line_count.height() == height + 2  # Fusion frame inset
        assert all(button.height() == height for button in (
            viewer._copy_all_btn, viewer._copy_btn, viewer._clear_btn,
            viewer._open_folder_btn,
        ))
    finally:
        viewer.close()


def test_preferences_tabs_are_flush_with_the_client_area(qapp):
    dialog = PreferencesDialog(copy.deepcopy(settings.build_defaults()))
    dialog.show()
    try:
        for index, tree, form in (
            (0, dialog._global_tree, dialog._global_stack),
            (1, dialog._preset_tree, dialog._preset_stack),
        ):
            dialog._tabs.setCurrentIndex(index)
            qapp.processEvents()
            tabs = dialog._tabs.tabBar()
            assert tabs.mapTo(dialog, QPoint()) == QPoint()
            assert tabs.width() == dialog.width()
            assert tree.mapTo(dialog, QPoint()).x() == WINDOW_MARGIN
            assert form.mapTo(dialog, QPoint(form.width(), 0)).x() == dialog.width() - WINDOW_MARGIN
        button_box = dialog.findChild(QDialogButtonBox)
        right = max(b.mapTo(dialog, QPoint(b.width(), 0)).x() for b in button_box.buttons())
        bottom = max(b.mapTo(dialog, QPoint(0, b.height())).y() for b in button_box.buttons())
        assert right == dialog.width() - WINDOW_MARGIN
        assert bottom == dialog.height() - WINDOW_MARGIN
    finally:
        dispose_window(dialog)


@pytest.fixture
def protools(qapp, monkeypatch):
    monkeypatch.setattr(ProToolsUtilsWindow, "_schedule_start_connect", lambda *_: None)
    window = ProToolsUtilsWindow(settings.build_defaults())
    window.resize(1600, 720)
    window.show()
    qapp.processEvents()
    yield window
    dispose_window(window)


def test_protools_tool_pages_share_the_window_inset(protools):
    assert _margins(protools) == (WINDOW_MARGIN,) * 4
    for tool in (protools._color_tool, protools._track_height_tool):
        assert _margins(tool) == (0, WINDOW_MARGIN, 0, 0)
    # The colour grid sits on the same bounds as the tool's other controls.
    assert _margins(protools._color_tool._grid) == (0, 0, 0, 0)


def test_preferences_colour_preview_grid_keeps_its_own_margin(qapp):
    page = ColorsPage()
    try:
        assert _margins(page._grid_preview) == (4, 4, 4, 4)  # only the Color Picker changed
    finally:
        dispose_window(page)


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
