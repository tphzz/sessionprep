from __future__ import annotations

import copy

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)

from qt_helpers import dispose_window
from sessionpreplib.models import SessionContext
from sessionprepgui.settings import build_defaults
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.tracks.table_widgets import _TAB_GROUPS, _TAB_SESSION
from sessionprepgui.change_marker import is_tab_marked


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _window(qapp):
    window = SessionPrepWindow()
    window.state.session = SessionContext([], {})
    window.phase2_page.detail_tabs.setTabEnabled(_TAB_GROUPS, True)
    window.phase2_page.detail_tabs.setTabEnabled(_TAB_SESSION, True)
    window.state.session_groups = copy.deepcopy(
        window.groups_controller.selected_group_preset_groups())
    window.groups_controller.populate_groups_tab()
    window.config_controller.init_session_config()
    window.config_controller.refresh_phase2_dirty_indicators()
    return window


def _set_different_value(widget):
    if isinstance(widget, QComboBox):
        original = widget.currentIndex()
        widget.setCurrentIndex(1 if original == 0 else 0)
        return lambda: widget.setCurrentIndex(original)
    if isinstance(widget, QCheckBox):
        original = widget.isChecked()
        widget.setChecked(not original)
        return lambda: widget.setChecked(original)
    if isinstance(widget, QDoubleSpinBox):
        original = widget.value()
        new_value = original + widget.singleStep()
        if new_value > widget.maximum():
            new_value = original - widget.singleStep()
        widget.setValue(new_value)
        return lambda: widget.setValue(original)
    if isinstance(widget, QSpinBox):
        original = widget.value()
        new_value = original + 1
        if new_value > widget.maximum():
            new_value = original - 1
        widget.setValue(new_value)
        return lambda: widget.setValue(original)
    if hasattr(widget, "set_value"):
        original = widget.value()
        widget.set_value(f"{original}_edited")
        return lambda: widget.set_value(original)
    if isinstance(widget, QLineEdit):
        original = widget.text()
        widget.setText(f"{original}_edited")
        return lambda: widget.setText(original)
    raise AssertionError(f"Unsupported widget type: {type(widget)!r}")


def _first_config_widget(window: SessionPrepWindow, section: str):
    return window.config_controller.session_widgets[section][0][1]


def _first_config_section(window: SessionPrepWindow, prefix: str) -> str:
    return next(
        key for key in window.config_controller.session_widgets if key.startswith(prefix))


def test_phase2_dirty_indicators_are_clean_on_preset_values(qapp):
    window = _window(qapp)
    try:
        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
        for item in window.config_controller.session_dirty_items.values():
            assert item.icon(1).isNull()
    finally:
        dispose_window(window)


def test_default_group_preset_uses_existing_color_names(qapp):
    defaults = build_defaults()
    color_names = {color["name"] for color in defaults["colors"]}

    assert {
        group["name"]: group["color"]
        for group in defaults["group_presets"]["Default"]
        if group["name"] in {"Loops", "A.Gtr", "VOX"}
    } == {
        "Loops": "Cafe Royale",
        "A.Gtr": "Lima",
        "VOX": "Dodger Blue",
    }
    assert all(
        group["color"] in color_names
        for group in defaults["group_presets"]["Default"]
    )


def test_legacy_group_color_names_resolve_to_current_palette(qapp):
    window = SessionPrepWindow()
    try:
        window.state.session = SessionContext([], {})
        window.phase2_page.detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        window.state.session_groups = [
            {"name": "Loops", "color": "Cafe Royale Light"},
            {"name": "A.Gtr", "color": "Lima Dark"},
            {"name": "VOX", "color": "Dodger Blue Dark"},
            {"name": "Unknown", "color": "Not A Color"},
        ]
        window.groups_controller.populate_groups_tab()

        assert window.groups_controller.page.table.cellWidget(0, 1).currentColor() == (
            "Cafe Royale")
        assert window.groups_controller.page.table.cellWidget(1, 1).currentColor() == (
            "Lima")
        assert window.groups_controller.page.table.cellWidget(2, 1).currentColor() == (
            "Dodger Blue")
        assert window.groups_controller.page.table.cellWidget(3, 1).currentColor() == (
            window.state.config["colors"][0]["name"])
        assert set(window.groups_controller.group_color_map()) == {
            "Loops", "A.Gtr", "VOX", "Unknown"}
    finally:
        dispose_window(window)


def test_phase2_groups_tab_stays_clean_before_first_analysis_groups_load(qapp):
    window = SessionPrepWindow()
    try:
        window.state.session = SessionContext([], {})
        window.state.session_groups = []
        window.phase2_page.detail_tabs.setTabEnabled(_TAB_GROUPS, False)

        window.config_controller.refresh_phase2_dirty_indicators()

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
    finally:
        dispose_window(window)


def test_phase2_groups_dirty_highlights_cell_and_tab(qapp):
    window = _window(qapp)
    try:
        item = window.groups_controller.page.table.item(0, 0)
        item.setText(item.text() + " Edited")

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        from sessionprepgui.tracks.groups_controller import _GROUP_DIRTY_BG
        assert item.background().color() == _GROUP_DIRTY_BG
        assert window.phase2_page.detail_tabs.tabToolTip(_TAB_GROUPS) == (
            f"Changed from group preset “{window.state.active_session_preset}”")

        window.groups_controller.merge_groups_from_preset()

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        assert (
            window.groups_controller.page.table.item(0, 0).background().style()
            == Qt.NoBrush
        )
    finally:
        dispose_window(window)


def test_phase2_groups_dirty_color_uses_centered_button_indicator(qapp):
    window = _window(qapp)
    try:
        widget = window.groups_controller.page.table.cellWidget(0, 1)
        original = widget.currentColor()
        alternate = next(
            color["name"]
            for color in window.state.config["colors"]
            if color["name"] != original
        )

        widget.setCurrentColor(alternate)
        widget.colorChanged.emit(alternate)

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        assert widget.property("sessionDirty") is True
        assert widget._dirty_indicator is True
        assert "\u25cf" not in widget.text()
    finally:
        dispose_window(window)


def test_phase2_groups_added_row_marks_full_row(qapp):
    window = _window(qapp)
    try:
        window.groups_controller.add_group_row()
        row = window.groups_controller.page.table.rowCount() - 1

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        assert window.groups_controller.groups_dirty_cells()[row] == set(range(6))
    finally:
        dispose_window(window)


def test_phase2_groups_removed_row_marks_tab(qapp):
    window = _window(qapp)
    try:
        window.groups_controller.page.table.selectRow(0)
        window.groups_controller.remove_selected_group_row()

        assert window.phase2_page.detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_GROUPS)
        assert window.groups_controller.groups_dirty_cells() == {}
    finally:
        dispose_window(window)


def test_phase2_config_analysis_dirty_marks_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        restore = _set_different_value(
            _first_config_widget(window, "analysis"))

        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
        assert not window.config_controller.session_dirty_items["analysis"].icon(1).isNull()
        expected = f"Changed from preset “{window.state.active_config_preset_name}”"
        assert window.phase2_page.detail_tabs.tabToolTip(_TAB_SESSION) == expected
        assert window.config_controller.session_dirty_items["analysis"].toolTip(1) == expected
        bar = window.phase2_page.detail_tabs.tabBar()
        if hasattr(bar, "accessibleTabName"):
            assert bar.accessibleTabName(_TAB_SESSION) == f"Config, {expected}"

        restore()

        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
        assert window.config_controller.session_dirty_items["analysis"].icon(1).isNull()
        assert window.phase2_page.detail_tabs.tabToolTip(_TAB_SESSION) == ""
        if hasattr(bar, "accessibleTabName"):
            assert bar.accessibleTabName(_TAB_SESSION) == "Config"
    finally:
        dispose_window(window)


def test_phase2_config_child_dirty_marks_parent(qapp):
    window = _window(qapp)
    try:
        section = _first_config_section(window, "detectors.")
        _set_different_value(_first_config_widget(window, section))

        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
        assert not window.config_controller.session_dirty_items[section].icon(1).isNull()
        assert not window.config_controller.session_dirty_items["_presentation"].icon(1).isNull()
    finally:
        dispose_window(window)


def test_phase2_config_revert_clears_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        _set_different_value(_first_config_widget(window, "analysis"))

        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)

        window.config_controller.on_session_config_reset()

        assert window.phase2_page.detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
        assert window.config_controller.session_dirty_items["analysis"].icon(1).isNull()
    finally:
        dispose_window(window)


def test_phase2_change_tooltips_name_their_own_comparison_preset(qapp):
    window = _window(qapp)
    try:
        window.state.active_session_preset = "Band groups"
        window.state.active_config_preset_name = "Mix config"
        assert window.config_controller.phase2_change_tooltip(_TAB_GROUPS) == "Changed from group preset “Band groups”"
        assert window.config_controller.phase2_change_tooltip(_TAB_SESSION) == "Changed from preset “Mix config”"
    finally:
        dispose_window(window)
