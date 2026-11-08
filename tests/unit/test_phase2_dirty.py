from __future__ import annotations

import copy
from types import SimpleNamespace

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
    window._session = SimpleNamespace(
        tracks=[],
        processors=[],
        transfer_manifest=[],
        output_tracks=[],
        daw_state={},
        prepare_state="none",
        config={},
    )
    window._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
    window._detail_tabs.setTabEnabled(_TAB_SESSION, True)
    window._session_groups = copy.deepcopy(
        window._selected_group_preset_groups())
    window._populate_groups_tab()
    window._session_config = copy.deepcopy(window._active_preset())
    window._load_session_widgets(window._session_config)
    window._refresh_phase2_dirty_indicators()
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
    return window._session_widgets[section][0][1]


def _first_config_section(window: SessionPrepWindow, prefix: str) -> str:
    return next(
        key for key in window._session_widgets if key.startswith(prefix))


def test_phase2_dirty_indicators_are_clean_on_preset_values(qapp):
    window = _window(qapp)
    try:
        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window._detail_tabs, _TAB_SESSION)
        for item in window._session_dirty_items.values():
            assert item.icon(1).isNull()
    finally:
        window.close()


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
        window._session = SimpleNamespace(
            tracks=[],
            processors=[],
            transfer_manifest=[],
            output_tracks=[],
            daw_state={},
            prepare_state="none",
            config={},
        )
        window._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        window._session_groups = [
            {"name": "Loops", "color": "Cafe Royale Light"},
            {"name": "A.Gtr", "color": "Lima Dark"},
            {"name": "VOX", "color": "Dodger Blue Dark"},
            {"name": "Unknown", "color": "Not A Color"},
        ]
        window._populate_groups_tab()

        assert window._groups_tab_table.cellWidget(0, 1).currentColor() == (
            "Cafe Royale")
        assert window._groups_tab_table.cellWidget(1, 1).currentColor() == (
            "Lima")
        assert window._groups_tab_table.cellWidget(2, 1).currentColor() == (
            "Dodger Blue")
        assert window._groups_tab_table.cellWidget(3, 1).currentColor() == (
            window._config["colors"][0]["name"])
        assert set(window._group_color_map()) == {
            "Loops", "A.Gtr", "VOX", "Unknown"}
    finally:
        window.close()


def test_phase2_groups_tab_stays_clean_before_first_analysis_groups_load(qapp):
    window = SessionPrepWindow()
    try:
        window._session = SimpleNamespace(
            tracks=[],
            processors=[],
            transfer_manifest=[],
            output_tracks=[],
            daw_state={},
            prepare_state="none",
            config={},
        )
        window._session_groups = []
        window._detail_tabs.setTabEnabled(_TAB_GROUPS, False)

        window._refresh_phase2_dirty_indicators()

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window._detail_tabs, _TAB_GROUPS)
    finally:
        window.close()


def test_phase2_groups_dirty_highlights_cell_and_tab(qapp):
    window = _window(qapp)
    try:
        item = window._groups_tab_table.item(0, 0)
        item.setText(item.text() + " Edited")

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert item.background().color() == window._GROUP_DIRTY_BG
        assert window._detail_tabs.tabToolTip(_TAB_GROUPS) == (
            f"Changed from group preset “{window._active_session_preset}”")

        window._on_groups_tab_reset()

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert not is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert (
            window._groups_tab_table.item(0, 0).background().style()
            == Qt.NoBrush
        )
    finally:
        window.close()


def test_phase2_groups_dirty_color_uses_centered_button_indicator(qapp):
    window = _window(qapp)
    try:
        widget = window._groups_tab_table.cellWidget(0, 1)
        original = widget.currentColor()
        alternate = next(
            color["name"]
            for color in window._config["colors"]
            if color["name"] != original
        )

        widget.setCurrentColor(alternate)
        widget.colorChanged.emit(alternate)

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert widget.property("sessionDirty") is True
        assert widget._dirty_indicator is True
        assert "\u25cf" not in widget.text()
    finally:
        window.close()


def test_phase2_groups_added_row_marks_full_row(qapp):
    window = _window(qapp)
    try:
        window._on_groups_tab_add()
        row = window._groups_tab_table.rowCount() - 1

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert window._groups_dirty_cells()[row] == set(range(6))
    finally:
        window.close()


def test_phase2_groups_removed_row_marks_tab(qapp):
    window = _window(qapp)
    try:
        window._groups_tab_table.selectRow(0)
        window._on_groups_tab_remove()

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
        assert is_tab_marked(window._detail_tabs, _TAB_GROUPS)
        assert window._groups_dirty_cells() == {}
    finally:
        window.close()


def test_phase2_config_analysis_dirty_marks_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        restore = _set_different_value(
            _first_config_widget(window, "analysis"))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window._detail_tabs, _TAB_SESSION)
        assert not window._session_dirty_items["analysis"].icon(1).isNull()
        expected = f"Changed from preset “{window._active_config_preset_name}”"
        assert window._detail_tabs.tabToolTip(_TAB_SESSION) == expected
        assert window._session_dirty_items["analysis"].toolTip(1) == expected
        bar = window._detail_tabs.tabBar()
        if hasattr(bar, "accessibleTabName"):
            assert bar.accessibleTabName(_TAB_SESSION) == f"Config, {expected}"

        restore()

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window._detail_tabs, _TAB_SESSION)
        assert window._session_dirty_items["analysis"].icon(1).isNull()
        assert window._detail_tabs.tabToolTip(_TAB_SESSION) == ""
        if hasattr(bar, "accessibleTabName"):
            assert bar.accessibleTabName(_TAB_SESSION) == "Config"
    finally:
        window.close()


def test_phase2_config_child_dirty_marks_parent(qapp):
    window = _window(qapp)
    try:
        section = _first_config_section(window, "detectors.")
        _set_different_value(_first_config_widget(window, section))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window._detail_tabs, _TAB_SESSION)
        assert not window._session_dirty_items[section].icon(1).isNull()
        assert not window._session_dirty_items["_presentation"].icon(1).isNull()
    finally:
        window.close()


def test_phase2_config_revert_clears_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        _set_different_value(_first_config_widget(window, "analysis"))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert is_tab_marked(window._detail_tabs, _TAB_SESSION)

        window._on_session_config_reset()

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert not is_tab_marked(window._detail_tabs, _TAB_SESSION)
        assert window._session_dirty_items["analysis"].icon(1).isNull()
    finally:
        window.close()


def test_phase2_change_tooltips_name_their_own_comparison_preset(qapp):
    window = _window(qapp)
    try:
        window._active_session_preset = "Band groups"
        window._active_config_preset_name = "Mix config"
        assert window._phase2_change_tooltip(_TAB_GROUPS) == "Changed from group preset “Band groups”"
        assert window._phase2_change_tooltip(_TAB_SESSION) == "Changed from preset “Mix config”"
    finally:
        window.close()
