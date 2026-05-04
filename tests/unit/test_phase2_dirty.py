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

from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.tracks.table_widgets import _TAB_GROUPS, _TAB_SESSION


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
        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        for item in window._session_dirty_items.values():
            assert item.icon(1).isNull()
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
    finally:
        window.close()


def test_phase2_groups_dirty_highlights_cell_and_tab(qapp):
    window = _window(qapp)
    try:
        item = window._groups_tab_table.item(0, 0)
        item.setText(item.text() + " Edited")

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups \u25cf"
        assert item.background().color() == window._GROUP_DIRTY_BG

        window._on_groups_tab_reset()

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups"
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

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups \u25cf"
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

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups \u25cf"
        assert window._groups_dirty_cells()[row] == set(range(6))
    finally:
        window.close()


def test_phase2_groups_removed_row_marks_tab(qapp):
    window = _window(qapp)
    try:
        window._groups_tab_table.selectRow(0)
        window._on_groups_tab_remove()

        assert window._detail_tabs.tabText(_TAB_GROUPS) == "Groups \u25cf"
        assert window._groups_dirty_cells() == {}
    finally:
        window.close()


def test_phase2_config_analysis_dirty_marks_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        restore = _set_different_value(
            _first_config_widget(window, "analysis"))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config \u25cf"
        assert not window._session_dirty_items["analysis"].icon(1).isNull()

        restore()

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert window._session_dirty_items["analysis"].icon(1).isNull()
    finally:
        window.close()


def test_phase2_config_child_dirty_marks_parent(qapp):
    window = _window(qapp)
    try:
        section = _first_config_section(window, "detectors.")
        _set_different_value(_first_config_widget(window, section))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config \u25cf"
        assert not window._session_dirty_items[section].icon(1).isNull()
        assert not window._session_dirty_items["_presentation"].icon(1).isNull()
    finally:
        window.close()


def test_phase2_config_revert_clears_tree_and_tab(qapp):
    window = _window(qapp)
    try:
        _set_different_value(_first_config_widget(window, "analysis"))

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config \u25cf"

        window._on_session_config_reset()

        assert window._detail_tabs.tabText(_TAB_SESSION) == "Config"
        assert window._session_dirty_items["analysis"].icon(1).isNull()
    finally:
        window.close()
