from __future__ import annotations

import copy

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)

from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.settings import build_defaults


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dialog(qapp):
    dlg = PreferencesDialog(copy.deepcopy(build_defaults()))
    return dlg


def _general_widget(dlg: PreferencesDialog, key: str):
    return next(widget for item_key, widget in dlg._general_page._widgets
                if item_key == key)


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


def _first_config_widget(dlg: PreferencesDialog, section: str):
    return dlg._cfg_widgets[section][0][1]


def _first_config_section(dlg: PreferencesDialog, prefix: str) -> str:
    return next(key for key in dlg._cfg_widgets if key.startswith(prefix))


def test_preferences_dirty_indicators_are_clean_on_open(qapp):
    dlg = _dialog(qapp)
    try:
        assert dlg._save_btn.text() == "Save Preferences"
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert dlg._global_items["general"].icon(1).isNull()
        assert dlg._global_items["colors"].icon(1).isNull()
        assert dlg._global_items["groups"].icon(1).isNull()
        for item in dlg._preset_items.values():
            assert item.icon(1).isNull()
    finally:
        dlg.close()


def test_preferences_general_dirty_dot_tracks_changed_and_restored_value(qapp):
    dlg = _dialog(qapp)
    try:
        combo = _general_widget(dlg, "report_verbosity")
        assert isinstance(combo, QComboBox)
        original = combo.currentIndex()
        combo.setCurrentIndex(1 if original == 0 else 0)

        assert not dlg._global_items["general"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global \u25cf"
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert dlg._save_btn.text() == "Save Preferences \u25cf"

        combo.setCurrentIndex(original)

        assert dlg._global_items["general"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert dlg._save_btn.text() == "Save Preferences"
    finally:
        dlg.close()


def test_preferences_colors_dirty_dot_tracks_palette_edit(qapp):
    dlg = _dialog(qapp)
    try:
        item = dlg._colors_page._table.item(0, 1)
        item.setText(item.text() + " Edited")

        assert not dlg._global_items["colors"].icon(1).isNull()
        assert dlg._save_btn.text() == "Save Preferences \u25cf"
    finally:
        dlg.close()


def test_preferences_groups_dirty_dot_tracks_group_edit(qapp):
    dlg = _dialog(qapp)
    try:
        table = dlg._groups_page._groups_widget.table
        item = table.item(0, 0)
        item.setText(item.text() + " Edited")

        assert not dlg._global_items["groups"].icon(1).isNull()
        assert dlg._save_btn.text() == "Save Preferences \u25cf"
    finally:
        dlg.close()


def test_preferences_config_analysis_dirty_dot_tracks_changed_and_restored_value(qapp):
    dlg = _dialog(qapp)
    try:
        restore = _set_different_value(_first_config_widget(dlg, "analysis"))

        assert not dlg._preset_items["analysis"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets \u25cf"
        assert dlg._save_btn.text() == "Save Preferences \u25cf"

        restore()

        assert dlg._preset_items["analysis"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert dlg._save_btn.text() == "Save Preferences"
    finally:
        dlg.close()


def test_preferences_config_detector_dirty_dot_marks_child_and_parent(qapp):
    dlg = _dialog(qapp)
    try:
        section = _first_config_section(dlg, "detectors.")
        _set_different_value(_first_config_widget(dlg, section))

        assert not dlg._preset_items[section].icon(1).isNull()
        assert not dlg._preset_items["_presentation"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets \u25cf"
        assert dlg._save_btn.text() == "Save Preferences \u25cf"
    finally:
        dlg.close()


def test_preferences_config_protools_templates_mark_child_and_parent(qapp):
    dlg = _dialog(qapp)
    try:
        widget = dlg._cfg_daw_custom_widgets["protools"]
        widget._on_add()
        widget._table.item(widget._table.rowCount() - 1, 0).setText("Mix")

        assert not dlg._preset_items["daw_processors.protools"].icon(1).isNull()
        assert not dlg._preset_items["_daw_processors"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets \u25cf"
        assert dlg._save_btn.text() == "Save Preferences \u25cf"
    finally:
        dlg.close()


def test_preferences_config_preset_list_changes_mark_config_tab(qapp):
    dlg = _dialog(qapp)
    try:
        dlg._on_cfg_added("Dirty Test")

        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets \u25cf"
        assert dlg._save_btn.text() == "Save Preferences \u25cf"
    finally:
        dlg.close()
