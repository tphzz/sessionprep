from __future__ import annotations

import copy

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QVBoxLayout,
)

from qt_helpers import dispose_window
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.prefs.config_pages import DawProjectTemplatesWidget
from sessionprepgui.prefs.param_form import _build_param_page
from sessionprepgui.prefs.preset_panel import NamedPresetPanel
from sessionprepgui.settings import build_defaults
from sessionprepgui.change_marker import is_button_marked, is_tab_marked
from sessionprepgui.theme import apply_dark_theme
from sessionpreplib.models import ParamSpec


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


@pytest.mark.parametrize("point_size", [9, 15])
def test_generated_reset_faces_follow_existing_field_paint_and_hit_target(qapp, form_font, point_size):
    qapp.setStyle("Fusion")
    font = qapp.font()
    font.setPointSize(point_size)
    qapp.setFont(font)
    dialog = QDialog()
    specs = [
        ParamSpec("integer", int, 6, "Integer", min=0, max=10),
        ParamSpec("decimal", (int, float), 1.15, "Decimal"),
        ParamSpec("choice", str, "avg", "Choice", choices=["avg", "sum"]),
        ParamSpec("text", str, "contents", "Text"),
        ParamSpec("path", str, "", "Temporary project directory",
                  widget_hint="path_picker_folder"),
        ParamSpec("enabled", bool, True, "Enabled"),
    ]
    page, editors = _build_param_page(specs, {})
    QVBoxLayout(dialog).addWidget(page)
    apply_dark_theme(dialog)
    dialog.resize(760, 640)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()

    def frame_edges(widget):
        image = dialog.grab().toImage()
        scale = image.devicePixelRatio()
        origin = widget.mapTo(dialog, QPoint())
        x = round((origin.x() + widget.width() * 0.70) * scale)
        rows = [
            y
            for y in range(round(origin.y() * scale),
                           round((origin.y() + widget.height()) * scale))
            if image.pixelColor(x, y).name() != "#1e1e1e"
        ]
        assert rows
        return min(rows), max(rows) + 1

    try:
        choice_height = dict(editors)["choice"].height()
        for spec, (_, editor) in zip(specs, editors):
            reset = next(
                b for b in dialog.findChildren(QPushButton)
                if b.accessibleName() == f"Reset {spec.label} to default"
            )
            target_height = 32 if spec.key == "enabled" else max(32, choice_height)
            assert (reset.width(), reset.height()) == (32, target_height)
            companion = editor
            if spec.key == "path":
                companion = next(b for b in editor.findChildren(QPushButton)
                                 if b.text().startswith("Browse"))
                assert companion.width() >= max(80, companion.sizeHint().width())
                assert editor.findChild(QLineEdit).height() == choice_height
            if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                assert editor.height() == choice_height + 2
            elif spec.key != "enabled":
                assert companion.height() == choice_height
            assert companion.height() == companion.sizeHint().height()
            top, bottom = frame_edges(reset)
            if spec.key == "enabled":
                option = QStyleOptionButton()
                editor.initStyleOption(option)
                indicator = editor.style().subElementRect(
                    QStyle.SE_CheckBoxIndicator, option, editor
                )
                scale = dialog.devicePixelRatioF()
                center = editor.mapTo(dialog, QPoint()).y() + indicator.y() + indicator.height() / 2
                assert abs(top - (center - 12) * scale) <= 1
                assert abs(bottom - (center + 12) * scale) <= 1
            else:
                assert all(abs(a - b) <= 1 for a, b in zip((top, bottom), frame_edges(companion)))
        integer = dict(editors)["integer"]
        int_reset = next(b for b in dialog.findChildren(QPushButton)
                         if b.accessibleName() == "Reset Integer to default")
        integer.setValue(8)
        QTest.mouseClick(int_reset, Qt.LeftButton, pos=QPoint(16, 1))
        assert integer.value() == 6
        integer.setValue(8)
        int_reset.setFocus(Qt.TabFocusReason)
        QTest.keyClick(int_reset, Qt.Key_Space)
        assert integer.value() == 6 and int_reset.hasFocus()

        dialog.resize(dialog.minimumSizeHint().width(), dialog.height())
        qapp.processEvents()
        browse = next(b for b in dict(editors)["path"].findChildren(QPushButton)
                      if b.text().startswith("Browse"))
        assert browse.width() >= browse.sizeHint().width()
    finally:
        dispose_window(dialog)


def test_template_form_row_fits_editors_and_round_trips_changes(qapp, form_font, monkeypatch):
    templates = DawProjectTemplatesWidget()
    apply_dark_theme(templates)
    original = {
        "name": "Mix Template",
        "template_path": "C:/Fixture/Template.dawproject",
        "fader_ceiling_db": 6.3,
    }
    templates.set_templates([original])
    templates.resize(740, 230)
    templates.show()
    qapp.processEvents()

    try:
        assert templates.get_templates() == [original]
        table = templates._table
        path_cell = table.cellWidget(0, 1)
        path = path_cell.findChild(QLineEdit)
        browse = path_cell.findChild(QPushButton)
        ceiling = table.findChild(QDoubleSpinBox)
        assert browse.height() == path.height()
        assert ceiling.height() == path.height() + 2
        assert ceiling.height() == ceiling.sizeHint().height()
        assert abs(ceiling.mapTo(table, ceiling.rect().center()).y()
                   - path.mapTo(table, path.rect().center()).y()) <= 1

        changed = QSignalSpy(templates.templates_changed)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: ("", ""))
        browse.click()
        assert templates.get_templates() == [original] and changed.count() == 0
        ceiling.setValue(8.4)
        assert templates.get_templates() == [dict(original, fader_ceiling_db=8.4)]
        assert changed.count() == 1
    finally:
        dispose_window(templates)


def test_named_preset_actions_match_the_selector_height(qapp, form_font):
    panel = NamedPresetPanel(["Default", "Mix"], protected=frozenset({"Default"}))
    restore = panel.add_trailing_button("Restore Defaults")
    apply_dark_theme(panel)
    panel.resize(760, 44)
    panel.show()
    qapp.processEvents()

    try:
        assert not panel._rename_btn.isEnabled()
        assert not panel._delete_btn.isEnabled()
        assert all(button.height() == panel._combo.height()
                   for button in panel.findChildren(QPushButton))
        panel._combo.setCurrentText("Mix")
        assert panel._rename_btn.isEnabled()
        assert panel._delete_btn.isEnabled() and restore.isEnabled()
    finally:
        panel.close()


def test_preferences_dirty_indicators_are_clean_on_open(qapp):
    dlg = _dialog(qapp)
    try:
        assert dlg._save_btn.text() == "Save Preferences"
        assert not is_button_marked(dlg._save_btn)
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert not is_tab_marked(dlg._tabs, dlg._global_tab_index)
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert not is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._global_items["general"].icon(1).isNull()
        assert dlg._global_items["colors"].icon(1).isNull()
        assert dlg._global_items["groups"].icon(1).isNull()
        for item in dlg._preset_items.values():
            assert item.icon(1).isNull()
    finally:
        dispose_window(dlg)


def test_preferences_general_dirty_dot_tracks_changed_and_restored_value(qapp):
    dlg = _dialog(qapp)
    try:
        combo = _general_widget(dlg, "report_verbosity")
        assert isinstance(combo, QComboBox)
        original = combo.currentIndex()
        combo.setCurrentIndex(1 if original == 0 else 0)

        assert not dlg._global_items["general"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert is_tab_marked(dlg._tabs, dlg._global_tab_index)
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert not is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)

        combo.setCurrentIndex(original)

        assert dlg._global_items["general"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert not is_tab_marked(dlg._tabs, dlg._global_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert not is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_colors_dirty_dot_tracks_palette_edit(qapp):
    dlg = _dialog(qapp)
    try:
        item = dlg._colors_page._table.item(0, 1)
        item.setText(item.text() + " Edited")

        assert not dlg._global_items["colors"].icon(1).isNull()
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_groups_dirty_dot_tracks_group_edit(qapp):
    dlg = _dialog(qapp)
    try:
        table = dlg._groups_page._groups_widget.table
        item = table.item(0, 0)
        item.setText(item.text() + " Edited")

        assert not dlg._global_items["groups"].icon(1).isNull()
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_config_analysis_dirty_dot_tracks_changed_and_restored_value(qapp):
    dlg = _dialog(qapp)
    try:
        restore = _set_different_value(_first_config_widget(dlg, "analysis"))

        assert not dlg._preset_items["analysis"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert not is_tab_marked(dlg._tabs, dlg._global_tab_index)
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)

        restore()

        assert dlg._preset_items["analysis"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert not is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert not is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_config_detector_dirty_dot_marks_child_and_parent(qapp):
    dlg = _dialog(qapp)
    try:
        section = _first_config_section(dlg, "detectors.")
        _set_different_value(_first_config_widget(dlg, section))

        assert not dlg._preset_items[section].icon(1).isNull()
        assert not dlg._preset_items["_presentation"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_config_protools_templates_mark_child_and_parent(qapp):
    dlg = _dialog(qapp)
    try:
        widget = dlg._cfg_daw_custom_widgets["protools"]
        widget._on_add()
        widget._table.item(widget._table.rowCount() - 1, 0).setText("Mix")

        assert not dlg._preset_items["daw_processors.protools"].icon(1).isNull()
        assert not dlg._preset_items["_daw_processors"].icon(1).isNull()
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)


def test_preferences_config_preset_list_changes_mark_config_tab(qapp):
    dlg = _dialog(qapp)
    try:
        dlg._on_cfg_added("Dirty Test")

        assert dlg._tabs.tabText(dlg._global_tab_index) == "Global"
        assert not is_tab_marked(dlg._tabs, dlg._global_tab_index)
        assert dlg._tabs.tabText(dlg._preset_tab_index) == "Config Presets"
        assert is_tab_marked(dlg._tabs, dlg._preset_tab_index)
        assert dlg._save_btn.text() == "Save Preferences"
        assert is_button_marked(dlg._save_btn)
    finally:
        dispose_window(dlg)
