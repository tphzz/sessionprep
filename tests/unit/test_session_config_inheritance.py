from __future__ import annotations

import copy
from html import unescape

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
)

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.change_marker import is_tab_marked
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.session.io import load_session
from sessionprepgui.tracks.table_widgets import _TAB_SESSION
from sessionpreplib.daw_processors import protools
from sessionpreplib.models import TrackContext, TransferEntry


PRO_TOOLS = "daw_processors.protools"
DAWPROJECT = "daw_processors.dawproject"
NAMED_PRESET = "Studio Mix"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def app_config(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    config = settings.build_defaults()
    preset = config["config_presets"]["Default"]
    preset["daw_processors"]["protools"].update(
        protools_project_dir=str(tmp_path / "protools-default"),
        protools_temp_dir=str(tmp_path / "protools-temp-default"),
        protools_port=31416,
        protools_templates=[{"group": "Mix", "name": "Pro Tools Template"}],
    )
    preset["daw_processors"]["dawproject"].update(
        dawproject_project_dir=str(tmp_path / "dawproject-default"),
        dawproject_templates=[{
            "name": "DAWproject Template",
            "template_path": str(tmp_path / "template.dawproject"),
            "fader_ceiling_db": 6.0,
        }],
    )
    named_preset = copy.deepcopy(preset)
    named_preset["daw_processors"]["protools"].update(
        protools_project_dir=str(tmp_path / "protools-studio"),
        protools_temp_dir=str(tmp_path / "protools-temp-studio"),
        protools_port=32416,
    )
    config["config_presets"][NAMED_PRESET] = named_preset
    settings.save_config(config)
    return config


@pytest.fixture
def window(qapp, app_config, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: QMessageBox.Ok)

    def unexpected_dialog(_parent, title, message, *_args):
        pytest.fail(f"{title}: {message}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_dialog)
    monkeypatch.setattr(QMessageBox, "critical", unexpected_dialog)

    win = SessionPrepWindow()
    try:
        yield win
    finally:
        dispose_window(win)


def _open_project(window, config, tmp_path, preset="Default", *, tracks=()):
    source = tmp_path / "audio"
    source.mkdir()
    window.persistence_controller.restore_session_state({
        "source_dir": str(source),
        "active_config_preset": preset,
        "session_config": copy.deepcopy(config["config_presets"][preset]),
        "session_groups": config["group_presets"]["Default"],
        "tracks": list(tracks),
        "prepare_state": "ready",
        "transfer_manifest": [TransferEntry("track-1", "Kick.wav", "Kick")],
        "daw_state": {"protools_0": {"assignments": {"track-1": "drums"}}},
    })


@pytest.fixture
def project(window, app_config, tmp_path):
    _open_project(window, app_config, tmp_path)
    return window


@pytest.fixture
def phase_work(window, monkeypatch):
    requests = []
    window.config_controller.reanalyze_phase1_requested.disconnect(
        window.persistence_controller.reanalyze_topology
    )
    window.config_controller.reanalyze_phase1_requested.connect(lambda: requests.append("layout"))
    window.config_controller.reanalyze_phase2_requested.disconnect(window.analysis_controller.analyze)
    window.config_controller.reanalyze_phase2_requested.connect(lambda: requests.append("analysis"))
    monkeypatch.setattr(window.persistence_controller, "reanalyze_topology", lambda: requests.append("layout"))
    monkeypatch.setattr(window.analysis_controller, "analyze", lambda: requests.append("analysis"))
    monkeypatch.setattr(window.analysis_controller, "prepare", lambda: requests.append("prepare"))
    return requests


def _field(widgets, section, key):
    return dict(widgets[section])[key]


def _reset_button(container, label):
    return next(
        button for button in container.findChildren(QPushButton)
        if button.accessibleName() in {
            f"Reset {label} to default", f"Use preset value for {label}",
        }
    )


def _save_preferences(window, monkeypatch, edit):
    def edit_dialog(dialog):
        edit(dialog)
        dialog._save_btn.click()
        assert dialog.saved
        return dialog.result()

    with monkeypatch.context() as modal:
        modal.setattr(PreferencesDialog, "exec", edit_dialog)
        window._on_preferences()


def _select_daw(window, processor_id):
    row = next(
        row for row in range(window.daw_controller.page.daw_combo.count())
        if window.state.daw_processors[window.daw_controller.page.daw_combo.itemData(row)].id == processor_id
    )
    window.daw_controller.page.daw_combo.setCurrentIndex(row)
    assert window.state.active_daw_processor.id == processor_id


def test_cancelled_preferences_keeps_project_fields_and_saved_defaults(
    project, tmp_path, monkeypatch, phase_work,
):
    original = project.state.active_daw_processor.project_dir
    before = settings.load_config()

    def edit_and_cancel(dialog):
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(
            str(tmp_path / "cancelled"))
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(PreferencesDialog, "exec", edit_and_cancel)
    project._on_preferences()

    assert project.state.active_daw_processor.project_dir == original
    assert settings.load_config() == before
    assert phase_work == []


def test_declining_preset_switch_keeps_project_selection_and_overrides(
    project, tmp_path, monkeypatch, phase_work,
):
    custom = str(tmp_path / "project-only")
    _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir").set_value(custom)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)

    project.config_controller.config_preset_combo.setCurrentText(NAMED_PRESET)

    assert project.config_controller.config_preset_combo.currentText() == "Default"
    assert project.state.active_daw_processor.project_dir == custom
    assert project.state.flat_config()["protools_port"] == 31416
    assert phase_work == []


def test_processor_enable_override_updates_processing_controls_without_reanalysis(
    project, phase_work,
):
    switch = _field(
        project.config_controller.session_widgets, "processors.bimodal_normalize", "bimodal_normalize_enabled")
    before = [processor.id for processor in project.state.session.processors]
    assert "bimodal_normalize" in before

    switch.setChecked(False)
    assert "bimodal_normalize" not in [processor.id for processor in project.state.session.processors]
    switch.setChecked(True)
    assert [processor.id for processor in project.state.session.processors] == before
    assert project.state.session.prepare_state == "stale"
    assert phase_work == []


def test_open_project_keeps_its_preset_when_preferences_saves_another_default(
    project, tmp_path, monkeypatch,
):
    new_default_dir = str(tmp_path / "default-updated")

    def edit(dialog):
        path = _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir")
        path.set_value(new_default_dir)
        dialog._cfg_panel._combo.setCurrentText(NAMED_PRESET)

    _save_preferences(project, monkeypatch, edit)

    assert settings.load_config()["app"]["active_config_preset"] == NAMED_PRESET
    assert project.config_controller.config_preset_combo.currentText() == "Default"
    assert project.state.flat_config()["protools_project_dir"] == new_default_dir
    assert project.state.active_daw_processor.project_dir == new_default_dir


def test_loading_named_project_selects_its_preset_in_config(window, app_config, tmp_path):
    _open_project(window, app_config, tmp_path, NAMED_PRESET)

    assert settings.load_config()["app"]["active_config_preset"] == "Default"
    assert window.config_controller.config_preset_combo.currentText() == NAMED_PRESET
    assert window.state.flat_config()["protools_port"] == 32416


def test_rename_and_delete_follow_preset_identity_and_preserve_project_edits(
    window, app_config, phase_work, tmp_path, monkeypatch,
):
    _open_project(window, app_config, tmp_path, NAMED_PRESET)
    custom_dir = str(tmp_path / "project-only")
    path = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir")
    port = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_port")
    path.set_value(custom_dir)
    renamed = "Renamed Studio Mix"
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: (renamed, True))

    def rename(dialog):
        panel = dialog._cfg_panel
        panel._combo.setCurrentText(NAMED_PRESET)
        panel._rename_btn.click()
        panel._combo.setCurrentText("Default")

    _save_preferences(window, monkeypatch, rename)

    assert window.config_controller.config_preset_combo.currentText() == renamed
    assert window.config_controller.config_preset_combo.findText(NAMED_PRESET) == -1
    assert port.value() == 32416
    assert path.value() == custom_dir
    assert window.state.active_daw_processor.project_dir == custom_dir

    def delete(dialog):
        panel = dialog._cfg_panel
        panel._combo.setCurrentText(renamed)
        panel._delete_btn.click()

    _save_preferences(window, monkeypatch, delete)

    assert window.config_controller.config_preset_combo.currentText() == "Default"
    assert renamed not in settings.load_config()["config_presets"]
    assert port.value() == 31416
    assert path.value() == custom_dir
    assert window.state.active_daw_processor.project_dir == custom_dir
    assert window.state.session.prepare_state == "ready"
    assert phase_work == []


@pytest.mark.parametrize("override_protools", [False, True], ids=["inherited", "override"])
def test_saving_two_daw_directories_updates_inherited_fields_and_keeps_preparation(
    project, phase_work, tmp_path, monkeypatch, override_protools,
):
    protools_dir = str(tmp_path / "protools-updated")
    dawproject_dir = str(tmp_path / "dawproject-updated")
    custom_dir = str(tmp_path / "project-only")
    protools_widget = _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir")
    dawproject_widget = _field(project.config_controller.session_widgets, DAWPROJECT, "dawproject_project_dir")
    local_groups = copy.deepcopy(project.state.session_groups)
    local_groups[0]["gain_linked"] = not local_groups[0]["gain_linked"]
    project.state.session_groups = copy.deepcopy(local_groups)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: pytest.fail(
        "A DAW-only save should not prompt to replace local groups"))
    assert project.state.session.prepare_state == "ready"
    if override_protools:
        protools_widget.set_value(custom_dir)
    _select_daw(project, "dawproject_0")

    def edit(dialog):
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(protools_dir)
        _field(dialog._cfg_widgets, DAWPROJECT, "dawproject_project_dir").set_value(dawproject_dir)

    _save_preferences(project, monkeypatch, edit)

    assert project.state.active_daw_processor.id == "dawproject_0"
    assert project.state.active_daw_processor.project_dir == dawproject_dir
    assert dawproject_widget.value() == dawproject_dir
    expected_protools_dir = custom_dir if override_protools else protools_dir
    assert protools_widget.value() == expected_protools_dir
    _select_daw(project, "protools_0")
    assert project.state.active_daw_processor.project_dir == expected_protools_dir
    assert project.state.session.prepare_state == "ready"
    assert project.state.session.daw_state["protools_0"]["assignments"] == {"track-1": "drums"}
    assert project.state.session_groups == local_groups
    assert project.daw_controller.page.transfer_action.isEnabled()
    assert phase_work == []


def test_mixed_preferences_save_refreshes_summary_and_transfer_without_reanalysis(
    window, app_config, phase_work, tmp_path, monkeypatch,
):
    track = TrackContext(
        filename="Kick.wav",
        filepath=str(tmp_path / "audio" / "Kick.wav"),
        audio_data=None,
        samplerate=48000,
        channels=1,
        total_samples=48000,
        bitdepth="24-bit PCM",
        subtype="PCM_24",
        duration_sec=1.0,
    )
    _open_project(window, app_config, tmp_path, tracks=[track])
    clean_finding = "No digital clipping detected"
    assert clean_finding not in window.phase2_page.summary_view.toPlainText()
    new_dir = str(tmp_path / "protools-updated")

    def edit(dialog):
        _field(dialog._cfg_widgets, "_presentation", "show_clean_detectors").setChecked(True)
        _field(dialog._cfg_widgets, "analysis", "window").setValue(800)
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(new_dir)

    _save_preferences(window, monkeypatch, edit)

    assert clean_finding in window.phase2_page.summary_view.toPlainText()
    assert window.state.flat_config()["window"] == 800
    assert window.state.active_daw_processor.project_dir == new_dir
    assert phase_work == []


def test_editing_port_back_to_preset_refreshes_phase3_and_pins_that_value(
    project, phase_work, monkeypatch,
):
    observed_ports = []

    def check_connection(connection, **_kwargs):
        observed_ports.append(connection.port)
        return True, "Test connection"

    monkeypatch.setattr(protools, "check_ptsl_connectivity", check_connection)
    port = _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_port")

    port.setValue(32416)
    project.state.active_daw_processor.check_connectivity()
    port.setValue(31416)
    project.state.active_daw_processor.check_connectivity()

    assert observed_ports == [32416, 31416]

    _save_preferences(
        project, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, PRO_TOOLS, "protools_port").setValue(33416),
    )
    project.state.active_daw_processor.check_connectivity()

    assert port.value() == 31416
    assert observed_ports[-1] == 31416
    assert phase_work == []


@pytest.mark.parametrize("equal_override", [False, True], ids=["different", "matching"])
def test_phase2_port_reset_uses_the_selected_project_preset(
    window, app_config, phase_work, tmp_path, monkeypatch, equal_override,
):
    _open_project(window, app_config, tmp_path, NAMED_PRESET)
    port = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_port")
    port.setValue(33416)
    if equal_override:
        port.setValue(32416)

    state = window.persistence_controller.capture_session_state()
    saved_overrides = state["session_config_overrides"]
    assert saved_overrides[f"{PRO_TOOLS}.protools_port"] == port.value()

    daw_tree_item = window.config_controller.session_dirty_items[PRO_TOOLS]
    daws_tree_item = window.config_controller.session_dirty_items["_daw_processors"]
    assert daw_tree_item.icon(1).isNull() == equal_override
    assert daws_tree_item.icon(1).isNull() == equal_override
    assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION) == (not equal_override)

    observed_ports = []

    def check_connection(connection, **_kwargs):
        observed_ports.append(connection.port)
        return True, "Test connection"

    monkeypatch.setattr(protools, "check_ptsl_connectivity", check_connection)
    reset = _reset_button(port.parentWidget(), "Port")
    tooltip_before_reset = reset.toolTip()
    subtext_before_reset = [
        label.text() for label in port.parentWidget().findChildren(QLabel)
    ]
    reset.click()

    window.state.active_daw_processor.check_connectivity()
    assert port.value() == 32416
    assert window.state.flat_config()["protools_port"] == 32416
    assert observed_ports == [32416]
    assert reset.accessibleName() == "Use preset value for Port"
    assert "32416" in tooltip_before_reset
    assert any("Preset: 32416" in text for text in subtext_before_reset)
    assert "32416" in reset.toolTip()
    assert "Preset:" in port.toolTip()
    assert "32416" in port.toolTip()
    assert daw_tree_item.icon(1).isNull()
    assert daws_tree_item.icon(1).isNull()
    assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
    assert window.persistence_controller.capture_session_state()["session_config_overrides"] == {}

    def update_preset(dialog):
        dialog._cfg_panel._combo.setCurrentText(NAMED_PRESET)
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_port").setValue(34416)

    _save_preferences(window, monkeypatch, update_preset)

    window.state.active_daw_processor.check_connectivity()
    assert port.value() == 34416
    assert observed_ports == [32416, 34416]
    assert "34416" in reset.toolTip()
    assert "Preset:" in port.toolTip()
    assert "34416" in port.toolTip()
    assert window.state.session.prepare_state == "ready"
    assert phase_work == []


@pytest.mark.parametrize("value", [0.1234, 0.12345678901])
def test_phase2_float_reset_displays_the_loaded_preset_precision(window, app_config, tmp_path, value):
    key = "protools_command_delay"
    app_config["config_presets"][NAMED_PRESET]["daw_processors"]["protools"][key] = value
    window.state.config = app_config
    _open_project(window, app_config, tmp_path, NAMED_PRESET)
    delay = _field(window.config_controller.session_widgets, PRO_TOOLS, key)
    delay.setValue(0.8)

    _reset_button(delay.parentWidget(), "Command Delay (s)").click()

    assert delay.value() == value
    assert window.state.flat_config()[key] == value
    assert f"{PRO_TOOLS}.{key}" not in window.state.session_overrides


def test_switching_config_presets_preserves_float_editor_step(qapp, app_config):
    key = "protools_command_delay"
    value = 0.12345678901
    app_config["config_presets"][NAMED_PRESET]["daw_processors"]["protools"][key] = value
    dialog = PreferencesDialog(app_config)
    try:
        delay = _field(dialog._cfg_widgets, PRO_TOOLS, key)
        original_step = delay.singleStep()
        original_decimals = delay.decimals()
        dialog._cfg_panel._combo.setCurrentText(NAMED_PRESET)
        assert delay.value() == value
        assert delay.singleStep() == original_step

        dialog._cfg_panel._combo.setCurrentText("Default")
        assert delay.decimals() == original_decimals
        assert delay.singleStep() == original_step
    finally:
        dispose_window(dialog)


@pytest.mark.parametrize("key, label", [
    ("protools_port", "Port"),
    ("protools_project_dir", "Project directory"),
])
def test_space_activates_the_focused_field_reset(qapp, project, key, label):
    editor = _field(project.config_controller.session_widgets, PRO_TOOLS, key)
    if key == "protools_port":
        editor.setValue(32416)
    else:
        editor.set_value("C:/override")
    project._phase_tabs.setCurrentIndex(1)
    project.phase2_page.detail_tabs.setCurrentIndex(_TAB_SESSION)
    project.config_controller.session_tree.setCurrentItem(project.config_controller.session_dirty_items[PRO_TOOLS])
    reset = _reset_button(project.config_controller.session_stack.currentWidget(), label)
    project.show()
    qapp.processEvents()
    reset.setFocus(Qt.TabFocusReason)
    qapp.processEvents()
    assert reset.hasFocus()

    QTest.keyClick(reset, Qt.Key_Space)

    assert f"{PRO_TOOLS}.{key}" not in project.state.session_overrides
    assert reset.hasFocus()


def test_phase2_project_directory_reset_follows_preset_on_reopen_and_switch(
    window, app_config, phase_work, tmp_path, monkeypatch,
):
    _open_project(window, app_config, tmp_path, NAMED_PRESET)
    path = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir")
    temporary_path = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_temp_dir")
    studio_dir = str(tmp_path / "protools-studio")
    custom_temp = str(tmp_path / "session-temp")
    path.set_value(str(tmp_path / "session-projects"))
    temporary_path.set_value(custom_temp)

    reset = _reset_button(path, "Project directory")
    tooltip_before_reset = reset.toolTip()
    reset.click()

    assert path.value() == studio_dir
    assert window.state.active_daw_processor.project_dir == studio_dir
    assert reset.accessibleName() == "Use preset value for Project directory"
    assert studio_dir in unescape(tooltip_before_reset)
    assert studio_dir in unescape(reset.toolTip())
    assert temporary_path.value() == custom_temp
    assert not window.config_controller.session_dirty_items[PRO_TOOLS].icon(1).isNull()
    assert is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)

    session_path = str(tmp_path / "reset.spsession")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (session_path, ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (session_path, ""))
    window.persistence_controller.save_session_dialog()
    state = load_session(session_path)
    assert state["active_config_preset"] == NAMED_PRESET
    assert state["session_config_overrides"] == {f"{PRO_TOOLS}.protools_temp_dir": custom_temp}
    assert state["session_config"]["daw_processors"]["protools"]["protools_project_dir"] == studio_dir

    updated_dir = str(tmp_path / "studio-updated")

    def update_preset(dialog):
        dialog._cfg_panel._combo.setCurrentText(NAMED_PRESET)
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(updated_dir)
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_temp_dir").set_value(
            str(tmp_path / "studio-temp-updated"))
        dialog._cfg_panel._combo.setCurrentText("Default")

    _save_preferences(window, monkeypatch, update_preset)
    assert path.value() == updated_dir
    assert window.state.active_daw_processor.project_dir == updated_dir
    assert updated_dir in unescape(reset.toolTip())

    window.persistence_controller.load_session_dialog()
    assert window.config_controller.config_preset_combo.currentText() == NAMED_PRESET
    assert path.value() == updated_dir
    assert window.state.active_daw_processor.project_dir == updated_dir
    assert temporary_path.value() == custom_temp

    window.config_controller.config_preset_combo.setCurrentText("Default")
    default_dir = str(tmp_path / "protools-default")
    assert path.value() == default_dir
    assert window.state.active_daw_processor.project_dir == default_dir
    assert default_dir in unescape(reset.toolTip())
    assert temporary_path.value() == custom_temp
    assert window.state.session.prepare_state == "ready"
    assert phase_work == []


def test_phase2_temporary_directory_reset_refreshes_the_phase3_fetch_path(
    window, app_config, phase_work, tmp_path, monkeypatch,
):
    _open_project(window, app_config, tmp_path, NAMED_PRESET)
    path = _field(window.config_controller.session_widgets, PRO_TOOLS, "protools_temp_dir")
    preset_path = str(tmp_path / "protools-temp-studio")
    path.set_value("")

    def unexpected_engine(_processor):
        pytest.fail("An invalid temporary project directory must fail before connecting")

    monkeypatch.setattr(protools.ProToolsDawProcessor, "_open_engine", unexpected_engine)
    _reset_button(path, "Temporary project directory").click()

    assert path.value() == preset_path
    assert window.state.flat_config()["protools_temp_dir"] == preset_path
    with pytest.raises(RuntimeError) as error:
        window.state.active_daw_processor.fetch(window.state.session)
    assert preset_path in str(error.value)
    assert not is_tab_marked(window.phase2_page.detail_tabs, _TAB_SESSION)
    assert window.state.session.prepare_state == "ready"
    assert phase_work == []


def test_phase2_presentation_reset_uses_the_current_presets_value(
    project, phase_work, monkeypatch,
):
    page = "_presentation"
    field = "show_clean_detectors"
    _save_preferences(
        project, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, page, field).setChecked(True),
    )
    checkbox = _field(project.config_controller.session_widgets, page, field)
    assert checkbox.isChecked()
    checkbox.setChecked(False)
    assert is_tab_marked(project.phase2_page.detail_tabs, _TAB_SESSION)

    _reset_button(checkbox.parentWidget(), "Show clean detector results").click()

    assert checkbox.isChecked()
    assert project.detail_controller.show_clean
    assert not is_tab_marked(project.phase2_page.detail_tabs, _TAB_SESSION)
    assert project.state.session.prepare_state == "ready"
    assert phase_work == []


def test_preferences_port_and_path_resets_still_use_built_in_defaults(qapp, app_config):
    dialog = PreferencesDialog(app_config)
    try:
        dialog._cfg_panel._combo.setCurrentText(NAMED_PRESET)
        port = _field(dialog._cfg_widgets, PRO_TOOLS, "protools_port")
        path = _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir")
        assert port.value() == 32416
        assert path.value()

        _reset_button(port.parentWidget(), "Port").click()
        _reset_button(path, "Project directory").click()
        dialog._save_btn.click()

        assert dialog.saved
        saved = dialog.result_config()["config_presets"][NAMED_PRESET]["daw_processors"]["protools"]
        assert saved["protools_port"] == 31416
        assert saved["protools_project_dir"] == ""
    finally:
        dispose_window(dialog)


def test_blank_project_directory_remains_an_override_until_revert(
    project, tmp_path, monkeypatch,
):
    path = _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir")
    path.set_value("")
    new_dir = str(tmp_path / "protools-updated")

    _save_preferences(
        project, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(new_dir),
    )

    assert path.value() == ""
    assert project.state.active_daw_processor.project_dir == ""

    project.config_controller.on_session_config_reset()

    assert path.value() == new_dir
    assert project.state.active_daw_processor.project_dir == new_dir

    later_dir = str(tmp_path / "protools-later")
    _save_preferences(
        project, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(later_dir),
    )
    assert path.value() == later_dir
    assert project.state.active_daw_processor.project_dir == later_dir


def test_removing_last_pro_tools_template_pins_the_empty_list(
    project, monkeypatch,
):
    templates = project.config_controller.session_daw_custom_widgets["protools"]
    templates._table.setCurrentCell(0, 0)
    remove = next(
        button for button in templates.findChildren(QPushButton)
        if button.text() == "Remove"
    )
    remove.click()

    assert project.state.flat_config()["protools_templates"] == []
    assert project.state.active_daw_processor.id == "dawproject_0"

    def edit(dialog):
        templates = dialog._cfg_daw_custom_widgets["protools"]
        templates._table.item(0, 1).setText("Updated Template")

    _save_preferences(project, monkeypatch, edit)

    assert project.state.flat_config()["protools_templates"] == []
    assert project.state.active_daw_processor.id == "dawproject_0"

    project.config_controller.on_session_config_reset()
    _select_daw(project, "protools_0")
    assert project.state.active_daw_processor.name.endswith("Updated Template")


@pytest.mark.parametrize("explicit_port", [False, True], ids=["inherited", "explicit-equal"])
def test_reopened_session_resolves_untouched_fields_against_current_preset(
    project, tmp_path, monkeypatch, explicit_port,
):
    port = _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_port")
    if explicit_port:
        port.setValue(32416)
        port.setValue(31416)
    session_path = str(tmp_path / "mix.spsession")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (session_path, ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (session_path, ""))
    project.persistence_controller.save_session_dialog()
    snapshot = load_session(session_path)["session_config"]
    assert snapshot["daw_processors"]["protools"]["protools_port"] == 31416

    updated_dir = str(tmp_path / "updated-projects")

    def edit(dialog):
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_port").setValue(33416)
        _field(dialog._cfg_widgets, PRO_TOOLS, "protools_project_dir").set_value(updated_dir)

    _save_preferences(project, monkeypatch, edit)
    project.persistence_controller.load_session_dialog()

    expected_port = 31416 if explicit_port else 33416
    assert port.value() == expected_port
    assert project.state.flat_config()["protools_port"] == expected_port
    assert project.state.active_daw_processor.project_dir == updated_dir


def test_switching_project_preset_preserves_overrides_and_refreshes_phase3_without_analysis(
    project, phase_work, tmp_path,
):
    custom_dir = str(tmp_path / "project-only")
    path = _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_project_dir")
    path.set_value(custom_dir)

    project.config_controller.config_preset_combo.setCurrentText(NAMED_PRESET)

    assert path.value() == custom_dir
    assert project.state.active_daw_processor.project_dir == custom_dir
    assert _field(project.config_controller.session_widgets, PRO_TOOLS, "protools_port").value() == 32416
    assert project.state.flat_config()["protools_port"] == 32416
    assert settings.load_config()["app"]["active_config_preset"] == "Default"
    assert project.state.session.prepare_state == "ready"
    assert phase_work == []
