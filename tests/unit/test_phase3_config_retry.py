from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.analysis.worker import DawCheckWorker, DawFetchWorker, DawTransferWorker
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.session.io import save_session
from sessionprepgui.tracks.table_widgets import _PHASE_SETUP
from sessionpreplib import config
from sessionpreplib.daw_processors import protools
from sessionpreplib.models import TransferEntry


PRO_TOOLS = "daw_processors.protools"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def messages(monkeypatch):
    seen = []

    def record(_parent, title, message, *_args):
        seen.append((title, message))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "critical", record)
    monkeypatch.setattr(QMessageBox, "warning", record)
    monkeypatch.setattr(QMessageBox, "information", record)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    return seen


@pytest.fixture
def phase3_project(qapp, messages, monkeypatch, tmp_path):
    app_dir = tmp_path / "settings"
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(app_dir))
    monkeypatch.setattr(config, "get_app_dir", lambda: str(app_dir))
    app_config = settings.build_defaults()
    preset = app_config["config_presets"]["Default"]
    preset["daw_processors"]["protools"].update(
        protools_project_dir=str(tmp_path / "projects"),
        protools_temp_dir=str(tmp_path / "temporary-projects"),
        protools_templates=[{"group": "SessionPrep", "name": "Template"}],
    )
    settings.save_config(app_config)
    source = tmp_path / "audio"
    source.mkdir()
    window = SessionPrepWindow()
    try:
        window._restore_session_state({
            "source_dir": str(source),
            "active_config_preset": "Default",
            "session_config": preset,
            "session_groups": app_config["group_presets"]["Default"],
            "project_name": "Mix",
            "tracks": [],
            "prepare_state": "ready",
            "transfer_manifest": [TransferEntry("kick", "Kick.wav", "Kick")],
            "daw_state": {"protools_0": {"assignments": {"kick": "drums"}}},
        })
        window._phase_tabs.setCurrentIndex(_PHASE_SETUP)
        yield window
    finally:
        dispose_window(window)


@pytest.fixture
def create_transport(monkeypatch):
    checked_ports = []
    transfers = []

    def check_connection(connection, **_kwargs):
        checked_ports.append(connection.port)
        return True, "Connected for test"

    def transfer(processor, session, _output_path, **_kwargs):
        transfers.append((
            processor.project_dir,
            session.config["protools_project_dir"],
            session.config["protools_port"],
        ))
        return []

    monkeypatch.setattr(protools, "check_ptsl_connectivity", check_connection)
    monkeypatch.setattr(protools.ProToolsDawProcessor, "transfer", transfer)
    monkeypatch.setattr(DawCheckWorker, "start", DawCheckWorker.run)
    monkeypatch.setattr(DawTransferWorker, "start", DawTransferWorker.run)
    return checked_ports, transfers


def _field(widgets, key):
    return dict(widgets[PRO_TOOLS])[key]


def _save_preferences(window, monkeypatch, edit):
    def edit_and_save(dialog):
        edit(dialog)
        dialog._save_btn.click()
        assert dialog.saved
        return dialog.result()

    with monkeypatch.context() as modal:
        modal.setattr(PreferencesDialog, "exec", edit_and_save)
        window._on_preferences()


def test_create_retries_with_phase2_project_directory_correction(
    phase3_project, create_transport, messages, tmp_path,
):
    window = phase3_project
    checked_ports, transfers = create_transport
    assert window._transfer_action.isEnabled()

    window._transfer_action.trigger()

    assert [title for title, _message in messages] == ["Project Directory Not Found"]
    assert str(tmp_path / "projects") in messages[0][1]
    assert checked_ports == []
    assert transfers == []
    assert window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()

    corrected = tmp_path / "corrected-projects"
    corrected.mkdir()
    _field(window._session_widgets, "protools_project_dir").set_value(str(corrected))
    window._transfer_action.trigger()

    assert checked_ports == [31416]
    assert transfers == [(str(corrected), str(corrected), 31416)]
    assert len(messages) == 1
    assert window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()
    assert window._session.prepare_state == "ready"


def test_fetch_retries_with_inherited_temporary_directory_correction(
    phase3_project, messages, tmp_path, monkeypatch,
):
    window = phase3_project
    home = tmp_path / "home"
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    template = (
        home / "Documents" / "Pro Tools" / "Session Templates"
        / "SessionPrep" / "Template.ptxt"
    )
    template.parent.mkdir(parents=True)
    template.write_text("template", encoding="utf-8")
    cache_file = tmp_path / "settings" / "pt_template_cache.json"
    cache_file.write_text(json.dumps({
        "SessionPrep/Template": {
            "mtime": template.stat().st_mtime,
            "folders": [{
                "id": "drums",
                "name": "Drums",
                "folder_type": "routing",
                "index": 1,
                "parent_id": None,
            }],
        },
    }), encoding="utf-8")
    monkeypatch.setattr(DawFetchWorker, "start", DawFetchWorker.run)

    def unexpected_engine(_processor):
        raise AssertionError("Fetching from the prepared cache must not open Pro Tools")

    monkeypatch.setattr(protools.ProToolsDawProcessor, "_open_engine", unexpected_engine)

    window._fetch_action.click()

    assert [title for title, _message in messages] == ["Fetch Failed"]
    assert str(tmp_path / "temporary-projects") in messages[0][1]
    assert window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()

    corrected = tmp_path / "corrected-temporary-projects"
    corrected.mkdir()
    _save_preferences(
        window, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, "protools_temp_dir").set_value(str(corrected)),
    )
    window._fetch_action.click()

    assert len(messages) == 1
    assert window._folder_tree.topLevelItem(0).text(0) == "Drums"
    assert window._session.daw_state["protools_0"]["assignments"] == {"kick": "drums"}
    assert window._session.prepare_state == "ready"
    assert window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()


def test_reordered_templates_keep_selection_and_assignments_until_replacement(
    phase3_project, messages, monkeypatch,
):
    window = phase3_project

    def add_alternative(dialog):
        templates = dialog._cfg_daw_custom_widgets["protools"]
        add = next(
            button for button in templates.findChildren(QPushButton)
            if button.text() == "Add"
        )
        add.click()
        templates._table.item(1, 0).setText("SessionPrep")
        templates._table.item(1, 1).setText("Alternative")

    _save_preferences(window, monkeypatch, add_alternative)

    def fetch_folders(processor, session, **_kwargs):
        session.daw_state[processor.id]["folders"] = [{
            "id": "drums",
            "name": "Drums",
            "folder_type": "routing",
            "index": 1,
            "parent_id": None,
        }]
        return session

    monkeypatch.setattr(protools.ProToolsDawProcessor, "fetch", fetch_folders)
    monkeypatch.setattr(DawFetchWorker, "start", DawFetchWorker.run)
    window._fetch_action.click()

    assert window._active_daw_processor.id == "protools_0"
    assert window._folder_tree.topLevelItem(0).text(0) == "Drums"

    def reorder(dialog):
        table = dialog._cfg_daw_custom_widgets["protools"]._table
        table.item(0, 1).setText("Alternative")
        table.item(1, 1).setText("Template")

    _save_preferences(window, monkeypatch, reorder)

    assert window._daw_combo.currentText().endswith(" / Template")
    assert window._active_daw_processor.id == "protools_1"
    assert window._session.daw_state["protools_1"]["assignments"] == {"kick": "drums"}
    assert "protools_0" not in window._session.daw_state
    assert window._folder_tree.topLevelItem(0).text(0) == "Drums"
    assert window._transfer_action.isEnabled()

    def replace(dialog):
        table = dialog._cfg_daw_custom_widgets["protools"]._table
        table.item(1, 1).setText("Replacement")

    _save_preferences(window, monkeypatch, replace)
    window._daw_combo.setCurrentIndex(1)

    assert window._daw_combo.currentText().endswith(" / Replacement")
    assert window._active_daw_processor.id == "protools_1"
    assert not window._session.daw_state.get("protools_1")
    assert window._folder_tree.topLevelItemCount() == 0
    assert not window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()
    assert window._session.prepare_state == "ready"
    assert messages == []


@pytest.mark.parametrize("change_source", ["phase2", "preferences"])
def test_pending_create_keeps_validated_config_and_next_attempt_uses_the_edit(
    phase3_project, create_transport, messages, tmp_path, monkeypatch, change_source,
):
    window = phase3_project
    checked_ports, transfers = create_transport
    pending_checks = []
    monkeypatch.setattr(DawCheckWorker, "start", lambda worker: pending_checks.append(worker))
    original = tmp_path / "projects"
    original.mkdir()
    corrected = tmp_path / "corrected-projects"
    corrected.mkdir()

    window._transfer_action.trigger()

    assert len(pending_checks) == 1
    assert not window._transfer_action.isEnabled()
    assert not window._fetch_action.isEnabled()
    assert transfers == []

    if change_source == "phase2":
        _field(window._session_widgets, "protools_project_dir").set_value(str(corrected))
        _field(window._session_widgets, "protools_port").setValue(32416)
    else:
        def edit(dialog):
            _field(dialog._cfg_widgets, "protools_project_dir").set_value(str(corrected))
            _field(dialog._cfg_widgets, "protools_port").setValue(32416)

        _save_preferences(window, monkeypatch, edit)

    assert not window._transfer_action.isEnabled()
    assert not window._fetch_action.isEnabled()
    pending_checks.pop().run()

    assert checked_ports == [31416]
    assert transfers == [(str(original), str(original), 31416)]
    assert window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()

    window._transfer_action.trigger()
    assert len(pending_checks) == 1
    pending_checks.pop().run()

    assert checked_ports == [31416, 32416]
    assert transfers[-1] == (str(corrected), str(corrected), 32416)
    assert messages == []
    assert window._session.prepare_state == "ready"


@pytest.mark.parametrize("workspace_change", ["open-empty-folder", "load-session", "close-window"])
def test_pending_create_cannot_transfer_after_leaving_the_original_project(
    phase3_project, create_transport, messages, tmp_path, monkeypatch, workspace_change,
):
    window = phase3_project
    checked_ports, transfers = create_transport
    pending_checks = []
    monkeypatch.setattr(DawCheckWorker, "start", lambda worker: pending_checks.append(worker))
    (tmp_path / "projects").mkdir()
    original = window._session

    window._transfer_action.trigger()
    assert len(pending_checks) == 1
    assert not window._transfer_action.isEnabled()

    if workspace_change == "close-window":
        assert window.close()
    else:
        next_source = tmp_path / "another-folder"
        next_source.mkdir()
        if workspace_change == "open-empty-folder":
            monkeypatch.setattr(
                QFileDialog, "getExistingDirectory", lambda *args: str(next_source),
            )
            window._on_open_path()
            assert window._session is None
        else:
            replacement = window._capture_session_state()
            replacement.update(source_dir=str(next_source), project_name="Replacement")
            path = str(tmp_path / "replacement.spsession")
            save_session(path, replacement)
            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))
            window._on_load_session()
            assert window._session is not original
            assert window._project_name_edit.text() == "Replacement"

    pending_checks.pop().run()

    assert checked_ports == [31416]
    assert transfers == []
    assert messages == []


def test_pending_fetch_failure_reports_the_processor_that_started_it(
    phase3_project, monkeypatch, messages,
):
    window = phase3_project
    pending = []
    monkeypatch.setattr(DawFetchWorker, "start", lambda worker: pending.append(worker))

    def fail(_processor, _session, **_kwargs):
        raise RuntimeError("Temporary project is unavailable")

    monkeypatch.setattr(protools.ProToolsDawProcessor, "fetch", fail)
    window._fetch_action.click()
    _save_preferences(
        window, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, "protools_enabled").setChecked(False),
    )
    assert window._active_daw_processor is None

    pending.pop().run()

    assert len(messages) == 1
    title, detail = messages[0]
    assert title == "Fetch Failed"
    assert "SessionPrep / Template" in detail
    assert "Temporary project is unavailable" in detail
    assert not window._fetch_action.isEnabled()
    assert window._session.daw_state["protools_0"]["assignments"] == {"kick": "drums"}


def test_pending_fetch_uses_captured_temp_directory_and_retry_uses_edited_default(
    phase3_project, monkeypatch, tmp_path, messages,
):
    window = phase3_project
    pending = []
    reads = []
    monkeypatch.setattr(DawFetchWorker, "start", lambda worker: pending.append(worker))

    def fetch(processor, session, **_kwargs):
        reads.append(processor._temp_dir)
        return session

    monkeypatch.setattr(protools.ProToolsDawProcessor, "fetch", fetch)
    window._fetch_action.click()
    corrected = str(tmp_path / "new-scratch")
    _save_preferences(
        window, monkeypatch,
        lambda dialog: _field(dialog._cfg_widgets, "protools_temp_dir").set_value(corrected),
    )
    assert not window._fetch_action.isEnabled()
    pending.pop().run()
    window._fetch_action.click()
    pending.pop().run()

    assert reads == [str(tmp_path / "temporary-projects"), corrected]
    assert messages == []
    assert window._session.prepare_state == "ready"


@pytest.mark.parametrize("change", ["template", "workspace"])
def test_pending_fetch_does_not_write_folders_to_a_replaced_target(
    phase3_project, monkeypatch, tmp_path, messages, change,
):
    window = phase3_project
    pending = []
    monkeypatch.setattr(DawFetchWorker, "start", lambda worker: pending.append(worker))

    def fetch(processor, session, **_kwargs):
        session.daw_state[processor.id] = {
            "folders": [{"id": "old", "name": "Obsolete folder"}],
            "assignments": {"kick": "old"},
        }
        return session

    monkeypatch.setattr(protools.ProToolsDawProcessor, "fetch", fetch)
    window._fetch_action.click()
    if change == "template":
        _save_preferences(
            window, monkeypatch,
            lambda dialog: dialog._cfg_daw_custom_widgets["protools"]._table.item(
                0, 1).setText("Replacement"),
        )
    else:
        replacement = window._capture_session_state()
        source = tmp_path / "new-audio"
        source.mkdir()
        replacement.update(source_dir=str(source), project_name="Replacement", daw_state={})
        window._restore_session_state(replacement)

    pending.pop().run()

    assert not window._session.daw_state.get("protools_0")
    assert not window._transfer_action.isEnabled()
    assert window._fetch_action.isEnabled()
    assert messages == []
