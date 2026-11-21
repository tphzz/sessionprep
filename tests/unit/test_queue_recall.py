"""Project and transfer queue recall through desktop file actions."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.batch.panel import BatchItem
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.session.io import CURRENT_VERSION, load_session
from sessionpreplib.models import DawCommand, DawCommandResult


def record(tmp_path, name, status="Pending"):
    return {
        "id": f"id-{name}", "project_name": name, "daw_processor_id": "dawproject_0",
        "daw_processor_name": "DAWproject", "output_path": str(tmp_path / f"{name}.dawproject"),
        "session_state": {"source_dir": str(tmp_path), "project_name": name,
                          "version": CURRENT_VERSION, "session_config": {
                              "daw_processors": {"protools": {"protools_temp_dir": str(tmp_path / name)}}}},
        "status": status,
    }


@pytest.fixture
def windows(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "prefs"))
    opened = []

    def new_window():
        window = SessionPrepWindow()
        opened.append(window)
        return window

    yield new_window
    for window in reversed(opened):
        window.batch_controller.dock.clear_all()
        dispose_window(window)
    app.processEvents()


def test_disk_snapshot_of_running_batch_recovers_without_overwriting_existing_export(
        tmp_path, windows, monkeypatch):
    path = tmp_path / "jobs.spbatch"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    window = windows()
    dock = window.batch_controller.dock
    for name, status in [("Done", "Success"), ("Interrupted", "Pending"), ("Next", "Pending")]:
        dock.add_item(BatchItem.from_dict(record(tmp_path, name, status)))
    dock.update_item("id-Interrupted", "Running", "Transferring...")
    window.batch_controller.save_queue()
    assert json.loads(path.read_text())[1]["status"] == "Running"

    restored = windows()
    restored.batch_controller.load_queue()
    queued = restored.batch_controller.dock.get_pending_items()
    assert [j.project_name for j in queued] == ["Interrupted", "Next"]
    assert queued[0].status == "Failed" and "interrupt" in queued[0].result_text.lower()
    manager = restored.batch_controller.manager
    config = manager._rehydrate_session(queued[0].session_state).config
    assert config["protools_temp_dir"] == str(tmp_path / "Interrupted")
    existing = Path(queued[0].output_path)
    existing.write_bytes(b"earlier export")
    attempts = []
    manager.item_finished.connect(lambda i, status, note: attempts.append((i, status, note)))
    checks = []
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_start_worker", lambda *args: checks.append(args))
        manager.start_single(queued[0])
    assert not checks
    assert [s for _, s, _ in attempts] == ["Running", "Failed"]
    assert str(existing) in attempts[-1][-1]
    assert existing.read_bytes() == b"earlier export"
    restored.batch_controller.save_queue()
    assert [row["status"] for row in json.loads(path.read_text())] == ["Success", "Failed", "Pending"]

    existing.rename(existing.with_suffix(".previous"))
    exported = []

    class OfflineAdapter:
        def check_connectivity(self):
            return True, "Ready"

        def transfer(self, session, output_path, **_kwargs):
            exported.append((session.project_name, session.config["protools_temp_dir"]))
            Path(output_path).write_text(session.project_name)
            return [DawCommandResult(DawCommand("export", output_path), True)]

    monkeypatch.setattr(manager, "_get_daw_processor", lambda *_args: OfflineAdapter())
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    manager.finished.connect(loop.quit)
    timeout.start(5000)
    restored.batch_controller.dock._run_btn.click()
    if manager.is_running:
        loop.exec()
    assert timeout.isActive() and not manager.is_running
    timeout.stop()
    assert exported == [(name, str(tmp_path / name)) for name in ("Interrupted", "Next")]
    assert [(i, s) for i, s, _ in attempts if s != "Running"] == [
        ("id-Interrupted", "Failed"), ("id-Interrupted", "Success"), ("id-Next", "Success")]
    assert existing.read_text() == "Interrupted"
    assert (tmp_path / "Done.dawproject").exists() is False


@pytest.mark.parametrize("choice", ["Append", "Replace"])
@pytest.mark.parametrize("problem", ["invalid row", "bad template", "repeated ID", "duplicate name"])
def test_failed_queue_load_preserves_complete_queue_and_workspace(tmp_path, windows, monkeypatch, choice, problem):
    window = windows()
    dock = window.batch_controller.dock
    dock.add_item(BatchItem.from_dict(record(tmp_path, "Earlier")))
    rows = [record(tmp_path, name) for name in ("One", "Two", "Three")]
    if problem == "invalid row":
        rows[1]["session_state"] = []
    elif problem == "bad template":
        rows[1]["session_state"]["session_config_overrides"] = {
            "daw_processors.dawproject.dawproject_templates": [
                {"name": "Mix", "template_path": "mix.dawproject", "fader_ceiling_db": "bad"}
            ]}
    elif problem == "repeated ID":
        rows[1]["id"] = "id-Earlier" if choice == "Append" else rows[0]["id"]
    else:
        rows[1]["project_name"] = "EARLIER" if choice == "Append" else "ONE"
    path = tmp_path / "bad.spbatch"
    path.write_text(json.dumps(rows))
    original = dock.get_state()
    messages = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    monkeypatch.setattr(QMessageBox, "exec", lambda _box: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: next(b for b in box.buttons() if b.text() == choice))
    for method in ("warning", "critical"):
        monkeypatch.setattr(QMessageBox, method, lambda _w, _t, message: messages.append(message))
    cleared = []
    window.batch_controller.clear_workspace_requested.disconnect(window.persistence_controller.clear_workspace)
    window.batch_controller.clear_workspace_requested.connect(lambda: cleared.append(True))
    window.batch_controller.load_queue()
    assert dock.get_state() == original
    assert len(messages) == 1 and "2" in messages[0]
    assert not cleared


def test_edit_queued_session_with_missing_sources_keeps_item(tmp_path, windows, monkeypatch):
    window = windows()
    row = record(tmp_path, "Offline")
    row["session_state"]["source_dir"] = str(tmp_path / "disconnected")
    item = BatchItem.from_dict(row)
    window.batch_controller.dock.add_item(item)
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _w, _t, message: messages.append(message))
    window.batch_controller.load_batch_item(item)
    assert window.batch_controller.dock.get_pending_items() == [item]
    assert str(tmp_path / "disconnected") in messages[0]
    window.batch_controller.manager.start_single(item)
    assert item.status == "Failed"
    assert str(tmp_path / "disconnected") in item.result_text


def test_dismissing_append_replace_prompt_keeps_current_queue(tmp_path, windows, monkeypatch):
    window = windows()
    dock = window.batch_controller.dock
    dock.add_item(BatchItem.from_dict(record(tmp_path, "Current")))
    queue = tmp_path / "import.spbatch"
    queue.write_text(json.dumps([record(tmp_path, "Imported")]))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(queue), ""))
    monkeypatch.setattr(QMessageBox, "exec", lambda box: box.reject())
    window.batch_controller.load_queue()
    assert [item.project_name for item in dock.get_pending_items()] == ["Current"]


@pytest.mark.parametrize("version", range(1, CURRENT_VERSION + 1))
def test_open_supported_project_versions_from_authored_json(tmp_path, version):
    project = {"version": version, "source_dir": str(tmp_path)}
    if version >= 5:
        project["project_name"] = "Monitor mix"
    path = tmp_path / "old.spsession"
    path.write_text(json.dumps(project))
    result = load_session(str(path))
    assert result["source_dir"] == str(tmp_path)
    assert result["project_name"] == ("Monitor mix" if version >= 5 else "")
    assert result["transfer_manifest"] == []
    assert result["session_config_overrides"] is None


def test_authored_project_reopens_with_pinned_directory_latest_default_and_saved_manifest(
        tmp_path, windows, monkeypatch):
    project = {
        "version": CURRENT_VERSION, "source_dir": str(tmp_path), "active_config_preset": "Studio",
        "session_config": {"daw_processors": {"protools": {
            "protools_project_dir": "My projects", "protools_temp_dir": "Old scratch",
            "protools_templates": [{"group": "Mix", "name": "Stereo"}]}}},
        "session_config_overrides": {"daw_processors.protools.protools_project_dir": "My projects"},
        "topology": {"entries": [{"output_filename": "Keys R.wav", "output_channels": 1,
                                   "sources": [{"input_filename": "Keys.wav", "routes": [
                                       {"source_channel": 1, "target_channel": 0}]}]}]},
        "transfer_manifest": [{"entry_id": "keys-right", "output_filename": "Keys R.wav",
                               "daw_track_name": "Keys R"}],
        "daw_state": {"protools_0": {"folders": [{"id": "keys", "name": "Keys"}],
                                     "assignments": {"keys-right": "keys"}}},
    }
    path = tmp_path / "mix.spsession"
    path.write_text(json.dumps(project))
    window = windows()
    preset = settings.resolve_config_preset(window.state.config, "Default")
    preset["daw_processors"]["protools"].update(
        protools_project_dir="Studio projects", protools_temp_dir="New scratch",
        protools_templates=[{"group": "Mix", "name": "Stereo"}])
    window.state.config["config_presets"]["Studio"] = preset
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    window.persistence_controller.load_session_dialog()
    assert window.state.active_daw_processor.project_dir == "My projects"
    assert window.state.active_daw_processor._temp_dir == "New scratch"
    entry, = window.state.session.transfer_manifest
    assert (entry.entry_id, entry.output_filename) == ("keys-right", "Keys R.wav")
    assert window.daw_controller.page.folder_tree.topLevelItem(0).text(0) == "Keys"
    assert window.daw_controller.page.folder_tree.topLevelItem(0).child(0).text(0) == "keys-right"
    route, = window.state.topo_topology.entries[0].sources[0].routes
    assert (route.source_channel, route.target_channel, route.gain) == (1, 0, 1.0)


@pytest.mark.parametrize("invalid", [
    {"version": CURRENT_VERSION + 1}, {"project_name": ["bad"]}, {"tracks": 3},
    {"session_config_overrides": {"analysis.window": "bad"}}, {"recursive_scan": "False"},
    {"session_groups": [42]}, {"session_config": {"detectors": {"clipping": []}}},
    {"session_groups": [{"name": 42}]},
    {"session_groups": [{"name": "Drums", "color": []}]},
    {"output_gain_adjustments": 1},
    {"output_gain_adjustments": {"drums.wav": {"peak_db": 1.0, "reduction_db": []}}},
    {"prepare_errors": [["drums.wav"]]},
    {"daw_state": {"protools_0": []}},
    {"daw_state": {"dawproject_0": {"folders": [{"id": "A", "name": ["bad"]}]}}},
    {"daw_state": {"protools_0": {"track_order": {"Drums": 1}}}},
    {"session_config": {"daw_processors": {"dawproject": {"dawproject_templates": [
        {"name": "Mix", "template_path": "mix.dawproject", "fader_ceiling_db": "bad"}]}}}},
])
def test_rejected_session_load_preserves_open_workspace(tmp_path, windows, monkeypatch, invalid):
    window = windows()
    original_folder = tmp_path / "original"
    original_folder.mkdir()
    window.persistence_controller.restore_session_state({"source_dir": str(original_folder), "tracks": []})
    before = window.state.session
    path = tmp_path / "broken.spsession"
    path.write_text(json.dumps({"version": CURRENT_VERSION, "source_dir": str(tmp_path), **invalid}))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _w, _t, message: messages.append(message))
    window.persistence_controller.load_session_dialog()
    assert messages and window.state.session is before and window.state.source_dir == str(original_folder)
    retained = tmp_path / "after-error.spsession"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(retained), ""))
    window.persistence_controller.save_session_dialog()
    assert json.loads(retained.read_text())["source_dir"] == str(original_folder)


def test_partial_legacy_records_and_unknown_daw_data_survive_validation(tmp_path):
    state = {
        "version": 6, "source_dir": str(tmp_path),
        "session_groups": [{"name": "Drums", "extension": ["legacy"]}],
        "daw_state": {"protools_0": {"assignments": {"kick": "drums"}, "custom": [1, 2]},
                      "custom_destination": ["opaque"]},
        "output_gain_adjustments": {"kick.wav": {"peak_db": .5, "reduction_db": .6}},
        "session_config": {"daw_processors": {"dawproject": {"dawproject_templates": [
            {"name": "Mix", "template_path": "mix.dawproject", "fader_ceiling_db": "6.0"}]}}},
    }
    path = tmp_path / "legacy.spsession"
    path.write_text(json.dumps(state))
    loaded = load_session(str(path))
    for key in ("daw_state", "session_groups", "output_gain_adjustments", "session_config"):
        assert loaded[key] == state[key]
