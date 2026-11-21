from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (
    QCoreApplication,
    QEventLoop,
    QObject,
    QThread,
    QTimer,
    Signal,
    Slot,
)

from sessionprepgui.batch import BatchItem
from sessionprepgui.batch import manager as batch
from sessionpreplib.daw_processors import protools
from sessionpreplib.models import DawCommand, DawCommandResult


class _WorkerGate(QObject):
    """Let the test observe and release a paused worker."""

    reached = Signal()

    def __init__(self):
        super().__init__()
        self.delivered = False
        self.release = Event()
        self.reached.connect(self._record_delivery)

    @Slot()
    def _record_delivery(self):
        self.delivered = True

    def pause(self):
        self.reached.emit()
        self.release.wait(timeout=10)


def _wait_until(predicate):
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(1)
    timeout = QTimer()
    timeout.setSingleShot(True)

    def quit_when_ready():
        if predicate():
            loop.quit()

    poll.timeout.connect(quit_when_ready)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start(4000)
    loop.exec()
    assert predicate(), "Timed out waiting for BatchManager signals"


def _item(name, port):
    return BatchItem(
        id=name,
        project_name=name,
        daw_processor_id="protools_0",
        daw_processor_name="Pro Tools",
        output_path=f"exports/{name}",
        session_state={
            "project_name": name,
            "session_config": {
                "daw_processors": {"protools": {
                    "protools_port": port,
                    "protools_project_dir": f"projects/{name}",
                    "protools_templates": [{"group": "Mix", "name": "Template"}],
                }},
            },
        },
    )


def _single_worker_lifetime():
    _app = QCoreApplication([])
    preflight = _WorkerGate()
    transfer = _WorkerGate()

    class HeldCheckWorker(batch.DawCheckWorker):
        def run(self):
            super().run()
            preflight.pause()

    class HeldTransferWorker(batch.DawTransferWorker):
        def run(self):
            super().run()
            transfer.pause()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "DawCheckWorker", HeldCheckWorker)
        patch.setattr(batch, "DawTransferWorker", HeldTransferWorker)
        patch.setattr(
            protools, "check_ptsl_connectivity", lambda *_args, **_kwargs: (True, "Connected"),
        )
        patch.setattr(
            protools.ProToolsDawProcessor, "transfer",
            lambda *_args, **_kwargs: [DawCommandResult(DawCommand("import", "Kick"), True)],
        )
        manager = batch.BatchManager(None)
        item = _item("Run Single", 32411)
        started = []
        completed = []
        updates = []
        manager.started.connect(lambda: started.append(True))
        manager.item_finished.connect(lambda *result: updates.append(result))
        manager.finished.connect(lambda: completed.append(True))

        try:
            manager.start_single(item)
            _wait_until(lambda: preflight.delivered)
            status_during_check = item.result_text
            preflight.release.set()
            _wait_until(lambda: transfer.delivered)
            terminal_while_running = (
                item.status in ("Success", "Failed") or bool(completed)
            )
        finally:
            preflight.release.set()
            transfer.release.set()

        _wait_until(lambda: bool(completed))
        assert status_during_check == "Checking DAW..."
        assert not terminal_while_running
        assert started == completed == [True]
        assert item.status == "Success"
        assert [(item_id, status) for item_id, status, _details in updates] == [
            (item.id, "Running"),
            (item.id, "Running"),
            (item.id, "Success"),
        ], updates


def _batch_continues_after_failures():
    app = QCoreApplication([])
    rejected = _item("Connection rejection", 32411)
    failed = _item("Transfer failure", 32412)
    good = _item("Finished mix", 32413)
    live = _item("Current Preferences", 32419)
    window = SimpleNamespace(_config={
        "app": {"active_config_preset": "Default"},
        "config_presets": {"Default": live.session_state["session_config"]},
    })
    checked = []
    transferred = []

    def check_connection(connection, **_kwargs):
        checked.append(connection.port)
        if connection.port == 32411:
            return False, "Connection refused"
        return True, "Connected"

    def transfer(processor, session, output_path, *, progress_cb, close_when_done):
        transferred.append((
            session.project_name,
            session.config["protools_port"],
            processor.project_dir,
            output_path,
            close_when_done,
        ))
        if session.project_name == failed.project_name:
            progress_cb(1, 2, "Importing Snare Top")
            return [
                DawCommandResult(DawCommand("import", "Kick"), True),
                DawCommandResult(DawCommand("import", "Snare Top"), False, "Audio file missing"),
            ]
        progress_cb(1, 2, "Importing Mix Bus")
        return [DawCommandResult(DawCommand("import", "Mix Bus"), True)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(protools, "check_ptsl_connectivity", check_connection)
        patch.setattr(protools.ProToolsDawProcessor, "transfer", transfer)
        manager = batch.BatchManager(window)
        started = []
        completed = []
        terminal = []
        progress = []
        messages = []
        signal_threads = []

        def record_item(item_id, status, details):
            signal_threads.append(QThread.currentThread() == app.thread())
            if status in ("Failed", "Success"):
                terminal.append((item_id, status, details))

        manager.started.connect(lambda: started.append(True))
        manager.finished.connect(lambda: completed.append(True))
        manager.item_finished.connect(record_item)
        manager.batch_progress_value.connect(lambda current, total: progress.append((current, total)))
        manager.batch_progress_message.connect(messages.append)

        manager.start_batch([rejected, failed, good])
        _wait_until(lambda: bool(completed))

        assert started == completed == [True]
        assert checked == [32411, 32412, 32413]
        assert transferred == [
            (failed.project_name, 32412, "projects/Transfer failure", failed.output_path, True),
            (good.project_name, 32413, "projects/Finished mix", good.output_path, True),
        ]
        assert [(item_id, status) for item_id, status, _details in terminal] == [
            (rejected.id, "Failed"),
            (failed.id, "Failed"),
            (good.id, "Success"),
        ], terminal
        assert "Connection refused" in rejected.result_text
        assert "Snare Top" in failed.result_text
        assert "Audio file missing" in failed.result_text
        assert all(signal_threads)
        assert [message for message in messages if "Importing" in message] == [
            "[Transfer failure] Importing Snare Top",
            "[Finished mix] Importing Mix Bus",
        ]
        assert (150, 300) in progress
        assert (250, 300) in progress


def _quit_after_current_batch_item():
    from PySide6.QtWidgets import QApplication, QMessageBox

    from sessionprepgui import settings
    from sessionprepgui.mainwindow import SessionPrepWindow

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    first = _item("Initial mix", 32411)
    active = _item("Active mix", 32412)
    next_item = _item("Remaining mix", 32413)
    first_gate = _WorkerGate()
    active_gate = _WorkerGate()
    gates = {first.project_name: first_gate, active.project_name: active_gate}
    checked = []
    transfers = []
    close_events = []
    prompts = []
    answers = iter([QMessageBox.No, QMessageBox.Yes])

    class WindowWithCloseEvents(SessionPrepWindow):
        def closeEvent(self, event):
            super().closeEvent(event)
            close_events.append(event.isAccepted())

    def confirm_quit(_parent, title, _message, *_args):
        prompts.append(title)
        return next(answers, QMessageBox.Yes)

    def check_connection(connection, **_kwargs):
        checked.append(connection.port)
        return True, "Connected"

    def transfer(_processor, session, _output_path, **_kwargs):
        transfers.append(session.project_name)
        gate = gates.get(session.project_name)
        if gate:
            gate.pause()
        return [DawCommandResult(DawCommand("import", "Kick"), True)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "get_app_dir", lambda: str(Path.cwd() / "settings"))
        patch.setattr(QMessageBox, "warning", confirm_quit)
        patch.setattr(protools, "check_ptsl_connectivity", check_connection)
        patch.setattr(protools.ProToolsDawProcessor, "transfer", transfer)
        window = WindowWithCloseEvents()
        dock = window.batch_controller.dock
        for item in (first, active, next_item):
            assert dock.add_item(item)
        completed = []
        window.batch_controller.manager.finished.connect(lambda: completed.append(True))

        try:
            dock._run_btn.click()
            _wait_until(lambda: first_gate.delivered)
            assert not window.close()
            assert close_events == [False]

            # Declining the first confirmation leaves the normal queue running.
            first_gate.release.set()
            _wait_until(lambda: active_gate.delivered)
            assert first.status == "Success"
            assert active.status == "Running"

            assert not window.close()
            assert close_events == [False, False]
            assert not completed
            assert next_item.status == "Pending"
        finally:
            first_gate.release.set()
            active_gate.release.set()

        _wait_until(lambda: close_events[-1:] == [True])
        assert close_events == [False, False, True]
        assert prompts == ["Pending Batch Items", "Pending Batch Items"]
        assert completed == [True]
        assert checked == [32411, 32412]
        assert transfers == [first.project_name, active.project_name]
        assert active.status == "Success"
        assert dock.get_pending_items() == [next_item]


def _run_isolated(scenario, tmp_path):
    # Qt aborts the entire process if a running thread is destroyed.
    script = """
import runpy
import sys

sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2])[sys.argv[3]]()
"""
    this_file = Path(__file__).resolve()
    result = subprocess.run(
        [
            sys.executable, "-X", "faulthandler", "-c", script,
            str(this_file.parents[2]), str(this_file), scenario.__name__,
        ],
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Worker scenario exited with {result.returncode}\n{result.stdout}\n{result.stderr}"
    )


def _quit_during_interactive_check():
    from PySide6.QtWidgets import QApplication, QMessageBox
    from sessionprepgui import settings
    from sessionprepgui.mainwindow import SessionPrepWindow

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    gate = _WorkerGate()
    exited = Event()
    created = []
    prompts = []

    class Adapter:
        name = "Offline adapter"

        def check_connectivity(self):
            gate.pause()
            exited.set()
            return False, "Disconnected while quitting"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "get_app_dir", lambda: str(Path.cwd() / "prefs"))
        patch.setattr(QMessageBox, "warning", lambda *args: prompts.append(args[1]))
        window = SessionPrepWindow()
        window.show()
        window.daw_controller.run_daw_check_then(Adapter(), lambda: created.append(True))
        _wait_until(lambda: gate.delivered)
        try:
            accepted = window.close()
            assert not accepted or exited.is_set(), "Window accepted close with a running DAW check"
        finally:
            gate.release.set()
        _wait_until(lambda: exited.is_set() and not window.isVisible())
        assert not created and not prompts, "A completed check triggered a new action after Quit"


def test_quit_owns_slow_interactive_daw_worker_until_its_thread_exits(tmp_path):
    _run_isolated(_quit_during_interactive_check, tmp_path)


def test_run_single_waits_for_both_worker_threads(tmp_path):
    _run_isolated(_single_worker_lifetime, tmp_path)


def test_run_batch_continues_after_connection_and_transfer_failures(tmp_path):
    _run_isolated(_batch_continues_after_failures, tmp_path)


def test_confirmed_quit_waits_for_active_batch_item_without_starting_the_next(tmp_path):
    _run_isolated(_quit_after_current_batch_item, tmp_path)


def test_batch_manager_parents_to_window_and_reads_state_config(_font_app, tmp_path, monkeypatch):
    class _FakeWindow(QObject):
        def __init__(self, cfg):
            super().__init__()
            self.state = SimpleNamespace(config=cfg)

    custom_colors = [{"name": "Neon", "argb": "#ff00ff00"}]
    cfg = {
        "app": {"phase1_output_folder": "custom_phase1_out"},
        "colors": custom_colors,
    }
    captured_folders: list[str] = []
    monkeypatch.setattr(
        batch,
        "restore_layout_sources",
        lambda state_dict, folder: captured_folders.append(folder) or None,
    )
    win = _FakeWindow(cfg)
    manager = batch.BatchManager(win)
    try:
        assert manager.parent() is win
        src_dir = tmp_path / "proj"
        src_dir.mkdir()
        item = BatchItem(
            id="rehydrate_test",
            project_name="Proj",
            daw_processor_id="dawproject_0",
            daw_processor_name="DAWproject",
            output_path=str(tmp_path / "out.dawproject"),
            session_state={
                "project_name": "Proj",
                "source_dir": str(src_dir),
                "session_config": {},
                "tracks": [],
            },
        )
        session = manager._rehydrate_session(item.session_state)
        assert session.config["gui"]["colors"] == custom_colors
        assert captured_folders == ["custom_phase1_out"]
    finally:
        win.deleteLater()
        _font_app.processEvents()
