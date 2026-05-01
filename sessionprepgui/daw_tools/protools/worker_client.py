"""Qt client for the persistent Pro Tools worker process."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from .connection_common import connection_timeout_message


WorkerCallback = Callable[[dict[str, Any]], None]


class JsonLineBuffer:
    """Incrementally parse newline-delimited JSON messages."""

    def __init__(self):
        self._buffer = ""

    def feed(self, text: str) -> list[dict[str, Any]]:
        self._buffer += text
        messages: list[dict[str, Any]] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                messages.append(payload)
        return messages


@dataclass
class _PendingRequest:
    method: str
    payload: dict[str, Any]
    callback: WorkerCallback
    timer: QTimer
    sent: bool = False


class ProToolsWorkerClient(QObject):
    """Asynchronous request client for the Pro Tools worker process."""

    worker_failed = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._next_id = 1
        self._pending: dict[int, _PendingRequest] = {}
        self._stdout = JsonLineBuffer()
        self._stderr = ""
        self._stopping = False
        self._connected = False

    @property
    def stderr_text(self) -> str:
        return self._stderr

    def start(self):
        if self._process is not None:
            return
        self._stopping = False
        process = QProcess(self)
        process.setProgram(sys.executable)
        if getattr(sys, "frozen", False):
            process.setArguments(["--ptsl-worker"])
        else:
            process.setArguments(
                ["-m", "sessionprepgui.daw_tools.protools.worker_process"]
            )
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.started.connect(self._on_started)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._on_error)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        callback: WorkerCallback,
        timeout_ms: int,
    ) -> int:
        self.start()
        request_id = self._next_id
        self._next_id += 1

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(timeout_ms)
        payload = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        timer.timeout.connect(lambda rid=request_id: self._on_timeout(rid))
        self._pending[request_id] = _PendingRequest(
            method=method,
            payload=payload,
            callback=callback,
            timer=timer,
        )

        if self._process is None:
            self._complete(
                request_id,
                self._failure(
                    "Worker process is not available.",
                    "Connection failed",
                    "Check the SessionPrep installation, then click Connect to try again.",
                ),
            )
        elif self._process.state() == QProcess.Running:
            self._send_pending(request_id)
        elif self._process.state() == QProcess.NotRunning:
            self._complete(
                request_id,
                self._failure(
                    "Worker process is not running.",
                    "Connection failed",
                    "Check the SessionPrep installation, then click Connect to try again.",
                ),
            )
        else:
            # The request will be written from _on_started.
            pass
        return request_id

    def stop(self):
        self._stopping = True
        self._fail_pending(
            "Disconnected from Pro Tools.",
            "Disconnected",
            "",
        )
        if self._process is not None:
            self._process.kill()
            self._process = None

    def _on_started(self):
        for request_id in list(self._pending):
            self._send_pending(request_id)

    def _send_pending(self, request_id: int):
        pending = self._pending.get(request_id)
        if pending is None or pending.sent or self._process is None:
            return
        data = (
            json.dumps(pending.payload, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        pending.sent = True
        pending.timer.start()
        self._process.write(data)

    def _read_stdout(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        try:
            messages = self._stdout.feed(text)
        except json.JSONDecodeError as exc:
            error = f"Invalid worker response: {exc}"
            title = "Connection failed"
            hint = "Restart the Pro Tools connection and try again."
            self._fail_pending(
                error,
                title,
                hint,
            )
            self.worker_failed.emit(error, title, hint)
            self.stop()
            return
        for message in messages:
            request_id = message.get("id")
            if isinstance(request_id, int):
                pending = self._pending.get(request_id)
                if pending is not None and pending.method == "connect" and message.get("ok"):
                    self._connected = True
                self._complete(request_id, message)

    def _read_stderr(self):
        if self._process is None:
            return
        self._stderr += bytes(self._process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )

    def _on_error(self, _error):
        error = self._stderr or "Could not start the Pro Tools worker process."
        title = "Connection failed"
        hint = "Check the SessionPrep installation, then click Connect to try again."
        self._fail_pending(
            error,
            title,
            hint,
        )
        self.worker_failed.emit(error, title, hint)

    def _on_finished(self, _exit_code, _exit_status):
        if not self._stopping:
            error = self._stderr or "The Pro Tools worker process exited."
            self._fail_pending(
                error,
                "Connection failed",
                "Start Pro Tools and click Connect to try again.",
            )
            if self._connected:
                self.worker_failed.emit(
                    error,
                    "Connection failed",
                    "Start Pro Tools and click Connect to try again.",
                )
        self._process = None
        self._connected = False

    def _on_timeout(self, request_id: int):
        pending = self._pending.get(request_id)
        if pending is None:
            return
        error, title, hint = connection_timeout_message()
        method = pending.method
        self._complete(request_id, self._failure(error, title, hint))
        if method != "connect" and self._connected:
            self.worker_failed.emit(error, title, hint)
        self.stop()

    def _complete(self, request_id: int, response: dict[str, Any]):
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        pending.timer.stop()
        pending.timer.deleteLater()
        pending.callback(response)

    def _fail_pending(self, error: str, title: str, hint: str):
        response = self._failure(error, title, hint)
        for request_id in list(self._pending):
            self._complete(request_id, response)

    @staticmethod
    def _failure(error: str, title: str, hint: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": error,
            "title": title,
            "hint": hint,
        }
