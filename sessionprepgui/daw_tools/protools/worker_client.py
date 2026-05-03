"""Qt client for the persistent Pro Tools worker process."""

from __future__ import annotations

import json
import logging
import os
import platform
import plistlib
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, QProcess, QTimer, Signal

from .connection_common import connection_timeout_message


WorkerCallback = Callable[[dict[str, Any]], None]
log = logging.getLogger(__name__)
_LOG_TEXT_LIMIT = 4000
_STDERR_SUMMARY_LINES = 8
_WORKER_LOG_RE = re.compile(
    r"^\[worker (?P<level>[A-Z]+)\] (?P<logger>[^:]+): (?P<message>.*)$"
)
_WORKER_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _tail_text(text: str, limit: int = _LOG_TEXT_LIMIT) -> str:
    """Return a bounded suffix for diagnostics."""
    if len(text) <= limit:
        return text
    return f"... <truncated {len(text) - limit} chars> ..." + text[-limit:]


def _complete_stderr_lines(partial: str, text: str) -> tuple[list[str], str]:
    """Split stderr text into complete lines plus a partial trailing line."""
    combined = partial + text
    if not combined:
        return [], ""
    chunks = combined.splitlines(keepends=True)
    if chunks and not chunks[-1].endswith(("\n", "\r")):
        partial = chunks.pop()
    else:
        partial = ""
    return [chunk.rstrip("\r\n") for chunk in chunks if chunk.rstrip("\r\n")], partial


def _stderr_summary(text: str, line_count: int = _STDERR_SUMMARY_LINES) -> str:
    """Return a compact single-line summary of recent stderr lines."""
    lines = [
        _format_stderr_summary_line(line.strip())
        for line in text.splitlines()
        if line.strip()
    ]
    if not lines:
        return "<empty>"
    selected = lines[-line_count:]
    summary = " | ".join(selected)
    if len(lines) > line_count:
        summary = f"... {len(lines) - line_count} earlier line(s) | {summary}"
    return _tail_text(summary)


def _format_stderr_summary_line(line: str) -> str:
    """Format one stderr line for inclusion in compact summaries."""
    match = _WORKER_LOG_RE.match(line)
    if not match:
        return line
    return (
        f"{_worker_logger_name(match.group('logger'))} "
        f"{match.group('level')}: {match.group('message')}"
    )


def _worker_logger_name(name: str) -> str:
    """Map worker process logger names into the GUI log namespace."""
    if name == "__main__":
        return "sessionprepgui.daw_tools.protools.worker_process"
    return name


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _qt_application_file_path() -> Path | None:
    """Return Qt's application file path when it points at an executable."""
    try:
        path = QCoreApplication.applicationFilePath()
    except RuntimeError:
        return None
    if not path:
        return None
    candidate = Path(path)
    return candidate if _is_executable_file(candidate) else None


def _macos_bundle_executable_from_path(path: str) -> Path | None:
    """Return the real executable for a macOS .app path inside Contents/MacOS."""
    candidate_path = Path(path)
    macos_dir = candidate_path.parent
    if macos_dir.name != "MacOS" or macos_dir.parent.name != "Contents":
        return None
    if not macos_dir.is_dir():
        return None

    info_plist = macos_dir.parent / "Info.plist"
    if info_plist.is_file():
        try:
            with info_plist.open("rb") as handle:
                bundle_executable = plistlib.load(handle).get("CFBundleExecutable")
        except Exception:
            bundle_executable = None
        if isinstance(bundle_executable, str) and bundle_executable:
            executable = macos_dir / bundle_executable
            if _is_executable_file(executable):
                return executable

    executables = [
        item
        for item in macos_dir.iterdir()
        if _is_executable_file(item) and not item.name.startswith("python")
    ]
    for executable in executables:
        if executable.name.startswith("sessionprep-gui"):
            return executable
    return executables[0] if executables else None


def _compiled_entrypoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    qt_application_path = _qt_application_file_path()
    if qt_application_path is not None:
        candidates.append(qt_application_path)

    argv0 = Path(sys.argv[0]) if sys.argv else None
    if argv0 is not None and argv0.is_absolute() and _is_executable_file(argv0):
        candidates.append(argv0)

    sys_executable = Path(sys.executable)
    if _is_executable_file(sys_executable):
        candidates.append(sys_executable)

    bundle_executable = _macos_bundle_executable_from_path(sys.executable)
    if bundle_executable is not None:
        candidates.append(bundle_executable)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _should_launch_current_app_as_worker() -> bool:
    return (
        bool(getattr(sys, "frozen", False))
        or "__compiled__" in globals()
        or not os.path.exists(sys.executable)
        or _macos_bundle_executable_from_path(sys.executable) is not None
    )


def _worker_launch_command() -> tuple[str, list[str], str]:
    """Return program, args, and mode for launching the Pro Tools worker."""
    if _should_launch_current_app_as_worker():
        candidates = _compiled_entrypoint_candidates()
        if candidates:
            return str(candidates[0]), ["--ptsl-worker"], "compiled-app"
        return sys.executable, ["--ptsl-worker"], "compiled-app-unresolved"
    return (
        sys.executable,
        ["-m", "sessionprepgui.daw_tools.protools.worker_process"],
        "python-module",
    )


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
        self._shutdown_request_id = 0
        self._pending: dict[int, _PendingRequest] = {}
        self._stdout = JsonLineBuffer()
        self._stderr = ""
        self._stderr_partial = ""
        self._stopping = False
        self._connected = False

    @property
    def stderr_text(self) -> str:
        return self._stderr

    def start(self):
        if self._process is not None:
            log.debug(
                "Pro Tools worker start skipped: existing process state=%s",
                self._process_state_text(),
            )
            return
        self._stopping = False
        process = QProcess(self)
        program, args, launch_mode = _worker_launch_command()
        process.setProgram(program)
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.started.connect(self._on_started)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._on_error)
        process.finished.connect(self._on_finished)
        self._process = process
        log.debug(
            "Starting Pro Tools worker: program=%r args=%r frozen=%s "
            "compiled=%s launch_mode=%s platform=%s cwd=%r argv=%r "
            "sys_executable=%r candidates=%r exists=%s executable=%s",
            process.program(),
            process.arguments(),
            bool(getattr(sys, "frozen", False)),
            "__compiled__" in globals(),
            launch_mode,
            platform.platform(),
            os.getcwd(),
            sys.argv,
            sys.executable,
            [str(candidate) for candidate in _compiled_entrypoint_candidates()],
            os.path.exists(process.program()),
            os.access(process.program(), os.X_OK),
        )
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
        log.debug(
            "Queued Pro Tools worker request: id=%s method=%s timeout_ms=%s "
            "process_state=%s",
            request_id,
            method,
            timeout_ms,
            self._process_state_text(),
        )

        if self._process is None:
            log.error("Pro Tools worker request failed: process object is missing")
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
            log.error(
                "Pro Tools worker request failed: process is not running; "
                "stderr_summary=%s",
                _stderr_summary(self._stderr),
            )
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
        log.debug(
            "Stopping Pro Tools worker: process_state=%s pending=%s connected=%s",
            self._process_state_text(),
            self._pending_summary(),
            self._connected,
        )
        self._stopping = True
        self._fail_pending(
            "Disconnected from Pro Tools.",
            "Disconnected",
            "",
        )
        process = self._process
        if process is None:
            return

        state = process.state()
        if state == QProcess.NotRunning:
            self._process = None
            return

        if state == QProcess.Running:
            self._send_shutdown_request(process)

        QTimer.singleShot(
            750,
            lambda proc=process: self._terminate_process_if_current(proc),
        )
        QTimer.singleShot(
            2500,
            lambda proc=process: self._kill_process_if_current(proc),
        )

    def _on_started(self):
        process_id = self._process.processId() if self._process is not None else None
        log.debug(
            "Pro Tools worker started: pid=%s pending=%s",
            process_id,
            self._pending_summary(),
        )
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
        log.debug(
            "Sending Pro Tools worker request: id=%s method=%s bytes=%s",
            request_id,
            pending.method,
            len(data),
        )
        self._process.write(data)

    def _send_shutdown_request(self, process: QProcess):
        self._shutdown_request_id -= 1
        payload = {
            "id": self._shutdown_request_id,
            "method": "disconnect",
            "params": {},
        }
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        log.debug(
            "Sending Pro Tools worker shutdown request: id=%s bytes=%s",
            self._shutdown_request_id,
            len(data),
        )
        process.write(data)
        process.closeWriteChannel()

    def _terminate_process_if_current(self, process: QProcess):
        if self._process is not process or process.state() == QProcess.NotRunning:
            return
        log.debug(
            "Terminating Pro Tools worker after graceful shutdown timeout: "
            "state=%s",
            process.state(),
        )
        process.terminate()

    def _kill_process_if_current(self, process: QProcess):
        if self._process is not process or process.state() == QProcess.NotRunning:
            return
        log.warning(
            "Killing Pro Tools worker after terminate timeout: state=%s",
            process.state(),
        )
        process.kill()

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
            log.exception(
                "Invalid Pro Tools worker stdout: error=%s text_tail=%r",
                exc,
                _tail_text(text),
            )
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
                if pending is None and request_id <= 0:
                    log.debug(
                        "Received Pro Tools worker shutdown response: id=%s ok=%s",
                        request_id,
                        message.get("ok"),
                    )
                    continue
                log.debug(
                    "Received Pro Tools worker response: id=%s method=%s ok=%s",
                    request_id,
                    pending.method if pending is not None else "<unknown>",
                    message.get("ok"),
                )
                if pending is not None and pending.method == "connect" and message.get("ok"):
                    self._connected = True
                self._complete(request_id, message)

    def _read_stderr(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )
        self._append_stderr_text(text)

    def _on_error(self, error_code):
        if self._stopping:
            log.debug(
                "Ignoring Pro Tools worker process error during stop: "
                "code=%s state=%s",
                error_code,
                self._process_state_text(),
            )
            return
        error_string = (
            self._process.errorString()
            if self._process is not None else ""
        )
        log.error(
            "Pro Tools worker process error: code=%s error_string=%r "
            "state=%s pending=%s stderr_summary=%s",
            error_code,
            error_string,
            self._process_state_text(),
            self._pending_summary(),
            _stderr_summary(self._stderr),
        )
        error = self._stderr or "Could not start the Pro Tools worker process."
        title = "Connection failed"
        hint = "Check the SessionPrep installation, then click Connect to try again."
        self._fail_pending(
            error,
            title,
            hint,
        )
        self.worker_failed.emit(error, title, hint)

    def _on_finished(self, exit_code, exit_status):
        self._flush_new_stderr_lines(force=True)
        log.debug(
            "Pro Tools worker finished: exit_code=%s exit_status=%s "
            "stopping=%s connected=%s pending=%s",
            exit_code,
            exit_status,
            self._stopping,
            self._connected,
            self._pending_summary(),
        )
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
        log.warning(
            "Pro Tools worker request timed out: id=%s method=%s "
            "pending=%s stderr_summary=%s",
            request_id,
            method,
            self._pending_summary(),
            _stderr_summary(self._stderr),
        )
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
        log.debug(
            "Completing Pro Tools worker request: id=%s method=%s ok=%s",
            request_id,
            pending.method,
            response.get("ok"),
        )
        pending.callback(response)

    def _fail_pending(self, error: str, title: str, hint: str):
        if self._pending:
            log.debug(
                "Failing pending Pro Tools worker requests: pending=%s "
                "title=%r error_summary=%s",
                self._pending_summary(),
                title,
                _stderr_summary(error),
            )
        response = self._failure(error, title, hint)
        for request_id in list(self._pending):
            self._complete(request_id, response)

    def _flush_new_stderr_lines(self, *, force: bool = False):
        if force and self._stderr_partial:
            self._log_worker_stderr_line(self._stderr_partial)
            self._stderr_partial = ""

    def _append_stderr_text(self, text: str):
        self._stderr += text
        lines, self._stderr_partial = _complete_stderr_lines(
            self._stderr_partial,
            text,
        )
        for line in lines:
            self._log_worker_stderr_line(line)

    def _log_worker_stderr_line(self, line: str):
        match = _WORKER_LOG_RE.match(line)
        if not match:
            log.debug("Pro Tools worker stderr: %s", line)
            return

        level = _WORKER_LOG_LEVELS.get(match.group("level"), logging.DEBUG)
        worker_logger = logging.getLogger(_worker_logger_name(match.group("logger")))
        worker_logger.log(level, "%s", match.group("message"))

    def _pending_summary(self) -> str:
        if not self._pending:
            return "<none>"
        return ", ".join(
            f"{request_id}:{pending.method}"
            for request_id, pending in sorted(self._pending.items())
        )

    def _process_state_text(self) -> str:
        if self._process is None:
            return "<none>"
        return str(self._process.state())

    @staticmethod
    def _failure(error: str, title: str, hint: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": error,
            "title": title,
            "hint": hint,
        }
