"""Persistent Pro Tools worker process for the GUI utilities."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import traceback
from typing import Any

from sessionpreplib.daw_processors import ptsl_helpers as ptslh

from .connection_common import (
    PTSL_HOST_READY_TIMEOUT_SECONDS,
    connection_failure_message,
    create_ptsl_engine_with_timeout,
)


log = logging.getLogger(__name__)


def _worker_log_level() -> int | None:
    raw = os.environ.get("SP_LOG_LEVEL", "").strip().upper()
    if raw == "NONE":
        return None
    if raw:
        level = getattr(logging, raw, None)
        if isinstance(level, int):
            return level
    return logging.INFO


def _configure_worker_logging() -> None:
    """Configure worker diagnostics on stderr without touching stdout."""
    level = _worker_log_level()
    if level is None:
        logging.disable(logging.CRITICAL)
        return

    logging.disable(logging.NOTSET)
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("[worker %(levelname)s] %(name)s: %(message)s")
    )
    root.handlers.clear()
    root.addHandler(handler)


class ProToolsWorker:
    """Owns the py-ptsl engine outside the Qt GUI process."""

    def __init__(self):
        self._engine = None
        log.debug("Pro Tools worker object created")

    def close(self):
        if self._engine is not None:
            log.debug("Closing Pro Tools worker engine")
            try:
                self._engine.close()
            except Exception:
                log.debug(
                    "Failed to close Pro Tools worker engine:\n%s",
                    traceback.format_exc(),
                )
            self._engine = None

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        log.debug("Handling Pro Tools worker request: method=%s", method)
        if method == "connect":
            if self._engine is None:
                log.debug("Creating Pro Tools PTSL engine")
                self._engine = create_ptsl_engine_with_timeout()
                log.debug("Created Pro Tools PTSL engine")
            log.debug(
                "Waiting for Pro Tools host readiness: timeout=%.1fs",
                PTSL_HOST_READY_TIMEOUT_SECONDS,
            )
            if not ptslh.wait_for_host_ready(
                self._engine,
                timeout=PTSL_HOST_READY_TIMEOUT_SECONDS,
                sleep_time=0.25,
            ):
                log.debug("Pro Tools host readiness check did not complete")
                raise RuntimeError("Pro Tools is still starting.")
            log.debug("Pro Tools host readiness confirmed")
            return {"connected": True}
        if method == "disconnect":
            self.close()
            return {"connected": False}
        if method == "ping":
            self._require_engine()
            return {"connected": True}
        if method == "run_command":
            self._require_engine()
            log.debug(
                "Running Pro Tools command through worker: command=%r",
                params.get("command"),
            )
            return ptslh.run_command(
                self._engine,
                params.get("command"),
                params.get("body") or {},
                batch_job_id=params.get("batch_job_id"),
                progress=int(params.get("progress") or 0),
            )
        if method == "get_color_palette":
            self._require_engine()
            target = params.get("target") or "CPTarget_Tracks"
            log.debug("Fetching Pro Tools color palette: target=%s", target)
            return ptslh.get_color_palette(self._engine, target=target)
        if method == "get_selected_track_names":
            self._require_engine()
            log.debug("Fetching selected Pro Tools track names")
            return ptslh.get_selected_track_names(self._engine)
        if method == "set_track_color":
            self._require_engine()
            log.debug("Setting Pro Tools track color through worker")
            return ptslh.set_track_color(
                self._engine,
                color_index=int(params["color_index"]),
                track_names=params.get("track_names"),
                track_ids=params.get("track_ids"),
            )
        if method == "get_track_list":
            self._require_engine()
            log.debug("Fetching Pro Tools track list")
            resp = ptslh.run_command(
                self._engine,
                "CId_GetTrackList",
                params.get("body") or {"pagination_request": {"limit": 0, "offset": 0}},
            )
            return list((resp or {}).get("track_list", []))
        if method == "set_track_height":
            self._require_engine()
            log.debug("Setting Pro Tools track height through worker")
            return ptslh.set_track_height(
                self._engine,
                params["height"],
                track_names=params.get("track_names"),
                track_ids=params.get("track_ids"),
            )
        raise ValueError(f"Unknown worker method: {method}")

    def _require_engine(self):
        if self._engine is None:
            raise RuntimeError("Not connected to Pro Tools")


def _write_response(payload: dict[str, Any]):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def self_test(*, require_ptsl: bool = True, configure_logging: bool = True) -> int:
    """Run a lightweight worker startup/protocol self-test."""
    if configure_logging:
        _configure_worker_logging()
    log.debug(
        "Pro Tools worker self-test started: executable=%r argv=%r platform=%s cwd=%r",
        sys.executable,
        sys.argv,
        platform.platform(),
        os.getcwd(),
    )
    try:
        if require_ptsl:
            __import__("ptsl")
        payload = {
            "id": 0,
            "ok": True,
            "result": {
                "self_test": True,
                "ptsl_imported": require_ptsl,
            },
        }
        _write_response(payload)
        log.debug("Pro Tools worker self-test completed")
        return 0
    except Exception as exc:
        log.error(
            "Pro Tools worker self-test failed: error=%s\n%s",
            exc,
            traceback.format_exc(),
        )
        _write_response(
            {
                "id": 0,
                "ok": False,
                "error": str(exc),
            }
        )
        return 2


def main() -> int:
    _configure_worker_logging()
    log.debug(
        "Pro Tools worker process started: executable=%r argv=%r platform=%s cwd=%r",
        sys.executable,
        sys.argv,
        platform.platform(),
        os.getcwd(),
    )
    worker = ProToolsWorker()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                request_id = request.get("id")
                log.debug(
                    "Received Pro Tools worker JSON request: id=%r method=%r",
                    request_id,
                    request.get("method"),
                )
                result = worker.handle(
                    str(request.get("method") or ""),
                    request.get("params") or {},
                )
                log.debug(
                    "Completed Pro Tools worker JSON request: id=%r method=%r",
                    request_id,
                    request.get("method"),
                )
                _write_response({"id": request_id, "ok": True, "result": result})
            except Exception as exc:
                log.error(
                    "Pro Tools worker request failed: request_id=%r error=%s\n%s",
                    request.get("id"),
                    exc,
                    traceback.format_exc(),
                )
                title, hint = connection_failure_message(exc)
                _write_response(
                    {
                        "id": request.get("id"),
                        "ok": False,
                        "error": str(exc),
                        "title": title,
                        "hint": hint,
                    }
                )
    finally:
        log.debug("Pro Tools worker process shutting down")
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
