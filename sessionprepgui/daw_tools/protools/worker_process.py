"""Persistent Pro Tools worker process for the GUI utilities."""

from __future__ import annotations

import json
import sys
from typing import Any

from sessionpreplib.daw_processors import ptsl_helpers as ptslh

from .connection_common import (
    PTSL_HOST_READY_TIMEOUT_SECONDS,
    connection_failure_message,
    create_ptsl_engine_with_timeout,
)


class ProToolsWorker:
    """Owns the py-ptsl engine outside the Qt GUI process."""

    def __init__(self):
        self._engine = None

    def close(self):
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        if method == "connect":
            if self._engine is None:
                self._engine = create_ptsl_engine_with_timeout()
            if not ptslh.wait_for_host_ready(
                self._engine,
                timeout=PTSL_HOST_READY_TIMEOUT_SECONDS,
                sleep_time=0.25,
            ):
                raise RuntimeError("Pro Tools is still starting.")
            return {"connected": True}
        if method == "disconnect":
            self.close()
            return {"connected": False}
        if method == "ping":
            self._require_engine()
            return {"connected": True}
        if method == "run_command":
            self._require_engine()
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
            return ptslh.get_color_palette(self._engine, target=target)
        if method == "get_selected_track_names":
            self._require_engine()
            return ptslh.get_selected_track_names(self._engine)
        if method == "set_track_color":
            self._require_engine()
            return ptslh.set_track_color(
                self._engine,
                color_index=int(params["color_index"]),
                track_names=params.get("track_names"),
                track_ids=params.get("track_ids"),
            )
        if method == "get_track_list":
            self._require_engine()
            resp = ptslh.run_command(
                self._engine,
                "CId_GetTrackList",
                params.get("body") or {"pagination_request": {"limit": 0, "offset": 0}},
            )
            return list((resp or {}).get("track_list", []))
        if method == "set_track_height":
            self._require_engine()
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


def main() -> int:
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
                result = worker.handle(
                    str(request.get("method") or ""),
                    request.get("params") or {},
                )
                _write_response({"id": request_id, "ok": True, "result": result})
            except Exception as exc:
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
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
