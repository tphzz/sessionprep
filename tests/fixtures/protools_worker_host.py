"""Disposable JSON-line worker with the production utility methods and an in-memory host."""

import json
import os
import sys
import time

from sessionprepgui.daw_tools.protools.worker_process import ProToolsWorker
from sessionpreplib.daw_processors import ptsl_helpers


def track(identifier, name, selection, visibility):
    return {"id": identifier, "name": name, "type": "TT_Audio",
            "track_attributes": {"is_selected": selection, "is_hidden": visibility}}


tracks = [
    track("one", "Kick", "SetExplicitly", "None"),
    track("two", "Snare", "TAState_SetExplicitly", "TAState_None"),
    track("three", "Hats", "SetImplicitly", "None"),
    track("four", "Hidden", "TAState_None", "TAState_SetExplicitly"),
]
writes = []


def transport(engine, command, body, **kwargs):
    if command == ptsl_helpers.command_id("CId_GetTrackList") or command == "CId_GetTrackList":
        return {"track_list": tracks}
    if command == "CId_GetColorPalette":
        return {"color_list": ["#ff000000", "#ff204080", "#ffffcc00"]}
    if command in ("CId_SetTrackColor", "CId_SetTrackHeight"):
        writes.append([command, body])
        return {}
    raise AssertionError(command)


def reply(request, result):
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)


def main():
    worker = ProToolsWorker()
    worker._engine = object()
    ptsl_helpers.run_command = transport
    deferred = None
    exit_on_inventory = False
    for line in sys.stdin:
        request = json.loads(line)
        method, params = request["method"], request["params"]
        if method == "disconnect":
            reply(request, None)
            return
        if method == "hold":
            deferred = request
        elif method == "release":
            reply(request, "second")
            reply(deferred, "first")
        elif method == "delay":
            time.sleep(.18)
            reply(request, "late")
        elif method == "arm_exit":
            exit_on_inventory = True
            reply(request, True)
        elif method == "get_track_list" and exit_on_inventory:
            sys.stdout.write('{"unfinished":')
            sys.stdout.flush()
            os._exit(11)
        elif method == "writes":
            reply(request, writes)
        else:
            reply(request, worker.handle(method, params))


if __name__ == "__main__":
    main()
