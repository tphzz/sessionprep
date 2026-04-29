"""
Diagnostic script for probing Pro Tools track I/O via PTSL.

Run via:
    uv run python tests/exploration/probe_track_io.py Kick

This script is intentionally exploratory.  PTSL 2026.04 exposes a direct
command for main output assignments, but the proto does not expose an
equivalent direct command for track input assignments.  To investigate input
routing, this script also exports Session Info for the selected track and
prints lines likely to contain input/output routing data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from ptsl import Engine
except ImportError:
    print("Error: py-ptsl not installed.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
PTSL_HELPERS_PATH = REPO_ROOT / "sessionpreplib" / "daw_processors" / "ptsl_helpers.py"

spec = importlib.util.spec_from_file_location("ptsl_helpers", PTSL_HELPERS_PATH)
if spec is None or spec.loader is None:
    print(f"Error: could not load {PTSL_HELPERS_PATH}")
    sys.exit(1)
ptsl_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptsl_helpers)


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _find_track_by_name(track_list, track_name: str):
    for track in track_list:
        if track.name == track_name:
            return track
    return None


def _export_selected_track_session_info(engine) -> str:
    resp = ptsl_helpers.run_command(
        engine,
        ptsl_helpers.command_id("CId_ExportSessionInfoAsText"),
        {
            "include_file_list": False,
            "include_clip_list": False,
            "include_markers": False,
            "include_plugin_list": False,
            "include_track_edls": True,
            "show_sub_frames": False,
            "include_user_timestamps": False,
            "track_list_type": "TListType_SelectedTracksOnly",
            "fade_handling_type": "FHType_ShowCrossfades",
            "location_type": "TLType_Samples",
            "text_as_file_format": "TFFormat_UTF8",
            "output_type": "ESIOType_String",
        },
    )
    return (resp or {}).get("session_info", "")


def _print_likely_io_lines(session_info: str) -> None:
    patterns = re.compile(
        r"\b(input|output|bus|path|i/o|io|interface|routing)\b",
        re.IGNORECASE,
    )
    matches = [
        (idx, line.rstrip())
        for idx, line in enumerate(session_info.splitlines(), start=1)
        if patterns.search(line)
    ]

    print()
    print("Session Info lines likely related to I/O:")
    if not matches:
        print("  No obvious input/output lines found in exported Session Info.")
        return

    for idx, line in matches:
        print(f"  {idx:>5}: {line}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a Pro Tools track and probe its I/O routing via PTSL."
    )
    parser.add_argument(
        "track_name",
        nargs="?",
        default="Kick",
        help="Exact Pro Tools track name to select and inspect. Default: Kick",
    )
    args = parser.parse_args()

    print("Connecting to Pro Tools Engine...")
    engine = None
    try:
        engine = Engine(
            company_name="SessionPrep Diagnostic",
            application_name="probe_track_io",
            address="localhost:31416",
        )

        session_name = engine.session_name()
        if not session_name:
            print("No active Pro Tools session open. Exiting.")
            return 1

        print(f"Connected to session: '{session_name}'")

        track = _find_track_by_name(engine.track_list(), args.track_name)
        if track is None:
            print(f"Track not found: {args.track_name!r}")
            print("Available tracks:")
            for candidate in engine.track_list():
                print(f"  - {candidate.name}")
            return 1

        print()
        print("Matched track:")
        print(f"  Name: {track.name}")
        print(f"  ID:   {track.id}")
        print(f"  Type: {track.type}")

        print()
        print(f"Selecting track {track.name!r}...")
        select_resp = ptsl_helpers.run_command(
            engine,
            ptsl_helpers.command_id("CId_SelectTracksByName"),
            {
                "track_names": [track.name],
                "selection_mode": "SMode_Replace",
                "pagination_request": {"limit": 0, "offset": 0},
            },
        )
        print("SelectTracksByName response:")
        print(_json_dump(select_resp or {}))

        print()
        print("Querying direct main output assignment IDs...")
        output_ids = ptsl_helpers.get_track_main_output_assignments(
            engine,
            [track.id],
        )
        print("GetTrackMainOutputAssignments response:")
        print(_json_dump({"signalpath_ids": output_ids}))

        print()
        print("Exporting Session Info for selected track...")
        session_info = _export_selected_track_session_info(engine)
        print(f"Exported Session Info length: {len(session_info)} characters")
        _print_likely_io_lines(session_info)

        print()
        print("Raw selected-track Session Info:")
        print(session_info if session_info else "  <empty>")

        return 0
    except Exception as exc:
        print(f"Unexpected error during script execution: {exc}")
        return 1
    finally:
        if engine is not None:
            print()
            print("Closing engine connection.")
            engine.close()


if __name__ == "__main__":
    sys.exit(main())
