# pylint: disable=cyclic-import
"""
SessionPrep GUI — PySide6 front-end for session analysis.

Usage:
    uv run python sessionprep-gui.py
"""

from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys


_GUI_RELAUNCHED = "_SESSIONPREP_GUI_RELAUNCHED"


def _bootstrap_gui_dependencies() -> None:
    """Relaunch a source run with the GUI extra if its imports are missing."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return
    if find_spec("PySide6") is not None and find_spec("sounddevice") is not None:
        return

    # uv passes the path to its own executable to child processes.
    uv = os.environ.get("UV")
    script = Path(__file__).resolve()
    if (not uv or os.environ.get(_GUI_RELAUNCHED)
            or not script.with_name("pyproject.toml").is_file()):
        raise SystemExit(
            "SessionPrep GUI requires PySide6 and sounddevice. From the source "
            "checkout, run `uv run --extra gui python sessionprep-gui.py`."
        )

    env = os.environ.copy()
    env[_GUI_RELAUNCHED] = "1"
    # Reuse the environment selected by the outer uv run, including overrides.
    try:
        result = subprocess.run(
            [uv, "run", "--project", str(script.parent), "--active",
             "--extra", "gui", "python", str(script), *sys.argv[1:]],
            env=env, check=False,
        )
    except OSError as exc:
        raise SystemExit(f"Could not start uv to install GUI dependencies: {exc}") from exc
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    _bootstrap_gui_dependencies()

    if "--ptsl-worker" in sys.argv:
        from sessionprepgui.daw_tools.protools.worker_process import main as worker_main

        sys.exit(worker_main())

    if "--ptsl-worker-self-test" in sys.argv:
        from sessionprepgui.daw_tools.protools.worker_process import self_test

        sys.exit(self_test())

    from sessionprepgui import main

    main()
