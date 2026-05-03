# pylint: disable=cyclic-import
"""
SessionPrep GUI — PySide6 front-end for session analysis.

Usage:
    python sessionprep-gui.py
    uv run python sessionprep-gui.py

Requires: PySide6 (install via `uv pip install PySide6`)
"""

if __name__ == "__main__":
    import sys

    if "--ptsl-worker" in sys.argv:
        from sessionprepgui.daw_tools.protools.worker_process import main as worker_main

        sys.exit(worker_main())

    if "--ptsl-worker-self-test" in sys.argv:
        from sessionprepgui.daw_tools.protools.worker_process import self_test

        sys.exit(self_test())

    from sessionprepgui import main

    main()
