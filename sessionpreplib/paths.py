"""Leaf path and shared default constants with zero intra-package dependencies."""

from __future__ import annotations

import os
import platform

DEFAULT_STEREO_WINDOW_MS = 250


def get_app_dir() -> str:
    """Return the OS-specific configuration directory for SessionPrep."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA")
        if not base:
            base = os.path.expanduser("~")
        return os.path.join(base, "sessionprep")
    if system == "Darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "sessionprep",
        )
    # Linux / BSD / …
    base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "sessionprep")
