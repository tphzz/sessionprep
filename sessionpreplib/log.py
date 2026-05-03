"""Lightweight debug logging helpers shared by GUI and library code."""

from __future__ import annotations

import inspect
import logging


def _caller_logger() -> logging.Logger:
    """Return a logger named after the calling module/class."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return logging.getLogger("sessionpreplib")
        mod = caller.f_globals.get("__name__", "sessionpreplib")
        return logging.getLogger(mod)
    finally:
        del frame


def dbg(msg: str) -> None:
    """Log a debug message, automatically detecting the caller."""
    _caller_logger().debug(msg)
