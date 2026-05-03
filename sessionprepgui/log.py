"""Lightweight debug logging for SessionPrep GUI.

Usage::

    from sessionprepgui.log import dbg

    dbg("Batch job created: {job_id}")
    dbg("Spectrogram cache invalidated")

This is a thin convenience wrapper around Python's standard
:mod:`logging` module.  Each call resolves the calling class or
module automatically and delegates to ``logging.getLogger(name).debug()``.

The log level and handlers are configured once at startup via
:func:`sessionpreplib.logging_setup.setup_logging`.
"""

from __future__ import annotations

from sessionpreplib.log import dbg

__all__ = ["dbg"]
