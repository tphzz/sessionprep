"""Centralized logging setup for SessionPrep.

Call :func:`setup_logging` once at application startup (CLI or GUI) to
configure the root logger with:

* A **RotatingFileHandler** writing to ``sessionprep.log`` in the
  OS-specific app-data directory (always active, append mode).
* A **StreamHandler** writing to *stderr* (only when a terminal is
  attached — suppressed in compiled / GUI-only builds).

Log level is controlled via the ``SP_LOG_LEVEL`` environment variable:

* ``DEBUG``, ``INFO`` (default), ``WARNING``, ``ERROR``, ``CRITICAL``
* ``NONE`` — disable logging entirely.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .config import get_app_dir

LOG_FILENAME = "sessionprep.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_NONE_LEVEL = logging.CRITICAL + 10

_initialized = False


def get_log_dir() -> str:
    """Return the directory where SessionPrep writes its log file."""
    return get_app_dir()


def get_log_path() -> str:
    """Return the active SessionPrep log file path."""
    return os.path.join(get_log_dir(), LOG_FILENAME)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)


def _is_sessionprep_file_handler(handler: logging.Handler) -> bool:
    return (
        isinstance(handler, RotatingFileHandler)
        and os.path.abspath(getattr(handler, "baseFilename", ""))
        == os.path.abspath(get_log_path())
    )


def ensure_file_logging() -> None:
    """Ensure the rotating SessionPrep file handler exists."""
    root = logging.getLogger()
    if any(_is_sessionprep_file_handler(handler) for handler in root.handlers):
        return

    try:
        os.makedirs(get_log_dir(), exist_ok=True)
        fh = RotatingFileHandler(
            get_log_path(),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        return
    fh.setFormatter(_make_formatter())
    fh.setLevel(root.level)
    root.addHandler(fh)


def get_current_log_level() -> int | None:
    """Return the current root log level, or None when logging is off."""
    if logging.root.manager.disable >= logging.CRITICAL:
        return None
    return logging.getLogger().level


def set_runtime_log_level(level: int | None) -> None:
    """Set the process-wide log level for the current runtime only."""
    root = logging.getLogger()
    if level is None:
        logging.disable(logging.CRITICAL)
        root.setLevel(_NONE_LEVEL)
        for handler in root.handlers:
            handler.setLevel(_NONE_LEVEL)
        return

    logging.disable(logging.NOTSET)
    root.setLevel(level)
    ensure_file_logging()
    for handler in root.handlers:
        handler.setLevel(level)


def setup_logging(level: int | None = None) -> None:
    """Configure the root logger for the entire application.

    Safe to call more than once — subsequent calls are no-ops.

    Parameters
    ----------
    level : int or None
        Explicit log level override.  When *None* the level is read
        from the ``SP_LOG_LEVEL`` environment variable.  If not set,
        ``INFO`` is used.  Set to ``NONE`` to disable logging.
    """
    global _initialized  # noqa: PLW0603  # pylint: disable=global-statement
    if _initialized:
        return
    _initialized = True

    if level is None:
        level = _level_from_env()

    if level == _NONE_LEVEL:  # NONE sentinel
        logging.disable(logging.CRITICAL)
        return

    root = logging.getLogger()
    logging.disable(logging.NOTSET)
    root.setLevel(level)

    formatter = _make_formatter()

    # ── File handler (always) ────────────────────────────────────────
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_path = get_log_path()

    try:
        fh = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError:
        # Cannot write to log file (permissions, etc.) — continue
        # with stderr only.
        pass

    # ── Stderr handler (only when a terminal is attached) ────────────
    if _has_stderr():
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # Visual separator so restarts are easy to find in long log files
    root.info("===================================================================")


def _level_from_env() -> int:
    """Determine log level from ``SP_LOG_LEVEL``."""
    raw = os.environ.get("SP_LOG_LEVEL", "").strip().upper()
    if raw == "NONE":
        return _NONE_LEVEL  # sentinel: skip handler setup
    if raw:
        numeric = getattr(logging, raw, None)
        if isinstance(numeric, int):
            return numeric
    return logging.INFO


def _has_stderr() -> bool:
    """Return True if stderr is connected to a real stream."""
    try:
        return sys.stderr is not None and hasattr(sys.stderr, "write")
    except Exception:
        return False
