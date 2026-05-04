from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from sessionpreplib import logging_setup


def _restore_logging(root, old_level, old_disable, old_handlers,
                     old_handler_levels):
    logging.disable(old_disable)
    root.setLevel(old_level)
    for handler in list(root.handlers):
        if handler not in old_handlers:
            root.removeHandler(handler)
            handler.close()
    for handler, level in old_handler_levels.items():
        handler.setLevel(level)
    root.handlers[:] = old_handlers


def _sessionprep_handlers():
    return [
        handler for handler in logging.getLogger().handlers
        if getattr(handler, "baseFilename", None) == logging_setup.get_log_path()
    ]


def test_log_path_uses_app_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "get_app_dir", lambda: str(tmp_path))

    assert logging_setup.get_log_dir() == str(tmp_path)
    assert logging_setup.get_log_path() == str(tmp_path / "sessionprep.log")


def test_runtime_log_level_updates_root_and_handlers(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "get_app_dir", lambda: str(tmp_path))
    root = logging.getLogger()
    old_level = root.level
    old_disable = logging.root.manager.disable
    old_handlers = list(root.handlers)
    old_handler_levels = {handler: handler.level for handler in old_handlers}

    try:
        logging_setup.set_runtime_log_level(logging.DEBUG)

        assert logging_setup.get_current_log_level() == logging.DEBUG
        assert root.level == logging.DEBUG
        assert logging.root.manager.disable == logging.NOTSET
        assert any(
            getattr(handler, "baseFilename", None) == logging_setup.get_log_path()
            for handler in root.handlers
        )

        logging_setup.set_runtime_log_level(None)

        assert logging_setup.get_current_log_level() is None
        assert logging.root.manager.disable >= logging.CRITICAL
    finally:
        _restore_logging(
            root, old_level, old_disable, old_handlers, old_handler_levels)


def test_file_handler_uses_bounded_rotating_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "get_app_dir", lambda: str(tmp_path))
    root = logging.getLogger()
    old_level = root.level
    old_disable = logging.root.manager.disable
    old_handlers = list(root.handlers)
    old_handler_levels = {handler: handler.level for handler in old_handlers}

    try:
        logging_setup.set_runtime_log_level(logging.INFO)
        handlers = _sessionprep_handlers()

        assert len(handlers) == 1
        handler = handlers[0]
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == logging_setup._MAX_BYTES
        assert handler.backupCount == logging_setup._BACKUP_COUNT
        assert handler.baseFilename == logging_setup.get_log_path()
    finally:
        _restore_logging(
            root, old_level, old_disable, old_handlers, old_handler_levels)


def test_rotating_file_handler_rolls_over_and_bounds_backups(
        monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "get_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(logging_setup, "_MAX_BYTES", 180)
    root = logging.getLogger()
    old_level = root.level
    old_disable = logging.root.manager.disable
    old_handlers = list(root.handlers)
    old_handler_levels = {handler: handler.level for handler in old_handlers}

    try:
        logging_setup.set_runtime_log_level(logging.INFO)
        logger = logging.getLogger("tests.logging_setup")
        for index in range(80):
            logger.info("rollover line %03d %s", index, "x" * 80)
        for handler in _sessionprep_handlers():
            handler.flush()

        assert (tmp_path / "sessionprep.log").exists()
        assert (tmp_path / "sessionprep.log.1").exists()
        log_files = sorted(path.name for path in tmp_path.glob("sessionprep.log*"))

        assert len(log_files) <= logging_setup._BACKUP_COUNT + 1
        assert "sessionprep.log.6" not in log_files
    finally:
        _restore_logging(
            root, old_level, old_disable, old_handlers, old_handler_levels)
