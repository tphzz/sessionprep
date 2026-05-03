from __future__ import annotations

import logging

from sessionpreplib import logging_setup


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
        logging.disable(old_disable)
        root.setLevel(old_level)
        for handler in list(root.handlers):
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler, level in old_handler_levels.items():
            handler.setLevel(level)
        root.handlers[:] = old_handlers
