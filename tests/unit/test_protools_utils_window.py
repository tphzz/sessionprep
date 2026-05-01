from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from sessionprepgui.daw_tools.protools.window import _connection_failure_message


def test_connection_failure_message_for_grpc_unavailable():
    exc = RuntimeError(
        "StatusCode.UNAVAILABLE: failed to connect to all addresses; "
        "Connection refused"
    )

    title, hint = _connection_failure_message(exc)

    assert title == "Pro Tools not available"
    assert "Start Pro Tools" in hint


def test_connection_failure_message_for_missing_ptsl():
    title, hint = _connection_failure_message(
        ImportError("No module named 'ptsl'")
    )

    assert title == "py-ptsl is not installed"
    assert "dependency" in hint


def test_connection_failure_message_for_unknown_error():
    title, hint = _connection_failure_message(RuntimeError("unexpected"))

    assert title == "Connection failed"
    assert "try again" in hint
