from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from sessionprepgui.daw_tools.protools.connection_common import (
    connection_button_state,
    connection_failure_message,
    connection_timeout_message,
)
from sessionprepgui.daw_tools.protools.worker_client import JsonLineBuffer
from sessionprepgui.widgets import _aspect_limited_grid_height


def test_connection_failure_message_for_grpc_unavailable():
    exc = RuntimeError(
        "StatusCode.UNAVAILABLE: failed to connect to all addresses; "
        "Connection refused"
    )

    title, hint = connection_failure_message(exc)

    assert title == "Pro Tools not available"
    assert "Start Pro Tools" in hint


def test_connection_failure_message_for_grpc_deadline():
    exc = RuntimeError("StatusCode.DEADLINE_EXCEEDED: deadline exceeded")

    title, hint = connection_failure_message(exc)

    assert title == "Pro Tools not available"
    assert "Start Pro Tools" in hint


def test_connection_failure_message_for_missing_ptsl():
    title, hint = connection_failure_message(
        ImportError("No module named 'ptsl'")
    )

    assert title == "py-ptsl is not installed"
    assert "dependency" in hint


def test_connection_failure_message_for_unknown_error():
    title, hint = connection_failure_message(RuntimeError("unexpected"))

    assert title == "Connection failed"
    assert "try again" in hint


def test_connection_button_state_for_connecting():
    text, color = connection_button_state("connecting")

    assert text == "Pro Tools: Connecting..."
    assert color == "#aaa"


def test_connection_button_state_for_checking():
    text, color = connection_button_state("checking")

    assert text == "Pro Tools: Checking..."
    assert color == "#aaa"


def test_connection_button_state_for_connected():
    text, color = connection_button_state("connected")

    assert text == "Pro Tools: Connected"
    assert color == "#4caf50"


def test_connection_button_state_for_failed():
    text, color = connection_button_state("failed")

    assert text == "Pro Tools: Offline"
    assert color == "#f44336"


def test_connection_timeout_message():
    error, title, hint = connection_timeout_message()

    assert "Timed out" in error
    assert title == "Pro Tools not available"
    assert "finished launching" in hint


def test_json_line_buffer_handles_split_messages():
    buffer = JsonLineBuffer()

    assert buffer.feed('{"id":1,') == []
    messages = buffer.feed('"ok":true}\n')

    assert messages == [{"id": 1, "ok": True}]


def test_json_line_buffer_handles_multiple_messages():
    buffer = JsonLineBuffer()

    messages = buffer.feed('{"id":1,"ok":true}\n{"id":2,"ok":false}\n')

    assert messages == [
        {"id": 1, "ok": True},
        {"id": 2, "ok": False},
    ]


def test_color_grid_aspect_limit_keeps_cells_square_or_wide():
    height = _aspect_limited_grid_height(
        2308,
        columns=23,
        row_count=3,
        cell_height=28,
        max_cell_height_to_width=1.0,
    )

    assert height == 307


def test_color_grid_aspect_limit_never_goes_below_min_cell_height():
    height = _aspect_limited_grid_height(
        300,
        columns=23,
        row_count=3,
        cell_height=28,
        max_cell_height_to_width=1.0,
    )

    assert height == 94


def test_color_grid_minimum_height_uses_min_cell_height():
    height = _aspect_limited_grid_height(
        1,
        columns=23,
        row_count=3,
        cell_height=28,
        max_cell_height_to_width=0.0,
    )

    assert height == 94
