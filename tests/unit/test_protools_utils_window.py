from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from sessionprepgui.daw_tools.protools.connection_common import (
    connection_button_state,
    connection_failure_message,
    connection_timeout_message,
)
from sessionprepgui.daw_tools.protools.track_height_tool import (
    _default_preset,
    _migrate_default_preset_scope,
)
from sessionprepgui.daw_tools.protools.worker_client import (
    JsonLineBuffer,
    ProToolsWorkerClient,
    _complete_stderr_lines,
    _stderr_summary,
    _tail_text,
)
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


def test_worker_stderr_splitter_preserves_partial_line():
    lines, partial = _complete_stderr_lines("", "first\nsec")

    assert lines == ["first"]
    assert partial == "sec"

    lines, partial = _complete_stderr_lines(partial, "ond\r\n")

    assert lines == ["second"]
    assert partial == ""


def test_worker_client_logs_complete_plain_stderr_lines(caplog):
    client = ProToolsWorkerClient()
    logger_name = "sessionprepgui.daw_tools.protools.worker_client"

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        client._append_stderr_text("worker started\npartial")
        client._append_stderr_text(" line\n")

    messages = [record.getMessage() for record in caplog.records]
    assert "Pro Tools worker stderr: worker started" in messages
    assert "Pro Tools worker stderr: partial line" in messages
    assert client.stderr_text == "worker started\npartial line\n"


def test_worker_client_logs_formatted_worker_stderr_as_normal_records(caplog):
    client = ProToolsWorkerClient()
    logger_name = "sessionprepgui.daw_tools.protools.worker_process"

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        client._append_stderr_text(
            "[worker DEBUG] __main__: Creating Pro Tools PTSL engine\n"
        )

    assert any(
        record.name == logger_name
        and record.levelno == logging.DEBUG
        and record.getMessage() == "Creating Pro Tools PTSL engine"
        for record in caplog.records
    )


def test_worker_stderr_summary_is_single_line():
    summary = _stderr_summary("one\ntwo\nthree\n", line_count=2)

    assert summary == "... 1 earlier line(s) | two | three"
    assert "\n" not in summary


def test_worker_stderr_summary_formats_worker_log_lines():
    summary = _stderr_summary(
        "[worker DEBUG] __main__: Handling Pro Tools worker request: method=connect\n"
        "[worker ERROR] sessionprepgui.daw_tools.protools.connection_common: failed\n"
    )

    assert "[worker" not in summary
    assert "__main__" not in summary
    assert (
        "sessionprepgui.daw_tools.protools.worker_process DEBUG: "
        "Handling Pro Tools worker request: method=connect"
    ) in summary
    assert (
        "sessionprepgui.daw_tools.protools.connection_common ERROR: failed"
    ) in summary


def test_worker_client_tail_text_bounds_large_diagnostics():
    text = _tail_text("x" * 20, limit=5)

    assert text.endswith("xxxxx")
    assert "truncated 15 chars" in text


def test_track_height_default_scope_is_all_tracks():
    assert _default_preset()["scope"] == "all"


def test_track_height_migrates_old_builtin_default_scope():
    old_default = _default_preset()
    old_default["scope"] = "selected"

    migrated = _migrate_default_preset_scope(old_default)

    assert migrated["scope"] == "all"


def test_track_height_does_not_migrate_custom_selected_scope():
    custom = _default_preset()
    custom["scope"] = "selected"
    custom["heights"]["TT_Audio"] = "THeight_Large"

    migrated = _migrate_default_preset_scope(custom)

    assert migrated["scope"] == "selected"
    assert migrated["heights"]["TT_Audio"] == "THeight_Large"


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
