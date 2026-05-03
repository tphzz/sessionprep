from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from sessionprepgui.log_viewer import parse_log_line, read_last_lines


def test_parse_log_line_splits_sessionprep_format():
    row = parse_log_line(
        "2026-05-03 16:10:11.123 [DEBUG] sessionprepgui.mainwindow: started"
    )

    assert row.time == "2026-05-03 16:10:11.123"
    assert row.level == "DEBUG"
    assert row.logger == "sessionprepgui.mainwindow"
    assert row.message == "started"


def test_parse_log_line_falls_back_for_continuation_line():
    row = parse_log_line("  traceback continuation")

    assert row.time == ""
    assert row.level == ""
    assert row.logger == ""
    assert row.message == "  traceback continuation"


def test_read_last_lines_returns_requested_tail(tmp_path):
    path = tmp_path / "sessionprep.log"
    path.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")

    assert read_last_lines(str(path), 3) == ["line 7", "line 8", "line 9"]


def test_read_last_lines_zero_starts_empty(tmp_path):
    path = tmp_path / "sessionprep.log"
    path.write_text("line 1\nline 2\n", encoding="utf-8")

    assert read_last_lines(str(path), 0) == []


def test_read_last_lines_missing_file_is_empty(tmp_path):
    assert read_last_lines(str(tmp_path / "missing.log"), 5000) == []


def test_clear_list_clears_table_without_deleting_log(monkeypatch, tmp_path):
    from sessionprepgui import log_viewer

    path = tmp_path / "sessionprep.log"
    path.write_text("line 1\nline 2\n", encoding="utf-8")

    class FakeTable:
        def __init__(self):
            self.row_count = 2

        def setRowCount(self, value):
            self.row_count = value

    class Harness:
        def __init__(self):
            self._table = FakeTable()
            self._partial_line = "partial"
            self.copy_updated = False

        def _update_copy_enabled(self):
            self.copy_updated = True

    monkeypatch.setattr(log_viewer, "get_log_path", lambda: str(path))
    harness = Harness()

    log_viewer.LogViewerWindow._clear_list(harness)

    assert path.read_text(encoding="utf-8") == "line 1\nline 2\n"
    assert harness._table.row_count == 0
    assert harness._partial_line == ""
    assert harness._offset == path.stat().st_size
    assert harness.copy_updated is True


def test_table_rows_text_copies_requested_rows_only():
    from sessionprepgui import log_viewer

    class Item:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

    class FakeTable:
        def __init__(self):
            self._data = [
                ["t0", "DEBUG", "logger.a", "first"],
                ["t1", "INFO", "logger.b", "second"],
                ["t2", "ERROR", "logger.c", "third"],
            ]

        def columnCount(self):
            return 4

        def item(self, row, column):
            return Item(self._data[row][column])

    class Harness:
        def __init__(self):
            self._table = FakeTable()

    text = log_viewer.LogViewerWindow._table_rows_text(Harness(), [0, 2])

    assert text == (
        "t0\tDEBUG\tlogger.a\tfirst\n"
        "t2\tERROR\tlogger.c\tthird"
    )
