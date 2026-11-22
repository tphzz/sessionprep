"""Exercise real QProcess callbacks and utility controls without a running DAW."""

from pathlib import Path
import sys
import time

import pytest

pytest.importorskip("ptsl")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QProcess
from PySide6.QtTest import QTest

from sessionprepgui.daw_tools.protools import worker_client
from sessionprepgui.daw_tools.protools.color_tool import ColorTool
from sessionprepgui.daw_tools.protools.track_height_tool import TrackHeightTool


def wait_for(condition):
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "Timed out waiting for Qt worker or widget"
        QTest.qWait(10)


def request(client, name):
    replies = []
    client.request(name, {}, replies.append, timeout_ms=7000)
    wait_for(lambda: replies)
    assert replies[0]["ok"], replies
    return replies[0]["result"]


@pytest.fixture
def client(_font_app, monkeypatch):
    helper = Path(__file__).resolve().parents[1] / "fixtures" / "protools_worker_host.py"
    monkeypatch.setattr(worker_client, "_worker_launch_command", lambda: (sys.executable, ["-u", str(helper)], "test"))
    client = worker_client.ProToolsWorkerClient()
    assert request(client, "writes") == []
    yield client
    client.stop()
    wait_for(lambda: client._process is None)


def test_exact_palette_and_all_height_scopes_use_the_selected_host_targets(client):
    color = ColorTool({"colors": [{"name": "Slate", "argb": "#ff204080"}]})
    color.set_client(client)
    color._on_color_clicked(0)
    wait_for(lambda: "Applied" in color._status.text())
    heights = TrackHeightTool({})
    heights.set_client(client)
    heights._mode_combo.setCurrentIndex(heights._mode_combo.findData("all"))
    heights._all_height_combo.setCurrentIndex(heights._all_height_combo.findData("THeight_Medium"))
    for scope in ("all", "visible", "selected"):
        heights._scope_combo.setCurrentIndex(heights._scope_combo.findData(scope))
        heights._apply_btn.click()
        assert not heights._apply_btn.isEnabled()
        wait_for(heights._apply_btn.isEnabled)
    writes = request(client, "writes")
    assert writes[0] == ["CId_SetTrackColor", {"color_index": 2, "track_names": ["Kick", "Snare"]}]
    assert writes[1:] == [
        ["CId_SetTrackHeight", {"height": "THeight_Medium", "track_ids": ids}]
        for ids in (["one", "two", "three", "four"], ["one", "two", "three"], ["one", "two"])
    ]


def test_reordered_and_expired_replies_complete_only_the_matching_request(client):
    replies = []
    first = client.request("hold", {}, replies.append, timeout_ms=10_000)
    second = client.request("release", {}, replies.append, timeout_ms=10_000)
    wait_for(lambda: len(replies) == 2)
    assert [(r["id"], r["result"]) for r in replies] == [(second, "second"), (first, "first")]
    timed_out = []
    client.request("delay", {}, timed_out.append, timeout_ms=50)
    wait_for(lambda: client._process is None)
    assert len(timed_out) == 1 and not timed_out[0]["ok"]
    assert request(client, "writes") == []
    assert len(timed_out) == 1


def test_worker_exit_mid_reply_restores_apply_and_restarts_cleanly(client):
    heights = TrackHeightTool({})
    heights.set_client(client)
    heights._mode_combo.setCurrentIndex(heights._mode_combo.findData("all"))
    heights._all_height_combo.setCurrentIndex(heights._all_height_combo.findData("THeight_Medium"))
    request(client, "arm_exit")
    heights._apply_btn.click()
    wait_for(lambda: client._process is None and heights._apply_btn.isEnabled())
    assert "failed" in heights._status.text().lower()
    heights._apply_btn.click()
    wait_for(heights._apply_btn.isEnabled)
    assert request(client, "writes") == [
        ["CId_SetTrackHeight", {"height": "THeight_Medium", "track_ids": ["one", "two", "three", "four"]}],
    ]


def test_failed_worker_launch_can_be_retried(client, monkeypatch, tmp_path):
    client.stop()
    wait_for(lambda: client._process is None)
    with monkeypatch.context() as patch:
        patch.setattr(worker_client, "_worker_launch_command", lambda: (str(tmp_path / "missing"), [], "test"))
        replies = []
        client.request("writes", {}, replies.append, timeout_ms=10_000)
        wait_for(lambda: replies)
        assert len(replies) == 1 and not replies[0]["ok"]
    assert request(client, "writes") == []


def test_reconnects_release_each_finished_qprocess(client):
    for _ in range(3):
        stopped = client._process
        client.stop()
        wait_for(lambda: client._process is None)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert client.findChildren(QProcess) == []
        assert request(client, "writes") == []
        # Both shutdown timers may still fire after the replacement starts.
        client._terminate_process_if_current(stopped)
        client._kill_process_if_current(stopped)
        assert client._process.state() == QProcess.Running
