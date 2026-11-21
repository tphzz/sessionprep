"""Unit and lifecycle tests for status-bar unverified Track Layout overwrite confirmation."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSizeF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import pytest
import soundfile as sf

from sessionprepgui.icons import icon, icon_path
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.status_prompt import StatusBarPromptWidget
from sessionprepgui.theme import apply_dark_theme
from sessionprepgui.tracks import _PHASE_ANALYSIS, _PHASE_TOPOLOGY
from sessionpreplib.audio import discover_track, file_revision, load_track
from sessionpreplib.layout_files import (
    UnverifiedLayoutFilesError,
    find_unverified_layout_files,
    publish_layout,
)
from sessionpreplib.models import SessionContext
from sessionpreplib.topology import build_default_topology


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _write_wav(path: Path, value: float, *, channels: int = 1, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((256, channels) if channels > 1 else 256, value, dtype=np.float64)
    sf.write(path, data, sr, subtype="PCM_24")


def _wait_until(qapp, predicate, *, timeout_ms: int = 5000) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(15)
        elapsed += 15
    assert predicate(), "Timed out waiting for condition"


# ── Library contract tests ───────────────────────────────────────────────────


def test_find_unverified_and_publish_with_pinned_override(tmp_path):
    layout = tmp_path / "sp_01_tracklayout"
    stage = tmp_path / "stage"
    existing = layout / "01_Kick.wav"
    unrelated = layout / "KeepMe.wav"
    _write_wav(existing, 0.125)
    _write_wav(unrelated, 0.375)
    unrelated_bytes = unrelated.read_bytes()

    unverified = find_unverified_layout_files(layout, ["01_Kick.wav"], [])
    assert list(unverified.keys()) == ["01_Kick.wav"]
    assert unverified["01_Kick.wav"] == file_revision(existing, existing.stat())

    _write_wav(stage / "01_Kick.wav", 0.5)
    track = replace(discover_track(str(stage / "01_Kick.wav")), filepath=str(existing))

    # Without override, publish refuses with typed UnverifiedLayoutFilesError
    with pytest.raises(UnverifiedLayoutFilesError):
        publish_layout(layout, stage, [track], [])

    # With pinned override, publish replaces target, leaves unrelated file alone, writes receipt
    publish_layout(layout, stage, [track], [], allowed_unverified=unverified)
    assert np.allclose(sf.read(existing)[0], 0.5, atol=1e-4)
    assert unrelated.read_bytes() == unrelated_bytes
    assert (layout / ".sessionprep-layout.json").is_file()

    # Subsequent apply succeeds without override because fresh receipt now verifies ownership
    assert find_unverified_layout_files(layout, ["01_Kick.wav"], []) == {}
    stage2 = tmp_path / "stage2"
    _write_wav(stage2 / "01_Kick.wav", 0.25)
    track2 = replace(discover_track(str(stage2 / "01_Kick.wav")), filepath=str(existing))
    publish_layout(layout, stage2, [track2], [])
    assert np.allclose(sf.read(existing)[0], 0.25, atol=1e-4)


def test_publish_refuses_if_confirmed_unverified_file_changed_before_commit(tmp_path):
    layout = tmp_path / "sp_01_tracklayout"
    stage = tmp_path / "stage"
    existing = layout / "01_Kick.wav"
    _write_wav(existing, 0.125)

    confirmed = find_unverified_layout_files(layout, ["01_Kick.wav"], [])

    # Simulate external edit while user is deciding or worker is rendering
    _write_wav(existing, 0.625, sr=44100)
    modified_bytes = existing.read_bytes()

    _write_wav(stage / "01_Kick.wav", 0.5)
    track = replace(discover_track(str(stage / "01_Kick.wav")), filepath=str(existing))

    with pytest.raises(UnverifiedLayoutFilesError):
        publish_layout(layout, stage, [track], [], allowed_unverified=confirmed)
    assert existing.read_bytes() == modified_bytes


def test_publish_refuses_new_unverified_file_and_rolls_back_atomically(tmp_path):
    layout = tmp_path / "sp_01_tracklayout"
    stage = tmp_path / "stage"
    first = layout / "01_Kick.wav"
    second = layout / "02_Snare.wav"
    _write_wav(first, 0.125)
    first_before = first.read_bytes()

    confirmed = find_unverified_layout_files(layout, ["01_Kick.wav", "02_Snare.wav"], [])
    assert list(confirmed.keys()) == ["01_Kick.wav"]

    # Another unverified file appears after pre-flight confirmation
    _write_wav(second, 0.375)
    second_before = second.read_bytes()

    _write_wav(stage / "01_Kick.wav", 0.5)
    _write_wav(stage / "02_Snare.wav", 0.75)
    t1 = replace(discover_track(str(stage / "01_Kick.wav")), filepath=str(first))
    t2 = replace(discover_track(str(stage / "02_Snare.wav")), filepath=str(second))

    with pytest.raises(UnverifiedLayoutFilesError):
        publish_layout(layout, stage, [t1, t2], [], allowed_unverified=confirmed)
    assert first.read_bytes() == first_before
    assert second.read_bytes() == second_before


def test_find_unverified_rejects_corrupt_receipt_as_hard_error(tmp_path):
    layout = tmp_path / "sp_01_tracklayout"
    _write_wav(layout / "01_Kick.wav", 0.125)
    (layout / ".sessionprep-layout.json").write_text('{"sha256": "not-a-dict"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid layout receipt"):
        find_unverified_layout_files(layout, ["01_Kick.wav"], [])


def test_prior_revision_matches_resolved_equivalent_path(tmp_path):
    layout = tmp_path / "sp_01_tracklayout"
    target = layout / "01_Kick.wav"
    _write_wav(target, 0.125)
    # Construct an equivalent non-normalized path (e.g. parent/../sp_01_tracklayout/01_Kick.wav)
    non_canonical = layout / ".." / "sp_01_tracklayout" / "01_Kick.wav"
    prior_track = replace(discover_track(str(target)), filepath=str(non_canonical))

    assert find_unverified_layout_files(layout, ["01_Kick.wav"], [prior_track]) == {}


# ── Widget, Icon, Keyboard & Elision tests ───────────────────────────────────


def test_warning_icon_asset_loads(qapp):
    assert icon_path("warning").is_file()
    pix = icon("warning").pixmap(16, 16)
    assert not pix.isNull()
    assert pix.deviceIndependentSize() == QSizeF(16.0, 16.0)


def test_prompt_elides_long_filename_in_middle_without_expanding_width(qapp):
    apply_dark_theme(qapp)
    prefix = "1 existing file in sp_01_tracklayout/ could not be verified ("
    filename = "01_Kick_Close_Mic_Super_Long_Take_Name_Final_Master_Take04.wav"
    suffix = "). Overwrite it?"
    prompt = StatusBarPromptWidget()
    prompt.show()
    prompt.show_prompt(prefix, filename, suffix, tooltip="Full details")
    qapp.processEvents()

    fm = prompt.message_label.fontMetrics()
    chrome_w = prompt.width() - prompt.message_label.width()
    fixed_w = fm.horizontalAdvance(f"{prefix}{suffix}")
    half_name_w = fm.horizontalAdvance(filename) // 2
    # Give enough room for boilerplate + half the filename so middle-elision activates
    prompt.resize(chrome_w + fixed_w + half_name_w, 34)
    qapp.processEvents()

    rendered = prompt.message_label.text()
    assert prompt.message_label.minimumSizeHint().width() == 0
    assert "sp_01_tracklayout/" in rendered
    assert "Overwrite it?" in rendered
    assert "\u2026" in rendered
    assert rendered != f"{prefix}{filename}{suffix}"

    # Even at ultra-narrow width, the label never forces the container wider
    prompt.resize(400, 34)
    qapp.processEvents()
    assert prompt.message_label.minimumSizeHint().width() == 0
    assert "\u2026" in prompt.message_label.text()
    prompt.close()
    prompt.deleteLater()


def test_status_prompt_space_cancels_and_suppresses_status_messages(qapp, monkeypatch):
    apply_dark_theme(qapp)
    window = SessionPrepWindow()
    try:
        window.show()
        window.resize(1920, 1080)
        QTest.qWait(30)
        qapp.processEvents()

        play_toggles: list[bool] = []
        monkeypatch.setattr(window.detail_controller, "toggle_play", lambda: play_toggles.append(True))

        accepted, cancelled = [], []
        apply_btn = window.topology_controller.page.toolbar.widgetForAction(
            window.topology_controller.page.apply_action
        )
        window.topology_controller.page.apply_action.setEnabled(True)

        window.show_status_prompt(
            "1 existing file in sp_01_tracklayout/ could not be verified (",
            "01_Kick.wav",
            "). Overwrite it?",
            tooltip="Folder: /tmp/sp_01_tracklayout\n• 01_Kick.wav",
            on_accept=lambda: accepted.append(True),
            on_cancel=lambda: cancelled.append(True),
            return_focus_widget=apply_btn,
        )
        qapp.processEvents()

        prompt_widget = window.status_prompt.widget
        assert window.status_prompt.is_active
        assert prompt_widget.isVisible()
        assert prompt_widget.cancel_btn.hasFocus()

        # Unrelated status updates must NOT paint over the active prompt
        window.show_status("Preferences saved.")
        assert window._status_bar.currentMessage() == ""

        # Pressing Space on focused Cancel activates Cancel without triggering Space playback shortcut
        QTest.keyClick(prompt_widget.cancel_btn, Qt.Key.Key_Space)
        qapp.processEvents()
        assert cancelled == [True]
        assert not accepted
        assert play_toggles == []
        assert not window.status_prompt.is_active
        assert apply_btn is not None and apply_btn.hasFocus()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_status_prompt_enter_accepts_and_escape_cancels(qapp, monkeypatch):
    apply_dark_theme(qapp)
    window = SessionPrepWindow()
    try:
        window.show()
        window.resize(1920, 1080)
        QTest.qWait(30)
        qapp.processEvents()

        play_toggles: list[bool] = []
        monkeypatch.setattr(window.detail_controller, "toggle_play", lambda: play_toggles.append(True))

        accepted, cancelled = [], []
        apply_btn = window.topology_controller.page.toolbar.widgetForAction(
            window.topology_controller.page.apply_action
        )
        window.topology_controller.page.apply_action.setEnabled(True)

        window.show_status_prompt(
            "1 existing file in sp_01_tracklayout/ could not be verified (",
            "01_Kick.wav",
            "). Overwrite it?",
            tooltip="Folder: /tmp/sp_01_tracklayout",
            on_accept=lambda: accepted.append(True),
            on_cancel=lambda: cancelled.append(True),
            return_focus_widget=apply_btn,
        )
        qapp.processEvents()
        prompt_widget = window.status_prompt.widget
        prompt_widget.accept_btn.setFocus()
        qapp.processEvents()
        QTest.keyClick(prompt_widget.accept_btn, Qt.Key.Key_Return)
        qapp.processEvents()
        assert accepted == [True]
        assert play_toggles == []
        assert not window.status_prompt.is_active

        window.show_status_prompt(
            "1 existing file in sp_01_tracklayout/ could not be verified (",
            "01_Kick.wav",
            "). Overwrite it?",
            tooltip="Folder: /tmp/sp_01_tracklayout",
            on_accept=lambda: accepted.append(True),
            on_cancel=lambda: cancelled.append(True),
            return_focus_widget=apply_btn,
        )
        qapp.processEvents()
        QTest.keyClick(prompt_widget.cancel_btn, Qt.Key.Key_Escape)
        qapp.processEvents()
        assert cancelled == [True]
        assert not window.status_prompt.is_active
        assert apply_btn is not None and apply_btn.hasFocus()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


# ── End-to-end GUI Apply & Stale-State Lifecycle tests ───────────────────────


def _setup_window_with_unverified_layout(tmp_path, *, filenames=("01_Kick.wav", "02_Snare.wav")):
    source_dir = tmp_path / "song"
    layout_dir = source_dir / "sp_01_tracklayout"
    tracks = []
    for idx, name in enumerate(filenames):
        src = source_dir / name
        _write_wav(src, 0.2 + 0.1 * idx)
        _write_wav(layout_dir / name, 0.05)
        tracks.append(load_track(str(src)))
    unrelated = layout_dir / "ExternalNotes.wav"
    _write_wav(unrelated, 0.42)

    session = SessionContext(tracks=tracks, config={})
    topo = build_default_topology(tracks)

    window = SessionPrepWindow()
    window.state.source_dir = str(source_dir)
    window.state.session = session
    window.state.topo_source_tracks = list(tracks)
    window.state.topo_topology = topo
    window.topology_controller.populate()
    window.topology_controller.page.apply_action.setEnabled(True)
    return window, source_dir, layout_dir, unrelated


def test_apply_prompts_on_unverified_then_cancel_and_overwrite(qapp, tmp_path, monkeypatch):
    window, _source_dir, layout_dir, unrelated = _setup_window_with_unverified_layout(tmp_path)
    applied_dirs: list[str] = []
    # Avoid launching Phase 2 analysis worker during this focused Phase 1 test
    monkeypatch.setattr(window.analysis_controller, "analyze", lambda: None)
    window.topology_controller.layout_applied.connect(applied_dirs.append)
    try:
        unrelated_before = unrelated.read_bytes()
        kick_before = (layout_dir / "01_Kick.wav").read_bytes()

        # 1. First Apply detects 2 unverified files off-thread and shows status-bar prompt
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)

        prompt = window.status_prompt.widget
        assert "2 existing files in sp_01_tracklayout/ could not be verified (first: 01_Kick.wav). Overwrite them?" == (
            prompt.message_label.full_text()
        )
        assert "01_Kick.wav" in prompt.message_label.toolTip()
        assert "02_Snare.wav" in prompt.message_label.toolTip()
        assert str(layout_dir.resolve()) in prompt.message_label.toolTip()
        assert not window.progress_coordinator.panel_for_phase(_PHASE_TOPOLOGY).isVisible()
        assert window.topology_controller.page.apply_action.isEnabled()

        # 2. Clicking Cancel leaves files untouched
        prompt.cancel_btn.click()
        qapp.processEvents()
        assert not window.status_prompt.is_active
        assert (layout_dir / "01_Kick.wav").read_bytes() == kick_before
        assert "cancelled" in window._status_bar.currentMessage().lower()

        # 3. Re-click Apply -> prompt reappears -> click Overwrite -> writes files & receipt
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        prompt.accept_btn.click()
        _wait_until(qapp, lambda: len(applied_dirs) == 1)

        assert not window.status_prompt.is_active
        assert (layout_dir / ".sessionprep-layout.json").is_file()
        assert np.allclose(sf.read(layout_dir / "01_Kick.wav")[0], 0.2, atol=1e-4)
        assert np.allclose(sf.read(layout_dir / "02_Snare.wav")[0], 0.3, atol=1e-4)
        assert unrelated.read_bytes() == unrelated_before

        # 4. Subsequent Apply succeeds immediately without prompting
        window.switch_to_phase(_PHASE_TOPOLOGY)
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: len(applied_dirs) == 2)
        assert not window.status_prompt.is_active
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_stale_state_events_dismiss_overwrite_prompt(qapp, tmp_path):
    window, _source_dir, layout_dir, _unrelated = _setup_window_with_unverified_layout(
        tmp_path, filenames=("01_Kick.wav",)
    )
    try:
        # Topology edit dismisses prompt
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        assert "1 existing file in sp_01_tracklayout/ could not be verified (01_Kick.wav). Overwrite it?" == (
            window.status_prompt.widget.message_label.full_text()
        )
        window.topology_controller.notify_topology_changed()
        assert not window.status_prompt.is_active

        # Phase switch away from Phase 1 dismisses prompt
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        window.set_phase_enabled(_PHASE_ANALYSIS, True)
        window.switch_to_phase(_PHASE_ANALYSIS)
        qapp.processEvents()
        assert not window.status_prompt.is_active
        window.switch_to_phase(_PHASE_TOPOLOGY)

        # Workspace clear dismisses prompt
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        window.topology_controller.clear_workspace()
        assert not window.status_prompt.is_active
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_reclicking_apply_after_moving_files_aside_succeeds_without_prompt(
    qapp, tmp_path, monkeypatch
):
    window, _source_dir, layout_dir, _unrelated = _setup_window_with_unverified_layout(
        tmp_path, filenames=("01_Kick.wav",)
    )
    applied_dirs: list[str] = []
    monkeypatch.setattr(window.analysis_controller, "analyze", lambda: None)
    window.topology_controller.layout_applied.connect(applied_dirs.append)
    try:
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)

        # User moves the conflicting file aside externally, then clicks Apply again
        os.unlink(layout_dir / "01_Kick.wav")
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: len(applied_dirs) == 1)

        assert not window.status_prompt.is_active
        assert (layout_dir / "01_Kick.wav").is_file()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_gui_overwrite_refuses_if_file_changes_while_prompt_is_open(
    qapp, tmp_path, monkeypatch
):
    window, _source_dir, layout_dir, _unrelated = _setup_window_with_unverified_layout(
        tmp_path, filenames=("01_Kick.wav",)
    )
    applied_dirs: list[str] = []
    monkeypatch.setattr(window.analysis_controller, "analyze", lambda: None)
    window.topology_controller.layout_applied.connect(applied_dirs.append)
    try:
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)

        # Modify the conflicting file externally while the prompt is open
        _write_wav(layout_dir / "01_Kick.wav", 0.65, sr=44100)
        modified_bytes = (layout_dir / "01_Kick.wav").read_bytes()

        window.status_prompt.widget.accept_btn.click()
        _wait_until(qapp, lambda: window.topology_controller.apply_worker is None)

        assert applied_dirs == []
        assert (layout_dir / "01_Kick.wav").read_bytes() == modified_bytes
        assert "could not be verified" in window._status_bar.currentMessage()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_inflight_preflight_discarded_after_phase_switch_or_batch_toggle(qapp, tmp_path):
    window, _source_dir, layout_dir, _unrelated = _setup_window_with_unverified_layout(
        tmp_path, filenames=("01_Kick.wav",)
    )
    try:
        window.set_phase_enabled(_PHASE_ANALYSIS, True)
        # Simulate an in-flight Pass 1 worker whose result arrives after phase switch
        class _DummyWorker:
            def wait(self):
                pass

        dummy = _DummyWorker()
        ctrl = window.topology_controller
        ctrl.apply_worker = dummy
        ctrl._apply_worker_gen = ctrl._overwrite_prompt_gen

        # Switching away from Phase 1 increments generation and invalidates the in-flight check
        window.switch_to_phase(_PHASE_ANALYSIS)
        qapp.processEvents()
        ctrl._on_overwrite_confirmation_required(
            dummy, str(layout_dir.resolve()), {"01_Kick.wav": "rev-1"}
        )
        assert not window.status_prompt.is_active

        # Toggling batch mode also dismisses an active prompt
        window.switch_to_phase(_PHASE_TOPOLOGY)
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        window.batch_controller.batch_mode_changed.emit(True)
        qapp.processEvents()
        assert not window.status_prompt.is_active

        # Applying while already in Batch Processing Mode still shows the prompt
        window._batch_mode_action.setChecked(True)
        qapp.processEvents()
        assert window.is_batch_mode()
        window.topology_controller.apply_topology()
        _wait_until(qapp, lambda: window.status_prompt.is_active)
        assert window.status_prompt.is_active
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
