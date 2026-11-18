"""Phase 2 preview compares two sample-aligned sources through one transport."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidgetAction

from sessionprepgui.detail.playback import PlaybackController
from sessionprepgui.waveform.panel import WaveformPanel
from sessionprepgui.waveform.source_selector import AuditionSourceSelector
from sessionpreplib.audition import RepairAudition
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture
def window(app, monkeypatch):
    from sessionprepgui import mainwindow, settings
    monkeypatch.setattr(mainwindow, "load_config", settings.build_defaults)
    monkeypatch.setattr(mainwindow, "save_config", lambda *args: None)
    monkeypatch.setattr(mainwindow.SessionPrepWindow, "_configure_daw_processors",
                        lambda self: setattr(self, "_daw_processors", []))
    owner = mainwindow.SessionPrepWindow()
    owner._phase_tabs.setTabEnabled(1, True)
    owner._phase_tabs.setCurrentIndex(1)
    owner.resize(1500, 800)
    owner.show()
    app.processEvents()
    yield owner
    owner.close()
    app.processEvents()


def _open_track(window, tmp_path, name, *, rate=96000):
    import soundfile
    from sessionpreplib.audio import load_track
    from sessionpreplib.models import SessionContext

    time = np.arange(rate) / rate
    tone = .55 * np.sin(2 * np.pi * 12 * time) + .3 * np.sin(2 * np.pi * 30000 * time)
    path = tmp_path / name
    soundfile.write(path, np.column_stack((tone, -tone)), rate, subtype="FLOAT")
    track = load_track(str(path))
    track.audio_data = None
    track.repair_recipe = RepairRecipe(low_cut_hz=30, ultrasonic=rate >= 50000)
    window._session = SessionContext([track], {})
    window._show_track_detail(track)
    return track


def _wait(app, window):
    import time
    limit = time.monotonic() + 8
    app.processEvents()
    while window._repair_preview.worker is not None and time.monotonic() < limit:
        app.processEvents()
        time.sleep(.005)
    assert window._repair_preview.worker is None


def test_real_file_panel_retains_zoom_cursor_and_native_spectral_scales(app, window, tmp_path):
    track = _open_track(window, tmp_path, "Stereo synth.wav")
    panel = window._wf_panel
    wave = panel.waveform
    _wait(app, window)
    assert window._repair_preview.audition is not None, panel.preview_message.text()
    wave.set_cursor(4096)
    wave.zoom_in()
    view = (wave._view_start, wave._view_end, wave._vscale)
    panel._play_mode_combo.setCurrentIndex(2)
    route = panel.play_mode
    panel.spec_action.trigger()
    panel.band_actions["high"].trigger()
    _wait(app, window)
    renderer = wave._spec_renderer
    before = renderer._spec_db.copy()
    assert (renderer.mel_view_min, renderer.mel_view_max) == (20000, 48000)
    assert before.max() > -14
    panel.source_selector.buttons[1].click()
    _wait(app, window)
    assert wave._corrected and renderer._spec_db.max() < before.max() - 20
    assert wave._cursor_sample == 4096 and view == (wave._view_start, wave._view_end, wave._vscale)
    assert panel.play_mode == route
    assert renderer._spec_db_floor == -100 and renderer._spec_db_ceil == 0
    panel.band_actions["low"].trigger()
    _wait(app, window)
    assert (renderer.mel_view_min, renderer.mel_view_max) == (0, 100)
    assert not wave.set_peak_data(object())
    assert window._current_track is track and track.audio_data is None


def test_replaced_source_does_not_play_cached_material_and_retry_recovers(app, window, tmp_path):
    import os
    path = tmp_path / "Recover.wav"
    _open_track(window, tmp_path, path.name)
    _wait(app, window)
    wave = window._waveform
    wave.set_cursor(7654)
    wave.zoom_in()
    original_view = (wave._view_start, wave._view_end, wave._vscale)
    original_stat = path.stat()
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))
    window._wf_panel.source_selector.buttons[1].click()
    _wait(app, window)
    panel = window._wf_panel
    assert panel.preview_retry_button.isVisible() and not panel.play_btn.isEnabled()
    assert window._repair_preview.audition is None and not panel.waveform._channels
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    panel.preview_retry_button.click()
    _wait(app, window)
    assert window._repair_preview.audition is not None and panel.waveform._corrected
    assert panel.play_btn.isEnabled() and not panel.preview_retry_button.isVisible()
    assert wave._cursor_sample == 7654
    assert (wave._view_start, wave._view_end, wave._vscale) == original_view


def test_late_old_file_or_recipe_cannot_replace_new_selection(app, window, tmp_path):
    first = _open_track(window, tmp_path, "First.wav")
    app.processEvents()
    second = _open_track(window, tmp_path, "Second.wav", rate=32000)
    second.repair_recipe = RepairRecipe(dc_remove=True)
    window._refresh_repair_preview(second)
    _wait(app, window)
    assert window._repair_preview.track is second and window._repair_preview.recipe == second.repair_recipe
    assert window._repair_preview.audition.sample_rate == 32000
    assert not window._wf_panel.band_actions["high"].isEnabled()
    assert window._waveform._samplerate != first.samplerate


def test_changed_repair_choice_immediately_withholds_old_audio_and_retry_replans(app, window, tmp_path, monkeypatch):
    track = _open_track(window, tmp_path, "Replan.wav")
    _wait(app, window)
    panel = window._wf_panel
    panel.source_selector.buttons[1].click()
    _wait(app, window)
    window._repair_plan_pending.add(track.filename)
    window._repair_preview.request()
    assert window._repair_preview.audition is None
    assert not panel.play_btn.isEnabled() and panel.preview_message.text() == "Updating repair preview…"
    window._repair_plan_pending.clear()
    track._cache["repair_error"] = "Source unavailable while planning"
    window._refresh_repair_preview(track)
    _wait(app, window)
    retries = []
    monkeypatch.setattr(window, "_queue_repair_plan", retries.extend)
    panel.preview_retry_button.click()
    assert retries == [track] and panel.corrected


def test_play_checks_disk_identity_after_render_and_cannot_start_stale_material(app, window, tmp_path, streams):
    track = _open_track(window, tmp_path, "Missing.wav")
    _wait(app, window)
    (tmp_path / track.filename).unlink()
    window._wf_panel.play_btn.click()
    _wait(app, window)
    assert not streams
    assert window._wf_panel.preview_retry_button.isVisible()
    assert "Restore the source" in window._wf_panel.preview_message.text()


def test_playing_source_switch_does_not_wait_for_waveform_computation(app, window, tmp_path, streams, monkeypatch):
    import time
    from threading import Event
    from sessionprepgui.waveform.compute import WaveformLoadWorker
    _open_track(window, tmp_path, "Audition.wav")
    _wait(app, window)
    window._wf_panel.play_btn.click()
    _wait(app, window)
    assert window._playback.is_playing
    resume = Event()
    compute = WaveformLoadWorker._compute

    def delayed_render(worker):
        resume.wait(3)
        compute(worker)

    monkeypatch.setattr(WaveformLoadWorker, "_compute", delayed_render)
    try:
        window._wf_panel.source_selector.buttons[1].click()
        until = time.monotonic() + 1.5
        while not window._playback._corrected and time.monotonic() < until:
            app.processEvents()
            time.sleep(.005)
        assert window._playback._corrected and window._repair_preview.worker is not None
        assert len(streams) == 1
    finally:
        resume.set()
        _wait(app, window)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_wide_and_overflow_select_the_same_audition_source(app):
    panel = WaveformPanel()
    reason = "Enable a repair to audition the corrected source."
    panel.set_audition_available(False, reason)
    assert panel.source_selector.buttons[1].toolTip() == reason
    panel.set_audition_available(True, reason)
    assert not panel.source_selector.buttons[1].toolTip()
    panel.resize(1500, 230)
    panel.show()
    app.processEvents()
    try:
        control = panel.source_selector
        assert control.isVisible()
        assert [button.text() for button in control.buttons] == ["Original", "Corrected"]
        QTest.keyClick(control.buttons[0], Qt.Key_Right)
        assert panel.corrected and control.buttons[1].isChecked()
        panel.resize(400, 230)
        app.processEvents()
        assert control.isHidden()
        row = panel._view_toolbar
        row._fill_menu()
        group = next(a.defaultWidget() for a in row.more_menu.actions()
                     if isinstance(a, QWidgetAction))
        editor = group.findChild(AuditionSourceSelector)
        assert editor.buttons[1].isChecked()
        assert not editor.buttons[1].toolTip()
        QTest.keyClick(editor.buttons[1], Qt.Key_Left)
        assert not panel.corrected and control.buttons[0].isChecked()
    finally:
        panel.close()


def test_keyboard_opens_overflow_and_operates_source_through_the_popup_window(app):
    panel = WaveformPanel()
    panel.set_audition_available(True)
    panel.resize(400, 230)
    panel.show()
    panel.activateWindow()
    app.processEvents()
    row = panel._view_toolbar
    menu = row.more_menu
    checks = []

    def exercise_popup():
        try:
            popup = QApplication.activePopupWidget()
            assert popup is menu
            native = popup.windowHandle()
            QTest.keyClick(native, Qt.Key_Down)
            action = menu.activeAction()
            assert isinstance(action, QWidgetAction)
            selector = action.defaultWidget().findChild(AuditionSourceSelector)
            assert selector.property("keyboardFocus") and selector._ring.isVisible()
            QTest.keyClick(native, Qt.Key_Right)
            assert panel.corrected and selector.buttons[1].isChecked()
            for key in (Qt.Key_Space, Qt.Key_Return):
                QTest.keyClick(native, key)
                assert menu.isVisible() and panel.corrected
            QTest.keyClick(native, Qt.Key_Down)
            assert menu.activeAction() is not action and not selector._ring.isVisible()
            QTest.keyClick(native, Qt.Key_Up)
            assert menu.activeAction() is action and selector._ring.isVisible()
            QTest.keyClick(native, Qt.Key_Left)
            assert not panel.corrected
            right = selector.buttons[1]
            QTest.mouseClick(native, Qt.LeftButton, Qt.NoModifier, right.mapTo(menu, right.rect().center()))
            assert panel.corrected and not selector._ring.isVisible()
            QTest.keyClick(native, Qt.Key_Escape)
            assert not menu.isVisible()
            checks.append(None)
        except BaseException as exc:
            checks.append(exc)
        finally:
            menu.close()

    try:
        row.more_button.setFocus(Qt.TabFocusReason)
        QTimer.singleShot(30, exercise_popup)
        QTest.keyClick(panel.windowHandle(), Qt.Key_Space)
        QTest.qWait(60)
        if not checks:
            pytest.fail("Keyboard activation did not open the toolbar overflow")
        if checks[0] is not None:
            raise checks[0]
        assert row.more_button.hasFocus()
    finally:
        panel.close()


@pytest.fixture
def streams(monkeypatch):
    streams = []

    class Stream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            streams.append(self)

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            pass

    monkeypatch.setattr("sessionprepgui.detail.playback.sd.OutputStream", Stream)
    return streams


def test_source_change_preserves_cursor_and_crossfades_across_callbacks(app, streams):
    signal = np.column_stack((np.full(100, 1.2), np.zeros(100)))
    audition = RepairAudition(signal, 1000, RepairRecipe(dc_remove=True))
    player = PlaybackController()
    player.play(audition.original, 1000, 5, mode="channel_mono", channel=0,
                audition=audition, corrected=False)
    try:
        player.set_audition_source(True)
        blocks = []
        for frames in (3, 4, 5):
            block = np.empty((frames, 1), dtype=np.float32)
            streams[0].callback(block, frames, None, None)
            blocks.append(block)
        expected = audition.switch_samples(5, False, True, 12)[:, 0]
        np.testing.assert_allclose(np.concatenate(blocks)[:, 0], expected, rtol=1e-6)
        assert len(streams) == 1 and player.current_sample() == 17
    finally:
        player.stop()


def test_finished_callback_returns_void_and_stale_notifications_leave_new_stream_playing(app, streams):
    player = PlaybackController()
    audio = np.zeros(100, dtype=np.float32)
    player.play(audio, 1000)
    stale = streams[0]
    player.play(audio, 1000)
    try:
        assert stale.finished_callback() is None
        app.processEvents()
        assert player.is_playing
        assert streams[1].finished_callback() is None
        app.processEvents()
        assert not player.is_playing
    finally:
        player.stop()


@pytest.mark.parametrize("mode, channel, expected", [
    ("as_is", None, [1., -.5]), ("mono", None, [.25]),
    ("channel_as_is", 2, [.5, 0.]), ("channel_mono", 1, [-.5]),
])
def test_double_monitor_pad_precedes_device_narrowing_on_both_sources(app, streams, mode, channel, expected):
    amplitude = float(np.finfo(np.float32).max) * 2
    audio = np.tile(np.array([1., -.5, 1., -.5]) * amplitude, (50, 1))
    pair = RepairAudition(audio, 1000, RepairRecipe())
    player = PlaybackController()
    player.play(audio, 1000, mode=mode, channel=channel, audition=pair)
    try:
        expected = np.asarray(expected) * pair.playback_samples(False)[0, 0]
        for switching, frames in ((False, 3), (True, 4), (True, 8), (True, 5)):
            player.set_audition_source(switching)
            block = np.empty((frames, len(expected)), dtype=np.float32)
            streams[0].callback(block, frames, None, None)
            assert np.isfinite(block).all()
            np.testing.assert_allclose(block, np.tile(expected, (frames, 1)), rtol=1.e-6, atol=1.e-7)
    finally:
        player.stop()
