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
        while not window._playback._monitor.selected and time.monotonic() < until:
            app.processEvents()
            time.sleep(.005)
        assert window._playback._monitor.selected and window._repair_preview.worker is not None
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
        streams[0].callback(np.empty((10, 1), dtype=np.float32), 10, None, None)
        player.set_audition_source(True)
        blocks = []
        for frames in (3, 4, 5):
            block = np.empty((frames, 1), dtype=np.float32)
            streams[0].callback(block, frames, None, None)
            blocks.append(block)
        expected = audition.switch_samples(15, False, True, 12)[:, 0]
        np.testing.assert_allclose(np.concatenate(blocks)[:, 0], expected, rtol=1e-6)
        assert len(streams) == 1 and player.current_sample() == 27
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
        # The fixture's 776.657388 dBFS peak owes precisely 776.7 dB of pad.
        expected = np.asarray(expected) * (amplitude * 10 ** (-776.7 / 20))
        streams[0].callback(np.empty((10, len(expected)), dtype=np.float32), 10, None, None)
        for switching, frames in ((False, 3), (True, 4), (True, 8), (True, 5)):
            player.set_audition_source(switching)
            block = np.empty((frames, len(expected)), dtype=np.float32)
            streams[0].callback(block, frames, None, None)
            assert np.isfinite(block).all()
            np.testing.assert_allclose(block, np.tile(expected, (frames, 1)), rtol=1.e-6, atol=1.e-7)
    finally:
        player.stop()


def _worker_spectra(track, band):
    from sessionprepgui.waveform.compute import WaveformLoadWorker

    pair = None
    for corrected in (False, True):
        worker = WaveformLoadWorker(None, track.samplerate, track.samplerate // 4,
                                    preview_track=track, audition=pair, corrected=corrected,
                                    band=band, spec_n_fft=512)
        results, errors = [], []
        worker.finished.connect(results.append)
        worker.failed.connect(errors.append)
        worker.run()
        assert not errors and len(results) == 1, errors
        pair = worker.audition
        yield results[0], pair.corrected if corrected else pair.original


@pytest.mark.parametrize("rate,hz,db,band,recipe", [
    (48000, 23997.5, -40, "high", RepairRecipe(low_cut_hz=40)),
    (48000, 23990., -60, "high", RepairRecipe(low_cut_hz=40)),
    (48000, 23997.5, -40, "full", RepairRecipe(low_cut_hz=40)),
    (48000, 41.2034446, -40, "full", RepairRecipe(low_cut_hz=40)),
    (48000, 23990., -40, "high", RepairRecipe(dc_remove=True)),
    (96000, .75, -40, "low", RepairRecipe(ultrasonic=True)),
])
def test_corrected_worker_calibrates_the_line_present_in_each_render(
        app, tmp_path, rate, hz, db, band, recipe):
    import soundfile as sf
    from sessionpreplib.audio import load_track
    from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx

    t = np.arange(4 * rate) / rate
    phase = 2 * np.pi * hz * t + .923
    path = tmp_path / "calibration.wav"
    sf.write(path, 10 ** (db / 20) * np.cos(phase), rate, subtype="PCM_16")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = recipe, None
    measured = (t >= 1.) & (t < 3.)
    basis = np.column_stack((np.cos(phase[measured]), np.sin(phase[measured]), np.ones(measured.sum())))
    for result, audio in _worker_spectra(track, band):
        a, b, _ = np.linalg.lstsq(basis, audio[measured], rcond=None)[0]
        target = 20 * np.log10(np.hypot(a, b))
        times = result["spec_times"]
        settled = np.flatnonzero((times >= 1.) & (times < 3.))
        peaks = result["spec_db"][:, settled].max(axis=0)
        np.testing.assert_allclose(peaks, target, atol=.3)
        renderer = SpectrogramRenderer()
        renderer.set_colormap("grayscale")
        renderer.set_inspection(result)
        for j in (settled[np.argmin(peaks)], settled[np.argmax(peaks)]):
            center = round(times[j] * rate)
            ctx = SpecRenderCtx(0, 30, 50, center - 1, center + 1, len(audio), rate)
            renderer._build_spec_image(ctx)
            brightest = max(renderer._spec_image.pixelColor(x, y).red() for x in range(30) for y in range(50))
            expected = 255 * (target - renderer._spec_db_floor) / (renderer._spec_db_ceil - renderer._spec_db_floor)
            assert abs(brightest - expected) <= 2


@pytest.mark.parametrize("rate,hz,band,recipe,reduction", [
    (48000, 0., "low", RepairRecipe(dc_remove=True), 100.),
    (48000, 1., "low", RepairRecipe(low_cut_hz=40), 55.),
    (96000, 47997.5, "high", RepairRecipe(ultrasonic=True), 70.),
    (96000, 47997.5, "full", RepairRecipe(ultrasonic=True), 70.),
])
def test_corrected_worker_only_reports_audio_left_by_the_filter(app, tmp_path, rate, hz, band, recipe, reduction):
    import soundfile as sf
    from sessionpreplib.audio import load_track

    t = np.arange(4 * rate) / rate
    wave = .1 * np.cos(2 * np.pi * hz * t)
    path = tmp_path / "filter.wav"
    sf.write(path, wave, rate, subtype="PCM_16")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = recipe, None
    levels = []
    for result, _ in _worker_spectra(track, band):
        times = result["spec_times"]
        settled = (times >= 1.5) & (times < 3.)
        carrier_row = np.argmin(abs(result["spec_freq"] - hz))
        levels.append(result["spec_db"][carrier_row, settled].max())
    assert levels[0] == pytest.approx(-20., abs=.3)
    assert levels[1] <= levels[0] - reduction


@pytest.mark.parametrize("rate,frequency,seconds,amplitude,noise_rms,seed,phase", [
    (48000, 23997.5, 4., .01, 8e-6, 43219, .923),
    (48000, 23996.5, 5.12, 10 ** (-37 / 20), 4e-6, 149, .683),
    (44144, 22064.5, 4., .05, 2e-5, 149, .683),
    (48000, 23992.5, 4., .05, 2e-5, 620, .49),
])
def test_float_worker_resolves_a_quiet_line_with_a_measured_noise_floor(
        app, tmp_path, rate, frequency, seconds, amplitude, noise_rms, seed, phase):
    import soundfile as sf
    from sessionpreplib.audio import load_track

    t = np.arange(round(seconds * rate)) / rate
    angle = 2 * np.pi * frequency * t + phase
    # A floating-point export of a noisy oscillator need not lie on a PCM grid.
    noise = np.random.default_rng(seed).normal(0., noise_rms, len(t))
    path = tmp_path / "floating.wav"
    sf.write(path, amplitude * np.cos(angle) + noise, rate, subtype="FLOAT")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = RepairRecipe(low_cut_hz=40), None
    for result, audio in _worker_spectra(track, "high"):
        selected = (t >= seconds / 4) & (t < 3 * seconds / 4)
        basis = np.column_stack((np.cos(angle[selected]), np.sin(angle[selected]), np.ones(selected.sum())))
        a, b, _ = np.linalg.lstsq(basis, audio[selected], rcond=None)[0]
        expected = 20 * np.log10(np.hypot(a, b))
        selected = (result["spec_times"] >= seconds / 4) & (result["spec_times"] < 3 * seconds / 4)
        np.testing.assert_allclose(result["spec_db"][:, selected].max(axis=0), expected, atol=.3)


@pytest.mark.parametrize("rate,level,offset", [(96000, .41, 350.), (95904, -.21, 314.)])
def test_filtered_nyquist_level_survives_a_louder_program_and_noise(app, tmp_path, rate, level, offset):
    import soundfile as sf
    from sessionpreplib.audio import load_track
    from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx

    index = np.arange(4 * rate)
    t = index / rate
    alternating = 1 - 2 * (index % 2)
    source = (level * alternating + .4 * abs(level) * np.cos(2 * np.pi * (rate / 2 - offset) * t + .61)
              + .12 * np.cos(2 * np.pi * 997 * t + .37))
    source += np.random.default_rng(22670).normal(0., abs(level) * 10 ** (-42 / 20), len(t))
    path = tmp_path / "nyquist.wav"
    sf.write(path, source, rate, subtype="PCM_16")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = RepairRecipe(ultrasonic=True), None
    for result, audio in _worker_spectra(track, "high"):
        settled = slice(rate, 3 * rate)
        reference = np.average(audio[settled] * alternating[settled], weights=np.hanning(2 * rate))
        expected = 20 * np.log10(abs(reference))
        columns = np.flatnonzero((result["spec_times"] >= 1.) & (result["spec_times"] < 3.))
        readings = result["spec_db"][-1, columns]
        np.testing.assert_allclose(readings, expected, atol=.3)
        renderer = SpectrogramRenderer()
        renderer.set_colormap("grayscale")
        renderer.set_inspection(result)
        renderer.set_db_floor(-120.)
        renderer.set_db_ceil(-30.)
        center = round(result["spec_times"][columns[np.argmax(readings)]] * rate)
        renderer._build_spec_image(SpecRenderCtx(0, 6, 2048, center - 1, center + 1, len(t), rate))
        gray = renderer._spec_image.pixelColor(3, 0).red()
        assert abs(gray - np.clip(255 * (expected + 120.) / 90., 0., 255.)) <= 2


@pytest.mark.parametrize("rate,hz,amplitude,band,recipe,expected_db", [
    (96000, 47990., 10 ** (-6 / 20), "high", RepairRecipe(ultrasonic=True), -109.77),
    (96000, 47999.75, 10 ** (-6 / 20), "high", RepairRecipe(ultrasonic=True), -109.76),
    (48000, .25, .5, "low", RepairRecipe(low_cut_hz=40), -94.19),
    (44100, .625, .3, "low", RepairRecipe(low_cut_hz=40), -82.70),
])
def test_corrected_worker_measures_a_pcm_stopband_line_near_or_below_the_source_code_step(
        app, tmp_path, rate, hz, amplitude, band, recipe, expected_db):
    import soundfile as sf
    from sessionpreplib.audio import load_track
    from sessionprepgui.waveform.spectrogram import SpectrogramRenderer, SpecRenderCtx

    t = np.arange(8 * rate) / rate
    angle = 2 * np.pi * hz * t + .49
    path = tmp_path / "stopband.wav"
    sf.write(path, amplitude * np.cos(angle), rate, subtype="PCM_16")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = recipe, None
    settled_samples = (t >= 2.) & (t < 6.)
    basis = np.column_stack((np.cos(angle[settled_samples]), np.sin(angle[settled_samples]),
                             np.ones(settled_samples.sum())))
    for result, audio in _worker_spectra(track, band):
        a, b, _ = np.linalg.lstsq(basis, audio[settled_samples], rcond=None)[0]
        measured = 20 * np.log10(np.hypot(a, b))
        columns = np.flatnonzero((result["spec_times"] >= 2.) & (result["spec_times"] < 6.))
        low, high = (0., 2.) if band == "low" else (rate / 2 - 100., rate / 2)
        near_line = (result["spec_freq"] >= low) & (result["spec_freq"] <= high)
        levels = result["spec_db"][near_line][:, columns].max(axis=0)
        if result["corrected"]:
            assert measured == pytest.approx(expected_db, abs=.1)
            renderer = SpectrogramRenderer()
            renderer.set_inspection(result)
            renderer.set_colormap("grayscale")
            renderer.set_db_floor(-120.)
            renderer.set_db_ceil(-30.)
            renderer._mel_view_min, renderer._mel_view_max = low, high
            for col in (columns[np.argmin(levels)], columns[np.argmax(levels)]):
                center = round(result["spec_times"][col] * rate)
                renderer._build_spec_image(SpecRenderCtx(0, 3, 12, center - 1, center + 1, len(t), rate))
                painted = max(renderer._spec_image.pixelColor(1, y).red() for y in range(12))
                assert abs(painted - 255 * (measured + 120.) / 90.) <= 2
        np.testing.assert_allclose(levels, measured, atol=.3)


def test_corrected_worker_keeps_the_high_band_dark_during_the_lowpass_tail_of_a_cut(app, tmp_path):
    import soundfile as sf
    from sessionpreplib.audio import load_track

    rate = 48000
    t = np.arange(6 * rate) / rate
    wave = .3 * np.cos(2 * np.pi * 23999.875 * t + np.pi / 2)
    wave[t >= 4.] = 0.
    path = tmp_path / "cut.wav"
    sf.write(path, wave, rate, subtype="PCM_U8")
    track = load_track(str(path))
    track.repair_recipe, track.audio_data = RepairRecipe(low_cut_hz=40), None
    for result, audio in _worker_spectra(track, "high"):
        if result["corrected"]:
            # The high-pass retains a low-frequency state after the cut.
            assert np.max(np.abs(audio[4 * rate:4 * rate + 480])) > 1e-4
        after_cut = result["spec_times"] >= 4.01
        assert np.max(result["spec_db"][:, after_cut]) == -200.
