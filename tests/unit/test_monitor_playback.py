"""Monitor assertions use the final float32 samples supplied to PortAudio."""
import gc
import tracemalloc

import numpy as np
import pytest
from scipy.signal.windows import blackmanharris
from scipy.integrate import cumulative_trapezoid

pytest.importorskip("PySide6")
import sounddevice as sd
from PySide6.QtWidgets import QApplication

from sessionprepgui.detail.playback import PlaybackController
from sessionpreplib.audition import RepairAudition
from sessionpreplib.repairs import RepairRecipe


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def devices(app, monkeypatch):
    streams = []

    class Output:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.active = False
            self.closed = False
            streams.append(self)

        def start(self):
            self.active = True

        def close(self):
            self.active = False
            self.closed = True

        def stop(self):
            self.active = False

        def read(self, frames):
            result = np.full((frames, self.channels), np.nan, dtype=np.float32)
            try:
                self.callback(result, frames, None, None)
            except sd.CallbackStop:
                self.active = False
                self.finished_callback()
            assert np.isfinite(result).all()
            return result

    monkeypatch.setattr("sessionprepgui.detail.playback.sd.OutputStream", Output)
    yield streams
    # Real PortAudio completes a stopped stream and queues the controller's
    # close on the GUI thread. The in-memory driver has no background clock.
    for stream in streams:
        if stream.active:
            stream.active = False
            stream.finished_callback()
    app.processEvents()
    assert all(stream.closed for stream in streams)


def _click_db(samples, frequency=100, rate=48000):
    window = blackmanharris(4096, sym=False)
    outside = np.abs(np.fft.rfftfreq(len(window), 1 / rate) - frequency) > 500
    frames = np.lib.stride_tricks.sliding_window_view(samples, len(window))[::128]
    amplitude = np.max(np.abs(np.fft.rfft(frames * window)[:, outside])) * 2 / window.sum()
    return 20 * np.log10(max(amplitude, 1e-12))


@pytest.mark.parametrize("event", ["start", "end", "stop", "seek"])
def test_transport_at_waveform_crest_meets_the_mute_click_limit(devices, app, event):
    rate = 48000
    signal = 10 ** (-6 / 20) * np.cos(2 * np.pi * 100 * np.arange(24001) / rate)
    player = PlaybackController()
    start = len(signal) - 9601 if event == "end" else 0
    player.play(signal, rate, start, audition=RepairAudition(signal, rate, RepairRecipe()))
    output = devices[0]
    lead = output.read(9600)[:, 0]
    if event == "start":
        heard = np.r_[np.zeros(5000), lead]
    elif event == "end":
        heard = np.r_[lead, output.read(2048)[:, 0], np.zeros(5000)]
    else:
        if event == "seek":
            player.play(signal, rate, 14400)
        else:
            player.stop()
        # The old stream must deliver its release before closing. A driver
        # that has already been stopped can only play silence at this point.
        release = output.read(2048)[:, 0] if output.active else np.zeros(2048)
        assert not output.active
        app.processEvents()
        after = devices[1].read(6000)[:, 0] if event == "seek" else np.zeros(6000)
        heard = np.r_[lead, release, after]
    assert _click_db(heard) <= -80
    player.stop()


@pytest.mark.parametrize("reverse_at", [None, 120, 240, 360])
@pytest.mark.parametrize("frequency, real_repair", [(100, False), (20, True), (100, True)])
def test_source_switch_meets_click_limit_including_partial_reversal(devices, reverse_at, frequency, real_repair):
    rate = 48000
    signal = 10 ** (-6 / 20) * np.cos(2 * np.pi * frequency * np.arange(rate) / rate)
    pair = RepairAudition(signal, rate, RepairRecipe(low_cut_hz=40) if real_repair else RepairRecipe())
    if not real_repair:
        pair.corrected = signal * 10 ** (-6 / 20)
    for phase in (range(8) if real_repair else (0,)):
        player = PlaybackController()
        start = rate // 2 - 6000 + round(phase * rate / (8 * frequency))
        player.play(signal, rate, start, audition=pair)
        stream = devices[-1]
        before = stream.read(6000)[1000:, 0]
        player.set_audition_source(True)
        if reverse_at:
            middle = stream.read(reverse_at)[:, 0]
            player.set_audition_source(False)
        else:
            middle = np.empty(0)
        after = stream.read(6000)[:, 0]
        assert _click_db(np.r_[before, middle, after], frequency) <= -80
        player.stop()


@pytest.mark.parametrize("mode,channel", [("as_is", None), ("mono", None), ("channel_as_is", 2)])
def test_callback_temporary_storage_does_not_grow_with_device_frame_count(devices, mode, channel):
    # Input, both sources, device buffer and stream setup are outside the trace.
    source = np.tile([2., -1., 1.5, -.75], (30000, 1))
    pair = RepairAudition(source, 48000, RepairRecipe(low_cut_hz=40))
    peaks = []
    for frames in (64, 8192):
        for state in ("steady", "switch", "reverse"):
            player = PlaybackController()
            player.play(source, 48000, audition=pair, mode=mode, channel=channel)
            output = devices[-1]
            output.read(1024)
            if state == "reverse":
                player.set_audition_source(True)
                output.read(128)
            player.set_audition_source(state == "switch")
            dest = np.empty((frames, output.channels), dtype=np.float32)
            gc.collect()
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            try:
                output.callback(dest, frames, None, None)
                peaks.append(tracemalloc.get_traced_memory()[1] - baseline)
            finally:
                tracemalloc.stop()
            player.stop()
    # Python slice objects have a small fixed cost. Even a single extra
    # float32 lane for the larger request would require 32 KiB.
    assert max(peaks) < 8192, peaks
    assert max(peaks[3:]) <= max(peaks[:3]) + 2048, peaks


def test_overflowing_finite_double_file_never_publishes_a_playable_pair(tmp_path, app):
    import soundfile as sf
    from sessionpreplib.audio import load_track
    from sessionprepgui.waveform.compute import WaveformLoadWorker

    signal = np.tile([1., -1., -1., 1.], 16) * np.finfo(np.float64).max
    path = tmp_path / "Extreme.wav"
    sf.write(path, signal, 96000, subtype="DOUBLE")
    track = load_track(str(path))
    track.repair_recipe = RepairRecipe(dc_remove=True)
    worker = WaveformLoadWorker(None, 96000, 400, preview_track=track, corrected=True, cached={})
    ready, errors = [], []
    worker.source_ready.connect(lambda: ready.append(True))
    worker.failed.connect(errors.append)
    with np.errstate(all="ignore"):
        worker.run()
    assert not ready
    assert len(errors) == 1 and "finite" in errors[0].lower()


def test_largest_safe_double_keeps_normal_monitor_coefficients_and_correct_device_level(devices):
    maximum = np.finfo(np.float64).max
    audio = np.tile([maximum, -maximum], (4096, 1))
    pair = RepairAudition(audio, 96000, RepairRecipe())
    player = PlaybackController()
    player.play(audio, 96000, audition=pair)
    # Check the actual persistent gain state in addition to the device result.
    # Some device threads flush subnormal operands to zero.
    assert player._monitor.monitor_pad is pair.monitor_pad
    assert np.finfo(np.float64).tiny <= pair.monitor_pad.half_gain < 1.
    assert pair.monitor_gain_db == -6165.1
    # Independent oracle: log of source peak, subtract the indicated pad,
    # then exponentiate the residual near unity.
    expected = 10 ** ((20 * np.log10(maximum) - 6165.1) / 20)
    np.testing.assert_allclose(pair.playback_samples(False)[0], [expected, -expected], rtol=1e-11)
    np.testing.assert_allclose(devices[0].read(2048)[-1], [expected, -expected], rtol=1e-6)
    player.stop()


def test_rapid_selections_match_a_smoothed_switch_independently_of_callback_partition(devices):
    rate = 48000
    t = np.arange(8192) / rate
    audio = .7 * np.cos(2 * np.pi * 20 * t) + .18 * np.sin(2 * np.pi * 997 * t)
    pair = RepairAudition(audio, rate, RepairRecipe(low_cut_hz=40))
    selections = np.zeros(len(audio))
    events = [(1751, True), (1773, False), (1800, True), (1817, False),
              (2017, True), (3501, False), (8000, True)]
    for position, selection in events:
        selections[position:] = selection
    fade = cumulative_trapezoid(np.blackman(480), initial=0.)
    fade /= fade[-1]
    weight = np.convolve(selections, np.diff(fade, prepend=0.))[:len(t)]
    expected = (1. - weight) * pair.playback_samples(False) + weight * pair.playback_samples(True)
    expected[:len(fade)] *= fade
    expected[-len(fade):] *= fade[::-1]
    device_outputs = []
    for pattern in ((29, 319, 1, 61), (4096, 17)):
        player = PlaybackController()
        player.play(audio, rate, audition=pair)
        output = devices[-1]
        blocks = []
        pos = 0
        for end, selection in [*events, (len(t), True)]:
            while pos < end:
                n = min(pattern[len(blocks) % len(pattern)], end - pos)
                blocks.append(output.read(n)[:, 0])
                pos += n
            player.set_audition_source(selection)
        device_outputs.append(np.concatenate(blocks))
        np.testing.assert_allclose(device_outputs[-1], expected, rtol=0, atol=2e-5)
        player.stop()
    np.testing.assert_array_equal(*device_outputs)


@pytest.mark.parametrize("rate", [48000, 192000])
@pytest.mark.parametrize("stimulus", ["silence", "impulse", "clipped_noise", "square", "bass", "dc_step"])
def test_monitor_output_is_finite_bounded_and_decays_on_digital_silence(devices, rate, stimulus):
    audio = np.zeros(11 * rate)
    t = np.arange(rate) / rate
    if stimulus == "impulse":
        audio[rate // 2] = 1.
    elif stimulus == "clipped_noise":
        audio[:rate] = np.clip(np.random.default_rng(13).normal(size=rate), -1., 1.)
    elif stimulus == "square":
        audio[:rate] = np.where(np.sin(2 * np.pi * 1000 * t) >= 0, 1., -1.)
    elif stimulus == "bass":
        audio[:rate] = np.sin(2 * np.pi * 20 * t)
    elif stimulus == "dc_step":
        audio[:rate] = .5
    recipe = RepairRecipe(dc_remove=True, low_cut_hz=40, ultrasonic=rate >= 50000)
    pair = RepairAudition(audio, rate, recipe)
    player = PlaybackController()
    player.play(audio, rate, audition=pair, corrected=True)
    stream = devices[-1]
    rng = np.random.default_rng(17)
    pos = 0
    while pos < rate:
        frames = min(rate - pos, round(rate * rng.uniform(.001, .05)))
        player.set_audition_source(bool(rng.integers(2)))
        assert np.max(np.abs(stream.read(frames))) <= 1.
        pos += frames
    player.set_audition_source(True)
    assert np.max(np.abs(stream.read(rate))) <= 1.
    tail = stream.read(9 * rate)
    assert np.max(np.abs(tail)) < 10 ** (-100 / 20)
    assert np.all((tail == 0.) | (np.abs(tail) >= np.finfo(np.float32).tiny))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_invalid_source_rejected_before_open_and_postvalidation_damage_fails_silent(devices, invalid):
    player = PlaybackController()
    src = np.full(48000, .25)
    src[1000] = invalid
    errors = []
    player.error.connect(errors.append)
    player.play(src, 48000)
    assert not devices and errors and "finite" in errors[-1].lower()
    src[1000] = .25
    pair = RepairAudition(src, 48000, RepairRecipe())
    player.play(src, 48000, audition=pair, corrected=True)
    pair.corrected[1000] = invalid
    samples = devices[0].read(1200)[:, 0]
    assert samples[1000] == 0
    np.testing.assert_array_equal(samples[[999, 1001]], [.25, .25])
    player.stop()


def test_postponed_seek_can_be_replaced_or_cancelled_while_old_stream_drains(devices, app):
    player = PlaybackController()
    signal = np.ones(48000) * .25
    player.play(signal, 48000)
    active = devices[0]
    active.read(1024)
    player.play(signal, 48000, 10000)
    player.play(signal, 48000, 20000)
    assert len(devices) == 1 and player.current_sample() == 20000
    active.read(513)
    app.processEvents()
    assert len(devices) == 2 and player.current_sample() == 20000
    player.play(signal, 48000, 30000)
    player.stop()
    devices[1].read(513)
    app.processEvents()
    assert len(devices) == 2 and not player.is_playing


def test_device_failure_after_deferred_seek_releases_transport_controls(devices, app, monkeypatch):
    player = PlaybackController()
    audio = np.zeros(48000)
    player.play(audio, 48000)
    old = devices[0]
    old.read(1024)
    player.play(audio, 48000, 20000)

    def disconnected(stream):
        raise sd.PortAudioError("Device disconnected")

    monkeypatch.setattr(type(old), "start", disconnected)
    finished, errors = [], []
    player.playback_finished.connect(lambda: finished.append(True))
    player.error.connect(errors.append)
    old.read(513)
    app.processEvents()
    assert errors == ["Device disconnected"]
    assert finished and not player.is_playing

