from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from qt_helpers import dispose_window
from sessionprepgui.analysis.worker import (
    AudioLoadWorker, TopoAudioResolveWorker, TopoMultiAudioWorker, TopologyApplyWorker,
)
from sessionprepgui import settings
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.session.io import load_session, save_session
from sessionprepgui.waveform.compute import PeakBuildWorker
from sessionprepgui.waveform.peakcache import build_peaks, get_source_mtime, load_peaks, peaks_path_for, save_peaks
from sessionprepgui.waveform.renderer import WaveformRenderCtx
from sessionprepgui.waveform.widget import WaveformWidget
from sessionpreplib.audio import load_track
from sessionpreplib.models import SessionContext
from sessionpreplib.topology import ChannelRoute, TopologyEntry, TopologyMapping, TopologySource


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    win = SessionPrepWindow()
    yield win
    dispose_window(win)


def wait_until(qapp, condition):
    until = monotonic() + 8
    while not condition() and monotonic() < until:
        qapp.processEvents()
        sleep(.005)
    assert condition()


@pytest.mark.parametrize("unreadable", [0, 1, 2])
def test_window_finishes_peak_build_and_publishes_only_readable_files(qapp, window, tmp_path, unreadable):
    tracks = []
    for i in range(2):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.full(4096, (i + 1) / 4), 48000, subtype="FLOAT")
        tracks.append(load_track(str(path)))
        if i < unreadable:
            path.write_bytes(b"invalid waveform")

    window.lifecycle_controller.start_peak_build(tracks, str(tmp_path))
    assert window.lifecycle_controller.peak_build_worker.wait(5000)
    qapp.processEvents()
    assert not window._progress_running
    assert (window._progress_current, window._progress_total) == (2, 2)
    assert window._progress_text == f"Peak cache built for {2 - unreadable} file(s)"
    assert sorted(float(p.levels[0].data.max()) for p in window.state.peak_cache.values()) == [
        (i + 1) / 4 for i in range(unreadable, 2)]


def test_window_keeps_readable_peaks_when_cache_folder_cannot_be_created(qapp, window, tmp_path):
    path = tmp_path / "Bass.wav"
    sf.write(path, np.full(4096, .5), 48000, subtype="FLOAT")
    (tmp_path / "sp_peaks").write_text("existing file")
    window.lifecycle_controller.start_peak_build([load_track(str(path))], str(tmp_path))
    assert window.lifecycle_controller.peak_build_worker.wait(5000)
    qapp.processEvents()
    assert window._progress_text == "Peak cache built for 1 file(s)"
    assert next(iter(window.state.peak_cache.values())).levels[0].data.max() == .5


def test_queued_peaks_cannot_finish_a_newer_foreground_operation(qapp, window, tmp_path):
    path = tmp_path / "Bass.wav"
    sf.write(path, np.full(4096, .5), 48000, subtype="FLOAT")
    window.lifecycle_controller.start_peak_build([load_track(str(path))], str(tmp_path))
    assert window.lifecycle_controller.peak_build_worker.wait(5000)
    window._progress_start("Preparing…")
    window._progress_value(1, 3)
    qapp.processEvents()
    assert window._progress_running and window._progress_text == "Preparing…"
    assert (window._progress_current, window._progress_total) == (1, 3)
    assert next(iter(window.state.peak_cache.values())).levels[0].data.max() == .5


def test_replacing_a_project_with_disk_caches_clears_its_unfinished_progress(qapp, window, tmp_path):
    path = tmp_path / "Bass.wav"
    sf.write(path, np.full(4096, .25), 48000, subtype="FLOAT")
    track = load_track(str(path))
    window.lifecycle_controller.start_peak_build([track], str(tmp_path))
    assert window.lifecycle_controller.peak_build_worker.wait(5000)
    state = window.lifecycle_controller.capture_session_state()
    state.update(source_dir=str(tmp_path), tracks=[track])
    window.lifecycle_controller.restore_session_state(state)
    qapp.processEvents()
    assert not window._progress_running
    assert next(iter(window.state.peak_cache.values())).levels[0].data.max() == .25


@pytest.mark.parametrize(("kind", "worker_type", "attribute", "prefix"), [
    ("input", AudioLoadWorker, "audio_worker", "Audio load error"),
    ("output", TopoAudioResolveWorker, "resolve_worker", "Preview error"),
    ("input-multi", TopoMultiAudioWorker, "multi_worker", "Multi-track load error"),
    ("output-multi", TopoMultiAudioWorker, "multi_worker", "Multi-track load error"),
])
def test_late_phase1_read_error_preserves_new_request(
        qapp, window, tmp_path, monkeypatch, kind, worker_type, attribute, prefix):
    path = tmp_path / "Bass.wav"
    sf.write(path, np.zeros(4096), 48000)
    track = load_track(str(path))
    item = QTreeWidgetItem()
    item.setData(0, Qt.UserRole, ("file", path.name))
    current_can_finish = Event()
    messages = iter(("previous file", "current file"))

    def fail_read(worker):
        message = next(messages)
        if message == "current file":
            current_can_finish.wait(5)
        worker.error.emit(message)

    def select():
        window.state.source_dir = str(tmp_path)
        window.state.topo_source_tracks = [track]
        window.state.topo_topology = TopologyMapping([TopologyEntry(path.name, 1, [
            TopologySource(path.name, [ChannelRoute(0, 0, 1.)])])])
        if kind == "input":
            window.topology_controller.load_input_waveform(path.name)
        elif kind == "output":
            window.topology_controller.load_output_waveform(path.name)
        elif kind == "input-multi":
            window.topology_controller.load_multi_input([item])
        else:
            window.topology_controller.load_multi_output([item])
        return getattr(window.topology_controller, attribute)

    monkeypatch.setattr(worker_type, "run", fail_read)
    old = select()
    assert old.wait(5000)
    window.lifecycle_controller.clear_workspace()
    current = select()
    try:
        qapp.processEvents()
        assert getattr(window.topology_controller, attribute) is current
        assert window.topology_controller.page.wf_panel.waveform._loading
        assert "previous file" not in window._status_bar.currentMessage()
    finally:
        current_can_finish.set()
        assert current.wait(5000)
    qapp.processEvents()
    assert getattr(window.topology_controller, attribute) is None
    assert not window.topology_controller.page.wf_panel.waveform._loading
    assert window._status_bar.currentMessage() == f"{prefix}: current file"


@pytest.mark.parametrize(("source_cached", "routed_cached"), [
    (False, False), (False, True), (True, False), (True, True),
])
def test_reopened_project_displays_its_own_source_and_routed_peaks(
        qapp, window, tmp_path, source_cached, routed_cached):
    source = tmp_path / "audio"
    routed = source / "sp_01_tracklayout"
    routed.mkdir(parents=True)
    name, rate, frames = "Bass.wav", 48000, 96000
    tone = np.cos(2 * np.pi * 375 * np.arange(frames) / rate)
    for folder, level, cached in ((source, .8, source_cached), (routed, .1, routed_cached)):
        path = folder / name
        sf.write(path, level * tone, rate, subtype="FLOAT")
        if cached:
            audio, _ = sf.read(path)
            save_peaks(build_peaks(audio, rate, get_source_mtime(str(path))),
                       peaks_path_for(str(folder / "sp_peaks"), name))

    state = window.lifecycle_controller.capture_session_state()
    state.update(source_dir=str(source), topology_dir=str(routed), topology_applied=True,
                 tracks=[load_track(str(routed / name))], topology=TopologyMapping([
                     TopologyEntry(name, 1, [TopologySource(name, [ChannelRoute(0, 0, .125)])])]))
    project = str(tmp_path / "routing.spsession")
    save_session(project, state)
    window.lifecycle_controller.restore_session_state(load_session(project))
    worker = window.lifecycle_controller.peak_build_worker
    if worker is not None:
        assert worker.wait(5000)
        qapp.processEvents()
    assert len(window.state.peak_cache) == 2

    def check_envelope(wave, expected):
        renderer = wave._wf_renderer
        assert renderer._peak_data is not None
        ctx = WaveformRenderCtx(0, 192, 128, 0, 0, frames, 1., wave._channels, 1,
                                False, False, False, False, 1)
        renderer._build_peaks(ctx)
        assert np.max(np.abs(wave._channels[0])) == pytest.approx(expected)
        assert np.max(renderer._peaks_cache[0][1]) == pytest.approx(expected)

    window.detail_controller.show_track_detail(window.state.session.tracks[0])
    wait_until(qapp, lambda: window._repair_preview.displayed and window._repair_preview.worker is None)
    check_envelope(window._wf_panel.waveform, .1)
    window._phase_tabs.setCurrentIndex(0)
    window.topology_controller.load_input_waveform(name)
    wave = window.topology_controller.page.wf_panel.waveform
    wait_until(qapp, lambda: bool(wave._channels) and len(wave._channels[0]) == frames
               and window.topology_controller.audio_worker is None and window.topology_controller.wf_worker is None)
    check_envelope(wave, .8)


def test_reapplying_same_length_route_rebuilds_peaks_from_new_audio(qapp, tmp_path):
    source = tmp_path / "Bass.wav"
    layout = tmp_path / "layout"
    cache = layout / "sp_peaks"
    sf.write(source, np.full(4096, .8), 48000, subtype="FLOAT")
    track = load_track(str(source))
    session = SessionContext([track], {})
    for gain in (1., .125):
        session.topology = TopologyMapping([TopologyEntry(source.name, 1, [
            TopologySource(source.name, [ChannelRoute(0, 0, gain)])])])
        worker = TopologyApplyWorker(session, str(layout), str(tmp_path), str(cache))
        worker.run()
        audio, _ = sf.read(layout / source.name)
        peaks = load_peaks(peaks_path_for(str(cache), source.name))
        assert peaks.levels[0].data.max() == pytest.approx(audio.max())
        assert audio.max() == pytest.approx(.8 * gain)


@pytest.mark.parametrize("dispatch_before_replacement", [False, True])
def test_new_project_ignores_queued_peaks_and_progress_for_the_same_filename(
        qapp, window, tmp_path, dispatch_before_replacement, monkeypatch):
    finished = []
    finish = window._progress_finish

    def record_finish(message):
        finished.append(message)
        finish(message)

    monkeypatch.setattr(window, "_progress_finish", record_finish)

    def start(folder_name, peak):
        folder = tmp_path / folder_name
        folder.mkdir()
        path = folder / "shared.wav"
        sf.write(path, np.full(4096, peak), 48000, subtype="FLOAT")
        window.lifecycle_controller.start_peak_build([load_track(str(path))], str(folder))
        return window.lifecycle_controller.peak_build_worker

    old_worker = start("first", .75)
    # Finish the real worker while its GUI callbacks are still queued.
    assert old_worker.wait(5000)
    window.lifecycle_controller.clear_workspace()
    if dispatch_before_replacement:
        qapp.processEvents()
        assert window.state.peak_cache == {} and finished == [] and not window._progress_running

    current_worker = start("second", .125)
    assert current_worker.wait(5000)
    # A second lookup in the same project may have no missing routed peaks.
    window.lifecycle_controller.start_peak_build([], str(tmp_path / "second"))
    assert window.lifecycle_controller.peak_build_worker is current_worker
    qapp.processEvents()
    assert finished == ["Peak cache built for 1 file(s)"]
    assert len(window.state.peak_cache) == 1
    assert Path(next(iter(window.state.peak_cache))) == tmp_path / "second" / "shared.wav"
    np.testing.assert_allclose(next(iter(window.state.peak_cache.values())).levels[0].data, .125)


def test_peak_completion_counts_only_readable_files(qapp, tmp_path):
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    sf.write(good, np.zeros(4096), 48000)
    bad.write_bytes(b"unreadable audio")
    worker = PeakBuildWorker([(str(p), p.name, str(p.with_suffix(".peaks"))) for p in (good, bad)])
    published, progress, done = [], [], []
    worker.file_done.connect(lambda job, filepath, data: published.append(Path(filepath)))
    worker.progress_value.connect(lambda job, current, total: progress.append((current, total)))
    worker.all_done.connect(lambda job, count: done.append(count))
    worker.run()
    assert published == [good]
    assert progress == [(1, 2), (2, 2)]
    assert done == [1]


def test_waveform_widget_rejects_mismatched_peak_data(qapp, caplog):
    del qapp
    widget = WaveformWidget()
    audio = np.column_stack([
        np.linspace(-1.0, 1.0, 4096, dtype=np.float64)
        for _ in range(4)
    ])
    widget.set_audio(audio, 48_000)
    stale_peak_data = build_peaks(audio[:, 0], 48_000, source_mtime=123)

    assert widget.set_peak_data(stale_peak_data) is False
    assert widget._wf_renderer._peak_data is None
    assert "Rejected waveform peak cache metadata mismatch" in caplog.text


def test_phase1_waveform_request_clears_peak_path_for_multi_selection(qapp, window):
    topology = window.topology_controller
    first_request = topology.begin_waveform_request(
        "input-file", peak_path="17_ElecGtr6.wav", file_count=1)
    second_request = topology.begin_waveform_request(
        "input-multi", file_count=4)

    assert second_request == first_request + 1
    assert topology.wf_peak_path is None
    assert not topology.is_current_waveform_request(first_request)
    assert topology.is_current_waveform_request(second_request)


def test_phase1_stale_multi_result_is_ignored(qapp, window):
    topology = window.topology_controller
    stale_request = topology.begin_waveform_request(
        "input-file", peak_path="old.wav")
    topology.begin_waveform_request("input-multi", file_count=2)
    sentinel = object()
    topology.multi_worker = sentinel

    topology.on_multi_done([], [], 48_000, [], stale_request)

    assert topology.multi_worker is sentinel
    assert topology.cached_audio is None
