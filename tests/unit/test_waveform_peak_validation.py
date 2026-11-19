from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from sessionprepgui.analysis.mixin import AnalysisMixin
from sessionprepgui.topology.mixin import TopologyMixin
from sessionprepgui.waveform.peakcache import build_peaks
from sessionprepgui.waveform.widget import WaveformWidget
from sessionpreplib.audio import load_track


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize("dispatch_before_replacement", [False, True])
def test_new_project_ignores_queued_peaks_and_progress_for_the_same_filename(
        qapp, tmp_path, dispatch_before_replacement):
    class Host(AnalysisMixin, QObject):
        def __init__(self):
            super().__init__()
            self._config = {}
            self.events = []

        def _progress_start(self, message):
            self.events.append(message)

        _progress_message = _progress_finish = _progress_start

        def _progress_value(self, current, total):
            self.events.append((current, total))

    host = Host()
    def start(folder_name, peak):
        folder = tmp_path / folder_name
        folder.mkdir()
        path = folder / "shared.wav"
        sf.write(path, np.full(4096, peak), 48000, subtype="FLOAT")
        host._start_peak_build([load_track(str(path))], str(folder))
        return host._peak_build_worker

    old_worker = start("first", .75)
    # Finish the real worker while its GUI callbacks are still queued.
    assert old_worker.wait(5000)
    host._cancel_worker("_peak_build_worker")
    host._peak_cache.clear()
    host.events.clear()
    if dispatch_before_replacement:
        qapp.processEvents()
        assert host._peak_cache == {} and host.events == []

    current_worker = start("second", .125)
    assert current_worker.wait(5000)
    # A second lookup in the same project may have no missing routed peaks.
    host._start_peak_build([], str(tmp_path / "second"))
    assert host._peak_build_worker is current_worker
    qapp.processEvents()
    assert host.events.count("Peak cache built for 1 file(s)") == 1
    np.testing.assert_allclose(host._peak_cache["shared.wav"].levels[0].data, .125)


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


def test_phase1_waveform_request_clears_peak_filename_for_multi_selection():
    class DummyTopology(TopologyMixin):
        pass

    topology = DummyTopology()
    first_request = topology._topo_begin_waveform_request(
        "input-file", peak_filename="17_ElecGtr6.wav", file_count=1)
    second_request = topology._topo_begin_waveform_request(
        "input-multi", file_count=4)

    assert second_request == first_request + 1
    assert topology._topo_wf_filename is None
    assert topology._topo_wf_peak_filename is None
    assert not topology._topo_is_current_waveform_request(first_request)
    assert topology._topo_is_current_waveform_request(second_request)


def test_phase1_stale_multi_result_is_ignored():
    class DummyTopology(TopologyMixin):
        pass

    topology = DummyTopology()
    stale_request = topology._topo_begin_waveform_request(
        "input-file", peak_filename="old.wav")
    topology._topo_begin_waveform_request("input-multi", file_count=2)
    topology._topo_multi_worker = object()

    topology._on_topo_multi_done([], [], 48_000, [], stale_request)

    assert topology._topo_multi_worker is not None
    assert not hasattr(topology, "_topo_cached_audio")
