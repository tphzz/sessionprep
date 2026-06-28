from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from sessionprepgui.topology.mixin import TopologyMixin
from sessionprepgui.waveform.peakcache import build_peaks
from sessionprepgui.waveform.widget import WaveformWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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