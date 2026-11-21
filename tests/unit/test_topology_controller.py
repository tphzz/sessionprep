"""Isolated unit tests for TopologyController (Phase 1 Track Layout)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QWidget

from sessionpreplib.audio import load_track
from sessionpreplib.models import SessionContext
from sessionpreplib.topology import ChannelRoute, TopologyEntry, TopologyMapping, TopologySource
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.topology import TopologyController


class _StubShell:
    def __init__(self) -> None:
        self.parent_widget = QWidget()
        self.status_messages: list[str] = []
        self.progress_events: list[tuple[str, object]] = []
        self.enabled_phases: dict[int, bool] = {}
        self.switched_phases: list[int] = []

    def show_status(self, message: str) -> None:
        self.status_messages.append(message)

    def progress_start(self, text: str, *, owner=None) -> None:
        self.progress_events.append(("start", text))

    def progress_message(self, text: str) -> None:
        self.progress_events.append(("message", text))

    def progress_value(self, current: int, total: int) -> None:
        self.progress_events.append(("value", (current, total)))

    def progress_finish(self, text: str) -> None:
        self.progress_events.append(("finish", text))

    def progress_fail(self, text: str) -> None:
        self.progress_events.append(("fail", text))

    def clear_progress(self) -> None:
        self.progress_events.append(("clear", None))

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None:
        self.enabled_phases[phase_index] = enabled

    def switch_to_phase(self, phase_index: int) -> None:
        self.switched_phases.append(phase_index)

    def current_phase(self) -> int:
        return 0

    def is_phase_enabled(self, phase_index: int) -> bool:
        return self.enabled_phases.get(phase_index, False)

    def register_phase_progress_layout(self, phase_index: int, layout) -> None:
        pass

    def dialog_parent(self) -> QWidget:
        return self.parent_widget

    def is_batch_mode(self) -> bool:
        return False

    def is_closing_after_daw(self) -> bool:
        return False

    def resume_close_after_daw(self) -> None:
        pass

    def set_window_title(self, title: str) -> None:
        pass


class _StubPlayback:
    def __init__(self) -> None:
        self.is_playing = False
        self.play_start_sample = 0

    def play(self, *args, **kwargs) -> None:
        self.is_playing = True

    def stop(self) -> None:
        self.is_playing = False


def test_topology_controller_stale_multi_request_discarded(_font_app):
    state = WorkspaceState(config=build_defaults())
    shell = _StubShell()
    ctrl = TopologyController(state, shell, _StubPlayback())
    try:
        first = ctrl.begin_waveform_request("input-file", peak_path="old.wav")
        second = ctrl.begin_waveform_request("input-multi", file_count=2)
        assert second == first + 1
        assert ctrl.wf_peak_path is None
        assert not ctrl.is_current_waveform_request(first)
        assert ctrl.is_current_waveform_request(second)

        sentinel = object()
        ctrl.multi_worker = sentinel
        ctrl.on_multi_done([], [], 48000, [], first)
        assert ctrl.multi_worker is sentinel
        assert ctrl.cached_audio is None
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_topology_controller_populate_and_edit_invalidates_downstream(_font_app, tmp_path):
    wav = tmp_path / "kick.wav"
    sf.write(wav, np.zeros(4800), 48000)
    track = load_track(str(wav))
    track.filename = "kick.wav"

    state = WorkspaceState(config=build_defaults())
    state.source_dir = str(tmp_path)
    state.topology_dir = str(tmp_path / "sp_01_tracklayout")
    state.topo_source_tracks = [track]
    state.session = SessionContext(tracks=[track], config={})

    shell = _StubShell()
    ctrl = TopologyController(state, shell, _StubPlayback())
    edited_events: list[bool] = []
    ctrl.topology_edited.connect(lambda: edited_events.append(True))
    try:
        ctrl.populate()
        assert state.topo_topology is not None
        assert ctrl.page.apply_action.isEnabled()
        assert "1 input" in ctrl.page.status_label.text()

        ctrl.notify_topology_changed()
        assert state.topology_dir is None
        assert shell.enabled_phases[1] is False
        assert shell.enabled_phases[2] is False
        assert edited_events == [True]
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()


def test_topology_controller_apply_error_recovers_ui(_font_app, tmp_path):
    state = WorkspaceState(config=build_defaults())
    state.source_dir = str(tmp_path)
    state.session = SessionContext(tracks=[], config={})
    state.topo_topology = TopologyMapping(
        [TopologyEntry("a.wav", 1, [TopologySource("a.wav", [ChannelRoute(0, 0, 1.0)])])]
    )

    shell = _StubShell()
    ctrl = TopologyController(state, shell, _StubPlayback())
    try:
        fake_worker = SimpleNamespace(wait=lambda: None)
        ctrl.apply_worker = fake_worker
        ctrl.page.apply_action.setEnabled(False)
        ctrl._on_apply_error(fake_worker, "disk full")
        assert ctrl.apply_worker is None
        assert ctrl.page.apply_action.isEnabled()
        assert ("fail", "disk full") in shell.progress_events
        assert "Apply topology error: disk full" in shell.status_messages
    finally:
        ctrl.dispose()
        ctrl.page.deleteLater()
        shell.parent_widget.deleteLater()
        _font_app.processEvents()
