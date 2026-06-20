from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QMenu

from sessionprepgui.detail.mixin import DetailMixin


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeButton:
    def __init__(self):
        self.text = ""

    def setText(self, text: str):
        self.text = text


class _FakeWaveform:
    def __init__(self):
        self.enabled_overlays: set[str] = set()

    def set_enabled_overlays(self, labels: set[str]):
        self.enabled_overlays = set(labels)


class _OverlayHost(DetailMixin):
    def __init__(self):
        self._overlay_menu = QMenu()
        self._overlay_btn = _FakeButton()
        self._waveform = _FakeWaveform()
        self._selected_detector_overlays: set[str] = set()
        self._session = None
        self._current_track = SimpleNamespace(detector_results={})


def _issue(label: str):
    return SimpleNamespace(label=label)


def _action_for(host: _OverlayHost, label: str):
    for action in host._overlay_menu.actions():
        if action.data() == label:
            return action
    raise AssertionError(f"No overlay action for {label!r}")


def test_overlay_selection_survives_menu_rebuild_for_same_label(qapp):
    host = _OverlayHost()
    host._update_overlay_menu([_issue("clipping")])

    _action_for(host, "clipping").setChecked(True)

    host._update_overlay_menu([_issue("clipping")])

    assert _action_for(host, "clipping").isChecked()
    assert host._waveform.enabled_overlays == {"clipping"}
    assert host._overlay_btn.text == "Detector Overlays (1)"


def test_overlay_selection_is_remembered_when_track_lacks_label(qapp):
    host = _OverlayHost()
    host._update_overlay_menu([_issue("clipping"), _issue("tail")])
    _action_for(host, "clipping").setChecked(True)

    host._update_overlay_menu([_issue("tail")])

    assert host._selected_detector_overlays == {"clipping"}
    assert host._waveform.enabled_overlays == set()
    assert host._overlay_btn.text == "Detector Overlays"

    host._update_overlay_menu([_issue("clipping")])

    assert _action_for(host, "clipping").isChecked()
    assert host._waveform.enabled_overlays == {"clipping"}
    assert host._overlay_btn.text == "Detector Overlays (1)"


def test_empty_issue_list_clears_active_overlays_but_keeps_memory(qapp):
    host = _OverlayHost()
    host._update_overlay_menu([_issue("clipping")])
    _action_for(host, "clipping").setChecked(True)

    host._update_overlay_menu([])

    assert host._selected_detector_overlays == {"clipping"}
    assert host._waveform.enabled_overlays == set()
    assert host._overlay_btn.text == "Detector Overlays"


def test_unchecking_current_label_does_not_forget_unavailable_labels(qapp):
    host = _OverlayHost()
    host._selected_detector_overlays = {"clipping", "tail"}
    host._update_overlay_menu([_issue("clipping")])

    _action_for(host, "clipping").setChecked(False)

    assert host._selected_detector_overlays == {"tail"}
    assert host._waveform.enabled_overlays == set()
    assert host._overlay_btn.text == "Detector Overlays"
