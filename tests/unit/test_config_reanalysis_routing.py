from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from sessionprepgui.mainwindow import SessionPrepWindow
from sessionpreplib.config import ConfigChangeImpact


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_phase1_config_impact_routes_to_track_layout_reanalysis(qapp, monkeypatch):
    window = SessionPrepWindow()
    calls: list[str] = []
    monkeypatch.setattr(
        window, "_on_topo_reanalyze", lambda: calls.append("phase1"))
    monkeypatch.setattr(window, "_on_analyze", lambda: calls.append("phase2"))

    try:
        window._apply_config_change_impact(
            ConfigChangeImpact(
                phase1_keys=frozenset({
                    "detectors.dual_mono.dual_mono_eps",
                })))
    finally:
        window.close()

    assert calls == ["phase1"]


def test_phase2_config_impact_routes_to_phase2_reanalysis(qapp, monkeypatch):
    window = SessionPrepWindow()
    calls: list[str] = []
    monkeypatch.setattr(
        window, "_on_topo_reanalyze", lambda: calls.append("phase1"))
    monkeypatch.setattr(window, "_on_analyze", lambda: calls.append("phase2"))

    try:
        window._apply_config_change_impact(
            ConfigChangeImpact(
                phase2_keys=frozenset({
                    "processors.bimodal_normalize.target_rms",
                })))
    finally:
        window.close()

    assert calls == ["phase2"]
