from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from qt_helpers import dispose_window
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
        dispose_window(window)

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
        dispose_window(window)

    assert calls == ["phase2"]


def test_mixed_preset_change_refreshes_daw_before_analysis_rerun(qapp, monkeypatch):
    window = SessionPrepWindow()
    calls: list[str] = []
    monkeypatch.setattr(window, "_on_analyze", lambda: calls.append("analysis"))
    monkeypatch.setattr(
        window, "_refresh_daw_processors_from_config", lambda: calls.append("daw"))

    try:
        window._apply_config_change_impact(
            ConfigChangeImpact(
                phase2_keys=frozenset({"processors.bimodal_normalize.target_rms"}),
                daw_keys=frozenset({"daw_processors.protools.protools_temp_dir"}),
            )
        )
    finally:
        dispose_window(window)

    assert calls == ["daw", "analysis"]


def test_daw_only_config_impact_refreshes_phase3_without_reanalysis(
        qapp, monkeypatch):
    window = SessionPrepWindow()
    calls: list[str] = []
    monkeypatch.setattr(
        window, "_on_topo_reanalyze", lambda: calls.append("phase1"))
    monkeypatch.setattr(window, "_on_analyze", lambda: calls.append("phase2"))
    monkeypatch.setattr(
        window,
        "_refresh_daw_processors_from_config",
        lambda: calls.append("daw"),
    )
    monkeypatch.setattr(
        window,
        "_refresh_config_changed_display",
        lambda: calls.append("display"),
    )

    try:
        window._apply_config_change_impact(
            ConfigChangeImpact(
                daw_keys=frozenset({
                    "daw_processors.protools.protools_project_dir",
                })))
    finally:
        dispose_window(window)

    assert calls == ["daw", "display"]
