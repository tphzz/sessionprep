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


def test_phase1_config_impact_routes_to_track_layout_reanalysis(qapp):
    window = SessionPrepWindow()
    calls: list[str] = []
    ctrl = window.config_controller
    ctrl.reanalyze_phase1_requested.disconnect(window.persistence_controller.reanalyze_topology)
    ctrl.reanalyze_phase2_requested.disconnect(window.analysis_controller.analyze)
    ctrl.reanalyze_phase1_requested.connect(lambda: calls.append("phase1"))
    ctrl.reanalyze_phase2_requested.connect(lambda: calls.append("phase2"))

    try:
        ctrl.apply_config_change_impact(
            ConfigChangeImpact(
                phase1_keys=frozenset({
                    "detectors.dual_mono.dual_mono_eps",
                })))
    finally:
        dispose_window(window)

    assert calls == ["phase1"]


def test_phase2_config_impact_routes_to_phase2_reanalysis(qapp):
    window = SessionPrepWindow()
    calls: list[str] = []
    ctrl = window.config_controller
    ctrl.reanalyze_phase1_requested.disconnect(window.persistence_controller.reanalyze_topology)
    ctrl.reanalyze_phase2_requested.disconnect(window.analysis_controller.analyze)
    ctrl.reanalyze_phase1_requested.connect(lambda: calls.append("phase1"))
    ctrl.reanalyze_phase2_requested.connect(lambda: calls.append("phase2"))

    try:
        ctrl.apply_config_change_impact(
            ConfigChangeImpact(
                phase2_keys=frozenset({
                    "processors.bimodal_normalize.target_rms",
                })))
    finally:
        dispose_window(window)

    assert calls == ["phase2"]


def test_mixed_preset_change_refreshes_daw_before_analysis_rerun(qapp):
    window = SessionPrepWindow()
    calls: list[str] = []
    ctrl = window.config_controller
    ctrl.reanalyze_phase2_requested.disconnect(window.analysis_controller.analyze)
    ctrl.daw_refresh_requested.disconnect(window._refresh_daw_processors_from_config)
    ctrl.reanalyze_phase2_requested.connect(lambda: calls.append("analysis"))
    ctrl.daw_refresh_requested.connect(lambda: calls.append("daw"))

    try:
        ctrl.apply_config_change_impact(
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
    ctrl = window.config_controller
    ctrl.reanalyze_phase1_requested.disconnect(window.persistence_controller.reanalyze_topology)
    ctrl.reanalyze_phase2_requested.disconnect(window.analysis_controller.analyze)
    ctrl.daw_refresh_requested.disconnect(window._refresh_daw_processors_from_config)
    ctrl.reanalyze_phase1_requested.connect(lambda: calls.append("phase1"))
    ctrl.reanalyze_phase2_requested.connect(lambda: calls.append("phase2"))
    ctrl.daw_refresh_requested.connect(lambda: calls.append("daw"))
    monkeypatch.setattr(ctrl, "refresh_config_changed_display", lambda: calls.append("display"))

    try:
        ctrl.apply_config_change_impact(
            ConfigChangeImpact(
                daw_keys=frozenset({
                    "daw_processors.protools.protools_project_dir",
                })))
    finally:
        dispose_window(window)

    assert calls == ["daw", "display"]
