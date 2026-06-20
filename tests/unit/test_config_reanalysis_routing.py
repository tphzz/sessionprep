from __future__ import annotations

import copy

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from sessionprepgui.mainwindow import SessionPrepWindow
from sessionpreplib.config import (
    ConfigChangeImpact,
    build_structured_defaults,
    classify_structured_config_change,
)


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


def _preset_with_protools_project_dir(path: str) -> dict:
    preset = build_structured_defaults()
    protools = preset["daw_processors"]["protools"]
    protools["protools_project_dir"] = path
    protools["protools_templates"] = [
        {"group": "SessionPrep", "name": "Default Template"},
    ]
    return preset


def test_daw_project_directory_keys_are_discovered_generically(qapp):
    window = SessionPrepWindow()
    try:
        keys = window._daw_project_dir_keys()
    finally:
        window.close()

    assert "daw_processors.protools.protools_project_dir" in keys
    assert "daw_processors.dawproject.dawproject_project_dir" in keys


def test_inherited_daw_project_dir_preference_merges_into_session(qapp):
    old_preset = _preset_with_protools_project_dir("")
    new_preset = _preset_with_protools_project_dir(r"C:\Audio Projects")
    impact = classify_structured_config_change(old_preset, new_preset)

    window = SessionPrepWindow()
    try:
        window._session_config = copy.deepcopy(old_preset)

        merged, preserved = window._merge_session_daw_project_dirs(
            old_preset, new_preset, impact)
        window._refresh_daw_processors_from_config()

        protools = next(
            dp for dp in window._daw_processors
            if dp.id.startswith("protools"))
    finally:
        window.close()

    assert merged == 1
    assert preserved == 0
    assert (
        window._session_config["daw_processors"]["protools"]
        ["protools_project_dir"]
        == r"C:\Audio Projects"
    )
    assert protools.project_dir == r"C:\Audio Projects"


def test_session_daw_project_dir_override_is_preserved(qapp):
    old_preset = _preset_with_protools_project_dir(r"C:\Old Projects")
    new_preset = _preset_with_protools_project_dir(r"C:\New Projects")
    impact = classify_structured_config_change(old_preset, new_preset)

    window = SessionPrepWindow()
    try:
        window._session_config = copy.deepcopy(old_preset)
        window._session_config["daw_processors"]["protools"][
            "protools_project_dir"
        ] = r"D:\Session Override"

        merged, preserved = window._merge_session_daw_project_dirs(
            old_preset, new_preset, impact)
    finally:
        window.close()

    assert merged == 0
    assert preserved == 1
    assert (
        window._session_config["daw_processors"]["protools"]
        ["protools_project_dir"]
        == r"D:\Session Override"
    )


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
        window.close()

    assert calls == ["daw", "display"]
