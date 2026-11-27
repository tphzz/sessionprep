"""Focused GUI unit tests for Pro Tools Offline .ptx Transfer Mode, template .ptxt path, and Phase 3 UX."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import platform
from typing import Any

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QMessageBox, QStackedWidget, QTreeWidget

from sessionpreplib.config import classify_structured_config_change, validate_structured_config
from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.models import SessionContext, TrackContext, TransferEntry
from sessionprepgui.daw.controller import DawTransferController
from sessionprepgui.prefs.config_pages import (
    build_config_pages,
    load_config_widgets,
    read_config_widgets,
)
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.prefs.protools_prefs import (
    PTSL_ONLY_PARAM_KEYS,
    ProToolsTemplatesWidget,
)
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from tests.unit.qt_helpers import dispose_window
from tests.unit.test_protools_ptx_orchestration import _build_nested_folder_ptx


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dispose(*widgets: Any) -> None:
    for w in widgets:
        if w is not None:
            dispose_window(w)
    QApplication.processEvents()


class _FakeShell:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def register_phase_progress_layout(self, _phase: int, _layout: Any) -> None:
        pass

    def is_batch_mode(self) -> bool:
        return False

    def is_closing_after_daw(self) -> bool:
        return False

    def resume_close_after_daw(self) -> None:
        pass

    def dialog_parent(self) -> Any:
        return None

    def progress_start(self, _msg: str) -> None:
        pass

    def progress_message(self, _msg: str) -> None:
        pass

    def progress_value(self, _cur: int, _tot: int) -> None:
        pass

    def progress_finish(self, _msg: str) -> None:
        pass

    def progress_fail(self, _msg: str) -> None:
        pass

    def show_status(self, msg: str) -> None:
        self.statuses.append(msg)


def _make_config(
    ptxt_path: str, proj_dir: str, backend: str = "offline_ptx", *, add_dawproject: bool = False
) -> dict[str, Any]:
    cfg = build_defaults()
    preset = cfg["config_presets"]["Default"]
    pt = preset["daw_processors"]["protools"]
    pt["protools_transfer_backend"] = backend
    pt["protools_project_dir"] = proj_dir
    pt["protools_templates"] = [{"group": "bnz", "name": "MixTpl", "ptxt_path": ptxt_path}]
    if add_dawproject:
        dp = preset["daw_processors"]["dawproject"]
        dp["dawproject_templates"] = [{"name": "DP", "template_path": str(Path(proj_dir) / "t.dawproject")}]
    return cfg


def test_transfer_backend_config_impact_and_ptxt_path_validation():
    cfg_a = build_defaults()
    p_a = cfg_a["config_presets"]["Default"]
    p_b = deepcopy(p_a)
    p_b["daw_processors"]["protools"]["protools_transfer_backend"] = "offline_ptx"
    impact = classify_structured_config_change(p_a, p_b)
    assert impact.daw_only is True
    assert "daw_processors.protools.protools_transfer_backend" in impact.daw_keys
    assert not impact.presentation_keys and not impact.requires_phase1 and not impact.requires_phase2

    # Valid & invalid optional ptxt_path in validate_structured_config
    p_b["daw_processors"]["protools"]["protools_templates"] = [{"group": "bnz", "name": "M", "ptxt_path": "C:/a.ptxt"}]
    assert validate_structured_config(p_b) == []
    p_b["daw_processors"]["protools"]["protools_templates"] = [{"group": "bnz", "name": "M", "ptxt_path": 123}]
    assert any("protools_templates[0]" in e.key for e in validate_structured_config(p_b))


def test_protools_prefs_transfer_mode_disables_all_7_ptsl_params_and_dirty_tracking(qapp):
    cfg = build_defaults()
    preset = cfg["config_presets"]["Default"]
    tree, stack = QTreeWidget(), QStackedWidget()
    widgets_dict: dict[str, Any] = {}
    dlg = PreferencesDialog(cfg)
    try:
        daw_widgets = build_config_pages(tree, preset, widgets_dict, lambda _i, w, _k: stack.addWidget(w))
        pt_fields = dict(widgets_dict["daw_processors.protools"])
        combo = pt_fields["protools_transfer_backend"]
        assert isinstance(combo, QComboBox)
        assert combo.currentData() == "ptsl_hybrid" and combo.currentText() == "Live PTSL + .ptx Hybrid"
        reset_btn = combo._param_row_widgets[2]
        assert "Live PTSL + .ptx Hybrid" in reset_btn.toolTip()
        assert all(pt_fields[k].isEnabled() for k in PTSL_ONLY_PARAM_KEYS)

        combo.setCurrentIndex(combo.findData("offline_ptx"))
        assert combo.currentText() == "Offline .ptx Writer"
        for key in PTSL_ONLY_PARAM_KEYS:
            w = pt_fields[key]
            assert not w.isEnabled()
            assert all(not c.isEnabled() for c in getattr(w, "_param_row_widgets", ()))

        out = read_config_widgets(widgets_dict, daw_widgets)
        assert out["daw_processors"]["protools"]["protools_transfer_backend"] == "offline_ptx"
        load_config_widgets(widgets_dict, preset, daw_widgets)
        assert combo.currentData() == "ptsl_hybrid" and all(pt_fields[k].isEnabled() for k in PTSL_ONLY_PARAM_KEYS)
        assert dlg._config_tree_dirty_keys() == set()
        dlg_pt = dict(dlg._cfg_widgets["daw_processors.protools"])
        dlg_combo = dlg_pt["protools_transfer_backend"]
        dlg_combo.setCurrentIndex(dlg_combo.findData("offline_ptx"))
        assert "daw_processors.protools" in dlg._config_tree_dirty_keys()

        # Direct deep-link into Config Presets › Pro Tools (cold-start recovery jump)
        dlg_jump = PreferencesDialog(cfg, initial_preset_section="daw_processors.protools")
        try:
            assert dlg_jump._tabs.currentIndex() == dlg_jump._preset_tab_index
            assert dlg_jump._preset_tree.currentItem() is dlg_jump._preset_items["daw_processors.protools"]
        finally:
            _dispose(dlg_jump)
    finally:
        _dispose(tree, stack, dlg)


def test_protools_templates_widget_3_columns_browse_clear_deselect_and_remove(qapp, monkeypatch):
    w = ProToolsTemplatesWidget()
    try:
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        w.set_templates([
            {"group": "bnz", "name": "MixA", "ptxt_path": "D:/Templates/MixA.ptxt"},
            {"group": "Mixing", "name": "MixB"},
        ])
        assert w._table.columnCount() == 3 and "Auto" in w._table.item(1, 2).text()
        assert w.get_templates() == [
            {"group": "bnz", "name": "MixA", "ptxt_path": "D:/Templates/MixA.ptxt"},
            {"group": "Mixing", "name": "MixB"},
        ]

        # Selection enables Browse/Clear; Clear Path reverts row 0; clearing selection disables buttons!
        w._table.selectRow(0)
        assert w._browse_btn.isEnabled() and w._clear_btn.isEnabled() and w._remove_btn.isEnabled()
        w._on_clear_path()
        assert w.get_templates()[0] == {"group": "bnz", "name": "MixA"} and not w._clear_btn.isEnabled()
        w._table.selectionModel().clearSelection()
        assert not w._browse_btn.isEnabled() and not w._remove_btn.isEnabled()

        # Add + Browse auto-fills empty Template Name from .ptxt stem, then Remove deletes row
        w._on_add()
        w._table.selectRow(2)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("/studio/NewMix.ptxt", ""))
        w._on_browse_ptxt()
        assert w.get_templates()[2] == {"group": "", "name": "NewMix", "ptxt_path": "/studio/NewMix.ptxt"}
        w._on_remove()
        assert len(w.get_templates()) == 2

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        w.set_templates([{"group": "Linux", "name": "MixL"}])
        assert "Linux" in w._table.item(0, 2).text() and w.get_templates() == [{"group": "Linux", "name": "MixL"}]
    finally:
        _dispose(w)


def test_phase3_offline_ptx_ready_not_ready_stale_race_fetch_and_switch(qapp, tmp_path: Path, monkeypatch):
    ptxt = tmp_path / "MixTpl.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=True))
    proj_dir = tmp_path / "Projects"
    proj_dir.mkdir()
    shell = _FakeShell()
    state = WorkspaceState(None, config=_make_config(str(ptxt), str(proj_dir), "offline_ptx", add_dawproject=True))
    ctrl = DawTransferController(
        state, shell,
        on_capture_session_state=lambda: {},
        on_enqueue_batch_item=lambda _i: False,
        on_is_prepare_running=lambda: False,
    )
    try:
        assert ctrl.page.daw_check_label.text() == "\u25cf Offline .ptx ready"
        assert not ctrl.page.offline_not_ready_banner.isVisibleTo(ctrl.page)
        assert "Pro Tools (Offline .ptx): Gain:" in ctrl.page.ptx_summary_label.text()

        t1 = TrackContext("Kick.wav", str(tmp_path / "Kick.wav"), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
        state.session = SessionContext(
            tracks=[t1], output_tracks=[t1], config={},
            transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
            daw_state={state.active_daw_processor.id: {"folders": [{"id": "f1", "name": "KIK"}], "assignments": {"Kick.wav": "f1"}}},
        )
        ctrl.update_lifecycle_buttons()
        assert ctrl.page.transfer_action.isEnabled()

        # Stale check result from a DIFFERENT processor must not overwrite Offline Ready badge or pop a modal
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No modal")))
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No critical")))
        stale_proc = ProToolsDawProcessor(instance_index=99, instance_group="g", instance_name="Old")
        ctrl.check_worker = type("_W", (), {"_processor": stale_proc})()  # type: ignore[assignment]
        ctrl._on_daw_check_result(False, "PTSL unreachable")
        assert ctrl.page.daw_check_label.text() == "\u25cf Offline .ptx ready"

        # Offline fetch result updates concise status bar; failure suppresses QMessageBox.critical
        ctrl.fetch_worker = type("_F", (), {"_processor": state.active_daw_processor, "_session": state.session})()  # type: ignore[assignment]
        ctrl._on_daw_fetch_result(True, "Fetched", state.session)
        assert any("Offline .ptx Writer" in s and "MixTpl.ptxt" in s for s in shell.statuses)
        ctrl.fetch_worker = type("_F", (), {"_processor": state.active_daw_processor, "_session": state.session})()  # type: ignore[assignment]
        ctrl._on_daw_fetch_result(False, "Corrupt .ptxt", None)

        # Valid .ptxt + missing Project directory -> Not-Ready banner visible, Create disabled
        state.config = _make_config(str(ptxt), str(tmp_path / "MissingDir"), "offline_ptx", add_dawproject=True)
        ctrl.refresh_from_config()
        assert ctrl.page.daw_check_label.text() == "\u25b2 Offline .ptx not ready"
        assert ctrl.page.offline_not_ready_banner.isVisibleTo(ctrl.page) and not ctrl.page.transfer_action.isEnabled()

        # Switch Offline Not-Ready -> DAWproject -> Offline Not-Ready hides & restores banner cleanly
        ctrl.page.daw_combo.setCurrentIndex(1)
        assert not ctrl.page.offline_not_ready_banner.isVisibleTo(ctrl.page)
        ctrl.page.daw_combo.setCurrentIndex(0)
        assert ctrl.page.offline_not_ready_banner.isVisibleTo(ctrl.page)
    finally:
        ctrl.dispose()
        _dispose(ctrl.page)


def test_protools_config_params_order_and_choice_labels():
    keys = [p.key for p in ProToolsDawProcessor.config_params()]
    assert keys[:3] == ["protools_enabled", "protools_transfer_backend", "protools_project_dir"]
    spec = ProToolsDawProcessor.config_params()[1]
    assert spec.choices == ["ptsl_hybrid", "offline_ptx"]
    assert spec.choice_labels == {
        "ptsl_hybrid": "Live PTSL + .ptx Hybrid",
        "offline_ptx": "Offline .ptx Writer",
    }
