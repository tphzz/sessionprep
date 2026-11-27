"""GUI unit tests for DAWproject Preferences, Routing & Folders, and Signal Flow Table/Graph."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QTableWidget, QTabWidget

from sessionpreplib.daw_processors.dawproject import DawProjectDawProcessor
from sessionpreplib.daw_processors.dawproject_signal_flow import (
    dawproject_config_params,
    preview_dawproject_signal_flow,
)
from sessionpreplib.daw_processors.dawproject_vst3 import (
    VstPresetChunk,
    VstPresetData,
    build_vstpreset,
)
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionprepgui.daw.controller import DawTransferController
from sessionprepgui.daw.page import DawSetupPage
from sessionprepgui.daw.signal_flow_controller import SignalFlowController
from sessionprepgui.daw.signal_flow_panel import SignalFlowPanel
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.tracks.table_controller import TrackTableController
from tests.unit.qt_helpers import dispose_window


def _qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


def _make_tc_vstpreset(gain_db: float = 0.0) -> bytes:
    params: dict[str, float] | str = "default" if abs(gain_db) < 1e-6 else {"kOpGain": (gain_db + 200.0) / 218.0}
    payload = json.dumps({"settings": {"currentAB": 0}, "ab": [{"parameters": params}]}).encode()
    comp = b"DMGAHDR\x00DGCT\x00\x00\x00\x00ZIP " + zlib.compress(payload)
    return build_vstpreset(
        VstPresetData(
            version=1,
            class_id="565354444D4741547243740000000001",
            chunks=(VstPresetChunk(b"Comp", comp),),
        )
    )


def _write_dawproject_template(path: Path, *, with_vst3_donor: bool = True) -> None:
    devices_xml = (
        '<Devices><Vst3Plugin id="p_tc" name="TrackControl" deviceName="TrackControl" '
        'deviceID="11111111-2222-3333-4444-555555555555" deviceVendor="DMG Audio" loaded="true">'
        '<Enabled id="p_tc_en" value="true"/><State path="plugins/tc.vstpreset"/></Vst3Plugin></Devices>'
        if with_vst3_donor
        else '<Devices><Vst2Plugin id="p_vst2" name="TrackControl" deviceID="1234" loaded="true"/></Devices>'
    )
    xml = f"""<Project version="1.0">
  <Application name="Cubase" version="14"/>
  <Structure>
    <Track id="f_vox" name="VOX" contentType="tracks" loaded="true">
      <Channel id="ch_vox" role="master" audioChannels="2" name="VOX" destination="ch_vox_bus">
        <Volume id="v_vox" value="1.0" unit="linear" min="0.0" max="2.0"/>
      </Channel>
      <Track id="f_sub_vox" name="Sub-Vox" contentType="tracks" loaded="true"/>
    </Track>
    <Track id="f_empty" name="EB" contentType="tracks" loaded="true">
      <Channel id="ch_eb" role="master" audioChannels="2" name="EB" destination="ch_eb_bus"/>
    </Track>
    <Channel id="ch_vox_bus" role="submix" audioChannels="2" name="cVOX" destination="ch_mix">
      {devices_xml}
      <Volume id="v_vb" value="1.0" unit="linear" min="0.0" max="2.0"/>
    </Channel>
    <Channel id="ch_eb_bus" role="submix" audioChannels="2" name="cBAS" destination="ch_mix"/>
    <Channel id="ch_mix" role="submix" audioChannels="2" name="MIX 1" destination="ch_master">
      <Volume id="v_mix" value="1.0" unit="linear" min="0.0" max="2.0"/>
    </Channel>
    <Channel id="ch_master" role="master" audioChannels="2" name="Stereo Out"/>
  </Structure>
</Project>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("project.xml", xml)
        zf.writestr("metadata.xml", "<MetaData/>")
        if with_vst3_donor:
            zf.writestr("plugins/tc.vstpreset", _make_tc_vstpreset(0.0))


def _make_vox_session(tmp_path: Path, db_peak: float = -12.0, gain_db: float = 8.0) -> tuple[SessionContext, TrackContext]:
    wav = tmp_path / "Lead_Vox.wav"
    t = np.arange(24000, dtype=np.float64) / 48000.0
    sig = (10.0 ** (db_peak / 20.0)) * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(wav, np.column_stack((sig, sig)), 48000, subtype="PCM_24")
    res = ProcessorResult(
        processor_id="bimodal_normalize", classification="Sustained", method="RMS",
        gain_db=gain_db, data={"fader_offset": -gain_db},
    )
    tc = TrackContext(
        filename="Lead_Vox.wav", filepath=str(wav), audio_data=None, samplerate=48000, channels=2,
        total_samples=24000, bitdepth="24-bit", subtype="PCM_24", duration_sec=0.5,
        processor_results={"bimodal_normalize": res},
    )
    sess = SessionContext(
        [tc], {"_use_processed": False, "target_rms": -18.0, "target_peak": -6.0},
        output_tracks=[tc],
        transfer_manifest=[TransferEntry("e1", tc.filename, "Lead_Vox")],
    )
    return sess, tc


class _DummyShell:
    def register_phase_progress_layout(self, _phase: int, _layout) -> None:
        pass

    def is_batch_mode(self) -> bool:
        return False

    def is_closing_after_daw(self) -> bool:
        return False

    def resume_close_after_daw(self) -> None:
        pass

    def dialog_parent(self):
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

    def show_status(self, _msg: str) -> None:
        pass


def _build_daw_fixture(tmp_path: Path, *, apply_trim_gain: bool = False):
    tpl_ok = tmp_path / "Cubase_Mix.dawproject"
    tpl_vst2 = tmp_path / "Legacy_VST2.dawproject"
    tpl_missing = tmp_path / "Missing_Template.dawproject"
    _write_dawproject_template(tpl_ok, with_vst3_donor=True)
    _write_dawproject_template(tpl_vst2, with_vst3_donor=False)
    session, _tc = _make_vox_session(tmp_path)
    cfg = build_defaults()
    dp_cfg = cfg["config_presets"]["Default"]["daw_processors"]["dawproject"]
    dp_cfg["dawproject_apply_trim_gain"] = apply_trim_gain
    dp_cfg["dawproject_templates"] = [
        {"name": "Cubase Mix", "template_path": str(tpl_ok), "fader_ceiling_db": 6.0},
        {"name": "Legacy VST2", "template_path": str(tpl_vst2), "fader_ceiling_db": 6.0},
        {"name": "Missing Tpl", "template_path": str(tpl_missing), "fader_ceiling_db": 6.0},
    ]
    state = WorkspaceState(config=cfg)
    state.session = session
    shell = _DummyShell()
    ctrl = DawTransferController(
        state, shell,  # type: ignore[arg-type]
        on_capture_session_state=lambda: {},
        on_enqueue_batch_item=lambda _item: False,
        on_is_prepare_running=lambda: False,
    )
    dummy_track_table = QTableWidget()
    table_ctrl = TrackTableController(
        state, shell,  # type: ignore[arg-type]
        track_table=dummy_track_table,  # type: ignore[arg-type]
        setup_table=ctrl.page.setup_table,
        on_get_dawproject_topology=lambda: ctrl.signal_flow_ctrl.current_topology,
    )
    return state, session, ctrl, table_ctrl, dummy_track_table


def _select_combo(ctrl: DawTransferController, substr: str) -> None:
    idx = next(i for i in range(ctrl.page.daw_combo.count()) if substr in ctrl.page.daw_combo.itemText(i))
    ctrl.page.daw_combo.setCurrentIndex(idx)


def test_preferences_exposes_all_6_dawproject_params_and_preserves_widgets():
    """Preferences exposes all 6 DAWproject ParamSpecs + DawProjectTemplatesWidget and existing nav items."""
    _qapp()
    dlg = PreferencesDialog(build_defaults())
    try:
        dp_widgets = dlg._cfg_widgets.get("daw_processors.dawproject", [])
        keys = [k for k, _w in dp_widgets]
        for spec in dawproject_config_params():
            assert spec.key in keys
        assert "dawproject" in dlg._cfg_daw_custom_widgets
        nav_labels = [item.text(0) for item in dlg._preset_items.values()]
        assert "DAWproject" in nav_labels
        assert "Subsonic Content" in nav_labels
        assert "Corrective processing" in nav_labels
    finally:
        dispose_window(dlg)


def test_dawproject_routing_tree_nested_basic_folder_and_unity_table(tmp_path):
    """Enriched 4-col tree handles nested Basic folders without false pruning or raw XML IDs, and unity table shows 0.0 dB."""
    _qapp()
    state, session, ctrl, table_ctrl, dummy_tbl = _build_daw_fixture(tmp_path, apply_trim_gain=False)
    try:
        _select_combo(ctrl, "Cubase Mix")
        dp = state.active_daw_processor
        assert isinstance(dp, DawProjectDawProcessor)
        session.daw_state[dp.id] = {
            "folders": [
                {"id": "f_vox", "name": "VOX", "folder_type": "routing", "index": 0, "parent_id": None},
                {"id": "f_sub_vox", "name": "Sub-Vox", "folder_type": "basic", "index": 1, "parent_id": "f_vox"},
                {"id": "f_empty", "name": "EB", "folder_type": "routing", "index": 2, "parent_id": None},
            ],
            "assignments": {"e1": "f_sub_vox"},
            "track_order": {"f_sub_vox": ["e1"]},
        }
        ctrl.sync_protools_ui_mode()
        ctrl.populate_folder_tree()
        table_ctrl.populate_setup_table()

        assert ctrl.page.setup_table.horizontalHeaderItem(4).text() == "Slot 1 Trim"
        assert ctrl.page.setup_table.item(0, 4).text() == "0.0 dB"
        assert ctrl.page.setup_table.item(0, 5).text() == "0.0 dB"
        assert "VST3 Trim: TrackControl (unity 0.0 dB" in ctrl.page.ptx_summary_label.text()
        assert ctrl.page.no_donor_banner.isHidden()

        vox_item = ctrl.page.folder_tree.topLevelItem(0)
        sub_item = vox_item.child(0)
        eb_item = ctrl.page.folder_tree.topLevelItem(1)
        # Parent VOX has 0 direct tracks but populated child Sub-Vox -> must NOT be marked pruned!
        assert "pruned" not in vox_item.text(3)
        assert "pruned" in eb_item.text(3)
        # Track inside Basic folder Sub-Vox resolves human-readable ancestor channel name "VOX", not raw ID "ch_vox"
        child_trk = sub_item.child(0)
        assert child_trk.text(2) == "TrackControl (VST3, 0.0 dB)"
        assert child_trk.text(3) == "Route \u2192 Ch: VOX"

        seen_daw: list[str] = []
        ctrl.edit_daw_config_requested.connect(seen_daw.append)
        ctrl.page.edit_config_btn.click()
        assert seen_daw == ["DAWproject"]
    finally:
        ctrl.dispose()
        dispose_window(dummy_tbl)
        dispose_window(ctrl.page)


def test_dawproject_calibrated_donorless_fallback_missing_tpl_and_daw_switch_hygiene(tmp_path):
    """Calibrated mode on VST2-only/missing templates fails closed to 0.0 dB; DAW switches clear stale state cleanly."""
    _qapp()
    state, session, ctrl, table_ctrl, dummy_tbl = _build_daw_fixture(tmp_path, apply_trim_gain=True)
    try:
        # Clean calibrated template shows non-zero Slot 1 Trim (+8.0 dB / -8.0 dB)
        _select_combo(ctrl, "Cubase Mix")
        table_ctrl.populate_setup_table()
        assert ctrl.page.setup_table.item(0, 4).text() == "+8.0 dB"
        assert ctrl.page.setup_table.item(0, 5).text() == "-8.0 dB"

        # Simulate Pro Tools rollback banner & leaf shift, then switch to VST2-only DAWproject template
        ctrl.page.show_rollback_notice("PTX rollback error")
        ctrl._last_leaf_shift_db = -3.0
        _select_combo(ctrl, "Legacy VST2")
        table_ctrl.populate_setup_table()
        assert ctrl.page.rollback_banner.isHidden()
        assert ctrl._last_leaf_shift_db == 0.0
        assert not ctrl.page.no_donor_banner.isHidden()
        assert "No VST3 Trim donor" in ctrl.page.no_donor_label.text()
        # Fail-closed parity: Setup Table displays unity 0.0 dB instead of misleading +8/-8 dB
        assert ctrl.page.setup_table.item(0, 4).text() == "0.0 dB"
        assert ctrl.page.setup_table.item(0, 5).text() == "0.0 dB"

        # Missing template file shows "Template unavailable", hides no-donor banner, and fails closed to 0.0 dB
        _select_combo(ctrl, "Missing Tpl")
        table_ctrl.populate_setup_table()
        assert ctrl.page.no_donor_banner.isHidden()
        assert "Template unavailable" in ctrl.page.ptx_summary_label.text()
        assert ctrl.page.setup_table.item(0, 4).text() == "0.0 dB"

        # Workspace clear resets amber no-donor banner cleanly
        _select_combo(ctrl, "Legacy VST2")
        assert not ctrl.page.no_donor_banner.isHidden()
        ctrl.clear_workspace()
        assert ctrl.page.no_donor_banner.isHidden()
    finally:
        ctrl.dispose()
        dispose_window(dummy_tbl)
        dispose_window(ctrl.page)


def test_dawproject_signal_flow_table_graph_and_unity_path_equation(tmp_path):
    """SignalFlowPanel renders Slot 1 / S1 / +6.0 dB ceiling in calibrated mode and avoids CG leak in unity mode."""
    _qapp()
    tpl = tmp_path / "Cubase_Mix.dawproject"
    _write_dawproject_template(tpl, with_vst3_donor=True)
    session, _tc = _make_vox_session(tmp_path, db_peak=-10.0, gain_db=6.0)
    folders = [{"id": "f_vox", "name": "VOX", "folder_type": "routing", "index": 0, "parent_id": None}]

    g_cal = preview_dawproject_signal_flow(
        tpl, session, {"e1": "f_vox"}, folders, gain_mode="trim_and_fader", apply_trim_gain=True, fader_ceiling_db=6.0,
    ).to_dict()
    g_unity = preview_dawproject_signal_flow(
        tpl, session, {"e1": "f_vox"}, folders, gain_mode="trim_and_fader", apply_trim_gain=False, fader_ceiling_db=6.0,
    ).to_dict()

    panel = SignalFlowPanel()
    try:
        panel.set_daw_context("dawproject", fader_ceiling_db=6.0, apply_trim_gain=True)
        panel.show_graph(g_cal, staging_enabled=True)
        assert panel.table.horizontalHeaderItem(3).text() == "Slot 1 Trim"
        assert "Trim: Calibrated (Exp.)" in panel.target_label.text()
        assert "Ceiling: +6.0 dB" in panel.target_label.text()
        assert "Atomic .dawproject ZIP export" in panel.footer_label.text()
        assert panel._spin_by_bus and all(s.maximum() == pytest.approx(6.0) for s in panel._spin_by_bus.values())
        for card in panel.graph_view._card_items.values():
            labels = [lbl for lbl, _val, _col in card._lines]
            assert "Slot A" not in labels and "CG / Fdr" not in labels
            if card.node_name == "cVOX":
                assert "S1" in labels

        # Unity mode Path Verification Bar uses Trim (never Pro Tools "CG")
        panel.set_daw_context("dawproject", fader_ceiling_db=6.0, apply_trim_gain=False)
        panel.show_graph(g_unity, staging_enabled=True)
        panel._update_path_bar("cVOX")
        assert "Trim" in panel.path_bar.text()
        assert " CG " not in panel.path_bar.text()
        assert "Unity 0.0 dB (Default)" in panel.target_label.text()
    finally:
        dispose_window(panel)


def test_dawproject_signal_flow_controller_async_worker(tmp_path):
    """SignalFlowController runs offline DAWproject preview asynchronously and updates tab chrome with 'buses'."""
    app = _qapp()
    tpl = tmp_path / "Cubase_Mix.dawproject"
    _write_dawproject_template(tpl, with_vst3_donor=True)
    session, _tc = _make_vox_session(tmp_path, db_peak=-10.0, gain_db=6.0)
    folders = [{"id": "f_vox", "name": "VOX", "folder_type": "routing", "index": 0, "parent_id": None}]

    tabs = QTabWidget()
    panel = SignalFlowPanel()
    tabs.addTab(DawSetupPage(), "Routing")
    tabs.addTab(panel, "Signal Flow")
    state = WorkspaceState(config=build_defaults())
    dp = DawProjectDawProcessor(template_path=str(tpl), template_fader_ceiling_db=6.0)
    state.active_daw_processor = dp
    session.daw_state[dp.id] = {"folders": folders, "assignments": {"e1": "f_vox"}}
    state.session = session

    sf_ctrl = SignalFlowController(
        state, panel, tabs,
        get_pt_options=lambda: {},
        get_dp_options=lambda: {
            "gain_mode": "trim_and_fader", "insert_track_trim": True, "apply_trim_gain": True,
            "stage_bus_faders": True, "pan_law_db": -3.0, "remove_unused_buses": True, "fader_ceiling_db": 6.0,
        },
    )
    try:
        sf_ctrl.refresh_template_topology()
        done: list[dict] = []
        sf_ctrl.signal_flow_updated.connect(lambda g: done.append(g) if isinstance(g, dict) else None)
        sf_ctrl.compute_preview()
        assert sf_ctrl._worker is not None
        sf_ctrl._worker.wait(5000)
        app.processEvents()
        assert done and "nodes" in done[-1]
        assert "buses staged" in tabs.tabText(1) or "1 bus staged" in tabs.tabText(1)
    finally:
        sf_ctrl.dispose()
        dispose_window(tabs)
