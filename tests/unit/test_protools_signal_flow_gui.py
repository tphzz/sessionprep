"""Unit tests for Pro Tools Signal Flow GUI, offline preview, overrides, and reporting."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import soundfile as sf
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.daw_processors.protools_ptx import GAIN_MODE_CLIP_AND_FADER
from sessionpreplib.daw_processors.protools_ptx_preview import (
    inspect_protools_template_topology,
    preview_protools_signal_flow,
)
from sessionpreplib.models import (
    DawCommand,
    DawCommandResult,
    ProcessorResult,
    SessionContext,
    TrackContext,
    TransferEntry,
)
from sessionpreplib.ptx import PtxSession, patch_track_volumes
from sessionpreplib.reports import save_json
from sessionprepgui.change_marker import MarkerTabBar
from sessionprepgui.daw.controller import DawTransferController
from sessionprepgui.daw.page import DawSetupPage
from sessionprepgui.daw.signal_flow_panel import BusFaderSpinBox, SignalFlowPanel
from sessionprepgui.session.io import _validate_daw_state
from sessionprepgui.settings import build_defaults
from sessionprepgui.state import WorkspaceState
from sessionprepgui.tracks.table_controller import TrackTableController
from tests.unit.test_protools_ptx_orchestration import _build_nested_folder_ptx
from tests.unit.test_ptx_signal_flow import _build_console_ptx, _sustained_tone, _transient_hit


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _make_shell(qapp):
    return SimpleNamespace(
        register_phase_progress_layout=lambda *_a: None,
        is_batch_mode=lambda: False,
        is_closing_after_daw=lambda: False,
        resume_close_after_daw=lambda: None,
        dialog_parent=lambda: None,
        progress_start=lambda *_a: None,
        progress_message=lambda *_a: None,
        progress_value=lambda *_a: None,
        progress_finish=lambda *_a: None,
        progress_fail=lambda *_a: None,
        show_status=lambda *_a: None,
    )


def _pump_until_idle(qapp, ctrl: DawTransferController, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while (ctrl.signal_flow_ctrl.is_running() or ctrl.signal_flow_ctrl._override_timer.isActive()) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_offline_template_topology_preview_overrides_vca_and_json_report(tmp_path):
    # Patch KIK-VCA to -6.0 dB so offline preview proves VCA offsets apply to virtual tracks
    ptx_bytes = _build_nested_folder_ptx(folder_has_input=True)
    ptx_bytes, _ = patch_track_volumes(ptx_bytes, {"KIK-VCA": -6.0})
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(ptx_bytes)

    topo = inspect_protools_template_topology(ptxt)
    kik_meta = topo.lookup("", "KIK")
    sub_meta = topo.lookup("", "Sub")
    assert kik_meta is not None and kik_meta.input_bus == "KIK" and kik_meta.vca_track == "KIK-VCA"
    assert sub_meta is not None and sub_meta.input_bus is None and sub_meta.resolved_routing_bus == "KIK"

    sr = 48000
    k_wav = tmp_path / "Kick1.wav"
    sf.write(k_wav, _transient_hit(sr, 0.5, 10.0 ** (-6.0 / 20.0), [0.1, 0.3]), sr, subtype="PCM_24")
    t_vca = TrackContext("Kick1.wav", str(k_wav), None, sr, 1, sr // 2, "24", "PCM_24", 0.5)
    ptx_ref0 = PtxSession.from_bytes(ptx_bytes)
    rf_id = ptx_ref0.track_by_name("KIK").uuid_braced
    sess_vca = SessionContext(
        tracks=[t_vca], output_tracks=[t_vca],
        config={"target_peak": -6.0, "target_rms": -18.0},
        transfer_manifest=[TransferEntry("Kick1.wav", "Kick1.wav", "Kick1")],
    )
    g_vca = preview_protools_signal_flow(
        ptxt, sess_vca, {"Kick1.wav": rf_id},
        [{"id": rf_id, "name": "KIK", "folder_type": "routing", "parent_id": None}],
        gain_mode=GAIN_MODE_CLIP_AND_FADER,
        auto_group_folders=True, link_template_vcas=True,
    )
    assert g_vca.nodes["Kick1"].vca_offset_db == pytest.approx(-6.0, abs=0.1)

    # Disabling auto_route prevents virtual leaf tracks from routing into parent folders
    g_noroute = preview_protools_signal_flow(
        ptxt, sess_vca, {"Kick1.wav": rf_id},
        [{"id": rf_id, "name": "KIK", "folder_type": "routing", "parent_id": None}],
        auto_route=False, gain_mode=GAIN_MODE_CLIP_AND_FADER,
    )
    assert g_noroute.nodes["KIK"].upstream_tracks == []

    # Offline preview on full console template with manual override, unreadable WAV, missing WAV, and NaN override
    console_ptxt = tmp_path / "Console.ptxt"
    console_ptxt.write_bytes(_build_console_ptx())
    ptx_ref = PtxSession.from_bytes(console_ptxt.read_bytes())
    k2_wav = tmp_path / "Kick2.wav"
    sf.write(k2_wav, _transient_hit(sr, 0.5, 10.0 ** (-6.0 / 20.0), [0.1, 0.3]), sr, subtype="PCM_24")
    bad_wav = tmp_path / "Bad.wav"
    bad_wav.write_bytes(b"NOT_A_WAV")
    missing_wav = tmp_path / "Missing.wav"

    t1 = TrackContext("Kick1.wav", str(k_wav), None, sr, 1, sr // 2, "24", "PCM_24", 0.5)
    t2 = TrackContext("Kick2.wav", str(k2_wav), None, sr, 1, sr // 2, "24", "PCM_24", 0.5)
    tbad = TrackContext("Bad.wav", str(bad_wav), None, sr, 1, sr // 2, "24", "PCM_24", 0.5)
    tmiss = TrackContext("Missing.wav", str(missing_wav), None, sr, 1, sr // 2, "24", "PCM_24", 0.5)
    session = SessionContext(
        tracks=[t1, t2, tbad, tmiss],
        output_tracks=[t1, t2, tbad, tmiss],
        config={"target_peak": -6.0, "target_rms": -18.0, "_source_dir": str(tmp_path)},
        transfer_manifest=[
            TransferEntry("Kick1.wav", "Kick1.wav", "Kick1"),
            TransferEntry("Kick2.wav", "Kick2.wav", "Kick2"),
            TransferEntry("Bad.wav", "Bad.wav", "Bad"),
            TransferEntry("Missing.wav", "Missing.wav", "Missing"),
        ],
    )
    kik_id = ptx_ref.track_by_name("KIK").uuid_braced
    snr_id = ptx_ref.track_by_name("SNR").uuid_braced
    folders = [
        {"id": kik_id, "name": "KIK", "folder_type": "routing", "parent_id": None},
        {"id": snr_id, "name": "SNR", "folder_type": "routing", "parent_id": None},
    ]
    assignments = {
        "Kick1.wav": kik_id, "Kick2.wav": kik_id, "Bad.wav": snr_id, "Missing.wav": snr_id,
    }

    graph = preview_protools_signal_flow(
        console_ptxt, session, assignments, folders,
        gain_mode=GAIN_MODE_CLIP_AND_FADER,
        bus_fader_overrides={"KIK": -2.5, "SNR": float("nan")},
    )
    assert "Bad.wav" in graph.excluded_tracks
    assert "Missing.wav" in graph.excluded_tracks
    assert graph.nodes["KIK"].staged is True
    assert graph.nodes["KIK"].recommended_fader_db == pytest.approx(-2.5, abs=0.05)
    assert graph.nodes["KIK"].is_overridden is True

    # Stale prepared audio raises a clear actionable error when Use Processed is enabled
    stale_sess = SessionContext(
        tracks=[t1], output_tracks=[t1],
        config={"_use_processed": True}, prepare_state="stale",
        transfer_manifest=[TransferEntry("Kick1.wav", "Kick1.wav", "Kick1")],
    )
    with pytest.raises(ValueError, match="Prepared audio is out of date"):
        preview_protools_signal_flow(console_ptxt, stale_sess, {"Kick1.wav": kik_id}, folders)

    # JSON report exports RFC 8259 compliant daw_signal_flow (sanitizes non-finite floats)
    sf_dict = graph.to_dict()
    sf_dict["nodes"]["KIK"]["pre_fader_rms_db"] = float("nan")
    session.daw_state["protools_0"] = {"signal_flow": sf_dict}
    out_json = tmp_path / "report.json"
    save_json(session, session.config, str(out_json), [])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "daw_signal_flow" in payload
    assert payload["daw_signal_flow"]["protools_0"]["nodes"]["KIK"]["recommended_fader_db"] == pytest.approx(-2.5, abs=0.05)
    assert payload["daw_signal_flow"]["protools_0"]["nodes"]["KIK"]["pre_fader_rms_db"] is None


def test_protools_fetch_preserves_overrides_and_session_io_validates_daw_state():
    dp = ProToolsDawProcessor()
    session = SessionContext(tracks=[], config={})
    session.daw_state[dp.id] = {
        "folders": [{"id": "f1", "name": "KIK", "folder_type": "routing", "parent_id": None, "index": 0}],
        "assignments": {"Kick.wav": "f1", "Gone.wav": "f_removed"},
        "bus_fader_overrides": {"KIK": -3.0},
        "track_order": {"f1": ["Kick.wav"], "f_removed": ["Gone.wav"]},
        "signal_flow": {"nodes": {}},
    }
    dp._apply_fetched_folders(
        session, [{"id": "f1", "name": "KIK", "folder_type": "routing", "parent_id": None, "index": 0}],
    )
    new_state = session.daw_state[dp.id]
    assert new_state["bus_fader_overrides"] == {"KIK": -3.0}
    assert new_state["track_order"] == {"f1": ["Kick.wav"]}
    assert new_state["assignments"] == {"Kick.wav": "f1"}
    assert "signal_flow" not in new_state

    # Valid bus_fader_overrides & signal_flow pass validation; corrupt ones fail
    _validate_daw_state({"protools_0": {"folders": [], "assignments": {}, "bus_fader_overrides": {"KIK": -2.0}, "signal_flow": {}}})
    with pytest.raises(ValueError, match="bus fader overrides"):
        _validate_daw_state({"protools_0": {"folders": [], "assignments": {}, "bus_fader_overrides": {"KIK": float("nan")}}})
    with pytest.raises(ValueError, match="signal flow"):
        _validate_daw_state({"protools_0": {"folders": [], "assignments": {}, "signal_flow": "corrupt"}})


def test_signal_flow_panel_states_spinbox_focus_wheel_narrow_and_formatting(qapp):
    panel = SignalFlowPanel()
    panel.resize(1100, 700)
    panel.show()

    # Controls bar splits actions across two narrow-safe rows (never sums buttons + checkbox + target on one line)
    row1_min_w = (
        panel.view_toggle.minimumSizeHint().width()
        + panel.refresh_btn.minimumSizeHint().width()
        + panel.reset_all_btn.minimumSizeHint().width()
        + 36
    )
    assert panel.controls_bar.minimumSizeHint().width() <= row1_min_w
    assert panel.controls_bar.minimumSizeHint().width() < row1_min_w + panel.active_only_cb.minimumSizeHint().width()

    # Empty state with 0 assigned tracks vs assigned tracks ready to compute
    panel.show_empty(assigned_count=0, folder_count=0)
    assert not panel.compute_empty_btn.isEnabled()
    panel.show_empty(assigned_count=3, folder_count=2, template_name="Mix.ptxt")
    assert panel.compute_empty_btn.isEnabled()

    # Busy state disables Refresh & Reset Overrides, and Escape emits cancel_requested
    cancelled: list[bool] = []
    panel.cancel_requested.connect(lambda: cancelled.append(True))
    panel.show_busy(5, 10, "Summing...")
    assert not panel.refresh_btn.isEnabled()
    assert not panel.reset_all_btn.isEnabled()
    esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(esc)
    assert cancelled == [True]

    # Error state card
    panel.show_error("Template missing", "Cannot find Mix.ptxt")
    assert "Template missing" in panel.error_title.text()

    # Graph with 0.0 dBFS Master peak, Slot A Trim aux, parked aux, validation strip, path bar, and graph view
    graph = {
        "target_peak_db": -6.0, "target_rms_db": -18.0, "pan_law_db": -3.0, "dbfs_convention": "standard",
        "topological_order": ["Kick", "KIK", "cKIK", "cBAS", "MIX 1", "Master", "RYT-PA"],
        "edges": [
            {"source": "Kick", "target": "KIK", "bus_name": "KIK", "kind": "main"},
            {"source": "KIK", "target": "cKIK", "bus_name": "cKIK", "kind": "main"},
            {"source": "cKIK", "target": "MIX 1", "bus_name": "MIX 1", "kind": "main"},
            {"source": "cBAS", "target": "MIX 1", "bus_name": "MIX 1", "kind": "main"},
            {"source": "MIX 1", "target": "Master", "bus_name": "Out 1-2", "kind": "main"},
        ],
        "cycles": [["FX-A", "FX-B", "FX-A"]],
        "excluded_tracks": {"Bad.wav": "corrupt"},
        "recommended_bus_faders": {"cBAS": -4.5, "MIX 1": -1.9},
        "recommended_slot_a_trims": {"cBAS": 4.5},
        "balance_validation": {
            "status": "verified",
            "converged": True,
            "passes": 1,
            "leaf_fader_shift_db": 0.0,
            "global_path_delta_db": -7.9,
            "max_balance_spread_db": 0.0,
            "active_leaf_count": 1,
            "staged_bus_count": 2,
            "summary": "Balance invariant verified (±0.0 dB across 1 tracks) · Global path Δ: -7.9 dB · 2 staged busses",
        },
        "nodes": {
            "Kick": {"track_name": "Kick", "track_type": "Audio", "classification": "Transient", "clip_gain_db": -8.0, "recommended_fader_db": 2.0, "pre_fader_peak_db": -6.0, "post_fader_peak_db": -4.0, "net_gain_delta_db": -6.0},
            "KIK": {"track_name": "KIK", "track_type": "RoutingFolder", "classification": "Transient", "staged": True, "is_overridden": False, "auto_fader_db": 0.0, "recommended_fader_db": 0.0, "pre_insert_peak_db": -8.0, "pre_insert_rms_db": -22.0, "pre_fader_peak_db": -8.0, "pre_fader_rms_db": -22.0, "post_fader_peak_db": -8.0, "post_fader_rms_db": -22.0, "net_gain_delta_db": 0.0, "upstream_tracks": ["Kick"], "active_inserts": []},
            "cKIK": {"track_name": "cKIK", "track_type": "Aux", "classification": "Transient", "staged": False, "is_overridden": False, "template_fader_db": -3.5, "recommended_fader_db": -3.5, "pre_insert_peak_db": -8.0, "pre_insert_rms_db": -22.0, "pre_fader_peak_db": -8.0, "pre_fader_rms_db": -22.0, "post_fader_peak_db": -11.5, "post_fader_rms_db": -25.5, "net_gain_delta_db": 0.0, "upstream_tracks": ["KIK"], "active_inserts": ["SSL 4K E"]},
            "cBAS": {"track_name": "cBAS", "track_type": "Aux", "classification": "Sustained", "staged": True, "is_overridden": False, "auto_fader_db": -4.5, "recommended_fader_db": -4.5, "slot_a_trim_plugin": "TrackControl", "recommended_trim_db": 4.5, "staging_strategy": "trim_and_fader", "pre_insert_peak_db": -14.5, "pre_insert_rms_db": -22.5, "pre_fader_peak_db": -10.0, "pre_fader_rms_db": -18.0, "post_fader_peak_db": -14.5, "post_fader_rms_db": -22.5, "net_gain_delta_db": 0.0, "upstream_tracks": ["EB"], "active_inserts": ["TrackControl", "SSL 4K E"]},
            "MIX 1": {"track_name": "MIX 1", "track_type": "Aux", "classification": "Transient", "is_terminal": True, "staged": True, "is_overridden": False, "auto_fader_db": -1.9, "recommended_fader_db": -1.9, "pre_insert_peak_db": -4.1, "pre_insert_rms_db": -19.5, "pre_fader_peak_db": -4.1, "pre_fader_rms_db": -19.5, "post_fader_peak_db": -6.0, "post_fader_rms_db": -21.4, "net_gain_delta_db": -1.9, "staging_strategy": "fader_only", "upstream_tracks": ["cKIK", "cBAS"], "active_inserts": ["SSL G-Master"]},
            "Master": {"track_name": "Master", "track_type": "Master", "classification": "Transient", "staged": False, "is_overridden": False, "template_fader_db": 0.0, "recommended_fader_db": 0.0, "pre_insert_peak_db": -4.0, "pre_insert_rms_db": -21.4, "pre_fader_peak_db": -4.0, "pre_fader_rms_db": -21.4, "post_fader_peak_db": 0.0, "post_fader_rms_db": -17.4, "net_gain_delta_db": 0.0, "upstream_tracks": ["MIX 1"], "active_inserts": []},
            "RYT-PA": {"track_name": "RYT-PA", "track_type": "Aux", "classification": "Silent", "staged": False, "is_overridden": False, "is_parallel_parked": True, "upstream_tracks": [], "active_inserts": []},
        },
    }
    panel.show_graph(graph, staging_enabled=True)
    assert not panel.reset_all_btn.isEnabled()
    assert panel.balance_banner.isVisible() and panel.cycle_banner.isVisible() and panel.excluded_banner.isVisible()
    assert "active inserts unsimulated" in panel._balance_lbl.text()
    assert "cBAS (SSL 4K E)" in panel.balance_banner.toolTip()
    assert "TrackControl" not in panel.balance_banner.toolTip()
    # 0.0 dBFS Master post-fader peak must not fall back to pre-fader -4.0 dB
    assert "Master Peak: 0.0 dB" in panel.footer_label.text()
    assert "cBAS (Trim +4.5 dB, Fader -4.5 dB)" in panel.footer_label.text()
    assert panel.path_bar.isVisible() and "Path:" in panel.path_bar.text()

    # Tier header rows are non-selectable; 9 calculation columns show Slot A Trim, Bus Fader, Net Δ, and Notes
    assert not (panel.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsSelectable)
    row_by_name = {
        panel.table.item(r, 0).text(): r
        for r in range(panel.table.rowCount())
        if panel.table.item(r, 0) is not None
    }
    assert panel.table.item(row_by_name["cKIK"], 5).text() == "-3.5 dB (fixed)"
    assert "+4.5 dB (TC)" in panel.table.item(row_by_name["cBAS"], 3).text()
    assert panel.table.item(row_by_name["cBAS"], 7).text() == "0.0 dB \u2713"
    assert "Slot A TrackControl: +4.5 dB" in panel.table.item(row_by_name["cBAS"], 8).text()
    assert "RYT-PA" not in row_by_name

    # _SelectedWhiteTextDelegate forces #ffffff text across all cells when row is selected
    from PySide6.QtWidgets import QStyle, QStyleOptionViewItem
    opt = QStyleOptionViewItem()
    opt.state |= QStyle.StateFlag.State_Selected
    panel.table.itemDelegate().initStyleOption(opt, panel.table.model().index(row_by_name["cBAS"], 3))
    assert opt.palette.color(opt.palette.ColorRole.Text).name().lower() == "#ffffff"
    assert opt.palette.color(opt.palette.ColorRole.HighlightedText).name().lower() == "#ffffff"

    # [Table | Graph] toggle switches views cleanly, claims Space ShortcutOverride, and renders DAG cards
    space_ovr = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    assert panel.view_toggle.table_btn.event(space_ovr) is True and space_ovr.isAccepted()
    panel.view_toggle.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    assert panel.view_stack.currentIndex() == 1
    assert "cBAS" in panel.graph_view._card_items
    # _NodeCardItem boundingRect encompasses focus ring (-2.5..+2.5)
    cbas_card = panel.graph_view._card_items["cBAS"]
    assert cbas_card.boundingRect().left() <= -3.5 and cbas_card.boundingRect().top() <= -3.5
    # Graph view claims Space/Escape ShortcutOverride and Escape clears selection + hides path_bar
    g_space = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    assert panel.graph_view.view.event(g_space) is True and g_space.isAccepted()
    panel.graph_view.select_node("MIX 1", emit=True)
    assert "MIX 1" in panel.path_bar.text()
    panel.graph_view.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert panel._selected_node is None and not panel.path_bar.isVisible()
    panel.graph_view.zoom_in()
    assert panel.graph_view.view.zoom_percent() > 100
    panel.graph_view.zoom_reset()
    assert panel.graph_view.view.zoom_percent() == 100
    # Narrow 360 px pane stays valid without clipping
    panel.resize(360, 640)
    qapp.processEvents()
    assert panel.width() == 360
    panel.view_toggle.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier))
    assert panel.view_stack.currentIndex() == 0

    # Unchecking Active path only reveals parked parallel aux
    panel.active_only_cb.setChecked(False)
    visible_all = [panel.table.item(r, 0).text() for r in range(panel.table.rowCount()) if panel.table.item(r, 0)]
    assert "RYT-PA" in visible_all

    # BusFaderSpinBox StrongFocus, keyboardTracking=False, ignores wheel when unfocused
    spin = BusFaderSpinBox()
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert spin.keyboardTracking() is False
    spin.setValue(-1.9)
    ev = QWheelEvent(
        QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )
    spin.wheelEvent(ev)
    assert not ev.isAccepted()
    assert spin.value() == pytest.approx(-1.9)

    # Manual override uses scoped #overrideCell tint, enables Reset Overrides, and sets accessible names
    graph["nodes"]["MIX 1"]["is_overridden"] = True
    graph["nodes"]["MIX 1"]["recommended_fader_db"] = -1.0
    graph["balance_validation"]["status"] = "override_drift"
    graph["balance_validation"]["summary"] = "Balance shifted by override (MIX 1 +0.9 dB · spread 0.9 dB)"
    panel.show_graph(graph, staging_enabled=True)
    assert panel.reset_all_btn.isEnabled()
    assert panel._balance_btn is not None and panel._balance_btn.isVisible()
    mix_row = next(r for r in range(panel.table.rowCount()) if panel.table.item(r, 0) and panel.table.item(r, 0).text() == "MIX 1")
    override_cell = panel.table.cellWidget(mix_row, 5)
    assert override_cell is not None and override_cell.objectName() == "overrideCell"
    assert panel._spin_by_bus["MIX 1"].accessibleName() == "MIX 1 target fader (dB)"

    # Blocked Slot A shows red SLOT A BLOCKED row & red balance strip even while Table & Graph remain visible
    graph["nodes"]["MIX 1"]["slot_a_blocked_error"] = "Insert Slot A is occupied by non-trim plugin 'SSL G-Master'."
    graph["balance_validation"]["status"] = "blocked"
    graph["balance_validation"]["summary"] = "Slot A blocked on MIX 1 (1 bus)"
    panel.show_graph(graph, staging_enabled=True)
    blk_row = next(r for r in range(panel.table.rowCount()) if panel.table.item(r, 0) and panel.table.item(r, 0).text() == "MIX 1")
    assert panel.table.item(blk_row, 3).text() == "SLOT A BLOCKED"
    assert panel.table.item(blk_row, 7).text() == "BLOCKED"
    graph["nodes"]["MIX 1"].pop("slot_a_blocked_error")
    graph["balance_validation"]["status"] = "verified"

    # Staging disabled -> read-only banner shown, Master stays "(fixed)", eligible bus shows "(staging off)", footer/notes hide staged text
    panel.show_graph(graph, staging_enabled=False)
    assert panel.disabled_banner.isVisible()
    assert not panel.reset_all_btn.isEnabled()
    assert "Staged on Create: none" in panel.footer_label.text()
    assert "SLOT A TRIM" not in panel.table.item(0, 0).text()
    row_by_name_off = {
        panel.table.item(r, 0).text(): r
        for r in range(panel.table.rowCount())
        if panel.table.item(r, 0) is not None
    }
    assert panel.table.item(row_by_name_off["Master"], 5).text() == "0.0 dB (fixed)"
    assert "(staging off)" in panel.table.item(row_by_name_off["KIK"], 5).text()
    assert "(staging off)" in panel.table.item(row_by_name_off["MIX 1"], 5).text()
    assert "Slot A TrackControl" not in panel.table.item(row_by_name_off["cBAS"], 8).text()
    panel.close()
    panel.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_daw_controller_protools_tabs_enriched_tree_marker_and_rollback(qapp, tmp_path, monkeypatch):
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=True))
    ptx_ref = PtxSession.from_bytes(ptxt.read_bytes())
    rf_id = ptx_ref.track_by_name("KIK").uuid_braced
    bf_id = ptx_ref.track_by_name("Sub").uuid_braced

    state = WorkspaceState()
    state.config = build_defaults()
    pt_cfg = state.config["config_presets"]["Default"]["daw_processors"]["protools"]
    pt_cfg["protools_enabled"] = True
    pt_cfg["protools_templates"] = [{"group": "Templates", "name": "Mix"}]
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    monkeypatch.setattr(ProToolsDawProcessor, "resolve_template_ptxt_path", lambda _self: ptxt)

    sr = 48000
    wav = tmp_path / "Kick.wav"
    sf.write(wav, _sustained_tone(sr, 0.25, 0.1), sr, subtype="PCM_24")
    tc = TrackContext("Kick.wav", str(wav), None, sr, 1, sr // 4, "24", "PCM_24", 0.25)
    tc.processor_results["bimodal_normalize"] = ProcessorResult("bimodal_normalize", 6.0, "Transient", "Peak", {"fader_offset": -6.0})
    state.session = SessionContext(
        tracks=[tc], output_tracks=[tc], config={"_use_processed": False}, project_name="Song",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )

    ctrl = DawTransferController(
        state, _make_shell(qapp),
        on_capture_session_state=lambda: {},
        on_enqueue_batch_item=lambda _i: True,
        on_is_prepare_running=lambda: False,
    )
    ctrl.configure_daw_processors()
    ctrl.populate_daw_combo()
    pt = ctrl.signal_flow_ctrl.active_protools_processor()
    assert pt is not None
    state.session.daw_state[pt.id] = {
        "folders": [
            {"id": rf_id, "name": "KIK", "folder_type": "routing", "parent_id": None, "index": 0},
            {"id": bf_id, "name": "Sub", "folder_type": "basic", "parent_id": rf_id, "index": 1},
        ],
        "assignments": {"Kick.wav": rf_id},
    }
    ctrl.page.resize(1200, 700)
    ctrl.page.right_stack.setCurrentIndex(1)
    ctrl.page.show()
    ctrl.sync_protools_ui_mode()
    qapp.processEvents()
    assert ctrl.page.right_tabs.documentMode() is True
    assert ctrl.page.right_tabs.tabBar().width() == ctrl.page.right_tabs.width()
    assert ctrl.page.right_tabs.isTabEnabled(1) is True
    assert ctrl.page.right_tabs.tabText(0) == "Routing && Folders"
    assert ctrl.page.signal_flow_panel.back_routing_btn.text() == "Back to Routing && Folders"
    ctrl.populate_folder_tree()

    # Pro Tools mode exposes 4-column enriched tree & tabs
    assert ctrl.page.folder_tree.columnCount() == 4
    kik_item = ctrl.page.folder_tree.topLevelItem(0)
    assert kik_item.text(1) == "Bus: KIK"
    assert "KIK-VCA" in kik_item.text(2)

    # Disabling auto_route & auto_group_folders updates enriched folder tree labels accurately
    pt_cfg["protools_auto_route_folders"] = False
    pt_cfg["protools_auto_group_folders"] = False
    ctrl.populate_folder_tree()
    kik_off = ctrl.page.folder_tree.topLevelItem(0)
    assert kik_off.text(2) == "Auto-grouping off"
    child_off = next(kik_off.child(i) for i in range(kik_off.childCount()) if kik_off.child(i).text(0) == "Kick.wav")
    assert child_off.text(3) == "Auto-route off"

    # Manual override sets MarkerTabBar blue dot on Signal Flow tab and counts both fader & Slot A Trim busses
    bar = ctrl.page.right_tabs.tabBar()
    assert isinstance(bar, MarkerTabBar)
    ctrl.signal_flow_ctrl._update_tab_chrome(
        {"recommended_bus_faders": {"KIK": -1.1}, "recommended_slot_a_trims": {"cKIK": 4.5}},
        {"KIK": -2.0},
    )
    assert bar.is_marked(1) is True
    assert "2 busses staged" in ctrl.page.right_tabs.tabText(1)
    # Left leaf-balance footer respects gain_mode and stage_bus_faders
    sf_sample = {
        "leaf_fader_shift_db": -3.5,
        "nodes": {
            "Kick": {"track_type": "Audio", "baked_gain_db": 0.0, "clip_gain_db": -8.0, "recommended_fader_db": 4.5, "post_fader_peak_db": -3.5},
        },
    }
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    pt_cfg["protools_stage_bus_faders"] = True
    ctrl._update_leaf_balance_footer(sf_sample)
    assert "headroom shift" in ctrl.page.setup_footer_label.text()
    pt_cfg["protools_stage_bus_faders"] = False
    ctrl._update_leaf_balance_footer(sf_sample)
    assert "headroom shift" not in ctrl.page.setup_footer_label.text()
    pt_cfg["protools_gain_mode"] = "clip_gain_unity"
    ctrl._update_leaf_balance_footer(sf_sample)
    assert "Unity-fader mode" in ctrl.page.setup_footer_label.text()
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    pt_cfg["protools_stage_bus_faders"] = True
    # Staging disabled hides staged count & marker dot in tab chrome
    ctrl.signal_flow_ctrl._update_tab_chrome(
        {"recommended_bus_faders": {"KIK": -1.1}, "recommended_slot_a_trims": {"cKIK": 4.5}},
        {"KIK": -2.0},
        staging_enabled=False,
    )
    assert bar.is_marked(1) is False
    assert ctrl.page.right_tabs.tabText(1) == "Signal Flow"

    # Rollback notice stays visible above tabs even when Signal Flow tab is active
    ctrl.page.right_tabs.setCurrentIndex(1)
    err_res = [DawCommandResult(DawCommand("reopen_session", "Song", {}), False, error="Reopen failed; rolled back")]
    ctrl._sync_transfer_signal_flow_and_rollback(
        SimpleNamespace(_processor=pt, _session=state.session, _origin_session=state.session), err_res,
    )
    assert not ctrl.page.rollback_banner.isHidden()

    # Switching to non-Pro-Tools hides tabs/summary/rollback, removes pane border & top margin, restores 1-col tree
    ctrl.page.set_protools_mode(False)
    assert ctrl.page.right_tabs.tabBar().isHidden()
    assert ctrl.page.right_tabs.isTabEnabled(1) is False
    assert ctrl.page.right_tabs.currentIndex() == 0
    assert ctrl.page.ptx_summary_bar.isHidden()
    assert ctrl.page.rollback_banner.isHidden()
    assert "border-top: none" in ctrl.page.right_tabs.styleSheet()
    assert ctrl.page._routing_tab_layout.contentsMargins().top() == 0
    assert ctrl.page.folder_tree.columnCount() == 1
    ctrl.page.close()
    ctrl.dispose()


def test_setup_table_clip_gain_baked_display_and_gain_modes(qapp, tmp_path):
    sr = 48000
    raw_wav = tmp_path / "Kick.wav"
    proc_wav = tmp_path / "Kick_proc.wav"
    sf.write(raw_wav, _sustained_tone(sr, 0.2, 0.1), sr, subtype="PCM_24")
    sf.write(proc_wav, _sustained_tone(sr, 0.2, 0.2), sr, subtype="PCM_24")

    tc = TrackContext("Kick.wav", str(raw_wav), None, sr, 1, int(sr * 0.2), "24", "PCM_24", 0.2)
    tc.processed_filepath = str(proc_wav)
    tc.processor_results["bimodal_normalize"] = ProcessorResult(
        "bimodal_normalize", 6.0, "Transient", "Peak", {"fader_offset": -6.0},
    )

    state = WorkspaceState()
    state.config = build_defaults()
    pt_cfg = state.config["config_presets"]["Default"]["daw_processors"]["protools"]
    pt_cfg["protools_enabled"] = True
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    state.active_daw_processor = ProToolsDawProcessor()
    state.session = SessionContext(
        tracks=[tc], output_tracks=[tc], config={"_use_processed": False}, project_name="Song",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )

    page = DawSetupPage()
    tbl_ctrl = TrackTableController(state, _make_shell(qapp), track_table=page.setup_table, setup_table=page.setup_table)

    # Raw WAV + clip_gain_and_fader -> +6.0 dB Clip Gain, -6.0 dB Fader
    tbl_ctrl.populate_setup_table()
    assert page.setup_table.item(0, 2).text() == "Kick.wav"
    assert page.setup_table.item(0, 4).text() == "+6.0 dB"
    assert page.setup_table.item(0, 5).text() == "-6.0 dB"

    # Stale prepared audio respects protools_gain_mode instead of crashing
    state.session.config["_use_processed"] = True
    state.session.prepare_state = "stale"
    pt_cfg["protools_gain_mode"] = "none"
    tbl_ctrl.populate_setup_table()
    assert page.setup_table.item(0, 4).text() == "0.0 dB"
    assert page.setup_table.item(0, 5).text() == "0.0 dB"

    # Prepared WAV (_use_processed=True, prepare_state="ready") -> 0.0 dB (baked) Clip Gain, -6.0 dB Fader
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    state.session.prepare_state = "ready"
    tbl_ctrl.populate_setup_table()
    assert page.setup_table.item(0, 2).text() == "Kick.wav"
    assert page.setup_table.item(0, 4).text() == "0.0 dB (baked)"
    assert page.setup_table.item(0, 5).text() == "-6.0 dB"

    # clip_gain_unity with raw WAV -> +6.0 dB Clip Gain, 0.0 dB Fader (no leading '+' on zero)
    state.session.config["_use_processed"] = False
    pt_cfg["protools_gain_mode"] = "clip_gain_unity"
    tbl_ctrl.populate_setup_table()
    assert page.setup_table.item(0, 4).text() == "+6.0 dB"
    assert page.setup_table.item(0, 5).text() == "0.0 dB"

    # Converged leaf_fader_shift_db updates assigned leaf Fader Gain cell + tooltip in clip_gain_and_fader mode
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    state.session.daw_state[state.active_daw_processor.id] = {
        "assignments": {"Kick.wav": "f1"},
        "signal_flow": {"leaf_fader_shift_db": -3.5},
    }
    tbl_ctrl.populate_setup_table()
    assert page.setup_table.item(0, 4).text() == "+6.0 dB"
    assert page.setup_table.item(0, 5).text() == "-9.5 dB"
    assert "headroom shift" in page.setup_table.item(0, 5).toolTip()


def test_signal_flow_controller_async_compute_cancel_overrides_and_stale(qapp, tmp_path, monkeypatch):
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=True))
    ptx_ref = PtxSession.from_bytes(ptxt.read_bytes())
    rf_id = ptx_ref.track_by_name("KIK").uuid_braced

    state = WorkspaceState()
    state.config = build_defaults()
    pt_cfg = state.config["config_presets"]["Default"]["daw_processors"]["protools"]
    pt_cfg["protools_enabled"] = True
    pt_cfg["protools_templates"] = [
        {"group": "Templates", "name": "Mix"},
        {"group": "Templates", "name": "AltMix"},
    ]
    pt_cfg["protools_gain_mode"] = "clip_gain_and_fader"
    pt_cfg["protools_stage_bus_faders"] = True
    monkeypatch.setattr(ProToolsDawProcessor, "resolve_template_ptxt_path", lambda _self: ptxt)

    sr = 48000
    wav1 = tmp_path / "Kick1.wav"
    wav2 = tmp_path / "Kick2.wav"
    sig = _sustained_tone(sr, 0.25, 0.35)
    sf.write(wav1, sig, sr, subtype="PCM_24")
    sf.write(wav2, sig, sr, subtype="PCM_24")
    tc1 = TrackContext("Kick1.wav", str(wav1), None, sr, 1, len(sig), "24", "PCM_24", 0.25)
    tc2 = TrackContext("Kick2.wav", str(wav2), None, sr, 1, len(sig), "24", "PCM_24", 0.25)
    for tc in (tc1, tc2):
        tc.processor_results["bimodal_normalize"] = ProcessorResult(
            "bimodal_normalize", 0.0, "Transient", "Peak", {"fader_offset": 0.0},
        )
    state.session = SessionContext(
        tracks=[tc1, tc2], output_tracks=[tc1, tc2], config={"_use_processed": False}, project_name="Song",
        transfer_manifest=[
            TransferEntry("Kick1.wav", "Kick1.wav", "Kick1"),
            TransferEntry("Kick2.wav", "Kick2.wav", "Kick2"),
        ],
    )

    ctrl = DawTransferController(
        state, _make_shell(qapp),
        on_capture_session_state=lambda: {},
        on_enqueue_batch_item=lambda _i: True,
        on_is_prepare_running=lambda: False,
    )
    ctrl.configure_daw_processors()
    ctrl.populate_daw_combo()
    pt = ctrl.signal_flow_ctrl.active_protools_processor()
    assert pt is not None
    state.session.daw_state[pt.id] = {
        "folders": [{"id": rf_id, "name": "KIK", "folder_type": "routing", "parent_id": None, "index": 0}],
        "assignments": {"Kick1.wav": rf_id, "Kick2.wav": rf_id},
    }
    ctrl.sync_protools_ui_mode()

    # Start preview and immediately cancel -> exits busy stack cleanly, does not stay stuck
    ctrl.signal_flow_ctrl.compute_preview()
    ctrl.signal_flow_ctrl.cancel_preview(wait=True, restore_ui=True, user_initiated=True)
    assert ctrl.page.signal_flow_panel.stack.currentIndex() != 1
    assert ctrl.page.signal_flow_panel.refresh_btn.isEnabled()

    # Full compute populates table and persists graph
    ctrl.signal_flow_ctrl.compute_preview()
    _pump_until_idle(qapp, ctrl)
    dp_state = state.session.daw_state[pt.id]
    assert "signal_flow" in dp_state
    assert ctrl.page.signal_flow_panel.table.rowCount() >= 2
    auto_kik = float(dp_state["signal_flow"]["nodes"]["KIK"]["auto_fader_db"])

    # Manual override uses debounced silent refresh (never flips stack to busy) and sets MarkerTabBar dot
    ctrl.signal_flow_ctrl.set_bus_override("KIK", -4.5)
    assert ctrl.page.signal_flow_panel.stack.currentIndex() == 3
    _pump_until_idle(qapp, ctrl)
    assert dp_state["bus_fader_overrides"]["KIK"] == pytest.approx(-4.5)
    bar = ctrl.page.right_tabs.tabBar()
    assert isinstance(bar, MarkerTabBar)
    assert bar.is_marked(1) is True

    # Returning override to auto_fader_db clears ghost override & unmarks tab dot
    ctrl.signal_flow_ctrl.set_bus_override("KIK", auto_kik)
    _pump_until_idle(qapp, ctrl)
    assert not dp_state.get("bus_fader_overrides")
    assert bar.is_marked(1) is False

    # Re-apply override, mark stale mid-refresh -> exits busy stack, re-enables Refresh, shows stale banner
    ctrl.signal_flow_ctrl.set_bus_override("KIK", -5.0)
    _pump_until_idle(qapp, ctrl)
    assert bar.is_marked(1) is True
    ctrl.signal_flow_ctrl.compute_preview()
    assert ctrl.page.signal_flow_panel.stack.currentIndex() == 1
    ctrl.signal_flow_ctrl.mark_stale()
    assert ctrl.page.signal_flow_panel.stack.currentIndex() == 3
    assert ctrl.page.signal_flow_panel.refresh_btn.isEnabled()
    assert not ctrl.page.signal_flow_panel.stale_banner.isHidden()

    # _snapshot_session_for_preview preserves session.tracks metadata when output_tracks are distinct objects
    from sessionprepgui.daw.signal_flow_controller import _snapshot_session_for_preview
    snap = _snapshot_session_for_preview(state.session, state.flat_config())
    assert len(snap.tracks) == len(state.session.tracks)
    assert snap.tracks[0] is not state.session.tracks[0]
    assert snap.tracks[0].processor_results.keys() == state.session.tracks[0].processor_results.keys()

    ctrl.signal_flow_ctrl.reset_all_overrides()
    _pump_until_idle(qapp, ctrl)
    assert not dp_state.get("bus_fader_overrides")
    assert bar.is_marked(1) is False

    # Switching to second template cancels any running preview & isolates daw_state
    ctrl.signal_flow_ctrl.compute_preview()
    ctrl.page.daw_combo.setCurrentIndex(1)
    _pump_until_idle(qapp, ctrl)
    alt_pt = ctrl.signal_flow_ctrl.active_protools_processor()
    assert alt_pt is not None and alt_pt.id != pt.id
    assert "signal_flow" not in state.session.daw_state.get(alt_pt.id, {})
    ctrl.dispose()
