"""GUI, Signal Flow panel, v7->v8 migration, and CLI tests for Peak-Capped RMS."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtCore import QLocale, QUrl
from PySide6.QtWidgets import QApplication

from sessionprepgui.daw.signal_flow_graph_view import _role_badge
from sessionprepgui.daw.signal_flow_panel import BusFaderSpinBox, SignalFlowPanel
from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.session.io import CURRENT_VERSION, deserialize_session_state
from sessionprepgui.tracks.phase2_page import Phase2Page
from sessionprepgui.tracks.table_controller import _mode_display_info
from sessionprepgui.widgets import TableCellDoubleSpinBox
from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mk_track(name: str = "Kick.wav", override: str | None = None) -> TrackContext:
    t = TrackContext(name, "", None, 48000, 1, 96000, "24-bit", "PCM_24", 2.0)
    t.classification_override = override
    return t


def test_phase2_mode_column_header_and_locale(qapp):
    page = Phase2Page({})
    assert page.track_table.horizontalHeaderItem(3).text() == "Mode"
    spin = TableCellDoubleSpinBox()
    assert spin.locale().decimalPoint() == QLocale.c().decimalPoint() == "."
    bus_spin = BusFaderSpinBox()
    assert bus_spin.locale().decimalPoint() == "."
    page.deleteLater()
    spin.deleteLater()
    bus_spin.deleteLater()


def test_mode_display_info_labels_and_group_tooltips():
    tr = _mk_track("KickIn.wav")
    pr_auto_pk = ProcessorResult("bimodal_normalize", -2.0, "Auto (Peak)", "", {"governing": "Peak"})
    lbl, ov, _col, tip = _mode_display_info(tr, pr_auto_pk)
    assert lbl == "Auto (Peak)" and ov is None and tip == ""

    pr_grp = ProcessorResult(
        "bimodal_normalize", -3.0, "Auto (Peak)", "",
        {"governing": "RMS", "group_limited": True, "group_governing_track": "KickOut.wav", "group_governing_constraint": "Peak"},
    )
    lbl, ov, _col, tip = _mode_display_info(tr, pr_grp)
    assert lbl == "Auto (Grp)" and ov is None
    assert "KickOut.wav" in tip and "peak" in tip

    tr.classification_override = "RMS"
    lbl, ov, _col, tip = _mode_display_info(tr, pr_grp)
    assert lbl == "RMS (Grp)" and ov == "RMS"
    assert "KickOut.wav" in tip and "Manual override: RMS" in tip

    tr.classification_override = None
    pr_nf = ProcessorResult("bimodal_normalize", 0.0, "Noise Floor", "", {"governing": "Noise Floor"})
    lbl, ov, _col, tip = _mode_display_info(tr, pr_nf)
    assert lbl == "Noise Floor" and ov is None and "-70.0 dBFS" in tip


def test_report_browser_peak_link_emits_peak_jump_requested(qapp):
    proc = BimodalNormalizeProcessor()
    proc.configure({})
    tr = _mk_track("Vocal.wav")
    tr.processor_results[proc.id] = ProcessorResult(
        proc.id, -2.4, "Auto (Peak)", "Peak (ceiling) → -6.0 dBFS",
        {
            "mode": "Auto", "governing": "Peak",
            "detected_peak_db": -3.6, "detected_rms_db": -22.0,
            "target_peak": -6.0, "target_rms": -18.0,
            "gain_for_peak": -2.4, "gain_for_rms": +4.0,
            "post_peak_db": -6.0, "post_rms_db": -24.4,
            "governing_peak_sample": 68160, "governing_peak_time_s": 1.42,
            "next_peak_db": -9.8, "next_peak_exclusion_ms": 100.0,
        },
    )
    sess = SessionContext([tr], {}, processors=[proc])
    browser = ReportBrowser({})
    jumped: list[int] = []
    browser.peakJumpRequested.connect(jumped.append)
    browser.present_track(tr, sess)
    assert 'href="peak:68160"' in browser.toHtml()
    browser.activate_report_link(QUrl("peak:68160"))
    assert jumped == [68160]
    browser.deleteLater()


def test_signal_flow_panel_governs_column_estimates_and_exact_insert_count(qapp):
    panel = SignalFlowPanel()
    graph = {
        "target_peak_db": -6.0,
        "target_rms_db": -18.0,
        "pan_law_db": -3.0,
        "dbfs_convention": "standard",
        "topological_order": ["KIK", "cBAS", "Master"],
        "balance_validation": {
            "status": "verified",
            "summary": "File-level balance verified (±0.0 dB across 2 paths)",
            "unsimulated_insert_count": 2,
        },
        "nodes": {
            "KIK": {
                "track_name": "KIK", "track_type": "RoutingFolder",
                "classification": "Peak", "governing": None, "staged": False,
                "pre_fader_peak_db": -6.0, "pre_fader_rms_db": -20.0,
                "post_fader_peak_db": -6.0, "post_fader_rms_db": -20.0,
                "net_gain_delta_db": 0.0,
            },
            "cBAS": {
                "track_name": "cBAS", "track_type": "Aux",
                "classification": "RMS", "governing": "RMS", "staged": True,
                "staging_strategy": "trim_and_fader", "slot_a_trim_plugin": "TrackControl",
                "active_inserts": ["TrackControl", "1176", "Neve 1073"],
                "recommended_trim_db": -4.0, "recommended_fader_db": 4.0, "auto_fader_db": 4.0,
                "pre_insert_peak_db": -8.0, "pre_insert_rms_db": -14.0,
                "pre_fader_peak_db": -12.0, "pre_fader_rms_db": -18.0,
                "post_fader_peak_db": -8.0, "post_fader_rms_db": -14.0,
                "net_gain_delta_db": 0.0,
            },
            "Master": {
                "track_name": "Master", "track_type": "Master",
                "classification": "RMS", "governing": None, "staged": False,
                "pre_fader_peak_db": -8.0, "pre_fader_rms_db": -14.0,
                "post_fader_peak_db": -8.0, "post_fader_rms_db": -14.0,
                "net_gain_delta_db": 0.0,
            },
        },
    }
    panel.show_graph(graph, staging_enabled=True)
    assert panel.table.horizontalHeaderItem(1).text() == "Governs"
    assert panel.table.horizontalHeaderItem(3).text() == "Slot A Trim (est.)"
    assert "Initial linear estimates" in panel.target_label.text()
    assert "2 active inserts unsimulated" in panel._balance_lbl.text()
    # Non-staged Tier 1 RoutingFolder & Master show em-dash; staged cBAS shows RMS
    assert panel.table.item(panel._row_by_node["KIK"], 1).text() == "\u2014"
    assert panel.table.item(panel._row_by_node["cBAS"], 1).text() == "RMS"
    assert panel.table.item(panel._row_by_node["Master"], 1).text() == "\u2014"
    # Graph view badge shows RMS for staged bus and blank for non-staged folder
    assert _role_badge("bus", graph["nodes"]["cBAS"])[0] == "RMS"
    assert _role_badge("folder", graph["nodes"]["KIK"])[0] == ""
    panel.deleteLater()


def test_v7_to_v8_session_migration_overrides_config_and_stale(tmp_path):
    assert CURRENT_VERSION == 8
    wav = tmp_path / "Loop.wav"
    sf.write(wav, np.zeros(4800), 48000)
    raw = {
        "version": 7,
        "source_dir": str(tmp_path),
        "prepare_state": "ready",
        "session_config": {
            "detectors": {"audio_classifier": {"crest_threshold": 12.0}},
            "processors": {"bimodal_normalize": {"force_transient": ["kick"], "force_sustained": ["pad"]}},
        },
        "session_config_overrides": {
            "detectors.audio_classifier.crest_threshold": 15.0,
            "processors.bimodal_normalize.force_transient": ["snare"],
        },
        "tracks": {
            "Loop.wav": {
                "status": "OK", "channels": 1, "samplerate": 48000, "total_samples": 4800,
                "classification_override": None,
                "detector_results": {"audio_classifier": {"detector_id": "audio_classifier", "severity": "clean", "data": {}}},
                "processor_results": {"bimodal_normalize": {"processor_id": "bimodal_normalize", "gain_db": 4.0, "classification": "Transient", "method": "Peak", "data": {}}},
            },
            "Overridden.wav": {
                "status": "OK", "channels": 1, "samplerate": 48000, "total_samples": 4800,
                "classification_override": "Transient",
            },
            "Sustained.wav": {
                "status": "OK", "channels": 1, "samplerate": 48000, "total_samples": 4800,
                "classification_override": "Sustained",
            },
        },
    }
    state = deserialize_session_state(raw)
    assert state["migrated_from_v7"] is True
    assert state["prepare_state"] == "stale"
    assert "audio_classifier" not in state["session_config"].get("detectors", {})
    assert state["session_config_overrides"].get("processors.bimodal_normalize.force_peak") == "snare"
    by_name = {t.filename: t for t in state["tracks"]}
    assert "audio_classifier" not in by_name["Loop.wav"].detector_results
    loop_pr = by_name["Loop.wav"].processor_results["bimodal_normalize"]
    assert loop_pr.classification == "Auto (Peak)"
    assert loop_pr.data["governing"] == "Peak"
    assert loop_pr.data["rms_anchor_active"] is False
    assert by_name["Overridden.wav"].classification_override == "Peak"
    assert by_name["Sustained.wav"].classification_override is None


def test_processor_result_nonfinite_roundtrip_and_preset_migration(tmp_path):
    import json
    from sessionprepgui.session.io import _deser_processor_result, _ser_processor_result
    from sessionpreplib.config import load_preset

    orig = ProcessorResult(
        "bimodal_normalize", 0.0, "Noise Floor", "Noise Floor Hold",
        {"detected_peak_db": float("-inf"), "detected_rms_db": float("-inf"), "post_peak_db": float("-inf")},
    )
    payload = _ser_processor_result(orig)
    json.dumps(payload, allow_nan=False)
    restored = _deser_processor_result(payload)
    assert restored.data["detected_peak_db"] == float("-inf")
    assert restored.data["detected_rms_db"] == float("-inf")

    preset_file = tmp_path / "legacy_preset.json"
    preset_file.write_text(json.dumps({
        "schema_version": "1.0",
        "crest_threshold": 14.0,
        "force_transient": ["kick*", "snare$"],
        "force_sustained": ["pad"],
    }), encoding="utf-8")
    loaded = load_preset(str(preset_file))
    assert "crest_threshold" not in loaded and "force_sustained" not in loaded
    assert loaded.get("force_peak")


def test_gui_live_aes17_toggle_and_preferences_validation_and_mode_refresh(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from sessionprepgui.prefs.dialog import PreferencesDialog
    from sessionprepgui.settings import build_defaults
    from sessionprepgui.state import WorkspaceState
    from sessionprepgui.tracks.table_controller import TrackTableController
    from sessionprepgui.tracks.table_widgets import TrackCol
    from sessionprepgui.analysis.config_controller import SessionConfigController
    from sessionpreplib.audio import AES17_OFFSET

    # 1. PreferencesDialog._on_save blocks invalid preset (target_rms > target_peak)
    cfg = build_defaults()
    cfg["config_presets"]["Default"]["processors"]["bimodal_normalize"]["target_rms"] = -2.0
    cfg["config_presets"]["Default"]["processors"]["bimodal_normalize"]["target_peak"] = -6.0
    warnings_shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings_shown.append(str(args[2])))
    dlg = PreferencesDialog(cfg)
    dlg._on_save()
    assert not dlg._saved and warnings_shown
    assert "must not exceed" in warnings_shown[0]
    dlg.deleteLater()

    # 2. Live dbfs_convention toggle updates session.config and processor _db_offset
    page = Phase2Page(build_defaults())
    try:
        state = WorkspaceState(config=build_defaults())
        proc = BimodalNormalizeProcessor()
        proc.configure({"dbfs_convention": "standard"})
        tr = _mk_track("Lead.wav")
        tr.processor_results[proc.id] = ProcessorResult(
            proc.id, 0.0, "Auto (RMS)", "RMS",
            {"mode": "Auto", "governing": "RMS", "detected_peak_db": -12.0, "detected_rms_db": -18.0},
        )
        state.session = SessionContext([tr], {"dbfs_convention": "standard"}, processors=[proc])
        state.current_track = tr

        class _DummyShell:
            def show_status(self, _msg): ...
            def is_phase_enabled(self, _p): return True
            def dialog_parent(self): return page
            def progress_start(self, _m): ...
            def progress_value(self, _a, _b): ...
            def progress_finish(self, _m): ...
            def progress_fail(self, _m): ...

        ctrl = SessionConfigController(state, _DummyShell(), detail_tabs=page.detail_tabs)
        # Invalid force_peak regex is rejected and not written to session_overrides
        ctrl.on_session_config_value_changed("processors.bimodal_normalize.force_peak", "[unclosed")
        assert "processors.bimodal_normalize.force_peak" not in state.session_overrides
        # Valid dbfs_convention toggle updates live processor offset & session.config
        state.session_overrides["analysis.dbfs_convention"] = "aes17"
        ctrl.refresh_presentation()
        assert state.session.config["dbfs_convention"] == "aes17"
        assert proc._db_offset == pytest.approx(AES17_OFFSET, abs=1e-12)

        # 3. refresh_mode_column updates existing BatchComboBox in-place without destroying it
        tctrl = TrackTableController(state, _DummyShell(), track_table=page.track_table)
        page.track_table.setRowCount(1)
        from sessionprepgui.tracks.table_widgets import _SortableItem
        page.track_table.setItem(0, TrackCol.FILE, _SortableItem("Lead.wav", "lead.wav"))
        tctrl.populate_track_row(0, tr, [])
        combo_before = page.track_table.cellWidget(0, TrackCol.CLASSIFICATION)
        tr.processor_results[proc.id].data["group_limited"] = True
        tr.processor_results[proc.id].data["group_governing_track"] = "Other.wav"
        tctrl.refresh_mode_column()
        combo_after = page.track_table.cellWidget(0, TrackCol.CLASSIFICATION)
        assert combo_before is combo_after
        assert combo_after.currentText() == "Auto (Grp)"
    finally:
        page.deleteLater()
        qapp.processEvents()


def test_cli_force_peak_and_deprecated_flag_warnings(tmp_path):
    sr = 48000
    sig = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sr) / sr)
    sf.write(tmp_path / "Kick.wav", sig, sr, subtype="PCM_24")
    cli = Path(__file__).resolve().parents[2] / "sessionprep.py"
    res = subprocess.run(
        [sys.executable, str(cli), str(tmp_path), "--force_peak", "kick", "--max_gain_db", "18.0",
         "--crest_threshold", "12.0", "--force_sustained", "pad"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "--crest_threshold is deprecated" in res.stderr
    assert "--force_sustained is deprecated" in res.stderr
