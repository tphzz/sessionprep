"""Verify Phase 2 classification and anchor edits against original and corrected samples."""
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox

from qt_helpers import dispose_window
from sessionprepgui import settings
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.session.io import load_session, save_session
from sessionpreplib.audio import load_track
from sessionpreplib.daw_processors import protools
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext
from sessionpreplib.pipeline import Pipeline
from sessionpreplib.processors import default_processors
from sessionpreplib.repairs import render_corrected


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(app, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    monkeypatch.setattr(protools, "check_ptsl_connectivity", lambda *args, **kwargs: (False, "offline"))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    settings.save_config(settings.build_defaults())
    window = SessionPrepWindow()
    source = tmp_path / "audio"
    source.mkdir()
    rate = 8000
    t = np.arange(8 * rate) / rate
    # A brief louder passage distinguishes the percentile and maximum anchors.
    envelope = np.where((t >= 4) & (t < 4.55), 0.22, 0.075)
    tracks = []
    for name, scale, bias in (("A.wav", 1.0, 0.38), ("B.wav", 0.8, 0.34), ("C.wav", 0.65, 0.0)):
        path = source / name
        sf.write(path, bias * np.minimum(t, 1.0) + scale * envelope * np.sin(2 * np.pi * 100 * t),
                 rate, subtype="PCM_16")
        track = load_track(str(path))
        if bias:
            track.repair_overrides["dc_remove"] = True
        tracks.append(track)
    config = window.state.flat_config()
    session = SessionContext(tracks, config)
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    window.persistence_controller.restore_session_state({"source_dir": str(source), "tracks": tracks,
                                   "active_config_preset": "Default", "session_config_overrides": {}})
    try:
        yield window
    finally:
        dispose_window(window)


def wait_for_edits(window):
    deadline = monotonic() + 8
    while window.analysis_controller.repair_plan_worker is not None or window.analysis_controller.repair_plan_pending:
        QApplication.processEvents()
        assert monotonic() < deadline, "The classification/anchor edit did not finish"
        sleep(0.005)


def choose(window, source, column, value, *, batch=False):
    combo = window.phase2_page.track_table.cellWidget(window.track_table_controller.find_table_row(source.filename), column)
    idx = combo.findText(value)
    if idx < 0 and value == "Auto":
        idx = 0
    assert idx >= 0
    # These are the flags that the table combobox sets on an Alt+Shift press.
    combo.batch_mode = batch
    combo.setProperty("_batch_mode", batch)
    combo.setCurrentIndex(idx)
    if column == 3:
        combo.activated.emit(idx)
    else:
        combo.textActivated.emit(combo.itemText(idx))
    wait_for_edits(window)


def levels(samples, rate, anchor):
    peak = 20 * np.log10(np.abs(samples).max())
    width = max(1, round(0.4 * rate))
    sums = np.r_[0.0, np.cumsum(samples ** 2)]
    means = (sums[width:] - sums[:-width]) / width
    # Every window of these continuously sounding fixtures passes the 40 dB gate.
    assert means.min() > means.max() * 1e-4
    selected = means.max() if anchor == "max" else np.percentile(means, float(anchor[1:]))
    return peak, 10 * np.log10(selected)


def assert_plan(track, anchor, mode):
    samples, rate = sf.read(track.filepath, dtype="float64")
    original_peak, original_rms = levels(samples, rate, anchor)
    corrected_peak, corrected_rms = levels(render_corrected(samples, rate, track.repair_recipe), rate, anchor)
    if track.repair_recipe.active:
        assert original_peak - corrected_peak > 5 and original_rms - corrected_rms > 5
    gain = track.processor_results["bimodal_normalize"]
    assert gain.data["mode"] == mode
    assert gain.data["rms_anchor_label"] == anchor
    assert gain.data["detected_peak_db"] == pytest.approx(corrected_peak, abs=1e-6)
    assert gain.data["detected_rms_db"] == pytest.approx(corrected_rms, abs=1e-6)
    if mode == "Peak":
        expected = -6 - corrected_peak
    elif mode == "RMS":
        expected = -18 - corrected_rms
    else:
        expected = min(-6 - corrected_peak, -18 - corrected_rms)
    assert gain.gain_db == pytest.approx(expected, abs=1e-6)


def test_individual_edits_rehydrate_source_and_keep_original_anchor(project, tmp_path):
    window = project
    saved = tmp_path / "classifications.spsession"
    save_session(str(saved), window.persistence_controller.capture_session_state())
    restored = load_session(str(saved))
    assert all(track.audio_data is None for track in restored["tracks"])
    window.persistence_controller.restore_session_state(restored)
    first, other, _ = window.state.session.tracks
    assert first.repair_recipe.dc_remove and first.classification_override is None
    assert first.audio_data is None
    choose(window, first, 3, "Peak")
    assert first.classification_override == "Peak"
    assert_plan(first, "p95", "Peak")
    choose(window, first, 3, "Auto")
    assert first.classification_override is None
    choose(window, first, 5, "Max")
    assert first.rms_anchor_override == "max" and other.rms_anchor_override is None
    assert_plan(first, "max", "Auto")
    max_gain = first.processor_results["bimodal_normalize"].gain_db
    choose(window, first, 5, "P85")
    assert_plan(first, "p85", "Auto")
    assert first.processor_results["bimodal_normalize"].gain_db - max_gain > 3


def test_batch_edits_replan_each_selected_row_and_preserve_selection(project):
    window = project
    first, second, plain = window.state.session.tracks
    assert first.repair_recipe.dc_remove and second.repair_recipe.dc_remove and not plain.repair_recipe.active
    window.phase2_page.track_table.restore_selection({first.filename, second.filename})
    choose(window, first, 5, "P85", batch=True)
    assert window.phase2_page.track_table.batch_selected_keys() == {first.filename, second.filename}
    assert (first.rms_anchor_override, second.rms_anchor_override, plain.rms_anchor_override) == ("p85", "p85", None)
    for track in (first, second):
        assert_plan(track, "p85", "Auto")
    window.phase2_page.track_table.restore_selection({first.filename, plain.filename})
    choose(window, first, 3, "Peak", batch=True)
    assert window.phase2_page.track_table.batch_selected_keys() == {first.filename, plain.filename}
    assert first.classification_override == plain.classification_override == "Peak"
    assert second.classification_override is None
    assert_plan(first, "p85", "Peak")
    assert_plan(plain, "p95", "Peak")
    choose(window, plain, 5, "Max")
    assert plain.rms_anchor_override == "max" and first.rms_anchor_override == "p85"
    assert_plan(plain, "max", "Peak")


def test_single_plain_classification_and_bypass_update_linked_gain_and_faders(project):
    window = project
    repaired, _, plain = window.state.session.tracks
    window.state.session_groups = [{"name": "Pair", "gain_linked": True}]
    repaired.group = plain.group = "Pair"
    window.groups_controller.apply_linked_group_levels()
    selected = "bimodal_normalize"
    shared = repaired.processor_results[selected].gain_db
    assert shared == plain.processor_results[selected].gain_db

    choose(window, plain, 3, "Peak")
    result = plain.processor_results[selected]
    individual = result.data["gain_db_individual"]
    assert individual > shared + 5
    assert result.gain_db == pytest.approx(shared)
    assert result.data["fader_offset"] == pytest.approx(-shared)

    button = window.phase2_page.track_table.cellWidget(window.track_table_controller.find_table_row(repaired.filename), 7)
    toggle = next(action for action in button.menu().actions() if action.data() == selected)
    toggle.setChecked(False)
    wait_for_edits(window)
    assert selected in repaired.processor_skip
    assert repaired.processor_results[selected].data["fader_offset"] == 0
    assert plain.processor_results[selected].gain_db == pytest.approx(individual)
    assert plain.processor_results[selected].data["fader_offset"] == pytest.approx(-individual)


def test_group_link_recomputes_named_anchor_after_edits_and_reopen(project, tmp_path):
    window = project
    session = window.state.session
    anchor, peer, _ = session.tracks
    window.state.session_overrides.update({"analysis.anchor": anchor.filename, "analysis.fader_headroom_db": 0})
    session.config.update(window.state.flat_config())
    session.config["_gain_linked_groups"] = set()
    for track, gain in zip(session.tracks, (6.0, -2.0, 1.0)):
        track.manual_gain_db = gain
    pipeline = Pipeline(default_detectors(), default_processors(), session.config, max_workers=1)
    pipeline.plan(session)
    window.state.session_groups = [{"name": "Pair", "gain_linked": False}]
    window.groups_controller.populate_groups_tab()
    window.groups_controller.refresh_group_combos()
    for track in (anchor, peer):
        choose(window, track, 6, "Pair")

    def check(session, gains, faders):
        results = [track.processor_results["bimodal_normalize"] for track in session.tracks]
        assert [r.gain_db for r in results] == pytest.approx(gains)
        assert [r.data["fader_offset"] for r in results] == pytest.approx(faders)

    check(session, [6, -2, 1], [0, 8, 5])
    link = window.groups_controller.page.table.cellWidget(0, 2).findChild(QCheckBox)
    link.click()
    check(session, [-2, -2, 1], [0, 0, -3])
    session.config["_gain_linked_groups"] = {"Pair"}
    pipeline.plan(session)
    check(session, [-2, -2, 1], [0, 0, -3])
    link.click()
    check(session, [6, -2, 1], [0, 8, 5])
    link.click()

    saved = tmp_path / "linked.spsession"
    save_session(str(saved), window.persistence_controller.capture_session_state())
    window.persistence_controller.restore_session_state(load_session(str(saved)))
    check(window.state.session, [-2, -2, 1], [0, 0, -3])
    restored_anchor = window.state.session.tracks[0]
    choose(window, restored_anchor, 3, "Peak")
    assert restored_anchor.classification_override == "Peak"
    check(window.state.session, [-2, -2, 1], [0, 0, -3])
    button = window.phase2_page.track_table.cellWidget(window.track_table_controller.find_table_row(restored_anchor.filename), 7)
    toggle = next(action for action in button.menu().actions() if action.data() == "bimodal_normalize")
    toggle.setChecked(False)
    wait_for_edits(window)
    assert [t.processor_results["bimodal_normalize"].data["fader_offset"]
            for t in window.state.session.tracks] == pytest.approx([0, 2, -1])
    toggle.setChecked(True)
    wait_for_edits(window)
    check(window.state.session, [-2, -2, 1], [0, 0, -3])
    link = window.groups_controller.page.table.cellWidget(0, 2).findChild(QCheckBox)
    link.click()
    check(window.state.session, [6, -2, 1], [0, 8, 5])


def test_reopened_group_edit_uses_the_selected_template_fader_ceiling(project, tmp_path):
    window = project
    window.state.config["config_presets"]["Default"]["daw_processors"]["dawproject"].update(
        dawproject_templates=[{"name": "Studio", "template_path": str(tmp_path / "studio.dawproject"),
                               "fader_ceiling_db": 6.0}],
    )
    window.daw_controller.configure_daw_processors()
    daw = next(dp for dp in window.state.daw_processors if dp.fader_ceiling_db == 6.0)
    window.daw_controller.populate_daw_combo(daw.id)
    saved = tmp_path / "headroom.spsession"
    save_session(str(saved), window.persistence_controller.capture_session_state())
    window.persistence_controller.restore_session_state(load_session(str(saved)))
    assert window.state.active_daw_processor.id == daw.id
    # With 8 dB of headroom on a +6 dB console, the highest fader is -2 dB.
    for track, gain in zip(window.state.session.tracks, (-10.0, 0.0, 1.0)):
        track.processor_results["bimodal_normalize"].data["gain_db_individual"] = gain
    window.groups_controller.apply_linked_group_levels()
    assert [t.processor_results["bimodal_normalize"].data["fader_offset"]
            for t in window.state.session.tracks] == pytest.approx([-2.0, -12.0, -13.0])
