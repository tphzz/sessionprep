"""Verify Phase 2 classification and anchor edits against original and corrected samples."""
from time import monotonic, sleep

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

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
    config = window._flat_config()
    session = SessionContext(tracks, config)
    pipeline = Pipeline(default_detectors(), default_processors(), config, max_workers=1)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)
    window._restore_session_state({"source_dir": str(source), "tracks": tracks,
                                   "active_config_preset": "Default", "session_config_overrides": {}})
    try:
        yield window
    finally:
        dispose_window(window)


def wait_for_edits(window):
    deadline = monotonic() + 8
    while window._repair_plan_worker is not None or window._repair_plan_pending:
        QApplication.processEvents()
        assert monotonic() < deadline, "The classification/anchor edit did not finish"
        sleep(0.005)


def choose(window, source, column, value, *, batch=False):
    combo = window._track_table.cellWidget(window._find_table_row(source.filename), column)
    assert combo.findText(value) >= 0
    # These are the flags that the table combobox sets on an Alt+Shift press.
    combo.batch_mode = batch
    combo.setProperty("_batch_mode", batch)
    combo.setCurrentText(value)
    combo.textActivated.emit(value)
    wait_for_edits(window)


def levels(samples, rate, anchor):
    peak = 20 * np.log10(np.abs(samples).max())
    width = int(0.4 * rate)
    sums = np.r_[0.0, np.cumsum(samples ** 2)]
    means = (sums[width:] - sums[:-width]) / width
    # Every window of these continuously sounding fixtures passes the 40 dB gate.
    assert means.min() > means.max() * 1e-4
    selected = means.max() if anchor == "max" else np.percentile(means, float(anchor[1:]))
    return peak, 10 * np.log10(selected)


def assert_plan(track, anchor, classification):
    samples, rate = sf.read(track.filepath, dtype="float64")
    original_peak, original_rms = levels(samples, rate, anchor)
    original = track.detector_results["audio_classifier"].data
    assert original["rms_anchor_label"] == anchor
    assert original["peak_db"] == pytest.approx(original_peak, abs=1e-6)
    assert original["rms_anchor_db"] == pytest.approx(original_rms, abs=1e-6)

    corrected_peak, corrected_rms = levels(render_corrected(samples, rate, track.repair_recipe), rate, anchor)
    if track.repair_recipe.active:
        assert original_peak - corrected_peak > 5 and original_rms - corrected_rms > 5
    gain = track.processor_results["bimodal_normalize"]
    assert gain.classification == classification
    assert gain.data["rms_anchor_label"] == anchor
    assert gain.data["detected_peak_db"] == pytest.approx(corrected_peak, abs=1e-6)
    assert gain.data["detected_rms_db"] == pytest.approx(corrected_rms, abs=1e-6)
    expected = -6 - corrected_peak
    if classification == "Sustained":
        expected = min(expected, -18 - corrected_rms)
    assert gain.gain_db == pytest.approx(expected, abs=1e-6)


def test_individual_edits_rehydrate_source_and_keep_original_anchor(project, tmp_path):
    window = project
    saved = tmp_path / "classifications.spsession"
    save_session(str(saved), window._capture_session_state())
    restored = load_session(str(saved))
    assert all(track.audio_data is None for track in restored["tracks"])
    window._restore_session_state(restored)
    first, other, _ = window._session.tracks
    assert first.repair_recipe.dc_remove and first.classification_override is None
    assert first.audio_data is None
    row = window._find_table_row(first.filename)
    displayed = window._track_table.cellWidget(row, 3).currentText()
    choose(window, first, 3, displayed)
    assert first.classification_override == displayed
    assert_plan(first, "p95", displayed)
    choose(window, first, 3, "Sustained")
    choose(window, first, 5, "Max")
    assert first.rms_anchor_override == "max" and other.rms_anchor_override is None
    assert_plan(first, "max", "Sustained")
    max_gain = first.processor_results["bimodal_normalize"].gain_db
    choose(window, first, 5, "P85")
    assert_plan(first, "p85", "Sustained")
    assert first.processor_results["bimodal_normalize"].gain_db - max_gain > 3


def test_batch_edits_replan_each_selected_row_and_preserve_selection(project):
    window = project
    first, second, plain = window._session.tracks
    assert first.repair_recipe.dc_remove and second.repair_recipe.dc_remove and not plain.repair_recipe.active
    window._track_table.restore_selection({first.filename, second.filename})
    choose(window, first, 5, "P85", batch=True)
    assert window._track_table.batch_selected_keys() == {first.filename, second.filename}
    assert (first.rms_anchor_override, second.rms_anchor_override, plain.rms_anchor_override) == ("p85", "p85", None)
    for track in (first, second):
        assert_plan(track, "p85", track.processor_results["bimodal_normalize"].classification)
    window._track_table.restore_selection({first.filename, plain.filename})
    choose(window, first, 3, "Transient", batch=True)
    assert window._track_table.batch_selected_keys() == {first.filename, plain.filename}
    assert first.classification_override == plain.classification_override == "Transient"
    assert second.classification_override is None
    assert_plan(first, "p85", "Transient")
    assert_plan(plain, "p95", "Transient")
    choose(window, plain, 5, "Max")
    assert plain.rms_anchor_override == "max" and first.rms_anchor_override == "p85"
    assert_plan(plain, "max", "Transient")


def test_single_plain_classification_and_bypass_update_linked_gain_and_faders(project):
    window = project
    repaired, _, plain = window._session.tracks
    window._session_groups = [{"name": "Pair", "gain_linked": True}]
    repaired.group = plain.group = "Pair"
    window._apply_linked_group_levels()
    selected = "bimodal_normalize"
    shared = repaired.processor_results[selected].gain_db
    assert shared == plain.processor_results[selected].gain_db

    choose(window, plain, 3, "Transient")
    result = plain.processor_results[selected]
    individual = result.data["gain_db_individual"]
    assert individual > shared + 5
    assert result.gain_db == pytest.approx(shared)
    assert result.data["fader_offset"] == pytest.approx(-shared)

    button = window._track_table.cellWidget(window._find_table_row(repaired.filename), 7)
    toggle = next(action for action in button.menu().actions() if action.data() == selected)
    toggle.setChecked(False)
    wait_for_edits(window)
    assert selected in repaired.processor_skip
    assert repaired.processor_results[selected].data["fader_offset"] == 0
    assert plain.processor_results[selected].gain_db == pytest.approx(individual)
    assert plain.processor_results[selected].data["fader_offset"] == pytest.approx(-individual)
