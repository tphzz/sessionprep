from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from sessionprepgui import settings
from sessionprepgui.analysis.worker import (
    AnalyzeWorker, Phase1AnalyzeWorker, PrepareWorker, TopologyApplyWorker,
)
from sessionprepgui.mainwindow import SessionPrepWindow
from sessionprepgui.session.io import deserialize_session_state, serialize_session_state
from sessionpreplib.models import TransferEntry


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_tone(path):
    samples = np.arange(4800) / 48000
    sf.write(path, 0.08 * np.sin(2 * np.pi * 1000 * samples), 48000, subtype="PCM_24")


@pytest.fixture
def project(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path / "settings"))
    config = settings.build_defaults()
    default = config["config_presets"]["Default"]
    default["daw_processors"]["protools"].update(
        protools_project_dir="Default Projects",
        protools_temp_dir="Default Scratch",
        protools_templates=[{"group": "Mix", "name": "Stereo"}],
    )
    studio = settings.resolve_config_preset(config, "Default")
    studio["daw_processors"]["protools"].update(
        protools_project_dir="Studio Projects",
        protools_temp_dir="Studio Scratch",
    )
    config["config_presets"]["Studio"] = studio
    settings.save_config(config)
    source = tmp_path / "audio"
    source.mkdir()
    _write_tone(source / "Tone.wav")

    window = SessionPrepWindow()
    monkeypatch.setattr(window, "_start_peak_build", lambda *_args: None)
    window._restore_session_state({
        "source_dir": str(source),
        "active_config_preset": "Studio",
        "session_config_overrides": {},
        "session_groups": config["group_presets"]["Default"],
        "tracks": [],
        "prepare_state": "ready",
        "transfer_manifest": [TransferEntry("track-1", "Tone.wav", "Tone")],
    })
    try:
        yield window
    finally:
        window.close()


def _set_project_override(window, path):
    fields = dict(window._session_widgets["daw_processors.protools"])
    fields["protools_project_dir"].set_value(path)


def _queued_state(window):
    return deserialize_session_state(serialize_session_state(window._capture_session_state()))


def test_track_layout_rerun_uses_phase1_override_and_preserves_session_preset(
    project, monkeypatch,
):
    field = dict(project._session_widgets["detectors.dual_mono"])["dual_mono_eps"]
    field.setValue(0.005)
    _set_project_override(project, "Session Projects")
    read_at_analysis = []

    def run(worker):
        read_at_analysis.append(worker.config["dual_mono_eps"])
        worker.run()

    monkeypatch.setattr(Phase1AnalyzeWorker, "start", run)
    project._on_topo_reanalyze()

    assert read_at_analysis == [0.005]
    assert project._config_preset_combo.currentText() == "Studio"
    assert project._active_daw_processor.project_dir == "Session Projects"
    assert project._flat_config()["protools_temp_dir"] == "Studio Scratch"


def test_opening_another_audio_folder_uses_app_preset_and_drops_previous_overrides(
    project, tmp_path, monkeypatch,
):
    _set_project_override(project, "Session Projects")
    other = tmp_path / "another-project"
    other.mkdir()
    _write_tone(other / "New.wav")
    for worker in (Phase1AnalyzeWorker, TopologyApplyWorker, AnalyzeWorker, PrepareWorker):
        monkeypatch.setattr(worker, "start", lambda instance: instance.run())

    project._load_directory(str(other))
    project._on_topo_apply()
    project._on_prepare()

    assert project._session.prepare_state == "ready"
    assert project._phase_tabs.currentIndex() == 2
    assert project._config_preset_combo.currentText() == "Default"
    assert project._active_daw_processor.project_dir == "Default Projects"
    assert project._active_daw_processor._temp_dir == "Default Scratch"
    assert project._capture_session_state()["session_config_overrides"] == {}


def test_queued_transfer_uses_captured_defaults_and_interactive_reopen_resolves_latest(
    project,
):
    saved = _queued_state(project)
    studio = project._config["config_presets"]["Studio"]["daw_processors"]["protools"]
    studio.update(protools_project_dir="Latest Projects", protools_temp_dir="Latest Scratch")

    queued = project._batch_manager._rehydrate_session(saved)
    queued_processor = project._batch_manager._get_daw_processor("protools_0", queued.config)
    project._restore_session_state(saved)

    assert queued_processor.project_dir == "Studio Projects"
    assert queued_processor._temp_dir == "Studio Scratch"
    assert project._active_daw_processor.project_dir == "Latest Projects"
    assert project._active_daw_processor._temp_dir == "Latest Scratch"
    assert project._session.prepare_state == "ready"


def test_opening_prepared_session_reflects_changed_analysis_defaults_and_marks_audio_stale(
    project,
):
    saved = _queued_state(project)
    project._config["config_presets"]["Studio"]["processors"]["bimodal_normalize"][
        "target_rms"
    ] = -21.0

    project._restore_session_state(saved)

    assert project._flat_config()["target_rms"] == -21.0
    assert project._session.prepare_state == "stale"
    assert project._session.transfer_manifest


def test_reopening_with_missing_preset_uses_default_and_preserves_explicit_values(project):
    _set_project_override(project, "Session Projects")
    saved = _queued_state(project)
    del project._config["config_presets"]["Studio"]

    project._restore_session_state(saved)

    assert project._config_preset_combo.currentText() == "Default"
    assert project._active_daw_processor.project_dir == "Session Projects"
    assert project._active_daw_processor._temp_dir == "Default Scratch"


def test_reopening_remaps_fetched_state_when_inherited_template_moves(project):
    project._session.daw_state["protools_0"] = {"assignments": {"track-1": "folder-1"}}
    saved = _queued_state(project)
    templates = project._config["config_presets"]["Studio"]["daw_processors"]["protools"][
        "protools_templates"
    ]
    templates.insert(0, {"group": "Mix", "name": "Alternative"})

    project._restore_session_state(saved)

    assert project._active_daw_processor.id == "protools_1"
    assert project._active_daw_processor.name.endswith(" / Stereo")
    assert project._session.daw_state == {
        "protools_1": {"assignments": {"track-1": "folder-1"}},
    }
    assert project._transfer_action.isEnabled()
