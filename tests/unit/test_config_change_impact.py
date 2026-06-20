from __future__ import annotations

import copy

from sessionpreplib.config import (
    build_structured_defaults,
    classify_structured_config_change,
)


def _changed(path: tuple[str, ...], value):
    old = build_structured_defaults()
    new = copy.deepcopy(old)
    target = new
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return classify_structured_config_change(old, new)


def test_dual_mono_epsilon_requires_phase1_reanalysis():
    impact = _changed(("detectors", "dual_mono", "dual_mono_eps"), 0.001)

    assert impact.requires_phase1
    assert not impact.requires_phase2
    assert "detectors.dual_mono.dual_mono_eps" in impact.phase1_keys


def test_one_sided_silence_threshold_requires_phase1_reanalysis():
    impact = _changed(
        ("detectors", "one_sided_silence", "one_sided_silence_db"), -70.0)

    assert impact.requires_phase1
    assert "detectors.one_sided_silence.one_sided_silence_db" in (
        impact.phase1_keys)


def test_phase2_detector_parameter_requires_phase2_reanalysis():
    impact = _changed(("detectors", "clipping", "clip_consecutive"), 9)

    assert not impact.requires_phase1
    assert impact.requires_phase2
    assert "detectors.clipping.clip_consecutive" in impact.phase2_keys


def test_processor_parameter_requires_phase2_reanalysis():
    impact = _changed(("processors", "bimodal_normalize", "target_rms"), -20.0)

    assert impact.requires_phase2
    assert "processors.bimodal_normalize.target_rms" in impact.phase2_keys


def test_presentation_only_changes_do_not_require_reanalysis():
    old = build_structured_defaults()
    old.setdefault("presentation", {"show_clean_detectors": False})
    new = copy.deepcopy(old)
    new["presentation"]["show_clean_detectors"] = True
    new["detectors"]["dual_mono"]["dual_mono_report_as"] = "skip"

    impact = classify_structured_config_change(old, new)

    assert impact.presentation_only
    assert not impact.requires_phase1
    assert not impact.requires_phase2
    assert impact.presentation_keys == frozenset({
        "presentation.show_clean_detectors",
        "detectors.dual_mono.dual_mono_report_as",
    })


def test_daw_processor_changes_are_daw_only():
    old = build_structured_defaults()
    new = copy.deepcopy(old)
    current = bool(new["daw_processors"]["protools"]["protools_enabled"])
    new["daw_processors"]["protools"]["protools_enabled"] = not current
    impact = classify_structured_config_change(old, new)

    assert impact.daw_only
    assert not impact.requires_phase1
    assert not impact.requires_phase2
