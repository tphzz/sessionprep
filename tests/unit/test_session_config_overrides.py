from sessionpreplib.config import infer_session_overrides, resolve_session_config


def test_explicit_preset_values_and_empty_values_override_later_defaults():
    preset = {
        "analysis": {"window": 800},
        "daw_processors": {
            "protools": {
                "protools_project_dir": "New Projects",
                "protools_temp_dir": "New Scratch",
                "protools_templates": [{"group": "Default", "name": "Mix"}],
            },
        },
    }
    overrides = {
        "analysis.window": 400,
        "daw_processors.protools.protools_temp_dir": "",
        "daw_processors.protools.protools_templates": [],
    }

    resolved = resolve_session_config(preset, overrides)

    assert resolved["analysis"]["window"] == 400
    assert resolved["daw_processors"]["protools"] == {
        "protools_project_dir": "New Projects",
        "protools_temp_dir": "",
        "protools_templates": [],
    }
    resolved["daw_processors"]["protools"]["protools_templates"].append({})
    assert overrides["daw_processors.protools.protools_templates"] == []
    assert preset["daw_processors"]["protools"]["protools_templates"] == [
        {"group": "Default", "name": "Mix"},
    ]


def test_legacy_snapshot_preserves_differences_and_missing_fields_inherit():
    preset = {
        "presentation": {"show_clean_detectors": False},
        "detectors": {"clipping": {"clip_report_as": "problem"}},
        "daw_processors": {
            "protools": {"protools_project_dir": "Projects", "protools_temp_dir": "Scratch"},
        },
    }
    legacy = {
        "presentation": {"show_clean_detectors": False},
        "daw_processors": {
            "protools": {"protools_project_dir": "Local Projects"},
        },
    }

    overrides = infer_session_overrides(preset, legacy)

    assert overrides == {"daw_processors.protools.protools_project_dir": "Local Projects"}
    assert resolve_session_config(preset, overrides)["daw_processors"]["protools"] == {
        "protools_project_dir": "Local Projects",
        "protools_temp_dir": "Scratch",
    }


def test_falsy_edits_and_plugin_fields_keep_their_exact_values():
    preset = {
        "analysis": {"anchor": "Kick", "fader_headroom_db": 8.0},
        "presentation": {"show_clean_detectors": True},
        "detectors": {"plugin.detector": {"threshold": 20}},
    }
    effective = resolve_session_config(preset, {
        "analysis.anchor": None,
        "analysis.fader_headroom_db": 0.0,
        "presentation.show_clean_detectors": False,
        "detectors.plugin.detector.threshold": 5,
        "obsolete": "ignored",
    })

    assert effective == {
        "analysis": {"anchor": None, "fader_headroom_db": 0.0},
        "presentation": {"show_clean_detectors": False},
        "detectors": {"plugin.detector": {"threshold": 5}},
    }
