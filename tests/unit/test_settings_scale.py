from __future__ import annotations

import json

from sessionprepgui import settings


def test_default_scale_factor_is_readability_default():
    assert settings.DEFAULT_SCALE_FACTOR == 1.15
    assert settings.build_defaults()["app"]["scale_factor"] == 1.15
    assert (
        settings.build_defaults()["app"]["_scale_factor_default_version"]
        == settings.SCALE_FACTOR_DEFAULT_VERSION
    )


def test_load_config_migrates_historical_default_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path))
    path = tmp_path / settings.CONFIG_FILENAME
    path.write_text(
        json.dumps({"app": {"scale_factor": 1.0}}),
        encoding="utf-8",
    )

    config = settings.load_config()

    assert config["app"]["scale_factor"] == 1.15
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["app"]["scale_factor"] == 1.15
    assert (
        saved["app"]["_scale_factor_default_version"]
        == settings.SCALE_FACTOR_DEFAULT_VERSION
    )


def test_load_config_preserves_explicit_non_default_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path))
    path = tmp_path / settings.CONFIG_FILENAME
    path.write_text(
        json.dumps({"app": {"scale_factor": 1.25}}),
        encoding="utf-8",
    )

    config = settings.load_config()

    assert config["app"]["scale_factor"] == 1.25


def test_load_config_preserves_post_migration_explicit_one(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_app_dir", lambda: str(tmp_path))
    path = tmp_path / settings.CONFIG_FILENAME
    path.write_text(
        json.dumps({
            "app": {
                "scale_factor": 1.0,
                "_scale_factor_default_version": settings.SCALE_FACTOR_DEFAULT_VERSION,
            }
        }),
        encoding="utf-8",
    )

    config = settings.load_config()

    assert config["app"]["scale_factor"] == 1.0


def test_startup_scale_factor_from_raw_config_migrates_historical_default():
    assert settings.startup_scale_factor_from_raw_config({}) == 1.15
    assert settings.startup_scale_factor_from_raw_config(
        {"app": {"scale_factor": 1.0}}
    ) == 1.15
    assert settings.startup_scale_factor_from_raw_config(
        {"app": {"scale_factor": 1.25}}
    ) == 1.25
    assert settings.startup_scale_factor_from_raw_config(
        {
            "app": {
                "scale_factor": 1.0,
                "_scale_factor_default_version": settings.SCALE_FACTOR_DEFAULT_VERSION,
            }
        }
    ) == 1.0
