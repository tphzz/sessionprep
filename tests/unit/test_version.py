from __future__ import annotations

import subprocess
import sys
import types
import pytest

from build_conf import BUILD_VERSION_MODULE, TARGETS
from sessionpreplib import _version


def _no_build_version(monkeypatch):
    monkeypatch.setattr(_version, "_build_version", lambda: None)


def _no_metadata(monkeypatch):
    monkeypatch.setattr(_version, "_metadata_version", lambda: None)


def _git_runner(responses: dict[tuple[str, ...], str]):
    def run(cmd, **_kwargs):
        args = tuple(cmd[1:])
        value = responses.get(args)
        if value is None:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{value}\n")

    return run


def test_exact_git_tag_is_version(monkeypatch):
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({
            ("describe", "--tags", "--exact-match", "HEAD"): "0.3.5",
        }),
    )

    assert _version.get_version(strict=True) == "0.3.5"


def test_git_branch_appends_dev_hash(monkeypatch):
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({
            ("branch", "--show-current"): "0.3.5",
            ("rev-parse", "--short=7", "HEAD"): "b2f7cf6",
        }),
    )

    assert _version.get_version(strict=True) == "0.3.5.dev0+gb2f7cf6"


def test_invalid_branch_base_fails(monkeypatch):
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({
            ("branch", "--show-current"): "feature/foo",
            ("rev-parse", "--short=7", "HEAD"): "b2f7cf6",
        }),
    )

    with pytest.raises(_version.VersionResolutionError):
        _version.get_version(strict=True)


def test_version_normalization_without_packaging(monkeypatch):
    monkeypatch.setattr(_version, "Version", None)

    assert _version._normalize_version("v0.3.5.dev0+gb2f7cf6", source="test") == (
        "0.3.5.dev0+gb2f7cf6"
    )


def test_invalid_version_without_packaging_fails(monkeypatch):
    monkeypatch.setattr(_version, "Version", None)

    with pytest.raises(_version.VersionResolutionError):
        _version._normalize_version("feature/foo", source="test")


def test_generated_build_version_is_used_without_git(monkeypatch, tmp_path):
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({}),
    )
    version_file = tmp_path / "_version.py"
    version_file.write_text("", encoding="utf-8")
    (tmp_path / "_build_version.py").write_text(
        'BUILD_VERSION = "0.3.5.dev0+gb2f7cf6"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(_version, "__file__", str(version_file))

    assert _version.get_version(strict=True) == "0.3.5.dev0+gb2f7cf6"


def test_static_build_version_module_is_used(monkeypatch):
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({}),
    )
    module = types.ModuleType(BUILD_VERSION_MODULE)
    module.BUILD_VERSION = "0.3.5.dev0+gb2f7cf6"
    monkeypatch.setitem(sys.modules, BUILD_VERSION_MODULE, module)

    assert _version.get_version(strict=True) == "0.3.5.dev0+gb2f7cf6"


def test_build_version_module_is_included_for_freezers():
    for target in TARGETS.values():
        assert BUILD_VERSION_MODULE in target["nuitka_include_modules"]
        assert BUILD_VERSION_MODULE in target["pyinstaller_hidden_imports"]


def test_installed_metadata_is_runtime_fallback(monkeypatch):
    _no_build_version(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({}),
    )
    monkeypatch.setattr(_version, "_metadata_version", lambda: "0.3.4")

    assert _version.get_version(strict=False) == "0.3.4"
