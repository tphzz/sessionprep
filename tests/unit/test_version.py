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


def _source_runtime(monkeypatch):
    monkeypatch.delattr(_version.sys, "frozen", raising=False)
    monkeypatch.delitem(_version.__dict__, "__compiled__", raising=False)


def _git_runner(responses: dict[tuple[str, ...], str]):
    def run(cmd, **_kwargs):
        args = tuple(cmd[1:])
        value = responses.get(args)
        if value is None:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{value}\n")

    return run


def test_source_runtime_prefers_git_over_build_version(monkeypatch):
    _source_runtime(monkeypatch)
    monkeypatch.setattr(_version, "_git_version", lambda *, strict: "0.3.5")
    monkeypatch.setattr(_version, "_build_version", lambda: "0.3.4")
    _no_metadata(monkeypatch)

    assert _version.get_version(strict=True) == "0.3.5"


def test_compiled_runtime_prefers_build_version_without_git(monkeypatch):
    monkeypatch.setattr(_version.sys, "frozen", True, raising=False)
    monkeypatch.setattr(_version, "_build_version", lambda: "0.3.5")
    _no_metadata(monkeypatch)

    def fail_git(*, strict):
        raise AssertionError("compiled runtime should not call Git")

    monkeypatch.setattr(_version, "_git_version", fail_git)

    assert _version.get_version(strict=True) == "0.3.5"


def test_compiled_runtime_falls_back_after_build_version(monkeypatch):
    monkeypatch.setattr(_version.sys, "frozen", True, raising=False)
    _no_build_version(monkeypatch)
    monkeypatch.setattr(_version, "_metadata_version", lambda: "0.3.4")

    def fail_git(*, strict):
        raise AssertionError("compiled runtime should not call Git")

    monkeypatch.setattr(_version, "_git_version", fail_git)

    assert _version.get_version(strict=True) == "0.3.4"


def test_compiled_runtime_without_metadata_returns_unknown_without_git(monkeypatch):
    monkeypatch.setattr(_version.sys, "frozen", True, raising=False)
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)

    def fail_git(*, strict):
        raise AssertionError("compiled runtime should not call Git")

    monkeypatch.setattr(_version, "_git_version", fail_git)

    assert _version.get_version(strict=False) == "0.0.0+unknown"


def test_compiled_runtime_strict_failure_does_not_call_git(monkeypatch):
    monkeypatch.setattr(_version.sys, "frozen", True, raising=False)
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)

    def fail_git(*, strict):
        raise AssertionError("compiled runtime should not call Git")

    monkeypatch.setattr(_version, "_git_version", fail_git)

    with pytest.raises(_version.VersionResolutionError):
        _version.get_version(strict=True)


@pytest.mark.skipif(
    not hasattr(subprocess, "STARTUPINFO"),
    reason="Windows subprocess startup flags are only available on Windows",
)
def test_git_subprocess_is_hidden_on_windows(monkeypatch, tmp_path):
    captured = {}

    def run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="0.3.5\n")

    monkeypatch.setattr(_version.sys, "platform", "win32")
    monkeypatch.setattr(_version.subprocess, "run", run)

    assert _version._git(["describe"], cwd=tmp_path) == "0.3.5"

    kwargs = captured["kwargs"]
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert kwargs["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
    assert kwargs["startupinfo"].wShowWindow == subprocess.SW_HIDE


def test_git_subprocess_has_no_windows_flags_off_windows(monkeypatch, tmp_path):
    captured = {}

    def run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="0.3.5\n")

    monkeypatch.setattr(_version.sys, "platform", "linux")
    monkeypatch.setattr(_version.subprocess, "run", run)

    assert _version._git(["describe"], cwd=tmp_path) == "0.3.5"

    assert "creationflags" not in captured["kwargs"]
    assert "startupinfo" not in captured["kwargs"]


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


def test_invalid_exact_git_tag_falls_back_to_branch(monkeypatch):
    _no_build_version(monkeypatch)
    _no_metadata(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({
            ("describe", "--tags", "--exact-match", "HEAD"): "branch-build-0.3.5",
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


def test_installed_metadata_is_runtime_fallback(monkeypatch):
    _no_build_version(monkeypatch)
    monkeypatch.setattr(
        _version.subprocess,
        "run",
        _git_runner({}),
    )
    monkeypatch.setattr(_version, "_metadata_version", lambda: "0.3.4")

    assert _version.get_version(strict=False) == "0.3.4"
