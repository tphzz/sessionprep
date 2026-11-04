import sysconfig

import pytest

import build_conf


@pytest.mark.parametrize("target, native_platform", [
    ("win-x64", "win-amd64"), ("win-arm64", "win-arm64"),
])
def test_windows_python_accepts_matching_native_interpreter(monkeypatch, target, native_platform):
    monkeypatch.setattr(build_conf.sys, "version_info", (3, 13, 14))
    monkeypatch.setattr(sysconfig, "get_platform", lambda: native_platform)
    build_conf.check_windows_python(target)


@pytest.mark.parametrize("target, native_platform", [
    ("win-x64", "win-arm64"), ("win-arm64", "win-amd64"),
])
def test_windows_python_rejects_emulated_interpreter_despite_msvc_target(monkeypatch, target, native_platform):
    monkeypatch.setattr(build_conf.sys, "version_info", (3, 13, 14))
    monkeypatch.setattr(sysconfig, "get_platform", lambda: native_platform)
    monkeypatch.setattr(build_conf.platform, "machine", lambda: "ARM64")
    monkeypatch.setenv("VSCMD_ARG_TGT_ARCH", "arm64")
    with pytest.raises(ValueError, match="CI_PYTHON") as error:
        build_conf.check_windows_python(target)
    assert native_platform in str(error.value)
    assert build_conf.sys.executable in str(error.value)


@pytest.mark.parametrize("version", [(3, 12, 9), (3, 14, 0)])
def test_windows_python_rejects_wrong_python_version(monkeypatch, version):
    monkeypatch.setattr(build_conf.sys, "version_info", version)
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "win-arm64")
    with pytest.raises(ValueError, match="3.13"):
        build_conf.check_windows_python("win-arm64")


@pytest.mark.parametrize("native_platform, expected", [
    ("win-amd64", "win-x64"), ("win-arm64", "win-arm64"),
])
def test_windows_artifact_suffix_describes_interpreter_not_physical_host(monkeypatch, native_platform, expected):
    monkeypatch.setattr(build_conf.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build_conf.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(sysconfig, "get_platform", lambda: native_platform)
    assert build_conf.get_platform_suffix() == expected
