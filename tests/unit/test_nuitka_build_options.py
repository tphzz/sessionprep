import sys

import pytest

import build_nuitka as build
import build_conf


@pytest.mark.parametrize("debug", [False, True])
def test_windows_debug_command_keeps_symbols_and_attaches_console(tmp_path, monkeypatch, debug):
    calls=[]
    versions=[]
    monkeypatch.setattr(build,"BASE_DIR",str(tmp_path))
    monkeypatch.setattr(build,"DIST_NUITKA",str(tmp_path/"dist"))
    monkeypatch.setattr(build.sys,"platform","win32")
    monkeypatch.setattr(build,"_check_dependencies",lambda *a: None)
    monkeypatch.setattr(build,"write_build_version_file",lambda version: versions.append(version) or version)
    monkeypatch.setattr(build,"remove_build_version_file",lambda: None)
    monkeypatch.setattr(build.subprocess,"check_call",lambda args: calls.append(args))
    monkeypatch.setattr(build,"fetch_and_bundle_rps",lambda *a: None)
    build.run_nuitka("gui",debug=debug,build_version="0.3.5.dev0+gabcdef0")
    command=calls[0]
    assert ("--unstripped" in command) is debug
    assert "--windows-console-mode="+("attach" if debug else "disable") in command
    assert "--debug" not in command and versions==["0.3.5.dev0+gabcdef0"]


def test_all_targets_share_explicit_ci_version(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(build.sys,"platform","win32")
    monkeypatch.setattr(build,"DIST_NUITKA",str(tmp_path/"dist"))
    monkeypatch.setattr(sys,"argv",["build_nuitka.py","all","--debug","--build-version","0.3.5"])
    monkeypatch.setattr(build,"run_nuitka",lambda target,**kwargs: calls.append((target,kwargs)))
    build.main()
    assert [target for target,_ in calls]==["cli","gui"]
    assert all(args["build_version"]=="0.3.5" and args["debug"] is True for _,args in calls)


def test_invalid_explicit_version_never_writes_build_metadata(monkeypatch,tmp_path):
    destination=tmp_path/"version.py"
    monkeypatch.setattr(build_conf,"BUILD_VERSION_FILE",str(destination))
    with pytest.raises(Exception,match="not a valid PEP 440"):
        build_conf.write_build_version_file('0.3.5"\ninvalid')
    assert not destination.exists()


@pytest.mark.parametrize("target", ["cli", "gui"])
@pytest.mark.parametrize("low_memory", [False, True])
def test_low_memory_option_reaches_each_nuitka_target(tmp_path, monkeypatch, target, low_memory):
    calls = []
    monkeypatch.setattr(build, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(build, "DIST_NUITKA", str(tmp_path / "dist"))
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build, "_check_dependencies", lambda *args: None)
    monkeypatch.setattr(build, "write_build_version_file", lambda version: version)
    monkeypatch.setattr(build, "remove_build_version_file", lambda: None)
    monkeypatch.setattr(build.subprocess, "check_call", lambda args: calls.append(args))
    monkeypatch.setattr(build, "fetch_and_bundle_rps", lambda *args: None)

    build.run_nuitka(target, low_memory=low_memory, debug=True, build_version="0.3.5")

    assert ("--low-memory" in calls[0]) is low_memory
    assert "--unstripped" in calls[0]
    assert "--lto=no" in calls[0]


@pytest.mark.parametrize("low_memory", [False, True])
def test_cli_passes_memory_mode_to_both_targets(tmp_path, monkeypatch, low_memory):
    calls = []
    arguments = ["build_nuitka.py", "all"]
    if low_memory:
        arguments.append("--low-memory")
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build, "DIST_NUITKA", str(tmp_path / "dist"))
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(build, "run_nuitka", lambda target, **kwargs: calls.append((target, kwargs)))

    build.main()

    assert [target for target, _ in calls] == ["cli", "gui"]
    assert all(options["low_memory"] is low_memory for _, options in calls)


@pytest.mark.parametrize("target", ["cli", "gui"])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("native_platform,python_platform", [
    ("win-arm64", "win32"), ("win-x64", "win32"),
    ("linux-arm64", "linux"), ("linux-x64", "linux"),
    ("macos-arm64", "darwin"), ("macos-x64", "darwin"),
])
def test_arm64_preserves_lebedev_as_bytecode(tmp_path, monkeypatch, target, debug,
                                           native_platform, python_platform):
    calls = []
    monkeypatch.setattr(build, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(build, "DIST_NUITKA", str(tmp_path / "dist"))
    monkeypatch.setattr(build.sys, "platform", python_platform)
    monkeypatch.setattr(build, "get_platform_suffix", lambda: native_platform)
    monkeypatch.setattr(build, "_check_dependencies", lambda *args: None)
    monkeypatch.setattr(build, "write_build_version_file", lambda version: version)
    monkeypatch.setattr(build, "remove_build_version_file", lambda: None)
    monkeypatch.setattr(build.subprocess, "check_call", lambda args: calls.append(args))
    monkeypatch.setattr(build, "fetch_and_bundle_rps", lambda *args: None)

    build.run_nuitka(target, low_memory=True, debug=debug, build_version="0.3.5")

    command = calls[0]
    expected = ["--noinclude-custom-mode=scipy.integrate._lebedev:bytecode"] if native_platform == "win-arm64" else []
    assert [arg for arg in command if arg.startswith("--noinclude-custom-mode=")] == expected
    assert not any(arg.startswith("--nofollow-import-to=scipy") for arg in command)
    assert "--low-memory" in command
