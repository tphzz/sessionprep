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

    build.run_nuitka(target, low_memory=low_memory, debug=True, build_version="0.3.5")

    assert ("--low-memory" in calls[0]) is low_memory
    assert ("--include-package=rich._unicode_data" in calls[0]) is (target == "cli")
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


@pytest.mark.parametrize("jobs", [None, 4])
def test_explicit_jobs_override_the_low_memory_single_job(tmp_path, monkeypatch, jobs):
    calls = []
    monkeypatch.setattr(build, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(build, "DIST_NUITKA", str(tmp_path / "dist"))
    monkeypatch.setattr(build, "_check_dependencies", lambda *args: None)
    monkeypatch.setattr(build, "write_build_version_file", lambda version: version)
    monkeypatch.setattr(build, "remove_build_version_file", lambda: None)
    monkeypatch.setattr(build.subprocess, "check_call", lambda args: calls.append(args))

    build.run_nuitka("cli", low_memory=True, jobs=jobs, build_version="0.3.5")

    assert "--low-memory" in calls[0]
    assert [arg for arg in calls[0] if arg.startswith("--jobs")] == ([] if jobs is None else ["--jobs=4"])


def test_cli_passes_jobs_to_both_targets(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build, "DIST_NUITKA", str(tmp_path / "dist"))
    monkeypatch.setattr(sys, "argv", ["build_nuitka.py", "all", "--low-memory", "--jobs", "4"])
    monkeypatch.setattr(build, "run_nuitka", lambda target, **kwargs: calls.append((target, kwargs)))

    build.main()

    assert [(target, options["jobs"]) for target, options in calls] == [("cli", 4), ("gui", 4)]


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

    build.run_nuitka(target, low_memory=True, debug=debug, build_version="0.3.5")

    command = calls[0]
    expected = ["--noinclude-custom-mode=scipy.integrate._lebedev:bytecode"] if native_platform == "win-arm64" else []
    assert [arg for arg in command if arg.startswith("--noinclude-custom-mode=")] == expected
    assert not any(arg.startswith("--nofollow-import-to=scipy") for arg in command)
    assert "--low-memory" in command
    is_win_arm64_gui = native_platform == "win-arm64" and target == "gui"
    assert ("--include-package=_sounddevice_data" in command) is is_win_arm64_gui
    user_cfgs = [arg for arg in command if arg.startswith("--user-package-configuration-file=")]
    assert len(user_cfgs) == (1 if is_win_arm64_gui else 0)
    if is_win_arm64_gui:
        assert user_cfgs[0].endswith("sounddevice-arm64.nuitka-package.config.yml")


def test_sounddevice_arm64_nuitka_config_merges_cleanly():
    from pathlib import Path
    from nuitka.Tracing import plugins_logger
    from nuitka.utils.Yaml import PackageConfigYaml, parsePackageYaml

    cfg_path = Path(build_conf.NUITKA_GUI_USER_PACKAGE_CONFIGS_BY_PLATFORM["win-arm64"][0])
    assert cfg_path.is_file()
    base = parsePackageYaml(
        logger=plugins_logger,
        package_name="nuitka.plugins.standard",
        filename="standard.nuitka-package.config.yml",
        assume_yes_for_downloads=False,
        check_checksums=False,
    )
    user = PackageConfigYaml(
        logger=plugins_logger,
        name=cfg_path.name,
        file_data=cfg_path.read_bytes(),
        assume_yes_for_downloads=False,
        check_checksums=False,
    )
    base.update(user)
    dlls = base.get("_sounddevice_data", "dlls")
    assert len(dlls) == 1
    assert dlls[0]["when"] == "win32 and arch_arm64"
    assert dlls[0]["from_filenames"]["prefixes"] == ["libportaudioarm64"]


def test_smoke_check_portaudio_bundle(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import shutil

    smoke_path = Path(build_conf.BASE_DIR) / "packaging" / "github" / "smoke.py"
    spec = importlib.util.spec_from_file_location("sessionprep_smoke", smoke_path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    fake_gui = tmp_path / "sessionprep-gui.dist" / "sessionprep-gui.exe"
    fake_gui.parent.mkdir(parents=True)
    smoke.check_portaudio_bundle(fake_gui, "linux-x64")
    smoke.check_portaudio_bundle(fake_gui, "linux-arm64")
    with pytest.raises(ValueError, match="Unsupported platform"):
        smoke.check_portaudio_bundle(fake_gui, "win-amd64")
    with pytest.raises(FileNotFoundError, match="libportaudioarm64.dll"):
        smoke.check_portaudio_bundle(fake_gui, "win-arm64")
    with pytest.raises(FileNotFoundError, match="libportaudio.dylib"):
        smoke.check_portaudio_bundle(fake_gui, "macos-arm64")

    # Missing Pa_GetVersion symbol raises RuntimeError on all platforms
    dummy_dll = fake_gui.parent / "_sounddevice_data" / "portaudio-binaries" / "libportaudioarm64.dll"
    dummy_dll.parent.mkdir(parents=True, exist_ok=True)
    dummy_dll.write_bytes(b"stub")
    with monkeypatch.context() as mp:
        mp.setattr(smoke.ctypes, "CDLL", lambda _p: type("_FakeLib", (), {"_handle": 0})())
        if sys.platform == "win32":
            import _ctypes

            mp.setattr(_ctypes, "FreeLibrary", lambda _h: None)
        with pytest.raises(RuntimeError, match="Pa_GetVersion"):
            smoke.check_portaudio_bundle(fake_gui, "win-arm64")
    dummy_dll.unlink()

    # Positive real-DLL load + unload check using _sounddevice_data (never calls Pa_Initialize)
    platform_suffix = build_conf.get_platform_suffix()
    expected_name = {
        "win-x64": "libportaudio64bit.dll",
        "win-arm64": "libportaudioarm64.dll",
        "macos-x64": "libportaudio.dylib",
        "macos-arm64": "libportaudio.dylib",
    }.get(platform_suffix)
    if expected_name is not None:
        import _sounddevice_data

        src = Path(next(iter(_sounddevice_data.__path__))) / "portaudio-binaries" / expected_name
        assert src.is_file(), f"Expected bundled PortAudio library at {src}"
        dest = fake_gui.parent / "_sounddevice_data" / "portaudio-binaries" / expected_name
        shutil.copy2(src, dest)
        smoke.check_portaudio_bundle(fake_gui, platform_suffix)
        # Verify Windows handle was released (no lingering file lock)
        dest.unlink()
