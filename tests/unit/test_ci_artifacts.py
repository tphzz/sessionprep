from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sessionprep_ci", ROOT / "packaging/github/build.py")
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)


def test_manifest_binds_bytes_configuration_and_commit(tmp_path):
    (tmp_path / "installer.exe").write_bytes(b"payload")
    ci.write_manifest(tmp_path, "win-x64", "0.3.5", "a" * 40, False)
    assert ci.verify_manifest(tmp_path, "win-x64", "0.3.5", "a" * 40, False)
    with pytest.raises(ValueError, match="identity"):
        ci.verify_manifest(tmp_path, "win-x64", "0.3.5", "b" * 40, False)
    (tmp_path / "installer.exe").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ci.verify_manifest(tmp_path, "win-x64", "0.3.5", "a" * 40, False)


def test_manifest_rejects_unlisted_files(tmp_path):
    (tmp_path / "package.deb").write_bytes(b"payload")
    ci.write_manifest(tmp_path, "linux-x64", "0.3.5", "a" * 40, False)
    (tmp_path / "unexpected.exe").write_bytes(b"extra")
    with pytest.raises(ValueError, match="file set"):
        ci.verify_manifest(tmp_path, "linux-x64", "0.3.5", "a" * 40, False)


def test_collector_checks_every_platform_before_creating_destination(tmp_path):
    destination = tmp_path / "release"
    with pytest.raises((ValueError, FileNotFoundError)):
        ci.collect(tmp_path / "artifacts", destination, "0.3.5", "a" * 40)
    assert not destination.exists()


@pytest.mark.parametrize("has_token", [True, False])
def test_release_preflight_checks_workflow_changes_before_builds(tmp_path, monkeypatch, has_token):
    import argparse
    import json
    monkeypatch.setattr(ci, "check_linux_audio_runtime", lambda: None)
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"repository":{"default_branch":"main"}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setattr(ci.subprocess, "check_output", lambda args, **kwargs: ".github/workflows/build.yml" if args[1]=="diff" else "a"*40)
    monkeypatch.setattr(ci, "resolve_ref_version", lambda *a: "0.3.5.dev0+gaaaaaaa")
    output=tmp_path/"outputs"
    args=argparse.Namespace(ref_type="branch",ref_name="feature/ci",sha="a"*40,github_output=str(output),
                            release_drafts=True,release_token_available=has_token)
    if has_token:
        ci.metadata(args)
        assert "version=" in output.read_text()
    else:
        with pytest.raises(ValueError, match="Workflows:write"):
            ci.metadata(args)
        assert not output.exists()


def test_collector_accepts_only_complete_matching_platform_artifacts(tmp_path):
    artifact_root=tmp_path/"artifacts"
    for platform in (*ci.PLATFORMS,"win-universal"):
        directory=artifact_root/f"sessionprep-{platform}-release"
        directory.mkdir(parents=True)
        suffixes=("tar.gz","deb","rpm") if platform.startswith("linux-") else ("dmg",) if platform.startswith("macos-") else ("exe",)
        for suffix in suffixes:
            (directory/f"sessionprep-0.3.5-{platform}.{suffix}").write_bytes(platform.encode())
        ci.write_manifest(directory,platform,"0.3.5","a"*40,False)
    ci.collect(artifact_root,tmp_path/"release","0.3.5","a"*40)
    delivered=list((tmp_path/"release").iterdir())
    assert len(delivered)==9
    assert len([p for p in delivered if p.suffix==".exe"])==1



def test_linux_audio_runtime_reports_missing_portaudio(monkeypatch):
    import ctypes
    import ctypes.util
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: pytest.fail("Missing library must not be loaded"))
    with pytest.raises(ValueError, match="libportaudio2"):
        ci.check_linux_audio_runtime()


def test_linux_audio_runtime_reports_unloadable_portaudio(monkeypatch):
    import ctypes
    import ctypes.util
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libportaudio.so.2")
    def load(name):
        raise OSError("Dependent shared object missing")
    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(ValueError, match="runner image") as error:
        ci.check_linux_audio_runtime()
    assert isinstance(error.value.__cause__, OSError)


def test_linux_audio_runtime_loads_without_opening_a_device(monkeypatch):
    import ctypes
    import ctypes.util
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libportaudio.so.2")
    loaded=[]
    monkeypatch.setattr(ctypes, "CDLL", lambda name: loaded.append(name) or object())
    ci.check_linux_audio_runtime()
    assert loaded == ["libportaudio.so.2"]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_linux_audio_runtime_leaves_other_platforms_untouched(monkeypatch, platform):
    import ctypes.util
    monkeypatch.setattr(ci.sys, "platform", platform)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: pytest.fail("Non-Linux runtime must not be probed"))
    ci.check_linux_audio_runtime()


def test_metadata_runtime_failure_leaves_no_success_outputs(tmp_path, monkeypatch):
    import argparse
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *a, **kw: "a"*40)
    def missing():
        raise ValueError("Missing PortAudio in runner image")
    monkeypatch.setattr(ci, "check_linux_audio_runtime", missing)
    output=tmp_path/"outputs"
    args=argparse.Namespace(ref_type="branch",ref_name="0.3.5",sha="a"*40,
                            github_output=str(output),release_drafts=False,release_token_available=False)
    with pytest.raises(ValueError, match="PortAudio"):
        ci.metadata(args)
    assert not output.exists()


def test_linux_compilation_stops_before_nuitka_without_audio_runtime(monkeypatch):
    import argparse
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ci, "get_platform_suffix", lambda: "linux-x64")
    monkeypatch.setattr(ci.shutil, "which", lambda name: "/provided/" + name)
    def missing():
        raise ValueError("Missing PortAudio in runner image")
    monkeypatch.setattr(ci, "check_linux_audio_runtime", missing)
    monkeypatch.setattr(ci, "run", lambda *a, **kw: pytest.fail("Nuitka must not start"))
    with pytest.raises(ValueError, match="PortAudio"):
        ci.compile_app(argparse.Namespace(platform="linux-x64",version="0.3.5",debug=False))



@pytest.mark.parametrize("missing", [
    ("ccache", "nfpm"),
    ("gcc", "ccache", "patchelf", "nfpm"),
])
def test_linux_compile_reports_all_missing_tools_before_starting(monkeypatch, missing):
    import argparse
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ci, "get_platform_suffix", lambda: "linux-x64")
    monkeypatch.setattr(ci, "check_linux_audio_runtime", lambda: None)
    monkeypatch.setattr(ci.shutil, "which", lambda name: None if name in missing else "/provided/" + name)
    monkeypatch.setattr(ci, "run", lambda *a, **kw: pytest.fail("Build must not start with missing prerequisites"))
    with pytest.raises(ValueError) as error:
        ci.compile_app(argparse.Namespace(platform="linux-x64", version="0.3.5", debug=False))
    message = str(error.value)
    for name in missing:
        assert name in message, message
    assert "runner image" in message



def test_linux_compile_starts_when_all_prerequisites_are_present(monkeypatch):
    import argparse
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setattr(ci, "get_platform_suffix", lambda: "linux-x64")
    monkeypatch.setattr(ci, "check_linux_audio_runtime", lambda: None)
    monkeypatch.setattr(ci.shutil, "which", lambda name: "/provided/" + name)
    started=[]
    monkeypatch.setattr(ci, "run", lambda *args, **kwargs: started.append(True))
    ci.compile_app(argparse.Namespace(platform="linux-x64", version="0.3.5", debug=False))
    assert started == [True]


@pytest.mark.parametrize("native_platform, python_platform, low_memory, jobs", [
    ("win-x64", "win32", True, "4"),
    ("win-arm64", "win32", True, None),
    ("linux-x64", "linux", False, None),
    ("linux-arm64", "linux", False, None),
    ("macos-arm64", "darwin", False, None),
    ("macos-x64", "darwin", False, None),
])
@pytest.mark.parametrize("debug", [False, True])
def test_ci_bounds_windows_compiler_memory(monkeypatch, native_platform,
                                          python_platform, low_memory, jobs, debug):
    import argparse
    started = []
    monkeypatch.setattr(ci.sys, "platform", python_platform)
    monkeypatch.setattr(ci, "get_platform_suffix", lambda: native_platform)
    monkeypatch.setattr(ci, "check_linux_audio_runtime", lambda: None)
    monkeypatch.setattr(ci.shutil, "which", lambda name: "/provided/" + name)
    monkeypatch.setattr(ci, "run", lambda *args, **kwargs: started.append((args, kwargs)))

    ci.compile_app(argparse.Namespace(platform=native_platform, version="0.3.5", debug=debug))

    assert len(started) == (2 if python_platform == "win32" else 1)
    if python_platform == "win32":
        assert started[0][0][1:] == (ROOT / "packaging/windows/build_installer.py", "check")
    command, options = started[-1]
    assert ("--low-memory" in command) is low_memory
    # x64 checks its warm clcache in parallel; ARM64 keeps one SciPy-safe job.
    assert (command[command.index("--jobs") + 1] if "--jobs" in command else None) == jobs
    assert ("--debug" in command) is debug
    assert command[command.index("--build-version") + 1] == "0.3.5"
    assert options["cwd"] == ROOT



def test_windows_missing_installer_stops_before_nuitka(monkeypatch):
    import argparse
    import subprocess
    monkeypatch.setattr(ci.sys, "platform", "win32")
    monkeypatch.setattr(ci, "get_platform_suffix", lambda: "win-x64")
    monkeypatch.setattr(ci, "check_linux_audio_runtime", lambda: None)
    monkeypatch.setattr(ci.shutil, "which", lambda name: "/provided/" + name)
    def fail_preflight(*command, **kwargs):
        assert command[1:] == (ROOT / "packaging/windows/build_installer.py", "check")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(ci, "run", fail_preflight)
    with pytest.raises(subprocess.CalledProcessError):
        ci.compile_app(argparse.Namespace(platform="win-x64", version="0.3.5", debug=False))
