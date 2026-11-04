from pathlib import Path

import json
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_manual_entrypoints_and_checkbox_defaults():
    for name in ("build-nuitka.yml", "build-windows-x64.yml", "build-windows-arm64.yml",
                 "build-linux-x64.yml", "build-linux-arm64.yml", "build-macos.yml"):
        workflow = load(name)
        assert set(workflow["on"]) == {"workflow_dispatch"}
        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        assert inputs["debug"]["type"] == "boolean" and inputs["debug"]["default"] is False
        if name in ("build-nuitka.yml", "build-macos.yml"):
            assert inputs["notarize_macos"]["default"] is False
        assert workflow["concurrency"]["cancel-in-progress"] is False
        assert workflow["concurrency"]["queue"] == "max"


def test_all_platforms_preserves_hardware_lanes_and_release_join():
    jobs = load("build-nuitka.yml")["jobs"]
    assert "linux-x64" in jobs["windows-x64"]["needs"]
    assert "linux-arm64" in jobs["windows-arm64"]["needs"]
    assert "windows-arm64" in jobs["macos"]["needs"]
    assert set(jobs["windows-installer"]["needs"]) >= {"windows-x64", "windows-arm64", "checks"}
    assert set(jobs["release"]["needs"]) >= {"windows-installer", "macos", "checks"}
    assert "!inputs.debug" in jobs["release"]["if"]
    assert jobs["release"]["permissions"]["contents"] == "write"


def test_only_all_platforms_can_publish():
    for path in WORKFLOWS.glob("*.yml"):
        workflow = load(path.name)
        if path.name != "build-nuitka.yml":
            assert workflow["permissions"]["contents"] == "read"
            assert "publish-draft-release" not in path.read_text()
            for job in workflow["jobs"].values():
                assert job.get("permissions", {}).get("contents") != "write"


def test_resource_locks_are_shared_and_never_cancel_running_work():
    for name in ("90-build-checks.yml", "91-build-platform.yml", "92-build-macos.yml", "93-windows-bundle.yml"):
        for job in load(name)["jobs"].values():
            lock = job["concurrency"]
            assert lock["queue"] == "max" and lock["cancel-in-progress"] is False
            assert "github.ref" not in lock["group"] and "github.workflow" not in lock["group"]
            if job["runs-on"] == "${{ fromJSON(inputs.runner_labels) }}":
                callers = [j for path in WORKFLOWS.glob("*.yml") for j in load(path.name)["jobs"].values()
                           if j.get("uses", "").endswith("91-build-platform.yml")]
                assert callers
                for caller in callers:
                    assert json.loads(caller["with"]["runner_labels"])[0] == "self-hosted"
                    expected = "sessionprep-apple-silicon-host" if caller["with"]["platform"].endswith("arm64") else "sessionprep-x64-host"
                    assert caller["with"]["resource_group"] == expected
            else:
                assert job["runs-on"][0] == "self-hosted"


def test_no_certificate_import_or_tag_forced_notarization():
    content = (WORKFLOWS / "92-build-macos.yml").read_text()
    assert "notarize_macos" in content
    assert "import-codesign-certs" not in content and "P12_BASE64" not in content
    assert "github.ref_type" not in content


def test_runner_context_is_only_used_after_job_is_scheduled():
    for path in WORKFLOWS.glob("*.yml"):
        for job in load(path.name)["jobs"].values():
            assert all("runner." not in str(value) for value in job.get("env", {}).values())


def test_windows_native_python_validation_precedes_uv_and_exports():
    for name in ("91-build-platform.yml", "93-windows-bundle.yml"):
        steps = next(iter(load(name)["jobs"].values()))["steps"]
        index, step = next((i, step) for i, step in enumerate(steps)
                           if step.get("name") == "Verify native Windows Python and expose Git Bash")
        script = step["run"]
        assert "check_windows_python" in script
        assert "$env:CI_PYTHON" in script
        assert script.index("check_windows_python") < script.index('"CI_PYTHON=$python"')
        assert index < next(i for i, step in enumerate(steps) if step.get("name") == "Set up uv")



def test_windows_python_powershell_guard_rejects_wrong_arch_before_export(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    import sysconfig
    import pytest
    powershell = shutil.which("powershell.exe")
    if sys.platform != "win32" or powershell is None:
        pytest.skip("Native PowerShell verification requires Windows")
    native = {"win-amd64": "win-x64", "win-arm64": "win-arm64"}.get(sysconfig.get_platform())
    if native is None:
        pytest.skip("Native PowerShell verification requires 64-bit Windows Python")
    for name, target in (
        ("91-build-platform.yml", "win-x64"),
        ("91-build-platform.yml", "win-arm64"),
        ("93-windows-bundle.yml", "win-x64"),
    ):
        succeeds = target == native
        steps = next(iter(load(name)["jobs"].values()))["steps"]
        script = next(step["run"] for step in steps
                      if step.get("name") == "Verify native Windows Python and expose Git Bash")
        script = script.split("$git =", 1)[0]
        destination = tmp_path / (name + target + ".env")
        environment = dict(os.environ, SOURCE_DIR=str(ROOT), BUILD_PLATFORM=target,
                           CI_PYTHON=sys.executable, GITHUB_ENV=str(destination),
                           PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                                env=environment, text=True, capture_output=True)
        assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
        if succeeds:
            assert "CI_PYTHON=" + sys.executable in destination.read_text(encoding="utf-8-sig")
        else:
            assert not destination.exists()
            assert target in result.stdout + result.stderr
