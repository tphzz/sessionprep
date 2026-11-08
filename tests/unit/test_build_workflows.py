from pathlib import Path

import json
import yaml
import pytest

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
        assert "windows_python.ps1" in script
        assert "Find-SessionPrepPython" in script
        assert script.index("Find-SessionPrepPython") < script.index('"CI_PYTHON=$python"')
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
    source = tmp_path / "source [literal]"
    (source / "packaging/github").mkdir(parents=True)
    shutil.copy2(ROOT / "build_conf.py", source)
    shutil.copy2(ROOT / "packaging/github/windows_python.ps1", source / "packaging/github")
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
        environment = dict(os.environ, SOURCE_DIR=source.name, BUILD_PLATFORM=target,
                           CI_PYTHON=sys.executable, GITHUB_ENV=str(destination),
                           PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                                env=environment, cwd=tmp_path, text=True, capture_output=True)
        assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
        if succeeds:
            assert "CI_PYTHON=" + sys.executable in destination.read_text(encoding="utf-8-sig")
        else:
            assert not destination.exists()
            assert target in result.stdout + result.stderr



def windows_cleanup_step(name):
    steps = next(iter(load(name)["jobs"].values()))["steps"]
    return next(step for step in steps if step.get("name") == "Clean only run-owned workspace (Windows)")


@pytest.mark.parametrize("name", ["93-windows-bundle.yml"])
def test_windows_cleanup_has_native_shell_even_before_preflight(name):
    step = windows_cleanup_step(name)
    assert step["shell"] == "powershell"
    assert "always()" in step["if"]


@pytest.mark.parametrize("name", ["93-windows-bundle.yml"])
def test_windows_cleanup_runs_without_bash_or_exported_ci_python(tmp_path, name):
    import os
    import shutil
    import stat
    import subprocess
    import sys
    powershell = shutil.which("powershell.exe")
    if sys.platform != "win32" or powershell is None:
        pytest.skip("Native PowerShell verification requires Windows")
    workspace = tmp_path / "workspace [literal]"
    temporary = tmp_path / "temporary [literal]"
    workspace.mkdir()
    temporary.mkdir()
    source = workspace / "src-123-1-win-arm64"
    environment = temporary / "sessionprep-env-123-1-win-arm64"
    for target in (source, environment):
        (target / "nested").mkdir(parents=True)
        readonly = target / "nested" / "readonly.dat"
        readonly.write_bytes(b"run-owned")
        readonly.chmod(stat.S_IREAD)
    unrelated = workspace / "src-other-run-win-arm64"
    unrelated.mkdir()
    (unrelated / "keep.dat").write_bytes(b"keep")
    variables = dict(os.environ, PATH=str(Path(sys.executable).parent),
                     GITHUB_WORKSPACE=str(workspace), RUNNER_TEMP=str(temporary),
                     GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1",
                     CLEANUP_NAMES=json.dumps(["win-arm64"]), PYTHONDONTWRITEBYTECODE="1")
    variables.pop("CI_PYTHON", None)
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command",
                             windows_cleanup_step(name)["run"]], env=variables, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not source.exists() and not environment.exists()
    assert (unrelated / "keep.dat").read_bytes() == b"keep"
    assert workspace.is_dir() and temporary.is_dir()


def test_windows_cleanup_rejects_paths_outside_run_roots(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    powershell = shutil.which("powershell.exe")
    if sys.platform != "win32" or powershell is None:
        pytest.skip("Native PowerShell verification requires Windows")
    workspace, temporary = tmp_path / "workspace", tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.dat").write_bytes(b"keep")
    variables = dict(os.environ, CI_PYTHON=sys.executable,
                     GITHUB_WORKSPACE=str(workspace), RUNNER_TEMP=str(temporary),
                     GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1",
                     CLEANUP_NAMES=json.dumps(["../../../outside"]), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command",
                             windows_cleanup_step("93-windows-bundle.yml")["run"]],
                            env=variables, text=True, capture_output=True)
    assert result.returncode != 0
    assert "escaped" in result.stdout + result.stderr
    assert (outside / "keep.dat").read_bytes() == b"keep"


# ── Build caching ───────────────────────────────────────────────────────────

def native_build_steps(name):
    job = next(iter(load(name)["jobs"].values()))
    return job, [step for step in job["steps"] if "build.py compile" in step.get("run", "")]


@pytest.mark.parametrize("name", ["91-build-platform.yml", "92-build-macos.yml"])
def test_native_builds_reuse_stable_folders_so_compiler_caches_hit(name):
    """Compiler caches key on absolute paths; run-scoped folders made every compile a miss."""
    job, builds = native_build_steps(name)
    assert builds
    folders = [job["env"]["SOURCE_DIR"]] + [step["env"].get("SOURCE_DIR", "") for step in builds]
    environments = [step["env"]["UV_PROJECT_ENVIRONMENT"] for step in builds]
    for value in folders + environments:
        assert "run_id" not in value and "run_attempt" not in value and "runner.temp" not in value
    assert all(value.startswith("${{ github.workspace }}/env-") for value in environments)
    assert len(set(environments)) == len(environments)          # one environment per architecture
    for step in job["steps"]:
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"].get("clean", True) is True        # stale outputs are removed
        assert not step.get("name", "").startswith("Clean only run-owned")


def test_self_hosted_builds_keep_caches_local():
    for path in WORKFLOWS.glob("*.yml"):
        for job in load(path.name)["jobs"].values():
            for step in job.get("steps", []):
                uses = step.get("uses", "")
                assert not uses.startswith("actions/cache@"), path.name
                if uses.startswith("astral-sh/setup-uv@"):
                    # Saving the remote uv cache prunes downloaded wheels locally.
                    assert step["with"]["enable-cache"] is False, path.name

