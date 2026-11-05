from pathlib import Path
import os
import shutil
import subprocess
import platform
import re
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def steps():
    workflow = yaml.safe_load((ROOT / ".github/workflows/92-build-macos.yml").read_text(encoding="utf-8"))
    return workflow["jobs"]["macos"]["steps"]


def resolver():
    return next(step for step in steps() if step.get("id") == "macos-python")


def test_macos_python_setup_is_user_managed_and_precedes_builds():
    workflow_steps = steps()
    assert not any(step.get("uses", "").startswith("actions/setup-python@") for step in workflow_steps)
    resolve = resolver()
    assert resolve["shell"] == "bash"
    assert "uv python install --no-bin" in resolve["run"]
    assert "sudo" not in resolve["run"]
    assert workflow_steps.index(resolve) > next(i for i, step in enumerate(workflow_steps)
                                               if step.get("name") == "Set up uv")
    for architecture in ("arm64", "x64"):
        build = next(step for step in workflow_steps
                     if step.get("name") == f"Build and package macOS {architecture}")
        assert f"steps.macos-python.outputs.{architecture}-python" in build["env"]["PYTHON_INSTALL"]
        assert 'python_path="$PYTHON_INSTALL"' in build["run"]
        assert 'uv run --frozen --python "$python_path"' in build["run"]
        assert workflow_steps.index(resolve) < workflow_steps.index(build)


@pytest.mark.parametrize("target,machine,version,system,implementation,accepts", [
    ("arm64", "arm64", (3, 13, 17), "Darwin", "CPython", True),
    ("x64", "x86_64", (3, 13, 17), "Darwin", "CPython", True),
    ("arm64", "x86_64", (3, 13, 17), "Darwin", "CPython", False),
    ("x64", "arm64", (3, 13, 17), "Darwin", "CPython", False),
    ("arm64", "arm64", (3, 12, 9), "Darwin", "CPython", False),
    ("arm64", "arm64", (3, 14, 0), "Darwin", "CPython", False),
    ("arm64", "arm64", (3, 13, 17), "Linux", "CPython", False),
    ("arm64", "arm64", (3, 13, 17), "Darwin", "PyPy", False),
])
def test_macos_python_guard_validates_actual_interpreter(monkeypatch, target, machine, version,
                                                         system, implementation, accepts):
    match = re.search(r"<<'PY'\n(.*?)\nPY", resolver()["run"], re.DOTALL)
    assert match is not None
    monkeypatch.setattr(sys, "argv", ["-", target])
    monkeypatch.setattr(sys, "version_info", version)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(platform, "python_implementation", lambda: implementation)
    if accepts:
        exec(compile(match[1], "macos-python-guard", "exec"), {})
    else:
        with pytest.raises(SystemExit, match="Python 3.13"):
            exec(compile(match[1], "macos-python-guard", "exec"), {})


@pytest.fixture
def bash():
    candidates = [Path(os.environ.get("PROGRAMFILES", "")) / "Git/bin/bash.exe"] if sys.platform == "win32" else []
    executable = next((str(path) for path in candidates if path.is_file()), shutil.which("bash"))
    if executable is None:
        pytest.skip("Bash is required for workflow execution tests")
    return executable


@pytest.mark.parametrize("selection,scenario", [
    ("both", "managed"), ("arm64", "managed"), ("x64", "managed"),
    ("both", "per-role"), ("arm64", "legacy"),
    ("both", "wrong-legacy"), ("x64", "wrong-role"),
    ("x64", "missing-override"), ("both", "intel-install-fails"),
    ("invalid", "managed"),
])
def test_macos_resolver_execution(tmp_path, bash, selection, scenario):
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls.log"
    output = tmp_path / "output.env"
    environment_file = tmp_path / "job.env"
    interpreters = {}
    for role in ("arm64", "x64"):
        interpreter = tmp_path / f"Python {role} [literal]" / "python3.13"
        interpreter.parent.mkdir()
        interpreter.write_text(f'''#!/bin/bash
cat >/dev/null
printf 'guard:{role}:%s\n' "$2" >> "$CALLS"
[[ "$2" == "{role}" ]]
''', encoding="utf-8", newline="\n")
        interpreter.chmod(0o755)
        interpreters[role] = interpreter.as_posix()
    fake_uv = tools / "uv"
    fake_uv.write_text(r'''#!/bin/bash
set -euo pipefail
printf 'uv:%s\n' "$*" >> "$CALLS"
request="${!#}"
case "$2" in
  install)
    if [[ "$SCENARIO" == "intel-install-fails" && "$request" == *x86_64* ]]; then exit 3; fi ;;
  find)
    case "$request" in
      *macos-aarch64-none) printf '%s\n' "$ARM_PYTHON" ;;
      *macos-x86_64-none) printf '%s\n' "$INTEL_PYTHON" ;;
      *) [[ -f "$request" ]] && printf '%s\n' "$request" ;;
    esac ;;
  *) exit 4 ;;
esac
''', encoding="utf-8", newline="\n")
    fake_uv.chmod(0o755)
    variables = dict(os.environ, PATH=tools.as_posix() + os.pathsep + os.environ["PATH"],
                     SELECTED_ARCH=selection, SCENARIO=scenario, CALLS=calls.as_posix(),
                     GITHUB_OUTPUT=output.as_posix(), GITHUB_ENV=environment_file.as_posix(),
                     ARM_PYTHON=interpreters["arm64"], INTEL_PYTHON=interpreters["x64"])
    for name in ("CI_PYTHON", "CI_PYTHON_MACOS_ARM64", "CI_PYTHON_MACOS_X64"):
        variables.pop(name, None)
    if scenario == "per-role":
        variables.update(CI_PYTHON_MACOS_ARM64=interpreters["arm64"],
                         CI_PYTHON_MACOS_X64=interpreters["x64"], CI_PYTHON="unused-invalid")
    elif scenario in ("legacy", "wrong-legacy"):
        variables["CI_PYTHON"] = interpreters["arm64"]
    elif scenario == "wrong-role":
        variables["CI_PYTHON_MACOS_X64"] = interpreters["arm64"]
    elif scenario == "missing-override":
        variables["CI_PYTHON_MACOS_X64"] = (tmp_path / "missing Python").as_posix()
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", resolver()["run"]],
                            cwd=tmp_path, env=variables, text=True, capture_output=True)
    succeeds = scenario in ("managed", "per-role", "legacy") and selection != "invalid"
    assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
    resolved = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
    log = calls.read_text().splitlines() if calls.exists() else []
    if succeeds:
        selected = ["arm64", "x64"] if selection == "both" else [selection]
        for role in ("arm64", "x64"):
            assert (role + "-python" in resolved) is (role in selected)
            if role in selected:
                assert resolved[role + "-python"] == interpreters[role]
        assert resolved["bootstrap-python"] == interpreters[selected[0]]
        assert "MACOS_BOOTSTRAP_PYTHON=" + interpreters[selected[0]] in environment_file.read_text()
        installs = [line for line in log if line.startswith("uv:python install")]
        assert len(installs) == (len(selected) if scenario == "managed" else 0)
    else:
        assert "x64-python" not in resolved
        if scenario in ("wrong-legacy", "intel-install-fails"):
            assert resolved["arm64-python"] == interpreters["arm64"]
            assert "MACOS_BOOTSTRAP_PYTHON=" + interpreters["arm64"] in environment_file.read_text()
        if scenario != "intel-install-fails":
            assert not any(line.startswith("uv:python install") for line in log)


@pytest.mark.parametrize("version,callback", [((3, 9, 6), "onerror"), ((3, 13, 17), "onexc")])
def test_macos_cleanup_keeps_fallback_python_compatibility(tmp_path, monkeypatch, version, callback):
    script = next(step["run"] for step in steps() if step.get("name") == "Clean only run-owned workspace")
    assert '${MACOS_BOOTSTRAP_PYTHON:-python3}' in script
    code = re.search(r"<<'PY'\n(.*?)\nPY", script, re.DOTALL)[1]
    workspace, temporary = tmp_path / "workspace", tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()
    source = workspace / "src-123-1-macos"
    source.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("RUNNER_TEMP", str(temporary))
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("CLEANUP_NAMES", '["macos"]')
    monkeypatch.setattr(sys, "version_info", version)
    calls = []
    def remove(path, **kwargs):
        calls.append((path, kwargs))
        assert set(kwargs) == {callback}
    monkeypatch.setattr(shutil, "rmtree", remove)
    exec(compile(code, "macos-cleanup", "exec"), {})
    assert [path for path, _ in calls] == [source.resolve()]
