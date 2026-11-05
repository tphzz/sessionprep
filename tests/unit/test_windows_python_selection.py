import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "packaging/github/windows_python.ps1"


@pytest.mark.parametrize("scenario,target", [
    ("machine", "win-arm64"),
    ("machine", "win-x64"),
    ("off-path", "win-arm64"),
    ("native-root", "win-arm64"),
    ("user", "win-arm64"),
    ("second-path", "win-arm64"),
    ("first-path", "win-x64"),
    ("explicit", "win-arm64"),
    ("wrong-override", "win-arm64"),
    ("missing-override", "win-arm64"),
    ("no-match", "win-arm64"),
])
def test_windows_selection_checks_candidates_without_replacing_explicit_override(tmp_path, scenario, target):
    powershell = shutil.which("powershell.exe")
    if sys.platform != "win32" or powershell is None:
        pytest.skip("Native interpreter discovery requires Windows PowerShell")
    wrong = tmp_path / "old Python" / "python.exe"
    second = tmp_path / "other Python" / "python.exe"
    folder = "Python313-arm64" if target == "win-arm64" else "Python313"
    machine = tmp_path / "Program Files [literal]" / folder / "python.exe"
    user = tmp_path / "Local AppData" / "Programs" / "Python" / folder / "python.exe"
    explicit = tmp_path / "custom Python" / "python.exe"
    for candidate in (wrong, second, machine, user, explicit):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
    expected = {"machine": machine, "user": user, "second-path": second,
                "first-path": wrong, "explicit": explicit}.get(scenario, machine)
    variables = dict(os.environ, PATH=os.pathsep.join(map(str, (wrong.parent, second.parent))),
                     FIXTURE_PROGRAMFILES=str(machine.parent.parent), FIXTURE_PROGRAMW6432="",
                     LOCALAPPDATA=str(user.parents[3]), TARGET=target, HELPER=str(HELPER),
                     EXPECTED_PYTHON=str(expected))
    variables.pop("CI_PYTHON", None)
    if scenario == "off-path":
        variables["PATH"] = str(tmp_path / "empty PATH")
    elif scenario == "native-root":
        variables["FIXTURE_PROGRAMW6432"] = variables["FIXTURE_PROGRAMFILES"]
        variables["FIXTURE_PROGRAMFILES"] = str(tmp_path / "emulated Program Files")
    if scenario in ("explicit", "wrong-override"):
        variables["CI_PYTHON"] = str(explicit)
    elif scenario == "missing-override":
        variables["CI_PYTHON"] = str(tmp_path / "missing" / "python.exe")
    script = r'''$ErrorActionPreference = 'Stop'
# PowerShell normalizes ProgramFiles at startup; isolate roots afterward.
$env:ProgramFiles = $env:FIXTURE_PROGRAMFILES
$env:ProgramW6432 = $env:FIXTURE_PROGRAMW6432
. $env:HELPER
# The installed executable boundary is mocked; candidate discovery and override
# precedence use the real filesystem and PATH. Native probes are tested below.
function Test-SessionPrepPython($Candidate, $Target) {
    Write-Host "Checked:$Candidate"
    if ($env:SCENARIO -eq 'no-match') { return $false }
    return $Candidate -eq $env:EXPECTED_PYTHON
}
$selected = Find-SessionPrepPython -Target $env:TARGET
Write-Host "Selected:$selected"
'''
    variables["SCENARIO"] = scenario
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                            env=variables, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if scenario in ("wrong-override", "missing-override", "no-match"):
        assert result.returncode != 0, output
        assert "Selected:" not in output
        assert "CI_PYTHON" in output
        if scenario != "no-match":
            assert "Checked:" + str(machine) not in output
    else:
        assert result.returncode == 0, output
        assert "Selected:" + str(expected) in output
        if scenario == "explicit":
            assert "Checked:" + str(wrong) not in output
        elif scenario not in ("first-path", "off-path"):
            assert "Checked:" + str(wrong) in output


def test_windows_discovery_reuses_native_python_guard():
    assert "from build_conf import check_windows_python" in HELPER.read_text(encoding="utf-8")
