"""Exercise the source launcher without starting uv or a Qt event loop."""

import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / "sessionprep-gui.py"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    gui = ModuleType("sessionprepgui")
    gui.main = Mock()
    monkeypatch.setitem(sys.modules, "sessionprepgui", gui)
    monkeypatch.setenv("UV", str(tmp_path / "uv executable"))
    monkeypatch.delenv("_SESSIONPREP_GUI_RELAUNCHED", raising=False)
    find_spec = importlib.util.find_spec

    def launch(*args, missing=None, globals_=None, script=LAUNCHER):
        def resolve_module(name, package=None):
            if name in ("PySide6", "sounddevice"):
                return None if name == missing else object()
            return find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", resolve_module)
        monkeypatch.setattr(sys, "argv", [str(script), *args])
        runpy.run_path(str(script), run_name="__main__", init_globals=globals_)

    return launch, gui.main


@pytest.mark.parametrize("missing", ["PySide6", "sounddevice"])
def test_missing_gui_dependency_relaunches_with_the_extra(
    launcher, monkeypatch, missing
):
    launch, main = launcher
    spawn = Mock(return_value=subprocess.CompletedProcess([], 17))
    monkeypatch.setattr(subprocess, "run", spawn)
    monkeypatch.setenv("SP_LOG_LEVEL", "DEBUG")

    with pytest.raises(SystemExit) as exited:
        launch("argument with spaces", missing=missing)

    assert exited.value.code == 17
    spawn.assert_called_once()
    command = spawn.call_args.args[0]
    assert command == [
        os.environ["UV"], "run", "--project", str(LAUNCHER.parent),
        "--active", "--extra", "gui", "python", str(LAUNCHER),
        "argument with spaces",
    ]
    child_env = spawn.call_args.kwargs["env"]
    assert child_env["_SESSIONPREP_GUI_RELAUNCHED"] == "1"
    assert child_env["SP_LOG_LEVEL"] == "DEBUG"
    main.assert_not_called()


def test_prepared_source_environment_starts_the_gui_directly(launcher, monkeypatch):
    launch, main = launcher
    spawn = Mock()
    monkeypatch.setattr(subprocess, "run", spawn)

    launch()

    main.assert_called_once_with()
    spawn.assert_not_called()


@pytest.mark.parametrize("compiled", ["nuitka", "frozen"])
def test_packaged_gui_bypasses_dependency_bootstrap(launcher, monkeypatch, compiled):
    launch, main = launcher
    spawn = Mock()
    monkeypatch.setattr(subprocess, "run", spawn)
    globals_ = None
    if compiled == "nuitka":
        globals_ = {"__compiled__": object()}
    else:
        monkeypatch.setattr(sys, "frozen", True, raising=False)

    launch(missing="PySide6", globals_=globals_)

    main.assert_called_once_with()
    spawn.assert_not_called()


@pytest.mark.parametrize("reason", ["outside_uv", "already_relaunched", "missing_project"])
def test_missing_dependency_has_install_instructions_when_no_retry_is_available(
    launcher, monkeypatch, tmp_path, reason
):
    launch, main = launcher
    spawn = Mock()
    monkeypatch.setattr(subprocess, "run", spawn)
    script = LAUNCHER
    if reason == "outside_uv":
        monkeypatch.delenv("UV")
    elif reason == "already_relaunched":
        monkeypatch.setenv("_SESSIONPREP_GUI_RELAUNCHED", "1")
    else:
        script = tmp_path / LAUNCHER.name
        script.write_bytes(LAUNCHER.read_bytes())

    with pytest.raises(SystemExit, match="uv run --extra gui"):
        launch(missing="PySide6", script=script)

    spawn.assert_not_called()
    main.assert_not_called()


def test_uv_start_failure_is_reported(launcher, monkeypatch):
    launch, main = launcher
    spawn = Mock(side_effect=FileNotFoundError("uv executable is missing"))
    monkeypatch.setattr(subprocess, "run", spawn)

    with pytest.raises(SystemExit, match="Could not start uv"):
        launch(missing="PySide6")

    spawn.assert_called_once()
    main.assert_not_called()
