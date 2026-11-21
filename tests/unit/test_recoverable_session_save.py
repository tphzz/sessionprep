"""An interrupted Save Session must leave its previous document usable."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from sessionprepgui.session.io import load_session, save_session


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("failure", ["encoding", "publication"])
def test_save_failure_and_retry(tmp_path, monkeypatch, failure, prior):
    target = tmp_path / "mix.spsession"
    first = {"source_dir": str(tmp_path), "project_name": "first"}
    if prior:
        save_session(str(target), first)
        original = target.read_bytes()

    def short_write(_value, stream, **_options):
        stream.write("{")
        raise OSError("partial write")

    def locked_target(_temporary, destination):
        assert Path(destination) == target
        raise PermissionError("destination in use")

    changed = dict(first, project_name="second")
    with monkeypatch.context() as fault:
        if failure == "encoding":
            fault.setattr(json, "dump", short_write)
        else:
            fault.setattr(os, "replace", locked_target)
        with pytest.raises(OSError):
            save_session(str(target), changed)

    assert list(tmp_path.iterdir()) == ([target] if prior else [])
    if prior:
        assert target.read_bytes() == original
        assert load_session(str(target))["project_name"] == "first"
    save_session(str(target), changed)
    assert load_session(str(target))["project_name"] == "second"
