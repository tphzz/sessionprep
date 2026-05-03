from __future__ import annotations

import sys
import types
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from sessionpreplib.daw_processors import protools
from sessionpreplib.models import SessionContext


class _FakeEngine:
    def __init__(self, *args, tracks=None, fail_first_track_list=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.closed = False
        self._tracks = tracks or []
        self._fail_first_track_list = fail_first_track_list
        self.track_list_calls = 0

    def track_list(self):
        self.track_list_calls += 1
        if self._fail_first_track_list and self.track_list_calls == 1:
            raise RuntimeError("track list not ready")
        return self._tracks

    def close(self):
        self.closed = True


@pytest.fixture
def fake_ptsl(monkeypatch):
    module = types.ModuleType("ptsl")
    module.PTSL_pb2 = SimpleNamespace(
        TrackType=SimpleNamespace(RoutingFolder=1, BasicFolder=2)
    )
    engines = []

    def engine_factory(*args, **kwargs):
        engine = _FakeEngine(
            *args,
            tracks=[
                SimpleNamespace(
                    type=1,
                    id="folder-1",
                    name="Routing",
                    index=1,
                    parent_folder_id="",
                )
            ],
            fail_first_track_list=True,
            **kwargs,
        )
        engines.append(engine)
        return engine

    module.Engine = engine_factory
    monkeypatch.setitem(sys.modules, "ptsl", module)
    return engines


def _processor(tmp_path, timeout=60.0):
    processor = protools.ProToolsDawProcessor(
        instance_index=0,
        instance_group="SessionPrep",
        instance_name="Template",
    )
    processor.configure(
        {
            "protools_temp_dir": str(tmp_path),
            "protools_template_create_timeout": timeout,
            "protools_command_delay": 0.01,
        }
    )
    return processor


def _session():
    return SessionContext(tracks=[], config={})


def _template_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    template_file = (
        home
        / "Documents"
        / "Pro Tools"
        / "Session Templates"
        / "SessionPrep"
        / "Template.ptxt"
    )
    template_file.parent.mkdir(parents=True)
    template_file.write_text("template", encoding="utf-8")
    return template_file


def _cache_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "app"
    cache_dir.mkdir()
    monkeypatch.setattr(protools, "get_app_dir", lambda: str(cache_dir), raising=False)

    import sessionpreplib.config as config

    monkeypatch.setattr(config, "get_app_dir", lambda: str(cache_dir))
    return cache_dir / "pt_template_cache.json"


def _allow_fresh_fetch(monkeypatch):
    monkeypatch.setattr(protools.ptslh, "wait_for_host_ready", lambda *a, **k: True)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda _engine: False)
    monkeypatch.setattr(protools.ptslh, "close_session", lambda _engine: None)
    monkeypatch.setattr(
        protools.ptslh,
        "create_session_from_template",
        lambda *args, **kwargs: None,
    )


def test_fetch_passes_template_timeout_and_retries_track_list(
    fake_ptsl, monkeypatch, tmp_path
):
    monkeypatch.setattr(protools.ptslh, "wait_for_host_ready", lambda *a, **k: True)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda _engine: False)
    monkeypatch.setattr(protools.ptslh, "close_session", lambda _engine: None)

    create_calls = []

    def fake_create_session_from_template(*args, **kwargs):
        create_calls.append((args, kwargs))

    monkeypatch.setattr(
        protools.ptslh,
        "create_session_from_template",
        fake_create_session_from_template,
    )

    progress = []
    result = _processor(tmp_path, timeout=60.0).fetch(
        _session(),
        progress_cb=lambda current, total, message: progress.append(message),
    )

    assert create_calls[0][1]["timeout"] == 60.0
    assert fake_ptsl[0].track_list_calls == 2
    assert result.daw_state["protools_0"]["folders"][0]["name"] == "Routing"
    assert progress[-1] == "Fetch complete"


def test_fetch_failure_does_not_emit_fetch_complete(
    fake_ptsl, monkeypatch, tmp_path
):
    monkeypatch.setattr(protools.ptslh, "wait_for_host_ready", lambda *a, **k: True)
    monkeypatch.setattr(protools.ptslh, "is_session_open", lambda _engine: False)
    monkeypatch.setattr(protools.ptslh, "close_session", lambda _engine: None)

    def fail_create_session_from_template(*args, **kwargs):
        raise RuntimeError("template creation failed")

    monkeypatch.setattr(
        protools.ptslh,
        "create_session_from_template",
        fail_create_session_from_template,
    )

    progress = []
    with pytest.raises(RuntimeError, match="template creation failed"):
        _processor(tmp_path, timeout=60.0).fetch(
            _session(),
            progress_cb=lambda current, total, message: progress.append(message),
        )

    assert "Fetch complete" not in progress


def test_fetch_uses_valid_template_cache_without_opening_ptsl(
    fake_ptsl, monkeypatch, tmp_path
):
    template_file = _template_file(tmp_path, monkeypatch)
    cache_file = _cache_file(tmp_path, monkeypatch)
    cache_file.write_text(
        json.dumps(
            {
                "SessionPrep/Template": {
                    "mtime": template_file.stat().st_mtime,
                    "folders": [
                        {
                            "id": "cached-folder",
                            "name": "Cached",
                            "folder_type": "routing",
                            "index": 1,
                            "parent_id": None,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = _processor(tmp_path).fetch(_session())

    assert fake_ptsl == []
    assert result.daw_state["protools_0"]["folders"][0]["name"] == "Cached"


def test_fetch_ignore_cache_opens_ptsl_despite_valid_cache(
    fake_ptsl, monkeypatch, tmp_path
):
    template_file = _template_file(tmp_path, monkeypatch)
    cache_file = _cache_file(tmp_path, monkeypatch)
    cache_file.write_text(
        json.dumps(
            {
                "SessionPrep/Template": {
                    "mtime": template_file.stat().st_mtime,
                    "folders": [
                        {
                            "id": "cached-folder",
                            "name": "Cached",
                            "folder_type": "routing",
                            "index": 1,
                            "parent_id": None,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    _allow_fresh_fetch(monkeypatch)

    result = _processor(tmp_path).fetch(_session(), ignore_cache=True)

    assert len(fake_ptsl) == 1
    assert result.daw_state["protools_0"]["folders"][0]["name"] == "Routing"


def test_fetch_ignore_cache_refreshes_template_cache(
    fake_ptsl, monkeypatch, tmp_path
):
    template_file = _template_file(tmp_path, monkeypatch)
    cache_file = _cache_file(tmp_path, monkeypatch)
    cache_file.write_text(
        json.dumps(
            {
                "SessionPrep/Template": {
                    "mtime": template_file.stat().st_mtime,
                    "folders": [
                        {
                            "id": "old-folder",
                            "name": "Old",
                            "folder_type": "routing",
                            "index": 1,
                            "parent_id": None,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    _allow_fresh_fetch(monkeypatch)

    _processor(tmp_path).fetch(_session(), ignore_cache=True)

    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
    folders = cache_data["SessionPrep/Template"]["folders"]
    assert folders[0]["name"] == "Routing"
    assert cache_data["SessionPrep/Template"]["mtime"] == template_file.stat().st_mtime
