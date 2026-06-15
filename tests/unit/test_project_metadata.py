from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gui_extra_does_not_require_git():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gui_dependencies = metadata["project"]["optional-dependencies"]["gui"]

    assert all("git+" not in dependency for dependency in gui_dependencies)
