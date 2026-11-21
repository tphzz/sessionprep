"""Phase 01 hygiene and contract tests: config defaults, DAWproject purity, CLI, license, build."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib

import yaml

import build_nuitka
from sessionpreplib import __version__
from sessionpreplib.config import (
    ANALYSIS_PARAMS,
    PRESENTATION_PARAMS,
    REPAIR_PARAMS,
    default_config,
)
from sessionpreplib.daw_processors.dawproject import DawProjectDawProcessor
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext
from sessionpreplib.processors import default_processors

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_matches_all_component_param_specs():
    """Every detector/processor/shared ParamSpec default must appear in default_config()."""
    cfg = default_config()
    for spec in (*ANALYSIS_PARAMS, *PRESENTATION_PARAMS, *REPAIR_PARAMS):
        assert spec.key in cfg, f"Missing shared ParamSpec key {spec.key!r} in default_config()"
        assert cfg[spec.key] == spec.default

    for component in (*default_detectors(), *default_processors()):
        for spec in component.config_params():
            assert spec.key in cfg, f"Missing {component.id}.{spec.key!r} in default_config()"
            assert cfg[spec.key] == spec.default


def test_components_configure_identically_for_empty_and_default_config():
    """Fallback defaults inside configure() must agree with default_config()."""
    cfg = default_config()
    for factory in (default_detectors, default_processors):
        empty_instances = factory()
        default_instances = factory()
        for empty_comp, default_comp in zip(empty_instances, default_instances, strict=True):
            empty_comp.configure({})
            default_comp.configure(cfg)
            assert empty_comp.__dict__ == default_comp.__dict__, (
                f"{empty_comp.id}.configure({{}}) diverged from configure(default_config())"
            )


def test_dawproject_default_output_path_is_pure_and_sanitized():
    """DawProjectDawProcessor computes output paths without Qt and sanitizes unsafe characters."""
    proc = DawProjectDawProcessor(instance_index=0, template_name="Rock/Mix:v1", template_path="")
    session = SessionContext(
        tracks=[],
        config={"_source_dir": "/tmp/song", "_output_folder": "sp_02_prepared"},
        project_name="My:Song/Take*1",
    )
    path = proc.default_output_path(session)
    assert path == os.path.join("/tmp/song", "sp_02_prepared", "My_Song_Take_1.dawproject")

    fallback = SessionContext(tracks=[], config={}, project_name="")
    assert proc.default_output_path(fallback) == "Rock_Mix_v1.dawproject"


def test_sessionpreplib_imports_without_pyside6():
    """Importing sessionpreplib and DAW processors must work when PySide6 is absent."""
    code = (
        "import sys; sys.modules['PySide6'] = None; "
        "import sessionpreplib; "
        "from sessionpreplib.daw_processors.dawproject import DawProjectDawProcessor; "
        "from sessionpreplib.models import SessionContext; "
        "p = DawProjectDawProcessor(instance_index=0, template_name='Demo'); "
        "assert p.default_output_path(SessionContext([], {})) == 'Demo.dawproject'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


def test_license_metadata_agrees_across_manifests():
    """pyproject.toml and Linux packaging metadata must declare the same SPDX license."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    nfpm = yaml.safe_load((REPO_ROOT / "packaging" / "linux" / "nfpm.yaml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"] == "LGPL-3.0-or-later"
    assert nfpm["license"] == "LGPL-3.0-or-later"


def test_cli_version_emits_clean_single_line_without_log_banner():
    """`sessionprep.py --version` must not print startup logging banners."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "sessionprep.py"), "--version"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert proc.stdout.strip() == f"sessionprep {__version__}"
    assert "[INFO" not in proc.stdout
    assert "[INFO" not in proc.stderr


def test_build_nuitka_has_no_unverified_rps_download():
    """Unused external RPS binary downloader must not exist in build_nuitka."""
    assert not hasattr(build_nuitka, "fetch_and_bundle_rps")
    assert not hasattr(build_nuitka, "RPS_VERSION")
