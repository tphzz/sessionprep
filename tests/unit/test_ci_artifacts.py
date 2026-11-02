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
