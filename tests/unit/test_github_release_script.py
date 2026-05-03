from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging" / "github" / "publish-draft-release.sh"


def run_script(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_allowlist(path: Path, *patterns: str) -> Path:
    path.write_text("\n".join(patterns) + "\n", encoding="utf-8")
    return path


def base_args(tmp_path: Path, artifact_root: Path, allowlist: Path) -> list[str]:
    return [
        "--artifact-root", str(artifact_root),
        "--mode", "branch",
        "--ref-name", "feature/release-test",
        "--target", "deadbeef",
        "--repo", "owner/repo",
        "--allowlist", str(allowlist),
        "--staging-dir", str(tmp_path / "release-assets"),
        "--dry-run",
    ]


def test_release_script_selects_allowed_files_from_dirs_and_zips(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(
        tmp_path / "allowlist.txt",
        "example-*-macos-arm64.dmg",
        "example-*-linux-x64.tar.gz",
    )

    direct_dir = artifact_root / "example-linux-x64"
    direct_dir.mkdir()
    (direct_dir / "example-1.2.3-linux-x64.tar.gz").write_bytes(b"tarball")
    (direct_dir / "example-linux-x64").write_bytes(b"loose binary")

    zip_path = artifact_root / "example-macos-arm64.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dist/example-1.2.3-macos-arm64.dmg", b"dmg")
        zf.writestr("dist/example-1.2.3-macos-arm64.build", b"build")

    result = run_script(tmp_path, *base_args(tmp_path, artifact_root, allowlist))

    assert result.returncode == 0, result.stderr + result.stdout
    staging = tmp_path / "release-assets"
    assert (staging / "example-1.2.3-linux-x64.tar.gz").read_bytes() == b"tarball"
    assert (staging / "example-1.2.3-macos-arm64.dmg").read_bytes() == b"dmg"
    assert not (staging / "example-linux-x64").exists()
    assert "Dry run complete" in result.stdout


def test_release_script_fails_on_duplicate_asset_names(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")

    first = artifact_root / "first"
    second = artifact_root / "second"
    first.mkdir()
    second.mkdir()
    (first / "example-1.2.3-macos-arm64.dmg").write_bytes(b"first")
    (second / "example-1.2.3-macos-arm64.dmg").write_bytes(b"second")

    result = run_script(tmp_path, *base_args(tmp_path, artifact_root, allowlist))

    assert result.returncode != 0
    assert "Duplicate release asset filename" in result.stderr


def test_release_script_fails_when_no_assets_match(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")
    (artifact_root / "example-linux-x64").write_bytes(b"loose binary")

    result = run_script(tmp_path, *base_args(tmp_path, artifact_root, allowlist))

    assert result.returncode != 0
    assert "No release assets matched" in result.stderr


def test_release_script_rejects_invalid_mode(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")

    args = base_args(tmp_path, artifact_root, allowlist)
    args[args.index("--mode") + 1] = "invalid"

    result = run_script(tmp_path, *args)

    assert result.returncode != 0
    assert "--mode must be 'branch' or 'tag'" in result.stderr


def test_release_script_rejects_invalid_download_run_id(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")

    args = base_args(tmp_path, artifact_root, allowlist)
    args[args.index("--artifact-root") : args.index("--artifact-root") + 2] = []
    args.extend(["--download-run-id", "not-a-number"])

    result = run_script(tmp_path, *args)

    assert result.returncode != 0
    assert "--download-run-id must be numeric" in result.stderr


def test_release_script_can_download_artifacts_before_dry_run(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    gh_log = tmp_path / "gh.log"
    fake_gh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {gh_log}
if [[ "$1 $2" == "auth status" ]]; then
  exit 0
fi
if [[ "$1 $2 $3" == "run download 12345" ]]; then
  out_dir=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir)
        out_dir="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  mkdir -p "$out_dir/example-linux-x64"
  printf artifact > "$out_dir/example-linux-x64/example-1.2.3-linux-x64.tar.gz"
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    artifact_root = tmp_path / "downloaded-artifacts"
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*-linux-x64.tar.gz")
    args = base_args(tmp_path, artifact_root, allowlist)
    args.extend(["--download-run-id", "12345"])

    result = run_script(
        tmp_path,
        *args,
        extra_env={"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "release-assets" / "example-1.2.3-linux-x64.tar.gz").read_bytes() == b"artifact"
    assert "--pattern" in gh_log.read_text(encoding="utf-8")


def test_release_script_rejects_staging_dir_inside_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")

    args = base_args(tmp_path, artifact_root, allowlist)
    args[args.index("--staging-dir") + 1] = str(artifact_root / "release-assets")

    result = run_script(tmp_path, *args)

    assert result.returncode != 0
    assert "--staging-dir must not be inside --artifact-root" in result.stderr


def test_release_script_fails_on_corrupt_zip(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*.dmg")
    (artifact_root / "example-macos-arm64.zip").write_bytes(b"not a zip")

    result = run_script(tmp_path, *base_args(tmp_path, artifact_root, allowlist))

    assert result.returncode != 0
    assert "Could not read ZIP artifact" in result.stderr
