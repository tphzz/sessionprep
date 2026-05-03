from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging" / "github" / "publish-draft-release.sh"


def run_script(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
    script: Path = SCRIPT,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(script), *args],
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


def write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)
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


def publish_args(
    tmp_path: Path,
    artifact_root: Path,
    allowlist: Path,
    *,
    mode: str = "branch",
    ref_name: str = "feature/release-test",
    release_title: str | None = None,
) -> list[str]:
    args = [
        "--artifact-root", str(artifact_root),
        "--mode", mode,
        "--ref-name", ref_name,
        "--target", "deadbeef",
        "--repo", "owner/repo",
        "--allowlist", str(allowlist),
        "--staging-dir", str(tmp_path / "release-assets"),
    ]
    if release_title is not None:
        args.extend(["--release-title", release_title])
    return args


def write_fake_curl(bin_dir: Path, log_path: Path) -> None:
    write_executable(
        bin_dir / "curl",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> {log_path}
        exit 0
        """,
    )


def test_release_script_defaults_to_adjacent_allowlist(tmp_path):
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    script = tool_dir / "publish-draft-release.sh"
    shutil.copy2(SCRIPT, script)
    write_allowlist(tool_dir / "release-assets.allowlist", "example-*.pkg")

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3.pkg").write_bytes(b"package payload")

    args = [
        "--artifact-root", str(artifact_root),
        "--mode", "branch",
        "--ref-name", "feature/release-test",
        "--target", "deadbeef",
        "--repo", "owner/repo",
        "--staging-dir", str(tmp_path / "release-assets"),
        "--dry-run",
    ]

    result = run_script(tmp_path, *args, script=script)

    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "release-assets" / "example-1.2.3.pkg").read_bytes() == b"package payload"


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


def test_release_script_replaces_branch_drafts_and_uploads_to_new_release_id(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh_log = tmp_path / "gh.log"
    curl_log = tmp_path / "curl.log"
    write_fake_curl(fake_bin, curl_log)
    write_executable(
        fake_bin / "gh",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> {gh_log}
        if [[ "$1 $2" == "auth status" ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/releases?per_page=100"* ]]; then
          printf '101\\ttrue\\tSessionPrep branch build: feature/release-test\\tuntagged-old\\n'
          printf '102\\ttrue\\tOther title\\tbranch-build-feature-release-test\\n'
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"-X DELETE repos/owner/repo/releases/"* ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"-X DELETE repos/owner/repo/git/refs/tags/branch-build-feature-release-test"* ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"-X POST repos/owner/repo/releases"* ]]; then
          printf '999\\thttps://github.com/owner/repo/releases/tag/untagged-new\\thttps://uploads.github.com/repos/owner/repo/releases/999/assets{{?name,label}}\\n'
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/releases/999/assets?per_page=100"* ]]; then
          printf 'example-1.2.3%%2Bbuild.pkg\\n'
          printf 'example-1.2.3.dmg\\n'
          exit 0
        fi
        exit 2
        """,
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3+build.pkg").write_bytes(b"pkg")
    (artifact_root / "example-1.2.3.dmg").write_bytes(b"dmg")
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*")

    result = run_script(
        tmp_path,
        *publish_args(
            tmp_path,
            artifact_root,
            allowlist,
            release_title="SessionPrep branch build: feature/release-test",
        ),
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "token",
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    gh_output = gh_log.read_text(encoding="utf-8")
    curl_output = curl_log.read_text(encoding="utf-8")
    assert "repos/owner/repo/releases/101" in gh_output
    assert "repos/owner/repo/releases/102" in gh_output
    assert "repos/owner/repo/releases/999/assets" in curl_output
    assert "example-1.2.3%2Bbuild.pkg" in curl_output
    assert "Draft release ready: id=999" in result.stdout


def test_release_script_refuses_matching_published_branch_release(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh_log = tmp_path / "gh.log"
    write_fake_curl(fake_bin, tmp_path / "curl.log")
    write_executable(
        fake_bin / "gh",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> {gh_log}
        if [[ "$1 $2" == "auth status" ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/releases?per_page=100"* ]]; then
          printf '101\\tfalse\\tSessionPrep branch build: feature/release-test\\tbranch-build-feature-release-test\\n'
          exit 0
        fi
        exit 2
        """,
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3.pkg").write_bytes(b"pkg")
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*")

    result = run_script(
        tmp_path,
        *publish_args(
            tmp_path,
            artifact_root,
            allowlist,
            release_title="SessionPrep branch build: feature/release-test",
        ),
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "token",
        },
    )

    assert result.returncode != 0
    assert "Refusing to replace published branch release" in result.stderr


def test_release_script_requires_existing_tag_in_tag_mode(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_curl(fake_bin, tmp_path / "curl.log")
    write_executable(
        fake_bin / "gh",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$1 $2" == "auth status" ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/git/matching-refs/tags/v1.2.3"* ]]; then
          printf '0\\n'
          exit 0
        fi
        exit 2
        """,
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3.pkg").write_bytes(b"pkg")
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*")

    result = run_script(
        tmp_path,
        *publish_args(tmp_path, artifact_root, allowlist, mode="tag", ref_name="v1.2.3"),
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "token",
        },
    )

    assert result.returncode != 0
    assert "Tag mode requires an existing Git tag: v1.2.3" in result.stderr


def test_release_script_fails_when_uploaded_asset_count_does_not_match(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_curl(fake_bin, tmp_path / "curl.log")
    write_executable(
        fake_bin / "gh",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$1 $2" == "auth status" ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/releases?per_page=100"* ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"-X DELETE repos/owner/repo/git/refs/tags/branch-build-feature-release-test"* ]]; then
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"-X POST repos/owner/repo/releases"* ]]; then
          printf '999\\thttps://github.com/owner/repo/releases/tag/untagged-new\\thttps://uploads.github.com/repos/owner/repo/releases/999/assets{?name,label}\\n'
          exit 0
        fi
        if [[ "$1" == "api" && "$*" == *"repos/owner/repo/releases/999/assets?per_page=100"* ]]; then
          printf 'example-1.2.3.pkg\\n'
          exit 0
        fi
        exit 2
        """,
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3.pkg").write_bytes(b"pkg")
    (artifact_root / "example-1.2.3.dmg").write_bytes(b"dmg")
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*")

    result = run_script(
        tmp_path,
        *publish_args(tmp_path, artifact_root, allowlist),
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "token",
        },
    )

    assert result.returncode != 0
    assert "Uploaded asset count mismatch for release 999: expected 2, got 1" in result.stderr


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
