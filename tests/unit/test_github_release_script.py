from __future__ import annotations

import json
import shlex
import os
import shutil
import sys
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
    command = [str(script), *args]
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash.exe")
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir(exist_ok=True)
        for tool in ("gh", "curl"):
            if not (fake_bin / tool).exists():
                write_executable(fake_bin / tool, "#!/bin/bash\necho Unexpected network call >&2\nexit 97\n")
        command = [bash, "-c", 'export PATH="$(/usr/bin/cygpath -u "$1"):$PATH"; shift; source "$@"', "publisher-tests",
                   fake_bin.as_posix(), script.as_posix(),
                   *(arg.replace("\\", "/") if len(arg) > 2 and arg[1] == ":" else arg for arg in args)]
        env.setdefault("SESSIONPREP_RELEASE_PYTHON", Path(sys.executable).as_posix())
    return subprocess.run(
        command,
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
    content = textwrap.dedent(content)
    if os.name == "nt":
        content = content.replace(str(path.parent.parent), path.parent.parent.as_posix())
    path.write_text(content, encoding="utf-8", newline="\n")
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


def test_release_script_defaults_to_adjacent_allowlist(tmp_path):
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    script = tool_dir / "publish-draft-release.sh"
    shutil.copy2(SCRIPT, script)
    shutil.copy2(SCRIPT.with_name("release_policy.py"), tool_dir / "release_policy.py")
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
    write_executable(fake_gh,
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{gh_log.as_posix()}"
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
    )

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
    prior = [owned_release(101), owned_release(102),
             dict(id=103, draft=True, tag_name="1.2.3", body="another branch")]
    state, environment = write_release_server(tmp_path, prior)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "example-1.2.3+build.pkg").write_bytes(b"pkg")
    (artifact_root / "example-1.2.3.dmg").write_bytes(b"dmg")
    allowlist = write_allowlist(tmp_path / "allowlist.txt", "example-*")
    result = run_script(tmp_path, *publish_args(tmp_path, artifact_root, allowlist), extra_env=environment)
    assert result.returncode == 0, result.stderr + result.stdout
    final = json.loads(state.read_text())
    assert final["deleted"] == [101, 102]
    assert final["releases"]["999"]["draft"] is True
    assert final["releases"]["999"]["prerelease"] is True
    assert final["releases"]["999"]["tag_name"].startswith("branch-build-feature-release-test-")
    assert {a["name"] for a in final["assets"]} == {"example-1.2.3+build.pkg", "example-1.2.3.dmg"}
    assert "Draft release ready: id=999" in result.stdout


def test_release_script_refuses_matching_published_branch_release(tmp_path):
    prior = owned_release(101)
    prior["draft"] = False
    state, environment = write_release_server(tmp_path, [prior])
    artifact_root, allowlist = publication_fixture(tmp_path)
    result = run_script(tmp_path, *publish_args(tmp_path, artifact_root, allowlist), extra_env=environment)
    assert result.returncode != 0 and "published branch release" in result.stderr
    assert not json.loads(state.read_text())["deleted"]


def test_release_script_requires_existing_tag_in_tag_mode(tmp_path):
    state, environment = write_release_server(tmp_path, [])
    artifact_root, allowlist = publication_fixture(tmp_path)
    result = run_script(tmp_path, *publish_args(tmp_path, artifact_root, allowlist, mode="tag", ref_name="v1.2.3"), extra_env=environment)
    assert result.returncode != 0 and "Tag missing" in result.stderr
    assert not json.loads(state.read_text())["deleted"]


def test_release_script_fails_when_uploaded_asset_count_does_not_match(tmp_path):
    state, environment = write_release_server(tmp_path, [owned_release(101)], mismatch=True)
    artifact_root, allowlist = publication_fixture(tmp_path)
    result = run_script(tmp_path, *publish_args(tmp_path, artifact_root, allowlist), extra_env=environment)
    assert result.returncode != 0 and "count/name/size mismatch" in result.stderr
    final = json.loads(state.read_text())
    assert final["deleted"] == [999] and "101" in final["releases"]


def test_release_create_permission_failure_keeps_prior_draft(tmp_path):
    state, environment = write_release_server(tmp_path, [owned_release(101)], fail_create=True)
    artifact_root, allowlist = publication_fixture(tmp_path)
    result = run_script(tmp_path, *publish_args(tmp_path, artifact_root, allowlist), extra_env=environment)
    assert result.returncode != 0 and "403" in result.stderr
    assert not json.loads(state.read_text())["deleted"]


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


def owned_release(identifier):
    return dict(id=identifier, draft=True, tag_name="old-"+str(identifier),
                body="<!-- github-draft-release-key: branch:feature/release-test -->")


def publication_fixture(tmp_path):
    artifact_root=tmp_path/"artifacts"
    artifact_root.mkdir()
    (artifact_root/"example-1.2.3.pkg").write_bytes(b"pkg")
    (artifact_root/"example-1.2.3.dmg").write_bytes(b"dmg")
    return artifact_root,write_allowlist(tmp_path/"allowlist.txt","example-*")


def write_release_server(tmp_path, releases, **settings):
    fake_bin=tmp_path/"bin"
    fake_bin.mkdir()
    state=tmp_path/"api-state.json"
    state.write_text(json.dumps(dict(releases={str(r["id"]):r for r in releases}, deleted=[], assets=[], **settings)))
    server=tmp_path/"api-server.py"
    server.write_text(r"""import json,sys,re
from pathlib import Path
from urllib.parse import urlparse,parse_qs
state=Path(sys.argv[1]); tool=sys.argv[2]; args=sys.argv[3:]
data=json.loads(state.read_text())
def finish(value=None,code=0,message=''):
    state.write_text(json.dumps(data))
    if value is not None: print(json.dumps(value))
    if message: print(message,file=sys.stderr)
    raise SystemExit(code)
if tool=='curl':
    source=args[args.index('--data-binary')+1].removeprefix('@')
    name=parse_qs(urlparse(args[-1]).query)['name'][0]
    data['assets'].append(dict(name=name,size=Path(source).stat().st_size))
    finish()
if args[:2]==['auth','status']: finish()
if not args or args[0]!='api': finish(code=97,message='Unexpected network command')
method=args[args.index('--method')+1] if '--method' in args else 'GET'
path=next(a for a in args if a.startswith('repos/'))
if '/git/matching-refs/' in path: finish([])
if '/git/ref/' in path: finish(code=1,message='Tag missing')
if path.endswith('/releases?per_page=100'): finish([list(data['releases'].values())])
if path.endswith('/releases') and method=='POST':
    if data.get('fail_create'): finish(code=1,message='403 Resource not accessible by integration')
    payload=json.load(sys.stdin)
    payload.update(id=999,upload_url='https://uploads.github.com/repos/owner/repo/releases/999/assets{?name,label}')
    data['releases']['999']=payload
    finish(payload)
match=re.search(r'/releases/([0-9]+)(/assets)?',path)
if match:
    identifier=match[1]
    if match[2]:
        assets=data['assets'][:1] if data.get('mismatch') else data['assets']
        finish([assets])
    if method=='GET': finish(data['releases'][identifier])
    if method=='DELETE':
        if not data['releases'][identifier]['draft']: finish(code=1,message='Refusing published deletion')
        data['deleted'].append(int(identifier)); del data['releases'][identifier]; finish()
    if method=='PATCH':
        data['releases'][identifier].update(json.load(sys.stdin)); finish(data['releases'][identifier])
finish(code=97,message='Unexpected API path: '+path)
""",encoding="utf-8")
    for tool in ("gh","curl"):
        command="#!/bin/bash\nexec "+shlex.quote(Path(sys.executable).as_posix())+" "+shlex.quote(server.as_posix())+" "+shlex.quote(state.as_posix())+" "+tool+' "$@"\n'
        write_executable(fake_bin/tool,command)
    # Native Windows Python cannot execute POSIX stub scripts. Route only its
    # network subprocesses to the same fake server; all other commands stay real.
    shim=tmp_path/"python-shim.py"
    shim.write_text("import sys,subprocess,runpy\n"
        +"original=subprocess.run\n"
        +"def isolated(command,*args,**kwargs):\n"
        +"    if command[0] in ('gh','curl'):\n"
        +"        command=[sys.executable,"+repr(server.as_posix())+","+repr(state.as_posix())+",*command]\n"
        +"    return original(command,*args,**kwargs)\n"
        +"subprocess.run=isolated\n"
        +"script=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(script,run_name='__main__')\n",encoding="utf-8")
    launcher=fake_bin/"release-python"
    write_executable(launcher,"#!/bin/bash\nexec "+shlex.quote(Path(sys.executable).as_posix())+" "+shlex.quote(shim.as_posix())+' "$@"\n')
    return state, {"PATH":str(fake_bin)+os.pathsep+os.environ["PATH"],"GH_TOKEN":"token",
                   "SESSIONPREP_RELEASE_PYTHON":launcher.as_posix()}
