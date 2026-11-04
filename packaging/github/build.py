"""Run-scoped CI build metadata, packaging and verified artifact collection."""
from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from build_conf import get_platform_suffix
from sessionpreplib._version import resolve_ref_version

PLATFORMS = ("linux-x64", "linux-arm64", "win-x64", "win-arm64", "macos-arm64", "macos-x64")
MANIFEST = "build-manifest.json"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(directory, platform, version, sha, debug):
    directory = Path(directory)
    files = {p.name: digest(p) for p in sorted(directory.iterdir())
             if p.is_file() and p.name != MANIFEST}
    if not files:
        raise ValueError("Cannot create an empty build artifact")
    data = dict(schema=1, platform=platform, version=version, sha=sha,
                debug=debug, files=files)
    (directory / MANIFEST).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def verify_manifest(directory, platform, version, sha, debug):
    directory = Path(directory)
    data = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    expected = dict(schema=1, platform=platform, version=version, sha=sha, debug=debug)
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Artifact identity mismatch: {directory}")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Invalid artifact file set: {directory}")
    if set(p.name for p in directory.iterdir()) != set(files) | {MANIFEST}:
        raise ValueError(f"Artifact file set mismatch: {directory}")
    for name, checksum in files.items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsafe artifact member: {name}")
        if digest(path) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    return [directory / name for name in files]


def check_linux_audio_runtime():
    """Check the real shared library without opening audio hardware."""
    if sys.platform != "linux":
        return
    message = ("Linux runner cannot load PortAudio. Install libportaudio2 in the "
               "Ubuntu/Debian runner image, rebuild it and recreate the runner "
               "containers for x64 and ARM64.")
    try:
        library = ctypes.util.find_library("portaudio")
        if not library:
            raise ValueError(message)
        ctypes.CDLL(library)
    except OSError as error:
        raise ValueError(message) from error


def metadata(args):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.sha:
        raise ValueError("Checkout does not match dispatched commit")
    if args.ref_type == "tag":
        tagged = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"refs/tags/{args.ref_name}^{{commit}}"], text=True).strip()
        if tagged != actual:
            raise ValueError("Dispatched tag no longer points to the build commit")
    if args.release_drafts and not args.release_token_available:
        event_file = os.environ.get("GITHUB_EVENT_PATH")
        if not event_file:
            raise ValueError("Release preflight requires GitHub event metadata")
        default_branch = json.loads(Path(event_file).read_text(encoding="utf-8"))["repository"]["default_branch"]
        changed = subprocess.check_output(["git", "diff", "--name-only", "refs/remotes/origin/" + default_branch,
                                           actual, "--", ".github/workflows/"], text=True)
        if changed.strip():
            raise ValueError("This ref changes workflows. Configure SESSIONPREP_RELEASE_TOKEN with Contents:write "
                             "and Workflows:write before building release drafts; GITHUB_TOKEN cannot provide that scope.")
    version = resolve_ref_version(args.ref_type, args.ref_name, actual)
    check_linux_audio_runtime()
    with Path(args.github_output).open("a", encoding="utf-8") as stream:
        stream.write(f"version={version}\nsha={actual}\n")
    print(f"Building {args.ref_type} {args.ref_name} at {actual}: {version}")


def compile_app(args):
    if get_platform_suffix() != args.platform:
        raise ValueError(f"Native Python architecture mismatch: {get_platform_suffix()} != {args.platform}")
    if sys.version_info[:2] != (3, 13):
        raise ValueError("Builds require native Python 3.13")
    check_linux_audio_runtime()
    required = (["gcc", "ccache", "patchelf", "nfpm"] if sys.platform == "linux"
                else ["cl.exe"] if sys.platform == "win32"
                else ["xcrun", "codesign", "ditto", "hdiutil"])
    missing = [tool for tool in required if not shutil.which(tool)]
    if missing:
        raise ValueError(f"Runner prerequisites missing: {', '.join(missing)}; provision them on the runner image")
    if sys.platform == "win32":
        run(sys.executable, ROOT / "packaging/windows/build_installer.py", "check", cwd=ROOT)
    command = [sys.executable, str(ROOT / "build_nuitka.py"), "all", "--build-version", args.version]
    if sys.platform == "win32":
        # Bound peak MSVC/SciPy compilation memory on shared Windows workers.
        command.append("--low-memory")
    if args.debug:
        command.append("--debug")
    run(*command, cwd=ROOT)


def linux_package(args):
    dist = ROOT / "dist_nuitka"
    shutil.copytree(dist / "sessionprep-gui.dist", dist / "sessionprep.dist", dirs_exist_ok=True)
    env = {**os.environ, "VERSION": args.version, "SUFFIX": args.platform,
           "NFPM_ARCH": "amd64" if args.platform.endswith("x64") else "arm64",
           "DIST_DIR": "dist_nuitka"}
    config = (ROOT / "packaging/linux/nfpm.yaml").read_text(encoding="utf-8")
    for key in ("VERSION", "SUFFIX", "NFPM_ARCH", "DIST_DIR"):
        config = config.replace("${" + key + "}", env[key])
    with tempfile.TemporaryDirectory(prefix="sessionprep-linux-") as temporary:
        config_path = Path(temporary) / "nfpm.yaml"
        config_path.write_text(config, encoding="utf-8")
        for kind in ("deb", "rpm"):
            run("nfpm", "package", "--config", config_path, "--packager", kind,
                "--target", args.output, cwd=ROOT, env=env)
        stage = Path(temporary) / "stage"
        stage.mkdir()
        shutil.copytree(dist / "sessionprep.dist", stage / "sessionprep.dist")
        for source in ("sessionprepgui/res/sessionprep.png", "packaging/linux/sessionprep.desktop",
                       "packaging/linux/install-sessionprep.sh"):
            shutil.copy2(ROOT / source, stage / Path(source).name)
        (stage / "install-sessionprep.sh").chmod(0o755)
        archive = args.output / f"sessionprep-{args.version}-{args.platform}.tar.gz"
        run("tar", "-czf", archive, "-C", stage, ".")


def package(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.platform.startswith("linux-"):
        linux_package(args)
    elif args.platform.startswith("win-"):
        run(sys.executable, ROOT / "packaging/windows/build_installer.py", "single",
            "--architecture", args.platform.removeprefix("win-"), "--version", args.version,
            "--sha", args.sha, "--configuration", "debug" if args.debug else "release",
            "--output", args.output)
    else:
        env = {**os.environ, "BUILD_VERSION": args.version, "BUILD_PLATFORM": args.platform,
               "ARTIFACT_DIR": str(args.output)}
        run(sys.executable, ROOT / "packaging/macos/ci_signing.py", "package", cwd=ROOT, env=env)
    if args.debug:
        for report in (ROOT / "dist_nuitka").glob("*-report.xml"):
            shutil.copy2(report, args.output / report.name)
    write_manifest(args.output, args.platform, args.version, args.sha, args.debug)


def collect(artifact_root, destination, version, sha):
    artifact_root, destination = Path(artifact_root), Path(destination)
    selected = []
    for platform in (*PLATFORMS, "win-universal"):
        paths = verify_manifest(artifact_root / f"sessionprep-{platform}-release",
                                platform, version, sha, False)
        if platform.startswith("linux-"):
            expected = {".deb", ".rpm", ".gz"}
        elif platform.startswith("macos-"):
            expected = {".dmg"}
        elif platform == "win-universal":
            expected = {".exe"}
        else:
            continue  # per-architecture Windows artifacts are assembly inputs
        deliverables = [path for path in paths if path.suffix in expected]
        if len(deliverables) != len(expected) or {p.suffix for p in deliverables} != expected:
            raise ValueError(f"Incomplete release deliverables: {platform}")
        selected.extend(deliverables)
    if len({p.name for p in selected}) != len(selected):
        raise ValueError("Duplicate release asset names")
    destination.mkdir(parents=True, exist_ok=False)
    for path in selected:
        shutil.copy2(path, destination / path.name)


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    meta = commands.add_parser("metadata")
    meta.add_argument("--ref-type", required=True, choices=("tag", "branch"))
    meta.add_argument("--ref-name", required=True)
    meta.add_argument("--sha", required=True)
    meta.add_argument("--github-output", required=True)
    meta.add_argument("--release-drafts", action="store_true")
    meta.add_argument("--release-token-available", action="store_true")
    for name in ("compile", "package", "collect", "manifest"):
        child = commands.add_parser(name)
        child.add_argument("--version", required=True)
        child.add_argument("--sha", required=True)
        if name == "collect":
            child.add_argument("--artifact-root", type=Path, required=True)
        else:
            child.add_argument("--platform", choices=(*PLATFORMS, "win-universal"), required=True)
            child.add_argument("--debug", action="store_true")
        if name != "compile":
            child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "metadata":
        metadata(args)
    elif args.command == "compile":
        compile_app(args)
    elif args.command == "package":
        package(args)
    elif args.command == "collect":
        collect(args.artifact_root, args.output, args.version, args.sha)
    else:
        write_manifest(args.output, args.platform, args.version, args.sha, args.debug)


if __name__ == "__main__":
    main()
