"""Package native Windows payloads and assemble one adaptive x64/ARM64 setup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def verify_pe(path, architecture):
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"Invalid PE executable: {path}")
        stream.seek(60)
        offset = stream.read(4)
        if len(offset) != 4:
            raise ValueError(f"Truncated PE executable: {path}")
        stream.seek(struct.unpack("<I", offset)[0])
        header = stream.read(6)
    expected = {"x64": 0x8664, "arm64": 0xAA64}[architecture]
    if len(header) != 6 or header[:4] != b"PE\0\0" or struct.unpack("<H", header[4:])[0] != expected:
        raise ValueError(f"Wrong executable architecture for {architecture}: {path}")


def verify_payload(root, architecture):
    for stem in ("sessionprep", "sessionprep-gui"):
        verify_pe(Path(root) / f"{stem}.dist" / f"{stem}-win-{architecture}.exe", architecture)


def extract_payload(archive, destination, version, sha, architecture, configuration):
    destination = Path(destination)
    expected = dict(schema=1, version=version, sha=sha, architecture=architecture, configuration=configuration)
    with zipfile.ZipFile(archive) as stream:
        for info in stream.infolist():
            name = info.filename.replace("\\", "/")
            resolved = (destination / name).resolve()
            if (not resolved.is_relative_to(destination.resolve()) or name.startswith("/")
                    or ":" in name or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
        actual = json.loads(stream.read("innosetup-rebuild.json"))
        if actual != expected:
            raise ValueError(f"Windows payload identity mismatch: {archive}")
        destination.mkdir(parents=True, exist_ok=False)
        stream.extractall(destination)
    verify_payload(destination / "dist_nuitka", architecture)
    return destination / "dist_nuitka"


def copy_definitions(stage):
    for relative in ("packaging/windows/sessionprep.iss", "packaging/windows/sessionprep-signed.iss",
                     "packaging/windows/sessionprep-common.isinc", "sessionprepgui/res/sessionprep.ico"):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def create_kit(stage, output, version, suffix, dist_dir):
    command = (f'ISCC.exe /DAPP_VERSION={version} /DDIST_DIR={dist_dir} '
               f'/DARCH_SUFFIX={suffix} packaging/windows/sessionprep.iss')
    (stage / "README-INNOSETUP-REBUILD.txt").write_text(
        "SessionPrep installer rebuild kit\n\nRequires Inno Setup 6 or 7 (ISCC.exe). Run from this extracted directory:\n"
        + command + "\n\nFor Certum signing use sessionprep-signed.iss and the existing\n"
        'sessionprepSignTool definition: signtool.exe sign /a /fd SHA256 /t http://time.certum.pl/ $f\n',
        encoding="utf-8")
    target = output / f"sessionprep-{version}-{suffix}-innosetup-input.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in stage.rglob("*"):
            if path.is_file():
                stream.write(path, path.relative_to(stage).as_posix())


def resolve_iscc():
    """Find the runner account's compiler, honoring explicit configuration first."""
    for name in ("ISCC_PATH", "ISCC"):
        explicit = os.environ.get(name, "").strip()
        if explicit:
            candidate = Path(explicit)
            if not candidate.is_file():
                raise ValueError(f"{name} does not point to an ISCC.exe compiler file: {candidate}")
            return str(candidate.resolve())
    found = shutil.which("ISCC.exe")
    if found:
        return str(Path(found).resolve())
    roots = []
    for name in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432"):
        value = os.environ.get(name)
        if value and Path(value) not in roots:
            roots.append(Path(value))
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "Programs")
    for version in ("6", "7"):
        for root in roots:
            candidate = root / ("Inno Setup " + version) / "ISCC.exe"
            if candidate.is_file():
                return str(candidate.resolve())
    raise ValueError("Inno Setup 6 or 7 is missing for this runner account; "
                     "set ISCC_PATH to the full ISCC.exe path or add it to PATH")


def compile_installer(stage, output, version, suffix, dist_dir):
    compiler = resolve_iscc()
    subprocess.run([compiler, f"/O{output}", f"/DAPP_VERSION={version}", f"/DDIST_DIR={dist_dir}",
                    f"/DARCH_SUFFIX={suffix}", str(stage / "packaging/windows/sessionprep.iss")],
                   cwd=stage, check=True)
    expected = output / f"sessionprep-{version}-{suffix}-setup.exe"
    if not expected.is_file():
        raise ValueError(f"Expected installer was not generated: {expected}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check", help="Check installer compiler discovery without building")
    for mode in ("single", "bundle"):
        child = modes.add_parser(mode)
        child.add_argument("--architecture", choices=("x64", "arm64"))
        child.add_argument("--version", required=True)
        child.add_argument("--sha", required=True)
        child.add_argument("--configuration", required=True, choices=("release", "debug"))
        child.add_argument("--artifact-root", type=Path)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = resolve_iscc()
    if args.mode == "check":
        print(f"Inno Setup compiler: {compiler}")
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sessionprep-inno-") as temporary:
        stage = Path(temporary)
        copy_definitions(stage)
        if args.mode == "single":
            if not args.architecture:
                parser.error("single mode requires --architecture")
            verify_payload(ROOT / "dist_nuitka", args.architecture)
            payload = stage / "dist_nuitka"
            payload.mkdir()
            for stem in ("sessionprep", "sessionprep-gui"):
                shutil.copytree(ROOT / "dist_nuitka" / f"{stem}.dist", payload / f"{stem}.dist")
            (stage / "innosetup-rebuild.json").write_text(json.dumps(dict(
                schema=1, version=args.version, sha=args.sha, architecture=args.architecture,
                configuration=args.configuration)), encoding="utf-8")
            suffix, dist_dir = "win-" + args.architecture, "dist_nuitka"
        else:
            if args.artifact_root is None:
                parser.error("bundle mode requires --artifact-root")
            # Validate the checksummed outer artifacts before extracting any payload.
            import importlib.util
            spec = importlib.util.spec_from_file_location("ci_artifacts", ROOT / "packaging/github/build.py")
            ci = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ci)
            (stage / "payloads").mkdir()
            for architecture in ("x64", "arm64"):
                artifact = args.artifact_root / f"sessionprep-win-{architecture}-{args.configuration}"
                ci.verify_manifest(artifact, "win-" + architecture, args.version, args.sha,
                                   args.configuration == "debug")
                archive = artifact / f"sessionprep-{args.version}-win-{architecture}-innosetup-input.zip"
                extracted = extract_payload(archive, stage / ("input-" + architecture), args.version,
                                            args.sha, architecture, args.configuration)
                shutil.move(str(extracted), stage / "payloads" / architecture)
                shutil.rmtree(stage / ("input-" + architecture))
            suffix, dist_dir = "win-universal", "payloads"
        compile_installer(stage, output, args.version, suffix, dist_dir)
        create_kit(stage, output, args.version, suffix, dist_dir)


if __name__ == "__main__":
    main()
