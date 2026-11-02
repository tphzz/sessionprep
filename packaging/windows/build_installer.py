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
        "SessionPrep installer rebuild kit\n\nRequires Inno Setup 6. Run from this extracted directory:\n"
        + command + "\n\nFor Certum signing use sessionprep-signed.iss and the existing\n"
        'sessionprepSignTool definition: signtool.exe sign /a /fd SHA256 /t http://time.certum.pl/ $f\n',
        encoding="utf-8")
    target = output / f"sessionprep-{version}-{suffix}-innosetup-input.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in stage.rglob("*"):
            if path.is_file():
                stream.write(path, path.relative_to(stage).as_posix())


def compile_installer(stage, output, version, suffix, dist_dir):
    compiler = (os.environ.get("ISCC") or shutil.which("ISCC.exe") or
                str(Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Inno Setup 6/ISCC.exe"))
    if not Path(compiler).is_file():
        raise ValueError("Inno Setup 6 is missing; provision ISCC.exe on the runner")
    subprocess.run([compiler, f"/O{output}", f"/DAPP_VERSION={version}", f"/DDIST_DIR={dist_dir}",
                    f"/DARCH_SUFFIX={suffix}", str(stage / "packaging/windows/sessionprep.iss")],
                   cwd=stage, check=True)
    expected = output / f"sessionprep-{version}-{suffix}-setup.exe"
    if not expected.is_file():
        raise ValueError(f"Expected installer was not generated: {expected}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("single", "bundle"))
    parser.add_argument("--architecture", choices=("x64", "arm64"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--configuration", required=True, choices=("release", "debug"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
