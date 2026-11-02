import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("windows_installer", ROOT / "packaging/windows/build_installer.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_payload_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        installer.extract_payload(archive, tmp_path / "stage", "0.3.5", "a" * 40, "x64", "release")
    assert not (tmp_path / "escape").exists()


def test_payload_extraction_rejects_wrong_commit_before_copying(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("innosetup-rebuild.json", json.dumps(dict(schema=1, version="0.3.5", sha="b" * 40,
                                                                  architecture="x64", configuration="release")))
    with pytest.raises(ValueError, match="identity"):
        installer.extract_payload(archive, tmp_path / "stage", "0.3.5", "a" * 40, "x64", "release")
    assert not (tmp_path / "stage").exists()


def test_pe_machine_check_distinguishes_x64_and_arm64(tmp_path):
    binary = tmp_path / "program.exe"
    content = bytearray(132)
    content[:2] = b"MZ"
    content[60:64] = (100).to_bytes(4, "little")
    content[100:104] = b"PE\0\0"
    content[104:106] = (0xAA64).to_bytes(2, "little")
    binary.write_bytes(content)
    installer.verify_pe(binary, "arm64")
    with pytest.raises(ValueError, match="architecture"):
        installer.verify_pe(binary, "x64")


@pytest.mark.parametrize("architecture", ["x64", "arm64", "universal"])
def test_inno_compiles_native_and_adaptive_definitions(tmp_path, architecture):
    compiler = Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")
    if not compiler.is_file():
        pytest.skip("Native Inno Setup compiler is unavailable")
    stage = tmp_path / "kit"
    stage.mkdir()
    installer.copy_definitions(stage)
    platforms = ("x64", "arm64") if architecture == "universal" else (architecture,)
    directory = "payloads" if architecture == "universal" else "dist_nuitka"
    for platform in platforms:
        base = stage / directory
        if architecture == "universal":
            base /= platform
        for stem in ("sessionprep", "sessionprep-gui"):
            target = base / f"{stem}.dist" / f"{stem}-win-{platform}.exe"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"test fixture; never executed")
    output = tmp_path / "output"
    output.mkdir()
    installer.compile_installer(stage, output, "0.3.5", "win-" + architecture, directory)
    assert (output / f"sessionprep-0.3.5-win-{architecture}-setup.exe").is_file()
