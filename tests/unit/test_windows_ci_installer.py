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
    try:
        installer.resolve_iscc()
    except ValueError as error:
        if installer.os.environ.get("ISCC_PATH") or installer.os.environ.get("ISCC"):
            raise
        pytest.skip(str(error))
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


@pytest.fixture
def inno_environment(tmp_path, monkeypatch):
    for name in ("ISCC_PATH", "ISCC", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "Program Files x86"))
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    return tmp_path


def fake_compiler(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"compiler fixture; never executed")
    return path


@pytest.mark.parametrize("name", ["ISCC_PATH", "ISCC"])
def test_inno_explicit_path_precedes_path_and_discovery(inno_environment, monkeypatch, name):
    selected = fake_compiler(inno_environment / "custom compiler" / "ISCC.exe")
    other = fake_compiler(inno_environment / "other" / "ISCC.exe")
    monkeypatch.setenv(name, str(selected))
    monkeypatch.setattr(installer.shutil, "which", lambda command: str(other))
    if name == "ISCC_PATH":
        monkeypatch.setenv("ISCC", str(other))
    assert installer.resolve_iscc() == str(selected.resolve())


@pytest.mark.parametrize("name", ["ISCC_PATH", "ISCC"])
def test_inno_invalid_explicit_path_fails_without_fallback(inno_environment, monkeypatch, name):
    monkeypatch.setenv(name, str(inno_environment / "missing.exe"))
    monkeypatch.setattr(installer.shutil, "which", lambda command: pytest.fail("Invalid override must not fall back"))
    with pytest.raises(ValueError, match=name):
        installer.resolve_iscc()


def test_inno_accepts_compiler_from_path(inno_environment, monkeypatch):
    selected = fake_compiler(inno_environment / "on path" / "ISCC.exe")
    monkeypatch.setattr(installer.shutil, "which", lambda name: str(selected))
    assert installer.resolve_iscc() == str(selected.resolve())


@pytest.mark.parametrize("root", ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "LOCALAPPDATA"])
@pytest.mark.parametrize("version", ["6", "7"])
def test_inno_discovers_machine_and_user_installations(inno_environment, monkeypatch, root, version):
    base = inno_environment / root
    monkeypatch.setenv(root, str(base))
    if root == "LOCALAPPDATA":
        base /= "Programs"
    selected = fake_compiler(base / ("Inno Setup " + version) / "ISCC.exe")
    assert installer.resolve_iscc() == str(selected.resolve())


def test_inno_missing_compiler_explains_configuration(inno_environment):
    with pytest.raises(ValueError, match="ISCC_PATH"):
        installer.resolve_iscc()


def test_inno_check_mode_requires_no_payload_arguments(inno_environment, monkeypatch, capsys):
    import sys
    selected = fake_compiler(inno_environment / "Inno Setup 7" / "ISCC.exe")
    monkeypatch.setenv("ISCC_PATH", str(selected))
    monkeypatch.setattr(sys, "argv", ["build_installer.py", "check"])
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: pytest.fail("Check must not compile an installer"))
    installer.main()
    assert str(selected.resolve()) in capsys.readouterr().out


def test_inno_compile_uses_iscc_path_as_one_argument(inno_environment, monkeypatch):
    selected = fake_compiler(inno_environment / "Inno Setup 7" / "ISCC.exe")
    monkeypatch.setenv("ISCC_PATH", str(selected))
    stage, output = inno_environment / "stage", inno_environment / "output"
    stage.mkdir()
    output.mkdir()
    calls = []
    def compile_fixture(command, **kwargs):
        calls.append(command)
        (output / "sessionprep-0.3.5-win-x64-setup.exe").write_bytes(b"installer fixture")
    monkeypatch.setattr(installer.subprocess, "run", compile_fixture)
    installer.compile_installer(stage, output, "0.3.5", "win-x64", "dist_nuitka")
    assert calls[0][0] == str(selected.resolve())
    assert calls[0][1] == "/O" + str(output)
