from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _inno_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, re.MULTILINE)
    assert match is not None, f"Missing Inno Setup key: {key}"
    return match.group(1).strip()


def test_windows_installer_identity_is_stable_across_versions():
    text = _read("packaging/windows/sessionprep-common.isinc")

    stable_keys = [
        "AppId",
        "AppName",
        "AppVerName",
        "DefaultDirName",
        "DefaultGroupName",
        "UninstallDisplayName",
    ]
    for key in stable_keys:
        assert "APP_VERSION" not in _inno_value(text, key)

    assert _inno_value(text, "AppName") == "{#AppName}"
    assert _inno_value(text, "AppVerName") == "{#AppName}"
    assert _inno_value(text, "UninstallDisplayName") == "{#AppName}"


def test_windows_installer_keeps_version_in_metadata_and_artifact_name():
    text = _read("packaging/windows/sessionprep-common.isinc")

    assert _inno_value(text, "AppVersion") == "{#APP_VERSION}"
    assert "APP_VERSION" in _inno_value(text, "OutputBaseFilename")


def test_linux_package_identity_is_stable_and_version_is_separate():
    nfpm = _read("packaging/linux/nfpm.yaml")
    desktop = _read("packaging/linux/sessionprep.desktop")

    assert re.search(r"^name: sessionprep$", nfpm, re.MULTILINE)
    assert re.search(r'^version: "\$\{VERSION\}"$', nfpm, re.MULTILINE)
    assert re.search(r"^Name=SessionPrep$", desktop, re.MULTILINE)


def test_macos_app_identity_is_stable_and_artifact_filename_is_versioned():
    build_conf = _read("build_conf.py")
    workflow = _read(".github/workflows/build-nuitka.yml")

    assert 'MACOS_APP_NAME = "SessionPrep"' in build_conf
    assert 'APP_PATH="dist_nuitka/SessionPrep.app"' in workflow
    assert 'FINAL_DMG="dist_nuitka/sessionprep-${VERSION}-${{ matrix.suffix }}.dmg"' in workflow