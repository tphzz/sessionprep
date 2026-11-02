import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("macos_signing", ROOT / "packaging/macos/ci_signing.py")
signing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signing)


def test_identity_selection_is_explicit_and_noninteractive():
    first, second = "A" * 40, "B" * 40
    identities = f'1) {first} "Developer ID Application: One (TEAM1)"\n2) {second} "Developer ID Application: Two (TEAM2)"'
    assert signing.select_identity(identities, first) == first
    assert signing.select_identity(identities, "Developer ID Application: Two (TEAM2)") == second
    with pytest.raises(ValueError, match="exactly one"):
        signing.select_identity(identities)
    with pytest.raises(ValueError, match="exactly one"):
        signing.select_identity(identities, "not-installed")


def test_non_developer_identity_is_not_selected():
    with pytest.raises(ValueError, match="exactly one"):
        signing.select_identity('1) ' + "A" * 40 + ' "Apple Development: Person"')


@pytest.mark.parametrize("notarize", [False, True])
def test_notarization_is_opt_in_and_uses_persistent_profile(tmp_path, notarize):
    app = tmp_path / "SessionPrep.app"
    app.mkdir()
    calls = []
    output = tmp_path / "output.dmg"
    def runner(*args, **kwargs):
        calls.append(tuple(str(a) for a in args))
        if args[0] == "hdiutil":
            output.write_bytes(b"dmg")
        if args[:3] == ("xcrun", "notarytool", "submit"):
            return '{"status":"Accepted"}'
        return ""
    # Directory symlinks require privileges on Windows; a placeholder suffices for this command test.
    from unittest.mock import patch
    with patch.object(Path, "symlink_to", lambda *a, **kw: None):
        signing.package_app(app, output, "A" * 40, tmp_path / "login.keychain-db", notarize, runner)
    submissions = [a for a in calls if a[:3] == ("xcrun", "notarytool", "submit")]
    assert len(submissions) == (2 if notarize else 0)
    assessments = [a for a in calls if a[0] == "spctl"]
    assert bool(assessments) is notarize
    if notarize:
        assert "open" in assessments[0] and "context:primary-signature" in assessments[0]
    assert all("developer-id-notary" in a and "--keychain" in a for a in submissions)
    assert not any("--apple-id" in a or "delete-keychain" in a or "import" in a for a in calls)
    assert not any(a[:3] == ("xcrun", "stapler", "staple") and a[-1].endswith(".zip") for a in calls)


def test_notary_rejection_fails_before_stapling(tmp_path):
    calls = []
    def runner(*args, **kwargs):
        calls.append(args)
        return '{"status":"Invalid"}'
    with pytest.raises(RuntimeError, match="not accepted"):
        signing.submit(tmp_path / "app.zip", tmp_path / "login.keychain-db", runner)
    assert len(calls) == 1
