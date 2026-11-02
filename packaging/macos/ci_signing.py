"""Use the runner's persistent Developer ID keychain for macOS CI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(*args, capture=False):
    result = subprocess.run([str(arg) for arg in args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None, timeout=1800)
    return result.stdout


def select_identity(output, selector=""):
    identities = re.findall(r'([A-Fa-f0-9]{40}) "(Developer ID Application: [^"\n]+)"', output)
    matches = [(fingerprint, name) for fingerprint, name in identities
               if not selector or selector.lower() == fingerprint.lower() or selector == name]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching valid Developer ID Application identity")
    return matches[0][0]


def prepare(notarize):
    keychain = Path.home() / "Library/Keychains/login.keychain-db"
    password_file = Path.home() / ".config/developer-id-signing/login-keychain-password"
    if not keychain.is_file() or not password_file.is_file():
        raise ValueError("Persistent signing keychain/password file is missing; provision the runner")
    mode = password_file.stat()
    if mode.st_uid != os.getuid() or mode.st_mode & 0o777 != 0o600:
        raise ValueError("Signing password file must be owned by the runner user with mode 600")
    lines = password_file.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("Signing password file must contain one nonempty line")
    password = lines[0]
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::add-mask::" + password, flush=True)
    try:
        subprocess.run(["security", "unlock-keychain", "-p", password, str(keychain)],
                       check=True, capture_output=True, timeout=30)
    except subprocess.SubprocessError:
        raise RuntimeError("Could not unlock the persistent signing keychain") from None
    del password, lines
    identity = select_identity(run("security", "find-identity", "-v", "-p", "codesigning", keychain,
                                   capture=True), os.environ.get("MACOS_DEV_ID_CERT_NAME", ""))
    if notarize:
        run("xcrun", "notarytool", "history", "--keychain-profile", "developer-id-notary",
            "--keychain", keychain, "--output-format", "json", capture=True)
    # Fail before the expensive builds if codesign cannot use the existing ACL.
    with tempfile.TemporaryDirectory(prefix="sessionprep-signing-") as temporary:
        source, binary = Path(temporary) / "probe.c", Path(temporary) / "probe"
        source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        run("xcrun", "clang", source, "-o", binary)
        run("codesign", "--force", "--timestamp", "--options", "runtime", "--keychain", keychain,
            "--sign", identity, binary)
        run("codesign", "--verify", "--strict", binary)
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
        stream.write(f"SIGNING_KEYCHAIN={keychain}\nMACOS_DEV_ID_CERT_NAME={identity}\n")


def submit(path, keychain, runner=run):
    response = runner("xcrun", "notarytool", "submit", path,
                      "--keychain-profile", "developer-id-notary", "--keychain", keychain,
                      "--wait", "--output-format", "json", capture=True)
    if json.loads(response).get("status") != "Accepted":
        raise RuntimeError("Apple notarization was not accepted; inspect the notarytool submission log")
    if Path(path).suffix != ".zip":
        runner("xcrun", "stapler", "staple", path)
        runner("xcrun", "stapler", "validate", path)


def package_app(app, output, identity, keychain, notarize=False, runner=run):
    app, output = Path(app), Path(output)
    if not app.is_dir():
        raise ValueError(f"App bundle is missing: {app}")
    runner("bash", ROOT / "packaging/macos/appbundle-sign.sh", "-a", app, "-c", identity)
    with tempfile.TemporaryDirectory(prefix="sessionprep-dmg-") as temporary:
        temporary = Path(temporary)
        if notarize:
            archive = temporary / "submit.zip"
            runner("ditto", "-c", "-k", "--keepParent", app, archive)
            submit(archive, keychain, runner)
            # The zip cannot be stapled; the accepted ticket belongs on the app.
            runner("xcrun", "stapler", "staple", app)
            runner("xcrun", "stapler", "validate", app)
        stage = temporary / "image"
        stage.mkdir()
        runner("ditto", app, stage / app.name)
        (stage / "Applications").symlink_to("/Applications", target_is_directory=True)
        runner("hdiutil", "create", "-volname", "SessionPrep", "-srcfolder", stage,
               "-ov", "-format", "UDZO", output)
    if not output.is_file():
        raise ValueError(f"DMG was not created: {output}")
    runner("codesign", "--force", "--timestamp", "--keychain", keychain, "--sign", identity, output)
    runner("codesign", "--verify", "--strict", output)
    if notarize:
        submit(output, keychain, runner)
        runner("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", "-vv", output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "package"))
    args = parser.parse_args()
    notarize = os.environ.get("NOTARIZE_MACOS", "false").lower() == "true"
    if args.mode == "prepare":
        prepare(notarize)
    else:
        version, platform = os.environ["BUILD_VERSION"], os.environ["BUILD_PLATFORM"]
        output = Path(os.environ["ARTIFACT_DIR"]) / f"sessionprep-{version}-{platform}.dmg"
        package_app(ROOT / "dist_nuitka/SessionPrep.app", output,
                    os.environ["MACOS_DEV_ID_CERT_NAME"], os.environ["SIGNING_KEYCHAIN"], notarize)


if __name__ == "__main__":
    main()
