# macOS signing and packaging

SessionPrep builds separate ARM64 and Intel Nuitka app bundles. The manual macOS
workflow can build either architecture or both; All Platforms builds both
sequentially on the Apple Silicon runner, using Rosetta/Intel Python for x64.

## CI Python interpreters

The workflow uses uv-managed CPython 3.13 in the runner account's Python cache.
It requests `cpython-3.13-macos-aarch64-none` for ARM64 and
`cpython-3.13-macos-x86_64-none` for Intel; only selected architectures are set up.
Missing managed interpreters are downloaded without an administrator installation
or changes to the default Python executable. Intel execution requires the
runner's existing Rosetta installation.

Optional runner environment variables `CI_PYTHON_MACOS_ARM64` and
`CI_PYTHON_MACOS_X64` select preinstalled interpreter executables. Each takes
precedence over the legacy `CI_PYTHON` override. A configured override is resolved
without downloads and must pass the same CPython 3.13, macOS and process-architecture
checks. An invalid override fails without selecting a different interpreter.
Both `uv sync` and `uv run` use the verified path for that architecture.

The first verified interpreter also handles signing preparation and run-owned
cleanup. If setup fails before one is selected, cleanup can use system `python3`,
including older macOS SDK Python versions. Managed Python installations remain
cached; cleanup removes only the run's checkouts and virtual environments.
See [uv's architecture selection documentation](https://docs.astral.sh/uv/concepts/python-versions/#transparent-x86_64-emulation-on-aarch64).

## Persistent CI signing

`ci_signing.py prepare` uses the runner's existing
`~/Library/Keychains/login.keychain-db` and reads its password from
`~/.config/developer-id-signing/login-keychain-password`. The password file must
belong to the runner account with mode 600 and contain one nonempty line.
The workflow unlocks and reuses this keychain; it does not import P12 files,
create/delete keychains, rewrite the search list or recreate notary credentials.

Exactly one valid Developer ID Application identity is required. Optional secret
`MACOS_DEV_ID_CERT_NAME` can select its exact name or fingerprint when several
identities exist. A small signing probe fails early if the existing key/ACL is
unavailable. The password is masked and never written to artifacts.

Both configurations sign the app and DMG. Notarization happens only when the
Notarize checkbox is checked; its default is off for branches, tags and Debug.
Checked runs use the already provisioned `developer-id-notary` keychain profile,
validate Apple's Accepted status, staple/validate tickets, and assess the DMG
with the disk-image Gatekeeper policy. No Apple ID/app-password or P12 GitHub
secrets are needed by this CI path. Neither building nor notarizing publishes a
release; only non-Debug All Platforms creates an unpublished draft.

`ci_signing.py package` calls `appbundle-sign.sh` for the established Nuitka/Qt
bundle layout and preserves the selected interpreter. Optional `SIGNING_KEYCHAIN`
restricts signing to the existing keychain. The shell signing helper remains
compatible with stock macOS Bash 3.2 and manual signing without that variable.

## Manual signing tools

For a local Nuitka app:

```bash
bash packaging/macos/appbundle-sign.sh -a dist_nuitka/SessionPrep.app -c "Developer ID Application: Your Name (TEAM_ID)"
```

The older interactive `appbundle-release.sh` remains available for local
credential-based notarization (`MACOS_DEV_ID_CERT_NAME`, `APPLE_NOTARY_USER`,
`APPLE_NOTARY_APP_PASSWORD`, `APPLE_TEAM_ID`). It always notarizes when invoked;
it is not called by CI and does not control the workflow checkbox.

See [the workflow guide](../github/README.sessionprep.md) for prerequisites,
artifact names, versioning and release ownership.
