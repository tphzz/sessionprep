# SessionPrep GitHub Release Publishing

This file documents how the generic release publisher in this directory is used
by SessionPrep.

## Artifact Names

The Nuitka workflow uploads one GitHub Actions artifact per platform:

```text
sessionprep-linux-x64
sessionprep-linux-arm64
sessionprep-macos-arm64
sessionprep-macos-x64
sessionprep-win-x64
sessionprep-win-arm64
```

The collector job should therefore pass:

```bash
--artifact-pattern "sessionprep-*"
```

## Release Assets

`release-assets.allowlist` is intentionally SessionPrep-specific and currently
allows these end-user deliverables:

```text
sessionprep-*-macos-arm64.dmg
sessionprep-*-macos-x64.dmg
sessionprep-*-linux-x64.tar.gz
sessionprep-*-linux-arm64.tar.gz
sessionprep_*.deb
sessionprep-*.rpm
sessionprep-*-win-x64-setup.exe
sessionprep-*-win-arm64-setup.exe
```

Intermediate build directories, loose executables, `.build` folders, and other
non-release files are intentionally excluded.

## Workflow Usage

The SessionPrep workflow calls the generic publisher from the final collector
job after all platform builds finish:

```bash
if [ "$RELEASE_MODE" = "tag" ]; then
  RELEASE_TITLE="SessionPrep $REF_NAME"
else
  RELEASE_TITLE="SessionPrep branch build: $REF_NAME"
fi

packaging/github/publish-draft-release.sh \
  --download-run-id "$RUN_ID" \
  --artifact-root downloaded-artifacts \
  --artifact-pattern "sessionprep-*" \
  --mode "$RELEASE_MODE" \
  --ref-name "$REF_NAME" \
  --target "$TARGET_SHA" \
  --repo "$REPOSITORY" \
  --release-title "$RELEASE_TITLE"
```

The workflow has a manual `notarize` input. It defaults to `false`, so branch
builds produce signed, non-notarized macOS development DMGs by default. If a
manual branch run is started with `notarize=true`, the macOS jobs use the same
notarized DMG packaging path as tag builds. Tag builds always notarize.

## Local Dry Run

Dry-run against an existing workflow run:

```bash
packaging/github/publish-draft-release.sh \
  --download-run-id 1234567890 \
  --artifact-root downloaded-artifacts \
  --artifact-pattern "sessionprep-*" \
  --mode branch \
  --ref-name 0.3.5 \
  --target "$(git rev-parse HEAD)" \
  --repo tphzz/sessionprep \
  --release-title "SessionPrep branch build: 0.3.5" \
  --dry-run
```

For a tag build, use:

```bash
--mode tag
--ref-name 0.3.5
--release-title "SessionPrep 0.3.5"
```

## Release Behavior

Branch builds use a release title such as `SessionPrep branch build: 0.3.5`.
Re-running a branch workflow replaces any matching draft release for that branch
and uploads assets to the newly created release by release ID. The script also
cleans up older duplicate draft releases that used the same title before the
hidden branch release key was introduced. Branch macOS DMGs are non-notarized
unless the manual workflow run explicitly enables `notarize`.

Tag builds use the real Git tag, such as `0.3.5`. Re-running a tag workflow
replaces the draft release for that tag, but published releases are protected
and are not modified by the automation. Tag macOS DMGs are always notarized.
