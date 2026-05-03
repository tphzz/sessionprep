# GitHub Draft Release Publishing

This directory contains the release publishing layer used by the Nuitka build
workflow. The build matrix runs on separate Linux, macOS, and Windows runners,
so no build runner ever has every release asset. Publishing is therefore done
by a single Ubuntu collector job after all matrix builds finish.

## Files

| File | Purpose |
| --- | --- |
| `publish-draft-release.sh` | Validates downloaded artifacts, stages selected release assets, and updates a GitHub draft release. |
| `release-assets.allowlist` | Explicit list of basename glob patterns that are allowed to become release assets. |

## Workflow Architecture

1. Each matrix build produces platform-specific packages in `dist_nuitka/`.
2. Each matrix build uploads one GitHub Actions artifact named
   `sessionprep-<platform-suffix>`.
3. The collector job runs `publish-draft-release.sh`.
4. The script downloads all `sessionprep-*` artifacts from the workflow run
   into one directory, unless an existing `--artifact-root` is supplied without
   `--download-run-id`.
5. `publish-draft-release.sh` scans the downloaded artifacts, copies only
   allowlisted files into `release-assets/`, and publishes those files.

The release script is the source of truth for filtering and replacement
semantics. The workflow should not duplicate release asset glob logic or
artifact-download behavior.

## Authentication

In GitHub Actions, no custom secret is required for publishing draft releases
to the same repository. The collector job must grant write access and expose the
built-in token:

```yaml
permissions:
  contents: write
  actions: read

env:
  GH_TOKEN: ${{ github.token }}
```

For local publishing, authenticate the GitHub CLI once:

```bash
gh auth login
gh auth status
```

Dry-run mode does not require GitHub authentication when it uses an existing
local `--artifact-root`. Dry-run mode with `--download-run-id` still needs
GitHub CLI authentication because it downloads Actions artifacts first.

## Local Dry Run

Use dry-run mode to validate artifact contents and allowlist matching without
creating or changing a release.

`--target` is the commit SHA for the workflow run being published. In branch
mode, the script creates the synthetic branch-build release tag at that exact
commit. In tag mode, GitHub already has a real tag from `--ref-name`; the script
uses `--target` only for validation, logging, and release notes so the draft is
traceable to the build run.

Against an already downloaded artifact directory:

```bash
packaging/github/publish-draft-release.sh \
  --artifact-root downloaded-artifacts \
  --mode branch \
  --ref-name 0.3.5 \
  --target "$(git rev-parse HEAD)" \
  --repo bzeiss/sessionprep \
  --dry-run
```

Against a real GitHub Actions run:

```bash
packaging/github/publish-draft-release.sh \
  --download-run-id 1234567890 \
  --artifact-root downloaded-artifacts \
  --mode branch \
  --ref-name 0.3.5 \
  --target "$(git rev-parse HEAD)" \
  --repo bzeiss/sessionprep \
  --dry-run
```

Dry-run mode still creates a clean staging directory, defaulting to
`release-assets/`, so the selected files can be inspected locally.

## Publishing

Branch build:

```bash
packaging/github/publish-draft-release.sh \
  --download-run-id 1234567890 \
  --artifact-root downloaded-artifacts \
  --mode branch \
  --ref-name 0.3.5 \
  --target "$(git rev-parse HEAD)" \
  --repo tphzz/sessionprep
```

Tag build:

```bash
packaging/github/publish-draft-release.sh \
  --download-run-id 1234567890 \
  --artifact-root downloaded-artifacts \
  --mode tag \
  --ref-name 0.3.5 \
  --target "$(git rev-parse HEAD)" \
  --repo tphzz/sessionprep
```

## Release Semantics

### Branch Mode

Branch mode creates a synthetic draft release tag derived from the branch name,
for example `branch-build-0.3.5`.

On every successful branch build, the script:

1. refuses to touch the release if it has already been published,
2. deletes the existing draft release if present,
3. deletes the existing synthetic tag if present,
4. recreates the synthetic draft release at the requested commit,
5. uploads the selected assets.

### Tag Mode

Tag mode uses the real Git tag from `--ref-name`.

On a tag build, the script:

1. creates a draft release if none exists,
2. deletes and recreates the draft release if one already exists,
3. preserves the real Git tag,
4. refuses to modify the release if it has already been published.

This means re-running a tag workflow replaces generated draft assets, but
published releases are protected from accidental mutation.

## Allowlist Format

`release-assets.allowlist` contains one basename glob pattern per line.
Blank lines and `#` comments are ignored.

Examples:

```text
sessionprep-*-macos-arm64.dmg
sessionprep-*-linux-x64.tar.gz
sessionprep_*.deb
sessionprep-*-win-x64-setup.exe
```

Patterns match file basenames only. Directory names inside downloaded artifacts
or ZIP archives do not matter. This avoids accidentally publishing build
directories, loose binaries, temporary DMGs, or intermediate files.

## Safety Checks

The script fails before publishing when it finds:

- missing required arguments,
- invalid release mode,
- invalid repository format,
- invalid target SHA,
- invalid download run ID,
- missing artifact root,
- missing or empty allowlist,
- missing required tools,
- unsafe staging directory,
- unreadable or corrupt ZIP artifacts,
- no files matching the allowlist,
- duplicate final asset filenames,
- empty selected assets,
- unauthenticated `gh` in publish mode,
- an existing published release for the target tag.

ZIP files are not extracted wholesale. The script lists ZIP entries and streams
only allowlisted files into the staging directory. This avoids unsafe archive
paths and keeps filtering deterministic.

## Troubleshooting

**No release assets matched**

Check the downloaded artifact layout and update `release-assets.allowlist` if
package filenames changed.

**Duplicate release asset filename**

Two artifacts produced the same basename. Rename one package output or make the
allowlist more specific.

**GitHub CLI is not authenticated**

In CI, ensure `GH_TOKEN: ${{ github.token }}` is set on the collector job. For
local publishing, run `gh auth login`.

**Refusing to modify published release**

The release has already been published. Create a new tag or manually manage the
published release. The automation intentionally protects published releases.

**Could not read ZIP artifact**

The artifact file is corrupt or not a ZIP. Re-download the artifact or inspect
the build upload step.
