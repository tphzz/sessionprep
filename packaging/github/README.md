# GitHub Draft Release Publishing

This directory contains a generic GitHub Actions release-publishing helper. It
is designed for build matrices where different operating systems produce
different artifacts and no individual build runner has the complete release
asset set.

The recommended shape is a final collector job that runs after all build jobs.
The collector job calls `publish-draft-release.sh`, which can download artifacts
from a workflow run, filter them through an allowlist, stage the selected files,
and update a GitHub draft release.

## Files

| File | Purpose |
| --- | --- |
| `publish-draft-release.sh` | Downloads or reads build artifacts, stages selected release assets, and updates a GitHub draft release. |
| `release-assets.allowlist` | Basename glob patterns that are allowed to become release assets. Replace or edit this for each repository. |

## Workflow Architecture

1. Matrix build jobs produce platform-specific packages.
2. Each matrix build uploads one or more GitHub Actions artifacts.
3. A collector job runs after the matrix succeeds.
4. The collector calls `publish-draft-release.sh`.
5. The script optionally downloads matching artifacts with `gh run download`.
6. The script scans the artifact tree, copies only allowlisted files into a
   staging directory, and uploads those staged files to a draft release.

The script is the source of truth for filtering and release replacement
semantics. Keep project-specific filename patterns in the allowlist or pass them
through command-line options rather than duplicating release selection logic in
workflow YAML.

## Authentication

In GitHub Actions, no custom secret is required for publishing draft releases to
the same repository. The collector job needs repository contents write access
and the built-in token exposed to the GitHub CLI:

```yaml
permissions:
  contents: write
  actions: read

env:
  GH_TOKEN: ${{ github.token }}
```

For local publishing or local artifact downloads, authenticate the GitHub CLI
once:

```bash
gh auth login
gh auth status
```

Dry-run mode does not require GitHub authentication when it uses an existing
local `--artifact-root`. Dry-run mode with `--download-run-id` still requires
GitHub CLI authentication because it downloads Actions artifacts first.

## Important Arguments

`--repo OWNER/REPO` is the repository that contains the workflow artifacts and
the draft release.

`--download-run-id ID` tells the script to download artifacts from a GitHub
Actions run before filtering. This is useful in CI and for local reproduction.

`--artifact-pattern GLOB` limits which GitHub Actions artifact names are
downloaded. The default is `*`. Most repositories should pass a narrower
project-specific pattern.

`--artifact-root DIR` is either an already downloaded artifact tree or the
destination directory used with `--download-run-id`.

`--target SHA` is the commit SHA for the workflow run being published. In branch
mode, the script creates the synthetic branch-build release tag at that exact
commit. In tag mode, GitHub already has a real tag from `--ref-name`; the script
uses `--target` for validation, logging, and release notes so the draft is
traceable to the build run.

`--allowlist FILE` points to basename glob patterns for files that may become
release assets. Directory names inside artifacts or ZIP files are ignored for
matching.

## Local Dry Run

Against an already downloaded artifact directory:

```bash
packaging/github/publish-draft-release.sh \
  --artifact-root downloaded-artifacts \
  --mode branch \
  --ref-name feature/test-release \
  --target "$(git rev-parse HEAD)" \
  --repo owner/repo \
  --allowlist packaging/github/release-assets.allowlist \
  --dry-run
```

Against a real GitHub Actions run:

```bash
packaging/github/publish-draft-release.sh \
  --download-run-id 1234567890 \
  --artifact-pattern "myapp-*" \
  --artifact-root downloaded-artifacts \
  --mode branch \
  --ref-name feature/test-release \
  --target "$(git rev-parse HEAD)" \
  --repo owner/repo \
  --allowlist packaging/github/release-assets.allowlist \
  --dry-run
```

Dry-run mode still creates a clean staging directory, defaulting to
`release-assets/`, so selected files can be inspected locally.

## CI Usage

Typical collector job:

```yaml
publish-draft-release:
  name: Publish draft release
  needs: build
  runs-on: ubuntu-latest
  permissions:
    contents: write
    actions: read
  concurrency:
    group: draft-release-${{ github.repository }}-${{ github.ref }}
    cancel-in-progress: false
  env:
    GH_TOKEN: ${{ github.token }}
    RELEASE_MODE: ${{ github.ref_type == 'tag' && 'tag' || 'branch' }}
    REF_NAME: ${{ github.ref_name }}
    TARGET_SHA: ${{ github.sha }}
    REPOSITORY: ${{ github.repository }}
    RUN_ID: ${{ github.run_id }}
  steps:
    - uses: actions/checkout@v5

    - name: Publish draft release
      shell: bash
      run: |
        packaging/github/publish-draft-release.sh \
          --download-run-id "$RUN_ID" \
          --artifact-pattern "myapp-*" \
          --artifact-root downloaded-artifacts \
          --mode "$RELEASE_MODE" \
          --ref-name "$REF_NAME" \
          --target "$TARGET_SHA" \
          --repo "$REPOSITORY"
```

## Release Semantics

### Branch Mode

Branch mode creates a synthetic draft release tag derived from the branch name,
for example `branch-build-feature-test-release`.

On every successful branch build, the script:

1. refuses to touch the release if it has already been published,
2. deletes the existing draft release if present,
3. deletes the existing synthetic tag if present,
4. recreates the synthetic draft release at `--target`,
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

`release-assets.allowlist` contains one basename glob pattern per line. Blank
lines and `#` comments are ignored.

Example:

```text
myapp-*-macos-arm64.dmg
myapp-*-linux-x64.tar.gz
myapp_*.deb
myapp-*-win-x64-setup.exe
```

Patterns match file basenames only. This avoids accidentally publishing build
directories, loose binaries, temporary files, or intermediate outputs.

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
- unauthenticated `gh` when GitHub access is needed,
- an existing published release for the target tag.

ZIP files are not extracted wholesale. The script lists ZIP entries and streams
only allowlisted files into the staging directory. This avoids unsafe archive
paths and keeps filtering deterministic.

## Troubleshooting

**No release assets matched**

Check the downloaded artifact layout and update the allowlist or
`--artifact-pattern`.

**Duplicate release asset filename**

Two artifacts produced the same basename. Rename one package output or make the
allowlist more specific.

**GitHub CLI is not authenticated**

In CI, ensure `GH_TOKEN: ${{ github.token }}` is set on the collector job. For
local publishing or artifact downloads, run `gh auth login`.

**Refusing to modify published release**

The release has already been published. Create a new tag or manually manage the
published release. The automation intentionally protects published releases.

**Could not read ZIP artifact**

The artifact file is corrupt or not a ZIP. Re-download the artifact or inspect
the build upload step.
