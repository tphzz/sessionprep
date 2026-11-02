# GitHub draft-release publisher

SessionPrep's [manual workflow guide](README.sessionprep.md) describes the
self-hosted build family, hardware lanes and All-Platforms-only release policy.

## Tools

- `build.py`: CI ref/version validation, native build/package orchestration,
  checksummed manifests and complete multi-platform collection.
- `publish-draft-release.sh`: portable Bash CLI for artifact download, allowlist
  filtering and staging; calls the adjacent Python publisher.
- `release_policy.py`: exact ownership, stale-run protection, remote tag checks,
  verified provisional drafts and guarded replacement.
- `release-assets.allowlist`: end-user filename patterns; intermediate payloads,
  rebuild kits and reports are excluded.

Keep the Bash entry point, Python helper and allowlist together when copying the
publisher. It requires Bash, Python 3, GitHub CLI, curl, unzip and ordinary POSIX
file utilities. `SESSIONPREP_RELEASE_PYTHON` can select a specific interpreter.

## CLI

Run from the repository root. All options are listed by `--help`.

```bash
bash packaging/github/publish-draft-release.sh \
  --artifact-root downloaded-artifacts \
  --staging-dir release-assets \
  --mode branch --ref-name feature/audio \
  --target COMMIT_SHA --repo OWNER/REPO \
  --producer-run-id RUN_ID --dry-run
```

Replace `COMMIT_SHA`, `OWNER/REPO` and `RUN_ID` with the actual build identity.
Dry-run stages allowlisted files and stops without release writes. Existing
local artifacts need no GitHub authentication; `--download-run-id` downloads
first and therefore requires authenticated read access.

| Option | Contract |
| --- | --- |
| `--artifact-root DIR` | Existing artifact tree, or download destination. |
| `--download-run-id ID` / `--artifact-pattern GLOB` | Optional artifact download from a specific workflow run. |
| `--mode branch|tag` / `--ref-name NAME` | Exact branch ownership or an existing real version tag. |
| `--target SHA` | Built commit; remote tag resolution must match it in tag mode. |
| `--repo OWNER/REPO` | Repository containing the artifacts/releases. |
| `--allowlist FILE` | Adjacent default allowlist or a custom basename-glob list. |
| `--staging-dir DIR` | Selected output; may not overlap artifacts or the working directory. |
| `--release-tag TAG` / `--release-title TITLE` | Optional technical tag/title override; ownership checks still apply. |
| `--producer-run-id ID` | Producing run ID; defaults to `GITHUB_RUN_ID`, otherwise zero. |
| `--dry-run` | Stage without release API writes. |

The generic CLI checks allowlisted files and duplicate basenames. SessionPrep's
All Platforms first applies the stronger complete-manifest collector; ordinary
individual workflow artifacts must not be passed directly as a complete release.

## Replacement and failure handling

Branch tags combine a readable slug with a hash of the exact branch name.
Legacy drafts with the same explicit branch ownership marker are recognized;
matching titles or version numbers alone never authorize deletion. Published
releases and a newer producer's preview are preserved.

Tag mode peels lightweight/annotated remote tags and checks the built commit,
including immediately before retirement. It preserves the real tag and unrelated
branch drafts. Both modes create unpublished drafts; branch drafts are prereleases.

Assets are uploaded to a provisional release ID and checked by name/count/size.
The original draft is retained until those checks pass. Fresh draft/tag/body
checks precede deletion and promotion. Failed preparation cleans up only its own
unchanged unpublished candidate. Once retirement is attempted, a verified
candidate is retained after any failure so it remains recoverable. Human changes
or ambiguous promotion results are left for review, not automatically erased.
A subsequent successful run can consolidate owned stale candidates.

The built-in Actions token suffices for ordinary same-repository targets with
Contents write. Targets that modify workflows relative to the default branch
also need Workflows write, which the built-in token cannot grant; SessionPrep
provides an optional release-secret hook and early preflight for that case.
See [the current GitHub API requirement](https://docs.github.com/en/rest/releases/releases#create-a-release).

Credentials are never embedded in artifact files or upload command arguments.
All publisher tests use fake API/upload endpoints; no real releases are created.
