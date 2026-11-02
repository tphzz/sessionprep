# SessionPrep build workflows

All builds are manually dispatched. Nuitka is the only executable builder.

## Entry points

| Actions workflow | Deliverables |
| --- | --- |
| 00 Full - All Platforms | Every platform, one adaptive Windows installer, then an unpublished release draft. |
| 10 Build - Windows x64 | x64 installer and rebuild kit attached to this workflow run. |
| 11 Build - Windows ARM64 | ARM64 installer and rebuild kit attached to this workflow run. |
| 12 Build - macOS | Separate ARM64/Intel DMGs; select both (default), ARM64 or x64. |
| 13 Build - Linux x64 | x64 tarball, DEB and RPM attached to this workflow run. |
| 14 Build - Linux ARM64 | ARM64 tarball, DEB and RPM attached to this workflow run. |

Every entry point offers Debug, default off. Debug retains native symbols and
build reports, attaches an existing Windows GUI console, and never creates a
release draft. All Platforms and macOS also offer Notarize, default off. No ref,
including a tag, implicitly enables notarization. Debug may be notarized when
explicitly selected, while remaining artifact-only.

## Hardware scheduling

All Platforms validates once, then runs two independent lanes:

```text
Linux x64   -> Windows x64
Linux ARM64 -> Windows ARM64 -> macOS ARM64 -> macOS Intel
```

The Windows assembly waits for both Windows architectures. The draft collector
waits for assembly and both macOS packages; transitive dependencies include Linux.
Only the collector has release-write permissions.

Linux runners use self-hosted/Linux/X64 or ARM64/jammy/docker. Windows uses
self-hosted/Windows/X64 or ARM64. Both macOS builds run sequentially on the
self-hosted/macOS/ARM64 runner; Intel uses Intel Python through Rosetta.

The x64 lane and Apple Silicon lane have shared repository-local concurrency
groups across all entry points. Running builds are not automatically cancelled;
`queue: max` preserves waiting manual jobs. These groups do not coordinate other
repositories. Source checkouts, virtual environments and outputs are run-scoped;
cleanup is restricted to those paths and never removes signing state.

## Runner prerequisites

- Linux: native compiler, ccache, patchelf, nfpm, tar, GitHub CLI and unzip.
- Windows: native Python 3.13, Git Bash, Visual Studio 18 C++ tools for the native
  architecture, and Inno Setup 6. CI does not change Defender configuration.
- macOS: Xcode at `/Applications/Xcode.app`, Rosetta for Intel, and the existing
  Developer ID signing setup described in [the macOS guide](../macos/README.md).

Python/uv and dependency setup follow the reusable helpers. Dependencies use the
frozen lockfile. Missing image prerequisites fail with actionable errors rather
than modifying system packages during builds.

## Versions and artifacts

All helpers check out the exact dispatched commit and receive the same validated
version/SHA. Valid version tags produce their normalized version. Version-named
branches retain their base plus a development suffix. Other branches use the
nearest reachable valid version tag, commit distance and SHA; synthetic preview
tags are ignored. Without a version ancestor, the development base is `0.0.0`.

Artifacts are named `sessionprep-<platform>-release` or `-debug`, where platform is
`linux-x64`, `linux-arm64`, `win-x64`, `win-arm64`, `macos-arm64`, or `macos-x64`.
All Platforms adds `sessionprep-win-universal-<configuration>`.

Each artifact includes a checksummed manifest binding its file set to platform,
configuration, version and commit. Windows assembly verifies both payload kits
and native PE architectures before compiling one installer. It selects ARM64 on
ARM64 Windows and x64 otherwise, retaining stable product identity, install
scope, shortcuts and PATH integration. Individual architecture installers remain
available in their run artifacts, alongside manual Certum rebuild inputs.

The collector verifies all seven artifact manifests before staging the nine
end-user downloads: two Linux archives, four Linux packages, two macOS DMGs and
one adaptive Windows installer. Rebuild kits and diagnostics are not release
assets. Windows CI installers remain unsigned; existing manual signing remains
available.

## Release ownership and credentials

Both tag releases and branch previews are **unpublished drafts**. Branch previews
are also marked prerelease. Only non-Debug All Platforms updates them.

One stable, hashed technical tag identifies each exact branch. Cleanup uses its
explicit ownership marker, never a shared version or display title. A release
branch's draft is not removed by another branch or a tag build with the same
version. Older runs cannot overwrite newer previews. Tag mode checks the remote
lightweight/annotated tag's commit again before retirement.

Replacement assets are uploaded and verified on a provisional draft before the
old draft is retired. Every destructive cleanup rechecks draft/tag/body state.
Published releases are protected. If retirement has begun and promotion fails,
the complete provisional draft is retained for review/retry. GitHub does not
provide a single atomic transaction across these operations; changed state or
API errors stop the operation rather than silently accepting it.

Ordinary same-repository publishing uses `GITHUB_TOKEN` with Contents write.
No additional secret is required for the normal/default-branch path. GitHub
requires Workflows write as well if the target differs from the default branch
under `.github/workflows/`; `GITHUB_TOKEN` cannot receive that permission.
For that exceptional case, an optional `SESSIONPREP_RELEASE_TOKEN` secret may
supply a fine-grained token restricted to this repository with Contents write
and Workflows write. Preflight checks this condition before long builds.
No token is exposed to platform builds. See [GitHub's release API rule](https://docs.github.com/en/rest/releases/releases#create-a-release).

See [the publisher CLI](README.md) for isolated local staging/dry-run commands.
