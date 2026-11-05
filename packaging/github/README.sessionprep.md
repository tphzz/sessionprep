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
Windows always-run cleanup uses PowerShell to invoke the existing guarded
Python cleanup directly. It can use Python on PATH when native validation
fails before exporting CI_PYTHON, without depending on Git Bash setup.
Cleanup is architecture-independent; the build still requires native Python.

## Runner prerequisites

- Linux: native compiler, ccache, patchelf, nfpm, tar, GitHub CLI, unzip,
  and the PortAudio runtime (`libportaudio2` on Ubuntu/Debian).
- Windows: native Python 3.13, Git Bash, Visual Studio 18 C++ tools for the native
  architecture, and Inno Setup 6 or 7. CI does not change Defender configuration.
- macOS: Xcode at `/Applications/Xcode.app`, Rosetta for Intel, and the existing
  Developer ID signing setup described in [the macOS guide](../macos/README.md).

Python/uv and dependency setup follow the reusable helpers. Dependencies use the
frozen lockfile. Missing image prerequisites fail with actionable errors rather
than modifying system packages during builds.

SoundFile must be at least 0.14.0: its Windows ARM64 wheel bundles the native
libsndfile DLL. The locked platform wheels and Nuitka's existing `_soundfile_data`
DLL rule provide this dependency; a separate Windows libsndfile installation is
not needed with the supported wheel.

Windows Python selection uses an explicit runner `CI_PYTHON` when set and rejects
an invalid override without falling back. Otherwise the shared PowerShell helper
checks each `python.exe` on PATH, then standard CPython 3.13 machine/per-user
installer locations (`Python313` for x64, `Python313-arm64` for ARM64). It skips
Windows Store execution aliases. Each candidate must pass the existing native
Python guard before dependency setup, MSVC activation or environment export:
`win-amd64` for x64 or `win-arm64` for ARM64, running Python 3.13. An x64 interpreter
under emulation does not satisfy an ARM64 build. Use `CI_PYTHON` for a custom
installation path; the scripts do not install or replace Python. The standard
locations follow the [CPython full installer](https://github.com/python/cpython/blob/3.13/Tools/msi/bundle/bundle.wxs).

Installer discovery prefers `ISCC_PATH` (the full `ISCC.exe` path), then the
legacy `ISCC` override, PATH, and Inno Setup 6/7 machine/per-user locations.
An invalid explicit override fails rather than selecting a different compiler.
Configure these variables for the runner account; restart its service after
changing inherited environment variables. Verify discovery without building:

```powershell
python packaging/windows/build_installer.py check
```

Windows native compilation checks installer discovery before running Nuitka.
The adaptive installer also checks it before staging/extracting payloads.

Windows CI uses Nuitka's `--low-memory` mode for both executables and both build
configurations. This selects one C compiler job and reduces peak memory use when
compiling SciPy with MSVC on shared workers; compilation can take longer. Local
builds can use `uv run python build_nuitka.py all --low-memory`. See
[Nuitka's documented memory controls](https://nuitka.net/user-documentation/common-issue-solutions.html#memory-issues-and-compiler-bugs).

The locked PySide6 6.10.2 ARM64 Linux wheels require glibc 2.39 or newer, so that
worker needs Ubuntu 24.04 (Noble) or a compatible newer runtime. x64 wheels work
with the existing Jammy baseline. Runner labels identify scheduling targets;
verify the actual running image with `cat /etc/os-release` and
`getconf GNU_LIBC_VERSION` inside the container. Rebuilding a Noble image does not
change a Compose service still referencing the old Jammy image: update that
service's `image:` reference and recreate it from the intended image.

Linux GUI test imports and builds require the real PortAudio shared library.
Installing the Python `sounddevice` package does not supply it on Linux; see
[its installation guide](https://python-sounddevice.readthedocs.io/en/0.5.5/installation.html).
Add `libportaudio2` to the existing apt package list in **both** x64 and ARM64
runner images, rebuild those images, then recreate the runner containers from
the updated images. Installing on the Docker host does not provision a container;
restarting an existing container does not update its image. Provision the
Ubuntu build/runtime package set together, for example in the existing apt layer:

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ccache patchelf libatomic1 libportaudio2 \
        git gh curl ca-certificates tar unzip \
    && rm -rf /var/lib/apt/lists/*
```

`ccache` is required by these CI helpers to retain compiler caching on the shared
workers; local Nuitka builds can run without it. Install `nfpm` separately using
[its official installation instructions](https://nfpm.goreleaser.com/docs/install/)
for the image's native architecture. The apt package list above does not install
nFPM. It is needed to create the DEB/RPM deliverables.

After provisioning all tools, verify the image before starting runner containers:

```dockerfile
RUN for tool in gcc ccache patchelf nfpm tar gh unzip; do \
        command -v "$tool" || exit 1; \
    done
```

Compilation preflight reports every missing required command in one error and
starts no compiler until all are present. It does not install missing tools.

The metadata command checks that PortAudio can be located and loaded before
writing success outputs or collecting GUI tests; compilation repeats the check
on its own worker. No audio device or playback is required by this check. A
missing/broken library fails with an image-provisioning instruction. It does not
skip tests, install packages or alter playback behavior.

## Versions and artifacts

All helpers check out the exact dispatched commit and receive the same validated
version/SHA. Valid version tags produce their normalized version. Version-named
branches retain their base plus a development suffix. Other branches use the
nearest reachable valid version tag, commit distance and SHA; synthetic preview
tags are ignored. Without a version ancestor, the development base is `0.0.0`.

Package setup validates GitHub's built-in ref/SHA variables against the source
checkout, so the first frozen `uv sync` works with detached HEAD and no generated
version file. CI branch and tag versions use the same resolver as the explicit
metadata command. Invalid CI identity fails before stale metadata can be reused.

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
