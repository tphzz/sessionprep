"""Git-derived SessionPrep version helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    InvalidVersion = ValueError
    Version = None


_DIST_NAME = "sessionprep"
_UNKNOWN_VERSION = "0.0.0+unknown"
_FALLBACK_VERSION_RE = re.compile(
    r"^v?\d+(?:\.\d+)*"
    r"(?:(?:a|b|rc|post|dev)\d*)?"
    r"(?:\.post\d+)?"
    r"(?:\.dev\d+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?$",
    re.IGNORECASE,
)


class VersionResolutionError(RuntimeError):
    """Raised when a build-time version cannot be resolved."""


def _normalize_version(value: str, *, source: str) -> str:
    text = value.strip()
    if not text:
        raise VersionResolutionError(f"Empty version from {source}")
    if Version is None:
        normalized = text[1:] if text[:1].lower() == "v" else text
        if _FALLBACK_VERSION_RE.fullmatch(text):
            return normalized
        raise VersionResolutionError(
            f"{source} value {text!r} is not a valid PEP 440 version"
        )
    try:
        return str(Version(text))
    except InvalidVersion as exc:
        raise VersionResolutionError(
            f"{source} value {text!r} is not a valid PEP 440 version"
        ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_compiled_runtime() -> bool:
    """Return True when running from a freezer/compiled app runtime."""
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def _hidden_subprocess_kwargs() -> dict:
    """Return Windows flags that prevent helper subprocess windows flashing."""
    if sys.platform != "win32":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


def _git(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            **_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value


def _build_version() -> str | None:
    loaded_module = sys.modules.get("sessionpreplib._build_version")
    BUILD_VERSION = getattr(loaded_module, "BUILD_VERSION", None)
    if BUILD_VERSION is None:
        BUILD_VERSION = _build_version_from_file()
    if BUILD_VERSION is None:
        try:
            from sessionpreplib._build_version import BUILD_VERSION
        except ImportError:
            BUILD_VERSION = None
    if BUILD_VERSION is None:
        return None
    return _normalize_version(str(BUILD_VERSION), source="_build_version.py")


def _build_version_from_file() -> str | None:
    path = Path(__file__).with_name("_build_version.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "sessionprep_generated_build_version", path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BUILD_VERSION = getattr(module, "BUILD_VERSION", None)
    return BUILD_VERSION


def resolve_ref_version(ref_type: str, ref_name: str, commit: str) -> str:
    """Resolve an explicit build ref, including a detached CI checkout.

    Version branches retain their declared base. Other branches use their
    nearest reachable version tag; synthetic preview tags are never bases.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise VersionResolutionError("Build commit must be a Git commit hash")
    if ref_type == "tag":
        return _normalize_version(ref_name, source="git tag")
    if ref_type != "branch" or not ref_name.strip():
        raise VersionResolutionError("Build ref must be a named branch or version tag")
    distance = "0"
    try:
        base = _normalize_version(ref_name, source="git branch")
    except VersionResolutionError:
        root = _repo_root()
        versions = {}
        reachable = _git(["tag", "--merged", commit], cwd=root)
        if reachable is None:
            raise VersionResolutionError("Could not inspect reachable version tags")
        for tag in reachable.splitlines():
            try:
                versions[tag] = _normalize_version(tag, source="ancestor tag")
            except VersionResolutionError:
                continue
        base = "0.0.0"
        if versions:
            args = ["describe", "--tags", "--abbrev=0", f"--candidates={len(versions)}"]
            for tag in versions:
                args.extend(["--match", tag])
            nearest = _git([*args, commit], cwd=root)
            if nearest not in versions:
                raise VersionResolutionError("Could not resolve the nearest version tag")
            base = versions[nearest]
            distance = _git(["rev-list", "--count", f"{nearest}..{commit}"], cwd=root)
            if distance is None or not distance.isdigit():
                raise VersionResolutionError("Could not count commits since the version tag")
    # Do not stack a second dev/local suffix on an already-development base.
    base = base.split("+", 1)[0].split(".dev", 1)[0]
    return _normalize_version(
        f"{base}.dev{distance}+g{commit[:7].lower()}", source="git branch and commit")


def _git_version(*, strict: bool) -> str | None:
    root = _repo_root()
    if os.environ.get("GITHUB_ACTIONS") == "true" and (root / ".git").exists():
        # checkout pins a commit, so branch --show-current is empty in CI.
        # Validate the checkout before using the event ref during package setup.
        commit = _git(["rev-parse", "HEAD"], cwd=root)
        expected = os.environ.get("GITHUB_SHA", "")
        if (not re.fullmatch(r"[0-9a-fA-F]{40}", expected)
                or commit is None or commit.lower() != expected.lower()):
            raise VersionResolutionError("GitHub Actions SHA does not match the source checkout")
        ref_type = os.environ.get("GITHUB_REF_TYPE", "")
        ref_name = os.environ.get("GITHUB_REF_NAME", "")
        if ref_type == "tag":
            tagged = _git(["rev-parse", "--verify", f"refs/tags/{ref_name}^{{commit}}"], cwd=root)
            if tagged != commit:
                raise VersionResolutionError("GitHub Actions tag does not match the source checkout")
        # Invalid CI identity must not silently reuse stale generated/installed metadata.
        return resolve_ref_version(ref_type, ref_name, commit)
    invalid_exact_tag_error: VersionResolutionError | None = None

    tag = _git(["describe", "--tags", "--exact-match", "HEAD"], cwd=root)
    if tag:
        try:
            return _normalize_version(tag, source="git tag")
        except VersionResolutionError as exc:
            invalid_exact_tag_error = exc

    branch = _git(["branch", "--show-current"], cwd=root)
    commit = _git(["rev-parse", "--short=7", "HEAD"], cwd=root)
    if branch and commit:
        try:
            return resolve_ref_version("branch", branch, commit)
        except VersionResolutionError:
            if strict:
                raise
            return None

    if invalid_exact_tag_error is not None and strict:
        raise invalid_exact_tag_error

    if strict:
        raise VersionResolutionError(
            "Could not resolve version from an exact Git tag or branch"
        )
    return None


def _metadata_version() -> str | None:
    try:
        return _normalize_version(
            importlib.metadata.version(_DIST_NAME),
            source="installed package metadata",
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def get_version(*, strict: bool = False) -> str:
    """Return the SessionPrep version.

    Source runs prefer live Git metadata. Compiled/frozen runs prefer the
    generated build-version module and never shell out to Git. A packaged app
    must not trigger platform developer-tool prompts on end-user machines.
    """
    if _is_compiled_runtime():
        version = (
            _build_version()
            or _metadata_version()
        )
    else:
        version = (
            _git_version(strict=False)
            or _build_version()
            or _metadata_version()
        )

    if version:
        return version
    if strict:
        return _raise_unknown_version()
    return _UNKNOWN_VERSION


def _raise_unknown_version() -> str:
    raise VersionResolutionError(
        "Could not resolve version from generated build metadata, installed "
        "package metadata, or source Git metadata"
    )


__version__ = get_version()
