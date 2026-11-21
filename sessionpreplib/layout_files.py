"""Commit a rendered Track Layout after every encoder has closed successfully."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from typing import Any

from .audio import AUDIO_EXTENSIONS, file_revision
from .file_io import atomic_writer, sha256_file


class UnverifiedLayoutFilesError(ValueError):
    """Raised when target layout files exist on disk without verified ownership."""

    def __init__(self, root: Path, unverified: dict[str, str]) -> None:
        self.root = root
        self.unverified = dict(unverified)
        first_name = next(iter(self.unverified))
        first_path = root / first_name
        super().__init__(
            f"Layout file already exists and could not be verified: {first_path}. "
            "Move it aside or choose a different layout output directory."
        )


def _check_output_path(path: Path, root: Path) -> None:
    if (
        path.suffix.lower() not in AUDIO_EXTENSIONS
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError(f"Invalid layout destination: {path}")


def _load_known_hashes(root: Path) -> dict[str, str]:
    receipt = root / ".sessionprep-layout.json"
    try:
        known = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
    except FileNotFoundError:
        return {}
    if not isinstance(known, dict):
        raise ValueError(f"Invalid layout receipt: {receipt}")
    return known


def _build_prior_revisions(root: Path, previous: Iterable[Any]) -> dict[Path, str]:
    prior: dict[Path, str] = {}
    for track in previous:
        if not track.source_file_revision:
            continue
        expected = (root / track.filename).resolve()
        try:
            resolved = Path(track.filepath).resolve()
        except OSError:
            continue
        if resolved == expected:
            prior[expected] = track.source_file_revision
    return prior


def _file_rev(path: Path, info: os.stat_result) -> str:
    rev = file_revision(str(path), info)
    return rev or ""


def _is_owned_existing_file(
    path: Path, name: str, known: dict[str, str], prior: dict[Path, str]
) -> bool:
    if not path.is_file():
        raise ValueError(f"Layout destination is not a file: {path}")
    if name in known:
        return sha256_file(path) == known[name]
    return path in prior and _file_rev(path, path.stat()) == prior[path]


def find_unverified_layout_files(
    root: Path, target_filenames: Iterable[str], previous: Iterable[Any]
) -> dict[str, str]:
    """Return ``{rel_posix: file_revision}`` for existing unverified target files."""
    resolved_root = Path(root).resolve()
    if not resolved_root.exists():
        return {}
    known = _load_known_hashes(resolved_root)
    prior = _build_prior_revisions(resolved_root, previous)
    unverified: dict[str, str] = {}
    for filename in dict.fromkeys(target_filenames):
        path = resolved_root / filename
        _check_output_path(path, resolved_root)
        if not path.exists():
            continue
        name = path.relative_to(resolved_root).as_posix()
        if not _is_owned_existing_file(path, name, known, prior):
            unverified[name] = _file_rev(path, path.stat())
    return unverified


def publish_layout(
    root: Path,
    staged: Path,
    outputs: Iterable[Any],
    previous: Iterable[Any],
    *,
    allowed_unverified: Mapping[str, str] | None = None,
) -> None:
    """Replace the selected outputs and retire only identified older renders."""
    resolved_root = Path(root).resolve()
    # Coordinate independent windows and processes. Closing the descriptor
    # releases the OS lock even if the process exits during an Apply.
    with (resolved_root / ".sessionprep-layout.lock").open("a+b") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError(f"Cannot lock the layout directory for Apply: {resolved_root}") from exc
        _publish(resolved_root, staged, outputs, previous, allowed_unverified=allowed_unverified)


def _allow_unverified_target(
    path: Path, name: str, allowed_unverified: Mapping[str, str] | None
) -> bool:
    if allowed_unverified is None:
        return False
    expected = allowed_unverified.get(name)
    if not expected:
        return False
    actual = _file_rev(path, path.stat())
    return bool(actual) and actual == expected


def _rollback_publish(
    installed: list[Path], saved: dict[Path, Path], recovery: Path
) -> None:
    try:
        for path in reversed(installed):
            if path not in saved:
                path.unlink()
        for path, old in reversed(saved.items()):
            os.replace(old, path)
    except OSError as exc:
        raise OSError(
            f"Layout recovery incomplete. Prior files remain in {recovery}: {exc}"
        ) from exc


def _publish(
    root: Path,
    staged: Path,
    outputs: Iterable[Any],
    previous: Iterable[Any],
    *,
    allowed_unverified: Mapping[str, str] | None = None,
) -> None:
    targets: dict[Path, Any] = {}
    for track in outputs:
        raw_path = Path(track.filepath)
        _check_output_path(raw_path, root)
        targets[raw_path.resolve()] = track
    known = _load_known_hashes(root)
    prior = _build_prior_revisions(root, previous)
    existing = [root / name for name in known]
    # A legacy saved session can identify active and obsolete layout files.
    existing.extend(prior)

    recovery = Path(mkdtemp(dir=root, prefix=".sp-layout-previous-"))
    saved: dict[Path, Path] = {}
    installed: list[Path] = []
    hashes: dict[str, str] = {}
    rollback_failed = False
    try:
        for index, path in enumerate(dict.fromkeys([*targets, *existing])):
            _check_output_path(path, root)
            name = path.relative_to(root).as_posix()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                owned = _is_owned_existing_file(path, name, known, prior)
                if not owned and not (path in targets and _allow_unverified_target(path, name, allowed_unverified)):
                    if path in targets:
                        raise UnverifiedLayoutFilesError(root, {name: _file_rev(path, path.stat())})
                    continue
                old = recovery / str(index)
                os.replace(path, old)
                saved[path] = old
            if path in targets:
                rendered = staged / targets[path].filename
                hashes[name] = sha256_file(rendered)
                os.replace(rendered, path)
                installed.append(path)
                info = path.stat()
                targets[path].source_file_signature = (info.st_size, info.st_mtime_ns)
                targets[path].source_file_revision = _file_rev(path, info)
        with atomic_writer(root / ".sessionprep-layout.json") as stream:
            json.dump({"sha256": hashes}, stream)
    except BaseException:
        try:
            _rollback_publish(installed, saved, recovery)
        except OSError:
            rollback_failed = True
            raise
        raise
    finally:
        # A failed rollback leaves its backups available for recovery.
        if not rollback_failed:
            try:
                shutil.rmtree(recovery)
            except OSError as exc:
                logging.getLogger(__name__).warning("Could not clean up layout backups in %s: %s", recovery, exc)
