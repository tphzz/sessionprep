"""Commit a rendered Track Layout after every encoder has closed successfully."""
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

from .audio import AUDIO_EXTENSIONS, file_revision
from .file_io import atomic_writer, sha256_file


def _check_output_path(path, root):
    if (path.suffix.lower() not in AUDIO_EXTENSIONS or path.is_symlink()
            or not path.resolve().is_relative_to(root)):
        raise ValueError(f"Invalid layout destination: {path}")


def publish_layout(root: Path, staged: Path, outputs, previous):
    """Replace the selected outputs and retire only identified older renders."""
    # Coordinate independent windows and processes. Closing the descriptor
    # releases the OS lock even if the process exits during an Apply.
    with (root / ".sessionprep-layout.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError(f"Cannot lock the layout directory for Apply: {root}") from exc
        _publish(root, staged, outputs, previous)


def _publish(root, staged, outputs, previous):
    targets = {Path(t.filepath): t for t in outputs}
    receipt = root / ".sessionprep-layout.json"
    try:
        known = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
        if not isinstance(known, dict):
            raise ValueError(f"Invalid layout receipt: {receipt}")
    except FileNotFoundError:
        known = {}
    prior = {Path(t.filepath): t.source_file_revision for t in previous
             if Path(t.filepath) == root / t.filename and t.source_file_revision}
    existing = [root / name for name in known]
    # A legacy saved session can identify active and obsolete layout files.
    existing.extend(prior)

    recovery = Path(mkdtemp(dir=root, prefix=".sp-layout-previous-"))
    saved = {}
    installed = []
    hashes = {}
    rollback_failed = False
    try:
        for index, path in enumerate(dict.fromkeys([*targets, *existing])):
            _check_output_path(path, root)
            name = path.relative_to(root).as_posix()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"Layout destination is not a file: {path}")
                if name in known:
                    owned = sha256_file(path) == known[name]
                else:
                    owned = path in prior and file_revision(path, path.stat()) == prior[path]
                if not owned:
                    if path in targets:
                        raise ValueError(f"Layout file already exists and could not be verified: {path}. "
                                         "Move it aside or choose a different layout output directory.")
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
                targets[path].source_file_revision = file_revision(path, info)
        with atomic_writer(receipt) as stream:
            json.dump({"sha256": hashes}, stream)
    except BaseException:
        try:
            for path in reversed(installed):
                if path not in saved:
                    path.unlink()
            for path, old in reversed(saved.items()):
                os.replace(old, path)
        except OSError as exc:
            rollback_failed = True
            raise OSError(f"Layout recovery incomplete. Prior files remain in {recovery}: {exc}") from exc
        raise
    finally:
        # A failed rollback leaves its backups available for recovery.
        if not rollback_failed:
            try:
                shutil.rmtree(recovery)
            except OSError as exc:
                logging.getLogger(__name__).warning("Could not clean up layout backups in %s: %s", recovery, exc)
