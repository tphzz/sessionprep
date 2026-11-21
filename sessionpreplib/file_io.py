"""Publish complete files without exposing an interrupted write."""
from contextlib import contextmanager
from hashlib import file_digest
import json
import os
from pathlib import Path
import shutil
import stat
from tempfile import NamedTemporaryFile

BACKUP_RECEIPT_SUFFIX = ".sessionprep-backup.json"


def _publish_new_file(temporary, destination):
    if os.name == "nt":
        os.rename(temporary, destination)  # Windows rename fails when the target exists.
    else:
        os.link(temporary, destination)


@contextmanager
def atomic_writer(destination, *, binary=False, replace_existing=True):
    """Close the document before replacing its previous directory entry."""
    destination = Path(destination)
    stream = NamedTemporaryFile(mode="wb" if binary else "w",
                                encoding=None if binary else "utf-8", dir=destination.parent,
                                prefix=".sp-", suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            yield stream
        if os.name != "nt":
            try:
                previous_mode = destination.lstat().st_mode
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISREG(previous_mode):
                    temporary.chmod(stat.S_IMODE(previous_mode))
        if replace_existing:
            os.replace(temporary, destination)
        else:
            _publish_new_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def ensure_audio_backup(source, destination):
    """Verify the first-generation copy and return a hook to bind the next render."""
    source, destination = Path(source), Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    current = sha256_file(source)
    if not destination.exists():
        with NamedTemporaryFile(dir=destination.parent, prefix=".sp-backup-",
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != current:
                raise OSError(f"Incomplete backup of {source}")
            try:
                _publish_new_file(temporary, destination)
            except FileExistsError:
                pass  # A concurrent first writer published its copy.
        finally:
            if os.name == "nt" and temporary.exists():
                temporary.chmod(stat.S_IWRITE)
            temporary.unlink(missing_ok=True)

    if destination.is_symlink() or source.samefile(destination):
        raise ValueError(f"Backup must be a separate audio file: {destination}")
    receipt = destination.with_name(destination.name + BACKUP_RECEIPT_SUFFIX)
    backup_digest = sha256_file(destination)
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Adopt legacy backups only while they can be compared with the original.
        if current != backup_digest:
            raise ValueError(f"Existing backup cannot be verified against the source: {destination}")
    else:
        if (recorded["original"] != backup_digest
                or current not in (backup_digest, recorded["previous"], recorded["rendered"])):
            raise ValueError(f"The source or its backup has been replaced: {destination}")

    def record_render(temporary):
        if destination.is_symlink() or source.samefile(destination) or sha256_file(destination) != backup_digest:
            raise ValueError(f"Backup changed before overwrite: {destination}")
        if sha256_file(source) != current:
            raise ValueError(f"Source changed before overwrite: {source}")
        # Record before publishing. A crash on either side of the rename leaves
        # a recognizable current source and a recoverable first generation.
        with atomic_writer(receipt) as stream:
            json.dump({"original": backup_digest, "previous": current, "rendered": sha256_file(temporary)}, stream)

    return record_render
