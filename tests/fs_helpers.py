"""Deterministic filesystem test helpers for cross-platform revision checks."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import time

from sessionpreplib.audio import file_revision


def rewrite_past_revision_tick(
    path: Path | str,
    write_fn: Callable[[], object],
    *,
    timeout_s: float = 2.0,
) -> None:
    """Rewrite *path* in place until its filesystem revision advances past a coarse clock tick."""
    target = Path(path)
    before_rev = file_revision(str(target), target.stat())
    deadline = time.monotonic() + timeout_s
    while True:
        write_fn()
        if file_revision(str(target), target.stat()) != before_rev:
            return
        assert time.monotonic() < deadline, f"file_revision did not advance for {target}"
        time.sleep(0.005)


def rewrite_preserving_mtime(
    path: Path | str,
    write_fn: Callable[[], object],
    *,
    mtime_offset_ns: int = 0,
    assert_same_size: bool = True,
    timeout_s: float = 2.0,
) -> None:
    """Rewrite *path* in place, set mtime with *mtime_offset_ns*, and advance revision when mtime is unchanged."""
    target = Path(path)
    before = target.stat()
    before_rev = file_revision(str(target), before)
    deadline = time.monotonic() + timeout_s
    while True:
        write_fn()
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + mtime_offset_ns))
        after = target.stat()
        if mtime_offset_ns != 0 or file_revision(str(target), after) != before_rev:
            if assert_same_size:
                assert after.st_size == before.st_size
            if mtime_offset_ns == 0:
                assert after.st_mtime_ns == before.st_mtime_ns
            return
        assert time.monotonic() < deadline, f"file_revision did not advance for {target}"
        time.sleep(0.005)
