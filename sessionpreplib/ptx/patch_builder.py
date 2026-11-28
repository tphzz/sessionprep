"""Fluent transactional batch patcher wrapping validated ``.ptx`` write operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from pathlib import Path
from typing import Any

from .models import PtxGroupAttributes
from .patcher import (
    _validate_patched_ptx,
    apply_atomic_ptx_patch,
    patch_add_group,
    patch_clear_track_inputs,
    patch_clip_gains,
    patch_delete_group,
    patch_modify_group,
    patch_slot_a_trim_gains,
    patch_track_volumes,
)

_UNSET: Any = object()
_StepFn = Callable[[bytes, dict[str, Any]], bytes]


def _empty_summary() -> dict[str, Any]:
    return {
        "cleared_inputs": {},
        "clip_gains": {},
        "slot_a_trims": {},
        "volumes": {},
        "groups": [],
    }


def _merge_track_summary(dest: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, val in updates.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(val, dict):
            merged = dict(val)
            for prev_key in ("previous_db", "previous_gain_db"):
                if prev_key in dest[key]:
                    merged[prev_key] = dest[key][prev_key]
            dest[key] = merged
        else:
            dest[key] = val


class PtxPatchBuilder:
    """Queue multiple validated ``.ptx`` mutations and apply them in one atomic pass."""

    def __init__(self, *, raw_bytes: bytes | None = None, file_path: str | Path | None = None):
        if raw_bytes is None and file_path is None:
            raise ValueError("Provide either raw_bytes or file_path to PtxPatchBuilder")
        if raw_bytes is not None and file_path is not None:
            raise ValueError("Provide either raw_bytes or file_path to PtxPatchBuilder, not both")
        self._raw_bytes = raw_bytes
        self._file_path = Path(file_path) if file_path is not None else None
        self._steps: list[_StepFn] = []

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> PtxPatchBuilder:
        return cls(raw_bytes=raw_bytes)

    @classmethod
    def from_file(cls, file_path: str | Path) -> PtxPatchBuilder:
        return cls(file_path=file_path)

    def __enter__(self) -> PtxPatchBuilder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    @property
    def has_pending_changes(self) -> bool:
        return bool(self._steps)

    def clear_track_inputs(self, track_names: Sequence[str]) -> PtxPatchBuilder:
        names = list(track_names)
        if not names:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_clear_track_inputs(cur, names)
            _merge_track_summary(summary["cleared_inputs"], res)
            return nxt

        self._steps.append(_step)
        return self

    def set_clip_gains(self, track_clip_gains_db: Mapping[str, float]) -> PtxPatchBuilder:
        gains = dict(track_clip_gains_db)
        if not gains:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_clip_gains(cur, gains)
            _merge_track_summary(summary["clip_gains"], res)
            return nxt

        self._steps.append(_step)
        return self

    def set_slot_a_trim_gains(
        self,
        track_trims_db: Mapping[str, float],
        *,
        activate_if_inactive: bool = True,
        unbypass_if_bypassed: bool = False,
        allow_auto_insert: bool = True,
    ) -> PtxPatchBuilder:
        trims = dict(track_trims_db)
        if not trims:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_slot_a_trim_gains(
                cur,
                trims,
                activate_if_inactive=activate_if_inactive,
                unbypass_if_bypassed=unbypass_if_bypassed,
                allow_auto_insert=allow_auto_insert,
            )
            _merge_track_summary(summary["slot_a_trims"], res)
            return nxt

        self._steps.append(_step)
        return self

    def set_track_volumes(self, track_volumes_db: Mapping[str, float]) -> PtxPatchBuilder:
        vols = dict(track_volumes_db)
        if not vols:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_track_volumes(cur, vols)
            _merge_track_summary(summary["volumes"], res)
            return nxt

        self._steps.append(_step)
        return self

    def add_group(
        self,
        name: str,
        member_tracks: Sequence[Any],
        *,
        vca_track: Any | None = None,
        group_type: str = "Mix/Edit",
        id_letter: str | None = None,
        is_edit_active: bool = True,
        is_mix_active: bool = True,
        follow_globals: bool = True,
        include_rulers: bool = False,
        attributes: PtxGroupAttributes | None = None,
        color_index: int = 0xFFFE,
    ) -> PtxPatchBuilder:
        members = list(member_tracks)
        attrs_copy = copy.deepcopy(attributes) if attributes is not None else None

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_add_group(
                cur, name, members, vca_track=vca_track, group_type=group_type,
                id_letter=id_letter, is_edit_active=is_edit_active, is_mix_active=is_mix_active,
                follow_globals=follow_globals, include_rulers=include_rulers,
                attributes=attrs_copy, color_index=color_index,
            )
            summary["groups"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def modify_group(
        self,
        name: str,
        *,
        new_name: str | None = None,
        member_tracks: Sequence[Any] | None = None,
        add_tracks: Sequence[Any] | None = None,
        remove_tracks: Sequence[Any] | None = None,
        vca_track: Any = _UNSET,
        group_type: str | None = None,
        is_edit_active: bool | None = None,
        is_mix_active: bool | None = None,
        follow_globals: bool | None = None,
        include_rulers: bool | None = None,
    ) -> PtxPatchBuilder:
        m_tracks = list(member_tracks) if member_tracks is not None else None
        a_tracks = list(add_tracks) if add_tracks is not None else None
        r_tracks = list(remove_tracks) if remove_tracks is not None else None

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            if vca_track is _UNSET:
                nxt, info = patch_modify_group(
                    cur, name, new_name=new_name, member_tracks=m_tracks,
                    add_tracks=a_tracks, remove_tracks=r_tracks, group_type=group_type,
                    is_edit_active=is_edit_active, is_mix_active=is_mix_active,
                    follow_globals=follow_globals, include_rulers=include_rulers,
                )
            else:
                nxt, info = patch_modify_group(
                    cur, name, new_name=new_name, member_tracks=m_tracks,
                    add_tracks=a_tracks, remove_tracks=r_tracks, vca_track=vca_track,
                    group_type=group_type, is_edit_active=is_edit_active, is_mix_active=is_mix_active,
                    follow_globals=follow_globals, include_rulers=include_rulers,
                )
            summary["groups"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def delete_group(self, name: str) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_delete_group(cur, name)
            summary["groups"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def _run_transform(self, raw: bytes) -> tuple[bytes, dict[str, Any]]:
        cur = raw
        summary = _empty_summary()
        for step in self._steps:
            cur = step(cur, summary)
        return cur, summary

    def build(self) -> tuple[bytes, dict[str, Any]]:
        """Apply queued patches in memory, validate structure, and return ``(patched_bytes, summary)``."""
        if self._raw_bytes is not None:
            source = self._raw_bytes
        elif self._file_path is not None:
            source = self._file_path.read_bytes()
        else:
            raise ValueError("PtxPatchBuilder has neither raw_bytes nor file_path")
        if not self._steps:
            return source, _empty_summary()
        patched, summary = self._run_transform(source)
        _validate_patched_ptx(source, patched)
        return patched, summary

    def commit(self) -> tuple[bytes, dict[str, Any]]:
        """Atomically apply queued patches to ``file_path``, clear the queue, and return ``(original_bytes, summary)``."""
        if self._file_path is None:
            raise ValueError("commit() requires PtxPatchBuilder.from_file(path); use build() for in-memory bytes")
        if not self._steps:
            return self._file_path.read_bytes(), _empty_summary()
        result = apply_atomic_ptx_patch(self._file_path, self._run_transform)
        self._steps.clear()
        return result
