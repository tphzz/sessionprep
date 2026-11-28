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
from .patcher_extended import (
    patch_add_fade,
    patch_add_marker,
    patch_clip_attributes,
    patch_delete_marker,
    patch_rename_clips,
    patch_rename_tracks,
    patch_session_settings,
    patch_track_states,
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
        "session_settings": {},
        "track_states": {},
        "renamed_tracks": {},
        "renamed_clips": {},
        "clip_attributes": [],
        "fades": [],
        "markers": [],
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

    def set_session_settings(
        self,
        *,
        sample_rate: int | None = None,
        bit_depth: int | None = None,
        audio_format: str | None = None,
        is_interleaved: bool | None = None,
        audio_pull_rate: str | None = None,
        video_pull_rate: str | None = None,
        fps_code: int | None = None,
        start_frames: int | None = None,
        feet_frames_rate: str | None = None,
        session_length: str | None = None,
        main_counter_format: str | None = None,
        sub_counter_format: str | None = None,
        edit_mode: str | None = None,
        grid_relative: bool | None = None,
        edit_tool: str | None = None,
        smart_tool_enabled: bool | None = None,
        pre_roll_enabled: bool | None = None,
        pre_roll_samples: int | None = None,
        post_roll_enabled: bool | None = None,
        post_roll_samples: int | None = None,
        timeline_in_sample: int | None = None,
        timeline_out_sample: int | None = None,
    ) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_session_settings(
                cur,
                sample_rate=sample_rate, bit_depth=bit_depth, audio_format=audio_format,
                is_interleaved=is_interleaved, audio_pull_rate=audio_pull_rate,
                video_pull_rate=video_pull_rate, fps_code=fps_code, start_frames=start_frames,
                feet_frames_rate=feet_frames_rate, session_length=session_length,
                main_counter_format=main_counter_format, sub_counter_format=sub_counter_format,
                edit_mode=edit_mode, grid_relative=grid_relative, edit_tool=edit_tool,
                smart_tool_enabled=smart_tool_enabled, pre_roll_enabled=pre_roll_enabled,
                pre_roll_samples=pre_roll_samples, post_roll_enabled=post_roll_enabled,
                post_roll_samples=post_roll_samples, timeline_in_sample=timeline_in_sample,
                timeline_out_sample=timeline_out_sample,
            )
            summary["session_settings"].update(res)
            return nxt

        self._steps.append(_step)
        return self

    def set_track_states(self, track_states: Mapping[str, Mapping[str, Any]]) -> PtxPatchBuilder:
        copied = {k: dict(v) for k, v in track_states.items()}
        if not copied:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_track_states(cur, copied)
            for k, v in res.items():
                summary["track_states"].setdefault(k, {}).update(v)
            return nxt

        self._steps.append(_step)
        return self

    def rename_tracks(self, renames: Mapping[str, str]) -> PtxPatchBuilder:
        copied = dict(renames)
        if not copied:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_rename_tracks(cur, copied)
            summary["renamed_tracks"].update(res)
            return nxt

        self._steps.append(_step)
        return self

    def rename_clips(self, renames: Mapping[str, str]) -> PtxPatchBuilder:
        copied = dict(renames)
        if not copied:
            return self

        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, res = patch_rename_clips(cur, copied)
            summary["renamed_clips"].update(res)
            return nxt

        self._steps.append(_step)
        return self

    def set_clip_attributes(
        self,
        track_name: str,
        start_sample: int,
        *,
        is_muted: bool | None = None,
        is_time_locked: bool | None = None,
        is_edit_locked: bool | None = None,
        color_index: int | None = None,
    ) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_clip_attributes(
                cur, track_name, start_sample,
                is_muted=is_muted, is_time_locked=is_time_locked,
                is_edit_locked=is_edit_locked, color_index=color_index,
            )
            summary["clip_attributes"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def add_fade(
        self,
        track_name: str,
        fade_type: str,
        start_sample: int,
        total_samples: int,
        *,
        pre_samples: int | None = None,
        shape: str = "standard",
        crossfade_link: str = "equal_power",
    ) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_add_fade(
                cur, track_name, fade_type, start_sample, total_samples,
                pre_samples=pre_samples, shape=shape, crossfade_link=crossfade_link,
            )
            summary["fades"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def add_marker(
        self,
        name: str,
        start_pos: int,
        *,
        number: int | None = None,
        end_pos: int | None = None,
        reference: str = "Absolute",
        time_properties: str = "Marker",
        comments: str = "",
        color_index: int = 0,
        color_rgb: tuple[int, int, int] | None = None,
        track: str | int | None = None,
        zoom_settings: bool = False,
        pre_post_roll_times: bool = False,
        track_visibility: bool = False,
        track_heights: bool = False,
        group_enables: bool = False,
        window_configuration: bool = False,
    ) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_add_marker(
                cur, name, start_pos,
                number=number, end_pos=end_pos, reference=reference,
                time_properties=time_properties, comments=comments,
                color_index=color_index, color_rgb=color_rgb, track=track,
                zoom_settings=zoom_settings, pre_post_roll_times=pre_post_roll_times,
                track_visibility=track_visibility, track_heights=track_heights,
                group_enables=group_enables, window_configuration=window_configuration,
            )
            summary["markers"].append(info)
            return nxt

        self._steps.append(_step)
        return self

    def delete_marker(self, number: int) -> PtxPatchBuilder:
        def _step(cur: bytes, summary: dict[str, Any]) -> bytes:
            nxt, info = patch_delete_marker(cur, number)
            summary["markers"].append(info)
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
