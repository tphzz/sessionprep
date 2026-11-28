"""Fluent transactional batch patcher wrapping validated ``.ptx`` write operations on a single ``PtxBuffer``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from pathlib import Path
from typing import Any

from ._document import PtxBuffer, validate_decrypted_ptx_pair
from .codec import atomic_write_bytes
from .models import (
    PtxClipAttributesPatch,
    PtxGroupAttributes,
    PtxPatchResult,
    PtxSessionSettings,
    PtxTrackStatePatch,
    PtxValidationError,
    TrackRef,
)
from .patch_clips import (
    patch_add_fade_on_buffer,
    patch_clip_attributes_on_buffer,
    patch_rename_clips_on_buffer,
)
from .patch_session import (
    patch_add_marker_on_buffer,
    patch_delete_marker_on_buffer,
    patch_session_settings_on_buffer,
)
from .patch_tracks import (
    patch_rename_tracks_on_buffer,
    patch_track_states_on_buffer,
)
from .patcher import (
    patch_add_group_on_buffer,
    patch_clear_track_inputs_on_buffer,
    patch_clip_gains_on_buffer,
    patch_delete_group_on_buffer,
    patch_modify_group_on_buffer,
    patch_track_volumes_on_buffer,
)
from .plugin_trim import patch_slot_a_trim_gains_on_buffer

_UNSET: Any = object()
_StepFn = Callable[[PtxBuffer, dict[str, Any]], None]


def empty_patch_summary() -> dict[str, Any]:
    """Return a freshly initialized ``PtxPatchBuilder`` summary dictionary."""
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
    """Queue multiple validated ``.ptx`` mutations and apply them in one decrypt/validate/encrypt pass."""

    def __init__(
        self,
        *,
        raw_bytes: bytes | None = None,
        file_path: str | Path | None = None,
        ptx_buffer: PtxBuffer | None = None,
    ) -> None:
        sources = sum(x is not None for x in (raw_bytes, file_path, ptx_buffer))
        if sources != 1:
            raise PtxValidationError("Provide exactly one of raw_bytes, file_path, or ptx_buffer to PtxPatchBuilder")
        self._raw_bytes = raw_bytes
        self._file_path = Path(file_path) if file_path is not None else None
        self._ptx_buffer = ptx_buffer
        self._steps: list[_StepFn] = []

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> PtxPatchBuilder:
        return cls(raw_bytes=raw_bytes)

    @classmethod
    def from_file(cls, file_path: str | Path) -> PtxPatchBuilder:
        return cls(file_path=file_path)

    @classmethod
    def from_buffer(cls, ptx_buffer: PtxBuffer) -> PtxPatchBuilder:
        """Create a builder operating directly on an already-decrypted ``PtxBuffer``."""
        return cls(ptx_buffer=ptx_buffer)

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

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            _merge_track_summary(summary["cleared_inputs"], patch_clear_track_inputs_on_buffer(ptx_buf, names))

        self._steps.append(_step)
        return self

    def set_clip_gains(self, track_clip_gains_db: Mapping[str, float]) -> PtxPatchBuilder:
        gains = dict(track_clip_gains_db)
        if not gains:
            return self

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            _merge_track_summary(summary["clip_gains"], patch_clip_gains_on_buffer(ptx_buf, gains))

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

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            res = patch_slot_a_trim_gains_on_buffer(
                ptx_buf, trims,
                activate_if_inactive=activate_if_inactive,
                unbypass_if_bypassed=unbypass_if_bypassed,
                allow_auto_insert=allow_auto_insert,
            )
            _merge_track_summary(summary["slot_a_trims"], res)

        self._steps.append(_step)
        return self

    def set_track_volumes(self, track_volumes_db: Mapping[str, float]) -> PtxPatchBuilder:
        vols = dict(track_volumes_db)
        if not vols:
            return self

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            _merge_track_summary(summary["volumes"], patch_track_volumes_on_buffer(ptx_buf, vols))

        self._steps.append(_step)
        return self

    def add_group(
        self,
        name: str,
        member_tracks: Sequence[TrackRef],
        *,
        vca_track: TrackRef | None = None,
        group_type: str = "Mix/Edit",
        id_letter: str | None = None,
        is_edit_active: bool = True,
        is_mix_active: bool = True,
        follow_globals: bool | None = None,
        include_rulers: bool | None = None,
        attributes: PtxGroupAttributes | None = None,
        color_index: int = 0xFFFE,
    ) -> PtxPatchBuilder:
        members = list(member_tracks)
        attrs_copy = copy.deepcopy(attributes) if attributes is not None else None

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            info = patch_add_group_on_buffer(
                ptx_buf, name, members, vca_track=vca_track, group_type=group_type,
                id_letter=id_letter, is_edit_active=is_edit_active, is_mix_active=is_mix_active,
                follow_globals=follow_globals, include_rulers=include_rulers,
                attributes=attrs_copy, color_index=color_index,
            )
            summary["groups"].append(info)

        self._steps.append(_step)
        return self

    def modify_group(
        self,
        name: str,
        *,
        new_name: str | None = None,
        member_tracks: Sequence[TrackRef] | None = None,
        add_tracks: Sequence[TrackRef] | None = None,
        remove_tracks: Sequence[TrackRef] | None = None,
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

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            kw: dict[str, Any] = {
                "new_name": new_name, "member_tracks": m_tracks, "add_tracks": a_tracks,
                "remove_tracks": r_tracks, "group_type": group_type,
                "is_edit_active": is_edit_active, "is_mix_active": is_mix_active,
                "follow_globals": follow_globals, "include_rulers": include_rulers,
            }
            if vca_track is not _UNSET:
                kw["vca_track"] = vca_track
            summary["groups"].append(patch_modify_group_on_buffer(ptx_buf, name, **kw))

        self._steps.append(_step)
        return self

    def delete_group(self, name: str) -> PtxPatchBuilder:
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            summary["groups"].append(patch_delete_group_on_buffer(ptx_buf, name))

        self._steps.append(_step)
        return self

    def set_session_settings(
        self,
        settings: PtxSessionSettings | None = None,
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
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            res = patch_session_settings_on_buffer(
                ptx_buf, settings,
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

        self._steps.append(_step)
        return self

    def set_track_states(
        self, track_states: Mapping[str, PtxTrackStatePatch | Mapping[str, Any]]
    ) -> PtxPatchBuilder:
        copied: dict[str, PtxTrackStatePatch | dict[str, Any]] = {
            k: (v if isinstance(v, PtxTrackStatePatch) else dict(v))
            for k, v in track_states.items()
        }
        if not copied:
            return self

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            for k, v in patch_track_states_on_buffer(ptx_buf, copied).items():
                summary["track_states"].setdefault(k, {}).update(v)

        self._steps.append(_step)
        return self

    def rename_tracks(self, renames: Mapping[str, str]) -> PtxPatchBuilder:
        copied = dict(renames)
        if not copied:
            return self

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            summary["renamed_tracks"].update(patch_rename_tracks_on_buffer(ptx_buf, copied))

        self._steps.append(_step)
        return self

    def rename_clips(self, renames: Mapping[str, str]) -> PtxPatchBuilder:
        copied = dict(renames)
        if not copied:
            return self

        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            summary["renamed_clips"].update(patch_rename_clips_on_buffer(ptx_buf, copied))

        self._steps.append(_step)
        return self

    def set_clip_attributes(
        self,
        track_name: str,
        start_sample: int,
        attrs: PtxClipAttributesPatch | None = None,
        *,
        is_muted: bool | None = None,
        is_time_locked: bool | None = None,
        is_edit_locked: bool | None = None,
        color_index: int | None = None,
        clip_name: str | None = None,
        clip_index: int | None = None,
    ) -> PtxPatchBuilder:
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            info = patch_clip_attributes_on_buffer(
                ptx_buf, track_name, start_sample, attrs,
                is_muted=is_muted, is_time_locked=is_time_locked,
                is_edit_locked=is_edit_locked, color_index=color_index,
                clip_name=clip_name, clip_index=clip_index,
            )
            summary["clip_attributes"].append(info)

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
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            info = patch_add_fade_on_buffer(
                ptx_buf, track_name, fade_type, start_sample, total_samples,
                pre_samples=pre_samples, shape=shape, crossfade_link=crossfade_link,
            )
            summary["fades"].append(info)

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
        track: TrackRef | None = None,
        zoom_settings: bool = False,
        pre_post_roll_times: bool = False,
        track_visibility: bool = False,
        track_heights: bool = False,
        group_enables: bool = False,
        window_configuration: bool = False,
    ) -> PtxPatchBuilder:
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            info = patch_add_marker_on_buffer(
                ptx_buf, name, start_pos,
                number=number, end_pos=end_pos, reference=reference,
                time_properties=time_properties, comments=comments,
                color_index=color_index, color_rgb=color_rgb, track=track,
                zoom_settings=zoom_settings, pre_post_roll_times=pre_post_roll_times,
                track_visibility=track_visibility, track_heights=track_heights,
                group_enables=group_enables, window_configuration=window_configuration,
            )
            summary["markers"].append(info)

        self._steps.append(_step)
        return self

    def delete_marker(self, number: int) -> PtxPatchBuilder:
        def _step(ptx_buf: PtxBuffer, summary: dict[str, Any]) -> None:
            summary["markers"].append(patch_delete_marker_on_buffer(ptx_buf, number))

        self._steps.append(_step)
        return self

    def _execute_on_buffer(
        self,
        ptx_buf: PtxBuffer,
        *,
        validate: bool,
        orig_session: Any = None,
    ) -> tuple[PtxBuffer, dict[str, Any]]:
        orig_dec = bytes(ptx_buf.data)
        summary = empty_patch_summary()
        if not self._steps:
            return ptx_buf, summary
        for step in self._steps:
            step(ptx_buf, summary)
        if validate:
            validate_decrypted_ptx_pair(
                orig_dec, ptx_buf.xor_delta, bytes(ptx_buf.data), ptx_buf.xor_delta,
                orig_session=orig_session,
            )
        return ptx_buf, summary

    def build_buffer(self, *, validate: bool = True) -> tuple[PtxBuffer, dict[str, Any]]:
        """Execute queued steps on a single decrypted ``PtxBuffer`` and return ``(buffer, summary)``."""
        orig_session = None
        if self._ptx_buffer is not None:
            orig_session = self._ptx_buffer._session
            ptx_buf = PtxBuffer(bytearray(self._ptx_buffer.data), self._ptx_buffer.xor_delta, self._ptx_buffer.k0)
        elif self._raw_bytes is not None:
            ptx_buf = PtxBuffer.from_encrypted(self._raw_bytes)
        elif self._file_path is not None:
            ptx_buf = PtxBuffer.from_encrypted(self._file_path.read_bytes())
        else:
            raise PtxValidationError("PtxPatchBuilder has no source")
        return self._execute_on_buffer(ptx_buf, validate=validate, orig_session=orig_session)

    def build_result(self) -> PtxPatchResult:
        """Apply queued patches in one pass, validate structure, and return a typed ``PtxPatchResult``."""
        orig_session = None
        if self._raw_bytes is not None:
            source = self._raw_bytes
            ptx_buf = PtxBuffer.from_encrypted(source) if self._steps else None
        elif self._file_path is not None:
            source = self._file_path.read_bytes()
            ptx_buf = PtxBuffer.from_encrypted(source) if self._steps else None
        elif self._ptx_buffer is not None:
            source = self._ptx_buffer.to_encrypted()
            orig_session = self._ptx_buffer._session
            ptx_buf = PtxBuffer(bytearray(self._ptx_buffer.data), self._ptx_buffer.xor_delta, self._ptx_buffer.k0) if self._steps else None
        else:
            raise PtxValidationError("PtxPatchBuilder has no source")
        if not self._steps or ptx_buf is None:
            return PtxPatchResult(patched_bytes=source, original_bytes=source, summary=empty_patch_summary())
        out_buf, summary = self._execute_on_buffer(ptx_buf, validate=True, orig_session=orig_session)
        return PtxPatchResult(patched_bytes=out_buf.to_encrypted(), original_bytes=source, summary=summary)

    def build(self) -> tuple[bytes, dict[str, Any]]:
        """Apply queued patches in memory, validate structure, and return ``(patched_bytes, summary)``."""
        res = self.build_result()
        return res.patched_bytes, res.summary

    def commit(self) -> tuple[bytes, dict[str, Any]]:
        """Atomically apply queued patches to ``file_path``, clear the queue, and return ``(original_bytes, summary)``."""
        if self._file_path is None:
            raise PtxValidationError("commit() requires PtxPatchBuilder.from_file(path); use build() for in-memory bytes")
        if not self._steps:
            return self._file_path.read_bytes(), empty_patch_summary()
        res = self.build_result()
        atomic_write_bytes(self._file_path, res.patched_bytes)
        self._steps.clear()
        return res.original_bytes, res.summary
