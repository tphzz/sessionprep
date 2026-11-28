"""Offline Pro Tools session creation runner bridging creation plans, media staging, and PTX writing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..ptx import (
    AudioFileWriteSpec,
    OfflineAudioTrackRequest,
    OfflineGroupRequest,
    OfflinePtxWriteRequest,
    OfflinePtxWriteResult,
    write_offline_ptx_session,
)
from .protools_ptx_media import (
    StagedSessionDir,
    cleanup_staged_session_dir,
    get_volume_info,
    stage_session_audio_files,
)
from .protools_ptx_plan import ProToolsCreationPlan


@dataclass(frozen=True)
class OfflineProToolsSessionOutput:
    """Completed offline Pro Tools session creation output."""

    plan: ProToolsCreationPlan
    staged: StagedSessionDir
    write_result: OfflinePtxWriteResult


def _build_track_and_clip_gain_requests(
    plan: ProToolsCreationPlan,
    staged: StagedSessionDir,
) -> tuple[tuple[OfflineAudioTrackRequest, ...], dict[str, float]]:
    file_index_by_dest = {sf.dest_path: idx for idx, sf in enumerate(staged.staged_files)}
    track_reqs: list[OfflineAudioTrackRequest] = []
    clip_gains: dict[str, float] = {}
    used_clip_names: set[str] = set()
    for pt in plan.tracks:
        sf = staged.lookup(pt.source_filepath)
        if sf is None:
            raise RuntimeError(f"Staged audio file missing for {pt.track_name!r}: {pt.source_filepath}")
        base_clip = Path(sf.filename).stem
        if base_clip.lower() in used_clip_names:
            base_clip = pt.track_name
        suffix = 2
        clip_name = base_clip
        while clip_name.lower() in used_clip_names:
            clip_name = f"{base_clip}_{suffix}"
            suffix += 1
        used_clip_names.add(clip_name.lower())
        track_reqs.append(
            OfflineAudioTrackRequest(
                track_name=pt.track_name,
                channels=pt.channels,
                parent_folder_uuid_hex=pt.parent_folder_uuid or None,
                output_bus_name=pt.output_bus_name,
                audio_file_index=file_index_by_dest[sf.dest_path],
                length_samples=sf.length_samples,
                clip_base_name=clip_name,
                color_index=pt.color_index,
            )
        )
        if abs(pt.staging.clip_gain_db) >= 0.05:
            clip_gains[pt.track_name] = round(pt.staging.clip_gain_db, 1)
    return tuple(track_reqs), clip_gains


def _build_write_request(
    plan: ProToolsCreationPlan,
    staged: StagedSessionDir,
) -> OfflinePtxWriteRequest:
    audio_files = tuple(
        AudioFileWriteSpec(
            filename=sf.filename,
            sample_rate=sf.sample_rate,
            channels=sf.channels,
            bit_depth=sf.bit_depth,
            length_samples=sf.length_samples,
            file_id8=sf.file_id8,
            smpte_umid32=sf.smpte_umid32,
            filetime_u64=sf.filetime_u64,
        )
        for sf in staged.staged_files
    )
    track_reqs, clip_gains = _build_track_and_clip_gain_requests(plan, staged)
    delete_names = tuple(
        [t.name for t in plan.cleanup_plan.empty_folders]
        + [t.name for t in plan.cleanup_plan.unused_vcas]
        + [t.name for t in plan.cleanup_plan.orphaned_auxes]
    )
    group_reqs = tuple(
        OfflineGroupRequest(
            name=ga.group_name,
            action="create" if ga.op == "add" else "add_members",
            member_tracks=tuple(ga.member_tracks),
            vca_track=ga.vca_track,
            group_type=ga.group_type,
        )
        for ga in plan.group_actions
        if ga.op in ("add", "modify")
    )
    vol_serial, vol_label = get_volume_info(staged.session_dir)
    return OfflinePtxWriteRequest(
        ptxt_path=Path(plan.ptxt_path),
        output_ptx_path=staged.ptx_path,
        session_dir=staged.session_dir,
        sample_rate=staged.spec.sample_rate,
        bit_depth=staged.spec.bit_depth,
        audio_files=audio_files,
        tracks=track_reqs,
        delete_track_names=delete_names,
        clip_gains_db=clip_gains,
        slot_a_trims_db=dict(plan.recommended_slot_a_trims_db),
        track_volumes_db=dict(plan.recommended_faders_db),
        group_requests=group_reqs,
        volume_serial_u32=vol_serial,
        volume_label=vol_label,
    )


def execute_offline_ptx_creation(
    plan: ProToolsCreationPlan,
    project_dir: str | Path,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> OfflineProToolsSessionOutput:
    """Stage audio media and write a complete ``.ptx`` session offline with atomic rollback on error."""
    if progress_cb:
        progress_cb(35, 100, f"Staging {len(plan.source_audio_paths)} audio files...")
    staged = stage_session_audio_files(
        project_dir, plan.session_name, plan.source_audio_paths, plan.audio_spec,
    )
    try:
        if progress_cb:
            progress_cb(70, 100, f"Writing {staged.ptx_path.name}...")
        req = _build_write_request(plan, staged)
        write_result = write_offline_ptx_session(req)
    except BaseException:
        cleanup_staged_session_dir(staged.session_dir)
        raise
    return OfflineProToolsSessionOutput(plan=plan, staged=staged, write_result=write_result)
