"""Declarative, backend-agnostic Pro Tools session creation planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from ..models import SessionContext, TrackContext
from ..ptx import (
    PtxSession,
    PtxSessionAudit,
    PtxSignalPathRef,
    PtxTrack,
    SignalFlowGraph,
    build_annotated_signal_flow,
    closest_pt_color_index,
)
from .protools_ptx import (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    EmptyTrackCleanupPlan,
    FolderGroupAction,
    ProToolsTrackStaging,
    plan_empty_folders_and_vcas,
    plan_folder_groups_and_vcas,
    resolve_protools_track_staging,
)
from .protools_ptx_media import TargetAudioSpec, resolve_target_audio_spec
from .protools_ptx_preview import (
    _apply_virtual_folder_groups,
    _find_folder_track,
    _load_preview_leaf,
    _prune_preview_empty_branches,
    _resolve_ancestor_routing_bus,
    _unique_virtual_name,
)


@dataclass(frozen=True)
class PlannedAudioTrack:
    """One planned Audio track to create inside a Pro Tools template folder."""

    entry_id: str
    track_name: str
    channels: int  # 1=Mono, 2=Stereo
    track_format: str  # "Mono" | "Stereo"
    parent_folder_name: str
    parent_folder_uuid: str
    output_bus_name: str | None
    output_bus_uuid_hex: str
    source_filepath: str
    staging: ProToolsTrackStaging
    color_index: int = 0xFFFE
    group_label: str = ""


@dataclass(frozen=True)
class ProToolsCreationPlan:
    """Complete deterministic plan for creating a Pro Tools session (shared by PTSL and Offline PTX)."""

    ptxt_path: str
    session_name: str
    audio_spec: TargetAudioSpec
    tracks: tuple[PlannedAudioTrack, ...]
    source_audio_paths: tuple[str, ...]
    group_actions: tuple[FolderGroupAction, ...]
    cleanup_plan: EmptyTrackCleanupPlan
    template_audit: PtxSessionAudit
    signal_flow: SignalFlowGraph | None = None
    recommended_faders_db: dict[str, float] = field(default_factory=dict)
    recommended_slot_a_trims_db: dict[str, float] = field(default_factory=dict)
    excluded_tracks: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _resolve_group_color_index(group_name: str | None, session: SessionContext) -> int:
    if not group_name:
        return 0xFFFE
    gui_cfg = session.config.get("gui")
    if not isinstance(gui_cfg, dict):
        return 0xFFFE
    groups = gui_cfg.get("groups") or ()
    colors = gui_cfg.get("colors") or ()
    color_name = next(
        (g.get("color") for g in groups if isinstance(g, dict) and g.get("name") == group_name),
        None,
    )
    if not color_name:
        return 0xFFFE
    argb = next(
        (c.get("argb") for c in colors if isinstance(c, dict) and c.get("name") == color_name),
        None,
    )
    if not argb:
        return 0xFFFE
    idx = closest_pt_color_index(str(argb))
    return idx if idx is not None else 0xFFFE


def _ordered_entry_pairs(
    assignments: Mapping[str, str],
    track_order: Mapping[str, list[str]] | None,
) -> list[tuple[str, str]]:
    work: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fid, ordered_ids in (track_order or {}).items():
        for eid in ordered_ids:
            if eid in assignments and assignments[eid] == fid and eid not in seen:
                work.append((eid, fid))
                seen.add(eid)
    for eid, fid in sorted(assignments.items()):
        if eid not in seen:
            work.append((eid, fid))
    return work


def _build_planned_track(
    ptx: PtxSession,
    session: SessionContext,
    eid: str,
    entry: Any,
    track: TrackContext,
    folder_trk: PtxTrack,
    staging: ProToolsTrackStaging,
    used_names: set[str],
    *,
    auto_route: bool,
    warnings: list[str],
) -> PlannedAudioTrack:
    raw_bus = _resolve_ancestor_routing_bus(ptx, folder_trk) if auto_route else None
    sp = ptx.signal_path_by_name(raw_bus) if raw_bus else None
    target_bus = raw_bus if (raw_bus and sp is not None) else None
    if auto_route and raw_bus and sp is None:
        warnings.append(
            f"Folder {folder_trk.name!r} input {raw_bus!r} is not an internal Pro Tools bus; {entry.daw_track_name!r} left unrouted."
        )
    elif auto_route and not raw_bus:
        warnings.append(
            f"Folder {folder_trk.name!r} has no ancestor Routing Folder bus; {entry.daw_track_name!r} left unrouted."
        )
    raw_name = (entry.daw_track_name or "").strip() or (track.filename or "").strip() or "Track"
    base_name = os.path.splitext(raw_name)[0].strip() or raw_name
    tname = _unique_virtual_name(base_name, used_names)
    fmt = "Stereo" if track.channels >= 2 else "Mono"
    main_out = PtxSignalPathRef(target_bus, "bus", sp.uuid_hex) if (target_bus and sp is not None) else None
    group_label = entry.group or track.group or ""
    ptx.tracks.append(
        PtxTrack(
            index=len(ptx.tracks), name=tname, track_type="Audio", format=fmt,
            uuid_hex="", uuid_braced="", parent_folder_name=folder_trk.name,
            parent_folder_uuid=folder_trk.uuid_braced, main_output=main_out, volume_db=staging.fader_db,
        )
    )
    return PlannedAudioTrack(
        entry_id=eid, track_name=tname, channels=track.channels, track_format=fmt,
        parent_folder_name=folder_trk.name, parent_folder_uuid=folder_trk.uuid_braced,
        output_bus_name=target_bus, output_bus_uuid_hex=sp.uuid_hex if (target_bus and sp is not None) else "",
        source_filepath=staging.filepath, staging=staging,
        color_index=_resolve_group_color_index(group_label, session),
        group_label=group_label,
    )


def _plan_audio_tracks(
    ptx: PtxSession,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    track_order: Mapping[str, list[str]] | None,
    *,
    auto_route: bool,
    gain_mode: str,
) -> tuple[
    list[PlannedAudioTrack],
    dict[str, tuple[TrackContext, ProToolsTrackStaging]],
    dict[str, tuple[float, float, float]],
    list[TrackContext],
    dict[str, str],
    list[str],
]:
    if session.config.get("_use_processed") is True and session.prepare_state != "ready":
        raise ValueError("Prepared audio is out of date. Run Prepare again (or uncheck Use Processed).")
    manifest_map = {e.entry_id: e for e in session.transfer_manifest}
    out_map = {t.filename: t for t in session.output_tracks}
    folder_lookup = {str(f.get("id", "")): f for f in folders}
    used_names = {t.name.lower() for t in ptx.tracks}

    planned: list[PlannedAudioTrack] = []
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]] = {}
    leaf_gains: dict[str, tuple[float, float, float]] = {}
    chosen_tracks: list[TrackContext] = []
    excluded: dict[str, str] = {}
    warnings: list[str] = []

    for eid, fid in _ordered_entry_pairs(assignments, track_order):
        entry = manifest_map.get(eid)
        track = out_map.get(entry.output_filename) if entry else None
        folder_trk = _find_folder_track(ptx, fid, folder_lookup)
        if (
            entry is None
            or track is None
            or folder_trk is None
            or folder_trk.track_type not in ("RoutingFolder", "BasicFolder")
        ):
            warnings.append(f"Skipped assignment {eid!r}: missing manifest entry, track, or template folder.")
            continue
        if track.channels < 1 or track.channels > 2:
            excluded[track.filename] = f"Unsupported channel count {track.channels}"
            continue
        try:
            staging = resolve_protools_track_staging(session, track, gain_mode=gain_mode)
        except (ValueError, OSError) as exc:
            excluded[track.filename] = str(exc)
            continue
        item = _build_planned_track(
            ptx, session, eid, entry, track, folder_trk, staging, used_names,
            auto_route=auto_route, warnings=warnings,
        )
        planned.append(item)
        leaf_entries[item.track_name.lower()] = (track, staging)
        leaf_gains[item.track_name] = (staging.clip_gain_db, staging.fader_db, staging.baked_gain_db)
        chosen_tracks.append(track)
    return planned, leaf_entries, leaf_gains, chosen_tracks, excluded, warnings


def _collect_fader_targets(
    planned: list[PlannedAudioTrack], graph: SignalFlowGraph | None
) -> tuple[dict[str, float], dict[str, float]]:
    faders: dict[str, float] = {}
    for pt in planned:
        if abs(pt.staging.fader_db) >= 0.05:
            faders[pt.track_name] = round(pt.staging.fader_db, 1)
    if graph is None:
        return faders, {}
    if abs(graph.leaf_fader_shift_db) >= 0.05:
        nodes_lc = {k.lower(): v for k, v in graph.nodes.items()}
        for pt in planned:
            node = nodes_lc.get(pt.track_name.lower())
            if node is not None:
                faders[pt.track_name] = round(node.recommended_fader_db, 1)
    for bus_name, bus_db in graph.recommended_bus_faders().items():
        faders.setdefault(bus_name, bus_db)
    return faders, graph.recommended_slot_a_trims()


def _compute_plan_signal_flow(
    ptx: PtxSession,
    session: SessionContext,
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]],
    leaf_gains: dict[str, tuple[float, float, float]],
    excluded: dict[str, str],
    *,
    gain_mode: str,
    stage_bus_faders: bool,
    pan_law_db: float,
    bus_fader_overrides: Mapping[str, float] | None,
    progress_cb: Callable[[int, int, str], None] | None,
    cancel_cb: Callable[[], bool] | None,
) -> SignalFlowGraph:
    sf_cfg = {**session.config, "protools_pan_law_db": pan_law_db}
    graph = build_annotated_signal_flow(
        ptx, None, sf_cfg,
        stage_bus_faders=stage_bus_faders,
        attenuate_only=(gain_mode != GAIN_MODE_CLIP_UNITY),
        strict_slot_a=True,
        allow_leaf_convergence=(gain_mode == GAIN_MODE_CLIP_AND_FADER and stage_bus_faders),
        leaf_loader=lambda name: _load_preview_leaf(name, leaf_entries, excluded),
        leaf_gains=leaf_gains,
        bus_fader_overrides=bus_fader_overrides,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    graph.excluded_tracks = dict(excluded)
    return graph


def _plan_groups_and_cleanup(
    ptx: PtxSession,
    created_names: list[str],
    *,
    auto_group_folders: bool,
    link_template_vcas: bool,
    group_type: str,
    remove_unused_aux_tracks: bool,
) -> tuple[list[FolderGroupAction], EmptyTrackCleanupPlan]:
    group_actions: list[FolderGroupAction] = []
    if auto_group_folders and created_names:
        group_actions = plan_folder_groups_and_vcas(
            ptx, created_names, link_vcas=link_template_vcas, group_type=group_type,
        )
        _apply_virtual_folder_groups(
            ptx, created_names, link_template_vcas=link_template_vcas, group_type=group_type,
        )
    cleanup_plan = plan_empty_folders_and_vcas(ptx, remove_unused_aux_tracks=remove_unused_aux_tracks)
    _prune_preview_empty_branches(ptx, remove_unused_aux_tracks=remove_unused_aux_tracks)
    return group_actions, cleanup_plan


def build_protools_creation_plan(
    ptxt_path: str | Path,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    *,
    track_order: Mapping[str, list[str]] | None = None,
    auto_route: bool = True,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
    auto_group_folders: bool = True,
    link_template_vcas: bool = True,
    group_type: str = "Mix/Edit",
    stage_bus_faders: bool = True,
    pan_law_db: float = -3.0,
    remove_unused_aux_tracks: bool = True,
    bus_fader_overrides: Mapping[str, float] | None = None,
    compute_signal_flow: bool = True,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> ProToolsCreationPlan:
    """Build a complete deterministic Pro Tools creation plan directly from a ``.ptxt`` template."""
    ptx = PtxSession.from_file(ptxt_path)
    if 0x2519 not in ptx.chunk_map:
        raise ValueError(f"Pro Tools template lacks 0x2519 track metadata: {ptxt_path}")
    audit = ptx.audit()
    planned, leaf_entries, leaf_gains, chosen_tracks, excluded, warnings = _plan_audio_tracks(
        ptx, session, assignments, folders, track_order, auto_route=auto_route, gain_mode=gain_mode,
    )
    audio_spec = resolve_target_audio_spec(
        chosen_tracks, template_sample_rate=ptx.info.sample_rate, template_bit_depth=ptx.info.bit_depth,
    )
    group_actions, cleanup_plan = _plan_groups_and_cleanup(
        ptx, [pt.track_name for pt in planned],
        auto_group_folders=auto_group_folders, link_template_vcas=link_template_vcas,
        group_type=group_type, remove_unused_aux_tracks=remove_unused_aux_tracks,
    )
    graph = (
        _compute_plan_signal_flow(
            ptx, session, leaf_entries, leaf_gains, excluded,
            gain_mode=gain_mode, stage_bus_faders=stage_bus_faders, pan_law_db=pan_law_db,
            bus_fader_overrides=bus_fader_overrides, progress_cb=progress_cb, cancel_cb=cancel_cb,
        )
        if compute_signal_flow and leaf_entries
        else None
    )
    faders, trims = _collect_fader_targets(planned, graph)
    return ProToolsCreationPlan(
        ptxt_path=str(ptxt_path),
        session_name=(session.project_name or "").strip() or "Session",
        audio_spec=audio_spec,
        tracks=tuple(planned), source_audio_paths=tuple(dict.fromkeys(pt.source_filepath for pt in planned)),
        group_actions=tuple(group_actions), cleanup_plan=cleanup_plan, template_audit=audit,
        signal_flow=graph, recommended_faders_db=faders, recommended_slot_a_trims_db=trims,
        excluded_tracks=excluded, warnings=tuple(warnings),
    )
