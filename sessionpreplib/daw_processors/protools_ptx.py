"""Pro Tools ``.ptx`` + PTSL post-import routing, clip-gain staging, and VCA group orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..models import ParamSpec, SessionContext, TrackContext
from ..ptx import (
    LeafAudioInput,
    PtxPatchBuilder,
    PtxSession,
    PtxSessionAudit,
    PtxTrack,
    SignalFlowGraph,
    build_annotated_signal_flow,
)
from ..ptx.models import PTX_MAGIC
from ..signal_flow import (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
    resolve_track_staging,
)
from . import ptsl_helpers as ptslh

log = logging.getLogger(__name__)

VALID_GAIN_MODES = (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
)


def ptx_config_params() -> list[ParamSpec]:
    """Return Pro Tools ``.ptx`` routing, clip-gain, VCA grouping, and bus-staging parameters."""
    return [
        ParamSpec(
            key="protools_auto_route_folders", type=bool, default=True,
            label="Auto-route Tracks to Folder Bus",
            description="Route each inserted audio track's Main Output to its parent Routing Folder's input bus.",
        ),
        ParamSpec(
            key="protools_gain_mode", type=str, default=GAIN_MODE_CLIP_AND_FADER,
            choices=list(VALID_GAIN_MODES),
            label="Gain Staging Mode",
            description=(
                "How Peak-Capped RMS normalization is applied in Pro Tools: Clip Gain pre-insert + inverse Fader "
                "balance (default), Clip Gain with unity faders, Fader only, or None."
            ),
        ),
        ParamSpec(
            key="protools_auto_group_folders", type=bool, default=True,
            label="Auto-group Folder Tracks",
            description="Create or update a Pro Tools group for the inserted audio tracks inside each folder.",
        ),
        ParamSpec(
            key="protools_link_template_vcas", type=bool, default=True,
            label="Link Groups to Template VCAs",
            description="Automatically wire each folder's Pro Tools group to its matching template VCA master.",
        ),
        ParamSpec(
            key="protools_group_type", type=str, default="Mix/Edit",
            choices=["Mix/Edit", "Mix", "Edit"],
            label="Pro Tools Group Type",
            description="Group type created for populated folders.",
        ),
        ParamSpec(
            key="protools_stage_bus_faders", type=bool, default=True,
            label="Stage Folder & Subgroup Bus Faders",
            description=(
                "Reconstruct the template signal-flow DAG, sum adjusted audio tracks, and pre-set "
                "Routing Folder and multi-input subgroup/stem Aux faders."
            ),
        ),
        ParamSpec(
            key="protools_pan_law_db", type=float, default=-3.0, min=-6.0, max=0.0,
            label="Signal-Flow Pan Law (dB)",
            description="Pro Tools pan depth used when summing mono/panned tracks through the signal-flow DAG.",
        ),
        ParamSpec(
            key="protools_remove_unused_aux_tracks", type=bool, default=True,
            label="Remove Orphaned Downstream Aux Busses",
            description=(
                "Automatically delete template Aux busses that have no surviving upstream tracks or "
                "sends feeding them after empty folder cleanup."
            ),
        ),
    ]


@dataclass(frozen=True)
class ProToolsTrackStaging:
    """Resolved media path plus pre-insert Clip Gain and post-insert Track Fader dB."""

    filepath: str
    clip_gain_db: float
    fader_db: float
    baked_gain_db: float
    legacy_ptsl_fader_db: float = 0.0


@dataclass(frozen=True)
class FolderRouteEntry:
    """One planned Main Output routing assignment for a created Pro Tools track."""

    track_name: str
    track_id: str
    routing_folder_name: str
    target_bus_name: str
    target_signalpath_id: str


@dataclass(frozen=True)
class FolderGroupAction:
    """One planned Pro Tools Edit/Mix group creation or update linked to a folder/VCA."""

    op: str  # "add" | "modify"
    group_name: str
    folder_name: str
    member_tracks: tuple[str, ...]
    vca_track: str | None
    group_type: str = "Mix/Edit"


@dataclass(frozen=True)
class EmptyTrackCleanupPlan:
    """Empty template folders (deepest-first), unused VCAs, and orphaned Auxes planned for removal."""

    empty_folders: tuple[PtxTrack, ...]
    unused_vcas: tuple[PtxTrack, ...]
    folder_depths: dict[str, int]
    orphaned_auxes: tuple[PtxTrack, ...] = ()


def resolve_protools_track_staging(
    session: SessionContext,
    track: TrackContext,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
) -> ProToolsTrackStaging:
    """Resolve transfer media plus pre-insert Clip Gain and post-insert Track Fader offsets.

    Delegates core staging resolution to :func:`sessionpreplib.signal_flow.resolve_track_staging`
    while clamping *gain_mode* to Pro Tools' supported modes.
    """
    mode = gain_mode if gain_mode in VALID_GAIN_MODES else GAIN_MODE_CLIP_AND_FADER
    plan = resolve_track_staging(session, track, mode)
    return ProToolsTrackStaging(
        filepath=plan.filepath,
        clip_gain_db=plan.clip_gain_db,
        fader_db=plan.fader_db,
        baked_gain_db=plan.baked_gain_db,
        legacy_ptsl_fader_db=plan.prepared_fader_db,
    )


def _live_field(track: Any, field: str, default: str = "") -> str:
    if isinstance(track, dict):
        return str(track.get(field, default) or default)
    val = getattr(track, field, default)
    return str(val or default)


def _is_routing_folder_live(live_track: Any) -> bool:
    ttype = _live_field(live_track, "type")
    if ttype in ("TType_RoutingFolder", "RoutingFolder", "TT_RoutingFolder"):
        return True
    try:
        from ptsl import PTSL_pb2 as pt
        return getattr(live_track, "type", None) == pt.TrackType.RoutingFolder
    except Exception:
        return False


def _nearest_routing_folder_id(
    track_id: str,
    live_by_id: dict[str, Any],
    ptx_by_uuid: dict[str, PtxTrack],
) -> str | None:
    """Walk ``parent_folder_id`` upward to find the nearest ancestor Routing Folder (cycle-safe)."""
    visited: set[str] = {track_id.lower()}
    cur = live_by_id.get(track_id.lower())
    while cur is not None:
        parent_id = _live_field(cur, "parent_folder_id")
        if not parent_id:
            return None
        key = parent_id.lower()
        if key in visited:
            return None
        visited.add(key)
        parent_live = live_by_id.get(key)
        parent_ptx = ptx_by_uuid.get(key)
        if (parent_live is not None and _is_routing_folder_live(parent_live)) or (
            parent_ptx is not None and parent_ptx.track_type == "RoutingFolder"
        ):
            return parent_id
        cur = parent_live
    return None


def plan_folder_output_routing(
    ptx: PtxSession,
    live_tracks: list[Any],
    base: int,
    created_track_ids: set[str] | None = None,
) -> tuple[list[FolderRouteEntry], list[str]]:
    """Build Main Output routing entries for tracks nested under Routing Folders."""
    live_by_id = {_live_field(t, "id").lower(): t for t in live_tracks if _live_field(t, "id")}
    ptx_by_uuid = {t.uuid_braced.lower(): t for t in ptx.tracks if t.uuid_braced}
    target_ids = {tid.lower() for tid in created_track_ids} if created_track_ids is not None else None
    plan: list[FolderRouteEntry] = []
    warnings: list[str] = []

    for lt in live_tracks:
        tid = _live_field(lt, "id")
        tname = _live_field(lt, "name")
        if not tid or (target_ids is not None and tid.lower() not in target_ids):
            continue
        rf_id = _nearest_routing_folder_id(tid, live_by_id, ptx_by_uuid)
        if rf_id is None:
            continue
        rf_ptx = ptx_by_uuid.get(rf_id.lower())
        if rf_ptx is None or rf_ptx.input_path is None or rf_ptx.input_path.ptsl_index is None:
            rf_name = rf_ptx.name if rf_ptx else rf_id
            warnings.append(f"Skipped output routing for '{tname}': folder '{rf_name}' has no routable bus input.")
            continue
        sp_id = hex(base + rf_ptx.input_path.ptsl_index)
        plan.append(FolderRouteEntry(tname, tid, rf_ptx.name, rf_ptx.input_path.name, sp_id))
    return plan, warnings


def execute_folder_output_routing(
    engine,
    ptx: PtxSession,
    live_tracks: list[Any],
    created_track_ids: set[str],
    *,
    batch_job_id: str | None = None,
    progress: int = 92,
) -> tuple[list[FolderRouteEntry], list[tuple[str, str]], list[str]]:
    """Calibrate PTSL signalpath base and assign Main Outputs for newly created tracks.

    Returns ``(routed_entries, route_errors, warnings)`` where each item in
    ``route_errors`` is ``(target_label, error_message)``.
    """
    # Short-circuit before calibration when no created tracks have a routable ancestor folder.
    probe_plan, probe_warnings = plan_folder_output_routing(ptx, live_tracks, 0, created_track_ids)
    if not probe_plan:
        return [], [], probe_warnings

    base, _verified, cal_warnings = ptslh.calibrate_signalpath_base(engine, ptx, live_tracks)
    plan, plan_warnings = plan_folder_output_routing(ptx, live_tracks, base, created_track_ids)
    by_target: dict[str, list[FolderRouteEntry]] = {}
    for entry in plan:
        by_target.setdefault(entry.target_signalpath_id, []).append(entry)
    routed: list[FolderRouteEntry] = []
    errors: list[tuple[str, str]] = []
    for target_sp_id, entries in by_target.items():
        try:
            ptslh.set_track_main_output_assignments(
                engine,
                [target_sp_id],
                track_ids=[e.track_id for e in entries],
                batch_job_id=batch_job_id,
                progress=progress,
            )
            routed.extend(entries)
        except Exception as exc:
            names = ", ".join(e.track_name for e in entries)
            errors.append((names, str(exc)))
    return routed, errors, cal_warnings + plan_warnings


def _find_folder_vca(ptx: PtxSession, folder_name: str) -> PtxTrack | None:
    """Find the matching template VCA for *folder_name* (inside folder first, then name match)."""
    inside = [t for t in ptx.tracks if t.track_type == "Vca" and t.parent_folder_name.lower() == folder_name.lower()]
    if inside:
        return inside[0]
    candidates = {f"{folder_name}-vca".lower(), f"{folder_name} vca".lower(), f"{folder_name}_vca".lower()}
    for trk in ptx.tracks:
        if trk.track_type == "Vca" and trk.name.lower() in candidates:
            return trk
    return None


def _unique_group_name(existing_names_lower: set[str], folder_name: str) -> str:
    if folder_name.lower() not in existing_names_lower:
        return folder_name
    base = f"{folder_name}-GRP"
    if base.lower() not in existing_names_lower:
        return base
    suffix = 2
    while f"{base}{suffix}".lower() in existing_names_lower:
        suffix += 1
    return f"{base}{suffix}"


def plan_folder_groups_and_vcas(
    ptx: PtxSession,
    created_track_names: list[str],
    *,
    link_vcas: bool = True,
    group_type: str = "Mix/Edit",
) -> list[FolderGroupAction]:
    """Plan per-folder Edit/Mix groups and template VCA links for *created_track_names*."""
    by_folder: dict[str, list[str]] = {}
    for tname in created_track_names:
        trk = ptx.track_by_name(tname)
        if trk is None or not trk.parent_folder_name:
            continue
        by_folder.setdefault(trk.parent_folder_name, []).append(trk.name)

    existing_lower = {g.name.lower() for g in ptx.groups}
    claimed_vcas: set[str] = set()
    actions: list[FolderGroupAction] = []
    for folder_name, members in by_folder.items():
        vca = _find_folder_vca(ptx, folder_name) if link_vcas and group_type.strip().lower() != "edit" else None
        if vca is not None and vca.name.lower() in claimed_vcas:
            log.warning("Skipping duplicate VCA %r for folder %r (already claimed).", vca.name, folder_name)
            vca = None
        if vca is not None:
            claimed_vcas.add(vca.name.lower())
        if vca is not None and vca.vca_group_name:
            actions.append(FolderGroupAction("modify", vca.vca_group_name, folder_name, tuple(members), vca.name, group_type))
            continue
        vca_name = vca.name if vca is not None else None
        existing = ptx.group_by_name(folder_name)
        if existing is not None and existing.name != "<ALL>" and (vca_name is None or existing.vca_track_name in (None, vca_name)):
            actions.append(FolderGroupAction("modify", existing.name, folder_name, tuple(members), vca_name, group_type))
            continue
        new_name = _unique_group_name(existing_lower, folder_name)
        existing_lower.add(new_name.lower())
        actions.append(FolderGroupAction("add", new_name, folder_name, tuple(members), vca_name, group_type))
    return actions


def _folder_depth(track_name_lower: str, parent_of: dict[str, str]) -> int:
    depth = 0
    seen: set[str] = {track_name_lower}
    cur = parent_of.get(track_name_lower, "")
    while cur and cur not in seen:
        seen.add(cur)
        depth += 1
        cur = parent_of.get(cur, "")
    return depth


def _mark_ancestors(folder_low: str, parent_of: dict[str, str], keep: set[str]) -> None:
    cur = folder_low
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        keep.add(cur)
        cur = parent_of.get(cur, "")


def _build_bus_family_map(ptx: PtxSession) -> dict[str, set[str]]:
    """Map each bus name (lowercase) to itself plus its parent bus or child subpaths."""
    family: dict[str, set[str]] = {}
    for sp in getattr(ptx, "signal_paths", ()):
        sp_low = sp.name.lower()
        family.setdefault(sp_low, {sp_low})
        if sp.parent_name:
            p_low = sp.parent_name.lower()
            family[sp_low].add(p_low)
            family.setdefault(p_low, {p_low}).add(sp_low)
        for sub in sp.subpaths:
            sub_low = sub.lower()
            family[sp_low].add(sub_low)
            family.setdefault(sub_low, {sub_low}).add(sp_low)
    return family


def _expand_bus(bus_low: str, bus_family: Mapping[str, set[str]]) -> set[str]:
    return bus_family.get(bus_low, {bus_low})


def _collect_non_empty_folders(
    ptx: PtxSession,
    excluded_lows: set[str] | None = None,
    bus_family: Mapping[str, set[str]] | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Return ``(non_empty_folder_names_lower, parent_of_lower)`` via fixed-point propagation."""
    excluded = excluded_lows or set()
    fam = bus_family if bus_family is not None else _build_bus_family_map(ptx)
    folder_types = ("RoutingFolder", "BasicFolder")
    parent_of = {t.name.lower(): t.parent_folder_name.lower() for t in ptx.tracks if t.parent_folder_name}
    keep: set[str] = set()
    for trk in ptx.tracks:
        if trk.name.lower() in excluded:
            continue
        if trk.track_type not in (*folder_types, "Vca") and trk.parent_folder_name:
            _mark_ancestors(trk.parent_folder_name.lower(), parent_of, keep)

    rf_by_input: dict[str, list[str]] = {}
    for trk in ptx.tracks:
        if trk.track_type == "RoutingFolder" and trk.input_path and trk.input_path.name:
            rf_by_input.setdefault(trk.input_path.name.lower(), []).append(trk.name.lower())

    changed = True
    while changed:
        changed = False
        for trk in ptx.tracks:
            low = trk.name.lower()
            if trk.track_type == "Vca" or low in excluded:
                continue
            if trk.track_type in folder_types and low not in keep:
                continue
            targets: list[str] = []
            if trk.main_output and trk.main_output.name:
                targets.append(trk.main_output.name.lower())
            targets.extend(s.destination.name.lower() for s in trk.sends if s.destination and s.destination.name)
            for bus_low in targets:
                for rel_bus in _expand_bus(bus_low, fam):
                    for rf_low in rf_by_input.get(rel_bus, ()):
                        if rf_low not in keep:
                            _mark_ancestors(rf_low, parent_of, keep)
                            changed = True
    return keep, parent_of


def _vca_matches_folder(vca_name_lower: str, folder_name_lower: str) -> bool:
    return vca_name_lower in (
        f"{folder_name_lower}-vca",
        f"{folder_name_lower} vca",
        f"{folder_name_lower}_vca",
    )


def _collect_unused_vcas(
    ptx: PtxSession,
    content_folders: set[str],
    candidate_empty_folders: list[PtxTrack],
    extra_removed_lows: set[str] | None = None,
) -> tuple[list[PtxTrack], set[str]]:
    """Return ``(removed_vcas, removed_vca_names_lower)`` via fixed-point pruning.

    Dedicated VCA-organizer folders (holding surviving VCAs alongside unlinked VCAs)
    never cause unrelated standalone VCAs inside them to be deleted, and VCAs tied to
    surviving content folders are always protected.
    """
    vca_members: dict[str, set[str]] = {}
    for grp in ptx.groups:
        if grp.name == "<ALL>" or not grp.vca_track_name:
            continue
        vca_members.setdefault(grp.vca_track_name.lower(), set()).update(m.lower() for m in grp.member_names)

    protected: set[str] = set()
    for trk in ptx.tracks:
        if trk.track_type in ("RoutingFolder", "BasicFolder") and trk.name.lower() in content_folders:
            vca = _find_folder_vca(ptx, trk.name)
            if vca is not None:
                protected.add(vca.name.lower())
        elif trk.track_type == "Vca" and trk.parent_folder_name.lower() in content_folders:
            protected.add(trk.name.lower())

    cand_folder_lows = {ef.name.lower() for ef in candidate_empty_folders}
    vcas_by_parent: dict[str, list[str]] = {}
    for trk in ptx.tracks:
        if trk.track_type == "Vca" and trk.parent_folder_name:
            vcas_by_parent.setdefault(trk.parent_folder_name.lower(), []).append(trk.name.lower())

    removed_lower = set(cand_folder_lows) | set(extra_removed_lows or ())
    removed_vcas: list[PtxTrack] = []
    removed_vca_lower: set[str] = set()

    changed = True
    while changed:
        changed = False
        for trk in ptx.tracks:
            low = trk.name.lower()
            if trk.track_type != "Vca" or low in removed_vca_lower or low in protected:
                continue
            members = vca_members.get(low, set())
            if members - removed_lower:
                continue
            parent_low = trk.parent_folder_name.lower()
            sole_in_cand_folder = parent_low in cand_folder_lows and len(vcas_by_parent.get(parent_low, ())) == 1
            matched_cand_folder = any(_vca_matches_folder(low, cf) for cf in cand_folder_lows)
            cascaded_super_vca = bool(members)
            if sole_in_cand_folder or matched_cand_folder or cascaded_super_vca:
                removed_vcas.append(trk)
                removed_vca_lower.add(low)
                removed_lower.add(low)
                changed = True
    return removed_vcas, removed_vca_lower


def _collect_orphaned_auxes(
    ptx: PtxSession,
    removed_lows: set[str],
    bus_family: Mapping[str, set[str]] | None = None,
) -> list[PtxTrack]:
    """Iteratively find Aux busses causally orphaned by removing empty folders/busses.

    An Aux bus is removed only when its input bus (``kind == "bus"``) was fed by
    at least one removed track (main output or send) AND is no longer fed by any
    surviving non-VCA track (accounting for Stereo/Mono bus subpaths). This protects
    on-demand FX return Auxes and hardware-input Auxes while pruning dead chains.
    """
    fam = bus_family if bus_family is not None else _build_bus_family_map(ptx)
    removed = set(removed_lows)
    orphaned: list[PtxTrack] = []
    changed = True
    while changed:
        changed = False
        surviving_fed: set[str] = set()
        removed_fed: set[str] = set()
        for trk in ptx.tracks:
            if trk.track_type == "Vca":
                continue
            target_set = removed_fed if trk.name.lower() in removed else surviving_fed
            if trk.main_output and trk.main_output.name:
                target_set.update(_expand_bus(trk.main_output.name.lower(), fam))
            for send in trk.sends:
                if send.destination and send.destination.name:
                    target_set.update(_expand_bus(send.destination.name.lower(), fam))
        for trk in ptx.tracks:
            low = trk.name.lower()
            if trk.track_type != "Aux" or low in removed:
                continue
            if trk.input_path is None or trk.input_path.kind != "bus" or not trk.input_path.name:
                continue
            in_low = trk.input_path.name.lower()
            if in_low in removed_fed and in_low not in surviving_fed:
                orphaned.append(trk)
                removed.add(low)
                changed = True
    return orphaned


def plan_empty_folders_and_vcas(
    ptx: PtxSession,
    *,
    remove_unused_aux_tracks: bool = True,
) -> EmptyTrackCleanupPlan:
    """Plan removal of empty folders (deepest-first), unused VCAs, and optionally orphaned Auxes."""
    folder_types = ("RoutingFolder", "BasicFolder")
    bus_family = _build_bus_family_map(ptx)
    orphaned_auxes: list[PtxTrack] = []
    removed_aux_lows: set[str] = set()
    while True:
        content_folders, parent_of = _collect_non_empty_folders(ptx, removed_aux_lows, bus_family)
        candidate_empty = [
            trk for trk in ptx.tracks
            if trk.track_type in folder_types and trk.name.lower() not in content_folders
        ]
        unused_vcas, removed_vca_lower = _collect_unused_vcas(
            ptx, content_folders, candidate_empty, removed_aux_lows,
        )
        keep_folders = set(content_folders)
        for trk in ptx.tracks:
            if trk.track_type == "Vca" and trk.name.lower() not in removed_vca_lower and trk.parent_folder_name:
                _mark_ancestors(trk.parent_folder_name.lower(), parent_of, keep_folders)
        empty_folders = [trk for trk in candidate_empty if trk.name.lower() not in keep_folders]
        if not remove_unused_aux_tracks:
            break
        removed_lows = {ef.name.lower() for ef in empty_folders} | removed_vca_lower | removed_aux_lows
        newly_orphaned = _collect_orphaned_auxes(ptx, removed_lows, bus_family)
        if not newly_orphaned:
            break
        orphaned_auxes.extend(newly_orphaned)
        removed_aux_lows.update(a.name.lower() for a in newly_orphaned)
    depths = {ef.name: _folder_depth(ef.name.lower(), parent_of) for ef in empty_folders}
    empty_folders.sort(key=lambda t: (-depths[t.name], t.index))
    return EmptyTrackCleanupPlan(tuple(empty_folders), tuple(unused_vcas), depths, tuple(orphaned_auxes))


def _delete_track_batch(
    engine,
    pairs: list[tuple[str, str]],
    *,
    batch_job_id: str | None,
    progress: int,
) -> tuple[list[str], list[str], str | None]:
    """Delete one batch of ``(name, id)`` pairs and verify ``success_count``."""
    try:
        deleted_count = ptslh.delete_tracks(
            engine, track_ids=[tid for _, tid in pairs],
            keep_folder_members=True, batch_job_id=batch_job_id, progress=progress,
        )
    except Exception as exc:
        return [], [name for name, _ in pairs], str(exc)
    if deleted_count >= len(pairs):
        return [name for name, _ in pairs], [], None
    remaining: set[str] | None = None
    if hasattr(engine, "track_list"):
        try:
            remaining = {_live_field(t, "id").lower() for t in engine.track_list() if _live_field(t, "id")}
        except Exception:
            remaining = None
    if remaining is not None:
        succeeded = [name for name, tid in pairs if tid.lower() not in remaining]
        failed = [name for name, tid in pairs if tid.lower() in remaining]
    else:
        succeeded, failed = [], [name for name, _ in pairs]
    err = f"Pro Tools deleted {deleted_count}/{len(pairs)} tracks."
    return succeeded, failed, err


def _run_cleanup_batch(
    engine,
    pairs: list[tuple[str, str]],
    op_name: str,
    parent_of: dict[str, str],
    blocked_folders: set[str],
    errors: list[tuple[str, str, str]],
    *,
    block_self: bool,
    batch_job_id: str | None,
    progress: int,
) -> list[str]:
    if not pairs:
        return []
    ok, failed, err = _delete_track_batch(engine, pairs, batch_job_id=batch_job_id, progress=progress)
    if err:
        errors.append((op_name, ", ".join(failed), err))
        for fname in failed:
            start = fname.lower() if block_self else parent_of.get(fname.lower(), "")
            _mark_ancestors(start, parent_of, blocked_folders)
    return ok


def execute_empty_folder_cleanup(
    engine,
    ptx: PtxSession,
    live_tracks: list[Any],
    *,
    remove_unused_aux_tracks: bool = True,
    batch_job_id: str | None = None,
    progress: int = 94,
) -> tuple[list[str], list[str], list[str], list[tuple[str, str, str]]]:
    """Delete unused VCAs, orphaned Aux busses, and empty folders via PTSL ``CId_DeleteTracks``.

    Deletes VCAs first, then orphaned Auxes, then empty folders grouped deepest-first with
    ``keep_folder_members=True`` so surviving tracks are never accidentally deleted.
    Returns ``(deleted_folder_names, deleted_vca_names, deleted_aux_names, errors)``.
    """
    plan = plan_empty_folders_and_vcas(ptx, remove_unused_aux_tracks=remove_unused_aux_tracks)
    if not plan.empty_folders and not plan.unused_vcas and not plan.orphaned_auxes:
        return [], [], [], []

    live_by_uuid = {_live_field(t, "id").lower(): _live_field(t, "id") for t in live_tracks if _live_field(t, "id")}
    live_by_name = {_live_field(t, "name").lower(): _live_field(t, "id") for t in live_tracks if _live_field(t, "id")}

    def _resolve_id(trk: PtxTrack) -> str | None:
        tid = live_by_uuid.get(trk.uuid_braced.lower()) or live_by_name.get(trk.name.lower())
        if tid is None:
            log.warning("Could not resolve live PTSL track ID for %r; skipping deletion.", trk.name)
        return tid

    errors: list[tuple[str, str, str]] = []
    blocked_folders: set[str] = set()
    parent_of = {t.name.lower(): t.parent_folder_name.lower() for t in ptx.tracks if t.parent_folder_name}

    vca_pairs = [(v.name, tid) for v in plan.unused_vcas if (tid := _resolve_id(v))]
    deleted_vcas = _run_cleanup_batch(
        engine, vca_pairs, "remove_unused_vca", parent_of, blocked_folders, errors,
        block_self=False, batch_job_id=batch_job_id, progress=progress,
    )
    aux_pairs = [(a.name, tid) for a in plan.orphaned_auxes if (tid := _resolve_id(a))]
    deleted_auxes = _run_cleanup_batch(
        engine, aux_pairs, "remove_unused_aux", parent_of, blocked_folders, errors,
        block_self=False, batch_job_id=batch_job_id, progress=progress,
    )

    deleted_folders: list[str] = []
    by_depth: dict[int, list[tuple[str, str]]] = {}
    for ef in plan.empty_folders:
        tid = _resolve_id(ef)
        if tid is None:
            _mark_ancestors(ef.name.lower(), parent_of, blocked_folders)
            continue
        by_depth.setdefault(plan.folder_depths.get(ef.name, 0), []).append((ef.name, tid))
    for depth in sorted(by_depth, reverse=True):
        pairs = [(n, tid) for n, tid in by_depth[depth] if n.lower() not in blocked_folders]
        deleted_folders.extend(_run_cleanup_batch(
            engine, pairs, "remove_empty_folder", parent_of, blocked_folders, errors,
            block_self=True, batch_job_id=batch_job_id, progress=progress,
        ))
    return deleted_folders, deleted_vcas, deleted_auxes, errors


def _queue_group_actions(builder: PtxPatchBuilder, group_actions: Sequence[FolderGroupAction]) -> None:
    for act in group_actions:
        if act.op == "add":
            builder.add_group(
                act.group_name,
                list(act.member_tracks),
                vca_track=act.vca_track,
                group_type=act.group_type,
                is_edit_active=False,
                is_mix_active=False,
            )
        elif act.vca_track is not None:
            builder.modify_group(
                act.group_name,
                add_tracks=list(act.member_tracks),
                vca_track=act.vca_track,
                is_edit_active=False,
                is_mix_active=False,
            )
        else:
            builder.modify_group(
                act.group_name,
                add_tracks=list(act.member_tracks),
                is_edit_active=False,
                is_mix_active=False,
            )


def log_ptx_audit_caveats(audit: PtxSessionAudit, context_label: str) -> None:
    """Log concise warnings when a ``.ptx``/``.ptxt`` contains dynamic features relevant to static staging."""
    caveats: list[str] = []
    if audit.automated_volume_tracks:
        caveats.append(f"automated volume on {', '.join(audit.automated_volume_tracks)}")
    if audit.automated_pan_tracks:
        caveats.append(f"automated pan on {', '.join(audit.automated_pan_tracks)}")
    if audit.automated_mute_tracks:
        caveats.append(f"automated mute on {', '.join(audit.automated_mute_tracks)}")
    if audit.dynamic_clip_gain_tracks:
        caveats.append(f"dynamic clip-gain lines on {', '.join(audit.dynamic_clip_gain_tracks)}")
    if audit.ara_tracks:
        caveats.append(f"ARA2 edits on {', '.join(audit.ara_tracks)}")
    if audit.has_unparsed_tempo_ramps:
        caveats.append("unparsed tempo ramps")
    if caveats:
        log.warning("Pro Tools session audit (%s): %s.", context_label, "; ".join(caveats))


def apply_post_close_ptx_patches(
    session_path: str | Path,
    *,
    clear_input_tracks: list[str] | None = None,
    clip_gains_db: dict[str, float] | None = None,
    slot_a_trims_db: dict[str, float] | None = None,
    track_volumes_db: dict[str, float] | None = None,
    group_actions: list[FolderGroupAction] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Apply No Input, Clip Gain, Slot A Trim, Track/Bus Faders, and Folder Groups/VCAs in one atomic pass."""
    with PtxPatchBuilder.from_file(session_path) as builder:
        if clear_input_tracks:
            builder.clear_track_inputs(clear_input_tracks)
        if clip_gains_db:
            builder.set_clip_gains(clip_gains_db)
        if slot_a_trims_db:
            builder.set_slot_a_trim_gains(slot_a_trims_db, activate_if_inactive=True)
        if track_volumes_db:
            builder.set_track_volumes(track_volumes_db)
        if group_actions:
            _queue_group_actions(builder, group_actions)
        return builder.commit()


def is_valid_ptx_file(path: str | Path) -> bool:
    """Return True when *path* exists on disk and begins with the Pro Tools ``.ptx`` magic header."""
    try:
        target = Path(path)
        if not target.is_file() or target.stat().st_size < 0x20:
            return False
        with target.open("rb") as handle:
            return handle.read(len(PTX_MAGIC)) == PTX_MAGIC
    except OSError:
        return False


def is_patchable_ptx_candidate(path: str | Path) -> bool:
    """Return True when *path* is a valid ``.ptx`` or a locked >=32-byte file awaiting close."""
    try:
        target = Path(path)
        if not target.is_file() or target.stat().st_size < 0x20:
            return False
        with target.open("rb") as handle:
            return handle.read(len(PTX_MAGIC)) == PTX_MAGIC
    except PermissionError:
        return True
    except OSError:
        return False


def _load_leaf_audio_input(track: TrackContext, staging: ProToolsTrackStaging) -> LeafAudioInput | None:
    """Load or reuse PCM samples for *staging.filepath* to feed the signal-flow DAG."""
    same_source = track.audio_data is not None and (
        os.path.normcase(os.path.realpath(staging.filepath))
        == os.path.normcase(os.path.realpath(track.filepath))
    )
    if same_source and track.audio_data is not None:
        return LeafAudioInput(
            samples=np.asarray(track.audio_data, dtype=np.float64),
            samplerate=track.samplerate,
            clip_gain_db=staging.clip_gain_db,
            fader_db=staging.fader_db,
            baked_gain_db=staging.baked_gain_db,
        )
    try:
        samples, sr = sf.read(staging.filepath, dtype="float64", always_2d=False)
    except Exception as exc:
        log.warning("Could not read audio for signal-flow summing (%s): %s", staging.filepath, exc)
        return None
    return LeafAudioInput(
        samples=np.asarray(samples, dtype=np.float64),
        samplerate=int(sr),
        clip_gain_db=staging.clip_gain_db,
        fader_db=staging.fader_db,
        baked_gain_db=staging.baked_gain_db,
    )


def _apply_group_actions_in_memory(raw_ptx: bytes, group_actions: list[FolderGroupAction]) -> PtxSession:
    """Apply planned group/VCA links to an in-memory ``.ptx`` copy so DAG summing sees active VCA offsets."""
    builder = PtxPatchBuilder.from_bytes(raw_ptx)
    _queue_group_actions(builder, group_actions)
    patched, _ = builder.build()
    return PtxSession.from_bytes(patched)


def _merge_graph_volumes(
    graph: SignalFlowGraph,
    created_names: list[str],
    track_volumes_db: dict[str, float],
) -> dict[str, float]:
    if abs(graph.leaf_fader_shift_db) >= 0.05:
        nodes_lc = {k.lower(): v for k, v in graph.nodes.items()}
        for created_name in created_names:
            leaf_node = nodes_lc.get(created_name.lower())
            if leaf_node is None:
                continue
            track_volumes_db[created_name] = round(leaf_node.recommended_fader_db, 1)
    for bus_name, bus_db in graph.recommended_bus_faders().items():
        if bus_name not in track_volumes_db:
            track_volumes_db[bus_name] = bus_db
    return graph.recommended_slot_a_trims()


def run_post_close_ptx_enhancement(
    session_path: str | Path,
    session: SessionContext,
    created_tracks: list[tuple[str, str, ProToolsTrackStaging, TrackContext]],
    *,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
    auto_group_folders: bool = True,
    link_template_vcas: bool = True,
    group_type: str = "Mix/Edit",
    stage_bus_faders: bool = True,
    pan_law_db: float = -3.0,
    bus_fader_overrides: Mapping[str, float] | None = None,
) -> tuple[bytes | None, dict[str, Any], SignalFlowGraph | None]:
    """Build signal-flow DAG, clip-gain/fader maps, and VCA groups, then patch ``.ptx`` atomically."""
    if not created_tracks or not is_valid_ptx_file(session_path):
        return None, {"cleared_inputs": {}, "clip_gains": {}, "slot_a_trims": {}, "volumes": {}, "groups": []}, None

    raw_bytes = Path(session_path).read_bytes()
    ptx = PtxSession.from_bytes(raw_bytes)
    log_ptx_audit_caveats(ptx.audit(), str(session_path))
    clip_gains_db: dict[str, float] = {}
    track_volumes_db: dict[str, float] = {}
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]] = {}
    leaf_gains: dict[str, tuple[float, float, float]] = {}
    created_names: list[str] = []

    for created_name, _tid, staging, track in created_tracks:
        created_names.append(created_name)
        if abs(staging.clip_gain_db) >= 0.05:
            clip_gains_db[created_name] = staging.clip_gain_db
        if abs(staging.fader_db) >= 0.05:
            track_volumes_db[created_name] = staging.fader_db
        leaf_entries[created_name.lower()] = (track, staging)
        leaf_gains[created_name] = (staging.clip_gain_db, staging.fader_db, staging.baked_gain_db)

    group_actions = (
        plan_folder_groups_and_vcas(ptx, created_names, link_vcas=link_template_vcas, group_type=group_type)
        if auto_group_folders
        else []
    )
    dag_ptx = _apply_group_actions_in_memory(raw_bytes, group_actions) if group_actions else ptx
    sf_cfg = {**session.config, "protools_pan_law_db": pan_law_db}
    attenuate_only = gain_mode != GAIN_MODE_CLIP_UNITY

    def _lazy_leaf(track_name: str) -> LeafAudioInput | None:
        pair = leaf_entries.get(track_name.lower())
        return _load_leaf_audio_input(pair[0], pair[1]) if pair is not None else None

    graph: SignalFlowGraph | None = None
    slot_a_trims_db: dict[str, float] = {}
    if leaf_entries:
        graph = build_annotated_signal_flow(
            dag_ptx, None, sf_cfg,
            stage_bus_faders=stage_bus_faders,
            attenuate_only=attenuate_only,
            strict_slot_a=True,
            allow_leaf_convergence=(gain_mode == GAIN_MODE_CLIP_AND_FADER and stage_bus_faders),
            leaf_loader=_lazy_leaf,
            leaf_gains=leaf_gains,
            bus_fader_overrides=bus_fader_overrides,
        )
        slot_a_trims_db = _merge_graph_volumes(graph, created_names, track_volumes_db)

    if not (created_names or clip_gains_db or slot_a_trims_db or track_volumes_db or group_actions):
        return None, {"cleared_inputs": {}, "clip_gains": {}, "slot_a_trims": {}, "volumes": {}, "groups": []}, graph

    orig_bytes, meta = apply_post_close_ptx_patches(
        session_path,
        clear_input_tracks=created_names or None,
        clip_gains_db=clip_gains_db or None,
        slot_a_trims_db=slot_a_trims_db or None,
        track_volumes_db=track_volumes_db or None,
        group_actions=group_actions or None,
    )
    return orig_bytes, meta, graph
