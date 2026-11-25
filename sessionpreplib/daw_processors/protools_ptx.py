"""Pro Tools ``.ptx`` + PTSL post-import routing, clip-gain staging, and VCA group orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..daw_processor import resolve_transfer_audio
from ..models import ParamSpec, SessionContext, TrackContext
from ..ptx import (
    LeafAudioInput,
    PtxSession,
    PtxTrack,
    SignalFlowGraph,
    apply_atomic_ptx_patch,
    build_annotated_signal_flow,
    patch_add_group,
    patch_clear_track_inputs,
    patch_clip_gains,
    patch_modify_group,
    patch_slot_a_trim_gains,
    patch_track_volumes,
)
from ..ptx.models import PTX_MAGIC
from . import ptsl_helpers as ptslh

log = logging.getLogger(__name__)

GAIN_MODE_CLIP_AND_FADER = "clip_gain_and_fader"
GAIN_MODE_CLIP_UNITY = "clip_gain_unity"
GAIN_MODE_FADER_ONLY = "fader_only"
GAIN_MODE_NONE = "none"
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
                "How bimodal normalization is applied in Pro Tools: Clip Gain pre-insert + inverse Fader "
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
    """Empty template folders (deepest-first) and unused VCAs planned for removal."""

    empty_folders: tuple[PtxTrack, ...]
    unused_vcas: tuple[PtxTrack, ...]
    folder_depths: dict[str, int]


def _planned_normalization_gains(session: SessionContext, track: TrackContext) -> tuple[float, float]:
    """Return ``(forward_gain_db, inverse_fader_db)`` planned for *track*."""
    proc_id = "bimodal_normalize"
    source = next((t for t in session.tracks if t.filename == track.filename), track)
    enabled = bool(session.config.get(f"{proc_id}_enabled", True))
    skipped = proc_id in track.processor_skip or proc_id in source.processor_skip
    result = track.processor_results.get(proc_id) or source.processor_results.get(proc_id)
    if enabled and not skipped and result is not None and not result.error and result.classification not in ("Skip", "Silent"):
        gain = float(result.gain_db)
        fader = float(result.data.get("fader_offset", -gain))
        if math.isfinite(gain) and math.isfinite(fader):
            return gain, fader
    manual = track.manual_gain_db if track.manual_gain_db is not None else source.manual_gain_db
    if manual is not None and math.isfinite(float(manual)):
        return float(manual), 0.0
    return 0.0, 0.0


def resolve_protools_track_staging(
    session: SessionContext,
    track: TrackContext,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
) -> ProToolsTrackStaging:
    """Resolve transfer media plus pre-insert Clip Gain and post-insert Track Fader offsets.

    Respects the user's selected audio source (``_use_processed``) while guaranteeing
    normalization gain is never double-applied or cancelled pre-insert:
      - Un-normalized audio (``_use_processed=False`` or no prepared WAV):
        * ``clip_gain_and_fader`` (default): Clip Gain ``+gain_db`` (pre-insert) and
          Track Fader ``fader_offset`` (post-insert balance).
        * ``clip_gain_unity``: Clip Gain ``+gain_db`` and Track Fader ``0.0 dB``.
        * ``fader_only``: Clip Gain ``0.0 dB`` and Track Fader ``0.0 dB``
          (avoids applying an unmatched inverse fader without pre-insert gain).
      - Prepared (already gain-baked) WAV (``_use_processed=True``):
        * Clip Gain stays ``0.0 dB`` (gain is already baked pre-insert) and Track Fader
          carries ``prepared_fader_db`` (unless mode is ``clip_gain_unity`` or ``none``).
      - ``none``: both Clip Gain and Track Fader stay ``0.0 dB``.
    """
    mode = gain_mode if gain_mode in VALID_GAIN_MODES else GAIN_MODE_CLIP_AND_FADER
    forward_gain, planned_fader = _planned_normalization_gains(session, track)
    media = resolve_transfer_audio(session, track)
    proc_path = track.processed_filepath
    is_prepared = proc_path is not None and bool(proc_path) and (
        os.path.normcase(os.path.realpath(media.filepath))
        == os.path.normcase(os.path.realpath(proc_path))
    )
    baked = forward_gain if is_prepared else 0.0
    if mode == GAIN_MODE_NONE:
        return ProToolsTrackStaging(media.filepath, 0.0, 0.0, baked_gain_db=baked, legacy_ptsl_fader_db=media.fader_db)
    if mode == GAIN_MODE_FADER_ONLY:
        # Inverse fader offsets only apply when normalization gain is actually baked into a prepared WAV.
        return ProToolsTrackStaging(
            media.filepath, 0.0, media.fader_db if is_prepared else 0.0,
            baked_gain_db=baked, legacy_ptsl_fader_db=media.fader_db,
        )
    clip_db = 0.0 if is_prepared else forward_gain
    fader_db = (media.fader_db if is_prepared else planned_fader) if mode == GAIN_MODE_CLIP_AND_FADER else 0.0
    return ProToolsTrackStaging(
        media.filepath, clip_db, fader_db, baked_gain_db=baked, legacy_ptsl_fader_db=media.fader_db,
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


def _collect_non_empty_folders(ptx: PtxSession) -> tuple[set[str], dict[str, str]]:
    """Return ``(non_empty_folder_names_lower, parent_of_lower)`` via fixed-point propagation."""
    folder_types = ("RoutingFolder", "BasicFolder")
    parent_of = {t.name.lower(): t.parent_folder_name.lower() for t in ptx.tracks if t.parent_folder_name}
    keep: set[str] = set()
    for trk in ptx.tracks:
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
            if trk.track_type == "Vca":
                continue
            if trk.track_type in folder_types and trk.name.lower() not in keep:
                continue
            targets: list[str] = []
            if trk.main_output and trk.main_output.name:
                targets.append(trk.main_output.name.lower())
            targets.extend(s.destination.name.lower() for s in trk.sends if s.destination and s.destination.name)
            for bus_low in targets:
                for rf_low in rf_by_input.get(bus_low, ()):
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
) -> tuple[list[PtxTrack], set[str]]:
    """Return ``(removed_vcas, removed_vca_names_lower)`` via fixed-point pruning.

    Dedicated VCA-organizer folders (holding surviving VCAs alongside unlinked VCAs)
    never cause unrelated standalone VCAs inside them to be deleted.
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

    removed_lower = set(cand_folder_lows)
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


def plan_empty_folders_and_vcas(ptx: PtxSession) -> EmptyTrackCleanupPlan:
    """Plan removal of empty folders (deepest-first) and unused/cascading VCA tracks."""
    content_folders, parent_of = _collect_non_empty_folders(ptx)
    folder_types = ("RoutingFolder", "BasicFolder")
    candidate_empty = [
        trk for trk in ptx.tracks
        if trk.track_type in folder_types and trk.name.lower() not in content_folders
    ]
    unused_vcas, removed_vca_lower = _collect_unused_vcas(ptx, content_folders, candidate_empty)

    keep_folders = set(content_folders)
    for trk in ptx.tracks:
        if trk.track_type == "Vca" and trk.name.lower() not in removed_vca_lower and trk.parent_folder_name:
            _mark_ancestors(trk.parent_folder_name.lower(), parent_of, keep_folders)

    empty_folders = [trk for trk in candidate_empty if trk.name.lower() not in keep_folders]
    depths = {ef.name: _folder_depth(ef.name.lower(), parent_of) for ef in empty_folders}
    empty_folders.sort(key=lambda t: (-depths[t.name], t.index))
    return EmptyTrackCleanupPlan(tuple(empty_folders), tuple(unused_vcas), depths)


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


def execute_empty_folder_cleanup(
    engine,
    ptx: PtxSession,
    live_tracks: list[Any],
    *,
    batch_job_id: str | None = None,
    progress: int = 94,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Delete empty folders and unused VCAs via PTSL ``CId_DeleteTracks``.

    Deletes VCAs first, then empty folders grouped deepest-first with
    ``keep_folder_members=True`` so surviving tracks are never accidentally deleted.
    Returns ``(deleted_folder_names, deleted_vca_names, errors)`` where each error is
    ``(operation, target_label, message)``.
    """
    plan = plan_empty_folders_and_vcas(ptx)
    if not plan.empty_folders and not plan.unused_vcas:
        return [], [], []

    live_by_uuid = {_live_field(t, "id").lower(): _live_field(t, "id") for t in live_tracks if _live_field(t, "id")}
    live_by_name = {_live_field(t, "name").lower(): _live_field(t, "id") for t in live_tracks if _live_field(t, "id")}

    def _resolve_id(trk: PtxTrack) -> str | None:
        tid = live_by_uuid.get(trk.uuid_braced.lower()) or live_by_name.get(trk.name.lower())
        if tid is None:
            log.warning("Could not resolve live PTSL track ID for %r; skipping deletion.", trk.name)
        return tid

    deleted_vcas: list[str] = []
    deleted_folders: list[str] = []
    errors: list[tuple[str, str, str]] = []
    blocked_folders: set[str] = set()
    parent_of = {t.name.lower(): t.parent_folder_name.lower() for t in ptx.tracks if t.parent_folder_name}

    vca_pairs = [(v.name, tid) for v in plan.unused_vcas if (tid := _resolve_id(v))]
    if vca_pairs:
        ok, failed, err = _delete_track_batch(engine, vca_pairs, batch_job_id=batch_job_id, progress=progress)
        deleted_vcas.extend(ok)
        if err:
            errors.append(("remove_unused_vca", ", ".join(failed), err))
            for fname in failed:
                _mark_ancestors(parent_of.get(fname.lower(), ""), parent_of, blocked_folders)

    by_depth: dict[int, list[tuple[str, str]]] = {}
    for ef in plan.empty_folders:
        tid = _resolve_id(ef)
        if tid is None:
            _mark_ancestors(ef.name.lower(), parent_of, blocked_folders)
            continue
        by_depth.setdefault(plan.folder_depths.get(ef.name, 0), []).append((ef.name, tid))
    for depth in sorted(by_depth, reverse=True):
        pairs = [(n, tid) for n, tid in by_depth[depth] if n.lower() not in blocked_folders]
        if not pairs:
            continue
        ok, failed, err = _delete_track_batch(engine, pairs, batch_job_id=batch_job_id, progress=progress)
        deleted_folders.extend(ok)
        if err:
            errors.append(("remove_empty_folder", ", ".join(failed), err))
            for fname in failed:
                _mark_ancestors(fname.lower(), parent_of, blocked_folders)

    return deleted_folders, deleted_vcas, errors


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
    empty_meta: dict[str, Any] = {
        "cleared_inputs": {}, "clip_gains": {}, "slot_a_trims": {}, "volumes": {}, "groups": [],
    }
    if not (clear_input_tracks or clip_gains_db or slot_a_trims_db or track_volumes_db or group_actions):
        return Path(session_path).read_bytes(), empty_meta

    def _transform(raw: bytes) -> tuple[bytes, dict[str, Any]]:
        cur = raw
        meta: dict[str, Any] = {
            "cleared_inputs": {}, "clip_gains": {}, "slot_a_trims": {}, "volumes": {}, "groups": [],
        }
        if clear_input_tracks:
            cur, meta["cleared_inputs"] = patch_clear_track_inputs(cur, clear_input_tracks)
        if clip_gains_db:
            cur, meta["clip_gains"] = patch_clip_gains(cur, clip_gains_db)
        if slot_a_trims_db:
            cur, meta["slot_a_trims"] = patch_slot_a_trim_gains(cur, slot_a_trims_db, activate_if_inactive=True)
        if track_volumes_db:
            cur, meta["volumes"] = patch_track_volumes(cur, track_volumes_db)
        for act in group_actions or ():
            if act.op == "add":
                cur, info = patch_add_group(
                    cur, act.group_name, list(act.member_tracks),
                    vca_track=act.vca_track, group_type=act.group_type,
                )
            else:
                kwargs: dict[str, Any] = {"add_tracks": list(act.member_tracks)}
                if act.vca_track is not None:
                    kwargs["vca_track"] = act.vca_track
                cur, info = patch_modify_group(cur, act.group_name, **kwargs)
            meta["groups"].append(info)
        return cur, meta

    return apply_atomic_ptx_patch(session_path, _transform)


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
    )


def _apply_group_actions_in_memory(raw_ptx: bytes, group_actions: list[FolderGroupAction]) -> PtxSession:
    """Apply planned group/VCA links to an in-memory ``.ptx`` copy so DAG summing sees active VCA offsets."""
    cur = raw_ptx
    for act in group_actions:
        if act.op == "add":
            cur, _ = patch_add_group(
                cur, act.group_name, list(act.member_tracks),
                vca_track=act.vca_track, group_type=act.group_type,
            )
        else:
            kwargs: dict[str, Any] = {"add_tracks": list(act.member_tracks)}
            if act.vca_track is not None:
                kwargs["vca_track"] = act.vca_track
            cur, _ = patch_modify_group(cur, act.group_name, **kwargs)
    return PtxSession.from_bytes(cur)


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
    clip_gains_db: dict[str, float] = {}
    track_volumes_db: dict[str, float] = {}
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]] = {}
    leaf_gains: dict[str, tuple[float, float]] = {}
    created_names: list[str] = []

    for created_name, _tid, staging, track in created_tracks:
        created_names.append(created_name)
        if abs(staging.clip_gain_db) >= 0.05:
            clip_gains_db[created_name] = staging.clip_gain_db
        if abs(staging.fader_db) >= 0.05:
            track_volumes_db[created_name] = staging.fader_db
        leaf_entries[created_name.lower()] = (track, staging)
        leaf_gains[created_name] = (staging.clip_gain_db, staging.fader_db)

    group_actions = (
        plan_folder_groups_and_vcas(ptx, created_names, link_vcas=link_template_vcas, group_type=group_type)
        if auto_group_folders
        else []
    )
    dag_ptx = _apply_group_actions_in_memory(raw_bytes, group_actions) if group_actions else ptx

    sf_cfg = dict(session.config)
    sf_cfg["protools_pan_law_db"] = pan_law_db
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
            leaf_loader=_lazy_leaf,
            leaf_gains=leaf_gains,
            bus_fader_overrides=bus_fader_overrides,
        )
        slot_a_trims_db = graph.recommended_slot_a_trims()
        for bus_name, bus_db in graph.recommended_bus_faders().items():
            if bus_name not in track_volumes_db:
                track_volumes_db[bus_name] = bus_db

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
