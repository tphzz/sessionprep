"""Offline Pro Tools ``.ptxt`` template topology inspection and signal-flow preview."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..models import SessionContext, TrackContext
from ..ptx import (
    LeafAudioInput,
    PtxGroup,
    PtxSession,
    PtxSignalPathRef,
    PtxTrack,
    SignalFlowGraph,
    build_annotated_signal_flow,
)
from .protools_ptx import (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    ProToolsTrackStaging,
    _find_folder_vca,
    plan_folder_groups_and_vcas,
    resolve_protools_track_staging,
)


@dataclass(frozen=True)
class TemplateFolderMeta:
    """Resolved offline routing and VCA metadata for one Pro Tools template folder."""

    folder_id: str
    folder_name: str
    folder_type: str  # "routing" | "basic"
    input_bus: str | None
    resolved_routing_bus: str | None
    vca_track: str | None
    output_chain: str


@dataclass(frozen=True)
class TemplateTopologySummary:
    """Parsed offline ``.ptxt`` folder metadata indexed by folder UUID and lowercase name."""

    ptxt_path: str
    by_id: dict[str, TemplateFolderMeta] = field(default_factory=dict)
    by_name: dict[str, TemplateFolderMeta] = field(default_factory=dict)

    def lookup(self, folder_id: str, folder_name: str) -> TemplateFolderMeta | None:
        return self.by_id.get(folder_id) or self.by_name.get(folder_name.lower())


def _resolve_ancestor_routing_bus(ptx: PtxSession, folder: PtxTrack) -> str | None:
    cur: PtxTrack | None = folder
    seen: set[str] = set()
    while cur is not None and cur.name.lower() not in seen:
        seen.add(cur.name.lower())
        if cur.track_type == "RoutingFolder":
            return cur.input_path.name if cur.input_path else None
        cur = ptx.track_by_name(cur.parent_folder_name) if cur.parent_folder_name else None
    return None


def _trace_output_chain(ptx: PtxSession, start: PtxTrack, max_hops: int = 4) -> str:
    by_input: dict[str, PtxTrack] = {}
    for trk in ptx.tracks:
        if trk.track_type in ("RoutingFolder", "Aux") and trk.input_path:
            by_input.setdefault(trk.input_path.name.lower(), trk)
    chain: list[str] = []
    cur: PtxTrack | None = start
    seen: set[str] = {start.name.lower()}
    for _ in range(max_hops):
        if cur is None or cur.main_output is None:
            break
        nxt = by_input.get(cur.main_output.name.lower())
        if nxt is None or nxt.name.lower() in seen:
            break
        seen.add(nxt.name.lower())
        chain.append(nxt.name)
        cur = nxt
    return ("→ " + " → ".join(chain)) if chain else ""


def inspect_protools_template_topology(ptxt_path: str | Path) -> TemplateTopologySummary:
    """Parse *ptxt_path* offline and return folder input bus, VCA, and output-chain metadata."""
    ptx = PtxSession.from_file(ptxt_path)
    by_id: dict[str, TemplateFolderMeta] = {}
    by_name: dict[str, TemplateFolderMeta] = {}
    for trk in ptx.tracks:
        if trk.track_type not in ("RoutingFolder", "BasicFolder"):
            continue
        ftype = "routing" if trk.track_type == "RoutingFolder" else "basic"
        in_bus = trk.input_path.name if (ftype == "routing" and trk.input_path) else None
        resolved_bus = _resolve_ancestor_routing_bus(ptx, trk)
        vca = _find_folder_vca(ptx, trk.name)
        vca_name = vca.name if vca is not None else None
        chain = _trace_output_chain(ptx, trk) if ftype == "routing" else (
            f"Walks up → Bus: {resolved_bus}" if resolved_bus else ""
        )
        meta = TemplateFolderMeta(
            folder_id=trk.uuid_braced,
            folder_name=trk.name,
            folder_type=ftype,
            input_bus=in_bus,
            resolved_routing_bus=resolved_bus,
            vca_track=vca_name,
            output_chain=chain,
        )
        if trk.uuid_braced:
            by_id[trk.uuid_braced] = meta
        by_name[trk.name.lower()] = meta
    return TemplateTopologySummary(ptxt_path=str(ptxt_path), by_id=by_id, by_name=by_name)


def _find_folder_track(ptx: PtxSession, folder_id: str, folder_lookup: dict[str, dict[str, Any]]) -> PtxTrack | None:
    by_uuid = {t.uuid_braced.lower(): t for t in ptx.tracks if t.uuid_braced}
    trk = by_uuid.get(folder_id.lower())
    if trk is not None:
        return trk
    fdict = folder_lookup.get(folder_id)
    if fdict and fdict.get("name"):
        return ptx.track_by_name(str(fdict["name"]))
    return None


def _apply_virtual_folder_groups(
    ptx: PtxSession,
    virtual_names: list[str],
    *,
    link_template_vcas: bool,
    group_type: str,
) -> None:
    """Mirror ``plan_folder_groups_and_vcas`` on virtual preview tracks so VCA offsets apply."""
    actions = plan_folder_groups_and_vcas(
        ptx, virtual_names, link_vcas=link_template_vcas, group_type=group_type,
    )
    by_name = {t.name.lower(): t for t in ptx.tracks}
    is_mix = group_type.strip().lower() != "edit"
    for act in actions:
        if act.op == "modify":
            grp = ptx.group_by_name(act.group_name)
            if grp is not None:
                for member in act.member_tracks:
                    if member not in grp.member_names:
                        grp.member_names.append(member)
                if act.vca_track and not grp.vca_track_name:
                    grp.vca_track_name = act.vca_track
        else:
            ptx.groups.append(
                PtxGroup(
                    index=len(ptx.groups),
                    name=act.group_name,
                    kind="mix" if is_mix else "edit",
                    group_type=act.group_type,
                    id_number=len(ptx.groups) + 1,
                    id_letter="",
                    is_mix_active=is_mix,
                    vca_track_name=act.vca_track,
                    member_names=list(act.member_tracks),
                )
            )
        for member in act.member_tracks:
            trk = by_name.get(member.lower())
            if trk is not None and act.group_name not in trk.groups:
                trk.groups.append(act.group_name)


def _inject_virtual_leaves(
    ptx: PtxSession,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    *,
    auto_route: bool,
    gain_mode: str,
    auto_group_folders: bool,
    link_template_vcas: bool,
    group_type: str,
    excluded: dict[str, str],
) -> tuple[dict[str, tuple[TrackContext, ProToolsTrackStaging]], dict[str, tuple[float, float]]]:
    if session.config.get("_use_processed") is True and session.prepare_state != "ready":
        raise ValueError("Prepared audio is out of date. Run Prepare again (or uncheck Use Processed).")
    out_map = {t.filename: t for t in session.output_tracks}
    folder_lookup = {str(f.get("id", "")): f for f in folders}
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]] = {}
    leaf_gains: dict[str, tuple[float, float]] = {}
    virtual_names: list[str] = []
    used_names = {t.name.lower() for t in ptx.tracks}

    for entry in session.transfer_manifest:
        fid = assignments.get(entry.entry_id)
        track = out_map.get(entry.output_filename) if fid else None
        folder_trk = _find_folder_track(ptx, fid, folder_lookup) if fid else None
        if track is None or folder_trk is None:
            continue
        target_bus = _resolve_ancestor_routing_bus(ptx, folder_trk) if auto_route else None
        if auto_route and not target_bus:
            continue
        try:
            staging = resolve_protools_track_staging(session, track, gain_mode=gain_mode)
        except (ValueError, OSError) as exc:
            excluded[track.filename] = str(exc)
            continue
        vname = _unique_virtual_name(entry.daw_track_name, used_names)
        main_out = PtxSignalPathRef(name=target_bus, kind="bus", uuid_hex="") if target_bus else None
        ptx.tracks.append(
            PtxTrack(
                index=len(ptx.tracks), name=vname, track_type="Audio",
                format="Stereo" if track.channels >= 2 else "Mono",
                uuid_hex="", uuid_braced="", parent_folder_name=folder_trk.name,
                main_output=main_out, volume_db=staging.fader_db,
            )
        )
        leaf_entries[vname.lower()] = (track, staging)
        leaf_gains[vname] = (staging.clip_gain_db, staging.fader_db)
        virtual_names.append(vname)

    if auto_group_folders and virtual_names:
        _apply_virtual_folder_groups(
            ptx, virtual_names, link_template_vcas=link_template_vcas, group_type=group_type,
        )
    return leaf_entries, leaf_gains


def _unique_virtual_name(base_name: str, used_names: set[str]) -> str:
    vname = base_name
    suffix = 2
    while vname.lower() in used_names:
        vname = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(vname.lower())
    return vname


def _load_preview_leaf(
    track_name: str,
    leaf_entries: dict[str, tuple[TrackContext, ProToolsTrackStaging]],
    excluded: dict[str, str],
) -> LeafAudioInput | None:
    pair = leaf_entries.get(track_name.lower())
    if pair is None:
        return None
    track, staging = pair
    same_src = track.audio_data is not None and (
        os.path.normcase(os.path.realpath(staging.filepath))
        == os.path.normcase(os.path.realpath(track.filepath))
    )
    if same_src and track.audio_data is not None:
        arr = np.asarray(track.audio_data, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            excluded[track.filename] = "Non-finite samples (NaN/Inf)"
            return None
        return LeafAudioInput(arr, track.samplerate, staging.clip_gain_db, staging.fader_db)
    try:
        samples, sr = sf.read(staging.filepath, dtype="float64", always_2d=False)
        arr = np.asarray(samples, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            excluded[track.filename] = "Non-finite samples (NaN/Inf)"
            return None
        return LeafAudioInput(arr, int(sr), staging.clip_gain_db, staging.fader_db)
    except Exception as exc:
        excluded[track.filename] = str(exc)
        return None


def preview_protools_signal_flow(
    ptxt_path: str | Path,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    *,
    auto_route: bool = True,
    gain_mode: str = GAIN_MODE_CLIP_AND_FADER,
    auto_group_folders: bool = True,
    link_template_vcas: bool = True,
    group_type: str = "Mix/Edit",
    stage_bus_faders: bool = True,
    pan_law_db: float = -3.0,
    bus_fader_overrides: Mapping[str, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SignalFlowGraph:
    """Compute an offline annotated Pro Tools signal-flow preview directly from a ``.ptxt`` template."""
    ptx = PtxSession.from_file(ptxt_path)
    excluded: dict[str, str] = {}
    leaf_entries, leaf_gains = _inject_virtual_leaves(
        ptx, session, assignments, folders,
        auto_route=auto_route, gain_mode=gain_mode,
        auto_group_folders=auto_group_folders, link_template_vcas=link_template_vcas,
        group_type=group_type, excluded=excluded,
    )
    sf_cfg = dict(session.config)
    sf_cfg["protools_pan_law_db"] = pan_law_db
    attenuate_only = gain_mode != GAIN_MODE_CLIP_UNITY
    graph = build_annotated_signal_flow(
        ptx, None, sf_cfg,
        stage_bus_faders=stage_bus_faders,
        attenuate_only=attenuate_only,
        leaf_loader=lambda name: _load_preview_leaf(name, leaf_entries, excluded),
        leaf_gains=leaf_gains,
        bus_fader_overrides=bus_fader_overrides,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    graph.excluded_tracks = excluded
    return graph
