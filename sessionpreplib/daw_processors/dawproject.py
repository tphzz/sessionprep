"""DAWproject file-based DAW processor."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
from threading import RLock
from functools import wraps
import zipfile
from typing import Any

from ..daw_processor import DawProcessor
from ..file_io import atomic_writer
from ..models import DawCommand, DawCommandResult, ParamSpec, SessionContext, TrackContext
from ..signal_flow import (
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    LeafAudioInput,
    SignalFlowGraph,
    TrackStagingPlan,
    build_signal_flow,
)
from .dawproject_signal_flow import (
    dawproject_config_params,
    dawproject_host_limits,
    dawproject_xml_to_mixer_session,
    patch_dawproject_channel_volumes_xml,
    prune_orphaned_submix_channels_xml,
    resolve_dawproject_track_staging,
)
from .dawproject_vst3 import (
    has_calibrated_vst3_trim_donor,
    insert_vst3_trim_slot1_xml,
    patch_vst3_trim_slot1_xml,
    select_vst3_trim_donor,
)


log = logging.getLogger(__name__)
_model_registry_lock = RLock()


def _with_model_registry(method):
    """The file-format library shares its object IDs across project instances."""
    @wraps(method)
    def locked(*args, **kwargs):
        with _model_registry_lock:
            return method(*args, **kwargs)
    return locked


def _is_folder_track(track) -> bool:
    """Return True when *track* is a DAWproject folder (``contentType`` includes ``tracks``)."""
    from dawproject import ContentType, Track

    if not isinstance(track, Track):
        return False
    for ct in getattr(track, "content_type", ()):
        if isinstance(ct, ContentType) and ct == ContentType.TRACKS:
            return True
        if not isinstance(ct, ContentType):
            try:
                if ContentType(ct) == ContentType.TRACKS:
                    return True
            except ValueError:
                pass
    return False


def _load_template_project(template: zipfile.ZipFile):
    """Parse ``project.xml`` after reserving all existing numeric ``id<N>`` attributes."""
    from dawproject import Project, Referenceable
    from lxml import etree

    root = etree.fromstring(template.read("project.xml"))
    Referenceable.reset_id()
    reserved = (n[2:] for n in root.xpath("//@id") if n.startswith("id") and n[2:].isdecimal())
    Referenceable.ID = max(map(int, reserved), default=-1) + 1
    project = Project.from_xml(root)
    return root, project


def _index_structure_elements(root, project) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map deserialized Track and Channel IDs back to their original XML elements."""
    from dawproject import Channel, Track

    folders: dict[str, Any] = {}
    channels: dict[str, Any] = {}

    def walk(elements, tracks):
        for element, track in zip(elements, tracks, strict=True):
            folders[track.id] = element
            if track.channel is not None:
                ch_elem = element.find("Channel")
                if ch_elem is not None:
                    channels[track.channel.id] = ch_elem
            walk(element.findall("Track"), [t for t in track.tracks if isinstance(t, Track)])

    walk(root.findall("Structure/Track"), [t for t in project.structure if isinstance(t, Track)])
    for ch_elem, ch in zip(
        root.findall("Structure/Channel"),
        [c for c in project.structure if isinstance(c, Channel)],
        strict=True,
    ):
        channels[ch.id] = ch_elem
    return folders, channels


def _copy_zip_member(source: zipfile.ZipFile, archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> None:
    """Copy a template archive member using a fresh ``ZipInfo`` (no stale extra/header state)."""
    dt = member.date_time
    if dt[0] < 1980:
        dt = (1980, 1, 1, 0, 0, 0)
    elif dt[0] > 2107:
        dt = (2107, 12, 31, 23, 59, 58)
    info = zipfile.ZipInfo(member.filename, date_time=dt)
    info.compress_type = member.compress_type
    info.external_attr = member.external_attr
    info.comment = member.comment
    info.file_size = member.file_size
    with source.open(member) as old, archive.open(info, "w") as new:
        shutil.copyfileobj(old, new)


def _is_folder_xml_element(elem: Any) -> bool:
    """Return True when an XML ``<Track>`` element is a folder track (``tracks`` contentType)."""
    raw = (elem.get("contentType") or "").replace(" ", ",")
    return any(tok.strip().lower() == "tracks" for tok in raw.split(","))


def _collect_dependent_timeline_nodes(root: Any, subtree_ids: set[str]) -> list[Any]:
    """Return Arrangement/Scene timeline nodes referencing any ID in *subtree_ids*."""
    if not subtree_ids:
        return []
    nodes: list[Any] = []
    seen: set[Any] = set()
    for node in root.xpath("//*[@track]"):
        if node.get("track") in subtree_ids and node not in seen:
            nodes.append(node)
            seen.add(node)
    for target in root.xpath("//Target[@parameter]"):
        if target.get("parameter") in subtree_ids:
            owner = target.getparent()
            if owner is not None and owner not in seen:
                nodes.append(owner)
                seen.add(owner)
    return nodes


def _has_external_routing_ref(
    root: Any,
    folder_elem: Any,
    dep_nodes: list[Any],
    all_removed_ids: set[str],
) -> bool:
    """Return True when any surviving element outside *folder_elem*/*dep_nodes* references *all_removed_ids*."""
    if not all_removed_ids:
        return False
    excluded = set(folder_elem.iter())
    for dep in dep_nodes:
        excluded.update(dep.iter())
    for attr in ("destination", "reference"):
        for node in root.xpath(f"//*[@{attr}]"):
            if node not in excluded and node.get(attr) in all_removed_ids:
                return True
    return False


def _remove_dependent_nodes(dep_nodes: list[Any]) -> None:
    """Remove dependent timeline nodes and prune any ``<Scene>`` left with zero children."""
    scenes_to_check: list[Any] = []
    for node in dep_nodes:
        parent = node.getparent()
        if parent is None:
            continue
        parent.remove(node)
        if parent.tag == "Scene":
            scenes_to_check.append(parent)
    for scene in scenes_to_check:
        if len(scene) == 0 and scene.getparent() is not None:
            scene.getparent().remove(scene)


def _is_vca_xml_element(elem: Any) -> bool:
    """Return True when an XML ``<Track>`` element is a VCA-only track (``Channel/@role='vca'``)."""
    if _is_folder_xml_element(elem):
        return False
    ch = elem.find("Channel")
    return ch is not None and (ch.get("role") or "").strip().lower() == "vca"


def _is_prunable_vca_only_folder(elem: Any, child_tracks: list[Any], other_folder_names: set[str]) -> bool:
    """Return True when a folder holding only VCA tracks is an empty folder (not a VCA-organizer folder)."""
    if not child_tracks:
        return True
    if elem.find("Channel") is None and len(child_tracks) > 1:
        return False
    folder_low = (elem.get("name") or "").strip().lower()
    for vca in child_tracks:
        vlow = (vca.get("name") or "").strip().lower()
        for other in other_folder_names:
            if other and other != folder_low and vlow in (f"{other}-vca", f"{other} vca", f"{other}_vca"):
                return False
    return True


def _prune_empty_dawproject_folders(root: Any) -> tuple[list[str], set[str]]:
    """Remove empty folder ``<Track>`` subtrees (bottom-up) and their dangling XML references.

    Preserves folders with non-VCA child tracks, dedicated VCA-organizer folders,
    and folders (or dependent lanes) referenced externally via ``@destination`` / ``@reference``.
    Returns ``(removed_folder_names, removed_channel_destinations)``.
    """
    structure = root.find("Structure")
    if structure is None:
        return [], set()
    removed_names: list[str] = []
    removed_dests: set[str] = set()
    changed = True
    while changed:
        changed = False
        all_folders = [el for el in structure.iter("Track") if el.getparent() is not None and _is_folder_xml_element(el)]
        for elem in reversed(all_folders):
            if elem.getparent() is None:
                continue
            child_tracks = elem.findall("Track")
            if any(not _is_vca_xml_element(child) for child in child_tracks):
                continue
            other_names = {
                (f.get("name") or "").strip().lower()
                for f in all_folders
                if f is not elem and f.getparent() is not None
            }
            if not _is_prunable_vca_only_folder(elem, child_tracks, other_names):
                continue
            subtree_ids = {nid for nid in elem.xpath(".//@id") if nid}
            dep_nodes = _collect_dependent_timeline_nodes(root, subtree_ids)
            all_removed_ids = set(subtree_ids)
            for dep in dep_nodes:
                all_removed_ids.update(nid for nid in dep.xpath(".//@id") if nid)
            if _has_external_routing_ref(root, elem, dep_nodes, all_removed_ids):
                continue
            for ch in elem.iter("Channel"):
                dest = (ch.get("destination") or "").strip()
                if dest:
                    removed_dests.add(dest)
            for snd in elem.iter("Send"):
                sdest = (snd.get("destination") or "").strip()
                if sdest:
                    removed_dests.add(sdest)
            name = elem.get("name") or elem.get("id") or "(unnamed)"
            elem.getparent().remove(elem)
            _remove_dependent_nodes(dep_nodes)
            removed_names.append(name)
            changed = True
    return removed_names, removed_dests


def _make_lxml_element(tag: str, attrib: dict[str, str] | None = None) -> Any:
    from lxml import etree

    return etree.Element(tag, attrib or {})


def _next_xml_id_factory(root: Any) -> Any:
    existing = [int(n[2:]) for n in root.xpath("//@id") if n.startswith("id") and n[2:].isdecimal()]
    counter = [max(existing, default=-1) + 1]

    def _alloc() -> str:
        val = f"id{counter[0]}"
        counter[0] += 1
        return val

    return _alloc


def _load_transfer_leaf_audio(track: TrackContext, plan: TrackStagingPlan) -> LeafAudioInput | None:
    import numpy as np
    import soundfile as sf

    same_src = track.audio_data is not None and (
        os.path.normcase(os.path.realpath(plan.filepath))
        == os.path.normcase(os.path.realpath(track.filepath))
    )
    if same_src and track.audio_data is not None:
        return LeafAudioInput(
            np.asarray(track.audio_data, dtype=np.float64),
            track.samplerate,
            clip_gain_db=0.0,
            fader_db=plan.fader_db,
            baked_gain_db=plan.baked_gain_db,
            trim_gain_db=plan.trim_gain_db,
        )
    try:
        samples, sr = sf.read(plan.filepath, dtype="float64", always_2d=False)
    except Exception as exc:
        log.warning("Could not read audio for DAWproject signal-flow summing (%s): %s", plan.filepath, exc)
        return None
    return LeafAudioInput(
        np.asarray(samples, dtype=np.float64),
        int(sr),
        clip_gain_db=0.0,
        fader_db=plan.fader_db,
        baked_gain_db=plan.baked_gain_db,
        trim_gain_db=plan.trim_gain_db,
    )


def _insert_leaf_vst3_trims(
    root: Any,
    leaf_xml_records: list[tuple[str, Any, TrackContext, TrackStagingPlan, str]],
    donors: tuple[Any, ...],
    *,
    apply_trim_gain: bool,
    next_id: Any,
    used_members: set[str],
    state_counter: list[int],
    extra_blobs: dict[str, bytes],
) -> dict[str, TrackStagingPlan]:
    """Insert Slot 1 VST3 trims onto leaf audio tracks; fail-closed leaf faders if gain patch fails."""
    plan_overrides: dict[str, TrackStagingPlan] = {}
    reset_vols: dict[str, float] = {}
    for _tname, trk_el, _tc, plan, fmt in leaf_xml_records:
        ch_el = trk_el.find("Channel")
        if ch_el is None:
            continue
        cid = (ch_el.get("id") or "").strip()
        need_gain = apply_trim_gain and abs(plan.trim_gain_db) >= 0.05
        donor = select_vst3_trim_donor(donors, fmt, require_calibrated=need_gain) if donors else None
        gain_applied = False
        if donor is not None:
            meta = insert_vst3_trim_slot1_xml(
                ch_el, donor, make_element=_make_lxml_element, next_xml_id=next_id,
                used_members=used_members, state_counter=state_counter,
                extra_archive_blobs=extra_blobs, apply_gain_db=plan.trim_gain_db if need_gain else None,
            )
            gain_applied = bool(meta.get("gain_applied"))
        if need_gain and not gain_applied and abs(plan.baked_gain_db) < 1e-9:
            log.warning("VST3 Trim gain patch skipped/failed on leaf %r; resetting fader to 0.0 dB.", cid)
            if cid:
                reset_vols[cid] = 0.0
                plan_overrides[cid] = TrackStagingPlan(
                    filepath=plan.filepath, clip_gain_db=0.0, trim_gain_db=0.0,
                    fader_db=0.0, baked_gain_db=0.0, prepared_fader_db=plan.prepared_fader_db,
                )
    if reset_vols:
        patch_dawproject_channel_volumes_xml(
            root, reset_vols, make_element=_make_lxml_element, next_xml_id=next_id,
        )
    return plan_overrides


def _patch_graph_bus_trims(
    root: Any,
    graph: SignalFlowGraph,
    mmap: Any,
    read_member: Any,
    *,
    next_id: Any,
    used_members: set[str],
    state_counter: list[int],
    extra_blobs: dict[str, bytes],
) -> set[str]:
    """Patch or auto-insert calibrated Slot 1 VST3 trims on staged busses; return failed channel IDs."""
    rec_trims = graph.recommended_slot_a_trims()
    if not rec_trims:
        return set()
    by_cid = {(ch.get("id") or "").strip(): ch for ch in root.iter("Channel") if (ch.get("id") or "").strip()}
    failed_cids: set[str] = set()
    for bus_name, target_db in rec_trims.items():
        cid = mmap.channel_id_by_node.get(bus_name)
        node = graph.nodes.get(bus_name)
        ch_el = by_cid.get(cid) if cid else None
        if not cid or node is None or ch_el is None:
            if cid:
                failed_cids.add(cid)
            continue
        if node.slot_a_trim_auto_insert:
            donor = select_vst3_trim_donor(mmap.donors, node.format, require_calibrated=True)
            if donor is None:
                log.warning("No calibrated VST3 donor to auto-insert Slot 1 Trim on bus %r.", bus_name)
                failed_cids.add(cid)
                continue
            meta = insert_vst3_trim_slot1_xml(
                ch_el, donor, make_element=_make_lxml_element, next_xml_id=next_id,
                used_members=used_members, state_counter=state_counter,
                extra_archive_blobs=extra_blobs, apply_gain_db=target_db,
            )
            if not meta.get("gain_applied"):
                failed_cids.add(cid)
        elif not patch_vst3_trim_slot1_xml(
            ch_el, target_db, read_member, make_element=_make_lxml_element,
            next_xml_id=next_id, used_members=used_members,
            state_counter=state_counter, extra_archive_blobs=extra_blobs,
        ):
            log.warning("Could not patch Slot 1 VST3 Trim on bus %r; skipping compensating fader.", bus_name)
            failed_cids.add(cid)
    return failed_cids


def _patch_graph_channel_volumes(
    root: Any,
    graph: SignalFlowGraph,
    mmap: Any,
    leaf_cids: list[str],
    next_id: Any,
    *,
    skip_cids: set[str] | None = None,
) -> None:
    blocked = skip_cids or set()
    vols_to_patch: dict[str, float] = {}
    if abs(graph.leaf_fader_shift_db) >= 0.05 and not blocked:
        for cid in leaf_cids:
            nm = mmap.node_by_channel_id.get(cid)
            if nm and nm in graph.nodes:
                vols_to_patch[cid] = round(graph.nodes[nm].recommended_fader_db, 1)
    for bus_name, bus_db in graph.recommended_bus_faders().items():
        cid = mmap.channel_id_by_node.get(bus_name)
        if cid and cid not in blocked and cid not in vols_to_patch:
            vols_to_patch[cid] = bus_db
    if vols_to_patch:
        patch_dawproject_channel_volumes_xml(
            root, vols_to_patch, make_element=_make_lxml_element, next_xml_id=next_id,
        )


def _collect_leaf_signal_flow_inputs(
    leaf_xml_records: list[tuple[str, Any, TrackContext, TrackStagingPlan, str]],
    plan_overrides: dict[str, TrackStagingPlan],
    mmap: Any,
) -> tuple[list[str], dict[str, tuple[TrackContext, TrackStagingPlan]], dict[str, tuple[float, float, float, float]]]:
    leaf_cids: list[str] = []
    leaf_entries: dict[str, tuple[TrackContext, TrackStagingPlan]] = {}
    leaf_gains: dict[str, tuple[float, float, float, float]] = {}
    for _tname, trk_el, tc, plan, _fmt in leaf_xml_records:
        ch_el = trk_el.find("Channel")
        cid = (ch_el.get("id") or "").strip() if ch_el is not None else ""
        eff_plan = plan_overrides.get(cid, plan)
        node_name = mmap.node_by_channel_id.get(cid) if cid else None
        if cid and node_name:
            leaf_cids.append(cid)
            leaf_entries[node_name.lower()] = (tc, eff_plan)
            leaf_gains[node_name] = (0.0, eff_plan.fader_db, eff_plan.baked_gain_db, eff_plan.trim_gain_db)
    return leaf_cids, leaf_entries, leaf_gains


def _apply_signal_flow_and_trims(
    source: zipfile.ZipFile,
    root: Any,
    session: SessionContext,
    leaf_xml_records: list[tuple[str, Any, TrackContext, TrackStagingPlan, str]],
    *,
    gain_mode: str,
    insert_track_trim: bool,
    apply_trim_gain: bool,
    stage_bus_faders: bool,
    pan_law_db: float,
    fader_ceiling_db: float,
    bus_fader_overrides: dict[str, float] | None,
    used_members: set[str],
    extra_blobs: dict[str, bytes],
) -> SignalFlowGraph | None:
    """Insert Slot 1 VST3 trims on leaves/busses, run signal-flow DAG, and patch staged channel volumes."""
    if not leaf_xml_records:
        return None
    names_lc = {m.replace("\\", "/").lower(): m for m in source.namelist()}

    def _read_member(rel: str) -> bytes | None:
        actual = names_lc.get(rel.replace("\\", "/").lower())
        return source.read(actual) if actual else None

    next_id = _next_xml_id_factory(root)
    state_counter = [0]
    mmap = dawproject_xml_to_mixer_session(root, _read_member, apply_trim_gain=apply_trim_gain)
    plan_overrides = (
        _insert_leaf_vst3_trims(
            root, leaf_xml_records, mmap.donors, apply_trim_gain=apply_trim_gain,
            next_id=next_id, used_members=used_members, state_counter=state_counter, extra_blobs=extra_blobs,
        )
        if insert_track_trim
        else {}
    )
    leaf_cids, leaf_entries, leaf_gains = _collect_leaf_signal_flow_inputs(leaf_xml_records, plan_overrides, mmap)
    cal_donor = select_vst3_trim_donor(mmap.donors, "Stereo", require_calibrated=True)
    limits = dawproject_host_limits(
        fader_ceiling_db,
        leaf_trim_min_db=cal_donor.trim_info.min_db if cal_donor else None,
        leaf_trim_max_db=cal_donor.trim_info.max_db if cal_donor else None,
    )
    allow_conv = (
        apply_trim_gain
        and has_calibrated_vst3_trim_donor(mmap.donors)
        and gain_mode == GAIN_MODE_TRIM_AND_FADER
        and stage_bus_faders
    )
    graph = build_signal_flow(
        mmap.mixer, None, dict(session.config),
        host_limits=limits, pan_law_db=pan_law_db,
        stage_bus_faders=stage_bus_faders, attenuate_only=(gain_mode != GAIN_MODE_TRIM_UNITY),
        strict_slot_a=False, allow_leaf_convergence=allow_conv,
        leaf_loader=lambda nm: _load_transfer_leaf_audio(*leaf_entries[nm.lower()]) if nm.lower() in leaf_entries else None,
        leaf_gains=leaf_gains, bus_fader_overrides=bus_fader_overrides,
    )
    failed_bus_cids = (
        _patch_graph_bus_trims(
            root, graph, mmap, _read_member, next_id=next_id,
            used_members=used_members, state_counter=state_counter, extra_blobs=extra_blobs,
        )
        if apply_trim_gain
        else set()
    )
    _patch_graph_channel_volumes(root, graph, mmap, leaf_cids, next_id, skip_cids=failed_bus_cids)
    return graph


def _write_dawproject_archive(
    stream: Any,
    source: zipfile.ZipFile,
    root: Any,
    extra_blobs: dict[str, bytes],
    embedded_files: list[tuple[str, str]],
) -> None:
    from lxml import etree

    extra_lc = {p.replace("\\", "/").lower() for p in extra_blobs}
    with zipfile.ZipFile(stream, "w", allowZip64=True, strict_timestamps=False) as archive:
        archive.comment = source.comment
        archive.writestr(
            "project.xml",
            etree.tostring(root, xml_declaration=True, encoding="UTF-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        if "metadata.xml" not in source.namelist():
            archive.writestr("metadata.xml", "<MetaData/>", compress_type=zipfile.ZIP_DEFLATED)
        for member in source.infolist():
            norm = member.filename.replace("\\", "/").lower()
            if norm == "project.xml" or norm in extra_lc:
                continue
            _copy_zip_member(source, archive, member)
        for blob_path, blob_bytes in extra_blobs.items():
            archive.writestr(blob_path, blob_bytes, compress_type=zipfile.ZIP_DEFLATED)
        for src_filepath, arcname in embedded_files:
            archive.write(src_filepath, arcname=arcname, compress_type=zipfile.ZIP_STORED)


def _save_template(
    source,
    root,
    output_path,
    project,
    additions,
    embedded_files,
    replace_existing,
    *,
    session: SessionContext | None = None,
    stage_options: dict[str, Any] | None = None,
    used_members: set[str] | None = None,
) -> tuple[list[str], list[str], SignalFlowGraph | None]:
    """Append new tracks, prune empty folders/buses, apply VST3 trims/signal flow, and write ZIP."""
    from lxml import etree

    template = Path(source.filename)
    opts = stage_options or {}
    members_seen = used_members if used_members is not None else {m.replace("\\", "/").lower() for m in source.namelist()}
    extra_blobs: dict[str, bytes] = {}
    with atomic_writer(output_path, binary=True, replace_existing=replace_existing) as stream:
        for item in root.iterfind(".//*[@path]"):
            path = item.get("path")
            if (item.get("external") in ("true", "1") and path and "://" not in path
                    and not Path(path).is_absolute() and not PureWindowsPath(path).is_absolute()):
                item.set("path", (template.parent / path).resolve().as_posix())

        folders, channels = _index_structure_elements(root, project)
        arrangement = root.find("Arrangement")
        if arrangement is None:
            arrangement = etree.Element("Arrangement")
            root.insert(root.index(root.find("Structure")) + 1, arrangement)
        lanes = arrangement.find("Lanes")
        if lanes is None:
            lanes = etree.Element("Lanes", timeUnit="seconds")
            arrangement.insert(0, lanes)

        leaf_xml_records: list[tuple[str, Any, TrackContext, TrackStagingPlan, str]] = []
        for folder, track, lane, dest_channel, out_tc, plan in additions:
            element = folders[folder.id]
            if dest_channel is not None and dest_channel.id in channels:
                channels[dest_channel.id].set("id", dest_channel.id)
            trk_el = track.to_xml()
            element.append(trk_el)
            lanes.append(lane.to_xml())
            fmt = "Stereo" if (out_tc.channels and out_tc.channels >= 2) else "Mono"
            leaf_xml_records.append((track.name or "", trk_el, out_tc, plan, fmt))

        removed_folders, removed_dests = _prune_empty_dawproject_folders(root)
        removed_buses = (
            prune_orphaned_submix_channels_xml(root, removed_dests)
            if bool(opts.get("remove_unused_buses", True))
            else []
        )
        graph: SignalFlowGraph | None = None
        if session is not None and leaf_xml_records:
            graph = _apply_signal_flow_and_trims(
                source, root, session, leaf_xml_records,
                gain_mode=str(opts.get("gain_mode", GAIN_MODE_TRIM_AND_FADER)),
                insert_track_trim=bool(opts.get("insert_track_trim", True)),
                apply_trim_gain=bool(opts.get("apply_trim_gain", False)),
                stage_bus_faders=bool(opts.get("stage_bus_faders", True)),
                pan_law_db=float(opts.get("pan_law_db", -3.0)),
                fader_ceiling_db=float(opts.get("fader_ceiling_db", 6.0)),
                bus_fader_overrides=opts.get("bus_fader_overrides"),
                used_members=members_seen,
                extra_blobs=extra_blobs,
            )
        _write_dawproject_archive(stream, source, root, extra_blobs, embedded_files)
    return removed_folders, removed_buses, graph


def _allocate_archive_audio_path(
    source_path: str,
    used_members: set[str],
    embedded_by_source: dict[str, str],
) -> tuple[str, bool]:
    """Return ``(archive_path, needs_write)`` inside ``audio/`` without collisions."""
    key = os.path.normcase(os.path.realpath(source_path))
    existing = embedded_by_source.get(key)
    if existing is not None:
        return existing, False
    base = (Path(source_path).name or "audio.wav").replace("\\", "_")
    stem, ext = os.path.splitext(base)
    candidate = f"audio/{base}"
    suffix = 2
    while candidate.lower() in used_members:
        candidate = f"audio/{stem}_{suffix}{ext}"
        suffix += 1
    used_members.add(candidate.lower())
    embedded_by_source[key] = candidate
    return candidate, True


def _find_master_channel(root, project) -> Any:
    """Return the template's main output master channel (Cubase standalone or Bitwig Track)."""
    from dawproject import Channel, MixerRole, Track

    _, channel_elements = _index_structure_elements(root, project)
    targeted_ids = set(root.xpath("//Channel/@destination"))
    candidates: list[Any] = []
    for item in getattr(project, "structure", ()):
        if isinstance(item, Channel):
            ch = item
        elif isinstance(item, Track) and not _is_folder_track(item) and item.channel is not None:
            ch = item.channel
        else:
            continue
        role = getattr(ch, "role", None)
        if role != MixerRole.MASTER and str(role) != "master":
            continue
        ch_elem = channel_elements.get(ch.id)
        if getattr(ch, "destination", None) is not None or (ch_elem is not None and ch_elem.get("destination")):
            continue
        candidates.append(ch)
    for ch in candidates:
        if getattr(ch, "id", None) in targeted_ids:
            return ch
    return candidates[0] if candidates else None


def _resolve_channel_destination(folder_track, folder_parent_map: dict[str, Any], master_channel) -> Any:
    """Resolve routing target: folder bus → nearest ancestor routing folder → master bus."""
    current = folder_track
    while current is not None:
        if getattr(current, "channel", None) is not None:
            return current.channel
        current = folder_parent_map.get(current.id)
    return master_channel


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear gain (0 dB → 1.0)."""
    return math.pow(10.0, db / 20.0)


def _argb_to_rgb_hex(argb: str) -> str | None:
    """Convert an ARGB hex string (e.g. 'FF3399CC') to '#rrggbb'."""
    argb = argb.lstrip("#")
    if len(argb) == 8:
        return f"#{argb[2:]}"
    if len(argb) == 6:
        return f"#{argb}"
    return None


class DawProjectDawProcessor(DawProcessor):
    """DAW processor that writes .dawproject files.

    DAWproject is an open interchange format for DAW sessions.
    This processor generates .dawproject files from the session state
    rather than communicating with a running DAW instance.

    Each configured template becomes a separate instance with its own
    ``id`` and ``name``, created via :meth:`create_instances`.
    """

    id = "dawproject"
    name = "DAWproject"
    fader_ceiling_db: float = 6.0
    output_file_filter: str = "DAWproject (*.dawproject);;All Files (*)"
    output_dialog_title: str = "Save DAWproject"

    @property
    def template_identity(self) -> tuple[str, ...]:
        return ("dawproject", self._template_name, self._template_path)

    @property
    def template_path(self) -> str:
        return self._template_path

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + dawproject_config_params()

    def __init__(
        self,
        *,
        instance_index: int | None = None,
        template_name: str = "",
        template_path: str = "",
        template_fader_ceiling_db: float = 6.0,
    ):
        self._instance_index = instance_index
        self._template_name = template_name
        self._template_path = template_path
        if instance_index is not None:
            self.id = f"dawproject_{instance_index}"
            self.name = f"DAWproject \u2013 {template_name}"
            self.fader_ceiling_db = template_fader_ceiling_db

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create_instances(
        cls, flat_config: dict[str, Any],
    ) -> list[DawProjectDawProcessor]:
        """Create one processor instance per configured template.

        Reads ``dawproject_templates`` from *flat_config*.  Each entry
        is a dict with keys ``name``, ``template_path``, and optionally
        ``fader_ceiling_db``.  Returns an empty list when no templates
        are configured (the base "DAWproject" entry in the dropdown is
        suppressed in that case).
        """
        templates = flat_config.get("dawproject_templates", [])
        if not isinstance(templates, list):
            return []
        instances: list[DawProjectDawProcessor] = []
        for idx, tpl in enumerate(templates):
            if not isinstance(tpl, dict):
                continue
            name = tpl.get("name", "").strip()
            path = tpl.get("template_path", "").strip()
            ceiling = float(tpl.get("fader_ceiling_db", 6.0))
            if not name or not path:
                continue
            instances.append(cls(
                instance_index=idx,
                template_name=name,
                template_path=path,
                template_fader_ceiling_db=ceiling,
            ))
        return instances

    # ── Config ─────────────────────────────────────────────────────────

    def configure(self, config: dict[str, Any]) -> None:
        # For template instances the enabled toggle is governed by the
        # base dawproject_enabled key.
        saved = config.get(f"{self.id}_enabled")
        if saved is None:
            config[f"{self.id}_enabled"] = config.get("dawproject_enabled", True)
        super().configure(config)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def check_connectivity(self) -> tuple[bool, str]:
        if not self._template_path:
            return False, "No template file configured."
        if not os.path.isfile(self._template_path):
            return False, f"Template not found: {self._template_path}"
        try:
            with zipfile.ZipFile(self._template_path, "r") as zf:
                if "project.xml" not in zf.namelist():
                    return False, "Template ZIP missing project.xml."
        except zipfile.BadZipFile:
            return False, "Template file is not a valid ZIP archive."
        return True, f"Template OK: {os.path.basename(self._template_path)}"

    @_with_model_registry
    def fetch(
        self,
        session: SessionContext,
        *,
        ignore_cache: bool = False,
    ) -> SessionContext:
        try:
            import dawproject as _dp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "dawproject package not installed. "
                "Install SessionPrep with its gui extra."
            ) from exc

        with zipfile.ZipFile(self._template_path) as template:
            _, project = _load_template_project(template)

        folders: list[dict[str, Any]] = []
        self._walk_structure(
            project.structure, folders, parent_id=None, counter=[0])

        # Preserve existing assignments, track order, and bus overrides where IDs still match
        dp_state = session.daw_state.get(self.id, {})
        old_assignments: dict[str, str] = dp_state.get("assignments", {})
        valid_ids = {f["id"] for f in folders}
        assignments = {
            fname: fid for fname, fid in old_assignments.items()
            if fid in valid_ids
        }
        old_order: dict[str, list[str]] = dp_state.get("track_order", {})
        track_order = {
            fid: [eid for eid in eids if assignments.get(eid) == fid]
            for fid, eids in old_order.items()
            if fid in valid_ids
        }
        track_order = {fid: eids for fid, eids in track_order.items() if eids}

        new_state: dict[str, Any] = {
            "folders": folders,
            "assignments": assignments,
        }
        if track_order:
            new_state["track_order"] = track_order
        if isinstance(dp_state.get("bus_fader_overrides"), dict) and dp_state["bus_fader_overrides"]:
            new_state["bus_fader_overrides"] = dict(dp_state["bus_fader_overrides"])
        session.daw_state[self.id] = new_state
        return session

    def _walk_structure(
        self,
        tracks: list,
        folders: list[dict[str, Any]],
        parent_id: str | None,
        counter: list[int],
    ) -> None:
        """Recursively collect folder tracks from the project structure."""
        for track in tracks:
            if _is_folder_track(track):
                folder_type = "routing" if track.channel else "basic"
                folders.append({
                    "id": track.id,
                    "name": track.name or "(unnamed)",
                    "folder_type": folder_type,
                    "index": counter[0],
                    "parent_id": parent_id,
                })
                counter[0] += 1
                self._walk_structure(
                    track.tracks, folders, parent_id=track.id,
                    counter=counter)

    def default_output_path(self, session: SessionContext) -> str:
        """Compute the default ``.dawproject`` output path (pure, Qt-free)."""
        source_dir = session.config.get("_source_dir", "")
        output_folder = session.config.get("_output_folder", "sp_02_prepared")
        output_dir = os.path.join(source_dir, output_folder) if source_dir else ""

        safe_name = (
            session.project_name
            if getattr(session, "project_name", "")
            else (self._template_name or self.name or "dawproject")
        )
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in safe_name
        )
        filename = f"{safe_name}.dawproject"
        return os.path.join(output_dir, filename) if output_dir else filename

    @_with_model_registry
    def transfer(
        self,
        session: SessionContext,
        output_path: str,
        progress_cb=None,
        close_when_done: bool = True,
    ) -> list[DawCommandResult]:
        if not session.daw_state.get(self.id, {}).get("assignments"):
            return []
        try:
            import dawproject as _dp  # noqa: F401
        except ImportError:
            return [DawCommandResult(
                command=DawCommand("transfer", "", {}),
                success=False,
                error="dawproject package not installed",
            )]

        try:
            template = zipfile.ZipFile(self._template_path)
        except (OSError, zipfile.BadZipFile) as exc:
            return [DawCommandResult(DawCommand("load_template", "", {}), False, str(exc))]
        with template:
            try:
                root, project = _load_template_project(template)
            except Exception as exc:
                return [DawCommandResult(DawCommand("load_template", "", {}), False, f"Failed to load template: {exc}")]
            return self._add_tracks(session, output_path, progress_cb, template, root, project)

    @staticmethod
    def _ordered_assignments(assignments: dict[str, str], track_order: dict[str, list[str]]) -> list[tuple[str, str]]:
        work: list[tuple[str, str]] = []
        seen: set[str] = set()
        for fid, ordered_names in track_order.items():
            for eid in ordered_names:
                if eid not in seen and assignments.get(eid) == fid:
                    work.append((eid, fid))
                    seen.add(eid)
        for eid, fid in sorted(assignments.items()):
            if eid not in seen:
                work.append((eid, fid))
        return work

    def _build_track_addition(
        self, session, folder_track, dest_channel, entry, out_tc, plan: TrackStagingPlan, archive_audio_path: str,
    ):
        from dawproject import (
            Audio, BoolParameter, Channel, Clip, Clips, ContentType,
            FileReference, Lanes, MixerRole, RealParameter, TimeUnit, Track, Unit,
        )

        track_name = os.path.splitext(entry.daw_track_name)[0]
        track_color = self._resolve_track_color(entry.group, session)
        volume = _db_to_linear(plan.fader_db)
        audio_channels = out_tc.channels if (out_tc.channels and out_tc.channels > 0) else 2
        channel = Channel(
            role=MixerRole.REGULAR,
            audio_channels=audio_channels,
            name=track_name,
            color=track_color,
            destination=dest_channel,
            mute=BoolParameter(value=False, name="Mute"),
            pan=RealParameter(value=0.5, unit=Unit.NORMALIZED, min_value=0.0, max_value=1.0, name="Pan"),
            volume=RealParameter(
                value=volume, unit=Unit.LINEAR, min_value=0.0, max_value=max(2.0, volume), name="Volume",
            ),
        )
        new_track = Track(
            name=track_name,
            color=track_color,
            content_type={ContentType.AUDIO},
            loaded=True,
            channel=channel,
        )
        audio = Audio(
            time_unit=TimeUnit.SECONDS,
            file=FileReference(path=archive_audio_path, external=False),
            sample_rate=out_tc.samplerate,
            channels=audio_channels,
            duration=out_tc.duration_sec,
        )
        clip = Clip(
            content=audio,
            time=0.0,
            duration=out_tc.duration_sec,
            content_time_unit=TimeUnit.SECONDS,
            name=track_name,
            color=track_color,
        )
        track_lane = Lanes(track=new_track, time_unit=TimeUnit.SECONDS, lanes=[Clips(clips=[clip])])
        return track_name, (folder_track, new_track, track_lane, dest_channel, out_tc, plan)

    def _stage_options_from_session(self, session: SessionContext) -> dict[str, Any]:
        cfg = session.config or {}
        dp_state = session.daw_state.get(self.id, {})
        overrides = dp_state.get("bus_fader_overrides")
        return {
            "gain_mode": str(cfg.get("dawproject_gain_mode", GAIN_MODE_TRIM_AND_FADER)),
            "insert_track_trim": bool(cfg.get("dawproject_insert_track_trim", True)),
            "apply_trim_gain": bool(cfg.get("dawproject_apply_trim_gain", False)),
            "stage_bus_faders": bool(cfg.get("dawproject_stage_bus_faders", True)),
            "pan_law_db": float(cfg.get("dawproject_pan_law_db", -3.0)),
            "remove_unused_buses": bool(cfg.get("dawproject_remove_unused_buses", True)),
            "fader_ceiling_db": float(self.fader_ceiling_db),
            "bus_fader_overrides": overrides if isinstance(overrides, dict) else None,
        }

    def _save_dawproject_output(
        self, session, output_path, template, root, project, additions, embedded_files, results,
        *, used_members: set[str] | None = None,
    ) -> None:
        try:
            path = Path(output_path)
            if path.resolve() == Path(template.filename).resolve():
                raise ValueError("Choose a different output path to preserve the DAWproject template.")
            path.parent.mkdir(parents=True, exist_ok=True)
            removed_folders, removed_buses, graph = _save_template(
                template, root, path, project, additions, embedded_files,
                replace_existing=not session.config.get("_batch_transfer", False),
                session=session,
                stage_options=self._stage_options_from_session(session),
                used_members=used_members,
            )
            if graph is not None:
                session.daw_state.setdefault(self.id, {})["signal_flow"] = graph.to_dict()
            for fname in removed_folders:
                results.append(DawCommandResult(DawCommand("remove_empty_folder", fname, {}), True))
            for bname in removed_buses:
                results.append(DawCommandResult(DawCommand("remove_unused_bus", bname, {}), True))
            results.append(DawCommandResult(DawCommand("save_project", output_path, {}), True))
            log.info("DAWproject saved to %s", output_path)
        except Exception as exc:
            for result in results:
                if result.success:
                    result.success = False
                    result.error = "Track could not be published because the project save failed."
            results.append(DawCommandResult(DawCommand("save_project", output_path, {}), False, f"Failed to save: {exc}"))

    @staticmethod
    def _resolve_leaf_trim_capability(
        template: zipfile.ZipFile,
        root: Any,
        opts: dict[str, Any],
    ) -> tuple[bool, float | None, float | None]:
        """Verify upfront whether calibrated leaf Trim gain can be applied without balance inversion."""
        if not (opts["apply_trim_gain"] and opts["insert_track_trim"]):
            return False, None, None
        names_lc = {m.replace("\\", "/").lower(): m for m in template.namelist()}

        def _read_member(rel: str) -> bytes | None:
            actual = names_lc.get(rel.replace("\\", "/").lower())
            return template.read(actual) if actual else None

        mmap = dawproject_xml_to_mixer_session(root, _read_member, apply_trim_gain=True)
        donor = select_vst3_trim_donor(mmap.donors, "Stereo", require_calibrated=True)
        if donor is None:
            return False, None, None
        return True, donor.trim_info.min_db, donor.trim_info.max_db

    def _add_tracks(self, session, output_path, progress_cb, template, root, project):
        dp_state = session.daw_state.get(self.id, {})
        opts = self._stage_options_from_session(session)
        eff_leaf_apply, lt_min, lt_max = self._resolve_leaf_trim_capability(template, root, opts)
        folder_track_map: dict[str, Any] = {}
        folder_parent_map: dict[str, Any] = {}
        self._build_folder_map(project.structure, folder_track_map, folder_parent_map)
        master_channel = _find_master_channel(root, project)
        folder_dict_map = {f["id"]: f for f in dp_state.get("folders", [])}
        manifest_map = {e.entry_id: e for e in session.transfer_manifest}
        out_track_map = {t.filename: t for t in session.output_tracks}
        used_members = {m.replace("\\", "/").lower() for m in template.namelist()}
        embedded_by_source: dict[str, str] = {}
        embedded_files: list[tuple[str, str]] = []

        work = self._ordered_assignments(dp_state.get("assignments", {}), dp_state.get("track_order", {}))
        total = len(work)
        if progress_cb:
            progress_cb(0, total, "Building DAWproject\u2026")

        results: list[DawCommandResult] = []
        additions = []
        for step, (eid, fid) in enumerate(work):
            folder_track = folder_track_map.get(fid)
            folder_dict = folder_dict_map.get(fid)
            entry = manifest_map.get(eid)
            out_tc = out_track_map.get(entry.output_filename) if entry else None
            if not folder_track or not folder_dict or not entry or not out_tc:
                results.append(DawCommandResult(
                    DawCommand("add_track", eid, {"folder_id": fid}), False,
                    f"Folder, manifest entry, or output track not found: {fid} / {eid}",
                ))
                continue
            try:
                plan = resolve_dawproject_track_staging(
                    session, out_tc,
                    gain_mode=opts["gain_mode"],
                    apply_trim_gain=eff_leaf_apply,
                    fader_ceiling_db=opts["fader_ceiling_db"],
                    leaf_trim_min_db=lt_min,
                    leaf_trim_max_db=lt_max,
                )
            except ValueError as exc:
                results.append(DawCommandResult(DawCommand("add_track", eid, {"folder_id": fid}), False, str(exc)))
                continue
            arcname, needs_write = _allocate_archive_audio_path(plan.filepath, used_members, embedded_by_source)
            if needs_write:
                embedded_files.append((plan.filepath, arcname))
            dest_channel = _resolve_channel_destination(folder_track, folder_parent_map, master_channel)
            log.debug("%s: file=%s -> %s, trim=%+.2f dB, fader=%+.2f dB", eid, plan.filepath, arcname, plan.trim_gain_db, plan.fader_db)
            track_name, addition = self._build_track_addition(
                session, folder_track, dest_channel, entry, out_tc, plan, arcname,
            )
            additions.append(addition)
            results.append(DawCommandResult(
                DawCommand("add_track", eid, {"folder": folder_dict.get("name", ""), "fader_db": plan.fader_db}), True,
            ))
            if progress_cb:
                progress_cb(step + 1, total, f"Added {track_name} ({step + 1}/{total})")

        if not (results and not any(result.success for result in results)):
            self._save_dawproject_output(
                session, output_path, template, root, project, additions, embedded_files, results,
                used_members=used_members,
            )
        session.daw_command_log.extend(results)
        return results

    def _build_folder_map(
        self,
        tracks: list,
        folder_map: dict[str, Any],
        parent_map: dict[str, Any] | None = None,
        parent: Any = None,
    ) -> None:
        """Recursively build folder ID → Track and folder ID → parent Track mappings."""
        for track in tracks:
            if _is_folder_track(track):
                folder_map[track.id] = track
                if parent_map is not None and parent is not None:
                    parent_map[track.id] = parent
                self._build_folder_map(track.tracks, folder_map, parent_map, parent=track)

    def _resolve_track_color(
        self, group_name: str | None, session: SessionContext,
    ) -> str | None:
        """Return ``#rrggbb`` for the track's group color, or ``None``."""
        if not group_name:
            return None
        groups = session.config.get("gui", {}).get("groups", [])
        color_name: str | None = None
        for g in groups:
            if g.get("name") == group_name:
                color_name = g.get("color")
                break
        if not color_name:
            return None
        colors = session.config.get("gui", {}).get("colors", [])
        for c in colors:
            if c.get("name") == color_name:
                argb = c.get("argb")
                if argb:
                    return _argb_to_rgb_hex(argb)
        return None

    def sync(self, session: SessionContext) -> list[DawCommandResult]:
        return []

    def execute_commands(
        self, session: SessionContext, commands: list[DawCommand],
    ) -> list[DawCommandResult]:
        return []
