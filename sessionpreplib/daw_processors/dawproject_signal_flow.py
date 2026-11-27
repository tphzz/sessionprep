"""DAWproject XML signal-flow adapter, offline template preview, and bus XML patching.

Uses Python standard-library XML + ZIP primitives so offline topology inspection,
signal-flow preview, and XML-level channel patching work cleanly without requiring
``dawproject-py`` at import time and remain compatible with both ``xml.etree.ElementTree``
and ``lxml.etree`` elements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
import soundfile as sf

from ..models import ParamSpec, SessionContext, TrackContext
from ..signal_flow import (
    DAWPROJECT_GAIN_LIMITS,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    HostGainLimits,
    LeafAudioInput,
    MixerInsert,
    MixerSend,
    MixerSession,
    MixerTrack,
    SignalFlowGraph,
    TrackStagingPlan,
    build_signal_flow,
    resolve_track_staging,
)
from .dawproject_vst3 import (
    Vst3TrimDonor,
    has_calibrated_vst3_trim_donor,
    inspect_vst3_slot1_trim,
    select_vst3_trim_donor,
)

log = logging.getLogger(__name__)

VALID_DAWPROJECT_GAIN_MODES = (
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
)


def dawproject_config_params() -> list[ParamSpec]:
    """Return DAWproject gain-staging, VST3 Trim, and signal-flow bus parameters."""
    return [
        ParamSpec(
            key="dawproject_gain_mode",
            type=str,
            default=GAIN_MODE_TRIM_AND_FADER,
            choices=list(VALID_DAWPROJECT_GAIN_MODES),
            label="Gain Staging Mode",
            description=(
                "How normalization gain is staged in DAWproject: Slot 1 VST3 Trim + inverse Fader "
                "(default), Slot 1 Trim with unity faders, Fader only, or None."
            ),
        ),
        ParamSpec(
            key="dawproject_insert_track_trim",
            type=bool,
            default=True,
            label="Insert VST3 Trim on Audio Tracks (Slot 1)",
            description=(
                "Clone the template's verified VST3 Trim plugin (TrackControl / Protility) into "
                "Slot 1 of newly created audio tracks."
            ),
        ),
        ParamSpec(
            key="dawproject_apply_trim_gain",
            type=bool,
            default=False,
            label="Apply Gain to Inserted VST3 Trim (Experimental)",
            description=(
                "Patch the VST3 .vstpreset gain value directly. Keep off to insert unity (0.0 dB) "
                "TrackControl/Protility instances until calibrated against your target DAW."
            ),
        ),
        ParamSpec(
            key="dawproject_stage_bus_faders",
            type=bool,
            default=True,
            label="Stage Folder & Submix Bus Faders",
            description=(
                "Reconstruct the template signal-flow DAG, sum adjusted audio tracks, and pre-set "
                "routing folder and submix bus faders within the template fader ceiling."
            ),
        ),
        ParamSpec(
            key="dawproject_pan_law_db",
            type=float,
            default=-3.0,
            min=-6.0,
            max=0.0,
            label="Signal-Flow Pan Law (dB)",
            description="Pan depth used when summing mono/panned tracks through the signal-flow DAG.",
        ),
        ParamSpec(
            key="dawproject_remove_unused_buses",
            type=bool,
            default=True,
            label="Remove Orphaned Submix Buses",
            description=(
                "Remove template submix channels that have no surviving upstream tracks or sends "
                "after empty folder cleanup."
            ),
        ),
    ]


@dataclass(frozen=True)
class DawProjectFolderMeta:
    """Resolved offline routing metadata for one DAWproject template folder."""

    folder_id: str
    folder_name: str
    folder_type: str  # "routing" | "basic"
    bus_channel_id: str | None
    bus_channel_name: str | None
    resolved_routing_channel_id: str | None
    output_chain: str
    resolved_routing_channel_name: str | None = None


@dataclass(frozen=True)
class DawProjectTopologySummary:
    """Parsed offline ``.dawproject`` template folder metadata indexed by ID and lowercase name."""

    template_path: str
    by_id: dict[str, DawProjectFolderMeta] = field(default_factory=dict)
    by_name: dict[str, DawProjectFolderMeta] = field(default_factory=dict)
    donor_plugin_name: str | None = None
    has_calibrated_donor: bool = False
    donor_min_db: float | None = None
    donor_max_db: float | None = None

    def lookup(self, folder_id: str, folder_name: str) -> DawProjectFolderMeta | None:
        return self.by_id.get(folder_id) or self.by_name.get(folder_name.lower())


@dataclass
class DawProjectMixerMap:
    """Parsed DAWproject mixer snapshot plus XML-to-node lookup tables for patching."""

    mixer: MixerSession
    channel_id_by_node: dict[str, str]
    node_by_channel_id: dict[str, str]
    master_channel_id: str | None
    donors: tuple[Vst3TrimDonor, ...]
    sample_rate: int


def dawproject_host_limits(
    fader_ceiling_db: float = 6.0,
    *,
    leaf_trim_min_db: float | None = None,
    leaf_trim_max_db: float | None = None,
) -> HostGainLimits:
    """Return :class:`HostGainLimits` calibrated to *fader_ceiling_db* (default ``+6.0 dB``)."""
    ceil = float(fader_ceiling_db) if math.isfinite(float(fader_ceiling_db)) else 6.0
    lt_min = (
        float(leaf_trim_min_db)
        if (leaf_trim_min_db is not None and math.isfinite(float(leaf_trim_min_db)))
        else DAWPROJECT_GAIN_LIMITS.leaf_trim_min_db
    )
    lt_max = (
        float(leaf_trim_max_db)
        if (leaf_trim_max_db is not None and math.isfinite(float(leaf_trim_max_db)))
        else DAWPROJECT_GAIN_LIMITS.leaf_trim_max_db
    )
    return HostGainLimits(
        fader_min_db=DAWPROJECT_GAIN_LIMITS.fader_min_db,
        fader_max_db=max(0.0, min(48.0, ceil)),
        clip_gain_min_db=DAWPROJECT_GAIN_LIMITS.clip_gain_min_db,
        clip_gain_max_db=DAWPROJECT_GAIN_LIMITS.clip_gain_max_db,
        leaf_trim_min_db=lt_min,
        leaf_trim_max_db=lt_max,
        leaf_shift_floor_db=DAWPROJECT_GAIN_LIMITS.leaf_shift_floor_db,
        parallel_inf_db=DAWPROJECT_GAIN_LIMITS.parallel_inf_db,
    )


def resolve_dawproject_track_staging(
    session: SessionContext,
    track: TrackContext,
    *,
    gain_mode: str = GAIN_MODE_TRIM_AND_FADER,
    apply_trim_gain: bool = False,
    fader_ceiling_db: float = 6.0,
    leaf_trim_min_db: float | None = None,
    leaf_trim_max_db: float | None = None,
) -> TrackStagingPlan:
    """Resolve DAWproject transfer media, Slot 1 Trim gain, and Track Fader dB.

    When *apply_trim_gain* is ``False`` (Phase 1 unity-insert mode) on un-normalized audio,
    pre-insert Trim and inverse Fader both remain ``0.0 dB`` so rough-mix balance is never
    inverted before VST3 preset gain calibration is enabled. On prepared (gain-baked) audio,
    Track Fader carries ``prepared_fader_db`` in balance-preserving modes.
    """
    mode = gain_mode if gain_mode in VALID_DAWPROJECT_GAIN_MODES else GAIN_MODE_TRIM_AND_FADER
    limits = dawproject_host_limits(
        fader_ceiling_db, leaf_trim_min_db=leaf_trim_min_db, leaf_trim_max_db=leaf_trim_max_db,
    )
    plan = resolve_track_staging(session, track, mode, limits=limits)
    if not apply_trim_gain and abs(plan.baked_gain_db) < 1e-9 and abs(plan.trim_gain_db) > 1e-9:
        return TrackStagingPlan(
            filepath=plan.filepath,
            clip_gain_db=0.0,
            trim_gain_db=0.0,
            fader_db=0.0,
            baked_gain_db=0.0,
            prepared_fader_db=plan.prepared_fader_db,
        )
    return plan


def _local_tag(tag: Any) -> str:
    text = str(tag or "")
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _is_folder_elem(elem: Any) -> bool:
    raw = (elem.get("contentType") or "").replace(" ", ",")
    return any(tok.strip().lower() == "tracks" for tok in raw.split(","))


def _parse_volume_db(vol_elem: Any) -> float:
    if vol_elem is None:
        return 0.0
    raw = vol_elem.get("value")
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        return 0.0
    if not math.isfinite(val):
        return 0.0
    unit = (vol_elem.get("unit") or "linear").strip().lower()
    if unit == "decibel":
        return max(-144.0, min(48.0, val))
    if val <= 0.0:
        return -144.0
    return max(-144.0, min(48.0, 20.0 * math.log10(val)))


def _parse_pan(pan_elem: Any) -> float | None:
    if pan_elem is None:
        return None
    raw = pan_elem.get("value")
    if raw is None:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    lo_s, hi_s = pan_elem.get("min"), pan_elem.get("max")
    try:
        lo = float(lo_s) if lo_s is not None else 0.0
        hi = float(hi_s) if hi_s is not None else 1.0
    except ValueError:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        return 0.0
    norm = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    return round((norm - 0.5) * 200.0, 2)


def _xml_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    low = val.strip().lower()
    if low in ("true", "1"):
        return True
    if low in ("false", "0"):
        return False
    return default


def _collect_raw_channels(root: Any) -> list[tuple[Any | None, Any, str, str]]:
    """Return ``(track_elem_or_None, channel_elem, parent_folder_name, synthetic_id)`` in structure order."""
    structure = root.find("Structure")
    if structure is None:
        return []
    out: list[tuple[Any | None, Any, str, str]] = []
    syn_idx = [0]

    def _next_syn(ch_el: Any) -> str:
        cid = (ch_el.get("id") or "").strip()
        if cid:
            return cid
        syn_idx[0] += 1
        return f"__dp_ch_{syn_idx[0]}"

    def _walk_track(trk_el: Any, parent_name: str) -> None:
        ch_el = trk_el.find("Channel")
        if ch_el is not None:
            out.append((trk_el, ch_el, parent_name, _next_syn(ch_el)))
        my_name = str(trk_el.get("name") or "") if _is_folder_elem(trk_el) else parent_name
        for child in trk_el.findall("Track"):
            _walk_track(child, my_name)

    for child in list(structure):
        tag = _local_tag(child.tag)
        if tag == "Track":
            _walk_track(child, "")
        elif tag == "Channel":
            out.append((None, child, "", _next_syn(child)))
    return out


def _identify_master_channel_id(
    entries: list[tuple[Any | None, Any, str, str]],
) -> str | None:
    targeted = {
        (ch.get("destination") or "").strip()
        for _, ch, _, _ in entries
        if (ch.get("destination") or "").strip()
    }
    candidates: list[str] = []
    for trk_el, ch_el, _, cid in entries:
        if trk_el is not None and _is_folder_elem(trk_el):
            continue
        role = (ch_el.get("role") or "").strip().lower()
        dest = (ch_el.get("destination") or "").strip()
        if role == "master" and not dest:
            candidates.append(cid)
    for cid in candidates:
        if cid in targeted:
            return cid
    return candidates[0] if candidates else None


def _parse_channel_devices(
    ch_el: Any,
    read_member: Callable[[str], bytes | None],
    *,
    track_format: str,
    cid: str,
    idx: int,
) -> tuple[tuple[MixerInsert, ...], Vst3TrimDonor | None]:
    devices_el = ch_el.find("Devices")
    if devices_el is None:
        return (), None
    inserts: list[MixerInsert] = []
    donor: Vst3TrimDonor | None = None
    device_elems = [el for el in devices_el if isinstance(getattr(el, "tag", None), str)]
    for slot_i, dev_el in enumerate(device_elems):
        dev_name = str(dev_el.get("deviceName") or dev_el.get("name") or _local_tag(dev_el.tag))
        dev_id = str(dev_el.get("deviceID") or "")
        loaded = _xml_bool(dev_el.get("loaded"), True)
        en_el = dev_el.find("Enabled")
        enabled = _xml_bool(en_el.get("value"), True) if en_el is not None else True
        inserts.append(
            MixerInsert(
                slot_index=slot_i,
                plugin_name=dev_name,
                is_active=loaded,
                is_bypassed=not enabled,
                plugin_identifier=dev_id,
            )
        )
        if slot_i == 0:
            donor = inspect_vst3_slot1_trim(
                dev_el, read_member, track_format=track_format, source_channel_id=cid, source_index=idx,
            )
    return tuple(inserts), donor


def _parse_channel_sends(ch_el: Any) -> tuple[MixerSend, ...]:
    sends_el = ch_el.find("Sends")
    if sends_el is None:
        return ()
    sends: list[MixerSend] = []
    for s_idx, snd_el in enumerate(sends_el.findall("Send"), 1):
        dest = (snd_el.get("destination") or "").strip()
        if not dest:
            continue
        en_el = snd_el.find("Enable")
        enabled = _xml_bool(en_el.get("value"), True) if en_el is not None else True
        stype = (snd_el.get("type") or "post").strip().lower()
        vol_db = _parse_volume_db(snd_el.find("Volume"))
        slot = str(snd_el.get("name") or f"Send {s_idx}")
        sends.append(
            MixerSend(
                destination_bus=dest,
                slot=slot,
                volume_db=vol_db,
                is_muted=not enabled,
                is_pre_fader=(stype == "pre"),
            )
        )
    return tuple(sends)


def _detect_template_sample_rate(root: Any) -> int:
    for audio_el in root.iter("Audio"):
        sr_s = audio_el.get("sampleRate")
        if sr_s:
            try:
                sr = int(float(sr_s))
                if sr > 0:
                    return sr
            except ValueError:
                pass
    return 48000


def _classify_channel_track_type(
    trk_el: Any | None,
    ch_el: Any,
    cid: str,
    master_cid: str | None,
) -> str | None:
    role = (ch_el.get("role") or "regular").strip().lower()
    if role == "vca":
        return None
    if cid == master_cid:
        return "Master"
    if trk_el is not None and _is_folder_elem(trk_el):
        return "RoutingFolder"
    if role in ("submix", "effect"):
        return "Aux"
    if role == "master":
        # Unreferenced secondary hardware output (e.g., Cubase AIO 5/6)
        return None if not (ch_el.get("destination") or "").strip() else "Aux"
    return "Audio" if trk_el is not None else "Aux"


def _unique_node_name(base: str, used_lc: set[str], suffix: str) -> str:
    name = base.strip() or suffix
    if name.lower() not in used_lc:
        used_lc.add(name.lower())
        return name
    cand = f"{name} ({suffix})"
    if cand.lower() not in used_lc:
        used_lc.add(cand.lower())
        return cand
    n = 2
    while f"{name}_{n}".lower() in used_lc:
        n += 1
    out = f"{name}_{n}"
    used_lc.add(out.lower())
    return out


def _build_mixer_track_from_xml(
    idx: int,
    trk_el: Any | None,
    ch_el: Any,
    parent_name: str,
    cid: str,
    master_cid: str | None,
    used_names: set[str],
    reader: Callable[[str], bytes | None],
    *,
    apply_trim_gain: bool,
) -> tuple[MixerTrack, Vst3TrimDonor | None] | None:
    ttype = _classify_channel_track_type(trk_el, ch_el, cid, master_cid)
    if ttype is None:
        return None
    raw_name = (trk_el.get("name") if trk_el is not None else None) or ch_el.get("name") or cid
    node_name = _unique_node_name(str(raw_name), used_names, "Bus" if trk_el is None else cid)
    ch_count_s = ch_el.get("audioChannels")
    try:
        ch_count = int(ch_count_s) if ch_count_s is not None else 2
    except ValueError:
        ch_count = 2
    fmt = "Mono" if ch_count == 1 else "Stereo"
    dest = (ch_el.get("destination") or "").strip()
    in_bus: str | None
    out_bus: str | None
    if ttype == "Master":
        in_bus, out_bus = None, cid
    else:
        in_bus = cid if ttype in ("RoutingFolder", "Aux") else None
        out_bus = dest or master_cid
    mute_el = ch_el.find("Mute")
    inserts, donor = _parse_channel_devices(ch_el, reader, track_format=fmt, cid=cid, idx=idx)
    track = MixerTrack(
        name=node_name,
        track_type=ttype,
        format=fmt,
        index=idx,
        input_bus=in_bus,
        input_is_bus=bool(in_bus),
        output_bus=out_bus,
        parent_folder=parent_name,
        volume_db=_parse_volume_db(ch_el.find("Volume")),
        pan_left=_parse_pan(ch_el.find("Pan")) if fmt == "Mono" else None,
        pan_right=None,
        is_muted=_xml_bool(mute_el.get("value"), False) if mute_el is not None else False,
        is_inactive=not _xml_bool(trk_el.get("loaded"), True) if trk_el is not None else False,
        vca_offset_db=0.0,
        inserts=inserts,
        sends=_parse_channel_sends(ch_el),
        slot_a_trim=donor.trim_info if (donor is not None and apply_trim_gain and donor.is_calibrated) else None,
    )
    return track, donor


def dawproject_xml_to_mixer_session(
    root: Any,
    read_member: Callable[[str], bytes | None] | None = None,
    *,
    apply_trim_gain: bool = False,
) -> DawProjectMixerMap:
    """Adapt a DAWproject ``project.xml`` element tree into a DAW-agnostic :class:`MixerSession`."""
    reader = read_member or (lambda _p: None)
    raw_entries = _collect_raw_channels(root)
    master_cid = _identify_master_channel_id(raw_entries)
    sample_rate = _detect_template_sample_rate(root)

    tracks: list[MixerTrack] = []
    donors: list[Vst3TrimDonor] = []
    ch_id_by_node: dict[str, str] = {}
    node_by_ch_id: dict[str, str] = {}
    used_names: set[str] = set()

    for idx, (trk_el, ch_el, parent_name, cid) in enumerate(raw_entries):
        built = _build_mixer_track_from_xml(
            idx, trk_el, ch_el, parent_name, cid, master_cid, used_names, reader,
            apply_trim_gain=apply_trim_gain,
        )
        if built is None:
            continue
        track, donor = built
        if donor is not None:
            donors.append(donor)
        tracks.append(track)
        ch_id_by_node[track.name] = cid
        node_by_ch_id[cid] = track.name

    donor_by_fmt: dict[str, Any] | None = None
    if apply_trim_gain and donors:
        donor_by_fmt = {}
        for fmt in ("Mono", "Stereo"):
            chosen = select_vst3_trim_donor(donors, fmt, require_calibrated=True)
            if chosen is not None:
                donor_by_fmt[fmt] = chosen.trim_info
        if not donor_by_fmt:
            donor_by_fmt = None

    mixer = MixerSession(
        sample_rate=sample_rate,
        tracks=tuple(tracks),
        donor_trim_by_format=donor_by_fmt,
    )
    return DawProjectMixerMap(
        mixer=mixer,
        channel_id_by_node=ch_id_by_node,
        node_by_channel_id=node_by_ch_id,
        master_channel_id=master_cid,
        donors=tuple(donors),
        sample_rate=sample_rate,
    )


def _format_output_chain(
    start_cid: str | None,
    ch_meta: Mapping[str, tuple[str, str | None]],
    master_cid: str | None,
) -> str:
    if not start_cid:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    cur: str | None = start_cid
    while cur and cur not in seen:
        seen.add(cur)
        name, nxt = ch_meta.get(cur, (cur, None))
        parts.append(name)
        if cur == master_cid:
            break
        cur = nxt or master_cid
    return " \u2192 ".join(parts)


def _inspect_topology_from_root(
    root: Any,
    mmap: DawProjectMixerMap,
    template_path: str | Path,
) -> DawProjectTopologySummary:
    raw_entries = _collect_raw_channels(root)
    ch_meta: dict[str, tuple[str, str | None]] = {}
    for trk_el, ch_el, _, cid in raw_entries:
        cname = str(
            (trk_el.get("name") if trk_el is not None else None) or ch_el.get("name") or cid
        )
        dest = (ch_el.get("destination") or "").strip() or None
        ch_meta[cid] = (cname, dest)

    by_id: dict[str, DawProjectFolderMeta] = {}
    by_name: dict[str, DawProjectFolderMeta] = {}
    structure = root.find("Structure")

    def _walk(elem: Any, ancestor_cid: str | None) -> None:
        for trk in elem.findall("Track"):
            if not _is_folder_elem(trk):
                continue
            fid = str(trk.get("id") or "")
            fname = str(trk.get("name") or "(unnamed)")
            ch_el = trk.find("Channel")
            own_cid = (ch_el.get("id") or "").strip() if ch_el is not None else None
            if ch_el is not None and not own_cid:
                for t_el, c_el, _, syn_cid in raw_entries:
                    if t_el is trk and c_el is ch_el:
                        own_cid = syn_cid
                        break
            resolved_cid = own_cid or ancestor_cid or mmap.master_channel_id
            bus_name = ch_meta[own_cid][0] if (own_cid and own_cid in ch_meta) else None
            resolved_name = ch_meta[resolved_cid][0] if (resolved_cid and resolved_cid in ch_meta) else bus_name
            chain_start = ch_meta[own_cid][1] if (own_cid and own_cid in ch_meta) else resolved_cid
            if own_cid and chain_start is None:
                chain_start = mmap.master_channel_id
            meta = DawProjectFolderMeta(
                folder_id=fid,
                folder_name=fname,
                folder_type="routing" if ch_el is not None else "basic",
                bus_channel_id=own_cid,
                bus_channel_name=bus_name,
                resolved_routing_channel_id=resolved_cid,
                output_chain=_format_output_chain(chain_start, ch_meta, mmap.master_channel_id),
                resolved_routing_channel_name=resolved_name,
            )
            if fid:
                by_id[fid] = meta
            by_name.setdefault(fname.lower(), meta)
            _walk(trk, own_cid or ancestor_cid)

    if structure is not None:
        _walk(structure, None)
    best_donor = select_vst3_trim_donor(mmap.donors, "Stereo", require_calibrated=False)
    cal_donor = select_vst3_trim_donor(mmap.donors, "Stereo", require_calibrated=True)
    return DawProjectTopologySummary(
        template_path=str(template_path),
        by_id=by_id,
        by_name=by_name,
        donor_plugin_name=best_donor.plugin_name if best_donor else None,
        has_calibrated_donor=cal_donor is not None,
        donor_min_db=cal_donor.trim_info.min_db if cal_donor is not None else None,
        donor_max_db=cal_donor.trim_info.max_db if cal_donor is not None else None,
    )


def inspect_dawproject_template_topology(template_path: str | Path) -> DawProjectTopologySummary:
    """Parse a ``.dawproject`` template ZIP offline and return per-folder routing metadata."""
    path = Path(template_path)
    with zipfile.ZipFile(path, "r") as zf:
        root = ET.fromstring(zf.read("project.xml"))
        names_lc = {m.replace("\\", "/").lower(): m for m in zf.namelist()}

        def _read_member(rel: str) -> bytes | None:
            actual = names_lc.get(rel.replace("\\", "/").lower())
            return zf.read(actual) if actual else None

        mmap = dawproject_xml_to_mixer_session(root, _read_member, apply_trim_gain=True)
    return _inspect_topology_from_root(root, mmap, path)


def _expand_param_bounds(vol_el: Any, val: float, *, fmt: str) -> None:
    for attr, cmp_gt in (("min", False), ("max", True)):
        raw = vol_el.get(attr)
        if raw is None:
            continue
        try:
            cur = float(raw)
        except ValueError:
            continue
        if (cmp_gt and val > cur) or (not cmp_gt and val < cur):
            vol_el.set(attr, fmt.format(val))


def patch_dawproject_channel_volumes_xml(
    root: Any,
    channel_volumes_db: Mapping[str, float],
    *,
    make_element: Callable[..., Any],
    next_xml_id: Callable[[], str],
) -> dict[str, float]:
    """Patch ``<Volume>`` on matching ``<Channel id="...">`` nodes in *root*."""
    if not channel_volumes_db:
        return {}
    by_cid: dict[str, Any] = {}
    for ch in root.iter("Channel"):
        cid = (ch.get("id") or "").strip()
        if cid:
            by_cid[cid] = ch
    patched: dict[str, float] = {}
    for cid, db_val in channel_volumes_db.items():
        if not math.isfinite(float(db_val)):
            continue
        ch_el = by_cid.get(cid)
        if ch_el is None:
            log.warning("Skipping DAWproject volume patch for unaddressed/synthetic channel %r", cid)
            continue
        vol_el = ch_el.find("Volume")
        if vol_el is None:
            vol_el = make_element("Volume", {"id": next_xml_id(), "unit": "linear", "min": "0.0", "max": "2.0"})
            ch_el.append(vol_el)
        unit = (vol_el.get("unit") or "linear").strip().lower()
        if unit == "decibel":
            val_db = round(float(db_val), 2)
            vol_el.set("value", f"{val_db:.2f}")
            _expand_param_bounds(vol_el, val_db, fmt="{:.2f}")
        else:
            lin = 0.0 if float(db_val) <= -144.0 else math.pow(10.0, float(db_val) / 20.0)
            vol_el.set("value", f"{lin:.6f}")
            _expand_param_bounds(vol_el, lin, fmt="{:.6f}")
        patched[cid] = round(float(db_val), 1)
    return patched


def _collect_submix_dependents(
    root: Any,
    subtree_ids: set[str],
    parent_map: dict[Any, Any],
) -> list[Any]:
    dep_nodes: list[Any] = []
    seen: set[Any] = set()
    for el in root.iter():
        if el.get("track") in subtree_ids and el not in seen:
            dep_nodes.append(el)
            seen.add(el)
        if _local_tag(getattr(el, "tag", "")) == "Target" and el.get("parameter") in subtree_ids:
            owner = parent_map.get(el)
            if owner is not None and owner not in seen:
                dep_nodes.append(owner)
                seen.add(owner)
    return dep_nodes


def _has_external_submix_refs(
    root: Any,
    ch_el: Any,
    dep_nodes: list[Any],
    all_removed_ids: set[str],
) -> bool:
    excluded = set(ch_el.iter())
    for dep in dep_nodes:
        excluded.update(dep.iter())
    for el in root.iter():
        if el in excluded:
            continue
        for attr in ("destination", "reference", "track", "parameter"):
            if el.get(attr) in all_removed_ids:
                return True
    return False


def _remove_xml_nodes_and_empty_scenes(nodes: list[Any], parent_map: dict[Any, Any]) -> None:
    scenes_to_check: list[Any] = []
    for node in nodes:
        parent = parent_map.get(node)
        if parent is None:
            continue
        try:
            parent.remove(node)
        except ValueError:
            continue
        if _local_tag(getattr(parent, "tag", "")) == "Scene":
            scenes_to_check.append(parent)
    for scene in scenes_to_check:
        if len(scene) == 0:
            sp = parent_map.get(scene)
            if sp is not None:
                try:
                    sp.remove(scene)
                except ValueError:
                    pass


def prune_orphaned_submix_channels_xml(
    root: Any,
    previously_fed_destinations: set[str],
) -> list[str]:
    """Remove standalone ``<Structure><Channel role='submix'>`` buses orphaned by empty folder cleanup.

    Protects effect returns, master channels, channels referenced externally via
    ``@destination`` / ``@reference`` / ``@track`` / ``@parameter``, and channels never fed
    by a removed folder/submix. Cleans dependent timeline automation nodes and emptied ``<Scene>``s.
    """
    structure = root.find("Structure")
    if structure is None or not previously_fed_destinations:
        return []
    removed_names: list[str] = []
    candidate_ids = set(previously_fed_destinations)
    changed = True
    while changed:
        changed = False
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for ch_el in list(structure.findall("Channel")):
            cid = (ch_el.get("id") or "").strip()
            role = (ch_el.get("role") or "").strip().lower()
            if not cid or role != "submix" or cid not in candidate_ids:
                continue
            subtree_ids = {el.get("id") for el in ch_el.iter() if el.get("id")}
            dep_nodes = _collect_submix_dependents(root, subtree_ids, parent_map)
            all_removed_ids = set(subtree_ids)
            for dep in dep_nodes:
                all_removed_ids.update(el.get("id") for el in dep.iter() if el.get("id"))
            if _has_external_submix_refs(root, ch_el, dep_nodes, all_removed_ids):
                continue
            dest = (ch_el.get("destination") or "").strip()
            if dest:
                candidate_ids.add(dest)
            for snd in ch_el.findall(".//Send"):
                sdest = (snd.get("destination") or "").strip()
                if sdest:
                    candidate_ids.add(sdest)
            name = ch_el.get("name") or cid
            structure.remove(ch_el)
            _remove_xml_nodes_and_empty_scenes(dep_nodes, parent_map)
            removed_names.append(str(name))
            changed = True
    return removed_names


def _load_preview_leaf(
    track_name: str,
    leaf_entries: Mapping[str, tuple[TrackContext, TrackStagingPlan]],
    excluded: dict[str, str],
) -> LeafAudioInput | None:
    pair = leaf_entries.get(track_name.lower())
    if pair is None:
        return None
    track, plan = pair
    same_src = track.audio_data is not None and (
        os.path.normcase(os.path.realpath(plan.filepath))
        == os.path.normcase(os.path.realpath(track.filepath))
    )
    if same_src and track.audio_data is not None:
        arr = np.asarray(track.audio_data, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            excluded[track.filename] = "Non-finite samples (NaN/Inf)"
            return None
        return LeafAudioInput(arr, track.samplerate, 0.0, plan.fader_db, plan.baked_gain_db, plan.trim_gain_db)
    try:
        samples, sr = sf.read(plan.filepath, dtype="float64", always_2d=False)
        arr = np.asarray(samples, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            excluded[track.filename] = "Non-finite samples (NaN/Inf)"
            return None
        return LeafAudioInput(arr, int(sr), 0.0, plan.fader_db, plan.baked_gain_db, plan.trim_gain_db)
    except Exception as exc:
        excluded[track.filename] = str(exc)
        return None


def _inject_virtual_preview_leaves(
    mmap: DawProjectMixerMap,
    topo: DawProjectTopologySummary,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    *,
    gain_mode: str,
    apply_trim_gain: bool,
    fader_ceiling_db: float,
    leaf_trim_min_db: float | None,
    leaf_trim_max_db: float | None,
    excluded: dict[str, str],
) -> tuple[MixerSession, dict[str, tuple[TrackContext, TrackStagingPlan]], dict[str, tuple[float, float, float, float]]]:
    if session.config.get("_use_processed") is True and session.prepare_state != "ready":
        raise ValueError("Prepared audio is out of date. Run Prepare again (or uncheck Use Processed).")
    out_map = {t.filename: t for t in session.output_tracks}
    folder_lookup = {str(f.get("id", "")): f for f in folders}
    used_lc = {t.name.lower() for t in mmap.mixer.tracks}
    extra_tracks: list[MixerTrack] = []
    leaf_entries: dict[str, tuple[TrackContext, TrackStagingPlan]] = {}
    leaf_gains: dict[str, tuple[float, float, float, float]] = {}

    for entry in session.transfer_manifest:
        fid = assignments.get(entry.entry_id)
        track = out_map.get(entry.output_filename) if fid else None
        if not fid or track is None:
            continue
        fdict = folder_lookup.get(str(fid), {})
        fmeta = topo.lookup(str(fid), str(fdict.get("name", "")))
        dest_cid = fmeta.resolved_routing_channel_id if fmeta else mmap.master_channel_id
        try:
            plan = resolve_dawproject_track_staging(
                session, track, gain_mode=gain_mode, apply_trim_gain=apply_trim_gain,
                fader_ceiling_db=fader_ceiling_db,
                leaf_trim_min_db=leaf_trim_min_db,
                leaf_trim_max_db=leaf_trim_max_db,
            )
        except (ValueError, OSError) as exc:
            excluded[track.filename] = str(exc)
            continue
        vname = _unique_node_name(os.path.splitext(entry.daw_track_name)[0], used_lc, "Track")
        fmt = "Stereo" if track.channels >= 2 else "Mono"
        extra_tracks.append(
            MixerTrack(
                name=vname,
                track_type="Audio",
                format=fmt,
                index=len(mmap.mixer.tracks) + len(extra_tracks),
                input_bus=None,
                input_is_bus=False,
                output_bus=dest_cid,
                parent_folder=fmeta.folder_name if fmeta else "",
                volume_db=plan.fader_db,
            )
        )
        leaf_entries[vname.lower()] = (track, plan)
        leaf_gains[vname] = (0.0, plan.fader_db, plan.baked_gain_db, plan.trim_gain_db)

    combined = MixerSession(
        sample_rate=mmap.mixer.sample_rate,
        tracks=mmap.mixer.tracks + tuple(extra_tracks),
        donor_trim_by_format=mmap.mixer.donor_trim_by_format,
    )
    return combined, leaf_entries, leaf_gains


def preview_dawproject_signal_flow(
    template_path: str | Path,
    session: SessionContext,
    assignments: Mapping[str, str],
    folders: list[dict[str, Any]],
    *,
    gain_mode: str = GAIN_MODE_TRIM_AND_FADER,
    insert_track_trim: bool = True,
    apply_trim_gain: bool = False,
    stage_bus_faders: bool = True,
    pan_law_db: float = -3.0,
    fader_ceiling_db: float = 6.0,
    bus_fader_overrides: Mapping[str, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SignalFlowGraph:
    """Compute an offline annotated DAWproject signal-flow preview directly from a ``.dawproject`` template."""
    path = Path(template_path)
    with zipfile.ZipFile(path, "r") as zf:
        root = ET.fromstring(zf.read("project.xml"))
        names_lc = {m.replace("\\", "/").lower(): m for m in zf.namelist()}

        def _read_member(rel: str) -> bytes | None:
            actual = names_lc.get(rel.replace("\\", "/").lower())
            return zf.read(actual) if actual else None

        mmap = dawproject_xml_to_mixer_session(root, _read_member, apply_trim_gain=apply_trim_gain)
    topo = _inspect_topology_from_root(root, mmap, path)
    cal_donor = select_vst3_trim_donor(mmap.donors, "Stereo", require_calibrated=True)
    effective_leaf_apply = bool(apply_trim_gain and insert_track_trim and cal_donor is not None)
    lt_min = cal_donor.trim_info.min_db if cal_donor is not None else None
    lt_max = cal_donor.trim_info.max_db if cal_donor is not None else None

    excluded: dict[str, str] = {}
    mixer, leaf_entries, leaf_gains = _inject_virtual_preview_leaves(
        mmap, topo, session, assignments, folders,
        gain_mode=gain_mode, apply_trim_gain=effective_leaf_apply,
        fader_ceiling_db=fader_ceiling_db,
        leaf_trim_min_db=lt_min, leaf_trim_max_db=lt_max,
        excluded=excluded,
    )
    limits = dawproject_host_limits(
        fader_ceiling_db, leaf_trim_min_db=lt_min, leaf_trim_max_db=lt_max,
    )
    attenuate_only = gain_mode != GAIN_MODE_TRIM_UNITY
    allow_convergence = (
        apply_trim_gain
        and has_calibrated_vst3_trim_donor(mmap.donors)
        and gain_mode == GAIN_MODE_TRIM_AND_FADER
        and stage_bus_faders
    )
    graph = build_signal_flow(
        mixer,
        None,
        dict(session.config),
        host_limits=limits,
        pan_law_db=pan_law_db,
        stage_bus_faders=stage_bus_faders,
        attenuate_only=attenuate_only,
        strict_slot_a=False,
        allow_leaf_convergence=allow_convergence,
        leaf_loader=lambda name: _load_preview_leaf(name, leaf_entries, excluded),
        leaf_gains=leaf_gains,
        bus_fader_overrides=bus_fader_overrides,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    graph.excluded_tracks = excluded
    return graph
