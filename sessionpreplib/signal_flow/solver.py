"""DAW-agnostic signal-flow DAG reconstruction, exact time-domain summing, and recursive bus staging."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import logging
import math
from typing import Any

import numpy as np

from ..audio import NOISE_FLOOR_PEAK_DB
from ..detectors.silence import SilenceDetector
from .analysis import validate_signal_flow_balance
from .math import (
    accumulate_into,
    adapt_for_target,
    clamp_baked_gain,
    clamp_clip_gain,
    clamp_fader_db,
    clamp_leaf_trim_gain,
    compute_bus_target_fader,
    compute_unconstrained_bus_gain,
    db_to_lin,
    empty_like_format,
    has_audible_energy,
    measure_buffer,
    normalize_track_buffer,
    pan_stereo,
    resample_if_needed,
    round_opt,
)
from .models import (
    AUDIO_TRACK_TYPES,
    MIN_SIGNAL_DB,
    PROTOOLS_GAIN_LIMITS,
    HostGainLimits,
    LeafAudioInput,
    MixerInsert,
    MixerSession,
    MixerTrack,
    SignalFlowEdge,
    SignalFlowGraph,
    SignalFlowNode,
    TrimPluginInfo,
)
from .trim_codecs import norm_zero, select_preferred_trim_donor

log = logging.getLogger(__name__)

_CONVERGENCE_TOL_DB = 0.15
_MAX_CONVERGENCE_PASSES = 4


def _build_routing_edges(
    mixer: MixerSession,
    *,
    include_sends: bool = False,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> tuple[dict[str, MixerTrack], list[SignalFlowEdge]]:
    audio_tracks = {t.name: t for t in mixer.tracks if t.track_type in AUDIO_TRACK_TYPES}
    receivers_by_bus: dict[str, list[MixerTrack]] = {}
    masters_by_output: dict[str, list[MixerTrack]] = {}
    for trk in audio_tracks.values():
        if trk.track_type == "Master" and trk.output_bus:
            masters_by_output.setdefault(trk.output_bus.lower(), []).append(trk)
        elif trk.input_bus and trk.input_is_bus:
            receivers_by_bus.setdefault(trk.input_bus.lower(), []).append(trk)

    edges: list[SignalFlowEdge] = []
    for trk in audio_tracks.values():
        if trk.track_type == "Master":
            continue
        if trk.output_bus:
            bus_key = trk.output_bus.lower()
            for dst in (*receivers_by_bus.get(bus_key, ()), *masters_by_output.get(bus_key, ())):
                if dst.name != trk.name:
                    edges.append(SignalFlowEdge(trk.name, dst.name, trk.output_bus, kind="main"))
        if include_sends:
            for snd in trk.sends:
                if snd.is_muted or (snd.volume_db is not None and snd.volume_db <= limits.parallel_inf_db):
                    continue
                for dst in receivers_by_bus.get(snd.destination_bus.lower(), ()):
                    if dst.name != trk.name:
                        edges.append(
                            SignalFlowEdge(
                                trk.name, dst.name, snd.destination_bus,
                                kind="send", send_slot=snd.slot, send_volume_db=snd.volume_db,
                                is_pre_fader=snd.is_pre_fader,
                            )
                        )
    return audio_tracks, edges


def _topological_sort(track_names: list[str], edges: list[SignalFlowEdge]) -> tuple[list[str], list[list[str]]]:
    in_deg = {name: 0 for name in track_names}
    adj: dict[str, list[str]] = {name: [] for name in track_names}
    for edge in edges:
        if edge.source in in_deg and edge.target in in_deg:
            adj[edge.source].append(edge.target)
            in_deg[edge.target] += 1
    queue = deque(name for name in track_names if in_deg[name] == 0)
    order: list[str] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in adj[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    remaining = [name for name in track_names if in_deg[name] > 0]
    return order, ([remaining] if remaining else [])


def _is_active_stageable_bus(trk: MixerTrack, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> bool:
    return (
        trk.track_type in ("RoutingFolder", "Aux")
        and not trk.is_inactive
        and not trk.is_muted
        and (trk.volume_db is None or trk.volume_db > limits.parallel_inf_db)
    )


def _has_usable_slot_a_trim(trk: MixerTrack, limits: HostGainLimits = PROTOOLS_GAIN_LIMITS) -> bool:
    return (
        _is_active_stageable_bus(trk, limits)
        and trk.slot_a_trim is not None
        and not trk.slot_a_trim.is_bypassed
    )


def _has_active_unbypassed_inserts(trk: MixerTrack) -> bool:
    return any(ins.is_active and not ins.is_bypassed for ins in trk.inserts)


def _has_empty_slot_a(trk: MixerTrack) -> bool:
    return not any(ins.slot_index == 0 for ins in trk.inserts)


def _bfs_active_nodes(
    start: str,
    main_adj: Mapping[str, list[str]],
    audio_tracks: Mapping[str, MixerTrack],
    stop_set: set[str] | None = None,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> set[str]:
    visited: set[str] = {start}
    hits: set[str] = set()
    queue: deque[str] = deque(main_adj[start])
    while queue:
        nxt = queue.popleft()
        if nxt in visited:
            continue
        visited.add(nxt)
        trk = audio_tracks[nxt]
        if trk.is_inactive or trk.is_muted:
            continue
        if trk.track_type in ("RoutingFolder", "Aux") and not _is_active_stageable_bus(trk, limits):
            continue
        hits.add(nxt)
        if stop_set is not None and nxt in stop_set:
            continue
        queue.extend(main_adj[nxt])
    return hits


def _elect_terminal_nodes(
    audio_tracks: Mapping[str, MixerTrack],
    main_adj: Mapping[str, list[str]],
    main_in_deg: Mapping[str, int],
    trim_nodes: set[str],
    raw_downstream_trim: set[str],
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
    *,
    has_trim_capability: bool = False,
) -> set[str]:
    conv_nodes = {
        name for name, trk in audio_tracks.items()
        if _is_active_stageable_bus(trk, limits) and main_in_deg[name] >= 2
    }
    terminal: set[str] = set()
    for name in conv_nodes:
        if _bfs_active_nodes(name, main_adj, audio_tracks, limits=limits) & (conv_nodes - {name}):
            continue
        if name in trim_nodes or name not in raw_downstream_trim:
            terminal.add(name)
        else:
            terminal.update(_bfs_active_nodes(name, main_adj, audio_tracks, trim_nodes, limits=limits) & trim_nodes)
    if terminal or not (trim_nodes or has_trim_capability):
        return terminal
    eligible = {
        name for name, trk in audio_tracks.items()
        if _is_active_stageable_bus(trk, limits) and main_in_deg[name] >= 1
        and (name in trim_nodes or _has_empty_slot_a(trk))
    }
    sinks = {
        name for name in eligible
        if not any(nxt in eligible for nxt in main_adj[name])
    }
    return {
        name for name in sinks
        if not any(name in _bfs_active_nodes(other, main_adj, audio_tracks, limits=limits) for other in sinks - {name})
    }


def _analyze_trim_topology(
    audio_tracks: dict[str, MixerTrack],
    edges: list[SignalFlowEdge],
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
    *,
    has_synthetic_donors: bool = False,
) -> tuple[set[str], set[str], set[str], bool]:
    """Return ``(has_downstream_trim, terminal_nodes, post_terminal_nodes, has_trim_capability)``."""
    main_in_deg: dict[str, int] = {name: 0 for name in audio_tracks}
    main_adj: dict[str, list[str]] = {name: [] for name in audio_tracks}
    for edge in edges:
        if edge.kind == "main" and edge.source in main_adj and edge.target in main_in_deg:
            src_trk = audio_tracks[edge.source]
            if src_trk.is_inactive or src_trk.is_muted:
                continue
            if src_trk.track_type in ("RoutingFolder", "Aux") and not _is_active_stageable_bus(src_trk, limits):
                continue
            main_adj[edge.source].append(edge.target)
            main_in_deg[edge.target] += 1

    trim_nodes = {name for name, trk in audio_tracks.items() if _has_usable_slot_a_trim(trk, limits)}
    has_trim_capability = bool(trim_nodes) or has_synthetic_donors
    raw_downstream = {
        name for name in audio_tracks
        if bool(_bfs_active_nodes(name, main_adj, audio_tracks, limits=limits) & (trim_nodes - {name}))
    }
    terminal_nodes = _elect_terminal_nodes(
        audio_tracks, main_adj, main_in_deg, trim_nodes, raw_downstream, limits,
        has_trim_capability=has_trim_capability,
    )
    post_terminal_nodes = {
        nxt for term in terminal_nodes
        for nxt in _bfs_active_nodes(term, main_adj, audio_tracks, limits=limits)
        if nxt not in terminal_nodes
    }
    effective_trim = set(trim_nodes)
    if has_trim_capability:
        for name, trk in audio_tracks.items():
            if not _is_active_stageable_bus(trk, limits) or name in post_terminal_nodes:
                continue
            if trk.slot_a_trim is not None and trk.slot_a_trim.is_bypassed:
                continue
            if name in terminal_nodes or (trk.track_type == "Aux" and _has_active_unbypassed_inserts(trk)):
                effective_trim.add(name)
    has_downstream_trim = {
        name for name in audio_tracks
        if bool(_bfs_active_nodes(name, main_adj, audio_tracks, limits=limits) & (effective_trim - {name}))
    }
    return has_downstream_trim, terminal_nodes, post_terminal_nodes, has_trim_capability


def _slot_a_blocked_message(trk: MixerTrack, slot_a_ins: MixerInsert) -> str:
    plug = slot_a_ins.plugin_name or slot_a_ins.plugin_identifier or "unknown plugin"
    if slot_a_ins.plugin_name in ("TrackControl", "Trim", "Protility"):
        return (
            f"Cannot apply Slot A gain staging on bus {trk.name!r}: Insert Slot A plugin {plug!r} "
            f"could not be validated (unsupported channel format, multi-mono, or non-default "
            f"switches such as +12 dB/Mute/Phase). Reset it to default or replace it."
        )
    return (
        f"Cannot apply Slot A gain staging on bus {trk.name!r}: Insert Slot A is occupied by "
        f"non-trim plugin {plug!r}. Move it to Slot B (or later) or disable bus staging."
    )


def _bus_requires_slot_a_trim(
    trk: MixerTrack,
    *,
    has_template_trims: bool,
    is_terminal: bool,
    is_post_terminal: bool,
) -> bool:
    if not has_template_trims or is_post_terminal or trk.track_type not in ("RoutingFolder", "Aux"):
        return False
    if trk.slot_a_trim is not None and trk.slot_a_trim.is_bypassed:
        return False
    return is_terminal or (trk.track_type == "Aux" and _has_active_unbypassed_inserts(trk))


def _resolve_bus_slot_a_trim(
    trk: MixerTrack,
    node: SignalFlowNode,
    *,
    needed_trim_db: float,
    has_template_trims: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    donor_trim_by_format: Mapping[str, TrimPluginInfo],
    strict_slot_a: bool = True,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> TrimPluginInfo | None:
    """Return existing or auto-inserted Slot A Trim spec for *trk*, raising or recording on blocked Slot A."""
    if is_post_terminal:
        return None
    if _has_usable_slot_a_trim(trk, limits) and trk.slot_a_trim is not None:
        return trk.slot_a_trim
    if not _bus_requires_slot_a_trim(
        trk, has_template_trims=has_template_trims, is_terminal=is_terminal, is_post_terminal=is_post_terminal,
    ):
        return None
    if abs(needed_trim_db) < 0.05:
        return None
    slot_a_ins = next((ins for ins in trk.inserts if ins.slot_index == 0), None)
    if slot_a_ins is not None:
        msg = _slot_a_blocked_message(trk, slot_a_ins)
        node.slot_a_blocked_error = msg
        if strict_slot_a:
            raise ValueError(msg)
        return None
    donor_trim = donor_trim_by_format.get(trk.format)
    if donor_trim is None:
        msg = (
            f"Cannot auto-insert Slot A Trim on {trk.format} bus {trk.name!r}: "
            f"no matching {trk.format} Slot A Trim donor exists in this template."
        )
        node.slot_a_blocked_error = msg
        if strict_slot_a:
            raise ValueError(msg)
        return None
    node.slot_a_trim_id = donor_trim.plugin_id
    node.slot_a_trim_plugin = donor_trim.plugin_name
    node.slot_a_trim_auto_insert = True
    node.slot_a_trim_min_db = donor_trim.min_db
    node.slot_a_trim_max_db = donor_trim.max_db
    node.slot_a_trim_precision = max(1, int(donor_trim.precision_decimals))
    return donor_trim


def _quantize_slot_a_trim(raw_db: float, trim_spec: TrimPluginInfo) -> float:
    raw = float(raw_db)
    if not math.isfinite(raw):
        return 0.0
    clamped = max(trim_spec.min_db, min(trim_spec.max_db, raw))
    if trim_spec.quantize_db is not None:
        q = float(trim_spec.quantize_db(clamped))
        if not math.isfinite(q):
            q = clamped
        return norm_zero(max(trim_spec.min_db, min(trim_spec.max_db, q)))
    return norm_zero(round(clamped, max(1, int(trim_spec.precision_decimals))))


def _should_stage_fader_only_bus(
    node: SignalFlowNode,
    active_feeders: int,
    *,
    has_downstream_trim: bool,
    is_terminal: bool,
) -> bool:
    if has_downstream_trim:
        return False
    if is_terminal and active_feeders >= 1 and node.track_type in ("RoutingFolder", "Aux"):
        return True
    if node.track_type == "RoutingFolder":
        return active_feeders >= 1
    if node.track_type == "Aux":
        return active_feeders >= 2
    return False


def _apply_fader_override(
    auto_fader_db: float,
    bus_override_db: float | None,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> tuple[float, bool]:
    if bus_override_db is not None and math.isfinite(bus_override_db):
        ovr = norm_zero(round(max(limits.fader_min_db, min(limits.fader_max_db, float(bus_override_db))), 1))
        return ovr, abs(ovr - auto_fader_db) >= 0.05
    return auto_fader_db, False


def _apply_trim_staging(
    node: SignalFlowNode,
    trim_spec: TrimPluginInfo,
    raw: float,
    base_fader: float,
    *,
    attenuate_only: bool,
    is_terminal: bool,
    bus_override_db: float | None,
    carry_db: float = 0.0,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> float:
    active_tpl = node.template_trim_db if node.slot_a_trim_active else 0.0
    node.staged = True
    node.trim_clamped_excess_db = 0.0
    out_carry = carry_db
    if attenuate_only and not is_terminal:
        target_trim = _quantize_slot_a_trim(raw, trim_spec)
        fader_floor = min(base_fader, limits.leaf_shift_floor_db)
        unc_fader = base_fader + active_tpl - target_trim - carry_db
        auto_fader = norm_zero(round(max(fader_floor, min(limits.fader_max_db, unc_fader)), 1))
        if unc_fader > limits.fader_max_db or unc_fader < fader_floor:
            auto_trim = _quantize_slot_a_trim(base_fader + active_tpl - auto_fader - carry_db, trim_spec)
        else:
            auto_trim = target_trim
        if unc_fader > limits.fader_max_db or raw < trim_spec.min_db:
            node.trim_clamped_excess_db = round(max(0.0, auto_trim - raw), 2)
        node.staging_strategy = "trim_and_fader"
        out_carry = round(carry_db + (auto_trim - active_tpl) + (auto_fader - base_fader), 4)
    elif attenuate_only and is_terminal:
        auto_trim = _quantize_slot_a_trim(min(active_tpl, raw), trim_spec)
        auto_fader = base_fader
        node.staging_strategy = "trim_only"
        out_carry = 0.0
    else:
        auto_trim = _quantize_slot_a_trim(raw, trim_spec)
        auto_fader = base_fader
        node.staging_strategy = "trim_only"
        out_carry = 0.0
    node.auto_trim_db = auto_trim
    node.recommended_trim_db = auto_trim
    node.auto_fader_db = auto_fader
    node.recommended_fader_db, node.is_overridden = _apply_fader_override(auto_fader, bus_override_db, limits)
    node.net_gain_delta_db = norm_zero(
        round((node.recommended_trim_db - active_tpl) + (node.recommended_fader_db - base_fader), 2)
    )
    return 0.0 if node.is_overridden else out_carry


def _compute_needed_trim_db(
    raw: float,
    base_fader: float,
    active_tpl: float,
    *,
    attenuate_only: bool,
    is_terminal: bool,
    limits: HostGainLimits,
) -> float:
    if attenuate_only and is_terminal:
        return norm_zero(round(min(active_tpl, raw) - active_tpl, 1))
    if attenuate_only:
        fader_floor = min(base_fader, limits.leaf_shift_floor_db)
        bounded = max(base_fader + active_tpl - limits.fader_max_db, min(base_fader + active_tpl - fader_floor, raw))
        return norm_zero(round(bounded - active_tpl, 1))
    return norm_zero(round(raw - active_tpl, 1))


def _plan_bus_node_staging(
    trk: MixerTrack,
    node: SignalFlowNode,
    pre_in_pk: float | None,
    pre_in_rms: float | None,
    *,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    max_gain_db: float,
    has_downstream_trim: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, TrimPluginInfo],
    target_peak_db: float,
    target_rms_db: float,
    bus_override_db: float | None,
    strict_slot_a: bool = True,
    carry_db: float = 0.0,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> float:
    if (
        not stage_bus_faders
        or is_post_terminal
        or node.is_muted
        or node.is_inactive
        or node.is_parallel_parked
        or node.non_finite_error
        or pre_in_pk is None
        or pre_in_pk <= NOISE_FLOOR_PEAK_DB
    ):
        return carry_db
    active_feeders = len(node.upstream_tracks)
    raw = compute_unconstrained_bus_gain(
        pre_in_pk, pre_in_rms, target_peak_db, target_rms_db, max_gain_db=max_gain_db,
    )
    base_fader = norm_zero(round(node.template_fader_db, 1))
    active_tpl = node.template_trim_db if node.slot_a_trim_active else 0.0
    needed_trim = _compute_needed_trim_db(
        raw, base_fader, active_tpl, attenuate_only=attenuate_only, is_terminal=is_terminal, limits=limits,
    )
    trim_spec = (
        _resolve_bus_slot_a_trim(
            trk, node, needed_trim_db=needed_trim, has_template_trims=has_template_trims,
            is_terminal=is_terminal, is_post_terminal=is_post_terminal,
            donor_trim_by_format=donor_trim_by_format, strict_slot_a=strict_slot_a, limits=limits,
        )
        if active_feeders >= 1
        else None
    )
    if trim_spec is not None:
        return _apply_trim_staging(
            node, trim_spec, raw, base_fader,
            attenuate_only=attenuate_only, is_terminal=is_terminal,
            bus_override_db=bus_override_db, carry_db=carry_db, limits=limits,
        )
    if _bus_requires_slot_a_trim(
        trk, has_template_trims=has_template_trims, is_terminal=is_terminal, is_post_terminal=is_post_terminal,
    ):
        return carry_db
    if _should_stage_fader_only_bus(
        node, active_feeders, has_downstream_trim=has_downstream_trim, is_terminal=is_terminal,
    ):
        node.staged = True
        node.staging_strategy = "fader_only"
        auto_db = norm_zero(compute_bus_target_fader(
            pre_in_pk, pre_in_rms, target_peak_db, target_rms_db,
            attenuate_only=attenuate_only, max_fader_db=max_bus_fader_db,
            base_fader_db=base_fader, max_gain_db=max_gain_db, limits=limits,
        ))
        node.auto_fader_db = auto_db
        node.recommended_fader_db, node.is_overridden = _apply_fader_override(auto_db, bus_override_db, limits)
        node.net_gain_delta_db = norm_zero(round(node.recommended_fader_db - base_fader, 2))
    return carry_db


def _safe_finite_db(val: float | None, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


def _make_initial_node(
    trk: MixerTrack,
    leaf_meta: tuple[float, ...] | None,
    *,
    is_terminal: bool = False,
    is_post_terminal: bool = False,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> SignalFlowNode:
    tpl_vol = _safe_finite_db(trk.volume_db, 0.0)
    eff_fader = clamp_fader_db(leaf_meta[1], limits) if (leaf_meta is not None and len(leaf_meta) >= 2) else round(tpl_vol, 1)
    eff_clip = round(clamp_clip_gain(leaf_meta[0], limits), 2) if (leaf_meta is not None and len(leaf_meta) >= 1) else 0.0
    eff_baked = round(clamp_baked_gain(leaf_meta[2]), 2) if (leaf_meta is not None and len(leaf_meta) >= 3) else 0.0
    eff_ltrim = round(clamp_leaf_trim_gain(leaf_meta[3], limits), 2) if (leaf_meta is not None and len(leaf_meta) >= 4) else 0.0
    active_ins = [ins.plugin_name or ins.plugin_identifier for ins in trk.inserts if ins.is_active and not ins.is_bypassed]
    trim = trk.slot_a_trim
    prec = max(1, int(trim.precision_decimals)) if trim else 1
    trim_gain = round(_safe_finite_db(trim.gain_db, 0.0), prec) if trim else 0.0
    vca_off = round(_safe_finite_db(trk.vca_offset_db, 0.0), 2)
    return SignalFlowNode(
        track_name=trk.name,
        track_type=trk.track_type,
        format=trk.format,
        input_bus=trk.input_bus,
        output_bus=trk.output_bus,
        parent_folder=trk.parent_folder or None,
        active_inserts=active_ins,
        is_muted=trk.is_muted,
        is_inactive=trk.is_inactive,
        is_parallel_parked=(eff_fader <= limits.parallel_inf_db),
        baked_gain_db=eff_baked,
        clip_gain_db=eff_clip,
        leaf_trim_db=eff_ltrim,
        slot_a_trim_id=trim.plugin_id if trim else None,
        slot_a_trim_plugin=trim.plugin_name if trim else None,
        slot_a_trim_active=bool(trim and trim.is_active and not trim.is_bypassed),
        slot_a_trim_bypassed=bool(trim and trim.is_bypassed),
        slot_a_trim_min_db=trim.min_db if trim else None,
        slot_a_trim_max_db=trim.max_db if trim else None,
        slot_a_trim_precision=prec,
        is_terminal=is_terminal,
        is_post_terminal=is_post_terminal,
        template_trim_db=trim_gain,
        recommended_trim_db=trim_gain,
        template_fader_db=eff_fader,
        vca_offset_db=vca_off,
        recommended_fader_db=eff_fader,
    )


def _evaluate_leaf_node(
    trk: MixerTrack,
    node: SignalFlowNode,
    leaf: LeafAudioInput,
    session_sr: int,
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    target_peak_db: float,
    target_rms_db: float,
    *,
    leaf_fader_shift_db: float = 0.0,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    is_mono = trk.format == "Mono"
    empty = empty_like_format(is_mono)
    norm = normalize_track_buffer(leaf.samples, is_mono)
    if norm is None:
        log.warning("Signal-flow DAG skipped non-finite audio on track %r", trk.name)
        node.classification = "Error"
        node.non_finite_error = "Non-finite samples in leaf audio"
        return empty, empty
    base_buf = resample_if_needed(norm, leaf.samplerate, session_sr)
    eff_clip = clamp_clip_gain(leaf.clip_gain_db, limits)
    eff_ltrim = clamp_leaf_trim_gain(leaf.trim_gain_db, limits)
    pre_buf = base_buf * db_to_lin(eff_clip + eff_ltrim, limits)
    pre_pk, pre_rms, cls_label, gov, err = measure_buffer(
        pre_buf, session_sr, silence_det, cfg, target_peak_db, target_rms_db,
    )
    node.pre_fader_peak_db, node.pre_fader_rms_db = round_opt(pre_pk), round_opt(pre_rms)
    node.classification, node.governing, node.is_transient = cls_label, gov, (gov == "Peak")
    node.non_finite_error = err
    eff_fader = clamp_fader_db(leaf.fader_db + leaf_fader_shift_db, limits)
    node.baked_gain_db = round(clamp_baked_gain(leaf.baked_gain_db), 2)
    node.clip_gain_db = round(eff_clip, 2)
    node.leaf_trim_db = round(eff_ltrim, 2)
    node.recommended_fader_db = eff_fader
    node.net_gain_delta_db = norm_zero(
        round(node.baked_gain_db + node.clip_gain_db + node.leaf_trim_db + eff_fader, 2)
    )
    if node.is_muted or node.is_inactive or err:
        return empty, empty
    total_db = eff_fader + node.vca_offset_db
    post_gain = db_to_lin(total_db, limits)
    if post_gain <= 0.0 or not has_audible_energy(pre_buf):
        return pre_buf, empty
    post_buf = pre_buf * post_gain
    if not has_audible_energy(post_buf):
        return pre_buf, empty
    node.post_fader_peak_db = round_opt(pre_pk + total_db) if pre_pk is not None else None
    node.post_fader_rms_db = round_opt(pre_rms + total_db) if pre_rms is not None else None
    return pre_buf, post_buf


def _apply_bus_trim_and_post_fader(
    trk: MixerTrack,
    node: SignalFlowNode,
    pre_buf: np.ndarray,
    pre_in_pk: float | None,
    pre_in_rms: float | None,
    session_sr: int,
    pan_law_db: float,
    target_peak_db: float,
    target_rms_db: float,
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    is_mono = trk.format == "Mono"
    empty = empty_like_format(is_mono)
    if node.staged and node.staging_strategy in ("trim_and_fader", "trim_only"):
        eff_trim_db = node.recommended_trim_db
    elif trk.slot_a_trim is not None and trk.slot_a_trim.is_active and not trk.slot_a_trim.is_bypassed:
        eff_trim_db = _safe_finite_db(trk.slot_a_trim.gain_db, 0.0)
    else:
        eff_trim_db = 0.0

    trim_lin = db_to_lin(eff_trim_db, limits)
    if trim_lin <= 0.0:
        post_trim_buf = empty
        node.pre_fader_peak_db, node.pre_fader_rms_db = None, None
    elif abs(eff_trim_db) < 0.005:
        post_trim_buf = pre_buf
        node.pre_fader_peak_db, node.pre_fader_rms_db = round_opt(pre_in_pk), round_opt(pre_in_rms)
    else:
        post_trim_buf = pre_buf * trim_lin
        node.pre_fader_peak_db = round_opt(pre_in_pk + eff_trim_db) if pre_in_pk is not None else None
        node.pre_fader_rms_db = round_opt(pre_in_rms + eff_trim_db) if pre_in_rms is not None else None

    if node.is_muted or node.is_inactive:
        return empty, empty
    total_db = node.recommended_fader_db + node.vca_offset_db
    post_gain = db_to_lin(total_db, limits)
    if post_gain <= 0.0 or not has_audible_energy(post_trim_buf):
        return post_trim_buf, empty
    post_buf = post_trim_buf * post_gain
    if not has_audible_energy(post_buf):
        return post_trim_buf, empty
    has_non_identity_stereo_pan = (not is_mono) and (
        (trk.pan_left is not None and trk.pan_left != -100)
        or (trk.pan_right is not None and trk.pan_right != 100)
    )
    if has_non_identity_stereo_pan:
        panned_post = pan_stereo(post_buf, trk.pan_left, trk.pan_right, pan_law_db)
        post_pk, post_rms, _, _, _ = measure_buffer(
            panned_post, session_sr, silence_det, cfg, target_peak_db, target_rms_db,
        )
        node.post_fader_peak_db, node.post_fader_rms_db = round_opt(post_pk), round_opt(post_rms)
    else:
        base_pk = pre_in_pk + eff_trim_db if pre_in_pk is not None else None
        base_rms = pre_in_rms + eff_trim_db if pre_in_rms is not None else None
        node.post_fader_peak_db = round_opt(base_pk + total_db) if base_pk is not None else None
        node.post_fader_rms_db = round_opt(base_rms + total_db) if base_rms is not None else None
    return post_trim_buf, post_buf


def _evaluate_bus_node(
    trk: MixerTrack,
    node: SignalFlowNode,
    pre_buf: np.ndarray | None,
    session_sr: int,
    pan_law_db: float,
    target_peak_db: float,
    target_rms_db: float,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    max_gain_db: float,
    has_downstream_trim: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, TrimPluginInfo],
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    *,
    bus_override_db: float | None = None,
    strict_slot_a: bool = True,
    carry_db: float = 0.0,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> tuple[np.ndarray, np.ndarray, float]:
    empty = empty_like_format(trk.format == "Mono")
    if pre_buf is None or pre_buf.shape[0] == 0:
        return empty, empty, carry_db
    pre_in_pk, pre_in_rms, cls_label, gov, err = measure_buffer(
        pre_buf, session_sr, silence_det, cfg, target_peak_db, target_rms_db,
    )
    node.pre_insert_peak_db, node.pre_insert_rms_db = round_opt(pre_in_pk), round_opt(pre_in_rms)
    node.classification, node.is_transient = cls_label, (gov == "Peak")
    node.non_finite_error = err
    if err:
        node.governing = None
        return empty, empty, carry_db

    out_carry = _plan_bus_node_staging(
        trk, node, pre_in_pk, pre_in_rms,
        stage_bus_faders=stage_bus_faders, attenuate_only=attenuate_only,
        max_bus_fader_db=max_bus_fader_db, max_gain_db=max_gain_db,
        has_downstream_trim=has_downstream_trim, is_terminal=is_terminal,
        is_post_terminal=is_post_terminal, has_template_trims=has_template_trims,
        donor_trim_by_format=donor_trim_by_format, target_peak_db=target_peak_db,
        target_rms_db=target_rms_db, bus_override_db=bus_override_db,
        strict_slot_a=strict_slot_a, carry_db=carry_db, limits=limits,
    )
    node.governing = gov if node.staged else None
    post_trim_buf, post_buf = _apply_bus_trim_and_post_fader(
        trk, node, pre_buf, pre_in_pk, pre_in_rms, session_sr, pan_law_db,
        target_peak_db, target_rms_db, silence_det, cfg, limits=limits,
    )
    return post_trim_buf, post_buf, out_carry


def _propagate_outgoing(
    source_name: str,
    src_trk: MixerTrack,
    pre_buf: np.ndarray,
    post_buf: np.ndarray,
    out_edges: list[SignalFlowEdge],
    audio_tracks: dict[str, MixerTrack],
    nodes: dict[str, SignalFlowNode],
    accumulators: dict[str, np.ndarray | None],
    pan_law_db: float,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> None:
    for edge in out_edges:
        if edge.kind == "main":
            raw_src = post_buf
        else:
            src_buf = pre_buf if edge.is_pre_fader else post_buf
            gain = db_to_lin(float(edge.send_volume_db or 0.0), limits)
            if src_buf.shape[0] == 0 or gain <= 0.0:
                continue
            raw_src = src_buf * gain
        if not has_audible_energy(raw_src):
            continue
        contrib = adapt_for_target(raw_src, src_trk, audio_tracks[edge.target], pan_law_db)
        if not has_audible_energy(contrib):
            continue
        if source_name not in nodes[edge.target].upstream_tracks:
            nodes[edge.target].upstream_tracks.append(source_name)
        accumulators[edge.target] = accumulate_into(accumulators.get(edge.target), contrib)


def _run_dag_traversal(
    order: list[str],
    audio_tracks: dict[str, MixerTrack],
    nodes: dict[str, SignalFlowNode],
    outgoing: dict[str, list[SignalFlowEdge]],
    leaf_map: dict[str, LeafAudioInput],
    gains_map: dict[str, tuple[float, ...]],
    overrides_map: dict[str, float],
    session_sr: int,
    pan_law_db: float,
    target_peak_db: float,
    target_rms_db: float,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    max_gain_db: float,
    downstream_trim_nodes: set[str],
    terminal_nodes: set[str],
    post_terminal_nodes: set[str],
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, TrimPluginInfo],
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    leaf_loader: Callable[[str], LeafAudioInput | None] | None,
    progress_cb: Callable[[int, int, str], None] | None,
    cancel_cb: Callable[[], bool] | None,
    *,
    strict_slot_a: bool = True,
    leaf_fader_shift_db: float = 0.0,
    leaf_cache: dict[str, LeafAudioInput | None] | None = None,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> None:
    accumulators: dict[str, np.ndarray | None] = {name: None for name in audio_tracks}
    carry_out: dict[str, float] = {}
    total = len(order)
    for idx, name in enumerate(order, 1):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Signal-flow computation cancelled.")
        if progress_cb is not None:
            progress_cb(idx, total, name)
        trk, node = audio_tracks[name], nodes[name]
        lk = name.lower()
        if lk in leaf_map:
            leaf: LeafAudioInput | None = leaf_map[lk]
        elif leaf_cache is not None and lk in leaf_cache:
            leaf = leaf_cache[lk]
        elif leaf_loader is not None and lk in gains_map:
            leaf = leaf_loader(name)
            if leaf_cache is not None:
                leaf_cache[lk] = leaf
        else:
            leaf = None
        if leaf is not None:
            pre_buf, post_buf = _evaluate_leaf_node(
                trk, node, leaf, session_sr, silence_det, cfg, target_peak_db, target_rms_db,
                leaf_fader_shift_db=leaf_fader_shift_db, limits=limits,
            )
            carry_out[name] = 0.0
        else:
            acc = accumulators.pop(name, None)
            in_carries = [carry_out.get(u, 0.0) for u in node.upstream_tracks]
            in_carry = (sum(in_carries) / len(in_carries)) if in_carries else 0.0
            pre_buf, post_buf, out_carry = _evaluate_bus_node(
                trk, node, acc, session_sr, pan_law_db, target_peak_db, target_rms_db,
                stage_bus_faders, attenuate_only, max_bus_fader_db, max_gain_db,
                name in downstream_trim_nodes, name in terminal_nodes,
                name in post_terminal_nodes, has_template_trims, donor_trim_by_format,
                silence_det, cfg,
                bus_override_db=overrides_map.get(lk), strict_slot_a=strict_slot_a,
                carry_db=in_carry, limits=limits,
            )
            carry_out[name] = out_carry
        _propagate_outgoing(
            name, trk, pre_buf, post_buf, outgoing[name], audio_tracks, nodes, accumulators, pan_law_db, limits,
        )


def _compute_next_leaf_shift_step(
    nodes: Mapping[str, SignalFlowNode],
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> float:
    """Return additional negative leaf-fader shift step (dB > 0) to relieve clamped intermediate busses."""
    if any(n.slot_a_blocked_error for n in nodes.values()):
        return 0.0
    max_excess = max((n.trim_clamped_excess_db for n in nodes.values() if not n.is_overridden), default=0.0)
    if max_excess <= _CONVERGENCE_TOL_DB:
        return 0.0
    margins: list[float] = []
    for n in nodes.values():
        if (
            n.track_type not in ("Audio", "Instrument")
            or n.post_fader_peak_db is None
            or n.is_muted
            or n.is_inactive
            or n.is_parallel_parked
            or n.vca_offset_db <= limits.parallel_inf_db
        ):
            continue
        fader_margin = min(n.recommended_fader_db, n.recommended_fader_db + n.vca_offset_db) - limits.leaf_shift_floor_db
        level_margin = n.post_fader_peak_db - (MIN_SIGNAL_DB + 1.0)
        margins.append(min(fader_margin, level_margin))
    if not margins:
        return 0.0
    max_safe_step = max(0.0, round(min(margins), 1))
    step = round(min(max_excess, max_safe_step), 1)
    return step if step >= 0.1 else 0.0


def _parse_overrides_map(bus_fader_overrides: Mapping[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (bus_fader_overrides or {}).items():
        if not isinstance(k, str) or v is None or isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[k.lower()] = fv
    return out


def _init_signal_flow_nodes(
    audio_tracks: Mapping[str, MixerTrack],
    leaf_map: Mapping[str, LeafAudioInput],
    gains_map: Mapping[str, tuple[float, ...]],
    terminal_nodes: set[str],
    post_terminal_nodes: set[str],
    leaf_fader_shift_db: float,
    limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
) -> dict[str, SignalFlowNode]:
    nodes: dict[str, SignalFlowNode] = {}
    for name, trk in audio_tracks.items():
        lk = name.lower()
        if lk in leaf_map:
            inp = leaf_map[lk]
            meta: tuple[float, ...] | None = (
                inp.clip_gain_db, inp.fader_db, inp.baked_gain_db, inp.trim_gain_db,
            )
        else:
            meta = gains_map.get(lk)
        if meta is not None and abs(leaf_fader_shift_db) >= 0.05:
            clip = meta[0] if len(meta) >= 1 else 0.0
            fdr = meta[1] if len(meta) >= 2 else 0.0
            baked = meta[2] if len(meta) >= 3 else 0.0
            ltrim = meta[3] if len(meta) >= 4 else 0.0
            meta = (clip, fdr + leaf_fader_shift_db, baked, ltrim)
        nodes[name] = _make_initial_node(
            trk, meta, is_terminal=(name in terminal_nodes),
            is_post_terminal=(name in post_terminal_nodes), limits=limits,
        )
    return nodes


def _resolve_donor_trims(
    mixer: MixerSession,
    audio_tracks: Mapping[str, MixerTrack],
    has_template_trims: bool,
) -> dict[str, TrimPluginInfo]:
    if mixer.donor_trim_by_format is not None:
        return dict(mixer.donor_trim_by_format)
    donor_trim_by_format: dict[str, TrimPluginInfo] = {}
    if has_template_trims:
        for fmt in {t.format for t in audio_tracks.values()}:
            donor = select_preferred_trim_donor(mixer.tracks, fmt)
            if donor is not None and donor.slot_a_trim is not None:
                donor_trim_by_format[fmt] = donor.slot_a_trim
    return donor_trim_by_format


def _solve_with_convergence(
    order: list[str],
    audio_tracks: dict[str, MixerTrack],
    outgoing: dict[str, list[SignalFlowEdge]],
    leaf_map: dict[str, LeafAudioInput],
    gains_map: dict[str, tuple[float, ...]],
    overrides_map: dict[str, float],
    session_sr: int,
    eff_pan_law: float,
    target_peak_db: float,
    target_rms_db: float,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    max_gain_db: float,
    downstream_trim_nodes: set[str],
    terminal_nodes: set[str],
    post_terminal_nodes: set[str],
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, TrimPluginInfo],
    silence_det: SilenceDetector,
    cfg: Mapping[str, Any],
    leaf_loader: Callable[[str], LeafAudioInput | None] | None,
    progress_cb: Callable[[int, int, str], None] | None,
    cancel_cb: Callable[[], bool] | None,
    *,
    strict_slot_a: bool,
    allow_leaf_convergence: bool,
    limits: HostGainLimits,
) -> tuple[dict[str, SignalFlowNode], float, int]:
    can_converge = allow_leaf_convergence and stage_bus_faders and attenuate_only
    leaf_cache: dict[str, LeafAudioInput | None] | None = {} if can_converge else None
    leaf_shift_db, passes, nodes = 0.0, 0, {}
    while True:
        passes += 1
        nodes = _init_signal_flow_nodes(
            audio_tracks, leaf_map, gains_map, terminal_nodes, post_terminal_nodes, leaf_shift_db, limits,
        )
        _run_dag_traversal(
            order, audio_tracks, nodes, outgoing, leaf_map, gains_map, overrides_map,
            session_sr, eff_pan_law, target_peak_db, target_rms_db,
            stage_bus_faders, attenuate_only, max_bus_fader_db, max_gain_db,
            downstream_trim_nodes, terminal_nodes, post_terminal_nodes,
            has_template_trims, donor_trim_by_format, silence_det, cfg,
            leaf_loader, progress_cb if passes == 1 else None, cancel_cb,
            strict_slot_a=strict_slot_a, leaf_fader_shift_db=leaf_shift_db,
            leaf_cache=leaf_cache, limits=limits,
        )
        if not can_converge or passes >= _MAX_CONVERGENCE_PASSES:
            break
        step = _compute_next_leaf_shift_step(nodes, limits)
        if step < 0.1:
            break
        leaf_shift_db = round(leaf_shift_db - step, 1)
    return nodes, leaf_shift_db, passes


def build_signal_flow(
    mixer: MixerSession,
    leaf_audio: Mapping[str, LeafAudioInput] | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    host_limits: HostGainLimits = PROTOOLS_GAIN_LIMITS,
    pan_law_db: float | None = None,
    stage_bus_faders: bool = True,
    attenuate_only: bool = False,
    include_sends: bool = False,
    strict_slot_a: bool = True,
    allow_leaf_convergence: bool = False,
    leaf_loader: Callable[[str], LeafAudioInput | None] | None = None,
    leaf_gains: Mapping[str, tuple[float, ...]] | None = None,
    bus_fader_overrides: Mapping[str, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SignalFlowGraph:
    """Reconstruct a DAW signal-flow DAG, sum audio in float64, and stage bus trims/faders."""
    cfg = dict(config or {})
    target_peak_db, target_rms_db = float(cfg.get("target_peak", -6.0)), float(cfg.get("target_rms", -18.0))
    max_gain_db = float(cfg.get("max_gain_db", 24.0))
    eff_pan_law = (
        float(pan_law_db)
        if pan_law_db is not None
        else float(cfg.get("pan_law_db", cfg.get("protools_pan_law_db", -3.0)))
    )
    max_bus_fader_db = max(0.0, host_limits.fader_max_db - max(0.0, float(cfg.get("fader_headroom_db", 8.0))))
    dbfs_conv = str(cfg.get("dbfs_convention", "standard"))

    silence_det = SilenceDetector()
    silence_det.configure(cfg)

    leaf_map = {k.lower(): v for k, v in (leaf_audio or {}).items()}
    gains_map = {k.lower(): v for k, v in (leaf_gains or {}).items()}
    overrides_map = _parse_overrides_map(bus_fader_overrides)
    session_sr = (
        mixer.sample_rate
        if mixer.sample_rate > 0
        else next((inp.samplerate for inp in leaf_map.values() if inp.samplerate > 0), 48000)
    )

    audio_tracks, edges = _build_routing_edges(mixer, include_sends=include_sends, limits=host_limits)
    order, cycles = _topological_sort(list(audio_tracks), edges)
    outgoing: dict[str, list[SignalFlowEdge]] = {name: [] for name in audio_tracks}
    for edge in edges:
        outgoing[edge.source].append(edge)

    downstream_trim_nodes, terminal_nodes, post_terminal_nodes, has_template_trims = _analyze_trim_topology(
        audio_tracks, edges, limits=host_limits, has_synthetic_donors=bool(mixer.donor_trim_by_format),
    )
    donor_trim_by_format = _resolve_donor_trims(mixer, audio_tracks, has_template_trims)
    nodes, leaf_shift_db, passes = _solve_with_convergence(
        order, audio_tracks, outgoing, leaf_map, gains_map, overrides_map,
        session_sr, eff_pan_law, target_peak_db, target_rms_db,
        stage_bus_faders, attenuate_only, max_bus_fader_db, max_gain_db,
        downstream_trim_nodes, terminal_nodes, post_terminal_nodes,
        has_template_trims, donor_trim_by_format, silence_det, cfg,
        leaf_loader, progress_cb, cancel_cb,
        strict_slot_a=strict_slot_a, allow_leaf_convergence=allow_leaf_convergence, limits=host_limits,
    )
    blocked = {name: n.slot_a_blocked_error for name, n in nodes.items() if n.slot_a_blocked_error}
    val_rep = validate_signal_flow_balance(
        mixer, nodes, edges, attenuate_only=attenuate_only, passes=passes,
        leaf_fader_shift_db=leaf_shift_db, parallel_inf_db=host_limits.parallel_inf_db,
    )
    return SignalFlowGraph(
        session_sr, target_peak_db, target_rms_db, eff_pan_law, dbfs_conv,
        order, nodes, edges, cycles, {}, round(leaf_shift_db, 1), blocked, val_rep.to_dict(),
    )
