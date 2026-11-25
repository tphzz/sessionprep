"""Pro Tools template signal-flow DAG reconstruction, exact time-domain bus summing, and fader staging."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..detectors.audio_classifier import AudioClassifierDetector
from ..detectors.silence import SilenceDetector
from ..models import TrackContext
from .models import PtxInsert, PtxTrack, SlotATrimInfo
from .parser import PtxSession
from .plugin_trim import avid_trim_db_to_step, avid_trim_step_to_db, select_preferred_slot_a_trim_donor
from .signal_flow_analysis import validate_signal_flow_balance

log = logging.getLogger(__name__)

_AUDIO_TRACK_TYPES = frozenset({"Audio", "RoutingFolder", "Aux", "Instrument", "Master"})
_PARALLEL_INF_DB = -140.0
_MIN_SIGNAL_DB = -120.0
_MIN_SIGNAL_LIN = 10.0 ** (_MIN_SIGNAL_DB / 20.0)
_FADER_MIN_DB = -144.0
_FADER_MAX_DB = 12.0
_CLIP_GAIN_MIN_DB = -144.0
_CLIP_GAIN_MAX_DB = 36.0
_LEAF_SHIFT_FLOOR_DB = -136.0
_CONVERGENCE_TOL_DB = 0.15
_MAX_CONVERGENCE_PASSES = 4


@dataclass(frozen=True)
class LeafAudioInput:
    """Time-aligned audio buffer (starting at sample 0) plus planned leaf gains."""

    samples: np.ndarray
    samplerate: int
    clip_gain_db: float = 0.0
    fader_db: float = 0.0
    baked_gain_db: float = 0.0


@dataclass(frozen=True)
class SignalFlowEdge:
    """Directed audio routing edge between two tracks via a named bus."""

    source: str
    target: str
    bus_name: str
    kind: str = "main"  # "main" | "send"
    send_slot: str = ""
    send_volume_db: float | None = None
    is_pre_fader: bool = False


@dataclass
class SignalFlowNode:
    """One track/bus node in the annotated signal-flow DAG.

    All Peak/RMS values use SessionPrep's internal standard dBFS scale
    (full-scale sine peak = ``0.0 dBFS``, full-scale sine RMS = ``-3.01 dBFS``);
    display layers apply ``dbfs_offset(config)`` when ``dbfs_convention == "aes17"``.
    """

    track_name: str
    track_type: str
    format: str
    input_bus: str | None = None
    output_bus: str | None = None
    parent_folder: str | None = None
    upstream_tracks: list[str] = field(default_factory=list)
    active_inserts: list[str] = field(default_factory=list)
    is_muted: bool = False
    is_inactive: bool = False
    is_parallel_parked: bool = False
    baked_gain_db: float = 0.0
    clip_gain_db: float = 0.0
    slot_a_trim_id: str | None = None
    slot_a_trim_plugin: str | None = None
    slot_a_trim_active: bool = False
    slot_a_trim_bypassed: bool = False
    slot_a_trim_auto_insert: bool = False
    slot_a_trim_min_db: float | None = None
    slot_a_trim_max_db: float | None = None
    slot_a_blocked_error: str | None = None
    is_terminal: bool = False
    is_post_terminal: bool = False
    template_trim_db: float = 0.0
    auto_trim_db: float = 0.0
    recommended_trim_db: float = 0.0
    trim_clamped_excess_db: float = 0.0
    net_gain_delta_db: float = 0.0
    staging_strategy: str = "none"
    template_fader_db: float = 0.0
    vca_offset_db: float = 0.0
    pre_insert_peak_db: float | None = None
    pre_insert_rms_db: float | None = None
    pre_fader_peak_db: float | None = None
    pre_fader_rms_db: float | None = None
    classification: str = "Silent"
    is_transient: bool = False
    staged: bool = False
    auto_fader_db: float = 0.0
    is_overridden: bool = False
    recommended_fader_db: float = 0.0
    post_fader_peak_db: float | None = None
    post_fader_rms_db: float | None = None


@dataclass
class SignalFlowGraph:
    """Complete annotated signal-flow DAG with topological order and cycle diagnostics."""

    samplerate: int
    target_peak_db: float
    target_rms_db: float
    pan_law_db: float
    dbfs_convention: str = "standard"
    topological_order: list[str] = field(default_factory=list)
    nodes: dict[str, SignalFlowNode] = field(default_factory=dict)
    edges: list[SignalFlowEdge] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    excluded_tracks: dict[str, str] = field(default_factory=dict)
    leaf_fader_shift_db: float = 0.0
    blocked_nodes: dict[str, str] = field(default_factory=dict)
    balance_validation: dict[str, Any] = field(default_factory=dict)

    def recommended_bus_faders(self) -> dict[str, float]:
        """Return ``{track_name: fader_db}`` for staged bus/folder nodes whose fader changed."""
        return {
            name: round(node.recommended_fader_db, 1)
            for name, node in self.nodes.items()
            if node.staged and (node.is_overridden or abs(node.recommended_fader_db - node.template_fader_db) >= 0.05)
        }

    def recommended_slot_a_trims(self) -> dict[str, float]:
        """Return ``{track_name: trim_db}`` for staged bus nodes whose Slot A Trim should be patched."""
        out: dict[str, float] = {}
        for name, node in self.nodes.items():
            if not (node.staged and node.slot_a_trim_plugin and node.staging_strategy in ("trim_and_fader", "trim_only")):
                continue
            eps = 0.02 if node.slot_a_trim_id == "trim" else 0.05
            if abs(node.recommended_trim_db - node.template_trim_db) >= eps or (
                abs(node.recommended_trim_db) >= eps and not node.slot_a_trim_active
            ):
                out[name] = round(node.recommended_trim_db, 2 if node.slot_a_trim_id == "trim" else 1)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "samplerate": self.samplerate,
            "target_peak_db": self.target_peak_db,
            "target_rms_db": self.target_rms_db,
            "pan_law_db": self.pan_law_db,
            "dbfs_convention": self.dbfs_convention,
            "topological_order": list(self.topological_order),
            "nodes": {name: asdict(node) for name, node in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "cycles": [list(c) for c in self.cycles],
            "excluded_tracks": dict(self.excluded_tracks),
            "leaf_fader_shift_db": self.leaf_fader_shift_db,
            "blocked_nodes": dict(self.blocked_nodes),
            "balance_validation": dict(self.balance_validation),
            "recommended_bus_faders": self.recommended_bus_faders(),
            "recommended_slot_a_trims": self.recommended_slot_a_trims(),
        }


def pan_gains(pan: int | None, pan_law_db: float = -3.0) -> tuple[float, float]:
    """Compute ``(left_gain, right_gain)`` for a Pro Tools pan value ``-100..+100``.

    Uses a 0 dB balance law at ``pan_law_db >= 0.0`` and a continuous raised-trigonometric
    law for negative pan depths (exact center gain ``10**(pan_law_db/20)`` at ``pan=0``,
    constant power at ``-3.01 dB``, constant voltage at ``-6.02 dB``, zero opposite bleed
    at ``-100`` / ``+100``).
    """
    p = 0.0 if pan is None else max(-100.0, min(100.0, float(pan)))
    u = (p + 100.0) / 200.0
    if u <= 0.0:
        return 1.0, 0.0
    if u >= 1.0:
        return 0.0, 1.0
    depth = float(pan_law_db)
    if depth >= 0.0:
        return min(1.0, 2.0 * (1.0 - u)), min(1.0, 2.0 * u)
    exponent = -depth / (10.0 * math.log10(2.0))
    angle = 0.5 * math.pi * u
    return math.cos(angle) ** exponent, math.sin(angle) ** exponent


def _clamp_clip_gain(db: float) -> float:
    try:
        val = float(db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(_CLIP_GAIN_MIN_DB, min(_CLIP_GAIN_MAX_DB, val))


def _clamp_baked_gain(db: float) -> float:
    try:
        val = float(db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(-144.0, min(144.0, val))


def _clamp_fader_db(db: float) -> float:
    try:
        val = float(db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return round(max(_FADER_MIN_DB, min(_FADER_MAX_DB, val)), 1)


def _db_to_lin(db: float) -> float:
    return 0.0 if db <= _FADER_MIN_DB else float(10.0 ** (db / 20.0))


def _normalize_track_buffer(samples: np.ndarray, is_mono: bool) -> np.ndarray | None:
    """Validate finite samples and shape to 1-D mono ``(N,)`` or 2-D stereo ``(N, 2)``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64) if is_mono else np.zeros((0, 2), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    if is_mono:
        if arr.ndim == 1:
            return arr.copy()
        if arr.shape[1] == 1:
            return arr[:, 0].copy()
        return 0.5 * (arr[:, 0] + arr[:, 1])
    if arr.ndim == 1:
        return np.column_stack((arr, arr))
    if arr.shape[1] == 1:
        col = arr[:, 0]
        return np.column_stack((col, col))
    return arr[:, :2].copy()


def _pan_stereo(stereo: np.ndarray, pan_left: int | None, pan_right: int | None, pan_law_db: float) -> np.ndarray:
    pl = -100 if pan_left is None else pan_left
    pr = 100 if pan_right is None else pan_right
    if pl == -100 and pr == 100:
        return stereo
    ll, lr = pan_gains(pl, pan_law_db)
    rl, rr = pan_gains(pr, pan_law_db)
    out = np.empty_like(stereo)
    out[:, 0] = stereo[:, 0] * ll + stereo[:, 1] * rl
    out[:, 1] = stereo[:, 0] * lr + stereo[:, 1] * rr
    return out


def _adapt_for_target(
    buf: np.ndarray,
    src_trk: PtxTrack,
    dst_trk: PtxTrack,
    pan_law_db: float,
) -> np.ndarray:
    """Apply Pro Tools output panning and channel adaptation from *src_trk* into *dst_trk*."""
    src_mono = buf.ndim == 1
    dst_mono = dst_trk.format == "Mono"
    if src_mono and dst_mono:
        return buf
    if src_mono and not dst_mono:
        g_l, g_r = pan_gains(src_trk.pan_left, pan_law_db)
        out = np.empty((buf.shape[0], 2), dtype=np.float64)
        out[:, 0] = buf * g_l
        out[:, 1] = buf * g_r
        return out
    panned = _pan_stereo(buf, src_trk.pan_left, src_trk.pan_right, pan_law_db)
    if dst_mono:
        return 0.5 * (panned[:, 0] + panned[:, 1])
    return panned


def _resample_if_needed(buf: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr or buf.shape[0] == 0:
        return buf
    gcd = math.gcd(src_sr, dst_sr)
    return np.asarray(resample_poly(buf, dst_sr // gcd, src_sr // gcd, axis=0), dtype=np.float64)


def _resolve_vca_offset_db(ptx: PtxSession, track: PtxTrack) -> float:
    """Recursively accumulate active VCA attenuation dB controlling *track* (cycle-safe)."""
    total_db = 0.0
    visited_tracks: set[str] = {track.name.lower()}
    queue: deque[PtxTrack] = deque([track])
    seen_vcas: set[str] = set()
    while queue:
        cur = queue.popleft()
        for gname in cur.groups:
            grp = ptx.group_by_name(gname)
            if grp is None or not grp.vca_track_name:
                continue
            vca = ptx.track_by_name(grp.vca_track_name)
            if vca is None or vca.is_inactive or vca.name.lower() in seen_vcas:
                continue
            seen_vcas.add(vca.name.lower())
            if vca.is_muted or (vca.volume_db is not None and vca.volume_db <= _PARALLEL_INF_DB):
                return _FADER_MIN_DB
            total_db += float(vca.volume_db or 0.0)
            if vca.name.lower() not in visited_tracks:
                visited_tracks.add(vca.name.lower())
                queue.append(vca)
    return total_db


def _build_routing_edges(
    ptx: PtxSession,
    *,
    include_sends: bool = False,
) -> tuple[dict[str, PtxTrack], list[SignalFlowEdge]]:
    audio_tracks = {t.name: t for t in ptx.tracks if t.track_type in _AUDIO_TRACK_TYPES}
    receivers_by_bus: dict[str, list[PtxTrack]] = {}
    masters_by_output: dict[str, list[PtxTrack]] = {}
    for trk in audio_tracks.values():
        if trk.track_type == "Master" and trk.main_output:
            masters_by_output.setdefault(trk.main_output.name.lower(), []).append(trk)
        elif trk.input_path and trk.input_path.kind == "bus":
            receivers_by_bus.setdefault(trk.input_path.name.lower(), []).append(trk)

    edges: list[SignalFlowEdge] = []
    for trk in audio_tracks.values():
        if trk.track_type == "Master":
            continue
        if trk.main_output:
            bus_key = trk.main_output.name.lower()
            for dst in (*receivers_by_bus.get(bus_key, ()), *masters_by_output.get(bus_key, ())):
                if dst.name != trk.name:
                    edges.append(SignalFlowEdge(trk.name, dst.name, trk.main_output.name, kind="main"))
        if include_sends:
            for snd in trk.sends:
                if snd.is_muted or (snd.volume_db is not None and snd.volume_db <= _PARALLEL_INF_DB):
                    continue
                for dst in receivers_by_bus.get(snd.destination.name.lower(), ()):
                    if dst.name != trk.name:
                        edges.append(
                            SignalFlowEdge(
                                trk.name, dst.name, snd.destination.name,
                                kind="send", send_slot=snd.slot, send_volume_db=snd.volume_db, is_pre_fader=snd.is_pre_fader,
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


def _measure_buffer(
    buf: np.ndarray,
    samplerate: int,
    silence_det: SilenceDetector,
    classifier_det: AudioClassifierDetector,
) -> tuple[float | None, float | None, str, bool]:
    if buf.size == 0 or not np.all(np.isfinite(buf)):
        return None, None, "Silent", False
    peak = float(np.max(np.abs(buf)))
    if peak <= 0.0:
        return None, None, "Silent", False
    channels = 1 if buf.ndim == 1 else buf.shape[1]
    temp = TrackContext(
        filename="",
        filepath="",
        audio_data=buf,
        samplerate=samplerate,
        channels=channels,
        total_samples=buf.shape[0],
        bitdepth="64-bit Float",
        subtype="DOUBLE",
        duration_sec=buf.shape[0] / float(samplerate),
    )
    sil = silence_det.analyze(temp)
    temp.detector_results["silence"] = sil
    if sil.data.get("is_silent"):
        return round(20.0 * math.log10(peak), 2), None, "Silent", False
    cls = classifier_det.analyze(temp)
    peak_db = cls.data.get("peak_db")
    rms_db = cls.data.get("rms_anchor_db")
    label = str(cls.data.get("classification", "Sustained"))
    is_trans = bool(cls.data.get("is_transient", False))
    return (
        round(float(peak_db), 2) if peak_db is not None and math.isfinite(float(peak_db)) else None,
        round(float(rms_db), 2) if rms_db is not None and math.isfinite(float(rms_db)) else None,
        label,
        is_trans,
    )


def _accumulate_into(acc: np.ndarray | None, contrib: np.ndarray) -> np.ndarray:
    if contrib.shape[0] == 0:
        return acc if acc is not None else contrib
    if acc is None or acc.shape[0] == 0:
        return contrib.copy()
    if contrib.shape[0] == acc.shape[0]:
        acc += contrib
        return acc
    target_len = max(acc.shape[0], contrib.shape[0])
    if acc.shape[0] < target_len:
        shape = (target_len,) if acc.ndim == 1 else (target_len, acc.shape[1])
        grown = np.zeros(shape, dtype=np.float64)
        grown[: acc.shape[0]] = acc
        acc = grown
    acc[: contrib.shape[0]] += contrib
    return acc


def _compute_unconstrained_bus_gain(
    is_transient: bool,
    pre_peak_db: float | None,
    pre_rms_db: float | None,
    target_peak_db: float,
    target_rms_db: float,
) -> float:
    if pre_peak_db is None or not math.isfinite(pre_peak_db) or pre_peak_db <= _MIN_SIGNAL_DB:
        return 0.0
    gain_for_peak = target_peak_db - pre_peak_db
    if is_transient or pre_rms_db is None or not math.isfinite(pre_rms_db):
        raw = gain_for_peak
    else:
        raw = min(target_rms_db - pre_rms_db, gain_for_peak)
    return raw if math.isfinite(raw) else 0.0


def _compute_bus_target_fader(
    is_transient: bool,
    pre_peak_db: float | None,
    pre_rms_db: float | None,
    target_peak_db: float,
    target_rms_db: float,
    *,
    attenuate_only: bool,
    max_fader_db: float,
    base_fader_db: float = 0.0,
) -> float:
    raw = _compute_unconstrained_bus_gain(is_transient, pre_peak_db, pre_rms_db, target_peak_db, target_rms_db)
    upper = base_fader_db if attenuate_only else max(base_fader_db, min(_FADER_MAX_DB, max_fader_db))
    return round(max(_FADER_MIN_DB, min(upper, raw)), 1)


def _norm_zero(val: float) -> float:
    return 0.0 if abs(val) < 1e-9 else val


def _is_active_stageable_bus(trk: PtxTrack) -> bool:
    return (
        trk.track_type in ("RoutingFolder", "Aux")
        and not trk.is_inactive
        and not trk.is_muted
        and (trk.volume_db is None or trk.volume_db > _PARALLEL_INF_DB)
    )


def _has_usable_slot_a_trim(trk: PtxTrack) -> bool:
    return (
        _is_active_stageable_bus(trk)
        and trk.slot_a_trim is not None
        and not trk.slot_a_trim.is_bypassed
    )


def _has_active_unbypassed_inserts(trk: PtxTrack) -> bool:
    return any(ins.is_active and not ins.is_bypassed for ins in trk.inserts)


def _has_empty_slot_a(trk: PtxTrack) -> bool:
    return not any(ins.slot_index == 0 for ins in trk.inserts)


def _bfs_active_nodes(
    start: str,
    main_adj: Mapping[str, list[str]],
    audio_tracks: Mapping[str, PtxTrack],
    stop_set: set[str] | None = None,
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
        if trk.track_type in ("RoutingFolder", "Aux") and not _is_active_stageable_bus(trk):
            continue
        hits.add(nxt)
        if stop_set is not None and nxt in stop_set:
            continue
        queue.extend(main_adj[nxt])
    return hits


def _elect_terminal_nodes(
    audio_tracks: Mapping[str, PtxTrack],
    main_adj: Mapping[str, list[str]],
    main_in_deg: Mapping[str, int],
    trim_nodes: set[str],
    raw_downstream_trim: set[str],
) -> set[str]:
    conv_nodes = {
        name for name, trk in audio_tracks.items()
        if _is_active_stageable_bus(trk) and main_in_deg[name] >= 2
    }
    terminal: set[str] = set()
    for name in conv_nodes:
        if _bfs_active_nodes(name, main_adj, audio_tracks) & (conv_nodes - {name}):
            continue
        if name in trim_nodes or name not in raw_downstream_trim:
            terminal.add(name)
        else:
            terminal.update(_bfs_active_nodes(name, main_adj, audio_tracks, trim_nodes) & trim_nodes)
    if terminal or not trim_nodes:
        return terminal
    eligible = {
        name for name, trk in audio_tracks.items()
        if _is_active_stageable_bus(trk) and main_in_deg[name] >= 1
        and (name in trim_nodes or _has_empty_slot_a(trk))
    }
    sinks = {
        name for name in eligible
        if not any(nxt in eligible for nxt in main_adj[name])
    }
    return {
        name for name in sinks
        if not any(name in _bfs_active_nodes(other, main_adj, audio_tracks) for other in sinks - {name})
    }


def _analyze_trim_topology(
    audio_tracks: dict[str, PtxTrack],
    edges: list[SignalFlowEdge],
) -> tuple[set[str], set[str], set[str], bool]:
    """Return ``(has_downstream_trim, terminal_nodes, post_terminal_nodes, has_template_trims)``."""
    main_in_deg: dict[str, int] = {name: 0 for name in audio_tracks}
    main_adj: dict[str, list[str]] = {name: [] for name in audio_tracks}
    for edge in edges:
        if edge.kind == "main" and edge.source in main_adj and edge.target in main_in_deg:
            src_trk = audio_tracks[edge.source]
            if src_trk.is_inactive or src_trk.is_muted:
                continue
            if src_trk.track_type in ("RoutingFolder", "Aux") and not _is_active_stageable_bus(src_trk):
                continue
            main_adj[edge.source].append(edge.target)
            main_in_deg[edge.target] += 1

    trim_nodes = {name for name, trk in audio_tracks.items() if _has_usable_slot_a_trim(trk)}
    raw_downstream = {
        name for name in audio_tracks
        if bool(_bfs_active_nodes(name, main_adj, audio_tracks) & (trim_nodes - {name}))
    }
    terminal_nodes = _elect_terminal_nodes(audio_tracks, main_adj, main_in_deg, trim_nodes, raw_downstream)
    post_terminal_nodes = {
        nxt for term in terminal_nodes
        for nxt in _bfs_active_nodes(term, main_adj, audio_tracks)
        if nxt not in terminal_nodes
    }
    effective_trim = set(trim_nodes)
    if trim_nodes:
        for name, trk in audio_tracks.items():
            if not _is_active_stageable_bus(trk) or name in post_terminal_nodes:
                continue
            if trk.slot_a_trim is not None and trk.slot_a_trim.is_bypassed:
                continue
            if name in terminal_nodes or (trk.track_type == "Aux" and _has_active_unbypassed_inserts(trk)):
                effective_trim.add(name)
    has_downstream_trim = {
        name for name in audio_tracks
        if bool(_bfs_active_nodes(name, main_adj, audio_tracks) & (effective_trim - {name}))
    }
    return has_downstream_trim, terminal_nodes, post_terminal_nodes, bool(trim_nodes)


def _slot_a_blocked_message(trk: PtxTrack, slot_a_ins: PtxInsert) -> str:
    plug = slot_a_ins.plugin_name or slot_a_ins.fourcc or "unknown plugin"
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
    trk: PtxTrack,
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
    trk: PtxTrack,
    node: SignalFlowNode,
    *,
    needed_trim_db: float,
    has_template_trims: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    donor_trim_by_format: Mapping[str, SlotATrimInfo],
    strict_slot_a: bool = True,
) -> SlotATrimInfo | None:
    """Return existing or auto-inserted Slot A Trim spec for *trk*, raising or recording on blocked Slot A."""
    if is_post_terminal:
        return None
    if _has_usable_slot_a_trim(trk) and trk.slot_a_trim is not None:
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
    return donor_trim


def _quantize_slot_a_trim(raw_db: float, trim_spec: SlotATrimInfo) -> float:
    clamped = max(trim_spec.min_db, min(trim_spec.max_db, float(raw_db)))
    if trim_spec.plugin_id == "trim":
        return _norm_zero(avid_trim_step_to_db(avid_trim_db_to_step(clamped)))
    return _norm_zero(round(clamped, 1))


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


def _apply_fader_override(auto_fader_db: float, bus_override_db: float | None) -> tuple[float, bool]:
    if bus_override_db is not None and math.isfinite(bus_override_db):
        ovr = _norm_zero(round(max(_FADER_MIN_DB, min(_FADER_MAX_DB, float(bus_override_db))), 1))
        return ovr, abs(ovr - auto_fader_db) >= 0.05
    return auto_fader_db, False


def _apply_trim_staging(
    node: SignalFlowNode,
    trim_spec: SlotATrimInfo,
    raw: float,
    base_fader: float,
    *,
    attenuate_only: bool,
    is_terminal: bool,
    bus_override_db: float | None,
    carry_db: float = 0.0,
) -> float:
    active_tpl = node.template_trim_db if node.slot_a_trim_active else 0.0
    node.staged = True
    node.trim_clamped_excess_db = 0.0
    out_carry = carry_db
    if attenuate_only and not is_terminal:
        target_trim = _quantize_slot_a_trim(raw, trim_spec)
        fader_floor = min(base_fader, _LEAF_SHIFT_FLOOR_DB)
        unc_fader = base_fader + active_tpl - target_trim - carry_db
        auto_fader = _norm_zero(round(max(fader_floor, min(_FADER_MAX_DB, unc_fader)), 1))
        if unc_fader > _FADER_MAX_DB or unc_fader < fader_floor:
            auto_trim = _quantize_slot_a_trim(base_fader + active_tpl - auto_fader - carry_db, trim_spec)
        else:
            auto_trim = target_trim
        if unc_fader > _FADER_MAX_DB or raw < trim_spec.min_db:
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
    node.recommended_fader_db, node.is_overridden = _apply_fader_override(auto_fader, bus_override_db)
    node.net_gain_delta_db = _norm_zero(
        round((node.recommended_trim_db - active_tpl) + (node.recommended_fader_db - base_fader), 2)
    )
    return 0.0 if node.is_overridden else out_carry


def _plan_bus_node_staging(
    trk: PtxTrack,
    node: SignalFlowNode,
    pre_in_pk: float | None,
    pre_in_rms: float | None,
    is_trans: bool,
    *,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    has_downstream_trim: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, SlotATrimInfo],
    target_peak_db: float,
    target_rms_db: float,
    bus_override_db: float | None,
    strict_slot_a: bool = True,
    carry_db: float = 0.0,
) -> float:
    if (
        not stage_bus_faders
        or node.is_muted
        or node.is_inactive
        or node.is_parallel_parked
        or pre_in_pk is None
        or pre_in_pk <= _MIN_SIGNAL_DB
    ):
        return carry_db
    active_feeders = len(node.upstream_tracks)
    raw = _compute_unconstrained_bus_gain(is_trans, pre_in_pk, pre_in_rms, target_peak_db, target_rms_db)
    base_fader = _norm_zero(round(node.template_fader_db, 1))
    active_tpl = node.template_trim_db if node.slot_a_trim_active else 0.0
    if attenuate_only and is_terminal:
        needed_trim = _norm_zero(round(min(active_tpl, raw) - active_tpl, 1))
    elif attenuate_only:
        fader_floor = min(base_fader, _LEAF_SHIFT_FLOOR_DB)
        needed_trim = _norm_zero(round(max(base_fader + active_tpl - _FADER_MAX_DB, min(base_fader + active_tpl - fader_floor, raw)) - active_tpl, 1))
    else:
        needed_trim = _norm_zero(round(raw - active_tpl, 1))
    trim_spec = (
        _resolve_bus_slot_a_trim(
            trk, node, needed_trim_db=needed_trim, has_template_trims=has_template_trims,
            is_terminal=is_terminal, is_post_terminal=is_post_terminal,
            donor_trim_by_format=donor_trim_by_format, strict_slot_a=strict_slot_a,
        )
        if active_feeders >= 1
        else None
    )
    if trim_spec is not None:
        return _apply_trim_staging(
            node, trim_spec, raw, base_fader,
            attenuate_only=attenuate_only, is_terminal=is_terminal,
            bus_override_db=bus_override_db, carry_db=carry_db,
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
        auto_db = _norm_zero(_compute_bus_target_fader(
            is_trans, pre_in_pk, pre_in_rms, target_peak_db, target_rms_db,
            attenuate_only=attenuate_only, max_fader_db=max_bus_fader_db, base_fader_db=base_fader,
        ))
        node.auto_fader_db = auto_db
        node.recommended_fader_db, node.is_overridden = _apply_fader_override(auto_db, bus_override_db)
        node.net_gain_delta_db = _norm_zero(round(node.recommended_fader_db - base_fader, 2))
    return carry_db


def _make_initial_node(
    ptx: PtxSession,
    trk: PtxTrack,
    leaf_meta: tuple[float, float] | tuple[float, float, float] | None,
    *,
    is_terminal: bool = False,
    is_post_terminal: bool = False,
) -> SignalFlowNode:
    tpl_vol = 0.0 if trk.volume_db is None else float(trk.volume_db)
    eff_fader = _clamp_fader_db(leaf_meta[1]) if leaf_meta is not None else round(tpl_vol, 1)
    eff_clip = round(_clamp_clip_gain(leaf_meta[0]), 2) if leaf_meta is not None else 0.0
    eff_baked = round(_clamp_baked_gain(leaf_meta[2]), 2) if (leaf_meta is not None and len(leaf_meta) >= 3) else 0.0
    active_ins = [ins.plugin_name or ins.fourcc for ins in trk.inserts if ins.is_active and not ins.is_bypassed]
    trim = trk.slot_a_trim
    return SignalFlowNode(
        track_name=trk.name,
        track_type=trk.track_type,
        format=trk.format,
        input_bus=trk.input_path.name if trk.input_path else None,
        output_bus=trk.main_output.name if trk.main_output else None,
        parent_folder=trk.parent_folder_name or None,
        active_inserts=active_ins,
        is_muted=trk.is_muted,
        is_inactive=trk.is_inactive,
        is_parallel_parked=(eff_fader <= _PARALLEL_INF_DB),
        baked_gain_db=eff_baked,
        clip_gain_db=eff_clip,
        slot_a_trim_id=trim.plugin_id if trim else None,
        slot_a_trim_plugin=trim.plugin_name if trim else None,
        slot_a_trim_active=bool(trim and trim.is_active and not trim.is_bypassed),
        slot_a_trim_bypassed=bool(trim and trim.is_bypassed),
        slot_a_trim_min_db=trim.min_db if trim else None,
        slot_a_trim_max_db=trim.max_db if trim else None,
        is_terminal=is_terminal,
        is_post_terminal=is_post_terminal,
        template_trim_db=round(trim.gain_db, 2 if (trim and trim.plugin_id == "trim") else 1) if trim else 0.0,
        recommended_trim_db=round(trim.gain_db, 2 if (trim and trim.plugin_id == "trim") else 1) if trim else 0.0,
        template_fader_db=eff_fader,
        vca_offset_db=round(_resolve_vca_offset_db(ptx, trk), 2),
        recommended_fader_db=eff_fader,
    )


def _empty_like_format(is_mono: bool) -> np.ndarray:
    return np.zeros(0, dtype=np.float64) if is_mono else np.zeros((0, 2), dtype=np.float64)


def _has_audible_energy(buf: np.ndarray) -> bool:
    return buf.size > 0 and float(np.max(np.abs(buf))) > _MIN_SIGNAL_LIN


def _evaluate_leaf_node(
    trk: PtxTrack,
    node: SignalFlowNode,
    leaf: LeafAudioInput,
    session_sr: int,
    silence_det: SilenceDetector,
    classifier_det: AudioClassifierDetector,
    *,
    leaf_fader_shift_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    is_mono = trk.format == "Mono"
    empty = _empty_like_format(is_mono)
    norm = _normalize_track_buffer(leaf.samples, is_mono)
    if norm is None:
        log.warning("Signal-flow DAG skipped non-finite audio on track %r", trk.name)
        return empty, empty
    base_buf = _resample_if_needed(norm, leaf.samplerate, session_sr)
    pre_buf = base_buf * _db_to_lin(_clamp_clip_gain(leaf.clip_gain_db))
    pre_pk, pre_rms, cls_label, is_trans = _measure_buffer(pre_buf, session_sr, silence_det, classifier_det)
    node.pre_fader_peak_db, node.pre_fader_rms_db, node.classification, node.is_transient = pre_pk, pre_rms, cls_label, is_trans
    eff_fader = _clamp_fader_db(leaf.fader_db + leaf_fader_shift_db)
    node.baked_gain_db = round(_clamp_baked_gain(leaf.baked_gain_db), 2)
    node.recommended_fader_db = eff_fader
    node.net_gain_delta_db = _norm_zero(round(node.baked_gain_db + node.clip_gain_db + eff_fader, 2))
    if node.is_muted or node.is_inactive:
        return empty, empty
    total_db = eff_fader + node.vca_offset_db
    post_gain = _db_to_lin(total_db)
    if post_gain <= 0.0 or not _has_audible_energy(pre_buf):
        return pre_buf, empty
    post_buf = pre_buf * post_gain
    if not _has_audible_energy(post_buf):
        return pre_buf, empty
    node.post_fader_peak_db = round(pre_pk + total_db, 2) if pre_pk is not None else None
    node.post_fader_rms_db = round(pre_rms + total_db, 2) if pre_rms is not None else None
    return pre_buf, post_buf


def _evaluate_bus_node(
    trk: PtxTrack,
    node: SignalFlowNode,
    pre_buf: np.ndarray | None,
    session_sr: int,
    pan_law_db: float,
    target_peak_db: float,
    target_rms_db: float,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    has_downstream_trim: bool,
    is_terminal: bool,
    is_post_terminal: bool,
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, SlotATrimInfo],
    silence_det: SilenceDetector,
    classifier_det: AudioClassifierDetector,
    *,
    bus_override_db: float | None = None,
    strict_slot_a: bool = True,
    carry_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    is_mono = trk.format == "Mono"
    empty = _empty_like_format(is_mono)
    if pre_buf is None or pre_buf.shape[0] == 0:
        return empty, empty, carry_db
    pre_in_pk, pre_in_rms, cls_label, is_trans = _measure_buffer(pre_buf, session_sr, silence_det, classifier_det)
    node.pre_insert_peak_db, node.pre_insert_rms_db = pre_in_pk, pre_in_rms
    node.classification, node.is_transient = cls_label, is_trans

    out_carry = _plan_bus_node_staging(
        trk, node, pre_in_pk, pre_in_rms, is_trans,
        stage_bus_faders=stage_bus_faders, attenuate_only=attenuate_only,
        max_bus_fader_db=max_bus_fader_db, has_downstream_trim=has_downstream_trim,
        is_terminal=is_terminal, is_post_terminal=is_post_terminal,
        has_template_trims=has_template_trims, donor_trim_by_format=donor_trim_by_format,
        target_peak_db=target_peak_db, target_rms_db=target_rms_db,
        bus_override_db=bus_override_db, strict_slot_a=strict_slot_a, carry_db=carry_db,
    )
    if node.staged and node.staging_strategy in ("trim_and_fader", "trim_only"):
        eff_trim_db = node.recommended_trim_db
    elif trk.slot_a_trim is not None and trk.slot_a_trim.is_active and not trk.slot_a_trim.is_bypassed:
        eff_trim_db = trk.slot_a_trim.gain_db
    else:
        eff_trim_db = 0.0

    trim_lin = _db_to_lin(eff_trim_db)
    if trim_lin <= 0.0:
        post_trim_buf = empty
        node.pre_fader_peak_db, node.pre_fader_rms_db = None, None
    elif abs(eff_trim_db) < 0.005:
        post_trim_buf = pre_buf
        node.pre_fader_peak_db, node.pre_fader_rms_db = pre_in_pk, pre_in_rms
    else:
        post_trim_buf = pre_buf * trim_lin
        node.pre_fader_peak_db = round(pre_in_pk + eff_trim_db, 2) if pre_in_pk is not None else None
        node.pre_fader_rms_db = round(pre_in_rms + eff_trim_db, 2) if pre_in_rms is not None else None

    if node.is_muted or node.is_inactive:
        return empty, empty, out_carry
    total_db = node.recommended_fader_db + node.vca_offset_db
    post_gain = _db_to_lin(total_db)
    if post_gain <= 0.0 or not _has_audible_energy(post_trim_buf):
        return post_trim_buf, empty, out_carry
    post_buf = post_trim_buf * post_gain
    if not _has_audible_energy(post_buf):
        return post_trim_buf, empty, out_carry
    has_non_identity_stereo_pan = (not is_mono) and (
        (trk.pan_left is not None and trk.pan_left != -100)
        or (trk.pan_right is not None and trk.pan_right != 100)
    )
    if has_non_identity_stereo_pan:
        panned_post = _pan_stereo(post_buf, trk.pan_left, trk.pan_right, pan_law_db)
        post_pk, post_rms, _, _ = _measure_buffer(panned_post, session_sr, silence_det, classifier_det)
        node.post_fader_peak_db, node.post_fader_rms_db = post_pk, post_rms
    else:
        node.post_fader_peak_db = round(node.pre_fader_peak_db + total_db, 2) if node.pre_fader_peak_db is not None else None
        node.post_fader_rms_db = round(node.pre_fader_rms_db + total_db, 2) if node.pre_fader_rms_db is not None else None
    return post_trim_buf, post_buf, out_carry


def _propagate_outgoing(
    source_name: str,
    src_trk: PtxTrack,
    pre_buf: np.ndarray,
    post_buf: np.ndarray,
    out_edges: list[SignalFlowEdge],
    audio_tracks: dict[str, PtxTrack],
    nodes: dict[str, SignalFlowNode],
    accumulators: dict[str, np.ndarray | None],
    pan_law_db: float,
) -> None:
    for edge in out_edges:
        if edge.kind == "main":
            raw_src = post_buf
        else:
            src_buf = pre_buf if edge.is_pre_fader else post_buf
            gain = _db_to_lin(float(edge.send_volume_db or 0.0))
            if src_buf.shape[0] == 0 or gain <= 0.0:
                continue
            raw_src = src_buf * gain
        if not _has_audible_energy(raw_src):
            continue
        contrib = _adapt_for_target(raw_src, src_trk, audio_tracks[edge.target], pan_law_db)
        if not _has_audible_energy(contrib):
            continue
        if source_name not in nodes[edge.target].upstream_tracks:
            nodes[edge.target].upstream_tracks.append(source_name)
        accumulators[edge.target] = _accumulate_into(accumulators.get(edge.target), contrib)


def _run_dag_traversal(
    order: list[str],
    audio_tracks: dict[str, PtxTrack],
    nodes: dict[str, SignalFlowNode],
    outgoing: dict[str, list[SignalFlowEdge]],
    leaf_map: dict[str, LeafAudioInput],
    gains_map: dict[str, tuple[float, float] | tuple[float, float, float]],
    overrides_map: dict[str, float],
    session_sr: int,
    pan_law_db: float,
    target_peak_db: float,
    target_rms_db: float,
    stage_bus_faders: bool,
    attenuate_only: bool,
    max_bus_fader_db: float,
    downstream_trim_nodes: set[str],
    terminal_nodes: set[str],
    post_terminal_nodes: set[str],
    has_template_trims: bool,
    donor_trim_by_format: Mapping[str, SlotATrimInfo],
    silence_det: SilenceDetector,
    classifier_det: AudioClassifierDetector,
    leaf_loader: Callable[[str], LeafAudioInput | None] | None,
    progress_cb: Callable[[int, int, str], None] | None,
    cancel_cb: Callable[[], bool] | None,
    *,
    strict_slot_a: bool = True,
    leaf_fader_shift_db: float = 0.0,
    leaf_cache: dict[str, LeafAudioInput | None] | None = None,
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
                trk, node, leaf, session_sr, silence_det, classifier_det, leaf_fader_shift_db=leaf_fader_shift_db,
            )
            carry_out[name] = 0.0
        else:
            acc = accumulators.pop(name, None)
            in_carries = [carry_out.get(u, 0.0) for u in node.upstream_tracks]
            in_carry = (sum(in_carries) / len(in_carries)) if in_carries else 0.0
            pre_buf, post_buf, out_carry = _evaluate_bus_node(
                trk, node, acc, session_sr, pan_law_db, target_peak_db, target_rms_db,
                stage_bus_faders, attenuate_only, max_bus_fader_db,
                name in downstream_trim_nodes, name in terminal_nodes,
                name in post_terminal_nodes, has_template_trims, donor_trim_by_format,
                silence_det, classifier_det,
                bus_override_db=overrides_map.get(lk), strict_slot_a=strict_slot_a, carry_db=in_carry,
            )
            carry_out[name] = out_carry
        _propagate_outgoing(name, trk, pre_buf, post_buf, outgoing[name], audio_tracks, nodes, accumulators, pan_law_db)


def _compute_next_leaf_shift_step(nodes: Mapping[str, SignalFlowNode]) -> float:
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
            or n.vca_offset_db <= _PARALLEL_INF_DB
        ):
            continue
        fader_margin = min(n.recommended_fader_db, n.recommended_fader_db + n.vca_offset_db) - _LEAF_SHIFT_FLOOR_DB
        level_margin = n.post_fader_peak_db - (_MIN_SIGNAL_DB + 1.0)
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
    ptx: PtxSession,
    audio_tracks: Mapping[str, PtxTrack],
    leaf_map: Mapping[str, LeafAudioInput],
    gains_map: Mapping[str, tuple[float, float] | tuple[float, float, float]],
    terminal_nodes: set[str],
    post_terminal_nodes: set[str],
    leaf_fader_shift_db: float,
) -> dict[str, SignalFlowNode]:
    nodes: dict[str, SignalFlowNode] = {}
    for name, trk in audio_tracks.items():
        lk = name.lower()
        if lk in leaf_map:
            inp = leaf_map[lk]
            meta: tuple[float, float] | tuple[float, float, float] | None = (
                inp.clip_gain_db, inp.fader_db, inp.baked_gain_db,
            )
        else:
            meta = gains_map.get(lk)
        if meta is not None and abs(leaf_fader_shift_db) >= 0.05:
            baked = meta[2] if len(meta) >= 3 else 0.0
            meta = (meta[0], meta[1] + leaf_fader_shift_db, baked)
        nodes[name] = _make_initial_node(
            ptx, trk, meta, is_terminal=(name in terminal_nodes), is_post_terminal=(name in post_terminal_nodes),
        )
    return nodes


def build_annotated_signal_flow(
    ptx: PtxSession,
    leaf_audio: Mapping[str, LeafAudioInput] | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    stage_bus_faders: bool = True,
    attenuate_only: bool = False,
    include_sends: bool = False,
    strict_slot_a: bool = True,
    allow_leaf_convergence: bool = False,
    leaf_loader: Callable[[str], LeafAudioInput | None] | None = None,
    leaf_gains: Mapping[str, tuple[float, float] | tuple[float, float, float]] | None = None,
    bus_fader_overrides: Mapping[str, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SignalFlowGraph:
    """Reconstruct the Pro Tools signal-flow DAG, sum audio in float64, and stage bus faders."""
    cfg = dict(config or {})
    target_peak_db, target_rms_db = float(cfg.get("target_peak", -6.0)), float(cfg.get("target_rms", -18.0))
    pan_law_db = float(cfg.get("protools_pan_law_db", -3.0))
    max_bus_fader_db = max(0.0, _FADER_MAX_DB - max(0.0, float(cfg.get("fader_headroom_db", 8.0))))
    dbfs_conv = str(cfg.get("dbfs_convention", "standard"))

    bus_cfg = {**cfg, "force_transient": [], "force_sustained": []}
    silence_det, classifier_det = SilenceDetector(), AudioClassifierDetector()
    silence_det.configure(bus_cfg)
    classifier_det.configure(bus_cfg)

    leaf_map = {k.lower(): v for k, v in (leaf_audio or {}).items()}
    gains_map = {k.lower(): v for k, v in (leaf_gains or {}).items()}
    overrides_map = _parse_overrides_map(bus_fader_overrides)
    session_sr = ptx.info.sample_rate or next((inp.samplerate for inp in leaf_map.values() if inp.samplerate > 0), 48000)

    audio_tracks, edges = _build_routing_edges(ptx, include_sends=include_sends)
    order, cycles = _topological_sort(list(audio_tracks), edges)
    outgoing: dict[str, list[SignalFlowEdge]] = {name: [] for name in audio_tracks}
    for edge in edges:
        outgoing[edge.source].append(edge)

    downstream_trim_nodes, terminal_nodes, post_terminal_nodes, has_template_trims = _analyze_trim_topology(audio_tracks, edges)
    donor_trim_by_format: dict[str, SlotATrimInfo] = {}
    if has_template_trims:
        for fmt in {t.format for t in audio_tracks.values()}:
            donor = select_preferred_slot_a_trim_donor(ptx, fmt)
            if donor is not None and donor.slot_a_trim is not None:
                donor_trim_by_format[fmt] = donor.slot_a_trim

    can_converge = allow_leaf_convergence and stage_bus_faders and attenuate_only
    leaf_cache: dict[str, LeafAudioInput | None] | None = {} if can_converge else None
    leaf_shift_db, passes, nodes = 0.0, 0, {}
    while True:
        passes += 1
        nodes = _init_signal_flow_nodes(ptx, audio_tracks, leaf_map, gains_map, terminal_nodes, post_terminal_nodes, leaf_shift_db)
        _run_dag_traversal(
            order, audio_tracks, nodes, outgoing, leaf_map, gains_map, overrides_map,
            session_sr, pan_law_db, target_peak_db, target_rms_db,
            stage_bus_faders, attenuate_only, max_bus_fader_db,
            downstream_trim_nodes, terminal_nodes, post_terminal_nodes,
            has_template_trims, donor_trim_by_format, silence_det, classifier_det,
            leaf_loader, progress_cb if passes == 1 else None, cancel_cb,
            strict_slot_a=strict_slot_a, leaf_fader_shift_db=leaf_shift_db, leaf_cache=leaf_cache,
        )
        if not can_converge or passes >= _MAX_CONVERGENCE_PASSES:
            break
        step = _compute_next_leaf_shift_step(nodes)
        if step < 0.1:
            break
        leaf_shift_db = round(leaf_shift_db - step, 1)

    blocked = {name: n.slot_a_blocked_error for name, n in nodes.items() if n.slot_a_blocked_error}
    val_rep = validate_signal_flow_balance(
        ptx, nodes, edges, attenuate_only=attenuate_only, passes=passes, leaf_fader_shift_db=leaf_shift_db,
    )
    return SignalFlowGraph(
        session_sr, target_peak_db, target_rms_db, pan_law_db, dbfs_conv,
        order, nodes, edges, cycles, {}, round(leaf_shift_db, 1), blocked, val_rep.to_dict(),
    )
