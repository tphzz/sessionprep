"""DAW-agnostic mixer, gain-limit, staging-plan, and signal-flow DAG data models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

GAIN_MODE_CLIP_AND_FADER = "clip_gain_and_fader"
GAIN_MODE_CLIP_UNITY = "clip_gain_unity"
GAIN_MODE_TRIM_AND_FADER = "trim_and_fader"
GAIN_MODE_TRIM_UNITY = "trim_unity"
GAIN_MODE_FADER_ONLY = "fader_only"
GAIN_MODE_NONE = "none"

VALID_GAIN_MODES = (
    GAIN_MODE_CLIP_AND_FADER,
    GAIN_MODE_CLIP_UNITY,
    GAIN_MODE_TRIM_AND_FADER,
    GAIN_MODE_TRIM_UNITY,
    GAIN_MODE_FADER_ONLY,
    GAIN_MODE_NONE,
)

AUDIO_TRACK_TYPES = frozenset({"Audio", "RoutingFolder", "Aux", "Instrument", "Master"})
MIN_SIGNAL_DB = -120.0
MIN_SIGNAL_LIN = 10.0 ** (MIN_SIGNAL_DB / 20.0)


@dataclass(frozen=True)
class HostGainLimits:
    """Host DAW fader, clip-gain, leaf-trim, and solver convergence limits (in dB)."""

    fader_min_db: float = -144.0
    fader_max_db: float = 12.0
    clip_gain_min_db: float = -144.0
    clip_gain_max_db: float = 36.0
    leaf_trim_min_db: float = -200.0
    leaf_trim_max_db: float = 200.0
    leaf_shift_floor_db: float = -136.0
    parallel_inf_db: float = -140.0


PROTOOLS_GAIN_LIMITS = HostGainLimits()
DAWPROJECT_GAIN_LIMITS = HostGainLimits(
    fader_min_db=-144.0,
    fader_max_db=6.0,
    clip_gain_min_db=-144.0,
    clip_gain_max_db=24.0,
    leaf_trim_min_db=-200.0,
    leaf_trim_max_db=18.0,
    leaf_shift_floor_db=-136.0,
    parallel_inf_db=-140.0,
)


@dataclass(frozen=True)
class TrimPluginInfo:
    """Validated pre-insert Trim plugin state and quantization profile (host-agnostic)."""

    plugin_id: str
    plugin_name: str
    gain_db: float
    min_db: float
    max_db: float
    is_active: bool
    is_bypassed: bool
    is_clean_donor: bool = False
    precision_decimals: int = 1
    quantize_db: Callable[[float], float] | None = None


@dataclass(frozen=True)
class MixerInsert:
    """One insert slot on a mixer track/bus."""

    slot_index: int
    plugin_name: str
    is_active: bool = True
    is_bypassed: bool = False
    plugin_identifier: str = ""


@dataclass(frozen=True)
class MixerSend:
    """One send routing from a mixer track/bus to a named destination bus."""

    destination_bus: str
    slot: str = ""
    volume_db: float | None = 0.0
    is_muted: bool = False
    is_pre_fader: bool = False


@dataclass(frozen=True)
class MixerTrack:
    """DAW-agnostic representation of a mixer track, folder, aux bus, or master."""

    name: str
    track_type: str
    format: str
    index: int = 0
    input_bus: str | None = None
    input_is_bus: bool = True
    output_bus: str | None = None
    parent_folder: str = ""
    volume_db: float | None = 0.0
    pan_left: float | None = None
    pan_right: float | None = None
    is_muted: bool = False
    is_inactive: bool = False
    vca_offset_db: float = 0.0
    inserts: tuple[MixerInsert, ...] = ()
    sends: tuple[MixerSend, ...] = ()
    slot_a_trim: TrimPluginInfo | None = None


@dataclass(frozen=True)
class MixerSession:
    """DAW-agnostic mixer snapshot consumed by the signal-flow solver and balance validator."""

    sample_rate: int = 48000
    tracks: tuple[MixerTrack, ...] = ()
    donor_trim_by_format: Mapping[str, TrimPluginInfo] | None = None


@dataclass(frozen=True)
class TrackStagingPlan:
    """Resolved transfer media path plus planned pre-insert, trim, and fader gain staging."""

    filepath: str
    clip_gain_db: float = 0.0
    trim_gain_db: float = 0.0
    fader_db: float = 0.0
    baked_gain_db: float = 0.0
    prepared_fader_db: float = 0.0


@dataclass(frozen=True)
class LeafAudioInput:
    """Time-aligned audio buffer (starting at sample 0) plus planned leaf gains."""

    samples: np.ndarray
    samplerate: int
    clip_gain_db: float = 0.0
    fader_db: float = 0.0
    baked_gain_db: float = 0.0
    trim_gain_db: float = 0.0


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
    leaf_trim_db: float = 0.0
    slot_a_trim_id: str | None = None
    slot_a_trim_plugin: str | None = None
    slot_a_trim_active: bool = False
    slot_a_trim_bypassed: bool = False
    slot_a_trim_auto_insert: bool = False
    slot_a_trim_min_db: float | None = None
    slot_a_trim_max_db: float | None = None
    slot_a_trim_precision: int = 1
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
    governing: str | None = None
    is_transient: bool = False
    non_finite_error: str | None = None
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
            prec = max(1, int(node.slot_a_trim_precision))
            eps = 0.02 if prec >= 2 else 0.05
            if abs(node.recommended_trim_db - node.template_trim_db) >= eps or (
                abs(node.recommended_trim_db) >= eps and not node.slot_a_trim_active
            ):
                out[name] = round(node.recommended_trim_db, prec)
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
