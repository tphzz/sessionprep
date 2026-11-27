"""Pro Tools ``.ptx`` adapter for SessionPrep's DAW-agnostic signal-flow DAG engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..signal_flow import (
    PROTOOLS_GAIN_LIMITS,
    LeafAudioInput,
    MixerInsert,
    MixerSend,
    MixerSession,
    MixerTrack,
    SignalFlowEdge,
    SignalFlowGraph,
    SignalFlowNode,
    TrimPluginInfo,
    build_signal_flow,
    pan_gains,
)
from .models import PtxTrack, SlotATrimInfo
from .parser import PtxSession
from .plugin_trim import avid_trim_db_to_step, avid_trim_step_to_db
from .queries import vcas_for_track

__all__ = [
    "LeafAudioInput",
    "SignalFlowEdge",
    "SignalFlowGraph",
    "SignalFlowNode",
    "build_annotated_signal_flow",
    "pan_gains",
    "ptx_to_mixer_session",
]


def _quantize_avid_trim_db(db: float) -> float:
    return avid_trim_step_to_db(avid_trim_db_to_step(db))


def _resolve_vca_offset_db(ptx: PtxSession, track: PtxTrack) -> float:
    """Recursively accumulate active Pro Tools VCA attenuation dB controlling *track* (cycle-safe)."""
    total_db = 0.0
    for vca in vcas_for_track(ptx, track):
        if vca.is_muted or (vca.volume_db is not None and vca.volume_db <= PROTOOLS_GAIN_LIMITS.parallel_inf_db):
            return PROTOOLS_GAIN_LIMITS.fader_min_db
        total_db += float(vca.volume_db or 0.0)
    return total_db


def _adapt_slot_a_trim(trim: SlotATrimInfo | None) -> TrimPluginInfo | None:
    if trim is None:
        return None
    is_avid = trim.plugin_id == "trim"
    return TrimPluginInfo(
        plugin_id=trim.plugin_id,
        plugin_name=trim.plugin_name,
        gain_db=trim.gain_db,
        min_db=trim.min_db,
        max_db=trim.max_db,
        is_active=trim.is_active,
        is_bypassed=trim.is_bypassed,
        is_clean_donor=trim.is_clean_donor,
        precision_decimals=2 if is_avid else 1,
        quantize_db=_quantize_avid_trim_db if is_avid else None,
    )


def ptx_to_mixer_session(ptx: PtxSession) -> MixerSession:
    """Adapt a parsed Pro Tools :class:`PtxSession` into a DAW-agnostic :class:`MixerSession`."""
    tracks: list[MixerTrack] = []
    for trk in ptx.tracks:
        inserts = tuple(
            MixerInsert(
                slot_index=ins.slot_index,
                plugin_name=ins.plugin_name,
                is_active=ins.is_active,
                is_bypassed=ins.is_bypassed,
                plugin_identifier=ins.fourcc,
            )
            for ins in trk.inserts
        )
        sends = tuple(
            MixerSend(
                destination_bus=snd.destination.name if snd.destination else "",
                slot=snd.slot,
                volume_db=snd.volume_db,
                is_muted=snd.is_muted,
                is_pre_fader=snd.is_pre_fader,
            )
            for snd in trk.sends
            if snd.destination and snd.destination.name
        )
        tracks.append(
            MixerTrack(
                name=trk.name,
                track_type=trk.track_type,
                format=trk.format,
                index=trk.index,
                input_bus=trk.input_path.name if trk.input_path else None,
                input_is_bus=bool(trk.input_path and trk.input_path.kind == "bus"),
                output_bus=trk.main_output.name if trk.main_output else None,
                parent_folder=trk.parent_folder_name,
                volume_db=trk.volume_db,
                pan_left=trk.pan_left,
                pan_right=trk.pan_right,
                is_muted=trk.is_muted,
                is_inactive=trk.is_inactive,
                vca_offset_db=_resolve_vca_offset_db(ptx, trk),
                inserts=inserts,
                sends=sends,
                slot_a_trim=_adapt_slot_a_trim(trk.slot_a_trim),
            )
        )
    return MixerSession(sample_rate=ptx.info.sample_rate or 0, tracks=tuple(tracks))


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
    leaf_gains: Mapping[str, tuple[float, ...]] | None = None,
    bus_fader_overrides: Mapping[str, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SignalFlowGraph:
    """Reconstruct the Pro Tools signal-flow DAG, sum audio in float64, and stage bus faders."""
    mixer = ptx_to_mixer_session(ptx)
    cfg = dict(config or {})
    pan_law_db = float(cfg.get("protools_pan_law_db", -3.0))
    return build_signal_flow(
        mixer,
        leaf_audio=leaf_audio,
        config=cfg,
        host_limits=PROTOOLS_GAIN_LIMITS,
        pan_law_db=pan_law_db,
        stage_bus_faders=stage_bus_faders,
        attenuate_only=attenuate_only,
        include_sends=include_sends,
        strict_slot_a=strict_slot_a,
        allow_leaf_convergence=allow_leaf_convergence,
        leaf_loader=leaf_loader,
        leaf_gains=leaf_gains,
        bus_fader_overrides=bus_fader_overrides,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
