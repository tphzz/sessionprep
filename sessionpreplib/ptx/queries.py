"""High-level query, folder-tree, mix-state evaluation, and session-audit APIs for ``.ptx``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .models import (
    PtxAudioFile,
    PtxClip,
    PtxCompoundClip,
    PtxGroup,
    PtxInsert,
    PtxSessionLike,
    PtxTrack,
)

_INF_DB = -140.0
_MIN_FADER_DB = -144.0


@dataclass
class PtxFolderNode:
    """Tree node representing a folder or leaf track in the session's folder hierarchy."""

    track: PtxTrack
    children: list[PtxFolderNode] = field(default_factory=list)


@dataclass(frozen=True)
class PtxSessionAudit:
    """Readiness and complexity summary for a parsed ``.ptx`` session."""

    automated_volume_tracks: tuple[str, ...] = ()
    automated_pan_tracks: tuple[str, ...] = ()
    automated_mute_tracks: tuple[str, ...] = ()
    automated_send_tracks: tuple[str, ...] = ()
    automated_plugin_tracks: tuple[str, ...] = ()
    dynamic_clip_gain_tracks: tuple[str, ...] = ()
    warped_tracks: tuple[str, ...] = ()
    ara_tracks: tuple[str, ...] = ()
    muted_clip_count: int = 0
    compound_clip_count: int = 0
    alternate_playlist_tracks: tuple[str, ...] = ()
    external_audio_files: tuple[str, ...] = ()
    unused_audio_files: tuple[str, ...] = ()
    has_tempo_changes: bool = False
    has_unparsed_tempo_ramps: bool = False
    has_meter_changes: bool = False
    parser_warnings: tuple[str, ...] = ()


def _resolve_track(session: PtxSessionLike, track_or_name: PtxTrack | str) -> PtxTrack | None:
    return track_or_name if isinstance(track_or_name, PtxTrack) else session.track_by_name(track_or_name)


def _subtree_track_names(session: PtxSessionLike, folder_name: str, *, recursive: bool) -> set[str]:
    target_low = folder_name.lower()
    children_by_parent: dict[str, list[str]] = {}
    for trk in session.tracks:
        if trk.parent_folder_name:
            children_by_parent.setdefault(trk.parent_folder_name.lower(), []).append(trk.name.lower())
    if not recursive:
        return set(children_by_parent.get(target_low, ()))
    allowed: set[str] = set()
    frontier: deque[str] = deque([target_low])
    seen_folders: set[str] = {target_low}
    while frontier:
        cur = frontier.popleft()
        for child_low in children_by_parent.get(cur, ()):
            allowed.add(child_low)
            if child_low in children_by_parent and child_low not in seen_folders:
                seen_folders.add(child_low)
                frontier.append(child_low)
    return allowed


def iter_tracks(
    session: PtxSessionLike,
    *,
    track_types: set[str] | None = None,
    include_hidden: bool = True,
    include_inactive: bool = True,
    folder: str | None = None,
    recursive: bool = False,
) -> list[PtxTrack]:
    """Filter tracks by type, visibility, active state, and optional parent folder."""
    allowed_types = {t.lower() for t in track_types} if track_types is not None else None
    folder_members = _subtree_track_names(session, folder, recursive=recursive) if folder is not None else None
    out: list[PtxTrack] = []
    for trk in session.tracks:
        if allowed_types is not None and trk.track_type.lower() not in allowed_types:
            continue
        if not include_hidden and trk.is_hidden:
            continue
        if not include_inactive and trk.is_inactive:
            continue
        if folder_members is not None and trk.name.lower() not in folder_members:
            continue
        out.append(trk)
    return out


def build_folder_tree(session: PtxSessionLike) -> list[PtxFolderNode]:
    """Build the nested top-level folder and track hierarchy (cycle-safe)."""
    nodes = {t.name.lower(): PtxFolderNode(track=t) for t in session.tracks}
    parent_of: dict[str, str] = {}
    for trk in session.tracks:
        low = trk.name.lower()
        p_low = trk.parent_folder_name.lower()
        if p_low and p_low != low and p_low in nodes:
            parent_of[low] = p_low

    def _creates_cycle(child_low: str, cand_parent: str) -> bool:
        cur: str | None = cand_parent
        seen: set[str] = {child_low}
        while cur is not None:
            if cur in seen:
                return True
            seen.add(cur)
            cur = parent_of.get(cur)
        return False

    roots: list[PtxFolderNode] = []
    for trk in session.tracks:
        low = trk.name.lower()
        parent_key = parent_of.get(low)
        if parent_key is not None and not _creates_cycle(low, parent_key):
            nodes[parent_key].children.append(nodes[low])
        else:
            parent_of.pop(low, None)
            roots.append(nodes[low])
    return roots


def groups_for_track(session: PtxSessionLike, track_or_name: PtxTrack | str, *, active_only: bool = False) -> list[PtxGroup]:
    """Return unique groups (preferring mix/edit entries) containing *track_or_name*."""
    trk = _resolve_track(session, track_or_name)
    if trk is None:
        return []
    out: list[PtxGroup] = []
    seen: set[str] = set()
    for gname in trk.groups:
        low = gname.lower()
        if low in seen:
            continue
        grp = session.group_by_name(gname)
        if grp is None:
            continue
        if active_only and not (grp.is_edit_active or grp.is_mix_active):
            continue
        seen.add(low)
        out.append(grp)
    return out


def vcas_for_track(session: PtxSessionLike, track_or_name: PtxTrack | str) -> list[PtxTrack]:
    """Return all active/controlling VCA master tracks recursively affecting *track_or_name*."""
    trk = _resolve_track(session, track_or_name)
    if trk is None:
        return []
    visited_tracks: set[str] = {trk.name.lower()}
    queue: deque[PtxTrack] = deque([trk])
    seen_vcas: set[str] = {trk.name.lower()}
    vcas: list[PtxTrack] = []
    while queue:
        cur = queue.popleft()
        for gname in cur.groups:
            grp = session.group_by_name(gname)
            if grp is None or not grp.vca_track_name:
                continue
            vca = session.track_by_name(grp.vca_track_name)
            if vca is None or vca.is_inactive or vca.name.lower() in seen_vcas:
                continue
            seen_vcas.add(vca.name.lower())
            vcas.append(vca)
            if vca.name.lower() not in visited_tracks:
                visited_tracks.add(vca.name.lower())
                queue.append(vca)
    return vcas


def find_inserts(
    session: PtxSessionLike,
    *,
    plugin_name: str | None = None,
    manufacturer_4cc: str | None = None,
    bundle_id: str | None = None,
    active_only: bool = False,
    include_bypassed: bool = True,
) -> list[tuple[PtxTrack, PtxInsert]]:
    """Search inserts across all tracks by plugin name, manufacturer 4CC, bundle ID, and state."""
    p_low = plugin_name.lower() if plugin_name is not None else None
    m_low = manufacturer_4cc.lower() if manufacturer_4cc is not None else None
    b_low = bundle_id.lower() if bundle_id is not None else None
    matches: list[tuple[PtxTrack, PtxInsert]] = []
    for trk in session.tracks:
        for ins in trk.inserts:
            if active_only and not ins.is_active:
                continue
            if not include_bypassed and ins.is_bypassed:
                continue
            if p_low is not None and p_low not in ins.plugin_name.lower():
                continue
            if m_low is not None and not ins.fourcc.lower().startswith(f"{m_low}/"):
                continue
            if b_low is not None and ins.bundle_id.lower() != b_low:
                continue
            matches.append((trk, ins))
    return matches


def _used_audio_file_indices(session: PtxSessionLike) -> set[int]:
    used_clip_ids: set[int] = set()
    comps_by_idx = {cp.index: cp for cp in session.compound_clips}
    for pl in session.playlists:
        for plc in pl.placements:
            if plc.is_compound:
                comp = comps_by_idx.get(plc.clip_index)
                if comp is not None:
                    used_clip_ids.update(mem.clip_index for mem in comp.members)
            else:
                used_clip_ids.add(plc.clip_index)
    clips_by_idx = {c.index: c for c in session.clips}
    used_af: set[int] = set()
    for cid in used_clip_ids:
        clip = clips_by_idx.get(cid)
        if clip is not None and clip.audio_file_index is not None:
            used_af.add(clip.audio_file_index)
    return used_af


def used_audio_files(session: PtxSessionLike) -> list[PtxAudioFile]:
    """Return audio files referenced by at least one placed clip or compound member."""
    used = _used_audio_file_indices(session)
    return [af for af in session.audio_files if af.index in used]


def unused_audio_files(session: PtxSessionLike) -> list[PtxAudioFile]:
    """Return audio files present in ``0x1004`` that are not placed on any playlist."""
    used = _used_audio_file_indices(session)
    return [af for af in session.audio_files if af.index not in used]


def external_audio_files(session: PtxSessionLike) -> list[PtxAudioFile]:
    """Return audio files residing outside the session folder (``relative_path is None``)."""
    return [af for af in session.audio_files if af.relative_path is None]


def _track_vol_at(session: PtxSessionLike, trk: PtxTrack, at_seconds: float) -> float | None:
    if trk.volume_automation is not None and trk.volume_automation.is_automated:
        return trk.volume_automation.value_at_seconds(at_seconds, session.timeline)
    if trk.volume_db is not None:
        return trk.volume_db
    return trk.volume_automation.initial_value if trk.volume_automation is not None else None


def _track_muted_at(session: PtxSessionLike, trk: PtxTrack, at_seconds: float) -> bool:
    if trk.mute_automation is not None and trk.mute_automation.is_automated:
        val = trk.mute_automation.value_at_seconds(at_seconds, session.timeline, stepped=True)
        return bool(val) if val is not None else trk.is_muted
    return trk.is_muted


def effective_track_volume_db(
    session: PtxSessionLike,
    track_or_name: PtxTrack | str,
    at_seconds: float = 0.0,
    *,
    include_vca: bool = True,
) -> float | None:
    """Evaluate track fader dB at *at_seconds*, optionally adding controlling VCA offsets."""
    trk = _resolve_track(session, track_or_name)
    if trk is None:
        return None
    base = _track_vol_at(session, trk, at_seconds)
    if base is None:
        return None
    if base <= _INF_DB:
        return _MIN_FADER_DB
    total = base
    if include_vca:
        for vca in vcas_for_track(session, trk):
            v_db = _track_vol_at(session, vca, at_seconds)
            if _track_muted_at(session, vca, at_seconds) or (v_db is not None and v_db <= _INF_DB):
                return _MIN_FADER_DB
            total += float(v_db or 0.0)
    return round(total, 2)


def is_track_audible(
    session: PtxSessionLike,
    track_or_name: PtxTrack | str,
    at_seconds: float = 0.0,
    *,
    include_vca: bool = True,
) -> bool:
    """Return True when the track is active, unmuted, and above ``-inf`` at *at_seconds*."""
    trk = _resolve_track(session, track_or_name)
    if trk is None or trk.is_inactive or _track_muted_at(session, trk, at_seconds):
        return False
    vol = effective_track_volume_db(session, trk, at_seconds, include_vca=include_vca)
    return vol is None or vol > _INF_DB


def effective_send_volume_db(
    session: PtxSessionLike,
    track_or_name: PtxTrack | str,
    send_slot: str,
    at_seconds: float = 0.0,
) -> float | None:
    """Evaluate a send's effective volume dB at *at_seconds* (``-144.0`` when muted/inactive/``-inf``)."""
    trk = _resolve_track(session, track_or_name)
    snd = trk.send_by_slot(send_slot) if trk is not None else None
    if snd is None:
        return None
    if not snd.is_active:
        return _MIN_FADER_DB
    muted = snd.is_muted
    if snd.mute_automation is not None and snd.mute_automation.is_automated:
        mval = snd.mute_automation.value_at_seconds(at_seconds, session.timeline, stepped=True)
        muted = bool(mval) if mval is not None else muted
    if muted:
        return _MIN_FADER_DB
    if snd.volume_automation is not None and snd.volume_automation.is_automated:
        val = snd.volume_automation.value_at_seconds(at_seconds, session.timeline)
    elif snd.volume_db is not None:
        val = snd.volume_db
    else:
        val = snd.volume_automation.initial_value if snd.volume_automation is not None else None
    if val is None:
        return None
    return _MIN_FADER_DB if val <= _INF_DB else round(val, 2)


def _track_member_clips(
    trk: PtxTrack, clips_by_idx: dict[int, PtxClip], comps_by_idx: dict[int, PtxCompoundClip]
) -> list[PtxClip]:
    out: list[PtxClip] = []
    for plc in trk.clips:
        if plc.is_compound:
            comp = comps_by_idx.get(plc.clip_index)
            if comp is not None:
                out.extend(clips_by_idx[m.clip_index] for m in comp.members if m.clip_index in clips_by_idx)
        elif plc.clip_index in clips_by_idx:
            out.append(clips_by_idx[plc.clip_index])
    return out


def _audit_track_lists(
    session: PtxSessionLike,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    vol_a, pan_a, mute_a, send_a, plug_a = [], [], [], [], []
    dyn_cg, warped, ara, alts = [], [], [], []
    clips_by_idx = {c.index: c for c in session.clips}
    comps_by_idx = {cp.index: cp for cp in session.compound_clips}
    curves_by_idx = {cg.index: cg for cg in session.clip_gain_curves}
    for trk in session.tracks:
        if trk.volume_automation and trk.volume_automation.is_automated:
            vol_a.append(trk.name)
        if (trk.pan_left_automation and trk.pan_left_automation.is_automated) or (
            trk.pan_right_automation and trk.pan_right_automation.is_automated
        ):
            pan_a.append(trk.name)
        if trk.mute_automation and trk.mute_automation.is_automated:
            mute_a.append(trk.name)
        if any(
            (s.volume_automation and s.volume_automation.is_automated)
            or (s.mute_automation and s.mute_automation.is_automated)
            or (s.lfe_or_trim_automation and s.lfe_or_trim_automation.is_automated)
            or (s.pan_left_automation and s.pan_left_automation.is_automated)
            or (s.pan_right_automation and s.pan_right_automation.is_automated)
            for s in trk.sends
        ):
            send_a.append(trk.name)
        if any(any(lane.is_automated for lane in ins.automation) for ins in trk.inserts):
            plug_a.append(trk.name)
        trk_clips = _track_member_clips(trk, clips_by_idx, comps_by_idx)
        if any(
            (c.clip_gain_curve_index in curves_by_idx and curves_by_idx[c.clip_gain_curve_index].is_dynamic)
            or c.clip_gain_chunk_size > 37
            for c in trk_clips
        ):
            dyn_cg.append(trk.name)
        if any(c.warp_markers or abs(c.playback_rate - 1.0) > 1e-6 for c in trk_clips):
            warped.append(trk.name)
        if trk.ara_plugin is not None or any(c.ara_plugin_name for c in trk_clips):
            ara.append(trk.name)
        if trk.alternate_playlists:
            alts.append(trk.name)
    return vol_a, pan_a, mute_a, send_a, plug_a, dyn_cg, warped, ara, alts


def audit_session(session: PtxSessionLike) -> PtxSessionAudit:
    """Build a structured readiness/complexity audit for a parsed :class:`PtxSession`."""
    vol_a, pan_a, mute_a, send_a, plug_a, dyn_cg, warped, ara, alts = _audit_track_lists(session)
    muted_clips = sum(1 for pl in session.playlists for p in pl.placements if p.is_muted)
    return PtxSessionAudit(
        automated_volume_tracks=tuple(vol_a),
        automated_pan_tracks=tuple(pan_a),
        automated_mute_tracks=tuple(mute_a),
        automated_send_tracks=tuple(send_a),
        automated_plugin_tracks=tuple(plug_a),
        dynamic_clip_gain_tracks=tuple(dyn_cg),
        warped_tracks=tuple(warped),
        ara_tracks=tuple(ara),
        muted_clip_count=muted_clips,
        compound_clip_count=len(session.compound_clips),
        alternate_playlist_tracks=tuple(alts),
        external_audio_files=tuple(af.filename for af in external_audio_files(session)),
        unused_audio_files=tuple(af.filename for af in unused_audio_files(session)),
        has_tempo_changes=len(session.tempo_events) > 1,
        has_unparsed_tempo_ramps=session.has_unparsed_tempo_ramps,
        has_meter_changes=len(session.meter_events) > 1,
        parser_warnings=tuple(session.warnings),
    )
