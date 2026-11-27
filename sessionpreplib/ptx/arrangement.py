"""Resolved timeline arrangement API joining clips, audio files, fades, compounds, MIDI, and video."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import (
    PtxAudioFile,
    PtxClip,
    PtxClipGainCurve,
    PtxClipPlacement,
    PtxCompoundClip,
    PtxFade,
    PtxFadePlacement,
    PtxMidiControlEvent,
    PtxMidiControlLane,
    PtxMidiNote,
    PtxSessionLike,
    PtxTrack,
    PtxWarpMarker,
)


@dataclass(frozen=True)
class ResolvedAudioEvent:
    """Fully joined audio clip placement on a track playlist channel."""

    track_name: str
    playlist_name: str
    channel: int
    clip_name: str
    audio_file: PtxAudioFile | None
    start_sample: int
    length_samples: int
    end_sample: int
    start_seconds: float
    duration_seconds: float
    source_offset_samples: int = 0
    sync_point_samples: int = 0
    playback_rate: float = 1.0
    warp_markers: tuple[PtxWarpMarker, ...] = ()
    clip_gain_db: float = 0.0
    clip_gain_curve: PtxClipGainCurve | None = None
    fade_in: PtxFade | None = None
    fade_out: PtxFade | None = None
    is_muted: bool = False
    is_covered: bool = False
    loop_count: int | None = None
    compound_group_name: str | None = None
    ara_plugin_name: str = ""
    ara_modification_uuid: str = ""


@dataclass(frozen=True)
class ResolvedMidiEvent:
    """Fully joined MIDI clip placement with notes and controllers projected onto timeline or clip ticks."""

    track_name: str
    clip_name: str
    start_tick: int
    length_ticks: int
    start_seconds: float
    duration_seconds: float
    is_muted: bool = False
    notes: tuple[PtxMidiNote, ...] = ()
    control_lanes: tuple[PtxMidiControlLane, ...] = ()


@dataclass(frozen=True)
class ResolvedVideoEvent:
    """Fully joined video clip placement on a video track."""

    track_name: str
    clip_name: str
    start_sample: int
    start_seconds: float
    offset_frames: int = 0
    length_frames: int = 0
    duration_seconds: float | None = None


def _resolve_track(session: PtxSessionLike, track_or_name: PtxTrack | str) -> PtxTrack | None:
    if isinstance(track_or_name, PtxTrack):
        return track_or_name
    return session.track_by_name(track_or_name)


def _select_playlist_data(
    session: PtxSessionLike,
    track: PtxTrack,
    playlist: str | None,
) -> tuple[str, list[PtxClipPlacement], list[PtxFadePlacement]]:
    if playlist is None:
        return track.name, track.clips, track.fade_placements
    low = playlist.lower()
    if low == track.name.lower():
        return track.name, track.clips, track.fade_placements
    for alt in track.alternate_playlists:
        if alt.name.lower() == low:
            return alt.name, alt.placements, alt.fade_placements
    trk_low = track.name.lower()
    for pl in session.playlists:
        if pl.name.lower() == low and pl.track_name.lower() in ("", trk_low):
            return pl.name, pl.placements, pl.fade_placements
    return playlist, [], []


def _build_audio_event(
    session: PtxSessionLike,
    track_name: str,
    pl_name: str,
    channel: int,
    start_smp: int,
    clip: PtxClip | None,
    *,
    audio_files_by_idx: dict[int, PtxAudioFile],
    curves_by_idx: dict[int, PtxClipGainCurve],
    fallback_name: str,
    fallback_len: int,
    is_muted: bool,
    is_covered: bool,
    loop_count: int | None,
    compound_group_name: str | None,
) -> ResolvedAudioEvent:
    tl = session.timeline
    rate = clip.playback_rate if (clip and clip.playback_rate > 0) else 1.0
    base_len = (clip.timeline_length_samples or clip.length_samples) if clip else fallback_len
    total_len = base_len * loop_count if (loop_count and loop_count > 1) else base_len
    sync = round((clip.sync_point_samples if clip else 0) / rate)
    raw_start = start_smp - sync
    src_off = clip.start_sample if clip else 0
    if raw_start < 0:
        head_trim = -raw_start
        adj_start = 0
        total_len = max(0, total_len - head_trim)
        src_off += round(head_trim * rate)
    else:
        adj_start = raw_start
    af = audio_files_by_idx.get(clip.audio_file_index) if (clip and clip.audio_file_index is not None) else None
    curve = curves_by_idx.get(clip.clip_gain_curve_index) if (clip and clip.clip_gain_curve_index >= 0) else None
    return ResolvedAudioEvent(
        track_name=track_name,
        playlist_name=pl_name,
        channel=channel,
        clip_name=clip.name if clip else fallback_name,
        audio_file=af,
        start_sample=adj_start,
        length_samples=total_len,
        end_sample=adj_start + total_len,
        start_seconds=round(tl.samples_to_seconds(adj_start), 6),
        duration_seconds=round(tl.samples_to_seconds(total_len), 6),
        source_offset_samples=src_off,
        sync_point_samples=sync,
        playback_rate=rate,
        warp_markers=tuple(clip.warp_markers) if clip else (),
        clip_gain_db=clip.gain_db if clip else 0.0,
        clip_gain_curve=curve,
        is_muted=is_muted,
        is_covered=is_covered,
        loop_count=loop_count,
        compound_group_name=compound_group_name,
        ara_plugin_name=clip.ara_plugin_name if clip else "",
        ara_modification_uuid=clip.ara_modification_uuid if clip else "",
    )


def _find_edge_match(
    events: list[ResolvedAudioEvent], channel: int, targets: tuple[int, ...], *, use_end: bool
) -> int | None:
    best_idx: int | None = None
    best_key: tuple[int, int] | None = None
    for idx, ev in enumerate(events):
        if ev.channel != channel:
            continue
        edge = ev.end_sample if use_end else ev.start_sample
        dist = min(abs(edge - tgt) for tgt in targets)
        if dist <= 2:
            key = (dist, 1 if ev.is_muted else 0)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
    return best_idx


def _attach_channel_fades(
    events: list[ResolvedAudioEvent],
    fade_placements: list[PtxFadePlacement],
    fades_by_idx: dict[int, PtxFade],
) -> list[ResolvedAudioEvent]:
    if not events or not fade_placements:
        return events
    updated = list(events)
    for fpl in fade_placements:
        fade = fades_by_idx.get(fpl.fade_index)
        if fade is None:
            continue
        if fade.fade_type == "in":
            hit = _find_edge_match(updated, fpl.channel, (fpl.start_sample,), use_end=False)
            if hit is not None:
                updated[hit] = replace(updated[hit], fade_in=fade)
        elif fade.fade_type == "out":
            targets = (fpl.start_sample, fpl.start_sample + fade.total_samples) if fade.total_samples > 0 else (fpl.start_sample,)
            hit = _find_edge_match(updated, fpl.channel, targets, use_end=True)
            if hit is not None:
                updated[hit] = replace(updated[hit], fade_out=fade)
        elif fade.fade_type == "crossfade":
            post = max(0, fade.total_samples - fade.pre_samples)
            left_targets = (fpl.start_sample, fpl.start_sample + post) if post > 0 else (fpl.start_sample,)
            right_targets = (fpl.start_sample, max(0, fpl.start_sample - fade.pre_samples)) if fade.pre_samples > 0 else (fpl.start_sample,)
            left_i = _find_edge_match(updated, fpl.channel, left_targets, use_end=True)
            right_i = _find_edge_match(updated, fpl.channel, right_targets, use_end=False)
            if left_i is not None:
                updated[left_i] = replace(updated[left_i], fade_out=fade)
            if right_i is not None and right_i != left_i:
                updated[right_i] = replace(updated[right_i], fade_in=fade)
    return updated


def _trim_head(ev: ResolvedAudioEvent, trim_samples: int, session: PtxSessionLike) -> ResolvedAudioEvent | None:
    new_len = ev.length_samples - trim_samples
    if new_len <= 0:
        return None
    new_start = ev.start_sample + trim_samples
    new_off = ev.source_offset_samples + round(trim_samples * ev.playback_rate)
    tl = session.timeline
    return replace(
        ev, start_sample=new_start, length_samples=new_len, end_sample=new_start + new_len,
        start_seconds=round(tl.samples_to_seconds(new_start), 6),
        duration_seconds=round(tl.samples_to_seconds(new_len), 6),
        source_offset_samples=new_off, fade_in=None,
    )


def _trim_tail(ev: ResolvedAudioEvent, new_end: int, session: PtxSessionLike) -> ResolvedAudioEvent | None:
    new_len = new_end - ev.start_sample
    if new_len <= 0:
        return None
    tl = session.timeline
    return replace(
        ev, length_samples=new_len, end_sample=new_end,
        duration_seconds=round(tl.samples_to_seconds(new_len), 6), fade_out=None,
    )


def _insert_resolved_event(
    resolved: list[ResolvedAudioEvent], curr: ResolvedAudioEvent, session: PtxSessionLike
) -> None:
    tail_split: ResolvedAudioEvent | None = None
    while resolved:
        prev = resolved[-1]
        ov = prev.end_sample - curr.start_sample
        xf_allow = min(prev.fade_out.total_samples if prev.fade_out else 0, curr.fade_in.total_samples if curr.fade_in else 0)
        if ov <= 2 or ov <= xf_allow + 2:
            break
        curr_wins = prev.is_covered or (not curr.is_covered and curr.start_sample != prev.start_sample)
        if curr_wins:
            resolved.pop()
            if curr.end_sample < prev.end_sample and tail_split is None:
                tail_split = _trim_head(prev, curr.end_sample - prev.start_sample, session)
            if curr.start_sample > prev.start_sample:
                trimmed_prev = _trim_tail(prev, curr.start_sample, session)
                if trimmed_prev is not None:
                    resolved.append(trimmed_prev)
                    break
        else:
            trimmed_curr = _trim_head(curr, ov, session)
            if trimmed_curr is None:
                return
            curr = trimmed_curr
            break
    resolved.append(curr)
    if tail_split is not None:
        resolved.append(tail_split)


def _trim_channel_overlaps(events: list[ResolvedAudioEvent], session: PtxSessionLike) -> list[ResolvedAudioEvent]:
    by_ch: dict[int, list[ResolvedAudioEvent]] = {}
    for ev in events:
        by_ch.setdefault(ev.channel, []).append(ev)
    result: list[ResolvedAudioEvent] = []
    for ch_events in by_ch.values():
        muted = [e for e in ch_events if e.is_muted]
        live = sorted(
            (e for e in ch_events if not e.is_muted),
            key=lambda e: (e.start_sample, 0 if e.is_covered else 1, e.end_sample),
        )
        resolved: list[ResolvedAudioEvent] = []
        for curr in live:
            _insert_resolved_event(resolved, curr, session)
        result.extend(resolved)
        result.extend(muted)
    result.sort(key=lambda e: (e.channel, e.start_sample))
    return result


def resolve_audio_events(
    session: PtxSessionLike,
    track_or_name: PtxTrack | str,
    *,
    playlist: str | None = None,
    expand_compounds: bool = True,
    attach_fades: bool = True,
    trim_overlaps: bool = False,
    include_muted: bool = True,
) -> list[ResolvedAudioEvent]:
    """Join placements, clips, files, fades, clip gain, and compound clips for one track."""
    track = _resolve_track(session, track_or_name)
    if track is None:
        return []
    pl_name, placements, fade_placements = _select_playlist_data(session, track, playlist)
    clips_by_idx = {c.index: c for c in session.clips}
    comps_by_idx: dict[int, PtxCompoundClip] = {cp.index: cp for cp in session.compound_clips}
    af_by_idx = {af.index: af for af in session.audio_files}
    cg_by_idx = {cg.index: cg for cg in session.clip_gain_curves}
    events: list[ResolvedAudioEvent] = []
    for plc in placements:
        if plc.is_compound and expand_compounds and plc.clip_index in comps_by_idx:
            comp = comps_by_idx[plc.clip_index]
            loops = max(1, plc.loop_count or 1)
            stride = comp.length_samples or plc.length_samples
            for loop_i in range(loops):
                for mem in comp.members:
                    ev_muted = plc.is_muted or mem.is_muted
                    if not include_muted and ev_muted:
                        continue
                    pos = max(0, plc.start_sample + loop_i * stride + (mem.start_sample - comp.offset_samples))
                    events.append(
                        _build_audio_event(
                            session, track.name, pl_name, plc.channel, pos, clips_by_idx.get(mem.clip_index),
                            audio_files_by_idx=af_by_idx, curves_by_idx=cg_by_idx,
                            fallback_name=f"clip_{mem.clip_index}", fallback_len=0,
                            is_muted=ev_muted, is_covered=plc.is_covered or mem.is_covered,
                            loop_count=None, compound_group_name=comp.name,
                        )
                    )
            continue
        if not include_muted and plc.is_muted:
            continue
        clip_obj = None if plc.is_compound else clips_by_idx.get(plc.clip_index)
        events.append(
            _build_audio_event(
                session, track.name, pl_name, plc.channel, plc.start_sample, clip_obj,
                audio_files_by_idx=af_by_idx, curves_by_idx=cg_by_idx,
                fallback_name=plc.clip_name, fallback_len=plc.length_samples,
                is_muted=plc.is_muted, is_covered=plc.is_covered,
                loop_count=plc.loop_count, compound_group_name=plc.clip_name if plc.is_compound else None,
            )
        )
    if attach_fades:
        events = _attach_channel_fades(events, fade_placements, {f.index: f for f in session.fades})
    if trim_overlaps:
        events = _trim_channel_overlaps(events, session)
    events.sort(key=lambda e: (e.channel, e.start_sample))
    return events


def resolve_midi_events(
    session: PtxSessionLike,
    track_or_name: PtxTrack | str,
    *,
    absolute_ticks: bool = True,
    include_muted: bool = True,
) -> list[ResolvedMidiEvent]:
    """Join MIDI placements, MIDI clips, and MIDI chunks with clipped notes and controllers."""
    track = _resolve_track(session, track_or_name)
    if track is None:
        return []
    clips_by_idx = {c.index: c for c in session.midi_clips}
    chunks_by_idx = {c.index: c for c in session.midi_chunks}
    tl = session.timeline
    out: list[ResolvedMidiEvent] = []
    for mpl in track.midi_clips:
        if not include_muted and mpl.is_muted:
            continue
        mclip = clips_by_idx.get(mpl.clip_index)
        chunk = chunks_by_idx.get(mclip.chunk_index) if mclip else None
        rstart = mclip.start_tick if mclip else 0
        rlen = mclip.length_ticks if mclip else mpl.length_ticks
        rend = rstart + rlen
        base_shift = mpl.start_tick if absolute_ticks else 0
        notes: list[PtxMidiNote] = []
        lanes: list[PtxMidiControlLane] = []
        if chunk is not None:
            for nt in chunk.notes:
                if rstart <= nt.tick < rend:
                    dur = min(nt.duration_ticks, rend - nt.tick)
                    notes.append(PtxMidiNote(base_shift + (nt.tick - rstart), nt.pitch, dur, nt.velocity))
            for lane in chunk.control_lanes:
                evs = [
                    PtxMidiControlEvent(base_shift + (ev.tick - rstart), ev.value)
                    for ev in lane.events
                    if rstart <= ev.tick < rend
                ]
                if evs:
                    lanes.append(PtxMidiControlLane(lane.lane_type_code, lane.lane_type, lane.controller_number, evs))
        t0 = tl.ticks_to_seconds(mpl.start_tick)
        t1 = tl.ticks_to_seconds(mpl.start_tick + rlen) if rlen > 0 else t0
        out.append(
            ResolvedMidiEvent(
                track.name, mpl.clip_name, mpl.start_tick, rlen,
                round(t0, 6), round(max(0.0, t1 - t0), 6), mpl.is_muted,
                tuple(notes), tuple(lanes),
            )
        )
    out.sort(key=lambda e: e.start_tick)
    return out


def resolve_video_events(session: PtxSessionLike, track_or_name: PtxTrack | str) -> list[ResolvedVideoEvent]:
    """Join video placements and video clips for a video track."""
    track = _resolve_track(session, track_or_name)
    if track is None:
        return []
    clips_by_idx = {c.index: c for c in session.video_clips}
    tl = session.timeline
    fps = session.info.frame_rate
    out: list[ResolvedVideoEvent] = []
    for vpl in track.video_clips:
        vc = clips_by_idx.get(vpl.clip_index)
        off_f = vc.offset_frames if vc else 0
        len_f = vc.length_frames if vc else 0
        dur = round(len_f / fps, 6) if (fps and fps > 0 and len_f > 0) else None
        out.append(
            ResolvedVideoEvent(
                track.name, vpl.clip_name, vpl.start_sample,
                round(tl.samples_to_seconds(vpl.start_sample), 6),
                off_f, len_f, dur,
            )
        )
    out.sort(key=lambda e: e.start_sample)
    return out
