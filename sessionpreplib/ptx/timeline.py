"""Timeline, tempo/meter integration, musical bar/beat, and timecode conversion API for ``.ptx``."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .codec import format_timecode
from .models import TICKS_PER_QUARTER, PtxMarker, PtxMeterEvent, PtxSessionLike


@dataclass(frozen=True)
class PtxBarBeat:
    """Musical bar, beat, and remainder tick position (1-based bar and beat)."""

    bar: int
    beat: int
    ticks: int


class PtxTimeline:
    """Unified time-domain converter and ruler query interface for a :class:`PtxSession`."""

    def __init__(self, session: PtxSessionLike):
        self._session = session

    @property
    def sample_rate(self) -> int:
        return self._session.info.sample_rate or 0

    @property
    def default_bpm(self) -> float:
        if self._session.tempo_events:
            bpm = self._session.tempo_events[0].bpm
            if math.isfinite(bpm) and bpm > 0:
                return bpm
        info_bpm = self._session.info.tempo_bpm
        return info_bpm if (info_bpm is not None and math.isfinite(info_bpm) and info_bpm > 0) else 120.0

    def samples_to_seconds(self, samples: int) -> float:
        sr = self.sample_rate
        return (samples / float(sr)) if sr > 0 else 0.0

    def seconds_to_samples(self, seconds: float) -> int:
        if not math.isfinite(seconds):
            raise ValueError(f"seconds must be finite, got {seconds!r}")
        sr = self.sample_rate
        return round(seconds * sr) if sr > 0 else 0

    def _tempo_segments(self) -> list[tuple[int, float, float]]:
        raw = sorted(
            (ev for ev in self._session.tempo_events if math.isfinite(ev.bpm) and ev.bpm > 0),
            key=lambda e: e.tick,
        )
        if not raw:
            return [(0, self.default_bpm, 0.0)]
        segs: list[tuple[int, float, float]] = []
        if raw[0].tick > 0:
            segs.append((0, raw[0].bpm, 0.0))
        else:
            segs.append((raw[0].tick, raw[0].bpm, 0.0))
        start_idx = 0 if raw[0].tick > 0 else 1
        for ev in raw[start_idx:]:
            prev_t, prev_bpm, prev_s = segs[-1]
            dt_ticks = max(0, ev.tick - prev_t)
            cum_s = prev_s + dt_ticks / float(TICKS_PER_QUARTER) * 60.0 / prev_bpm
            segs.append((ev.tick, ev.bpm, cum_s))
        return segs

    def tempo_at_tick(self, tick: int) -> float:
        segs = self._tempo_segments()
        active = segs[0][1]
        for seg_tick, seg_bpm, _seg_s in segs:
            if seg_tick <= tick:
                active = seg_bpm
            else:
                break
        return active

    def meter_at_tick(self, tick: int) -> tuple[int, int]:
        meters = self._normalized_meters()
        active = (meters[0].numerator, meters[0].denominator)
        for ev in meters:
            if ev.tick <= tick:
                active = (ev.numerator, ev.denominator)
            else:
                break
        return active

    def ticks_to_seconds(self, tick: int) -> float:
        segs = self._tempo_segments()
        active_t, active_bpm, active_s = segs[0]
        if tick < active_t:
            return active_s + (tick - active_t) / float(TICKS_PER_QUARTER) * 60.0 / active_bpm
        for seg_t, seg_bpm, seg_s in segs:
            if seg_t <= tick:
                active_t, active_bpm, active_s = seg_t, seg_bpm, seg_s
            else:
                break
        return active_s + (tick - active_t) / float(TICKS_PER_QUARTER) * 60.0 / active_bpm

    def seconds_to_ticks(self, seconds: float) -> int:
        if not math.isfinite(seconds):
            raise ValueError(f"seconds must be finite, got {seconds!r}")
        segs = self._tempo_segments()
        active_t, active_bpm, active_s = segs[0]
        if seconds < active_s:
            return active_t + round((seconds - active_s) * active_bpm / 60.0 * TICKS_PER_QUARTER)
        for seg_t, seg_bpm, seg_s in segs:
            if seg_s <= seconds:
                active_t, active_bpm, active_s = seg_t, seg_bpm, seg_s
            else:
                break
        return active_t + round((seconds - active_s) * active_bpm / 60.0 * TICKS_PER_QUARTER)

    def ticks_to_samples(self, tick: int) -> int:
        return self.seconds_to_samples(self.ticks_to_seconds(tick))

    def samples_to_ticks(self, samples: int) -> int:
        return self.seconds_to_ticks(self.samples_to_seconds(samples))

    def seconds_to_timecode(self, seconds: float, *, include_session_start: bool = True) -> str:
        if not math.isfinite(seconds):
            return ""
        info = self._session.info
        if info.fps_code is None or not info.frame_rate or info.frame_rate <= 0:
            return ""
        base_frames = (info.start_frames or 0) if include_session_start else 0
        total_frames = base_frames + round(seconds * info.frame_rate)
        if total_frames < 0:
            return ""
        return format_timecode(total_frames, info.fps_code)

    def samples_to_timecode(self, samples: int, *, include_session_start: bool = True) -> str:
        return self.seconds_to_timecode(
            self.samples_to_seconds(samples), include_session_start=include_session_start,
        )

    def _normalized_meters(self) -> list[PtxMeterEvent]:
        meters = sorted(self._session.meter_events, key=lambda m: m.tick)
        if not meters or meters[0].tick > 0:
            return [PtxMeterEvent(tick=0, numerator=4, denominator=4), *meters]
        return meters

    @staticmethod
    def _bar_and_beat_ticks(numerator: int, denominator: int) -> tuple[int, int]:
        den = max(1, denominator)
        beat_ticks = round(TICKS_PER_QUARTER * 4 / den)
        bar_ticks = max(1, beat_ticks * max(1, numerator))
        return bar_ticks, max(1, beat_ticks)

    def ticks_to_bar_beat(self, tick: int) -> PtxBarBeat:
        target = max(0, int(tick))
        meters = self._normalized_meters()
        cur_bar = 1
        for idx, cur in enumerate(meters):
            bar_ticks, beat_ticks = self._bar_and_beat_ticks(cur.numerator, cur.denominator)
            next_tick = meters[idx + 1].tick if idx + 1 < len(meters) else None
            if next_tick is None or target < next_tick:
                delta = target - cur.tick
                bars_elapsed, rem_bar = divmod(delta, bar_ticks)
                beat_idx, rem_tick = divmod(rem_bar, beat_ticks)
                return PtxBarBeat(bar=cur_bar + bars_elapsed, beat=beat_idx + 1, ticks=rem_tick)
            seg_ticks = max(0, next_tick - cur.tick)
            cur_bar += (seg_ticks + bar_ticks - 1) // bar_ticks
        return PtxBarBeat(bar=1, beat=1, ticks=0)

    def bar_beat_to_ticks(self, bar: int, beat: int = 1, ticks: int = 0) -> int:
        if bar < 1 or beat < 1 or ticks < 0:
            raise ValueError(f"Invalid musical position bar={bar}, beat={beat}, ticks={ticks}")
        meters = self._normalized_meters()
        cur_bar = 1
        for idx, cur in enumerate(meters):
            bar_ticks, beat_ticks = self._bar_and_beat_ticks(cur.numerator, cur.denominator)
            next_tick = meters[idx + 1].tick if idx + 1 < len(meters) else None
            if next_tick is not None:
                seg_bars = (max(0, next_tick - cur.tick) + bar_ticks - 1) // bar_ticks
                if bar >= cur_bar + seg_bars:
                    cur_bar += seg_bars
                    continue
            return cur.tick + (bar - cur_bar) * bar_ticks + (beat - 1) * beat_ticks + ticks
        return ticks

    def markers_in_range(self, start_sec: float, end_sec: float) -> list[PtxMarker]:
        out: list[PtxMarker] = []
        for marker in self._session.markers:
            m_sec = (
                marker.start_seconds
                if marker.start_seconds is not None
                else self.samples_to_seconds(marker.start_sample)
            )
            if start_sec <= m_sec <= end_sec:
                out.append(marker)
        return out

    def marker_by_number(self, number: int) -> PtxMarker | None:
        return next((m for m in self._session.markers if m.number == number), None)

    def marker_by_name(self, name: str) -> PtxMarker | None:
        exact = next((m for m in self._session.markers if m.name == name), None)
        if exact is not None:
            return exact
        low = name.lower()
        return next((m for m in self._session.markers if m.name.lower() == low), None)

    def session_duration_seconds(self) -> float:
        max_sec = 0.0
        for pl in self._session.playlists:
            for plc in pl.placements:
                eff_len = plc.length_samples * plc.loop_count if (plc.loop_count and plc.loop_count > 1) else plc.length_samples
                end_smp = max(plc.end_sample, plc.start_sample + eff_len)
                max_sec = max(max_sec, self.samples_to_seconds(end_smp))
        for midi_list in self._session.midi_playlists.values():
            for mpl in midi_list:
                end_t = self.ticks_to_seconds(mpl.start_tick + mpl.length_ticks)
                max_sec = max(max_sec, end_t)
        fps = self._session.info.frame_rate
        vid_by_idx = {vc.index: vc for vc in self._session.video_clips}
        for vid_list in self._session.video_playlists.values():
            for vpl in vid_list:
                v_start = self.samples_to_seconds(vpl.start_sample)
                vc = vid_by_idx.get(vpl.clip_index)
                v_dur = (vc.length_frames / fps) if (vc and fps and fps > 0 and vc.length_frames > 0) else 0.0
                max_sec = max(max_sec, v_start + v_dur)
        for marker in self._session.markers:
            end_s = max(marker.start_sample, marker.end_sample)
            max_sec = max(max_sec, self.samples_to_seconds(end_s))
        return round(max_sec, 6)
