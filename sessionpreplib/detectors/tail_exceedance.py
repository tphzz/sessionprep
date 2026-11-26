from __future__ import annotations

import numpy as np

from ..models import ParamSpec
from ..detector import TrackDetector
from ..models import DetectorResult, IssueLocation, Severity, TrackContext
from ..audio import (
    get_rms_window_means,
    get_gated_rms_data,
    get_window_samples,
    is_silent,
    linear_to_db,
    format_duration,
    parse_rms_anchor_override,
    resolve_rms_anchor,
)


class TailExceedanceDetector(TrackDetector):
    id = "tail_exceedance"
    name = "Tail Regions Exceeded Anchor"
    shorthand = "TE"
    depends_on = ["silence"]

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="tail_min_exceed_db", type=(int, float), default=3.0,
                min=0.0, min_exclusive=True,
                label="Minimum tail exceedance (dB)",
                description="RMS windows exceeding the anchor by more than this many dB are flagged.",
            ),
            ParamSpec(
                key="tail_max_regions", type=int, default=20, min=1,
                label="Max tail exceedance regions",
                description="Maximum number of exceedance regions to report per track.",
            ),
            ParamSpec(
                key="tail_hop_ms", type=int, default=10, min=1,
                label="Tail hop size (ms)",
                description="Hop between RMS window evaluations for tail detection.",
            ),
        ]

    @classmethod
    def html_help(cls) -> str:
        return (
            "<b>Description</b><br/>"
            "Identifies regions where the RMS level significantly exceeds the "
            "percentile-based anchor RMS. Uses a hop-based scan to find "
            "contiguous loud sections."
            "<br/><br/>"
            "<b>Results</b><br/>"
            "<b>OK</b> – No tail exceedances found.<br/>"
            "<b>ATTENTION</b> – Regions found that exceed the anchor by more "
            "than the configured threshold (reported with count and dB values)."
            "<br/><br/>"
            "<b>Interpretation</b><br/>"
            "Tail exceedance regions are significantly louder than the track's "
            "typical level. This may indicate automation rides, unexpected "
            "volume spikes, or sections that need level matching. Consider "
            "section-based gain riding."
        )

    def is_relevant(self, result: DetectorResult, track: TrackContext | None = None) -> bool:
        """Tail exceedance is only meaningful when effective normalization is RMS-anchored."""
        if track is not None:
            if "bimodal_normalize" in (track.processor_skip or ()):
                return False
            if track.processor_results:
                pr = track.processor_results.get("bimodal_normalize")
                if pr is not None:
                    if "rms_anchor_active" in pr.data:
                        return bool(pr.data["rms_anchor_active"])
                    if pr.method and "RMS" not in pr.method:
                        return False
        return True

    def configure(self, config):
        super().configure(config)
        self.window_ms = config.get("window", 400)
        self.stereo_mode = config.get("stereo_mode", "avg")
        self.rms_anchor_mode = config.get("rms_anchor", "percentile")
        self.rms_percentile = config.get("rms_percentile", 95.0)
        self.gate_relative_db = config.get("gate_relative_db", 40.0)
        self.tail_min_exceed_db = config.get("tail_min_exceed_db", 3.0)
        self.tail_max_regions = config.get("tail_max_regions", 20)
        self.tail_hop_ms = config.get("tail_hop_ms", 10)

    def _non_applicable(self, summary: str, anchor_db: float = float(-np.inf)) -> DetectorResult:
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.CLEAN,
            summary=summary,
            applicable=False,
            data={
                "tail_regions": [],
                "tail_summary": {
                    "regions": 0,
                    "total_duration_sec": 0.0,
                    "max_exceed_db": 0.0,
                    "anchor_db": anchor_db,
                },
            },
        )

    def _scan_tail_regions(self, track: TrackContext, anchor_mean: float) -> list[dict]:
        window_means = get_rms_window_means(track, self.window_ms, self.stereo_mode)
        if window_means.size == 0:
            return []
        gated = get_gated_rms_data(
            track, self.window_ms, self.stereo_mode, self.gate_relative_db
        )
        active_mask = gated["active_mask"]
        window_samples = get_window_samples(track, self.window_ms)
        samplerate = track.samplerate
        floor = np.finfo(np.float64).tiny

        # Peak-preserving max-pooling per hop ensures no mid-hop RMS burst is missed.
        hop_samples = max(1, round(self.tail_hop_ms * samplerate / 1000))
        starts_all = np.arange(0, window_means.size, hop_samples, dtype=np.int64)
        active_means = np.where(active_mask, window_means, 0.0)
        pooled_means = np.maximum.reduceat(active_means, starts_all)
        hop_active = np.maximum.reduceat(active_mask.astype(np.uint8), starts_all).astype(bool)
        active_hop_indices = starts_all[hop_active]
        sampled_means = pooled_means[hop_active]
        if sampled_means.size == 0:
            return []

        exceed_db = 10.0 * np.log10(np.maximum(sampled_means, floor) / anchor_mean)
        idx = np.flatnonzero(exceed_db > float(self.tail_min_exceed_db))
        if idx.size == 0:
            return []

        splits = np.flatnonzero(np.diff(active_hop_indices[idx]) > hop_samples)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [idx.size - 1]))

        data_shape_0 = int(track.audio_data.shape[0]) if track.audio_data is not None else 0
        thresh_mean = anchor_mean * (10.0 ** (float(self.tail_min_exceed_db) / 10.0))
        regions = []
        for s_i, e_i in zip(starts, ends):
            i_start, i_end = int(idx[s_i]), int(idx[e_i])
            h_start = int(active_hop_indices[i_start])
            h_end = int(active_hop_indices[i_end])
            # Refine region bounds to exact first/last exceeding active windows in boundary hops.
            s_slice = active_means[h_start:min(h_start + hop_samples, window_means.size)]
            s_hits = np.flatnonzero(s_slice > thresh_mean)
            sample_start = h_start + (int(s_hits[0]) if s_hits.size else 0)
            e_slice = active_means[h_end:min(h_end + hop_samples, window_means.size)]
            e_hits = np.flatnonzero(e_slice > thresh_mean)
            last_win = h_end + (int(e_hits[-1]) if e_hits.size else 0)
            sample_end = min(last_win + window_samples, data_shape_0)
            regions.append({
                "start_sample": sample_start,
                "end_sample": sample_end,
                "start_time": format_duration(sample_start, samplerate),
                "end_time": format_duration(sample_end, samplerate),
                "max_exceed_db": float(np.max(exceed_db[i_start:i_end + 1])),
                "max_rms_db": linear_to_db(np.sqrt(float(np.max(sampled_means[i_start:i_end + 1])))),
            })

        regions.sort(key=lambda r: r["max_exceed_db"], reverse=True)
        regions = regions[:int(self.tail_max_regions)]
        regions.sort(key=lambda r: r["start_sample"])
        return regions

    def _build_tail_result(
        self,
        regions: list[dict],
        samplerate: int,
        rms_anchor_db: float,
        rms_anchor_label: str = "",
    ) -> DetectorResult:
        covered_samples = 0
        covered_end = 0
        for region in regions:
            start = max(covered_end, region["start_sample"])
            covered_samples += max(0, region["end_sample"] - start)
            covered_end = max(covered_end, region["end_sample"])
        max_exceed = max((r["max_exceed_db"] for r in regions), default=0.0)

        tail_summary = {
            "regions": len(regions),
            "total_duration_sec": covered_samples / samplerate,
            "max_exceed_db": float(max_exceed),
            "anchor_db": float(rms_anchor_db),
            "anchor_label": rms_anchor_label,
        }
        summary_text = (
            f"{len(regions)} tail region(s) exceed anchor "
            f"by >{self.tail_min_exceed_db:g} dB (max +{max_exceed:.1f} dB)"
        )
        issues = [
            IssueLocation(
                sample_start=r["start_sample"],
                sample_end=r["end_sample"] - 1,
                channel=None,
                severity=Severity.ATTENTION,
                label="tail_exceedance",
                description=f"tail exceeds anchor by +{r['max_exceed_db']:.1f} dB "
                            f"({r['start_time']}–{r['end_time']})",
            )
            for r in regions
        ]
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.ATTENTION,
            summary=summary_text,
            data={
                "tail_regions": regions,
                "tail_summary": tail_summary,
                "rms_anchor_label": rms_anchor_label,
            },
            hint="check for section-based riding",
            issues=issues,
        )

    def analyze(self, track: TrackContext) -> DetectorResult:
        if is_silent(track):
            return self._non_applicable("silent track")

        effective_mode, _ = parse_rms_anchor_override(
            getattr(track, "rms_anchor_override", None), self.rms_anchor_mode, self.rms_percentile,
        )
        if effective_mode != "percentile":
            return self._non_applicable("tail reporting only for percentile anchoring")

        anchor = resolve_rms_anchor(
            track,
            self.window_ms,
            self.stereo_mode,
            self.gate_relative_db,
            self.rms_anchor_mode,
            self.rms_percentile,
        )
        anchor_mean = anchor["rms_anchor_mean"]
        rms_anchor_db = anchor["rms_anchor_db"]
        rms_anchor_label = str(anchor.get("rms_anchor_label", ""))
        if anchor_mean <= 0:
            return self._non_applicable("anchor is zero", anchor_db=rms_anchor_db)

        regions = self._scan_tail_regions(track, anchor_mean)
        if not regions:
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.CLEAN,
                summary="no significant tail exceedances",
                data={
                    "tail_regions": [],
                    "rms_anchor_label": rms_anchor_label,
                    "tail_summary": {
                        "regions": 0,
                        "total_duration_sec": 0.0,
                        "max_exceed_db": 0.0,
                        "anchor_db": float(rms_anchor_db),
                        "anchor_label": rms_anchor_label,
                    },
                },
            )
        return self._build_tail_result(regions, track.samplerate, rms_anchor_db, rms_anchor_label)
