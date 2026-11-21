from __future__ import annotations

import numpy as np

from ..models import ParamSpec
from ..detector import TrackDetector
from ..models import DetectorResult, Severity, TrackContext
from ..audio import (
    get_peak_db,
    get_gated_rms_data,
    is_silent,
    linear_to_db,
)
from ..utils import matches_keywords


_PEAK_TIE_DB = 0.001
_DECAY_LIMIT_DB = 120.0


def _decay_envelope(power: np.ndarray, width: int) -> np.ndarray:
    """Mean power in every 10 ms window, including quiet tails after loud notes."""
    if len(power) <= width:
        return np.array([power.mean()])
    # A window straddles at most two width-sized blocks. Add their suffix
    # and prefix sums so a quiet tail never subtracts two large track totals.
    count = len(power) - width + 1
    envelope = np.empty(count)
    batch = width * max(1, 65536 // width)
    for start in range(0, count, batch):
        chunk = power[start:start + batch + width]
        padding = -len(chunk) % width
        blocks = (np.pad(chunk, (0, padding)) if padding else chunk).reshape(-1, width)
        sums = np.cumsum(blocks[:, ::-1], axis=1)[:, ::-1]
        sums[:-1, 1:] += np.cumsum(blocks[1:, :-1], axis=1)
        end = min(start + batch, count)
        envelope[start:end] = sums.ravel()[:end - start] / width
    return envelope


class AudioClassifierDetector(TrackDetector):
    id = "audio_classifier"
    name = "Audio Classifier"
    shorthand = "AC"
    depends_on = ["silence"]

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="crest_threshold", type=(int, float), default=12.0,
                min=0.0, min_exclusive=True,
                label="Crest factor threshold (dB)",
                description="Crest factor above this classifies a track as transient.",
            ),
            ParamSpec(
                key="decay_lookahead_ms", type=int, default=200,
                min=50, max=1000,
                label="Decay lookahead (ms)",
                description=(
                    "Time window after the loudest moment to measure energy "
                    "decay. Used to distinguish true transients (fast decay) "
                    "from sustained content with sharp attacks."
                ),
            ),
            ParamSpec(
                key="decay_db_threshold", type=(int, float), default=12.0,
                min=3.0, max=30.0,
                label="Decay threshold (dB)",
                description=(
                    "Energy drop within the lookahead window that confirms a "
                    "transient. Higher values require a sharper drop."
                ),
            ),
            ParamSpec(
                key="sparse_density_threshold", type=float, default=0.25,
                min=0.0, max=1.0,
                label="Sparse track density threshold",
                description=(
                    "A track below this fraction of active RMS windows is "
                    "classified as transient if its crest factor or decay "
                    "exceeds the corresponding threshold."
                ),
            ),
        ]

    @classmethod
    def html_help(cls) -> str:
        return (
            "<b>Description</b><br/>"
            "Classifies a track as Transient or Sustained using three "
            "metrics: crest factor (peak-to-RMS ratio), envelope decay "
            "rate (how fast energy drops after the loudest moment), and "
            "density (fraction of the track containing active content). "
            "Very sparse tracks are classified as Transient when either "
            "crest or decay exceeds its threshold. Otherwise, crest and decay vote together "
            "with decay acting as tiebreaker."
            "<br/><br/>"
            "<b>Results</b><br/>"
            "Classification metrics (crest, decay, density) are shown "
            "under Bimodal Normalization when report verbosity is set "
            "to verbose."
            "<br/><br/>"
            "<b>Interpretation</b><br/>"
            "Sparse track with high crest or fast decay → Transient.<br/>"
            "High crest + fast decay → drums, percussion (Transient).<br/>"
            "Low crest + slow decay → pads, bass, vocals (Sustained).<br/>"
            "High crest + slow decay → plucked/piano (Sustained).<br/>"
            "Low crest + fast decay → compressed drums (Transient).<br/>"
            "The classification determines which normalization "
            "strategy is applied during processing."
        )

    def is_relevant(self, result: DetectorResult, track: TrackContext | None = None) -> bool:
        return False

    def configure(self, config):
        super().configure(config)
        self.window_ms = config.get("window", 400)
        self.stereo_mode = config.get("stereo_mode", "avg")
        self.rms_anchor_mode = config.get("rms_anchor", "percentile")
        self.rms_percentile = config.get("rms_percentile", 95.0)
        self.gate_relative_db = config.get("gate_relative_db", 40.0)
        self.crest_threshold = config.get("crest_threshold", 12.0)
        self.decay_lookahead_ms = config.get("decay_lookahead_ms", 200)
        self.decay_db_threshold = config.get("decay_db_threshold", 12.0)
        self.sparse_density_threshold = config.get("sparse_density_threshold", 0.25)
        self.force_transient = config.get("force_transient", [])
        self.force_sustained = config.get("force_sustained", [])

    def _resolve_rms_anchor(
        self,
        track: TrackContext,
        active_means: np.ndarray,
        max_mean: float,
    ) -> tuple[float, float, str]:
        ov = getattr(track, "rms_anchor_override", None)
        if ov == "max":
            anchor_mode, percentile = "max", self.rms_percentile
        elif ov and ov.startswith("p"):
            anchor_mode, percentile = "percentile", float(ov[1:])
        else:
            anchor_mode, percentile = self.rms_anchor_mode, self.rms_percentile

        anchor_mean = (
            max_mean if anchor_mode == "max"
            else float(np.percentile(active_means, percentile))
        )
        rms_anchor_db = float(linear_to_db(np.sqrt(anchor_mean)))
        anchor_label = "max" if anchor_mode == "max" else f"p{percentile:g}"
        return anchor_mean, rms_anchor_db, anchor_label

    def _compute_decay(self, audio: np.ndarray, sr: int) -> tuple[float, bool]:
        decay_env_samples = max(1, int((10 / 1000.0) * sr))
        if audio.ndim > 1 and audio.shape[1] > 1:
            mono_sq = np.mean(audio.astype(np.float64) ** 2, axis=1)
        else:
            mono_sq = (audio.flatten().astype(np.float64)) ** 2

        env = _decay_envelope(mono_sq, decay_env_samples)
        peak_power = float(np.max(env))
        peak_env_idx = int(np.argmax(env >= peak_power * 10 ** (-_PEAK_TIE_DB / 10)))
        lookahead_env = max(1, int((self.decay_lookahead_ms / 1000.0) * sr))
        decay_end = min(len(env), peak_env_idx + 1 + lookahead_env)

        if decay_end > peak_env_idx + 1:
            tail_slice = env[peak_env_idx + 1:decay_end]
            quarter = max(1, len(tail_slice) // 4)
            tail_median = float(np.median(tail_slice[-quarter:]))
        else:
            tail_median = float(env[peak_env_idx])

        decay_db = 0.0
        if peak_power > 0:
            decay_db = (
                float(10 * np.log10(peak_power) - 10 * np.log10(tail_median))
                if tail_median > 0 else _DECAY_LIMIT_DB
            )
        decay_at_floor = decay_db >= _DECAY_LIMIT_DB
        return min(decay_db, _DECAY_LIMIT_DB), decay_at_floor

    def _classify_track(
        self,
        filename: str,
        crest: float,
        decay_db: float,
        density: float,
    ) -> tuple[str, bool, bool]:
        crest_says_transient = crest > self.crest_threshold
        decay_says_transient = decay_db > self.decay_db_threshold
        is_sparse = density < self.sparse_density_threshold

        if matches_keywords(filename, self.force_transient):
            is_transient, classification = True, "Transient (Forced)"
        elif matches_keywords(filename, self.force_sustained):
            is_transient, classification = False, "Sustained (Forced)"
        elif is_sparse and (crest_says_transient or decay_says_transient):
            is_transient, classification = True, "Transient"
        elif crest_says_transient and decay_says_transient:
            is_transient, classification = True, "Transient"
        elif not crest_says_transient and not decay_says_transient:
            is_transient, classification = False, "Sustained"
        elif crest_says_transient and not decay_says_transient:
            is_transient, classification = False, "Sustained"
        else:
            is_transient, classification = True, "Transient"

        forced = "Forced" in classification
        near_threshold = (
            not forced
            and np.isfinite(crest)
            and (
                abs(crest - self.crest_threshold) < 2.0
                or abs(decay_db - self.decay_db_threshold) < 2.0
            )
        )
        return classification, is_transient, near_threshold

    def analyze(self, track: TrackContext) -> DetectorResult:
        if is_silent(track):
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.CLEAN,
                summary="silent track",
                applicable=False,
                data={
                    "peak_db": float(-np.inf),
                    "rms_max_db": float(-np.inf),
                    "rms_anchor_db": float(-np.inf),
                    "rms_anchor_mean": 0.0,
                    "crest": 0.0,
                    "decay_db": 0.0,
                    "density": 0.0,
                    "classification": "Silent",
                    "is_transient": False,
                    "near_threshold": False,
                },
            )

        peak_db = get_peak_db(track)
        gated = get_gated_rms_data(
            track, self.window_ms, self.stereo_mode, self.gate_relative_db
        )
        active_means = gated["active_means"]
        window_means = gated["window_means"]

        max_mean = float(np.max(active_means)) if active_means.size else 0.0
        rms_max_db = float(linear_to_db(np.sqrt(max_mean)))
        anchor_mean, rms_anchor_db, anchor_label = self._resolve_rms_anchor(
            track, active_means, max_mean
        )
        crest = (
            float(peak_db - rms_max_db)
            if np.isfinite(peak_db) and np.isfinite(rms_max_db)
            else 0.0
        )
        decay_db, decay_at_floor = self._compute_decay(track.audio_data, track.samplerate)
        density = (
            float(len(active_means) / len(window_means))
            if len(window_means) > 0 else 1.0
        )
        classification, is_transient, near_threshold = self._classify_track(
            track.filename, crest, decay_db, density
        )
        summary = (
            f"crest {crest:.1f} dB, decay {'≥' if decay_at_floor else ''}{decay_db:.1f} dB, "
            f"density {density:.0%}, {classification}"
        )
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.INFO,
            summary=summary,
            data={
                "peak_db": float(peak_db),
                "rms_max_db": float(rms_max_db),
                "rms_anchor_db": float(rms_anchor_db),
                "rms_anchor_mean": float(anchor_mean),
                "rms_anchor_label": anchor_label,
                "crest": float(crest),
                "decay_db": float(decay_db),
                "decay_at_floor": decay_at_floor,
                "density": float(density),
                "classification": classification,
                "is_transient": bool(is_transient),
                "near_threshold": bool(near_threshold),
            },
        )
