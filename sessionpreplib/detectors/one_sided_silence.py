from __future__ import annotations

import numpy as np

from ..models import ParamSpec, LifecyclePhase
from ..detector import TrackDetector
from ..models import DetectorResult, IssueLocation, Severity, TrackContext
from ..audio import dbfs_offset, is_silent, linear_to_db, stereo_layout_levels


class OneSidedSilenceDetector(TrackDetector):
    id = "one_sided_silence"
    name = "One-Sided Silence"
    shorthand = "OS"
    phase = LifecyclePhase.PHASE1
    depends_on = ["silence"]

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="one_sided_silence_db", type=(int, float), default=-80.0,
                max=0.0,
                label="One-sided silence threshold (dB)",
                description="RMS below this level on one channel triggers a warning.",
            ),
        ]

    @classmethod
    def html_help(cls) -> str:
        return (
            "<b>Description</b><br/>"
            "Detects stereo files where one channel is silent while the other "
            "contains audio content."
            "<br/><br/>"
            "<b>Results</b><br/>"
            "<b>OK</b> – Both channels have audio content.<br/>"
            "<b>ATTENTION</b> – One channel is silent (reported with side and "
            "per-channel RMS levels)."
            "<br/><br/>"
            "<b>Interpretation</b><br/>"
            "One-sided silence usually indicates a mono source exported as "
            "stereo with incorrect channel routing, or a missing channel. "
            "Check the stereo export settings and channel routing in the DAW."
        )

    def configure(self, config):
        super().configure(config)
        self.threshold_db = config.get("one_sided_silence_db", -80.0)
        self._db_offset = dbfs_offset(config)

    def _non_applicable(self, summary: str) -> DetectorResult:
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.CLEAN,
            summary=summary,
            applicable=False,
            data={
                "one_sided_silence": False,
                "one_sided_silence_side": None,
                "l_rms_db": float(-np.inf),
                "r_rms_db": float(-np.inf),
            },
        )

    def _build_one_sided_result(
        self,
        track: TrackContext,
        side: str | None,
        l_rms_db: float,
        r_rms_db: float,
        data: dict,
    ) -> DetectorResult:
        data["topology_action"] = "extract_channel"
        data["topology_channel"] = 1 if side == "L" else 0
        off = self._db_offset

        def fmt_db(x: float) -> str:
            return f"{float(x) + off:.1f}" if np.isfinite(x) else "-inf"

        side_prefix = f"one-sided silence ({side})" if side else "one-sided silence"
        summary = f"{side_prefix} (L {fmt_db(l_rms_db)} dBFS, R {fmt_db(r_rms_db)} dBFS)"
        ch_idx = 0 if side == "L" else 1 if side == "R" else None
        issues = [IssueLocation(
            sample_start=0,
            sample_end=track.total_samples - 1,
            channel=ch_idx,
            severity=Severity.ATTENTION,
            label="one_sided_silence",
            description=f"channel {side} is silent",
        )]
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.ATTENTION,
            summary=summary,
            data=data,
            hint="check stereo export / channel routing",
            issues=issues,
        )

    def analyze(self, track: TrackContext) -> DetectorResult:
        if is_silent(track):
            return self._non_applicable("silent track")

        levels = stereo_layout_levels(track)
        if levels is None:
            return self._non_applicable("requires a stereo track")

        _, l_rms_lin, r_rms_lin = levels
        l_rms_db, r_rms_db = linear_to_db(l_rms_lin), linear_to_db(r_rms_lin)
        silence_lin = float(10.0 ** (self.threshold_db / 20.0))

        one_sided = False
        side = None
        if l_rms_lin <= silence_lin < r_rms_lin:
            one_sided, side = True, "L"
        elif r_rms_lin <= silence_lin < l_rms_lin:
            one_sided, side = True, "R"

        data = {
            "one_sided_silence": bool(one_sided),
            "one_sided_silence_side": side,
            "l_rms_db": l_rms_db,
            "r_rms_db": r_rms_db,
        }
        if one_sided:
            return self._build_one_sided_result(track, side, l_rms_db, r_rms_db, data)

        return DetectorResult(
            detector_id=self.id,
            severity=Severity.CLEAN,
            summary="no one-sided silence",
            data=data,
        )

    def clean_message(self) -> str | None:
        return "No one-sided silent stereo files detected"
