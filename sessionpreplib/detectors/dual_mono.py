from __future__ import annotations

from ..models import ParamSpec, LifecyclePhase
from ..detector import TrackDetector
from ..models import DetectorResult, Severity, TrackContext
from ..audio import is_silent, stereo_layout_levels


class DualMonoDetector(TrackDetector):
    id = "dual_mono"
    name = "Dual-Mono (Identical L/R)"
    shorthand = "DM"
    phase = LifecyclePhase.PHASE1
    depends_on = ["silence"]

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="dual_mono_eps", type=(int, float), default=1e-5,
                min=0.0, min_exclusive=True,
                label="Dual-mono epsilon",
                description="Max sample difference to consider L/R identical.",
            ),
        ]

    @classmethod
    def html_help(cls) -> str:
        return (
            "<b>Description</b><br/>"
            "Checks whether the left and right channels of a stereo file are "
            "identical (dual-mono)."
            "<br/><br/>"
            "<b>Results</b><br/>"
            "<b>OK</b> – Channels differ (true stereo or mono file).<br/>"
            "<b>INFO</b> – Left and right are identical."
            "<br/><br/>"
            "<b>Interpretation</b><br/>"
            "A dual-mono file carries the same content on both channels, "
            "wasting storage and bandwidth. It may be intentional (mono source "
            "panned center) or indicate a routing error. Consider exporting as "
            "a mono file."
        )

    def configure(self, config):
        super().configure(config)
        self.eps = config.get("dual_mono_eps", 1e-5)

    def analyze(self, track: TrackContext) -> DetectorResult:
        if is_silent(track):
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.CLEAN,
                summary="silent track",
                applicable=False,
                data={"dual_mono": False},
            )

        levels = stereo_layout_levels(track)
        if levels is None:
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.CLEAN,
                summary="requires a stereo track",
                applicable=False,
                data={"dual_mono": False},
            )

        max_difference, _, _ = levels
        if max_difference <= self.eps:
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.INFO,
                summary="dual-mono (identical L/R)",
                data={
                    "dual_mono": True,
                    "topology_action": "extract_channel",
                    "topology_channel": 0,
                },
            )

        return DetectorResult(
            detector_id=self.id,
            severity=Severity.CLEAN,
            summary="not dual-mono",
            data={"dual_mono": False},
        )
