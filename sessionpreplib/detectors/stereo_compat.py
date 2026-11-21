from __future__ import annotations

import math


from ..models import ParamSpec
from ..detector import TrackDetector, detector_badge_cell_html
from ..models import DetectorResult, IssueLocation, Severity, TrackContext
from ..audio import is_silent, windowed_stereo_correlation
from ..paths import DEFAULT_STEREO_WINDOW_MS


class StereoCompatDetector(TrackDetector):
    id = "stereo_compat"
    name = "Stereo Compatibility"
    shorthand = "SC"
    depends_on = ["silence"]

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="corr_warn", type=(int, float), default=-0.3,
                min=-1.0, max=1.0,
                label="Stereo correlation warning threshold",
                description=(
                    "Correlation below this value triggers a stereo-compatibility "
                    "warning. Values near +1 indicate highly correlated (mono-"
                    "compatible) stereo; values near 0 indicate wide stereo; "
                    "negative values indicate phase cancellation."
                ),
            ),
            ParamSpec(
                key="mono_loss_warn_db", type=(int, float), default=6.0,
                min=0.0, min_exclusive=True,
                label="Mono folddown loss warning (dB)",
                description=(
                    "Mono fold-down loss above this level triggers a warning. "
                    "This measures how much RMS level is lost when summing "
                    "stereo to mono via (L+R)/2."
                ),
            ),
            ParamSpec(
                key="corr_windowed", type=bool, default=True,
                label="Windowed analysis",
                description=(
                    "When enabled, stereo compatibility is analyzed per window "
                    "to localize regions with poor mono compatibility. The "
                    "whole-file summary is always computed regardless."
                ),
            ),
            ParamSpec(
                key="corr_window_ms", type=int, default=DEFAULT_STEREO_WINDOW_MS,
                min=100, max=5000,
                label="Analysis window (ms)",
                description=(
                    "Window length for windowed stereo analysis. 250 ms gives "
                    "roughly 16th-note resolution at moderate tempos — short "
                    "enough to catch reverb tails and widener artifacts, long "
                    "enough for stable correlation values."
                ),
            ),
            ParamSpec(
                key="corr_max_regions", type=int, default=20,
                min=1, max=200,
                label="Max reported regions",
                description="Maximum number of low-correlation regions to report per file.",
            ),
        ]

    @classmethod
    def html_help(cls) -> str:
        return (
            "<b>Description</b><br/>"
            "Analyzes stereo compatibility by computing the Pearson correlation "
            "coefficient between left and right channels and measuring mono "
            "fold-down loss ((L+R)/2 vs. stereo RMS)."
            "<br/><br/>"
            "<b>Correlation values</b><br/>"
            "<table style='margin:4px 0; font-size:8pt;'>"
            "<tr><td><b>+1.0</b></td><td style='padding-left:8px;'>"
            "Perfect positive correlation; channel levels may differ.</td></tr>"
            "<tr><td><b>+0.5 to +1.0</b></td><td style='padding-left:8px;'>"
            "Typical stereo \u2014 good mono compatibility</td></tr>"
            "<tr><td><b>0.0</b></td><td style='padding-left:8px;'>"
            "Uncorrelated \u2014 wide stereo, ~3 dB mono loss</td></tr>"
            "<tr><td><b>&lt; 0</b></td><td style='padding-left:8px;'>"
            "Out of phase \u2014 mono cancellation, level loss or full "
            "cancellation</td></tr>"
            "</table>"
            "<br/>"
            "<b>Mono folddown loss</b><br/>"
            "Measures how much RMS level is lost when summing L+R to mono. "
            "High loss means significant cancellation affecting phone speakers, "
            "mono PA systems, and broadcast."
            "<br/><br/>"
            "<b>Windowed analysis</b> (optional)<br/>"
            "When enabled, the signal is split into windows and both metrics "
            "are computed per window. Contiguous windows that exceed either "
            "threshold are merged into regions with precise sample ranges. "
            "This helps identify localized phase problems (a widener plugin, "
            "phase-inverted reverb tail, mid/side EQ) that a whole-file "
            "average might mask."
            "<br/><br/>"
            "<b>Results</b><br/>"
            "<b>OK</b> \u2013 Stereo compatibility is acceptable.<br/>"
            "<b>INFO</b> \u2013 Whole-file correlation or mono loss exceeds threshold.<br/>"
            "<b>ATTENTION</b> \u2013 Localized regions with poor mono compatibility."
            "<br/><br/>"
            "<b>Interpretation</b><br/>"
            "Review stereo widening, panning, or mid/side processing in the "
            "flagged regions. Consider checking the mix in mono to verify "
            "acceptability."
        )

    def configure(self, config):
        super().configure(config)
        self.corr_warn = config.get("corr_warn", -0.3)
        self.mono_loss_warn_db = config.get("mono_loss_warn_db", 6.0)
        self.windowed = config.get("corr_windowed", True)
        self.window_ms = config.get("corr_window_ms", DEFAULT_STEREO_WINDOW_MS)
        self.max_regions = config.get("corr_max_regions", 20)

    def _non_applicable(self, summary: str) -> DetectorResult:
        return DetectorResult(
            detector_id=self.id,
            severity=Severity.CLEAN,
            summary=summary,
            applicable=False,
            data={"lr_corr": None, "mono_loss_db": None,
                  "corr_warn": False, "mono_warn": False},
        )

    def _build_issue_result(
        self,
        track: TrackContext,
        severity: Severity,
        result_data: dict,
        whole_mono_loss: float,
        corr_threshold: float,
        mono_threshold: float,
        corr_warn: bool,
        mono_warn: bool,
        has_attention_regions: bool,
        issues: list[IssueLocation],
        detail_lines: list[str],
    ) -> DetectorResult:
        summary_parts: list[str] = []
        if corr_warn and result_data["lr_corr"] is not None:
            summary_parts.append(f"corr {result_data['lr_corr']:.2f} (< {corr_threshold:g})")
        if mono_warn and result_data["mono_loss_db"] is not None:
            summary_parts.append(
                f"mono loss {format_mono_loss(whole_mono_loss)} (> {mono_threshold:g} dB)"
            )
        if has_attention_regions and not summary_parts:
            mc = result_data.get("min_window_corr")
            if mc is not None:
                summary_parts.append(f"regions: min corr {mc:.2f}")

        summary = ", ".join(summary_parts) if summary_parts else "stereo compat issue"
        if not issues:
            issues.append(IssueLocation(
                sample_start=0,
                sample_end=track.total_samples - 1,
                channel=None,
                severity=severity,
                label=self.id,
                description=summary,
            ))

        hint_parts = []
        if corr_warn or has_attention_regions:
            hint_parts.append("review stereo widening / mid-side processing")
        if mono_warn:
            hint_parts.append("check mix in mono")

        return DetectorResult(
            detector_id=self.id,
            severity=severity,
            summary=summary,
            data=result_data,
            detail_lines=detail_lines,
            hint=", ".join(hint_parts) if hint_parts else None,
            issues=issues,
        )

    def analyze(self, track: TrackContext) -> DetectorResult:
        if is_silent(track):
            return self._non_applicable("silent track")

        data = track.audio_data
        if data is None or data.ndim < 2 or data.shape[1] < 2:
            return self._non_applicable("not stereo")
        if data.shape[1] > 2:
            return self._non_applicable("multichannel (>2ch), skipped")

        whole_corr, whole_mono_loss, win_results = windowed_stereo_correlation(
            data[:, 0], data[:, 1], track.samplerate,
            window_ms=int(self.window_ms), include_windows=self.windowed,
        )
        corr_threshold = float(self.corr_warn)
        mono_threshold = float(self.mono_loss_warn_db)
        corr_warn = not math.isnan(whole_corr) and whole_corr < corr_threshold
        mono_warn = whole_mono_loss > mono_threshold

        result_data: dict = {
            "lr_corr": whole_corr if not math.isnan(whole_corr) else None,
            "mono_loss_db": None if math.isnan(whole_mono_loss) else whole_mono_loss,
            "corr_warn": corr_warn,
            "mono_warn": mono_warn,
        }

        issues: list[IssueLocation] = []
        detail_lines: list[str] = []
        has_attention_regions = False
        if self.windowed and win_results:
            result_data.update(self._windowed_analysis(
                track, win_results, corr_threshold, mono_threshold,
                issues, detail_lines,
            ))
            has_attention_regions = bool(issues)

        if has_attention_regions:
            severity = Severity.ATTENTION
        elif corr_warn or mono_warn:
            severity = Severity.INFO
        else:
            return DetectorResult(
                detector_id=self.id,
                severity=Severity.CLEAN,
                summary="stereo compatibility OK",
                data=result_data,
            )

        return self._build_issue_result(
            track, severity, result_data, whole_mono_loss,
            corr_threshold, mono_threshold, corr_warn, mono_warn,
            has_attention_regions, issues, detail_lines,
        )

    # ------------------------------------------------------------------
    # Windowed analysis helper
    # ------------------------------------------------------------------

    def _windowed_analysis(  # pylint: disable=too-many-positional-arguments
        self,
        track: TrackContext,
        win_results: list[tuple[int, int, float, float]],
        corr_threshold: float,
        mono_threshold: float,
        issues: list[IssueLocation],
        detail_lines: list[str],
    ) -> dict:
        """Merge per-window results into contiguous regions.

        A window is flagged if its correlation is below *corr_threshold*
        **or** its mono loss exceeds *mono_threshold*.

        """
        max_reg = int(self.max_regions)

        # Primary: merge windows exceeding either threshold
        regions = self._merge_regions(
            win_results, corr_threshold, mono_threshold)

        if not regions:
            return {"windowed_regions": []}
        min_corr = min((r["min_corr"] for r in regions if r["min_corr"] is not None), default=None)
        max_loss = max(r["max_mono_loss_db"] for r in regions)
        # Complete cancellations come first. For equal correlations prefer
        # the greater mono loss. The display cap never changes the extrema.
        regions.sort(key=lambda r: (r["max_mono_loss_db"] != math.inf,
                                    r["min_corr"] if r["min_corr"] is not None else math.inf,
                                    -r["max_mono_loss_db"]))
        regions = regions[:max_reg]

        # Build issues and detail lines
        for reg in regions:
            corr_str = _format_correlation(reg["min_corr"])
            loss = reg["max_mono_loss_db"]
            if loss > 0.01:
                loss_str = f" / mono loss {format_mono_loss(loss)}"
            else:
                loss_str = ""

            t_start = reg["sample_start"] / track.samplerate
            t_end = reg["sample_end"] / track.samplerate
            time_str = f"{_fmt_time(t_start)}\u2013{_fmt_time(t_end)}"

            desc = f"{corr_str}{loss_str} ({time_str})"
            issues.append(IssueLocation(
                sample_start=reg["sample_start"],
                sample_end=reg["sample_end"],
                channel=None,
                severity=Severity.ATTENTION,
                label=self.id,
                description=desc,
            ))

        if regions:
            worst = regions[0]
            detail_lines.append(
                f"Windowed: {len(regions)} region(s) with poor stereo compat")
            loss = worst["max_mono_loss_db"]
            loss_part = f" / mono loss {format_mono_loss(loss)}"
            t_s = worst["sample_start"] / track.samplerate
            t_e = worst["sample_end"] / track.samplerate
            detail_lines.append(
                f"Worst: {_format_correlation(worst['min_corr'])}{loss_part} "
                f"at {_fmt_time(t_s)}\u2013{_fmt_time(t_e)}")

        return {"windowed_regions": regions, "min_window_corr": min_corr,
                "max_window_mono_loss_db": max_loss}

    @staticmethod
    def _merge_regions(
        win_results: list[tuple[int, int, float, float]],
        corr_threshold: float,
        mono_threshold: float,
    ) -> list[dict]:
        """Merge contiguous windows that exceed either threshold."""
        regions: list[dict] = []
        current: dict | None = None
        for s_start, s_end, corr, mono_loss in win_results:
            if corr < corr_threshold or mono_loss > mono_threshold:
                if current is None:
                    current = {
                        "sample_start": s_start,
                        "sample_end": s_end,
                        "min_corr": None if math.isnan(corr) else corr,
                        "max_mono_loss_db": mono_loss,
                    }
                else:
                    current["sample_end"] = s_end
                    if not math.isnan(corr) and (current["min_corr"] is None or corr < current["min_corr"]):
                        current["min_corr"] = corr
                    current["max_mono_loss_db"] = max(current["max_mono_loss_db"], mono_loss)
            else:
                if current is not None:
                    regions.append(current)
                    current = None
        if current is not None:
            regions.append(current)
        return regions

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def render_html(self, result: DetectorResult,
                    track: TrackContext | None = None) -> str:
        eff = self.effective_severity(result)
        if eff is None:
            return ""
        if not self.is_relevant(result, track):
            return ""

        sev = eff.value
        sev_color, sev_label = {
            "problem":     ("#ff4444", "PROBLEM"),
            "attention":   ("#ffaa00", "ATTENTION"),
            "information": ("#4499ff", "INFO"),
            "info":        ("#4499ff", "INFO"),
            "clean":       ("#44cc44", "OK"),
        }.get(sev, ("#4499ff", "INFO"))

        d = result.data or {}
        parts: list[str] = []

        lr_corr = d.get("lr_corr")
        if lr_corr is not None:
            parts.append(f"corr {lr_corr:.2f}")

        mono_loss = d.get("mono_loss_db")
        if mono_loss is not None and not math.isnan(mono_loss):
            parts.append(f"mono loss {format_mono_loss(mono_loss)}")

        regions = d.get("windowed_regions", [])
        if regions:
            parts.append(f"{len(regions)} region(s)")

        summary = " \u00b7 ".join(parts) if parts else str(result.summary)

        return (
            f'<tr>'
            f'{detector_badge_cell_html(sev_color, sev_label)}'
            f'<td style="padding-left:6px; white-space:nowrap;">'
            f'<a href="detector:{self.id}" style="color:#dddddd;'
            f' text-decoration:none;"><b>{self.id}</b></a></td>'
            f'<td style="padding-left:6px; color:#888888;">'
            f'{summary}</td>'
            f'</tr>'
        )

    def clean_message(self) -> str | None:
        return "No stereo compatibility issues detected"


def format_mono_loss(value: float) -> str:
    return "∞ dB" if value == math.inf else f"{value:.1f} dB"


def _format_correlation(value: float | None) -> str:
    return "correlation unavailable" if value is None else f"corr {value:.2f}"


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS.s"""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"
