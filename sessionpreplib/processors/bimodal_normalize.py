from __future__ import annotations

import math

import numpy as np

from ..models import ParamSpec
from ..processor import AudioProcessor, PRIORITY_NORMALIZE
from ..models import ProcessorResult, TrackContext
from ..audio import db_to_linear, dbfs_offset


class BimodalNormalizeProcessor(AudioProcessor):
    id = "bimodal_normalize"
    name = "Bimodal Normalization"
    shorthand = "BN"
    priority = PRIORITY_NORMALIZE

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        return super().config_params() + [
            ParamSpec(
                key="target_rms", type=(int, float), default=-18.0,
                min=-80.0, max=0.0,
                label="Target RMS (dBFS)",
                description="Sustained tracks are RMS-normalized to this level.",
            ),
            ParamSpec(
                key="target_peak", type=(int, float), default=-6.0,
                min=-80.0, max=0.0,
                label="Target peak (dBFS)",
                description="Transient tracks are peak-normalized to this level.",
            ),
        ]

    def configure(self, config):
        super().configure(config)
        self.target_rms = config.get("target_rms", -18.0)
        self.target_peak = config.get("target_peak", -6.0)
        self._db_offset = dbfs_offset(config)
        anchor_mode = config.get("rms_anchor", "percentile")
        if anchor_mode == "max":
            self._rms_anchor_label = "max"
        else:
            pct = config.get("rms_percentile", 95.0)
            self._rms_anchor_label = f"p{pct:g}"

    def process(self, track: TrackContext) -> ProcessorResult:
        # Read from audio_classifier detector
        crest_result = track.detector_results.get("audio_classifier")
        silence_result = track.detector_results.get("silence")

        if silence_result and silence_result.data.get("is_silent"):
            return ProcessorResult(
                processor_id=self.id,
                gain_db=0.0,
                classification="Silent",
                method="None",
                data={"gain_db_individual": 0.0},
            )

        if crest_result is None:
            return ProcessorResult(
                processor_id=self.id,
                gain_db=0.0,
                classification="Unknown",
                method="None",
                data={"gain_db_individual": 0.0},
                error="audio_classifier detector result missing",
            )

        # Check for user override from the GUI dropdown
        override = track.classification_override
        if override == "Skip":
            return ProcessorResult(
                processor_id=self.id,
                gain_db=0.0,
                classification="Skip",
                method="None",
                data={"gain_db_individual": 0.0},
            )

        peak_db = crest_result.data["peak_db"]
        rms_anchor_db = crest_result.data["rms_anchor_db"]

        if override in ("Transient", "Sustained"):
            classification = override
            is_transient = override == "Transient"
        else:
            classification = crest_result.data["classification"]
            is_transient = crest_result.data["is_transient"]

        # Compute both gain paths for transparency
        gain_for_peak = self.target_peak - peak_db
        gain_for_rms = self.target_rms - rms_anchor_db

        if is_transient:
            gain = gain_for_peak
            method = f"Peak → {self.target_peak:.0f} dB"
        else:
            gain = min(gain_for_rms, gain_for_peak)
            if gain == gain_for_rms:
                method = f"RMS → {self.target_rms:.0f} dB"
            else:
                method = "Peak Limited"

        # Use per-track anchor label from detector if available
        anchor_label = crest_result.data.get(
            "rms_anchor_label", self._rms_anchor_label)

        return ProcessorResult(
            processor_id=self.id,
            gain_db=float(gain),
            classification=classification,
            method=method,
            data={
                "gain_db_individual": float(gain),
                "target_peak": self.target_peak,
                "target_rms": self.target_rms,
                "detected_peak_db": peak_db,
                "detected_rms_db": rms_anchor_db,
                "rms_anchor_label": anchor_label,
                "gain_for_peak": float(gain_for_peak),
                "gain_for_rms": float(gain_for_rms),
            },
        )

    def render_html(self, result: ProcessorResult, track=None, *, verbose: bool = False,
                    compact: bool = False, narrow: bool = False) -> str:
        """Render the normalization summary line and comparison table.

        The table follows the report's finding-table style. The chosen or
        limiting measure is named in text (not by colour alone). ``compact``
        renders percentage columns with the note under the measure name;
        ``narrow`` stacks one entry per measure under a legend band.
        """
        d = result.data
        cls_text = result.classification or "Unknown"
        surface, separator = "#2d2d2d", "#3a3a3a"
        text, metadata, muted = "#dddddd", "#aaaaaa", "#888888"

        # Classification color
        if "Transient" in cls_text:
            type_color = "#cc77ff"
        elif "Sustained" in cls_text:
            type_color = "#44cccc"
        else:
            type_color = muted

        # Skip / Silent / Unknown: single-line, no breakdown
        if cls_text in ("Skip", "Silent", "Unknown"):
            return (
                f'<div><span style="color:{type_color}; font-weight:bold;">{cls_text}</span>'
                f' &nbsp;&middot;&nbsp; <span style="color:{metadata};">No normalization</span>'
                f'</div>'
            )

        off = self._db_offset

        def fmt_inf(val):
            return "+&infin;" if val > 0 else "&minus;&infin;"

        def minus(text):
            return "&minus;" + text[1:] if text.startswith("-") else text

        def fmt_abs(val):
            if not math.isfinite(val):
                return fmt_inf(val)
            return minus(f"{val + off:.1f}")

        def fmt_rel(val):
            if not math.isfinite(val):
                return fmt_inf(val)
            return minus(f"{val:+.1f}")

        det_peak = d.get("detected_peak_db", float("-inf"))
        det_rms = d.get("detected_rms_db", float("-inf"))
        tgt_peak = d.get("target_peak", self.target_peak)
        tgt_rms = d.get("target_rms", self.target_rms)
        anchor_label = d.get("rms_anchor_label", self._rms_anchor_label)
        gain_for_peak = d.get("gain_for_peak", 0.0)
        gain_for_rms = d.get("gain_for_rms", 0.0)

        rms_metric = f"RMS ({anchor_label})" if anchor_label else "RMS"
        is_transient = "Transient" in cls_text
        is_peak_limited = result.method == "Peak Limited"

        peak_active = is_transient or is_peak_limited
        rms_active = not is_transient and not is_peak_limited

        if is_peak_limited:
            method = f"Peak (limited) &rarr; {minus(f'{tgt_peak:.1f}')} dBFS"
        elif is_transient:
            method = f"Peak &rarr; {minus(f'{tgt_peak:.1f}')} dBFS"
        else:
            method = f"RMS &rarr; {minus(f'{tgt_rms:.1f}')} dBFS"

        # --- Summary line ---
        html = (
            f'<div><span style="color:{type_color}; font-weight:bold;">{cls_text}</span>'
            f' &nbsp;&middot;&nbsp; {method}'
            f' &nbsp;&middot;&nbsp; <b>{minus(f"{result.gain_db:+.1f}")} dB</b></div>'
        )

        def note(chosen, label):
            if not label:
                return ""
            style = (f"color:{type_color}; font-weight:600;" if chosen
                     else f"color:{metadata};")
            return f'<span style="{style}">{label}</span>'

        pk_note = note(True, "Chosen &middot; limiting" if is_peak_limited
                       else "Chosen" if is_transient else "")
        rms_note = (note(False, "Would exceed peak") if is_peak_limited
                    else note(True, "Chosen") if rms_active else "")

        # --- Comparison table (finding-table style) ---
        head = (f'padding:4px 12px; background-color:{surface}; color:{text};'
                f' border-bottom:1px solid {separator};')
        cell = f'padding:4px 12px; border-bottom:1px solid {separator};'

        def value_cells(values, active, width_attrs):
            weight = " font-weight:bold;" if active else ""
            return "".join(
                f'<td{width} style="{cell} white-space:nowrap; color:{text};{weight}">{value}</td>'
                for value, width in zip(values, width_attrs))

        peak_values = (f"{fmt_abs(det_peak)} dBFS", f"{minus(f'{tgt_peak:.1f}')} dBFS",
                       f"{fmt_rel(gain_for_peak)} dB")
        rms_values = (f"{fmt_abs(det_rms)} dBFS", f"{minus(f'{tgt_rms:.1f}')} dBFS",
                      f"{fmt_rel(gain_for_rms)} dB")
        if narrow:
            header = (f'<td style="{head}">Measure<br>'
                      f'Detected &rarr; Target &middot; Gain</td>')

            def row(metric, values, active, row_note):
                weight = " font-weight:bold;" if active else ""
                det, tgt, gain = (f"<nobr>{value}</nobr>" for value in values)
                return (f'<tr><td style="{cell}"><span style="color:{metadata};">{metric}</span>'
                        + (f"<br>{row_note}" if row_note else "")
                        + f'<br><span style="color:{text};{weight}">'
                        f'{det} &rarr; {tgt} &middot; {gain}</span></td></tr>')
            table_attrs = 'width="100%" '
        elif compact:
            widths = (' width="30%"', ' width="24%"', ' width="24%"', ' width="22%"')
            header = "".join(f'<td{w} style="{head}">{label}</td>' for label, w in
                             zip(("Measure", "Detected", "Target", "Gain"), widths))

            def row(metric, values, active, row_note):
                measure = metric + (f"<br>{row_note}" if row_note else "")
                return (f'<tr><td{widths[0]} style="{cell} color:{metadata};">{measure}</td>'
                        + value_cells(values, active, widths[1:]) + '</tr>')
            table_attrs = 'width="100%" '
        else:
            widths = (' width="112"', ' width="104"', ' width="104"', ' width="88"', ' width="150"')
            header = "".join(f'<td{w} style="{head}">{label}</td>' for label, w in
                             zip(("Measure", "Detected", "Target", "Gain", "&nbsp;"), widths))

            def row(metric, values, active, row_note):
                return (f'<tr><td{widths[0]} style="{cell} color:{metadata};">{metric}</td>'
                        + value_cells(values, active, widths[1:4])
                        + f'<td{widths[4]} style="{cell} white-space:nowrap;">{row_note or "&nbsp;"}</td></tr>')
            table_attrs = ""

        html += (
            f'<table {table_attrs}cellpadding="0" cellspacing="0"'
            f' style="margin-top:8px; border-collapse:collapse;">'
            f'<tr>{header}</tr>'
            + row("Peak", peak_values, peak_active, pk_note)
            + row(rms_metric, rms_values, rms_active, rms_note)
            + '</table>'
        )

        # Fader rebalance note
        rebalance = d.get("fader_rebalance_shift", 0.0)
        if rebalance != 0.0:
            fader_off = d.get("fader_offset", 0.0)
            html += (
                f'<div style="margin-top:8px; color:{metadata};">'
                f'Fader rebalanced: {minus(f"{rebalance:+.1f}")} dB shift'
                f' &rarr; fader offset {minus(f"{fader_off:+.1f}")} dB'
                f'</div>'
            )

        # Verbose: append classification analysis metrics
        if verbose and track is not None:
            cr = track.detector_results.get("audio_classifier")
            if cr is not None:
                crest = cr.data.get("crest", 0.0)
                decay = cr.data.get("decay_db", 0.0)
                density = cr.data.get("density", 0.0)
                html += (
                    f'<div style="margin-top:8px; color:{metadata};">'
                    f'Analysis: Crest {crest:.1f} dB'
                    f' &middot; Decay {decay:.1f} dB'
                    f' &middot; Density {density:.0%}'
                    f'</div>'
                )

        return html

    def apply(self, track: TrackContext, result: ProcessorResult) -> np.ndarray:
        if result.classification == "Silent" or result.gain_db == 0.0:
            return track.audio_data
        linear_gain = db_to_linear(float(result.gain_db))
        return track.audio_data * linear_gain
