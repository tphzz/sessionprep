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

    def _no_gain_result(self, classification: str, error: str | None = None) -> ProcessorResult:
        return ProcessorResult(
            processor_id=self.id,
            gain_db=0.0,
            classification=classification,
            method="None",
            data={"gain_db_individual": 0.0},
            error=error,
        )

    def process(self, track: TrackContext) -> ProcessorResult:
        crest_result = track.detector_results.get("audio_classifier")
        silence_result = track.detector_results.get("silence")

        if silence_result and silence_result.data.get("is_silent"):
            return self._no_gain_result("Silent")
        if crest_result is None:
            return self._no_gain_result("Unknown", error="audio_classifier detector result missing")

        override = track.classification_override
        if override == "Skip":
            return self._no_gain_result("Skip")

        peak_db = crest_result.data["peak_db"]
        rms_anchor_db = crest_result.data["rms_anchor_db"]

        if override in ("Transient", "Sustained"):
            classification = override
            is_transient = override == "Transient"
        else:
            classification = crest_result.data["classification"]
            is_transient = crest_result.data["is_transient"]

        gain_for_peak = self.target_peak - peak_db
        gain_for_rms = self.target_rms - rms_anchor_db

        if is_transient:
            gain = gain_for_peak
            method = f"Peak → {self.target_peak:.0f} dB"
        else:
            gain = min(gain_for_rms, gain_for_peak)
            method = f"RMS → {self.target_rms:.0f} dB" if gain == gain_for_rms else "Peak Limited"

        anchor_label = crest_result.data.get("rms_anchor_label", self._rms_anchor_label)
        classifier_metrics = {
            key: crest_result.data[key]
            for key in ("crest", "decay_db", "decay_at_floor", "density")
            if key in crest_result.data
        }
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
                "classifier_metrics": classifier_metrics,
                "gain_for_peak": float(gain_for_peak),
                "gain_for_rms": float(gain_for_rms),
            },
        )

    def render_html(self, result: ProcessorResult, track=None, *, verbose: bool = False,
                    compact: bool = False, narrow: bool = False) -> str:
        """Render the normalization summary line and comparison table."""
        d = result.data
        cls_text = result.classification or "Unknown"
        metadata, muted = "#aaaaaa", "#888888"
        type_color = (
            "#cc77ff" if "Transient" in cls_text
            else "#44cccc" if "Sustained" in cls_text
            else muted
        )
        if cls_text in ("Skip", "Silent", "Unknown"):
            return (
                f'<div><span style="color:{type_color}; font-weight:bold;">{cls_text}</span>'
                f' &nbsp;&middot;&nbsp; <span style="color:{metadata};">No normalization</span>'
                f'</div>'
            )

        off = self._db_offset
        tgt_peak = d.get("target_peak", self.target_peak)
        tgt_rms = d.get("target_rms", self.target_rms)
        anchor_label = d.get("rms_anchor_label", self._rms_anchor_label)
        rms_metric = f"RMS ({anchor_label})" if anchor_label else "RMS"
        is_transient = "Transient" in cls_text
        is_peak_limited = result.method == "Peak Limited"
        peak_active = is_transient or is_peak_limited
        rms_active = not peak_active

        if is_peak_limited:
            method = f"Peak (limited) &rarr; {_fmt_abs_db(tgt_peak, off)} dBFS"
        elif is_transient:
            method = f"Peak &rarr; {_fmt_abs_db(tgt_peak, off)} dBFS"
        else:
            method = f"RMS &rarr; {_fmt_abs_db(tgt_rms, off)} dBFS"

        html = (
            f'<div><span style="color:{type_color}; font-weight:bold;">{cls_text}</span>'
            f' &nbsp;&middot;&nbsp; {method}'
            f' &nbsp;&middot;&nbsp; <b>{_html_minus(f"{result.gain_db:+.1f}")} dB</b></div>'
        )
        pk_note = _note_html(
            True,
            "Chosen &middot; limiting" if is_peak_limited else "Chosen" if is_transient else "",
            type_color, metadata,
        )
        rms_note = (
            _note_html(False, "Would exceed peak", type_color, metadata) if is_peak_limited
            else _note_html(True, "Chosen", type_color, metadata) if rms_active else ""
        )
        peak_values = (
            f"{_fmt_abs_db(d.get('detected_peak_db', float('-inf')), off)} dBFS",
            f"{_fmt_abs_db(tgt_peak, off)} dBFS",
            f"{_fmt_rel_db(d.get('gain_for_peak', 0.0))} dB",
        )
        rms_values = (
            f"{_fmt_abs_db(d.get('detected_rms_db', float('-inf')), off)} dBFS",
            f"{_fmt_abs_db(tgt_rms, off)} dBFS",
            f"{_fmt_rel_db(d.get('gain_for_rms', 0.0))} dB",
        )
        html += _render_comparison_table(
            peak_values, rms_values, peak_active, rms_active,
            pk_note, rms_note, rms_metric, compact=compact, narrow=narrow,
        )
        rebalance = d.get("fader_rebalance_shift", 0.0)
        if rebalance != 0.0:
            fader_off = d.get("fader_offset", 0.0)
            html += (
                f'<div style="margin-top:8px; color:{metadata};">'
                f'Fader rebalanced: {_fmt_rel_db(-rebalance)} dB shift'
                f' &rarr; fader offset {_html_minus(f"{fader_off:+.1f}")} dB</div>'
            )
        if verbose:
            html += _render_verbose_metrics(d, track, metadata)
        return html

    def apply(self, track: TrackContext, result: ProcessorResult) -> np.ndarray:
        if result.classification == "Silent" or result.gain_db == 0.0:
            return track.audio_data
        linear_gain = db_to_linear(float(result.gain_db))
        return track.audio_data * linear_gain


def _fmt_inf(val: float) -> str:
    return "+&infin;" if val > 0 else "&minus;&infin;"


def _html_minus(text: str) -> str:
    return "&minus;" + text[1:] if text.startswith("-") else text


def _fmt_abs_db(val: float, offset: float) -> str:
    if not math.isfinite(val):
        return _fmt_inf(val)
    return _html_minus(f"{val + offset:.1f}")


def _fmt_rel_db(val: float) -> str:
    if not math.isfinite(val):
        return _fmt_inf(val)
    return _html_minus(f"{val:+.1f}")


def _note_html(chosen: bool, label: str, type_color: str, metadata: str) -> str:
    if not label:
        return ""
    style = f"color:{type_color}; font-weight:600;" if chosen else f"color:{metadata};"
    return f'<span style="{style}">{label}</span>'


def _render_narrow_rows(rows: list[tuple[str, tuple[str, str, str], bool, str]],
                        head: str, cell: str, text: str, metadata: str) -> tuple[str, str, str]:
    header = f'<td style="{head}">Measure<br>Detected &rarr; Target &middot; Gain</td>'
    body = ""
    for metric, values, active, row_note in rows:
        weight = " font-weight:bold;" if active else ""
        det, tgt, gain = (f"<nobr>{value}</nobr>" for value in values)
        body += (
            f'<tr><td style="{cell}"><span style="color:{metadata};">{metric}</span>'
            + (f"<br>{row_note}" if row_note else "")
            + f'<br><span style="color:{text};{weight}">'
            f'{det} &rarr; {tgt} &middot; {gain}</span></td></tr>'
        )
    return 'width="100%" ', header, body


def _render_comparison_table(
    peak_values: tuple[str, str, str],
    rms_values: tuple[str, str, str],
    peak_active: bool,
    rms_active: bool,
    pk_note: str,
    rms_note: str,
    rms_metric: str,
    *,
    compact: bool,
    narrow: bool,
) -> str:
    surface, separator = "#2d2d2d", "#3a3a3a"
    text, metadata = "#dddddd", "#aaaaaa"
    head = (
        f'padding:4px 12px; background-color:{surface}; color:{text};'
        f' border-bottom:1px solid {separator};'
    )
    cell = f'padding:4px 12px; border-bottom:1px solid {separator};'
    rows = [("Peak", peak_values, peak_active, pk_note), (rms_metric, rms_values, rms_active, rms_note)]

    def value_cells(values, active, width_attrs):
        weight = " font-weight:bold;" if active else ""
        return "".join(
            f'<td{width} style="{cell} white-space:nowrap; color:{text};{weight}">{value}</td>'
            for value, width in zip(values, width_attrs)
        )

    if narrow:
        table_attrs, header, body = _render_narrow_rows(rows, head, cell, text, metadata)
    elif compact:
        widths = (' width="30%"', ' width="24%"', ' width="24%"', ' width="22%"')
        header = "".join(
            f'<td{w} style="{head}">{label}</td>'
            for label, w in zip(("Measure", "Detected", "Target", "Gain"), widths)
        )
        body = "".join(
            f'<tr><td{widths[0]} style="{cell} color:{metadata};">'
            f'{metric}{f"<br>{row_note}" if row_note else ""}</td>'
            + value_cells(values, active, widths[1:]) + '</tr>'
            for metric, values, active, row_note in rows
        )
        table_attrs = 'width="100%" '
    else:
        widths = (' width="112"', ' width="104"', ' width="104"', ' width="88"', ' width="150"')
        header = "".join(
            f'<td{w} style="{head}">{label}</td>'
            for label, w in zip(("Measure", "Detected", "Target", "Gain", "&nbsp;"), widths)
        )
        body = "".join(
            f'<tr><td{widths[0]} style="{cell} color:{metadata};">{metric}</td>'
            + value_cells(values, active, widths[1:4])
            + f'<td{widths[4]} style="{cell} white-space:nowrap;">{row_note or "&nbsp;"}</td></tr>'
            for metric, values, active, row_note in rows
        )
        table_attrs = ""

    return (
        f'<table {table_attrs}cellpadding="0" cellspacing="0"'
        f' style="margin-top:8px; border-collapse:collapse;">'
        f'<tr>{header}</tr>{body}</table>'
    )


def _render_verbose_metrics(d: dict, track, metadata: str) -> str:
    metrics = d.get("classifier_metrics")
    if metrics is None and track is not None:
        cr = track.detector_results.get("audio_classifier")
        if track.repair_recipe.active:
            checked = track.corrected_analysis
            cr = (
                checked.results.get("audio_classifier")
                or checked.support_results.get("audio_classifier")
                if checked else None
            )
        metrics = cr.data if cr and not cr.error else {}
    if not (metrics and {"crest", "decay_db", "density"} <= metrics.keys()):
        return ""
    crest = metrics["crest"]
    decay = metrics["decay_db"]
    decay_prefix = "≥" if metrics.get("decay_at_floor") else ""
    density = metrics["density"]
    return (
        f'<div style="margin-top:8px; color:{metadata};">'
        f'Analysis: Crest {crest:.1f} dB'
        f' &middot; Decay {decay_prefix}{decay:.1f} dB'
        f' &middot; Density {density:.0%}'
        f'</div>'
    )
