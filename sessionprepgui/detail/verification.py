"""Presentation of a detector's two measured results in the File and repair views."""
from __future__ import annotations

import math

from sessionpreplib.audio import dbfs_offset
from sessionpreplib.models import Severity
from sessionpreplib.repair_verification import CheckOutcome

from ..helpers import esc
from ..theme import COLORS


def measurement_table(rows, *, heading="Measurement", editor=False, action=None):
    """A Qt rich-text table with aligned source columns, including in a wrapped row."""
    widths = (43, 24, 33) if editor else (43, 32, 25)
    if action is not None:
        # The remaining 26% holds File's native Review repair action.
        widths = (32, 23, 19)
    header = "".join(
        f'<td width="{width}%" style="padding:5px 8px 5px 0;'
        f' border-bottom:1px solid {COLORS["border"]}; color:{COLORS["metadata"]};'
        f' font-weight:600;">{esc(title)}</td>'
        for title, width in zip((heading, "Original", "Corrected"), widths)
    )
    body = []
    if action is not None:
        header += '<td width="26%"></td>'
    for index, (label, original, corrected) in enumerate(rows):
        values = [esc(label), esc(original), esc(corrected)]
        if editor:
            values[2] = f'<b>{values[2]}</b>'
        cells = "".join(
            f'<td valign="top" style="padding:6px 8px 6px 0;">{value}</td>' for value in values
        )
        if action is not None:
            content = action if index == len(rows) // 2 else ""
            cells += f'<td align="center" valign="middle" style="padding-left:8px;">{content}</td>'
        body.append("<tr>" + cells + "</tr>")
    return ('<table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">'
            f'<tr>{header}</tr>{"".join(body)}</table>')


def compact_measurements(rows, *, stacked=False):
    """Keep the source attached to its reading, even beside an open Help pane."""
    body = []
    for label, original, corrected in rows:
        body.append(f'<tr><td colspan="2" style="padding:8px 0 4px 0;'
                    f' border-top:1px solid {COLORS["border"]};"><b>{esc(label)}</b></td></tr>')
        cells = []
        for source, value in (("Original", original), ("Corrected", corrected)):
            join = "<br>" if stacked else " &nbsp;"
            reading = (f'<span style="color:{COLORS["metadata"]};">{source}</span>'
                       f'{join}{esc(value)}')
            if stacked:
                body.append(f'<tr><td colspan="2" style="padding:3px 0;">{reading}</td></tr>')
            else:
                cells.append(f'<td width="50%" valign="top" style="padding:3px 6px 4px 0;">{reading}</td>')
        if cells:
            body.append("<tr>" + "".join(cells) + "</tr>")
    return '<table width="100%" cellspacing="0" cellpadding="0">' + "".join(body) + '</table>'


def measurement_note(note, *, stacked=False):
    if not note:
        return ""
    value = esc(note)
    if stacked:
        value = value.replace(" · ", "<br>").replace("Warning threshold ", "Warning threshold<br>")
    return f'<p style="margin:10px 0 4px 0; color:{COLORS["metadata"]};">{value}</p>'


def verification_status(outcome):
    """Return the shared status label and colour when severity alone is insufficient."""
    return {
        CheckOutcome.RESOLVED: ("✓ Correction applied", COLORS["clean"]),
        CheckOutcome.PENDING: ("Checking…", COLORS["metadata"]),
        CheckOutcome.FAILED: ("Unable to verify", COLORS["attention"]),
        CheckOutcome.INAPPLICABLE: ("Not applicable", COLORS["metadata"]),
    }.get(outcome)


def _level(value, unit, places=1):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(value):
        return "—"
    return f"{value:.{places}f}".replace("inf", "∞").replace("-", "−") + unit


def subsonic_note(config):
    return "Relative to program energy · Warning threshold " + _level(config.get("subsonic_warn_ratio_db", -20), " dB")


def _channels(result, track, config):
    if result.severity == Severity.CLEAN:
        return []
    if "affected_channels" in result.data:
        return [i for i in result.data["affected_channels"] if type(i) is int and 0 <= i < track.channels]
    if result.data.get("dual_mono"):
        return list(range(track.channels))
    if result.detector_id == "dc_offset" and "signed_dc" in result.data:
        threshold = 10 ** (config.get("dc_offset_warn_db", -40) / 20)
        return [i for i, mean in enumerate(result.data["signed_dc"]) if abs(mean) > threshold]
    if result.detector_id == "subsonic" and "per_channel" in result.data:
        return [i for i in range(track.channels)
                if result.data["per_channel"].get(i, result.data["per_channel"].get(str(i), {})).get("warn")]
    if any(issue.channel is None for issue in result.issues):
        return list(range(track.channels))
    return sorted({issue.channel for issue in result.issues if 0 <= issue.channel < track.channels})


def channel_names(channels, total):
    if not channels:
        return "None"
    if total == 2:
        return " and ".join(("left", "right")[i] for i in sorted(channels)).capitalize()
    return ", ".join(str(i + 1) for i in channels)


def comparison_readings(check, track, config):
    """Build comparable readings, including clean numeric results hidden by detector summaries."""
    before, after = check.original, check.corrected
    detector_id = (before or after).detector_id
    offset = dbfs_offset(config)
    cutoff = config.get("subsonic_hz", 30)
    fields = {
        "dc_offset": [("DC offset", "dc_db", " dBFS", 1)],
        "subsonic": [(f"Energy at or below {cutoff:g} Hz", "subsonic_ratio_db", " dB", 1)],
        "stereo_compat": [("Minimum correlation", "min_window_corr", "", 2),
                          ("Whole-track correlation", "lr_corr", "", 2),
                          ("Mono loss", "mono_loss_db", " dB", 1)],
        "ultrasonic": [("Energy above 24.1 kHz", "band_dbfs", " dBFS RMS", 1),
                       ("Relative to program energy", "ratio_db", " dB", 1)],
        "clipping": [("Clipped ranges", "runs", "", 0)],
        "one_sided_silence": [("Left RMS", "l_rms_db", " dBFS", 1), ("Right RMS", "r_rms_db", " dBFS", 1)],
    }.get(detector_id, [])
    sources = (before, after)
    def reading(result, formatter):
        if result is None:
            return "Pending" if check.outcome == CheckOutcome.PENDING else "Unavailable"
        if result.error:
            return f"Unable to verify · {result.error}"
        if not result.applicable:
            return f"Not applicable · {result.summary}"
        return formatter(result)

    rows = []
    for name, key, unit, places in fields:
        if any(r and key in r.data for r in sources):
            def format_value(r, _key=key, _unit=unit, _places=places):
                value = r.data.get(_key)
                if _key in {"dc_db", "l_rms_db", "r_rms_db"} and isinstance(value, (int, float)):
                    value += offset
                return _level(value, _unit, _places)
            rows.append((name, *(reading(r, format_value) for r in sources)))
    if not rows:
        rows.append(("Result", *(reading(r, lambda v: v.summary) for r in sources)))
    counts = [len(r.issues) if r and r.applicable and not r.error else None for r in sources]
    channels = [_channels(r, track, config) if r and r.applicable and not r.error else None for r in sources]
    rows.append(("Affected channels", *(reading(r, lambda v: channel_names(_channels(v, track, config), track.channels))
                                         for r in sources)))
    rows.append(("Locations", *(reading(r, lambda v: str(len(v.issues))) for r in sources)))
    if detector_id == "dc_offset":
        rows.insert(1, ("Affected region", *(reading(r, lambda v: "Whole track" if v.issues else "None")
                                              for r in sources)))
    detail = ""
    if detector_id == "dc_offset":
        detail = "Warning threshold " + _level(config.get("dc_offset_warn_db", -40) + offset, " dBFS")
    elif detector_id == "subsonic":
        detail = subsonic_note(config)
    location = ""
    if None not in counts:
        count = counts[1]
        location = (f"{count} location" + ("" if count == 1 else "s") if counts[0] == count
                    else f"{counts[0]} → {count} locations")
    partial = ""
    if check.outcome == CheckOutcome.REMAINING and None not in channels and len(channels[0]) > len(channels[1]) > 0:
        partial = f"{len(channels[0])} affected channels → {len(channels[1])}"
        if track.channels == 2 and len(channels[1]) == 1:
            detail = f"Still detected on the {channel_names(channels[1], 2).lower()} channel" + (f" · {detail}" if detail else "")
    return rows, detail, location, partial
