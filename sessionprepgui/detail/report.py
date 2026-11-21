"""HTML report rendering for the GUI."""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from numbers import Integral
from urllib.parse import quote
from ..icons import icon_path

from sessionpreplib.chunks import chunk_sizes, STANDARD_CHUNKS, detect_origin
from sessionpreplib.detector import TrackDetector
from sessionpreplib.repair_verification import CheckOutcome, audition_results, detector_check, original_report_track, separate_rumble_finding
from .repair_options import track_repair_options
from ..theme import COLORS, FILE_COLOR_TRANSIENT, FILE_COLOR_SUSTAINED
from ..helpers import esc
from .report_style import REPORT_SECTION_HEADING, REPORT_SECTION_SIZE, REPORT_TITLE_SIZE
from .verification import (
    compact_measurements, comparison_readings, measurement_note, measurement_table, verification_status,
)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _severity_meta(severity):
    key = "information" if severity == "info" else severity
    return {
        "problem": (COLORS["problems"], "Problem"),
        "attention": (COLORS["attention"], "Attention"),
        "information": (COLORS["information"], "Information"),
        "clean": (COLORS["clean"], "Clean"),
    }.get(key, (COLORS["information"], "Information"))


def _guidance(hint):
    if not hint:
        return "\u2014"
    if hint == "request reprint / check limiting":
        return "Request a clean reprint or check limiting."
    return hint[0].upper() + hint[1:]


def _location_text(result):
    ranges = [(i.sample_start, i.sample_end, i.channel) for i in result.issues]
    if not ranges:
        ranges = [(r[0], r[1], r[2] if len(r) > 2 else None) for r in (result.data.get("ranges") or [])
                  if isinstance(r, (tuple, list)) and len(r) >= 2]
    valid = []
    for start, end, channel in ranges:
        try:
            start = int(start)
            end = start if end is None else int(end)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= start <= end:
            value = f"Samples {start}\u2013{end}" if end != start else f"Sample {start}"
            if isinstance(channel, Integral):
                value += f" · Ch {channel + 1}"
            valid.append(value)
    if not valid:
        return "\u2014", ""
    return (valid[0] if len(valid) == 1 else f"{len(valid)} locations",
            "; ".join(valid))


def _signed(value):
    """Signed one-decimal value using a true minus sign (normalization section)."""
    text = f"{value:+.1f}"
    return "&minus;" + text[1:] if text.startswith("-") else text


def _location_html(location):
    """Keep the sample numbers together while allowing the channel label to wrap."""
    for prefix in ("Samples ", "Sample "):
        if location.startswith(prefix):
            samples, separator, channel = location[len(prefix):].partition(" · ")
            return f"{prefix}<nobr>{esc(samples)}</nobr>{esc(separator + channel)}"
    return esc(location)


def _image(name):
    return f'<img src="{esc(icon_path(name).as_uri())}" width="16" height="16">'


def _link(scheme, key, label, title=""):
    target = scheme + ":" + quote(str(key), safe="")
    return (f'<a href="{esc(target)}" title="{esc(title)}" '
            f'style="color:{COLORS["information"]}; text-decoration:none;">{label}</a>')


def _table_header(labels, widths, *, table_width="100%"):
    cells = [_finding_cell(esc(label), width, header=True, index=i, columns=len(widths))
             for i, (label, width) in enumerate(zip(labels, widths))]
    # Qt drops a heading's bottom margin before a table; the table carries it.
    return (f'<table width="{table_width}" cellpadding="0" cellspacing="0" '
            'style="margin-top:8px; border-collapse:collapse;">'
            '<tr>' + "".join(cells) + '</tr>')


def render_summary_html(summary, *, show_faders=True, show_clean=True, session=None):
    """Render compact findings while keeping file and finding counts distinct."""
    categories = ("problems", "attention", "information", "clean")
    counts = {key: sum(max(len(g.get("items") or []), int(bool(g.get("standalone"))))
                       for g in summary.get(key) or []) for key in categories}
    clean, total = int(summary.get("clean_count", 0)), int(summary.get("total_ok", 0))
    parts = [f'<p style="font-size:{REPORT_TITLE_SIZE}; font-weight:600;'
             f' margin-top:0; margin-bottom:8px;">Session health</p>',
             f'<p style="margin-top:0; margin-bottom:8px;"><b>{clean} of {total} files clean</b></p>']
    overview = summary.get("overview") or {}
    if overview.get("most_common_sr") is not None:
        parts.append(f'<p style="color:{COLORS["metadata"]}; margin-top:0;">'
                     f'Session format: {esc(overview["most_common_sr"])} Hz / '
                     f'{esc(overview.get("most_common_bd", "?"))}</p>')
    parts.append(f'<p style="color:{COLORS["metadata"]}; margin-top:0; margin-bottom:8px;">'
                 f'Attention {counts["attention"]} &middot; Information {counts["information"]}</p>')
    tracks = {t.filename: t for t in session.tracks} if session else {}
    for key, label in (("verification_pending", "Checking repaired audio"), ("verification_failed", "Unable to verify")):
        if names := summary.get(key):
            parts.append(f'<p style="color:{COLORS["attention"]};">{label} · {esc(", ".join(names))}</p>')
    widths = ("22.13%", "22.72%", "42.62%", "12.53%")
    findings_anchor = '<a name="report-findings"></a>'
    for category in categories:
        if category == "clean" and not show_clean:
            continue
        groups = summary.get(category) or []
        if not counts[category]:
            continue
        severity = {"problems": "problem", "clean": "clean"}.get(category, category)
        color, label = _severity_meta(severity)
        count = counts[category]
        parts.append(f'<p style="font-size:{REPORT_SECTION_SIZE}; color:{color};'
                     f' font-weight:bold; margin-bottom:6px;">{findings_anchor}'
                     f'{label if category != "problems" else "Problems"} &middot; '
                     f'{count} {"finding" if count == 1 else "findings"}</p>')
        findings_anchor = ""
        parts.append(_table_header(("Finding", "Affected file", "Next step", "Action"), widths))
        for group in groups:
            items = group.get("items") or []
            refs = group.get("item_refs") or []
            if not items and not group.get("standalone"):
                continue
            for index in range(max(1, len(items))):
                ref = refs[index] if index < len(refs) else None
                filename = ref.get("filename") if ref else None
                detail = ref.get("detail", "") if ref else (items[index] if items else "\u2014")
                track = tracks.get(filename)
                result = (detector_check(track, "clipping", pending=bool(track._cache.get("repair_verification_pending"))).remaining_result
                          if track and group.get("detector_id") == "clipping" else None)
                if result is not None and group.get("detector_id") == "clipping":
                    runs = result.data.get("runs")
                    if isinstance(runs, int):
                        detail = f'{runs} clipped {"range" if runs == 1 else "ranges"}'
                finding = f'<b>{esc(group.get("title", ""))}</b><br><span style="color:{color};">{label}</span>'
                file_text = (f'<b>{esc(filename)}</b><br>' if filename else "")
                file_text += f'<span style="color:{COLORS["metadata"]};">{esc(detail)}</span>'
                action = _link("report-review", filename, "Review file", f"Review {filename}") if filename else ""
                values = (finding, file_text, esc(_guidance(group.get("hint"))), action)
                parts.append("<tr>" + "".join(
                    _finding_cell(value, width, index=i, columns=len(widths))
                    for i, (value, width) in enumerate(zip(values, widths))) + "</tr>")
        parts.append("</table>")
    if findings_anchor:
        parts.append(f'<p style="color:{COLORS["metadata"]};">'
                     f'{findings_anchor}No findings to show.</p>')
    return "\n".join(parts)


_log = logging.getLogger(__name__)


def _layout_variant(render_html, args, verbose, fallback, **layout):
    """Render an optional narrower layout, keeping ``fallback`` when unavailable.

    Only keywords the override declares (or ``**kwargs``) are passed, and a
    failing variant never takes the File report down with it.
    """
    if not all(_accepts(render_html, name) for name in layout):
        return fallback
    try:
        return render_html(*args, verbose=verbose, **layout) or fallback
    except Exception:  # noqa: BLE001 - third-party renderers must not break the report
        _log.exception("Processor layout variant %s failed; using the wider one", layout)
        return fallback


def _accepts(render_html, name):
    """Processor overrides written before a layout keyword existed keep working."""
    try:
        parameters = inspect.signature(render_html).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in parameters)


def _finding_line(body, height):
    # Qt supports table height, but not CSS min-height on report cells.
    return (f'<table height="{height}" cellpadding="0" cellspacing="0"><tr>'
            f'<td valign="middle">{body}</td></tr></table>')


def _finding_cell(body, width, *, header=False, active=False, index=0, columns=1, expanded=False, valign="top"):
    size = f' width="{width}"' if width is not None else ""
    background = COLORS["surface"] if header or active else COLORS["bg"]
    padding = 4
    left = 12 if index == 0 else 6
    right = 12 if index == columns - 1 else 6
    rule = "" if expanded else f' border-bottom:1px solid {COLORS["accent"]};'
    return (f'<td{size} valign="{valign}" style="padding:{padding}px {right}px {padding}px {left}px;{rule}'
            f' background-color:{background}; color:{COLORS["text"]};">'
            f'{body}</td>')


def visualizable_issues(result, track):
    """Return existing locations that can safely be shown in this waveform."""
    if result is None or result.error or not result.applicable or track.total_samples <= 0 or track.channels <= 0:
        return []
    valid = []
    for issue in result.issues or []:
        start, end, channel = issue.sample_start, issue.sample_end, issue.channel
        if (not isinstance(start, Integral) or isinstance(start, bool)
                or not 0 <= start < track.total_samples):
            continue
        if end is not None and (not isinstance(end, Integral) or isinstance(end, bool)
                                or not start <= end < track.total_samples):
            continue
        if channel is not None and (not isinstance(channel, Integral) or isinstance(channel, bool)
                                    or not 0 <= channel < track.channels):
            continue
        if isinstance(issue.label, str) and issue.label:
            valid.append(issue)
    return valid


def _finding_actions(row, compact):
    links = []
    if row.get("retry"):
        links.append((48, _link("report-retry", row["id"], "Retry")))
    if source := row.get("location_source"):
        links.append((144, _link(f"report-{source}", row["id"], f"Show {source} locations")))
    elif row.get("visualize_labels"):
        links.append((80, _link("report-visualize", row["id"], "<nobr>Visualize</nobr>",
                                "Show only this detector's locations and fit the whole file")))
    if row["help"]:
        links.append((48, _link("report-help", row["id"], "<nobr>Help</nobr>",
                                f'Help for {row["title"]}')))
    if compact:
        cells = [f'<tr><td height="32" width="{width}" align="center" valign="middle">'
                 f'{link}</td></tr>' for width, link in links]
        body = '<tr><td height="8"></td></tr>'.join(cells)
    else:
        cells = [f'<td height="32" width="{width}" align="center" valign="middle">'
                 f'{link}</td>' for width, link in links]
        body = "<tr>" + '<td width="8"></td>'.join(cells) + "</tr>"
    return '<table cellpadding="0" cellspacing="0">' + body + "</table>" if links else ""


def _repair_action(scheme, key, label, action_size):
    # A native form action occupies this cell in the interactive browser.
    # The link also provides the static report and keyboard reading order.
    size = action_size(scheme, key, label) if action_size else None
    dimensions = f' width="{size.width()}" height="{size.height()}"' if size else ''
    link = _link(scheme, key, f"<nobr>{label}</nobr>")
    return (f'<table{dimensions} cellspacing="0" cellpadding="0"><tr>'
            f'<td{dimensions} align="center" valign="middle">{link}</td></tr></table>')


def _expanded_finding(row, narrow, action_size, *, compact=False):
    comparison = row.get("comparison")
    action = (_repair_action("report-repair", row["id"], "Review repair", action_size)
              if row.get("review_repair") else "")
    if comparison:
        if narrow:
            result = compact_measurements(row.get("compact_readings", comparison), stacked=True)
        else:
            result = measurement_table(comparison, action=action if action and not compact else None)
        result += measurement_note(row.get("comparison_note"), stacked=narrow)
        result += measurement_note(row.get("repair_attribution"), stacked=narrow)
        if narrow or compact:
            result += action
        return result
    result = "<b>Result</b> " + esc(row["summary"])
    if action:
        if narrow:
            result += action
        else:
            result = (f'<table width="100%" cellspacing="0" cellpadding="0"><tr><td valign="middle">{result}</td>'
                      f'<td align="right">{action}</td></tr></table>')
    parts = [result, measurement_note(row.get("repair_attribution"), stacked=narrow)]
    if row["locations"]:
        parts.append("<div>" + "; ".join(_location_html(item) for item in row["locations"].split("; ")) + "</div>")
    parts.append('<table cellpadding="3" cellspacing="0">' + row["fragment"] + "</table>")
    return "".join(parts)


def _render_narrow_findings(rows, expanded, active_help, action_size):
    """One stacked column: severity, title band, location and next step, actions."""
    parts = ['<table width="100%" cellpadding="0" cellspacing="0"'
             ' style="margin-top:8px; border-collapse:collapse;">']
    for row in rows:
        key = row["id"]
        token = quote(key, safe="")
        color, label = _row_status(row)
        arrow = _image("chevron-down" if key in expanded else "chevron-right")
        title = _link("report-details", key, f'{arrow}&nbsp;&nbsp; <b>{esc(row["title"])}</b>',
                      "Toggle detector details")
        # The count leads the metadata lines, so the title band holds the title
        # only; an absent location is left out rather than shown as a dash.
        details = [f'<nobr>{esc(row["count"])}</nobr>'] if row["count"] else []
        if row["location"] and row["location"] != "—":
            details.append(_location_html(row["location"]))
        if row["hint"]:
            details.append(f'<span style="color:{COLORS["text"]};">{esc(row["hint"])}</span>')
        # One inner table with explicit rows: Qt adds stray empty blocks around
        # paragraphs/divs that hold tables, and ignores table/cell heights this
        # deep, so 6 px padding centres a line in the 32 px title/action bands.
        lines = [f'<tr><td><a name="finding-{token}"></a>'
                 f'<span style="color:{color};">{label}</span></td></tr>',
                 f'<tr><td style="padding:6px 0;">{title}</td></tr>']
        if details:
            lines.append(f'<tr><td style="padding-left:24px; color:{COLORS["metadata"]};">'
                         + "<br>".join(details) + '</td></tr>')
        actions = _finding_actions(row, bool(row.get("comparison")))
        if actions:
            lines.append(f'<tr><td style="padding:6px 0;">{actions}</td></tr>')
        body = ('<table width="100%" cellpadding="0" cellspacing="0">'
                + "".join(lines) + '</table>')
        active = key == active_help and color != COLORS["problems"]
        compared = bool(row.get("comparison") and key in expanded)
        parts.append("<tr>" + _finding_cell(body, None, active=active, expanded=compared) + "</tr>")
        if key in expanded:
            padding = "4px 4px 4px 8px" if row.get("comparison") else "4px 12px 4px 36px"
            rule = f' border-bottom:1px solid {COLORS["border"]};' if compared else ""
            parts.append(f'<tr><td style="padding:{padding};{rule}">'
                         + _expanded_finding(row, True, action_size) + '</td></tr>')
    parts.append("</table>")
    return parts


def _render_finding_row_parts(
    row: dict,
    *,
    widths: tuple,
    compact: bool,
    expanded,
    active_help,
    action_size,
) -> list[str]:
    key = row["id"]
    token = quote(key, safe="")
    color, label = _row_status(row)
    arrow = _image("chevron-down" if key in expanded else "chevron-right")
    title = _link(
        "report-details",
        key,
        f'{arrow}&nbsp;&nbsp; <b>{esc(row["title"])}</b>',
        "Toggle detector details",
    )
    if row["count"]:
        title += f' &middot; <span style="color:{COLORS["metadata"]};">{esc(row["count"])}</span>'
    finding = f'<a name="finding-{token}"></a>' + _finding_line(title, 32)
    if compact:
        finding += (
            f'<p style="margin:4px 0 0 24px; color:{COLORS["metadata"]};">'
            f'{_location_html(row["location"])}<br>'
            f'<span style="color:{COLORS["text"]};">{esc(row["hint"])}</span></p>'
        )
    severity = _finding_line(f'<span style="color:{color};">{label}</span>', 32)
    actions = _finding_actions(row, compact)
    location = _location_html(row["location"])
    count_in_cell = not compact and row["location"].endswith((" location", " locations"))
    location = f"<nobr>{location}</nobr>" if count_in_cell else _finding_line(location, 32)
    values = (
        (severity, finding, actions)
        if compact
        else (severity, finding, location, _finding_line(esc(row["hint"]), 32), actions)
    )
    compared = bool(row.get("comparison") and key in expanded)
    out = [
        "<tr>"
        + "".join(
            _finding_cell(
                value,
                width,
                active=key == active_help and color != COLORS["problems"],
                index=i,
                columns=len(widths),
                expanded=compared,
                valign="middle" if count_in_cell and i == 2 else "top",
            )
            for i, (value, width) in enumerate(zip(values, widths))
        )
        + "</tr>"
    ]
    if compared:
        rule = f'border-bottom:1px solid {COLORS["border"]};'
        lead = "" if compact else f'<td style="{rule}"></td>'
        span = len(widths) if compact else len(widths) - 1
        out.append(
            f'<tr>{lead}<td colspan="{span}" style="padding:8px 12px 12px 30px; {rule}">'
            + _expanded_finding(row, False, action_size, compact=compact)
            + "</td></tr>"
        )
    elif key in expanded:
        out.append(
            f'<tr><td colspan="{len(widths)}" style="padding:4px 12px 4px 136px;">'
            + _expanded_finding(row, False, action_size)
            + "</td></tr>"
        )
    return out


def render_prepared_track_report(data, *, expanded=(), compact=False, narrow=False,
                                 active_help=None, action_size=None):
    """Render cached data with severity-leading rows; Help lives in its own pane.

    ``narrow`` (below the compact layout) stacks each finding in one column.
    """
    if narrow:
        header = data.get("header_narrow", data.get("header_compact", data["header"]))
    elif compact:
        header = data.get("header_compact", data["header"])
    else:
        header = data["header"]
    heading = (
        f'<p style="{REPORT_SECTION_HEADING} color:{COLORS["heading"]};">'
        '<a name="report-findings"></a>Detector findings</p>'
    )
    heading_row = f'<td valign="middle">{heading}</td>'
    if data.get("can_review_repairs"):
        if narrow:
            heading_row += "</tr><tr>"
        side = "left" if narrow else "right"
        action = _repair_action("report-repairs", "track", "Track repairs…", action_size)
        heading_row += f'<td align="{side}">{action}</td>'
    parts = [
        header,
        '<div><table width="100%" cellspacing="0" cellpadding="0" style="margin-top:16px;"><tr>'
        + heading_row
        + "</tr></table>",
    ]
    if failure := data.get("source_failure"):
        parts.append(
            f'<p style="color:{COLORS["attention"]};">Unable to verify · {esc(failure)} &nbsp; '
            + _link("report-retry", "track", "Retry")
            + "</p>"
        )
    rows = data["rows"]
    if not rows:
        parts.append(f'<p style="color:{COLORS["metadata"]};">No findings to show.</p></div>')
        return "\n".join(parts)
    if narrow:
        parts.extend(_render_narrow_findings(rows, expanded, active_help, action_size))
        parts.append("</div>")
        return "\n".join(parts)
    paired = any("comparison" in row for row in rows)
    status = "Status" if paired else "Severity"
    width = "146" if paired else "100"
    widths = (
        (width, None, "100" if paired else "80")
        if compact
        else (("17.5%", "23.5%", "14%", "20%", "25%") if paired else (width, None, "96", "216", "136"))
    )
    labels = (
        (status, "Finding, location and next step", "Actions")
        if compact
        else (status, "Finding", "Location", "Next step", "Actions")
    )
    parts.append(_table_header(labels, widths, table_width="99.9%" if paired and not compact else "100%"))
    for row in rows:
        parts.extend(
            _render_finding_row_parts(
                row,
                widths=widths,
                compact=compact,
                expanded=expanded,
                active_help=active_help,
                action_size=action_size,
            )
        )
    parts.append("</table></div>")
    return "\n".join(parts)


def render_track_detail_html(track, session=None, *, show_clean=True, verbose=False):
    """Render a static compact view; interactive browsers cache prepared data."""
    return render_prepared_track_report(
        prepare_track_report(track, session, show_clean=show_clean, verbose=verbose))


def _track_findings(track, config):
    """Keep the source detector and a qualifying local repair observation distinct."""
    signals = audition_results(track, corrected=True, pending=bool(track._cache.get("repair_verification_pending")))
    for key in dict.fromkeys((*track.detector_results, *signals)):
        result = track.detector_results.get(key) or signals[key]
        yield key, key, result, None
    local = separate_rumble_finding(track, config)
    if local:
        evidence = local.corrected if local.corrected and local.corrected.issues else local.original
        yield "subsonic-local", "subsonic", evidence, local


def _row_status(row):
    outcome = row.get("outcome")
    special = verification_status(outcome)
    if special:
        label, color = special
        return color, label
    color, label = _severity_meta(row["severity"])
    if outcome == CheckOutcome.NEW:
        label += " · New after repair"
    return color, label


def select_report_source(data, corrected):
    """Select cached row evidence without rereading the file or recomputing detectors."""
    for row in data["rows"]:
        if "source_labels" in row:
            original, repaired = row["source_labels"]
            row["visualize_labels"] = repaired if corrected else original
            row["location_source"] = None
            if not row["visualize_labels"]:
                if corrected and original and row.get("outcome") == CheckOutcome.RESOLVED:
                    row["location_source"] = "original"
                elif not corrected and repaired and row.get("outcome") == CheckOutcome.NEW:
                    row["location_source"] = "corrected"


# ---------------------------------------------------------------------------
# Fader table
# ---------------------------------------------------------------------------

def render_fader_table_html(session) -> str:
    """Render the fader offset table as an HTML table."""
    rows = []
    for t in session.tracks:
        if t.status != "OK":
            rows.append(
                f'<tr><td style="color:{COLORS["problems"]}">{esc(t.filename)}</td>'
                f'<td>Error</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>'
                f'<td style="color:{COLORS["problems"]}">ERR</td></tr>'
            )
            continue

        pr = next(iter(t.processor_results.values()), None) if t.processor_results else None
        fmt_str = f"{t.samplerate/1000:.0f}k/{t.bitdepth}"

        if pr and pr.classification == "Silent":
            rows.append(
                f'<tr><td style="color:{COLORS["metadata"]}">{esc(t.filename)}</td>'
                f'<td>{fmt_str}</td><td>Silent</td><td>0.0 dB</td><td>0.0 dB</td>'
                f'<td style="color:{COLORS["attention"]}">SILENT</td></tr>'
            )
            continue

        classification = pr.classification if pr else "Unknown"
        gain_db = pr.gain_db if pr else 0.0
        fader_offset = pr.data.get("fader_offset", 0.0) if pr else 0.0

        is_clipped = False
        clip_r = t.detector_results.get("clipping")
        if clip_r:
            is_clipped = bool(clip_r.data.get("is_clipped"))

        type_color = FILE_COLOR_TRANSIENT.name() if "Transient" in classification else FILE_COLOR_SUSTAINED.name()
        status_color = COLORS["problems"] if is_clipped else COLORS["clean"]
        status_label = "CLIP" if is_clipped else "OK"

        rows.append(
            f'<tr>'
            f'<td>{esc(t.filename)}</td>'
            f'<td>{fmt_str}</td>'
            f'<td style="color:{type_color}">{esc(classification)}</td>'
            f'<td>{gain_db:+.1f} dB</td>'
            f'<td style="font-weight:bold; color:{COLORS["clean"]}">{fader_offset:+.1f} dB</td>'
            f'<td style="color:{status_color}">{status_label}</td>'
            f'</tr>'
        )

    header = (
        '<table cellpadding="4" cellspacing="0" style="border-collapse:collapse; width:100%;">'
        '<tr style="border-bottom:1px solid #555;">'
        f'<th align="left" style="color:{COLORS["heading"]}">Track</th>'
        f'<th align="left" style="color:{COLORS["heading"]}">Format</th>'
        f'<th align="center" style="color:{COLORS["heading"]}">Type</th>'
        f'<th align="right" style="color:{COLORS["heading"]}">Gain</th>'
        f'<th align="right" style="color:{COLORS["heading"]}">Fader</th>'
        f'<th align="right" style="color:{COLORS["heading"]}">Status</th>'
        '</tr>'
    )
    return header + "\n".join(rows) + "</table>"


# ---------------------------------------------------------------------------
# Per-track detail
# ---------------------------------------------------------------------------

def _fmt_chunk_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _append_track_metadata_parts(parts: list, track) -> None:
    fmt = f"{track.samplerate} Hz / {track.bitdepth} / {track.channels}ch"
    dur = f"{track.duration_sec:.2f}s ({track.total_samples} samples)"
    parts.append(f'<div style="color:{COLORS["metadata"]}; margin-top:6px;">{fmt}</div>')
    parts.append(f'<div style="color:{COLORS["metadata"]};">{dur}</div>')

    try:
        notable = [(name, size) for name, size in chunk_sizes(track.filepath) if name not in STANDARD_CHUNKS]
    except (ValueError, OSError):
        notable = []
    origin = detect_origin(track.chunk_ids, track.filepath)
    meta_parts = []
    if notable:
        chunk_parts = [f"{esc(name.strip())} ({_fmt_chunk_size(size)})" for name, size in notable]
        meta_parts.append(f'Chunks: {" &middot; ".join(chunk_parts)}')
    if origin:
        meta_parts.append(f"Origin: {esc(origin)}")
    if meta_parts:
        parts.append(f'<div style="color:{COLORS["metadata"]}; margin-top:4px;">{" / ".join(meta_parts)}</div>')


def _append_processor_parts(parts: list, track, proc_map: dict, *, verbose: bool) -> None:
    for proc_id, pr in (track.processor_results or {}).items():
        proc_inst = proc_map.get(proc_id)
        if proc_inst:
            html_fragment = proc_inst.render_html(pr, track, verbose=verbose)
            if not html_fragment:
                continue
            render = proc_inst.render_html
            compact_fragment = _layout_variant(render, (pr, track), verbose, html_fragment, compact=True)
            narrow_fragment = _layout_variant(
                render, (pr, track), verbose, compact_fragment, compact=True, narrow=True
            )
        else:
            html_fragment = compact_fragment = narrow_fragment = (
                f'<div>{esc(pr.classification or "")} &middot; {esc(pr.method)} &middot; {pr.gain_db:+.1f} dB</div>'
            )
        parts.append(
            f'<p style="{REPORT_SECTION_HEADING} color:{COLORS["heading"]};">'
            '<a name="report-processors"></a>'
            f'{esc(proc_map[proc_id].name if proc_id in proc_map else proc_id)}</p>'
        )
        parts.append((html_fragment, compact_fragment, narrow_fragment))
        if track.group:
            original = pr.data.get("original_gain_db", pr.gain_db)
            delta = pr.gain_db - original
            parts.append(
                f'<div style="color:{COLORS["metadata"]}; margin-top:8px;">'
                f'Group: <b>{esc(track.group)}</b>'
                f' &nbsp;&middot;&nbsp; Group level: <b>{_signed(pr.gain_db)} dB</b>'
                f' (individual: {_signed(original)} dB,'
                f' &Delta; {_signed(delta)} dB)</div>'
            )


def _attach_check_details(row: dict, check, det_id: str, track, config: dict) -> None:
    row["outcome"] = check.outcome
    row["retry"] = check.outcome == CheckOutcome.FAILED and not track._cache.get("repair_error")
    row["source_labels"] = tuple(
        tuple(sorted({i.label for i in visualizable_issues(r, track)}))
        for r in (check.original, check.corrected)
    )
    readings, note, counts, partial = comparison_readings(check, track, config)
    row["comparison"] = [entry for entry in readings if not counts or entry[0] != "Locations"]
    row["comparison_note"] = note
    if det_id == "subsonic" and check.original and check.corrected:
        _, left, right = readings[0]
        compact = []
        for r, value in ((check.original, left), (check.corrected, right)):
            if r.applicable and not r.error and "per_channel" in r.data:
                names = [int(i) for i, data in r.data["per_channel"].items() if data.get("warn")]
                if len(names) == 1 and track.channels == 2:
                    value += " · " + ("Left", "Right")[names[0]]
                else:
                    value += f" · {len(names)} channels"
            compact.append(value)
        row["compact_readings"] = [(readings[0][0], *compact)]
    if counts:
        row["location"], row["count"] = counts, ""
    if check.outcome == CheckOutcome.RESOLVED:
        row["hint"] = ""
    elif partial:
        row["hint"] = partial
    elif check.outcome == CheckOutcome.FAILED:
        row["hint"] = check.corrected.error if check.corrected else ""


def _resolve_paired_check(
    det_id: str, result, *, track, original_track, detector, config: dict, pending: bool
):
    """Resolve paired before/after verification check for a track detector row."""
    if not config.get(f"{det_id}_enabled", True) or getattr(detector, "_report_as", "") == "skip":
        return None, None, True
    check = detector_check(track, det_id, pending=pending)
    is_relevant = getattr(detector, "is_relevant", None)
    if check.corrected is not None and is_relevant and not is_relevant(check.corrected, track):
        if check.original is None or not is_relevant(check.original, original_track):
            return None, None, True
        check = replace(
            check,
            outcome=CheckOutcome.INAPPLICABLE,
            corrected=replace(
                check.corrected, applicable=False, summary="Inactive for this normalization method"
            ),
        )
    if check.outcome in {CheckOutcome.NEW, CheckOutcome.REMAINING}:
        result = check.corrected
    return check, result, False


def _build_finding_row_dict(
    row_id: str, det_id: str, result, *, is_local: bool, track, detector, repairs, fragment: str, severity: str
) -> dict:
    location, locations = _location_text(result)
    repair_target = repairs.targets.get(row_id)
    if det_id == "dc_offset" and result.data.get("dc_warn"):
        location = "Whole track"
    runs = result.data.get("runs")
    count = f'{runs} {"range" if runs == 1 else "ranges"}' if isinstance(runs, int) else ""
    help_html = getattr(detector, "html_help", "") if detector else ""
    if callable(help_html):
        help_html = help_html()
    title = detector.name if detector else det_id
    return {
        "id": row_id,
        "title": f"{title} · local energy" if is_local else title,
        "severity": severity,
        "summary": result.summary,
        "fragment": fragment,
        "location": location,
        "locations": locations,
        "count": count,
        "hint": _guidance(result.hint),
        "review_repair": bool(repair_target and not result.error),
        "repair_attribution": repairs.attribution(row_id),
        "help": help_html,
        "visualize_labels": tuple(sorted({issue.label for issue in visualizable_issues(result, track)})),
    }


def _prepare_finding_row(
    row_id: str,
    det_id: str,
    result,
    local_check,
    *,
    track,
    original_track,
    detector,
    repairs,
    config: dict,
    pending: bool,
    show_clean: bool,
) -> dict | None:
    is_local = local_check is not None
    check = None
    paired = track.repair_recipe.active or pending or track._cache.get("repair_error")
    if paired and not is_local and (
        isinstance(detector, TrackDetector)
        or (track.corrected_analysis and det_id in track.corrected_analysis.results)
    ):
        check, result, skip = _resolve_paired_check(
            det_id,
            result,
            track=track,
            original_track=original_track,
            detector=detector,
            config=config,
            pending=pending,
        )
        if skip:
            return None

    severity = (
        detector.effective_severity(result)
        if detector and hasattr(detector, "effective_severity")
        else result.severity
    )
    if severity is None:
        return None
    severity = getattr(severity, "value", str(severity))
    if not show_clean and severity == "clean" and not (check and check.corrected and check.corrected.error):
        return None
    if detector:
        context = original_track if check and result is check.original else track
        fragment = detector.render_html(result, context)
        if not fragment:
            return None
    else:
        fragment = f"<tr><td>{esc(result.summary)}</td></tr>"

    row = _build_finding_row_dict(
        row_id,
        det_id,
        result,
        is_local=is_local,
        track=track,
        detector=detector,
        repairs=repairs,
        fragment=fragment,
        severity=severity,
    )
    if check:
        _attach_check_details(row, check, det_id, track, config)
    elif is_local:
        row["outcome"] = local_check.outcome
        row["source_labels"] = tuple(
            tuple(sorted({i.label for i in visualizable_issues(r, track)}))
            for r in (local_check.original, local_check.corrected)
        )
    return row


def prepare_track_report(track, session=None, *, show_clean: bool = True,
                         verbose: bool = False) -> dict:
    """Prepare cached per-track report data."""
    parts: list = []
    rows: list[dict] = []
    repairs = track_repair_options(track, session) if track.status == "OK" else None
    parts.append(
        f'<div style="color:{COLORS["heading"]}; font-size:{REPORT_TITLE_SIZE}; font-weight:600;">'
        f"{esc(track.filename)}</div>"
    )
    if track.status != "OK":
        parts.append(
            f'<div style="color:{COLORS["problems"]}; margin-top:8px;">Status: {esc(track.status)}</div>'
        )
    else:
        _append_track_metadata_parts(parts, track)
        proc_map = {p.id: p for p in (session.processors if session else [])}
        det_map = {d.id: d for d in (session.detectors if session else [])}
        _append_processor_parts(parts, track, proc_map, verbose=verbose)

        severity_order = {"problem": 0, "attention": 1, "information": 2, "info": 2, "clean": 3}
        pending = bool(track._cache.get("repair_verification_pending"))
        config = session.config if session else {}
        original_track = original_report_track(track, session)
        for row_id, det_id, result, local_check in _track_findings(track, config):
            row = _prepare_finding_row(
                row_id,
                det_id,
                result,
                local_check,
                track=track,
                original_track=original_track,
                detector=det_map.get(det_id),
                repairs=repairs,
                config=config,
                pending=pending,
                show_clean=show_clean,
            )
            if row is not None:
                rows.append(row)
        rows.sort(
            key=lambda r: (
                r.get("outcome") == CheckOutcome.RESOLVED,
                severity_order.get(r["severity"], 99),
                r["title"].lower(),
            )
        )
    wide = [p if isinstance(p, str) else p[0] for p in parts]
    compact = [p if isinstance(p, str) else p[1] for p in parts]
    narrow = [p if isinstance(p, str) else p[2] for p in parts]
    data = {
        "header": "\n".join(wide),
        "header_compact": "\n".join(compact),
        "header_narrow": "\n".join(narrow),
        "rows": rows,
        "can_review_repairs": bool(repairs and repairs.available),
        "source_failure": track._cache.get("repair_error"),
    }
    select_report_source(data, False)
    return data
