"""HTML report rendering for the GUI."""

from __future__ import annotations

from numbers import Integral
from urllib.parse import quote
from ..icons import icon_path

from sessionpreplib.chunks import read_chunks, STANDARD_CHUNKS, detect_origin
from ..theme import COLORS, FILE_COLOR_TRANSIENT, FILE_COLOR_SUSTAINED
from ..helpers import esc
from .report_style import REPORT_SECTION_SIZE, REPORT_TITLE_SIZE


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
    ranges = [(i.sample_start, i.sample_end) for i in result.issues]
    if not ranges:
        ranges = [r[:2] for r in (result.data.get("ranges") or [])
                  if isinstance(r, (tuple, list)) and len(r) >= 2]
    valid = []
    for start, end in ranges:
        try:
            start = int(start)
            end = start if end is None else int(end)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= start <= end:
            value = f"Samples {start}\u2013{end}" if end != start else f"Sample {start}"
            if value not in valid:
                valid.append(value)
    if not valid:
        return "\u2014", ""
    return (valid[0] if len(valid) == 1 else f"{len(valid)} locations",
            "; ".join(valid))


def _image(name):
    return f'<img src="{esc(icon_path(name).as_uri())}" width="16" height="16">'


def _link(scheme, key, label, title=""):
    target = scheme + ":" + quote(str(key), safe="")
    return (f'<a href="{esc(target)}" title="{esc(title)}" '
            f'style="color:{COLORS["information"]}; text-decoration:none;">{label}</a>')


def _cell(body, width, index=0, header=False):
    left = 12 if index == 0 else 6
    right = 12 if index == 3 else 6
    padding = 4 if header else 8
    background = " background-color:#2d2d2d;" if header else ""
    return (f'<td width="{width}%" valign="top" style="padding:{padding}px {right}px '
            f'{padding}px {left}px; border-bottom:1px solid #3a3a3a;'
            f' color:{COLORS["text"]};{background}">'
            f'<p style="margin-top:0; margin-bottom:0; line-height:120%;">{body}</p></td>')


def _table_header(labels, widths):
    cells = [_cell(esc(label), width, i, True)
             for i, (label, width) in enumerate(zip(labels, widths))]
    return ('<table width="100%" cellpadding="0" cellspacing="0" '
            'style="border-collapse:collapse;">'
            '<tr>' + "".join(cells) + '</tr>')


def render_summary_html(summary, *, show_faders=True, show_clean=True, session=None):
    """Render compact findings while keeping file and finding counts distinct."""
    categories = ("problems", "attention", "information", "clean")
    counts = {key: sum(max(len(g.get("items") or []), int(bool(g.get("standalone"))))
                       for g in summary.get(key) or []) for key in categories}
    clean, total = int(summary.get("clean_count", 0)), int(summary.get("total_ok", 0))
    parts = [f'<p style="font-size:{REPORT_TITLE_SIZE}; font-weight:bold;'
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
    widths = (22.13, 22.72, 42.62, 12.53)
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
                     f' font-weight:bold; margin-bottom:6px;">'
                     f'{label if category != "problems" else "Problems"} &middot; '
                     f'{count} {"finding" if count == 1 else "findings"}</p>')
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
                result = track.detector_results.get(group.get("detector_id")) if track else None
                if result is not None and group.get("detector_id") == "clipping":
                    runs = result.data.get("runs")
                    if isinstance(runs, int):
                        detail = f'{runs} clipped {"range" if runs == 1 else "ranges"}'
                finding = f'<b>{esc(group.get("title", ""))}</b><br><span style="color:{color};">{label}</span>'
                file_text = (f'<b>{esc(filename)}</b><br>' if filename else "")
                file_text += f'<span style="color:{COLORS["metadata"]};">{esc(detail)}</span>'
                action = _link("report-review", filename, "Review file", f"Review {filename}") if filename else ""
                values = (finding, file_text, esc(_guidance(group.get("hint"))), action)
                parts.append('<tr>' + "".join(_cell(value, width, i)
                             for i, (value, width) in enumerate(zip(values, widths))) + '</tr>')
        parts.append("</table>")
    if not any(counts[k] for k in categories if k != "clean" or show_clean):
        parts.append(f'<p style="color:{COLORS["metadata"]};">No findings to show.</p>')
    return "\n".join(parts)


def _finding_line(body, height):
    # Qt supports table height, but not CSS min-height on report cells.
    return (f'<table height="{height}" cellpadding="0" cellspacing="0"><tr>'
            f'<td valign="middle">{body}</td></tr></table>')


def _finding_cell(body, width, *, header=False, active=False, index=0, columns=1):
    size = f' width="{width}"' if width is not None else ""
    background = COLORS["surface"] if header or active else COLORS["bg"]
    padding = 4 if header else 8
    left = 12 if index == 0 else 6
    right = 12 if index == columns - 1 else 6
    return (f'<td{size} valign="top" style="padding:{padding}px {right}px {padding}px {left}px;'
            f' border-bottom:1px solid {COLORS["accent"]};'
            f' background-color:{background}; color:{COLORS["text"]};">'
            f'{body}</td>')


def visualizable_issues(result, track):
    """Return existing locations that can safely be shown in this waveform."""
    if result is None or result.error or track.total_samples <= 0 or track.channels <= 0:
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
    if row.get("visualize_labels"):
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


def render_prepared_track_report(data, *, expanded=(), compact=False, active_help=None):
    """Render cached data with severity-leading rows; Help lives in its own pane."""
    parts = [data["header"],
             '<div>'
             '<p style="font-weight:bold; margin-top:20px; margin-bottom:8px;">'
             '<a name="report-findings"></a>Detector findings</p>']
    rows = data["rows"]
    if not rows:
        parts.append(f'<p style="color:{COLORS["metadata"]};">No findings to show.</p></div>')
        return "\n".join(parts)
    widths = ("100", None, "80") if compact else ("100", None, "96", "216", "136")
    labels = (("Severity", "Finding, location and next step", "Actions") if compact
              else ("Severity", "Finding", "Location", "Next step", "Actions"))
    parts.append('<table width="100%" cellpadding="0" cellspacing="0"'
                 ' style="border-collapse:collapse;"><tr>')
    parts.extend(_finding_cell("<nobr>Help</nobr>" if label == "Help" else esc(label),
                               width, header=True, index=i, columns=len(widths))
                 for i, (label, width) in enumerate(zip(labels, widths)))
    parts.append("</tr>")
    for row in rows:
        key = row["id"]
        token = quote(key, safe="")
        color, label = _severity_meta(row["severity"])
        arrow = _image("chevron-down" if key in expanded else "chevron-right")
        title = _link("report-details", key, f'{arrow}&nbsp;&nbsp; <b>{esc(row["title"])}</b>',
                      "Toggle detector details")
        if row["count"]:
            title += f' &middot; <span style="color:{COLORS["metadata"]};">{esc(row["count"])}</span>'
        finding = f'<a name="finding-{token}"></a>' + _finding_line(title, 32 if compact else 40)
        if compact:
            finding += (f'<p style="margin:4px 0 0 24px; color:{COLORS["metadata"]};">'
                        f'{esc(row["location"])}<br>'
                        f'<span style="color:{COLORS["text"]};">{esc(row["hint"])}</span></p>')
        severity = _finding_line(f'<span style="color:{color};">{label}</span>',
                                 32 if compact else 40)
        actions = _finding_actions(row, compact)
        values = ((severity, finding, actions) if compact else
                  (severity, finding, _finding_line(esc(row["location"]), 40),
                   _finding_line(esc(row["hint"]), 40), actions))
        parts.append("<tr>" + "".join(
            _finding_cell(value, width, active=key == active_help and color != COLORS["problems"],
                          index=i, columns=len(widths))
            for i, (value, width) in enumerate(zip(values, widths))) + "</tr>")
        if key in expanded:
            parts.append(f'<tr><td colspan="{len(widths)}" style="padding:4px 12px 8px 136px;">'
                         '<b>Result</b> ' + esc(row["summary"]) +
                         ('<br>' + esc(row["locations"]) if row["locations"] else "") +
                         '<table cellpadding="3" cellspacing="0">' + row["fragment"] +
                         '</table></td></tr>')
    parts.append("</table></div>")
    return "\n".join(parts)


def render_track_detail_html(track, session=None, *, show_clean=True, verbose=False):
    """Render a static compact view; interactive browsers cache prepared data."""
    return render_prepared_track_report(
        prepare_track_report(track, session, show_clean=show_clean, verbose=verbose))


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

def prepare_track_report(track, session=None, *, show_clean: bool = True,
                             verbose: bool = False) -> dict:
    """Prepare cached per-track report data.

    Parameters
    ----------
    track : TrackContext
    session : SessionContext | None
        If provided, configured detector/processor instances from the
        session are used for self-rendering via their ``render_html()``
        methods.
    show_clean : bool
        If False, detectors with severity ``clean`` are hidden.
    verbose : bool
        When True, processors may include additional analytical detail.
    """
    parts = []
    rows = []
    parts.append(f'<div style="color:{COLORS["heading"]}; font-size:{REPORT_TITLE_SIZE}; font-weight:bold;">'
                 f'{esc(track.filename)}</div>')

    if track.status != "OK":
        parts.append(f'<div style="color:{COLORS["problems"]}; margin-top:8px;">'
                     f'Status: {esc(track.status)}</div>')
    else:
        # File info
        fmt = f"{track.samplerate} Hz / {track.bitdepth} / {track.channels}ch"
        dur = f"{track.duration_sec:.2f}s ({track.total_samples} samples)"
        parts.append(f'<div style="color:{COLORS["metadata"]}; margin-top:6px;">{fmt}</div>')
        parts.append(f'<div style="color:{COLORS["metadata"]};">{dur}</div>')

        # Chunk metadata + DAW origin (single line)
        try:
            _container, all_chunks = read_chunks(track.filepath)
            notable = [ch for ch in all_chunks if ch.id not in STANDARD_CHUNKS]
        except (ValueError, OSError):
            notable = []
        origin = detect_origin(track.chunk_ids, track.filepath)
        meta_parts = []
        if notable:
            def _fmt_size(n: int) -> str:
                if n < 1024:
                    return f"{n} B"
                if n < 1024 * 1024:
                    return f"{n / 1024:.1f} KB"
                return f"{n / (1024 * 1024):.1f} MB"
            chunk_parts = [
                f'{esc(ch.id.strip())} ({_fmt_size(ch.size)})'
                for ch in notable
            ]
            meta_parts.append(f'Chunks: {" &middot; ".join(chunk_parts)}')
        if origin:
            meta_parts.append(f'Origin: {esc(origin)}')
        if meta_parts:
            parts.append(
                f'<div style="color:{COLORS["metadata"]}; margin-top:4px;">'
                f'{" / ".join(meta_parts)}</div>'
            )

        # Build lookup maps from session instances
        proc_map = {p.id: p for p in (session.processors if session else [])}
        det_map = {d.id: d for d in (session.detectors if session else [])}

        # Processor results
        for proc_id, pr in (track.processor_results or {}).items():
            proc_inst = proc_map.get(proc_id)
            if proc_inst:
                html_fragment = proc_inst.render_html(pr, track, verbose=verbose)
                if not html_fragment:
                    continue  # processor chose to suppress output (e.g. no-op)
            else:
                # Fallback: basic display
                html_fragment = (
                    f'<div style="margin-left:8px;">'
                    f'{esc(pr.classification or "")} &middot; {esc(pr.method)} '
                    f'&middot; {pr.gain_db:+.1f} dB</div>'
                )
            parts.append(f'<div style="margin-top:12px; color:{COLORS["heading"]}; '
                         f'font-weight:bold;">{esc(proc_map[proc_id].name if proc_id in proc_map else proc_id)}</div>')
            parts.append(html_fragment)
            if track.group:
                original = pr.data.get("original_gain_db", pr.gain_db)
                delta = pr.gain_db - original
                parts.append(
                    f'<div style="margin-left:8px; margin-top:4px;">'
                    f'Group: <b>{esc(track.group)}</b>'
                    f' &nbsp;&middot;&nbsp; Group level: <b>{pr.gain_db:+.1f} dB</b>'
                    f' (individual: {original:+.1f} dB,'
                    f' &Delta; {delta:+.1f} dB)</div>'
                )

        severity_order = {"problem": 0, "attention": 1, "information": 2, "info": 2, "clean": 3}
        for det_id, result in track.detector_results.items():
            detector = det_map.get(det_id)
            severity = (detector.effective_severity(result)
                        if detector and hasattr(detector, "effective_severity")
                        else result.severity)
            if severity is None:
                continue
            severity = getattr(severity, "value", str(severity))
            if not show_clean and severity == "clean":
                continue
            if detector:
                fragment = detector.render_html(result, track)
                if not fragment:
                    continue
            else:
                fragment = f'<tr><td>{esc(result.summary)}</td></tr>'
            location, locations = _location_text(result)
            runs = result.data.get("runs")
            count = f'{runs} {"range" if runs == 1 else "ranges"}' if isinstance(runs, int) else ""
            help_html = getattr(detector, "html_help", "") if detector else ""
            if callable(help_html):
                help_html = help_html()
            rows.append({
                "id": det_id, "title": detector.name if detector else det_id,
                "severity": severity, "summary": result.summary, "fragment": fragment,
                "location": location, "locations": locations, "count": count,
                "hint": _guidance(result.hint),
                "help": help_html,
                "visualize_labels": tuple(sorted({issue.label
                    for issue in visualizable_issues(result, track)})),
            })
        rows.sort(key=lambda row: (severity_order.get(row["severity"], 99), row["title"].lower()))
    return {"header": "\n".join(parts), "rows": rows}
