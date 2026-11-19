"""Pure helper functions used across the GUI — no widget dependencies."""

from __future__ import annotations

from .theme import COLORS
from sessionpreplib.repair_verification import remaining_results


# ---------------------------------------------------------------------------
# Severity ranking
# ---------------------------------------------------------------------------

SEVERITY_RANK = {"problem": 0, "attention": 1, "information": 2, "clean": 3}
SEVERITY_LABELS = {
    "problem": ("PROBLEMS", COLORS["problems"]),
    "attention": ("ATTENTION", COLORS["attention"]),
    "information": ("OK", COLORS["information"]),
    "clean": ("OK", COLORS["clean"]),
}

_SEV_COLOR = {
    "problem": COLORS["problems"],
    "attention": COLORS["attention"],
    "information": COLORS["information"],
}
_SEV_LETTER = {
    "problem": "P",
    "attention": "A",
    "information": "I",
}


def track_analysis_label(track, detectors=None) -> tuple[str, str, str, int]:
    """Return severity-count label for a track's detector results.

    Returns ``(plain_text, html_text, worst_color, sort_key)`` where:

    - *plain_text*: e.g. ``"2P 1A 5I"`` or ``"OK"``
    - *html_text*: rich-text with per-severity coloring
    - *worst_color*: hex color of the worst severity (for fallback)
    - *sort_key*: numeric key (lower = worse) for table sorting

    Parameters
    ----------
    track : TrackContext
    detectors : list | None
        If provided, each detector's ``is_relevant()`` and
        ``effective_severity()`` are used to filter and remap results.
    """
    if track.status != "OK":
        err_html = f'<span style="color:{COLORS["problems"]}">Error</span>'
        return "Error", err_html, COLORS["problems"], 0

    det_map = {d.id: d for d in detectors} if detectors else {}

    counts = {"problem": 0, "attention": 0, "information": 0}
    pending = bool(track._cache.get("repair_verification_pending") or
                   (track.repair_recipe.active and track.corrected_analysis is None))
    failed = bool(track._cache.get("repair_error") or
                  (track.corrected_analysis and any(r.error for r in track.corrected_analysis.results.values())))
    for det_id, result in remaining_results(track, detectors or [], pending=pending).items():
        det_inst = det_map.get(det_id)
        if det_inst and hasattr(det_inst, 'is_relevant') and not det_inst.is_relevant(result, track):
            continue
        if det_inst and hasattr(det_inst, 'effective_severity'):
            eff = det_inst.effective_severity(result)
            if eff is None:
                continue
            sev = eff.value
        else:
            sev = result.severity.value if hasattr(result.severity, "value") else str(result.severity)
        if sev == "info":
            sev = "information"
        if sev in counts:
            counts[sev] += 1

    # Build label parts (omit zero counts)
    parts_plain = []
    parts_html = []
    for sev_key in ("problem", "attention", "information"):
        c = counts[sev_key]
        if c > 0:
            letter = _SEV_LETTER[sev_key]
            color = _SEV_COLOR[sev_key]
            parts_plain.append(f"{c}{letter}")
            parts_html.append(
                f'<span style="color:{color}; font-weight:bold;">{c}{letter}</span>')

    if pending or failed:
        label = "Unable to verify" if failed else "Checking…"
        color = COLORS["attention" if failed else "metadata"]
        parts_html.append(f'<span style="color:{color}">{label}</span>')
        if not parts_plain:
            return label, parts_html[-1], color, 2999
        parts_plain.append(label)

    if parts_plain:
        plain = " ".join(parts_plain)
        html = "&nbsp;".join(parts_html)
        # Sort key: worst severity rank * 1000 + total non-clean count
        # Lower = worse (problems first)
        worst_rank = 3  # clean
        for sev_key in ("problem", "attention", "information"):
            if counts[sev_key] > 0:
                worst_rank = min(worst_rank, SEVERITY_RANK[sev_key])
                break
        total = sum(counts.values())
        sort_key = worst_rank * 1000 - total
        worst_color = _SEV_COLOR.get(
            next(s for s in ("problem", "attention", "information") if counts[s] > 0),
            COLORS["clean"])
        return plain, html, worst_color, sort_key

    plain = "OK"
    html = f'<span style="color:{COLORS["clean"]}; font-weight:bold;">OK</span>'
    return plain, html, COLORS["clean"], 3000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
