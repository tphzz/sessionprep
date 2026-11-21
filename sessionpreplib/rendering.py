"""Plain-text rendering helpers (and backward-compatible summary re-export)."""

from __future__ import annotations

from typing import Any

from .summary import build_diagnostic_summary as build_diagnostic_summary


def render_diagnostic_summary_text(summary: dict[str, Any]) -> str:
    """Render the diagnostic summary as plain text (for sessionprep.txt)."""
    problems = summary.get("problems") or []
    attention = summary.get("attention") or []
    information = summary.get("information") or []
    clean = summary.get("clean") or []
    clean_count = int(summary.get("clean_count", 0))
    total_ok = int(summary.get("total_ok", 0))

    def item_count(groups):
        return sum(len(g.get("items") or []) for g in groups)

    def render_groups(groups, compact=False):
        lines = []
        for g in groups:
            title = g.get("title")
            hint = g.get("hint")
            items = g.get("items") or []
            if not items and not g.get("standalone"):
                continue
            header = f"{title} -> {hint}" if hint else f"{title}"
            lines.append(f"   - {header}")
            for item in items:
                lines.append(f"     * {item}")
            if not compact:
                lines.append("")
        return lines

    sections = [
        (f"\U0001f534 PROBLEMS ({item_count(problems)})", problems, False),
        (f"\U0001f7e1 ATTENTION ({item_count(attention)})", attention, False),
        (f"\U0001f535 INFORMATION ({item_count(information)})", information, False),
        ("\U0001f7e2 CLEAN", clean, True),
    ]
    lines = [f"\U0001f9fe Session Health Summary: {clean_count}/{total_ok} file(s) CLEAN"]
    for heading, groups, compact in sections:
        lines.extend(["", heading])
        rendered = render_groups(groups, compact=compact)
        lines.extend(rendered if rendered else ["   - None"])
    return "\n".join(lines)
