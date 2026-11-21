"""Stateless helpers for session groups, color tinting, and gain-link maps."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QIcon, QPixmap

from ..state import WorkspaceState
from ..theme import PT_DEFAULT_COLORS, argb_to_qcolor as _argb_to_qcolor, normalize_color_name

GROUP_NONE_LABEL = "(None)"
LINK_INDICATOR = " 🔗"
_TINT_FACTOR = 0.15


def color_names_from_config(config: dict[str, Any]) -> list[str]:
    colors = config.get("colors", PT_DEFAULT_COLORS)
    return [c["name"] for c in colors if c.get("name")]


def color_argb_by_name(config: dict[str, Any], name: str) -> str | None:
    colors = config.get("colors", PT_DEFAULT_COLORS)
    normalized = normalize_color_name(name, colors)
    for palette in (colors, PT_DEFAULT_COLORS):
        for entry in palette:
            if entry.get("name") == normalized:
                return entry.get("argb")
    return None


def color_swatch_icon(argb: str, size: int = 16) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(_argb_to_qcolor(argb))
    return QIcon(pm)


def group_color_map(state: WorkspaceState) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in state.session_groups:
        argb = color_argb_by_name(state.config, group.get("color", ""))
        if argb:
            result[group["name"]] = argb
    return result


def tint_group_color(
    state: WorkspaceState,
    group_name: str | None,
    gcm: dict[str, str] | None = None,
) -> QColor | None:
    color_map = group_color_map(state) if gcm is None else gcm
    argb = color_map.get(group_name) if group_name else None
    if not argb:
        return None
    qc = _argb_to_qcolor(argb)
    alpha = (qc.alpha() / 255.0) * _TINT_FACTOR
    bg_r, bg_g, bg_b = 0x1E, 0x1E, 0x1E
    return QColor(
        int(qc.red() * alpha + bg_r * (1 - alpha)),
        int(qc.green() * alpha + bg_g * (1 - alpha)),
        int(qc.blue() * alpha + bg_b * (1 - alpha)),
    )


def apply_row_group_color(
    table: Any,
    state: WorkspaceState,
    row: int,
    group_name: str | None,
    gcm: dict[str, str] | None = None,
) -> None:
    table.apply_row_color(row, tint_group_color(state, group_name, gcm))


def normalize_session_groups(
    config: dict[str, Any], groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    colors = config.get("colors", PT_DEFAULT_COLORS)
    normalized: list[dict[str, Any]] = []
    for group in groups:
        name = str(group.get("name", "")).strip()
        if not name:
            continue
        normalized.append({
            "name": name,
            "color": normalize_color_name(group.get("color", ""), colors),
            "gain_linked": bool(group.get("gain_linked", False)),
            "daw_target": group.get("daw_target", ""),
            "match_method": group.get("match_method", "contains"),
            "match_pattern": group.get("match_pattern", ""),
        })
    return normalized


def group_combo_items(state: WorkspaceState) -> list[str]:
    return [GROUP_NONE_LABEL] + [g["name"] for g in state.session_groups]


def gain_linked_map(state: WorkspaceState) -> dict[str, bool]:
    return {g["name"]: g.get("gain_linked", False) for g in state.session_groups}


def group_display_name(
    state: WorkspaceState, name: str, glm: dict[str, bool] | None = None
) -> str:
    linked_map = gain_linked_map(state) if glm is None else glm
    return name + LINK_INDICATOR if linked_map.get(name, False) else name


def group_rank_map(state: WorkspaceState) -> dict[str, int]:
    return {g["name"]: i for i, g in enumerate(state.session_groups)}
