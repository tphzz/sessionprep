"""Color Picker tool for Pro Tools."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.daw_processors import ptsl_helpers as ptslh

from ...widgets import ColorGridPanel
from .connection_common import PTSL_MUTATION_TIMEOUT_MS, PTSL_READ_TIMEOUT_MS


class ColorTool(QWidget):
    """Interactive color picker that pushes colors to Pro Tools."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._client = None
        self._pt_palette: list[str] = []
        self._init_ui()
        self._load_palette()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._desc = QLabel(
            "Click a color to apply it to the selected track(s) in Pro Tools. "
            "Colors are perceptually matched to the Pro Tools palette."
        )
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: #aaa; font-size: 9pt; margin-bottom: 6px;")
        layout.addWidget(self._desc)

        self._grid = ColorGridPanel(
            cell_height=28,
            stretch_vertical=True,
            max_cell_height_to_width=1.0,
            parent=self,
        )
        self._grid.colorClicked.connect(self._on_color_clicked)
        layout.addWidget(self._grid)
        layout.addStretch(1)

        status_row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 8pt;")
        status_row.addWidget(self._status)
        status_row.addStretch()
        layout.addLayout(status_row)

    def preferred_compact_height_for_width(
        self,
        width: int,
        *,
        use_minimum_grid_height: bool = False,
    ) -> int:
        """Return a compact height that keeps the palette cells square-to-wide."""
        layout = self.layout()
        if layout is None:
            return self.sizeHint().height()
        margins = layout.contentsMargins()
        spacing = layout.spacing()
        inner_width = max(1, width - margins.left() - margins.right())
        desc_height = (
            self._desc.heightForWidth(inner_width)
            if self._desc.hasHeightForWidth()
            else self._desc.sizeHint().height()
        )
        grid_height = (
            self._grid.minimum_grid_height()
            if use_minimum_grid_height
            else self._grid.aspect_limited_height_for_width(inner_width)
        )
        status_height = self._status.sizeHint().height()
        return (
            margins.top()
            + margins.bottom()
            + desc_height
            + grid_height
            + status_height
            + spacing * 3
        )

    def set_engine(self, engine):
        """Compatibility shim for older callers."""
        self.set_client(engine)

    def set_client(self, client):
        """Set or clear the Pro Tools worker client."""
        self._client = client
        self._pt_palette = []

    def update_config(self, config: dict):
        """Refresh the palette grid from an updated config."""
        self._config = config
        self._load_palette()

    def _load_palette(self):
        colors = self._config.get("colors", [])
        self._grid.set_colors(colors)

    def _set_status(self, text: str, color: str):
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 8pt;")

    def _fetch_pt_palette(self):
        if self._client is None:
            return
        self._client.request(
            "get_color_palette",
            {"target": "CPTarget_Tracks"},
            self._on_palette_response,
            timeout_ms=PTSL_READ_TIMEOUT_MS,
        )

    def _on_palette_response(self, response: dict):
        if not response.get("ok"):
            self._set_status(
                f"Failed to fetch PT palette: {response.get('error') or ''}",
                "#f44336",
            )
            return
        self._pt_palette = list(response.get("result") or [])
        count = len(self._pt_palette)
        if count:
            self._set_status(f"PT palette loaded ({count} colors)", "#4caf50")
        else:
            self._set_status("PT palette empty", "#ff9800")

    def _on_color_clicked(self, index: int):
        if self._client is None:
            self._set_status("Not connected to Pro Tools", "#f44336")
            return

        colors = self._config.get("colors", [])
        if index < 0 or index >= len(colors):
            return

        entry = colors[index]
        argb = entry.get("argb", "")
        name = entry.get("name", "")
        if not argb:
            return

        if not self._pt_palette:
            self._set_status("Loading Pro Tools palette...", "#aaa")
            self._client.request(
                "get_color_palette",
                {"target": "CPTarget_Tracks"},
                lambda response, idx=index: self._on_palette_then_apply(
                    response,
                    idx,
                ),
                timeout_ms=PTSL_READ_TIMEOUT_MS,
            )
            return

        self._apply_color(argb, name)

    def _on_palette_then_apply(self, response: dict, index: int):
        self._on_palette_response(response)
        if not self._pt_palette:
            return
        colors = self._config.get("colors", [])
        if index < 0 or index >= len(colors):
            return
        entry = colors[index]
        self._apply_color(entry.get("argb", ""), entry.get("name", ""))

    def _apply_color(self, argb: str, name: str):
        pt_index = ptslh.closest_palette_index(argb, self._pt_palette)
        if pt_index is None:
            self._set_status("Could not match color", "#f44336")
            return

        self._set_status("Applying color...", "#aaa")
        self._client.request(
            "get_selected_track_names",
            {},
            lambda response, color_index=pt_index + 1, label=name or argb: (
                self._on_selected_tracks_for_color(response, color_index, label)
            ),
            timeout_ms=PTSL_READ_TIMEOUT_MS,
        )

    def _on_selected_tracks_for_color(
        self,
        response: dict,
        color_index: int,
        label: str,
    ):
        if not response.get("ok"):
            self._set_status(f"Error: {response.get('error') or ''}", "#f44336")
            return
        selected = list(response.get("result") or [])
        if not selected:
            self._set_status("No tracks selected in Pro Tools", "#ff9800")
            return
        self._client.request(
            "set_track_color",
            {"color_index": color_index, "track_names": selected},
            lambda apply_response, count=len(selected), text=label: (
                self._on_color_applied(apply_response, count, text)
            ),
            timeout_ms=PTSL_MUTATION_TIMEOUT_MS,
        )

    def _on_color_applied(self, response: dict, count: int, label: str):
        if not response.get("ok"):
            self._set_status(f"Error: {response.get('error') or ''}", "#f44336")
            return
        self._set_status(
            f"Applied '{label}' ({count} track{'s' if count != 1 else ''})",
            "#4caf50",
        )
