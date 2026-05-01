"""Track height preset tool for Pro Tools."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...prefs.preset_panel import NamedPresetPanel
from ...settings import save_config
from .connection_common import PTSL_MUTATION_TIMEOUT_MS, PTSL_READ_TIMEOUT_MS


COL_TYPE = 0
COL_MATCHED = 1
COL_CURRENT = 2
COL_PRESET = 3

SCOPE_CHOICES = [
    ("all", "All tracks"),
    ("visible", "All visible tracks"),
    ("selected", "Selected tracks"),
]

MODE_CHOICES = [
    ("all", "All"),
    ("track_type", "Track Type"),
]

HEIGHT_CHOICES = [
    ("", "Leave unchanged"),
    ("THeight_Micro", "Micro"),
    ("THeight_Mini", "Mini"),
    ("THeight_Small", "Small"),
    ("THeight_Medium", "Medium"),
    ("THeight_Large", "Large"),
    ("THeight_Jumbo", "Jumbo"),
    ("THeight_Extreme", "Extreme"),
    ("THeight_FitToWindow", "Fit to Window"),
]

TRACK_TYPES = [
    ("TT_Audio", "Audio"),
    ("TT_Aux", "Aux"),
    ("TT_Instrument", "Instrument"),
    ("TT_Midi", "MIDI"),
    ("TT_Master", "Master"),
    ("TT_Vca", "VCA"),
    ("TT_BasicFolder", "Basic Folder"),
    ("TT_RoutingFolder", "Routing Folder"),
    ("TT_Video", "Video"),
    ("TT_Tempo", "Tempo"),
    ("TT_Markers", "Markers"),
    ("TT_Meter", "Meter"),
    ("TT_KeySignature", "Key Signature"),
    ("TT_ChordSymbols", "Chord Symbols"),
    ("TT_CompLane", "Comp Lane"),
]

_INT_TRACK_TYPES = {
    1: "TT_Midi",
    2: "TT_Audio",
    3: "TT_Aux",
    4: "TT_Video",
    5: "TT_Vca",
    6: "TT_Tempo",
    7: "TT_Markers",
    8: "TT_Meter",
    9: "TT_KeySignature",
    10: "TT_ChordSymbols",
    11: "TT_Instrument",
    12: "TT_Master",
    14: "TT_BasicFolder",
    15: "TT_RoutingFolder",
    16: "TT_CompLane",
}

_TYPE_ALIASES = {
    "TType_Midi": "TT_Midi",
    "TType_Audio": "TT_Audio",
    "TType_Aux": "TT_Aux",
    "TType_Video": "TT_Video",
    "TType_Vca": "TT_Vca",
    "TType_Tempo": "TT_Tempo",
    "TType_Markers": "TT_Markers",
    "TType_Meter": "TT_Meter",
    "TType_KeySignature": "TT_KeySignature",
    "TType_ChordSymbols": "TT_ChordSymbols",
    "TType_Instrument": "TT_Instrument",
    "TType_Master": "TT_Master",
    "TType_BasicFolder": "TT_BasicFolder",
    "TType_RoutingFolder": "TT_RoutingFolder",
    "TType_CompLane": "TT_CompLane",
    "Audio": "TT_Audio",
    "Aux": "TT_Aux",
    "Instrument": "TT_Instrument",
    "MIDI": "TT_Midi",
    "Midi": "TT_Midi",
    "Master": "TT_Master",
    "VCA": "TT_Vca",
    "Vca": "TT_Vca",
    "BasicFolder": "TT_BasicFolder",
    "RoutingFolder": "TT_RoutingFolder",
}


class _NumericItem(QTableWidgetItem):
    """Table item that sorts by numeric value stored in UserRole."""

    def __lt__(self, other):
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if isinstance(left, int) and isinstance(right, int):
            return left < right
        return super().__lt__(other)


def _default_preset() -> dict[str, Any]:
    return {
        "scope": "all",
        "mode": "track_type",
        "all_height": "THeight_Small",
        "heights": {
            "TT_Audio": "THeight_Small",
            "TT_Aux": "THeight_Small",
            "TT_Instrument": "THeight_Small",
            "TT_Midi": "THeight_Small",
            "TT_Master": "THeight_Medium",
            "TT_Vca": "THeight_Medium",
            "TT_BasicFolder": "THeight_Medium",
            "TT_RoutingFolder": "THeight_Medium",
            "TT_Video": "THeight_Mini",
            "TT_Tempo": "THeight_Mini",
            "TT_Markers": "THeight_Mini",
            "TT_Meter": "THeight_Mini",
            "TT_KeySignature": "THeight_Mini",
            "TT_ChordSymbols": "THeight_Mini",
            "TT_CompLane": "THeight_Mini",
        },
    }


def _normalize_track_type(value: Any) -> str:
    if isinstance(value, int):
        return _INT_TRACK_TYPES.get(value, "TT_Unknown")
    text = str(value)
    if text.isdigit():
        return _INT_TRACK_TYPES.get(int(text), "TT_Unknown")
    return _TYPE_ALIASES.get(text, text)


def _normalize_preset(data: Any) -> dict[str, Any]:
    preset = _default_preset()
    if not isinstance(data, dict):
        return preset
    scope = data.get("scope")
    if scope in {key for key, _ in SCOPE_CHOICES}:
        preset["scope"] = scope
    mode = data.get("mode")
    if mode in {key for key, _ in MODE_CHOICES}:
        preset["mode"] = mode
    all_height = data.get("all_height")
    if all_height in {key for key, _ in HEIGHT_CHOICES}:
        preset["all_height"] = all_height
    heights = data.get("heights", {})
    if isinstance(heights, dict):
        valid = {key for key, _ in HEIGHT_CHOICES}
        for track_type, height in heights.items():
            if isinstance(track_type, str) and height in valid:
                preset["heights"][track_type] = height
    return preset


class TrackHeightTool(QWidget):
    """Preset-based Pro Tools track height utility."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._client = None
        self._presets_data: dict[str, dict[str, Any]] = {}
        self._loading = False
        self._combos: dict[str, QComboBox] = {}
        self._preview_tracks: list[dict[str, Any]] = []
        self._sort_column = COL_TYPE
        self._sort_order = Qt.AscendingOrder
        self._init_ui()
        self._load_from_config()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        desc = QLabel(
            "Configure Pro Tools track heights by track type, then apply the "
            "active preset to selected, visible, or all tracks."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 9pt;")
        layout.addWidget(desc)

        self._preset_panel = NamedPresetPanel([], label="Preset:", parent=self)
        self._preset_panel.preset_switching.connect(self._on_preset_switching)
        self._preset_panel.preset_added.connect(self._on_preset_added)
        self._preset_panel.preset_duplicated.connect(self._on_preset_duplicated)
        self._preset_panel.preset_renamed.connect(self._on_preset_renamed)
        self._preset_panel.preset_deleted.connect(self._on_preset_deleted)
        layout.addWidget(self._preset_panel)
        preset_layout = self._preset_panel.layout()
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save_preset)
        preset_layout.insertWidget(3, self._save_btn)
        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._on_reset_defaults)
        preset_layout.addWidget(self._reset_btn)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        for value, label in MODE_CHOICES:
            self._mode_combo.addItem(label, value)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        controls.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        for value, label in SCOPE_CHOICES:
            self._scope_combo.addItem(label, value)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        controls.addWidget(self._scope_combo)

        self._all_height_label = QLabel("Height:")
        controls.addWidget(self._all_height_label)
        self._all_height_combo = QComboBox()
        for value, text in HEIGHT_CHOICES:
            self._all_height_combo.addItem(text, value)
        self._all_height_combo.currentIndexChanged.connect(
            self._on_all_height_changed
        )
        controls.addWidget(self._all_height_combo)
        self._all_summary = QLabel("")
        self._all_summary.setStyleSheet("color: #aaa; font-size: 9pt;")
        controls.addWidget(self._all_summary)

        controls.addStretch()
        self._refresh_btn = QPushButton("Preview")
        self._refresh_btn.clicked.connect(self._refresh_preview)
        controls.addWidget(self._refresh_btn)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_preset)
        controls.addWidget(self._apply_btn)
        layout.addLayout(controls)

        self._table = QTableWidget(len(TRACK_TYPES), 4, self)
        self._table.setHorizontalHeaderLabels(
            ["Track Type", "Matched", "Current Heights", "Preset Height"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_header_clicked)
        header.setSectionResizeMode(COL_TYPE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_MATCHED, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CURRENT, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_PRESET, QHeaderView.ResizeToContents)
        layout.addWidget(self._table, 1)

        for row, (track_type, label) in enumerate(TRACK_TYPES):
            type_item = QTableWidgetItem(label)
            type_item.setData(Qt.UserRole, track_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_TYPE, type_item)
            count_item = _NumericItem("0")
            count_item.setData(Qt.UserRole, 0)
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_MATCHED, count_item)
            current_item = QTableWidgetItem("")
            current_item.setFlags(current_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_CURRENT, current_item)
            combo = QComboBox()
            for value, text in HEIGHT_CHOICES:
                combo.addItem(text, value)
            combo.currentIndexChanged.connect(
                lambda _idx, tt=track_type: self._on_height_changed(tt)
            )
            self._combos[track_type] = combo
            self._table.setCellWidget(row, COL_PRESET, combo)

        self._sort_table(COL_TYPE, Qt.AscendingOrder)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(self._status)
        self._bottom_stretch = QSpacerItem(
            0, 0, QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        layout.addItem(self._bottom_stretch)
        self._update_mode_visibility()
        self._update_dirty_state()

    def _set_status(self, text: str, color: str):
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 8pt;")

    def _set_error_status(self, text: str):
        self._set_status(text, "#f44336")

    def _set_warning_status(self, text: str):
        self._set_status(text, "#ff9800")

    def _set_success_status(self, text: str):
        self._set_status(text, "#4caf50")

    def set_engine(self, engine):
        self.set_client(engine)

    def set_client(self, client):
        self._client = client
        self._preview_tracks = []
        self._update_connection_state()

    def update_config(self, config: dict):
        self._config = config
        self._load_from_config()

    def _load_from_config(self):
        app = self._config.setdefault("app", {})
        raw_presets = app.get("protools_track_height_presets", {})
        if not isinstance(raw_presets, dict) or not raw_presets:
            raw_presets = {"Default": _default_preset()}
        self._presets_data = {
            name: _normalize_preset(data)
            for name, data in raw_presets.items()
            if isinstance(name, str) and name
        }
        if "Default" not in self._presets_data:
            self._presets_data["Default"] = _default_preset()

        current = app.get("active_protools_track_height_preset", "Default")
        if current not in self._presets_data:
            current = "Default"

        self._preset_panel.reset(list(self._presets_data), current=current)
        self._load_preset(current)

    def _persist_config(self):
        app = self._config.setdefault("app", {})
        app["protools_track_height_presets"] = copy.deepcopy(self._presets_data)
        app["active_protools_track_height_preset"] = self._preset_panel.current_name
        save_config(self._config)

    def _current_widget_preset(self) -> dict[str, Any]:
        preset = _default_preset()
        preset["mode"] = self._mode_combo.currentData() or "track_type"
        preset["scope"] = self._scope_combo.currentData() or "all"
        preset["all_height"] = self._all_height_combo.currentData() or ""
        heights = {}
        for track_type, combo in self._combos.items():
            heights[track_type] = combo.currentData() or ""
        preset["heights"] = heights
        return preset

    def _save_current_widgets(self, name: str | None = None):
        target = name or self._preset_panel.current_name
        if not target:
            return
        self._presets_data[target] = self._current_widget_preset()

    def _load_widgets(self, preset: dict[str, Any]):
        self._loading = True
        mode_idx = self._mode_combo.findData(preset["mode"])
        self._mode_combo.setCurrentIndex(max(mode_idx, 0))
        scope_idx = self._scope_combo.findData(preset["scope"])
        self._scope_combo.setCurrentIndex(max(scope_idx, 0))
        all_height_idx = self._all_height_combo.findData(preset["all_height"])
        self._all_height_combo.setCurrentIndex(max(all_height_idx, 0))
        for track_type, combo in self._combos.items():
            height = preset["heights"].get(track_type, "")
            idx = combo.findData(height)
            combo.setCurrentIndex(max(idx, 0))
        self._loading = False
        self._update_mode_visibility()
        self._update_connection_state()
        self._update_dirty_state()

    def _load_preset(self, name: str):
        preset = _normalize_preset(self._presets_data.get(name))
        self._presets_data[name] = preset
        self._load_widgets(preset)

    def _on_preset_switching(self, old_name: str, new_name: str):
        self._load_preset(new_name)

    def _on_preset_added(self, name: str):
        self._presets_data[name] = _default_preset()
        self._load_preset(name)
        self._persist_config()

    def _on_preset_duplicated(self, source: str, new_name: str):
        self._presets_data[new_name] = self._current_widget_preset()
        self._load_preset(new_name)
        self._persist_config()

    def _on_preset_renamed(self, old_name: str, new_name: str):
        self._presets_data[new_name] = self._presets_data.pop(old_name, _default_preset())
        self._persist_config()

    def _on_preset_deleted(self, name: str):
        self._presets_data.pop(name, None)
        self._load_preset(self._preset_panel.current_name)
        self._persist_config()

    def _on_scope_changed(self):
        if self._loading:
            return
        self._update_dirty_state()

    def _on_mode_changed(self):
        if self._loading:
            return
        self._update_mode_visibility()
        self._update_dirty_state()

    def _on_all_height_changed(self):
        if self._loading:
            return
        self._update_dirty_state()

    def _on_height_changed(self, _track_type: str):
        if self._loading:
            return
        self._update_dirty_state()

    def _update_dirty_state(self):
        if not hasattr(self, "_save_btn"):
            return
        saved = _normalize_preset(
            self._presets_data.get(self._preset_panel.current_name)
        )
        current = _normalize_preset(self._current_widget_preset())
        self._save_btn.setEnabled(current != saved)

    def _on_save_preset(self):
        self._save_current_widgets()
        self._persist_config()
        self._update_dirty_state()
        self._set_success_status(f"Saved preset: {self._preset_panel.current_name}")

    def _on_reset_defaults(self):
        self._load_widgets(_default_preset())
        self._set_warning_status("Defaults loaded. Press Save to update the preset.")

    def _update_mode_visibility(self):
        is_all = self._mode_combo.currentData() == "all"
        self._table.setVisible(not is_all)
        if hasattr(self, "_bottom_stretch"):
            self._bottom_stretch.changeSize(
                0,
                0 if not is_all else 1,
                QSizePolicy.Minimum,
                QSizePolicy.Fixed if not is_all else QSizePolicy.Expanding,
            )
            widget_layout = self.layout()
            if widget_layout is not None:
                widget_layout.invalidate()
        self._all_height_label.setVisible(is_all)
        self._all_height_combo.setVisible(is_all)
        self._all_summary.setVisible(is_all)

    def _on_header_clicked(self, column: int):
        if column not in (COL_TYPE, COL_MATCHED):
            return
        if column == self._sort_column:
            order = (
                Qt.DescendingOrder
                if self._sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            order = Qt.AscendingOrder
        self._sort_table(column, order)

    def _sort_table(self, column: int, order: Qt.SortOrder):
        if column not in (COL_TYPE, COL_MATCHED):
            return
        self._sort_column = column
        self._sort_order = order
        self._table.sortItems(column, order)
        self._table.horizontalHeader().setSortIndicator(column, order)

    def _tracks_for_scope(
        self,
        tracks: list[dict[str, Any]],
        preset: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scope = (preset or self._current_widget_preset()).get("scope", "all")
        if scope == "all":
            return tracks
        if scope == "visible":
            return [
                t for t in tracks
                if t.get("track_attributes", {}).get("is_hidden")
                in ("None", "TAState_None")
            ]
        return [
            t for t in tracks
            if t.get("track_attributes", {}).get("is_selected")
            in ("SetExplicitly", "TAState_SetExplicitly")
        ]

    def _refresh_preview(self):
        if self._client is None:
            self._set_error_status("Not connected to Pro Tools")
            return
        self._refresh_btn.setEnabled(False)
        self._set_status("Loading preview...", "#aaa")
        self._client.request(
            "get_track_list",
            {},
            self._on_preview_tracks,
            timeout_ms=PTSL_READ_TIMEOUT_MS,
        )

    def _on_preview_tracks(self, response: dict):
        self._refresh_btn.setEnabled(True)
        if not response.get("ok"):
            self._set_error_status(f"Preview failed: {response.get('error') or ''}")
            return
        self._preview_tracks = list(response.get("result") or [])
        preset = self._current_widget_preset()
        self._render_preview(self._preview_tracks, preset)

    def _render_preview(self, tracks: list[dict[str, Any]], preset: dict[str, Any]):
        scoped = self._tracks_for_scope(tracks, preset)
        if preset.get("mode") == "all":
            heights = Counter(
                track.get("height") or "THeight_Unknown"
                for track in scoped
            )
            text = ", ".join(
                f"{height.replace('THeight_', '')}: {count}"
                for height, count in sorted(heights.items())
            )
            suffix = f" ({text})" if text else ""
            self._all_summary.setText(
                f"{len(scoped)} of {len(tracks)} tracks match scope{suffix}"
            )
            self._set_success_status(
                f"Preview loaded: {len(scoped)} of {len(tracks)} tracks match scope"
            )
            return

        counts: Counter[str] = Counter()
        current: dict[str, Counter[str]] = defaultdict(Counter)
        for track in scoped:
            track_type = _normalize_track_type(track.get("type", ""))
            counts[track_type] += 1
            height = track.get("height") or "THeight_Unknown"
            current[track_type][height] += 1

        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_TYPE)
            track_type = item.data(Qt.UserRole)
            count = counts.get(track_type, 0)
            count_item = self._table.item(row, COL_MATCHED)
            count_item.setText(str(count))
            count_item.setData(Qt.UserRole, count)
            heights = current.get(track_type, Counter())
            text = ", ".join(
                f"{height.replace('THeight_', '')}: {count}"
                for height, count in sorted(heights.items())
            )
            self._table.item(row, COL_CURRENT).setText(text)

        self._sort_table(self._sort_column, self._sort_order)
        self._set_success_status(
            f"Preview loaded: {len(scoped)} of {len(tracks)} tracks match scope"
        )

    def _apply_preset(self):
        if self._client is None:
            self._set_error_status("Not connected to Pro Tools")
            return

        self._apply_btn.setEnabled(False)
        self._set_status("Applying height preset...", "#aaa")
        self._client.request(
            "get_track_list",
            {},
            self._on_apply_tracks,
            timeout_ms=PTSL_READ_TIMEOUT_MS,
        )

    def _on_apply_tracks(self, response: dict):
        if not response.get("ok"):
            self._apply_btn.setEnabled(True)
            self._set_error_status(f"Apply failed: {response.get('error') or ''}")
            return

        tracks = list(response.get("result") or [])
        preset = self._current_widget_preset()
        scoped = self._tracks_for_scope(tracks, preset)
        if preset.get("mode") == "all":
            all_height = preset.get("all_height", "")
            if not all_height:
                self._apply_btn.setEnabled(True)
                self._set_warning_status("No height selected")
                return
            track_ids = [track.get("id") for track in scoped if track.get("id")]
            if not track_ids:
                self._apply_btn.setEnabled(True)
                self._set_warning_status("No matching tracks in scope")
                return
            self._request_height_changes({all_height: track_ids})
            return

        heights = preset.get("heights", {})
        by_height: dict[str, list[str]] = defaultdict(list)
        for track in scoped:
            track_id = track.get("id")
            track_type = _normalize_track_type(track.get("type", ""))
            height = heights.get(track_type, "")
            if track_id and height:
                by_height[height].append(track_id)

        if not by_height:
            self._apply_btn.setEnabled(True)
            self._set_warning_status("No matching tracks with configured heights")
            return
        self._request_height_changes(by_height)

    def _request_height_changes(self, by_height: dict[str, list[str]]):
        pending = len(by_height)
        changed = sum(len(track_ids) for track_ids in by_height.values())
        failures: list[str] = []

        def on_height_response(response: dict):
            nonlocal pending
            if not response.get("ok"):
                failures.append(str(response.get("error") or "Unknown error"))
            pending -= 1
            if pending:
                return
            self._apply_btn.setEnabled(True)
            if failures:
                self._set_error_status(f"Apply failed: {failures[0]}")
                return
            self._set_success_status(f"Applied height preset to {changed} track(s)")
            self._refresh_preview()

        for height, track_ids in by_height.items():
            self._client.request(
                "set_track_height",
                {"height": height, "track_ids": track_ids},
                on_height_response,
                timeout_ms=PTSL_MUTATION_TIMEOUT_MS,
            )

    def _update_connection_state(self):
        if self._client is None:
            self._set_status("", "#888")
        elif self._status.text() == "Disconnected":
            self._set_status("", "#888")
