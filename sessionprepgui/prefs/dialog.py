"""Preferences dialog — thin orchestrator.

Creates the two-tab shell (Global / Config Presets), wires up the
self-contained page classes, and owns config-preset CRUD via NamedPresetPanel.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..settings import _build_default_config_preset
from .config_pages import build_config_pages, load_config_widgets, read_config_widgets
from .page_colors import ColorsPage
from .page_general import GeneralPage
from .page_groups import GroupsPage
from .preset_panel import NamedPresetPanel


class PreferencesDialog(QDialog):
    """Hierarchical preferences dialog with tree navigation."""

    def __init__(self, config: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(1150, 700)
        self._config = copy.deepcopy(config)
        self._saved = False
        self._dirty_icon = self._make_dirty_icon()
        self._original_global_state: dict[str, Any] = {}
        self._original_config_state: dict[str, Any] = {}
        self._global_items: dict[str, QTreeWidgetItem] = {}
        self._preset_items: dict[str, QTreeWidgetItem] = {}
        self._loading_cfg_widgets = False

        # Config presets working copy {name: structured_dict}
        self._config_presets_data: dict[str, dict[str, Any]] = copy.deepcopy(
            self._config.get("config_presets", {}))
        if "Default" not in self._config_presets_data:
            self._config_presets_data["Default"] = _build_default_config_preset()

        # Pipeline widget registry (built by build_config_pages)
        self._cfg_widgets: dict = {}
        self._cfg_daw_custom_widgets: dict[str, QWidget] = {}

        # Pages
        self._general_page = GeneralPage()
        self._colors_page = ColorsPage()
        self._groups_page = GroupsPage(
            color_provider=self._colors_page.color_provider,
            all_colors_provider=self._colors_page.all_colors_provider)
        self._colors_page.colorsChanged.connect(
            self._groups_page.refresh_colors)

        self._init_ui()

        # Load all pages after UI is built
        self._general_page.load(self._config)
        self._colors_page.load(self._config)
        self._groups_page.load(self._config)

        active_cfg = self._config.get("app", {}).get(
            "active_config_preset", "Default")
        self._cfg_panel.set_current(active_cfg)
        self._load_cfg_preset_widgets(active_cfg)
        self._original_global_state = self._current_global_state()
        self._original_config_state = self._current_config_state()
        self._connect_dirty_signals()
        self._update_dirty_indicators()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def saved(self) -> bool:
        return self._saved

    def result_config(self) -> dict[str, Any]:
        """Return the edited config (only valid after Save)."""
        return self._config

    # ── UI construction ───────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        root.addWidget(self._tabs, 1)

        self._global_tab_index = self._tabs.addTab(
            self._build_global_tab(), "Global")
        self._preset_tab_index = self._tabs.addTab(
            self._build_preset_tab(), "Config Presets")

        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        self._save_btn = btn_box.button(QDialogButtonBox.Save)
        self._save_btn.setText("Save Preferences")
        self._save_btn.setDefault(True)
        btn_box.accepted.connect(self._on_save)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def _build_global_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        self._global_tree = QTreeWidget()
        self._global_tree.setColumnCount(2)
        self._global_tree.setHeaderHidden(True)
        self._global_tree.setMinimumWidth(140)
        self._global_tree.setMaximumWidth(200)
        self._global_tree.header().setStretchLastSection(False)
        self._global_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._global_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self._global_tree.setColumnWidth(1, 18)
        splitter.addWidget(self._global_tree)

        self._global_stack = QStackedWidget()
        splitter.addWidget(self._global_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._global_page_index: dict[int, int] = {}

        def add(label: str, page: QWidget) -> QTreeWidgetItem:
            item = QTreeWidgetItem(self._global_tree, [label, ""])
            item.setFont(0, QFont("", -1, QFont.Bold))
            self._register_page(item, page,
                                self._global_stack, self._global_page_index)
            return item

        self._global_items["general"] = add("General", self._general_page)
        self._global_items["colors"] = add("Colors", self._colors_page)
        self._global_items["groups"] = add("Groups", self._groups_page)

        self._global_tree.expandAll()
        first = self._global_tree.topLevelItem(0)
        if first:
            self._global_tree.setCurrentItem(first)
        self._global_tree.currentItemChanged.connect(
            lambda cur, _: self._on_tree_selection(
                cur, self._global_stack, self._global_page_index))

        return tab

    def _build_preset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 4, 0, 0)

        self._cfg_panel = NamedPresetPanel(
            list(self._config_presets_data),
            label="Config Preset:",
            protected=frozenset({"Default"}),
        )
        self._cfg_panel.preset_switching.connect(self._on_cfg_switching)
        self._cfg_panel.preset_added.connect(self._on_cfg_added)
        self._cfg_panel.preset_duplicated.connect(self._on_cfg_duplicated)
        self._cfg_panel.preset_renamed.connect(self._on_cfg_renamed)
        self._cfg_panel.preset_deleted.connect(self._on_cfg_deleted)
        layout.addWidget(self._cfg_panel)

        splitter = QSplitter(Qt.Horizontal)
        self._preset_tree = QTreeWidget()
        self._preset_tree.setColumnCount(2)
        self._preset_tree.setHeaderHidden(True)
        self._preset_tree.setMinimumWidth(180)
        self._preset_tree.setMaximumWidth(250)
        self._preset_tree.header().setStretchLastSection(False)
        self._preset_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._preset_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self._preset_tree.setColumnWidth(1, 18)
        splitter.addWidget(self._preset_tree)

        self._preset_stack = QStackedWidget()
        splitter.addWidget(self._preset_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._preset_page_index: dict[int, int] = {}
        self._cfg_daw_custom_widgets = build_config_pages(
            self._preset_tree,
            self._active_preset(),
            self._cfg_widgets,
            self._register_preset_page,
        )

        self._preset_tree.expandAll()
        first = self._preset_tree.topLevelItem(0)
        if first:
            self._preset_tree.setCurrentItem(first)
        self._preset_tree.currentItemChanged.connect(
            lambda cur, _: self._on_tree_selection(
                cur, self._preset_stack, self._preset_page_index))

        return tab

    # ── Tree/stack navigation ─────────────────────────────────────────

    def _register_page(self, item: QTreeWidgetItem, page: QWidget,
                       stack: QStackedWidget,
                       index: dict[int, int]) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(page)
        idx = stack.addWidget(scroll)
        index[id(item)] = idx

    def _register_preset_page(self, item: QTreeWidgetItem, page: QWidget,
                              dirty_key: str | None = None) -> None:
        self._register_page(
            item, page, self._preset_stack, self._preset_page_index)
        if dirty_key:
            self._preset_items[dirty_key] = item

    def _on_tree_selection(self, current: QTreeWidgetItem | None,
                           stack: QStackedWidget,
                           index: dict[int, int]) -> None:
        if current is None:
            return
        idx = index.get(id(current))
        if idx is not None:
            stack.setCurrentIndex(idx)

    # ДД Dirty indicators ДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДДД

    @staticmethod
    def _make_dirty_icon() -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#e53935"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(3, 3, 6, 6)
        painter.end()
        return QIcon(pixmap)

    def _connect_dirty_signals(self) -> None:
        self._general_page.preferencesChanged.connect(
            self._update_dirty_indicators)
        self._colors_page.colorsChanged.connect(self._update_dirty_indicators)
        self._groups_page.groupsChanged.connect(self._update_dirty_indicators)
        self._connect_config_dirty_signals()

    def _connect_config_dirty_signals(self) -> None:
        for widgets in self._cfg_widgets.values():
            for _key, widget in widgets:
                self._connect_widget_dirty_signal(widget)

        for widget in self._cfg_daw_custom_widgets.values():
            if hasattr(widget, "templates_changed"):
                widget.templates_changed.connect(self._update_dirty_indicators)

    def _connect_widget_dirty_signal(self, widget: QWidget) -> None:
        if hasattr(widget, "path_changed"):
            widget.path_changed.connect(
                lambda *_: self._update_dirty_indicators())
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(
                lambda *_: self._update_dirty_indicators())
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda *_: self._update_dirty_indicators())
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(
                lambda *_: self._update_dirty_indicators())
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(
                lambda *_: self._update_dirty_indicators())

    def _current_global_state(self) -> dict[str, Any]:
        return {
            "general": copy.deepcopy(self._general_page.current_values()),
            "colors": copy.deepcopy(self._colors_page.current_values()),
            "groups": copy.deepcopy(self._groups_page.current_values()),
        }

    def _current_config_state(self) -> dict[str, Any]:
        presets = copy.deepcopy(self._config_presets_data)
        current_name = self._cfg_panel.current_name
        if current_name:
            current_preset = presets.setdefault(current_name, {})
            current_preset.update(self._read_current_config_widgets())
        return {
            "active": current_name,
            "presets": presets,
        }

    def _read_current_config_widgets(self) -> dict[str, Any]:
        return read_config_widgets(
            self._cfg_widgets, self._cfg_daw_custom_widgets)

    def _set_tree_icon(self, item: QTreeWidgetItem | None,
                       dirty: bool) -> None:
        if item is not None:
            item.setIcon(1, self._dirty_icon if dirty else QIcon())

    def _set_tab_dirty(self, index: int, label: str, dirty: bool) -> None:
        self._tabs.setTabText(index, f"{label} \u25cf" if dirty else label)

    def _update_dirty_indicators(self) -> None:
        if self._loading_cfg_widgets:
            return

        current_global = self._current_global_state()
        global_dirty = False
        for key, item in self._global_items.items():
            dirty = current_global.get(key) != self._original_global_state.get(key)
            self._set_tree_icon(item, dirty)
            global_dirty = global_dirty or dirty

        config_state = self._current_config_state()
        config_dirty = config_state != self._original_config_state
        self._update_config_tree_dirty_indicators()

        self._set_tab_dirty(self._global_tab_index, "Global", global_dirty)
        self._set_tab_dirty(
            self._preset_tab_index, "Config Presets", config_dirty)

        self._save_btn.setText(
            "Save Preferences \u25cf"
            if global_dirty or config_dirty
            else "Save Preferences")

    def _update_config_tree_dirty_indicators(self) -> None:
        dirty_keys = self._config_tree_dirty_keys()
        for key, item in self._preset_items.items():
            self._set_tree_icon(item, key in dirty_keys)

    def _config_tree_dirty_keys(self) -> set[str]:
        current_name = self._cfg_panel.current_name
        original_presets = self._original_config_state.get("presets", {})
        original_preset = original_presets.get(current_name)
        if original_preset is None:
            return set(self._preset_items)

        current_preset = copy.deepcopy(
            self._config_presets_data.get(current_name, {}))
        current_preset.update(self._read_current_config_widgets())

        dirty_keys: set[str] = set()
        if current_preset.get("analysis") != original_preset.get("analysis"):
            dirty_keys.add("analysis")
        if current_preset.get("presentation") != original_preset.get("presentation"):
            dirty_keys.add("_presentation")

        for key in self._cfg_widgets:
            if key.startswith("detectors."):
                section_id = key.removeprefix("detectors.")
                current = current_preset.get("detectors", {}).get(section_id)
                original = original_preset.get("detectors", {}).get(section_id)
                if current != original:
                    dirty_keys.add(key)
            elif key.startswith("processors."):
                section_id = key.removeprefix("processors.")
                current = current_preset.get("processors", {}).get(section_id)
                original = original_preset.get("processors", {}).get(section_id)
                if current != original:
                    dirty_keys.add(key)
            elif key.startswith("daw_processors."):
                section_id = key.removeprefix("daw_processors.")
                current = current_preset.get(
                    "daw_processors", {}).get(section_id)
                original = original_preset.get(
                    "daw_processors", {}).get(section_id)
                if current != original:
                    dirty_keys.add(key)

        if any(key.startswith("detectors.") for key in dirty_keys):
            dirty_keys.add("_presentation")
        if any(key.startswith("processors.") for key in dirty_keys):
            dirty_keys.add("_processors")
        if any(key.startswith("daw_processors.") for key in dirty_keys):
            dirty_keys.add("_daw_processors")
        return dirty_keys

    # ── Config preset helpers ─────────────────────────────────────────

    def _active_preset(self) -> dict[str, Any]:
        name = self._cfg_panel.current_name if hasattr(self, "_cfg_panel") else "Default"
        return self._config_presets_data.get(
            name, self._config_presets_data.get("Default", {}))

    def _save_cfg_preset_widgets(self, name: str | None = None) -> None:
        if name is None:
            name = self._cfg_panel.current_name
        if not name:
            return
        preset = self._config_presets_data.setdefault(name, {})
        preset.update(read_config_widgets(
            self._cfg_widgets, self._cfg_daw_custom_widgets))

    def _load_cfg_preset_widgets(self, name: str) -> None:
        preset = self._config_presets_data.get(name, {})
        self._loading_cfg_widgets = True
        try:
            load_config_widgets(
                self._cfg_widgets, preset, self._cfg_daw_custom_widgets)
        finally:
            self._loading_cfg_widgets = False

    # ── Config preset signal handlers ────────────────────────────────

    def _on_cfg_switching(self, old: str, new: str) -> None:
        if old and old in self._config_presets_data:
            self._save_cfg_preset_widgets(old)  # old name before combo changed
        self._load_cfg_preset_widgets(new)
        self._update_dirty_indicators()

    def _on_cfg_added(self, name: str) -> None:
        self._config_presets_data[name] = _build_default_config_preset()
        self._load_cfg_preset_widgets(name)
        self._update_dirty_indicators()

    def _on_cfg_duplicated(self, source: str, new: str) -> None:
        self._save_cfg_preset_widgets(source)  # capture any unsaved widget edits
        self._config_presets_data[new] = copy.deepcopy(
            self._config_presets_data.get(source, {}))
        self._load_cfg_preset_widgets(new)
        self._update_dirty_indicators()

    def _on_cfg_renamed(self, old: str, new: str) -> None:
        self._config_presets_data[new] = self._config_presets_data.pop(old, {})
        self._update_dirty_indicators()

    def _on_cfg_deleted(self, name: str) -> None:
        self._config_presets_data.pop(name, None)
        self._load_cfg_preset_widgets(self._cfg_panel.current_name)
        self._update_dirty_indicators()

    # ── Save ─────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        err = self._general_page.validate()
        if err:
            QMessageBox.warning(self, "Invalid Settings", err)
            return

        self._general_page.commit(self._config)
        self._colors_page.commit(self._config)

        # Validate regex patterns across all group presets before committing
        self._groups_page._save_current()
        for preset_name, groups in self._groups_page._presets_data.items():
            for g in groups:
                if g.get("match_method") == "regex" and g.get("match_pattern", ""):
                    try:
                        re.compile(g["match_pattern"])
                    except re.error as e:
                        QMessageBox.warning(
                            self, "Invalid Regex Pattern",
                            f"Group \u201c{g['name']}\u201d in preset "
                            f"\u201c{preset_name}\u201d has an invalid "
                            f"regular expression:\n\n"
                            f"{g['match_pattern']}\n\n{e}")
                        return

        self._groups_page.commit(self._config)

        self._save_cfg_preset_widgets()
        self._config["config_presets"] = copy.deepcopy(self._config_presets_data)
        self._config.setdefault("app", {})[
            "active_config_preset"] = self._cfg_panel.current_name

        # Remove legacy keys
        for legacy in ("gui", "analysis", "detectors", "processors", "daw_processors"):
            self._config.pop(legacy, None)

        self._saved = True
        self.accept()
