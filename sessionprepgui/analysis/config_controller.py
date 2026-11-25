"""Phase 2 Config tab, preset switching, overrides, dirty markers, and impact routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.config import (
    ConfigChangeImpact,
    classify_structured_config_change,
    resolve_session_config,
)
from sessionpreplib.processors import default_processors
from sessionpreplib.rendering import build_diagnostic_summary

from ..change_marker import MARKER_COLUMN_WIDTH, set_item_marker, set_tab_marker
from ..controls import content_splitter, fit_navigation_pane, inset_pane, settings_scroll_area
from ..prefs.config_pages import _config_section_values
from ..prefs.param_form import ConfigResetButton, _read_widget, _set_widget_value
from ..prefs.param_widgets import build_config_pages, load_config_widgets
from ..settings import build_defaults, save_config
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..tracks.table_widgets import (
    DETAIL_TAB_INSET,
    _PHASE_ANALYSIS,
    _TAB_GROUPS,
    _TAB_SESSION,
)
from .analysis_controller import repair_settings_can_apply

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTabWidget


class SessionConfigController(QObject):
    """Owns Phase 2 Config tab, per-session overrides, preset switching, and impact routing."""

    config_impact_applied = Signal(object)  # ConfigChangeImpact
    daw_refresh_requested = Signal()
    prepare_stale_requested = Signal()
    repair_plan_requested = Signal(object)  # list[TrackContext]
    reanalyze_phase1_requested = Signal()
    reanalyze_phase2_requested = Signal()
    groups_dirty_refresh_requested = Signal()
    processing_column_refresh_requested = Signal()
    analysis_column_refresh_requested = Signal()
    summary_render_requested = Signal()
    file_report_refresh_requested = Signal(object)  # TrackContext
    waveform_issues_refresh_requested = Signal()
    waveform_colormap_changed = Signal(str)

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        *,
        detail_tabs: QTabWidget | None = None,
        on_is_repair_pending_or_running: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._detail_tabs = detail_tabs
        self._on_is_repair_pending_or_running = on_is_repair_pending_or_running

        self.session_widgets: dict[str, list[tuple[str, QWidget]]] = {}
        self.session_page_index: dict[int, int] = {}
        self.session_dirty_items: dict[str, Any] = {}
        self.session_daw_custom_widgets: dict[str, Any] = {}
        self._loading_widgets = False

        self.page = self._build_tab()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_tab(self) -> QWidget:
        page = QWidget()
        page.setAutoFillBackground(True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET, 0)
        layout.setSpacing(0)

        desc = QLabel(
            "Session-local config adjustments. Changes here apply only to the current session."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(desc)
        layout.addSpacing(DETAIL_TAB_INSET)

        layout.addLayout(self._build_header())
        layout.addWidget(self._build_tree_and_stack(), 1)
        self._build_session_pages()

        self.session_tree.expandAll()
        first = self.session_tree.topLevelItem(0)
        if first:
            self.session_tree.setCurrentItem(first)
        return page

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(QLabel("Config Preset:"))
        self.config_preset_combo = QComboBox()
        self.config_preset_combo.setMinimumWidth(140)
        self.populate_config_preset_combo()
        self.config_preset_combo.currentTextChanged.connect(self.on_config_preset_changed)
        header.addWidget(self.config_preset_combo)
        header.addStretch()
        reset_btn = ConfigResetButton("Revert to Preset")
        reset_btn.setToolTip(
            "Discard all session-specific changes and reload from the selected config preset."
        )
        reset_btn.clicked.connect(self.on_session_config_reset)
        header.addWidget(reset_btn)
        return header

    def _build_tree_and_stack(self):
        splitter = content_splitter()
        self.session_tree = QTreeWidget()
        self.session_tree.setColumnCount(2)
        self.session_tree.setHeaderHidden(True)
        self.session_tree.setMinimumWidth(160)
        self.session_tree.setMaximumWidth(220)
        self.session_tree.header().setStretchLastSection(False)
        self.session_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.session_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.session_tree.setColumnWidth(1, MARKER_COLUMN_WIDTH)
        self.session_tree.currentItemChanged.connect(self._on_tree_selection)
        splitter.addWidget(
            inset_pane(self.session_tree, (0, DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET))
        )

        self.session_stack = QStackedWidget()
        splitter.addWidget(inset_pane(self.session_stack, (0, 0, 0, DETAIL_TAB_INSET)))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        fit_navigation_pane(splitter)
        return splitter

    def _build_session_pages(self) -> None:
        def _register_page(tree_item: Any, page: QWidget, dirty_key: str | None = None) -> None:
            idx = self.session_stack.addWidget(settings_scroll_area(page))
            self.session_page_index[id(tree_item)] = idx
            if dirty_key:
                self.session_dirty_items[dirty_key] = tree_item

        self.session_daw_custom_widgets = build_config_pages(
            self.session_tree,
            self._state.active_preset(),
            self.session_widgets,
            _register_page,
            on_reset=self.on_session_config_field_reset,
        )
        self._connect_dirty_signals()

    def _on_tree_selection(self, current: Any, _previous: Any) -> None:
        if current is None:
            return
        idx = self.session_page_index.get(id(current))
        if idx is not None:
            self.session_stack.setCurrentIndex(idx)

    # ── Preset combo & widget loading ────────────────────────────────────

    def select_daw_processor_page(self, label: str = "Pro Tools") -> bool:
        """Select the named DAW processor entry in the Phase 2 Config tree."""
        for i in range(self.session_tree.topLevelItemCount()):
            top = self.session_tree.topLevelItem(i)
            if top is None:
                continue
            for j in range(top.childCount()):
                child = top.child(j)
                if child is not None and child.text(0).strip().lower() == label.strip().lower():
                    self.session_tree.setCurrentItem(child)
                    return True
        return False

    def populate_config_preset_combo(self) -> None:
        combo = self.config_preset_combo
        presets = self._state.config.get(
            "config_presets", build_defaults().get("config_presets", {})
        )
        active = self._state.active_config_preset_name
        combo.blockSignals(True)
        combo.clear()
        for name in presets:
            combo.addItem(name)
        idx = combo.findText(active)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def init_session_config(self) -> None:
        self.populate_config_preset_combo()
        self.load_session_widgets(self._state.effective_session_config())
        if self._detail_tabs is not None:
            self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self.refresh_phase2_dirty_indicators()

    def load_session_widgets(self, preset: dict[str, Any]) -> None:
        self._loading_widgets = True
        try:
            load_config_widgets(
                self.session_widgets,
                preset,
                self.session_daw_custom_widgets,
                reference_preset=self._state.active_preset(),
            )
        finally:
            self._loading_widgets = False
        self.refresh_dirty_indicators()

    # ── Reset & edit handlers ────────────────────────────────────────────

    @Slot()
    def on_session_config_reset(self) -> None:
        old_config = self._state.effective_session_config()
        self._state.session_overrides.clear()
        preset = self._state.active_preset()
        self.load_session_widgets(preset)
        self.refresh_effective_config(classify_structured_config_change(old_config, preset))
        self._shell.show_status("Session config reverted to preset.")

    def on_session_config_field_reset(self, path: str, widget: Any) -> None:
        if not self._state.session or self._loading_widgets:
            return
        old_config = self._state.effective_session_config()
        section, key = path.rsplit(".", 1)
        values = _config_section_values(self._state.active_preset(), section)
        preset_value = values.get(key, widget._param_spec.default)
        self._state.session_overrides.pop(path, None)
        self._loading_widgets = True
        try:
            _set_widget_value(widget, preset_value)
            widget._set_param_reference(preset_value=preset_value)
        finally:
            self._loading_widgets = False
        self.apply_session_config_edit(old_config)

    def _connect_dirty_signals(self) -> None:
        for section, widgets in self.session_widgets.items():
            prefix = "presentation" if section == "_presentation" else section
            for key, widget in widgets:
                self._connect_widget_dirty_signal(f"{prefix}.{key}", widget)
        for processor_id, widget in self.session_daw_custom_widgets.items():
            path = f"daw_processors.{processor_id}.{processor_id}_templates"
            widget.templates_changed.connect(
                lambda p=path, w=widget: self.on_session_config_value_changed(p, w.get_templates())
            )

    def _connect_widget_dirty_signal(self, path: str, widget: Any) -> None:
        def edited(*_args: Any) -> None:
            self.on_session_config_value_changed(path, _read_widget(widget))

        if hasattr(widget, "path_changed"):
            widget.path_changed.connect(edited)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(edited)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(edited)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(edited)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(edited)

    def on_session_config_value_changed(self, path: str, value: Any) -> None:
        if not self._state.session or self._loading_widgets:
            return
        old_config = self._state.effective_session_config()
        self._state.session_overrides[path] = value
        self.apply_session_config_edit(old_config)

    def apply_session_config_edit(self, old_config: dict[str, Any]) -> None:
        impact = classify_structured_config_change(old_config, self._state.effective_session_config())
        self.refresh_dirty_indicators()
        self.refresh_effective_config(impact)
        repair_busy = bool(
            self._on_is_repair_pending_or_running and self._on_is_repair_pending_or_running()
        )
        if (impact.requires_phase1 or impact.requires_phase2) and repair_busy:
            phase = "Track Layout" if impact.requires_phase1 else "Phase 2 analysis"
            self._shell.show_status(
                f"Repair checks update automatically. Rerun {phase} to refresh the full session."
            )
        elif impact.requires_phase1:
            self._shell.show_status(
                "Config changed: rerun Track Layout to apply Phase 1 detector settings."
            )
        elif impact.requires_phase2:
            self._shell.show_status("Config changed: press Reanalyze to update Phase 2.")
        elif impact.daw_keys:
            self._shell.show_status("Config changed: Phase 3 DAW settings updated.")

    # ── Preset switching & impact routing ────────────────────────────────

    @Slot(str)
    def on_config_preset_changed(self, name: str) -> None:
        presets = self._state.config.get(
            "config_presets", build_defaults().get("config_presets", {})
        )
        if name not in presets:
            return
        old_config = self._state.effective_session_config()
        new_config = resolve_session_config(presets[name], self._state.session_overrides)
        impact = classify_structured_config_change(old_config, new_config)

        if self._state.session is not None and not self._confirm_preset_switch(name, impact):
            self.config_preset_combo.blockSignals(True)
            self.config_preset_combo.setCurrentText(self._state.active_config_preset_name)
            self.config_preset_combo.blockSignals(False)
            return

        self._state.active_config_preset_name = name
        if self._state.session is None:
            self._state.config.setdefault("app", {})["active_config_preset"] = name
            save_config(self._state.config)
        self.load_session_widgets(new_config)
        self.apply_config_change_impact(impact)

    def _confirm_preset_switch(self, name: str, impact: ConfigChangeImpact) -> bool:
        intro = (
            f"Switching to \u201c{name}\u201d updates inherited settings.\n\n"
            "Session overrides and group assignments will be preserved."
        )
        if impact.requires_phase1:
            intro += "\n\nThe new preset changes Track Layout analysis settings."
        elif impact.requires_phase2:
            intro += "\n\nThe new preset changes Phase 2 analysis or gain planning settings."
        if impact.requires_phase1 or impact.requires_phase2:
            return self.confirm_config_change_reanalysis(impact, intro=intro)
        ans = QMessageBox.question(
            self._shell.dialog_parent(),
            "Switch config preset?",
            intro + "\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def confirm_config_change_reanalysis(self, impact: ConfigChangeImpact, *, intro: str) -> bool:
        parent = self._shell.dialog_parent()
        if impact.requires_phase1:
            ans = QMessageBox.question(
                parent,
                "Rerun Track Layout?",
                intro
                + "\n\nThese changes affect Track Layout checks such as "
                "dual-mono or one-sided-silence detection. Rerun Track "
                "Layout now to apply them to the current session?\n\n"
                "Topology may change, so review Phase 1 before continuing.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return ans == QMessageBox.StandardButton.Yes
        if impact.requires_phase2:
            ans = QMessageBox.question(
                parent,
                "Reanalyze Phase 2?",
                intro
                + "\n\nThese changes affect Phase 2 analysis or gain planning. "
                "Reanalyze Phase 2 now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            return ans == QMessageBox.StandardButton.Yes
        return True

    def apply_config_change_impact(self, impact: ConfigChangeImpact) -> None:
        reanalyzing = impact.requires_phase1 or impact.requires_phase2
        self.refresh_effective_config(impact, reanalyzing=reanalyzing)
        if impact.requires_phase1:
            self.reanalyze_phase1_requested.emit()
        elif impact.requires_phase2:
            self.reanalyze_phase2_requested.emit()
        elif not (impact.presentation_keys and self._state.session):
            self.refresh_config_changed_display()
        self.config_impact_applied.emit(impact)

    def refresh_effective_config(self, impact: ConfigChangeImpact, *, reanalyzing: bool = False) -> None:
        if impact.daw_keys:
            self.daw_refresh_requested.emit()
        if any(key.startswith("processors.") and key.endswith("_enabled") for key in impact.phase2_keys):
            self.refresh_enabled_processors()
        if impact.requires_phase1 or impact.requires_phase2:
            self.prepare_stale_requested.emit()
        session = self._state.session
        if session and self._shell.is_phase_enabled(_PHASE_ANALYSIS) and not reanalyzing:
            if impact.repair_keys:
                self.repair_plan_requested.emit(session.tracks)
            elif impact.requires_phase1 or impact.requires_phase2:
                flat = self._state.flat_config()
                tracks = [
                    t for t in session.tracks
                    if t.repair_recipe.active or repair_settings_can_apply(t, flat)
                ]
                if tracks:
                    self.repair_plan_requested.emit(tracks)
        if impact.presentation_keys and session:
            self.refresh_presentation()

    def refresh_config_changed_display(self) -> None:
        if self._state.session:
            self.summary_render_requested.emit()
        cmap = self._state.config.get("app", {}).get("spectrogram_colormap", "magma")
        self.waveform_colormap_changed.emit(cmap)
        if self._state.current_track:
            self.file_report_refresh_requested.emit(self._state.current_track)

    def refresh_enabled_processors(self) -> None:
        session = self._state.session
        if not session:
            return
        flat = self._state.flat_config()
        new_processors = []
        for proc in default_processors():
            proc.configure(flat)
            if proc.enabled:
                new_processors.append(proc)
        new_processors.sort(key=lambda p: p.priority)
        session.processors = new_processors
        self.processing_column_refresh_requested.emit()

    def refresh_presentation(self) -> None:
        session = self._state.session
        if not session:
            return
        flat = self._state.flat_config()
        for det in session.detectors:
            det.configure(flat)
        self._state.summary = build_diagnostic_summary(session, remaining=True)
        self.summary_render_requested.emit()
        self.analysis_column_refresh_requested.emit()
        if self._state.current_track:
            self.file_report_refresh_requested.emit(self._state.current_track)
            self.waveform_issues_refresh_requested.emit()
        cmap = self._state.config.get("app", {}).get("spectrogram_colormap", "magma")
        self.waveform_colormap_changed.emit(cmap)

    # ── Dirty indicators ─────────────────────────────────────────────────

    def phase2_change_tooltip(self, index: int) -> str:
        name = (
            self._state.active_session_preset
            if index == _TAB_GROUPS
            else self._state.active_config_preset_name
        ) or ""
        kind = "group preset" if index == _TAB_GROUPS else "preset"
        return f"Changed from {kind} \u201c{name}\u201d" if name else f"Changed from {kind}"

    @Slot(int, str, bool)
    def set_detail_tab_dirty(self, index: int, _label: str, dirty: bool) -> None:
        if self._detail_tabs is not None:
            set_tab_marker(self._detail_tabs, index, dirty, self.phase2_change_tooltip(index))

    def refresh_phase2_dirty_indicators(self) -> None:
        self.groups_dirty_refresh_requested.emit()
        self.refresh_dirty_indicators()

    def refresh_dirty_indicators(self) -> None:
        dirty_keys = self.session_config_dirty_keys()
        tooltip = self.phase2_change_tooltip(_TAB_SESSION)
        for key, item in self.session_dirty_items.items():
            set_item_marker(item, 1, key in dirty_keys, tooltip)
        self.set_detail_tab_dirty(_TAB_SESSION, "Config", bool(dirty_keys))

    def session_config_dirty_keys(self) -> set[str]:
        if not self._state.session or not self.session_widgets:
            return set()
        current = self._state.effective_session_config()
        preset = self._state.active_preset()
        dirty_keys: set[str] = set()
        if current.get("analysis") != preset.get("analysis"):
            dirty_keys.add("analysis")
        if current.get("presentation") != preset.get("presentation"):
            dirty_keys.add("_presentation")

        for key in self.session_widgets:
            for prefix in ("detectors.", "processors.", "daw_processors."):
                if key.startswith(prefix):
                    group_key = prefix[:-1]
                    section_id = key.removeprefix(prefix)
                    if current.get(group_key, {}).get(section_id) != preset.get(group_key, {}).get(section_id):
                        dirty_keys.add(key)

        if any(key.startswith("detectors.") for key in dirty_keys):
            dirty_keys.add("_presentation")
        if any(key.startswith("processors.") for key in dirty_keys):
            dirty_keys.add("_processors")
        if any(key.startswith("daw_processors.") for key in dirty_keys):
            dirty_keys.add("_daw_processors")
        return dirty_keys
