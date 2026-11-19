# pylint: disable=too-many-lines
"""Analysis mixin: open/save/load session, analyze, prepare, session config tab."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

import copy
from decimal import Decimal, ROUND_FLOOR
import os
from typing import Any

from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.audio import AUDIO_EXTENSIONS, discover_audio_files, discover_track
from sessionpreplib.config import (
    classify_structured_config_change,
    default_config,
    REPAIR_PARAMS,
    infer_session_overrides,
    resolve_session_config,
)
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import SessionContext, TrackContext
from sessionpreplib.processors import default_processors
from sessionpreplib.topology import build_default_topology
from sessionpreplib.utils import protools_sort_key

from ..helpers import track_analysis_label
from ..prefs.config_pages import _config_section_values
from ..prefs.param_form import ConfigResetButton, _read_widget, _set_widget_value
from ..prefs.param_widgets import build_config_pages, load_config_widgets
from ..session.io import (
    load_session as _load_session_file, restore_layout_sources, save_session as _save_session_file,
)
from ..settings import build_defaults
from ..tracks.table_widgets import (
    DETAIL_TAB_INSET,
    _SETUP_RIGHT_TREE,
    _SortableItem, _make_analysis_cell,
    _TAB_FILE, _TAB_GROUPS, _TAB_SESSION, _TAB_SUMMARY,
    _PAGE_TABS,
    _PHASE_ANALYSIS, _PHASE_TOPOLOGY, _PHASE_SETUP,
)
from ..change_marker import MARKER_COLUMN_WIDTH, set_item_marker, set_tab_marker
from ..controls import (
    FormDoubleSpinBox, content_splitter, fit_navigation_pane, inset_pane, set_action_busy, settings_scroll_area,
)
from ..theme import COLORS, FILE_COLOR_OK, FILE_COLOR_ERROR
from .worker import AnalyzeWorker, PrepareWorker, Phase1AnalyzeWorker, RepairPlanWorker


class AnalysisMixin:  # pylint: disable=too-few-public-methods
    """Session lifecycle: open, save, load, analyze, prepare, session config tab.

    Mixed into ``SessionPrepWindow`` — not meant to be used standalone.
    """

    # ── Session config tab (per-session overrides) ────────────────────────

    def _build_session_settings_tab(self) -> QWidget:
        """Build a tree+stack config editor for per-session overrides."""
        page = QWidget()
        page.setAutoFillBackground(True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET, 0)
        layout.setSpacing(0)

        desc = QLabel(
            "Session-local config adjustments. Changes here apply only to "
            "the current session."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(desc)
        layout.addSpacing(DETAIL_TAB_INSET)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(QLabel("Config Preset:"))
        self._config_preset_combo = QComboBox()
        self._config_preset_combo.setMinimumWidth(140)
        self._populate_config_preset_combo()
        self._config_preset_combo.currentTextChanged.connect(
            self._on_config_preset_changed)
        header.addWidget(self._config_preset_combo)
        header.addStretch()
        reset_btn = ConfigResetButton("Revert to Preset")
        reset_btn.setToolTip(
            "Discard all session-specific changes and reload from the "
            "selected config preset.")
        reset_btn.clicked.connect(self._on_session_config_reset)
        header.addWidget(reset_btn)
        layout.addLayout(header)

        # Tree + Stack
        splitter = content_splitter()

        self._session_tree = QTreeWidget()
        self._session_tree.setColumnCount(2)
        self._session_tree.setHeaderHidden(True)
        self._session_tree.setMinimumWidth(160)
        self._session_tree.setMaximumWidth(220)
        self._session_tree.header().setStretchLastSection(False)
        self._session_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._session_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self._session_tree.setColumnWidth(1, MARKER_COLUMN_WIDTH)
        self._session_tree.currentItemChanged.connect(
            self._on_session_tree_selection)
        splitter.addWidget(inset_pane(
            self._session_tree, (0, DETAIL_TAB_INSET, DETAIL_TAB_INSET, DETAIL_TAB_INSET)))

        self._session_stack = QStackedWidget()
        splitter.addWidget(inset_pane(self._session_stack, (0, 0, 0, DETAIL_TAB_INSET)))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        fit_navigation_pane(splitter)

        layout.addWidget(splitter, 1)

        # Build initial pages from the active global preset
        self._session_page_index: dict[int, int] = {}
        self._session_dirty_items: dict[str, Any] = {}
        self._build_session_pages()

        self._session_tree.expandAll()
        first = self._session_tree.topLevelItem(0)
        if first:
            self._session_tree.setCurrentItem(first)

        return page

    def _build_session_pages(self):
        """Populate the session config tree + stack from the active preset."""

        def _register_page(tree_item, page, dirty_key=None):
            scroll = settings_scroll_area(page)
            idx = self._session_stack.addWidget(scroll)
            self._session_page_index[id(tree_item)] = idx
            if dirty_key:
                self._session_dirty_items[dirty_key] = tree_item

        self._session_daw_custom_widgets = build_config_pages(
            self._session_tree,
            self._active_preset(),
            self._session_widgets,
            _register_page,
            on_reset=self._on_session_config_field_reset,
        )
        self._connect_session_config_dirty_signals()

    def _on_session_tree_selection(self, current, _previous):
        if current is None:
            return
        idx = self._session_page_index.get(id(current))
        if idx is not None:
            self._session_stack.setCurrentIndex(idx)

    def _init_session_config(self):
        """Display the project's effective config and enable its editor."""
        self._populate_config_preset_combo()
        self._load_session_widgets(self._effective_session_config())
        self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._refresh_phase2_dirty_indicators()

    def _load_session_widgets(self, preset: dict[str, Any]):
        """Load values from a config preset dict into session widgets."""
        self._loading_session_widgets = True
        try:
            self._load_session_widgets_inner(preset)
        finally:
            self._loading_session_widgets = False
        self._refresh_session_config_dirty_indicators()

    def _load_session_widgets_inner(self, preset: dict[str, Any]):
        """Inner loader — sets widget values without triggering column refresh."""
        load_config_widgets(
            self._session_widgets, preset,
            self._session_daw_custom_widgets,
            reference_preset=self._active_preset(),
        )

    def _on_session_config_reset(self):
        """Revert session config to the selected config preset."""
        old_config = self._effective_session_config()
        self._session_overrides.clear()
        preset = self._active_preset()
        self._load_session_widgets(preset)
        self._refresh_effective_config(classify_structured_config_change(old_config, preset))
        self._status_bar.showMessage("Session config reverted to preset.")

    def _on_session_config_field_reset(self, path: str, widget: QWidget) -> None:
        """Release one override and restore the field from the selected preset."""
        if not self._session or getattr(self, "_loading_session_widgets", False):
            return
        old_config = self._effective_session_config()
        section, key = path.rsplit(".", 1)
        values = _config_section_values(self._active_preset(), section)
        preset_value = values.get(key, widget._param_spec.default)
        self._session_overrides.pop(path, None)
        self._loading_session_widgets = True
        try:
            _set_widget_value(widget, preset_value)
            widget._set_param_reference(preset_value=preset_value)
        finally:
            self._loading_session_widgets = False
        self._apply_session_config_edit(old_config)

    # 哪 Phase-2 preset difference indicators 哪哪哪哪哪哪哪哪哪哪哪哪哪哪

    def _phase2_change_tooltip(self, index: int) -> str:
        # Name the preset the dirty state is compared against.
        name = getattr(self, "_active_session_preset" if index == _TAB_GROUPS
                       else "_active_config_preset_name", None) or ""
        kind = "group preset" if index == _TAB_GROUPS else "preset"
        return f"Changed from {kind} \u201c{name}\u201d" if name else f"Changed from {kind}"

    def _set_detail_tab_dirty(self, index: int, label: str, dirty: bool) -> None:
        if hasattr(self, "_detail_tabs"):
            set_tab_marker(self._detail_tabs, index, dirty,
                           self._phase2_change_tooltip(index))

    def _connect_session_config_dirty_signals(self) -> None:
        for section, widgets in self._session_widgets.items():
            prefix = "presentation" if section == "_presentation" else section
            for key, widget in widgets:
                self._connect_session_widget_dirty_signal(f"{prefix}.{key}", widget)
        for processor_id, widget in self._session_daw_custom_widgets.items():
            path = f"daw_processors.{processor_id}.{processor_id}_templates"
            widget.templates_changed.connect(
                lambda p=path, w=widget: self._on_session_config_value_changed(
                    p, w.get_templates()))

    def _connect_session_widget_dirty_signal(self, path: str, widget: QWidget) -> None:
        def edited(*_args):
            self._on_session_config_value_changed(path, _read_widget(widget))

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

    def _on_session_config_value_changed(self, path: str, value: Any) -> None:
        if not self._session or getattr(self, "_loading_session_widgets", False):
            return
        old_config = self._effective_session_config()
        self._session_overrides[path] = value
        self._apply_session_config_edit(old_config)

    def _apply_session_config_edit(self, old_config: dict[str, Any]) -> None:
        impact = classify_structured_config_change(
            old_config, self._effective_session_config())
        self._refresh_session_config_dirty_indicators()
        self._refresh_effective_config(impact)
        if impact.requires_phase1:
            self._status_bar.showMessage(
                "Config changed: rerun Track Layout to apply Phase 1 "
                "detector settings.")
        elif impact.requires_phase2:
            self._status_bar.showMessage(
                "Config changed: press Reanalyze to update Phase 2.")
        elif impact.daw_keys:
            self._status_bar.showMessage(
                "Config changed: Phase 3 DAW settings updated.")

    def _refresh_phase2_dirty_indicators(self) -> None:
        if hasattr(self, "_refresh_groups_dirty_indicators"):
            self._refresh_groups_dirty_indicators()
        self._refresh_session_config_dirty_indicators()

    def _refresh_session_config_dirty_indicators(self) -> None:
        if not hasattr(self, "_session_dirty_items"):
            return
        dirty_keys = self._session_config_dirty_keys()
        tooltip = self._phase2_change_tooltip(_TAB_SESSION)
        for key, item in self._session_dirty_items.items():
            set_item_marker(item, 1, key in dirty_keys, tooltip)
        self._set_detail_tab_dirty(_TAB_SESSION, "Config", bool(dirty_keys))

    def _session_config_dirty_keys(self) -> set[str]:
        if not getattr(self, "_session", None):
            return set()
        if not getattr(self, "_session_widgets", None):
            return set()
        current = self._effective_session_config()
        preset = self._active_preset()
        dirty_keys: set[str] = set()

        if current.get("analysis") != preset.get("analysis"):
            dirty_keys.add("analysis")
        if current.get("presentation") != preset.get("presentation"):
            dirty_keys.add("_presentation")

        for key in self._session_widgets:
            if key.startswith("detectors."):
                section_id = key.removeprefix("detectors.")
                if current.get("detectors", {}).get(section_id) != (
                        preset.get("detectors", {}).get(section_id)):
                    dirty_keys.add(key)
            elif key.startswith("processors."):
                section_id = key.removeprefix("processors.")
                if current.get("processors", {}).get(section_id) != (
                        preset.get("processors", {}).get(section_id)):
                    dirty_keys.add(key)
            elif key.startswith("daw_processors."):
                section_id = key.removeprefix("daw_processors.")
                if current.get("daw_processors", {}).get(section_id) != (
                        preset.get("daw_processors", {}).get(section_id)):
                    dirty_keys.add(key)

        if any(key.startswith("detectors.") for key in dirty_keys):
            dirty_keys.add("_presentation")
        if any(key.startswith("processors.") for key in dirty_keys):
            dirty_keys.add("_processors")
        if any(key.startswith("daw_processors.") for key in dirty_keys):
            dirty_keys.add("_daw_processors")
        return dirty_keys

    # ── Slots: file / analysis ────────────────────────────────────────────

    def _cancel_worker(self, worker_attr: str) -> None:
        """Safely detach and cancel a background worker to avoid QThread GC destruction."""
        worker = getattr(self, worker_attr, None)
        if isinstance(worker, (RepairPlanWorker, PrepareWorker)):
            setattr(self, worker_attr, None)
            worker.disconnect(self)
            if isinstance(worker, RepairPlanWorker):
                worker.cancel()
            else:
                worker.requestInterruption()
            worker.finished.connect(worker.deleteLater)
            if worker.isFinished():
                worker.deleteLater()
            return
        if worker is not None:
            if hasattr(worker, "isRunning") and worker.isRunning():
                # Disconnect all signals so it stops updating UI
                try:
                    worker.disconnect(self)
                except RuntimeError:
                    pass
                if hasattr(worker, "cancel"):
                    worker.cancel()
                else:
                    worker.requestInterruption()
                # Maintain python reference until it finishes so it doesn't get GC'd
                if not hasattr(self, "_zombie_workers"):
                    self._zombie_workers = set()
                self._zombie_workers.add(worker)

                # Cleanup reference once finished
                def _cleanup(*args, w=worker):
                    if hasattr(self, "_zombie_workers"):
                        self._zombie_workers.discard(w)

                if hasattr(worker, "finished"):
                    worker.finished.connect(_cleanup)
                if hasattr(worker, "error"):
                    worker.error.connect(_cleanup)
            setattr(self, worker_attr, None)

    def _clear_workspace(self, *, keep_session_config: bool = False):
        """Clear the UI and reset session state."""
        self._on_stop()
        self._cancel_worker("_p1_worker")
        self._cancel_worker("_worker")
        self._cancel_worker("_prepare_worker")
        self._cancel_worker("_repair_plan_worker")
        self._repair_plan_pending = set()
        self._repair_plan_anchors = set()
        self._batch_filenames = set()
        self._track_table.setSortingEnabled(True)
        for attr in ("_prepare_action", "_topo_apply_action", "_transfer_action"):
            action = getattr(self, attr, None)
            if action is not None:
                set_action_busy(action, False)
        self._cancel_worker("_peak_build_worker")
        self._peak_cache = {}
        self._session = None
        self._use_processed_cb.setChecked(False)
        self._update_use_processed_action()
        self._summary = None
        self._current_track = None
        self._selected_detector_overlays = set()
        self._topology_dir = None
        self._source_dir = None
        self._topo_source_tracks = []
        self._topo_topology = None

        if getattr(self, "_topo_input_tree", None) is not None:
            self._topo_input_tree.clear()
            self._topo_input_tree.set_source_dir(None)
        if getattr(self, "_topo_output_tree", None) is not None:
            self._topo_output_tree.clear()
        if getattr(self, "_topo_status_label", None) is not None:
            self._topo_status_label.clear()
        if getattr(self, "_topo_apply_action", None) is not None:
            self._topo_apply_action.setEnabled(False)

        if getattr(self, "_analyze_action", None) is not None:
            self._analyze_action.setEnabled(False)
        if getattr(self, "_topo_reanalyze_action", None) is not None:
            self._topo_reanalyze_action.setEnabled(False)
        if getattr(self, "_topo_reset_action", None) is not None:
            self._topo_reset_action.setEnabled(False)
        if getattr(self, "_save_session_action", None) is not None:
            self._save_session_action.setEnabled(False)

        if getattr(self, "_waveform", None) is not None:
            self._waveform.set_audio(None, 44100)

        self._phase_tabs.setCurrentIndex(_PHASE_TOPOLOGY)
        self._phase_tabs.setTabEnabled(_PHASE_ANALYSIS, False)
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, False)
        self._track_table.setRowCount(0)
        self._track_table.set_source_dir(None)
        self._setup_table.setRowCount(0)
        self._summary_view.clear()
        self._file_report.clear()
        self._wf_container.set_content_available(False)
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, False)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, False)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._refresh_phase2_dirty_indicators()
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        if not keep_session_config:
            self._session_overrides.clear()
            self._active_config_preset_name = self._default_config_preset_name()
        self._session_groups = []
        self._groups_tab_table.setRowCount(0)
        self._folder_tree.clear()
        self._setup_right_stack.setCurrentIndex(0)
        self._project_name_edit.clear()
        self.setWindowTitle("SessionPrep")

    @Slot()
    def _on_open_path(self):
        start_dir = self._config.get("app", {}).get("default_project_dir", "") or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Session Directory", start_dir,
            QFileDialog.ShowDirsOnly,
        )
        if not path:
            return

        self._load_directory(path)

    @Slot()
    def _on_topo_reanalyze(self):
        if self._source_dir:
            self._load_directory(self._source_dir, keep_session_config=True)

    def _load_directory(self, path: str, *, keep_session_config: bool = False):
        log.info("Open folder: %s", path)
        self._clear_workspace(keep_session_config=keep_session_config)
        self._source_dir = path
        self._track_table.set_source_dir(path)
        self._topo_input_tree.set_source_dir(path)
        self._populate_config_preset_combo()
        self._load_session_widgets(self._effective_session_config())
        self._refresh_daw_processors_from_config()

        app_cfg = self._config.get("app", {})
        skip_folders = {
            app_cfg.get("phase1_output_folder", "sp_01_tracklayout"),
            app_cfg.get("phase2_output_folder", "sp_02_prepared"),
            app_cfg.get("peak_cache_folder", "sp_peaks"),
        }
        wav_files = discover_audio_files(
            path, recursive=self._recursive_scan,
            skip_folders=skip_folders)

        if not wav_files:
            self._status_bar.showMessage(f"No audio files found in {path}")
            self._analyze_action.setEnabled(False)
            return

        # Lightweight file discovery — metadata only, no audio data loaded
        tracks = []
        for rel in wav_files:
            try:
                tc = discover_track(os.path.join(path, rel))
                tc.filename = rel  # use relative path as canonical key
                tracks.append(tc)
            except Exception:
                pass  # skip unreadable files

        if not tracks:
            self._status_bar.showMessage(f"No readable audio files in {path}")
            self._analyze_action.setEnabled(False)
            return

        # Preserve original source tracks for Phase 1 topology input table
        self._topo_source_tracks = list(tracks)

        # Phase 1 topology will be built intelligently after analysis finishes.
        self._topo_topology = None

        # Create session with discovered tracks (no topology on session)
        self._session = SessionContext(tracks=tracks, config={})

        self._run_phase1_analysis()

    def _run_phase1_analysis(self):
        """Asynchronously run Phase 1 format and structural checks on newly loaded tracks."""
        self._analyze_action.setEnabled(False)
        if getattr(self, "_topo_apply_action", None):
            self._topo_apply_action.setEnabled(False)

        self._progress_start("Analyzing Format & Layout\u2026")

        config = self._flat_config()
        config["_source_dir"] = self._source_dir

        self._p1_worker = Phase1AnalyzeWorker(self._session, config)
        self._p1_worker.progress.connect(self._on_worker_progress)
        self._p1_worker.progress_value.connect(self._on_worker_progress_value)
        self._p1_worker.finished.connect(self._on_phase1_done)
        self._p1_worker.error.connect(self._on_analyze_error)
        self._p1_worker.start()

    @Slot(object)
    def _on_phase1_done(self, session):
        self._session = session
        if self._p1_worker is not None:
            self._p1_worker.deleteLater()
            self._p1_worker = None

        # Rebuild topology intelligently using Phase 1 results
        self._topo_topology = build_default_topology(session.tracks)

        # Populate Phase 1 topology tables using Phase 1 output issues
        self._populate_topology_tab()

        self._analyze_action.setEnabled(True)
        if getattr(self, "_topo_reanalyze_action", None) is not None:
            self._topo_reanalyze_action.setEnabled(True)
        if getattr(self, "_topo_reset_action", None) is not None:
            self._topo_reset_action.setEnabled(True)

        self._status_bar.showMessage(
            f"Discovered {len(session.tracks)} file(s) from {self._source_dir} \u2014 "
            "review layout, then click Apply"
        )
        self._progress_finish(
            f"Discovered {len(session.tracks)} file(s)")
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._save_session_action.setEnabled(True)
        self.setWindowTitle("SessionPrep")

        # Eagerly build peak cache for all source tracks in the background
        self._start_peak_build(session.tracks, self._source_dir)

    @Slot()
    def _on_save_session(self):
        """Save the current session state to a .spsession file."""
        if not self._session or not self._source_dir:
            return
        if self._prepare_worker is not None or self._repair_plan_worker is not None:
            return
        log.info("Save session")
        default_path = os.path.join(self._source_dir, "session.spsession")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", default_path,
            "SessionPrep Session (*.spsession);;All Files (*)",
        )
        if not path:
            return

        try:
            state = self._capture_session_state()
            _save_session_file(path, state)
            self._status_bar.showMessage(f"Session saved to {path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Session Failed",
                f"Could not save session:\n\n{exc}",
            )

    def _capture_session_state(self) -> dict:
        """Capture the current session state into a dictionary."""
        # Ensure project name is synced from widget
        if self._session:
            self._session.project_name = self._project_name_edit.text().strip()

        active_dp_id = getattr(self, "_active_daw_processor", None)
        active_dp_id = active_dp_id.id if active_dp_id else None

        return {
            "source_dir": self._source_dir,
            "active_config_preset": self._active_config_preset_name,
            "session_config": self._effective_session_config(),
            "session_config_overrides": copy.deepcopy(self._session_overrides),
            "session_groups": self._session_groups,
            "daw_state": self._session.daw_state if self._session else {},
            "active_daw_processor_id": active_dp_id,
            "tracks": self._session.tracks if self._session else [],
            "output_tracks": getattr(self._session, "output_tracks", []) if self._session else [],
            "topology": self._topo_topology,
            "transfer_manifest": self._session.transfer_manifest if self._session else [],
            "topology_applied": self._topology_dir is not None,
            "topology_dir": self._topology_dir,
            "prepare_state": self._session.prepare_state if self._session else "none",
            "prepare_errors": self._session.config.get("_prepare_errors", []) if self._session else [],
            "output_gain_adjustments": self._session.config.get("_output_gain_adjustments", {}) if self._session else {},
            "base_transfer_manifest": self._session.base_transfer_manifest if self._session else [],
            "use_processed": self._use_processed_cb.isChecked(),
            "recursive_scan": self._recursive_scan,
            "project_name": self._session.project_name if self._session else "",
        }

    @Slot()
    def _on_load_session(self):
        """Load a .spsession file and restore the full session state."""
        start_dir = self._source_dir or self._config.get("app", {}).get(
            "default_project_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", start_dir,
            "SessionPrep Session (*.spsession);;All Files (*)",
        )
        if not path:
            return

        log.info("Load session: %s", path)
        try:
            data = _load_session_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Session Failed",
                f"Could not load session:\n\n{exc}",
            )
            return

        self._restore_session_state(data)

    def _restore_session_state(self, data: dict):
        """Restore session state from a dictionary."""
        source_dir = data["source_dir"]
        if not os.path.isdir(source_dir):
            QMessageBox.warning(
                self, "Load Session",
                f"The session's audio directory no longer exists:\n\n{source_dir}\n\n"
                "Please move the files back or open the directory manually.",
            )
            return

        # ── Reset UI (same as _on_open_path but without auto-analyze) ────────
        self._clear_workspace()
        self._source_dir = source_dir
        self._track_table.set_source_dir(source_dir)

        # Re-discover original source tracks for Phase 1 topology input table
        was_recursive = data.get("recursive_scan", False)
        self._recursive_scan = was_recursive
        self._recursive_cb.setChecked(was_recursive)
        app_cfg = self._config.get("app", {})
        skip_folders = {
            app_cfg.get("phase1_output_folder", "sp_01_tracklayout"),
            app_cfg.get("phase2_output_folder", "sp_02_prepared"),
            app_cfg.get("peak_cache_folder", "sp_peaks"),
        }
        source_tracks = []
        for rel in discover_audio_files(
            source_dir, recursive=was_recursive,
            skip_folders=skip_folders,
        ):
            try:
                tc = discover_track(os.path.join(source_dir, rel))
                tc.filename = rel
                source_tracks.append(tc)
            except Exception:
                pass
        self._topo_source_tracks = source_tracks
        self._topo_topology = data.get("topology")

        # ── Restore session-level state ───────────────────────────────────────
        preset_name = data.get("active_config_preset", "Default")
        self._active_config_preset_name = self._existing_config_preset_name(preset_name)
        overrides = data.get("session_config_overrides")
        if isinstance(overrides, dict):
            self._session_overrides = copy.deepcopy(overrides)
        else:
            self._session_overrides = infer_session_overrides(
                self._active_preset(), data.get("session_config"))
        self._session_groups = data.get("session_groups", [])

        self._populate_config_preset_combo()
        effective = self._effective_session_config()
        self._load_session_widgets(effective)

        # Ensure the DAW processors and combo reflect the newly loaded session config
        self._configure_daw_processors()
        daw_state, active_daw_id = self._restore_daw_bindings(data)
        self._populate_daw_combo(active_daw_id)

        # ── Reconstruct SessionContext from saved tracks ──────────────────────
        from sessionpreplib.models import SessionContext
        from sessionpreplib.rendering import build_diagnostic_summary

        tracks = data["tracks"]

        layout_dir = restore_layout_sources(data, app_cfg.get("phase1_output_folder", "sp_01_tracklayout"))

        flat = self._flat_config()

        # Re-instantiate detectors and processors (needed for label filtering)
        all_detectors = default_detectors()
        for d in all_detectors:
            d.configure(flat)
        all_processors = []
        for proc in default_processors():
            proc.configure(flat)
            if proc.enabled:
                all_processors.append(proc)
        all_processors.sort(key=lambda p: p.priority)

        session_config_flat = dict(default_config())
        session_config_flat.update(flat)
        session_config_flat["_source_dir"] = source_dir
        session_config_flat["_prepare_errors"] = data.get("prepare_errors", [])
        session_config_flat["_output_gain_adjustments"] = data.get("output_gain_adjustments", {})

        prepare_state = data.get("prepare_state", "none")
        saved_repairs = _config_section_values(data.get("session_config") or {}, "processors.corrective")
        current_repairs = _config_section_values(effective, "processors.corrective")
        repair_defaults_changed = any(saved_repairs.get(p.key, p.default) !=
                                      current_repairs.get(p.key, p.default) for p in REPAIR_PARAMS)
        if prepare_state == "ready":
            since_saved = infer_session_overrides(effective, data.get("session_config"))
            saved = resolve_session_config(effective, since_saved)
            impact = classify_structured_config_change(saved, effective)
            if impact.requires_phase1 or impact.requires_phase2 or repair_defaults_changed:
                prepare_state = "stale"

        session = SessionContext(
            tracks=tracks,
            config=session_config_flat,
            groups={},
            detectors=all_detectors,
            processors=all_processors,
            daw_state=daw_state,
            prepare_state=prepare_state,
            transfer_manifest=data.get("transfer_manifest", []),
            base_transfer_manifest=data.get("base_transfer_manifest", []),
            project_name=data.get("project_name", ""),
        )

        self._session = session

        # Restore project name widget
        self._project_name_edit.blockSignals(True)
        self._project_name_edit.setText(session.project_name)
        self._project_name_edit.blockSignals(False)

        self._summary = build_diagnostic_summary(session)

        # ── Populate file list in track table ─────────────────────────────────
        self._track_table.setSortingEnabled(False)
        self._track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            item = _SortableItem(track.filename, protools_sort_key(track.filename))
            item.setForeground(FILE_COLOR_OK if track.status == "OK" else FILE_COLOR_ERROR)
            self._track_table.setItem(row, 0, item)
            for col in range(1, 8):
                cell = _SortableItem("", "")
                cell.setForeground(QColor(COLORS["dim"]))
                self._track_table.setItem(row, col, cell)
        self._track_table.setSortingEnabled(True)

        # ── Populate all table widgets and tabs ───────────────────────────────
        self._populate_groups_tab()
        self._populate_group_preset_combo()
        self._populate_table(session)
        self._render_summary()

        # ── Restore topology_dir if topology was applied ─────────────────────
        if layout_dir and os.path.isdir(layout_dir):
            self._topology_dir = layout_dir

        # ── Restore or reconstruct output_tracks ───────────────
        if data.get("output_tracks"):
            session.output_tracks = data["output_tracks"]
        elif self._topology_dir and self._topo_topology:
            import soundfile as sf
            prep_folder = self._config.get("app", {}).get(
                "phase2_output_folder", "sp_02_prepared")
            prep_dir = os.path.join(source_dir, prep_folder)
            # Build group lookup from transfer manifest
            manifest_group: dict[str, str | None] = {
                e.output_filename: e.group
                for e in session.transfer_manifest
            }
            rebuilt: list[TrackContext] = []
            for entry in self._topo_topology.entries:
                topo_path = os.path.join(self._topology_dir,
                                         entry.output_filename)
                if not os.path.isfile(topo_path):
                    continue
                try:
                    info = sf.info(topo_path)
                except Exception:
                    continue
                proc_path = os.path.join(prep_dir, entry.output_filename)
                out_tc = TrackContext(
                    filename=entry.output_filename,
                    filepath=topo_path,
                    audio_data=None,
                    samplerate=info.samplerate,
                    channels=info.channels,
                    total_samples=info.frames,
                    bitdepth=str(info.subtype_info) if hasattr(info, 'subtype_info') else "",
                    subtype=info.subtype,
                    duration_sec=info.duration,
                    processed_filepath=(
                        proc_path if os.path.isfile(proc_path) else None),
                )
                out_tc.group = manifest_group.get(entry.output_filename)
                rebuilt.append(out_tc)
            session.output_tracks = rebuilt

        # ── Enable post-analysis UI ───────────────────────────────────────────
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._refresh_phase2_dirty_indicators()
        self._populate_topology_tab()
        self._phase_tabs.setTabEnabled(_PHASE_ANALYSIS, True)
        self._phase_tabs.setCurrentIndex(_PHASE_ANALYSIS)
        has_manifest = bool(session.transfer_manifest)
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._populate_setup_table()
        self._analyze_action.setEnabled(True)
        self._save_session_action.setEnabled(True)
        self._update_prepare_button()
        # Restore "Use Processed" checkbox state
        use_processed = data.get("use_processed", False)
        session.config["_use_processed"] = use_processed
        self._use_processed_cb.setChecked(use_processed)
        self._update_use_processed_action()

        self._update_daw_lifecycle_buttons()
        # Show folder tree if daw_state already has assignments
        if has_manifest and self._active_daw_processor:
            dp_state = session.daw_state.get(
                self._active_daw_processor.id, {})
            if dp_state.get("folders"):
                self._populate_folder_tree()
                self._setup_right_stack.setCurrentIndex(_SETUP_RIGHT_TREE)
        self._auto_fit_track_table()

        ok_count = sum(1 for t in tracks if t.status == "OK")
        prepare_hint = ""
        if not has_manifest:
            prepare_hint = " — run Prepare to enable Session Setup"
        self._status_bar.showMessage(
            f"Session loaded: {ok_count}/{len(tracks)} tracks OK"
            f"{prepare_hint}"
        )
        self.setWindowTitle("SessionPrep")

        # Eagerly build peak cache for source tracks (Phase 1 waveforms)
        if source_dir and source_tracks:
            self._start_peak_build(source_tracks, source_dir)
        # And for analysis tracks (Phase 2 waveforms) if topology was applied
        if self._topology_dir and tracks:
            self._start_peak_build(tracks, self._topology_dir)
        if repair_defaults_changed:
            self._queue_repair_plan(tracks)

    # ── Analyze ──────────────────────────────────────────────────────────

    @Slot()
    def _on_analyze(self):
        if self._prepare_worker is not None:
            return
        if not self._source_dir:
            return

        # Phase 2 analysis reads from sp_01_tracklayout/ if available
        analyze_dir = self._topology_dir or self._source_dir
        log.info("Analyze: %s", analyze_dir)
        self._cancel_worker("_repair_plan_worker")
        self._repair_plan_pending = set()
        self._repair_plan_anchors = set()
        self._batch_filenames = set()
        self._track_table.setSortingEnabled(True)

        # Snapshot existing group assignments and user overrides so we can
        # restore them after re-analysis (filenames that survive are matched).
        track_settings = {}
        if self._session:
            for t in self._session.tracks:
                overrides = {}
                if t.classification_override:
                    overrides["classification_override"] = t.classification_override
                if t.rms_anchor_override:
                    overrides["rms_anchor_override"] = t.rms_anchor_override
                if t.processor_skip or "audio_classifier" in t.detector_results:
                    overrides["processor_skip"] = set(t.processor_skip)
                if t.repair_overrides:
                    overrides["repair_overrides"] = dict(t.repair_overrides)
                if t.manual_gain_db is not None:
                    overrides["manual_gain_db"] = t.manual_gain_db
                if t.group is not None:
                    overrides["group"] = t.group
                if overrides:
                    track_settings[t.filename] = overrides

        self._analyze_action.setEnabled(False)
        self._current_track = None
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)
        self._track_table.setVisible(False)

        if not self._detail_tabs.isTabEnabled(_TAB_SESSION):
            self._init_session_config()

        # Show progress page
        self._progress_start("Analyzing\u2026")

        config = self._flat_config()
        config["_source_dir"] = self._source_dir
        if self._active_daw_processor:
            config["_fader_ceiling_db"] = self._active_daw_processor.fader_ceiling_db

        self._worker = AnalyzeWorker(analyze_dir, config,
                                     recursive=self._recursive_scan, track_settings=track_settings)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.progress_value.connect(self._on_worker_progress_value)
        self._worker.track_analyzed.connect(self._on_track_analyzed)
        self._worker.track_planned.connect(self._on_track_planned)
        self._worker.finished.connect(self._on_analyze_done)
        self._worker.error.connect(self._on_analyze_error)
        self._worker.start()

    @Slot(str)
    def _on_worker_progress(self, message: str):
        self._progress_message(message)

    @Slot(int, int)
    def _on_worker_progress_value(self, current: int, total: int):
        self._progress_value(current, total)

    @Slot(str, object)
    def _on_track_analyzed(self, filename: str, track):
        """Update the severity column for a track after detectors complete."""
        row = self._find_table_row(filename)
        if row < 0:
            return
        # Ch column
        ch_item = _SortableItem(str(track.channels), track.channels)
        ch_item.setForeground(QColor(COLORS["metadata"]))
        self._track_table.setItem(row, 1, ch_item)
        # Analysis column
        _plain, html, _color, sort_key = track_analysis_label(track)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._track_table.setItem(row, 2, item)
        self._track_table.setCellWidget(row, 2, lbl)

    @Slot(str, object)
    def _on_track_planned(self, filename: str, track):
        """Update classification and gain columns after processors complete."""
        row = self._find_table_row(filename)
        if row < 0:
            return

        # Re-evaluate severity now that processor results inform is_relevant()
        dets = self._session.detectors if self._session else None
        _plain, html, _color, sort_key = track_analysis_label(track, dets)
        lbl, item = _make_analysis_cell(html, sort_key)
        self._track_table.setItem(row, 2, item)
        self._track_table.setCellWidget(row, 2, lbl)

        # Remove previous cell widgets
        self._track_table.removeCellWidget(row, 3)
        self._track_table.removeCellWidget(row, 4)
        self._track_table.removeCellWidget(row, 5)

        from ..widgets import BatchComboBox, TableCellDoubleSpinBox
        from ..theme import (
            FILE_COLOR_SILENT, FILE_COLOR_TRANSIENT, FILE_COLOR_SUSTAINED,
        )

        pr = (
            next(iter(track.processor_results.values()), None)
            if track.processor_results
            else None
        )
        if track.status != "OK":
            cls_item = _SortableItem("Error", "error")
            cls_item.setForeground(FILE_COLOR_ERROR)
            self._track_table.setItem(row, 3, cls_item)
            gain_item = _SortableItem("", 0.0)
            gain_item.setForeground(QColor(COLORS["dim"]))
            self._track_table.setItem(row, 4, gain_item)
        elif pr and pr.classification == "Silent":
            cls_item = _SortableItem("Silent", "silent")
            cls_item.setForeground(FILE_COLOR_SILENT)
            self._track_table.setItem(row, 3, cls_item)
            gain_item = _SortableItem("0.0 dB", 0.0)
            gain_item.setForeground(QColor(COLORS["dim"]))
            self._track_table.setItem(row, 4, gain_item)
        elif pr:
            cls_text = pr.classification or "Unknown"
            if "Transient" in cls_text:
                base_cls = "Transient"
            elif cls_text == "Skip":
                base_cls = "Skip"
            elif "Sustained" in cls_text:
                base_cls = "Sustained"
            else:
                base_cls = "Sustained"

            sort_item = _SortableItem(base_cls, base_cls.lower())
            self._track_table.setItem(row, 3, sort_item)

            combo = BatchComboBox()
            combo.addItems(["Transient", "Sustained", "Skip"])
            combo.blockSignals(True)
            combo.setCurrentText(base_cls)
            combo.blockSignals(False)
            combo.setProperty("track_filename", track.filename)
            self._style_classification_combo(combo, base_cls)
            combo.textActivated.connect(
                lambda text, c=combo: self._on_classification_changed(text, c))
            self._track_table.setCellWidget(row, 3, combo)

            gain_db = pr.gain_db
            gain_sort = _SortableItem(f"{gain_db:+.1f}", gain_db)
            self._track_table.setItem(row, 4, gain_sort)

            spin = TableCellDoubleSpinBox()
            spin.setRange(-60.0, 60.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setSuffix(" dB")
            spin.blockSignals(True)
            spin.setValue(gain_db)
            spin.blockSignals(False)
            spin.setProperty("track_filename", track.filename)
            spin.setEnabled(base_cls != "Skip")
            spin.valueChanged.connect(
                lambda value, s=spin: self._on_gain_changed(value, s))
            self._track_table.setCellWidget(row, 4, spin)

            # RMS Anchor combo (column 5)
            self._create_anchor_combo(row, track)

            # Group combo (column 6)
            self._create_group_combo(row, track)

            # Row background from group color
            self._apply_row_group_color(row, track.group)

        self._auto_fit_group_column()

    @Slot(object, object)
    def _on_analyze_done(self, session, summary):
        # Preserve Phase 1 topology in GUI attribute (already set).
        # Do NOT carry it onto the analysis session — Prepare must see
        # topology=None so it builds a default passthrough from the
        # Phase 1 output files that are now session.tracks.
        if self._session and self._session.topology and not self._topo_topology:
            self._topo_topology = self._session.topology

        # Preserve project name across analysis runs
        current_project_name = self._project_name_edit.text().strip()
        session.project_name = current_project_name

        if self._session and self._session.output_tracks:
            session.output_tracks = self._session.output_tracks
            session.prepare_state = "stale"
            session.transfer_manifest = self._session.transfer_manifest
            session.base_transfer_manifest = self._session.base_transfer_manifest
            session.daw_state = self._session.daw_state
        self._session = session
        self._summary = summary
        self._analyze_action.setEnabled(True)
        self._track_table.setVisible(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if not self._session_groups:
            # First analysis — load from Default group preset
            self._active_session_preset = "Default"
            self._merge_groups_from_preset()
            self._populate_group_preset_combo()
        else:
            # AnalyzeWorker applied the saved choices before planning gain.
            self._populate_groups_tab()
            self._refresh_group_combos()

        # Rebuild track table rows from the new session — required because
        # analysis may have run on sp_01_tracklayout/ whose filenames differ
        # from the rows created during _on_open_path.
        tracks = session.tracks
        self._track_table.setSortingEnabled(False)
        self._track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            item = _SortableItem(
                track.filename, protools_sort_key(track.filename))
            item.setForeground(
                FILE_COLOR_OK if track.status == "OK" else FILE_COLOR_ERROR)
            self._track_table.setItem(row, 0, item)
            for col in range(1, self._track_table.columnCount()):
                cell = _SortableItem("", "")
                cell.setForeground(QColor(COLORS["dim"]))
                self._track_table.setItem(row, col, cell)
        self._track_table.setSortingEnabled(True)

        self._populate_table(session)
        self._render_summary()

        # Switch to tabs — summary tab
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, True)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, True)
        self._refresh_phase2_dirty_indicators()

        # Refresh topology tab (always enabled as Phase 1 landing page)
        self._populate_topology_tab()

        # Enable Session Setup phase only if transfer_manifest is populated
        # (i.e. Prepare has been run at least once). Otherwise, user must
        # run Prepare first to populate output_tracks + transfer_manifest.
        has_manifest = bool(
            self._session and self._session.transfer_manifest)
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, has_manifest)
        if has_manifest:
            self._populate_setup_table()

        # Enable Prepare button; mark stale if previously prepared
        if session.prepare_state == "ready":
            session.prepare_state = "stale"
        self._update_prepare_button()

        self._save_session_action.setEnabled(True)

        ok_count = sum(1 for t in session.tracks if t.status == "OK")
        log.info("Analyze complete: %d/%d tracks OK", ok_count, len(session.tracks))
        self._status_bar.showMessage(
            f"Analysis complete: {ok_count}/{len(session.tracks)} tracks OK"
        )
        self._progress_finish(
            f"Analysis complete: {ok_count}/{len(session.tracks)} tracks OK")

        # Eagerly build peak cache for Phase 2 tracks
        analyze_dir = self._topology_dir or self._source_dir
        if analyze_dir:
            self._start_peak_build(session.tracks, analyze_dir)

    # ── Peak cache ────────────────────────────────────────────────────────

    def _start_peak_build(self, tracks, base_dir: str):
        """Launch a background PeakBuildWorker for all tracks under *base_dir*.

        Before launching, scan existing ``sp_peaks/`` for valid ``.peaks``
        files and load them into the in-memory cache — only stale or missing
        files are queued for background building.
        """
        from ..waveform.compute import PeakBuildWorker
        from ..waveform.peakcache import (
            peaks_path_for, load_peaks, get_source_mtime,
        )

        app_cfg = self._config.get("app", {})
        peaks_folder = app_cfg.get("peak_cache_folder", "sp_peaks")
        peaks_dir = os.path.join(base_dir, peaks_folder)

        if not hasattr(self, "_peak_cache"):
            self._peak_cache = {}

        items = []
        for track in tracks:
            filepath = getattr(track, "filepath", None)
            if not filepath:
                filepath = os.path.join(base_dir, track.filename)
            if not os.path.isfile(filepath):
                continue
            pp = peaks_path_for(peaks_dir, track.filename)
            # Try to reuse existing .peaks from disk
            mtime = get_source_mtime(filepath)
            existing = load_peaks(pp, expected_mtime=mtime)
            if existing is not None:
                self._peak_cache[track.filename] = existing
            else:
                items.append((filepath, track.filename, pp))

        if not items:
            log.info("Peak cache: all %d files already cached", len(tracks))
            return

        log.info(
            "Peak cache: %d cached from disk, %d to build",
            len(tracks) - len(items), len(items),
        )

        self._cancel_worker("_peak_build_worker")
        self._progress_start(f"Building peak cache: 0/{len(items)}")
        worker = PeakBuildWorker(items)
        worker.file_done.connect(self._on_peak_file_done)
        worker.progress.connect(self._progress_message)
        worker.progress_value.connect(self._progress_value)
        worker.all_done.connect(
            lambda: self._on_peak_build_done(len(items)))
        self._peak_build_worker = worker
        worker.start()

    def _on_peak_build_done(self, count: int) -> None:
        log.info("Peak cache build complete (%d files)", count)
        self._progress_finish(f"Peak cache built for {count} file(s)")

    @Slot(str, object)
    def _on_peak_file_done(self, filename: str, peak_data):
        """Cache a newly built PeakData in memory, and push to UI if active."""
        if not hasattr(self, "_peak_cache"):
            self._peak_cache = {}
        self._peak_cache[filename] = peak_data

        # Push to Phase 1 topology preview if active
        if getattr(self, "_topo_wf_filename", None) == filename:
            wf = getattr(self, "_topo_wf_panel", None)
            if wf and hasattr(wf, "waveform"):
                wf.waveform.set_peak_data(peak_data)
                if getattr(wf.waveform, "_loading", False):
                    track = None
                    if hasattr(self, "_topo_track_map"):
                        track = self._topo_track_map().get(filename)
                    if track:
                        wf.waveform.set_preview_mode(
                            track.channels, track.total_samples,
                            track.samplerate, peak_data
                        )

        # Push to Phase 2 analysis preview if active
        cur_track = getattr(self, "_current_track", None)
        if cur_track and cur_track.filename == filename:
            wf = getattr(self, "_waveform", None)
            preview = getattr(self, "_repair_preview", None)
            if preview and preview.track is cur_track and (
                    preview.panel.corrected or not preview.displayed or preview.worker is not None):
                return
            if wf:
                wf.set_peak_data(peak_data)
                if getattr(wf, "_loading", False):
                    wf.set_preview_mode(
                        cur_track.channels, cur_track.total_samples,
                        cur_track.samplerate, peak_data
                    )

    def _prioritize_peak(self, filename: str):
        """If a peak build is in progress, move *filename* to the front."""
        worker = getattr(self, "_peak_build_worker", None)
        if worker is not None and worker.isRunning():
            worker.prioritize(filename)

    @Slot(str)
    def _on_analyze_error(self, message: str):
        self._analyze_action.setEnabled(True)
        self._track_table.setVisible(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        from ..helpers import esc

        self._right_stack.setCurrentIndex(_PAGE_TABS)
        self._detail_tabs.setCurrentIndex(_TAB_SUMMARY)
        self._summary_view.setHtml(self._wrap_html(
            f'<div style="color:{COLORS["problems"]}; font-weight:bold;">'
            f'Analysis Error</div>'
            f'<div style="margin-top:8px;">{esc(message)}</div>'
        ))
        self._status_bar.showMessage(f"Error: {message}")
        self._progress_fail(message)

    def _set_track_repair_choice(self, track, key, value):
        if self._session is None or not any(t is track for t in self._session.tracks):
            return
        if value is None:
            track.repair_overrides.pop(key, None)
        else:
            track.repair_overrides[key] = value
        self._queue_repair_plan([track])

    def _queue_repair_plan(self, tracks, *, refresh_anchors=False):
        """Coalesce edits while DSP is running and publish only the newest choice for each file."""
        if self._session is None:
            return
        pending = getattr(self, "_repair_plan_pending", set())
        changed = {t.filename for t in tracks if t.status == "OK"}
        pending.update(changed)
        preview = getattr(self, "_repair_preview", None)
        if preview and preview.track is not None and preview.track.filename in changed:
            preview.planning()
        if refresh_anchors:
            self._repair_plan_anchors.update(changed)
        self._repair_plan_pending = pending
        if not pending:
            return
        self._mark_prepare_stale()
        self._save_session_action.setEnabled(False)
        self._prepare_action.setEnabled(False)
        if getattr(self, "_repair_plan_worker", None) is not None or self._prepare_worker is not None:
            return
        tracks = [t for t in self._session.tracks if t.filename in pending]
        pending.clear()
        config = {**self._session.config, **self._flat_config()}
        self._repair_plan_worker = RepairPlanWorker(
            self._session, tracks, config, self, refresh_anchors=self._repair_plan_anchors)
        if self._batch_filenames:
            self._repair_plan_worker.progress_value.connect(self._on_worker_progress_value)
        self._repair_plan_worker.completed.connect(self._repair_plan_finished, Qt.QueuedConnection)
        self._repair_plan_worker.start()

    @Slot(object)
    def _repair_plan_finished(self, worker):
        if self._repair_plan_worker is not worker or self._session is not worker.session:
            return
        worker.wait()
        self._repair_plan_worker = None
        worker.deleteLater()
        current = {t.filename: t for t in self._session.tracks}
        committed = []
        for planned in worker.tracks:
            t = current.get(planned.filename)
            if t is None or planned.filename in self._repair_plan_pending:
                continue
            problem = worker.failure or planned._cache.get("repair_error")
            if problem:
                t._cache["repair_error"] = problem
                self._status_bar.showMessage(f"{t.filename}: {problem}")
            else:
                t._cache.pop("repair_error", None)
            t.repair_recipe = planned.repair_recipe
            t.repair_measurements = planned.repair_measurements
            t.processor_results = planned.processor_results
            if planned.filename in worker.refresh_anchors:
                t.detector_results.update(planned.detector_results)
                self._repair_plan_anchors.discard(planned.filename)
            elif "ultrasonic" in planned.detector_results:
                t.detector_results["ultrasonic"] = planned.detector_results["ultrasonic"]
            committed.append(t)
        self._session.config.update(self._flat_config())
        self._apply_linked_group_levels()
        for t in committed:
            self._update_track_row(t.filename)
            row = self._find_table_row(t.filename)
            button = self._track_table.cellWidget(row, 7) if row >= 0 else None
            if button:
                self._update_processing_button_label(button, t, self._session.processors)
            if t is self._current_track:
                self._refresh_file_tab(t)
                self._refresh_track_repair_editor(t)
                self._refresh_repair_preview(t)
        if self._repair_plan_pending:
            self._queue_repair_plan([])
        else:
            self._save_session_action.setEnabled(True)
            self._update_prepare_button()
            if self._batch_filenames:
                self._on_batch_done()

    # ── Prepare handlers ─────────────────────────────────────────────────

    @Slot()
    def _on_prepare(self):
        """Run the Prepare pipeline to generate processed audio files."""
        if not self._session or not self._source_dir:
            return
        if self._prepare_worker is not None or getattr(self, "_repair_plan_worker", None) is not None:
            return  # already running
        log.info("Prepare: %s", self._source_dir)

        output_folder = self._config.get("app", {}).get(
            "phase2_output_folder", "sp_02_prepared")
        output_dir = os.path.join(self._source_dir, output_folder)

        # Refresh pipeline config from current session widgets so that
        # processor enabled/disabled changes made after analysis take effect.
        self._session.config.update(self._flat_config())

        # Use the session's configured processors
        processors = list(self._session.processors) if self._session.processors else []
        set_action_busy(self._prepare_action, True, "Preparing…")
        self._prepare_action.setEnabled(False)
        self._progress_start("Preparing\u2026")

        self._prepare_worker = PrepareWorker(
            self._session, processors, output_dir)
        self._prepare_worker.setParent(self)
        self._save_session_action.setEnabled(False)
        self._analyze_action.setEnabled(False)
        self._update_daw_lifecycle_buttons()
        self._prepare_worker.progress.connect(self._on_prepare_progress)
        self._prepare_worker.progress_value.connect(
            self._on_prepare_progress_value)
        self._prepare_worker.completed.connect(self._on_prepare_done, Qt.QueuedConnection)
        self._prepare_worker.start()

    @Slot(str)
    def _on_prepare_progress(self, message: str):
        self._progress_message(message)

    @Slot(int, int)
    def _on_prepare_progress_value(self, current: int, total: int):
        self._progress_value(current, total)

    @Slot(object)
    def _on_prepare_done(self, worker):
        if self._prepare_worker is not worker or worker.owner is not self._session:
            return
        worker.wait()
        if worker.failure:
            self._on_prepare_error(worker.failure)
            return
        worker.publish()
        outdated = worker.outdated
        self._prepare_worker = None
        worker.deleteLater()
        set_action_busy(self._prepare_action, False)
        self._analyze_action.setEnabled(not self._batch_filenames)
        self._save_session_action.setEnabled(not self._repair_plan_pending)
        self._update_daw_lifecycle_buttons()
        self._update_prepare_button()
        self._update_use_processed_action()

        prepared = len(self._session.config.get("_prepared_files", []))
        errors = self._session.config.get("_prepare_errors", [])
        if self._session.transfer_manifest:
            self._phase_tabs.setTabEnabled(_PHASE_SETUP, True)
        if self._current_track:
            self._refresh_track_repair_editor(self._current_track)
        if outdated:
            msg = "Settings changed during Prepare. Run Prepare again for the current settings."
            self._progress_finish(msg)
            self._status_bar.showMessage(msg)
        elif errors:
            msg = f"Prepare incomplete: {prepared} file(s) written, {len(errors)} error(s)"
            self._progress_fail(msg)
            self._status_bar.showMessage(msg)
            detail = "\n".join(f"• {fn}: {err}" for fn, err in errors)
            adjustments = self._session.config.get("_output_gain_adjustments", {})
            track = next((t for t in self._session.tracks if t.filename in adjustments), None)
            if track is None:
                QMessageBox.warning(self, "Prepare — errors", detail)
            else:
                box = QMessageBox(QMessageBox.Warning, "Output gain needs adjustment",
                                  detail, QMessageBox.Close, self)
                adjust = box.addButton("Adjust Gain", QMessageBox.ActionRole)
                box.exec()
                if box.clickedButton() is adjust:
                    self._adjust_failed_output_gain(track)
        else:
            msg = f"Prepare complete: {prepared} file(s) written"
            log.info("Prepare complete: %d file(s)", prepared)
            self._progress_finish(msg)
            self._status_bar.showMessage(msg)
            if self._session.transfer_manifest and not self._repair_plan_pending:
                self._phase_tabs.setCurrentIndex(_PHASE_SETUP)
        self._populate_setup_table()
        if self._repair_plan_pending:
            self._queue_repair_plan([])

    def _adjust_failed_output_gain(self, track):
        """Offer the measured safe trim. Only Apply changes the track's gain."""
        adjustment = self._session.config.get("_output_gain_adjustments", {}).get(track.filename)
        if not adjustment:
            return
        norm = track.processor_results.get("bimodal_normalize")
        normalizing = (norm is not None and "bimodal_normalize" not in track.processor_skip
                       and norm.classification not in ("Skip", "Silent"))
        current = norm.gain_db if normalizing else float(track.manual_gain_db or 0)
        previous = adjustment.get("gain_db", current)
        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Gain" if normalizing else "Manual output gain")
        form = QFormLayout(dialog)
        form.addRow(QLabel(track.filename))
        gain = FormDoubleSpinBox()
        gain.setRange(-160, 60)
        gain.setDecimals(1)
        gain.setSingleStep(0.1)
        safe = Decimal(str(previous)) - Decimal(str(adjustment["reduction_db"]))
        gain.setValue(float(safe.quantize(Decimal("0.1"), rounding=ROUND_FLOOR)))
        form.addRow("Output gain (dB)", gain)
        forecast = QLabel()
        def show_peak():
            peak = adjustment["peak_db"] + gain.value() - previous
            forecast.setText(f"At this gain, the prepared peak is {peak:+.2f} dBFS.")
        show_peak()
        gain.valueChanged.connect(show_peak)
        form.addRow(forecast)
        form.addRow(QLabel("Original and Corrected audition both stay at unity before output gain."))
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        apply = buttons.addButton("Apply gain", QDialogButtonBox.AcceptRole)
        apply.setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            track.manual_gain_db = gain.value()
            self._queue_repair_plan([track])

    @Slot(str)
    def _on_prepare_error(self, message: str):
        set_action_busy(self._prepare_action, False)
        log.error("Prepare failed: %s", message)
        if self._prepare_worker is not None:
            self._prepare_worker.deleteLater()
            self._prepare_worker = None
        self._prepare_action.setEnabled(True)
        self._analyze_action.setEnabled(not self._batch_filenames)
        self._save_session_action.setEnabled(not self._repair_plan_pending)
        self._update_daw_lifecycle_buttons()
        self._progress_fail(message)
        self._status_bar.showMessage(f"Prepare failed: {message}")
        if self._repair_plan_pending:
            self._queue_repair_plan([])

    def _update_prepare_button(self):
        """Update the Prepare button text and enabled state based on prepare_state."""
        if getattr(self, "_prepare_worker", None) is not None or getattr(self, "_repair_plan_worker", None) is not None:
            return
        if not self._session:
            self._prepare_action.setEnabled(False)
            self._prepare_action.setText("Prepare")
            self._auto_group_action.setEnabled(False)
            return

        state = self._session.prepare_state
        self._prepare_action.setEnabled(True)
        self._auto_group_action.setEnabled(True)
        if state == "ready":
            self._prepare_action.setText("Prepare \u2713")
        elif state == "stale":
            self._prepare_action.setText("Prepare (!)")
        else:
            self._prepare_action.setText("Prepare")

    def _mark_prepare_stale(self):
        """Mark prepared files as stale if they were previously ready."""
        if getattr(self, "_prepare_worker", None) is not None:
            self._prepare_worker.outdated = True
        if self._session and self._session.prepare_state == "ready":
            self._session.prepare_state = "stale"
            self._update_prepare_button()
            self._update_use_processed_action()

    def _refresh_enabled_processors(self):
        """Live-update session.processors and Processing column when a
        processor enabled toggle changes in the session config widgets."""
        if not self._session:
            return
        # Re-evaluate which processors are enabled from current widget values
        flat = self._flat_config()
        new_processors = []
        for proc in default_processors():
            proc.configure(flat)
            if proc.enabled:
                new_processors.append(proc)
        new_processors.sort(key=lambda p: p.priority)
        self._session.processors = new_processors
        self._refresh_processing_column()

    def _refresh_processing_column(self):
        """Rebuild all Processing column buttons from the current
        session.processors list."""
        if not self._session:
            return
        processors = self._session.processors
        for row in range(self._track_table.rowCount()):
            fname_item = self._track_table.item(row, 0)
            if not fname_item:
                continue
            track = next(
                (t for t in self._session.tracks if t.filename == fname_item.text()),
                None,
            )
            if not track or track.status != "OK":
                continue
            # Remove old widget and recreate
            self._track_table.removeCellWidget(row, 7)
            self._create_processing_button(row, track)

    # ── Presentation refresh ─────────────────────────────────────────────

    def _refresh_presentation(self):
        """Re-render all UI after presentation-only config changes (e.g. report_as).

        Reconfigures detector instances in-place, rebuilds the diagnostic
        summary, and refreshes all visible components — without re-reading
        audio or re-running analysis.
        """
        if not self._session:
            return

        # 1. Reconfigure detector instances with updated flat config
        flat = self._flat_config()
        for d in self._session.detectors:
            d.configure(flat)

        # 2. Rebuild diagnostic summary (bucketing depends on report_as)
        from sessionpreplib.rendering import build_diagnostic_summary
        self._summary = build_diagnostic_summary(self._session)

        # 3. Re-render summary HTML
        self._render_summary()

        # 4. Refresh track table Analysis column
        self._refresh_analysis_column()

        # 5. Re-render current track detail
        if self._current_track:
            self._refresh_file_report(self._current_track)

        # 6. Refresh overlay menu (skipped detectors filtered out)
        if self._current_track:
            self._refresh_waveform_issues()

        # 7. Apply any concurrent GUI-only changes
        cmap = self._config.get("app", {}).get("spectrogram_colormap", "magma")
        self._waveform.set_colormap(cmap)

    def _refresh_analysis_column(self):
        """Update the Analysis column for all rows using current detector config."""
        if not self._session:
            return
        track_map = {t.filename: t for t in self._session.tracks}
        dets = self._session.detectors if hasattr(self._session, 'detectors') else None
        self._track_table.setSortingEnabled(False)
        for row in range(self._track_table.rowCount()):
            fname_item = self._track_table.item(row, 0)
            if not fname_item:
                continue
            track = track_map.get(fname_item.text())
            if not track:
                continue
            _plain, html, _color, sort_key = track_analysis_label(track, dets)
            lbl, item = _make_analysis_cell(html, sort_key)
            self._track_table.setItem(row, 2, item)
            self._track_table.setCellWidget(row, 2, lbl)
        self._track_table.setSortingEnabled(True)
