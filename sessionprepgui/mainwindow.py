"""Main application window for SessionPrep GUI."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from PySide6.QtCore import Qt, Slot, QSize, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.config import ConfigChangeImpact, classify_structured_config_change
from sessionpreplib.detectors import detector_help_map
from sessionpreplib.file_io import atomic_writer

from .analysis import SessionConfigController, SessionLifecycleController
from .batch import BatchQueueDock, BatchManager
from .change_marker import install_tab_marker, use_marker_tab_bar
from .controls import bind_primary_action, content_splitter, inset_pane
from .daw import DawTransferController
from .detail import PlaybackController, TrackDetailController
from .detail.file_review import FileReviewPane
from .detail.help_panel import FileReportPane
from .detail.repair_preview import RepairPreview
from .detail.report_style import wrap_report_html
from .log import dbg
from .prefs import PreferencesDialog
from .settings import DEFAULT_SCALE_FACTOR, build_defaults, load_config, save_config
from .state import WorkspaceState
from .theme import apply_dark_theme
from .topology import TopologyController
from .tracks import (
    GroupsController,
    TrackTableController,
    _DraggableTrackTable,
    _PAGE_TABS,
    _PHASE_ANALYSIS,
    _PHASE_SETUP,
    _PHASE_TOPOLOGY,
    _TAB_FILE,
    _TAB_GROUPS,
    _TAB_SESSION,
)
from .waveform import WaveformPanel
from .widgets import ProgressPanel
from .window_geometry import (
    FALLBACK_STARTUP_SIZE,
    calculate_startup_geometry,
    screen_for_startup,
)

log = logging.getLogger(__name__)

MAIN_WINDOW_MIN_HEIGHT_HINT = 360


def _size_text(size: QSize) -> str:
    return f"{size.width()}x{size.height()}"


def _rect_text(rect) -> str:
    return f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"


def _window_state_text(state) -> str:
    value = getattr(state, "value", None)
    if value is None:
        return str(state)
    return f"{state} ({value})"


class _CurrentPageTabWidget(QTabWidget):
    """Tab widget whose minimum hints are based on the active page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def minimumSizeHint(self) -> QSize:
        hint = self._current_page_hint(super().minimumSizeHint(), minimum=True)
        return QSize(hint.width(), min(hint.height(), MAIN_WINDOW_MIN_HEIGHT_HINT))

    def sizeHint(self) -> QSize:
        return self._current_page_hint(super().sizeHint(), minimum=False)

    def _current_page_hint(self, base: QSize, *, minimum: bool) -> QSize:
        current = self.currentWidget()
        stack = self.findChild(QStackedWidget, "qt_tabwidget_stackedwidget")
        if current is None or stack is None:
            return base

        stack_hint = stack.minimumSizeHint() if minimum else stack.sizeHint()
        page_hint = current.minimumSizeHint() if minimum else current.sizeHint()
        tab_hint = (
            self.tabBar().minimumSizeHint()
            if minimum
            else self.tabBar().sizeHint()
        )

        chrome_w = max(0, base.width() - stack_hint.width())
        chrome_h = max(0, base.height() - stack_hint.height())
        return QSize(
            max(page_hint.width(), tab_hint.width()) + chrome_w,
            page_hint.height() + chrome_h,
        )


class SessionPrepWindow(QMainWindow):
    """Thin top-level composition shell hosting WorkspaceState and phase controllers."""

    def __init__(self):
        t_init = time.perf_counter()
        super().__init__()
        self.setWindowTitle("SessionPrep")
        self.setWindowIcon(_app_icon())

        t0 = time.perf_counter()
        loaded_config = load_config()
        dbg(f"load_config: {(time.perf_counter() - t0) * 1000:.1f} ms")
        self._state = WorkspaceState(self, config=loaded_config)
        self._state.recursive_scan = loaded_config.get("app", {}).get("recursive_scan", False)

        self._pt_utils_window = None
        self._log_viewer_window = None
        self._report_screen_connection = None

        t0 = time.perf_counter()
        self._detector_help = detector_help_map()
        dbg(f"detector_help_map: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._batch_manager = BatchManager(self)
        self._close_after_batch = False
        self._close_after_daw = False
        self._batch_manager.finished.connect(self._on_batch_finished)
        self._batch_manager.item_finished.connect(self._on_batch_item_finished)

        t0 = time.perf_counter()
        self._playback = PlaybackController(self)
        self._playback.cursor_updated.connect(lambda pos: self._detail_ctrl.on_cursor_updated(pos))
        self._playback.playback_finished.connect(lambda: self._detail_ctrl.on_playback_finished())
        self._playback.error.connect(lambda msg: self._detail_ctrl.on_playback_error(msg))
        dbg(f"PlaybackController (sounddevice): {(time.perf_counter() - t0) * 1000:.1f} ms")

        t0 = time.perf_counter()
        self._init_ui()
        dbg(f"_init_ui: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._init_batch_dock()

        t0 = time.perf_counter()
        apply_dark_theme(self)
        dbg(f"apply_dark_theme: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._apply_initial_window_geometry()

        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.activated.connect(lambda: self._detail_ctrl.toggle_play())

        dbg(f"SessionPrepWindow.__init__ total: {(time.perf_counter() - t_init) * 1000:.1f} ms")

    def _init_batch_dock(self) -> None:
        self._batch_dock = BatchQueueDock(self)
        self._batch_dock.load_requested.connect(self._on_load_batch_item)
        self._batch_dock.run_batch_requested.connect(self._batch_manager.start_batch)
        self._batch_dock.run_single_requested.connect(self._batch_manager.start_single)
        self._batch_dock.open_project_requested.connect(self._on_open_batch_project_folder)
        self.addDockWidget(Qt.RightDockWidgetArea, self._batch_dock)
        self._batch_dock.hide()

        self._batch_manager.started.connect(self._on_batch_started)
        self._batch_manager.batch_progress_value.connect(self._batch_dock.update_progress)
        self._batch_manager.batch_progress_message.connect(self._batch_dock.update_progress_message)
        self._batch_manager.batch_progress_message.connect(self._status_bar.showMessage)

    # ── Public controller & state accessors ──────────────────────────────

    @property
    def state(self) -> WorkspaceState:
        """Shared cross-phase desktop workspace state."""
        return self._state

    @property
    def topology_controller(self) -> TopologyController:
        """Phase 1 (Track Layout) controller."""
        return self._topology_ctrl

    @property
    def daw_controller(self) -> DawTransferController:
        """Phase 3 (DAW Transfer) controller."""
        return self._daw_ctrl

    @property
    def detail_controller(self) -> TrackDetailController:
        """Phase 2 (File detail, waveform & playback) controller."""
        return self._detail_ctrl

    @property
    def track_table_controller(self) -> TrackTableController:
        """Phase 2 & Phase 3 track table controller."""
        return self._table_ctrl

    @property
    def groups_controller(self) -> GroupsController:
        """Phase 2 Groups controller."""
        return self._groups_ctrl

    @property
    def config_controller(self) -> SessionConfigController:
        """Phase 2 Session Config controller."""
        return self._config_ctrl

    @property
    def lifecycle_controller(self) -> SessionLifecycleController:
        """Session open/save/load and Phase 1/2 analysis & Prepare lifecycle controller."""
        return self._lifecycle_ctrl

    # ── Startup geometry ─────────────────────────────────────────────────

    def _apply_initial_window_geometry(self):
        """Size and center the main window on the most relevant screen."""
        screen = screen_for_startup()
        if screen is None:
            self.resize(FALLBACK_STARTUP_SIZE)
            log.debug(
                "Startup geometry: no screen detected; fallback_size=%s actual=%s",
                _size_text(FALLBACK_STARTUP_SIZE),
                _rect_text(self.geometry()),
            )
            return

        available = screen.availableGeometry()
        full = screen.geometry()
        minimum_hint = self.minimumSizeHint()
        minimum_size = self.minimumSize()
        geometry = calculate_startup_geometry(available, minimum_hint)
        log.debug(
            "Startup geometry request: screen=%r dpr=%.2f full=%s available=%s "
            "minimum_hint=%s minimum_size=%s requested=%s",
            screen.name(),
            screen.devicePixelRatio(),
            _rect_text(full),
            _rect_text(available),
            _size_text(minimum_hint),
            _size_text(minimum_size),
            _rect_text(geometry),
        )
        self.setGeometry(geometry)
        log.debug(
            "Startup geometry applied before show: requested=%s actual=%s frame=%s",
            _rect_text(geometry),
            _rect_text(self.geometry()),
            _rect_text(self.frameGeometry()),
        )
        self._log_startup_size_hints()

    def _log_startup_size_hints(self):
        """Log the largest propagated size hints for startup diagnosis."""
        phase_info = ""
        if hasattr(self, "_phase_tabs"):
            phase_info = (
                f" phase_current={self._phase_tabs.currentIndex()}"
                f" phase_min_hint={_size_text(self._phase_tabs.minimumSizeHint())}"
                f" phase_size_hint={_size_text(self._phase_tabs.sizeHint())}"
            )
        log.debug(
            "Startup geometry hints: window_min_hint=%s window_size_hint=%s "
            "central_min_hint=%s central_size_hint=%s%s",
            _size_text(self.minimumSizeHint()),
            _size_text(self.sizeHint()),
            _size_text(self.centralWidget().minimumSizeHint()) if self.centralWidget() else "none",
            _size_text(self.centralWidget().sizeHint()) if self.centralWidget() else "none",
            phase_info,
        )
        if hasattr(self, "_phase_tabs"):
            for index in range(self._phase_tabs.count()):
                page = self._phase_tabs.widget(index)
                if page is None:
                    continue
                log.debug(
                    "Startup geometry phase tab: index=%d title=%r visible=%s "
                    "min_hint=%s size_hint=%s min_size=%s",
                    index,
                    self._phase_tabs.tabText(index),
                    page.isVisibleTo(self),
                    _size_text(page.minimumSizeHint()),
                    _size_text(page.sizeHint()),
                    _size_text(page.minimumSize()),
                )

        contributors = []
        for widget in self.findChildren(QWidget):
            hint = widget.minimumSizeHint()
            size_hint = widget.sizeHint()
            if hint.height() <= 120 and size_hint.height() <= 180:
                continue
            contributors.append((
                max(hint.height(), size_hint.height()),
                type(widget).__name__,
                widget.objectName() or "-",
                _size_text(hint),
                _size_text(size_hint),
                _size_text(widget.minimumSize()),
                widget.isVisibleTo(self),
            ))
        contributors.sort(reverse=True)
        for _, cls_name, object_name, hint, size_hint, min_size, visible in contributors[:12]:
            log.debug(
                "Startup geometry hint contributor: class=%s object=%s "
                "visible=%s min_hint=%s size_hint=%s min_size=%s",
                cls_name,
                object_name,
                visible,
                hint,
                size_hint,
                min_size,
            )

    # ── MainWindowShell protocol methods ─────────────────────────────────

    def show_status(self, message: str) -> None:
        self._status_bar.showMessage(message)

    def progress_start(self, text: str, *, owner: Any = None) -> None:
        if owner is None:
            self._progress_start(text)
        else:
            self._progress_start(text, owner=owner)

    def progress_message(self, text: str) -> None:
        self._progress_message(text)

    def progress_value(self, current: int, total: int) -> None:
        self._progress_value(current, total)

    def progress_finish(self, text: str) -> None:
        self._progress_finish(text)

    def progress_fail(self, text: str) -> None:
        self._progress_fail(text)

    def clear_progress(self) -> None:
        self._clear_progress()

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None:
        self._phase_tabs.setTabEnabled(phase_index, enabled)

    def switch_to_phase(self, phase_index: int) -> None:
        self._phase_tabs.setCurrentIndex(phase_index)

    def current_phase(self) -> int:
        return self._phase_tabs.currentIndex()

    def is_phase_enabled(self, phase_index: int) -> bool:
        return self._phase_tabs.isTabEnabled(phase_index)

    def register_phase_progress_layout(self, phase_index: int, layout: Any) -> None:
        self._register_phase_progress_layout(phase_index, layout)

    def dialog_parent(self) -> QWidget:
        return self

    def is_batch_mode(self) -> bool:
        action = getattr(self, "_batch_mode_action", None)
        return bool(action and action.isChecked())

    def is_closing_after_daw(self) -> bool:
        return bool(self._close_after_daw)

    def resume_close_after_daw(self) -> None:
        self._resume_close_after_daw()

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    # ── UI setup & controller composition ────────────────────────────────

    def _init_ui(self):
        self._init_menus()
        self._phase_progress_layouts = {}
        self._phase_progress_panels = {}
        self._clear_progress()

        self._phase_tabs = _CurrentPageTabWidget()
        self._phase_tabs.setObjectName("phaseTabs")
        self._phase_tabs.setDocumentMode(True)

        self._topology_ctrl = TopologyController(
            self._state,
            self,
            self._playback,
            on_prioritize_peak=lambda fp: self._lifecycle_ctrl.prioritize_peak(fp),
            parent=self,
        )
        self._topology_ctrl.open_folder_requested.connect(
            lambda: self._lifecycle_ctrl.open_directory_dialog()
        )
        self._topology_ctrl.reanalyze_requested.connect(
            lambda: self._lifecycle_ctrl.reanalyze_topology()
        )
        self._topology_ctrl.topology_edited.connect(
            lambda: self._lifecycle_ctrl.mark_prepare_stale()
        )
        self._topology_ctrl.layout_applied.connect(
            lambda _out_dir: self._lifecycle_ctrl.analyze()
        )
        self._phase_tabs.addTab(self._topology_ctrl.page, "Phase 1: Track Layout")

        analysis_page = self._build_analysis_page()
        self._phase_tabs.addTab(analysis_page, "Phase 2: Analysis && Preparation")
        self._phase_tabs.setTabEnabled(_PHASE_ANALYSIS, False)

        self._daw_ctrl = self._create_daw_controller()
        self._table_ctrl = self._create_track_table_controller()
        self._lifecycle_ctrl = self._create_lifecycle_controller()
        self._phase_tabs.addTab(self._daw_ctrl.page, "Phase 3: DAW Transfer")
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, False)
        self._phase_tabs.currentChanged.connect(self._on_phase_tab_changed)

        self.setCentralWidget(self._phase_tabs)

        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(
            "QStatusBar { background-color: #1e1e1e; border-top: 1px solid #444; }"
            "QStatusBar::item { border: none; }"
        )
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Open a directory containing .wav / .aif files to begin.")

    def _build_analysis_page(self) -> QWidget:
        analysis_page = QWidget()
        analysis_layout = QVBoxLayout(analysis_page)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(0)
        self._init_analysis_toolbar()
        analysis_layout.addWidget(self._analysis_toolbar)
        self._main_splitter = content_splitter()
        self._main_splitter.addWidget(self._build_left_panel())
        self._main_splitter.addWidget(self._build_right_panel())
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 2)
        self._main_splitter.setSizes([620, 480])
        analysis_layout.addWidget(self._main_splitter, 1)
        self._register_phase_progress_layout(_PHASE_ANALYSIS, analysis_layout)
        return analysis_page

    def _create_daw_controller(self) -> DawTransferController:
        return DawTransferController(
            self._state,
            self,
            on_populate_setup_table=lambda: self._table_ctrl.populate_setup_table(),
            on_group_color_map=lambda: self._groups_ctrl.group_color_map(),
            on_tint_group_color=lambda name, gcm=None: self._groups_ctrl.tint_group_color(name, gcm),
            on_normalize_groups=lambda groups: self._groups_ctrl.normalize_session_groups(groups),
            on_capture_session_state=lambda: self._lifecycle_ctrl.capture_session_state(),
            on_enqueue_batch_item=lambda item: self._batch_dock.add_item(item),
            on_clear_workspace=lambda: self._lifecycle_ctrl.clear_workspace(),
            on_is_prepare_running=lambda: (
                getattr(self, "_lifecycle_ctrl", None) is not None
                and self._lifecycle_ctrl.prepare_worker is not None
            ),
            parent=self,
        )

    def _create_track_table_controller(self) -> TrackTableController:
        return TrackTableController(
            self._state,
            self,
            track_table=self._track_table,
            setup_table=self._daw_ctrl.page.setup_table,
            main_splitter=self._main_splitter,
            right_stack=self._right_stack,
            detail_tabs=self._detail_tabs,
            on_show_track_detail=lambda t: self._detail_ctrl.show_track_detail(t),
            on_refresh_file_report=lambda t=None: self._detail_ctrl.refresh_file_report(t),
            on_refresh_waveform_issues=lambda: self._detail_ctrl.refresh_waveform_issues(),
            on_open_track_repairs=lambda t, det=None: self._detail_ctrl.open_track_repairs(t, det),
            on_create_group_combo=lambda row, t: self._groups_ctrl.create_group_combo(row, t),
            on_apply_row_group_color=lambda r, g, gcm=None, table=None: self._groups_ctrl.apply_row_group_color(
                r, g, gcm, table=table
            ),
            on_apply_linked_group_levels=lambda: self._groups_ctrl.apply_linked_group_levels(),
            on_group_color_map=lambda: self._groups_ctrl.group_color_map(),
            on_group_rank_map=lambda: self._groups_ctrl.group_rank_map(),
            on_gain_linked_map=lambda: self._groups_ctrl.gain_linked_map(),
            on_group_display_name=lambda n, glm=None: self._groups_ctrl.group_display_name(n, glm),
            on_auto_fit_group_column=lambda: self._table_ctrl.auto_fit_track_table(),
            on_apply_setup_column_modes=self._daw_ctrl.page.apply_table_column_modes,
            on_schedule_setup_splitter_fit=self._daw_ctrl.page.schedule_splitter_fit,
            on_set_setup_table_populating=lambda v: setattr(self._daw_ctrl, "setup_table_populating", v),
            on_mark_prepare_stale=lambda: self._lifecycle_ctrl.mark_prepare_stale(),
            on_queue_repair_plan=lambda *a, **kw: self._lifecycle_ctrl.queue_repair_plan(*a, **kw),
            on_is_analysis_running=lambda: bool(
                self._lifecycle_ctrl.analyze_worker and self._lifecycle_ctrl.analyze_worker.isRunning()
            ),
            on_is_repair_running=lambda: self._lifecycle_ctrl.repair_plan_worker is not None,
            on_set_analyze_enabled=self._analyze_action.setEnabled,
            parent=self,
        )

    def _create_lifecycle_controller(self) -> SessionLifecycleController:
        return SessionLifecycleController(
            self._state,
            self,
            topology_ctrl=self._topology_ctrl,
            daw_ctrl=self._daw_ctrl,
            detail_ctrl=self._detail_ctrl,
            table_ctrl=self._table_ctrl,
            groups_ctrl=self._groups_ctrl,
            config_ctrl=self._config_ctrl,
            track_table=self._track_table,
            summary_view=self._summary_view,
            file_report=self._file_report,
            detail_tabs=self._detail_tabs,
            right_stack=self._right_stack,
            analyze_action=self._analyze_action,
            auto_group_action=self._auto_group_action,
            prepare_action=self._prepare_action,
            save_session_action=self._save_session_action,
            repair_preview=self._repair_preview,
            on_progress_owner=lambda: getattr(self, "_progress_owner", None),
            on_update_tools_menu=self._update_tools_menu,
            parent=self,
        )

    # ── Global phase progress bar ────────────────────────────────────────

    def _register_phase_progress_layout(self, phase_index: int, layout: QVBoxLayout) -> None:
        self._phase_progress_layouts[phase_index] = layout
        panel = ProgressPanel()
        panel.setObjectName(f"phase{phase_index}ProgressPanel")
        self._phase_progress_panels[phase_index] = panel
        layout.addWidget(panel)

    def _progress_panel_for_phase(self, phase_index: int | None = None):
        if phase_index is None:
            phase_index = self._phase_tabs.currentIndex()
        return self._phase_progress_panels[phase_index]

    def _show_progress_on_phase(self, phase_index: int) -> None:
        panel = self._progress_panel_for_phase(phase_index)
        for other_phase, other_panel in self._phase_progress_panels.items():
            if other_phase != phase_index:
                other_panel.setVisible(False)
        panel.start(self._progress_text)
        panel.set_progress(self._progress_current, self._progress_total)
        self._progress_active_phase = phase_index

    def _clear_progress(self) -> None:
        for panel in self._phase_progress_panels.values():
            panel.hide()
        self._progress_active_phase = None
        self._progress_owner = None
        self._progress_running = False
        self._progress_text = ""
        self._progress_current = 0
        self._progress_total = 1

    def _progress_start(self, text: str, *, owner=None) -> None:
        self._progress_owner = owner
        self._progress_running = True
        self._progress_text = text
        self._progress_current = 0
        self._progress_total = 1
        self._show_progress_on_phase(self._phase_tabs.currentIndex())

    def _progress_message(self, text: str) -> None:
        self._progress_text = text
        self._progress_panel_for_phase(self._progress_active_phase).set_message(text)

    def _progress_value(self, current: int, total: int) -> None:
        self._progress_current = current
        self._progress_total = max(total, 1)
        self._progress_panel_for_phase(self._progress_active_phase).set_progress(current, total)

    def _progress_finish(self, text: str) -> None:
        self._progress_owner = None
        self._progress_running = False
        self._progress_text = text
        self._progress_panel_for_phase(self._progress_active_phase).finish(text)

    def _progress_fail(self, text: str) -> None:
        self._progress_owner = None
        self._progress_running = False
        self._progress_text = text
        self._progress_panel_for_phase(self._progress_active_phase).fail(text)

    # ── Menus & batch queue handlers ─────────────────────────────────────

    def _init_menus(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open &Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(lambda: self._lifecycle_ctrl.open_directory_dialog())
        file_menu.addAction(open_action)

        load_session_action = QAction("&Load Session...", self)
        load_session_action.setShortcut("Ctrl+Shift+O")
        load_session_action.triggered.connect(lambda: self._lifecycle_ctrl.load_session_dialog())
        file_menu.addAction(load_session_action)

        load_batch_action = QAction("Load Batch Queue...", self)
        load_batch_action.triggered.connect(self._on_load_batch_queue)
        file_menu.addAction(load_batch_action)

        file_menu.addSeparator()

        self._save_session_action = QAction("&Save Session...", self)
        self._save_session_action.setShortcut("Ctrl+S")
        self._save_session_action.setEnabled(False)
        self._save_session_action.triggered.connect(
            lambda: self._lifecycle_ctrl.save_session_dialog()
        )
        file_menu.addAction(self._save_session_action)

        self._save_batch_action = QAction("Save Batch Queue...", self)
        self._save_batch_action.triggered.connect(self._on_save_batch_queue)
        file_menu.addAction(self._save_batch_action)

        file_menu.addSeparator()

        self._batch_mode_action = QAction("Batch Processing Mode", self)
        self._batch_mode_action.setCheckable(True)
        self._batch_mode_action.toggled.connect(self._on_batch_mode_toggled)
        file_menu.addAction(self._batch_mode_action)

        file_menu.addSeparator()

        prefs_action = QAction("&Preferences...", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        prefs_action.triggered.connect(self._on_preferences)
        file_menu.addAction(prefs_action)

        file_menu.addSeparator()

        about_action = QAction("&About SessionPrep", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self._on_about)
        file_menu.addAction(about_action)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = self.menuBar().addMenu("&Tools")

        self._log_viewer_action = QAction("Log Viewer", self)
        self._log_viewer_action.triggered.connect(self._on_open_log_viewer)
        tools_menu.addAction(self._log_viewer_action)

        tools_menu.addSeparator()

        self._pt_utils_action = QAction("Pro Tools Utils\u2026", self)
        self._pt_utils_action.triggered.connect(self._on_open_pt_utils)
        tools_menu.addAction(self._pt_utils_action)
        self._update_tools_menu()

    @Slot()
    def _on_save_batch_queue(self):
        if not self._batch_dock.has_items:
            QMessageBox.information(self, "Save Batch Queue", "The batch queue is empty.")
            return

        start_dir = self._state.config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Batch Queue",
            start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)",
        )
        if not path:
            return

        try:
            state = self._batch_dock.get_state()
            with atomic_writer(path) as f:
                json.dump(state, f, indent=2)
            self._status_bar.showMessage(f"Batch queue saved to {path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Batch Queue Failed", f"Could not save batch queue:\n\n{exc}"
            )

    @Slot()
    def _on_load_batch_queue(self):
        start_dir = self._state.config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Batch Queue",
            start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Batch Queue Failed", f"Could not load batch queue:\n\n{exc}"
            )
            return

        if not isinstance(state, list):
            QMessageBox.critical(
                self, "Invalid File", "The selected file is not a valid batch queue format."
            )
            return

        append = False
        if self._batch_dock.has_items:
            msg = QMessageBox(self)
            msg.setWindowTitle("Load Batch Queue")
            msg.setText("How would you like to load the batch queue?")
            replace_btn = msg.addButton("Replace", QMessageBox.AcceptRole)
            append_btn = msg.addButton("Append", QMessageBox.AcceptRole)
            msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.exec()
            choice = msg.clickedButton()
            if choice == append_btn:
                append = True
            elif choice != replace_btn:
                return

        try:
            self._batch_dock.load_state(state, append=append)
        except ValueError as exc:
            QMessageBox.critical(self, "Load Batch Queue Failed", str(exc))
            return

        self._batch_mode_action.setChecked(True)
        self._lifecycle_ctrl.clear_workspace()
        self._status_bar.showMessage(f"Batch queue loaded from {path}")

    @Slot(bool)
    def _on_batch_mode_toggled(self, checked: bool):
        self._batch_dock.setVisible(checked)
        self._daw_ctrl.update_lifecycle_buttons()

    @Slot(object)
    def _on_load_batch_item(self, item):
        if self._state.session:
            reply = QMessageBox.question(
                self,
                "Load Session",
                "You have an active session in the workspace. Discard current workspace and load from queue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        if self._lifecycle_ctrl.restore_session_state(item.session_state):
            self._batch_dock.remove_item(item.id)

    @Slot(object)
    def _on_open_batch_project_folder(self, item):
        daw_processor = next(
            (dp for dp in self._state.daw_processors if dp.id == item.daw_processor_id),
            None,
        )
        if not daw_processor:
            QMessageBox.warning(
                self,
                "Open Project Folder",
                f"The configured DAW processor '{item.daw_processor_name}' is not available.",
            )
            return

        project_dir = getattr(daw_processor, "project_dir", "").strip()
        if not project_dir:
            QMessageBox.information(
                self,
                "Open Project Folder",
                f"No project directory configured for {daw_processor.name}.",
            )
            return

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if item.output_path and os.path.exists(os.path.dirname(item.output_path)):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(item.output_path)))
            return

        target_path = (
            os.path.join(project_dir, item.project_name) if item.project_name else project_dir
        )
        if os.path.isdir(target_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_path))
        elif os.path.isdir(project_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(project_dir))
        else:
            QMessageBox.warning(
                self,
                "Open Project Folder",
                f"The directory does not exist:\n\n{project_dir}",
            )

    @Slot()
    def _on_batch_started(self):
        self._batch_dock.set_running_state(True)

    @Slot()
    def _on_batch_finished(self):
        if self._close_after_batch:
            QTimer.singleShot(0, self.close)
        else:
            self._batch_dock.set_running_state(False)

    @Slot(str, str, str)
    def _on_batch_item_finished(self, item_id: str, status: str, result_text: str):
        self._batch_dock.update_item(item_id, status, result_text)

    # ── Phase 2 toolbar & panel builders ─────────────────────────────────

    def _init_analysis_toolbar(self):
        self._analysis_toolbar = QToolBar("Analysis")
        self._analysis_toolbar.setIconSize(QSize(16, 16))
        self._analysis_toolbar.setMovable(False)
        self._analysis_toolbar.setFloatable(False)

        self._analyze_action = QAction("Reanalyze", self)
        self._analyze_action.setEnabled(False)
        self._analyze_action.triggered.connect(lambda: self._lifecycle_ctrl.analyze())
        self._analysis_toolbar.addAction(self._analyze_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._analysis_toolbar.addWidget(spacer)

        self._auto_group_action = QAction("Auto-Group", self)
        self._auto_group_action.setEnabled(False)
        self._auto_group_action.triggered.connect(lambda: self._groups_ctrl.auto_group())
        self._analysis_toolbar.addAction(self._auto_group_action)

        self._prepare_action = QAction("Prepare", self)
        self._prepare_action.setEnabled(False)
        self._prepare_action.triggered.connect(lambda: self._lifecycle_ctrl.prepare())
        self._analysis_toolbar.addAction(self._prepare_action)
        bind_primary_action(self._analysis_toolbar, self._prepare_action)

    def _build_left_panel(self) -> QWidget:
        self._track_table = _DraggableTrackTable()
        self._track_table.setColumnCount(8)
        self._track_table.setHorizontalHeaderLabels(
            ["File", "Ch", "Analysis", "Classification", "Gain", "RMS Anchor", "Group", "Processing"]
        )
        self._track_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._track_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._track_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._track_table.verticalHeader().setDefaultSectionSize(24)
        self._track_table.verticalHeader().setVisible(False)
        self._track_table.setMinimumWidth(300)
        self._track_table.setShowGrid(True)
        self._track_table.setAlternatingRowColors(True)
        self._track_table.setSortingEnabled(True)

        header = self._track_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        for col in range(2, 8):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
        for col, width in ((1, 30), (2, 150), (3, 120), (4, 90), (5, 100), (6, 140), (7, 130)):
            header.resizeSection(col, width)
        header.sectionResized.connect(
            lambda index, *_: index and self._track_table.fit_file_column()
        )
        self._track_table.fit_file_column()

        self._track_table.cellClicked.connect(
            lambda row, col: self._table_ctrl.on_row_clicked(row, col)
        )
        self._track_table.currentCellChanged.connect(
            lambda r, c, pr, pc: self._table_ctrl.on_current_cell_changed(r, c, pr, pc)
        )
        return inset_pane(self._track_table, (0, 0, 0, 0))

    def _build_file_tab_widgets(self) -> QWidget:
        """Build the File detail splitter (report + repair review + waveform)."""
        self._file_splitter = content_splitter(Qt.Vertical)

        self._file_report = self._make_report_browser()
        self._file_report_pane = FileReportPane(self._file_report)
        self._file_review = FileReviewPane(self._file_report_pane, self._file_splitter)
        self._repair_editor = self._file_review.editor
        self._file_report.repairRequested.connect(
            lambda key: self._detail_ctrl.open_track_repairs(self._state.current_track, key)
        )
        self._repair_editor.choiceChanged.connect(
            lambda fn, det, ch: self._lifecycle_ctrl.set_track_repair_choice(fn, det, ch)
        )
        self._repair_editor.adjustGainRequested.connect(
            lambda fn: self._lifecycle_ctrl.adjust_failed_output_gain(fn)
        )
        self._file_review.processingFocusRequested.connect(
            lambda fn: self._detail_ctrl.focus_processing_menu(fn)
        )
        self._file_report.visualizeRequested.connect(
            lambda key: self._detail_ctrl.visualize_detector(key)
        )
        self._file_report.sourceLocationsRequested.connect(
            lambda key, corr: self._detail_ctrl.show_source_locations(key, corr)
        )
        self._file_report.verificationRetryRequested.connect(
            lambda: self._detail_ctrl.retry_detector_verification()
        )
        self._file_report.reportChanged.connect(
            lambda: self._detail_ctrl.on_file_report_context_changed()
        )
        self._file_splitter.addWidget(self._file_review)

        self._wf_panel = WaveformPanel(analysis_mode=True)
        self._wf_panel.play_clicked.connect(lambda: self._detail_ctrl.play())
        self._wf_panel.stop_clicked.connect(lambda: self._detail_ctrl.stop())
        self._wf_panel.position_clicked.connect(lambda pos: self._detail_ctrl.on_waveform_seek(pos))
        self._wf_panel.display_mode_changed.connect(
            lambda mode: self._detail_ctrl.on_display_mode_changed(mode)
        )
        self._wf_panel.waveform.set_invert_scroll(
            self._state.config.get("app", {}).get("invert_scroll", "default")
        )

        self._repair_preview = RepairPreview(
            self._state,
            self._wf_panel,
            self._file_report,
            self._playback,
            get_detail_ctrl=lambda: self._detail_ctrl,
            get_lifecycle_ctrl=lambda: self._lifecycle_ctrl,
            get_table_ctrl=lambda: self._table_ctrl,
            parent=self,
        )
        self._detail_ctrl = self._create_detail_controller()
        self._wf_panel.fft_group.triggered.connect(self._detail_ctrl.on_spec_fft_changed)
        self._wf_panel.win_group.triggered.connect(self._detail_ctrl.on_spec_window_changed)
        self._wf_panel.cmap_group.triggered.connect(self._detail_ctrl.on_spec_cmap_changed)
        self._wf_panel.floor_group.triggered.connect(self._detail_ctrl.on_spec_floor_changed)
        self._wf_panel.ceil_group.triggered.connect(self._detail_ctrl.on_spec_ceil_changed)

        self._file_splitter.addWidget(self._wf_panel)
        self._file_splitter.setCollapsible(1, False)
        self._file_splitter.setStretchFactor(0, 3)
        self._file_splitter.setStretchFactor(1, 1)
        self._file_splitter.setSizes([620, 272])
        return self._file_splitter

    def _create_detail_controller(self) -> TrackDetailController:
        return TrackDetailController(
            self._state,
            self,
            summary_view=self._summary_view,
            file_report=self._file_report,
            file_review=self._file_review,
            repair_editor=self._repair_editor,
            detail_tabs=self._detail_tabs,
            track_table=self._track_table,
            wf_panel=self._wf_panel,
            playback=self._playback,
            repair_preview=self._repair_preview,
            topology_ctrl=self._topology_ctrl,
            on_find_table_row=lambda fn: self._table_ctrl.find_table_row(fn),
            on_queue_repair_plan=lambda *a, **kw: self._lifecycle_ctrl.queue_repair_plan(*a, **kw),
            on_prioritize_peak=lambda fp: self._lifecycle_ctrl.prioritize_peak(fp),
            parent=self,
        )

    def _build_right_panel(self) -> QWidget:
        """Build the right-hand side analysis detail tabs."""
        self._right_stack = QStackedWidget()
        self._right_stack.setMinimumWidth(400)

        self._detail_tabs = QTabWidget()
        use_marker_tab_bar(self._detail_tabs)
        self._detail_tabs.setDocumentMode(True)
        self._detail_tabs.currentChanged.connect(self._on_detail_tab_changed)

        self._summary_view = self._make_report_browser()
        self._detail_tabs.addTab(self._summary_view, "Summary")

        self._detail_tabs.addTab(self._build_file_tab_widgets(), "File")
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)

        self._groups_ctrl = GroupsController(
            self._state,
            self,
            track_table=self._track_table,
            detail_tabs=self._detail_tabs,
            on_find_table_row=lambda fn: self._table_ctrl.find_table_row(fn),
            on_populate_setup_table=lambda: self._table_ctrl.populate_setup_table(),
            on_refresh_file_tab=lambda t: self._table_ctrl.refresh_file_tab(t),
            on_mark_prepare_stale=lambda: self._lifecycle_ctrl.mark_prepare_stale(),
            on_set_detail_tab_dirty=lambda idx, lbl, dirty: self._config_ctrl.set_detail_tab_dirty(
                idx, lbl, dirty
            ),
            on_auto_fit_track_table=lambda: self._table_ctrl.auto_fit_track_table(),
            parent=self,
        )
        self._detail_tabs.addTab(self._groups_ctrl.page, "Groups")
        install_tab_marker(self._detail_tabs, _TAB_GROUPS)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, False)

        self._config_ctrl = SessionConfigController(
            self._state,
            self,
            detail_tabs=self._detail_tabs,
            on_refresh_daw_from_config=self._refresh_daw_processors_from_config,
            on_mark_prepare_stale=lambda: self._lifecycle_ctrl.mark_prepare_stale(),
            on_queue_repair_plan=lambda *a, **kw: self._lifecycle_ctrl.queue_repair_plan(*a, **kw),
            on_repair_settings_can_apply=lambda t, cfg: self._lifecycle_ctrl.repair_settings_can_apply(
                t, cfg
            ),
            on_is_repair_pending_or_running=lambda: bool(
                self._lifecycle_ctrl.repair_plan_worker or self._lifecycle_ctrl.repair_plan_pending
            ),
            on_reanalyze_phase1=lambda: self._lifecycle_ctrl.reanalyze_topology(),
            on_reanalyze_phase2=lambda: self._lifecycle_ctrl.analyze(),
            on_refresh_groups_dirty=self._groups_ctrl.refresh_dirty_indicators,
            on_refresh_processing_column=lambda: self._table_ctrl.refresh_processing_column(),
            on_refresh_analysis_column=lambda: self._table_ctrl.refresh_analysis_column(),
            on_render_summary=self._detail_ctrl.render_summary,
            on_refresh_file_report=self._detail_ctrl.refresh_file_report,
            on_refresh_waveform_issues=self._detail_ctrl.refresh_waveform_issues,
            on_set_waveform_colormap=self._wf_panel.waveform.set_colormap,
            parent=self,
        )
        self._detail_tabs.addTab(self._config_ctrl.page, "Config")
        install_tab_marker(self._detail_tabs, _TAB_SESSION)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, False)

        self._right_stack.addWidget(self._detail_tabs)
        self._right_stack.setCurrentIndex(_PAGE_TABS)
        return self._right_stack

    @Slot(int)
    def _on_detail_tab_changed(self, index: int):
        """Explicitly hide the File tab content when leaving it."""
        fs = getattr(self, "_file_splitter", None)
        if fs is not None:
            fs.setVisible(index == _TAB_FILE)

    def _make_report_browser(self):
        """Create a consistently styled QTextBrowser for reports."""
        from .detail.browser import ReportBrowser

        browser = ReportBrowser(self._detector_help)
        browser.reviewRequested.connect(lambda fn: self._detail_ctrl.review_summary_file(fn))
        browser.refresh_font(self.screen() or screen_for_startup())
        return browser

    def _refresh_report_browser_fonts(self, screen=None):
        """Re-apply report font sizing for the current monitor."""
        target_screen = screen or self.screen() or screen_for_startup()
        for attr in ("_summary_view", "_file_report"):
            browser = getattr(self, attr, None)
            if browser is not None:
                browser.refresh_font(target_screen)
        pane = getattr(self, "_file_report_pane", None)
        if pane is not None:
            pane.refresh_fonts(target_screen)

    def _install_report_screen_refresh(self):
        """Refresh report font scale when the window moves between screens."""
        handle = self.windowHandle()
        if handle is None:
            return
        if self._report_screen_connection is None:
            self._report_screen_connection = handle.screenChanged.connect(
                self._refresh_report_browser_fonts
            )
        self._refresh_report_browser_fonts(handle.screen())

    @Slot(int)
    def _on_phase_tab_changed(self, index: int):
        if index == _PHASE_TOPOLOGY and self._topology_ctrl.pending_sources:
            self._lifecycle_ctrl.check_reopened_layout_sources()
        if getattr(self, "_progress_running", False):
            self._show_progress_on_phase(index)
        if index == _PHASE_SETUP:
            self._daw_ctrl.page.schedule_splitter_fit()

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Wrap HTML content in a styled <body> tag."""
        return wrap_report_html(body)

    # ── Preferences & Tools ──────────────────────────────────────────────

    def _refresh_daw_processors_from_config(self) -> None:
        """Rebuild DAW processors and related Phase 3 UI from current config."""
        self._daw_ctrl.refresh_from_config()
        self._update_tools_menu()

    @Slot()
    def _on_preferences(self):
        old_scale = self._state.config.get("app", {}).get("scale_factor", DEFAULT_SCALE_FACTOR)
        old_config = self._state.effective_session_config()
        old_group_preset = self._state.config.get("group_presets", {}).get(
            self._state.active_session_preset
        )

        dlg = PreferencesDialog(self._state.config, parent=self)
        dlg.exec()
        if not dlg.saved:
            return

        self._state.config = dlg.result_config()
        save_config(self._state.config)
        if self._state.session is not None:
            name = dlg.preset_name_after_save(self._state.active_config_preset_name)
            self._state.active_config_preset_name = self._state.existing_config_preset_name(name)
        else:
            self._state.active_config_preset_name = self._state.default_config_preset_name()

        new_config = self._state.effective_session_config()
        impact = classify_structured_config_change(old_config, new_config)
        self._status_bar.showMessage("Preferences saved.")
        self._wf_panel.waveform.set_invert_scroll(
            self._state.config.get("app", {}).get("invert_scroll", "default")
        )

        self._groups_ctrl.populate_group_preset_combo()
        self._config_ctrl.populate_config_preset_combo()
        self._config_ctrl.load_session_widgets(new_config)
        self._config_ctrl.refresh_effective_config(impact)
        self._maybe_merge_updated_group_preset(old_group_preset)
        self._update_tools_menu()
        if self._pt_utils_window is not None:
            self._pt_utils_window.update_config(self._state.config)
        self._notify_preferences_impact(impact, old_scale)

    def _maybe_merge_updated_group_preset(self, old_group_preset: Any) -> None:
        if not self._state.session:
            return
        preset_name = self._state.active_session_preset
        presets = self._state.config.get(
            "group_presets", build_defaults().get("group_presets", {})
        )
        if preset_name not in presets:
            return
        preset_groups = presets[preset_name]
        if preset_groups == old_group_preset or preset_groups == self._state.session_groups:
            return
        ans = QMessageBox.question(
            self,
            "Update session groups?",
            f'The group preset \u201c{preset_name}\u201d has changed.\n\n'
            "Update the current session\u2019s groups to match?\n\n"
            "\u2022 Track assignments will be preserved where group names match.\n"
            "\u2022 Unmatched tracks will be set to (None).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if ans == QMessageBox.Yes:
            self._groups_ctrl.merge_groups_from_preset()
            self._status_bar.showMessage("Session groups updated from preset.")

    def _notify_preferences_impact(self, impact: ConfigChangeImpact, old_scale: Any) -> None:
        if not (impact.presentation_keys and self._state.session):
            self._config_ctrl.refresh_config_changed_display()
        if self._state.source_dir and (impact.requires_phase1 or impact.requires_phase2):
            phase = "Track Layout" if impact.requires_phase1 else "Phase 2 analysis"
            repair_active = bool(
                self._lifecycle_ctrl.repair_plan_worker or self._lifecycle_ctrl.repair_plan_pending
            )
            message = (
                f"Repair checks update automatically. Rerun {phase} to refresh the full session."
                if repair_active
                else f"Rerun {phase} to update results for the current session."
            )
            QMessageBox.information(
                self,
                "Config preset updated",
                f"Inherited settings have been updated from Preferences. {message}",
            )
        elif impact.daw_keys:
            self._status_bar.showMessage("Preferences saved. Phase 3 settings updated.")

        new_scale = self._state.config.get("app", {}).get("scale_factor", DEFAULT_SCALE_FACTOR)
        if new_scale != old_scale:
            QMessageBox.information(
                self,
                "Restart required",
                f"HiDPI scale factor changed from {old_scale} to {new_scale}.\n"
                "Please restart SessionPrep for the new scaling to take effect.",
            )

    def _update_tools_menu(self):
        """Enable/disable Tools menu entries based on active config."""
        preset = self._state.active_preset()
        pt_section = preset.get("daw_processors", {}).get("protools", {})
        self._pt_utils_action.setEnabled(pt_section.get("protools_enabled", False))

    @Slot()
    def _on_open_log_viewer(self):
        """Open (or activate) the detached log viewer window."""
        from .log_viewer import LogViewerWindow

        if self._log_viewer_window is None:
            self._log_viewer_window = LogViewerWindow()
        self._log_viewer_window.show()
        self._log_viewer_window.raise_()
        self._log_viewer_window.activateWindow()

    @Slot()
    def _on_open_pt_utils(self):
        """Open (or activate) the Pro Tools Utils window."""
        from .daw_tools.protools.window import ProToolsUtilsWindow

        if self._pt_utils_window is None:
            self._pt_utils_window = ProToolsUtilsWindow(self._state.config)
        else:
            self._pt_utils_window.update_config(self._state.config)
        self._pt_utils_window.prepare_for_show()
        self._pt_utils_window.show()
        self._pt_utils_window.raise_()
        self._pt_utils_window.activateWindow()

    @Slot()
    def _on_about(self):
        from sessionpreplib import __version__ as ver

        QMessageBox.about(
            self,
            "About SessionPrep",
            f"<h2>SessionPrep</h2>"
            f"<p>Version {ver}</p>"
            f"<p>Batch audio analyzer and normalizer<br/>"
            f"for mix session preparation.</p>",
        )

    @Slot()
    def _resume_close_after_daw(self):
        if self._close_after_daw:
            self.close()

    def closeEvent(self, event):
        if self._batch_dock.has_items and not (self._close_after_batch or self._close_after_daw):
            reply = QMessageBox.warning(
                self,
                "Pending Batch Items",
                "You have pending sessions in the batch queue. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        if self._batch_manager.is_running:
            self._close_after_batch = True
            self._batch_manager.stop_after_current()
            event.ignore()
            return
        self._daw_ctrl.pending_after_check = None
        if self._daw_ctrl.has_running_workers():
            self._close_after_daw = True
            event.ignore()
            return

        self._close_after_daw = self._close_after_batch = False
        self._playback.close()
        self._repair_preview.close()

        self._lifecycle_ctrl.dispose()
        self._topology_ctrl.dispose()
        self._daw_ctrl.dispose()
        self._detail_ctrl.dispose()
        from PySide6.QtCore import QThread

        for worker in self.findChildren(QThread):
            if worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                else:
                    worker.requestInterruption()
                worker.wait()

        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _app_icon() -> QIcon:
    """Load the application icon from the res/ directory."""
    res_dir = os.path.join(os.path.dirname(__file__), "res")
    icon = QIcon()
    svg = os.path.join(res_dir, "sessionprep.svg")
    png = os.path.join(res_dir, "sessionprep.png")
    if os.path.isfile(svg):
        icon = QIcon(svg)
    if os.path.isfile(png):
        icon.addFile(png)
    return icon


def _apply_startup_scale_env() -> None:
    """Apply configured HiDPI scale factor before QApplication is constructed."""
    from .settings import (
        DEFAULT_SCALE_FACTOR as _default_scale,
        config_path as _cfg_path,
        startup_scale_factor_from_raw_config as _startup_scale,
    )

    scale = _default_scale
    try:
        with open(_cfg_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
        scale = _startup_scale(raw)
    except Exception:
        pass
    if scale is not None and float(scale) != 1.0:
        os.environ["QT_SCALE_FACTOR"] = str(float(scale))


def _schedule_smoke_check(app: QApplication, window: SessionPrepWindow, smoke_files: Any) -> None:
    def check_window() -> None:
        try:
            if not window.isVisible() or app.windowIcon().pixmap(32, 32).isNull():
                raise RuntimeError("Qt main window or packaged icon did not load")
            from hashlib import sha256
            from sessionpreplib import __version__
            from sessionpreplib.audio import load_track

            audio_file, receipt_file = smoke_files
            track = load_track(str(audio_file))
            with receipt_file.open("x", encoding="utf-8") as receipt:
                json.dump(
                    dict(
                        version=__version__,
                        rate=track.samplerate,
                        frames=track.total_samples,
                        pcm_sha256=sha256(
                            track.audio_data.astype("<f8", copy=False)
                        ).hexdigest(),
                    ),
                    receipt,
                )
        except Exception:
            log.exception("GUI smoke test failed")
            app.exit(1)
        else:
            window.close()

    QTimer.singleShot(0, check_window)


def main(*, smoke_files=None):
    from sessionpreplib.logging_setup import setup_logging

    setup_logging()
    t_main = time.perf_counter()
    _apply_startup_scale_env()

    t0 = time.perf_counter()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(_app_icon())
    dbg(f"QApplication created: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    window = SessionPrepWindow()
    dbg(f"SessionPrepWindow created: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    window.show()
    window._install_report_screen_refresh()
    dbg(f"window.show: {(time.perf_counter() - t0) * 1000:.1f} ms")
    log.debug(
        "Startup geometry after show: actual=%s frame=%s window_state=%s",
        _rect_text(window.geometry()),
        _rect_text(window.frameGeometry()),
        _window_state_text(window.windowState()),
    )
    QTimer.singleShot(
        0,
        lambda: log.debug(
            "Startup geometry after event loop starts: actual=%s frame=%s window_state=%s",
            _rect_text(window.geometry()),
            _rect_text(window.frameGeometry()),
            _window_state_text(window.windowState()),
        ),
    )

    if smoke_files is not None:
        _schedule_smoke_check(app, window, smoke_files)

    dbg(f"main() total: {(time.perf_counter() - t_main) * 1000:.1f} ms")
    sys.exit(app.exec())
