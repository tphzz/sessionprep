"""Main application window for SessionPrep GUI."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
import os
import sys
import time
from typing import Any

from PySide6.QtCore import QSize, QThread, QTimer, Qt, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from sessionpreplib.config import ConfigChangeImpact, classify_structured_config_change
from sessionpreplib.detectors import detector_help_map

from .analysis import (
    AnalysisPrepareController,
    SessionConfigController,
    SessionPersistenceController,
)
from .batch import BatchQueueController
from .daw import DawTransferController
from .detail import PlaybackController, TrackDetailController
from .detail.repair_preview import RepairPreview
from .detail.report_style import wrap_report_html
from .log import dbg
from .prefs import PreferencesDialog
from .progress import ProgressCoordinator
from .settings import DEFAULT_SCALE_FACTOR, build_defaults, load_config, save_config
from .state import WorkspaceState
from .status_prompt import StatusBarPromptCoordinator
from .theme import apply_dark_theme
from .topology import TopologyController
from .tracks import (
    GroupsController,
    Phase2Page,
    TrackTableController,
    _PHASE_ANALYSIS,
    _PHASE_SETUP,
    _PHASE_TOPOLOGY,
    _TAB_SESSION,
)
from .window_geometry import (
    FALLBACK_STARTUP_SIZE,
    calculate_startup_geometry,
    screen_for_startup,
)

log = logging.getLogger(__name__)

MAIN_WINDOW_MIN_HEIGHT_HINT = 360


def _size_text(size: QSize) -> str:
    return f"{size.width()}x{size.height()}"


def _rect_text(rect: Any) -> str:
    return f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"


def _window_state_text(state: Any) -> str:
    value = getattr(state, "value", None)
    if value is None:
        return str(state)
    return f"{state} ({value})"


class _CurrentPageTabWidget(QTabWidget):
    """Tab widget whose minimum hints are based on the active page."""

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self) -> None:
        t_init = time.perf_counter()
        super().__init__()
        self.setWindowTitle("SessionPrep")
        self.setWindowIcon(_app_icon())

        t0 = time.perf_counter()
        loaded_config = load_config()
        dbg(f"load_config: {(time.perf_counter() - t0) * 1000:.1f} ms")
        self._state = WorkspaceState(self, config=loaded_config)
        self._state.recursive_scan = loaded_config.get("app", {}).get("recursive_scan", False)

        self._pt_utils_window: Any = None
        self._log_viewer_window: Any = None
        self._report_screen_connection: Any = None
        self._close_after_daw = False

        t0 = time.perf_counter()
        self._detector_help = detector_help_map()
        dbg(f"detector_help_map: {(time.perf_counter() - t0) * 1000:.1f} ms")

        t0 = time.perf_counter()
        self._playback = PlaybackController(self)
        dbg(f"PlaybackController (sounddevice): {(time.perf_counter() - t0) * 1000:.1f} ms")

        t0 = time.perf_counter()
        self._init_ui()
        dbg(f"_init_ui: {(time.perf_counter() - t0) * 1000:.1f} ms")

        t0 = time.perf_counter()
        apply_dark_theme(self)
        dbg(f"apply_dark_theme: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._apply_initial_window_geometry()
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.activated.connect(self._detail_ctrl.toggle_play)

        dbg(f"SessionPrepWindow.__init__ total: {(time.perf_counter() - t_init) * 1000:.1f} ms")

    # ── Public controller, page & state accessors ────────────────────────

    @property
    def state(self) -> WorkspaceState:
        """Shared cross-phase desktop workspace state."""
        return self._state

    @property
    def progress_coordinator(self) -> ProgressCoordinator:
        """Per-phase global progress bar coordinator."""
        return self._progress

    @property
    def status_prompt(self) -> StatusBarPromptCoordinator:
        """Status bar inline prompt coordinator."""
        return self._status_prompt

    @property
    def phase2_page(self) -> Phase2Page:
        """Phase 2 (Analysis & Preparation) page container."""
        return self._phase2_page

    @property
    def batch_controller(self) -> BatchQueueController:
        """Batch queue dock, manager, and persistence controller."""
        return self._batch_ctrl

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
    def analysis_controller(self) -> AnalysisPrepareController:
        """Phase 1/2 analysis, repair-planning, and Prepare controller."""
        return self._analysis_ctrl

    @property
    def persistence_controller(self) -> SessionPersistenceController:
        """Folder open, .spsession save/load, workspace clear, and peak cache controller."""
        return self._persistence_ctrl

    @property
    def repair_preview(self) -> RepairPreview:
        """File-tab repair audition preview coordinator."""
        return self._repair_preview

    # ── Startup geometry ─────────────────────────────────────────────────

    def _apply_initial_window_geometry(self) -> None:
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

    def _log_startup_size_hints(self) -> None:
        """Log the largest propagated size hints for startup diagnosis."""
        phase_info = (
            f" phase_current={self._phase_tabs.currentIndex()}"
            f" phase_min_hint={_size_text(self._phase_tabs.minimumSizeHint())}"
            f" phase_size_hint={_size_text(self._phase_tabs.sizeHint())}"
        )
        central = self.centralWidget()
        log.debug(
            "Startup geometry hints: window_min_hint=%s window_size_hint=%s "
            "central_min_hint=%s central_size_hint=%s%s",
            _size_text(self.minimumSizeHint()),
            _size_text(self.sizeHint()),
            _size_text(central.minimumSizeHint()) if central else "none",
            _size_text(central.sizeHint()) if central else "none",
            phase_info,
        )
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
        self._log_large_hint_contributors()

    def _log_large_hint_contributors(self) -> None:
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
        for _, cls_name, object_name, hint_str, size_hint_str, min_size_str, visible in contributors[:12]:
            log.debug(
                "Startup geometry hint contributor: class=%s object=%s "
                "visible=%s min_hint=%s size_hint=%s min_size=%s",
                cls_name,
                object_name,
                visible,
                hint_str,
                size_hint_str,
                min_size_str,
            )

    # ── MainWindowShell protocol methods ─────────────────────────────────

    def show_status(self, message: str) -> None:
        self._status_prompt.show_status(message)

    def show_status_prompt(
        self,
        prefix: str,
        highlight: str,
        suffix: str,
        *,
        tooltip: str,
        on_accept: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        accept_label: str = "Overwrite",
        cancel_label: str = "Cancel",
        accept_description: str = "",
        cancel_description: str = "",
        return_focus_widget: QWidget | None = None,
    ) -> None:
        self._status_prompt.show_prompt(
            prefix,
            highlight,
            suffix,
            tooltip=tooltip,
            on_accept=on_accept,
            on_cancel=on_cancel,
            accept_label=accept_label,
            cancel_label=cancel_label,
            accept_description=accept_description,
            cancel_description=cancel_description,
            return_focus_widget=return_focus_widget,
        )

    def dismiss_status_prompt(
        self, *, status_message: str | None = None, restore_focus: bool = False
    ) -> None:
        self._status_prompt.dismiss(
            status_message=status_message,
            restore_focus=restore_focus,
        )

    def progress_start(self, text: str, *, owner: Any = None) -> None:
        self._progress.start(text, owner=owner)

    def progress_message(self, text: str) -> None:
        self._progress.message(text)

    def progress_value(self, current: int, total: int) -> None:
        self._progress.value(current, total)

    def progress_finish(self, text: str) -> None:
        self._progress.finish(text)

    def progress_fail(self, text: str) -> None:
        self._progress.fail(text)

    def clear_progress(self) -> None:
        self._progress.clear()

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None:
        self._phase_tabs.setTabEnabled(phase_index, enabled)

    def switch_to_phase(self, phase_index: int) -> None:
        self._phase_tabs.setCurrentIndex(phase_index)

    def current_phase(self) -> int:
        return self._phase_tabs.currentIndex()

    def is_phase_enabled(self, phase_index: int) -> bool:
        return self._phase_tabs.isTabEnabled(phase_index)

    def register_phase_progress_layout(self, phase_index: int, layout: QVBoxLayout) -> None:
        self._progress.register_phase_layout(phase_index, layout)

    def dialog_parent(self) -> QWidget:
        return self

    def is_batch_mode(self) -> bool:
        return bool(getattr(self, "_batch_ctrl", None) and self._batch_ctrl.is_batch_mode)

    def is_closing_after_daw(self) -> bool:
        return bool(self._close_after_daw)

    def resume_close_after_daw(self) -> None:
        self._resume_close_after_daw()

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    # ── UI setup & controller composition ────────────────────────────────

    def _init_ui(self) -> None:
        self._init_status_bar()
        self._init_menus()

        self._phase_tabs = _CurrentPageTabWidget()
        self._phase_tabs.setObjectName("phaseTabs")
        self._phase_tabs.setDocumentMode(True)
        self._progress = ProgressCoordinator(self._phase_tabs.currentIndex, parent=self)

        self._create_controllers()
        self._wire_controllers()

        self._phase_tabs.addTab(self._topology_ctrl.page, "Phase 1: Track Layout")
        self._phase_tabs.addTab(self._phase2_page, "Phase 2: Analysis && Preparation")
        self._phase_tabs.setTabEnabled(_PHASE_ANALYSIS, False)
        self._phase_tabs.addTab(self._daw_ctrl.page, "Phase 3: DAW Transfer")
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, False)
        self._active_phase: int = self._phase_tabs.currentIndex()
        self._phase_tabs.currentChanged.connect(self._on_phase_tab_changed)

        self.setCentralWidget(self._phase_tabs)
        self.show_status("Open a directory containing .wav / .aif files to begin.")

    def _init_status_bar(self) -> None:
        self._status_bar = QStatusBar(self)
        self._status_prompt = StatusBarPromptCoordinator(self._status_bar)
        self.setStatusBar(self._status_bar)

    def _create_controllers(self) -> None:
        self._create_phase_controllers()
        self._create_lifecycle_controllers()

    def _create_phase_controllers(self) -> None:
        self._topology_ctrl = TopologyController(self._state, self, self._playback, parent=self)

        invert_scroll = self._state.config.get("app", {}).get("invert_scroll", "default")
        self._phase2_page = Phase2Page(self._detector_help, invert_scroll=invert_scroll, parent=self)
        self.register_phase_progress_layout(_PHASE_ANALYSIS, self._phase2_page.main_layout)

        self._groups_ctrl = GroupsController(
            self._state,
            self,
            track_table=self._phase2_page.track_table,
            detail_tabs=self._phase2_page.detail_tabs,
            on_find_table_row=lambda fn: self._table_ctrl.find_table_row(fn),
            parent=self,
        )
        self._phase2_page.add_groups_tab(self._groups_ctrl.page)

        self._config_ctrl = SessionConfigController(
            self._state,
            self,
            detail_tabs=self._phase2_page.detail_tabs,
            on_is_repair_pending_or_running=lambda: self._analysis_ctrl.is_repair_pending_or_running(),
            parent=self,
        )
        self._phase2_page.add_config_tab(self._config_ctrl.page)

        self._repair_preview = RepairPreview(
            self._state,
            self._phase2_page.wf_panel,
            self._phase2_page.file_report,
            self._playback,
            get_detail_ctrl=lambda: self._detail_ctrl,
            is_track_repair_pending=self._is_track_repair_pending,
            parent=self,
        )
        self._detail_ctrl = TrackDetailController(
            self._state,
            self,
            summary_view=self._phase2_page.summary_view,
            file_report=self._phase2_page.file_report,
            file_review=self._phase2_page.file_review,
            repair_editor=self._phase2_page.repair_editor,
            detail_tabs=self._phase2_page.detail_tabs,
            track_table=self._phase2_page.track_table,
            wf_panel=self._phase2_page.wf_panel,
            playback=self._playback,
            repair_preview=self._repair_preview,
            topology_ctrl=self._topology_ctrl,
            on_find_table_row=lambda fn: self._table_ctrl.find_table_row(fn),
            parent=self,
        )

    def _create_lifecycle_controllers(self) -> None:
        self._batch_ctrl = BatchQueueController(
            self._state,
            self,
            self,
            on_restore_session=lambda data: self._persistence_ctrl.restore_session_state(data),
            parent=self,
        )
        self._daw_ctrl = DawTransferController(
            self._state,
            self,
            on_capture_session_state=lambda: self._persistence_ctrl.capture_session_state(),
            on_enqueue_batch_item=self._batch_ctrl.enqueue_item,
            on_is_prepare_running=lambda: (
                getattr(self, "_analysis_ctrl", None) is not None
                and self._analysis_ctrl.is_prepare_running()
            ),
            parent=self,
        )
        self._table_ctrl = TrackTableController(
            self._state,
            self,
            track_table=self._phase2_page.track_table,
            setup_table=self._daw_ctrl.page.setup_table,
            main_splitter=self._phase2_page.main_splitter,
            right_stack=self._phase2_page.right_stack,
            detail_tabs=self._phase2_page.detail_tabs,
            on_create_group_combo=self._groups_ctrl.create_group_combo,
            on_is_analysis_running=lambda: self._analysis_ctrl.is_analysis_running(),
            on_is_repair_running=lambda: self._analysis_ctrl.is_repair_running(),
            parent=self,
        )
        self._analysis_ctrl = AnalysisPrepareController(
            self._state,
            self,
            phase2_page=self._phase2_page,
            topology_ctrl=self._topology_ctrl,
            detail_ctrl=self._detail_ctrl,
            table_ctrl=self._table_ctrl,
            groups_ctrl=self._groups_ctrl,
            config_ctrl=self._config_ctrl,
            repair_preview=self._repair_preview,
            get_project_name=lambda: self._daw_ctrl.page.project_name_edit.text().strip(),
            parent=self,
        )
        self._persistence_ctrl = SessionPersistenceController(
            self._state,
            self,
            phase2_page=self._phase2_page,
            topology_ctrl=self._topology_ctrl,
            daw_ctrl=self._daw_ctrl,
            detail_ctrl=self._detail_ctrl,
            table_ctrl=self._table_ctrl,
            groups_ctrl=self._groups_ctrl,
            config_ctrl=self._config_ctrl,
            save_session_action=self._save_session_action,
            on_progress_owner=lambda: self._progress.owner,
            on_is_save_blocked=self._analysis_ctrl.is_save_blocked,
            parent=self,
        )

    def _wire_controllers(self) -> None:
        self._wire_menus_and_playback()
        self._wire_phase1_and_phase2_pages()
        self._wire_controller_signals()

    def _wire_menus_and_playback(self) -> None:
        self._open_action.triggered.connect(self._persistence_ctrl.open_directory_dialog)
        self._load_session_action.triggered.connect(self._persistence_ctrl.load_session_dialog)
        self._load_batch_action.triggered.connect(self._batch_ctrl.load_queue)
        self._save_session_action.triggered.connect(self._persistence_ctrl.save_session_dialog)
        self._save_batch_action.triggered.connect(self._batch_ctrl.save_queue)
        self._batch_ctrl.bind_batch_mode_action(self._batch_mode_action)

        self._playback.cursor_updated.connect(self._detail_ctrl.on_cursor_updated)
        self._playback.playback_finished.connect(self._detail_ctrl.on_playback_finished)
        self._playback.error.connect(self._detail_ctrl.on_playback_error)
        self._state.config_changed.connect(self._update_tools_menu)

    def _wire_phase1_and_phase2_pages(self) -> None:
        self._topology_ctrl.open_folder_requested.connect(self._persistence_ctrl.open_directory_dialog)
        self._topology_ctrl.reanalyze_requested.connect(self._persistence_ctrl.reanalyze_topology)
        self._topology_ctrl.topology_edited.connect(self._analysis_ctrl.mark_prepare_stale)
        self._topology_ctrl.layout_applied.connect(lambda _out_dir: self._analysis_ctrl.analyze())
        self._topology_ctrl.peak_prioritize_requested.connect(self._persistence_ctrl.prioritize_peak)

        p2 = self._phase2_page
        p2.analyze_action.triggered.connect(self._analysis_ctrl.analyze)
        p2.auto_group_action.triggered.connect(self._groups_ctrl.auto_group)
        p2.prepare_action.triggered.connect(self._analysis_ctrl.prepare)
        p2.track_table.cellClicked.connect(self._table_ctrl.on_row_clicked)
        p2.track_table.currentCellChanged.connect(self._table_ctrl.on_current_cell_changed)
        p2.summary_view.reviewRequested.connect(self._detail_ctrl.review_summary_file)
        p2.file_report.reviewRequested.connect(self._detail_ctrl.review_summary_file)
        p2.file_report.repairRequested.connect(
            lambda key: self._detail_ctrl.open_track_repairs(self._state.current_track, key)
        )
        p2.file_report.visualizeRequested.connect(self._detail_ctrl.visualize_detector)
        p2.file_report.sourceLocationsRequested.connect(self._detail_ctrl.show_source_locations)
        p2.file_report.verificationRetryRequested.connect(self._detail_ctrl.retry_detector_verification)
        p2.file_report.reportChanged.connect(self._detail_ctrl.on_file_report_context_changed)
        p2.repair_editor.choiceChanged.connect(self._analysis_ctrl.set_track_repair_choice)
        p2.repair_editor.adjustGainRequested.connect(self._analysis_ctrl.adjust_failed_output_gain)
        p2.file_review.processingFocusRequested.connect(self._detail_ctrl.focus_processing_menu)

        wf = p2.wf_panel
        wf.play_clicked.connect(self._detail_ctrl.play)
        wf.stop_clicked.connect(self._detail_ctrl.stop)
        wf.position_clicked.connect(self._detail_ctrl.on_waveform_seek)
        wf.display_mode_changed.connect(self._detail_ctrl.on_display_mode_changed)
        wf.fft_group.triggered.connect(self._detail_ctrl.on_spec_fft_changed)
        wf.win_group.triggered.connect(self._detail_ctrl.on_spec_window_changed)
        wf.cmap_group.triggered.connect(self._detail_ctrl.on_spec_cmap_changed)
        wf.floor_group.triggered.connect(self._detail_ctrl.on_spec_floor_changed)
        wf.ceil_group.triggered.connect(self._detail_ctrl.on_spec_ceil_changed)

    def _wire_controller_signals(self) -> None:
        self._wire_table_and_group_signals()
        self._wire_config_and_detail_signals()
        self._wire_lifecycle_and_daw_signals()

    def _wire_table_and_group_signals(self) -> None:
        self._table_ctrl.track_selected.connect(self._detail_ctrl.show_track_detail)
        self._table_ctrl.file_report_refresh_requested.connect(self._detail_ctrl.refresh_file_report)
        self._table_ctrl.waveform_issues_refresh_requested.connect(self._detail_ctrl.refresh_waveform_issues)
        self._table_ctrl.track_repairs_requested.connect(
            lambda track: self._detail_ctrl.open_track_repairs(track)
        )
        self._table_ctrl.linked_group_levels_requested.connect(self._groups_ctrl.apply_linked_group_levels)
        self._table_ctrl.setup_column_modes_requested.connect(self._daw_ctrl.page.apply_table_column_modes)
        self._table_ctrl.setup_splitter_fit_requested.connect(self._daw_ctrl.page.schedule_splitter_fit)
        self._table_ctrl.setup_table_populating_changed.connect(self._daw_ctrl.set_setup_table_populating)
        self._table_ctrl.prepare_stale_requested.connect(self._analysis_ctrl.mark_prepare_stale)
        self._table_ctrl.prepare_stale_requested.connect(self._daw_ctrl.mark_signal_flow_stale)
        self._table_ctrl.repair_plan_requested.connect(
            lambda tracks, anchors: self._analysis_ctrl.queue_repair_plan(tracks, refresh_anchors=anchors)
        )
        self._table_ctrl.analyze_enabled_requested.connect(self._phase2_page.analyze_action.setEnabled)

        self._groups_ctrl.setup_table_refresh_requested.connect(self._table_ctrl.populate_setup_table)
        self._groups_ctrl.file_tab_refresh_requested.connect(self._table_ctrl.refresh_file_tab)
        self._groups_ctrl.prepare_stale_requested.connect(self._analysis_ctrl.mark_prepare_stale)
        self._groups_ctrl.prepare_stale_requested.connect(self._daw_ctrl.mark_signal_flow_stale)
        self._groups_ctrl.detail_tab_dirty_changed.connect(self._config_ctrl.set_detail_tab_dirty)
        self._groups_ctrl.auto_fit_requested.connect(self._table_ctrl.auto_fit_track_table)

    def _wire_config_and_detail_signals(self) -> None:
        self._config_ctrl.daw_refresh_requested.connect(self._refresh_daw_processors_from_config)
        self._config_ctrl.prepare_stale_requested.connect(self._analysis_ctrl.mark_prepare_stale)
        self._config_ctrl.prepare_stale_requested.connect(self._daw_ctrl.mark_signal_flow_stale)
        self._config_ctrl.config_impact_applied.connect(lambda _imp: self._daw_ctrl.mark_signal_flow_stale())
        self._config_ctrl.repair_plan_requested.connect(self._analysis_ctrl.queue_repair_plan)
        self._config_ctrl.reanalyze_phase1_requested.connect(self._persistence_ctrl.reanalyze_topology)
        self._config_ctrl.reanalyze_phase2_requested.connect(self._analysis_ctrl.analyze)
        self._config_ctrl.groups_dirty_refresh_requested.connect(self._groups_ctrl.refresh_dirty_indicators)
        self._config_ctrl.processing_column_refresh_requested.connect(self._table_ctrl.refresh_processing_column)
        self._config_ctrl.analysis_column_refresh_requested.connect(self._table_ctrl.refresh_analysis_column)
        self._config_ctrl.summary_render_requested.connect(self._detail_ctrl.render_summary)
        self._config_ctrl.file_report_refresh_requested.connect(self._detail_ctrl.refresh_file_report)
        self._config_ctrl.waveform_issues_refresh_requested.connect(self._detail_ctrl.refresh_waveform_issues)
        self._config_ctrl.waveform_colormap_changed.connect(self._phase2_page.wf_panel.waveform.set_colormap)

        self._detail_ctrl.repair_plan_requested.connect(
            lambda tracks, reval: self._analysis_ctrl.queue_repair_plan(tracks, revalidate_source=reval)
        )
        self._detail_ctrl.peak_prioritize_requested.connect(self._persistence_ctrl.prioritize_peak)
        self._repair_preview.revalidate_repair_requested.connect(
            lambda track: self._analysis_ctrl.queue_repair_plan([track], revalidate_source=True)
        )
        self._repair_preview.invalid_source_detected.connect(self._on_repair_preview_invalid_source)

    def _wire_lifecycle_and_daw_signals(self) -> None:
        self._daw_ctrl.setup_table_refresh_requested.connect(self._table_ctrl.populate_setup_table)
        self._daw_ctrl.clear_workspace_requested.connect(self._persistence_ctrl.clear_workspace)
        self._daw_ctrl.edit_protools_config_requested.connect(self._open_protools_session_config)
        self._daw_ctrl.open_log_viewer_requested.connect(self._on_open_log_viewer)
        self._batch_ctrl.batch_mode_changed.connect(lambda _c: self._daw_ctrl.update_lifecycle_buttons())
        self._batch_ctrl.batch_mode_changed.connect(
            lambda _c: self._topology_ctrl.dismiss_overwrite_prompt()
        )
        self._batch_ctrl.clear_workspace_requested.connect(self._persistence_ctrl.clear_workspace)
        self._batch_ctrl.close_ready.connect(self.close)

        self._persistence_ctrl.workspace_clearing.connect(self._detail_ctrl.stop)
        self._persistence_ctrl.workspace_clearing.connect(self._topology_ctrl.clear_workspace)
        self._persistence_ctrl.workspace_clearing.connect(self._analysis_ctrl.cancel_all_workers)
        self._persistence_ctrl.workspace_cleared.connect(lambda _k: self._daw_ctrl.clear_workspace())
        self._persistence_ctrl.workspace_cleared.connect(lambda _k: self._detail_ctrl.clear_workspace())
        self._persistence_ctrl.workspace_cleared.connect(lambda _k: self._groups_ctrl.clear_workspace())
        self._persistence_ctrl.phase1_analysis_requested.connect(self._analysis_ctrl.run_phase1_analysis)
        self._persistence_ctrl.peak_file_ready.connect(self._topology_ctrl.on_peak_file_done)
        self._persistence_ctrl.peak_file_ready.connect(self._detail_ctrl.on_peak_file_done)
        self._persistence_ctrl.repair_plan_requested.connect(
            lambda tracks, vo: self._analysis_ctrl.queue_repair_plan(tracks, verification_only=vo)
        )
        self._persistence_ctrl.prepare_button_refresh_requested.connect(
            self._analysis_ctrl.update_prepare_button
        )
        self._persistence_ctrl.tools_menu_refresh_requested.connect(self._update_tools_menu)

        self._analysis_ctrl.save_enabled_changed.connect(self._save_session_action.setEnabled)
        self._analysis_ctrl.peak_build_requested.connect(self._persistence_ctrl.on_peak_build_requested)
        self._analysis_ctrl.prepare_state_changed.connect(self._daw_ctrl.update_lifecycle_buttons)
        self._analysis_ctrl.prepare_state_changed.connect(self._daw_ctrl.update_use_processed_action)
        self._analysis_ctrl.phase1_completed.connect(lambda _s: self._daw_ctrl.sync_protools_ui_mode())
        self._analysis_ctrl.analysis_completed.connect(self._on_analysis_completed_for_daw)
        self._analysis_ctrl.prepare_completed.connect(lambda _s: self._table_ctrl.populate_setup_table())
        self._analysis_ctrl.prepare_completed.connect(lambda _s: self._daw_ctrl.mark_signal_flow_stale())

    @Slot(object, object)
    def _on_analysis_completed_for_daw(self, _session: Any, _summary: Any) -> None:
        self._daw_ctrl.sync_protools_ui_mode()
        self._daw_ctrl.mark_signal_flow_stale()

    def _is_track_repair_pending(self, filename: str) -> bool:
        plan = self._analysis_ctrl.repair_plan_worker
        return filename in self._analysis_ctrl.repair_plan_pending or bool(
            plan and any(t.filename == filename for t in plan.tracks)
        )

    @Slot(object)
    def _on_repair_preview_invalid_source(self, track: Any) -> None:
        self._analysis_ctrl.mark_prepare_stale()
        self._table_ctrl.refresh_file_tab(track)
        self._detail_ctrl.refresh_track_repair_editor(track)
        self._table_ctrl.update_track_row(track.filename)
        self._detail_ctrl.refresh_verified_summary()
        self._analysis_ctrl.update_prepare_button()

    # ── Menus ────────────────────────────────────────────────────────────

    def _init_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self._open_action = QAction("Open &Folder...", self)
        self._open_action.setShortcut("Ctrl+O")
        file_menu.addAction(self._open_action)

        self._load_session_action = QAction("&Load Session...", self)
        self._load_session_action.setShortcut("Ctrl+Shift+O")
        file_menu.addAction(self._load_session_action)

        self._load_batch_action = QAction("Load Batch Queue...", self)
        file_menu.addAction(self._load_batch_action)

        file_menu.addSeparator()

        self._save_session_action = QAction("&Save Session...", self)
        self._save_session_action.setShortcut("Ctrl+S")
        self._save_session_action.setEnabled(False)
        file_menu.addAction(self._save_session_action)

        self._save_batch_action = QAction("Save Batch Queue...", self)
        file_menu.addAction(self._save_batch_action)

        file_menu.addSeparator()

        self._batch_mode_action = QAction("Batch Processing Mode", self)
        self._batch_mode_action.setCheckable(True)
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

    # ── Screen fonts, phase tab changes & HTML helper ────────────────────

    def _refresh_report_browser_fonts(self, screen: Any = None) -> None:
        self._phase2_page.refresh_report_fonts(screen or self.screen() or screen_for_startup())

    def _install_report_screen_refresh(self) -> None:
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
    def _on_phase_tab_changed(self, index: int) -> None:
        prev_phase = self._active_phase
        self._active_phase = index
        if prev_phase == _PHASE_TOPOLOGY and index != _PHASE_TOPOLOGY:
            self._topology_ctrl.stop_preview()
        elif prev_phase == _PHASE_ANALYSIS and index != _PHASE_ANALYSIS:
            self._detail_ctrl.stop()
        if index != _PHASE_TOPOLOGY:
            self._topology_ctrl.dismiss_overwrite_prompt()
        if hasattr(self, "_space_shortcut"):
            self._space_shortcut.setEnabled(index != _PHASE_SETUP)
        if index == _PHASE_TOPOLOGY and self._topology_ctrl.pending_sources:
            self._analysis_ctrl.check_reopened_layout_sources()
        self._progress.on_phase_changed(index)
        if index == _PHASE_SETUP:
            self._daw_ctrl.page.schedule_splitter_fit()

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Wrap HTML content in a styled <body> tag."""
        return wrap_report_html(body)

    # ── Preferences & Tools ──────────────────────────────────────────────

    @Slot()
    def _refresh_daw_processors_from_config(self) -> None:
        """Rebuild DAW processors and related Phase 3 UI from current config."""
        self._daw_ctrl.refresh_from_config()
        self._update_tools_menu()

    @Slot()
    def _open_protools_session_config(self) -> None:
        """Jump from Phase 3's read-only summary bar to Phase 2 › Config › Pro Tools."""
        if self._state.session is None:
            return
        self._phase_tabs.setCurrentIndex(_PHASE_ANALYSIS)
        self._phase2_page.detail_tabs.setCurrentIndex(_TAB_SESSION)
        self._config_ctrl.select_daw_processor_page("Pro Tools")

    @Slot()
    def _on_preferences(self) -> None:
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
        self._topology_ctrl.dismiss_overwrite_prompt()
        self.show_status("Preferences saved.")
        self._phase2_page.wf_panel.waveform.set_invert_scroll(
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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._groups_ctrl.merge_groups_from_preset()
            self.show_status("Session groups updated from preset.")

    def _notify_preferences_impact(self, impact: ConfigChangeImpact, old_scale: Any) -> None:
        if not (impact.presentation_keys and self._state.session):
            self._config_ctrl.refresh_config_changed_display()
        if self._state.source_dir and (impact.requires_phase1 or impact.requires_phase2):
            phase = "Track Layout" if impact.requires_phase1 else "Phase 2 analysis"
            repair_active = self._analysis_ctrl.is_repair_pending_or_running()
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
            self.show_status("Preferences saved. Phase 3 settings updated.")

        new_scale = self._state.config.get("app", {}).get("scale_factor", DEFAULT_SCALE_FACTOR)
        if new_scale != old_scale:
            QMessageBox.information(
                self,
                "Restart required",
                f"HiDPI scale factor changed from {old_scale} to {new_scale}.\n"
                "Please restart SessionPrep for the new scaling to take effect.",
            )

    @Slot()
    def _update_tools_menu(self) -> None:
        """Enable/disable Tools menu entries based on active config."""
        preset = self._state.active_preset()
        pt_section = preset.get("daw_processors", {}).get("protools", {})
        self._pt_utils_action.setEnabled(pt_section.get("protools_enabled", False))

    @Slot()
    def _on_open_log_viewer(self) -> None:
        """Open (or activate) the detached log viewer window."""
        from .log_viewer import LogViewerWindow

        if self._log_viewer_window is None:
            self._log_viewer_window = LogViewerWindow()
        self._log_viewer_window.show()
        self._log_viewer_window.raise_()
        self._log_viewer_window.activateWindow()

    @Slot()
    def _on_open_pt_utils(self) -> None:
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
    def _on_about(self) -> None:
        from sessionpreplib import __version__ as ver

        QMessageBox.about(
            self,
            "About SessionPrep",
            f"<h2>SessionPrep</h2><p>Version {ver}</p>",
        )

    @Slot()
    def _resume_close_after_daw(self) -> None:
        if self._close_after_daw:
            self.close()

    def closeEvent(self, event: Any) -> None:
        if self._batch_ctrl.confirm_or_defer_close(closing_after_daw=self._close_after_daw):
            event.ignore()
            return
        self._daw_ctrl.pending_after_check = None
        if self._daw_ctrl.has_running_workers():
            self._close_after_daw = True
            event.ignore()
            return

        self._close_after_daw = False
        self._topology_ctrl.dismiss_overwrite_prompt()
        self._batch_ctrl.reset_close_flag()
        for detached in (self._pt_utils_window, self._log_viewer_window):
            if detached is not None and isValid(detached):
                detached.close()
        self._playback.close()
        self._repair_preview.close()

        self._analysis_ctrl.dispose()
        self._persistence_ctrl.dispose()
        self._topology_ctrl.dispose()
        self._daw_ctrl.dispose()
        self._detail_ctrl.dispose()

        for worker in self.findChildren(QThread):
            if isValid(worker) and worker.isRunning():
                cancel_fn = getattr(worker, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
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
            assert track.audio_data is not None
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


def main(*, smoke_files: Any = None) -> None:
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
