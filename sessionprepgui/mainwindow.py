"""Main application window for SessionPrep GUI."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from PySide6.QtCore import Qt, Slot, QSize, QTimer
from PySide6.QtGui import (
    QAction, QIcon, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QStatusBar,
    QWidget,
)

from sessionpreplib.config import (
    ConfigChangeImpact,
    classify_structured_config_change,
    default_config,
    flatten_structured_config,
    resolve_session_config,
)
from sessionpreplib.detectors import detector_help_map

from .settings import (
    DEFAULT_SCALE_FACTOR,
    load_config, save_config,
    resolve_config_preset, build_defaults,
)
from .controls import bind_primary_action, content_splitter, inset_pane
from .theme import COLORS, apply_dark_theme
from .log import dbg
from .prefs import PreferencesDialog
from .detail import PlaybackController, DetailMixin
from .change_marker import install_tab_marker, use_marker_tab_bar
from .detail.help_panel import FileReportPane
from .detail.file_review import FileReviewPane
from .detail.report_style import wrap_report_html
from .waveform import WaveformPanel, WaveformLoadWorker
from .widgets import ProgressPanel
from .window_geometry import (
    FALLBACK_STARTUP_SIZE,
    calculate_startup_geometry,
    screen_for_startup,
)
from .analysis import (
    AnalysisMixin,
    AudioLoadWorker, DawCheckWorker,
    DawFetchWorker, DawTransferWorker, PrepareWorker,
)
from .tracks import (
    TrackColumnsMixin, GroupsMixin,
    _DraggableTrackTable, _TAB_FILE, _TAB_GROUPS, _TAB_SESSION,
    _PAGE_TABS,
    _PHASE_ANALYSIS, _PHASE_SETUP,
)
from .daw import DawMixin
from .topology import TopologyMixin
from .batch import BatchQueueDock, BatchManager

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
            if minimum else self.tabBar().sizeHint()
        )

        chrome_w = max(0, base.width() - stack_hint.width())
        chrome_h = max(0, base.height() - stack_hint.height())
        return QSize(
            max(page_hint.width(), tab_hint.width()) + chrome_w,
            page_hint.height() + chrome_h,
        )


class SessionPrepWindow(  # pylint: disable=too-many-ancestors
    QMainWindow,
    AnalysisMixin, TrackColumnsMixin,
                        GroupsMixin, DawMixin, TopologyMixin, DetailMixin):
    def __init__(self):
        t_init = time.perf_counter()
        super().__init__()
        self.setWindowTitle("SessionPrep")
        self.setWindowIcon(_app_icon())

        self._session = None
        self._summary = None
        self._source_dir = None
        self._topology_dir = None  # path to sp_01_tracklayout/ after Phase 1 Apply
        self._topo_source_tracks = []  # original source tracks for Phase 1 input table
        self._topo_topology = None  # Phase 1 topology (separate from session.topology)
        self._worker = None
        self._batch_filenames: set[str] = set()
        self._wf_worker: WaveformLoadWorker | None = None
        self._audio_load_worker: AudioLoadWorker | None = None
        self._current_track = None
        self._selected_detector_overlays: set[str] = set()
        self._pending_detector_visualization: tuple[int, str] | None = None
        self._waveform_context_generation: int | None = None
        self._waveform_unavailable_generation: int | None = None
        self._session_groups: list[dict] = []
        self._active_session_preset: str = "Default"
        self._recursive_scan: bool = False
        self._session_overrides: dict[str, Any] = {}
        self._session_widgets: dict[str, list[tuple[str, QWidget]]] = {}
        self._pt_utils_window = None  # singleton Pro Tools Utils window
        self._log_viewer_window = None  # singleton detached log viewer
        self._report_screen_connection = None

        t0 = time.perf_counter()
        self._detector_help = detector_help_map()
        dbg(f"detector_help_map: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._daw_check_worker: DawCheckWorker | None = None
        self._pending_after_check = None
        self._daw_fetch_worker: DawFetchWorker | None = None
        self._daw_transfer_worker: DawTransferWorker | None = None
        self._prepare_worker: PrepareWorker | None = None
        self._repair_plan_worker = None
        self._repair_plan_pending = set()
        self._repair_plan_anchors = set()
        self._repair_plan_revalidate = set()

        self._batch_manager = BatchManager(self)
        self._close_after_batch = False
        self._batch_manager.finished.connect(self._on_batch_finished)
        self._batch_manager.item_finished.connect(self._on_batch_item_finished)

        # Load persistent GUI configuration (four-section structure)
        t0 = time.perf_counter()
        self._config = load_config()
        dbg(f"load_config: {(time.perf_counter() - t0) * 1000:.1f} ms")
        self._active_config_preset_name = self._default_config_preset_name()
        self._recursive_scan = self._config.get(
            "app", {}).get("recursive_scan", False)

        # Instantiate and configure DAW processors
        t0 = time.perf_counter()
        self._daw_processors: list = []
        self._active_daw_processor = None
        self._configure_daw_processors()
        dbg(f"daw_processors: {(time.perf_counter() - t0) * 1000:.1f} ms")

        # Playback controller
        t0 = time.perf_counter()
        self._playback = PlaybackController(self)
        self._playback.cursor_updated.connect(self._on_cursor_updated)
        self._playback.playback_finished.connect(self._on_playback_finished)
        self._playback.error.connect(self._on_playback_error)
        dbg(f"PlaybackController (sounddevice): "
            f"{(time.perf_counter() - t0) * 1000:.1f} ms")

        t0 = time.perf_counter()
        self._init_ui()
        dbg(f"_init_ui: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._batch_dock = BatchQueueDock(self)
        self._batch_dock.load_requested.connect(self._on_load_batch_item)
        self._batch_dock.run_batch_requested.connect(self._batch_manager.start_batch)
        self._batch_dock.run_single_requested.connect(self._batch_manager.start_single)
        self._batch_dock.open_project_requested.connect(self._on_open_batch_project_folder)
        self.addDockWidget(Qt.RightDockWidgetArea, self._batch_dock)
        self._batch_dock.hide()  # hidden by default

        self._batch_manager.started.connect(self._on_batch_started)
        self._batch_manager.batch_progress_value.connect(self._batch_dock.update_progress)
        self._batch_manager.batch_progress_message.connect(
            self._batch_dock.update_progress_message)
        self._batch_manager.batch_progress_message.connect(self._status_bar.showMessage)

        t0 = time.perf_counter()
        apply_dark_theme(self)
        dbg(f"apply_dark_theme: {(time.perf_counter() - t0) * 1000:.1f} ms")

        self._apply_initial_window_geometry()

        # Spacebar toggles play/stop
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.activated.connect(self._on_toggle_play)

        dbg(f"SessionPrepWindow.__init__ total: "
            f"{(time.perf_counter() - t_init) * 1000:.1f} ms")

    # ── Config helpers ───────────────────────────────────────────────────

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
        geometry = calculate_startup_geometry(
            available,
            minimum_hint,
        )
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
            _size_text(self.centralWidget().minimumSizeHint())
            if self.centralWidget() else "none",
            _size_text(self.centralWidget().sizeHint())
            if self.centralWidget() else "none",
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

    def _flat_config(self) -> dict[str, Any]:
        """Return the flattened effective configuration for runtime consumers."""
        flat = default_config()
        flat.update(flatten_structured_config(self._effective_session_config()))
        return flat

    def _effective_session_config(self) -> dict[str, Any]:
        return resolve_session_config(self._active_preset(), self._session_overrides)

    def _existing_config_preset_name(self, name: str) -> str:
        if isinstance(name, str) and name in self._config.get("config_presets", {}):
            return name
        return "Default"

    def _default_config_preset_name(self) -> str:
        return self._existing_config_preset_name(
            self._config.get("app", {}).get("active_config_preset", "Default"))

    def _active_preset(self) -> dict[str, Any]:
        """Return the active config preset's structured dict."""
        return resolve_config_preset(
            self._config, self._active_config_preset_name)

    # ── UI setup ──────────────────────────────────────────────────────────

    def _init_ui(self):
        self._init_menus()
        self._phase_progress_layouts = {}
        self._phase_progress_panels = {}
        self._clear_progress()

        # ── Top-level phase tabs ──────────────────────────────────────────
        self._phase_tabs = _CurrentPageTabWidget()
        self._phase_tabs.setObjectName("phaseTabs")
        self._phase_tabs.setDocumentMode(True)

        # Tab 0 — Phase 1: Track Layout (landing page)
        self._phase_tabs.addTab(
            self._build_topology_page(),
            "Phase 1: Track Layout")

        # Tab 1 — Phase 2: Analysis & Preparation
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
        self._phase_tabs.addTab(
            analysis_page, "Phase 2: Analysis && Preparation")
        self._phase_tabs.setTabEnabled(_PHASE_ANALYSIS, False)

        # Tab 2 — Phase 3: DAW Transfer
        self._phase_tabs.addTab(
            self._build_setup_page(), "Phase 3: DAW Transfer")
        self._phase_tabs.setTabEnabled(_PHASE_SETUP, False)
        self._phase_tabs.currentChanged.connect(self._on_phase_tab_changed)

        self.setCentralWidget(self._phase_tabs)

        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(
            "QStatusBar { background-color: #1e1e1e; border-top: 1px solid #444; }"
            "QStatusBar::item { border: none; }")
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Open a directory containing .wav / .aif files to begin.")

    def _register_phase_progress_layout(self, phase_index: int,
                                        layout: QVBoxLayout) -> None:
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
        self._progress_panel_for_phase(
            self._progress_active_phase).set_progress(current, total)

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

    def _init_menus(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open &Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_path)
        file_menu.addAction(open_action)

        load_session_action = QAction("&Load Session...", self)
        load_session_action.setShortcut("Ctrl+Shift+O")
        load_session_action.triggered.connect(self._on_load_session)
        file_menu.addAction(load_session_action)

        load_batch_action = QAction("Load Batch Queue...", self)
        load_batch_action.triggered.connect(self._on_load_batch_queue)
        file_menu.addAction(load_batch_action)

        file_menu.addSeparator()

        self._save_session_action = QAction("&Save Session...", self)
        self._save_session_action.setShortcut("Ctrl+S")
        self._save_session_action.setEnabled(False)
        self._save_session_action.triggered.connect(self._on_save_session)
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

        # ── Tools menu ────────────────────────────────────────────────
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

        start_dir = self._config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Queue", start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)"
        )
        if not path:
            return

        try:
            state = self._batch_dock.get_state()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self._status_bar.showMessage(f"Batch queue saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Batch Queue Failed", f"Could not save batch queue:\n\n{exc}")

    @Slot()
    def _on_load_batch_queue(self):
        start_dir = self._config.get("app", {}).get("default_project_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Batch Queue", start_dir,
            "Batch Queue Files (*.spbatch);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load Batch Queue Failed", f"Could not load batch queue:\n\n{exc}")
            return

        if not isinstance(state, list):
            QMessageBox.critical(self, "Invalid File", "The selected file is not a valid batch queue format.")
            return

        append = False
        if self._batch_dock.has_items:
            msg = QMessageBox(self)
            msg.setWindowTitle("Load Batch Queue")
            msg.setText("How would you like to load the batch queue?")

            replace_btn = msg.addButton("Replace", QMessageBox.AcceptRole)
            append_btn = msg.addButton("Append", QMessageBox.AcceptRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

            msg.exec()

            if msg.clickedButton() == cancel_btn:
                return
            if msg.clickedButton() == append_btn:
                append = True

        self._batch_dock.load_state(state, append=append)

        self._batch_mode_action.setChecked(True)
        self._clear_workspace()
        self._status_bar.showMessage(f"Batch queue loaded from {path}")

    @Slot(bool)
    def _on_batch_mode_toggled(self, checked: bool):
        self._batch_dock.setVisible(checked)
        self._update_daw_lifecycle_buttons()

    @Slot(object)
    def _on_load_batch_item(self, item):
        if self._session:
            reply = QMessageBox.question(
                self, "Load Session",
                "You have an active session in the workspace. Discard current workspace and load from queue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self._restore_session_state(item.session_state)
        self._batch_dock.remove_item(item.id)

    @Slot(object)
    def _on_open_batch_project_folder(self, item):
        daw_processor = None
        for dp in self._daw_processors:
            if dp.id == item.daw_processor_id:
                daw_processor = dp
                break

        if not daw_processor:
            QMessageBox.warning(
                self, "Open Project Folder",
                f"The configured DAW processor '{item.daw_processor_name}' is not available."
            )
            return

        project_dir = getattr(daw_processor, "project_dir", "").strip()

        if not project_dir:
            QMessageBox.information(
                self, "Open Project Folder",
                f"No project directory configured for {daw_processor.name}."
            )
            return

        import os
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        # For DAWs like DAWproject that specify a full output path (e.g. .dawproject file)
        # we try to just open the directory containing it if the output path exists.
        if item.output_path and os.path.exists(os.path.dirname(item.output_path)):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(item.output_path)))
            return

        # Otherwise, try the conventional project directory approach
        target_path = os.path.join(project_dir, item.project_name) if item.project_name else project_dir

        if os.path.isdir(target_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_path))
        elif os.path.isdir(project_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(project_dir))
        else:
            QMessageBox.warning(
                self, "Open Project Folder",
                f"The directory does not exist:\n\n{project_dir}"
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

    def _init_analysis_toolbar(self):
        self._analysis_toolbar = QToolBar("Analysis")
        self._analysis_toolbar.setIconSize(QSize(16, 16))
        self._analysis_toolbar.setMovable(False)
        self._analysis_toolbar.setFloatable(False)

        self._analyze_action = QAction("Reanalyze", self)
        self._analyze_action.setEnabled(False)
        self._analyze_action.triggered.connect(self._on_analyze)
        self._analysis_toolbar.addAction(self._analyze_action)

        # Preset selectors live in their respective Phase 2 tabs.
        # ── Spacer ─────────────────────────────────────────────────────
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._analysis_toolbar.addWidget(spacer)

        # ── Right: Auto-Group + Prepare buttons ──────────────────────
        self._auto_group_action = QAction("Auto-Group", self)
        self._auto_group_action.setEnabled(False)
        self._auto_group_action.triggered.connect(self._on_auto_group)
        self._analysis_toolbar.addAction(self._auto_group_action)

        self._prepare_action = QAction("Prepare", self)
        self._prepare_action.setEnabled(False)
        self._prepare_action.triggered.connect(self._on_prepare)
        self._analysis_toolbar.addAction(self._prepare_action)
        bind_primary_action(self._analysis_toolbar, self._prepare_action)

    def _populate_config_preset_combo(self):
        """Fill the config-preset combo from config, preserving the current selection."""
        if not hasattr(self, "_config_preset_combo"):
            return
        presets = self._config.get("config_presets",
                                   build_defaults().get("config_presets", {}))
        active = self._active_config_preset_name
        self._config_preset_combo.blockSignals(True)
        self._config_preset_combo.clear()
        for name in presets:
            self._config_preset_combo.addItem(name)
        idx = self._config_preset_combo.findText(active)
        if idx >= 0:
            self._config_preset_combo.setCurrentIndex(idx)
        elif self._config_preset_combo.count() > 0:
            self._config_preset_combo.setCurrentIndex(0)
        self._config_preset_combo.blockSignals(False)

    def _populate_group_preset_combo(self):
        """Fill the group-preset combo from config, preserving the current selection."""
        if not hasattr(self, "_group_preset_combo"):
            return
        presets = self._config.get("group_presets",
                                   build_defaults().get("group_presets", {}))
        active = self._active_session_preset
        self._group_preset_combo.blockSignals(True)
        self._group_preset_combo.clear()
        for name in presets:
            self._group_preset_combo.addItem(name)
        idx = self._group_preset_combo.findText(active)
        if idx >= 0:
            self._group_preset_combo.setCurrentIndex(idx)
        elif self._group_preset_combo.count() > 0:
            self._group_preset_combo.setCurrentIndex(0)
        self._group_preset_combo.blockSignals(False)

    # ── Panel builders ────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        self._track_table = _DraggableTrackTable()
        self._track_table.setColumnCount(8)
        self._track_table.setHorizontalHeaderLabels(
            ["File", "Ch", "Analysis", "Classification", "Gain",
             "RMS Anchor", "Group", "Processing"]
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
        # File fills the free width (≥ 160 px, see fit_file_column); other
        # columns keep their widths and the table scrolls when it is too narrow.
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)
        header.setSectionResizeMode(7, QHeaderView.Interactive)
        header.resizeSection(1, 30)
        header.resizeSection(2, 150)
        header.resizeSection(3, 120)
        header.resizeSection(4, 90)
        header.resizeSection(5, 100)
        header.resizeSection(6, 140)
        header.resizeSection(7, 130)
        header.sectionResized.connect(
            lambda index, *_: index and self._track_table.fit_file_column())
        self._track_table.fit_file_column()

        self._track_table.cellClicked.connect(self._on_row_clicked)
        self._track_table.currentCellChanged.connect(self._on_current_cell_changed)
        # Reanalysis hides the table while keeping its splitter pane visible.
        return inset_pane(self._track_table, (0, 0, 0, 0))

    def _build_right_panel(self) -> QWidget:
        """Build the right-hand side analysis detail tabs."""
        self._right_stack = QStackedWidget()
        self._right_stack.setMinimumWidth(400)

        # ── Tabs (Summary / File) ─────────────────────────────────────────
        self._detail_tabs = QTabWidget()
        use_marker_tab_bar(self._detail_tabs)
        self._detail_tabs.setDocumentMode(True)
        self._detail_tabs.currentChanged.connect(self._on_detail_tab_changed)

        # Summary tab — single QTextBrowser
        self._summary_view = self._make_report_browser()
        self._detail_tabs.addTab(self._summary_view, "Summary")

        # File tab — vertical splitter (report + waveform)
        self._file_splitter = content_splitter(Qt.Vertical)

        self._file_report = self._make_report_browser()
        self._file_report_pane = FileReportPane(self._file_report)
        self._file_review = FileReviewPane(self._file_report_pane, self._file_splitter)
        self._repair_editor = self._file_review.editor
        self._file_report.repairRequested.connect(lambda key: self._open_track_repairs(self._current_track, key))
        self._repair_editor.choiceChanged.connect(self._set_track_repair_choice)
        self._repair_editor.adjustGainRequested.connect(self._adjust_failed_output_gain)
        self._file_review.processingFocusRequested.connect(self._focus_processing_menu)
        self._file_report.visualizeRequested.connect(self._visualize_detector)
        self._file_report.sourceLocationsRequested.connect(self._show_source_locations)
        self._file_report.verificationRetryRequested.connect(self._retry_detector_verification)
        self._file_report.reportChanged.connect(self._on_file_report_context_changed)
        self._file_splitter.addWidget(self._file_review)

        # Waveform panel (toolbar + waveform + transport)
        self._wf_panel = WaveformPanel(analysis_mode=True)
        self._wf_panel.play_clicked.connect(self._on_play)
        self._wf_panel.stop_clicked.connect(self._on_stop)
        self._wf_panel.position_clicked.connect(self._on_waveform_seek)
        self._wf_panel.display_mode_changed.connect(self._on_display_mode_changed)
        self._wf_panel.waveform.set_invert_scroll(
            self._config.get("app", {}).get("invert_scroll", "default"))

        # Backward-compat aliases for DetailMixin / other mixins
        self._waveform = self._wf_panel.waveform
        self._wf_container = self._wf_panel
        self._overlay_btn = self._wf_panel.overlay_btn
        self._overlay_menu = self._wf_panel.overlay_menu
        self._markers_toggle = self._wf_panel.markers_toggle
        self._rms_lr_toggle = self._wf_panel.rms_lr_toggle
        self._rms_avg_toggle = self._wf_panel.rms_avg_toggle
        self._wf_settings_btn = self._wf_panel.wf_settings_btn
        self._spec_settings_btn = self._wf_panel.spec_settings_btn
        self._wf_action = self._wf_panel.wf_action
        self._spec_action = self._wf_panel.spec_action
        self._display_mode_btn = self._wf_panel.display_mode_btn
        self._cmap_group = self._wf_panel.cmap_group
        self._play_btn = self._wf_panel.play_btn
        self._stop_btn = self._wf_panel.stop_btn
        self._time_label = self._wf_panel.time_label
        from .detail.repair_preview import RepairPreview
        self._repair_preview = RepairPreview(self)

        # Connect spectrogram action groups to DetailMixin slots
        self._wf_panel.fft_group.triggered.connect(self._on_spec_fft_changed)
        self._wf_panel.win_group.triggered.connect(self._on_spec_window_changed)
        self._wf_panel.cmap_group.triggered.connect(self._on_spec_cmap_changed)
        self._wf_panel.floor_group.triggered.connect(self._on_spec_floor_changed)
        self._wf_panel.ceil_group.triggered.connect(self._on_spec_ceil_changed)

        self._file_splitter.addWidget(self._wf_panel)
        self._file_splitter.setCollapsible(1, False)

        self._file_splitter.setStretchFactor(0, 3)
        self._file_splitter.setStretchFactor(1, 1)
        self._file_splitter.setSizes([620, 272])

        self._detail_tabs.addTab(self._file_splitter, "File")
        self._detail_tabs.setTabEnabled(_TAB_FILE, False)

        # Groups tab — session-local group editor
        self._detail_tabs.addTab(self._build_groups_tab(), "Groups")
        install_tab_marker(self._detail_tabs, _TAB_GROUPS)
        self._detail_tabs.setTabEnabled(_TAB_GROUPS, False)

        # Config tab — per-session config overrides
        self._detail_tabs.addTab(
            self._build_session_settings_tab(), "Config")
        install_tab_marker(self._detail_tabs, _TAB_SESSION)
        self._detail_tabs.setTabEnabled(_TAB_SESSION, False)

        self._right_stack.addWidget(self._detail_tabs)

        # Start on the tabs page (summary empty until first analysis)
        self._right_stack.setCurrentIndex(_PAGE_TABS)

        return self._right_stack

    @Slot(int)
    def _on_detail_tab_changed(self, index: int):
        """Explicitly hide the File tab content when leaving it.

        QTabWidget with setDocumentMode(True) can fail to fully hide
        the previous tab's widget (QSplitter + custom-painted waveform),
        causing visual bleed-through on other tabs.  Work around this by
        explicitly managing _file_splitter visibility.
        """
        fs = getattr(self, "_file_splitter", None)
        if fs is not None:
            fs.setVisible(index == _TAB_FILE)

    def _make_report_browser(self):
        """Create a consistently styled QTextBrowser for reports."""
        from .detail.browser import ReportBrowser
        browser = ReportBrowser(self._detector_help)
        browser.reviewRequested.connect(self._review_summary_file)
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

    # ── Slots: phase tabs ─────────────────────────────────────────────────

    @Slot(int)
    def _on_phase_tab_changed(self, index: int):
        if getattr(self, "_progress_running", False):
            self._show_progress_on_phase(index)
        if index == _PHASE_SETUP:
            self._schedule_setup_splitter_fit()

    # ── HTML helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Wrap HTML content in a styled <body> tag."""
        return wrap_report_html(body)

    # ── Preferences ───────────────────────────────────────────────────────

    def _refresh_config_changed_display(self) -> None:
        """Refresh visible reports for config changes that do not reanalyze."""
        if self._session:
            self._render_summary()
        cmap = self._config.get("app", {}).get("spectrogram_colormap", "magma")
        self._waveform.set_colormap(cmap)
        if self._current_track:
            self._refresh_file_report(self._current_track)

    def _confirm_config_change_reanalysis(
        self,
        impact: ConfigChangeImpact,
        *,
        intro: str,
    ) -> bool:
        """Ask whether to apply config changes that require reanalysis."""
        if impact.requires_phase1:
            ans = QMessageBox.question(
                self,
                "Rerun Track Layout?",
                intro
                + "\n\nThese changes affect Track Layout checks such as "
                "dual-mono or one-sided-silence detection. Rerun Track "
                "Layout now to apply them to the current session?\n\n"
                "Topology may change, so review Phase 1 before continuing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return ans == QMessageBox.Yes

        if impact.requires_phase2:
            ans = QMessageBox.question(
                self,
                "Reanalyze Phase 2?",
                intro
                + "\n\nThese changes affect Phase 2 analysis or gain "
                "planning. Reanalyze Phase 2 now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            return ans == QMessageBox.Yes

        return True

    def _apply_config_change_impact(self, impact: ConfigChangeImpact) -> None:
        """Apply a preset switch, including any confirmed reanalysis."""
        self._refresh_effective_config(impact, reanalyzing=impact.requires_phase1 or impact.requires_phase2)
        if impact.requires_phase1:
            self._on_topo_reanalyze()
        elif impact.requires_phase2:
            self._on_analyze()
        elif not (impact.presentation_keys and self._session):
            self._refresh_config_changed_display()

    def _refresh_effective_config(self, impact: ConfigChangeImpact, *, reanalyzing=False) -> None:
        """Refresh consumers that can apply settings without reanalysis."""
        if impact.daw_keys:
            self._refresh_daw_processors_from_config()
        if any(key.startswith("processors.") and key.endswith("_enabled")
               for key in impact.phase2_keys):
            self._refresh_enabled_processors()
        if impact.requires_phase1 or impact.requires_phase2:
            self._mark_prepare_stale()
        if self._session and self._phase_tabs.isTabEnabled(_PHASE_ANALYSIS) and not reanalyzing:
            if impact.repair_keys:
                self._queue_repair_plan(self._session.tracks)
            elif impact.requires_phase1 or impact.requires_phase2:
                flat = self._flat_config()
                tracks = [t for t in self._session.tracks if t.repair_recipe.active
                          or self._repair_settings_can_apply(t, flat)]
                if tracks:
                    self._queue_repair_plan(tracks)
        if impact.presentation_keys and self._session:
            self._refresh_presentation()

    def _refresh_daw_processors_from_config(self) -> None:
        """Rebuild DAW processors and related Phase 3 UI from current config."""
        active_id = getattr(self._active_daw_processor, "id", None)
        previous = self._daw_template_bindings
        self._configure_daw_processors()
        self._populate_daw_combo(self._matching_daw_id(active_id, previous))
        if self._session and (
            previous != self._daw_template_bindings
            or active_id != getattr(self._active_daw_processor, "id", None)
        ):
            self._populate_folder_tree()
            self._populate_setup_table()
        self._daw_check_label.setText("")
        self._update_daw_lifecycle_buttons()
        self._update_tools_menu()

    @Slot()
    def _on_preferences(self):
        old_scale = self._config.get("app", {}).get(
            "scale_factor", DEFAULT_SCALE_FACTOR)
        old_config = self._effective_session_config()
        old_group_preset = self._config.get("group_presets", {}).get(
            self._active_session_preset)

        dlg = PreferencesDialog(self._config, parent=self)
        dlg.exec()
        if dlg.saved:
            self._config = dlg.result_config()
            save_config(self._config)
            if self._session is not None:
                name = dlg.preset_name_after_save(self._active_config_preset_name)
                self._active_config_preset_name = self._existing_config_preset_name(name)
            else:
                self._active_config_preset_name = self._default_config_preset_name()
            new_config = self._effective_session_config()
            impact = classify_structured_config_change(
                old_config, new_config)
            self._status_bar.showMessage("Preferences saved.")
            self._waveform.set_invert_scroll(
                self._config.get("app", {}).get("invert_scroll", "default"))

            # Refresh preset combos (presets may have been added/removed/renamed)
            self._populate_group_preset_combo()
            self._populate_config_preset_combo()
            self._load_session_widgets(new_config)
            self._refresh_effective_config(impact)

            # Offer to merge if the session's active preset was modified
            if self._session:
                preset_name = self._active_session_preset
                presets = self._config.get("group_presets",
                                          build_defaults().get("group_presets", {}))
                if preset_name in presets:
                    preset_groups = presets[preset_name]
                    if (preset_groups != old_group_preset
                            and preset_groups != self._session_groups):
                        ans = QMessageBox.question(
                            self, "Update session groups?",
                            f'The group preset \u201c{preset_name}\u201d'
                            " has changed.\n\n"
                            "Update the current session\u2019s groups"
                            " to match?\n\n"
                            "\u2022 Track assignments will be preserved"
                            " where group names match.\n"
                            "\u2022 Unmatched tracks will be set to"
                            " (None).",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if ans == QMessageBox.Yes:
                            self._merge_groups_from_preset()
                            self._status_bar.showMessage(
                                "Session groups updated from preset.")

            self._update_tools_menu()

            # Update Pro Tools Utils window if open
            if self._pt_utils_window is not None:
                self._pt_utils_window.update_config(self._config)

            if not (impact.presentation_keys and self._session):
                self._refresh_config_changed_display()
            if self._source_dir and (impact.requires_phase1 or impact.requires_phase2):
                phase = "Track Layout" if impact.requires_phase1 else "Phase 2 analysis"
                message = (f"Rerun {phase} to update results for the current session."
                           if not (self._repair_plan_worker or self._repair_plan_pending)
                           else f"Repair checks update automatically. Rerun {phase} to refresh the full session.")
                QMessageBox.information(
                    self, "Config preset updated",
                    f"Inherited settings have been updated from Preferences. {message}",
                )
            elif impact.daw_keys:
                self._status_bar.showMessage("Preferences saved. Phase 3 settings updated.")

            # Prompt restart if scale factor changed
            new_scale = self._config.get("app", {}).get(
                "scale_factor", DEFAULT_SCALE_FACTOR)
            if new_scale != old_scale:
                QMessageBox.information(
                    self, "Restart required",
                    f"HiDPI scale factor changed from {old_scale} to {new_scale}.\n"
                    "Please restart SessionPrep for the new scaling to take effect.",
                )
    # ── Tools menu ─────────────────────────────────────────────────────────

    def _update_tools_menu(self):
        """Enable/disable Tools menu entries based on active config."""
        preset = self._active_preset()
        pt_section = preset.get("daw_processors", {}).get("protools", {})
        pt_enabled = pt_section.get("protools_enabled", False)
        self._pt_utils_action.setEnabled(pt_enabled)

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
            self._pt_utils_window = ProToolsUtilsWindow(self._config)
        else:
            self._pt_utils_window.update_config(self._config)
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

    def closeEvent(self, event):
        if self._batch_dock.has_items and not self._close_after_batch:
            reply = QMessageBox.warning(
                self, "Pending Batch Items",
                "You have pending sessions in the batch queue. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        if self._batch_manager.is_running:
            self._close_after_batch = True
            self._batch_manager.stop_after_current()
            event.ignore()
            return
        self._close_after_batch = False

        self._playback.close()
        self._pending_after_check = None
        self._repair_preview.close()
        # Qt destroys child threads along with their window. A long filter can
        # finish its current track after cancellation, including on a closed project.
        from .analysis.worker import RepairPlanWorker
        for worker in self.findChildren(RepairPlanWorker):
            worker.cancel()
            worker.wait()
        self._repair_plan_worker = None
        for worker in self.findChildren(PrepareWorker):
            worker.requestInterruption()
            worker.wait()
        self._prepare_worker = None

        # Stop active background tasks cleanly so ThreadPoolExecutor can terminate
        from PySide6.QtCore import QThread
        for worker_attr in (
            "_p1_worker",
            "_worker",
            "_peak_build_worker",
            "_topo_apply_worker",
            "_daw_check_worker",
            "_daw_fetch_worker",
            "_daw_transfer_worker",
            "_wf_worker",
            "_audio_load_worker",
        ):
            worker = getattr(self, worker_attr, None)
            if isinstance(worker, QThread) and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                else:
                    worker.requestInterruption()
                worker.wait(500)
            setattr(self, worker_attr, None)
        if hasattr(self, "_topo_cancel_workers"):
            self._topo_cancel_workers()

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


def main():
    from sessionpreplib.logging_setup import setup_logging
    setup_logging()

    t_main = time.perf_counter()

    # Apply HiDPI scale factor before QApplication is created.
    # Read directly from JSON to avoid the validate-and-overwrite path
    # in load_config() which could reset the file to defaults.
    import json as _json
    from .settings import (
        DEFAULT_SCALE_FACTOR as _default_scale,
        config_path as _cfg_path,
        startup_scale_factor_from_raw_config as _startup_scale,
    )
    scale = _default_scale
    try:
        with open(_cfg_path(), "r", encoding="utf-8") as _f:
            _raw = _json.load(_f)
        scale = _startup_scale(_raw)
    except Exception:
        pass
    if scale is not None and float(scale) != 1.0:
        os.environ["QT_SCALE_FACTOR"] = str(float(scale))

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
    QTimer.singleShot(0, lambda: log.debug(
        "Startup geometry after event loop starts: actual=%s frame=%s "
        "window_state=%s",
        _rect_text(window.geometry()),
        _rect_text(window.frameGeometry()),
        _window_state_text(window.windowState()),
    ))

    dbg(f"main() total: {(time.perf_counter() - t_main) * 1000:.1f} ms")
    sys.exit(app.exec())
