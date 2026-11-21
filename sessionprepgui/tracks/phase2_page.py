"""Phase 2 (Analysis & Preparation) pure page widget container."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..change_marker import install_tab_marker, use_marker_tab_bar
from ..controls import bind_primary_action, content_splitter, inset_pane
from ..detail.browser import ReportBrowser
from ..detail.file_review import FileReviewPane
from ..detail.help_panel import FileReportPane
from ..waveform import WaveformPanel
from ..window_geometry import screen_for_startup
from .table_widgets import (
    TrackCol,
    _DraggableTrackTable,
    _PAGE_TABS,
    _TAB_FILE,
    _TAB_GROUPS,
    _TAB_SESSION,
)


class Phase2Page(QWidget):
    """Owns the Phase 2 toolbar, track table, detail tabs, file review, and waveform panel."""

    def __init__(
        self,
        detector_help: dict[str, Any],
        *,
        invert_scroll: Any = "default",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._detector_help = detector_help

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self._build_toolbar()
        self.main_layout.addWidget(self.toolbar)

        self.main_splitter = content_splitter()
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_right_panel(invert_scroll))
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([620, 480])
        self.main_layout.addWidget(self.main_splitter, 1)

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("Analysis", self)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        self.analyze_action = QAction("Reanalyze", self)
        self.analyze_action.setEnabled(False)
        self.toolbar.addAction(self.analyze_action)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        self.auto_group_action = QAction("Auto-Group", self)
        self.auto_group_action.setEnabled(False)
        self.toolbar.addAction(self.auto_group_action)

        self.prepare_action = QAction("Prepare", self)
        self.prepare_action.setEnabled(False)
        self.toolbar.addAction(self.prepare_action)
        bind_primary_action(self.toolbar, self.prepare_action)

    def _build_left_panel(self) -> QWidget:
        table = _DraggableTrackTable()
        table.setColumnCount(len(TrackCol))
        table.setHorizontalHeaderLabels(
            ["File", "Ch", "Analysis", "Classification", "Gain", "RMS Anchor", "Group", "Processing"]
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setDefaultSectionSize(24)
        table.verticalHeader().setVisible(False)
        table.setMinimumWidth(300)
        table.setShowGrid(True)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)

        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(TrackCol.FILE, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(TrackCol.CHANNELS, QHeaderView.ResizeMode.Fixed)
        for col in range(TrackCol.ANALYSIS, len(TrackCol)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        for col, width in (
            (TrackCol.CHANNELS, 30),
            (TrackCol.ANALYSIS, 150),
            (TrackCol.CLASSIFICATION, 120),
            (TrackCol.GAIN, 90),
            (TrackCol.ANCHOR, 100),
            (TrackCol.GROUP, 140),
            (TrackCol.PROCESSING, 130),
        ):
            header.resizeSection(col, width)
        def _on_section_resized(index: int, *_args: Any) -> None:
            if index:
                table.fit_file_column()

        header.sectionResized.connect(_on_section_resized)
        table.fit_file_column()
        self.track_table = table
        return inset_pane(table, (0, 0, 0, 0))

    def _build_right_panel(self, invert_scroll: Any) -> QWidget:
        self.right_stack = QStackedWidget()
        self.right_stack.setMinimumWidth(400)

        self.detail_tabs = QTabWidget()
        use_marker_tab_bar(self.detail_tabs)
        self.detail_tabs.setDocumentMode(True)
        self.detail_tabs.currentChanged.connect(self._on_detail_tab_changed)

        self.summary_view = self._make_report_browser()
        self.detail_tabs.addTab(self.summary_view, "Summary")

        self.detail_tabs.addTab(self._build_file_tab_widgets(invert_scroll), "File")
        self.detail_tabs.setTabEnabled(_TAB_FILE, False)

        self.right_stack.addWidget(self.detail_tabs)
        self.right_stack.setCurrentIndex(_PAGE_TABS)
        return self.right_stack

    def _build_file_tab_widgets(self, invert_scroll: Any) -> QWidget:
        self.file_splitter = content_splitter(Qt.Orientation.Vertical)

        self.file_report = self._make_report_browser()
        self.file_report_pane = FileReportPane(self.file_report)
        self.file_review = FileReviewPane(self.file_report_pane, self.file_splitter)
        self.repair_editor = self.file_review.editor
        self.file_splitter.addWidget(self.file_review)

        self.wf_panel = WaveformPanel(analysis_mode=True)
        self.wf_panel.waveform.set_invert_scroll(invert_scroll)
        self.file_splitter.addWidget(self.wf_panel)
        self.file_splitter.setCollapsible(1, False)
        self.file_splitter.setStretchFactor(0, 3)
        self.file_splitter.setStretchFactor(1, 1)
        self.file_splitter.setSizes([620, 272])
        return self.file_splitter

    def add_groups_tab(self, widget: QWidget) -> None:
        self.detail_tabs.addTab(widget, "Groups")
        install_tab_marker(self.detail_tabs, _TAB_GROUPS)
        self.detail_tabs.setTabEnabled(_TAB_GROUPS, False)

    def add_config_tab(self, widget: QWidget) -> None:
        self.detail_tabs.addTab(widget, "Config")
        install_tab_marker(self.detail_tabs, _TAB_SESSION)
        self.detail_tabs.setTabEnabled(_TAB_SESSION, False)

    def _make_report_browser(self) -> ReportBrowser:
        browser = ReportBrowser(self._detector_help)
        browser.refresh_font(self.screen() or screen_for_startup())
        return browser

    def refresh_report_fonts(self, screen: Any = None) -> None:
        target = screen or self.screen() or screen_for_startup()
        self.summary_view.refresh_font(target)
        self.file_report.refresh_font(target)
        self.file_report_pane.refresh_fonts(target)

    @Slot(int)
    def _on_detail_tab_changed(self, index: int) -> None:
        fs = getattr(self, "file_splitter", None)
        if fs is not None:
            fs.setVisible(index == _TAB_FILE)
