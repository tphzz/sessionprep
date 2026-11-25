"""Phase 3 (DAW Transfer) page widget layout."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ..change_marker import install_tab_marker, use_marker_tab_bar
from ..controls import FormActionButton, bind_primary_action, content_splitter
from ..theme import COLORS, CONTENT_INSET, tune_toolbar_checkbox
from ..tracks.table_widgets import (
    _SETUP_RIGHT_PLACEHOLDER,
    _FolderDropTree,
    _SetupDragTable,
)
from .signal_flow_panel import SignalFlowPanel

_SETUP_TABLE_EXTRA_WIDTH = 30
_SETUP_RIGHT_MIN_WIDTH = 360

_FETCH_BUTTON_STYLE = """
QToolButton#fetchSplitButton {
    color: #dddddd;
    padding: 4px 18px 4px 8px;
    background: transparent;
    border: none;
}
QToolButton#fetchSplitButton:hover {
    background-color: #3a3a3a;
}
QToolButton#fetchSplitButton:pressed {
    background-color: #2a6db5;
}
QToolButton#fetchSplitButton:disabled {
    color: #aaaaaa;
    background: transparent;
}
QToolButton#fetchSplitButton::menu-button {
    width: 14px;
    border: none;
    background: transparent;
    subcontrol-origin: padding;
    subcontrol-position: right center;
}
QToolButton#fetchSplitButton::menu-button:hover {
    background-color: #3a3a3a;
}
"""


class DawSetupPage(QWidget):
    """Pure UI container for Phase 3 (DAW Transfer)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toolbar = self._build_toolbar()
        self.main_layout.addWidget(self.toolbar)

        self.splitter = content_splitter()
        self.setup_table = self._build_setup_table()
        self.right_stack = self._build_right_stack()
        self.splitter.addWidget(self.setup_table)
        self.splitter.addWidget(self.right_stack)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.schedule_splitter_fit()

        self.main_layout.addWidget(self.splitter, 1)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Session Setup")
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        self.daw_combo = QComboBox()
        self.daw_combo.setMinimumWidth(140)
        self.daw_combo.setMaximumWidth(300)
        toolbar.addWidget(self.daw_combo)

        self.daw_check_label = QLabel("")
        self.daw_check_label.setContentsMargins(6, 0, 0, 0)
        self.daw_check_label.setMaximumWidth(260)
        toolbar.addWidget(self.daw_check_label)

        self.reset_manifest_action = QAction("Reset DAW Track List", self)
        self.reset_manifest_action.setToolTip(
            "Remove duplicate DAW tracks and restore default DAW track names"
        )
        self.reset_manifest_action.setEnabled(False)

        toolbar.addSeparator()

        self.use_processed_cb = QCheckBox("Use Processed")
        tune_toolbar_checkbox(self.use_processed_cb)
        self.use_processed_cb.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.use_processed_cb.setEnabled(False)
        toolbar.addWidget(self.use_processed_cb)
        toolbar.addAction(self.reset_manifest_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._build_fetch_buttons(toolbar)

        self.auto_assign_action = QAction("Auto-Assign", self)
        self.auto_assign_action.setEnabled(False)
        toolbar.addAction(self.auto_assign_action)

        self.unassign_all_action = QAction("Unassign All", self)
        self.unassign_all_action.setToolTip("Clear all local DAW folder assignments")
        self.unassign_all_action.setEnabled(False)
        toolbar.addAction(self.unassign_all_action)

        self.transfer_action = QAction("Create", self)
        self.transfer_action.setEnabled(False)
        toolbar.addAction(self.transfer_action)
        bind_primary_action(toolbar, self.transfer_action)
        return toolbar

    def _build_fetch_buttons(self, toolbar: QToolBar) -> None:
        self.fetch_action = QToolButton(self)
        self.fetch_action.setObjectName("fetchSplitButton")
        self.fetch_action.setText("Fetch")
        self.fetch_action.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.fetch_action.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.fetch_action.setStyleSheet(_FETCH_BUTTON_STYLE)
        self.fetch_action.setEnabled(False)

        fetch_menu = QMenu(self.fetch_action)
        self.fetch_cached_action = fetch_menu.addAction("Fetch (cached)")
        self.fetch_ignore_cache_action = fetch_menu.addAction("Fetch (ignore cache)")

        self.fetch_menu_button = QToolButton(self)
        self.fetch_menu_button.setText("v")
        self.fetch_menu_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.fetch_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.fetch_menu_button.setMenu(fetch_menu)
        self.fetch_menu_button.setFixedWidth(14)
        self.fetch_menu_button.setEnabled(False)
        self.fetch_menu_button.hide()

        self.fetch_action.setMenu(fetch_menu)
        toolbar.addWidget(self.fetch_action)

    def _build_setup_table(self) -> _SetupDragTable:
        table = _SetupDragTable()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["Track Name", "\u2713", "File", "Ch", "Clip Gain", "Fader Gain", "Group"]
        )
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.verticalHeader().setDefaultSectionSize(24)
        table.verticalHeader().setVisible(False)
        table.setMinimumWidth(300)
        table.setShowGrid(True)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)

        sh = table.horizontalHeader()
        sh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        sh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        sh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        sh.resizeSection(1, 24)
        sh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        sh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        sh.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        sh.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        sh.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        sh.resizeSection(2, 180)
        sh.resizeSection(3, 30)
        sh.resizeSection(4, 90)
        sh.resizeSection(5, 90)
        sh.resizeSection(6, 110)
        return table

    def _build_right_stack(self) -> QStackedWidget:
        stack = QStackedWidget()

        placeholder = QWidget()
        p_layout = QVBoxLayout(placeholder)
        p_layout.setContentsMargins(*(CONTENT_INSET,) * 4)
        p_layout.addStretch(2)
        p_label = QLabel("Connect to a DAW to configure routing")
        p_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p_label.setStyleSheet(f"color: {COLORS['dim']}; font-size: 13pt;")
        p_layout.addWidget(p_label)
        p_layout.addStretch(3)
        stack.addWidget(placeholder)

        tree_page = QWidget()
        t_layout = QVBoxLayout(tree_page)
        t_layout.setContentsMargins(*(CONTENT_INSET,) * 4)
        t_layout.setSpacing(CONTENT_INSET)

        proj_row = QWidget()
        proj_layout = QHBoxLayout(proj_row)
        proj_layout.setContentsMargins(0, 0, 0, 0)
        proj_layout.setSpacing(8)
        proj_layout.addWidget(QLabel("<b>Project Name:</b>"))
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Enter project name...")
        proj_layout.addWidget(self.project_name_edit, 1)
        self.open_project_btn = FormActionButton("Open Project Folder")
        proj_layout.addWidget(self.open_project_btn)
        t_layout.addWidget(proj_row)

        self.rollback_banner = self._build_rollback_banner()
        t_layout.addWidget(self.rollback_banner)

        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("QTabWidget::pane { border: none; margin: 0; padding: 0; }")
        use_marker_tab_bar(self.right_tabs)
        self.right_tabs.setDocumentMode(True)

        routing_tab = self._build_routing_tab()
        self.signal_flow_panel = SignalFlowPanel()
        self.right_tabs.addTab(routing_tab, "Routing && Folders")
        self.right_tabs.addTab(self.signal_flow_panel, "Signal Flow")
        install_tab_marker(self.right_tabs, 1)
        self.right_tabs.setTabEnabled(1, False)
        self.right_tabs.tabBar().hide()
        t_layout.addWidget(self.right_tabs, 1)

        stack.addWidget(tree_page)
        stack.setCurrentIndex(_SETUP_RIGHT_PLACEHOLDER)
        return stack

    def _build_rollback_banner(self) -> QWidget:
        banner = QWidget()
        banner.setObjectName("rollbackBanner")
        banner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        banner.setStyleSheet(
            f"QWidget#rollbackBanner {{ background-color: #2e1a1a; border: 1px solid #6e3030; "
            f"border-left: 3px solid {COLORS['problems']}; border-radius: 2px; }}"
        )
        r_lay = QHBoxLayout(banner)
        r_lay.setContentsMargins(8, 6, 8, 6)
        r_lay.setSpacing(8)
        self.rollback_label = QLabel("")
        self.rollback_label.setWordWrap(True)
        self.rollback_label.setStyleSheet(f"border: none; background: transparent; color: {COLORS['text']};")
        r_lay.addWidget(self.rollback_label, 1)
        self.open_log_btn = FormActionButton("Open Log Viewer")
        self.open_log_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.open_log_btn.setStyleSheet(
            f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
        )
        r_lay.addWidget(self.open_log_btn)
        banner.hide()
        return banner

    def _build_routing_tab(self) -> QWidget:
        tab = QWidget()
        self._routing_tab_layout = QVBoxLayout(tab)
        self._routing_tab_layout.setContentsMargins(0, 0, 0, 0)
        self._routing_tab_layout.setSpacing(6)

        self.ptx_summary_bar = QWidget()
        self.ptx_summary_bar.setObjectName("ptxSummaryBar")
        self.ptx_summary_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.ptx_summary_bar.setStyleSheet(
            f"QWidget#ptxSummaryBar {{ background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['accent']}; border-radius: 2px; }}"
        )
        s_lay = QHBoxLayout(self.ptx_summary_bar)
        s_lay.setContentsMargins(8, 4, 8, 4)
        s_lay.setSpacing(8)
        self.ptx_summary_label = QLabel("")
        self.ptx_summary_label.setStyleSheet(f"border: none; background: transparent; color: {COLORS['metadata']};")
        self.ptx_summary_label.setWordWrap(True)
        s_lay.addWidget(self.ptx_summary_label, 1)
        self.edit_config_btn = QPushButton("Edit in Session Config")
        self.edit_config_btn.setFlat(True)
        self.edit_config_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_config_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.edit_config_btn.setStyleSheet(
            f"QPushButton {{ border: 2px solid transparent; background: transparent; "
            f"color: {COLORS['information']}; font-weight: bold; padding: 0px 2px; border-radius: 2px; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}"
            f"QPushButton:focus {{ border: 2px solid {COLORS['information']}; }}"
            f"QPushButton:disabled {{ color: {COLORS['dim']}; }}"
        )
        s_lay.addWidget(self.edit_config_btn)
        self.ptx_summary_bar.hide()
        self._routing_tab_layout.addWidget(self.ptx_summary_bar)

        self.folder_tree = _FolderDropTree()
        self.folder_tree.setHeaderLabels(["Folder / Track"])
        self.folder_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.folder_tree.setAlternatingRowColors(True)
        self.folder_tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.folder_tree.setStyleSheet(
            "QTreeWidget::item:selected { background-color: rgba(42, 109, 181, 128); }"
        )
        self._routing_tab_layout.addWidget(self.folder_tree, 1)
        return tab

    def set_protools_mode(self, enabled: bool) -> None:
        """Toggle Pro Tools right-pane tabs, summary bar, and 4-column enriched folder tree."""
        self.right_tabs.setTabEnabled(1, enabled)
        self.right_tabs.tabBar().setVisible(enabled)
        self.ptx_summary_bar.setVisible(enabled)
        pane_border = f"1px solid {COLORS['border']}" if enabled else "none"
        self.right_tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; border-top: {pane_border}; margin: 0; padding: 0; }}"
        )
        self._routing_tab_layout.setContentsMargins(0, 6 if enabled else 0, 0, 0)
        if not enabled:
            self.right_tabs.setCurrentIndex(0)
            self.rollback_banner.hide()
            self.folder_tree.setColumnCount(1)
            self.folder_tree.setHeaderLabels(["Folder / Track"])
            self.folder_tree.header().setStretchLastSection(True)
            return
        self.folder_tree.setColumnCount(4)
        self.folder_tree.setHeaderLabels(["Folder / Track", "Input Bus", "Planned Group & VCA", "Output Routing"])
        hdr = self.folder_tree.header()
        for col, width in enumerate((220, 115, 195, 210)):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            self.folder_tree.setColumnWidth(col, width)
        hdr.setStretchLastSection(True)

    def show_rollback_notice(self, detail: str) -> None:
        self.rollback_label.setText(f"\u2715  {detail}")
        self.rollback_banner.show()

    def clear_rollback_notice(self) -> None:
        self.rollback_banner.hide()

    def apply_table_column_modes(self) -> None:
        """Restore the intended setup-table resize behavior after auto-fit."""
        header = self.setup_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 24)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        for col in range(4, self.setup_table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)

    def table_content_width(self) -> int:
        header = self.setup_table.horizontalHeader()
        for col in range(self.setup_table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.setup_table.resizeColumnsToContents()
        width = sum(header.sectionSize(col) for col in range(self.setup_table.columnCount()))
        self.apply_table_column_modes()
        return width + self.setup_table.verticalHeader().width() + _SETUP_TABLE_EXTRA_WIDTH

    def fit_splitter_to_table(self) -> None:
        total_width = self.splitter.width() - self.splitter.handleWidth()
        if total_width <= 0:
            return
        table_width = self.table_content_width()
        min_left = self.setup_table.minimumWidth()
        max_left = max(min_left, total_width - _SETUP_RIGHT_MIN_WIDTH - 2 * CONTENT_INSET)
        left_width = min(max(table_width, min_left), max_left)
        self.splitter.setSizes([left_width, max(0, total_width - left_width)])

    def schedule_splitter_fit(self) -> None:
        QTimer.singleShot(0, self.fit_splitter_to_table)
