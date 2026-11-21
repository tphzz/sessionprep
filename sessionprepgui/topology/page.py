"""Track Layout (Phase 1) page widget layout."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..controls import bind_primary_action, content_splitter
from ..theme import COLORS, CONTENT_INSET, tune_toolbar_checkbox
from ..waveform import WaveformPanel
from .input_tree import InputTree
from .output_tree import OutputTree


class TopologyPage(QWidget):
    """Pure UI container for Phase 1 (Track Layout)."""

    def __init__(self, *, recursive_scan: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toolbar = self._build_toolbar(recursive_scan)
        self.main_layout.addWidget(self.toolbar)

        h_splitter = self._build_trees_splitter()
        self.wf_panel = self._build_waveform_panel()
        self.wf_toggle = self.wf_panel.disclosure_button

        v_splitter = content_splitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self.wf_panel)
        v_splitter.setCollapsible(1, False)
        v_splitter.setSizes([700, 300])
        self.wf_panel.set_expanded(False)
        self.main_layout.addWidget(v_splitter, 1)

    def _build_toolbar(self, recursive_scan: bool) -> QToolBar:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())

        self.open_action = QAction("Open Folder", self)
        self.open_action.setToolTip("Open a directory containing audio files")
        toolbar.addAction(self.open_action)

        self.recursive_cb = QCheckBox("Scan subfolders")
        tune_toolbar_checkbox(self.recursive_cb)
        self.recursive_cb.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.recursive_cb.setToolTip(
            "When checked, Open Folder will recursively discover "
            "audio files in all subdirectories"
        )
        self.recursive_cb.setChecked(recursive_scan)
        toolbar.addWidget(self.recursive_cb)

        toolbar.addSeparator()

        self.reanalyze_action = QAction("Reanalyze", self)
        self.reanalyze_action.setToolTip("Rescan the current folder for new or changed files")
        self.reanalyze_action.setEnabled(False)
        toolbar.addAction(self.reanalyze_action)

        self.reset_action = QAction("Reset to Default", self)
        self.reset_action.setToolTip("Rebuild the default passthrough topology from input tracks")
        self.reset_action.setEnabled(False)
        toolbar.addAction(self.reset_action)

        self.add_output_action = QAction("Add Output", self)
        self.add_output_action.setToolTip("Create one or more empty output tracks")
        toolbar.addAction(self.add_output_action)

        self.remove_empty_action = QAction("Remove Empty", self)
        self.remove_empty_action.setToolTip("Remove output tracks that have no wired source channels")
        toolbar.addAction(self.remove_empty_action)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"QLabel {{ color: {COLORS['dim']}; padding: 0 8px; }}")
        toolbar.addWidget(self.status_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.collapse_action = QAction("Collapse All", self)
        self.collapse_action.setToolTip("Collapse all output tree nodes")
        toolbar.addAction(self.collapse_action)

        self.expand_action = QAction("Expand All", self)
        self.expand_action.setToolTip("Expand all output tree nodes")
        toolbar.addAction(self.expand_action)

        toolbar.addSeparator()

        self.apply_action = QAction("Apply", self)
        self.apply_action.setToolTip("Write channel-rerouted files to the topology output folder")
        self.apply_action.setEnabled(False)
        toolbar.addAction(self.apply_action)
        bind_primary_action(toolbar, self.apply_action)

        return toolbar

    def _build_trees_splitter(self):
        h_splitter = content_splitter()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(*(CONTENT_INSET,) * 4)
        left_label = QLabel("Input Tracks")
        left_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        left_label.setStyleSheet(f"QLabel {{ color: {COLORS['text']}; font-weight: bold; }}")
        left_layout.addWidget(left_label)

        self.input_tree = InputTree()
        left_layout.addWidget(self.input_tree)
        h_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(*(CONTENT_INSET,) * 4)
        right_label = QLabel("Output Tracks")
        right_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_label.setStyleSheet(f"QLabel {{ color: {COLORS['text']}; font-weight: bold; }}")
        right_layout.addWidget(right_label)

        self.output_tree = OutputTree()
        right_layout.addWidget(self.output_tree)
        h_splitter.addWidget(right_panel)

        h_splitter.setSizes([400, 400])
        return h_splitter

    @staticmethod
    def _build_waveform_panel() -> WaveformPanel:
        panel = WaveformPanel(analysis_mode=False)
        panel.setVisible(True)
        return panel
