"""Pro Tools Utils — standalone utility window.

Hosts per-tool tabs and manages a shared PTSL engine connection.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from ...theme import apply_dark_theme

from .color_tool import ColorTool
from .track_height_tool import TrackHeightTool


def _connection_failure_message(exc: Exception) -> tuple[str, str]:
    """Return a compact user-facing connection failure title and hint."""
    if isinstance(exc, ImportError):
        return (
            "py-ptsl is not installed",
            "Install the Pro Tools scripting dependency, then click Connect.",
        )

    text = str(exc)
    lowered = text.lower()
    unavailable_markers = (
        "statuscode.unavailable",
        "connection refused",
        "failed to connect to all addresses",
        "connectex",
    )
    if any(marker in lowered for marker in unavailable_markers):
        return (
            "Pro Tools not available",
            "Start Pro Tools and make sure scripting is enabled, then click Connect.",
        )

    return (
        "Connection failed",
        "Check Pro Tools and PTSL, then click Connect to try again.",
    )


class ProToolsUtilsWindow(QDialog):
    """Detached utility window for Pro Tools interactive tools."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pro Tools Utils")
        self.setWindowFlag(Qt.Window, True)
        self.setMinimumSize(600, 300)
        self.setAttribute(Qt.WA_DeleteOnClose, False)  # reuse window

        self._config = config
        self._engine = None
        self._last_connection_error = ""

        self._init_ui()
        apply_dark_theme(self)

    # ── UI ────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Connection header
        header = QHBoxLayout()
        header.setSpacing(8)

        status_col = QVBoxLayout()
        status_col.setSpacing(2)
        self._status_label = QLabel("Disconnected")
        self._status_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        status_col.addWidget(self._status_label)
        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color: #888; font-size: 8pt;")
        status_col.addWidget(self._hint_label)
        header.addLayout(status_col, 1)

        self._details_btn = QPushButton("Details")
        self._details_btn.clicked.connect(self._show_connection_details)
        self._details_btn.setVisible(False)
        header.addWidget(self._details_btn)

        self._on_top_cb = QCheckBox("Always on Top")
        self._on_top_cb.setStyleSheet("color: #aaa; font-size: 8pt;")
        self._on_top_cb.toggled.connect(self._toggle_on_top)
        header.addWidget(self._on_top_cb)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._toggle_connection)
        header.addWidget(self._connect_btn)

        layout.addLayout(header)

        # Tab widget for tools
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs, 1)

        # Register tools
        self._color_tool = ColorTool(self._config, self)
        self._tabs.addTab(self._color_tool, "Color Picker")
        self._track_height_tool = TrackHeightTool(self._config, self)
        self._tabs.addTab(self._track_height_tool, "Track Heights")

    # ── Connection management ────────────────────────────────────────

    def _toggle_on_top(self, checked: bool):
        geo = self.geometry()
        was_visible = self.isVisible()
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        # Ensure standard title-bar buttons survive the flag change
        flags |= Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint
        self.setWindowFlags(flags)
        if was_visible:
            self.setGeometry(geo)
            self.show()

    def _toggle_connection(self):
        if self._engine is not None:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        try:
            from ptsl import Engine
            self._engine = Engine(
                company_name="SessionPrep",
                application_name="Pro Tools Utils",
            )
            self._status_label.setText("Connected")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 9pt;")
            self._hint_label.setText("")
            self._details_btn.setVisible(False)
            self._last_connection_error = ""
            self._connect_btn.setText("Disconnect")
            self._color_tool.set_engine(self._engine)
            self._track_height_tool.set_engine(self._engine)
        except Exception as e:
            title, hint = _connection_failure_message(e)
            self._status_label.setText(title)
            self._status_label.setStyleSheet("color: #f44336; font-size: 9pt;")
            self._hint_label.setText(hint)
            self._last_connection_error = str(e)
            self._details_btn.setVisible(bool(self._last_connection_error))
            self._connect_btn.setText("Connect")
            self._engine = None
            self._color_tool.set_engine(None)
            self._track_height_tool.set_engine(None)

    def _disconnect(self):
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None
        self._status_label.setText("Disconnected")
        self._status_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._hint_label.setText("")
        self._details_btn.setVisible(False)
        self._last_connection_error = ""
        self._connect_btn.setText("Connect")
        self._color_tool.set_engine(None)
        self._track_height_tool.set_engine(None)

    def _show_connection_details(self):
        if not self._last_connection_error:
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Pro Tools Connection Details")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(self._status_label.text())
        msg.setInformativeText(self._hint_label.text())
        msg.setDetailedText(self._last_connection_error)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def update_config(self, config: dict):
        """Update the config (e.g. after preferences change)."""
        self._config = config
        self._color_tool.update_config(config)
        self._track_height_tool.update_config(config)

    def showEvent(self, event):
        super().showEvent(event)
        if self._engine is None:
            self._connect()

    def closeEvent(self, event):
        self._disconnect()
        super().closeEvent(event)
