"""Pro Tools Utils — standalone utility window.

Hosts per-tool tabs and manages a shared PTSL engine connection.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
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
        self._connection_title = "Disconnected"
        self._connection_hint = ""
        self._connection_state = "disconnected"

        self._init_ui()
        apply_dark_theme(self)

    # ── UI ────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Connection header
        header = QHBoxLayout()
        header.setSpacing(8)

        header.addStretch()

        self._connection_btn = QPushButton("Pro Tools: Offline")
        self._connection_btn.clicked.connect(self._show_connection_dialog)
        header.addWidget(self._connection_btn)

        self._on_top_cb = QCheckBox("Always on Top")
        self._on_top_cb.setStyleSheet("color: #aaa; font-size: 8pt;")
        self._on_top_cb.toggled.connect(self._toggle_on_top)
        header.addWidget(self._on_top_cb)

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
        self._update_connection_button()

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
            self._connection_state = "connected"
            self._connection_title = "Connected to Pro Tools"
            self._connection_hint = ""
            self._last_connection_error = ""
            self._update_connection_button()
            self._color_tool.set_engine(self._engine)
            self._track_height_tool.set_engine(self._engine)
        except Exception as e:
            title, hint = _connection_failure_message(e)
            self._connection_state = "failed"
            self._connection_title = title
            self._connection_hint = hint
            self._last_connection_error = str(e)
            self._engine = None
            self._update_connection_button()
            self._color_tool.set_engine(None)
            self._track_height_tool.set_engine(None)

    def _disconnect(self):
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None
        self._connection_state = "disconnected"
        self._connection_title = "Disconnected"
        self._connection_hint = ""
        self._last_connection_error = ""
        self._update_connection_button()
        self._color_tool.set_engine(None)
        self._track_height_tool.set_engine(None)

    def _update_connection_button(self):
        if self._connection_state == "connected":
            text = "Pro Tools: Connected"
            color = "#4caf50"
        elif self._connection_state == "failed":
            text = "Pro Tools: Offline"
            color = "#f44336"
        else:
            text = "Pro Tools: Offline"
            color = "#aaa"
        self._connection_btn.setText(text)
        self._connection_btn.setToolTip(self._connection_title)
        self._connection_btn.setStyleSheet(
            f"color: {color}; font-weight: 600; padding: 4px 12px;"
        )

    def _show_connection_dialog(self):
        dialog = _ConnectionDialog(self)
        dialog.exec()

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


class _ConnectionDialog(QDialog):
    """Connection status and diagnostics for Pro Tools Utils."""

    def __init__(self, window: ProToolsUtilsWindow):
        super().__init__(window)
        self._window = window
        self.setWindowTitle("Pro Tools Connection")
        self.setMinimumSize(760, 360)
        self._init_ui()
        self._refresh()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._title = QLabel("")
        self._title.setStyleSheet("font-size: 11pt; font-weight: 600;")
        layout.addWidget(self._title)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aaa; font-size: 9pt;")
        layout.addWidget(self._hint)

        self._details_label = QLabel("Technical Details")
        self._details_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        layout.addWidget(self._details_label)

        self._details = QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._details.setFont(QFont("Consolas", 9))
        layout.addWidget(self._details, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.clicked.connect(self._copy_details)
        buttons.addWidget(self._copy_btn)
        self._action_btn = QPushButton("Connect")
        self._action_btn.clicked.connect(self._on_action)
        buttons.addWidget(self._action_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _refresh(self):
        window = self._window
        self._title.setText(window._connection_title)
        self._hint.setText(window._connection_hint)
        has_error = bool(window._last_connection_error)
        self._details_label.setVisible(has_error)
        self._details.setVisible(has_error)
        self._details.setPlainText(window._last_connection_error)
        self._copy_btn.setVisible(has_error)

        if window._engine is None:
            self._action_btn.setText("Connect")
        else:
            self._action_btn.setText("Disconnect")

    def _on_action(self):
        if self._window._engine is None:
            self._window._connect()
        else:
            self._window._disconnect()
        self._refresh()

    def _copy_details(self):
        if self._window._last_connection_error:
            QApplication.clipboard().setText(self._window._last_connection_error)
