"""Pro Tools Utils — standalone utility window.

Hosts per-tool tabs and manages a shared PTSL engine connection.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal
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
from .connection_common import (
    PTSL_CONNECT_TIMEOUT_MS,
    connection_button_state as _connection_button_state,
)
from .track_height_tool import TrackHeightTool
from .worker_client import ProToolsWorkerClient


class ProToolsUtilsWindow(QDialog):
    """Detached utility window for Pro Tools interactive tools."""

    connection_state_changed = Signal()

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
        self._connection_attempt_id = 0
        self._client: ProToolsWorkerClient | None = None
        self._closing = False
        self._suppress_next_show_connect = False

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
            self._suppress_next_show_connect = True
            self.show()

    def _toggle_connection(self):
        if self._connection_state == "connected":
            self._disconnect()
        else:
            self._start_connect()

    def _start_connect(self):
        if self._connection_state == "connecting":
            return

        self._connection_attempt_id += 1
        attempt_id = self._connection_attempt_id
        self._connection_state = "connecting"
        self._connection_title = "Connecting to Pro Tools..."
        self._connection_hint = ""
        self._last_connection_error = ""
        self._engine = None
        self._color_tool.set_client(None)
        self._track_height_tool.set_client(None)
        self._update_connection_button()
        self.connection_state_changed.emit()

        self._client = ProToolsWorkerClient(self)
        self._client.worker_failed.connect(self._on_worker_failed)
        self._client.request(
            "connect",
            {},
            lambda response, current_attempt=attempt_id: self._on_connect_response(
                response,
                current_attempt,
            ),
            timeout_ms=PTSL_CONNECT_TIMEOUT_MS,
        )

    def _on_connect_response(self, response: dict, attempt_id: int):
        if self._is_stale_connect_result(attempt_id):
            return
        if response.get("ok"):
            self._engine = self._client
            self._connection_state = "connected"
            self._connection_title = "Connected to Pro Tools"
            self._connection_hint = ""
            self._last_connection_error = ""
            self._update_connection_button()
            self._color_tool.set_client(self._client)
            self._track_height_tool.set_client(self._client)
            self.connection_state_changed.emit()
        else:
            self._set_connection_failed(
                str(response.get("error") or ""),
                str(response.get("title") or "Connection failed"),
                str(response.get("hint") or ""),
            )

    def _on_worker_failed(self, error: str, title: str, hint: str):
        if self._closing or self._connection_state == "disconnected":
            return
        self._set_connection_failed(error, title, hint)

    def _set_connection_failed(self, error: str, title: str, hint: str):
        self._engine = None
        if self._client is not None:
            self._client.stop()
            self._client = None
        self._connection_state = "failed"
        self._connection_title = title
        self._connection_hint = hint
        self._last_connection_error = error
        self._update_connection_button()
        self._color_tool.set_client(None)
        self._track_height_tool.set_client(None)
        self.connection_state_changed.emit()

    def _is_stale_connect_result(self, attempt_id: int) -> bool:
        return (
            self._closing
            or attempt_id != self._connection_attempt_id
            or self._connection_state != "connecting"
        )

    def _disconnect(self):
        self._connection_attempt_id += 1
        if self._client is not None:
            self._client.stop()
            self._client = None
        self._engine = None
        self._connection_state = "disconnected"
        self._connection_title = "Disconnected"
        self._connection_hint = ""
        self._last_connection_error = ""
        self._update_connection_button()
        self._color_tool.set_client(None)
        self._track_height_tool.set_client(None)
        self.connection_state_changed.emit()

    def _update_connection_button(self):
        text, color = _connection_button_state(self._connection_state)
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
        self._closing = False
        if self._suppress_next_show_connect:
            self._suppress_next_show_connect = False
            return
        if (
            self._engine is None
            and self._connection_state != "connecting"
        ):
            self._start_connect()

    def closeEvent(self, event):
        self._closing = True
        self._disconnect()
        super().closeEvent(event)


class _ConnectionDialog(QDialog):
    """Connection status and diagnostics for Pro Tools Utils."""

    def __init__(self, window: ProToolsUtilsWindow):
        super().__init__(window)
        self._window = window
        self.setWindowTitle("Pro Tools Connection")
        self.setMinimumSize(760, 360)
        self._close_on_connected = False
        self._init_ui()
        self._window.connection_state_changed.connect(self._refresh)
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
        if (
            self._close_on_connected
            and window._connection_state == "connected"
        ):
            self.accept()
            return
        self._title.setText(window._connection_title)
        self._hint.setText(window._connection_hint)
        has_error = bool(window._last_connection_error)
        self._details_label.setVisible(has_error)
        self._details.setVisible(has_error)
        self._details.setPlainText(window._last_connection_error)
        self._copy_btn.setVisible(has_error)

        if window._engine is None:
            if window._connection_state == "connecting":
                self._action_btn.setText("Connecting...")
                self._action_btn.setEnabled(False)
            else:
                self._action_btn.setText("Connect")
                self._action_btn.setEnabled(True)
        else:
            self._action_btn.setText("Disconnect")
            self._action_btn.setEnabled(True)

    def _on_action(self):
        if self._window._engine is None:
            self._close_on_connected = True
            self._window._start_connect()
        else:
            self._close_on_connected = False
            self._window._disconnect()
        self._refresh()

    def _copy_details(self):
        if self._window._last_connection_error:
            QApplication.clipboard().setText(self._window._last_connection_error)
