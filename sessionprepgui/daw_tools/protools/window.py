"""Pro Tools Utils — standalone utility window.

Hosts per-tool tabs and manages a shared PTSL engine connection.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtGui import QFont
from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from sessionpreplib.daw_processors.ptsl_connection import ProToolsConnectionSettings

from ...theme import apply_dark_theme
from ...window_geometry import (
    STARTUP_SCREEN_MARGIN,
    calculate_startup_geometry,
    screen_for_startup,
)

from .color_tool import ColorTool
from .connection_common import (
    PTSL_CONNECT_TIMEOUT_MS,
    connection_button_state as _connection_button_state,
)
from .track_height_tool import TrackHeightTool
from .worker_client import ProToolsWorkerClient

PT_UTILS_WIDTH_FRACTION = 0.82
PT_UTILS_COLOR_HEIGHT_FRACTION = 0.45
PT_UTILS_COMFORT_HEIGHT = 16


class _ToolTabs(QWidget):
    """Tab bar plus one active tool widget.

    Unlike QTabWidget, inactive tools are not kept in a stacked layout, so
    they cannot contribute hidden-page minimum sizes.
    """

    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: list[QWidget] = []
        self._current_index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setDocumentMode(True)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tab_bar)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        layout.addWidget(self._content, 1)

    def setDocumentMode(self, enabled: bool):
        self._tab_bar.setDocumentMode(enabled)

    def tabBar(self) -> QTabBar:
        return self._tab_bar

    def addTab(self, widget: QWidget, label: str):
        index = len(self._widgets)
        self._widgets.append(widget)
        widget.setParent(None)
        self._tab_bar.addTab(label)
        if self._current_index < 0:
            self._tab_bar.setCurrentIndex(index)
            self._set_current_index(index)
        return index

    def currentWidget(self) -> QWidget | None:
        if 0 <= self._current_index < len(self._widgets):
            return self._widgets[self._current_index]
        return None

    def setCurrentWidget(self, widget: QWidget):
        try:
            index = self._widgets.index(widget)
        except ValueError:
            return
        self._tab_bar.setCurrentIndex(index)
        self._set_current_index(index)

    def _on_tab_changed(self, index: int):
        self._set_current_index(index)
        self.currentChanged.emit(index)

    def _set_current_index(self, index: int):
        if index == self._current_index or not (0 <= index < len(self._widgets)):
            return

        current = self.currentWidget()
        if current is not None:
            self._content_layout.removeWidget(current)
            current.setParent(None)

        self._current_index = index
        widget = self._widgets[index]
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._content_layout.addWidget(widget)
        widget.show()
        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        return self._current_tool_hint(super().minimumSizeHint(), minimum=True)

    def sizeHint(self) -> QSize:
        return self._current_tool_hint(super().sizeHint(), minimum=False)

    def _current_tool_hint(self, base: QSize, *, minimum: bool) -> QSize:
        current = self.currentWidget()
        if current is None:
            return base
        page_hint = current.minimumSizeHint() if minimum else current.sizeHint()
        tab_hint = (
            self._tab_bar.minimumSizeHint()
            if minimum else self._tab_bar.sizeHint()
        )
        return QSize(
            max(page_hint.width(), tab_hint.width()),
            page_hint.height() + tab_hint.height(),
        )


class ProToolsUtilsWindow(QDialog):
    """Detached utility window for Pro Tools interactive tools."""

    connection_state_changed = Signal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pro Tools Utils")
        self.setWindowFlag(Qt.Window, True)
        self.setMinimumSize(600, 220)
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
        self._initial_geometry_applied = False
        self._applying_geometry = False

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
        self._tabs = _ToolTabs()
        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs, 1)

        # Register tools
        self._color_tool = ColorTool(self._config, self)
        self._tabs.addTab(self._color_tool, "Color Picker")
        self._track_height_tool = TrackHeightTool(self._config, self)
        self._tabs.addTab(self._track_height_tool, "Track Heights")
        self._tabs.currentChanged.connect(self._on_tool_tab_changed)
        self._update_connection_button()

    # ── Window sizing ────────────────────────────────────────

    def _available_geometry(self):
        screen = self.screen() or screen_for_startup()
        if screen is None:
            return None
        return screen.availableGeometry()

    def _tool_content_width(self, window_width: int) -> int:
        layout = self.layout()
        if layout is None:
            return max(1, window_width)
        margins = layout.contentsMargins()
        return max(1, window_width - margins.left() - margins.right())

    def _chrome_height_for_tool(self, tool_height: int) -> int:
        layout = self.layout()
        if layout is None:
            return tool_height
        margins = layout.contentsMargins()
        spacing = layout.spacing()
        header_height = max(
            self._connection_btn.sizeHint().height(),
            self._on_top_cb.sizeHint().height(),
        )
        tab_height = self._tabs.tabBar().sizeHint().height()
        return (
            margins.top()
            + margins.bottom()
            + header_height
            + tab_height
            + tool_height
            + spacing * 2
            + PT_UTILS_COMFORT_HEIGHT
        )

    def _color_picker_window_height(self, window_width: int) -> int:
        content_width = self._tool_content_width(window_width)
        tool_height = self._color_tool.preferred_compact_height_for_width(
            content_width,
            use_minimum_grid_height=True,
        )
        return self._chrome_height_for_tool(tool_height)

    def _track_heights_window_height(self, window_width: int) -> int:
        content_width = self._tool_content_width(window_width)
        tool_height = self._track_height_tool.preferred_expanded_height_for_width(
            content_width
        )
        return self._chrome_height_for_tool(tool_height)

    def _clamp_window_height(self, height: int, available) -> int:
        if available is None:
            return max(self.minimumHeight(), height)
        max_height = max(1, available.height() - STARTUP_SCREEN_MARGIN * 2)
        return min(max(self.minimumHeight(), height), max_height)

    def _apply_initial_geometry(self):
        available = self._available_geometry()
        if available is None:
            self._set_geometry_programmatically(QRect(0, 0, 1280, 420))
            self._initial_geometry_applied = True
            return

        base = calculate_startup_geometry(
            available,
            self.minimumSizeHint(),
            width_fraction=PT_UTILS_WIDTH_FRACTION,
            height_fraction=PT_UTILS_COLOR_HEIGHT_FRACTION,
        )
        width = base.width()
        height = self._clamp_window_height(
            self._color_picker_window_height(width),
            available,
        )
        geometry = QRect(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
            width,
            height,
        )
        self._set_geometry_programmatically(geometry)
        self._initial_geometry_applied = True

    def _on_tool_tab_changed(self, _index: int):
        if not self._closing:
            QTimer.singleShot(0, self._resize_for_current_tab)

    def _resize_for_current_tab(self):
        if self._closing:
            return

        available = self._available_geometry()
        current = self.geometry()
        current_tool = self._tabs.currentWidget()
        if current_tool is self._track_height_tool:
            target_height = self._track_heights_window_height(current.width())
        else:
            target_height = self._color_picker_window_height(current.width())

        desired_height = self._clamp_window_height(
            target_height,
            available,
        )
        if desired_height == current.height():
            return

        new_geometry = QRect(current)
        new_geometry.setHeight(desired_height)
        if available is not None:
            max_bottom = available.bottom() - STARTUP_SCREEN_MARGIN
            if new_geometry.bottom() > max_bottom:
                new_geometry.moveTop(
                    max(available.top(), max_bottom - desired_height + 1)
                )
            if new_geometry.left() < available.left():
                new_geometry.moveLeft(available.left())
            if new_geometry.right() > available.right():
                new_geometry.moveRight(available.right())
        self._set_geometry_programmatically(new_geometry)

    def _set_geometry_programmatically(self, geometry: QRect):
        self._applying_geometry = True
        self.setGeometry(geometry)
        QTimer.singleShot(150, self._clear_programmatic_geometry_flag)

    def _clear_programmatic_geometry_flag(self):
        self._applying_geometry = False

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
            self._set_geometry_programmatically(geo)
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
            {"settings": self._connection_settings().to_dict()},
            lambda response, current_attempt=attempt_id: self._on_connect_response(
                response,
                current_attempt,
            ),
            timeout_ms=PTSL_CONNECT_TIMEOUT_MS,
        )

    def _connection_settings(self) -> ProToolsConnectionSettings:
        settings = ProToolsConnectionSettings.from_config(self._config)
        return replace(settings, host_ready_timeout=5.0)

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
        if not self._initial_geometry_applied:
            self._apply_initial_geometry()
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
        self._reset_layout_state_for_next_show()
        super().closeEvent(event)

    def _reset_layout_state_for_next_show(self):
        self._applying_geometry = True
        try:
            self._tabs.setCurrentWidget(self._color_tool)
            self._initial_geometry_applied = False
        finally:
            self._applying_geometry = False


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
