"""Reusable waveform panel: preview header (with view controls) + WaveformWidget + transport bar."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal, QSize
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
    QSplitter,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icons import icon
from .widget import WaveformWidget


class _PreviewDisclosureButton(QToolButton):
    """Keep keyboard focus visible without making pointer clicks look selected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("waveformPreviewDisclosure")
        self.setProperty("keyboardFocus", False)

    def _set_keyboard_focus(self, enabled: bool) -> None:
        if self.property("keyboardFocus") == enabled:
            return
        self.setProperty("keyboardFocus", enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def focusInEvent(self, event):
        self._set_keyboard_focus(event.reason() in (
            Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason))
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self._set_keyboard_focus(False)
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        self._set_keyboard_focus(False)
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        self._set_keyboard_focus(True)
        super().keyPressEvent(event)


class _ViewControlRow(QWidget):
    """Waveform view controls in the preview header row, with a » overflow menu.

    The panel decides which controls are wanted (display mode, analysis mode);
    the row decides which of those fit. Controls that do not fit move, from the
    end, into the » menu: dropdowns become submenus and toggles checkable items
    that click the real button, so every existing signal path is kept.
    """

    SPACING = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controls: list[QToolButton] = []
        self._wanted: dict[QToolButton, bool] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self.SPACING, 0, self.SPACING, 0)
        layout.setSpacing(self.SPACING)
        self.more_button = QToolButton(self)
        self.more_button.setText("\u00bb")
        self.more_button.setToolTip("More view controls")
        self.more_button.setAccessibleName("More view controls")
        self.more_button.setAutoRaise(True)
        self.more_button.setPopupMode(QToolButton.InstantPopup)
        self.more_button.setObjectName("waveformMoreViewControls")
        # "»" already says "more"; no extra menu arrow (approved design).
        self.more_button.setStyleSheet(
            "QToolButton#waveformMoreViewControls::menu-indicator { image: none; width: 0; }")
        self.more_button.setFocusPolicy(Qt.StrongFocus)
        self.more_menu = QMenu(self.more_button)
        self.more_menu.setToolTipsVisible(True)
        self.more_menu.aboutToShow.connect(self._fill_menu)
        self.more_button.setMenu(self.more_menu)
        self.more_button.hide()
        layout.addWidget(self.more_button)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def add_control(self, control: QToolButton) -> None:
        layout = self.layout()
        layout.insertWidget(layout.indexOf(self.more_button), control)
        self._controls.append(control)
        self._wanted[control] = True

    @property
    def controls(self) -> list[QToolButton]:
        return list(self._controls)

    def set_wanted(self, control: QToolButton, wanted: bool) -> None:
        """Show or hide a control by panel logic (it may still overflow)."""
        self._wanted[control] = bool(wanted)
        self.updateGeometry()  # sizeHint depends on the wanted controls
        self._relayout()

    def is_wanted(self, control: QToolButton) -> bool:
        return self._wanted.get(control, False)

    def overflowed(self) -> list[QToolButton]:
        """Wanted controls currently moved into the » menu."""
        return [c for c in self._controls if self._wanted[c] and c.isHidden()]

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        margins = self.layout().contentsMargins()
        return QSize(margins.left() + self.more_button.sizeHint().width() + margins.right(),
                     hint.height())

    def sizeHint(self):
        hint = super().sizeHint()
        margins = self.layout().contentsMargins()
        wanted = [c for c in self._controls if self._wanted[c]]
        width = sum(c.sizeHint().width() for c in wanted) + self.SPACING * max(0, len(wanted) - 1)
        return QSize(margins.left() + width + margins.right(), hint.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def event(self, event):
        # A control's text change (e.g. "Detector Overlays (n)") posts a layout
        # request without resizing the row: re-check what fits.
        result = super().event(event)
        if event.type() == QEvent.LayoutRequest:
            self._relayout()
        return result

    def _relayout(self) -> None:
        margins = self.layout().contentsMargins()
        available = self.width() - margins.left() - margins.right()
        wanted = [c for c in self._controls if self._wanted[c]]
        widths = [c.sizeHint().width() for c in wanted]
        total = sum(widths) + self.SPACING * max(0, len(wanted) - 1)
        fits = len(wanted)
        if total > available:
            # Reserve the » button, then keep the leading controls that fit.
            room = available - self.more_button.sizeHint().width()
            used, fits = 0, 0
            for width in widths:
                needed = used + width + self.SPACING
                if needed > room:
                    break
                used, fits = needed, fits + 1
        shown = set(wanted[:fits])
        # Show » first so focus leaving an overflowing control can land on it.
        overflow = fits < len(wanted)
        self.more_button.setVisible(overflow)
        focused = QApplication.focusWidget()
        for control in self._controls:
            visible = control in shown
            if control.isHidden() == visible:
                control.setVisible(visible)
        if overflow and focused in self._controls and focused.isHidden():
            self.more_button.setFocus(Qt.OtherFocusReason)

    def _fill_menu(self) -> None:
        self.more_menu.clear()
        for control in self.overflowed():
            menu = control.menu()
            if menu is not None:
                menu.setTitle(control.text())
                action = self.more_menu.addMenu(menu)
            else:
                action = self.more_menu.addAction(control.text())
                action.setCheckable(control.isCheckable())
                action.setChecked(control.isChecked())
                action.triggered.connect(lambda _checked=False, c=control: c.click())
            action.setEnabled(control.isEnabled())
            action.setToolTip(control.toolTip())


class WaveformPanel(QWidget):
    """Composite widget: preview header + WaveformWidget + transport bar.

    The view controls sit in the preview header row after the disclosure (one
    row in both phases); controls that do not fit move into a » menu.

    Parameters
    ----------
    analysis_mode : bool
        When True, analysis-specific toolbar controls (Detector Overlays,
        Peak/RMS Max, RMS L/R, RMS AVG) are visible. When False they are
        hidden — suitable for Phase 1 or other contexts without analysis data.
    """

    # Signals forwarded from inner widgets / transport
    play_clicked = Signal()
    stop_clicked = Signal()
    position_clicked = Signal(int)
    display_mode_changed = Signal(str)

    def __init__(self, analysis_mode: bool = True, parent=None):
        super().__init__(parent)
        self._analysis_mode = analysis_mode
        self._expanded = True
        self._content_available = False
        self._expanded_splitter_sizes = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.waveform = WaveformWidget()
        self.waveform.position_clicked.connect(self.position_clicked)
        self.preview_header = self._build_preview_header()
        layout.addWidget(self.preview_header)
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self._view_toolbar = self._build_toolbar()
        self.preview_header.layout().insertWidget(1, self._view_toolbar, 1)
        self._set_header_tab_order()
        body_layout.addWidget(self.waveform, 1)
        body_layout.addWidget(self._build_transport())
        layout.addWidget(self._body, 1)
        self.disclosure_button.toggled.connect(self.set_expanded)

    def _build_preview_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("waveformPreviewHeader")
        header.setFixedHeight(32)
        header.setStyleSheet("""
            #waveformPreviewHeader { background: #2d2d2d; border-bottom: 1px solid #555; }
            QToolButton { color: #ddd; background: transparent; border: 2px solid transparent;
                          padding: 0 6px; }
            QToolButton:hover { background: #3a3a3a; }
            QToolButton:focus { border-color: #4499ff; }
            QToolButton#waveformPreviewDisclosure:focus { border-color: transparent; }
            QToolButton#waveformPreviewDisclosure[keyboardFocus="true"]:focus {
                border-color: #4499ff;
            }
            QToolButton:disabled { color: #aaa; }
        """)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 8, 0)
        row.setSpacing(0)
        self.disclosure_button = _PreviewDisclosureButton()
        self.disclosure_button.setText("Waveform preview")
        self.disclosure_button.setAccessibleName("Waveform preview")
        self.disclosure_button.setToolTip("Show or hide the waveform preview")
        self.disclosure_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.disclosure_button.setIcon(icon("chevron-down"))
        self.disclosure_button.setIconSize(QSize(16, 16))
        self.disclosure_button.setCheckable(True)
        self.disclosure_button.setChecked(True)
        self.disclosure_button.setFocusPolicy(Qt.StrongFocus)
        self.disclosure_button.setFixedHeight(32)
        row.addWidget(self.disclosure_button)
        # The view-control row (inserted at index 1, stretch 1) takes the free
        # width; while it is hidden (collapsed) this stretch keeps the
        # disclosure left and the zoom group packed at the right.
        row.addStretch()
        self._zoom_buttons: list[QToolButton] = []
        for name, tooltip, callback in (
            ("fit", "Fit waveform", self.waveform.zoom_fit),
            ("zoom-in", "Zoom in at cursor", self.waveform.zoom_in),
            ("zoom-out", "Zoom out at cursor", self.waveform.zoom_out),
            ("scale-up", "Scale up (vertical)", self.waveform.scale_up),
            ("scale-down", "Scale down (vertical)", self.waveform.scale_down),
        ):
            button = QToolButton()
            button.setIcon(icon(name))
            button.setIconSize(QSize(20, 20))
            button.setFixedSize(32, 32)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.setFocusPolicy(Qt.StrongFocus)
            button.clicked.connect(callback)
            row.addWidget(button)
            self._zoom_buttons.append(button)
        return header

    def _set_header_tab_order(self) -> None:
        """Keyboard order follows the row: disclosure, view controls, », zoom."""
        chain = [self.disclosure_button, *self._view_toolbar.controls,
                 self._view_toolbar.more_button, *self._zoom_buttons]
        for control in self._view_toolbar.controls:
            control.setFocusPolicy(Qt.StrongFocus)
        for first, second in zip(chain, chain[1:]):
            QWidget.setTabOrder(first, second)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_content_available(self, available: bool) -> None:
        """Update data availability without overriding the user's disclosure."""
        self._content_available = available
        self.disclosure_button.setAccessibleDescription(
            "Audio preview available" if available else "No preview audio available")
        if not available:
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def set_expanded(self, expanded: bool) -> None:
        """Change visibility only; retain the waveform, workers and viewport."""
        expanded = bool(expanded)
        if self._expanded == expanded:
            return
        parent = self.parentWidget()
        splitter = parent if isinstance(parent, QSplitter) else None
        index = splitter.indexOf(self) if splitter else -1
        if splitter and index >= 0:
            splitter.setCollapsible(index, False)
        if not expanded and splitter and index >= 0:
            self._expanded_splitter_sizes = splitter.sizes()
        self._expanded = expanded
        self.disclosure_button.setChecked(expanded)
        self.disclosure_button.setIcon(icon("chevron-down" if expanded else "chevron-up"))
        self._body.setVisible(expanded)
        self._view_toolbar.setVisible(expanded)
        self.setMaximumHeight(16777215 if expanded else 32)
        self.setSizePolicy(QSizePolicy.Preferred,
                           QSizePolicy.Expanding if expanded else QSizePolicy.Fixed)
        if not expanded and splitter and index >= 0:
            sizes = list(self._expanded_splitter_sizes)
            header_height = self.preview_header.height()
            remaining = max(0, sum(sizes) - header_height)
            other_height = max(1, sum(size for i, size in enumerate(sizes) if i != index))
            splitter.setSizes([
                header_height if i == index else round(size * remaining / other_height)
                for i, size in enumerate(sizes)
            ])
        if expanded and splitter and index >= 0:
            sizes = self._expanded_splitter_sizes
            if sizes is None:
                sizes = splitter.sizes()
                total = sum(sizes)
                height = min(280, max(180, total // 3))
                others = max(1, total - sizes[index])
                sizes = [max(0, round(size * (total - height) / others))
                         if i != index else height for i, size in enumerate(sizes)]
            available = sum(splitter.sizes())
            previous = sum(sizes)
            if available > 0 and previous > 0:
                sizes = [round(size * available / previous) for size in sizes]
            splitter.setSizes(sizes)
        self.updateGeometry()

    def _build_toolbar(self) -> "_ViewControlRow":
        toolbar = _ViewControlRow()

        toggle_style = (
            "QToolButton:checked { background-color: #2a6db5;"
            " color: #ffffff; }")

        # The chevron sits about 4 px after the label with 8 px right padding
        # (approved header design); the padding reserves its slot.
        dropdown_style = (
            "QToolButton { padding-right: 21px; }"
            "QToolButton::menu-indicator { subcontrol-position: right center;"
            " subcontrol-origin: padding; right: 8px; }")

        # Display mode dropdown
        self.display_mode_btn = QToolButton()
        self.display_mode_btn.setText("Waveform")
        self.display_mode_btn.setToolTip(
            "Switch between Waveform and Spectrogram display")
        self.display_mode_btn.setPopupMode(QToolButton.InstantPopup)
        self.display_mode_btn.setAutoRaise(True)
        self.display_mode_btn.setStyleSheet(dropdown_style)
        display_menu = QMenu(self.display_mode_btn)
        self.wf_action = display_menu.addAction("Waveform")
        self.spec_action = display_menu.addAction("Spectrogram")
        self.wf_action.setCheckable(True)
        self.wf_action.setChecked(True)
        self.spec_action.setCheckable(True)
        self._display_group = QActionGroup(self)
        self._display_group.addAction(self.wf_action)
        self._display_group.addAction(self.spec_action)
        self._display_group.triggered.connect(self._on_display_mode_changed)
        self.display_mode_btn.setMenu(display_menu)
        self.display_mode_btn.setAccessibleName("Waveform display mode")
        toolbar.add_control(self.display_mode_btn)

        # Spectrogram settings dropdown
        self.spec_settings_btn = QToolButton()
        self.spec_settings_btn.setText("Display")
        self.spec_settings_btn.setToolTip(
            "Configure spectrogram display parameters")
        self.spec_settings_btn.setPopupMode(QToolButton.InstantPopup)
        self.spec_settings_btn.setAutoRaise(True)
        self.spec_settings_btn.setStyleSheet(dropdown_style)
        spec_menu = QMenu(self.spec_settings_btn)

        # -- FFT Size submenu --
        fft_menu = spec_menu.addMenu("FFT Size")
        self.fft_group = QActionGroup(self)
        for sz in (512, 1024, 2048, 4096, 8192):
            act = fft_menu.addAction(str(sz))
            act.setCheckable(True)
            act.setData(sz)
            if sz == 2048:
                act.setChecked(True)
            self.fft_group.addAction(act)

        # -- Window submenu --
        win_menu = spec_menu.addMenu("Window")
        self.win_group = QActionGroup(self)
        _WINDOW_MAP = [("Hann", "hann"), ("Hamming", "hamming"),
                       ("Blackman-Harris", "blackmanharris")]
        for label, key in _WINDOW_MAP:
            act = win_menu.addAction(label)
            act.setCheckable(True)
            act.setData(key)
            if key == "hann":
                act.setChecked(True)
            self.win_group.addAction(act)

        # -- Color Theme submenu --
        cmap_menu = spec_menu.addMenu("Color Theme")
        self.cmap_group = QActionGroup(self)
        for name in ("Magma", "Viridis", "Grayscale"):
            act = cmap_menu.addAction(name)
            act.setCheckable(True)
            act.setData(name.lower())
            if name == "Magma":
                act.setChecked(True)
            self.cmap_group.addAction(act)

        # -- dB Floor submenu --
        floor_menu = spec_menu.addMenu("dB Floor")
        self.floor_group = QActionGroup(self)
        for val in (-120, -100, -80, -60, -50, -40, -30, -20):
            act = floor_menu.addAction(f"{val} dB")
            act.setCheckable(True)
            act.setData(val)
            if val == -80:
                act.setChecked(True)
            self.floor_group.addAction(act)

        # -- dB Ceiling submenu --
        ceil_menu = spec_menu.addMenu("dB Ceiling")
        self.ceil_group = QActionGroup(self)
        for val in (-30, -20, -10, -5, 0):
            act = ceil_menu.addAction(f"{val} dB")
            act.setCheckable(True)
            act.setData(val)
            if val == 0:
                act.setChecked(True)
            self.ceil_group.addAction(act)

        self.spec_settings_btn.setMenu(spec_menu)
        self.spec_settings_btn.setAccessibleName("Spectrogram display options")
        toolbar.add_control(self.spec_settings_btn)
        toolbar.set_wanted(self.spec_settings_btn, False)

        # Waveform settings dropdown
        self.wf_settings_btn = QToolButton()
        self.wf_settings_btn.setText("Display")
        self.wf_settings_btn.setToolTip(
            "Configure waveform display parameters")
        self.wf_settings_btn.setPopupMode(QToolButton.InstantPopup)
        self.wf_settings_btn.setAutoRaise(True)
        self.wf_settings_btn.setStyleSheet(dropdown_style)
        wf_menu = QMenu(self.wf_settings_btn)

        # -- Anti-Aliased Lines toggle --
        self.wf_aa_action = wf_menu.addAction("Anti-Aliased Lines")
        self.wf_aa_action.setCheckable(True)
        self.wf_aa_action.setChecked(False)
        self.wf_aa_action.toggled.connect(self.waveform.set_wf_antialias)

        # -- Line Thickness submenu --
        thick_menu = wf_menu.addMenu("Line Thickness")
        self.wf_thick_group = QActionGroup(self)
        for label, val in [("Thin (1px)", 1), ("Normal (2px)", 2)]:
            act = thick_menu.addAction(label)
            act.setCheckable(True)
            act.setData(val)
            if val == 1:
                act.setChecked(True)
            self.wf_thick_group.addAction(act)
        self.wf_thick_group.triggered.connect(
            lambda a: self.waveform.set_wf_line_width(int(a.data())))

        self.wf_settings_btn.setMenu(wf_menu)
        self.wf_settings_btn.setAccessibleName("Waveform display options")
        toolbar.add_control(self.wf_settings_btn)

        # Overlay dropdown (analysis-only)
        self.overlay_btn = QToolButton()
        self.overlay_btn.setText("Detector Overlays")
        self.overlay_btn.setToolTip(
            "Select detector overlays to display on the waveform")
        self.overlay_btn.setPopupMode(QToolButton.InstantPopup)
        self.overlay_btn.setAutoRaise(True)
        self.overlay_btn.setStyleSheet(dropdown_style)
        self.overlay_menu = QMenu(self.overlay_btn)
        self.overlay_btn.setMenu(self.overlay_menu)
        self.overlay_btn.setAccessibleName("Detector overlays")
        toolbar.add_control(self.overlay_btn)

        # Markers toggle (analysis-only)
        self.markers_toggle = QToolButton()
        self.markers_toggle.setText("Peak / RMS Max")
        self.markers_toggle.setToolTip(
            "Toggle peak and maximum RMS markers on the waveform")
        self.markers_toggle.setCheckable(True)
        self.markers_toggle.setChecked(False)
        self.markers_toggle.setAutoRaise(True)
        self.markers_toggle.setStyleSheet(toggle_style)
        self.markers_toggle.toggled.connect(self.waveform.toggle_markers)
        self.markers_toggle.setAccessibleName("Peak / RMS Max markers")
        toolbar.add_control(self.markers_toggle)

        # RMS L/R toggle (analysis-only)
        self.rms_lr_toggle = QToolButton()
        self.rms_lr_toggle.setText("RMS L/R")
        self.rms_lr_toggle.setToolTip(
            "Toggle per-channel RMS envelope overlay")
        self.rms_lr_toggle.setCheckable(True)
        self.rms_lr_toggle.setAutoRaise(True)
        self.rms_lr_toggle.setStyleSheet(toggle_style)
        self.rms_lr_toggle.toggled.connect(self.waveform.toggle_rms_lr)
        self.rms_lr_toggle.setAccessibleName("RMS L/R envelopes")
        toolbar.add_control(self.rms_lr_toggle)

        # RMS AVG toggle (analysis-only)
        self.rms_avg_toggle = QToolButton()
        self.rms_avg_toggle.setText("RMS AVG")
        self.rms_avg_toggle.setToolTip(
            "Toggle combined (average) RMS envelope overlay")
        self.rms_avg_toggle.setCheckable(True)
        self.rms_avg_toggle.setAutoRaise(True)
        self.rms_avg_toggle.setStyleSheet(toggle_style)
        self.rms_avg_toggle.toggled.connect(self.waveform.toggle_rms_avg)
        self.rms_avg_toggle.setAccessibleName("RMS AVG envelope")
        toolbar.add_control(self.rms_avg_toggle)

        # Hide analysis-only controls when not in analysis mode
        if not self._analysis_mode:
            for control in self._analysis_controls():
                toolbar.set_wanted(control, False)
        # The header draws the background and the single bottom border.
        toolbar.setFixedHeight(32)
        return toolbar

    def _analysis_controls(self) -> tuple[QToolButton, ...]:
        return (self.overlay_btn, self.markers_toggle, self.rms_lr_toggle, self.rms_avg_toggle)

    # ------------------------------------------------------------------
    # Transport bar
    # ------------------------------------------------------------------

    _COMBO_MIN_WIDTH = 120
    _ICON_ONLY_WIDTH = 32
    _COMPACT_HYSTERESIS = 8

    def _build_transport(self) -> QWidget:
        transport = QWidget()
        transport.setObjectName("wfTransport")
        transport.installEventFilter(self)
        self._transport = transport
        self._transport_compact = False
        transport.setFixedHeight(32)
        transport.setStyleSheet(
            "#wfTransport { background-color: #2d2d2d;"
            " border-top: 1px solid #555; }")
        layout = QHBoxLayout(transport)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        _BTN_H = 24

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(_BTN_H)
        self.play_btn.setIcon(icon('play'))
        self.play_btn.setIconSize(QSize(16, 16))
        self.play_btn.setAccessibleName('Play')
        self.play_btn.setToolTip('Play')
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_clicked)
        layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(_BTN_H)
        self.stop_btn.setIcon(icon('stop'))
        self.stop_btn.setIconSize(QSize(16, 16))
        self.stop_btn.setAccessibleName('Stop')
        self.stop_btn.setToolTip('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked)
        layout.addWidget(self.stop_btn)

        # Play-mode combo (matches QComboBox style used in analysis toolbar)
        self._play_mode_combo = QComboBox()
        self._play_mode_combo.setFixedHeight(_BTN_H)
        self._play_mode_combo.setToolTip("Select playback channel routing")
        self._play_mode_combo.setMinimumWidth(self._COMBO_MIN_WIDTH)
        self._play_mode_combo.addItem("As-is", ("as_is", None))
        self._play_mode_combo.addItem("Mono", ("mono", None))
        self._play_mode_combo.currentIndexChanged.connect(
            self._on_play_mode_changed)
        layout.addWidget(self._play_mode_combo)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(
            "color: #888888; font-family: Consolas, monospace;"
            " font-size: 9pt; padding: 0 8px;")
        layout.addWidget(self.time_label)
        layout.addStretch()

        return transport

    def _transport_full_width(self) -> int:
        """Width the transport needs with labelled Play/Stop buttons."""
        layout = self._transport.layout()
        margins = layout.contentsMargins()
        buttons = sum(self._labelled_width(b) for b in (self.play_btn, self.stop_btn))
        widgets = sum(1 for i in range(layout.count()) if layout.itemAt(i).widget())
        return (margins.left() + margins.right() + layout.spacing() * max(0, widgets - 1) + buttons
                + self._COMBO_MIN_WIDTH + self.time_label.sizeHint().width())

    def _labelled_width(self, button: QPushButton) -> int:
        if not self._transport_compact:
            button._labelled_width = button.sizeHint().width()
        return getattr(button, "_labelled_width", button.sizeHint().width())

    def _update_transport_compact(self) -> None:
        """Play/Stop become icon-only when the transport would not fit."""
        full = self._transport_full_width()
        width = self._transport.width()
        if self._transport_compact:
            compact = width < full + self._COMPACT_HYSTERESIS
        else:
            compact = width < full
        if compact == self._transport_compact:
            return
        self._transport_compact = compact
        for button, label in ((self.play_btn, "Play"), (self.stop_btn, "Stop")):
            if compact:
                button.setText("")
                button.setFixedWidth(self._ICON_ONLY_WIDTH)
            else:
                button.setMinimumWidth(0)
                button.setMaximumWidth(16777215)
                button.setText(label)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_transport", None) and event.type() == QEvent.Resize:
            self._update_transport_compact()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Play-mode helpers
    # ------------------------------------------------------------------

    def _on_play_mode_changed(self, index: int):
        pass  # nothing extra needed; play_mode reads current data directly

    @property
    def play_mode(self) -> tuple[str, int | None]:
        """Return the current play mode as (mode_str, channel_or_None)."""
        data = self._play_mode_combo.currentData()
        if data is None:
            return ("as_is", None)
        return data

    # Microsoft WAV channel order labels for common layouts
    _CHANNEL_LABELS: dict[int, list[str]] = {
        1: ["M"],
        2: ["L", "R"],
        4: ["L", "R", "BL", "BR"],
        6: ["L", "R", "C", "LFE", "BL", "BR"],
        8: ["L", "R", "C", "LFE", "BL", "BR", "SL", "SR"],
    }

    @staticmethod
    def _channel_name(n_channels: int, ch: int) -> str:
        """Return a display label for channel *ch* in an *n_channels* file."""
        labels = WaveformPanel._CHANNEL_LABELS.get(n_channels)
        if labels and ch < len(labels):
            return f"Ch {ch} ({labels[ch]})"
        return f"Ch {ch}"

    def update_play_mode_channels(self, n_channels: int,
                                   labels: list[str] | None = None):
        """Rebuild per-channel entries in the play-mode combo.

        Parameters
        ----------
        n_channels : int
            Total number of display channels.
        labels : list[str] or None
            Custom per-channel labels (e.g. for multi-track stacked display).
            If *None*, uses standard Microsoft WAV channel labels.
        """
        # Remember current selection
        current = self.play_mode

        self._play_mode_combo.blockSignals(True)

        # Keep first two items (As-is, Mono), remove the rest
        while self._play_mode_combo.count() > 2:
            self._play_mode_combo.removeItem(self._play_mode_combo.count() - 1)

        if n_channels > 1:
            # QComboBox has no real separator — use a disabled item as divider
            self._play_mode_combo.addItem("─────")
            sep_idx = self._play_mode_combo.count() - 1
            model = self._play_mode_combo.model()
            model.item(sep_idx).setEnabled(False)

            for ch in range(n_channels):
                if labels and ch < len(labels):
                    name = f"Ch {ch} ({labels[ch]})"
                else:
                    name = self._channel_name(n_channels, ch)
                self._play_mode_combo.addItem(
                    f"{name} / As-is", ("channel_as_is", ch))
                self._play_mode_combo.addItem(
                    f"{name} / Mono", ("channel_mono", ch))

        # Try to restore previous selection
        restored = False
        for i in range(self._play_mode_combo.count()):
            if self._play_mode_combo.itemData(i) == current:
                self._play_mode_combo.setCurrentIndex(i)
                restored = True
                break
        if not restored:
            self._play_mode_combo.setCurrentIndex(0)

        self._play_mode_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Display mode (internal)
    # ------------------------------------------------------------------

    def _on_display_mode_changed(self, action: QAction):
        """Switch waveform display mode and toggle toolbar controls."""
        is_waveform = action == self.wf_action
        mode = "waveform" if is_waveform else "spectrogram"
        self.display_mode_btn.setText(action.text())
        self.waveform.set_display_mode(mode)

        # Hide waveform-only toolbar controls in spectrogram mode
        row = self._view_toolbar
        row.set_wanted(self.wf_settings_btn, is_waveform)
        if self._analysis_mode:
            row.set_wanted(self.markers_toggle, is_waveform)
            row.set_wanted(self.rms_lr_toggle, is_waveform)
            row.set_wanted(self.rms_avg_toggle, is_waveform)
        # Show spectrogram-only controls
        row.set_wanted(self.spec_settings_btn, not is_waveform)
        self.display_mode_changed.emit(mode)

    # ------------------------------------------------------------------
    # analysis_mode setter
    # ------------------------------------------------------------------

    def set_analysis_mode(self, enabled: bool):
        """Show or hide analysis-specific toolbar controls."""
        self._analysis_mode = enabled
        waveform_mode = self.wf_action.isChecked()
        for control in self._analysis_controls():
            # Waveform-only toggles stay hidden in spectrogram mode.
            wanted = enabled and (control is self.overlay_btn or waveform_mode)
            self._view_toolbar.set_wanted(control, wanted)
