"""Per-track repair choices and measurements for the File review page."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QSize, Qt, QSignalBlocker, Signal
from PySide6.QtGui import QFont, QTextOption
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QSizePolicy, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from sessionpreplib.config import REPAIR_PARAMS
from sessionpreplib.reports import format_boundary_settling
from sessionpreplib.repair_verification import CheckOutcome, detector_check, local_band_regions, local_rumble_check, source_low_band
from .repair_options import DETECTOR_REPAIRS, REPAIR_NAMES, track_repair_options
from .verification import (
    channel_names, compact_measurements, comparison_readings, measurement_note, measurement_table,
    subsonic_note, verification_status,
)
from ..controls import FormActionButton, FormDoubleSpinBox
from ..icons import icon
from ..prefs.param_form import _FieldResetButton, _spin_decimals
from ..theme import COLORS, REPORT_INSET


_CUTOFF = next(spec for spec in REPAIR_PARAMS if spec.key == "repair_low_cut_hz")


def _label(text="", *, muted=False):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    if muted:
        label.setStyleSheet(f"color:{COLORS['metadata']};")
    return label


def _control_pair(control, name):
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(control, 0, Qt.AlignVCenter)
    reset = _FieldResetButton(control)
    reset.setToolTip(f"Reset {name} to preset")
    reset.setAccessibleName(reset.toolTip())
    layout.addWidget(reset, 0, Qt.AlignVCenter)
    layout.addStretch()
    return box, reset


def _number(value, places, *, sign=False):
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "—"
    if math.isinf(value):
        return "−∞" if value < 0 else "+∞"
    return format(value, f"{'+' if sign else ''}.{places}f").replace("-", "−")


def _qualifying_window_seconds(regions, rate):
    """Display elapsed time once when channels report overlapping locations."""
    end = -1
    samples = 0
    for first, last, _channel in sorted(regions):
        samples += max(0, last - max(first, end + 1) + 1)
        end = max(end, last)
    return samples / rate


def _region_count(regions):
    return f"{len(regions)} {'region' if len(regions) == 1 else 'regions'}"


def _reading(value, places, units):
    number = _number(value, places)
    return number + units if number != "—" else number


def _local_readings(track):
    """Each channel list describes the regions found on that signal alone."""
    bands = (source_low_band(track), source_low_band(track, corrected=True) if track.repair_recipe.active else None)
    if not any(isinstance(band, dict) and band.get("regions") for band in bands):
        return []
    levels, counts, names = [], [], []
    for band in bands:
        regions = local_band_regions(band, track)
        level = band.get("band_dbfs") if isinstance(band, dict) else None
        levels.append(_reading(level, 1, " dBFS RMS"))
        counts.append(str(len(regions)) if regions is not None else "—")
        names.append(channel_names(sorted({ch for _, _, ch in regions}), track.channels)
                     if regions is not None else "—")
    return [("Band level", *levels), ("Regions", *counts), ("Affected channels", *names)]


class _CutoffSpinBox(FormDoubleSpinBox):
    def textFromValue(self, value):
        text = super().textFromValue(value)
        return text.rstrip("0").rstrip(self.locale().decimalPoint()) if self.locale().decimalPoint() in text else text


def _disclosure(text):
    button = QToolButton()
    button.setText(text)
    button.setCheckable(True)
    button.setAutoRaise(True)
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setFocusPolicy(Qt.StrongFocus)
    button.setIconSize(QSize(16, 16))
    button.setIcon(icon("chevron-right"))
    button.toggled.connect(lambda opened: button.setIcon(icon("chevron-down" if opened else "chevron-right")))
    return button


class _RepairRow(QFrame):
    def __init__(self, key, title, on_change):
        super().__init__()
        self.setObjectName("trackRepairRow")
        self.toggle = _disclosure(title)
        font = QFont(self.toggle.font())
        font.setWeight(QFont.DemiBold)
        self.toggle.setFont(font)
        self.description = _label(muted=True)
        self.state = _label(muted=True)
        self.verdict = _label()
        self.verdict.setFont(font)
        self.verdict.hide()
        self.state.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.choice = QComboBox()
        self.choice.setAccessibleName(title)
        for text, value in (("Use preset (Off)", None), ("On", True), ("Off", False)):
            self.choice.addItem(text, value)
        self.choice.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.choice.setFocusPolicy(Qt.StrongFocus)
        self.controls, self.reset = _control_pair(self.choice, title)
        self.choice.currentIndexChanged.connect(lambda _: on_change(key, self.choice.currentData()))
        self.reset.clicked.connect(lambda: on_change(key, None))

        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 6, 0, 6)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.band = QFrame()
        self.band.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.band.setStyleSheet(f"background:{COLORS['surface']}; border:0;")
        self.grid.addWidget(self.band, 0, 0, 1, 4)
        self.grid.addWidget(self.toggle, 0, 0, Qt.AlignLeft)
        self.toggle.setStyleSheet(
            f"QToolButton {{background:transparent; color:{COLORS['information']}; border:1px solid transparent;}}"
            f"QToolButton:checked {{color:{COLORS['text']};}}"
            f"QToolButton:disabled {{color:{COLORS['metadata']};}}"
            f"QToolButton:hover {{background:{COLORS['accent']};}}"
            f"QToolButton:focus {{border-color:{COLORS['information']};}}")
        self.details = QWidget()
        self.fields = QVBoxLayout(self.details)
        self.fields.setSpacing(10)
        self.evidence = _label()
        self.evidence.setTextFormat(Qt.RichText)
        self.evidence.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.evidence.setMaximumWidth(640)
        self.evidence.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.fields.addWidget(self.evidence)
        self.grid.addWidget(self.details, 4, 0, 1, 4)
        self.editable = True
        self.unavailable = ""
        self._narrow = False
        self._available_width = 0
        self._shape = None
        self._wide_evidence = self._compact_evidence = ""
        self.update_disclosure()

    def set_evidence(self, wide, compact):
        self._wide_evidence, self._compact_evidence = wide, compact
        self.evidence.setText(compact if self._narrow else wide)
        self.evidence.setVisible(bool(wide))

    def arrange(self, narrow, available_width):
        if self._narrow != narrow:
            self.evidence.setText(self._compact_evidence if narrow else self._wide_evidence)
        self._narrow = narrow
        self._available_width = available_width
        self.update_disclosure()

    def update_disclosure(self):
        opened = self.toggle.isChecked()
        title_font = QFont(self.font())
        if opened and self._narrow:
            title_font.setPointSizeF(title_font.pointSizeF() * 1.3)
            title_font.setWeight(QFont.Bold)
        else:
            title_font.setWeight(QFont.DemiBold)
        if self.toggle.font() != title_font:
            self.toggle.setFont(title_font)
        self.state.setText(self.choice.currentText() if self.editable else
                           self.unavailable if self._narrow else "Unavailable")
        gap = self.grid.horizontalSpacing()
        title_width = self.toggle.sizeHint().width()
        verdict_width = self.verdict.fontMetrics().horizontalAdvance(self.verdict.text())
        state_width = self.state.fontMetrics().horizontalAdvance(self.state.text())
        has_verdict = bool(self.verdict.text())
        wrap_verdict = self._narrow and has_verdict and title_width + gap + verdict_width > self._available_width
        header_width = title_width + (gap + verdict_width if has_verdict and not wrap_verdict else 0)
        wrap_state = self._narrow and not opened and header_width + gap + state_width > self._available_width
        self.grid.setColumnMinimumWidth(0, 0 if self._narrow else min(214, self._available_width // 5))
        shape = self._narrow, opened, has_verdict, wrap_state, wrap_verdict
        if self._shape != shape:
            self._shape = shape
            for widget in (self.description, self.state, self.controls, self.verdict):
                self.grid.removeWidget(widget)
            self.grid.setColumnStretch(2, 1)
            if wrap_verdict:
                self.grid.addWidget(self.verdict, 1, 0, 1, 4)
            else:
                self.grid.addWidget(self.verdict, 0, 1)
            if wrap_state:
                self.grid.addWidget(self.state, 2 if wrap_verdict else 1, 0, 1, 4)
            else:
                self.grid.addWidget(self.state, 0, 3)
            self.state.setWordWrap(self._narrow)
            self.state.setAlignment((Qt.AlignLeft if wrap_state else Qt.AlignRight) | Qt.AlignVCenter)
            if opened:
                self.grid.addWidget(self.description, 2 if wrap_verdict else 1, 0, 1, 4)
            elif has_verdict:
                self.grid.addWidget(self.description, 0, 2)
            else:
                self.grid.addWidget(self.description, 0, 1, 1, 2)
            if self._narrow:
                self.grid.addWidget(self.controls, 3, 0, 1, 4, Qt.AlignLeft)
            else:
                self.grid.addWidget(self.controls, 0, 3, Qt.AlignRight)
            indent = 0 if self._narrow else 38
            self.fields.setContentsMargins(indent, 6, 0, 8)
            self.grid.setRowMinimumHeight(0, 0 if self._narrow else 38)
            self.description.setIndent(indent if opened else 0)
        self.verdict.setVisible(has_verdict)
        self.description.setVisible(bool(self.description.text()) and (opened or not self._narrow))
        self.band.setVisible(opened and not self._narrow)
        self.controls.setVisible(opened and self.editable)
        self.state.setVisible(not opened)
        self.details.setVisible(opened)


class TrackRepairEditor(QFrame):
    choiceChanged = Signal(object, str, object)
    adjustGainRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.track = self.session = None
        self._opening_finding = None
        self._narrow = False
        self.setObjectName("trackRepairEditor")
        self.setAccessibleName("Track repairs")
        self.setFont(QApplication.font())
        self.setStyleSheet(f"QFrame#trackRepairRow {{border:0; border-top:1px solid {COLORS['accent']};}}")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(REPORT_INSET, REPORT_INSET, REPORT_INSET, REPORT_INSET)
        self.body.setSpacing(6)
        self._build_header_and_rows()
        self._build_cutoff_and_peak()
        self._build_recovery_section()
        self._wire_tab_order_and_events()

    def _build_header_and_rows(self) -> None:
        header = QHBoxLayout()
        self.back_button = FormActionButton("Back to findings")
        self.back_button.setToolTip("Back to findings (Alt+Left)")
        header.addWidget(self.back_button, 0, Qt.AlignLeft | Qt.AlignTop)
        self.filename = _label(muted=True)
        self.filename.setWordWrap(False)
        self.filename.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self.filename, 1)
        self.body.addLayout(header)
        self.title = _label("Track repairs")
        self.title.setWordWrap(False)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        font = QFont(self.title.font())
        font.setWeight(QFont.Bold)
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.title.setFont(font)
        self.body.addWidget(self.title)
        self.body.addWidget(_label("Whole track · before normalization", muted=True))
        self.empty = _label("No repairable findings.", muted=True)
        self.body.addWidget(self.empty)
        self.rows = {}
        for key, name in REPAIR_NAMES.items():
            row = _RepairRow(key, name, self._change)
            row.toggle.clicked.connect(lambda opened, key=key: self.select_repair(key if opened else None))
            self.rows[key] = row
            self.body.addWidget(row)

    def _build_cutoff_and_peak(self) -> None:
        self.cutoff = _CutoffSpinBox()
        self.cutoff.setRange(_CUTOFF.min, _CUTOFF.max)
        self.cutoff.setSingleStep(1)
        self.cutoff.setKeyboardTracking(False)
        self.cutoff.setAccessibleName("Cutoff (Hz)")
        cutoff_pair, self.cutoff_reset = _control_pair(self.cutoff, "cutoff")
        self.cutoff.valueChanged.connect(lambda value: self._change("low_cut_hz", value))
        self.cutoff_reset.clicked.connect(lambda: self._change("low_cut_hz", None))
        self.cutoff_box = QWidget()
        cutoff_layout = QGridLayout(self.cutoff_box)
        cutoff_layout.setContentsMargins(0, 3, 0, 3)
        cutoff_layout.setSpacing(6)
        cutoff_layout.addWidget(_label("Cutoff (Hz)"), 0, 0)
        cutoff_layout.addWidget(cutoff_pair, 0, 1)
        cutoff_layout.setColumnStretch(1, 1)
        self.cutoff_default = _label(muted=True)
        cutoff_layout.addWidget(self.cutoff_default, 1, 0, 1, 2)
        self.rows["low_cut"].fields.insertWidget(0, self.cutoff_box)
        self.peak_name = _label("Sample peak before gain", muted=True)
        self.peak_name.setWordWrap(False)
        self.peak = _label()
        self.peak_box = QWidget()
        self.peak_layout = QGridLayout(self.peak_box)
        self.peak_layout.setContentsMargins(0, 6, 0, 6)
        self.peak_layout.setColumnStretch(1, 1)
        self.peak_layout.addWidget(self.peak_name, 0, 0)
        self.peak_layout.addWidget(self.peak, 0, 1)
        self.body.addWidget(self.peak_box)

    def _build_recovery_section(self) -> None:
        self.boundary = _disclosure("Boundary settling")
        self.body.addWidget(self.boundary, 0, Qt.AlignLeft)
        self.boundary_details = _label()
        self.boundary_details.setStyleSheet(f"color:{COLORS['attention']};")
        self.body.addWidget(self.boundary_details)
        self.boundary.toggled.connect(self.boundary_details.setVisible)
        self.recovery_title = _label("Output gain needs adjustment")
        title_font = QFont(self.recovery_title.font())
        title_font.setWeight(QFont.DemiBold)
        self.recovery_title.setFont(title_font)
        self.body.addWidget(self.recovery_title)
        self.error = QTextBrowser()
        self.error.setAccessibleName("Repair error")
        self.error.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.error.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.error.setFocusPolicy(Qt.ClickFocus)
        self.error.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.error.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.error.setStyleSheet(f"color:{COLORS['attention']}; background:transparent; border:0; padding:0;")
        self.error.document().setDocumentMargin(0)
        self.error.document().documentLayout().documentSizeChanged.connect(
            lambda size: self.error.setFixedHeight(math.ceil(size.height()))
        )
        self.body.addWidget(self.error)
        self.recovery = _label()
        self.body.addWidget(self.recovery)
        self.recovery_detail = _label(muted=True)
        self.body.addWidget(self.recovery_detail)
        self.adjust_gain = FormActionButton("Adjust Gain")
        self.adjust_gain.clicked.connect(lambda: self.adjustGainRequested.emit(self.track))
        self.body.addWidget(self.adjust_gain, 0, Qt.AlignLeft)
        self.body.addStretch()

    def _wire_tab_order_and_events(self) -> None:
        order = [self.back_button]
        for key, row in self.rows.items():
            order.extend((row.toggle, row.choice, row.reset))
            if key == "low_cut":
                order.extend((self.cutoff, self.cutoff_reset))
        order.extend((self.boundary, self.adjust_gain))
        for first, second in zip(order, order[1:]):
            QWidget.setTabOrder(first, second)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Space:
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _change(self, key, value):
        if self.track is not None:
            self.choiceChanged.emit(self.track, key, value)

    def select_repair(self, key):
        for candidate, row in self.rows.items():
            row.toggle.setChecked(candidate == key)
            row.update_disclosure()
        self._place_peak()

    def _place_peak(self):
        # In the compact view the before-gain peak follows the open repair.
        rows = list(self.rows.values())
        row = next((r for r in rows if self._narrow and r.toggle.isChecked()), rows[-1])
        pos = self.body.indexOf(row)
        current = self.body.indexOf(self.peak_box)
        if current != pos + 1:
            self.body.removeWidget(self.peak_box)
            self.body.insertWidget(self.body.indexOf(row) + 1, self.peak_box)

    def show_track(self, track, session, detector, options=None):
        self.track, self.session = track, session
        self._opening_finding = detector
        self.boundary.setChecked(False)
        options = options or track_repair_options(track, session)
        first = next((key for key in self.rows if key in options.rows), None)
        self.select_repair(options.targets.get(detector, first))
        self.refresh(track, session, options)

    def _sync_visible_rows_and_focus(self, options) -> None:
        focus = QApplication.focusWidget()
        previous = next((key for key, row in self.rows.items() if row.toggle.isChecked()), None)
        focus_removed = focus is not None and any(
            key not in options.rows and row.isAncestorOf(focus) for key, row in self.rows.items()
        )
        for key, row in self.rows.items():
            row.setVisible(key in options.rows)
        if previous is not None and previous not in options.rows:
            self.select_repair(next((key for key in self.rows if key in options.rows), None))
        if focus_removed:
            target = next((row.toggle for key, row in self.rows.items() if key in options.rows), self.back_button)
            target.setFocus(Qt.OtherFocusReason)
        self.empty.setVisible(not options.available)

    def _update_row_verdicts_and_choices(
        self, track, options, checks, low_check, selected, inherited, overrides, pending
    ) -> None:
        for key, row in self.rows.items():
            check = low_check if key == "low_cut" else checks[key]
            status = verification_status(check.outcome)
            if check.outcome in {CheckOutcome.REMAINING, CheckOutcome.NEW}:
                status = (
                    "Still detected" if check.outcome == CheckOutcome.REMAINING else "New after repair",
                    COLORS["attention"] if check.corrected.severity.value != "info" else COLORS["information"],
                )
            show_status = bool(
                status
                and (check.original or check.corrected or pending)
                and (key != "ultrasonic" or track.samplerate >= 50000)
            )
            row.verdict.setText(status[0] if show_status else "")
            if show_status:
                row.verdict.setStyleSheet(f"color:{status[1]};")
            value = overrides.get(key)
            with QSignalBlocker(row.choice):
                row.choice.setItemText(
                    0, f"Use preset ({'On' if inherited.get(key, selected[key] and value is None) else 'Off'})"
                )
                row.choice.setCurrentIndex(0 if value is None else 1 if value else 2)
            enabled = key != "ultrasonic" or track.samplerate >= 50000
            row.editable = enabled or value is not None or selected[key]
            row.toggle.setEnabled(row.editable)
            row.unavailable = f"Unavailable at {track.samplerate / 1000:g} kHz" if not enabled else ""
            row.choice.model().item(1).setEnabled(enabled)
            row.reset.setEnabled(key in overrides)

    def _update_repair_sections(self, track, options, before, after, suggested, selected, config, overrides, checks) -> None:
        dc = self.rows["dc_remove"]
        dc.description.setText(
            "Suggested · 2 Hz high-pass"
            if options.targets.get("dc_offset") == "dc_remove" and not track.repair_recipe.dc_remove
            else "2 Hz high-pass"
        )
        dc_after = after.get("signed_dc", [])
        dc_values = [
            (
                ("Left", "Right")[i] if track.channels == 2 and i < 2 else f"Ch {i + 1}",
                _reading(mean * 100, 2, "%"),
                _reading(dc_after[i] * 100, 2, "%") if i < len(dc_after) else "—",
            )
            for i, mean in enumerate(before.get("signed_dc", []))
        ]
        dc_table = measurement_table(dc_values, heading="DC mean (% FS)", editor=True) if dc_values else ""
        dc.set_evidence(dc_table, dc_table)

        low = self.rows["low_cut"]
        low.description.setText(
            "Whole-track high-pass"
            if selected["low_cut"]
            else "Suggested"
            if "low_cut" in options.targets.values()
            else options.attribution("subsonic")
        )
        self._set_low_evidence(before, after, checks["low_cut"])
        preset = config.get(_CUTOFF.key, _CUTOFF.default)
        cutoff_value = overrides.get("low_cut_hz", preset)
        with QSignalBlocker(self.cutoff):
            self.cutoff.setDecimals(_spin_decimals(_CUTOFF, cutoff_value))
            self.cutoff.setValue(cutoff_value)
        self.cutoff_reset.setEnabled("low_cut_hz" in overrides)
        self.cutoff_default.setText(f"Preset: {preset:g} Hz")
        self.cutoff_box.setVisible(
            selected["low_cut"] or overrides.get("low_cut") is True or "low_cut_hz" in overrides
        )

        high = self.rows["ultrasonic"]
        original_high = before.get("ultrasonic", {})
        if track.samplerate < 50000:
            high.description.setText(f"Requires at least 50 kHz · source is {track.samplerate / 1000:g} kHz")
        elif suggested.get("ultrasonic"):
            duration = _qualifying_window_seconds(original_high.get("regions", []), track.samplerate)
            high.description.setText(f"Suggested · Qualifying windows span {duration:.2f} s")
        else:
            high.description.setText("Audible passband through 20 kHz")
        if original_high.get("eligible"):
            high_values = [
                (
                    "Energy ≥24.1 kHz",
                    _reading(original_high.get("band_dbfs"), 1, " dBFS RMS"),
                    _reading(after.get("ultrasonic", {}).get("band_dbfs"), 1, " dBFS RMS"),
                )
            ]
            high.set_evidence(measurement_table(high_values, editor=True), compact_measurements(high_values))
        else:
            high.set_evidence("", "")

    def _update_peak_and_recovery(self, track, options, measures, before, after, config, unavailable) -> None:
        recipe = track.repair_recipe

        def peak(data):
            value = data.get("peak_dbfs")
            return _number(value, 2 if value is not None and abs(value) < 1 else 1, sign=True)

        self.peak.setText(f"{peak(before)} → {peak(after)} dBFS")
        self.peak_box.setVisible(recipe.active and "peak_dbfs" in before)
        errors = options.errors
        adjustment = options.gain_adjustment
        self.recovery_title.setVisible(bool(adjustment))
        details = []
        if adjustment:
            errors = [msg for msg in errors if not msg.startswith("Output peak ")]
            errors.insert(0, f"Output peak {_number(adjustment['peak_db'], 2, sign=True)} dBFS")
            self.recovery.setText(
                f"Lower output Gain by at least {adjustment['reduction_db']:.1f} dB, then retry Prepare."
            )
            from sessionpreplib.audio import INACTIVE_NORMALIZATION_CLASSIFICATIONS

            norm = track.processor_results.get("bimodal_normalize")
            if (
                not config.get("bimodal_normalize_enabled", True)
                or "bimodal_normalize" in track.processor_skip
                or norm is None
                or norm.classification in INACTIVE_NORMALIZATION_CLASSIFICATIONS
            ):
                details.append("With normalization bypassed, Adjust Gain opens a manual output gain for this track.")
            if options.has_stale_output:
                details.append("The previous prepared output is marked stale.")
        self.recovery_detail.setText(" ".join(details))
        self.recovery_detail.setVisible(bool(details))
        regions = measures.get("boundary_settling", []) if recipe.active and not unavailable else []
        self.boundary.setVisible(bool(regions))
        self.boundary.setText("Boundary settling · " + _region_count(regions))
        self.boundary_details.setText("\n".join(format_boundary_settling(regions)))
        self.boundary_details.setVisible(bool(regions) and self.boundary.isChecked())
        self.error.setPlainText("\n".join(errors))
        self.error.setVisible(bool(errors))
        self.recovery.setVisible(bool(adjustment))
        self.adjust_gain.setVisible(bool(adjustment))

    def refresh(self, track, session, options=None):
        if track is not self.track or session is not self.session:
            return
        options = options or track_repair_options(track, session)
        self._sync_visible_rows_and_focus(options)
        measures = track.repair_measurements
        before, after = measures.get("original", {}), measures.get("corrected", {})
        suggested, inherited = measures.get("suggested", {}), measures.get("inherited", {})
        config, overrides = session.config, track.repair_overrides
        recipe = track.repair_recipe
        selected = {
            "dc_remove": recipe.dc_remove,
            "low_cut": recipe.low_cut_hz is not None,
            "ultrasonic": recipe.ultrasonic,
        }
        pending = bool(track._cache.get("repair_verification_pending"))
        unavailable = pending or bool(track._cache.get("repair_error"))
        if unavailable or not recipe.active:
            after = {}
        checks = {
            key: detector_check(track, detector_id, pending=pending)
            for detector_id, key in DETECTOR_REPAIRS.items()
            if detector_id != "subsonic-local"
        }
        whole = checks["low_cut"]
        low_check = whole
        if options.targets.get("subsonic-local") == "low_cut":
            local = local_rumble_check(track, config, whole)
            outstanding = {CheckOutcome.NEW, CheckOutcome.REMAINING}
            if (
                self._opening_finding == "subsonic-local"
                or options.targets.get("subsonic") != "low_cut"
                or (
                    self._opening_finding != "subsonic"
                    and local.outcome in outstanding
                    and whole.outcome not in outstanding
                )
            ):
                low_check = local

        self._update_row_verdicts_and_choices(
            track, options, checks, low_check, selected, inherited, overrides, pending
        )
        self._update_repair_sections(
            track, options, before, after, suggested, selected, config, overrides, checks
        )
        for row in self.rows.values():
            row.update_disclosure()
        self._update_peak_and_recovery(track, options, measures, before, after, config, unavailable)

    def _set_low_evidence(self, before, after, check):
        config, track = self.session.config, self.track
        frequency = config.get("subsonic_hz", 30)
        rows = []
        note = subsonic_note(config)
        if check.corrected is not None:
            verified, note, _, _ = comparison_readings(check, track, config)
            rows = [entry for entry in verified if entry[0] not in ("Locations", "Result")]
        if not rows and any("subsonic_ratio_db" in side for side in (before, after)):
            rows = [(f"Energy at or below {frequency:g} Hz",
                     *(_reading(side.get("subsonic_ratio_db"), 1, " dB") for side in (before, after)))]
        wide = compact = ""
        if rows:
            wide = measurement_table(rows, editor=True) + measurement_note(note)
            compact = compact_measurements(rows) + measurement_note(note)
        local = _local_readings(track)
        if local:
            band = f"Local energy ≤{min(frequency, 20):g} Hz"
            wide += measurement_table(local, heading=band, editor=True)
            compact += measurement_table(local, heading=band, editor=True)
        self.rows["low_cut"].set_evidence(wide, compact)

    def set_viewport_width(self, width):
        # Use the scroll viewport's available space. A child's old minimum
        # size can exceed it precisely when a compact row needs to wrap.
        narrow = width < 590
        available = max(0, width - 2 * REPORT_INSET)
        name = self.track.filename if self.track is not None else ""
        label = self.title if narrow else self.filename
        name_width = available if narrow else max(0, available - self.back_button.sizeHint().width() - 12)
        label.setText(label.fontMetrics().elidedText(name, Qt.ElideMiddle, name_width))
        label.setAccessibleName(name)
        label.setToolTip(name)
        if not narrow:
            self.title.setText("Track repairs")
            self.title.setAccessibleName("Track repairs")
            self.title.setToolTip("")
        for row in self.rows.values():
            row.arrange(narrow, available)
        if narrow == self._narrow:
            return
        self._narrow = narrow
        self.filename.setVisible(not narrow)
        self.peak_layout.removeWidget(self.peak)
        self.peak_layout.addWidget(self.peak, int(narrow), int(not narrow))
        self._place_peak()
