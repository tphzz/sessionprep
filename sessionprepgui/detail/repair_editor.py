"""Native track controls at the end of the independently scrolling File report."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)

from sessionpreplib.config import REPAIR_PARAMS
from sessionpreplib.reports import format_boundary_settling
from ..controls import FormActionButton, FormDoubleSpinBox
from ..prefs.param_form import _FieldResetButton, _spin_decimals
from ..theme import COLORS, REPORT_INSET


_DETECTORS = {"dc_offset": "dc_remove", "subsonic": "low_cut", "subsonic-local": "low_cut",
              "ultrasonic": "ultrasonic"}
_REPAIRS = {"dc_remove": "DC removal", "low_cut": "Subsonic filter", "ultrasonic": "Ultrasonic filter"}
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
    for first, last, channel in sorted(regions):
        samples += max(0, last - max(first, end + 1) + 1)
        end = max(end, last)
    return samples / rate


def _region_count(regions):
    return f"{len(regions)} {'region' if len(regions) == 1 else 'regions'}"


class _CutoffSpinBox(FormDoubleSpinBox):
    def textFromValue(self, value):
        text = super().textFromValue(value)
        return text.rstrip("0").rstrip(self.locale().decimalPoint()) if self.locale().decimalPoint() in text else text


class _RepairRow(QFrame):
    def __init__(self, key, title, on_change):
        super().__init__()
        self.setObjectName("trackRepairRow")
        self.name = _label(title)
        self.description = _label()
        self.evidence_title = _label(muted=True)
        self.evidence = _label()
        self.choice = QComboBox()
        self.choice.setAccessibleName(title)
        for text, value in (("Use preset (Off)", None), ("On", True), ("Off", False)):
            self.choice.addItem(text, value)
        self.choice.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.choice.setFocusPolicy(Qt.StrongFocus)
        self.controls, self.reset = _control_pair(self.choice, title)
        self.choice.currentIndexChanged.connect(lambda _: on_change(key, self.choice.currentData()))
        self.reset.clicked.connect(lambda: on_change(key, None))
        self.cutoff_box = None
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 8, 0, 8)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(5)

    def arrange(self, narrow):
        fields = [self.name, self.description, self.controls, self.evidence_title, self.evidence]
        if self.cutoff_box is not None:
            fields.insert(3, self.cutoff_box)
        for widget in fields:
            self.grid.removeWidget(widget)
        self.grid.setColumnMinimumWidth(0, 0 if narrow else 164)
        self.grid.setVerticalSpacing(8 if narrow else 5)
        label_width = self.choice.fontMetrics().horizontalAdvance("Use preset (Off)")
        self.choice.setFixedWidth(max(143 if narrow else 150, label_width + 40))
        self.grid.setColumnStretch(0, int(narrow))
        self.grid.setColumnStretch(1, int(not narrow))
        if narrow:
            for index, widget in enumerate(fields):
                self.grid.addWidget(widget, index, 0)
        else:
            for column, widget in enumerate(fields[:3]):
                self.grid.addWidget(widget, 0, column, Qt.AlignVCenter)
            if self.cutoff_box is not None:
                self.grid.addWidget(self.cutoff_box, 1, 1, 1, 2)
            self.grid.addWidget(self.evidence_title, 2, 0)
            self.grid.addWidget(self.evidence, 2, 1, 1, 2)


class TrackRepairEditor(QFrame):
    choiceChanged = Signal(object, str, object)
    adjustGainRequested = Signal(object)
    processingFocusRequested = Signal(str)

    def __init__(self, report):
        super().__init__(report.viewport())
        self.report = report
        self.track = self.session = self.detector = None
        self._low_narrow = self._low_wide = ""
        self._narrow = None
        self._laying_out = False
        self.setObjectName("trackRepairEditor")
        self.setAccessibleName("Track repairs")
        self.setFont(QApplication.font())
        self.setStyleSheet(f"QFrame#trackRepairEditor {{background:{COLORS['bg_alt']};}}"
                           f"QFrame#trackRepairRow {{border:0; border-top:1px solid {COLORS['accent']};}}")
        self.body = QVBoxLayout(self)
        self.body.setSpacing(4)
        header = QHBoxLayout()
        title = _label("Track repairs")
        font = QFont(title.font())
        font.setWeight(QFont.Bold)
        font.setPointSizeF(font.pointSizeF() * 1.25)
        title.setFont(font)
        header.addWidget(title, 1)
        self.close_button = FormActionButton("Close")
        self.close_button.setAccessibleName("Close track repairs")
        self.close_button.setToolTip("Close track repairs (Esc)")
        self.close_button.setMinimumWidth(64)
        self.close_button.clicked.connect(self.close_editor)
        header.addWidget(self.close_button)
        self.header = header
        self.body.addLayout(header)
        self.scope = _label(muted=True)
        self.body.addWidget(self.scope)
        self.rows = {}
        for key, name in _REPAIRS.items():
            row = _RepairRow(key, name, self._change)
            self.rows[key] = row
            self.body.addWidget(row)
        self.cutoff = _CutoffSpinBox()
        self.cutoff.setRange(_CUTOFF.min, _CUTOFF.max)
        self.cutoff.setSingleStep(1)
        self.cutoff.setKeyboardTracking(False)
        self.cutoff.setAccessibleName("Cutoff (Hz)")
        cutoff_pair, self.cutoff_reset = _control_pair(self.cutoff, "cutoff")
        self.cutoff.valueChanged.connect(lambda value: self._change("low_cut_hz", value))
        self.cutoff_reset.clicked.connect(lambda: self._change("low_cut_hz", None))
        self.cutoff_box = QWidget()
        cutoff_layout = QVBoxLayout(self.cutoff_box)
        cutoff_layout.setContentsMargins(0, 3, 0, 3)
        cutoff_layout.setSpacing(3)
        cutoff_layout.addWidget(_label("Cutoff (Hz)"))
        cutoff_layout.addWidget(cutoff_pair)
        self.cutoff_default = _label(muted=True)
        cutoff_layout.addWidget(self.cutoff_default)
        self.rows["low_cut"].cutoff_box = self.cutoff_box
        for row in self.rows.values():
            row.arrange(False)
        self.peak_name = _label("Sample peak", muted=True)
        self.peak = _label()
        self.peak_layout = QGridLayout()
        self.peak_layout.addWidget(self.peak_name, 0, 0)
        self.peak_layout.addWidget(self.peak, 0, 1)
        self.body.addLayout(self.peak_layout)
        self.recovery_title = _label("Output gain needs adjustment")
        title_font = QFont(self.recovery_title.font())
        title_font.setWeight(QFont.DemiBold)
        self.recovery_title.setFont(title_font)
        self.body.addWidget(self.recovery_title)
        self.error = _label()
        self.error.setStyleSheet(f"color:{COLORS['attention']};")
        self.body.addWidget(self.error)
        self.recovery = _label()
        self.body.addWidget(self.recovery)
        self.recovery_detail = _label(muted=True)
        self.body.addWidget(self.recovery_detail)
        self.adjust_gain = FormActionButton("Adjust Gain")
        self.adjust_gain.clicked.connect(lambda: self.adjustGainRequested.emit(self.track))
        self.body.addWidget(self.adjust_gain, 0, Qt.AlignLeft)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.timeout.connect(self.reflow)
        report.viewport().installEventFilter(self)
        self.hide()
        report.reportChanged.connect(self._context_changed)
        report.reportLayoutChanged.connect(self.reflow)
        report.verticalScrollBar().valueChanged.connect(self._position)
        QApplication.instance().focusChanged.connect(self._keep_focus_visible)

    def eventFilter(self, watched, event):
        if watched is self.report.viewport():
            if event.type() == QEvent.Resize and self.track is not None:
                self._viewport_timer.start(0)
            return super().eventFilter(watched, event)
        if event.type() in (QEvent.ShortcutOverride, QEvent.KeyPress):
            if event.key() == Qt.Key_Escape and QApplication.activePopupWidget() is None:
                event.accept()
                if event.type() == QEvent.KeyPress:
                    self.close_editor()
                return True
            if event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Space:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _change(self, key, value):
        if self.track is not None:
            self.choiceChanged.emit(self.track, key, value)

    def open_track(self, track, session, detector=None):
        if session is not self.report.report_session or track.filename != self.report.track_filename:
            return
        self.track, self.session, self.detector = track, session, detector
        self._narrow = None
        self.show()
        self.refresh(track, session)
        row = self.rows[_DETECTORS.get(detector, "dc_remove")]
        focus = row.choice if row.choice.isVisible() and row.choice.isEnabled() else self.close_button
        bar = self.report.verticalScrollBar()
        bar.setValue(bar.value() + max(0, self.y() - 16))
        focus.setFocus(Qt.OtherFocusReason)
        self._keep_focus_visible(None, focus)

    def _context_changed(self):
        if self.track is None:
            return
        if self.session is not self.report.report_session or self.track.filename != self.report.track_filename:
            self.close_editor(restore_focus=False)

    def close_editor(self, *, restore_focus=True):
        if self.track is None:
            return
        filename, detector = self.track.filename, self.detector
        self.track = self.session = self.detector = None
        self.hide()
        self.reflow()
        if restore_focus:
            if detector is None or self.report._anchor_cursor(href="report-repair:" + detector) is None:
                self.processingFocusRequested.emit(filename)
            else:
                self.report.focus_report_link("report-repair", detector)
                self.report.ensureCursorVisible()

    def refresh(self, track, session):
        if self.track is None or session is not self.session or self.track.filename != track.filename:
            return
        self.track = track
        measures = track.repair_measurements
        before, after = measures.get("original", {}), measures.get("corrected", {})
        suggested, inherited = measures.get("suggested", {}), measures.get("inherited", {})
        config, overrides = session.config, track.repair_overrides
        recipe = track.repair_recipe
        selected = {"dc_remove": recipe.dc_remove, "low_cut": recipe.low_cut_hz is not None, "ultrasonic": recipe.ultrasonic}
        for key, row in self.rows.items():
            value = overrides.get(key)
            with QSignalBlocker(row.choice):
                row.choice.setItemText(0, f"Use preset ({'On' if inherited.get(key, selected[key] and value is None) else 'Off'})")
                row.choice.setCurrentIndex(0 if value is None else 1 if value else 2)
            enabled = key != "ultrasonic" or track.samplerate >= 50000
            row.controls.setVisible(enabled or value is not None)
            row.choice.setVisible(enabled or value is not None)
            row.choice.model().item(1).setEnabled(enabled)
            row.reset.setEnabled(key in overrides)
        dc = self.rows["dc_remove"]
        dc.description.setText("Suggested · 2 Hz high-pass" if suggested.get("dc_remove") and not recipe.dc_remove else "2 Hz high-pass")
        dc.evidence_title.setText("DC mean (% FS)")
        dc_values = []
        for i, (a, b) in enumerate(zip(before.get("signed_dc", []), after.get("signed_dc", []))):
            label = ("Left", "Right")[i] if track.channels == 2 else f"Ch {i + 1}"
            dc_values.append(f"{label}  {_number(a * 100, 2)} → {_number(b * 100, 2)}%")
        dc.evidence.setText("  ·  ".join(dc_values) or "—")
        low = self.rows["low_cut"]
        resolved = bool(recipe.dc_remove and recipe.low_cut_hz is None and measures.get("subsonic_resolved"))
        low.description.setText("Resolved by DC removal" if resolved else "Suggested" if suggested.get("low_cut") else "")
        low.description.setStyleSheet(f"color:{COLORS['hints' if resolved else 'metadata']};")
        frequency = config.get("subsonic_hz", 30)
        low.evidence_title.setText(f"Energy ≤{frequency:g} Hz")
        low_evidence = []
        if "subsonic_ratio_db" in before:
            low_evidence.append(f"{_number(before['subsonic_ratio_db'], 1)} → "
                                f"{_number(after.get('subsonic_ratio_db'), 1)} dB")
        local_before, local_after = before.get("low_band", {}), after.get("low_band", {})
        spans, corrected_spans = local_before.get("regions", []), local_after.get("regions", [])
        if spans or corrected_spans:
            channels = sorted({ch for _, _, ch in [*spans, *corrected_spans]})
            names = [("Left", "Right")[ch] if track.channels == 2 and ch in (0, 1) else f"Ch {ch + 1}"
                     for ch in channels]
            low_evidence.append(f"Local ≤{min(frequency, 20):g} Hz · {_number(local_before.get('band_dbfs'), 1)} → "
                                f"{_number(local_after.get('band_dbfs'), 1)} dBFS RMS")
            low_evidence.append(f"{_region_count(spans)} → {_region_count(corrected_spans)} · {', '.join(names)}")
        self._low_narrow = "\n".join(low_evidence)
        if "subsonic_ratio_db" in before:
            low_evidence[0] += " relative to program energy"
        self._low_wide = "\n".join(low_evidence)
        preset = config.get(_CUTOFF.key, _CUTOFF.default)
        cutoff_value = overrides.get("low_cut_hz", preset)
        with QSignalBlocker(self.cutoff):
            self.cutoff.setDecimals(_spin_decimals(_CUTOFF, cutoff_value))
            self.cutoff.setValue(cutoff_value)
        self.cutoff_reset.setEnabled("low_cut_hz" in overrides)
        self.cutoff_default.setText(f"Preset: {preset:g} Hz")
        self.cutoff_box.setVisible(selected["low_cut"] or overrides.get("low_cut") is True)
        high = self.rows["ultrasonic"]
        original_high = before.get("ultrasonic", {})
        if track.samplerate < 50000:
            high.description.setText(f"Requires at least 50 kHz · source is {track.samplerate / 1000:g} kHz")
        elif suggested.get("ultrasonic"):
            duration = _qualifying_window_seconds(original_high.get("regions", []), track.samplerate)
            high.description.setText(f"Suggested · Qualifying windows span {duration:.2f} s")
        else:
            high.description.setText("Audible passband through 20 kHz")
        high.evidence_title.setText("Energy ≥24.1 kHz")
        high.evidence.setText(f"{_number(original_high.get('band_dbfs'), 1)} → "
                              f"{_number(after.get('ultrasonic', {}).get('band_dbfs'), 1)} dBFS RMS")
        for row, visible in ((low, bool(low_evidence)), (dc, bool(dc_values)),
                             (high, bool(original_high.get("eligible")))):
            row.evidence_title.setVisible(visible)
            row.evidence.setVisible(visible)
            row.description.setVisible(bool(row.description.text()))
        def peak(data):
            value = data.get("peak_dbfs")
            return _number(value, 2 if value is not None and abs(value) < 1 else 1, sign=True)
        self.peak.setText(f"{peak(before)} → {peak(after)} dBFS")
        self.peak_name.setVisible("peak_dbfs" in before)
        self.peak.setVisible("peak_dbfs" in before)
        names = {track.filename}
        if session.topology:
            names.update(entry.output_filename for entry in session.topology.entries
                         if any(src.input_filename == track.filename for src in entry.sources))
        errors = [str(msg) for name, msg in config.get("_prepare_errors", []) if name in names]
        if track._cache.get("repair_error") and track._cache["repair_error"] not in errors:
            errors.append(track._cache["repair_error"])
        adjustment = config.get("_output_gain_adjustments", {}).get(track.filename)
        self.recovery_title.setVisible(bool(adjustment))
        details = []
        if adjustment:
            errors = [msg for msg in errors if not msg.startswith("Output peak ")]
            errors.insert(0, f"Output peak {_number(adjustment['peak_db'], 2, sign=True)} dBFS")
            self.recovery.setText(f"Lower output Gain by at least {adjustment['reduction_db']:.1f} dB, then retry Prepare.")
            norm = track.processor_results.get("bimodal_normalize")
            if (not config.get("bimodal_normalize_enabled", True) or "bimodal_normalize" in track.processor_skip
                    or norm is None or norm.classification in ("Skip", "Silent")):
                details.append("With normalization bypassed, Adjust Gain opens a manual output gain for this track.")
            if session.prepare_state == "stale" and any(t.filename in names and t.processed_filepath
                                                        for t in session.output_tracks):
                details.append("The previous prepared output is marked stale.")
        self.recovery_detail.setText(" ".join(details))
        self.recovery_detail.setVisible(bool(details))
        messages = errors + format_boundary_settling(measures.get("boundary_settling", []))
        self.error.setText("\n".join(messages))
        self.error.setVisible(bool(messages))
        self.recovery.setVisible(bool(adjustment))
        self.adjust_gain.setVisible(bool(adjustment))
        self.reflow()

    def reflow(self):
        """Reserve native controls inside the report's bottom margin and scroll them with it."""
        if self._laying_out:
            return
        self._laying_out = True
        try:
            root = self.report.document().rootFrame()
            fmt = root.frameFormat()
            height = 0
            if self.track is not None:
                width = max(1, self.report.viewport().width() - 2 * REPORT_INSET)
                narrow = width < 590
                if narrow:
                    width += REPORT_INSET // 2
                self.rows["low_cut"].evidence.setText(self._low_narrow if narrow else self._low_wide)
                if narrow != self._narrow:
                    self._narrow = narrow
                    self.body.setSpacing(8 if narrow else 4)
                    self.body.setContentsMargins(0 if narrow else 12, 8, 0 if narrow else 12, 8)
                    self.scope.setText("Whole track" if narrow else "Whole track · before normalization")
                    self.body.removeWidget(self.close_button)
                    self.header.removeWidget(self.close_button)
                    if narrow:
                        self.body.addWidget(self.close_button, 0, Qt.AlignLeft)
                    else:
                        self.header.addWidget(self.close_button)
                    for key, row in self.rows.items():
                        row.arrange(narrow)
                        row.setVisible(not narrow or self.detector is None or _DETECTORS.get(self.detector) == key)
                    self.peak_layout.addWidget(self.peak_name, 0, 0)
                    self.peak_layout.addWidget(self.peak, int(narrow), int(not narrow))
                    self.peak_layout.setColumnMinimumWidth(0, 0 if narrow else 164)
                    self.peak_layout.setColumnStretch(1, int(not narrow))
                height = max(self.body.minimumSize().height(), self.body.totalHeightForWidth(width))
                self.resize(width, height)
                self.body.activate()
            if fmt.bottomMargin() != height + REPORT_INSET:
                fmt.setBottomMargin(height + REPORT_INSET)
                root.setFrameFormat(fmt)
            self._position()
        finally:
            self._laying_out = False

    def _position(self):
        if self.track is not None:
            bottom = math.ceil(self.report.document().size().height()) - REPORT_INSET
            self.move(REPORT_INSET, bottom - self.height() - self.report.verticalScrollBar().value())

    def _keep_focus_visible(self, _old, widget):
        if widget is None or self.track is None or not self.isAncestorOf(widget):
            return
        top = widget.mapTo(self.report.viewport(), widget.rect().topLeft()).y()
        bar = self.report.verticalScrollBar()
        if top < 0:
            bar.setValue(bar.value() + top - 4)
        elif top + widget.height() > self.report.viewport().height():
            bar.setValue(bar.value() + top + widget.height() - self.report.viewport().height() + 4)
