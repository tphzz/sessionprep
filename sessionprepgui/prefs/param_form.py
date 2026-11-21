"""Generic PySide6 widget factory for parameter-spec-driven forms.

Uses shared desktop controls with any object satisfying the ParamSpec protocol.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from functools import partial
from html import escape
from typing import Any, Protocol, runtime_checkable
from collections.abc import Callable

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPixmap

from ..controls import FormActionButton, FormBrowseButton, FormDoubleSpinBox, FormSpinBox
from ..icons import icon
from ..theme import CONTENT_INSET, argb_to_qcolor as _argb_to_qcolor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QStyleOptionSpinBox,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# ParamSpec protocol  (duck-typed — sessionpreplib.config.ParamSpec satisfies it)
# ---------------------------------------------------------------------------

@runtime_checkable
class ParamSpec(Protocol):  # pylint: disable=too-few-public-methods
    key: str
    label: str
    type: type
    default: Any
    choices: list | None
    min: float | None
    max: float | None
    description: str | None
    widget_hint: str | None   # rendering hint consumed by _build_widget; never read by the library


# ---------------------------------------------------------------------------
# PathPickerMode
# ---------------------------------------------------------------------------

class PathPickerMode(enum.Enum):
    """Controls which QFileDialog variant PathPicker opens."""

    FOLDER    = "folder"      # QFileDialog.getExistingDirectory
    OPEN_FILE = "open_file"   # QFileDialog.getOpenFileName
    SAVE_FILE = "save_file"   # QFileDialog.getSaveFileName


# ---------------------------------------------------------------------------
# ARGB / color helpers
# ---------------------------------------------------------------------------


def _color_swatch_icon(argb: str, size: int = 16) -> QIcon:
    """Create a small square QIcon filled with the given ARGB color."""
    pm = QPixmap(size, size)
    pm.fill(_argb_to_qcolor(argb))
    return QIcon(pm)


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

# String settings fields are at least this wide, so default values such as
# folder names are fully visible; longer values scroll inside the field.
LINE_EDIT_MIN_WIDTH = 200
LINE_EDIT_MAX_MIN_WIDTH = 360   # beyond this a value scrolls inside its field


def _fit_line_edit(w: QLineEdit) -> None:
    text_width = w.fontMetrics().horizontalAdvance(w.text() or w.placeholderText())
    w.setMinimumWidth(min(max(LINE_EDIT_MIN_WIDTH, text_width + 24), LINE_EDIT_MAX_MIN_WIDTH))


def _decimal_places(value: Any) -> int:
    if value is None:
        return 0
    exponent = Decimal(str(float(value))).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _spin_decimals(spec: Any, value: Any) -> int:
    refs = (spec.default, spec.min, spec.max, value)
    return max(2, *map(_decimal_places, refs))


_PATH_HINT_MODES = {
    "path_picker_folder": PathPickerMode.FOLDER,
    "path_picker_file": PathPickerMode.OPEN_FILE,
    "path_picker_save": PathPickerMode.SAVE_FILE,
}


def _build_double_spin_widget(spec: Any, value: Any) -> QWidget:
    w = FormDoubleSpinBox()
    lo = float(spec.min) if spec.min is not None else -999999.0
    hi = float(spec.max) if spec.max is not None else 999999.0
    decimals = _spin_decimals(spec, value)
    w.setDecimals(decimals)
    w.setMinimum(lo)
    w.setMaximum(hi)
    span = hi - lo
    if decimals >= 4:
        w.setSingleStep(10 ** -decimals)
    elif span <= 5:
        w.setSingleStep(0.25)
    elif span <= 20:
        w.setSingleStep(0.5)
    elif span <= 200:
        w.setSingleStep(1.0)
    else:
        w.setSingleStep(5.0)
    w.setValue(float(value) if value is not None else float(spec.default))
    return w


def _build_widget(
    spec: Any,
    value: Any,
    *,
    reset_to_preset: Callable[[QWidget], None] | None = None,
) -> QWidget:
    """Create an appropriate input widget for a ParamSpec and set its value."""
    hint = getattr(spec, "widget_hint", None)
    if hint in _PATH_HINT_MODES:
        picker = PathPicker(spec, mode=_PATH_HINT_MODES[hint], reset_to_preset=reset_to_preset)
        _set_widget_value(picker, value)
        return picker

    if spec.choices is not None:
        w: QWidget = QComboBox()
        for c in spec.choices:
            w.addItem(str(c), c)
        idx = w.findData(value)
        if idx >= 0:
            w.setCurrentIndex(idx)
    elif spec.type is bool:
        w = QCheckBox()
        w.setChecked(bool(value))
    elif spec.type is int:
        w = FormSpinBox()
        w.setMinimum(int(spec.min) if spec.min is not None else -999999)
        w.setMaximum(int(spec.max) if spec.max is not None else 999999)
        w.setValue(int(value) if value is not None else int(spec.default))
    elif spec.type in ((int, float), float):
        w = _build_double_spin_widget(spec, value)
    elif spec.type is list:
        w = QLineEdit()
        w.setText(", ".join(str(x) for x in value) if isinstance(value, list) else (str(value) if value else ""))
        w.setPlaceholderText("comma-separated values")
        _fit_line_edit(w)
    else:
        w = QLineEdit()
        w.setText(str(value) if value is not None else "")
        _fit_line_edit(w)
    w._param_spec = spec
    return w


def _set_widget_value(widget: QWidget, value: Any) -> None:
    """Set a widget's value programmatically."""
    if isinstance(widget, PathPicker):   # checked before QLineEdit (PathPicker contains one)
        widget.set_value(str(value) if value is not None else "")
        return
    if isinstance(widget, QComboBox):
        idx = widget.findData(value)
        if idx >= 0:
            widget.setCurrentIndex(idx)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setDecimals(_spin_decimals(widget._param_spec, value))
        widget.setValue(float(value))
    elif isinstance(widget, QLineEdit):
        if isinstance(value, list):
            widget.setText(", ".join(str(x) for x in value))
        else:
            widget.setText(str(value) if value is not None else "")


def _read_widget(widget: QWidget) -> Any:
    """Read the current value from a widget created by _build_widget."""
    if isinstance(widget, PathPicker):   # checked before QLineEdit (PathPicker contains one)
        return widget.value()
    spec = widget._param_spec
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QLineEdit):
        text = widget.text().strip()
        if spec.type is list:
            if not text:
                return []
            return [s.strip() for s in text.split(",") if s.strip()]
        return text
    return None


# ---------------------------------------------------------------------------
# Tooltip / subtext builders
# ---------------------------------------------------------------------------

def _type_label(t: Any) -> str:
    """Human-readable type name."""
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__


_BUILT_IN = object()


def _reference(spec: Any, preset_value: Any) -> tuple[str, Any]:
    if preset_value is _BUILT_IN:
        return "Default", spec.default
    return "Preset", "(empty)" if preset_value == "" else preset_value


def _build_tooltip(spec: Any, *, preset_value: Any = _BUILT_IN) -> str:
    """Build a rich tooltip with key, reference value, and range info."""
    parts = [f"<b>{spec.label}</b>"]
    if spec.description:
        parts.append(f"<br/>{spec.description}")
    parts.append(f"<br/><br/>Config key: <code>{spec.key}</code>")
    label, value = _reference(spec, preset_value)
    parts.append(f"<br/>{label}: <b>{escape(str(value))}</b>")
    if spec.min is not None or spec.max is not None:
        lo = str(spec.min) if spec.min is not None else "\u2212\u221e"
        hi = str(spec.max) if spec.max is not None else "\u221e"
        parts.append(f"<br/>Range: {lo} \u2013 {hi}")
    if spec.choices:
        parts.append(f"<br/>Choices: {', '.join(str(c) for c in spec.choices)}")
    return "".join(parts)


def _build_subtext(spec: Any, *, preset_value: Any = _BUILT_IN) -> str:
    """Build visible subtext with description, type, range, and reference."""
    parts = []
    if spec.description:
        parts.append(spec.description)
    meta = [f"Type: {_type_label(spec.type)}"]
    if spec.min is not None or spec.max is not None:
        lo = str(spec.min) if spec.min is not None else "\u2212\u221e"
        hi = str(spec.max) if spec.max is not None else "\u221e"
        meta.append(f"Range: {lo} \u2013 {hi}")
    if spec.choices:
        meta.append(f"Choices: {', '.join(str(c) for c in spec.choices)}")
    label, value = _reference(spec, preset_value)
    meta.append(f"{label}: {value}")
    parts.append("  \u2022  ".join(meta))
    return "\n".join(parts)


class ConfigResetButton(FormActionButton):
    """Let focused reset buttons use Space even in the playback window."""

    def event(self, event):
        if event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Space:
            event.accept()
            return True
        return super().event(event)


class _FieldResetButton(ConfigResetButton):
    """Paint a compact face while the full button stays available to input."""

    def __init__(self, companion: QWidget):
        super().__init__()
        self._companion = companion
        self.setIcon(icon("reset"))
        self.setIconSize(QSize(16, 16))
        self.setFixedWidth(32)
        self.setMinimumHeight(32)

    def sizeHint(self) -> QSize:
        companion = self._companion
        height = companion.sizeHint().height()
        if isinstance(companion, QAbstractSpinBox):
            height -= 2 * self._spin_frame_width()
        elif isinstance(companion, QCheckBox):
            height = 24
        return QSize(32, max(self.minimumHeight(), height))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, _event):
        companion = self._companion
        top = companion.y() - self.y()
        option = QStyleOptionButton()
        self.initStyleOption(option)
        if isinstance(companion, QAbstractSpinBox):
            self._paint_spin_box_face(option, top)
            return
        if isinstance(companion, QCheckBox):
            check = QStyleOptionButton()
            companion.initStyleOption(check)
            indicator = companion.style().subElementRect(
                QStyle.SE_CheckBoxIndicator, check, companion
            )
            height = 24
            top += indicator.y() + (indicator.height() - height) // 2
        else:
            height = min(companion.height(), self.height())
            top += (companion.height() - height) // 2
        option.rect = QRect(0, top, self.width(), height)
        with QStylePainter(self) as painter:
            painter.drawControl(QStyle.CE_PushButton, option)

    def _paint_spin_box_face(self, option: QStyleOptionButton, top: int) -> None:
        spin_box = self._companion
        # Fusion places a locally painted spin frame on the form. Use the same
        # raster origin and truncate the logical height of the rounded pixmap.
        scale = self.devicePixelRatioF()
        surface = QPixmap(QSize(self.width(), spin_box.height()) * scale)
        surface.setDevicePixelRatio(scale)
        surface.fill(Qt.transparent)

        spin_option = QStyleOptionSpinBox()
        spin_box.initStyleOption(spin_option)
        style = spin_box.style()
        frame = style.subControlRect(QStyle.CC_SpinBox, spin_option, QStyle.SC_SpinBoxFrame, spin_box)
        frame.setHeight(min(frame.height(), int(surface.deviceIndependentSize().height())))
        inset = self._spin_frame_width(spin_option)
        frame = frame.adjusted(0, inset, 0, -inset)
        height = min(frame.height(), self.height())
        option.rect = QRect(0, frame.y() + (frame.height() - height) // 2, self.width(), height)

        with QStylePainter(surface, self) as painter:
            painter.drawControl(QStyle.CE_PushButton, option)
        with QPainter(self) as painter:
            painter.drawPixmap(QPoint(0, top), surface)

    def _spin_frame_width(self, option=None) -> int:
        companion = self._companion
        return companion.style().pixelMetric(QStyle.PM_DefaultFrameWidth, option, companion)


def _set_param_reference(
    spec: Any, editor: QWidget, label: QLabel, subtext: QLabel, reset: QPushButton,
    *, preset_value: Any = _BUILT_IN,
) -> None:
    tooltip = _build_tooltip(spec, preset_value=preset_value)
    for widget in (editor, label, subtext):
        widget.setToolTip(tooltip)
    subtext.setText(_build_subtext(spec, preset_value=preset_value))
    name = spec.label or spec.key
    if preset_value is _BUILT_IN:
        reset.setAccessibleName(f"Reset {name} to default")
        value = "empty" if spec.default == "" else spec.default
        reset.setToolTip(f"Reset to default ({value})")
    else:
        reset.setAccessibleName(f"Use preset value for {name}")
        _, value = _reference(spec, preset_value)
        reset.setToolTip(f"<p>Use preset value for {escape(name)}<br/>{escape(str(value))}</p>")


# ---------------------------------------------------------------------------
# PathPicker — self-contained path / file picker widget
# ---------------------------------------------------------------------------

class PathPicker(QWidget):
    """Generic path/file picker row driven by a ParamSpec.

    Renders a bold label, a ``QLineEdit``, a Browse button, and a reset
    button.  When *show_recursive* is ``True`` and *mode* is ``FOLDER`` an
    "Include subfolders" checkbox is appended below the input row.  A grey
    description subtext mirrors the style of ``_build_param_page`` rows,
    making the widget a visual drop-in for any form built with this module.

    This class depends on the shared GUI icon helper, not sessionpreplib.

    Parameters
    ----------
    spec:
        ``ParamSpec`` providing *key*, *label*, *description*, and *default*.
    mode:
        Which ``QFileDialog`` variant the Browse button opens.
    file_filter:
        Qt filter string (e.g. ``"Audio (*.wav *.flac);;All (*.*)"``).
        Only meaningful for ``OPEN_FILE`` / ``SAVE_FILE`` modes.
    show_recursive:
        When ``True`` and *mode* is ``FOLDER``, adds an
        "Include subfolders" ``QCheckBox`` below the input row.
    """

    #: Emitted whenever the path text changes (browse, clear, or manual edit).
    path_changed = Signal(str)

    def __init__(
        self,
        spec: Any,
        *,
        mode: PathPickerMode = PathPickerMode.FOLDER,
        file_filter: str = "",
        show_recursive: bool = False,
        reset_to_preset: Callable[[QWidget], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._param_spec = spec
        self._mode = mode
        self._file_filter = file_filter
        self._show_recursive = show_recursive
        self._reset_to_preset = reset_to_preset
        self._line_edit: QLineEdit
        self._recursive_cb: QCheckBox | None = None
        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────

    def value(self) -> str:
        """Return the current path text (stripped)."""
        return self._line_edit.text().strip()

    def set_value(self, path: str) -> None:
        """Set the path text and emit :attr:`path_changed`."""
        self._line_edit.setText(path)
        self.path_changed.emit(path)

    def recursive(self) -> bool:
        """Return the "Include subfolders" checkbox state.

        Always returns ``False`` when *show_recursive* was not set.
        """
        return self._recursive_cb.isChecked() if self._recursive_cb else False

    def set_recursive(self, on: bool) -> None:
        """Set the "Include subfolders" checkbox state."""
        if self._recursive_cb:
            self._recursive_cb.setChecked(on)

    # ── Private ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        spec = self._spec
        placeholder = "" if spec.default else "(system default)"

        # Input row: label | line-edit | Browse… | ↺
        self._line_edit = QLineEdit()
        self._line_edit.setPlaceholderText(placeholder)
        self._line_edit.textEdited.connect(self._on_text_edited)

        browse_btn = FormBrowseButton()
        browse_btn.setToolTip("Open file browser")
        browse_btn.clicked.connect(self._browse)

        reset_btn = _FieldResetButton(browse_btn)
        reset_btn.clicked.connect(self._reset_path)

        name_lbl = QLabel(f"<b>{spec.label}</b>")

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        input_row.addWidget(name_lbl, 0)
        input_row.addWidget(self._line_edit, 1)
        input_row.addWidget(browse_btn)
        input_row.addWidget(reset_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addLayout(input_row)

        # Optional recursive checkbox (FOLDER mode only)
        if self._show_recursive and self._mode is PathPickerMode.FOLDER:
            self._recursive_cb = QCheckBox("Include subfolders")
            outer.addWidget(self._recursive_cb)

        # Description subtext — matches _build_param_page visual style
        sub = QLabel()
        sub.setTextFormat(Qt.TextFormat.PlainText)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #888; font-size: 9pt;")
        outer.addWidget(sub)
        self._set_param_reference = partial(
            _set_param_reference, spec, self._line_edit, name_lbl, sub, reset_btn)
        self._set_param_reference()

    def _reset_path(self) -> None:
        if self._reset_to_preset is None:
            self.set_value(self._spec.default)
        else:
            self._reset_to_preset(self)

    def _browse(self) -> None:
        start = self.value() or ""
        path = ""
        if self._mode is PathPickerMode.FOLDER:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {self._spec.label}", start,
                QFileDialog.Option.ShowDirsOnly,
            )
        elif self._mode is PathPickerMode.OPEN_FILE:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self._spec.label}", start, self._file_filter,
            )
        elif self._mode is PathPickerMode.SAVE_FILE:
            path, _ = QFileDialog.getSaveFileName(
                self, f"Select {self._spec.label}", start, self._file_filter,
            )
        if path:
            self.set_value(path)

    def _on_text_edited(self, text: str) -> None:
        self.path_changed.emit(text.strip())


# ---------------------------------------------------------------------------
# Page builder
# ---------------------------------------------------------------------------

SETTINGS_PAGE_MARGIN = CONTENT_INSET


def _build_param_page(
    params: list[Any],
    values: dict[str, Any],
    *, on_reset: Callable[[str, QWidget], None] | None = None,
) -> tuple[QWidget, list[tuple[str, QWidget]]]:
    """Build a scrollable form page for a list of ParamSpecs.

    Returns ``(page_widget, [(key, widget), ...])``
    """
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(
        SETTINGS_PAGE_MARGIN, SETTINGS_PAGE_MARGIN, SETTINGS_PAGE_MARGIN, SETTINGS_PAGE_MARGIN)
    outer.setSpacing(12)
    widgets: list[tuple[str, QWidget]] = []
    for spec in params:
        val = values.get(spec.key, spec.default)
        reset_to_preset = partial(on_reset, spec.key) if on_reset is not None else None
        w = _build_widget(spec, val, reset_to_preset=reset_to_preset)
        reference = val if reset_to_preset is not None else _BUILT_IN

        # Self-contained widgets (e.g. PathPicker) already render their own
        # label, subtext, and reset button — add them directly.
        if isinstance(w, PathPicker):
            w._set_param_reference(preset_value=reference)
            outer.addWidget(w)
            widgets.append((spec.key, w))
            continue

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name_label = QLabel(f"<b>{spec.label}</b>")
        row.addWidget(name_label, 1)
        row.addWidget(w, 0)

        reset_btn = _FieldResetButton(w)
        reset = (partial(_set_widget_value, w, spec.default)
                 if reset_to_preset is None else partial(reset_to_preset, w))
        reset_btn.clicked.connect(lambda _checked=False, action=reset: action())
        row.addWidget(reset_btn)

        param_box = QVBoxLayout()
        param_box.setContentsMargins(0, 0, 0, 0)
        param_box.setSpacing(2)
        param_box.addLayout(row)

        sub_label = QLabel()
        sub_label.setTextFormat(Qt.TextFormat.PlainText)
        sub_label.setWordWrap(True)
        sub_label.setStyleSheet("color: #888; font-size: 9pt;")
        param_box.addWidget(sub_label)
        w._set_param_reference = partial(
            _set_param_reference, spec, w, name_label, sub_label, reset_btn)
        w._set_param_reference(preset_value=reference)

        outer.addLayout(param_box)
        widgets.append((spec.key, w))

    outer.addStretch()
    return page, widgets


# ---------------------------------------------------------------------------
# Output folder validation
# ---------------------------------------------------------------------------

_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)
_ILLEGAL_CHARS = frozenset('<>:"|?*')


def sanitize_output_folder(name: str) -> str | None:
    """Validate and clean an output folder name.

    Returns the stripped name on success, or ``None`` if invalid.
    Rejects empty strings, path traversals, separators, illegal Windows
    characters, control characters, and reserved names.
    """
    name = name.strip()
    if not name:
        return None
    if ".." in name:
        return None
    if "/" in name or "\\" in name:
        return None
    if any(c in _ILLEGAL_CHARS for c in name):
        return None
    if any(ord(c) < 32 for c in name):
        return None
    if name.upper() in _WINDOWS_RESERVED:
        return None
    return name
