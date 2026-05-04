"""GeneralPage — application-level settings (app + waveform prefs)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QVBoxLayout,
    QSpinBox,
    QWidget,
)

from sessionpreplib.models import ParamSpec

from .param_form import (
    _build_param_page,
    _read_widget,
    _set_widget_value,
    sanitize_output_folder,
)

# ---------------------------------------------------------------------------
# Page-level param specs (data, not UI)
# ---------------------------------------------------------------------------

_APP_PARAMS = [
    ParamSpec(
        key="default_project_dir", type=str, default="",
        label="Default project directory",
        description=(
            "When set, the Open Folder dialog starts in this directory. "
            "Leave empty to use the system default."
        ),
        widget_hint="path_picker_folder",
    ),
    ParamSpec(
        key="scale_factor", type=(int, float), default=1.0,
        min=0.5, max=4.0,
        label="HiDPI scale factor",
        description=(
            "Scale factor for the application UI. "
            "Requires a restart to take effect."
        ),
    ),
    ParamSpec(
        key="report_verbosity", type=str, default="normal",
        choices=["normal", "verbose"],
        label="Report verbosity",
        description=(
            "Controls the level of detail shown in track reports. "
            "Verbose mode includes additional analytical data such as "
            "classification metrics."
        ),
    ),
    ParamSpec(
        key="phase1_output_folder", type=str, default="sp_01_tracklayout",
        label="Phase 1: track layout folder name",
        description=(
            "Name of the subfolder (relative to the project directory) "
            "where track-layout audio files are written after "
            "applying the channel topology. "
            "Must be a simple folder name without path separators."
        ),
    ),
    ParamSpec(
        key="phase2_output_folder", type=str, default="sp_02_prepared",
        label="Phase 2: prepared folder name",
        description=(
            "Name of the subfolder (relative to the project directory) "
            "where processed audio files are written after "
            "applying analysis processors (e.g. normalization). "
            "Must be a simple folder name without path separators."
        ),
    ),
    ParamSpec(
        key="spectrogram_colormap", type=str, default="magma",
        choices=["magma", "viridis", "grayscale"],
        label="Spectrogram color theme",
        description="Color palette used for the spectrogram display.",
    ),
    ParamSpec(
        key="invert_scroll", type=str, default="default",
        choices=["default", "horizontal", "vertical", "both"],
        label="Invert mouse-wheel scrolling",
        description=(
            "Reverses the scroll direction in the waveform/spectrogram view. "
            "'horizontal' inverts Shift+wheel (timeline panning), "
            "'vertical' inverts Shift+Alt+wheel (frequency panning), "
            "'both' inverts both axes."
        ),
    ),
]


class GeneralPage(QWidget):
    """App-level preference form.

    Implements the standard page interface:
        load(config)   — populate from config["app"]
        commit(config) — write back to config["app"]
        validate()     — returns error string or None
    """

    preferencesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: list[tuple[str, QWidget]] = []
        self._init_ui()

    # ── Page interface ────────────────────────────────────────────────

    def load(self, config: dict) -> None:
        values = config.get("app", {})
        for key, widget in self._widgets:
            if key in values:
                _set_widget_value(widget, values[key])

    def commit(self, config: dict) -> None:
        app = config.setdefault("app", {})
        app.update(self.current_values())

    def current_values(self) -> dict:
        """Return the current app preference values managed by this page."""
        return {key: _read_widget(widget) for key, widget in self._widgets}

    def validate(self) -> str | None:
        """Return an error message if any output folder name is invalid."""
        for key, widget in self._widgets:
            if key in ("phase1_output_folder", "phase2_output_folder"):
                raw = _read_widget(widget)
                if sanitize_output_folder(str(raw)) is None:
                    label = ("Phase 1 track layout" if "phase1" in key
                             else "Phase 2 prepared")
                    return (
                        f"The {label} folder name is invalid.\n\n"
                        "It must be a simple folder name without path "
                        "separators, special characters, or reserved names."
                    )
        return None

    # ── UI setup ─────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        page, self._widgets = _build_param_page(_APP_PARAMS, {})
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)
        for _key, widget in self._widgets:
            self._connect_change_signal(widget)

    def _connect_change_signal(self, widget: QWidget) -> None:
        if hasattr(widget, "path_changed"):
            widget.path_changed.connect(lambda *_: self.preferencesChanged.emit())
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(
                lambda *_: self.preferencesChanged.emit())
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda *_: self.preferencesChanged.emit())
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda *_: self.preferencesChanged.emit())
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda *_: self.preferencesChanged.emit())
