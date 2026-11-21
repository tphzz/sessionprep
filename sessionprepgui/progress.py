"""Global per-phase progress bar coordinator for SessionPrepWindow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QVBoxLayout

from .widgets import ProgressPanel


class ProgressCoordinator(QObject):
    """Manages per-phase ProgressPanel widgets and active foreground/background ownership."""

    def __init__(
        self,
        current_phase_fn: Callable[[], int],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._current_phase_fn = current_phase_fn
        self.phase_layouts: dict[int, QVBoxLayout] = {}
        self.phase_panels: dict[int, ProgressPanel] = {}
        self.active_phase: int | None = None
        self.owner: Any = None
        self.running: bool = False
        self.text: str = ""
        self.current: int = 0
        self.total: int = 1

    def register_phase_layout(self, phase_index: int, layout: QVBoxLayout) -> None:
        self.phase_layouts[phase_index] = layout
        panel = ProgressPanel()
        panel.setObjectName(f"phase{phase_index}ProgressPanel")
        self.phase_panels[phase_index] = panel
        layout.addWidget(panel)

    def panel_for_phase(self, phase_index: int | None = None) -> ProgressPanel:
        if phase_index is None:
            phase_index = self._current_phase_fn()
        return self.phase_panels[phase_index]

    def show_on_phase(self, phase_index: int) -> None:
        panel = self.panel_for_phase(phase_index)
        for other_phase, other_panel in self.phase_panels.items():
            if other_phase != phase_index:
                other_panel.setVisible(False)
        panel.start(self.text)
        panel.set_progress(self.current, self.total)
        self.active_phase = phase_index

    def on_phase_changed(self, phase_index: int) -> None:
        if self.running:
            self.show_on_phase(phase_index)

    def clear(self) -> None:
        for panel in self.phase_panels.values():
            panel.hide()
        self.active_phase = None
        self.owner = None
        self.running = False
        self.text = ""
        self.current = 0
        self.total = 1

    def start(self, text: str, *, owner: Any = None) -> None:
        self.owner = owner
        self.running = True
        self.text = text
        self.current = 0
        self.total = 1
        self.show_on_phase(self._current_phase_fn())

    def message(self, text: str) -> None:
        self.text = text
        self.panel_for_phase(self.active_phase).set_message(text)

    def value(self, current: int, total: int) -> None:
        self.current = current
        self.total = max(total, 1)
        self.panel_for_phase(self.active_phase).set_progress(current, total)

    def finish(self, text: str) -> None:
        self.owner = None
        self.running = False
        self.text = text
        self.panel_for_phase(self.active_phase).finish(text)

    def fail(self, text: str) -> None:
        self.owner = None
        self.running = False
        self.text = text
        self.panel_for_phase(self.active_phase).fail(text)
