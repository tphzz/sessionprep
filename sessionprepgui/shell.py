"""Protocol contract exposing safe MainWindow shell services to controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from PySide6.QtWidgets import QWidget


class MainWindowShell(Protocol):
    """Minimal surface that GUI controllers may call on the host window."""

    def show_status(self, message: str) -> None: ...

    def show_status_prompt(
        self,
        prefix: str,
        highlight: str,
        suffix: str,
        *,
        tooltip: str,
        on_accept: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        accept_label: str = "Overwrite",
        cancel_label: str = "Cancel",
        accept_description: str = "",
        cancel_description: str = "",
        return_focus_widget: QWidget | None = None,
    ) -> None: ...

    def dismiss_status_prompt(
        self, *, status_message: str | None = None, restore_focus: bool = False
    ) -> None: ...

    def progress_start(self, text: str, *, owner: Any = None) -> None: ...

    def progress_message(self, text: str) -> None: ...

    def progress_value(self, current: int, total: int) -> None: ...

    def progress_finish(self, text: str) -> None: ...

    def progress_fail(self, text: str) -> None: ...

    def clear_progress(self) -> None: ...

    def set_phase_enabled(self, phase_index: int, enabled: bool) -> None: ...

    def switch_to_phase(self, phase_index: int) -> None: ...

    def current_phase(self) -> int: ...

    def is_phase_enabled(self, phase_index: int) -> bool: ...

    def register_phase_progress_layout(self, phase_index: int, layout: Any) -> None: ...

    def dialog_parent(self) -> QWidget: ...

    def is_batch_mode(self) -> bool: ...

    def is_closing_after_daw(self) -> bool: ...

    def resume_close_after_daw(self) -> None: ...

    def set_window_title(self, title: str) -> None: ...
