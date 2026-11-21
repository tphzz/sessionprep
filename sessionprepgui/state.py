"""Explicit shared workspace state for the SessionPrep desktop GUI."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from sessionpreplib.config import (
    default_config,
    flatten_structured_config,
    resolve_session_config,
)
from sessionpreplib.models import SessionContext, TrackContext

from .settings import resolve_config_preset


class WorkspaceState(QObject):
    """Single source of truth for cross-phase desktop workspace state.

    Worker handles (``QThread`` instances) are intentionally excluded: each
    controller owns its own background workers, cancellation, and cleanup.
    """

    session_changed = Signal(object)          # SessionContext | None
    current_track_changed = Signal(object)    # TrackContext | None
    summary_changed = Signal(object)          # dict | None
    topology_changed = Signal()
    config_changed = Signal()
    groups_changed = Signal()
    daw_processor_changed = Signal(object)    # DawProcessor | None
    overlays_changed = Signal(object)         # set[str]
    busy_changed = Signal(bool, str)          # (is_busy, reason)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._session: SessionContext | None = None
        self._summary: dict[str, Any] | None = None
        self._source_dir: str | None = None
        self._topology_dir: str | None = None
        self._topo_source_tracks: list[TrackContext] = []
        self._topo_topology: Any = None
        self._recursive_scan: bool = False
        self._current_track: TrackContext | None = None
        self._selected_detector_overlays: set[str] = set()
        self._config: dict[str, Any] = config if config is not None else {}
        self._active_config_preset_name: str = self.default_config_preset_name()
        self._active_session_preset: str = "Default"
        self._session_overrides: dict[str, Any] = {}
        self._session_groups: list[dict[str, Any]] = []
        self._daw_processors: list[Any] = []
        self._active_daw_processor: Any = None
        self._daw_template_bindings: dict[str, Any] = {}
        self._peak_cache: dict[str, Any] = {}

    # ── Session & selection properties ────────────────────────────────────

    @property
    def session(self) -> SessionContext | None:
        return self._session

    @session.setter
    def session(self, value: SessionContext | None) -> None:
        if self._session is not value:
            self._session = value
            self.session_changed.emit(value)

    @property
    def summary(self) -> dict[str, Any] | None:
        return self._summary

    @summary.setter
    def summary(self, value: dict[str, Any] | None) -> None:
        self._summary = value
        self.summary_changed.emit(value)

    @property
    def source_dir(self) -> str | None:
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value: str | None) -> None:
        self._source_dir = value

    @property
    def topology_dir(self) -> str | None:
        return self._topology_dir

    @topology_dir.setter
    def topology_dir(self, value: str | None) -> None:
        self._topology_dir = value

    @property
    def topo_source_tracks(self) -> list[TrackContext]:
        return self._topo_source_tracks

    @topo_source_tracks.setter
    def topo_source_tracks(self, value: list[TrackContext]) -> None:
        self._topo_source_tracks = value

    @property
    def topo_topology(self) -> Any:
        return self._topo_topology

    @topo_topology.setter
    def topo_topology(self, value: Any) -> None:
        self._topo_topology = value
        self.topology_changed.emit()

    @property
    def recursive_scan(self) -> bool:
        return self._recursive_scan

    @recursive_scan.setter
    def recursive_scan(self, value: bool) -> None:
        self._recursive_scan = bool(value)

    @property
    def current_track(self) -> TrackContext | None:
        return self._current_track

    @current_track.setter
    def current_track(self, value: TrackContext | None) -> None:
        if self._current_track is not value:
            self._current_track = value
            self.current_track_changed.emit(value)

    @property
    def selected_detector_overlays(self) -> set[str]:
        return self._selected_detector_overlays

    @selected_detector_overlays.setter
    def selected_detector_overlays(self, value: set[str]) -> None:
        self._selected_detector_overlays = set(value)
        self.overlays_changed.emit(self._selected_detector_overlays)

    # ── Config, groups & DAW processor properties ─────────────────────────

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        self._config = value
        self.config_changed.emit()

    @property
    def active_config_preset_name(self) -> str:
        return self._active_config_preset_name

    @active_config_preset_name.setter
    def active_config_preset_name(self, value: str) -> None:
        self._active_config_preset_name = value
        self.config_changed.emit()

    @property
    def active_session_preset(self) -> str:
        return self._active_session_preset

    @active_session_preset.setter
    def active_session_preset(self, value: str) -> None:
        self._active_session_preset = value

    @property
    def session_overrides(self) -> dict[str, Any]:
        return self._session_overrides

    @session_overrides.setter
    def session_overrides(self, value: dict[str, Any]) -> None:
        self._session_overrides = value
        self.config_changed.emit()

    @property
    def session_groups(self) -> list[dict[str, Any]]:
        return self._session_groups

    @session_groups.setter
    def session_groups(self, value: list[dict[str, Any]]) -> None:
        self._session_groups = value
        self.groups_changed.emit()

    @property
    def daw_processors(self) -> list[Any]:
        return self._daw_processors

    @daw_processors.setter
    def daw_processors(self, value: list[Any]) -> None:
        self._daw_processors = value

    @property
    def active_daw_processor(self) -> Any:
        return self._active_daw_processor

    @active_daw_processor.setter
    def active_daw_processor(self, value: Any) -> None:
        if self._active_daw_processor is not value:
            self._active_daw_processor = value
            self.daw_processor_changed.emit(value)

    @property
    def daw_template_bindings(self) -> dict[str, Any]:
        return self._daw_template_bindings

    @daw_template_bindings.setter
    def daw_template_bindings(self, value: dict[str, Any]) -> None:
        self._daw_template_bindings = value

    @property
    def peak_cache(self) -> dict[str, Any]:
        return self._peak_cache

    @peak_cache.setter
    def peak_cache(self, value: dict[str, Any]) -> None:
        self._peak_cache = value

    # ── Derived config helpers ────────────────────────────────────────────

    def existing_config_preset_name(self, name: str) -> str:
        if isinstance(name, str) and name in self._config.get("config_presets", {}):
            return name
        return "Default"

    def default_config_preset_name(self) -> str:
        return self.existing_config_preset_name(
            self._config.get("app", {}).get("active_config_preset", "Default")
        )

    def active_preset(self) -> dict[str, Any]:
        """Return the active config preset's structured dict."""
        return resolve_config_preset(self._config, self._active_config_preset_name)

    def effective_session_config(self) -> dict[str, Any]:
        return resolve_session_config(self.active_preset(), self._session_overrides)

    def flat_config(self) -> dict[str, Any]:
        """Return the flattened effective configuration for runtime consumers."""
        flat = default_config()
        flat.update(flatten_structured_config(self.effective_session_config()))
        if self._active_daw_processor is not None:
            flat["_fader_ceiling_db"] = self._active_daw_processor.fader_ceiling_db
        return flat

    # ── Workspace reset ───────────────────────────────────────────────────

    def reset_workspace(self, *, keep_session_config: bool = False) -> None:
        """Reset all per-project workspace fields to an empty state."""
        self.peak_cache = {}
        self.session = None
        self.summary = None
        self.current_track = None
        self.selected_detector_overlays = set()
        self.topology_dir = None
        self.source_dir = None
        self.topo_source_tracks = []
        self.topo_topology = None
        if not keep_session_config:
            self._session_overrides.clear()
            self.active_config_preset_name = self.default_config_preset_name()
        self.session_groups = []
