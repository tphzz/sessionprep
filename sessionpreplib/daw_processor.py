"""Abstract base class for DAW integration processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import os
import stat
from typing import Any

from .models import ParamSpec
from .models import DawCommand, DawCommandResult, SessionContext, TrackContext


@dataclass(frozen=True)
class TransferAudio:
    """Media path and corresponding DAW fader offset for a transfer entry."""

    filepath: str
    fader_db: float


def resolve_transfer_audio(session: SessionContext, track: TrackContext) -> TransferAudio:
    """Resolve and verify the selected output before creating a DAW track."""
    choice = session.config.get("_use_processed")
    prepared = bool(track.processed_filepath) if choice is None else choice
    path = track.processed_filepath if prepared else track.filepath
    if not path:
        raise ValueError("The selected audio is unavailable. Run Prepare again.")
    if prepared:
        if session.prepare_state == "stale" or (choice is True and session.prepare_state != "ready"):
            raise ValueError("Prepared audio is out of date. Run Prepare again.")
    elif track.processed_filepath and _same_audio_path(path, track.processed_filepath):
        # Older saved outputs stored the prepared path in both fields.
        source = next((t for t in session.tracks if t.filename == track.filename), None)
        if source is None or _same_audio_path(source.filepath, track.processed_filepath):
            raise ValueError("The routed source is unavailable. Apply Track Layout and Prepare again.")
        path = source.filepath

    try:
        info = os.stat(path)
    except OSError as exc:
        raise ValueError(f"The selected audio file could not be opened. {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"The selected audio path is not a file. {path}")
    signature = track.prepared_file_signature if prepared else track.source_file_signature
    if signature is not None and signature != (info.st_size, info.st_mtime_ns):
        if prepared:
            raise ValueError("The prepared file has changed. Run Prepare again.")
        raise ValueError("The routed source file has changed. Reanalyze this track.")

    fader_db = 0.0
    if prepared and track.prepared_fader_db is not None:
        fader_db = track.prepared_fader_db
    elif prepared or choice is None:
        fader_db = _legacy_fader_db(session, track)
    if not isinstance(fader_db, (int, float)) or not math.isfinite(fader_db):
        raise ValueError("Invalid saved fader offset. Run Prepare again.")
    return TransferAudio(os.path.abspath(path), fader_db)


def _same_audio_path(first: str, second: str) -> bool:
    return os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second))


def _legacy_fader_db(session: SessionContext, track: TrackContext) -> float:
    """Read legacy normalization evidence, including skips saved only on the source."""
    proc_id = "bimodal_normalize"
    source = next((t for t in session.tracks if t.filename == track.filename), track)
    if (not session.config.get(f"{proc_id}_enabled", True)
            or proc_id in track.processor_skip or proc_id in source.processor_skip):
        return 0.0
    result = track.processor_results.get(proc_id) or source.processor_results.get(proc_id)
    if result is None or result.error or result.classification in ("Skip", "Silent"):
        return 0.0
    return result.data.get("fader_offset", 0.0)


class DawProcessor(ABC):
    """Abstract base for DAW integration processors.

    Each concrete subclass handles one DAW (e.g. ProToolsProcessor,
    DAWProjectProcessor).  The processor owns all communication logic
    internally — the ABC only defines the lifecycle contract.

    Execution model (Option B):
        transfer()/sync() build a list of DawCommand objects internally,
        execute them via processor-private dispatch, and return
        DawCommandResult objects.  Commands are plain data; the processor
        is the executor.

    Lifecycle (called by GUI/CLI, not Pipeline):
        1. configure(config)        — read ParamSpec values
        2. check_connectivity()     — verify DAW is reachable
        3. fetch(session)           — pull DAW state into session.daw_state
        4. transfer(session)        — initial full push to DAW
        5. sync(session)            — incremental delta push

    Ad-hoc commands (called by GUI tools, outside lifecycle):
        6. execute_commands(session, commands) — run externally-built commands
    """

    id: str = ""
    name: str = ""
    fader_ceiling_db: float = 12.0

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        """Base returns the enabled toggle and project dir. Subclasses call super() + [...]."""
        return [
            ParamSpec(
                key=f"{cls.id}_enabled",
                type=bool,
                default=True,
                label="Enabled",
                description=(
                    "Whether this DAW processor is available for selection "
                    "in the toolbar. Disable if you never use this DAW."
                ),
            ),
            ParamSpec(
                key=f"{cls.id}_project_dir",
                type=str,
                default="",
                label="Project directory",
                description=(
                    f"Directory where newly created {cls.name} projects are saved. "
                    "Leave empty to prompt for a location each time."
                ),
                widget_hint="path_picker_folder",
            ),
        ]

    def configure(self, config: dict[str, Any]) -> None:
        """Read config values. Subclasses should call super().configure(config)."""
        self._enabled: bool = config.get(f"{self.id}_enabled", True)
        self._connected: bool = False
        self._project_dir: str = config.get(f"{type(self).id}_project_dir", "")

    @property
    def enabled(self) -> bool:
        """Whether this processor is available for selection."""
        return self._enabled

    @property
    def project_dir(self) -> str:
        """The directory where new projects should be created."""
        return self._project_dir

    @property
    def template_identity(self) -> tuple[str, ...]:
        """Identity of the template that owns fetched folders and assignments."""
        return (self.id,)

    @property
    def connected(self) -> bool:
        """Whether the last connectivity check succeeded."""
        return self._connected

    @abstractmethod
    def check_connectivity(self) -> tuple[bool, str]:
        """Test whether the DAW is reachable.

        Returns (ok, message).  For socket-based DAWs (Pro Tools PTSL)
        this checks the connection.  For file-based DAWs (DAWProject)
        this might validate the output path.
        """

    @abstractmethod
    def fetch(
        self,
        session: SessionContext,
        *,
        ignore_cache: bool = False,
    ) -> SessionContext:
        """Pull external state from the DAW into the session.

        Populates session.daw_state[self.id] with fetched data
        (routing folders, track list, colors, etc.).  The GUI can
        then display this data in the Session Setup panel.

        ``ignore_cache`` requests a fresh fetch when the processor has an
        internal cache. Processors without a cache can ignore it.
        """

    def resolve_output_path(
        self,
        session: SessionContext,
        parent_widget=None,
    ) -> str | None:
        """Resolve the output path for a transfer, optionally prompting the user.

        Called by the GUI before starting a transfer worker.

        Return values
        -------------
        ``None``
            User cancelled — the GUI should abort.
        ``""``
            No file path needed (e.g. gRPC-based processors like Pro Tools).
        non-empty str
            Absolute path to write the output file to.

        The default implementation returns ``""`` (no path, no dialog).
        File-based processors (e.g. DAWproject) override this to compute a
        default path and show a save-file dialog.
        """
        return ""

    @abstractmethod
    def transfer(
        self,
        session: SessionContext,
        output_path: str,
        progress_cb=None,
        close_when_done: bool = True,
    ) -> list[DawCommandResult]:
        """Initial full push of session data to the DAW.

        Parameters
        ----------
        session:
            The current session context.
        output_path:
            Explicit destination path for the output file (e.g.
            ``/path/to/session.dawproject``).  The caller is responsible
            for choosing this path — supervised callers show a file dialog;
            unsupervised (batch) callers derive it from track/session data.
        progress_cb:
            Optional ``(current, total, message)`` callback.

        Returns the list of DawCommandResult for this batch.
        """

    @abstractmethod
    def sync(self, session: SessionContext) -> list[DawCommandResult]:
        """Incremental update — send only what changed since last transfer.

        Compares current session state against the snapshot stored by
        transfer() (in session.daw_state[self.id]) and sends only the
        deltas.  Same internal dispatch as transfer().
        """

    @abstractmethod
    def execute_commands(
        self, session: SessionContext, commands: list[DawCommand],
    ) -> list[DawCommandResult]:
        """Execute ad-hoc commands built by the GUI/CLI.

        Same internal dispatch as transfer()/sync(), but the commands
        are constructed externally — e.g. a GUI color picker builds
        DawCommand("set_color", "Kick", {"color_name": "Red"}) and
        hands it to the processor.

        Results are appended to session.daw_command_log.
        """
