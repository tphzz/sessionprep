"""Pro Tools DAW processor (PTSL-based)."""

from __future__ import annotations


import os
import time
from typing import Any

from ..models import ParamSpec
from ..daw_processor import DawProcessor
from ..models import DawCommand, DawCommandResult, SessionContext, TrackContext
from ..log import dbg
from ..ptx import PtxSession, atomic_write_bytes
from . import ptsl_helpers as ptslh
from .protools_ptx import (
    GAIN_MODE_CLIP_AND_FADER,
    ProToolsTrackStaging,
    execute_empty_folder_cleanup,
    execute_folder_output_routing,
    is_patchable_ptx_candidate,
    is_valid_ptx_file,
    ptx_config_params,
    resolve_protools_track_staging,
    run_post_close_ptx_enhancement,
)
from .protools_ptx_offline import execute_offline_ptx_creation
from .protools_ptx_plan import build_protools_creation_plan
from .protools_ptx_preview import read_template_folders_from_ptxt
from .ptsl_connection import (
    ProToolsConnectionSettings,
    check_connectivity as check_ptsl_connectivity,
    create_ptsl_engine,
)

import logging

log = logging.getLogger(__name__)

# Re-export color helpers from ptsl_helpers (private aliases for
# backward compatibility within this module).
_parse_argb = ptslh.parse_argb
_srgb_to_linear = ptslh.srgb_to_linear
_rgb_to_lab = ptslh.rgb_to_lab
_closest_palette_index = ptslh.closest_palette_index


def _ptsl_connection_params() -> list[ParamSpec]:
    return [
        ParamSpec(
            key="protools_temp_dir",
            type=str,
            default="",
            label="Temporary project directory",
            description=(
                "Directory for temporary Pro Tools sessions during Live PTSL inspection "
                "(PTSL only; leave empty for system temp)."
            ),
            widget_hint="path_picker_folder",
        ),
        ParamSpec(
            key="protools_company_name",
            type=str,
            default="github.com",
            label="Company Name",
            description="Company name sent during the PTSL handshake (PTSL only).",
        ),
        ParamSpec(
            key="protools_application_name",
            type=str,
            default="sessionprep",
            label="Application Name",
            description="Application name sent during the PTSL handshake (PTSL only).",
        ),
        ParamSpec(
            key="protools_host",
            type=str,
            default="localhost",
            label="Host",
            description="Hostname or IP address of the Pro Tools PTSL server (PTSL only).",
        ),
        ParamSpec(
            key="protools_port",
            type=int,
            default=31416,
            label="Port",
            description="Port number of the Pro Tools PTSL server (PTSL only).",
            min=1,
            max=65535,
        ),
        ParamSpec(
            key="protools_command_delay",
            type=float,
            default=0.5,
            label="Command Delay (s)",
            description=(
                "Seconds to wait between Pro Tools commands "
                "(folder select, import, etc.; PTSL only)."
            ),
            min=0.1,
            max=5.0,
        ),
        ParamSpec(
            key="protools_template_create_timeout",
            type=float,
            default=60.0,
            label="Template Create Timeout (s)",
            description=(
                "Seconds to wait for Pro Tools to create and open a session "
                "from a template (PTSL only)."
            ),
            min=5.0,
            max=300.0,
        ),
    ]


class ProToolsDawProcessor(DawProcessor):
    """DAW processor for Avid Pro Tools via the PTSL scripting SDK.

    Communicates with Pro Tools over a gRPC connection specified by
    host and port.  The company_name and application_name are sent
    during the PTSL handshake to identify the client.
    """

    id = "protools"
    name = "Pro Tools"

    def __init__(
        self,
        *,
        instance_index: int | None = None,
        instance_group: str = "",
        instance_name: str = "",
        instance_ptxt_path: str = "",
    ):
        self._instance_index = instance_index
        self._instance_group = instance_group
        self._instance_ptxt_path = instance_ptxt_path.strip()
        self._instance_name = (
            instance_name.strip()
            or (os.path.splitext(os.path.basename(self._instance_ptxt_path))[0] if self._instance_ptxt_path else "")
        )
        if instance_index is not None:
            self.id = f"protools_{instance_index}"
            if instance_group:
                self.name = f"Pro Tools \u2013 {instance_group} / {self._instance_name}"
            else:
                self.name = f"Pro Tools \u2013 {self._instance_name}"

    @classmethod
    def create_instances(
        cls,
        flat_config: dict[str, Any],
    ) -> list[ProToolsDawProcessor]:
        """Create one processor instance per configured template.

        Reads ``protools_templates`` from *flat_config*.  Each entry
        may provide ``group`` + ``name`` and/or an explicit ``ptxt_path`` (``path``).
        Returns an empty list when no templates are configured.
        """
        templates = flat_config.get("protools_templates", [])
        if not isinstance(templates, list):
            return []
        instances: list[ProToolsDawProcessor] = []
        for idx, tpl in enumerate(templates):
            if not isinstance(tpl, dict):
                continue
            group = str(tpl.get("group", "") or "").strip()
            name = str(tpl.get("name", "") or "").strip()
            ptxt_path = str(tpl.get("ptxt_path") or tpl.get("path") or "").strip()
            if not name and ptxt_path:
                name = os.path.splitext(os.path.basename(ptxt_path))[0]
            if not name or (not group and not ptxt_path):
                continue
            instances.append(
                cls(
                    instance_index=idx,
                    instance_group=group,
                    instance_name=name,
                    instance_ptxt_path=ptxt_path,
                )
            )
        return instances

    @classmethod
    def config_params(cls) -> list[ParamSpec]:
        base = super().config_params()
        transfer_mode = ParamSpec(
            key="protools_transfer_backend",
            type=str,
            default="offline_ptx",
            label="Transfer Mode",
            description=(
                "Write .ptx sessions and Audio Files/ directly offline (no Pro Tools required), "
                "or control a running Pro Tools instance via PTSL + post-close .ptx patching."
            ),
            choices=["offline_ptx", "ptsl_hybrid"],
            choice_labels={
                "offline_ptx": "Offline .ptx Writer",
                "ptsl_hybrid": "Live PTSL + .ptx Hybrid",
            },
        )
        return [base[0], transfer_mode, *base[1:]] + _ptsl_connection_params() + ptx_config_params()

    def configure(self, config: dict[str, Any]) -> None:
        saved = config.get(f"{self.id}_enabled")
        if saved is None:
            config[f"{self.id}_enabled"] = config.get("protools_enabled", True)
        super().configure(config)
        self._project_dir: str = config.get("protools_project_dir", "")
        self._temp_dir: str = config.get("protools_temp_dir", "")
        self._company_name: str = config.get("protools_company_name", "github.com")
        self._application_name: str = config.get(
            "protools_application_name", "sessionprep"
        )
        self._host: str = config.get("protools_host", "localhost")
        self._port: int = config.get("protools_port", 31416)
        self._command_delay: float = config.get("protools_command_delay", 0.5)
        self._template_create_timeout: float = config.get(
            "protools_template_create_timeout", 60.0
        )
        self._auto_route_folders: bool = bool(config.get("protools_auto_route_folders", True))
        self._gain_mode: str = str(config.get("protools_gain_mode", GAIN_MODE_CLIP_AND_FADER))
        self._auto_group_folders: bool = bool(config.get("protools_auto_group_folders", True))
        self._link_template_vcas: bool = bool(config.get("protools_link_template_vcas", True))
        self._group_type: str = str(config.get("protools_group_type", "Mix/Edit"))
        self._stage_bus_faders: bool = bool(config.get("protools_stage_bus_faders", True))
        self._pan_law_db: float = float(config.get("protools_pan_law_db", -3.0))
        self._remove_unused_aux_tracks: bool = bool(config.get("protools_remove_unused_aux_tracks", True))
        raw_backend = config.get("protools_transfer_backend")
        self._transfer_backend: str = (str(raw_backend).strip().lower() if raw_backend else "offline_ptx")
        self._connection_settings = ProToolsConnectionSettings.from_config(config)

    def check_connectivity(self) -> tuple[bool, str]:
        if self._transfer_backend == "offline_ptx":
            template_file, _mtime = self._template_file_and_mtime()
            if template_file is None or not is_valid_ptx_file(template_file):
                self._connected = False
                return False, f"Pro Tools template file not found or invalid: {template_file or self._instance_name}"
            if not self._project_dir or not os.path.isdir(self._project_dir):
                self._connected = False
                return False, "Pro Tools 'Project directory' is not configured or does not exist."
            self._connected = True
            return True, f"Offline .ptx ready ({template_file.name})"
        if self._transfer_backend != "ptsl_hybrid":
            self._connected = False
            return False, f"Unknown Pro Tools transfer backend: {self._transfer_backend!r}"
        ok, message = check_ptsl_connectivity(
            self._connection_settings,
            ready_timeout=25.0,
        )
        self._connected = ok
        return ok, message

    @property
    def transfer_backend(self) -> str:
        """Return the configured Pro Tools transfer backend (``'offline_ptx'`` or ``'ptsl_hybrid'``)."""
        return getattr(self, "_transfer_backend", "offline_ptx")

    @property
    def template_identity(self) -> tuple[str, ...]:
        if self._instance_ptxt_path:
            return ("protools", self._instance_group, self._instance_name, self._instance_ptxt_path)
        return ("protools", self._instance_group, self._instance_name)

    def _track_list_with_retry(self, engine, *, timeout: float, sleep_time: float):
        """Return the Pro Tools track list, retrying while a new session settles."""
        start_time = time.time()
        attempt = 0
        last_error: Exception | None = None
        while time.time() - start_time < timeout:
            attempt += 1
            try:
                dbg(f"PTSL track_list attempt {attempt}")
                tracks = engine.track_list()
                dbg(
                    "PTSL track_list succeeded: "
                    f"attempt={attempt} elapsed={time.time() - start_time:.1f}s "
                    f"tracks={len(tracks)}"
                )
                return tracks
            except Exception as exc:
                last_error = exc
                dbg(
                    "PTSL track_list not ready: "
                    f"attempt={attempt} elapsed={time.time() - start_time:.1f}s "
                    f"error={exc}"
                )
                time.sleep(sleep_time)

        msg = (
            "Pro Tools created the temporary session, but did not provide the "
            f"track list within {timeout:.1f} seconds."
        )
        if last_error is not None:
            msg = f"{msg} Last error: {last_error}"
        raise RuntimeError(msg)

    def resolve_template_ptxt_path(self):
        """Return the local ``.ptxt`` template path if supported on this OS, else ``None``."""
        template_file, _mtime = self._template_file_and_mtime()
        return template_file

    def _template_file_and_mtime(self):
        import platform
        from pathlib import Path

        if self._instance_ptxt_path:
            explicit = Path(self._instance_ptxt_path).expanduser()
            try:
                return explicit, (explicit.stat().st_mtime if explicit.is_file() else None)
            except OSError:
                return explicit, None
        if platform.system() not in ("Windows", "Darwin"):
            return None, None
        template_file = (
            Path.home() / "Documents" / "Pro Tools" / "Session Templates"
            / self._instance_group / f"{self._instance_name}.ptxt"
        )
        try:
            return template_file, (template_file.stat().st_mtime if template_file.is_file() else None)
        except OSError:
            return template_file, None

    def _load_template_cache(self):
        import json
        from pathlib import Path
        from ..paths import get_app_dir

        cache_file = Path(get_app_dir()) / "pt_template_cache.json"
        if not cache_file.is_file():
            return cache_file, {}
        try:
            with open(cache_file, encoding="utf-8") as f:
                return cache_file, json.load(f)
        except Exception:
            return cache_file, {}

    def _apply_fetched_folders(self, session: SessionContext, folders: list[dict[str, Any]]) -> None:
        pt_state = session.daw_state.get(self.id, {})
        old_assignments: dict[str, str] = pt_state.get("assignments", {})
        valid_ids = {f["id"] for f in folders}
        assignments = {fname: fid for fname, fid in old_assignments.items() if fid in valid_ids}
        new_state: dict[str, Any] = {"folders": folders, "assignments": assignments}
        old_overrides = pt_state.get("bus_fader_overrides")
        if isinstance(old_overrides, dict) and old_overrides:
            new_state["bus_fader_overrides"] = dict(old_overrides)
        old_order = pt_state.get("track_order")
        if isinstance(old_order, dict) and old_order:
            pruned_order = {fid: list(order) for fid, order in old_order.items() if fid in valid_ids and isinstance(order, list)}
            if pruned_order:
                new_state["track_order"] = pruned_order
        session.daw_state[self.id] = new_state

    def _delete_temp_session_folder(self, temp_session_name: str, temp_session_path: str) -> None:
        import shutil

        target_dir = os.path.join(self._temp_dir, temp_session_name)
        if not (
            os.path.isdir(target_dir)
            and os.path.dirname(os.path.abspath(target_dir)) == os.path.abspath(self._temp_dir)
            and not os.path.islink(target_dir)
            and os.path.isfile(temp_session_path)
        ):
            return
        for _ in range(10):
            try:
                dbg(f"Deleting temporary Pro Tools session folder: {target_dir!r}")
                shutil.rmtree(target_dir, ignore_errors=True)
                if not os.path.exists(target_dir):
                    dbg(f"Deleted temporary Pro Tools session folder: {target_dir!r}")
                    break
            except Exception:
                pass
            time.sleep(0.5)

    def _read_template_folders_from_engine(self, engine, temp_session_path: str, pt) -> list[dict[str, Any]]:
        if not ptslh.session_matches_path(engine, temp_session_path):
            raise RuntimeError("Pro Tools switched sessions before SessionPrep could read the template folders.")
        all_tracks = self._track_list_with_retry(
            engine, timeout=self._template_create_timeout, sleep_time=self._command_delay,
        )
        if not ptslh.session_matches_path(engine, temp_session_path):
            raise RuntimeError("Pro Tools switched sessions while SessionPrep was reading the template folders.")
        folders: list[dict[str, Any]] = []
        for track in all_tracks:
            if track.type in (pt.TrackType.RoutingFolder, pt.TrackType.BasicFolder):
                folders.append({
                    "id": track.id,
                    "name": track.name,
                    "folder_type": "routing" if track.type == pt.TrackType.RoutingFolder else "basic",
                    "index": track.index,
                    "parent_id": track.parent_folder_id or None,
                })
        dbg(f"Read Pro Tools folder structure: folders={len(folders)}")
        return folders

    def _close_temp_fetch_session(self, engine, temp_session_name: str, temp_session_path: str) -> tuple[bool, Exception | None]:
        try:
            if not ptslh.session_matches_path(engine, temp_session_path):
                raise RuntimeError("Pro Tools switched sessions before Fetch could close its temporary project.")
            dbg(f"Closing temporary Pro Tools session: {temp_session_name!r}")
            ptslh.close_session(engine)
            if ptslh.is_session_open(engine):
                raise RuntimeError("Pro Tools still reports an open session after CloseSession.")
            return True, None
        except Exception as exc:
            log.debug(f"Failed to close temp session: {exc}")
            dbg(f"Failed to close temporary Pro Tools session: {exc}")
            return False, exc

    def _fetch_folders_live(self, progress_cb) -> list[dict[str, Any]]:
        import uuid

        try:
            from ptsl import PTSL_pb2 as pt
        except ImportError as exc:
            raise RuntimeError("Pro Tools Fetch requires py-ptsl.") from exc

        engine = None
        temp_session_name = None
        temp_session_path = None
        temp_closed = False
        close_error = None
        try:
            if progress_cb:
                progress_cb(10, 100, "Connecting to Pro Tools...")
            engine = self._open_engine()
            if progress_cb:
                progress_cb(15, 100, "Waiting for Pro Tools to become ready...")
            if not ptslh.wait_for_host_ready(engine, timeout=25.0, sleep_time=self._command_delay):
                raise RuntimeError("Pro Tools is busy or not ready. Please bring its window to the front to wake it.")
            if ptslh.is_session_open(engine):
                raise RuntimeError("PRO_TOOLS_SESSION_OPEN")

            temp_session_name = f"SessionPrep_Temp_{uuid.uuid4().hex[:8]}"
            project_folder = os.path.join(self._temp_dir, temp_session_name)
            if os.path.lexists(project_folder):
                raise FileExistsError(f"The temporary project folder already exists: {project_folder}")
            if progress_cb:
                progress_cb(30, 100, f"Creating temporary session from template '{self._instance_group} / {self._instance_name}'...")
            ptslh.create_session_from_template(
                engine, temp_session_name, self._temp_dir,
                self._instance_group, self._instance_name, timeout=self._template_create_timeout,
            )
            temp_session_path = os.path.join(project_folder, f"{temp_session_name}.ptx")
            if progress_cb:
                progress_cb(65, 100, "Waiting for Pro Tools session to become ready...")
            if not ptslh.wait_for_host_ready(engine, timeout=self._template_create_timeout, sleep_time=self._command_delay):
                raise RuntimeError(
                    f"Pro Tools created the temporary session, but did not become ready within {self._template_create_timeout:.1f} seconds."
                )
            if progress_cb:
                progress_cb(70, 100, "Reading track folder structure...")
            folders = self._read_template_folders_from_engine(engine, temp_session_path, pt)
            if progress_cb:
                progress_cb(90, 100, "Cleaning up temporary session...")
        finally:
            if engine is not None:
                if temp_session_path and temp_session_name:
                    temp_closed, close_error = self._close_temp_fetch_session(engine, temp_session_name, temp_session_path)
                try:
                    engine.close()
                except Exception:
                    pass
            if temp_closed and temp_session_name and temp_session_path:
                self._delete_temp_session_folder(temp_session_name, temp_session_path)

        if close_error is not None:
            raise RuntimeError(f"Failed to close the temporary Pro Tools session: {close_error}") from close_error
        return folders

    def _save_template_cache(self, cache_file, cache_data: dict[str, Any], cache_key: str, mtime: float, folders: list[dict[str, Any]]) -> None:
        import json

        cache_data[cache_key] = {"mtime": mtime, "folders": folders}
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            dbg(f"Wrote Pro Tools template cache: {str(cache_file)!r}")
        except Exception as e:
            log.debug(f"Failed to write template cache: {e}")
            dbg(f"Failed to write Pro Tools template cache: {e}")

    def _try_fetch_folders_offline(self, template_file, progress_cb) -> list[dict[str, Any]] | None:
        host_low = self._host.strip().lower()
        local_allowed = (
            self._transfer_backend == "offline_ptx"
            or bool(self._instance_ptxt_path)
            or host_low in ("", "localhost", "127.0.0.1", "::1")
        )
        if not local_allowed or template_file is None or not is_valid_ptx_file(template_file):
            return None
        try:
            folders = read_template_folders_from_ptxt(template_file)
            if progress_cb:
                progress_cb(80, 100, f"Read template '{self._instance_name}' offline.")
            dbg(f"Read Pro Tools template folders offline from {str(template_file)!r}: folders={len(folders)}")
            return folders
        except Exception as exc:
            log.warning("Offline .ptxt template fetch failed for %s (%s); falling back to PTSL.", template_file, exc)
            return None

    def fetch(
        self,
        session: SessionContext,
        progress_cb=None,
        ignore_cache: bool = False,
    ) -> SessionContext:
        template_file, current_mtime = self._template_file_and_mtime()
        cache_file, cache_data = self._load_template_cache()
        if self._instance_ptxt_path:
            norm_path = os.path.normcase(os.path.normpath(os.path.expanduser(self._instance_ptxt_path)))
            cache_key = f"{self._instance_group}/{self._instance_name}|{norm_path}"
        else:
            cache_key = f"{self._instance_group}/{self._instance_name}"
        dbg(f"Pro Tools template fetch requested: key={cache_key} ignore_cache={ignore_cache}")

        if not ignore_cache and current_mtime is not None and cache_key in cache_data:
            entry = cache_data[cache_key]
            if entry.get("mtime") == current_mtime:
                dbg(f"Pro Tools template cache hit: key={cache_key!r} mtime={current_mtime}")
                if progress_cb:
                    progress_cb(100, 100, f"Loaded structure for '{self._instance_name}' from cache.")
                self._apply_fetched_folders(session, entry.get("folders", []))
                return session

        offline_folders = self._try_fetch_folders_offline(template_file, progress_cb) if current_mtime is not None else None
        if offline_folders is not None:
            folders = offline_folders
        elif self._transfer_backend == "offline_ptx" or (self._instance_ptxt_path and not self._instance_group):
            raise RuntimeError(
                f"Pro Tools template file not found or invalid for offline fetch: {template_file or self._instance_name}"
            )
        else:
            if not self._temp_dir:
                raise RuntimeError("The 'Temporary project directory' is not configured in Preferences.")
            if not os.path.isdir(self._temp_dir):
                raise RuntimeError(f"The configured temporary project directory does not exist: {self._temp_dir}")
            dbg(
                f"Pro Tools template cache miss: key={cache_key!r} "
                f"template_file={str(template_file) if template_file else None!r} mtime={current_mtime}"
            )
            folders = self._fetch_folders_live(progress_cb)
        self._apply_fetched_folders(session, folders)
        if current_mtime is not None:
            self._save_template_cache(cache_file, cache_data, cache_key, current_mtime, folders)
        if progress_cb:
            progress_cb(100, 100, "Fetch complete")
        return session

    def _resolve_group_color(
        self,
        group_name: str | None,
        session: SessionContext,
    ) -> str | None:
        """Return the ARGB hex for *group_name*, or ``None``."""
        if not group_name:
            return None
        groups = session.config.get("gui", {}).get("groups", [])
        color_name: str | None = None
        for g in groups:
            if g.get("name") == group_name:
                color_name = g.get("color")
                break
        if not color_name:
            return None
        colors = session.config.get("gui", {}).get("colors", [])
        for c in colors:
            if c.get("name") == color_name:
                return c.get("argb")
        return None

    def _open_engine(self):
        """Create and return a connected PTSL Engine."""
        return create_ptsl_engine(self._connection_settings)

    def _get_optimal_session_specs(self, session: SessionContext) -> tuple[str, str]:
        """Determine most common sample rate and bit depth from output tracks.

        Returns (sample_rate_enum, bit_depth_enum).
        """
        from collections import Counter

        rates = [t.samplerate for t in session.output_tracks if t.samplerate > 0]
        # bitdepth is string, e.g. "PCM_24". Try to extract numeric part.
        depths = []
        for t in session.output_tracks:
            bd = str(t.bitdepth).upper()
            if "32" in bd:
                depths.append(32)
            elif "24" in bd:
                depths.append(24)
            elif "16" in bd:
                depths.append(16)

        # Default fallback
        common_rate = Counter(rates).most_common(1)[0][0] if rates else 48000
        common_depth = Counter(depths).most_common(1)[0][0] if depths else 24

        rate_map = {
            44100: "SR_44100",
            48000: "SR_48000",
            88200: "SR_88200",
            96000: "SR_96000",
            176400: "SR_176400",
            192000: "SR_192000",
        }
        depth_map = {16: "Bit16", 24: "Bit24", 32: "Bit32Float"}

        return (
            rate_map.get(common_rate, "SR_48000"),
            depth_map.get(common_depth, "Bit24"),
        )

    def _ordered_transfer_work(
        self, session: SessionContext, pt_state: dict[str, Any], record,
    ) -> tuple[dict[str, Any], list[tuple[str, ProToolsTrackStaging, str, TrackContext]]]:
        assignments: dict[str, str] = pt_state.get("assignments", {})
        folder_map = {f["id"]: f for f in pt_state.get("folders", [])}
        track_order = pt_state.get("track_order", {})
        manifest_map = {e.entry_id: e for e in session.transfer_manifest}
        out_track_map = {t.filename: t for t in session.output_tracks}

        work: list[tuple[str, str]] = []
        seen: set[str] = set()
        for fid, ordered_names in track_order.items():
            for eid in ordered_names:
                if eid in assignments and assignments[eid] == fid:
                    work.append((eid, fid))
                    seen.add(eid)
        for eid, fid in sorted(assignments.items()):
            if eid not in seen:
                work.append((eid, fid))

        valid_work: list[tuple[str, ProToolsTrackStaging, str, TrackContext]] = []
        for entry_id, folder_id in work:
            entry = manifest_map.get(entry_id)
            if entry is None:
                record("validate_track", entry_id, error="The assigned track is missing from the transfer list.")
                continue
            if folder_id not in folder_map:
                record("validate_track", entry.daw_track_name,
                       error="The assigned Pro Tools folder is missing. Fetch and assign this track again.")
                continue
            track = out_track_map.get(entry.output_filename)
            if track is None:
                record("validate_track", entry.daw_track_name, error="The output track is missing.")
                continue
            try:
                staging = resolve_protools_track_staging(session, track, self._gain_mode)
            except ValueError as exc:
                record("validate_track", entry.daw_track_name, error=str(exc))
                continue
            valid_work.append((folder_id, staging, os.path.splitext(entry.daw_track_name)[0], track))
        return folder_map, valid_work

    def _create_transfer_session(self, engine, session: SessionContext, progress_cb, record) -> str:
        if not self._project_dir:
            raise RuntimeError("Pro Tools 'Project directory' is not configured.")
        if not os.path.isdir(self._project_dir):
            raise RuntimeError(f"Pro Tools 'Project directory' does not exist: {self._project_dir}")
        if not session.project_name:
            raise RuntimeError("Project name is empty.")
        if ptslh.is_session_open(engine):
            raise RuntimeError("A Pro Tools session is currently open. Save and close it in Pro Tools, then try again.")
        if progress_cb:
            progress_cb(5, 100, "Calculating audio specifications...")
        rate_enum, depth_enum = self._get_optimal_session_specs(session)
        if progress_cb:
            progress_cb(10, 100, f"Creating session '{session.project_name}'...")
        ptslh.create_session_from_template(
            engine, session.project_name, self._project_dir,
            self._instance_group, self._instance_name,
            sample_rate=rate_enum, bit_depth=depth_enum, timeout=self._template_create_timeout,
        )
        created_path = os.path.join(self._project_dir, session.project_name, f"{session.project_name}.ptx")
        record("create_session", session.project_name)
        if not ptslh.wait_for_host_ready(engine, timeout=self._template_create_timeout, sleep_time=self._command_delay):
            raise RuntimeError(
                f"Pro Tools did not become ready within {self._template_create_timeout:g} seconds after creating the session."
            )
        if not ptslh.session_matches_path(engine, created_path):
            raise RuntimeError("Pro Tools is no longer in the session created for this transfer.")
        return created_path

    def _build_group_palette(self, engine, session: SessionContext) -> dict[str, int]:
        pt_palette = ptslh.get_color_palette(engine)
        group_palette_idx: dict[str, int] = {}
        if not pt_palette:
            return group_palette_idx
        for entry in session.transfer_manifest:
            if entry.group and entry.group not in group_palette_idx:
                argb = self._resolve_group_color(entry.group, session)
                if argb:
                    idx = _closest_palette_index(argb, pt_palette)
                    if idx is not None:
                        group_palette_idx[entry.group] = idx
        return group_palette_idx

    def _import_transfer_audio(self, engine, valid_work, batch_job_id, record):
        all_filepaths = list(dict.fromkeys(media.filepath for _, media, _, _ in valid_work))
        clip_id_map: dict[str, list[str]] = {}
        import_failures: dict[str, str] = {}
        try:
            import_resp = ptslh.batch_import_audio(engine, all_filepaths, batch_job_id=batch_job_id, progress=25)
            time.sleep(self._command_delay)
            if import_resp:
                for entry in import_resp.get("file_list", []):
                    orig = entry.get("original_input_path", "")
                    dest_list = entry.get("destination_file_list", [])
                    if dest_list and dest_list[0].get("clip_id_list"):
                        clip_id_map[os.path.normcase(orig)] = list(dest_list[0]["clip_id_list"])
                for fail in import_resp.get("failure_list", []):
                    fail_path = fail.get("file_path") or fail.get("original_input_path", "")
                    import_failures[os.path.normcase(fail_path)] = (
                        fail.get("failure_message") or "Pro Tools rejected the audio file."
                    )
        except Exception as exc:
            record("batch_import", error=str(exc))
            return None, None
        return clip_id_map, import_failures

    def _spot_transfer_tracks(
        self, engine, valid_work, folder_map, clip_id_map, import_failures,
        group_palette_idx, batch_job_id, progress_cb, record,
    ):
        work_by_folder: dict[str, list[tuple[str, TrackContext, list[str], ProToolsTrackStaging]]] = {}
        for folder_id, staging, name, track in valid_work:
            key = os.path.normcase(staging.filepath)
            clip_ids = clip_id_map.get(key)
            if not clip_ids or key in import_failures:
                record("import_audio", name, error=import_failures.get(
                    key, f"Pro Tools returned no clips for '{staging.filepath}'."))
                continue
            work_by_folder.setdefault(folder_id, []).append((name, track, clip_ids, staging))

        total_spot = sum(len(items) for items in work_by_folder.values())
        if total_spot == len(valid_work):
            record("batch_import")

        color_groups: dict[int, list[str]] = {}
        created_tracks: list[tuple[str, str, ProToolsTrackStaging, TrackContext]] = []
        attempted = 0
        for fid, folder_items in work_by_folder.items():
            folder_name = folder_map[fid]["name"]
            for name, track, clip_ids, staging in reversed(folder_items):
                attempted += 1
                pct = 30 + int(50 * attempted / total_spot)
                track_format = "TF_Mono" if track.channels == 1 else "TF_Stereo"
                try:
                    tid, created_name = ptslh.create_track_with_name(
                        engine, name, track_format,
                        insertion_point_track_name=folder_name, insertion_point_position="TIPoint_First",
                        batch_job_id=batch_job_id, progress=pct,
                    )
                    ptslh.spot_clips(engine, clip_ids, tid, batch_job_id=batch_job_id, progress=pct)
                    record("spot_clip", created_name)
                    created_tracks.append((created_name, tid, staging, track))
                    if track.group in group_palette_idx:
                        color_groups.setdefault(group_palette_idx[track.group], []).append(created_name)
                except Exception as exc:
                    record("spot_clip", name, error=str(exc))
                if progress_cb:
                    progress_cb(pct, 100, f"Transferred {len(created_tracks)}/{total_spot} tracks")
        return color_groups, created_tracks

    def _apply_transfer_colors_and_faders(self, engine, color_groups, created_tracks, batch_job_id, record) -> None:
        for cidx, names in color_groups.items():
            try:
                ptslh.colorize_tracks(engine, names, cidx, batch_job_id=batch_job_id, progress=90)
                record("colorize_tracks", ", ".join(names))
            except Exception as exc:
                record("colorize_tracks", ", ".join(names), error=str(exc))

        if not ptslh.VOLUME_AUTOMATION_ENABLED:
            return
        for name, t_id, staging, _track in created_tracks:
            fader_db = staging.legacy_ptsl_fader_db
            if fader_db == 0.0:
                continue
            try:
                ptslh.set_track_volume(engine, t_id, fader_db, batch_job_id=batch_job_id, progress=95)
                record("set_fader", name, fader_db=fader_db)
            except Exception as exc:
                detail = str(exc)
                error = f"Could not write volume automation at sample 0. {detail}"
                command_error = detail.casefold()
                unavailable = (
                    "private api has not been enabled in this session" in command_error
                    or "pt_unsupportedcommand" in command_error
                )
                if unavailable:
                    error += " SessionPrep stopped volume automation writes for this transfer."
                record("set_fader", name, fader_db=fader_db, error=error)
                if unavailable:
                    break

    def _apply_folder_output_routing(
        self, engine, created_session_path: str, created_tracks, progress_cb, record,
    ) -> None:
        if not self._auto_route_folders or not created_tracks or not is_valid_ptx_file(created_session_path):
            return
        if not hasattr(engine, "track_list"):
            return
        try:
            if not ptslh.session_matches_path(engine, created_session_path):
                raise RuntimeError("Pro Tools switched sessions before SessionPrep could route track outputs.")
            if progress_cb:
                progress_cb(92, 100, "Routing tracks to parent folder busses...")
            ptslh.save_session(engine)
            ptx = PtxSession.from_file(created_session_path)
            live_tracks = engine.track_list()
            created_ids = {tid for _, tid, _, _ in created_tracks}
            routed, route_errors, warnings = execute_folder_output_routing(engine, ptx, live_tracks, created_ids)
            for warn in warnings:
                log.warning("Pro Tools routing: %s", warn)
            for entry in routed:
                record("route_output", entry.track_name, bus=entry.target_bus_name, signalpath_id=entry.target_signalpath_id)
            for target_label, err in route_errors:
                record("route_output", target_label, error=err)
        except Exception as exc:
            record("route_output", "", error=str(exc))

    def _remove_empty_folders_and_vcas(
        self, engine, created_session_path: str, created_tracks, progress_cb, record,
    ) -> None:
        if not created_tracks or not is_valid_ptx_file(created_session_path):
            return
        if not hasattr(engine, "track_list"):
            return
        try:
            if not ptslh.session_matches_path(engine, created_session_path):
                raise RuntimeError("Pro Tools switched sessions before SessionPrep could remove empty folders.")
            if progress_cb:
                progress_cb(94, 100, "Removing empty folders, unused VCAs, and orphaned Auxes...")
            ptslh.save_session(engine)
            ptx = PtxSession.from_file(created_session_path)
            live_tracks = engine.track_list()
            del_folders, del_vcas, del_auxes, del_errors = execute_empty_folder_cleanup(
                engine, ptx, live_tracks, remove_unused_aux_tracks=self._remove_unused_aux_tracks,
            )
            for fname in del_folders:
                record("remove_empty_folder", fname)
            for vname in del_vcas:
                record("remove_unused_vca", vname)
            for aname in del_auxes:
                record("remove_unused_aux", aname)
            for op, target_label, err in del_errors:
                record(op, target_label, error=err)
        except Exception as exc:
            record("remove_empty_folder", "", error=str(exc))

    def _rollback_and_reopen(
        self, engine, session_name: str, created_session_path: str, orig_bytes: bytes | None, exc: Exception, record,
    ) -> None:
        if orig_bytes is None:
            record("reopen_session", session_name, error=f"Could not reopen session: {exc}")
            return
        try:
            atomic_write_bytes(created_session_path, orig_bytes)
        except Exception as rb_exc:
            record(
                "reopen_session", session_name,
                error=f"Reopen failed ({exc}); rollback of original .ptx also failed: {rb_exc}",
            )
            return
        try:
            ptslh.open_session(engine, created_session_path)
        except Exception as reopen_exc:
            log.warning("Reopen of rolled-back .ptx also failed: %s", reopen_exc)
        record("reopen_session", session_name, error=f"Reopen failed after .ptx patch; rolled back: {exc}")

    def _apply_post_close_ptx_pass(
        self, engine, session: SessionContext, created_session_path: str, created_tracks,
        close_when_done: bool, progress_cb, record,
    ) -> None:
        if progress_cb:
            progress_cb(96, 100, "Applying Clip Gain, VCA groups, and bus staging...")
        orig_bytes: bytes | None = None
        meta: dict[str, Any] = {"clip_gains": {}, "volumes": {}, "groups": []}
        graph = None
        overrides = session.daw_state.get(self.id, {}).get("bus_fader_overrides")
        try:
            orig_bytes, meta, graph = run_post_close_ptx_enhancement(
                created_session_path, session, created_tracks,
                gain_mode=self._gain_mode,
                auto_group_folders=self._auto_group_folders,
                link_template_vcas=self._link_template_vcas,
                group_type=self._group_type,
                stage_bus_faders=self._stage_bus_faders,
                pan_law_db=self._pan_law_db,
                bus_fader_overrides=overrides if isinstance(overrides, dict) else None,
            )
        except Exception as exc:
            record("patch_ptx", session.project_name, error=str(exc))
            if not close_when_done:
                try:
                    ptslh.open_session(engine, created_session_path)
                except Exception as reopen_exc:
                    record("reopen_session", session.project_name, error=f"Could not reopen session after patch failure: {reopen_exc}")
            return
        if not close_when_done:
            try:
                ptslh.open_session(engine, created_session_path)
            except Exception as exc:
                self._rollback_and_reopen(engine, session.project_name, created_session_path, orig_bytes, exc, record)
                return
        if graph is not None:
            session.daw_state.setdefault(self.id, {})["signal_flow"] = graph.to_dict()
        for tname, info in (meta.get("cleared_inputs") or {}).items():
            if info.get("cleared"):
                record("clear_input", tname, previous_input=info.get("previous_input"))
        for tname, info in meta["clip_gains"].items():
            record("set_clip_gain", tname, clip_gain_db=info["target_db"])
        for tname, info in (meta.get("slot_a_trims") or {}).items():
            record(
                "set_slot_a_trim", tname,
                plugin=info.get("plugin"), trim_db=info.get("target_db"),
                activated=info.get("activated"), auto_inserted=info.get("auto_inserted", False),
            )
        for tname, info in meta["volumes"].items():
            record("set_fader_ptx", tname, fader_db=info["target_db"])
        for grp in meta["groups"]:
            record("set_group", grp["name"], vca_track=grp.get("vca_track"), members=grp.get("member_tracks", []))

    def _needs_post_close_ptx_pass(self, created_session_path: str, created_tracks) -> bool:
        return bool(created_tracks and is_patchable_ptx_candidate(created_session_path))

    def _finalize_transfer_session(
        self, engine, session: SessionContext, created_session_path: str, created_tracks,
        close_when_done: bool, progress_cb, record,
    ) -> None:
        needs_ptx_pass = self._needs_post_close_ptx_pass(created_session_path, created_tracks)
        operation = "close_session" if close_when_done else "save_session"
        try:
            if progress_cb:
                msg = "Saving and closing session..." if close_when_done else "Saving session..."
                progress_cb(98, 100, msg)
            if not ptslh.session_matches_path(engine, created_session_path):
                raise RuntimeError("Pro Tools switched sessions before SessionPrep could save this project.")
            if close_when_done or needs_ptx_pass:
                ptslh.close_session(engine, save_on_close=True, delay=self._command_delay)
                if needs_ptx_pass and ptslh.is_session_open(engine):
                    raise RuntimeError("Pro Tools still reports an open session after CloseSession.")
            else:
                ptslh.save_session(engine)
            record(operation, session.project_name)
        except Exception as exc:
            record(operation, session.project_name, error=str(exc))
            return
        if needs_ptx_pass:
            if is_valid_ptx_file(created_session_path):
                self._apply_post_close_ptx_pass(
                    engine, session, created_session_path, created_tracks, close_when_done, progress_cb, record,
                )
            elif not close_when_done:
                try:
                    ptslh.open_session(engine, created_session_path)
                except Exception as exc:
                    record("reopen_session", session.project_name, error=str(exc))

    def _execute_transfer_ptsl_hybrid(
        self,
        session: SessionContext,
        folder_map: dict[str, Any],
        valid_work: list[tuple[str, ProToolsTrackStaging, str, TrackContext]],
        progress_cb,
        close_when_done: bool,
        record,
    ) -> None:
        engine = None
        batch_job_id: str | None = None
        created_session_path: str | None = None
        created_tracks: list[tuple[str, str, ProToolsTrackStaging, TrackContext]] = []
        try:
            if progress_cb:
                progress_cb(0, 100, "Connecting to Pro Tools...")
            engine = self._open_engine()
            if progress_cb:
                progress_cb(2, 100, "Waiting for Pro Tools to become ready...")
            if not ptslh.wait_for_host_ready(engine, timeout=25.0, sleep_time=self._command_delay):
                raise RuntimeError("Pro Tools is busy or not ready. Please bring its window to the front to wake it.")

            created_session_path = self._create_transfer_session(engine, session, progress_cb, record)
            group_palette_idx = self._build_group_palette(engine, session)
            batch_job_id = ptslh.create_batch_job(engine, "SessionPrep Create", f"Importing {len(valid_work)} tracks")
            if progress_cb:
                progress_cb(20, 100, "Importing audio to clip list...")

            clip_id_map, import_failures = self._import_transfer_audio(engine, valid_work, batch_job_id, record)
            if clip_id_map is None:
                return
            color_groups, created_tracks = self._spot_transfer_tracks(
                engine, valid_work, folder_map, clip_id_map, import_failures,
                group_palette_idx, batch_job_id, progress_cb, record,
            )
            self._apply_transfer_colors_and_faders(engine, color_groups, created_tracks, batch_job_id, record)
            if batch_job_id:
                try:
                    ptslh.complete_batch_job(engine, batch_job_id)
                    batch_job_id = None
                except Exception as exc:
                    record("complete_batch", error=str(exc))
            self._apply_folder_output_routing(engine, created_session_path, created_tracks, progress_cb, record)
            self._remove_empty_folders_and_vcas(engine, created_session_path, created_tracks, progress_cb, record)
        except Exception as exc:
            record("transfer" if created_session_path else "create_session", session.project_name, error=str(exc))
        finally:
            if engine is not None:
                if batch_job_id:
                    ptslh.cancel_batch_job(engine, batch_job_id)
                if created_session_path:
                    self._finalize_transfer_session(
                        engine, session, created_session_path, created_tracks, close_when_done, progress_cb, record,
                    )
                engine.close()

    def _record_offline_write_results(self, session: SessionContext, output, record) -> None:
        plan, staged, res = output.plan, output.staged, output.write_result
        record("create_session", plan.session_name, path=str(staged.ptx_path))
        for sf in staged.staged_files:
            record("stage_audio", sf.filename, converted=sf.converted, sample_rate=sf.sample_rate, bit_depth=sf.bit_depth)
        for pt in plan.tracks:
            record("create_track", pt.track_name, folder=pt.parent_folder_name, bus=pt.output_bus_name)
        for ftrk in plan.cleanup_plan.empty_folders:
            record("remove_empty_folder", ftrk.name)
        for vtrk in plan.cleanup_plan.unused_vcas:
            record("remove_unused_vca", vtrk.name)
        for atrk in plan.cleanup_plan.orphaned_auxes:
            record("remove_unused_aux", atrk.name)
        meta = res.patch_summary
        for tname, info in (meta.get("clip_gains") or {}).items():
            record("set_clip_gain", tname, clip_gain_db=info["target_db"])
        for tname, info in (meta.get("slot_a_trims") or {}).items():
            record(
                "set_slot_a_trim", tname, plugin=info.get("plugin"),
                trim_db=info.get("target_db"), activated=info.get("activated"),
                auto_inserted=info.get("auto_inserted", False),
            )
        for tname, info in (meta.get("volumes") or {}).items():
            record("set_fader_ptx", tname, fader_db=info["target_db"])
        for grp in meta.get("groups") or ():
            record("set_group", grp["name"], vca_track=grp.get("vca_track"), members=grp.get("member_tracks", []))
        if plan.signal_flow is not None:
            session.daw_state.setdefault(self.id, {})["signal_flow"] = plan.signal_flow.to_dict()

    def _execute_transfer_offline_ptx(
        self,
        session: SessionContext,
        folder_map: dict[str, Any],
        valid_work: list[tuple[str, ProToolsTrackStaging, str, TrackContext]],
        progress_cb,
        record,
    ) -> None:
        del folder_map, valid_work
        template_file, _mtime = self._template_file_and_mtime()
        if template_file is None or not is_valid_ptx_file(template_file):
            record(
                "create_session", session.project_name,
                error=f"Pro Tools template file not found or invalid: {template_file or self._instance_name}",
            )
            return
        if not self._project_dir or not os.path.isdir(self._project_dir):
            record(
                "create_session", session.project_name,
                error="Pro Tools 'Project directory' is not configured or does not exist.",
            )
            return
        pt_state = session.daw_state.get(self.id, {})
        overrides = pt_state.get("bus_fader_overrides")
        try:
            if progress_cb:
                progress_cb(5, 100, "Planning offline Pro Tools session...")
            plan = build_protools_creation_plan(
                template_file, session,
                pt_state.get("assignments") or {},
                list(pt_state.get("folders") or ()),
                track_order=pt_state.get("track_order"),
                auto_route=self._auto_route_folders,
                gain_mode=self._gain_mode,
                auto_group_folders=self._auto_group_folders,
                link_template_vcas=self._link_template_vcas,
                group_type=self._group_type,
                stage_bus_faders=self._stage_bus_faders,
                pan_law_db=self._pan_law_db,
                remove_unused_aux_tracks=self._remove_unused_aux_tracks,
                bus_fader_overrides=overrides if isinstance(overrides, dict) else None,
            )
            for fname, reason in plan.excluded_tracks.items():
                record("plan_track", fname, error=reason)
            for warn in plan.warnings:
                log.warning("Offline Pro Tools planning: %s", warn)
            if not plan.tracks:
                record(
                    "create_session", session.project_name,
                    error="No valid audio tracks remained to create after offline Pro Tools planning.",
                )
                return
            output = execute_offline_ptx_creation(plan, self._project_dir, progress_cb=progress_cb)
            self._record_offline_write_results(session, output, record)
        except Exception as exc:
            record("create_session", session.project_name, error=str(exc))

    def transfer(
        self,
        session: SessionContext,
        output_path: str,
        progress_cb=None,
        close_when_done: bool = True,
    ) -> list[DawCommandResult]:
        """Create a new Pro Tools session from a template, import audio, route, and stage gains."""
        log.debug("transfer() called (backend=%s)", self._transfer_backend)
        if self._transfer_backend not in ("ptsl_hybrid", "offline_ptx"):
            return [
                DawCommandResult(
                    DawCommand("transfer", session.project_name, {}),
                    False,
                    f"Unknown Pro Tools transfer backend: {self._transfer_backend!r}",
                )
            ]
        if self._transfer_backend != "offline_ptx":
            try:
                from ptsl import PTSL_pb2 as pt  # noqa: F401 – validates install
            except ImportError:
                log.debug("py-ptsl not installed")
                return [DawCommandResult(DawCommand("transfer", "", {}), False, "py-ptsl package not installed")]

        pt_state = session.daw_state.get(self.id, {})
        if not pt_state.get("assignments"):
            log.debug("No assignments, returning early")
            return []

        results: list[DawCommandResult] = []

        def record(operation: str, target: str = "", *, error: str | None = None, **params: Any) -> None:
            results.append(DawCommandResult(DawCommand(operation, target, params), error is None, error))

        folder_map, valid_work = self._ordered_transfer_work(session, pt_state, record)
        if not valid_work:
            return results

        if self._transfer_backend == "offline_ptx":
            self._execute_transfer_offline_ptx(session, folder_map, valid_work, progress_cb, record)
        else:
            self._execute_transfer_ptsl_hybrid(session, folder_map, valid_work, progress_cb, close_when_done, record)

        if progress_cb and all(result.success for result in results):
            progress_cb(100, 100, "Project creation complete")
        return results

    def sync(self, session: SessionContext) -> list[DawCommandResult]:
        return []

    def execute_commands(
        self,
        session: SessionContext,
        commands: list[DawCommand],
    ) -> list[DawCommandResult]:
        return []
