"""DAWproject file-based DAW processor."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
from threading import RLock
from functools import wraps
import zipfile
from typing import Any

from ..daw_processor import DawProcessor, resolve_transfer_audio
from ..file_io import atomic_writer
from ..models import DawCommand, DawCommandResult, SessionContext


log = logging.getLogger(__name__)
_model_registry_lock = RLock()


def _with_model_registry(method):
    """The file-format library shares its object IDs across project instances."""
    @wraps(method)
    def locked(*args, **kwargs):
        with _model_registry_lock:
            return method(*args, **kwargs)
    return locked


def _save_template(source, root, output_path, project, additions, replace_existing):
    """Append new tracks to the original XML, retaining template assets and routing."""
    from lxml import etree

    template = Path(source.filename)
    with atomic_writer(output_path, binary=True, replace_existing=replace_existing) as stream:
        # Relative external files in the template are relative to its original directory.
        for item in root.iterfind(".//*[@path]"):
            path = item.get("path")
            if (item.get("external") in ("true", "1") and path and "://" not in path
                    and not Path(path).is_absolute() and not PureWindowsPath(path).is_absolute()):
                item.set("path", (template.parent / path).resolve().as_posix())

        folders = {}

        def index_tracks(elements, tracks):
            for element, track in zip(elements, tracks, strict=True):
                folders[track.id] = element
                index_tracks(element.findall("Track"), track.tracks)

        index_tracks(root.findall("Structure/Track"), (t for t in project.structure if hasattr(t, "tracks")))
        arrangement = root.find("Arrangement")
        if arrangement is None:
            arrangement = etree.Element("Arrangement")
            root.insert(root.index(root.find("Structure")) + 1, arrangement)
        lanes = arrangement.find("Lanes")
        if lanes is None:
            lanes = etree.Element("Lanes", timeUnit="seconds")
            arrangement.insert(0, lanes)
        for folder, track, lane in additions:
            element = folders[folder.id]
            if folder.channel is not None:
                element.find("Channel").set("id", folder.channel.id)
            element.append(track.to_xml())
            lanes.append(lane.to_xml())

        with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
            archive.comment = source.comment
            archive.writestr("project.xml", etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
            if "metadata.xml" not in source.namelist():
                archive.writestr("metadata.xml", "<MetaData/>")
            for member in source.infolist():
                if member.filename == "project.xml":
                    continue
                with source.open(member) as old, archive.open(member, "w") as new:
                    shutil.copyfileobj(old, new)


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear gain (0 dB → 1.0)."""
    return math.pow(10.0, db / 20.0)


def _argb_to_rgb_hex(argb: str) -> str | None:
    """Convert an ARGB hex string (e.g. 'FF3399CC') to '#rrggbb'."""
    argb = argb.lstrip("#")
    if len(argb) == 8:
        return f"#{argb[2:]}"
    if len(argb) == 6:
        return f"#{argb}"
    return None


class DawProjectDawProcessor(DawProcessor):
    """DAW processor that writes .dawproject files.

    DAWproject is an open interchange format for DAW sessions.
    This processor generates .dawproject files from the session state
    rather than communicating with a running DAW instance.

    Each configured template becomes a separate instance with its own
    ``id`` and ``name``, created via :meth:`create_instances`.
    """

    id = "dawproject"
    name = "DAWproject"
    fader_ceiling_db: float = 6.0

    @property
    def template_identity(self) -> tuple[str, ...]:
        return ("dawproject", self._template_name, self._template_path)

    def __init__(
        self,
        *,
        instance_index: int | None = None,
        template_name: str = "",
        template_path: str = "",
        template_fader_ceiling_db: float = 6.0,
    ):
        self._instance_index = instance_index
        self._template_name = template_name
        self._template_path = template_path
        if instance_index is not None:
            self.id = f"dawproject_{instance_index}"
            self.name = f"DAWproject \u2013 {template_name}"
            self.fader_ceiling_db = template_fader_ceiling_db

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create_instances(
        cls, flat_config: dict[str, Any],
    ) -> list[DawProjectDawProcessor]:
        """Create one processor instance per configured template.

        Reads ``dawproject_templates`` from *flat_config*.  Each entry
        is a dict with keys ``name``, ``template_path``, and optionally
        ``fader_ceiling_db``.  Returns an empty list when no templates
        are configured (the base "DAWproject" entry in the dropdown is
        suppressed in that case).
        """
        templates = flat_config.get("dawproject_templates", [])
        if not isinstance(templates, list):
            return []
        instances: list[DawProjectDawProcessor] = []
        for idx, tpl in enumerate(templates):
            if not isinstance(tpl, dict):
                continue
            name = tpl.get("name", "").strip()
            path = tpl.get("template_path", "").strip()
            ceiling = float(tpl.get("fader_ceiling_db", 6.0))
            if not name or not path:
                continue
            instances.append(cls(
                instance_index=idx,
                template_name=name,
                template_path=path,
                template_fader_ceiling_db=ceiling,
            ))
        return instances

    # ── Config ─────────────────────────────────────────────────────────

    def configure(self, config: dict[str, Any]) -> None:
        # For template instances the enabled toggle is governed by the
        # base dawproject_enabled key.
        saved = config.get(f"{self.id}_enabled")
        if saved is None:
            config[f"{self.id}_enabled"] = config.get("dawproject_enabled", True)
        super().configure(config)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def check_connectivity(self) -> tuple[bool, str]:
        if not self._template_path:
            return False, "No template file configured."
        if not os.path.isfile(self._template_path):
            return False, f"Template not found: {self._template_path}"
        try:
            with zipfile.ZipFile(self._template_path, "r") as zf:
                if "project.xml" not in zf.namelist():
                    return False, "Template ZIP missing project.xml."
        except zipfile.BadZipFile:
            return False, "Template file is not a valid ZIP archive."
        return True, f"Template OK: {os.path.basename(self._template_path)}"

    @_with_model_registry
    def fetch(
        self,
        session: SessionContext,
        *,
        ignore_cache: bool = False,
    ) -> SessionContext:
        try:
            from dawproject import (  # noqa: F401
                ContentType, DawProject, Referenceable,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dawproject package not installed. "
                "Install SessionPrep with its gui extra."
            ) from exc

        Referenceable.reset_id()
        project = DawProject.load_project(self._template_path)

        folders: list[dict[str, Any]] = []
        self._walk_structure(
            project.structure, folders, parent_id=None, counter=[0])

        # Preserve existing assignments where folder IDs still match
        dp_state = session.daw_state.get(self.id, {})
        old_assignments: dict[str, str] = dp_state.get("assignments", {})
        valid_ids = {f["id"] for f in folders}
        assignments = {
            fname: fid for fname, fid in old_assignments.items()
            if fid in valid_ids
        }

        session.daw_state[self.id] = {
            "folders": folders,
            "assignments": assignments,
        }
        return session

    def _walk_structure(
        self,
        tracks: list,
        folders: list[dict[str, Any]],
        parent_id: str | None,
        counter: list[int],
    ) -> None:
        """Recursively collect folder tracks from the project structure."""
        from dawproject import ContentType

        for track in tracks:
            ct_values = set()
            for ct in getattr(track, "content_type", []):
                if isinstance(ct, ContentType):
                    ct_values.add(ct)
                else:
                    try:
                        ct_values.add(ContentType(ct))
                    except ValueError:
                        pass

            if ContentType.TRACKS in ct_values:
                folder_type = "routing" if track.channel else "basic"
                folders.append({
                    "id": track.id,
                    "name": track.name or "(unnamed)",
                    "folder_type": folder_type,
                    "index": counter[0],
                    "parent_id": parent_id,
                })
                counter[0] += 1
                # Recurse into nested tracks
                self._walk_structure(
                    track.tracks, folders, parent_id=track.id,
                    counter=counter)

    def resolve_output_path(
        self,
        session: SessionContext,
        parent_widget=None,
    ) -> str | None:
        """Compute default .dawproject path and show a save-file dialog.

        Returns the chosen path, or ``None`` if the user cancelled.
        """
        from PySide6.QtWidgets import QFileDialog
        import os

        source_dir = session.config.get("_source_dir", "")
        output_folder = session.config.get("_output_folder", "sp_02_prepared")
        output_dir = os.path.join(source_dir, output_folder) if source_dir else ""

        # Use project name if set, otherwise fallback to template name
        safe_name = session.project_name if getattr(session, "project_name", "") else (self._template_name or self.name or "dawproject")
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in safe_name)
        default_path = os.path.join(output_dir, f"{safe_name}.dawproject") \
            if output_dir else f"{safe_name}.dawproject"

        # If in batch mode, we do NOT want a blocking dialog
        batch_mode = False
        if parent_widget:
            batch_action = getattr(parent_widget, "_batch_mode_action", None)
            if batch_action and batch_action.isChecked():
                batch_mode = True

        if batch_mode:
            # Just return the computed path directly, do not block with a dialog.
            return default_path

        path, _ = QFileDialog.getSaveFileName(
            parent_widget, "Save DAWproject", default_path,
            "DAWproject (*.dawproject);;All Files (*)",
        )
        return path if path else None

    @_with_model_registry
    def transfer(
        self,
        session: SessionContext,
        output_path: str,
        progress_cb=None,
        close_when_done: bool = True,
    ) -> list[DawCommandResult]:
        if not session.daw_state.get(self.id, {}).get("assignments"):
            return []
        try:
            from dawproject import Project, Referenceable
            from lxml import etree
        except ImportError:
            return [DawCommandResult(
                command=DawCommand("transfer", "", {}),
                success=False,
                error="dawproject package not installed",
            )]

        try:
            template = zipfile.ZipFile(self._template_path)
        except (OSError, zipfile.BadZipFile) as exc:
            return [DawCommandResult(DawCommand("load_template", "", {}), False, str(exc))]
        with template:
            try:
                root = etree.fromstring(template.read("project.xml"))
                Referenceable.reset_id()
                project = Project.from_xml(root)
                # The template may contain device or automation IDs that the
                # library does not deserialize. Reserve those IDs as well.
                reserved = (n[2:] for n in root.xpath("//@id") if n.startswith("id") and n[2:].isdecimal())
                Referenceable.ID = max(Referenceable.ID, max(map(int, reserved), default=-1) + 1)
            except Exception as exc:
                return [DawCommandResult(DawCommand("load_template", "", {}), False, f"Failed to load template: {exc}")]
            return self._add_tracks(session, output_path, progress_cb, template, root, project)

    def _add_tracks(self, session, output_path, progress_cb, template, root, project):
        from dawproject import Audio, ContentType, FileReference, Lanes, MixerRole, TimeUnit, Utility

        dp_state = session.daw_state.get(self.id, {})
        assignments: dict[str, str] = dp_state.get("assignments", {})
        daw_folders = dp_state.get("folders", [])
        track_order = dp_state.get("track_order", {})

        results: list[DawCommandResult] = []

        # Build folder ID → Track object lookup from the loaded project
        folder_track_map: dict[str, Any] = {}
        self._build_folder_map(project.structure, folder_track_map)

        # Build lookups
        folder_dict_map = {f["id"]: f for f in daw_folders}
        manifest_map = {
            e.entry_id: e for e in session.transfer_manifest}
        out_track_map = {
            t.filename: t for t in session.output_tracks}

        # Build ordered work list: [(entry_id, folder_id), ...]
        work: list[tuple[str, str]] = []
        seen: set[str] = set()
        for fid, ordered_names in track_order.items():
            for eid in ordered_names:
                if eid not in seen and assignments.get(eid) == fid:
                    work.append((eid, fid))
                    seen.add(eid)
        for eid, fid in sorted(assignments.items()):
            if eid not in seen:
                work.append((eid, fid))

        total = len(work)
        if progress_cb:
            progress_cb(0, total, "Building DAWproject\u2026")

        additions = []

        # ── Create tracks and clips ─────────────────────────────
        for step, (eid, fid) in enumerate(work):
            folder_track = folder_track_map.get(fid)
            folder_dict = folder_dict_map.get(fid)
            entry = manifest_map.get(eid)
            out_tc = (
                out_track_map.get(entry.output_filename)
                if entry else None
            )

            if not folder_track or not folder_dict or not entry or not out_tc:
                results.append(DawCommandResult(
                    command=DawCommand("add_track", eid,
                                       {"folder_id": fid}),
                    success=False,
                    error=f"Folder, manifest entry, or output track not found: "
                          f"{fid} / {eid}"))
                continue

            try:
                media = resolve_transfer_audio(session, out_tc)
            except ValueError as exc:
                results.append(DawCommandResult(
                    command=DawCommand("add_track", eid, {"folder_id": fid}),
                    success=False, error=str(exc)))
                continue
            log.debug("%s: file=%s, fader=%+.2f dB", eid, media.filepath, media.fader_db)
            volume_linear = _db_to_linear(media.fader_db)

            # Resolve group color → #rrggbb
            track_color = self._resolve_track_color(entry.group, session)

            # Create the track with channel
            track_name = os.path.splitext(entry.daw_track_name)[0]
            new_track = Utility.create_track(
                name=track_name,
                content_types={ContentType.AUDIO},
                mixer_role=MixerRole.REGULAR,
                volume=volume_linear,
                pan=0.5,
            )

            # Set color
            if track_color:
                new_track.color = track_color

            # Route to folder's channel
            if folder_track.channel is not None:
                new_track.channel.destination = folder_track.channel

            # Create audio clip in arrangement
            audio = Audio(
                time_unit=TimeUnit.SECONDS,
                file=FileReference(
                    path=media.filepath.replace("\\", "/"),
                    external=True),
                sample_rate=out_tc.samplerate,
                channels=out_tc.channels,
                duration=out_tc.duration_sec,
            )
            clip_content = audio

            clip = Utility.create_clip(
                content=clip_content, time=0.0, duration=out_tc.duration_sec)
            clips = Utility.create_clips(clip)

            # Build lane contents for this track
            lane_contents = [clips]

            # Create a Lanes entry for this track in the arrangement
            track_lane = Lanes(
                track=new_track,
                time_unit=TimeUnit.SECONDS,
                lanes=lane_contents,
            )
            additions.append((folder_track, new_track, track_lane))

            results.append(DawCommandResult(
                command=DawCommand("add_track", eid,
                                   {"folder": folder_dict.get("name", ""),
                                    "fader_db": media.fader_db}),
                success=True))

            if progress_cb:
                progress_cb(step + 1, total,
                            f"Added {track_name} ({step + 1}/{total})")

        if results and not any(result.success for result in results):
            session.daw_command_log.extend(results)
            return results

        # ── Save ──────────────────────────────────────────────────
        try:
            path = Path(output_path)
            if path.resolve() == Path(template.filename).resolve():
                raise ValueError("Choose a different output path to preserve the DAWproject template.")
            path.parent.mkdir(parents=True, exist_ok=True)
            _save_template(template, root, path, project, additions,
                           replace_existing=not session.config.get("_batch_transfer", False))
            results.append(DawCommandResult(
                command=DawCommand("save_project", output_path, {}),
                success=True))
            log.info("DAWproject saved to %s", output_path)
        except Exception as e:
            for result in results:
                if result.success:
                    result.success = False
                    result.error = "Track could not be published because the project save failed."
            results.append(DawCommandResult(
                command=DawCommand("save_project", output_path, {}),
                success=False, error=f"Failed to save: {e}"))

        session.daw_command_log.extend(results)
        return results

    def _build_folder_map(
        self, tracks: list, folder_map: dict[str, Any],
    ) -> None:
        """Recursively build a mapping of folder ID → Track object."""
        from dawproject import ContentType

        for track in tracks:
            ct_values = set()
            for ct in getattr(track, "content_type", []):
                if isinstance(ct, ContentType):
                    ct_values.add(ct)
                else:
                    try:
                        ct_values.add(ContentType(ct))
                    except ValueError:
                        pass
            if ContentType.TRACKS in ct_values:
                folder_map[track.id] = track
                self._build_folder_map(track.tracks, folder_map)

    def _resolve_track_color(
        self, group_name: str | None, session: SessionContext,
    ) -> str | None:
        """Return ``#rrggbb`` for the track's group color, or ``None``."""
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
                argb = c.get("argb")
                if argb:
                    return _argb_to_rgb_hex(argb)
        return None

    def sync(self, session: SessionContext) -> list[DawCommandResult]:
        return []

    def execute_commands(
        self, session: SessionContext, commands: list[DawCommand],
    ) -> list[DawCommandResult]:
        return []
