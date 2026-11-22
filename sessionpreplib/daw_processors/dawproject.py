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


def _is_folder_track(track) -> bool:
    """Return True when *track* is a DAWproject folder (``contentType`` includes ``tracks``)."""
    from dawproject import ContentType, Track

    if not isinstance(track, Track):
        return False
    for ct in getattr(track, "content_type", ()):
        if isinstance(ct, ContentType) and ct == ContentType.TRACKS:
            return True
        if not isinstance(ct, ContentType):
            try:
                if ContentType(ct) == ContentType.TRACKS:
                    return True
            except ValueError:
                pass
    return False


def _load_template_project(template: zipfile.ZipFile):
    """Parse ``project.xml`` after reserving all existing numeric ``id<N>`` attributes."""
    from dawproject import Project, Referenceable
    from lxml import etree

    root = etree.fromstring(template.read("project.xml"))
    Referenceable.reset_id()
    reserved = (n[2:] for n in root.xpath("//@id") if n.startswith("id") and n[2:].isdecimal())
    Referenceable.ID = max(map(int, reserved), default=-1) + 1
    project = Project.from_xml(root)
    return root, project


def _index_structure_elements(root, project) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map deserialized Track and Channel IDs back to their original XML elements."""
    from dawproject import Channel, Track

    folders: dict[str, Any] = {}
    channels: dict[str, Any] = {}

    def walk(elements, tracks):
        for element, track in zip(elements, tracks, strict=True):
            folders[track.id] = element
            if track.channel is not None:
                ch_elem = element.find("Channel")
                if ch_elem is not None:
                    channels[track.channel.id] = ch_elem
            walk(element.findall("Track"), [t for t in track.tracks if isinstance(t, Track)])

    walk(root.findall("Structure/Track"), [t for t in project.structure if isinstance(t, Track)])
    for ch_elem, ch in zip(
        root.findall("Structure/Channel"),
        [c for c in project.structure if isinstance(c, Channel)],
        strict=True,
    ):
        channels[ch.id] = ch_elem
    return folders, channels


def _copy_zip_member(source: zipfile.ZipFile, archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> None:
    """Copy a template archive member using a fresh ``ZipInfo`` (no stale extra/header state)."""
    dt = member.date_time
    if dt[0] < 1980:
        dt = (1980, 1, 1, 0, 0, 0)
    elif dt[0] > 2107:
        dt = (2107, 12, 31, 23, 59, 58)
    info = zipfile.ZipInfo(member.filename, date_time=dt)
    info.compress_type = member.compress_type
    info.external_attr = member.external_attr
    info.comment = member.comment
    info.file_size = member.file_size
    with source.open(member) as old, archive.open(info, "w") as new:
        shutil.copyfileobj(old, new)


def _save_template(
    source, root, output_path, project, additions, embedded_files, replace_existing,
):
    """Append new tracks and embed audio into a self-contained DAWproject archive."""
    from lxml import etree

    template = Path(source.filename)
    with atomic_writer(output_path, binary=True, replace_existing=replace_existing) as stream:
        # Relative external files in the template are relative to its original directory.
        for item in root.iterfind(".//*[@path]"):
            path = item.get("path")
            if (item.get("external") in ("true", "1") and path and "://" not in path
                    and not Path(path).is_absolute() and not PureWindowsPath(path).is_absolute()):
                item.set("path", (template.parent / path).resolve().as_posix())

        folders, channels = _index_structure_elements(root, project)
        arrangement = root.find("Arrangement")
        if arrangement is None:
            arrangement = etree.Element("Arrangement")
            root.insert(root.index(root.find("Structure")) + 1, arrangement)
        lanes = arrangement.find("Lanes")
        if lanes is None:
            lanes = etree.Element("Lanes", timeUnit="seconds")
            arrangement.insert(0, lanes)
        for folder, track, lane, dest_channel in additions:
            element = folders[folder.id]
            if dest_channel is not None and dest_channel.id in channels:
                channels[dest_channel.id].set("id", dest_channel.id)
            element.append(track.to_xml())
            lanes.append(lane.to_xml())

        with zipfile.ZipFile(stream, "w", allowZip64=True, strict_timestamps=False) as archive:
            archive.comment = source.comment
            archive.writestr(
                "project.xml",
                etree.tostring(root, xml_declaration=True, encoding="UTF-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            if "metadata.xml" not in source.namelist():
                archive.writestr("metadata.xml", "<MetaData/>", compress_type=zipfile.ZIP_DEFLATED)
            for member in source.infolist():
                if member.filename == "project.xml":
                    continue
                _copy_zip_member(source, archive, member)
            for src_filepath, arcname in embedded_files:
                archive.write(src_filepath, arcname=arcname, compress_type=zipfile.ZIP_STORED)


def _allocate_archive_audio_path(
    source_path: str,
    used_members: set[str],
    embedded_by_source: dict[str, str],
) -> tuple[str, bool]:
    """Return ``(archive_path, needs_write)`` inside ``audio/`` without collisions."""
    key = os.path.normcase(os.path.realpath(source_path))
    existing = embedded_by_source.get(key)
    if existing is not None:
        return existing, False
    base = (Path(source_path).name or "audio.wav").replace("\\", "_")
    stem, ext = os.path.splitext(base)
    candidate = f"audio/{base}"
    suffix = 2
    while candidate.lower() in used_members:
        candidate = f"audio/{stem}_{suffix}{ext}"
        suffix += 1
    used_members.add(candidate.lower())
    embedded_by_source[key] = candidate
    return candidate, True


def _find_master_channel(root, project) -> Any:
    """Return the template's main output master channel (Cubase standalone or Bitwig Track)."""
    from dawproject import Channel, MixerRole, Track

    _, channel_elements = _index_structure_elements(root, project)
    targeted_ids = set(root.xpath("//Channel/@destination"))
    candidates: list[Any] = []
    for item in getattr(project, "structure", ()):
        if isinstance(item, Channel):
            ch = item
        elif isinstance(item, Track) and not _is_folder_track(item) and item.channel is not None:
            ch = item.channel
        else:
            continue
        role = getattr(ch, "role", None)
        if role != MixerRole.MASTER and str(role) != "master":
            continue
        ch_elem = channel_elements.get(ch.id)
        if getattr(ch, "destination", None) is not None or (ch_elem is not None and ch_elem.get("destination")):
            continue
        candidates.append(ch)
    for ch in candidates:
        if getattr(ch, "id", None) in targeted_ids:
            return ch
    return candidates[0] if candidates else None


def _resolve_channel_destination(folder_track, folder_parent_map: dict[str, Any], master_channel) -> Any:
    """Resolve routing target: folder bus → nearest ancestor routing folder → master bus."""
    current = folder_track
    while current is not None:
        if getattr(current, "channel", None) is not None:
            return current.channel
        current = folder_parent_map.get(current.id)
    return master_channel


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
    output_file_filter: str = "DAWproject (*.dawproject);;All Files (*)"
    output_dialog_title: str = "Save DAWproject"

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
            import dawproject as _dp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "dawproject package not installed. "
                "Install SessionPrep with its gui extra."
            ) from exc

        with zipfile.ZipFile(self._template_path) as template:
            _, project = _load_template_project(template)

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
        for track in tracks:
            if _is_folder_track(track):
                folder_type = "routing" if track.channel else "basic"
                folders.append({
                    "id": track.id,
                    "name": track.name or "(unnamed)",
                    "folder_type": folder_type,
                    "index": counter[0],
                    "parent_id": parent_id,
                })
                counter[0] += 1
                self._walk_structure(
                    track.tracks, folders, parent_id=track.id,
                    counter=counter)

    def default_output_path(self, session: SessionContext) -> str:
        """Compute the default ``.dawproject`` output path (pure, Qt-free)."""
        source_dir = session.config.get("_source_dir", "")
        output_folder = session.config.get("_output_folder", "sp_02_prepared")
        output_dir = os.path.join(source_dir, output_folder) if source_dir else ""

        safe_name = (
            session.project_name
            if getattr(session, "project_name", "")
            else (self._template_name or self.name or "dawproject")
        )
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in safe_name
        )
        filename = f"{safe_name}.dawproject"
        return os.path.join(output_dir, filename) if output_dir else filename

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
            import dawproject as _dp  # noqa: F401
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
                root, project = _load_template_project(template)
            except Exception as exc:
                return [DawCommandResult(DawCommand("load_template", "", {}), False, f"Failed to load template: {exc}")]
            return self._add_tracks(session, output_path, progress_cb, template, root, project)

    @staticmethod
    def _ordered_assignments(assignments: dict[str, str], track_order: dict[str, list[str]]) -> list[tuple[str, str]]:
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
        return work

    def _build_track_addition(
        self, session, folder_track, dest_channel, entry, out_tc, media, archive_audio_path: str,
    ):
        from dawproject import (
            Audio, BoolParameter, Channel, Clip, Clips, ContentType,
            FileReference, Lanes, MixerRole, RealParameter, TimeUnit, Track, Unit,
        )

        track_name = os.path.splitext(entry.daw_track_name)[0]
        track_color = self._resolve_track_color(entry.group, session)
        volume = _db_to_linear(media.fader_db)
        audio_channels = out_tc.channels if (out_tc.channels and out_tc.channels > 0) else 2
        channel = Channel(
            role=MixerRole.REGULAR,
            audio_channels=audio_channels,
            name=track_name,
            color=track_color,
            destination=dest_channel,
            mute=BoolParameter(value=False, name="Mute"),
            pan=RealParameter(value=0.5, unit=Unit.NORMALIZED, min_value=0.0, max_value=1.0, name="Pan"),
            volume=RealParameter(
                value=volume, unit=Unit.LINEAR, min_value=0.0, max_value=max(2.0, volume), name="Volume",
            ),
        )
        new_track = Track(
            name=track_name,
            color=track_color,
            content_type={ContentType.AUDIO},
            loaded=True,
            channel=channel,
        )
        audio = Audio(
            time_unit=TimeUnit.SECONDS,
            file=FileReference(path=archive_audio_path, external=False),
            sample_rate=out_tc.samplerate,
            channels=audio_channels,
            duration=out_tc.duration_sec,
        )
        clip = Clip(
            content=audio,
            time=0.0,
            duration=out_tc.duration_sec,
            content_time_unit=TimeUnit.SECONDS,
            name=track_name,
            color=track_color,
        )
        track_lane = Lanes(track=new_track, time_unit=TimeUnit.SECONDS, lanes=[Clips(clips=[clip])])
        return track_name, (folder_track, new_track, track_lane, dest_channel)

    def _save_dawproject_output(
        self, session, output_path, template, root, project, additions, embedded_files, results,
    ) -> None:
        try:
            path = Path(output_path)
            if path.resolve() == Path(template.filename).resolve():
                raise ValueError("Choose a different output path to preserve the DAWproject template.")
            path.parent.mkdir(parents=True, exist_ok=True)
            _save_template(
                template, root, path, project, additions, embedded_files,
                replace_existing=not session.config.get("_batch_transfer", False),
            )
            results.append(DawCommandResult(DawCommand("save_project", output_path, {}), True))
            log.info("DAWproject saved to %s", output_path)
        except Exception as exc:
            for result in results:
                if result.success:
                    result.success = False
                    result.error = "Track could not be published because the project save failed."
            results.append(DawCommandResult(DawCommand("save_project", output_path, {}), False, f"Failed to save: {exc}"))

    def _add_tracks(self, session, output_path, progress_cb, template, root, project):
        dp_state = session.daw_state.get(self.id, {})
        folder_track_map: dict[str, Any] = {}
        folder_parent_map: dict[str, Any] = {}
        self._build_folder_map(project.structure, folder_track_map, folder_parent_map)
        master_channel = _find_master_channel(root, project)
        folder_dict_map = {f["id"]: f for f in dp_state.get("folders", [])}
        manifest_map = {e.entry_id: e for e in session.transfer_manifest}
        out_track_map = {t.filename: t for t in session.output_tracks}
        used_members = {m.replace("\\", "/").lower() for m in template.namelist()}
        embedded_by_source: dict[str, str] = {}
        embedded_files: list[tuple[str, str]] = []

        work = self._ordered_assignments(dp_state.get("assignments", {}), dp_state.get("track_order", {}))
        total = len(work)
        if progress_cb:
            progress_cb(0, total, "Building DAWproject\u2026")

        results: list[DawCommandResult] = []
        additions = []
        for step, (eid, fid) in enumerate(work):
            folder_track = folder_track_map.get(fid)
            folder_dict = folder_dict_map.get(fid)
            entry = manifest_map.get(eid)
            out_tc = out_track_map.get(entry.output_filename) if entry else None
            if not folder_track or not folder_dict or not entry or not out_tc:
                results.append(DawCommandResult(
                    DawCommand("add_track", eid, {"folder_id": fid}), False,
                    f"Folder, manifest entry, or output track not found: {fid} / {eid}",
                ))
                continue
            try:
                media = resolve_transfer_audio(session, out_tc)
            except ValueError as exc:
                results.append(DawCommandResult(DawCommand("add_track", eid, {"folder_id": fid}), False, str(exc)))
                continue
            arcname, needs_write = _allocate_archive_audio_path(media.filepath, used_members, embedded_by_source)
            if needs_write:
                embedded_files.append((media.filepath, arcname))
            dest_channel = _resolve_channel_destination(folder_track, folder_parent_map, master_channel)
            log.debug("%s: file=%s -> %s, fader=%+.2f dB", eid, media.filepath, arcname, media.fader_db)
            track_name, addition = self._build_track_addition(
                session, folder_track, dest_channel, entry, out_tc, media, arcname,
            )
            additions.append(addition)
            results.append(DawCommandResult(
                DawCommand("add_track", eid, {"folder": folder_dict.get("name", ""), "fader_db": media.fader_db}), True,
            ))
            if progress_cb:
                progress_cb(step + 1, total, f"Added {track_name} ({step + 1}/{total})")

        if not (results and not any(result.success for result in results)):
            self._save_dawproject_output(
                session, output_path, template, root, project, additions, embedded_files, results,
            )
        session.daw_command_log.extend(results)
        return results

    def _build_folder_map(
        self,
        tracks: list,
        folder_map: dict[str, Any],
        parent_map: dict[str, Any] | None = None,
        parent: Any = None,
    ) -> None:
        """Recursively build folder ID → Track and folder ID → parent Track mappings."""
        for track in tracks:
            if _is_folder_track(track):
                folder_map[track.id] = track
                if parent_map is not None and parent is not None:
                    parent_map[track.id] = parent
                self._build_folder_map(track.tracks, folder_map, parent_map, parent=track)

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
