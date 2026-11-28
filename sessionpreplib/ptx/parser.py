"""Pure-Python Pro Tools ``.ptx`` / ``.ptxt`` binary session parser (v10+)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import os
from pathlib import Path
import re
import struct
from typing import TYPE_CHECKING, Any

from .arrangement import (
    ResolvedAudioEvent,
    ResolvedMidiEvent,
    ResolvedVideoEvent,
    resolve_audio_events,
    resolve_midi_events,
    resolve_video_events,
)
from .codec import (
    decode_slot_mask,
    decrypt_ptx,
    find_subchunks,
    format_ptsl_uuid,
    group_id_to_label,
    iter_top_chunks,
    lookup_pt_color_rgb,
    parse_chunk_tree,
    read_lp_str,
)
from .models import (
    AUDIO_FORMAT_MAP,
    GROUP_TYPE_MAP,
    IO_KIND_MAP,
    TRACK_FORMAT_MAP,
    TRACK_TYPE_MAP,
    UUID_PREFIX,
    format_pull_rate,
    PtxAraBinding,
    PtxAraPlugin,
    PtxAudioFile,
    PtxChunk,
    PtxClip,
    PtxClipGainCurve,
    PtxCompoundClip,
    PtxFade,
    PtxGroup,
    PtxGroupAttributes,
    PtxMarker,
    PtxMeterEvent,
    PtxMidiChunk,
    PtxMidiClip,
    PtxMidiPlacement,
    PtxPlaylist,
    PtxPluginInfo,
    PtxSessionInfo,
    PtxSignalPath,
    PtxSignalPathRef,
    PtxTempoEvent,
    PtxTrack,
    PtxVideoClip,
    PtxVideoPlacement,
)
from .parser_mixer import (
    parse_ara_bindings_4820,
    parse_ara_registry_4823,
    parse_track_flags,
    parse_track_input,
    parse_track_inserts,
    parse_track_outputs_and_sends,
    parse_vca_mixer,
)
from .parser_timeline import (
    parse_audio_files_1004,
    parse_clip_gain_curves_2637,
    parse_clips_262a,
    parse_compound_clips,
    parse_fades_2630,
    parse_markers_2030,
    parse_meter_2029,
    parse_midi_data,
    parse_playlists_1015,
    parse_session_start_204d,
    parse_session_ui_state,
    parse_tempo_2028,
    parse_timeline_lanes_1054,
    parse_video_data,
)
from .queries import (
    PtxFolderNode,
    PtxSessionAudit,
    audit_session,
    build_folder_tree,
    effective_send_volume_db,
    effective_track_volume_db,
    external_audio_files,
    find_inserts,
    groups_for_track,
    is_track_audible,
    iter_tracks,
    unused_audio_files,
    used_audio_files,
    vcas_for_track,
)
from .timeline import PtxTimeline

if TYPE_CHECKING:
    from .models import PtxInsert


def split_202b(raw_202b: bytes) -> tuple[list[bytes], list[bytes], bytes]:
    """Split a ``0x202b`` chunk into ``(edit_202a_list, mix_202a_list, trailer)``."""
    pos = 9
    edit_list, pos = _read_202a_list(raw_202b, pos, "edit")
    mix_list, pos = _read_202a_list(raw_202b, pos, "mix")
    return edit_list, mix_list, raw_202b[pos:]


def _read_202a_list(raw: bytes, pos: int, label: str) -> tuple[list[bytes], int]:
    if pos + 4 > len(raw):
        raise RuntimeError(f"Truncated {label} group count inside 0x202b")
    count = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    items: list[bytes] = []
    for _ in range(count):
        if pos + 9 > len(raw) or raw[pos] != 0x5A:
            raise RuntimeError(f"Corrupt {label} group list inside 0x202b")
        sz = struct.unpack_from("<I", raw, pos + 3)[0]
        end = pos + 7 + sz
        if sz < 2 or end > len(raw):
            raise RuntimeError(f"Truncated {label} group entry inside 0x202b")
        items.append(raw[pos:end])
        pos = end
    return items, pos


def _parse_group_attrs_258e(chunk_258e: bytes) -> PtxGroupAttributes:
    body = chunk_258e[9:]
    if len(body) < 52:
        return PtxGroupAttributes()
    known = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 25, 26, 33, 34, 41, 42, 49, 50, 61}
    extra = {i: body[i] for i in range(len(body)) if i not in known and body[i] != 0}
    return PtxGroupAttributes(
        main_volume=bool(body[0]),
        main_mute=bool(body[1]),
        main_pan=bool(body[2]),
        main_lfe=bool(body[3]),
        follow_globals=bool(body[4]),
        record_enable=bool(body[5]),
        input_monitoring=bool(body[6]),
        solos=bool(body[7]),
        automation_mode=bool(body[8]),
        include_rulers=bool(body[61]) if len(body) > 61 else False,
        insert_controls=decode_slot_mask(struct.unpack_from("<H", body, 9)[0]),
        insert_bypass=decode_slot_mask(struct.unpack_from("<H", body, 17)[0]),
        send_level=decode_slot_mask(struct.unpack_from("<H", body, 25)[0]),
        send_mute=decode_slot_mask(struct.unpack_from("<H", body, 33)[0]),
        send_pan=decode_slot_mask(struct.unpack_from("<H", body, 41)[0]),
        send_lfe=decode_slot_mask(struct.unpack_from("<H", body, 49)[0]),
        extra_mix_flags=extra,
    )


def _parse_signal_path_entry(chunk: bytes, idx: int, source_chunk: str) -> PtxSignalPath | None:
    if len(chunk) < 24:
        return None
    f0, f1 = chunk[9], chunk[10]
    name, p = read_lp_str(chunk, 11, max_len=256)
    if not name:
        return None
    ch_count = 0
    bus_chs: list[int] = []
    if p + 4 <= len(chunk):
        ch_count = struct.unpack_from("<I", chunk, p)[0]
        p += 4
        if 0 < ch_count <= 64 and p + ch_count * 2 <= len(chunk):
            bus_chs = list(struct.unpack_from(f"<{ch_count}H", chunk, p))
            p += ch_count * 2
    parent_io_idx, is_active, mapped_to_out = _parse_sp_flags(chunk, p, source_chunk)
    raw_uuid, parent_uuid = _parse_sp_uuids(chunk, p, source_chunk)
    output_mappings = _parse_sp_output_mappings(chunk) if source_chunk == "0x2603" else []
    if output_mappings:
        mapped_to_out = True
    is_stereo = (f1 == 1) or (ch_count == 2)
    return PtxSignalPath(
        index=idx,
        name=name,
        kind=IO_KIND_MAP.get(f0, "bus"),
        is_stereo=is_stereo,
        channel_count=ch_count if 0 < ch_count <= 64 else (2 if is_stereo else 1),
        uuid_hex=raw_uuid.hex(),
        uuid_braced=format_ptsl_uuid(raw_uuid),
        source_chunk=source_chunk,
        bus_channels=bus_chs,
        is_active=is_active,
        is_subpath=(parent_io_idx is not None) or bool(parent_uuid),
        parent_io_index=parent_io_idx,
        parent_uuid_hex=parent_uuid.hex(),
        mapped_to_output=mapped_to_out,
        output_mappings=output_mappings,
    )


def _parse_sp_flags(chunk: bytes, pos: int, source_chunk: str) -> tuple[int | None, bool, bool]:
    parent_io_idx: int | None = None
    is_active = True
    mapped_to_out = False
    if pos + 3 <= len(chunk):
        raw_parent = struct.unpack_from("<H", chunk, pos)[0]
        if raw_parent != 0xFFFF:
            parent_io_idx = raw_parent
        is_active = bool(chunk[pos + 2])
        if source_chunk == "0x2603" and pos + 4 <= len(chunk):
            mapped_to_out = bool(chunk[pos + 3])
    return parent_io_idx, is_active, mapped_to_out


def _parse_sp_uuids(chunk: bytes, pos: int, source_chunk: str) -> tuple[bytes, bytes]:
    uoff = chunk.find(UUID_PREFIX, pos)
    raw_uuid = chunk[uoff : uoff + 16] if (uoff != -1 and uoff + 16 <= len(chunk)) else b""
    parent_uuid = b""
    if source_chunk == "0x2603" and uoff != -1 and uoff + 32 <= len(chunk):
        cand = chunk[uoff + 16 : uoff + 32]
        if cand != b"\x00" * 16:
            parent_uuid = cand
    return raw_uuid, parent_uuid


def _parse_sp_output_mappings(chunk: bytes) -> list[str]:
    mappings: list[str] = []
    for m2600 in find_subchunks(chunk, 0x2600):
        mu = m2600.find(UUID_PREFIX)
        if mu != -1 and mu + 20 <= len(m2600):
            mname, _ = read_lp_str(m2600, mu + 16, max_len=256)
            if mname:
                mappings.append(mname)
    return mappings


def _parse_ptx_paths_2067(raw: bytes) -> tuple[str, list[str]]:
    session_id_code = ""
    ptx_paths: list[str] = []
    for match in re.finditer(rb"[\x20-\x7e]+\.ptx", raw):
        fname = match.group().decode("utf-8", errors="replace")
        seg_region = raw[max(0, match.start() - 220) : match.start()]
        segs = [s.decode("utf-8", errors="replace") for s in re.findall(rb"[A-Za-z0-9_ .:\\/-]{2,64}", seg_region)]
        parts: list[str] = []
        for seg in segs:
            if seg.startswith("Info #"):
                parts.clear()
            elif len(seg) == 16 and seg.isalnum() and not any(ch in seg for ch in " \\/."):
                session_id_code = session_id_code or seg
            elif seg.endswith(":\\") or seg.startswith("/"):
                parts = [seg]
            elif parts:
                parts.append(seg)
        if parts:
            ptx_paths.append(os.path.join(*parts, fname))
    return session_id_code, ptx_paths


class PtxSession:
    """General-purpose decoded Pro Tools ``.ptx`` or ``.ptxt`` session (v10+)."""

    def __init__(self, decrypted: bytes, xor_delta: int = 0, file_path: str | None = None):
        self.decrypted = decrypted
        self.info = PtxSessionInfo(file_path=file_path, xor_delta=xor_delta, file_size=len(decrypted))
        self.chunks: list[PtxChunk] = iter_top_chunks(decrypted)
        self.chunk_map: dict[int, PtxChunk] = {c.tag: c for c in self.chunks}
        self.warnings: list[str] = []
        self.tempo_events: list[PtxTempoEvent] = []
        self.tempo_event_count_declared: int = 0
        self.has_unparsed_tempo_ramps: bool = False
        self.meter_events: list[PtxMeterEvent] = []
        self.markers: list[PtxMarker] = []
        self.io_paths: list[PtxSignalPath] = []
        self.signal_paths: list[PtxSignalPath] = []
        self.plugins: list[PtxPluginInfo] = []
        self.ara_plugins: list[PtxAraPlugin] = []
        self.ara_bindings: list[PtxAraBinding] = []
        self.tracks: list[PtxTrack] = []
        self.groups: list[PtxGroup] = []
        self.group_globals: PtxGroupAttributes | None = None
        self.group_presets: list[PtxGroupAttributes] = []
        self.audio_files: list[PtxAudioFile] = []
        self.clip_gain_curves: list[PtxClipGainCurve] = []
        self.clips: list[PtxClip] = []
        self.fades: list[PtxFade] = []
        self.compound_clips: list[PtxCompoundClip] = []
        self.playlists: list[PtxPlaylist] = []
        self.midi_chunks: list[PtxMidiChunk] = []
        self.midi_clips: list[PtxMidiClip] = []
        self.midi_playlists: dict[str, list[PtxMidiPlacement]] = {}
        self.video_files: list[str] = []
        self.video_clips: list[PtxVideoClip] = []
        self.video_playlists: dict[str, list[PtxVideoPlacement]] = {}
        self._io1022_by_uuid: dict[bytes, PtxSignalPath] = {}
        self._sp2603_by_uuid: dict[bytes, PtxSignalPath] = {}
        self._sp2603_by_name: dict[str, PtxSignalPath] = {}
        self._tracks_missing_tail_active: set[str] = set()
        self._timeline: PtxTimeline | None = None
        self._parse_all()

    @classmethod
    def from_file(cls, path: str | Path) -> PtxSession:
        raw = Path(path).read_bytes()
        dec, delta = decrypt_ptx(raw)
        return cls(dec, xor_delta=delta, file_path=os.path.abspath(str(path)))

    @classmethod
    def from_bytes(cls, raw: bytes, file_path: str | None = None) -> PtxSession:
        dec, delta = decrypt_ptx(raw)
        return cls(dec, xor_delta=delta, file_path=file_path)

    @property
    def timeline(self) -> PtxTimeline:
        """Return the unified time-domain converter and ruler query interface."""
        if self._timeline is None:
            self._timeline = PtxTimeline(self)
        return self._timeline

    def chunk_tree(self) -> list[PtxChunk]:
        """Return a recursively parsed nested Z-chunk tree for low-level inspection."""
        return parse_chunk_tree(self.decrypted)

    def ticks_to_seconds(self, tick: int) -> float:
        """Convert a relative Pro Tools tick value to seconds using the decoded tempo map."""
        return self.timeline.ticks_to_seconds(tick)

    def resolve_audio_events(
        self,
        track_or_name: PtxTrack | str,
        *,
        playlist: str | None = None,
        expand_compounds: bool = True,
        attach_fades: bool = True,
        trim_overlaps: bool = False,
        include_muted: bool = True,
    ) -> list[ResolvedAudioEvent]:
        """Join placements, clips, files, fades, clip gain, and compound clips for one track."""
        return resolve_audio_events(
            self, track_or_name, playlist=playlist, expand_compounds=expand_compounds,
            attach_fades=attach_fades, trim_overlaps=trim_overlaps, include_muted=include_muted,
        )

    def resolve_midi_events(
        self,
        track_or_name: PtxTrack | str,
        *,
        absolute_ticks: bool = True,
        include_muted: bool = True,
    ) -> list[ResolvedMidiEvent]:
        """Join MIDI placements, clips, and chunks with clipped notes and control lanes."""
        return resolve_midi_events(
            self, track_or_name, absolute_ticks=absolute_ticks, include_muted=include_muted,
        )

    def resolve_video_events(self, track_or_name: PtxTrack | str) -> list[ResolvedVideoEvent]:
        """Join video placements and video clips for a video track."""
        return resolve_video_events(self, track_or_name)

    def iter_tracks(
        self,
        *,
        track_types: set[str] | None = None,
        include_hidden: bool = True,
        include_inactive: bool = True,
        folder: str | None = None,
        recursive: bool = False,
    ) -> list[PtxTrack]:
        """Filter tracks by type, visibility, active state, and optional parent folder."""
        return iter_tracks(
            self, track_types=track_types, include_hidden=include_hidden,
            include_inactive=include_inactive, folder=folder, recursive=recursive,
        )

    def folder_tree(self) -> list[PtxFolderNode]:
        """Build the nested top-level folder and track hierarchy."""
        return build_folder_tree(self)

    def groups_for_track(self, track_or_name: PtxTrack | str, *, active_only: bool = False) -> list[PtxGroup]:
        """Return unique groups containing *track_or_name*."""
        return groups_for_track(self, track_or_name, active_only=active_only)

    def vcas_for_track(self, track_or_name: PtxTrack | str) -> list[PtxTrack]:
        """Return all active VCA master tracks recursively controlling *track_or_name*."""
        return vcas_for_track(self, track_or_name)

    def find_inserts(
        self,
        *,
        plugin_name: str | None = None,
        manufacturer_4cc: str | None = None,
        bundle_id: str | None = None,
        active_only: bool = False,
        include_bypassed: bool = True,
    ) -> list[tuple[PtxTrack, PtxInsert]]:
        """Search inserts across all tracks by plugin name, manufacturer 4CC, bundle ID, and state."""
        return find_inserts(
            self, plugin_name=plugin_name, manufacturer_4cc=manufacturer_4cc,
            bundle_id=bundle_id, active_only=active_only, include_bypassed=include_bypassed,
        )

    def used_audio_files(self) -> list[PtxAudioFile]:
        """Return audio files referenced by at least one placed clip or compound member."""
        return used_audio_files(self)

    def unused_audio_files(self) -> list[PtxAudioFile]:
        """Return audio files in ``0x1004`` that are not placed on any playlist."""
        return unused_audio_files(self)

    def external_audio_files(self) -> list[PtxAudioFile]:
        """Return audio files residing outside the session folder."""
        return external_audio_files(self)

    def effective_track_volume_db(
        self, track_or_name: PtxTrack | str, at_seconds: float = 0.0, *, include_vca: bool = True
    ) -> float | None:
        """Evaluate track fader dB at *at_seconds*, optionally adding controlling VCA offsets."""
        return effective_track_volume_db(self, track_or_name, at_seconds, include_vca=include_vca)

    def is_track_audible(
        self, track_or_name: PtxTrack | str, at_seconds: float = 0.0, *, include_vca: bool = True
    ) -> bool:
        """Return True when the track is active, unmuted, and above ``-inf`` at *at_seconds*."""
        return is_track_audible(self, track_or_name, at_seconds, include_vca=include_vca)

    def effective_send_volume_db(
        self, track_or_name: PtxTrack | str, send_slot: str, at_seconds: float = 0.0
    ) -> float | None:
        """Evaluate a track send's volume dB at *at_seconds*."""
        return effective_send_volume_db(self, track_or_name, send_slot, at_seconds)

    def audit(self) -> PtxSessionAudit:
        """Return a structured readiness and complexity audit for the session."""
        return audit_session(self)

    def signal_path_by_name(self, name: str) -> PtxSignalPath | None:
        exact = self._sp2603_by_name.get(name)
        if exact:
            return exact
        low = name.lower()
        return next((sp for sp in self.signal_paths if sp.name.lower() == low), None)

    def signal_path_by_index(self, idx: int) -> PtxSignalPath | None:
        return self.signal_paths[idx] if 0 <= idx < len(self.signal_paths) else None

    def track_by_name(self, name: str) -> PtxTrack | None:
        exact = next((t for t in self.tracks if t.name == name), None)
        if exact:
            return exact
        low = name.lower()
        return next((t for t in self.tracks if t.name.lower() == low), None)

    def track_by_uuid(self, uuid_braced_or_hex: str) -> PtxTrack | None:
        norm = uuid_braced_or_hex.strip().lower().replace("{", "").replace("}", "").replace("-", "")
        return next((t for t in self.tracks if t.uuid_hex.lower() == norm), None)

    def group_by_name(self, name: str, kind: str = "edit") -> PtxGroup | None:
        low = name.lower()
        for grp in self.groups:
            if grp.name.lower() == low and grp.kind == kind:
                return grp
        return next((grp for grp in self.groups if grp.name.lower() == low), None)

    def io_configuration(self, ptsl_base: int | None = None) -> dict[str, Any]:
        """Return a structured summary of the session's I/O and bus configuration."""
        def _fmt(sp: PtxSignalPath, include_ptsl: bool = False) -> dict[str, Any]:
            width = "Stereo" if sp.is_stereo else (f"{sp.channel_count}ch" if sp.channel_count > 2 else "Mono")
            entry: dict[str, Any] = {
                "index": sp.index, "name": sp.name, "width": width,
                "bus_channels": sp.bus_channels, "is_active": sp.is_active,
                "is_subpath": sp.is_subpath, "parent": sp.parent_name or None, "subpaths": sp.subpaths,
            }
            if sp.mapped_to_output or sp.output_mappings:
                entry["mapped_to_output"] = sp.mapped_to_output
                entry["output_mappings"] = sp.output_mappings
            if include_ptsl and ptsl_base is not None:
                entry["ptsl_signalpath_id"] = hex(ptsl_base + sp.index)
            return entry

        hw_in = [_fmt(sp) for sp in self.io_paths if sp.kind == "input"]
        hw_out = [_fmt(sp) for sp in self.io_paths if sp.kind == "output"]
        mapped = [_fmt(sp, True) for sp in self.signal_paths if sp.mapped_to_output]
        internal = [_fmt(sp, True) for sp in self.signal_paths if not sp.mapped_to_output]
        return {
            "counts": {
                "hardware_inputs": len(hw_in), "hardware_outputs": len(hw_out),
                "output_mapped_busses": len(mapped), "internal_busses": len(internal),
                "total_routable_2603": len(self.signal_paths), "total_io_1022": len(self.io_paths),
            },
            "hardware_inputs": hw_in, "hardware_outputs": hw_out,
            "output_mapped_busses": mapped, "internal_busses": internal,
        }

    def groups_summary(self) -> dict[str, Any]:
        """Return groups, VCA wiring, group globals, and presets 1-6."""
        return {
            "globals": asdict(self.group_globals) if self.group_globals else None,
            "presets": [asdict(p) for p in self.group_presets],
            "groups": [asdict(g) for g in self.groups if g.kind == "edit"],
            "vca_wiring": [
                {
                    "vca_track": t.name, "vca_track_index": t.index,
                    "group_name": t.vca_group_name, "controlled_tracks": list(t.vca_controlled_tracks),
                    "volume_db": t.volume_db, "muted": t.is_muted,
                }
                for t in self.tracks if t.track_type == "Vca"
            ],
        }

    def routing_matrix(self, ptsl_base: int | None = None) -> dict[str, Any]:
        """Build a routing matrix & bus connectivity graph for the session."""
        bus_graph: dict[str, dict[str, list[str]]] = {}

        def _ensure_bus(bus_name: str) -> dict[str, list[str]]:
            return bus_graph.setdefault(
                bus_name, {"fed_by_main": [], "fed_by_send": [], "feeds_input": [], "feeds_sidechain": []},
            )

        rows = [self._routing_row(t, ptsl_base, _ensure_bus) for t in self.tracks]
        return {
            "session": asdict(self.info),
            "ptsl_base": hex(ptsl_base) if ptsl_base is not None else None,
            "track_count": len(self.tracks),
            "signal_path_count": len(self.signal_paths),
            "tracks": rows,
            "bus_graph": bus_graph,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete decoded session into plain JSON-compatible dictionaries."""
        return {
            "info": asdict(self.info),
            "warnings": list(self.warnings),
            "tempo_events": [asdict(e) for e in self.tempo_events],
            "meter_events": [asdict(m) for m in self.meter_events],
            "markers": [asdict(m) for m in self.markers],
            "signal_paths": [asdict(sp) for sp in self.signal_paths],
            "io_paths": [asdict(sp) for sp in self.io_paths],
            "plugins": [asdict(p) for p in self.plugins],
            "ara_plugins": [asdict(a) for a in self.ara_plugins],
            "ara_bindings": [asdict(b) for b in self.ara_bindings],
            "tracks": [asdict(t) for t in self.tracks],
            "group_globals": asdict(self.group_globals) if self.group_globals else None,
            "group_presets": [asdict(p) for p in self.group_presets],
            "groups": [asdict(g) for g in self.groups],
            "audio_files": [asdict(af) for af in self.audio_files],
            "clip_gain_curves": [asdict(cg) for cg in self.clip_gain_curves],
            "clips": [asdict(c) for c in self.clips],
            "fades": [asdict(f) for f in self.fades],
            "compound_clips": [asdict(cp) for cp in self.compound_clips],
            "playlists": [asdict(pl) for pl in self.playlists],
            "midi_chunks": [asdict(mc) for mc in self.midi_chunks],
            "midi_clips": [asdict(mc) for mc in self.midi_clips],
            "midi_playlists": {k: [asdict(ev) for ev in v] for k, v in self.midi_playlists.items()},
            "video_files": list(self.video_files),
            "video_clips": [asdict(vc) for vc in self.video_clips],
            "video_playlists": {k: [asdict(ev) for ev in v] for k, v in self.video_playlists.items()},
        }

    def _routing_row(
        self, track: PtxTrack, ptsl_base: int | None, ensure_bus: Callable[[str], dict[str, list[str]]]
    ) -> dict[str, Any]:
        if track.input_path:
            ensure_bus(track.input_path.name)["feeds_input"].append(track.name)
        if track.main_output:
            ensure_bus(track.main_output.name)["fed_by_main"].append(track.name)
        for snd in track.sends:
            ensure_bus(snd.destination.name)["fed_by_send"].append(f"{track.name}[Send {snd.slot}]")
        for ins in track.inserts:
            if ins.sidechain:
                ensure_bus(ins.sidechain.name)["feeds_sidechain"].append(
                    f"{track.name}[Ins {ins.slot}: {ins.plugin_name or ins.fourcc}]"
                )
        return {
            "index": track.index, "name": track.name, "id": track.uuid_braced,
            "type": track.track_type, "format": track.format,
            "parent_folder": track.parent_folder_name or None,
            "input": self._format_sp_ref(track.input_path, ptsl_base),
            "main_output": self._format_sp_ref(track.main_output, ptsl_base),
            "volume_db": track.volume_db,
            "pan": [track.pan_left, track.pan_right] if (track.pan_left is not None or track.pan_right is not None) else None,
            "muted": track.is_muted, "inactive": track.is_inactive,
            "vca_group": track.vca_group_name, "vca_controlled_tracks": list(track.vca_controlled_tracks),
            "clip_count": len(track.clips),
        }

    @staticmethod
    def _format_sp_ref(ref: PtxSignalPathRef | None, ptsl_base: int | None) -> dict[str, Any] | None:
        if ref is None:
            return None
        data: dict[str, Any] = {"name": ref.name, "kind": ref.kind, "ptsl_index": ref.ptsl_index}
        if ptsl_base is not None and ref.ptsl_index is not None:
            data["ptsl_signalpath_id"] = hex(ptsl_base + ref.ptsl_index)
        return data

    def _run_section(self, label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            msg = f"Failed to parse {label}: {exc}"
            if msg not in self.warnings:
                self.warnings.append(msg)

    def _parse_all(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = [
            ("header", self._parse_version_and_header),
            ("session start", self._parse_session_start),
            ("tempo", self._parse_tempo),
            ("meter", self._parse_meter),
            ("I/O setup", self._parse_io_setup_1022),
            ("signal paths", self._parse_signal_paths_2603),
            ("plugins", self._parse_plugins_1018),
            ("ARA2", self._parse_ara_chunks),
            ("tracks", self._parse_tracks_2519),
            ("folders", self._parse_folder_tree_210c),
            ("mixer", self._parse_mixer_2624),
            ("track UI", self._parse_track_ui_2587),
            ("group globals", self._parse_group_globals_and_presets),
            ("groups", self._parse_groups_202b),
            ("audio files", self._parse_audio_files),
            ("clip gain", self._parse_clip_gain_curves),
            ("clips", self._parse_clips),
            ("fades", self._parse_fades),
            ("compound clips", self._parse_compound_clips),
            ("playlists", self._parse_playlists_and_placements),
            ("MIDI", self._parse_midi),
            ("video", self._parse_video),
            ("markers", self._parse_markers),
        ]
        for label, fn in steps:
            self._run_section(label, fn)

    def _parse_version_and_header(self) -> None:
        c0003 = self.chunk_map.get(0x0003)
        if c0003:
            strs = [s.decode("utf-8", errors="replace") for s in re.findall(rb"[\x20-\x7e]{2,80}", c0003.body)]
            if len(strs) >= 1:
                self.info.product_name = strs[0]
            if len(strs) >= 2:
                self.info.protools_version = strs[1]
            if len(strs) >= 3:
                self.info.build_type = strs[2]
            if len(strs) >= 5:
                self.info.os_name = strs[4]
        c1028 = self.chunk_map.get(0x1028)
        if c1028 and len(c1028.raw) >= 15:
            raw28 = c1028.raw
            self.info.audio_format_code = raw28[9]
            self.info.audio_format_name = AUDIO_FORMAT_MAP.get(raw28[9], f"Format_{raw28[9]}")
            self.info.bit_depth = raw28[10]
            self.info.sample_rate = struct.unpack_from("<I", raw28, 11)[0]
            pull_01 = raw28[15] if len(raw28) > 15 else 0
            if len(raw28) >= 31:
                pull_4 = raw28[-15]
                vpull_01 = raw28[-14]
                self.info.audio_pull_rate = format_pull_rate(pull_4, pull_01)
                self.info.video_pull_rate = format_pull_rate(0, vpull_01)
                self.info.is_interleaved = bool(raw28[-7])
                if raw28[-6] in (16, 24, 32):
                    self.info.bit_depth = raw28[-6]
            else:
                self.info.audio_pull_rate = format_pull_rate(0, pull_01)
        c2067 = self.chunk_map.get(0x2067)
        if c2067:
            sid, paths = _parse_ptx_paths_2067(c2067.raw)
            self.info.session_id_code = sid
            if len(paths) >= 1:
                self.info.saved_session_path = paths[0]
            if len(paths) >= 2:
                self.info.source_template_path = paths[1]

    def _parse_session_start(self) -> None:
        c204d = self.chunk_map.get(0x204D)
        if c204d:
            parse_session_start_204d(c204d.raw, self.info)
        parse_session_ui_state(self.chunk_map, self.info)

    def _parse_tempo(self) -> None:
        c = self.chunk_map.get(0x2028)
        if not c:
            return
        events, declared, bpm = parse_tempo_2028(c.raw)
        self.tempo_events = events
        self.tempo_event_count_declared = declared
        self.has_unparsed_tempo_ramps = declared > len(events)
        self.info.tempo_bpm = bpm
        if self.has_unparsed_tempo_ramps:
            self.warnings.append(
                f"Tempo map declares {declared} events but {len(events)} constant events were parsed (tempo ramps unparsed)."
            )

    def _parse_meter(self) -> None:
        c = self.chunk_map.get(0x2029)
        if c:
            self.meter_events = parse_meter_2029(c.raw)

    def _parse_io_setup_1022(self) -> None:
        c = self.chunk_map.get(0x1022)
        if not c or len(c.raw) < 13:
            return
        pos, idx = 13, 0
        while pos + 9 <= len(c.raw) and c.raw[pos] == 0x5A:
            csz = struct.unpack_from("<I", c.raw, pos + 3)[0]
            end = pos + 7 + csz
            if csz < 2 or end > len(c.raw):
                break
            sp = _parse_signal_path_entry(c.raw[pos:end], idx, "0x1022")
            if sp:
                self.io_paths.append(sp)
                if sp.uuid_hex:
                    self._io1022_by_uuid[bytes.fromhex(sp.uuid_hex)] = sp
            pos = end
            idx += 1
        for sp in self.io_paths:
            if sp.parent_io_index is not None and 0 <= sp.parent_io_index < len(self.io_paths):
                parent = self.io_paths[sp.parent_io_index]
                sp.parent_name = parent.name
                sp.parent_uuid_hex = parent.uuid_hex
                if sp.name not in parent.subpaths:
                    parent.subpaths.append(sp.name)

    def _parse_signal_paths_2603(self) -> None:
        c = self.chunk_map.get(0x2603)
        if not c or len(c.raw) < 13:
            return
        pos, idx = 13, 0
        while pos + 9 <= len(c.raw) and c.raw[pos] == 0x5A:
            csz = struct.unpack_from("<I", c.raw, pos + 3)[0]
            end = pos + 7 + csz
            if csz < 2 or end > len(c.raw):
                break
            sp = _parse_signal_path_entry(c.raw[pos:end], idx, "0x2603")
            if sp:
                self.signal_paths.append(sp)
                if sp.uuid_hex:
                    self._sp2603_by_uuid[bytes.fromhex(sp.uuid_hex)] = sp
                self._sp2603_by_name.setdefault(sp.name, sp)
            pos = end
            idx += 1
        for sp in self.signal_paths:
            if sp.parent_uuid_hex:
                parent = self._sp2603_by_uuid.get(bytes.fromhex(sp.parent_uuid_hex))
                if parent:
                    sp.parent_name = parent.name
                    if sp.name not in parent.subpaths:
                        parent.subpaths.append(sp.name)

    def _parse_plugins_1018(self) -> None:
        c = self.chunk_map.get(0x1018)
        if not c or len(c.raw) < 13:
            return
        count = struct.unpack_from("<I", c.raw, 9)[0]
        pos = 13
        for idx in range(count):
            if pos + 9 > len(c.raw) or c.raw[pos] != 0x5A:
                break
            csz = struct.unpack_from("<I", c.raw, pos + 3)[0]
            end = pos + 7 + csz
            if csz < 2 or end > len(c.raw):
                break
            sub = c.raw[pos:end]
            name, after_name = read_lp_str(sub, 10, max_len=160)
            m4 = p4 = t4 = bundle = ""
            if after_name + 12 <= len(sub):
                m4 = sub[after_name : after_name + 4][::-1].decode("ascii", errors="replace")
                p4 = sub[after_name + 4 : after_name + 8][::-1].decode("ascii", errors="replace")
                t4 = sub[after_name + 8 : after_name + 12][::-1].decode("ascii", errors="replace")
            bm = re.search(rb"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){2,}", sub[after_name:])
            if bm:
                bundle = bm.group().decode("ascii", errors="replace")
            self.plugins.append(PtxPluginInfo(idx, name, m4, p4, t4, bundle))
            pos = end

    def _parse_ara_chunks(self) -> None:
        c4823 = self.chunk_map.get(0x4823)
        if c4823:
            self.ara_plugins = parse_ara_registry_4823(c4823.raw)
        c4820 = self.chunk_map.get(0x4820)
        if c4820:
            self.ara_bindings = parse_ara_bindings_4820(c4820.raw)

    def _parse_tracks_2519(self) -> None:
        c = self.chunk_map.get(0x2519)
        if not c:
            return
        seen_uuids: set[bytes] = set()
        for sub in find_subchunks(c.raw, 0x251A):
            track = self._decode_track_251a(sub, seen_uuids)
            if track is not None:
                self.tracks.append(track)

    def _decode_track_251a(self, sub: bytes, seen_uuids: set[bytes]) -> PtxTrack | None:
        if len(sub) < 40:
            return None
        ver = struct.unpack_from("<H", sub, 1)[0]
        type_code = sub[9]
        name, after_name = read_lp_str(sub, 11, max_len=256)
        if not name or after_name + 18 > len(sub):
            return None
        fmt_byte = sub[after_name]
        uoff = sub.find(UUID_PREFIX, after_name)
        if uoff == -1 or uoff + 16 > len(sub):
            return None
        raw_uuid = sub[uoff : uoff + 16]
        if raw_uuid in seen_uuids:
            return None
        seen_uuids.add(raw_uuid)
        has_v8_tail = ver >= 8 and (uoff + 16 + 11 <= len(sub))
        if has_v8_tail:
            is_folder = bool(sub[-11])
            is_elastic = bool(sub[-10])
            is_hidden = not bool(sub[-7])
            is_inactive = not bool(sub[-6])
            color_idx = struct.unpack_from("<H", sub, len(sub) - 2)[0]
        else:
            is_folder = False
            disp_idx = struct.unpack_from("<I", sub, after_name + 18)[0] if after_name + 22 <= len(sub) else 1
            is_hidden = (disp_idx == 0) and (type_code != 8)
            is_elastic = bool(sub[after_name + 22]) if after_name + 23 <= len(sub) else False
            is_inactive = False
            color_idx = 0xFFFE
            self._tracks_missing_tail_active.add(raw_uuid.hex())
        ttype = "RoutingFolder" if (type_code == 2 and is_folder) else TRACK_TYPE_MAP.get(type_code, f"Type_{type_code}")
        if ttype in ("Vca", "BasicFolder", "Midi", "Video"):
            fmt_str, ch_count = "None", 0
        else:
            ext_code = sub[-9] if (has_v8_tail and sub[-9] != 0 and sub[-9] in TRACK_FORMAT_MAP) else fmt_byte
            fmt_entry = TRACK_FORMAT_MAP.get(ext_code) or TRACK_FORMAT_MAP.get(fmt_byte)
            fmt_str, ch_count = fmt_entry if fmt_entry is not None else (f"Format_{ext_code}", 1)
        return PtxTrack(
            index=len(self.tracks), name=name, uuid_hex=raw_uuid.hex(), uuid_braced=format_ptsl_uuid(raw_uuid),
            track_type=ttype, format=fmt_str, channel_count=ch_count, is_folder=is_folder, is_elastic_or_ara=is_elastic,
            color_index=color_idx, is_hidden=is_hidden, is_inactive=is_inactive,
            color_rgb=lookup_pt_color_rgb(color_idx),
        )

    def _parse_folder_tree_210c(self) -> None:
        c = self.chunk_map.get(0x210C)
        if not c or len(c.raw) < 13:
            return
        by_uuid = {bytes.fromhex(t.uuid_hex): t for t in self.tracks if t.uuid_hex}
        pos = 9
        folder_count = struct.unpack_from("<I", c.raw, pos)[0]
        pos += 4
        for _ in range(folder_count):
            if pos + 21 > len(c.raw):
                break
            folder_uuid = c.raw[pos + 1 : pos + 17]
            child_count = struct.unpack_from("<I", c.raw, pos + 17)[0]
            pos += 21
            folder_track = by_uuid.get(folder_uuid)
            for _ in range(child_count):
                if pos + 17 > len(c.raw):
                    break
                child_track = by_uuid.get(c.raw[pos + 1 : pos + 17])
                pos += 17
                if folder_track and child_track:
                    child_track.parent_folder_name = folder_track.name
                    child_track.parent_folder_uuid = folder_track.uuid_braced
                    folder_track.child_track_names.append(child_track.name)

    def _resolve_sp_uuid(self, raw_uuid: bytes, fallback_name: str = "") -> PtxSignalPathRef | None:
        sp2603 = self._sp2603_by_uuid.get(raw_uuid)
        if sp2603:
            return PtxSignalPathRef(sp2603.name, sp2603.kind, sp2603.uuid_hex, sp2603.index)
        io1022 = self._io1022_by_uuid.get(raw_uuid)
        if io1022:
            ptsl_idx = None
            if io1022.kind != "input":
                match = self._sp2603_by_name.get(io1022.name)
                if match:
                    ptsl_idx = match.index
            return PtxSignalPathRef(io1022.name, io1022.kind, io1022.uuid_hex, ptsl_idx)
        if fallback_name:
            return PtxSignalPathRef(fallback_name, "unknown", raw_uuid.hex(), None)
        return None

    def _parse_mixer_2624(self) -> None:
        c = self.chunk_map.get(0x2624)
        if not c or len(c.raw) < 13:
            return
        by_uuid = {bytes.fromhex(t.uuid_hex): t for t in self.tracks if t.uuid_hex}
        tcount = struct.unpack_from("<I", c.raw, 9)[0]
        pos = 13
        for ti in range(tcount):
            if pos + 9 > len(c.raw) or c.raw[pos] != 0x5A:
                break
            csz = struct.unpack_from("<I", c.raw, pos + 3)[0]
            tag = struct.unpack_from("<H", c.raw, pos + 7)[0]
            end = pos + 7 + csz
            if csz < 2 or end > len(c.raw):
                break
            blk = c.raw[pos:end]
            pos = end
            track = self._match_mixer_track(blk, by_uuid, ti)
            if track is None:
                continue
            if track.track_type == "Video" and (tag == 0x261F or bool(find_subchunks(blk, 0x2610))):
                track.track_type = "Vca"
                track.format = "None"
            parse_track_flags(track, blk, self.ara_plugins)
            if track.track_type == "Vca":
                parse_vca_mixer(track, blk)
                continue
            parse_track_input(track, blk, self._resolve_sp_uuid, self.io_paths)
            parse_track_outputs_and_sends(
                track, blk, self._resolve_sp_uuid,
                derive_inactive_from_output=(track.uuid_hex in self._tracks_missing_tail_active),
            )
            parse_track_inserts(track, blk, self.plugins, self._resolve_sp_uuid)

    def _match_mixer_track(self, blk: bytes, by_uuid: dict[bytes, PtxTrack], index: int) -> PtxTrack | None:
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        if uoff != -1:
            found = by_uuid.get(blk[uoff : uoff + 16])
            if found is not None:
                return found
        return self.tracks[index] if index < len(self.tracks) else None

    def _parse_group_globals_and_presets(self) -> None:
        c204a = self.chunk_map.get(0x204A)
        if c204a:
            s8e = find_subchunks(c204a.raw, 0x258E)
            if s8e:
                self.group_globals = _parse_group_attrs_258e(s8e[0])
        c2510 = self.chunk_map.get(0x2510)
        if c2510:
            self.group_presets = [_parse_group_attrs_258e(x) for x in find_subchunks(c2510.raw, 0x258E)[:6]]

    def _parse_groups_202b(self) -> None:
        c = self.chunk_map.get(0x202B)
        if not c or len(c.raw) < 13:
            return
        try:
            edit_chunks, mix_chunks, _ = split_202b(c.raw)
            ordered: list[tuple[str, bytes]] = [("edit", g) for g in edit_chunks] + [("mix", g) for g in mix_chunks]
        except Exception:
            g_chunks = find_subchunks(c.raw, 0x202A)
            half = len(g_chunks) // 2 if len(g_chunks) >= 2 and len(g_chunks) % 2 == 0 else len(g_chunks)
            ordered = [("edit" if i < half else "mix", g) for i, g in enumerate(g_chunks)]
        wired_names: set[str] = set()
        for gi, (kind, g_raw) in enumerate(ordered):
            grp = self._decode_group_202a(gi, kind, g_raw)
            if grp is not None:
                self.groups.append(grp)
                self._wire_group_backrefs(grp, wired_names)

    def _decode_group_202a(self, index: int, kind: str, raw: bytes) -> PtxGroup | None:
        gname, off = read_lp_str(raw, 9, max_len=128)
        if not gname or off + 11 > len(raw):
            return None
        gtype_code = raw[off]
        id_num = struct.unpack_from("<i", raw, off + 1)[0]
        edit_act, mix_act = bool(raw[off + 5]), bool(raw[off + 6])
        mcnt = struct.unpack_from("<I", raw, off + 7)[0]
        tail_off = off + 11
        members: list[int] = []
        if mcnt <= 2048 and tail_off + 2 * mcnt <= len(raw):
            members = list(struct.unpack_from(f"<{mcnt}H", raw, tail_off))
            tail_off += 2 * mcnt
        tail = raw[tail_off:]
        color_idx = struct.unpack_from("<H", tail, 0)[0] if len(tail) >= 2 else 0xFFFE
        vca_idx: int | None = None
        vca_name: str | None = None
        if len(tail) >= 12:
            raw_vca = struct.unpack_from("<H", tail, 10)[0]
            if raw_vca != 0xFFFF and raw_vca < len(self.tracks):
                vca_idx, vca_name = raw_vca, self.tracks[raw_vca].name
        s8e = find_subchunks(tail, 0x258E, start=0)
        attrs = _parse_group_attrs_258e(s8e[0]) if s8e else None
        follow_glob = attrs.follow_globals if attrs else False
        effective = self.group_globals if (follow_glob and self.group_globals) else attrs
        mnames = [self.tracks[m].name for m in members if m < len(self.tracks)]
        return PtxGroup(
            index=index, name=gname, kind=kind,
            group_type=GROUP_TYPE_MAP.get(gtype_code, f"Type_{gtype_code}"),
            id_number=id_num, id_letter=group_id_to_label(id_num),
            is_edit_active=edit_act, is_mix_active=mix_act, color_index=color_idx,
            vca_track_index=vca_idx, vca_track_name=vca_name,
            member_indices=members, member_names=mnames, follow_globals=follow_glob,
            include_rulers=attrs.include_rulers if attrs else False,
            attributes=attrs, effective_attributes=effective,
        )

    def _wire_group_backrefs(self, grp: PtxGroup, wired_names: set[str]) -> None:
        if grp.name in wired_names:
            return
        wired_names.add(grp.name)
        if grp.name != "<ALL>":
            for m in grp.member_indices:
                if m < len(self.tracks) and grp.name not in self.tracks[m].groups:
                    self.tracks[m].groups.append(grp.name)
        if grp.vca_track_index is not None and grp.vca_track_index < len(self.tracks):
            self.tracks[grp.vca_track_index].vca_group_name = grp.name
            self.tracks[grp.vca_track_index].vca_controlled_tracks = list(grp.member_names)

    def _parse_audio_files(self) -> None:
        c = self.chunk_map.get(0x1004)
        if not c:
            return
        session_root, files = parse_audio_files_1004(c.raw)
        if session_root:
            self.info.session_root = session_root
        self.audio_files = files

    def _parse_clip_gain_curves(self) -> None:
        c = self.chunk_map.get(0x2637)
        if c:
            self.clip_gain_curves = parse_clip_gain_curves_2637(c.raw, self.info.sample_rate)

    def _parse_clips(self) -> None:
        c = self.chunk_map.get(0x262A)
        if c:
            self.clips = parse_clips_262a(c.raw, self.audio_files, self.clip_gain_curves)

    def _parse_fades(self) -> None:
        c = self.chunk_map.get(0x2630)
        if c:
            self.fades = parse_fades_2630(c.raw)

    def _parse_compound_clips(self) -> None:
        c262c = self.chunk_map.get(0x262C)
        c2428 = self.chunk_map.get(0x2428)
        if c262c or c2428:
            self.compound_clips = parse_compound_clips(
                c262c.raw if c262c else None, c2428.raw if c2428 else None,
            )

    def _parse_playlists_and_placements(self) -> None:
        c1054 = self.chunk_map.get(0x1054)
        lanes = (
            parse_timeline_lanes_1054(c1054.raw, self.clips, self.fades, self.compound_clips, self.info.sample_rate)
            if c1054
            else []
        )
        c1015 = self.chunk_map.get(0x1015)
        if c1015:
            self.playlists = parse_playlists_1015(c1015.raw, lanes, self.tracks)

    def _parse_midi(self) -> None:
        c2000 = self.chunk_map.get(0x2000)
        c2634 = self.chunk_map.get(0x2634)
        c1058 = self.chunk_map.get(0x1058)
        if not (c2000 or c2634 or c1058):
            return
        self.midi_chunks, self.midi_clips, self.midi_playlists = parse_midi_data(
            c2000.raw if c2000 else None,
            c2634.raw if c2634 else None,
            c1058.raw if c1058 else None,
            self.tempo_events,
            self.info.tempo_bpm or 120.0,
        )
        for trk in self.tracks:
            if trk.track_type in ("Midi", "Instrument") and trk.name in self.midi_playlists:
                trk.midi_clips = list(self.midi_playlists[trk.name])

    def _parse_video(self) -> None:
        c1006 = self.chunk_map.get(0x1006)
        c262e = self.chunk_map.get(0x262E)
        c1055 = self.chunk_map.get(0x1055)
        if not (c1006 or c262e or c1055):
            return
        self.video_files, self.video_clips, self.video_playlists = parse_video_data(
            c1006.raw if c1006 else None,
            c262e.raw if c262e else None,
            c1055.raw if c1055 else None,
            self.info.sample_rate,
        )
        for trk in self.tracks:
            if trk.track_type == "Video" and trk.name in self.video_playlists:
                trk.video_clips = list(self.video_playlists[trk.name])

    def _parse_track_ui_2587(self) -> None:
        c2587 = self.chunk_map.get(0x2587)
        if not c2587:
            return
        entries = find_subchunks(c2587.raw, 0x2589)
        for entry in entries:
            if len(entry) < 20:
                continue
            tidx = struct.unpack_from("<H", entry, 9)[0]
            if tidx >= len(self.tracks):
                continue
            trk = self.tracks[tidx]
            if trk.is_folder or trk.track_type in ("BasicFolder", "RoutingFolder"):
                trk.is_folder_open = not bool(entry[-7])

    def _parse_markers(self) -> None:
        self.markers = parse_markers_2030(self.chunks, self.info.sample_rate, tracks=self.tracks)
