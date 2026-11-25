"""Pure-Python Pro Tools ``.ptx`` / ``.ptxt`` binary session parser."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

from .codec import (
    decode_slot_mask,
    decrypt_ptx,
    find_subchunks,
    format_ptsl_uuid,
    format_rfc_uuid,
    group_id_to_label,
    iter_top_chunks,
    read_lp_str,
)
from .models import (
    AUTOMATION_MODE_MAP,
    GROUP_TYPE_MAP,
    IO_KIND_MAP,
    SEND_SLOTS,
    TRACK_TYPE_MAP,
    UUID_PREFIX,
    PtxAudioFile,
    PtxChunk,
    PtxClip,
    PtxClipPlacement,
    PtxGroup,
    PtxGroupAttributes,
    PtxInsert,
    PtxPlaylist,
    PtxPluginInfo,
    PtxSend,
    PtxSessionInfo,
    PtxSignalPath,
    PtxSignalPathRef,
    PtxTrack,
)


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


def _extract_260d_params(
    chunk_260d: bytes,
) -> tuple[float | None, bool, float | None, int | None, int | None, str]:
    cpos = chunk_260d.find(b"\x0c\x26")
    head_part = chunk_260d[:cpos] if cpos != -1 else chunk_260d
    direct_a = find_subchunks(head_part, 0x260A)
    vol_db: float | None = None
    is_muted = False
    lfe_db: float | None = None
    timebase = "Samples"
    if len(direct_a) >= 1 and len(direct_a[0]) >= 39:
        vol_db = round(struct.unpack_from("<h", direct_a[0], len(direct_a[0]) - 4)[0] / 10.0, 1)
        if direct_a[0][17] == 1:
            timebase = "Ticks"
    if len(direct_a) >= 2 and len(direct_a[1]) >= 39:
        is_muted = bool(struct.unpack_from("<h", direct_a[1], len(direct_a[1]) - 4)[0])
    if len(direct_a) >= 3 and len(direct_a[2]) >= 39:
        lfe_db = round(struct.unpack_from("<h", direct_a[2], len(direct_a[2]) - 4)[0] / 10.0, 1)

    pan_l, pan_r = _extract_pan_pair(chunk_260d)
    return vol_db, is_muted, lfe_db, pan_l, pan_r, timebase


def _extract_pan_pair(chunk_260d: bytes) -> tuple[int | None, int | None]:
    pan_l: int | None = None
    pan_r: int | None = None
    c_list = find_subchunks(chunk_260d, 0x260C)
    if len(c_list) >= 1:
        pa = find_subchunks(c_list[0], 0x260A)
        if pa and len(pa[0]) >= 39:
            pan_l = struct.unpack_from("<h", pa[0], len(pa[0]) - 4)[0]
    if len(c_list) >= 2:
        pa = find_subchunks(c_list[1], 0x260A)
        if pa and len(pa[0]) >= 39:
            pan_r = struct.unpack_from("<h", pa[0], len(pa[0]) - 4)[0]
    return pan_l, pan_r


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
    """Decoded Pro Tools ``.ptx`` or ``.ptxt`` session."""

    def __init__(self, decrypted: bytes, xor_delta: int = 0, file_path: str | None = None):
        self.decrypted = decrypted
        self.info = PtxSessionInfo(file_path=file_path, xor_delta=xor_delta)
        self.chunks: list[PtxChunk] = iter_top_chunks(decrypted)
        self.chunk_map: dict[int, PtxChunk] = {c.tag: c for c in self.chunks}
        self.io_paths: list[PtxSignalPath] = []
        self.signal_paths: list[PtxSignalPath] = []
        self.plugins: list[PtxPluginInfo] = []
        self.tracks: list[PtxTrack] = []
        self.groups: list[PtxGroup] = []
        self.group_globals: PtxGroupAttributes | None = None
        self.group_presets: list[PtxGroupAttributes] = []
        self.audio_files: list[PtxAudioFile] = []
        self.clips: list[PtxClip] = []
        self.playlists: list[PtxPlaylist] = []
        self._io1022_by_uuid: dict[bytes, PtxSignalPath] = {}
        self._sp2603_by_uuid: dict[bytes, PtxSignalPath] = {}
        self._sp2603_by_name: dict[str, PtxSignalPath] = {}
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

    def routing_matrix(self, ptsl_base: int | None = None) -> dict[str, Any]:
        """Build a routing matrix & bus connectivity graph for the session."""
        bus_graph: dict[str, dict[str, list[str]]] = {}

        def _ensure_bus(bus_name: str) -> dict[str, list[str]]:
            return bus_graph.setdefault(
                bus_name,
                {"fed_by_main": [], "fed_by_send": [], "feeds_input": [], "feeds_sidechain": []},
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

    def _routing_row(self, track: PtxTrack, ptsl_base: int | None, ensure_bus: Any) -> dict[str, Any]:
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
            "index": track.index,
            "name": track.name,
            "id": track.uuid_braced,
            "type": track.track_type,
            "format": track.format,
            "parent_folder": track.parent_folder_name or None,
            "input": self._format_sp_ref(track.input_path, ptsl_base),
            "main_output": self._format_sp_ref(track.main_output, ptsl_base),
            "volume_db": track.volume_db,
            "pan": [track.pan_left, track.pan_right]
            if (track.pan_left is not None or track.pan_right is not None)
            else None,
            "muted": track.is_muted,
            "inactive": track.is_inactive,
            "vca_group": track.vca_group_name,
            "vca_controlled_tracks": list(track.vca_controlled_tracks),
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

    def _parse_all(self) -> None:
        self._parse_version_and_header()
        self._parse_tempo()
        self._parse_io_setup_1022()
        self._parse_signal_paths_2603()
        self._parse_plugins_1018()
        self._parse_tracks_2519()
        self._parse_folder_tree_210c()
        self._parse_mixer_2624()
        self._parse_group_globals_and_presets()
        self._parse_groups_202b()
        self._parse_audio_files_1004()
        self._parse_clips_262a()
        self._parse_playlists_and_placements()

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
            self.info.audio_format_code = c1028.raw[9]
            self.info.bit_depth = c1028.raw[10]
            self.info.sample_rate = struct.unpack_from("<I", c1028.raw, 11)[0]
        c2067 = self.chunk_map.get(0x2067)
        if c2067:
            sid, paths = _parse_ptx_paths_2067(c2067.raw)
            self.info.session_id_code = sid
            if len(paths) >= 1:
                self.info.saved_session_path = paths[0]
            if len(paths) >= 2:
                self.info.source_template_path = paths[1]

    def _parse_tempo(self) -> None:
        c = self.chunk_map.get(0x2028)
        if not c:
            return
        sent = c.raw.find(b"\x00\x10\xa5\xd4\xe8\x00")
        if sent == -1 or sent + 22 > len(c.raw):
            return
        for off in range(sent + 6, len(c.raw) - 7):
            val = struct.unpack_from("<d", c.raw, off)[0]
            if 20.0 <= val <= 400.0 and abs(val * 100 - round(val * 100)) < 1e-4:
                self.info.tempo_bpm = round(val, 4)
                return

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
        is_folder = bool(sub[-11])
        is_elastic = bool(sub[-10])
        is_hidden = not bool(sub[-7])
        is_inactive = not bool(sub[-6])
        color_idx = struct.unpack_from("<H", sub, len(sub) - 2)[0]
        ttype = "RoutingFolder" if (type_code == 2 and is_folder) else TRACK_TYPE_MAP.get(type_code, f"Type_{type_code}")
        fmt_str = "None" if ttype in ("Vca", "BasicFolder", "Midi", "Video") else ("Stereo" if fmt_byte == 1 else "Mono")
        return PtxTrack(
            index=len(self.tracks),
            name=name,
            uuid_hex=raw_uuid.hex(),
            uuid_braced=format_ptsl_uuid(raw_uuid),
            track_type=ttype,
            format=fmt_str,
            is_folder=is_folder,
            is_elastic_or_ara=is_elastic,
            color_index=color_idx,
            is_hidden=is_hidden,
            is_inactive=is_inactive,
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

    def _extract_260e_ref(self, chunk_260e: bytes) -> PtxSignalPathRef | None:
        uoff = chunk_260e.find(UUID_PREFIX)
        if uoff == -1 or uoff + 16 > len(chunk_260e):
            return None
        return self._resolve_sp_uuid(chunk_260e[uoff : uoff + 16])

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
            end = pos + 7 + csz
            if csz < 2 or end > len(c.raw):
                break
            blk = c.raw[pos:end]
            pos = end
            track = self._match_mixer_track(blk, by_uuid, ti)
            if track is None:
                continue
            self._parse_track_flags(track, blk)
            if track.track_type == "Vca":
                self._parse_vca_mixer(track, blk)
                continue
            self._parse_track_input(track, blk)
            self._parse_track_outputs_and_sends(track, blk)
            self._parse_track_inserts(track, blk)

    def _match_mixer_track(self, blk: bytes, by_uuid: dict[bytes, PtxTrack], index: int) -> PtxTrack | None:
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        if uoff != -1:
            found = by_uuid.get(blk[uoff : uoff + 16])
            if found is not None:
                return found
        return self.tracks[index] if index < len(self.tracks) else None

    @staticmethod
    def _parse_track_flags(track: PtxTrack, blk: bytes) -> None:
        s4301 = find_subchunks(blk[:512], 0x4301)
        if s4301:
            h4301 = blk.find(s4301[0])
            if h4301 != -1 and h4301 + 61 <= len(blk):
                track.is_soloed = bool(blk[h4301 + 59])
                am_code = blk[h4301 + 60]
                track.automation_mode_code = am_code
                track.automation_mode = AUTOMATION_MODE_MAP.get(am_code, f"Mode_{am_code}")
        tail_blk = blk[max(0, len(blk) - 900) :]
        s2038 = find_subchunks(tail_blk, 0x2038, start=0)
        if s2038 and len(s2038[0]) >= 20:
            track.is_record_enabled = bool(s2038[0][18])
            track.is_record_safe = bool(s2038[0][19])
        if track.track_type == "Audio" and len(blk) > 33:
            track.is_input_monitor = bool(blk[-33])

    @staticmethod
    def _parse_vca_mixer(track: PtxTrack, blk: bytes) -> None:
        s2610 = find_subchunks(blk, 0x2610)
        if not s2610:
            return
        a_list = find_subchunks(s2610[0], 0x260A)
        if len(a_list) >= 1 and len(a_list[0]) >= 39:
            track.volume_db = round(struct.unpack_from("<h", a_list[0], len(a_list[0]) - 4)[0] / 10.0, 1)
        if len(a_list) >= 2 and len(a_list[1]) >= 39:
            track.is_muted = bool(struct.unpack_from("<h", a_list[1], len(a_list[1]) - 4)[0])

    def _parse_track_input(self, track: PtxTrack, blk: bytes) -> None:
        s4856_list = find_subchunks(blk, 0x4856)
        if not s4856_list:
            return
        h4856 = blk.rfind(s4856_list[-1])
        pre = blk[max(0, h4856 - 480) : h4856]
        for match in re.finditer(re.escape(UUID_PREFIX), pre):
            ref = self._resolve_sp_uuid(pre[match.start() : match.start() + 16])
            if ref:
                track.input_path = ref
                return

    def _parse_track_outputs_and_sends(self, track: PtxTrack, blk: bytes) -> None:
        for di, d_chunk in enumerate(find_subchunks(blk, 0x260D)):
            e_list = find_subchunks(d_chunk, 0x260E)
            if not e_list:
                continue
            dest = self._extract_260e_ref(e_list[0])
            if not dest:
                continue
            vol_db, is_muted, lfe_db, pan_l, pan_r, tbase = _extract_260d_params(d_chunk)
            if di == 0:
                track.main_output = dest
                track.volume_db = vol_db
                track.is_muted = is_muted
                track.lfe_db = lfe_db
                track.pan_left = pan_l
                track.pan_right = pan_r
                track.timebase = tbase
            else:
                slot_idx = di - 1
                slot = SEND_SLOTS[slot_idx] if slot_idx < len(SEND_SLOTS) else str(slot_idx)
                s1029 = find_subchunks(d_chunk, 0x1029)
                is_pre = bool(s1029 and len(s1029[0]) > 9 and s1029[0][9] == 0)
                track.sends.append(
                    PtxSend(slot, slot_idx, dest, vol_db, is_muted, is_pre, lfe_db, pan_l, pan_r)
                )

    def _parse_track_inserts(self, track: PtxTrack, blk: bytes) -> None:
        s2627 = find_subchunks(blk, 0x2627)
        if not s2627:
            return
        b27 = s2627[0]
        # Real Pro Tools 0x2627 has a 2-byte slot count at +9 followed by sequential 0x2616/0x2625 slots at +11.
        pos = 11 if (len(b27) > 11 and b27[11] == 0x5A) else 9
        si = 0
        while pos + 9 <= len(b27) and b27[pos] == 0x5A:
            ver = struct.unpack_from("<H", b27, pos + 1)[0]
            csz = struct.unpack_from("<I", b27, pos + 3)[0]
            tag = struct.unpack_from("<H", b27, pos + 7)[0]
            end = pos + 7 + csz
            if ver >= 0x1000 or csz < 2 or end > len(b27):
                break
            if tag == 0x2625:
                si += 1
                pos = end
                continue
            if tag == 0x2616:
                slot_blk = b27[pos:end]
                s2626 = find_subchunks(slot_blk, 0x2626)
                if s2626:
                    plug_idx, plug_name, bundle_id, fourcc = self._resolve_insert_plugin(slot_blk, s2626[0])
                else:
                    plug_idx, plug_name, bundle_id, fourcc = None, "(Hardware / Unknown Insert)", "", ""
                sc_chunks = find_subchunks(slot_blk, 0x260E)
                sc_ref = self._extract_260e_ref(sc_chunks[0]) if sc_chunks else None
                is_active, is_bypassed = self._parse_insert_flags(slot_blk)
                slot = SEND_SLOTS[si] if si < len(SEND_SLOTS) else str(si)
                track.inserts.append(
                    PtxInsert(slot, si, plug_idx, plug_name, bundle_id, fourcc, is_active, is_bypassed, sc_ref)
                )
                if si == 0 and plug_idx is not None and plug_idx < len(self.plugins):
                    from .plugin_trim import inspect_slot_a_trim
                    track.slot_a_trim = inspect_slot_a_trim(
                        slot_blk, self.plugins[plug_idx],
                        is_active=is_active, is_bypassed=is_bypassed,
                        track_format=track.format, has_sidechain=(sc_ref is not None),
                    )
                si += 1
            pos = end

    def _resolve_insert_plugin(self, slot_blk: bytes, s2626: bytes) -> tuple[int | None, str, str, str]:
        idx_pos = slot_blk.find(s2626) + len(s2626)
        if idx_pos + 2 <= len(slot_blk):
            cand = struct.unpack_from("<H", slot_blk, idx_pos)[0]
            if cand < len(self.plugins):
                pinfo = self.plugins[cand]
                fourcc = f"{pinfo.manufacturer_4cc}/{pinfo.product_4cc}/{pinfo.type_4cc}"
                return cand, pinfo.name, pinfo.bundle_id, fourcc
        return None, "", "", ""

    @staticmethod
    def _parse_insert_flags(slot_blk: bytes) -> tuple[bool, bool]:
        is_active = True
        is_bypassed = False
        s2613 = find_subchunks(slot_blk, 0x2613)
        s1038 = find_subchunks(slot_blk, 0x1038)
        if s2613:
            b13 = s2613[0]
            if s1038:
                pos_after38 = b13.find(s1038[0]) + len(s1038[0])
                if pos_after38 < len(b13):
                    is_active = bool(b13[pos_after38])
            pos_after13 = slot_blk.find(b13) + len(b13)
            if pos_after13 < len(slot_blk):
                is_bypassed = bool(slot_blk[pos_after13])
        return is_active, is_bypassed

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
            index=index,
            name=gname,
            kind=kind,
            group_type=GROUP_TYPE_MAP.get(gtype_code, f"Type_{gtype_code}"),
            id_number=id_num,
            id_letter=group_id_to_label(id_num),
            is_edit_active=edit_act,
            is_mix_active=mix_act,
            color_index=color_idx,
            vca_track_index=vca_idx,
            vca_track_name=vca_name,
            member_indices=members,
            member_names=mnames,
            follow_globals=follow_glob,
            include_rulers=attrs.include_rulers if attrs else False,
            attributes=attrs,
            effective_attributes=effective,
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

    def _parse_audio_files_1004(self) -> None:
        c = self.chunk_map.get(0x1004)
        if not c:
            return
        directory, filenames = self._parse_audio_filenames(c.raw)
        s1003 = find_subchunks(c.raw, 0x1003)
        for i in range(max(len(filenames), len(s1003))):
            fname = filenames[i] if i < len(filenames) else f"file_{i}"
            file_num, sr, chs, bits, length, fmt_code = self._decode_audio_entry(s1003[i], i + 1) if i < len(s1003) else (i + 1, 0, 0, 0, 0, 0)
            dur = round(length / sr, 4) if (sr > 0 and length > 0) else None
            self.audio_files.append(PtxAudioFile(i, file_num, fname, directory, sr, chs, bits, length, dur, fmt_code))

    @staticmethod
    def _parse_audio_filenames(raw: bytes) -> tuple[str, list[str]]:
        directory = ""
        filenames: list[str] = []
        s103a = find_subchunks(raw, 0x103A)
        if s103a:
            blob = s103a[0]
            if re.search(rb"Audio Files", blob):
                directory = "Audio Files"
            for m in re.finditer(rb"[\x20-\x7e]{1,200}\.(?:wav|WAV|aif|aiff|AIF|AIFF|mp3|m4a|flac|sd2)", blob):
                fn = m.group().decode("utf-8", errors="replace").strip()
                if fn not in filenames:
                    filenames.append(fn)
        return directory, filenames

    @staticmethod
    def _decode_audio_entry(b3: bytes, default_num: int) -> tuple[int, int, int, int, int, int]:
        file_num = struct.unpack_from("<I", b3, 9)[0] if len(b3) >= 13 else default_num
        s1001 = find_subchunks(b3, 0x1001)
        if s1001 and len(s1001[0]) >= 24:
            b1 = s1001[0]
            return file_num, struct.unpack_from("<I", b1, 9)[0], b1[13], b1[14], struct.unpack_from("<Q", b1, 15)[0], b1[23]
        return file_num, 0, 0, 0, 0, 0

    def _parse_clips_262a(self) -> None:
        curves_db = self._parse_clip_gain_curves_2637()
        c = self.chunk_map.get(0x262A)
        if not c:
            return
        for ci, sub in enumerate(find_subchunks(c.raw, 0x2629)):
            clip = self._decode_clip_2629(ci, sub, curves_db)
            if clip is not None:
                self.clips.append(clip)

    def _parse_clip_gain_curves_2637(self) -> list[float]:
        """Extract per-curve static Clip Gain (dB) from the top-level ``0x2637`` chunk."""
        c = self.chunk_map.get(0x2637)
        if not c or len(c.raw) < 13:
            return []
        raw = c.raw
        cnt = struct.unpack_from("<I", raw, 9)[0]
        pos = 13
        curves: list[float] = []
        for _ in range(cnt):
            if pos + 8 > len(raw):
                break
            _flags, magic, _ver, plen = struct.unpack_from("<BBHI", raw, pos)
            end = pos + 8 + plen
            if magic != 0x46 or end > len(raw):
                break
            gdb = struct.unpack_from("<f", raw, pos + 8 + 18)[0] if plen >= 22 else 0.0
            curves.append(float(gdb) if math.isfinite(gdb) else 0.0)
            pos = end
        return curves

    def _decode_clip_2629(self, index: int, sub: bytes, curves_db: list[float]) -> PtxClip | None:
        s28 = find_subchunks(sub, 0x2628)
        if not s28:
            return None
        b28 = s28[0]
        cname, after_name = read_lp_str(b28, 9, max_len=256)
        if not cname:
            return None
        start_smp, len_smp = self._decode_clip_range(b28, after_name)
        clip_id = format_rfc_uuid(b28[-16:]) if len(b28) >= 16 else ""
        b28_ver = struct.unpack_from("<H", b28, 1)[0] if len(b28) >= 3 else 0
        cg_idx = (
            struct.unpack_from("<i", b28, len(b28) - 22)[0]
            if (b28_ver >= 5 and len(b28) >= 31 and after_name <= len(b28) - 22)
            else -1
        )
        b28_end = sub.find(b28) + len(b28)
        file_idx: int | None = None
        file_name = ""
        ch_idx = 0
        if b28_end + 6 <= len(sub):
            fi, chi = struct.unpack_from("<IH", sub, b28_end)
            if fi < len(self.audio_files):
                file_idx, file_name = fi, self.audio_files[fi].filename
                len_smp = len_smp or self.audio_files[fi].length_samples
            ch_idx = chi
        gain_lin, gain_db = 1.0, 0.0
        if 0 <= cg_idx < len(curves_db):
            gain_db = round(curves_db[cg_idx], 2)
            gain_lin = 0.0 if gain_db <= -144.0 else round(10.0 ** (gain_db / 20.0), 6)
        return PtxClip(index, cname, clip_id, file_idx, file_name, ch_idx, start_smp, len_smp, gain_lin, gain_db, cg_idx)

    @staticmethod
    def _decode_clip_range(b28: bytes, after_name: int) -> tuple[int, int]:
        start_smp = len_smp = 0
        if after_name + 5 <= len(b28):
            flag = b28[after_name + 2]
            len_w, start_w = (flag >> 4) & 0x0F, flag & 0x0F
            pos = after_name + 5
            if start_w <= 8 and len_w <= 8 and pos + start_w + len_w <= len(b28):
                if start_w > 0:
                    start_smp = int.from_bytes(b28[pos : pos + start_w], "little")
                    pos += start_w
                if len_w > 0:
                    len_smp = int.from_bytes(b28[pos : pos + len_w], "little")
        return start_smp, len_smp

    def _parse_playlists_and_placements(self) -> None:
        lanes = self._parse_timeline_lanes_1054()
        c1015 = self.chunk_map.get(0x1015)
        if not c1015:
            return
        tracks_by_name = {t.name: t for t in self.tracks if t.track_type == "Audio"}
        for pi, s14 in enumerate(find_subchunks(c1015.raw, 0x1014)):
            playlist = self._decode_playlist_1014(pi, s14, lanes)
            if playlist is None:
                continue
            self.playlists.append(playlist)
            trk = tracks_by_name.get(playlist.name)
            if trk and not trk.clips:
                trk.clips = list(playlist.placements)

    def _parse_timeline_lanes_1054(self) -> list[tuple[str, list[PtxClipPlacement]]]:
        lanes: list[tuple[str, list[PtxClipPlacement]]] = []
        c1054 = self.chunk_map.get(0x1054)
        if not c1054:
            return lanes
        clips_by_idx = {clip.index: clip for clip in self.clips}
        for s52 in find_subchunks(c1054.raw, 0x1052):
            lname, _ = read_lp_str(s52, 9, max_len=256)
            lane_clips: list[PtxClipPlacement] = []
            for s4f in find_subchunks(s52, 0x104F):
                if len(s4f) < 27:
                    continue
                clip_idx = struct.unpack_from("<I", s4f, 11)[0]
                start_smp = struct.unpack_from("<Q", s4f, 16)[0]
                color_idx = struct.unpack_from("<H", s4f, 25)[0]
                clip_obj = clips_by_idx.get(clip_idx)
                clen = clip_obj.length_samples if clip_obj else 0
                lane_clips.append(
                    PtxClipPlacement(
                        channel=0,
                        clip_index=clip_idx,
                        clip_name=clip_obj.name if clip_obj else f"clip_{clip_idx}",
                        audio_file_name=clip_obj.audio_file_name if clip_obj else "",
                        start_sample=start_smp,
                        length_samples=clen,
                        end_sample=start_smp + clen,
                        color_index=color_idx,
                    )
                )
            lanes.append((lname, lane_clips))
        return lanes

    @staticmethod
    def _decode_playlist_1014(
        index: int, s14: bytes, lanes: list[tuple[str, list[PtxClipPlacement]]]
    ) -> PtxPlaylist | None:
        pname, off = read_lp_str(s14, 9, max_len=256)
        if not pname or off + 5 > len(s14):
            return None
        ch_count = struct.unpack_from("<I", s14, off + 1)[0]
        lp = off + 5
        lane_idxs = list(struct.unpack_from(f"<{ch_count}H", s14, lp)) if (0 < ch_count <= 64 and lp + 2 * ch_count <= len(s14)) else []
        uoff = s14.find(UUID_PREFIX, lp)
        pl_uuid = format_ptsl_uuid(s14[uoff : uoff + 16]) if (uoff != -1 and uoff + 16 <= len(s14)) else ""
        placements: list[PtxClipPlacement] = []
        for ch_i, lidx in enumerate(lane_idxs):
            if 0 <= lidx < len(lanes):
                for plc in lanes[lidx][1]:
                    placements.append(
                        PtxClipPlacement(
                            ch_i, plc.clip_index, plc.clip_name, plc.audio_file_name,
                            plc.start_sample, plc.length_samples, plc.end_sample, plc.color_index,
                        )
                    )
        return PtxPlaylist(index, pname, pl_uuid, ch_count if ch_count <= 64 else len(lane_idxs), lane_idxs, placements)
