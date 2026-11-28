"""Validated in-memory and atomic-file ``.ptx`` patchers (volumes, clip gain, groups & VCAs)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import math
from pathlib import Path
import struct
from typing import Any

from .codec import (
    atomic_write_bytes,
    decrypt_ptx,
    encode_slot_mask,
    encrypt_ptx,
    find_subchunks,
    group_id_to_label,
    iter_top_chunks,
    read_lp_str,
    shift_index_0002,
    splice_top_chunk,
    validate_index_0002,
)
from .models import GROUP_TYPE_MAP, MAX_GROUP_ID, UUID_PREFIX, PtxGroup, PtxGroupAttributes
from .parser import PtxSession, split_202b
from .plugin_trim import patch_slot_a_trim_gains_on_session

_UNSET: Any = object()
_GROUP_TYPE_TO_CODE = {v.lower(): k for k, v in GROUP_TYPE_MAP.items()}


def _require_finite_db(value: float, label: str) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return val


def _patch_mixer_block_volume(buf: bytearray, blk_off: int, blk: bytes, val_tenths: int, tname: str) -> float:
    d_list = find_subchunks(blk, 0x260D)
    if d_list:
        d0 = d_list[0]
        d0_off = blk_off + blk.find(d0)
        prev_db: float | None = None
        wrote = False
        p1029_list = find_subchunks(d0, 0x1029)
        if p1029_list and len(p1029_list[0]) >= 100:
            p1029_off = d0_off + d0.find(p1029_list[0])
            prev_db = round(struct.unpack_from("<i", buf, p1029_off + 10)[0] / 10.0, 1)
            struct.pack_into("<i", buf, p1029_off + 10, val_tenths)
            struct.pack_into("<i", buf, p1029_off + 96, val_tenths)
            wrote = True
        cpos = d0.find(b"\x0c\x26")
        head_part = d0[:cpos] if cpos != -1 else d0
        a_list = find_subchunks(head_part, 0x260A)
        if a_list and len(a_list[0]) >= 39:
            a0_off = d0_off + head_part.find(a_list[0])
            if prev_db is None:
                prev_db = round(struct.unpack_from("<h", buf, a0_off + len(a_list[0]) - 4)[0] / 10.0, 1)
            struct.pack_into("<h", buf, a0_off + len(a_list[0]) - 4, val_tenths)
            wrote = True
        if not wrote:
            raise RuntimeError(f"Track {tname!r} 0x260d block has no writable volume fields")
        return prev_db if prev_db is not None else 0.0

    s2610 = find_subchunks(blk, 0x2610)
    if s2610:
        s10_off = blk_off + blk.find(s2610[0])
        a_list = find_subchunks(s2610[0], 0x260A)
        if a_list and len(a_list[0]) >= 39:
            a0_off = s10_off + s2610[0].find(a_list[0])
            prev = round(struct.unpack_from("<h", buf, a0_off + len(a_list[0]) - 4)[0] / 10.0, 1)
            struct.pack_into("<h", buf, a0_off + len(a_list[0]) - 4, val_tenths)
            return prev
    raise RuntimeError(f"Track {tname!r} has neither 0x260d nor 0x2610 volume section")


def patch_track_volumes(
    ptx_raw: bytes,
    track_volumes_db: dict[str, float],
) -> tuple[bytes, dict[str, dict[str, float]]]:
    """Patch static main-fader volumes (``-144.0`` to ``+12.0 dB``, ``0.1 dB`` resolution)."""
    if not track_volumes_db:
        raise ValueError("track_volumes_db must be a non-empty {track_name: db} dict")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    session = PtxSession(dec, xor_delta=delta)

    targets_by_uuid: dict[bytes, tuple[str, float]] = {}
    for tname, db in track_volumes_db.items():
        val_db = _require_finite_db(db, f"Track volume for {tname!r}")
        trk = session.track_by_name(tname)
        if trk is None:
            raise KeyError(f"Track {tname!r} not found in .ptx session")
        targets_by_uuid[bytes.fromhex(trk.uuid_hex)] = (trk.name, max(-144.0, min(12.0, val_db)))

    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(buf):
        raise RuntimeError("Missing or truncated 0x2624 mixer chunk in .ptx")
    chunk_end = min(len(buf), c2624.offset + 7 + c2624.size)
    tcount = struct.unpack_from("<I", buf, c2624.offset + 9)[0]
    pos = c2624.offset + 13
    changes: dict[str, dict[str, float]] = {}
    for _ in range(tcount):
        if pos + 9 > chunk_end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        blk_end = pos + 7 + sz
        if sz < 2 or blk_end > chunk_end:
            break
        blk = bytes(buf[pos:blk_end])
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        if uoff != -1 and blk[uoff : uoff + 16] in targets_by_uuid:
            tname, target_db = targets_by_uuid[blk[uoff : uoff + 16]]
            val_tenths = int(round(target_db * 10.0))
            prev_db = _patch_mixer_block_volume(buf, pos, blk, val_tenths, tname)
            changes[tname] = {"previous_db": prev_db, "target_db": round(val_tenths / 10.0, 1)}
        pos = blk_end

    missing = [nm for nm, _ in targets_by_uuid.values() if nm not in changes]
    if missing:
        raise RuntimeError(f"Could not locate mixer blocks for track(s): {missing}")
    return encrypt_ptx(bytes(buf), delta, k0=k0), changes


def _parse_2637_records(raw: bytes) -> tuple[list[bytearray], bytes]:
    if len(raw) < 13:
        raise RuntimeError("Truncated 0x2637 clip-gain curves chunk in .ptx")
    cnt = struct.unpack_from("<I", raw, 9)[0]
    pos = 13
    records: list[bytearray] = []
    for i in range(cnt):
        if pos + 8 > len(raw):
            raise RuntimeError(f"Truncated 0x2637 curve header at index {i}")
        _flags, magic, _ver, plen = struct.unpack_from("<BBHI", raw, pos)
        end = pos + 8 + plen
        if magic != 0x46 or end > len(raw):
            raise RuntimeError(f"Malformed 0x2637 curve record at index {i}")
        records.append(bytearray(raw[pos:end]))
        pos = end
    return records, raw[pos:]


def _build_static_clip_gain_record(gain_db: float, length_samples: int = 0) -> bytearray:
    """Build a 30-byte Pro Tools ``0x2637`` single-point static Clip Gain record."""
    clamped_len = min(max(0, int(length_samples)), 0xFFFFFFFF)
    return bytearray(
        struct.pack("<BBHIHIHHIIf", 1, 0x46, 1, 22, 0, 1, 4, 0, clamped_len, 0, float(gain_db))
    )


def _build_2637_chunk(records: list[bytearray], version: int = 1, trailer: bytes = b"") -> bytes:
    body = struct.pack("<I", len(records)) + b"".join(records) + trailer
    return b"Z" + struct.pack("<HIH", version, 2 + len(body), 0x2637) + body


def patch_clip_gains(
    ptx_raw: bytes,
    track_clip_gains_db: dict[str, float],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Patch static Clip Gain (``-144.0`` to ``+36.0 dB``) via ``0x2637`` curves & ``0x2628`` links."""
    if not track_clip_gains_db:
        raise ValueError("track_clip_gains_db must be a non-empty {track_name: db} dict")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    session = PtxSession(dec, xor_delta=delta)
    clip_targets, summary = _resolve_clip_gain_targets(session, track_clip_gains_db)

    c2637 = session.chunk_map.get(0x2637)
    if c2637 is None:
        raise RuntimeError("Missing 0x2637 clip-gain curves chunk in .ptx")
    c262a = session.chunk_map.get(0x262A)
    if c262a is None:
        raise RuntimeError("Missing 0x262a clips chunk in .ptx")
    records, trailer = _parse_2637_records(c2637.raw)
    curve_refcounts: dict[int, int] = {}
    clip_lengths: dict[int, int] = {}
    for clip in session.clips:
        clip_lengths[clip.index] = clip.length_samples
        if clip.clip_gain_curve_index >= 0:
            curve_refcounts[clip.clip_gain_curve_index] = curve_refcounts.get(clip.clip_gain_curve_index, 0) + 1
    chunk_end = min(len(buf), c262a.offset + 7 + c262a.size)
    patched_count = _apply_clip_gain_writes(
        buf, c262a.offset, chunk_end, clip_targets, records, curve_refcounts, clip_lengths, summary,
    )
    if patched_count != len(clip_targets):
        raise RuntimeError(f"Patched {patched_count}/{len(clip_targets)} target clips in 0x262a")
    splice_top_chunk(buf, 0x2637, _build_2637_chunk(records, version=c2637.version, trailer=trailer))
    return encrypt_ptx(bytes(buf), delta, k0=k0), summary


def _resolve_clip_gain_targets(
    session: PtxSession, track_clip_gains_db: dict[str, float]
) -> tuple[dict[int, tuple[list[str], float]], dict[str, dict[str, Any]]]:
    clip_targets: dict[int, tuple[list[str], float]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tname, db in track_clip_gains_db.items():
        val_db = _require_finite_db(db, f"Clip gain for {tname!r}")
        trk = session.track_by_name(tname)
        if trk is None:
            raise KeyError(f"Track {tname!r} not found in .ptx session")
        clamped = max(-144.0, min(36.0, val_db))
        indices = sorted({pl.clip_index for pl in trk.clips})
        if not indices:
            raise ValueError(f"Track {trk.name!r} has no placed clips to patch")
        for ci in indices:
            existing = clip_targets.get(ci)
            if existing is not None:
                prev_tracks, prev_db = existing
                if abs(prev_db - clamped) > 1e-6:
                    raise ValueError(
                        f"Clip index {ci} is shared by {prev_tracks} ({prev_db} dB) and {trk.name!r} ({clamped} dB)"
                    )
                prev_tracks.append(trk.name)
            else:
                clip_targets[ci] = ([trk.name], clamped)
        summary[trk.name] = {"target_db": round(clamped, 2), "clamped": clamped != val_db, "clip_count": len(indices)}
    return clip_targets, summary


def _apply_clip_gain_writes(
    buf: bytearray,
    base: int,
    end_262a: int,
    clip_targets: dict[int, tuple[list[str], float]],
    records: list[bytearray],
    curve_refcounts: dict[int, int],
    clip_lengths: dict[int, int],
    summary: dict[str, dict[str, Any]],
) -> int:
    pos = base + 9
    ci = 0
    patched = 0
    while pos + 9 <= end_262a:
        if buf[pos] == 0x5A:
            version = struct.unpack_from("<H", buf, pos + 1)[0]
            sz = struct.unpack_from("<I", buf, pos + 3)[0]
            tag = struct.unpack_from("<H", buf, pos + 7)[0]
            sub_end = pos + 7 + sz
            if tag == 0x2629 and version < 0x1000 and sz >= 2 and sub_end <= end_262a:
                if ci in clip_targets:
                    _write_single_clip_gain(
                        buf, pos, sub_end, ci, clip_targets[ci], records,
                        curve_refcounts, clip_lengths.get(ci, 0), summary,
                    )
                    patched += 1
                ci += 1
                pos = sub_end
                continue
        pos += 1
    return patched


def _is_exclusive_static_curve(records: list[bytearray], curve_refcounts: dict[int, int], idx: int) -> bool:
    if not (0 <= idx < len(records)) or curve_refcounts.get(idx, 0) != 1:
        return False
    rec = records[idx]
    return len(rec) == 30 and struct.unpack_from("<I", rec, 10)[0] == 1


def _write_single_clip_gain(
    buf: bytearray,
    pos: int,
    sub_end: int,
    ci: int,
    target: tuple[list[str], float],
    records: list[bytearray],
    curve_refcounts: dict[int, int],
    length_samples: int,
    summary: dict[str, dict[str, Any]],
) -> None:
    track_names, target_db = target
    sub = bytes(buf[pos:sub_end])
    s28 = find_subchunks(sub, 0x2628)
    if not s28:
        raise RuntimeError(f"Clip {ci} ({track_names[0]}) missing 0x2628 header")
    b28 = s28[0]
    b28_ver = struct.unpack_from("<H", b28, 1)[0] if len(b28) >= 3 else 0
    _cname, after_name = read_lp_str(b28, 9, max_len=256)
    if b28_ver < 5 or len(b28) < 31 or after_name > len(b28) - 22:
        raise RuntimeError(f"Clip {ci} ({track_names[0]}) uses unsupported 0x2628 version {b28_ver}")
    b28_rel = sub.find(b28, 9)
    idx_off = pos + b28_rel + len(b28) - 22
    cur_idx = struct.unpack_from("<i", buf, idx_off)[0]
    if _is_exclusive_static_curve(records, curve_refcounts, cur_idx):
        struct.pack_into("<f", records[cur_idx], 26, float(target_db))
    elif abs(target_db) >= 1e-4 or cur_idx != -1:
        if cur_idx >= 0 and cur_idx in curve_refcounts:
            curve_refcounts[cur_idx] = max(0, curve_refcounts[cur_idx] - 1)
        new_idx = len(records)
        records.append(_build_static_clip_gain_record(target_db, length_samples))
        curve_refcounts[new_idx] = 1
        struct.pack_into("<i", buf, idx_off, new_idx)
    new_lin = 0.0 if target_db <= -144.0 else float(10.0 ** (target_db / 20.0))
    for nm in track_names:
        summary[nm]["gain_linear"] = round(new_lin, 6)


def patch_clear_track_inputs(
    ptx_raw: bytes,
    track_names: list[str] | set[str],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Clear the assigned input path (to ``No Input``) on the named Audio tracks in ``0x2624``."""
    if not track_names:
        return ptx_raw, {}
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    buf = bytearray(dec)
    session = PtxSession(dec, xor_delta=delta)
    targets_by_uuid: dict[bytes, str] = {}
    for name in track_names:
        trk = session.track_by_name(name)
        if trk is None or trk.track_type != "Audio" or not trk.uuid_hex:
            continue
        targets_by_uuid[bytes.fromhex(trk.uuid_hex)] = trk.name
    if not targets_by_uuid:
        return ptx_raw, {}

    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(buf):
        raise RuntimeError("Missing or truncated 0x2624 mixer chunk in .ptx")
    edits, changes = _locate_input_clear_edits(buf, c2624.offset, min(len(buf), c2624.offset + 7 + c2624.size), targets_by_uuid)
    for blk_off, p_in, nlen, p261b_off in reversed(edits):
        edit_end = blk_off + p_in + 4 + nlen + 16
        buf[blk_off + p_in : edit_end] = b"\x00" * 20
        if nlen != 0:
            for anc_off in (p261b_off, blk_off, c2624.offset):
                if anc_off is None:
                    continue
                old_sz = struct.unpack_from("<I", buf, anc_off + 3)[0]
                if old_sz < nlen + 2:
                    raise RuntimeError(f"Enclosing Z-chunk at 0x{anc_off:x} too small to shrink by {nlen}")
                struct.pack_into("<I", buf, anc_off + 3, old_sz - nlen)
            shift_index_0002(buf, edit_end, -nlen)
    return encrypt_ptx(bytes(buf), delta, k0=k0), changes


def _find_enclosing_261b_in_block(blk: bytes, blk_off: int, p_in: int, nlen: int) -> int | None:
    """Return absolute offset of an enclosing ``0x261B`` container inside ``0x261C``, if present."""
    slot_end = p_in + 4 + nlen + 16
    for sub in find_subchunks(blk, 0x261B):
        rel = blk.find(sub, 9)
        if rel != -1 and rel < p_in and rel + len(sub) >= slot_end:
            return blk_off + rel
    return None


def _locate_input_clear_edits(
    buf: bytearray,
    c2624_off: int,
    chunk_end: int,
    targets_by_uuid: dict[bytes, str],
) -> tuple[list[tuple[int, int, int, int | None]], dict[str, dict[str, Any]]]:
    tcount = struct.unpack_from("<I", buf, c2624_off + 9)[0]
    pos = c2624_off + 13
    edits: list[tuple[int, int, int, int | None]] = []
    changes: dict[str, dict[str, Any]] = {}
    for _ in range(tcount):
        if pos + 9 > chunk_end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        blk_end = pos + 7 + sz
        if sz < 2 or blk_end > chunk_end:
            break
        blk = bytes(buf[pos:blk_end])
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        if tag == 0x261C and uoff != -1 and blk[uoff : uoff + 16] in targets_by_uuid:
            tname = targets_by_uuid[blk[uoff : uoff + 16]]
            edit = _find_261c_input_slot(blk)
            if edit is None:
                changes[tname] = {"previous_input": None, "cleared": False}
            else:
                p_in, nlen, prev_name = edit
                p261b_off = _find_enclosing_261b_in_block(blk, pos, p_in, nlen)
                edits.append((pos, p_in, nlen, p261b_off))
                changes[tname] = {"previous_input": prev_name, "cleared": True}
        pos = blk_end
    return edits, changes


def _find_261c_input_slot(blk: bytes) -> tuple[int, int, str] | None:
    """Return ``(rel_offset, name_len, prev_name)`` for an assigned input inside a ``0x261c`` block."""
    s485a = find_subchunks(blk, 0x485A)
    for cand in reversed(s485a):
        h = blk.rfind(cand)
        p_hdr = h + len(cand)
        if p_hdr + 57 > len(blk):
            continue
        if blk[p_hdr : p_hdr + 4] != b"\x01\x01\x0a\x00" or blk[p_hdr + 44 : p_hdr + 53] != b"\x00\x00\x00\x00\x08\x00\x00\x00\x00":
            continue
        p_in = p_hdr + 53
        nlen = struct.unpack_from("<I", blk, p_in)[0]
        if nlen > 256 or p_in + 4 + nlen + 27 > len(blk):
            continue
        if blk[p_in + 4 + nlen + 23 : p_in + 4 + nlen + 27] != b"\xff\xff\xff\xff":
            continue
        u16 = blk[p_in + 4 + nlen : p_in + 4 + nlen + 16]
        if nlen == 0 and u16 == b"\x00" * 16:
            return None
        prev_name = blk[p_in + 4 : p_in + 4 + nlen].decode("utf-8", errors="replace")
        return p_in, nlen, prev_name
    return None


def _set_vca_track_group_names(buf: bytearray, vca_updates_by_uuid: dict[bytes, tuple[str, int]]) -> None:
    if not vca_updates_by_uuid:
        return
    c2624 = next((c for c in reversed(iter_top_chunks(bytes(buf), strict=True)) if c.tag == 0x2624), None)
    if c2624 is None or c2624.offset + 13 > len(buf):
        raise RuntimeError("Missing or truncated 0x2624 mixer chunk in .ptx")
    chunk_end = min(len(buf), c2624.offset + 7 + c2624.size)
    matches = _locate_vca_blocks(buf, c2624.offset, chunk_end, vca_updates_by_uuid)
    if len(matches) != len(vca_updates_by_uuid):
        raise RuntimeError(f"Located {len(matches)}/{len(vca_updates_by_uuid)} target VCA 0x261f blocks")
    for blk_off, blk_end, new_gname, new_gid in reversed(matches):
        blk = bytes(buf[blk_off:blk_end])
        s2610 = find_subchunks(blk, 0x2610)
        if not s2610:
            raise RuntimeError("Missing 0x2610 inside VCA 0x261f block")
        p10_rel = blk.find(s2610[0])
        uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
        if uoff != -1 and p10_rel >= uoff + 21 and new_gname:
            buf[blk_off + p10_rel - 5] = 1
        if p10_rel >= 4:
            struct.pack_into("<I", buf, blk_off + p10_rel - 4, max(0, int(new_gid)))
        tail_rel = p10_rel + len(s2610[0])
        gbytes = new_gname.encode("utf-8")
        new_tail = struct.pack("<I", len(gbytes)) + gbytes + b"\x00"
        delta = len(new_tail) - (len(blk) - tail_rel)
        buf[blk_off + tail_rel : blk_end] = new_tail
        struct.pack_into("<I", buf, blk_off + 3, (tail_rel + len(new_tail)) - 7)
        old_2624_sz = struct.unpack_from("<I", buf, c2624.offset + 3)[0]
        struct.pack_into("<I", buf, c2624.offset + 3, old_2624_sz + delta)
        shift_index_0002(buf, blk_end, delta)


def _locate_vca_blocks(
    buf: bytearray, c2624_offset: int, chunk_end: int, vca_updates_by_uuid: dict[bytes, tuple[str, int]]
) -> list[tuple[int, int, str, int]]:
    tcount = struct.unpack_from("<I", buf, c2624_offset + 9)[0]
    pos = c2624_offset + 13
    matches: list[tuple[int, int, str, int]] = []
    for _ in range(tcount):
        if pos + 9 > chunk_end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        blk_end = pos + 7 + sz
        if sz < 2 or blk_end > chunk_end:
            break
        if struct.unpack_from("<H", buf, pos + 7)[0] == 0x261F:
            blk = bytes(buf[pos:blk_end])
            uoff = blk.find(UUID_PREFIX, 0, min(256, len(blk)))
            if uoff != -1 and blk[uoff : uoff + 16] in vca_updates_by_uuid:
                gname, gid = vca_updates_by_uuid[blk[uoff : uoff + 16]]
                matches.append((pos, blk_end, gname, gid))
        pos = blk_end
    return matches


def _serialize_group_attrs_258e(attrs: PtxGroupAttributes) -> bytes:
    b = bytearray(62)
    for k, v in attrs.extra_mix_flags.items():
        if 0 <= k < 62:
            b[k] = v & 0xFF
    b[0], b[1], b[2], b[3] = int(attrs.main_volume), int(attrs.main_mute), int(attrs.main_pan), int(attrs.main_lfe)
    b[4] = int(attrs.follow_globals)
    b[5], b[6], b[7], b[8] = int(attrs.record_enable), int(attrs.input_monitoring), int(attrs.solos), int(attrs.automation_mode)
    struct.pack_into("<H", b, 9, encode_slot_mask(attrs.insert_controls))
    struct.pack_into("<H", b, 17, encode_slot_mask(attrs.insert_bypass))
    struct.pack_into("<H", b, 25, encode_slot_mask(attrs.send_level))
    struct.pack_into("<H", b, 33, encode_slot_mask(attrs.send_mute))
    struct.pack_into("<H", b, 41, encode_slot_mask(attrs.send_pan))
    struct.pack_into("<H", b, 49, encode_slot_mask(attrs.send_lfe))
    b[61] = int(attrs.include_rulers)
    return struct.pack("<BHIH", 0x5A, 5, len(b) + 2, 0x258E) + bytes(b)


def _serialize_group_202a(
    name: str,
    group_type_code: int,
    id_number: int,
    is_edit_active: bool,
    is_mix_active: bool,
    member_indices: list[int],
    color_index: int = 0xFFFE,
    flags: int = 0x0001,
    vca_track_index: int | None = None,
    attributes: PtxGroupAttributes | None = None,
) -> bytes:
    name_b = name.encode("utf-8")
    attrs = attributes if attributes is not None else PtxGroupAttributes(follow_globals=True)
    vca_u16 = vca_track_index if vca_track_index is not None else 0xFFFF
    body = b"".join([
        struct.pack("<I", len(name_b)),
        name_b,
        struct.pack("<BiBBI", group_type_code, id_number, int(bool(is_edit_active)), int(bool(is_mix_active)), len(member_indices)),
        struct.pack(f"<{len(member_indices)}H", *member_indices) if member_indices else b"",
        struct.pack("<HH6sH", color_index & 0xFFFF, flags & 0xFFFF, b"\x00" * 6, vca_u16 & 0xFFFF),
        _serialize_group_attrs_258e(attrs),
    ])
    return struct.pack("<BHIH", 0x5A, 5, len(body) + 2, 0x202A) + body


def _build_202b(edit_list: list[bytes], mix_list: list[bytes], trailer: bytes = b"\x00\x00", version: int = 4) -> bytes:
    body = struct.pack("<I", len(edit_list)) + b"".join(edit_list) + struct.pack("<I", len(mix_list)) + b"".join(mix_list) + trailer
    return struct.pack("<BHIH", 0x5A, version, len(body) + 2, 0x202B) + body


def _resolve_track_index(session: PtxSession, track_ref: Any) -> int:
    if isinstance(track_ref, int):
        if not (0 <= track_ref < len(session.tracks)):
            raise IndexError(f"Track index {track_ref} out of range")
        return track_ref
    trk = session.track_by_name(str(track_ref))
    if trk is None:
        raise KeyError(f"Track {track_ref!r} not found in .ptx session")
    return trk.index


def _resolve_group_id_number(session: PtxSession, id_letter: str | None, exclude_name: str | None = None) -> tuple[int, str]:
    used = {
        g.id_number for g in session.groups
        if g.name != "<ALL>" and (exclude_name is None or g.name.lower() != exclude_name.lower())
    }
    if id_letter is not None:
        s = id_letter.strip().lower()
        if len(s) == 1 and "a" <= s <= "z":
            num = ord(s) - ord("a") + 1
        elif len(s) == 2 and s[0] in "234" and "a" <= s[1] <= "z":
            num = (int(s[0]) - 1) * 26 + (ord(s[1]) - ord("a") + 1)
        else:
            raise ValueError(f"Invalid group id_letter {id_letter!r}")
        if num in used:
            raise ValueError(f"Group ID {s!r} ({num}) is already in use")
        return num, s
    for n in range(1, MAX_GROUP_ID + 1):
        if n not in used:
            return n, group_id_to_label(n)
    raise ValueError(f"All {MAX_GROUP_ID} Pro Tools group IDs are in use")


def _resolve_vca_assignment(
    session: PtxSession, vca_track: Any, member_indices: list[int], group_name: str, id_num: int, gtype_code: int,
    *, allow_existing_group: str | None = None,
) -> tuple[int | None, str | None, dict[bytes, tuple[str, int]]]:
    if vca_track is None or vca_track == "":
        return None, None, {}
    if gtype_code == 0:
        raise ValueError("Cannot assign a VCA master to an Edit-only group")
    vca_idx = _resolve_track_index(session, vca_track)
    vtrk = session.tracks[vca_idx]
    if vtrk.track_type != "Vca":
        raise ValueError(f"Track {vtrk.name!r} is type {vtrk.track_type!r}, not 'Vca'")
    if vtrk.vca_group_name and (allow_existing_group is None or vtrk.vca_group_name.lower() != allow_existing_group.lower()):
        raise ValueError(f"VCA track {vtrk.name!r} is already assigned to group {vtrk.vca_group_name!r}")
    if vca_idx in member_indices:
        raise ValueError(f"VCA track {vtrk.name!r} cannot be a member of the group it controls")
    return vca_idx, vtrk.name, {bytes.fromhex(vtrk.uuid_hex): (group_name, id_num)}


def patch_add_group(
    ptx_raw: bytes,
    name: str,
    member_tracks: list[Any],
    *,
    vca_track: Any = None,
    group_type: str = "Mix/Edit",
    id_letter: str | None = None,
    is_edit_active: bool = True,
    is_mix_active: bool = True,
    follow_globals: bool = True,
    include_rulers: bool = False,
    attributes: PtxGroupAttributes | None = None,
    color_index: int = 0xFFFE,
) -> tuple[bytes, dict[str, Any]]:
    """Add a new Edit/Mix/Mix-Edit group (with optional VCA link) to a ``.ptx`` buffer."""
    gname = name.strip()
    if not gname or gname == "<ALL>":
        raise ValueError(f"Invalid group name {name!r}")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    if any(g.name.lower() == gname.lower() for g in session.groups):
        raise ValueError(f"Group {gname!r} already exists in session")
    gtype_code = _GROUP_TYPE_TO_CODE.get(group_type.strip().lower())
    if gtype_code is None:
        raise ValueError(f"Invalid group_type {group_type!r}")
    id_num, resolved_letter = _resolve_group_id_number(session, id_letter)
    member_indices = sorted({_resolve_track_index(session, t) for t in member_tracks})
    vca_idx, vca_name, vca_updates = _resolve_vca_assignment(session, vca_track, member_indices, gname, id_num, gtype_code)
    attrs = PtxGroupAttributes(**asdict(attributes)) if attributes is not None else PtxGroupAttributes()
    attrs.follow_globals, attrs.include_rulers = follow_globals, include_rulers
    new_202a = _serialize_group_202a(
        gname, gtype_code, id_num, is_edit_active, is_mix_active, member_indices, color_index, 0x0001, vca_idx, attrs,
    )
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise RuntimeError("Missing 0x202b groups chunk in .ptx")
    edit_list, mix_list, trailer = split_202b(c202b.raw)
    if gtype_code in (0, 2):
        edit_list.append(new_202a)
    if gtype_code in (1, 2):
        mix_list.append(new_202a)
    buf = bytearray(dec)
    splice_top_chunk(buf, 0x202B, _build_202b(edit_list, mix_list, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(buf, vca_updates)
    return encrypt_ptx(bytes(buf), delta, k0=k0), {
        "action": "add_group",
        "name": gname,
        "id_number": id_num,
        "id_letter": resolved_letter,
        "group_type": GROUP_TYPE_MAP[gtype_code],
        "member_tracks": [session.tracks[i].name for i in member_indices],
        "vca_track": vca_name,
    }


def patch_delete_group(ptx_raw: bytes, name: str) -> tuple[bytes, dict[str, Any]]:
    """Delete an existing group from a ``.ptx`` buffer and clear any linked VCA assignment."""
    gname = name.strip()
    if not gname or gname == "<ALL>":
        raise ValueError(f"Cannot delete group {name!r}")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    existing = session.group_by_name(gname)
    if existing is None:
        raise KeyError(f"Group {gname!r} not found in .ptx session")
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise RuntimeError("Missing 0x202b groups chunk in .ptx")
    edit_list, mix_list, trailer = split_202b(c202b.raw)
    low = existing.name.lower()
    new_edit = [g for g in edit_list if read_lp_str(g, 9, max_len=128)[0].lower() != low]
    new_mix = [g for g in mix_list if read_lp_str(g, 9, max_len=128)[0].lower() != low]
    vca_updates: dict[bytes, tuple[str, int]] = {}
    cleared_vca: str | None = None
    if existing.vca_track_index is not None and existing.vca_track_index < len(session.tracks):
        vtrk = session.tracks[existing.vca_track_index]
        vca_updates[bytes.fromhex(vtrk.uuid_hex)] = ("", 0)
        cleared_vca = vtrk.name
    buf = bytearray(dec)
    splice_top_chunk(buf, 0x202B, _build_202b(new_edit, new_mix, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(buf, vca_updates)
    return encrypt_ptx(bytes(buf), delta, k0=k0), {
        "action": "delete_group",
        "name": existing.name,
        "cleared_vca_track": cleared_vca,
    }


def _replace_in_202a_list(lst: list[bytes], old_low: str, new_202a: bytes, should_exist: bool) -> list[bytes]:
    out: list[bytes] = []
    replaced = False
    for item in lst:
        if read_lp_str(item, 9, max_len=128)[0].lower() == old_low:
            if should_exist and not replaced:
                out.append(new_202a)
                replaced = True
        else:
            out.append(item)
    if should_exist and not replaced:
        out.append(new_202a)
    return out


def patch_modify_group(
    ptx_raw: bytes,
    name: str,
    *,
    new_name: str | None = None,
    member_tracks: list[Any] | None = None,
    add_tracks: list[Any] | None = None,
    remove_tracks: list[Any] | None = None,
    vca_track: Any = _UNSET,
    group_type: str | None = None,
    is_edit_active: bool | None = None,
    is_mix_active: bool | None = None,
    follow_globals: bool | None = None,
    include_rulers: bool | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Modify an existing group's name, members, or VCA wiring inside a ``.ptx`` buffer."""
    if name.strip() == "<ALL>":
        raise ValueError("Cannot modify built-in <ALL> group")
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    session = PtxSession(dec, xor_delta=delta)
    existing = session.group_by_name(name)
    if existing is None:
        raise KeyError(f"Group {name!r} not found in .ptx session")
    target_name = existing.name if new_name is None else new_name.strip()
    if not target_name or target_name == "<ALL>":
        raise ValueError(f"Invalid new_name {new_name!r}")
    if target_name.lower() != existing.name.lower() and any(g.name.lower() == target_name.lower() for g in session.groups):
        raise ValueError(f"Group {target_name!r} already exists in session")
    gtype_code = _GROUP_TYPE_TO_CODE.get((group_type or existing.group_type).strip().lower())
    if gtype_code is None:
        raise ValueError(f"Invalid group_type {group_type!r}")
    members = set(existing.member_indices) if member_tracks is None else {_resolve_track_index(session, t) for t in member_tracks}
    if add_tracks:
        members.update(_resolve_track_index(session, t) for t in add_tracks)
    if remove_tracks:
        members.difference_update(_resolve_track_index(session, t) for t in remove_tracks)
    member_indices = sorted(members)
    vca_idx, vca_name, vca_updates = _compute_modified_vca(session, existing, vca_track, member_indices, target_name, gtype_code)
    attrs = PtxGroupAttributes(**asdict(existing.attributes)) if existing.attributes else PtxGroupAttributes(follow_globals=True)
    if follow_globals is not None:
        attrs.follow_globals = follow_globals
    if include_rulers is not None:
        attrs.include_rulers = include_rulers
    edit_ex = session.group_by_name(name, kind="edit") or existing
    mix_ex = session.group_by_name(name, kind="mix") or existing
    e_edit_act = edit_ex.is_edit_active if is_edit_active is None else bool(is_edit_active)
    e_mix_act = edit_ex.is_mix_active if is_mix_active is None else bool(is_mix_active)
    m_edit_act = mix_ex.is_edit_active if is_edit_active is None else bool(is_edit_active)
    m_mix_act = mix_ex.is_mix_active if is_mix_active is None else bool(is_mix_active)
    new_edit_202a = _serialize_group_202a(
        target_name, gtype_code, existing.id_number, e_edit_act, e_mix_act,
        member_indices, existing.color_index, 0x0001, vca_idx, attrs,
    )
    new_mix_202a = new_edit_202a if (m_edit_act, m_mix_act) == (e_edit_act, e_mix_act) else _serialize_group_202a(
        target_name, gtype_code, existing.id_number, m_edit_act, m_mix_act,
        member_indices, existing.color_index, 0x0001, vca_idx, attrs,
    )
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise RuntimeError("Missing 0x202b groups chunk in .ptx")
    edit_list, mix_list, trailer = split_202b(c202b.raw)
    old_low = existing.name.lower()
    new_edit = _replace_in_202a_list(edit_list, old_low, new_edit_202a, gtype_code in (0, 2))
    new_mix = _replace_in_202a_list(mix_list, old_low, new_mix_202a, gtype_code in (1, 2))
    buf = bytearray(dec)
    splice_top_chunk(buf, 0x202B, _build_202b(new_edit, new_mix, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(buf, vca_updates)
    return encrypt_ptx(bytes(buf), delta, k0=k0), {
        "action": "modify_group",
        "name": target_name,
        "member_tracks": [session.tracks[i].name for i in member_indices],
        "vca_track": vca_name,
    }


def _compute_modified_vca(
    session: PtxSession, existing: PtxGroup, vca_track: Any, member_indices: list[int], target_name: str, gtype_code: int
) -> tuple[int | None, str | None, dict[bytes, tuple[str, int]]]:
    vca_updates: dict[bytes, tuple[str, int]] = {}
    if vca_track is _UNSET:
        vca_idx = existing.vca_track_index
        vca_name = existing.vca_track_name
        if vca_idx is not None and vca_idx < len(session.tracks):
            if gtype_code == 0:
                raise ValueError("Cannot convert a VCA-linked group to Edit-only without unlinking its VCA")
            vtrk = session.tracks[vca_idx]
            if vca_idx in member_indices:
                raise ValueError(f"VCA track {vtrk.name!r} cannot be a member of the group it controls")
            if target_name != existing.name:
                vca_updates[bytes.fromhex(vtrk.uuid_hex)] = (target_name, existing.id_number)
        return vca_idx, vca_name, vca_updates

    if existing.vca_track_index is not None and existing.vca_track_index < len(session.tracks):
        old_vtrk = session.tracks[existing.vca_track_index]
        vca_updates[bytes.fromhex(old_vtrk.uuid_hex)] = ("", 0)
    new_idx, new_name, new_upd = _resolve_vca_assignment(
        session, vca_track, member_indices, target_name, existing.id_number, gtype_code,
        allow_existing_group=existing.name,
    )
    vca_updates.update(new_upd)
    return new_idx, new_name, vca_updates


def _validate_patched_ptx(original_bytes: bytes, patched_bytes: bytes) -> None:
    """Strictly verify structural integrity of *patched_bytes* against *original_bytes* before writing."""
    orig_dec, orig_delta = decrypt_ptx(original_bytes)
    new_dec, new_delta = decrypt_ptx(patched_bytes)
    orig_chunks = iter_top_chunks(orig_dec, strict=True)
    new_chunks = iter_top_chunks(new_dec, strict=True)
    orig_tags = {c.tag for c in orig_chunks}
    new_tags = {c.tag for c in new_chunks}
    missing_tags = orig_tags - new_tags
    extra_tags = new_tags - orig_tags - {0x2630}
    if missing_tags or extra_tags:
        raise RuntimeError(f"Patched .ptx top-level chunk tags changed: missing={missing_tags}, extra={extra_tags}")
    orig_has_idx = any(c.tag == 0x0002 for c in orig_chunks)
    if orig_has_idx and not validate_index_0002(new_dec):
        raise RuntimeError("Patched .ptx lost valid 0x0002 chunk offset index")
    orig_sess = PtxSession(orig_dec, xor_delta=orig_delta)
    new_sess = PtxSession(new_dec, xor_delta=new_delta)
    if len(orig_sess.tracks) != len(new_sess.tracks):
        raise RuntimeError(f"Patched .ptx track count changed ({len(orig_sess.tracks)} -> {len(new_sess.tracks)})")


def patch_slot_a_trim_gains(
    raw_ptx: bytes,
    track_gains_db: dict[str, float],
    *,
    activate_if_inactive: bool = True,
    unbypass_if_bypassed: bool = False,
    allow_auto_insert: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Patch (or auto-insert into empty Slot A) validated Slot A Trim plugin gains across tracks."""
    if not track_gains_db:
        return raw_ptx, {}
    session = PtxSession.from_bytes(raw_ptx)
    return patch_slot_a_trim_gains_on_session(
        raw_ptx, session, track_gains_db,
        activate_if_inactive=activate_if_inactive,
        unbypass_if_bypassed=unbypass_if_bypassed,
        allow_auto_insert=allow_auto_insert,
    )


def apply_atomic_ptx_patch(
    session_path: str | Path,
    transform_fn: Callable[[bytes], tuple[bytes, Any]],
) -> tuple[bytes, Any]:
    """Validate and atomically write patched ``.ptx`` bytes, returning ``(original_bytes, metadata)``."""
    target = Path(session_path)
    original_bytes = target.read_bytes()
    patched_bytes, metadata = transform_fn(original_bytes)
    _validate_patched_ptx(original_bytes, patched_bytes)
    atomic_write_bytes(target, patched_bytes)
    return original_bytes, metadata
