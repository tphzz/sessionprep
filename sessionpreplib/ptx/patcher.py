"""Validated in-memory and atomic-file ``.ptx`` patchers (volumes, clip gain, groups & VCAs)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import math
from pathlib import Path
import struct
from typing import Any

from ._document import (
    PtxBuffer,
    assemble_top_chunks,
    iter_mixer_strips,
    patch_static_0x260a_curve,
    validate_decrypted_ptx_pair,
)
from .codec import (
    atomic_write_bytes,
    decrypt_ptx,
    encode_slot_mask,
    find_subchunk_spans,
    find_zchunk_offset,
    group_id_to_label,
    iter_top_chunks,
    read_lp_str,
    resize_zchunk_ancestors,
    shift_index_0002,
    splice_top_chunk,
)
from .models import (
    GROUP_TYPE_MAP,
    MAX_GROUP_ID,
    PtxCorruptFileError,
    PtxGroup,
    PtxGroupAttributes,
    PtxNotFoundError,
    PtxValidationError,
    TrackRef,
)
from .parser import PtxSession, split_202b
from .plugin_trim import patch_slot_a_trim_gains_on_buffer
from .writer_index import apply_inserted_top_chunk_index_0002
from .writer_primitives import build_zchunk, parse_index_0002

_UNSET: Any = object()
_GROUP_TYPE_TO_CODE = {v.lower(): k for k, v in GROUP_TYPE_MAP.items()}


def _require_finite_db(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise PtxValidationError(f"{label} must be numeric, got {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise PtxValidationError(f"{label} must be finite, got {value!r}")
    return val


def _patch_mixer_block_volume(buf: bytearray, blk_off: int, blk: bytes, val_tenths: int, tname: str) -> float:
    d_spans = find_subchunk_spans(blk, 0x260D)
    if d_spans:
        d0_rel, d0 = d_spans[0]
        d0_off = blk_off + d0_rel
        prev_db: float | None = None
        wrote = False
        cpos = find_zchunk_offset(d0, 0x260C, 9)
        head_end = cpos if cpos != -1 else len(d0)
        a_spans = find_subchunk_spans(d0, 0x260A, 9, head_end)
        if a_spans:
            a0_rel, a0 = a_spans[0]
            prev_raw = patch_static_0x260a_curve(
                buf, d0_off + a0_rel, a0, val_tenths, param_label=f"Track {tname!r} volume",
            )
            prev_db = round(prev_raw / 10.0, 1)
            wrote = True
        p1029_spans = find_subchunk_spans(d0, 0x1029)
        if p1029_spans and len(p1029_spans[0][1]) >= 100:
            p1029_off = d0_off + p1029_spans[0][0]
            prev_db = round(struct.unpack_from("<i", buf, p1029_off + 10)[0] / 10.0, 1)
            struct.pack_into("<i", buf, p1029_off + 10, val_tenths)
            struct.pack_into("<i", buf, p1029_off + 96, val_tenths)
            wrote = True
        if not wrote:
            raise PtxCorruptFileError(f"Track {tname!r} 0x260d block has no writable volume fields")
        return prev_db if prev_db is not None else 0.0

    s2610_spans = find_subchunk_spans(blk, 0x2610)
    if s2610_spans:
        s10_rel, s10 = s2610_spans[0]
        a_spans = find_subchunk_spans(s10, 0x260A)
        if a_spans:
            a0_rel, a0 = a_spans[0]
            prev_raw = patch_static_0x260a_curve(
                buf, blk_off + s10_rel + a0_rel, a0, val_tenths, param_label=f"Track {tname!r} volume",
            )
            return round(prev_raw / 10.0, 1)
    raise PtxCorruptFileError(f"Track {tname!r} has neither 0x260d nor 0x2610 volume section")


def patch_track_volumes_on_buffer(
    ptx_buf: PtxBuffer,
    track_volumes_db: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Patch static main-fader volumes in-place on a decrypted ``PtxBuffer``."""
    if not track_volumes_db:
        raise PtxValidationError("track_volumes_db must be a non-empty {track_name: db} dict")
    session = ptx_buf.session
    targets_by_uuid: dict[bytes, tuple[str, float]] = {}
    for tname, db in track_volumes_db.items():
        val_db = _require_finite_db(db, f"Track volume for {tname!r}")
        trk = session.track_by_name(tname)
        if trk is None:
            raise PtxNotFoundError(f"Track {tname!r} not found in .ptx session")
        targets_by_uuid[bytes.fromhex(trk.uuid_hex)] = (trk.name, max(-144.0, min(12.0, val_db)))

    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(ptx_buf.data):
        raise PtxCorruptFileError("Missing or truncated 0x2624 mixer chunk in .ptx")
    changes: dict[str, dict[str, float]] = {}
    for span in iter_mixer_strips(ptx_buf.data, c2624.offset):
        if span.uuid16 is not None and span.uuid16 in targets_by_uuid:
            tname, target_db = targets_by_uuid[span.uuid16]
            val_tenths = int(round(target_db * 10.0))
            blk = bytes(ptx_buf.data[span.offset : span.end])
            prev_db = _patch_mixer_block_volume(ptx_buf.data, span.offset, blk, val_tenths, tname)
            changes[tname] = {"previous_db": prev_db, "target_db": round(val_tenths / 10.0, 1)}

    missing = [nm for nm, _ in targets_by_uuid.values() if nm not in changes]
    if missing:
        raise PtxCorruptFileError(f"Could not locate mixer blocks for track(s): {missing}")
    ptx_buf.invalidate()
    return changes


def patch_track_volumes(
    ptx_raw: bytes,
    track_volumes_db: dict[str, float],
) -> tuple[bytes, dict[str, dict[str, float]]]:
    """Patch static main-fader volumes (``-144.0`` to ``+12.0 dB``, ``0.1 dB`` resolution)."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    changes = patch_track_volumes_on_buffer(ptx_buf, track_volumes_db)
    return ptx_buf.to_encrypted(), changes


def _parse_2637_records(raw: bytes) -> tuple[list[bytearray], bytes]:
    if len(raw) < 13:
        raise PtxCorruptFileError("Truncated 0x2637 clip-gain curves chunk in .ptx")
    cnt = struct.unpack_from("<I", raw, 9)[0]
    pos = 13
    records: list[bytearray] = []
    for i in range(cnt):
        if pos + 8 > len(raw):
            raise PtxCorruptFileError(f"Truncated 0x2637 curve header at index {i}")
        _flags, magic, _ver, plen = struct.unpack_from("<BBHI", raw, pos)
        end = pos + 8 + plen
        if magic != 0x46 or end > len(raw):
            raise PtxCorruptFileError(f"Malformed 0x2637 curve record at index {i}")
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
    return build_zchunk(0x2637, body, version=version)


def patch_clip_gains_on_buffer(
    ptx_buf: PtxBuffer,
    track_clip_gains_db: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Patch static Clip Gain (``-144.0`` to ``+36.0 dB``) in-place on a decrypted ``PtxBuffer``."""
    if not track_clip_gains_db:
        raise PtxValidationError("track_clip_gains_db must be a non-empty {track_name: db} dict")
    session = ptx_buf.session
    clip_targets, summary = _resolve_clip_gain_targets(session, track_clip_gains_db)
    c262a = session.chunk_map.get(0x262A)
    if c262a is None:
        raise PtxCorruptFileError("Missing 0x262a clips chunk in .ptx")
    c2637 = session.chunk_map.get(0x2637)
    records, trailer = _parse_2637_records(c2637.raw) if c2637 is not None else ([], b"")
    curve_refcounts: dict[int, int] = {}
    clip_lengths: dict[int, int] = {}
    for clip in session.clips:
        clip_lengths[clip.index] = clip.length_samples
        if clip.clip_gain_curve_index >= 0:
            curve_refcounts[clip.clip_gain_curve_index] = curve_refcounts.get(clip.clip_gain_curve_index, 0) + 1
    chunk_end = min(len(ptx_buf.data), c262a.offset + 7 + c262a.size)
    patched_count = _apply_clip_gain_writes(
        ptx_buf.data, c262a.offset, chunk_end, clip_targets, records, curve_refcounts, clip_lengths, summary,
    )
    if patched_count != len(clip_targets):
        raise PtxCorruptFileError(f"Patched {patched_count}/{len(clip_targets)} target clips in 0x262a")
    new_2637 = _build_2637_chunk(records, version=(c2637.version if c2637 is not None else 1), trailer=trailer)
    if c2637 is not None:
        splice_top_chunk(ptx_buf.data, 0x2637, new_2637)
    else:
        old_dec = bytes(ptx_buf.data)
        template_idx = parse_index_0002(old_dec) if 0x0002 in session.chunk_map else None
        new_buf = assemble_top_chunks(old_dec, {}, insertions=((0x2637, new_2637, (0x204D, 0x2031)),))
        if template_idx is not None:
            apply_inserted_top_chunk_index_0002(new_buf, template_idx, 0x2637, new_2637)
        ptx_buf.replace_data(new_buf)
    ptx_buf.invalidate()
    return summary


def patch_clip_gains(
    ptx_raw: bytes,
    track_clip_gains_db: dict[str, float],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Patch static Clip Gain (``-144.0`` to ``+36.0 dB``) via ``0x2637`` curves & ``0x2628`` links."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    summary = patch_clip_gains_on_buffer(ptx_buf, track_clip_gains_db)
    return ptx_buf.to_encrypted(), summary


def _resolve_clip_gain_targets(
    session: PtxSession, track_clip_gains_db: Mapping[str, float]
) -> tuple[dict[int, tuple[list[str], float]], dict[str, dict[str, Any]]]:
    clip_targets: dict[int, tuple[list[str], float]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tname, db in track_clip_gains_db.items():
        val_db = _require_finite_db(db, f"Clip gain for {tname!r}")
        trk = session.track_by_name(tname)
        if trk is None:
            raise PtxNotFoundError(f"Track {tname!r} not found in .ptx session")
        clamped = max(-144.0, min(36.0, val_db))
        indices = sorted({pl.clip_index for pl in trk.clips})
        if not indices:
            raise PtxValidationError(f"Track {trk.name!r} has no placed clips to patch")
        for ci in indices:
            existing = clip_targets.get(ci)
            if existing is not None:
                prev_tracks, prev_db = existing
                if abs(prev_db - clamped) > 1e-6:
                    raise PtxValidationError(
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
    s28_spans = find_subchunk_spans(sub, 0x2628)
    if not s28_spans:
        raise PtxCorruptFileError(f"Clip {ci} ({track_names[0]}) missing 0x2628 header")
    b28_rel, b28 = s28_spans[0]
    b28_ver = struct.unpack_from("<H", b28, 1)[0] if len(b28) >= 3 else 0
    _cname, after_name = read_lp_str(b28, 9, max_len=256)
    if b28_ver < 5 or len(b28) < 31 or after_name > len(b28) - 22:
        raise PtxCorruptFileError(f"Clip {ci} ({track_names[0]}) uses unsupported 0x2628 version {b28_ver}")
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


def patch_clear_track_inputs_on_buffer(
    ptx_buf: PtxBuffer,
    track_names: Sequence[str] | set[str],
) -> dict[str, dict[str, Any]]:
    """Clear the assigned input path (to ``No Input``) in-place on a decrypted ``PtxBuffer``."""
    if not track_names:
        return {}
    session = ptx_buf.session
    targets_by_uuid: dict[bytes, str] = {}
    for name in track_names:
        trk = session.track_by_name(name)
        if trk is None or trk.track_type != "Audio" or not trk.uuid_hex:
            continue
        targets_by_uuid[bytes.fromhex(trk.uuid_hex)] = trk.name
    if not targets_by_uuid:
        return {}

    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None or c2624.offset + 13 > len(ptx_buf.data):
        raise PtxCorruptFileError("Missing or truncated 0x2624 mixer chunk in .ptx")
    edits, changes = _locate_input_clear_edits(ptx_buf.data, c2624.offset, targets_by_uuid)
    for blk_off, p_in, nlen, p261b_off in reversed(edits):
        edit_end = blk_off + p_in + 4 + nlen + 16
        ptx_buf.data[blk_off + p_in : edit_end] = b"\x00" * 20
        if nlen != 0:
            ancestors = [a for a in (p261b_off, blk_off, c2624.offset) if a is not None]
            resize_zchunk_ancestors(ptx_buf.data, ancestors, -nlen)
            shift_index_0002(ptx_buf.data, edit_end, -nlen)
    if edits:
        ptx_buf.invalidate()
    return changes


def patch_clear_track_inputs(
    ptx_raw: bytes,
    track_names: list[str] | set[str],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Clear the assigned input path (to ``No Input``) on the named Audio tracks in ``0x2624``."""
    if not track_names:
        return ptx_raw, {}
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    changes = patch_clear_track_inputs_on_buffer(ptx_buf, track_names)
    if not changes:
        return ptx_raw, {}
    return ptx_buf.to_encrypted(), changes


def _find_enclosing_261b_in_block(blk: bytes, blk_off: int, p_in: int, nlen: int) -> int | None:
    """Return absolute offset of an enclosing ``0x261B`` container inside ``0x261C``, if present."""
    slot_end = p_in + 4 + nlen + 16
    for rel, sub in find_subchunk_spans(blk, 0x261B):
        if rel < p_in and rel + len(sub) >= slot_end:
            return blk_off + rel
    return None


def _locate_input_clear_edits(
    buf: bytearray,
    c2624_off: int,
    targets_by_uuid: dict[bytes, str],
) -> tuple[list[tuple[int, int, int, int | None]], dict[str, dict[str, Any]]]:
    edits: list[tuple[int, int, int, int | None]] = []
    changes: dict[str, dict[str, Any]] = {}
    for span in iter_mixer_strips(buf, c2624_off):
        if span.tag == 0x261C and span.uuid16 is not None and span.uuid16 in targets_by_uuid:
            tname = targets_by_uuid[span.uuid16]
            blk = bytes(buf[span.offset : span.end])
            edit = _find_261c_input_slot(blk)
            if edit is None:
                changes[tname] = {"previous_input": None, "cleared": False}
            else:
                p_in, nlen, prev_name = edit
                p261b_off = _find_enclosing_261b_in_block(blk, span.offset, p_in, nlen)
                edits.append((span.offset, p_in, nlen, p261b_off))
                changes[tname] = {"previous_input": prev_name, "cleared": True}
    return edits, changes


def _find_261c_input_slot(blk: bytes) -> tuple[int, int, str] | None:
    """Return ``(rel_offset, name_len, prev_name)`` for an assigned input inside a ``0x261c`` block."""
    for h, cand in reversed(find_subchunk_spans(blk, 0x485A)):
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
        raise PtxCorruptFileError("Missing or truncated 0x2624 mixer chunk in .ptx")
    matches = [
        (span.offset, span.end, *vca_updates_by_uuid[span.uuid16])
        for span in iter_mixer_strips(buf, c2624.offset)
        if span.tag == 0x261F and span.uuid16 is not None and span.uuid16 in vca_updates_by_uuid
    ]
    if len(matches) != len(vca_updates_by_uuid):
        raise PtxCorruptFileError(f"Located {len(matches)}/{len(vca_updates_by_uuid)} target VCA 0x261f blocks")
    for blk_off, blk_end, new_gname, new_gid in reversed(matches):
        blk = bytes(buf[blk_off:blk_end])
        s2610_spans = find_subchunk_spans(blk, 0x2610)
        if not s2610_spans:
            raise PtxCorruptFileError("Missing 0x2610 inside VCA 0x261f block")
        p10_rel, s2610_raw = s2610_spans[0]
        if p10_rel >= 21 and new_gname:
            buf[blk_off + p10_rel - 5] = 1
        if p10_rel >= 4:
            struct.pack_into("<I", buf, blk_off + p10_rel - 4, max(0, int(new_gid)))
        tail_rel = p10_rel + len(s2610_raw)
        gbytes = new_gname.encode("utf-8")
        new_tail = struct.pack("<I", len(gbytes)) + gbytes + b"\x00"
        delta = len(new_tail) - (len(blk) - tail_rel)
        buf[blk_off + tail_rel : blk_end] = new_tail
        resize_zchunk_ancestors(buf, (blk_off, c2624.offset), delta)
        shift_index_0002(buf, blk_end, delta)


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
    return build_zchunk(0x258E, bytes(b), version=5)


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
    return build_zchunk(0x202A, body, version=5)


def _build_202b(edit_list: list[bytes], mix_list: list[bytes], trailer: bytes = b"\x00\x00", version: int = 4) -> bytes:
    body = struct.pack("<I", len(edit_list)) + b"".join(edit_list) + struct.pack("<I", len(mix_list)) + b"".join(mix_list) + trailer
    return build_zchunk(0x202B, body, version=version)


def _resolve_track_index(session: PtxSession, track_ref: TrackRef) -> int:
    if isinstance(track_ref, bool):
        raise PtxValidationError(f"Invalid track reference: {track_ref!r}")
    if isinstance(track_ref, int):
        if not (0 <= track_ref < len(session.tracks)):
            raise PtxNotFoundError(f"Track index {track_ref} out of range")
        return track_ref
    trk = session.track_by_name(str(track_ref))
    if trk is None:
        raise PtxNotFoundError(f"Track {track_ref!r} not found in .ptx session")
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
            raise PtxValidationError(f"Invalid group id_letter {id_letter!r}")
        if num in used:
            raise PtxValidationError(f"Group ID {s!r} ({num}) is already in use")
        return num, s
    for n in range(1, MAX_GROUP_ID + 1):
        if n not in used:
            return n, group_id_to_label(n)
    raise PtxValidationError(f"All {MAX_GROUP_ID} Pro Tools group IDs are in use")


def _resolve_vca_assignment(
    session: PtxSession, vca_track: TrackRef | None, member_indices: list[int], group_name: str, id_num: int, gtype_code: int,
    *, allow_existing_group: str | None = None,
) -> tuple[int | None, str | None, dict[bytes, tuple[str, int]]]:
    if vca_track is None or vca_track == "":
        return None, None, {}
    if gtype_code == 0:
        raise PtxValidationError("Cannot assign a VCA master to an Edit-only group")
    vca_idx = _resolve_track_index(session, vca_track)
    vtrk = session.tracks[vca_idx]
    if vtrk.track_type != "Vca":
        raise PtxValidationError(f"Track {vtrk.name!r} is type {vtrk.track_type!r}, not 'Vca'")
    if vtrk.vca_group_name and (allow_existing_group is None or vtrk.vca_group_name.lower() != allow_existing_group.lower()):
        raise PtxValidationError(f"VCA track {vtrk.name!r} is already assigned to group {vtrk.vca_group_name!r}")
    if vca_idx in member_indices:
        raise PtxValidationError(f"VCA track {vtrk.name!r} cannot be a member of the group it controls")
    return vca_idx, vtrk.name, {bytes.fromhex(vtrk.uuid_hex): (group_name, id_num)}


def _resolve_new_group_attributes(
    attributes: PtxGroupAttributes | None,
    follow_globals: bool | None,
    include_rulers: bool | None,
) -> PtxGroupAttributes:
    if attributes is None:
        return PtxGroupAttributes(
            follow_globals=True if follow_globals is None else follow_globals,
            include_rulers=False if include_rulers is None else include_rulers,
        )
    attrs = PtxGroupAttributes(**asdict(attributes))
    if follow_globals is not None:
        attrs.follow_globals = follow_globals
    if include_rulers is not None:
        attrs.include_rulers = include_rulers
    return attrs


def patch_add_group_on_buffer(
    ptx_buf: PtxBuffer,
    name: str,
    member_tracks: Sequence[TrackRef],
    *,
    vca_track: TrackRef | None = None,
    group_type: str = "Mix/Edit",
    id_letter: str | None = None,
    is_edit_active: bool = True,
    is_mix_active: bool = True,
    follow_globals: bool | None = None,
    include_rulers: bool | None = None,
    attributes: PtxGroupAttributes | None = None,
    color_index: int = 0xFFFE,
) -> dict[str, Any]:
    """Add a new Edit/Mix/Mix-Edit group (with optional VCA link) in-place on a ``PtxBuffer``."""
    gname = name.strip()
    if not gname or gname == "<ALL>":
        raise PtxValidationError(f"Invalid group name {name!r}")
    session = ptx_buf.session
    if any(g.name.lower() == gname.lower() for g in session.groups):
        raise PtxValidationError(f"Group {gname!r} already exists in session")
    gtype_code = _GROUP_TYPE_TO_CODE.get(group_type.strip().lower())
    if gtype_code is None:
        raise PtxValidationError(f"Invalid group_type {group_type!r}")
    id_num, resolved_letter = _resolve_group_id_number(session, id_letter)
    member_indices = sorted({_resolve_track_index(session, t) for t in member_tracks})
    vca_idx, vca_name, vca_updates = _resolve_vca_assignment(session, vca_track, member_indices, gname, id_num, gtype_code)
    attrs = _resolve_new_group_attributes(attributes, follow_globals, include_rulers)
    new_202a = _serialize_group_202a(
        gname, gtype_code, id_num, is_edit_active, is_mix_active, member_indices, color_index, 0x0001, vca_idx, attrs,
    )
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise PtxCorruptFileError("Missing 0x202b groups chunk in .ptx")
    edit_list, mix_list, trailer = split_202b(c202b.raw)
    if gtype_code in (0, 2):
        edit_list.append(new_202a)
    if gtype_code in (1, 2):
        mix_list.append(new_202a)
    splice_top_chunk(ptx_buf.data, 0x202B, _build_202b(edit_list, mix_list, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(ptx_buf.data, vca_updates)
    ptx_buf.invalidate()
    return {
        "action": "add_group", "name": gname, "id_number": id_num, "id_letter": resolved_letter,
        "group_type": GROUP_TYPE_MAP[gtype_code],
        "member_tracks": [session.tracks[i].name for i in member_indices], "vca_track": vca_name,
    }


def patch_add_group(
    ptx_raw: bytes,
    name: str,
    member_tracks: list[TrackRef],
    *,
    vca_track: TrackRef | None = None,
    group_type: str = "Mix/Edit",
    id_letter: str | None = None,
    is_edit_active: bool = True,
    is_mix_active: bool = True,
    follow_globals: bool | None = None,
    include_rulers: bool | None = None,
    attributes: PtxGroupAttributes | None = None,
    color_index: int = 0xFFFE,
) -> tuple[bytes, dict[str, Any]]:
    """Add a new Edit/Mix/Mix-Edit group (with optional VCA link) to a ``.ptx`` buffer."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_add_group_on_buffer(
        ptx_buf, name, member_tracks, vca_track=vca_track, group_type=group_type,
        id_letter=id_letter, is_edit_active=is_edit_active, is_mix_active=is_mix_active,
        follow_globals=follow_globals, include_rulers=include_rulers,
        attributes=attributes, color_index=color_index,
    )
    return ptx_buf.to_encrypted(), info


def patch_delete_group_on_buffer(ptx_buf: PtxBuffer, name: str) -> dict[str, Any]:
    """Delete an existing group in-place on a ``PtxBuffer`` and clear any linked VCA assignment."""
    gname = name.strip()
    if not gname or gname == "<ALL>":
        raise PtxValidationError(f"Cannot delete group {name!r}")
    session = ptx_buf.session
    existing = session.group_by_name(gname)
    if existing is None:
        raise PtxNotFoundError(f"Group {gname!r} not found in .ptx session")
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise PtxCorruptFileError("Missing 0x202b groups chunk in .ptx")
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
    splice_top_chunk(ptx_buf.data, 0x202B, _build_202b(new_edit, new_mix, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(ptx_buf.data, vca_updates)
    ptx_buf.invalidate()
    return {"action": "delete_group", "name": existing.name, "cleared_vca_track": cleared_vca}


def patch_delete_group(ptx_raw: bytes, name: str) -> tuple[bytes, dict[str, Any]]:
    """Delete an existing group from a ``.ptx`` buffer and clear any linked VCA assignment."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_delete_group_on_buffer(ptx_buf, name)
    return ptx_buf.to_encrypted(), info


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


def _serialize_modified_group_pair(
    session: PtxSession,
    existing: PtxGroup,
    name: str,
    target_name: str,
    gtype_code: int,
    member_indices: list[int],
    vca_idx: int | None,
    *,
    is_edit_active: bool | None,
    is_mix_active: bool | None,
    follow_globals: bool | None,
    include_rulers: bool | None,
) -> tuple[bytes, bytes]:
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
    new_edit = _serialize_group_202a(
        target_name, gtype_code, existing.id_number, e_edit_act, e_mix_act,
        member_indices, existing.color_index, 0x0001, vca_idx, attrs,
    )
    if (m_edit_act, m_mix_act) == (e_edit_act, e_mix_act):
        return new_edit, new_edit
    new_mix = _serialize_group_202a(
        target_name, gtype_code, existing.id_number, m_edit_act, m_mix_act,
        member_indices, existing.color_index, 0x0001, vca_idx, attrs,
    )
    return new_edit, new_mix


def patch_modify_group_on_buffer(
    ptx_buf: PtxBuffer,
    name: str,
    *,
    new_name: str | None = None,
    member_tracks: Sequence[TrackRef] | None = None,
    add_tracks: Sequence[TrackRef] | None = None,
    remove_tracks: Sequence[TrackRef] | None = None,
    vca_track: Any = _UNSET,
    group_type: str | None = None,
    is_edit_active: bool | None = None,
    is_mix_active: bool | None = None,
    follow_globals: bool | None = None,
    include_rulers: bool | None = None,
) -> dict[str, Any]:
    """Modify an existing group's name, members, or VCA wiring in-place on a ``PtxBuffer``."""
    if name.strip() == "<ALL>":
        raise PtxValidationError("Cannot modify built-in <ALL> group")
    session = ptx_buf.session
    existing = session.group_by_name(name)
    if existing is None:
        raise PtxNotFoundError(f"Group {name!r} not found in .ptx session")
    target_name = existing.name if new_name is None else new_name.strip()
    if not target_name or target_name == "<ALL>":
        raise PtxValidationError(f"Invalid new_name {new_name!r}")
    if target_name.lower() != existing.name.lower() and any(g.name.lower() == target_name.lower() for g in session.groups):
        raise PtxValidationError(f"Group {target_name!r} already exists in session")
    gtype_code = _GROUP_TYPE_TO_CODE.get((group_type or existing.group_type).strip().lower())
    if gtype_code is None:
        raise PtxValidationError(f"Invalid group_type {group_type!r}")
    members = set(existing.member_indices) if member_tracks is None else {_resolve_track_index(session, t) for t in member_tracks}
    if add_tracks:
        members.update(_resolve_track_index(session, t) for t in add_tracks)
    if remove_tracks:
        members.difference_update(_resolve_track_index(session, t) for t in remove_tracks)
    member_indices = sorted(members)
    vca_idx, vca_name, vca_updates = _compute_modified_vca(session, existing, vca_track, member_indices, target_name, gtype_code)
    new_edit_202a, new_mix_202a = _serialize_modified_group_pair(
        session, existing, name, target_name, gtype_code, member_indices, vca_idx,
        is_edit_active=is_edit_active, is_mix_active=is_mix_active,
        follow_globals=follow_globals, include_rulers=include_rulers,
    )
    c202b = session.chunk_map.get(0x202B)
    if c202b is None:
        raise PtxCorruptFileError("Missing 0x202b groups chunk in .ptx")
    edit_list, mix_list, trailer = split_202b(c202b.raw)
    old_low = existing.name.lower()
    new_edit = _replace_in_202a_list(edit_list, old_low, new_edit_202a, gtype_code in (0, 2))
    new_mix = _replace_in_202a_list(mix_list, old_low, new_mix_202a, gtype_code in (1, 2))
    splice_top_chunk(ptx_buf.data, 0x202B, _build_202b(new_edit, new_mix, trailer=trailer, version=c202b.version))
    _set_vca_track_group_names(ptx_buf.data, vca_updates)
    ptx_buf.invalidate()
    return {
        "action": "modify_group",
        "name": target_name,
        "member_tracks": [session.tracks[i].name for i in member_indices],
        "vca_track": vca_name,
    }


def patch_modify_group(
    ptx_raw: bytes,
    name: str,
    *,
    new_name: str | None = None,
    member_tracks: list[TrackRef] | None = None,
    add_tracks: list[TrackRef] | None = None,
    remove_tracks: list[TrackRef] | None = None,
    vca_track: Any = _UNSET,
    group_type: str | None = None,
    is_edit_active: bool | None = None,
    is_mix_active: bool | None = None,
    follow_globals: bool | None = None,
    include_rulers: bool | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Modify an existing group's name, members, or VCA wiring inside a ``.ptx`` buffer."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_modify_group_on_buffer(
        ptx_buf, name, new_name=new_name, member_tracks=member_tracks, add_tracks=add_tracks,
        remove_tracks=remove_tracks, vca_track=vca_track, group_type=group_type,
        is_edit_active=is_edit_active, is_mix_active=is_mix_active,
        follow_globals=follow_globals, include_rulers=include_rulers,
    )
    return ptx_buf.to_encrypted(), info


def _compute_modified_vca(
    session: PtxSession, existing: PtxGroup, vca_track: Any, member_indices: list[int], target_name: str, gtype_code: int
) -> tuple[int | None, str | None, dict[bytes, tuple[str, int]]]:
    vca_updates: dict[bytes, tuple[str, int]] = {}
    if vca_track is _UNSET:
        vca_idx = existing.vca_track_index
        vca_name = existing.vca_track_name
        if vca_idx is not None and vca_idx < len(session.tracks):
            if gtype_code == 0:
                raise PtxValidationError("Cannot convert a VCA-linked group to Edit-only without unlinking its VCA")
            vtrk = session.tracks[vca_idx]
            if vca_idx in member_indices:
                raise PtxValidationError(f"VCA track {vtrk.name!r} cannot be a member of the group it controls")
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
    validate_decrypted_ptx_pair(orig_dec, orig_delta, new_dec, new_delta)


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
    ptx_buf = PtxBuffer.from_encrypted(raw_ptx)
    meta = patch_slot_a_trim_gains_on_buffer(
        ptx_buf, track_gains_db,
        activate_if_inactive=activate_if_inactive,
        unbypass_if_bypassed=unbypass_if_bypassed,
        allow_auto_insert=allow_auto_insert,
    )
    return ptx_buf.to_encrypted(), meta


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
