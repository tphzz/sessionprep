"""Clip and fade ``.ptx`` patchers: clip renames, placement attributes, and fade/crossfade synthesis."""

from __future__ import annotations

from collections.abc import Mapping
import struct
from typing import Any

from ._document import PtxBuffer, assemble_top_chunks
from .codec import (
    find_zchunk_offset,
    read_zchunk_at,
    resize_zchunk_ancestors,
    walk_sequential_zchunks,
)
from .models import (
    TICK_MASK_56,
    PtxClipAttributesPatch,
    PtxCorruptFileError,
    PtxNotFoundError,
    PtxTrack,
    PtxValidationError,
)
from .parser import PtxSession
from .writer_chunks import serialize_fade_262f, serialize_placement_104f
from .writer_index import apply_rebuilt_index_0002
from .writer_primitives import build_zchunk, encode_lp_str, parse_index_0002


def _rewrite_clip_2629_name(c2629: bytes, new_name: str) -> bytes:
    p28 = find_zchunk_offset(c2629, 0x2628, 9)
    if p28 < 0 or p28 + 13 > len(c2629):
        raise PtxCorruptFileError("Missing or truncated 0x2628 inside 0x2629 clip")
    end_28 = p28 + 7 + struct.unpack_from("<I", c2629, p28 + 3)[0]
    old_nlen = struct.unpack_from("<I", c2629, p28 + 9)[0]
    if p28 + 13 + old_nlen > end_28:
        raise PtxCorruptFileError("Truncated 0x2628 clip name length")
    new_lp = encode_lp_str(new_name)
    delta = len(new_lp) - (4 + old_nlen)
    buf = bytearray(c2629)
    buf[p28 + 9 : p28 + 13 + old_nlen] = new_lp
    resize_zchunk_ancestors(buf, (0, p28), delta)
    return bytes(buf)


def patch_rename_clips_on_buffer(ptx_buf: PtxBuffer, renames: Mapping[str, str]) -> dict[str, str]:
    """Rename one or more audio clips inside ``0x262A`` -> ``0x2629`` -> ``0x2628`` in-place on *ptx_buf*."""
    if not renames:
        raise PtxValidationError("renames must be a non-empty {old_name: new_name} mapping")
    session = ptx_buf.session
    c262a = session.chunk_map.get(0x262A)
    if c262a is None or len(c262a.raw) < 13:
        raise PtxCorruptFileError("Missing or truncated 0x262A clip container chunk in .ptx")
    seen_keys: set[str] = set()
    rename_by_idx: dict[int, str] = {}
    summary: dict[str, str] = {}
    for old_nm, new_nm in renames.items():
        low_key = old_nm.lower()
        if low_key in seen_keys:
            raise PtxValidationError(f"Duplicate rename key for clip {old_nm!r}")
        seen_keys.add(low_key)
        cleaned = new_nm.strip()
        if not cleaned:
            raise PtxValidationError(f"New clip name for {old_nm!r} must not be empty")
        matches = [c for c in session.clips if c.name == old_nm] or [c for c in session.clips if c.name.lower() == low_key]
        if not matches:
            raise PtxNotFoundError(f"Clip {old_nm!r} not found in .ptx session")
        for clip in matches:
            if clip.index in rename_by_idx:
                raise PtxValidationError(f"Duplicate clip rename target for clip index {clip.index} ({clip.name!r})")
            rename_by_idx[clip.index] = cleaned
            summary[clip.name] = cleaned
    ver = struct.unpack_from("<H", c262a.raw, 1)[0]
    cnt = struct.unpack_from("<I", c262a.raw, 9)[0]
    items, end_pos = walk_sequential_zchunks(c262a.raw, 13, cnt, expected_tag=0x2629)
    new_items = [_rewrite_clip_2629_name(it, rename_by_idx[i]) if i in rename_by_idx else it for i, it in enumerate(items)]
    new_262a = build_zchunk(0x262A, struct.pack("<I", len(new_items)) + b"".join(new_items) + c262a.raw[end_pos:], version=ver)
    old_dec = bytes(ptx_buf.data)
    template_idx = parse_index_0002(old_dec)
    out = assemble_top_chunks(old_dec, {0x262A: new_262a})
    if template_idx is not None:
        apply_rebuilt_index_0002(out, template_idx, old_dec)
    ptx_buf.replace_data(out)
    return summary


def patch_rename_clips(ptx_raw: bytes, renames: Mapping[str, str]) -> tuple[bytes, dict[str, str]]:
    """Rename one or more audio clips inside ``0x262A`` -> ``0x2629`` -> ``0x2628`` and rebuild ``0x0002``."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    summary = patch_rename_clips_on_buffer(ptx_buf, renames)
    return ptx_buf.to_encrypted(), summary


def _resolve_audio_track_lanes(session: PtxSession, track_name: str) -> tuple[PtxTrack, set[int]]:
    trk = session.track_by_name(track_name)
    if trk is None or trk.track_type != "Audio":
        raise PtxNotFoundError(f"Audio track {track_name!r} not found in .ptx session")
    pl = next((p for p in session.playlists if not p.is_alternate and (p.track_name or p.name).lower() == trk.name.lower()), None)
    if pl is None or not pl.lane_indices:
        raise PtxCorruptFileError(f"Audio track {trk.name!r} has no active playlist lanes in 0x1015")
    return trk, set(pl.lane_indices)


def _patch_104f_entry(
    buf: bytearray,
    hit: int,
    fsz: int,
    start_sample: int,
    allowed_clip_indices: set[int] | None,
    *,
    is_muted: bool | None,
    is_time_locked: bool | None,
    is_edit_locked: bool | None,
    color_index: int | None,
) -> bool:
    if fsz < 37:
        return False
    clip_idx = struct.unpack_from("<I", buf, hit + 11)[0]
    if allowed_clip_indices is not None and clip_idx not in allowed_clip_indices:
        return False
    if (struct.unpack_from("<Q", buf, hit + 16)[0] & TICK_MASK_56) != start_sample:
        return False
    if is_muted is not None:
        buf[hit + 9] = 1 if is_muted else 0
    if is_edit_locked is not None:
        buf[hit + 10] = 1 if is_edit_locked else 0
        if is_edit_locked:
            buf[hit + 39] = 1
    if is_time_locked is not None:
        buf[hit + 39] = 1 if (is_time_locked or bool(buf[hit + 10])) else 0
    if color_index is not None:
        struct.pack_into("<H", buf, hit + 25, color_index)
    return True


def _patch_104f_in_lane(
    buf: bytearray,
    lane_pos: int,
    l_end: int,
    start_sample: int,
    allowed_clip_indices: set[int] | None,
    *,
    is_muted: bool | None,
    is_time_locked: bool | None,
    is_edit_locked: bool | None,
    color_index: int | None,
) -> int:
    """Structurally walk counted items in a ``0x1052`` lane (handling wrapped ``0x1050`` and bare ``0x104F``)."""
    if lane_pos + 17 > l_end:
        return 0
    nlen = struct.unpack_from("<I", buf, lane_pos + 9)[0]
    cnt_off = lane_pos + 13 + nlen
    if cnt_off + 4 > l_end:
        return 0
    item_cnt = struct.unpack_from("<I", buf, cnt_off)[0]
    cur = cnt_off + 4
    patched = 0
    for _ in range(item_cnt):
        if cur + 9 > l_end or buf[cur] != 0x5A:
            break
        itag, _raw, item_end = read_zchunk_at(buf, cur, limit=l_end)
        if itag == 0x1050:
            hit = find_zchunk_offset(buf, 0x104F, cur + 9, item_end)
            if hit >= 0:
                fsz = struct.unpack_from("<I", buf, hit + 3)[0]
                f_end = hit + 7 + fsz
                is_fade = f_end < item_end and buf[f_end] == 1
                if not is_fade and _patch_104f_entry(
                    buf, hit, fsz, start_sample, allowed_clip_indices,
                    is_muted=is_muted, is_time_locked=is_time_locked,
                    is_edit_locked=is_edit_locked, color_index=color_index,
                ):
                    patched += 1
        elif itag == 0x104F:
            fsz = struct.unpack_from("<I", buf, cur + 3)[0]
            if _patch_104f_entry(
                buf, cur, fsz, start_sample, allowed_clip_indices,
                is_muted=is_muted, is_time_locked=is_time_locked,
                is_edit_locked=is_edit_locked, color_index=color_index,
            ):
                patched += 1
        cur = item_end
    return patched


def _normalize_clip_attr_args(
    start_sample: int,
    attrs: PtxClipAttributesPatch | None,
    is_muted: bool | None,
    is_time_locked: bool | None,
    is_edit_locked: bool | None,
    color_index: int | None,
    clip_name: str | None,
    clip_index: int | None,
) -> tuple[bool | None, bool | None, bool | None, int | None, str | None, int | None]:
    if attrs is not None:
        is_muted = attrs.is_muted if is_muted is None else is_muted
        is_time_locked = attrs.is_time_locked if is_time_locked is None else is_time_locked
        is_edit_locked = attrs.is_edit_locked if is_edit_locked is None else is_edit_locked
        color_index = attrs.color_index if color_index is None else color_index
        clip_name = attrs.clip_name if clip_name is None else clip_name
        clip_index = attrs.clip_index if clip_index is None else clip_index
    if isinstance(start_sample, bool) or not (0 <= start_sample <= TICK_MASK_56):
        raise PtxValidationError(f"start_sample out of 56-bit range: {start_sample!r}")
    if is_edit_locked is True and is_time_locked is False:
        raise PtxValidationError("Contradictory clip lock flags: is_edit_locked=True requires is_time_locked=True")
    if color_index is not None and (isinstance(color_index, bool) or not (0 <= color_index <= 0xFFFF)):
        raise PtxValidationError(f"color_index out of u16 range: {color_index!r}")
    if clip_index is not None and clip_name is not None:
        raise PtxValidationError("Specify either clip_index or clip_name, not both")
    if clip_index is not None and (isinstance(clip_index, bool) or clip_index < 0):
        raise PtxValidationError(f"Invalid clip_index: {clip_index!r}")
    return is_muted, is_time_locked, is_edit_locked, color_index, clip_name, clip_index


def patch_clip_attributes_on_buffer(
    ptx_buf: PtxBuffer,
    track_name: str,
    start_sample: int,
    attrs: PtxClipAttributesPatch | None = None,
    *,
    is_muted: bool | None = None,
    is_time_locked: bool | None = None,
    is_edit_locked: bool | None = None,
    color_index: int | None = None,
    clip_name: str | None = None,
    clip_index: int | None = None,
) -> dict[str, Any]:
    """Patch ``0x104F`` clip placement mute, time-lock, edit-lock, and/or color_index in-place on *ptx_buf*."""
    is_muted, is_time_locked, is_edit_locked, color_index, clip_name, clip_index = _normalize_clip_attr_args(
        start_sample, attrs, is_muted, is_time_locked, is_edit_locked, color_index, clip_name, clip_index,
    )
    session = ptx_buf.session
    trk, target_lanes = _resolve_audio_track_lanes(session, track_name)
    allowed_indices: set[int] | None = None
    if clip_index is not None:
        allowed_indices = {clip_index}
    elif clip_name is not None:
        low = clip_name.lower()
        allowed_indices = {c.index for c in session.clips if c.name.lower() == low}
        if not allowed_indices:
            raise PtxNotFoundError(f"Clip {clip_name!r} not found in .ptx session")
    c1054 = session.chunk_map.get(0x1054)
    if c1054 is None or len(c1054.raw) < 13:
        raise PtxCorruptFileError("Missing or truncated 0x1054 timeline lanes chunk in .ptx")
    cnt = struct.unpack_from("<I", ptx_buf.data, c1054.offset + 9)[0]
    pos, end = c1054.offset + 13, c1054.offset + len(c1054.raw)
    patched_count = 0
    for lane_idx in range(cnt):
        if pos + 9 > end or ptx_buf.data[pos] != 0x5A:
            break
        l_end = pos + 7 + struct.unpack_from("<I", ptx_buf.data, pos + 3)[0]
        if l_end > end:
            break
        if lane_idx in target_lanes:
            patched_count += _patch_104f_in_lane(
                ptx_buf.data, pos, l_end, start_sample, allowed_indices,
                is_muted=is_muted, is_time_locked=is_time_locked,
                is_edit_locked=is_edit_locked, color_index=color_index,
            )
        pos = l_end
    if patched_count == 0:
        raise PtxNotFoundError(f"No clip placement found on track {trk.name!r} at start_sample={start_sample}")
    ptx_buf.invalidate()
    return {"track_name": trk.name, "start_sample": start_sample, "lanes_patched": patched_count}


def patch_clip_attributes(
    ptx_raw: bytes,
    track_name: str,
    start_sample: int,
    attrs: PtxClipAttributesPatch | None = None,
    *,
    is_muted: bool | None = None,
    is_time_locked: bool | None = None,
    is_edit_locked: bool | None = None,
    color_index: int | None = None,
    clip_name: str | None = None,
    clip_index: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Patch ``0x104F`` clip placement mute, time-lock, edit-lock, and/or color_index in place."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_clip_attributes_on_buffer(
        ptx_buf, track_name, start_sample, attrs,
        is_muted=is_muted, is_time_locked=is_time_locked,
        is_edit_locked=is_edit_locked, color_index=color_index,
        clip_name=clip_name, clip_index=clip_index,
    )
    return ptx_buf.to_encrypted(), info


def _insert_fade_into_lane_1052(lane_raw: bytes, fade_index: int, start_sample: int) -> bytes:
    ver = struct.unpack_from("<H", lane_raw, 1)[0]
    nlen = struct.unpack_from("<I", lane_raw, 9)[0]
    cnt_off = 13 + nlen
    if cnt_off + 4 > len(lane_raw):
        raise PtxCorruptFileError("Truncated 0x1052 lane header")
    old_cnt = struct.unpack_from("<I", lane_raw, cnt_off)[0]
    old_items, after = walk_sequential_zchunks(lane_raw, cnt_off + 4, old_cnt, expected_tag=0x1050)
    tail = lane_raw[after:]
    fade_104f = serialize_placement_104f(fade_index, start_sample, flags=1, color_index=0xFFFE)
    fade_1050 = build_zchunk(0x1050, fade_104f + b"\x01\x01\x01", version=3)

    def _ev_key(w1050: bytes) -> tuple[int, int]:
        p4f = find_zchunk_offset(w1050, 0x104F, 9)
        if p4f >= 0 and p4f + 24 <= len(w1050):
            smp = struct.unpack_from("<Q", w1050, p4f + 16)[0] & TICK_MASK_56
            f_end = p4f + 7 + struct.unpack_from("<I", w1050, p4f + 3)[0]
            is_f = 0 if (f_end < len(w1050) and w1050[f_end] == 1) else 1
            return smp, is_f
        return 0, 1

    new_items = sorted([*old_items, fade_1050], key=_ev_key)
    body = lane_raw[9:cnt_off] + struct.pack("<I", len(new_items)) + b"".join(new_items) + tail
    return build_zchunk(0x1052, body, version=ver)


def patch_add_fade_on_buffer(
    ptx_buf: PtxBuffer,
    track_name: str,
    fade_type: str,
    start_sample: int,
    total_samples: int,
    *,
    pre_samples: int | None = None,
    shape: str = "standard",
    crossfade_link: str = "equal_power",
) -> dict[str, Any]:
    """Append a fade-in, fade-out, or crossfade to ``0x2630`` and place it in ``0x1054`` in-place on *ptx_buf*."""
    session = ptx_buf.session
    trk, target_lanes = _resolve_audio_track_lanes(session, track_name)
    c1054 = session.chunk_map.get(0x1054)
    if c1054 is None or len(c1054.raw) < 13:
        raise PtxCorruptFileError("Missing or truncated 0x1054 timeline lanes chunk in .ptx")
    fade_chunk = serialize_fade_262f(
        fade_type, total_samples, pre_samples=pre_samples, shape=shape, crossfade_link=crossfade_link,
    )
    c2630 = session.chunk_map.get(0x2630)
    if c2630 is not None and len(c2630.raw) >= 13:
        fver, fcnt = struct.unpack_from("<H", c2630.raw, 1)[0], struct.unpack_from("<I", c2630.raw, 9)[0]
        old_fades, f_end = walk_sequential_zchunks(c2630.raw, 13, fcnt, expected_tag=0x262F)
        fade_idx = len(old_fades)
        new_2630 = build_zchunk(0x2630, struct.pack("<I", fade_idx + 1) + b"".join([*old_fades, fade_chunk]) + c2630.raw[f_end:], version=fver)
    else:
        fade_idx = 0
        new_2630 = build_zchunk(0x2630, struct.pack("<I", 1) + fade_chunk, version=1)
    lver, lcnt = struct.unpack_from("<H", c1054.raw, 1)[0], struct.unpack_from("<I", c1054.raw, 9)[0]
    old_lanes, l_end = walk_sequential_zchunks(c1054.raw, 13, lcnt, expected_tag=0x1052)
    new_lanes = [_insert_fade_into_lane_1052(ln, fade_idx, start_sample) if i in target_lanes else ln for i, ln in enumerate(old_lanes)]
    new_1054 = build_zchunk(0x1054, struct.pack("<I", len(new_lanes)) + b"".join(new_lanes) + c1054.raw[l_end:], version=lver)
    old_dec = bytes(ptx_buf.data)
    template_idx = parse_index_0002(old_dec)
    replacements = {0x1054: new_1054}
    insertions: tuple[tuple[int, bytes, tuple[int, ...]], ...] = ()
    if 0x2630 in session.chunk_map:
        replacements[0x2630] = new_2630
    else:
        insertions = ((0x2630, new_2630, (0x2504, 0x262A)),)
    out = assemble_top_chunks(old_dec, replacements, insertions=insertions)
    if template_idx is not None:
        apply_rebuilt_index_0002(out, template_idx, old_dec)
    ptx_buf.replace_data(out)
    return {
        "action": "add_fade", "track_name": trk.name, "fade_type": fade_type,
        "fade_index": fade_idx, "start_sample": start_sample, "total_samples": total_samples,
    }


def patch_add_fade(
    ptx_raw: bytes,
    track_name: str,
    fade_type: str,
    start_sample: int,
    total_samples: int,
    *,
    pre_samples: int | None = None,
    shape: str = "standard",
    crossfade_link: str = "equal_power",
) -> tuple[bytes, dict[str, Any]]:
    """Append a fade-in, fade-out, or crossfade to ``0x2630`` and place it on *track_name* in ``0x1054``."""
    ptx_buf = PtxBuffer.from_encrypted(ptx_raw)
    info = patch_add_fade_on_buffer(
        ptx_buf, track_name, fade_type, start_sample, total_samples,
        pre_samples=pre_samples, shape=shape, crossfade_link=crossfade_link,
    )
    return ptx_buf.to_encrypted(), info
