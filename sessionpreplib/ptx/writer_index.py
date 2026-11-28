"""Deterministic ``0x0002`` chunk-offset index rebuilder for Pro Tools ``.ptx`` writing."""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .codec import find_zchunk_offset, iter_top_chunks
from .models import PtxChunk
from .writer_primitives import (
    PtxIndex0002,
    PtxIndexEntry,
    PtxIndexOccurrence,
    PtxIndexPathStep,
    replace_index_0002_chunk,
)

_ZERO5 = b"\x00" * 5
_ZERO6 = b"\x00" * 6


@dataclass(frozen=True)
class _Parsed2519Offsets:
    chunk_offset: int
    pass1_offsets: tuple[int, ...]
    pass2_offsets: tuple[int, ...]
    c251b_offset: int | None
    c251c_offsets: tuple[int, ...]
    c2716_offset: int | None


@dataclass(frozen=True)
class _Parsed2624StripRack:
    ordinal_1based: int
    strip_tag: int
    strip_offset: int
    c261b_offset: int | None
    c2627_offset: int


def _occ(offset: int) -> PtxIndexOccurrence:
    return PtxIndexOccurrence(1, (offset,), _ZERO6)


def _meta_ord(ordinal: int) -> bytes:
    return b"\x01" + struct.pack("<I", ordinal)


def _walk_counted_children(
    buf: bytes,
    start: int,
    end: int,
    count: int,
) -> tuple[list[tuple[int, int, int]], int]:
    """Deterministically walk *count* sequential Z-chunks starting at *start*."""
    items: list[tuple[int, int, int]] = []
    pos = start
    for _ in range(count):
        if pos + 9 > end or buf[pos] != 0x5A:
            raise RuntimeError(f"Expected Z-chunk at offset 0x{pos:x} while rebuilding 0x0002")
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        nxt = pos + 7 + sz
        if sz < 2 or nxt > end:
            raise RuntimeError(f"Truncated Z-chunk 0x{tag:04x} at offset 0x{pos:x}")
        items.append((tag, pos, nxt))
        pos = nxt
    return items, pos


def _scan_counted_container(buf: bytes, chunk: PtxChunk) -> tuple[list[tuple[int, int, int]], int | None, int | None]:
    """Return ``(items, trailing_tag, trailing_offset)`` for counted containers (``0x0F3D``/``0x1015``/``0x1054``)."""
    end = chunk.offset + 7 + chunk.size
    if chunk.offset + 13 > end:
        return [], None, None
    count = struct.unpack_from("<I", buf, chunk.offset + 9)[0]
    items, pos = _walk_counted_children(buf, chunk.offset + 13, end, count)
    if pos + 9 <= end and buf[pos] == 0x5A:
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        if sz >= 2 and pos + 7 + sz <= end:
            return items, tag, pos
    return items, None, None


def _parse_2519_offsets(buf: bytes, chunk: PtxChunk) -> _Parsed2519Offsets:
    end = chunk.offset + 7 + chunk.size
    pos = chunk.offset + 19  # skip 9B header + 10B prefix
    cnt_a = struct.unpack_from("<I", buf, pos)[0]
    pos += 4 + cnt_a * 2
    tcnt = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    for _ in range(tcnt):
        pos += 2
        nlen = struct.unpack_from("<I", buf, pos)[0]
        pos += 4 + nlen + 21
    p1_items, pos = _walk_counted_children(buf, pos, end, tcnt)
    tcnt2 = struct.unpack_from("<I", buf, pos)[0]
    p2_items, pos = _walk_counted_children(buf, pos + 4, end, tcnt2)

    c2716_off: int | None = None
    c251b_off: int | None = None
    c251c_offs: list[int] = []
    # Scan sequential tail chunks after the fixed gap bytes following Pass 2.
    scan = buf.find(b"\x5A", pos, end)
    while scan != -1 and scan + 9 <= end:
        ver = struct.unpack_from("<H", buf, scan + 1)[0]
        sz = struct.unpack_from("<I", buf, scan + 3)[0]
        tag = struct.unpack_from("<H", buf, scan + 7)[0]
        nxt = scan + 7 + sz
        if ver < 0x1000 and sz >= 2 and nxt <= end:
            if tag == 0x2716 and c2716_off is None:
                c2716_off = scan
            elif tag == 0x251B and c251b_off is None:
                c251b_off = scan
                if scan + 13 <= nxt:
                    sub_cnt = struct.unpack_from("<I", buf, scan + 9)[0]
                    subs, _ = _walk_counted_children(buf, scan + 13, nxt, sub_cnt)
                    c251c_offs = [off for t, off, _ in subs if t == 0x251C]
            scan = buf.find(b"\x5A", nxt, end)
        else:
            scan = buf.find(b"\x5A", scan + 1, end)
    return _Parsed2519Offsets(
        chunk_offset=chunk.offset,
        pass1_offsets=tuple(off for _, off, _ in p1_items),
        pass2_offsets=tuple(off for _, off, _ in p2_items),
        c251b_offset=c251b_off,
        c251c_offsets=tuple(c251c_offs),
        c2716_offset=c2716_off,
    )


def _find_2627_after_head(buf: bytes, start: int, end: int) -> int | None:
    pos = start
    if pos + 9 <= end and buf[pos] == 0x5A:
        sz0 = struct.unpack_from("<I", buf, pos + 3)[0]
        t0 = struct.unpack_from("<H", buf, pos + 7)[0]
        if sz0 >= 2 and pos + 7 + sz0 <= end:
            if t0 == 0x2627:
                return pos
            pos = pos + 7 + sz0
    for cand in (pos, pos + 4):
        if cand + 9 <= end and buf[cand] == 0x5A:
            c_sz = struct.unpack_from("<I", buf, cand + 3)[0]
            c_tag = struct.unpack_from("<H", buf, cand + 7)[0]
            if c_tag == 0x2627 and c_sz >= 2 and cand + 7 + c_sz <= end:
                return cand
    return None


def _scan_2624_offsets(buf: bytes, chunk: PtxChunk) -> tuple[list[int], list[_Parsed2624StripRack]]:
    end = chunk.offset + 7 + chunk.size
    count = struct.unpack_from("<I", buf, chunk.offset + 9)[0]
    strips, _ = _walk_counted_children(buf, chunk.offset + 13, end, count)
    strip_offsets = [off for _, off, _ in strips]
    racks: list[_Parsed2624StripRack] = []
    for ordinal, (strip_tag, strip_off, strip_end) in enumerate(strips, start=1):
        if strip_off + 18 > strip_end or buf[strip_off + 9] != 0x5A:
            continue
        b_sz = struct.unpack_from("<I", buf, strip_off + 12)[0]
        b_tag = struct.unpack_from("<H", buf, strip_off + 16)[0]
        b_end = strip_off + 9 + 7 + b_sz
        if b_tag == 0x261B and b_sz >= 2 and b_end <= strip_end:
            c27 = _find_2627_after_head(buf, strip_off + 18, b_end)
            if c27 is not None:
                racks.append(_Parsed2624StripRack(ordinal, strip_tag, strip_off, strip_off + 9, c27))
        elif b_tag == 0x102D and b_sz >= 2 and b_end <= strip_end:
            # Strips without 0x261B (e.g. Instrument 0x2621) place 0x102D + gap + 0x2627 directly inside strip.
            c27 = _find_2627_after_head(buf, strip_off + 9, strip_end)
            if c27 is not None:
                racks.append(_Parsed2624StripRack(ordinal, strip_tag, strip_off, None, c27))
    return strip_offsets, racks


def _build_2519_251a_entries(p2519: _Parsed2519Offsets) -> list[PtxIndexEntry]:
    entries = [
        PtxIndexEntry(
            path=[
                PtxIndexPathStep(0x2519, -1, _ZERO5),
                PtxIndexPathStep(0x251A, p2519.chunk_offset, _ZERO5),
            ],
            occurrences=[PtxIndexOccurrence(1, p2519.pass1_offsets, _ZERO6)],
        )
    ]
    for ordinal, off in enumerate(p2519.pass2_offsets, start=1):
        entries.append(
            PtxIndexEntry(
                path=[
                    PtxIndexPathStep(0x2519, -1, _meta_ord(ordinal)),
                    PtxIndexPathStep(0x251A, p2519.chunk_offset, _ZERO5),
                ],
                occurrences=[_occ(off)],
            )
        )
    return entries


def _build_2519_tail_entry(entry: PtxIndexEntry, p2519: _Parsed2519Offsets) -> PtxIndexEntry | None:
    tags = tuple(s.tag for s in entry.path)
    m0 = _meta_ord(len(p2519.pass2_offsets))
    if tags == (0x2519, 0x251B) and p2519.c251b_offset is not None:
        return PtxIndexEntry(
            path=[
                PtxIndexPathStep(0x2519, -1, m0),
                PtxIndexPathStep(0x251B, p2519.chunk_offset, entry.path[1].meta),
            ],
            occurrences=[_occ(p2519.c251b_offset), *(_occ(o) for o in p2519.c251c_offsets)],
        )
    if tags == (0x2519, 0x251B, 0x251C) and p2519.c251b_offset is not None:
        ord_k = struct.unpack_from("<I", entry.path[1].meta, 1)[0]
        if 1 <= ord_k <= len(p2519.c251c_offsets):
            return PtxIndexEntry(
                path=[
                    PtxIndexPathStep(0x2519, -1, m0),
                    PtxIndexPathStep(0x251B, p2519.chunk_offset, entry.path[1].meta),
                    PtxIndexPathStep(0x251C, p2519.c251b_offset, _ZERO5),
                ],
                occurrences=[_occ(p2519.c251c_offsets[ord_k - 1])],
            )
        return None
    if tags == (0x2519, 0x2716) and p2519.c2716_offset is not None:
        return PtxIndexEntry(
            path=[
                PtxIndexPathStep(0x2519, -1, m0),
                PtxIndexPathStep(0x2716, p2519.chunk_offset, _ZERO5),
            ],
            occurrences=[_occ(p2519.c2716_offset)],
        )
    return None


def _build_2624_rack_entries(c2624_off: int, racks: list[_Parsed2624StripRack]) -> list[PtxIndexEntry]:
    entries: list[PtxIndexEntry] = []
    for r in racks:
        if r.c261b_offset is not None:
            path = [
                PtxIndexPathStep(0x2624, -1, _meta_ord(r.ordinal_1based)),
                PtxIndexPathStep(r.strip_tag, c2624_off, _ZERO5),
                PtxIndexPathStep(0x261B, r.strip_offset, _ZERO5),
                PtxIndexPathStep(0x2627, r.c261b_offset, _ZERO5),
            ]
        else:
            path = [
                PtxIndexPathStep(0x2624, -1, _meta_ord(r.ordinal_1based)),
                PtxIndexPathStep(r.strip_tag, c2624_off, _ZERO5),
                PtxIndexPathStep(0x2627, r.strip_offset, _ZERO5),
            ]
        entries.append(PtxIndexEntry(path=path, occurrences=[_occ(r.c2627_offset)]))
    return entries


def _index_top_chunks_by_key(chunks: list[PtxChunk]) -> dict[tuple[int, int], PtxChunk]:
    counts: dict[int, int] = {}
    out: dict[tuple[int, int], PtxChunk] = {}
    for ch in chunks:
        ord_idx = counts.get(ch.tag, 0)
        counts[ch.tag] = ord_idx + 1
        out[(ch.tag, ord_idx)] = ch
    return out


def _find_entry_delta(
    entry: PtxIndexEntry,
    old_by_key: dict[tuple[int, int], PtxChunk],
    new_by_key: dict[tuple[int, int], PtxChunk],
) -> int | None:
    if not entry.path:
        return None
    top_tag = entry.path[0].tag
    ref_off = next((o for occ in entry.occurrences for o in occ.offsets), None)
    if ref_off is not None:
        for (tag, ord_idx), old_ch in old_by_key.items():
            if tag == top_tag and old_ch.offset <= ref_off < old_ch.offset + 7 + old_ch.size:
                new_ch = new_by_key.get((tag, ord_idx))
                if new_ch is not None:
                    return new_ch.offset - old_ch.offset
    old_first = old_by_key.get((top_tag, 0))
    new_first = new_by_key.get((top_tag, 0))
    if old_first is not None and new_first is not None:
        return new_first.offset - old_first.offset
    return None


def _compute_2587_internal_shift(
    old_chunks: list[PtxChunk],
    new_chunks: list[PtxChunk],
) -> tuple[int, int] | None:
    """Return ``(old_258a_end_abs, inner_delta)`` when the last ``0x2587`` resized its ``0x258A`` track list."""
    old_ch = next((c for c in reversed(old_chunks) if c.tag == 0x2587), None)
    new_ch = next((c for c in reversed(new_chunks) if c.tag == 0x2587), None)
    if old_ch is None or new_ch is None:
        return None
    inner_delta = len(new_ch.raw) - len(old_ch.raw)
    if inner_delta == 0:
        return None
    p2551 = find_zchunk_offset(old_ch.raw, 0x2551, 9)
    if p2551 < 0:
        return None
    p2551_end = p2551 + 7 + struct.unpack_from("<I", old_ch.raw, p2551 + 3)[0]
    p258b = find_zchunk_offset(old_ch.raw, 0x258B, p2551 + 9, p2551_end)
    if p258b < 0:
        return None
    p258b_end = p258b + 7 + struct.unpack_from("<I", old_ch.raw, p258b + 3)[0]
    p258a = find_zchunk_offset(old_ch.raw, 0x258A, p258b + 9, p258b_end)
    if p258a < 0:
        return None
    old_258a_end_rel = p258a + 7 + struct.unpack_from("<I", old_ch.raw, p258a + 3)[0]
    return old_ch.offset + old_258a_end_rel, inner_delta


def _shift_static_entry(
    entry: PtxIndexEntry,
    old_by_key: dict[tuple[int, int], PtxChunk],
    new_by_key: dict[tuple[int, int], PtxChunk],
    shift_2587: tuple[int, int] | None = None,
) -> PtxIndexEntry:
    delta = _find_entry_delta(entry, old_by_key, new_by_key)
    if delta is None:
        return entry
    is_2587 = bool(entry.path and entry.path[0].tag == 0x2587 and shift_2587 is not None)
    if delta == 0 and not is_2587:
        return entry
    thresh_258a, inner_delta = shift_2587 if (is_2587 and shift_2587 is not None) else (0, 0)

    def _adj(off: int) -> int:
        return off + delta + (inner_delta if (is_2587 and off >= thresh_258a) else 0)

    new_path = [
        PtxIndexPathStep(s.tag, (-1 if s.offset == -1 else _adj(s.offset)), s.meta)
        for s in entry.path
    ]
    new_occs = [
        PtxIndexOccurrence(o.flag, tuple(_adj(x) for x in o.offsets), o.tail)
        for o in entry.occurrences
    ]
    return PtxIndexEntry(path=new_path, occurrences=new_occs)


@dataclass
class _RebuildContext:
    c0f3d_off: int | None
    c0f3d_subs: list[int]
    c1015_off: int | None
    c1015_subs: list[int]
    c4829_off: int | None
    c1054_off: int | None
    c1054_subs: list[int]
    c4828_off: int | None
    c1058_off: int | None
    c1058_subs: list[int]
    p2519: _Parsed2519Offsets | None
    c2624_off: int | None
    c2624_strips: list[int]
    c2624_racks: list[_Parsed2624StripRack]
    has_template_rack_entry: bool
    done_251a: bool = False
    done_2624: bool = False


def _prepare_rebuild_context(new_buf: bytes, new_chunks: list[PtxChunk], template_index: PtxIndex0002) -> _RebuildContext:
    by_tag = {c.tag: c for c in new_chunks}
    c0f3d, c1015, c1054, c1058 = by_tag.get(0x0F3D), by_tag.get(0x1015), by_tag.get(0x1054), by_tag.get(0x1058)
    c2519, c2624 = by_tag.get(0x2519), by_tag.get(0x2624)

    f3d_items, _, _ = _scan_counted_container(new_buf, c0f3d) if c0f3d else ([], None, None)
    i15_items, t4829, o4829 = _scan_counted_container(new_buf, c1015) if c1015 else ([], None, None)
    i54_items, t4828, o4828 = _scan_counted_container(new_buf, c1054) if c1054 else ([], None, None)
    i58_items, _, _ = _scan_counted_container(new_buf, c1058) if c1058 else ([], None, None)
    p2519 = _parse_2519_offsets(new_buf, c2519) if c2519 else None
    strips, racks = _scan_2624_offsets(new_buf, c2624) if c2624 else ([], [])
    has_rack = any(
        e.path and e.path[0].tag == 0x2624 and (
            (len(e.path) == 4 and e.path[2].tag == 0x261B and e.path[3].tag == 0x2627)
            or (len(e.path) == 3 and e.path[2].tag == 0x2627)
        )
        for e in template_index.entries
    )
    return _RebuildContext(
        c0f3d_off=c0f3d.offset if c0f3d else None,
        c0f3d_subs=[o for t, o, _ in f3d_items if t == 0x0F3C],
        c1015_off=c1015.offset if c1015 else None,
        c1015_subs=[o for t, o, _ in i15_items if t == 0x1014],
        c4829_off=o4829 if t4829 == 0x4829 else None,
        c1054_off=c1054.offset if c1054 else None,
        c1054_subs=[o for t, o, _ in i54_items if t == 0x1052],
        c4828_off=o4828 if t4828 == 0x4828 else None,
        c1058_off=c1058.offset if c1058 else None,
        c1058_subs=[o for t, o, _ in i58_items if t == 0x1057],
        p2519=p2519,
        c2624_off=c2624.offset if c2624 else None,
        c2624_strips=strips,
        c2624_racks=racks,
        has_template_rack_entry=has_rack,
    )


def _rebuild_dynamic_entry(entry: PtxIndexEntry, ctx: _RebuildContext) -> list[PtxIndexEntry] | None:
    if not entry.path:
        return None
    tags = tuple(s.tag for s in entry.path)
    if tags == (0x0F3D,) and ctx.c0f3d_off is not None:
        return [PtxIndexEntry(list(entry.path), [_occ(ctx.c0f3d_off), *(_occ(o) for o in ctx.c0f3d_subs)])]
    if tags == (0x1015,) and ctx.c1015_off is not None:
        return [PtxIndexEntry(list(entry.path), [_occ(ctx.c1015_off), *(_occ(o) for o in ctx.c1015_subs)])]
    if tags == (0x1015, 0x4829) and ctx.c1015_off is not None:
        if ctx.c4829_off is None:
            return []
        return [PtxIndexEntry([
            PtxIndexPathStep(0x1015, -1, entry.path[0].meta),
            PtxIndexPathStep(0x4829, ctx.c1015_off, entry.path[1].meta),
        ], [_occ(ctx.c4829_off)])]
    if tags == (0x1054,) and ctx.c1054_off is not None:
        return [PtxIndexEntry(list(entry.path), [_occ(ctx.c1054_off), *(_occ(o) for o in ctx.c1054_subs)])]
    if tags == (0x1054, 0x4828) and ctx.c1054_off is not None:
        if ctx.c4828_off is None:
            return []
        return [PtxIndexEntry([
            PtxIndexPathStep(0x1054, -1, entry.path[0].meta),
            PtxIndexPathStep(0x4828, ctx.c1054_off, entry.path[1].meta),
        ], [_occ(ctx.c4828_off)])]
    if tags == (0x1058,) and ctx.c1058_off is not None:
        return [PtxIndexEntry(list(entry.path), [_occ(ctx.c1058_off), *(_occ(o) for o in ctx.c1058_subs)])]
    if tags[0] == 0x2519 and ctx.p2519 is not None:
        if tags == (0x2519,):
            return [PtxIndexEntry(list(entry.path), [_occ(ctx.p2519.chunk_offset), *(_occ(o) for o in ctx.p2519.pass2_offsets)])]
        if tags == (0x2519, 0x251A):
            if ctx.done_251a:
                return []
            ctx.done_251a = True
            return _build_2519_251a_entries(ctx.p2519)
        tail = _build_2519_tail_entry(entry, ctx.p2519)
        return [tail] if tail is not None else []
    if tags == (0x2624,) and ctx.c2624_off is not None:
        res = [PtxIndexEntry(list(entry.path), [_occ(ctx.c2624_off), *(_occ(o) for o in ctx.c2624_strips)])]
        if not ctx.has_template_rack_entry and ctx.c2624_racks and not ctx.done_2624:
            ctx.done_2624 = True
            res.extend(_build_2624_rack_entries(ctx.c2624_off, ctx.c2624_racks))
        return res
    if (
        tags[0] == 0x2624
        and ctx.c2624_off is not None
        and ((len(tags) == 4 and tags[2:] == (0x261B, 0x2627)) or (len(tags) == 3 and tags[2] == 0x2627))
    ):
        if ctx.done_2624:
            return []
        ctx.done_2624 = True
        return _build_2624_rack_entries(ctx.c2624_off, ctx.c2624_racks)
    return None


def _append_synthesized_chunk_entries(
    out_entries: list[PtxIndexEntry],
    old_tags: set[int],
    new_chunks: list[PtxChunk],
    ctx: _RebuildContext,
) -> None:
    """Append canonical ``0x0002`` entries only for chunks newly synthesized onto virgin templates."""
    present_paths = {tuple(s.tag for s in e.path) for e in out_entries if e.path}
    by_tag = {c.tag: c for c in new_chunks}
    m0 = _meta_ord(0)
    if 0x0F3D not in old_tags and (0x0F3D,) not in present_paths and ctx.c0f3d_off is not None:
        out_entries.append(PtxIndexEntry(
            [PtxIndexPathStep(0x0F3D, -1, m0)],
            [_occ(ctx.c0f3d_off), *(_occ(o) for o in ctx.c0f3d_subs)],
        ))
    for tag in (0x1004, 0x2637, 0x262A, 0x2630, 0x210C):
        if tag not in old_tags and (tag,) not in present_paths and tag in by_tag:
            out_entries.append(PtxIndexEntry([PtxIndexPathStep(tag, -1, _ZERO5)], [_occ(by_tag[tag].offset)]))
    if 0x1058 not in old_tags and (0x1058,) not in present_paths and ctx.c1058_off is not None:
        out_entries.append(PtxIndexEntry(
            [PtxIndexPathStep(0x1058, -1, m0)],
            [_occ(ctx.c1058_off), *(_occ(o) for o in ctx.c1058_subs)],
        ))
    if 0x1054 not in old_tags and ctx.c1054_off is not None:
        if (0x1054,) not in present_paths:
            out_entries.append(PtxIndexEntry(
                [PtxIndexPathStep(0x1054, -1, m0)],
                [_occ(ctx.c1054_off), *(_occ(o) for o in ctx.c1054_subs)],
            ))
        if (0x1054, 0x4828) not in present_paths and ctx.c4828_off is not None:
            out_entries.append(PtxIndexEntry(
                [PtxIndexPathStep(0x1054, -1, m0), PtxIndexPathStep(0x4828, ctx.c1054_off, m0)],
                [_occ(ctx.c4828_off)],
            ))
    if 0x1015 not in old_tags and ctx.c1015_off is not None:
        if (0x1015,) not in present_paths:
            out_entries.append(PtxIndexEntry(
                [PtxIndexPathStep(0x1015, -1, m0)],
                [_occ(ctx.c1015_off), *(_occ(o) for o in ctx.c1015_subs)],
            ))
        if (0x1015, 0x4829) not in present_paths and ctx.c4829_off is not None:
            out_entries.append(PtxIndexEntry(
                [PtxIndexPathStep(0x1015, -1, m0), PtxIndexPathStep(0x4829, ctx.c1015_off, m0)],
                [_occ(ctx.c4829_off)],
            ))


def rebuild_index_0002(
    template_index: PtxIndex0002,
    old_buf: bytes,
    new_buf: bytes,
) -> PtxIndex0002:
    """Rebuild ``0x0002`` deterministically for *new_buf* from *template_index* and *old_buf*."""
    old_chunks = iter_top_chunks(old_buf, strict=True)
    new_chunks = iter_top_chunks(new_buf, strict=True)
    old_by_key = _index_top_chunks_by_key(old_chunks)
    new_by_key = _index_top_chunks_by_key(new_chunks)
    shift_2587 = _compute_2587_internal_shift(old_chunks, new_chunks)
    ctx = _prepare_rebuild_context(new_buf, new_chunks, template_index)

    out_entries: list[PtxIndexEntry] = []
    for entry in template_index.entries:
        dynamic = _rebuild_dynamic_entry(entry, ctx)
        if dynamic is not None:
            out_entries.extend(dynamic)
        else:
            out_entries.append(_shift_static_entry(entry, old_by_key, new_by_key, shift_2587))
    _append_synthesized_chunk_entries(out_entries, {c.tag for c in old_chunks}, new_chunks, ctx)
    return PtxIndex0002(version=template_index.version, entries=out_entries)


def apply_rebuilt_index_0002(
    buf: bytearray,
    template_index: PtxIndex0002,
    old_buf: bytes,
) -> None:
    """Rebuild and splice the trailing ``0x0002`` index into *buf* in place."""
    rebuilt = rebuild_index_0002(template_index, old_buf, bytes(buf))
    replace_index_0002_chunk(buf, rebuilt)
