"""Binary codec, Z-chunk traversal, control curves, and ``0x0002`` offset-index maintenance for ``.ptx``."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import struct
import tempfile

from .models import (
    FPS_CODE_MAP,
    PT12_COLOR_PALETTE,
    PTX_MAGIC,
    SLOT_LETTERS_LOWER,
    ZERO_TICKS,
    PtxAutomationPoint,
    PtxChunk,
    PtxCorruptFileError,
    PtxUnsupportedTemplateError,
    PtxValidationError,
)

_XOR_TABLES: tuple[bytes, ...] = tuple(bytes(b ^ k for b in range(256)) for k in range(256))


def format_ptsl_uuid(raw16: bytes) -> str:
    """Format a 16-byte PTX object UUID into PTSL's braced ``{XXXXXXXX-...}`` form."""
    if len(raw16) != 16:
        return ""
    a, b, c, d = raw16[0:4].hex(), raw16[4:8].hex(), raw16[8:12].hex(), raw16[12:16].hex()
    return f"{{{a}-{b}-{c}-{d}}}"


def format_rfc_uuid(raw16: bytes) -> str:
    """Format a 16-byte big-endian UUID into standard 8-4-4-4-12 hyphenated form."""
    if len(raw16) != 16:
        return ""
    h = raw16.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def decode_slot_mask(mask16: int) -> list[str]:
    """Decode a 10-bit Insert/Send slot bitmask into lowercase slot letters ``a..j``."""
    return [SLOT_LETTERS_LOWER[i] for i in range(10) if mask16 & (1 << i)]


def encode_slot_mask(slots: list[str]) -> int:
    """Encode lowercase slot letters ``a..j`` into a 10-bit bitmask."""
    mask = 0
    for ch in slots:
        idx = SLOT_LETTERS_LOWER.find(str(ch).lower())
        if 0 <= idx < 10:
            mask |= 1 << idx
    return mask


def group_id_to_label(id_number: int) -> str:
    """Format a Pro Tools numeric group ID (``-1`` = ``!``, ``1..26`` = ``a..z``, ``27..104`` = ``2a..4z``)."""
    if id_number == -1:
        return "!"
    if 1 <= id_number <= 26:
        return chr(ord("a") + id_number - 1)
    if 27 <= id_number <= 104:
        bank = (id_number - 1) // 26 + 1
        letter = chr(ord("a") + (id_number - 1) % 26)
        return f"{bank}{letter}"
    return str(id_number)


def lookup_pt_color_rgb(color_index: int) -> tuple[int, int, int] | None:
    """Return ``(r, g, b)`` for a known Pro Tools 12 track colour palette index, or ``None``."""
    return PT12_COLOR_PALETTE.get(color_index)


def decode_ticks_relative(raw_ticks: int) -> int:
    """Convert an unsigned Pro Tools tick timestamp into ticks relative to ``ZERO_TICKS``."""
    return max(0, int(raw_ticks) - ZERO_TICKS) if raw_ticks >= ZERO_TICKS else int(raw_ticks)


def format_timecode(start_frames: int, fps_code: int) -> str:
    """Format an integer frame offset using a Pro Tools ``0x204d`` frame-rate code."""
    entry = FPS_CODE_MAP.get(fps_code)
    if entry is None or start_frames < 0:
        return ""
    _real_fps, nominal_fps, _label, is_drop = entry
    frames = start_frames
    if is_drop:
        ten_min, one_min = 17982, 1798
        q, rem = divmod(frames, ten_min)
        frames += 18 * q + (2 * ((rem - 2) // one_min) if rem > 1 else 0)
    hours = frames // (3600 * nominal_fps)
    minutes = (frames // (60 * nominal_fps)) % 60
    seconds = (frames // nominal_fps) % 60
    rem_frames = frames % nominal_fps
    sep = ";" if is_drop else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{rem_frames:02d}"


def _derive_ptx_xor_delta(xor_value: int) -> int:
    """Derive the deterministic 4 KB rolling-XOR delta for ``xor_type == 0x05``."""
    for i in range(256):
        if ((i * 11) & 0xFF) == xor_value:
            return (-i) & 0xFF
    return 0


def _estimate_mode_xor_delta(raw: bytes, nblocks: int) -> int:
    modes = [
        Counter(raw[b * 0x1000 : min(len(raw), (b + 1) * 0x1000)]).most_common(1)[0][0]
        for b in range(min(6, nblocks))
    ]
    deltas = [(modes[i + 1] - modes[i]) & 0xFF for i in range(len(modes) - 1)]
    return Counter(deltas).most_common(1)[0][0]


def _apply_block_xor(data: bytes, delta: int, k0: int) -> bytes:
    if len(data) <= 0x14:
        return bytes(data)
    parts: list[bytes] = [data[:0x14]]
    nblocks = (len(data) + 0xFFF) // 0x1000
    for block_idx in range(nblocks):
        key = (k0 + block_idx * delta) & 0xFF
        start = 0x14 if block_idx == 0 else block_idx * 0x1000
        end = min(len(data), (block_idx + 1) * 0x1000)
        chunk = data[start:end]
        parts.append(chunk if key == 0 else chunk.translate(_XOR_TABLES[key]))
    return b"".join(parts)


def _is_plausible_ptx_plaintext(buf: bytes) -> bool:
    if len(buf) <= 0x1F or buf[0x14] != 0x5A or buf[0x1F] != 0x5A:
        return False
    chunks = iter_top_chunks(buf)
    if not chunks:
        return False
    last = chunks[-1]
    return last.offset + 7 + last.size == len(buf)


def decrypt_ptx(raw: bytes) -> tuple[bytes, int]:
    """Decrypt a Pro Tools v10+ ``.ptx``/``.ptxt`` buffer and return ``(decrypted, xor_delta)``."""
    if len(raw) < 0x20:
        raise PtxValidationError(f"File too small to be a .ptx session ({len(raw)} bytes)")
    if not raw.startswith(PTX_MAGIC):
        raise PtxValidationError(f"Unrecognized .ptx magic header: {raw[:20]!r}")
    if raw[0x11] != 0:
        raise PtxUnsupportedTemplateError("Big-endian legacy Pro Tools sessions are not supported (.ptx v10+ only)")
    xor_type = raw[0x12]
    if xor_type == 0x01:
        raise PtxUnsupportedTemplateError("Legacy Pro Tools v5-v9 (.pts/.ptf) encryption is not supported (.ptx v10+ only)")

    k0 = raw[0x14] ^ 0x5A
    nblocks = (len(raw) + 0xFFF) // 0x1000
    if nblocks <= 1:
        return _apply_block_xor(raw, 0, k0), 0

    if xor_type == 0x05:
        alg_delta = _derive_ptx_xor_delta(raw[0x13])
        alg_dec = _apply_block_xor(raw, alg_delta, k0)
        if _is_plausible_ptx_plaintext(alg_dec):
            return alg_dec, alg_delta

    mode_delta = _estimate_mode_xor_delta(raw, nblocks)
    return _apply_block_xor(raw, mode_delta, k0), mode_delta


def encrypt_ptx(decrypted: bytes, delta: int, k0: int = 0) -> bytes:
    """Re-apply Pro Tools' 4 KB rolling block-XOR obfuscation."""
    return _apply_block_xor(decrypted, delta, k0)


def iter_top_chunks(buf: bytes, *, strict: bool = False) -> list[PtxChunk]:
    """Enumerate all top-level Z-chunks beginning at offset ``0x1F``."""
    chunks: list[PtxChunk] = []
    pos = 0x1F
    limit = len(buf)
    while pos < limit:
        if pos + 9 > limit or buf[pos] != 0x5A:
            if strict:
                raise PtxCorruptFileError(f"Invalid top-level Z-chunk header at offset 0x{pos:x}")
            break
        version = struct.unpack_from("<H", buf, pos + 1)[0]
        size = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        end = pos + 7 + size
        if size < 2 or end > limit:
            if strict:
                raise PtxCorruptFileError(f"Truncated Z-chunk 0x{tag:04x} at 0x{pos:x} (size={size})")
            break
        chunks.append(PtxChunk(offset=pos, version=version, size=size, tag=tag, raw=buf[pos:end]))
        pos = end
    return chunks


def _parse_chunk_node(buf: bytes, pos: int, max_end: int, depth: int) -> PtxChunk | None:
    if depth > 32 or pos + 9 > max_end or buf[pos] != 0x5A:
        return None
    version = struct.unpack_from("<H", buf, pos + 1)[0]
    if version >= 0x1000:
        return None
    size = struct.unpack_from("<I", buf, pos + 3)[0]
    end = pos + 7 + size
    if size < 2 or end > max_end:
        return None
    tag = struct.unpack_from("<H", buf, pos + 7)[0]
    children: list[PtxChunk] = []
    scan = pos + 9
    while scan < end:
        hit = buf.find(b"\x5A", scan, end)
        if hit == -1:
            break
        child = _parse_chunk_node(buf, hit, end, depth + 1)
        if child is not None:
            children.append(child)
            scan = child.offset + 7 + child.size
        else:
            scan = hit + 1
    return PtxChunk(offset=pos, version=version, size=size, tag=tag, raw=buf[pos:end], children=children)


def parse_chunk_tree(buf: bytes, start: int = 0x1F) -> list[PtxChunk]:
    """Recursively parse nested Z-chunks starting at *start* for inspection and debugging."""
    nodes: list[PtxChunk] = []
    pos = start
    limit = len(buf)
    while pos < limit:
        node = _parse_chunk_node(buf, pos, limit, 0)
        if node is None:
            break
        nodes.append(node)
        pos = node.offset + 7 + node.size
    return nodes


def normalize_ptx_uuid16(value: str | bytes) -> bytes:
    """Normalize raw 16-byte, hex, hyphenated, or PTSL braced UUID into 16 raw bytes."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 16:
            raise PtxValidationError(f"UUID bytes must be 16 bytes, got {len(value)}")
        return bytes(value)
    cleaned = str(value).strip().strip("{}").replace("-", "")
    if len(cleaned) != 32:
        raise PtxValidationError(f"Invalid PTX UUID string: {value!r}")
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise PtxValidationError(f"Invalid PTX UUID string: {value!r}") from exc


def find_zchunk_offset(
    buf: bytearray | bytes,
    tag: int,
    start: int = 0,
    end: int | None = None,
) -> int:
    """Return the first valid Z-chunk offset matching *tag* within ``buf[start:end]``, or ``-1``."""
    limit = len(buf) if end is None else min(len(buf), end)
    needle = struct.pack("<H", tag)
    pos = max(0, start) + 7
    while pos < limit:
        hit = buf.find(needle, pos, limit)
        if hit == -1:
            return -1
        off = hit - 7
        if off >= start and buf[off] == 0x5A:
            ver = struct.unpack_from("<H", buf, off + 1)[0]
            sz = struct.unpack_from("<I", buf, off + 3)[0]
            if ver < 0x1000 and 2 <= sz and off + 7 + sz <= limit:
                return off
        pos = hit + 1
    return -1


def read_zchunk_at(
    raw: bytes | bytearray,
    pos: int,
    *,
    expected_tag: int | None = None,
    limit: int | None = None,
) -> tuple[int, bytes, int]:
    """Read one validated Z-chunk at *pos* and return ``(tag, chunk_bytes, next_pos)``."""
    end_lim = len(raw) if limit is None else min(len(raw), limit)
    if pos < 0 or pos + 9 > end_lim or raw[pos] != 0x5A:
        raise PtxCorruptFileError(f"Expected Z-chunk header at offset 0x{pos:x}")
    sz = struct.unpack_from("<I", raw, pos + 3)[0]
    tag = struct.unpack_from("<H", raw, pos + 7)[0]
    end = pos + 7 + sz
    if sz < 2 or end > end_lim:
        raise PtxCorruptFileError(f"Truncated Z-chunk 0x{tag:04x} at offset 0x{pos:x}")
    if expected_tag is not None and tag != expected_tag:
        raise PtxCorruptFileError(f"Expected Z-chunk 0x{expected_tag:04x} at 0x{pos:x}, found 0x{tag:04x}")
    return tag, bytes(raw[pos:end]), end


def walk_sequential_zchunks(
    raw: bytes | bytearray,
    start: int,
    count: int,
    *,
    expected_tag: int | None = None,
    limit: int | None = None,
) -> tuple[list[bytes], int]:
    """Deterministically walk *count* contiguous Z-chunks from *start*."""
    items: list[bytes] = []
    pos = start
    for _ in range(count):
        _tag, chunk_raw, pos = read_zchunk_at(raw, pos, expected_tag=expected_tag, limit=limit)
        items.append(chunk_raw)
    return items, pos


def find_subchunk_spans(
    buf: bytes | bytearray,
    tag16: int,
    start: int = 9,
    end: int | None = None,
) -> list[tuple[int, bytes]]:
    """Return ``(offset, raw_bytes)`` for all valid Z-subchunks with *tag16* inside ``buf[start:end]``."""
    results: list[tuple[int, bytes]] = []
    needle = struct.pack("<H", tag16)
    min_start = max(0, start)
    pos = min_start + 7
    limit = len(buf) if end is None else min(len(buf), end)
    while pos < limit:
        hit = buf.find(needle, pos, limit)
        if hit == -1:
            break
        cstart = hit - 7
        if cstart >= min_start and buf[cstart] == 0x5A:
            version = struct.unpack_from("<H", buf, cstart + 1)[0]
            size = struct.unpack_from("<I", buf, cstart + 3)[0]
            if 2 <= size <= limit - cstart - 7 and version < 0x1000:
                results.append((cstart, bytes(buf[cstart : cstart + 7 + size])))
                pos = cstart + 7 + size + 7
                continue
        pos = hit + 1
    return results


def find_subchunks(buf: bytes, tag16: int, start: int = 9) -> list[bytes]:
    """Return all valid Z-subchunks with ``tag16`` inside *buf* starting at *start*."""
    return [raw for _, raw in find_subchunk_spans(buf, tag16, start)]


def resize_zchunk_ancestors(buf: bytearray, ancestor_offsets: list[int] | tuple[int, ...], delta: int) -> None:
    """Adjust the ``<I`` size headers of enclosing Z-chunks at *ancestor_offsets* by *delta*."""
    if delta == 0:
        return
    for off in ancestor_offsets:
        if off < 0 or off + 7 > len(buf) or buf[off] != 0x5A:
            raise PtxCorruptFileError(f"Invalid ancestor Z-chunk at offset 0x{off:x}")
        new_sz = struct.unpack_from("<I", buf, off + 3)[0] + delta
        if not (2 <= new_sz <= 0xFFFF_FFFF):
            raise PtxCorruptFileError(f"Ancestor Z-chunk size out of u32 bounds at 0x{off:x}: {new_sz}")
        struct.pack_into("<I", buf, off + 3, new_sz)


def read_lp_str(buf: bytes, offset: int, max_len: int = 512) -> tuple[str, int]:
    """Read a length-prefixed UTF-8 string (``<I`` length + bytes) at *offset*."""
    if offset < 0 or offset + 4 > len(buf):
        return "", max(0, offset)
    length = struct.unpack_from("<I", buf, offset)[0]
    if length == 0 or length > max_len or offset + 4 + length > len(buf):
        return "", offset + 4
    text = buf[offset + 4 : offset + 4 + length].decode("utf-8", errors="replace")
    return text, offset + 4 + length


def _decode_control_value(buf: bytes, pos: int, vsize: int, *, signed_int32: bool) -> float:
    if vsize == 2:
        return float(struct.unpack_from("<h", buf, pos)[0])
    if vsize == 4:
        if signed_int32:
            return float(struct.unpack_from("<i", buf, pos)[0])
        return float(struct.unpack_from("<f", buf, pos)[0])
    if vsize == 8:
        return float(struct.unpack_from("<Q", buf, pos)[0])
    raise PtxCorruptFileError(f"Unsupported control-curve value width {vsize}")


def parse_control_curve(
    buf: bytes,
    offset: int = 0,
    *,
    signed_int32: bool = False,
) -> tuple[str, list[PtxAutomationPoint], int] | None:
    """Decode a Pro Tools ``0x46`` (``'F'``) control curve at *offset*.

    Returns ``(timebase, points, next_offset)`` or ``None`` if no valid curve starts at *offset*.
    """
    if offset < 0 or offset + 8 > len(buf) or buf[offset + 1] != 0x46:
        return None
    version = struct.unpack_from("<H", buf, offset + 2)[0]
    payload_len = struct.unpack_from("<I", buf, offset + 4)[0]
    end = offset + 8 + payload_len
    if version != 1 or payload_len < 18 or end > len(buf):
        return None
    q = offset + 8
    tb_code = struct.unpack_from("<H", buf, q)[0]
    vsize = struct.unpack_from("<H", buf, q + 6)[0]
    extra_count = struct.unpack_from("<H", buf, q + 8)[0]
    if vsize not in (2, 4, 8) or q + 18 + vsize > end:
        return None
    timebase = "Ticks" if tb_code == 1 else "Samples"
    pos0_raw = struct.unpack_from("<Q", buf, q + 10)[0]
    pos0 = decode_ticks_relative(pos0_raw) if tb_code == 1 else pos0_raw
    val0 = _decode_control_value(buf, q + 18, vsize, signed_int32=signed_int32)
    points = [PtxAutomationPoint(position=pos0, value=val0)]
    stride = 4 + vsize
    cursor = q + 18 + vsize
    max_extra = (end - cursor) // stride
    for _ in range(min(extra_count, max_extra)):
        delta_pos = struct.unpack_from("<I", buf, cursor)[0]
        val = _decode_control_value(buf, cursor + 4, vsize, signed_int32=signed_int32)
        points.append(PtxAutomationPoint(position=pos0 + delta_pos, value=val))
        cursor += stride
    return timebase, points, end


def _validate_index_entry(buf: bytes, pos: int, body_end: int) -> int:
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 chunk offset index")
    depth = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if depth > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 path depth {depth} exceeds chunk bounds")
    for _ in range(depth):
        poff = struct.unpack_from("<i", buf, pos + 2)[0]
        if poff != -1 and not (0 <= poff < len(buf) and buf[poff] == 0x5A):
            raise PtxCorruptFileError(f"0x0002 path offset 0x{poff:x} does not point to a Z-chunk")
        pos += 11
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 occurrence count")
    occ_cnt = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if occ_cnt > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 occurrence count {occ_cnt} exceeds chunk bounds")
    for _ in range(occ_cnt):
        if pos + 5 > body_end:
            raise PtxCorruptFileError("Truncated 0x0002 occurrence header")
        _b0, byte_len, off_cnt = struct.unpack_from("<BHH", buf, pos)
        pos += 5
        if byte_len != 4 * off_cnt or pos + byte_len + 6 > body_end:
            raise PtxCorruptFileError(f"Invalid 0x0002 occurrence bounds (byte_len={byte_len}, count={off_cnt})")
        for j in range(off_cnt):
            off = struct.unpack_from("<I", buf, pos + 4 * j)[0]
            if not (0 <= off < len(buf) and buf[off] == 0x5A):
                raise PtxCorruptFileError(f"0x0002 chunk offset 0x{off:x} does not point to a Z-chunk")
        pos += byte_len + 6
    return pos


def _shift_index_entry(buf: bytearray, pos: int, body_end: int, threshold: int, delta: int) -> int:
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 chunk offset index")
    depth = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if depth > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 path depth {depth} exceeds chunk bounds")
    for _ in range(depth):
        poff = struct.unpack_from("<i", buf, pos + 2)[0]
        if poff != -1:
            if poff >= threshold:
                poff += delta
            if not (0 <= poff < len(buf) and buf[poff] == 0x5A):
                raise PtxCorruptFileError(f"0x0002 path offset 0x{poff:x} does not point to a Z-chunk")
            struct.pack_into("<i", buf, pos + 2, poff)
        pos += 11
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 occurrence count")
    occ_cnt = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if occ_cnt > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 occurrence count {occ_cnt} exceeds chunk bounds")
    for _ in range(occ_cnt):
        if pos + 5 > body_end:
            raise PtxCorruptFileError("Truncated 0x0002 occurrence header")
        _b0, byte_len, off_cnt = struct.unpack_from("<BHH", buf, pos)
        pos += 5
        if byte_len != 4 * off_cnt or pos + byte_len + 6 > body_end:
            raise PtxCorruptFileError(f"Invalid 0x0002 occurrence bounds (byte_len={byte_len}, count={off_cnt})")
        for j in range(off_cnt):
            off_pos = pos + 4 * j
            off = struct.unpack_from("<I", buf, off_pos)[0]
            if off >= threshold:
                off += delta
            if not (0 <= off < len(buf) and buf[off] == 0x5A):
                raise PtxCorruptFileError(f"0x0002 chunk offset 0x{off:x} does not point to a Z-chunk")
            struct.pack_into("<I", buf, off_pos, off)
        pos += byte_len + 6
    return pos


def validate_index_0002(buf: bytes) -> bool:
    """Validate that ``0x0002`` (if referenced by ``0x1B``) exists and every offset hits ``0x5A``."""
    chunks = iter_top_chunks(buf, strict=True)
    c2 = next((c for c in reversed(chunks) if c.tag == 0x0002), None)
    if c2 is None:
        return False
    if len(buf) >= 0x1F:
        ptr = struct.unpack_from("<I", buf, 0x1B)[0]
        if ptr != c2.offset:
            raise PtxCorruptFileError(f"Root 0x1B pointer 0x{ptr:x} does not match 0x0002 at 0x{c2.offset:x}")
    body_end = c2.offset + 7 + c2.size
    pos = c2.offset + 9
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 body")
    entry_count = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    for _ in range(entry_count):
        pos = _validate_index_entry(buf, pos, body_end)
    if pos != body_end:
        raise PtxCorruptFileError(f"0x0002 parse ended at {pos}, expected {body_end}")
    return True


def shift_index_0002(buf: bytearray, threshold: int, delta: int) -> None:
    """Shift all ``>= threshold`` file offsets in ``0x0002`` by *delta* and update ``0x1B``."""
    if delta == 0:
        return
    chunks = iter_top_chunks(bytes(buf), strict=True)
    c2 = next((c for c in reversed(chunks) if c.tag == 0x0002), None)
    if c2 is None:
        if len(buf) >= 0x1F and struct.unpack_from("<I", buf, 0x1B)[0] != 0:
            raise PtxCorruptFileError("Header 0x1B referenced 0x0002 index, but 0x0002 chunk is missing")
        return
    if len(buf) >= 0x1F:
        struct.pack_into("<I", buf, 0x1B, c2.offset)
    body_end = c2.offset + 7 + c2.size
    pos = c2.offset + 9
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 body")
    entry_count = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    for _ in range(entry_count):
        pos = _shift_index_entry(buf, pos, body_end, threshold, delta)
    if pos != body_end:
        raise PtxCorruptFileError(f"0x0002 parse ended at {pos}, expected {body_end}")


def splice_top_chunk(buf: bytearray, tag: int, new_chunk_bytes: bytes) -> int:
    """Replace top-level chunk *tag* with *new_chunk_bytes* and shift ``0x0002`` offsets."""
    chunks = iter_top_chunks(bytes(buf), strict=True)
    target = next((c for c in reversed(chunks) if c.tag == tag), None)
    if target is None:
        raise PtxCorruptFileError(f"Top-level chunk 0x{tag:04x} not found in .ptx")
    old_end = target.offset + 7 + target.size
    delta = len(new_chunk_bytes) - (old_end - target.offset)
    buf[target.offset : old_end] = new_chunk_bytes
    shift_index_0002(buf, old_end, delta)
    return delta


def atomic_write_bytes(target_path: str | Path, payload: bytes) -> None:
    """Write *payload* to *target_path* atomically via same-directory temp file + fsync + replace."""
    target = Path(target_path)
    fd, tmp_name = tempfile.mkstemp(prefix=".sp_ptx_", suffix=".tmp", dir=str(target.parent))
    fd_owned = True
    handle = None
    try:
        handle = os.fdopen(fd, "wb")
        fd_owned = False
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(tmp_name, target)
    except BaseException:
        if handle is not None:
            handle.close()
        elif fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise
