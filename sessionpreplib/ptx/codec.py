"""Binary codec, Z-chunk traversal, and ``0x0002`` offset-index maintenance for ``.ptx``."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import struct
import tempfile

from .models import PTX_MAGIC, SLOT_LETTERS_LOWER, PtxChunk


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


def _apply_block_xor(data: bytes, delta: int, k0: int) -> bytes:
    out = bytearray(data)
    nblocks = (len(out) + 0xFFF) // 0x1000
    for block_idx in range(nblocks):
        key = (k0 + block_idx * delta) & 0xFF
        if key == 0:
            continue
        start = 0x14 if block_idx == 0 else block_idx * 0x1000
        end = min(len(out), (block_idx + 1) * 0x1000)
        for i in range(start, end):
            out[i] ^= key
    return bytes(out)


def decrypt_ptx(raw: bytes) -> tuple[bytes, int]:
    """Decrypt a Pro Tools v10+ ``.ptx``/``.ptxt`` buffer and return ``(decrypted, xor_delta)``."""
    if len(raw) < 0x20:
        raise ValueError(f"File too small to be a .ptx session ({len(raw)} bytes)")
    if not raw.startswith(PTX_MAGIC):
        raise ValueError(f"Unrecognized .ptx magic header: {raw[:20]!r}")

    k0 = raw[0x14] ^ 0x5A
    nblocks = (len(raw) + 0xFFF) // 0x1000
    if nblocks <= 1:
        return _apply_block_xor(raw, 0, k0), 0

    modes = [
        Counter(raw[b * 0x1000 : min(len(raw), (b + 1) * 0x1000)]).most_common(1)[0][0]
        for b in range(min(6, nblocks))
    ]
    deltas = [(modes[i + 1] - modes[i]) & 0xFF for i in range(len(modes) - 1)]
    delta = Counter(deltas).most_common(1)[0][0]
    return _apply_block_xor(raw, delta, k0), delta


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
                raise RuntimeError(f"Invalid top-level Z-chunk header at offset 0x{pos:x}")
            break
        version = struct.unpack_from("<H", buf, pos + 1)[0]
        size = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        end = pos + 7 + size
        if size < 2 or end > limit:
            if strict:
                raise RuntimeError(f"Truncated Z-chunk 0x{tag:04x} at 0x{pos:x} (size={size})")
            break
        chunks.append(PtxChunk(offset=pos, version=version, size=size, tag=tag, raw=buf[pos:end]))
        pos = end
    return chunks


def find_subchunks(buf: bytes, tag16: int, start: int = 9) -> list[bytes]:
    """Return all valid Z-subchunks with ``tag16`` inside *buf* starting at *start*."""
    results: list[bytes] = []
    needle = struct.pack("<H", tag16)
    pos = start
    limit = len(buf)
    while True:
        hit = buf.find(needle, pos)
        if hit == -1:
            break
        if hit >= 7 and buf[hit - 7] == 0x5A:
            cstart = hit - 7
            version = struct.unpack_from("<H", buf, cstart + 1)[0]
            size = struct.unpack_from("<I", buf, cstart + 3)[0]
            if 2 <= size <= limit - cstart - 7 and version < 0x1000:
                results.append(buf[cstart : cstart + 7 + size])
                pos = cstart + 7 + size
                continue
        pos = hit + 1
    return results


def read_lp_str(buf: bytes, offset: int, max_len: int = 512) -> tuple[str, int]:
    """Read a length-prefixed UTF-8 string (``<I`` length + bytes) at *offset*."""
    if offset + 4 > len(buf):
        return "", offset
    length = struct.unpack_from("<I", buf, offset)[0]
    if length == 0 or length > max_len or offset + 4 + length > len(buf):
        return "", offset + 4
    text = buf[offset + 4 : offset + 4 + length].decode("utf-8", errors="replace")
    return text, offset + 4 + length


def _shift_index_entry(buf: bytearray, pos: int, body_end: int, threshold: int, delta: int) -> int:
    if pos + 4 > body_end:
        raise RuntimeError("Truncated 0x0002 chunk offset index")
    depth = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if depth > (body_end - pos) // 11:
        raise RuntimeError(f"0x0002 path depth {depth} exceeds chunk bounds")
    for _ in range(depth):
        poff = struct.unpack_from("<i", buf, pos + 2)[0]
        if poff != -1:
            if poff >= threshold:
                poff += delta
            if not (0 <= poff < len(buf) and buf[poff] == 0x5A):
                raise RuntimeError(f"0x0002 path offset 0x{poff:x} does not point to a Z-chunk")
            struct.pack_into("<i", buf, pos + 2, poff)
        pos += 11
    if pos + 4 > body_end:
        raise RuntimeError("Truncated 0x0002 occurrence count")
    occ_cnt = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if occ_cnt > (body_end - pos) // 11:
        raise RuntimeError(f"0x0002 occurrence count {occ_cnt} exceeds chunk bounds")
    for _ in range(occ_cnt):
        if pos + 5 > body_end:
            raise RuntimeError("Truncated 0x0002 occurrence header")
        _b0, byte_len, off_cnt = struct.unpack_from("<BHH", buf, pos)
        pos += 5
        if byte_len != 4 * off_cnt or pos + byte_len + 6 > body_end:
            raise RuntimeError(f"Invalid 0x0002 occurrence bounds (byte_len={byte_len}, count={off_cnt})")
        for j in range(off_cnt):
            off_pos = pos + 4 * j
            off = struct.unpack_from("<I", buf, off_pos)[0]
            if off >= threshold:
                off += delta
            if not (0 <= off < len(buf) and buf[off] == 0x5A):
                raise RuntimeError(f"0x0002 chunk offset 0x{off:x} does not point to a Z-chunk")
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
            raise RuntimeError(f"Root 0x1B pointer 0x{ptr:x} does not match 0x0002 at 0x{c2.offset:x}")
    # Dry-run bounds & Z-chunk check using shift helper on a copy with delta=0
    body_end = c2.offset + 7 + c2.size
    pos = c2.offset + 9
    if pos + 4 > body_end:
        raise RuntimeError("Truncated 0x0002 body")
    entry_count = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    tmp = bytearray(buf)
    for _ in range(entry_count):
        pos = _shift_index_entry(tmp, pos, body_end, len(buf) + 1, 0)
    if pos != body_end:
        raise RuntimeError(f"0x0002 parse ended at {pos}, expected {body_end}")
    return True


def shift_index_0002(buf: bytearray, threshold: int, delta: int) -> None:
    """Shift all ``>= threshold`` file offsets in ``0x0002`` by *delta* and update ``0x1B``."""
    if delta == 0:
        return
    chunks = iter_top_chunks(bytes(buf), strict=True)
    c2 = next((c for c in reversed(chunks) if c.tag == 0x0002), None)
    if c2 is None:
        if len(buf) >= 0x1F and struct.unpack_from("<I", buf, 0x1B)[0] != 0:
            raise RuntimeError("Header 0x1B referenced 0x0002 index, but 0x0002 chunk is missing")
        return
    if len(buf) >= 0x1F:
        struct.pack_into("<I", buf, 0x1B, c2.offset)
    body_end = c2.offset + 7 + c2.size
    pos = c2.offset + 9
    if pos + 4 > body_end:
        raise RuntimeError("Truncated 0x0002 body")
    entry_count = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    for _ in range(entry_count):
        pos = _shift_index_entry(buf, pos, body_end, threshold, delta)
    if pos != body_end:
        raise RuntimeError(f"0x0002 parse ended at {pos}, expected {body_end}")


def splice_top_chunk(buf: bytearray, tag: int, new_chunk_bytes: bytes) -> int:
    """Replace top-level chunk *tag* with *new_chunk_bytes* and shift ``0x0002`` offsets."""
    chunks = iter_top_chunks(bytes(buf), strict=True)
    target = next((c for c in reversed(chunks) if c.tag == tag), None)
    if target is None:
        raise RuntimeError(f"Top-level chunk 0x{tag:04x} not found in .ptx")
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
    except Exception:
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
