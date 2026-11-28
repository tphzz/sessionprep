"""Shared decrypted ``.ptx`` working buffer and deduplicated binary traversal primitives."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import struct

from .codec import (
    decrypt_ptx,
    encrypt_ptx,
    iter_top_chunks,
    parse_control_curve,
    validate_index_0002,
)
from .models import UUID_PREFIX, PtxChunk, PtxCorruptFileError, PtxValidationError
from .parser import PtxSession


@dataclass(frozen=True)
class MixerStripSpan:
    """Bounds and identity of one mixer strip inside ``0x2624``."""

    index: int
    tag: int
    offset: int
    end: int
    uuid16: bytes | None


def iter_mixer_strips(buf: bytes | bytearray, c2624_offset: int) -> Iterator[MixerStripSpan]:
    """Yield every validated Z-chunk mixer strip inside ``0x2624`` starting at *c2624_offset*."""
    if c2624_offset < 0 or c2624_offset + 13 > len(buf) or buf[c2624_offset] != 0x5A:
        raise PtxCorruptFileError(f"Missing or truncated 0x2624 mixer chunk at 0x{c2624_offset:x}")
    sz_all = struct.unpack_from("<I", buf, c2624_offset + 3)[0]
    chunk_end = min(len(buf), c2624_offset + 7 + sz_all)
    tcount = struct.unpack_from("<I", buf, c2624_offset + 9)[0]
    pos = c2624_offset + 13
    for idx in range(tcount):
        if pos + 9 > chunk_end or buf[pos] != 0x5A:
            break
        sz = struct.unpack_from("<I", buf, pos + 3)[0]
        tag = struct.unpack_from("<H", buf, pos + 7)[0]
        blk_end = pos + 7 + sz
        if sz < 2 or blk_end > chunk_end:
            break
        head_end = min(blk_end, pos + 256)
        uoff = buf.find(UUID_PREFIX, pos, head_end)
        u16 = bytes(buf[uoff : uoff + 16]) if (uoff != -1 and uoff + 16 <= blk_end) else None
        yield MixerStripSpan(index=idx, tag=tag, offset=pos, end=blk_end, uuid16=u16)
        pos = blk_end


def patch_static_0x260a_curve(
    buf: bytearray,
    curve_abs_off: int,
    curve_bytes: bytes,
    value_i16: int,
    *,
    param_label: str,
) -> int:
    """Patch a single-point ``i16`` ``0x260A`` control curve in-place and return its previous raw i16 value.

    Refuses to corrupt multi-point automation curves or non-2-byte control curves.
    """
    parsed = parse_control_curve(curve_bytes, 9)
    if parsed is None:
        raise PtxCorruptFileError(f"{param_label}: unreadable 0x260A control curve")
    _tb, pts, _end = parsed
    if len(pts) != 1:
        raise PtxValidationError(
            f"{param_label}: refuses to overwrite multi-point automation ({len(pts)} breakpoints)"
        )
    vsize = struct.unpack_from("<H", curve_bytes, 23)[0]
    if vsize != 2 or len(curve_bytes) < 37:
        raise PtxCorruptFileError(f"{param_label}: unsupported 0x260A value width ({vsize})")
    val_off = curve_abs_off + 35
    prev_raw = struct.unpack_from("<h", buf, val_off)[0]
    struct.pack_into("<h", buf, val_off, value_i16)
    return int(prev_raw)


def assemble_top_chunks(
    dec_base: bytes,
    replacements: Mapping[int, bytes],
    *,
    insertions: Sequence[tuple[int, bytes, tuple[int, ...]]] = (),
) -> bytearray:
    """Rebuild top-level decrypted ``.ptx`` bytes with chunk replacements and canonical insertions.

    Each tuple in *insertions* is ``(tag, chunk_raw, after_tags)``; if *tag* is not already present
    in *dec_base*, *chunk_raw* is inserted immediately after the last present tag in *after_tags*,
    or directly before ``0x0002`` if none of *after_tags* exist.
    """
    chunks = iter_top_chunks(dec_base, strict=True)
    last_idx_by_tag: dict[int, int] = {ch.tag: idx for idx, ch in enumerate(chunks)}
    after_buckets: dict[int, list[bytes]] = {}
    before_0002: list[bytes] = []
    for tag, raw, after_tags in insertions:
        if tag in last_idx_by_tag:
            continue
        anchor_idx = next((last_idx_by_tag[a] for a in reversed(after_tags) if a in last_idx_by_tag), None)
        if anchor_idx is not None:
            after_buckets.setdefault(anchor_idx, []).append(raw)
        else:
            before_0002.append(raw)

    parts: list[bytes] = [dec_base[:0x1F]]
    inserted_tail = False
    for idx, ch in enumerate(chunks):
        if ch.tag == 0x0002 and before_0002 and not inserted_tail:
            parts.extend(before_0002)
            inserted_tail = True
        if ch.tag in replacements and idx == last_idx_by_tag[ch.tag]:
            parts.append(replacements[ch.tag])
        else:
            parts.append(ch.raw)
        if idx in after_buckets:
            parts.extend(after_buckets.pop(idx))
    if before_0002 and not inserted_tail:
        parts.extend(before_0002)
    return bytearray(b"".join(parts))


def validate_decrypted_ptx_pair(
    orig_dec: bytes,
    orig_delta: int,
    new_dec: bytes,
    new_delta: int,
    *,
    allowed_added_tags: frozenset[int] = frozenset({0x2630, 0x2030, 0x2637}),
    orig_session: PtxSession | None = None,
) -> PtxSession:
    """Strictly verify structural integrity, ``0x0002`` validity, track count, and parser warnings."""
    orig_chunks: list[PtxChunk] = iter_top_chunks(orig_dec, strict=True)
    new_chunks: list[PtxChunk] = iter_top_chunks(new_dec, strict=True)
    orig_tags = {c.tag for c in orig_chunks}
    new_tags = {c.tag for c in new_chunks}
    missing_tags = orig_tags - new_tags
    extra_tags = new_tags - orig_tags - allowed_added_tags
    if missing_tags or extra_tags:
        raise PtxCorruptFileError(
            f"Patched .ptx top-level chunk tags changed: missing={missing_tags}, extra={extra_tags}"
        )
    if any(c.tag == 0x0002 for c in orig_chunks) and not validate_index_0002(new_dec):
        raise PtxCorruptFileError("Patched .ptx lost valid 0x0002 chunk offset index")
    base_sess = orig_session if orig_session is not None else PtxSession(orig_dec, xor_delta=orig_delta)
    new_sess = PtxSession(new_dec, xor_delta=new_delta)
    if len(base_sess.tracks) != len(new_sess.tracks):
        raise PtxCorruptFileError(
            f"Patched .ptx track count changed ({len(base_sess.tracks)} -> {len(new_sess.tracks)})"
        )
    base_counts = Counter(base_sess.warnings)
    new_counts = Counter(new_sess.warnings)
    extra_warnings = list((new_counts - base_counts).elements())
    if extra_warnings:
        raise PtxCorruptFileError(f"Patched .ptx introduced new parser warnings: {extra_warnings}")
    return new_sess


class PtxBuffer:
    """Mutable decrypted ``.ptx`` buffer with lazy cached ``PtxSession`` for single-pass patching."""

    def __init__(self, data: bytearray, xor_delta: int, k0: int, *, session: PtxSession | None = None) -> None:
        self.data = data
        self.xor_delta = xor_delta
        self.k0 = k0
        self._session: PtxSession | None = session

    @classmethod
    def from_encrypted(cls, raw_ptx: bytes) -> PtxBuffer:
        """Decrypt *raw_ptx* once into a mutable ``PtxBuffer``."""
        dec, delta = decrypt_ptx(raw_ptx)
        k0 = raw_ptx[0x14] ^ 0x5A
        return cls(bytearray(dec), delta, k0)

    def to_encrypted(self) -> bytes:
        """Re-encrypt the current decrypted buffer preserving original ``xor_delta`` and ``k0``."""
        return encrypt_ptx(bytes(self.data), self.xor_delta, k0=self.k0)

    @property
    def session(self) -> PtxSession:
        """Return a lazily parsed ``PtxSession`` view of the current decrypted buffer."""
        if self._session is None:
            self._session = PtxSession(bytes(self.data), xor_delta=self.xor_delta)
        return self._session

    def invalidate(self) -> None:
        """Invalidate the cached ``PtxSession`` after mutating ``self.data``."""
        self._session = None

    def replace_data(self, new_data: bytearray) -> None:
        """Replace ``self.data`` with *new_data* and invalidate the cached session."""
        self.data = new_data
        self._session = None
