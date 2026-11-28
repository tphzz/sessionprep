"""Low-level binary serializers and ``0x0002`` offset-index roundtrip codec for Pro Tools ``.ptx``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import secrets
import struct

from .codec import iter_top_chunks
from .models import UUID_PREFIX, ZERO_TICKS, PtxAutomationPoint, PtxCorruptFileError, PtxValidationError


@dataclass(frozen=True)
class PtxIndexPathStep:
    """One 11-byte path step inside a ``0x0002`` chunk-offset index entry."""

    tag: int
    offset: int  # Signed i32 file offset, or -1 for wildcard/container step
    meta: bytes = b"\x00" * 5  # 5 trailing bytes preserved verbatim


@dataclass(frozen=True)
class PtxIndexOccurrence:
    """One occurrence group inside a ``0x0002`` index entry."""

    flag: int
    offsets: tuple[int, ...]
    tail: bytes = b"\x00" * 6  # 6 trailing bytes preserved verbatim


@dataclass
class PtxIndexEntry:
    """One decoded entry in the trailing ``0x0002`` chunk-offset index."""

    path: list[PtxIndexPathStep] = field(default_factory=list)
    occurrences: list[PtxIndexOccurrence] = field(default_factory=list)


@dataclass
class PtxIndex0002:
    """Structured representation of a ``.ptx`` file's trailing ``0x0002`` chunk-offset index."""

    version: int = 2
    entries: list[PtxIndexEntry] = field(default_factory=list)


def build_zchunk(tag: int, body: bytes, *, version: int = 1) -> bytes:
    """Wrap *body* in a Pro Tools Z-chunk header (``0x5A``, version, size, tag)."""
    if isinstance(version, bool) or not (0 <= version <= 0xFFFF):
        raise PtxValidationError(f"Z-chunk version out of u16 range: {version}")
    if isinstance(tag, bool) or not (0 <= tag <= 0xFFFF):
        raise PtxValidationError(f"Z-chunk tag out of u16 range: {tag}")
    if 2 + len(body) > 0xFFFF_FFFF:
        raise PtxValidationError(f"Z-chunk body size exceeds u32 range: {len(body)}")
    return b"Z" + struct.pack("<HIH", version, 2 + len(body), tag) + body


def encode_lp_str(text: str) -> bytes:
    """Encode *text* as a Pro Tools ``<u32`` length-prefixed UTF-8 string."""
    raw = text.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def generate_ptx_uuid() -> bytes:
    """Generate a fresh 16-byte Pro Tools object UUID (``UUID_PREFIX`` + 8 random bytes)."""
    return UUID_PREFIX + secrets.token_bytes(8)


def encode_packed_uint(value: int) -> tuple[int, bytes]:
    """Encode a non-negative integer into the minimal 1/2/4/8-byte LE representation."""
    if isinstance(value, bool) or value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        raise PtxValidationError(f"Packed uint out of u64 range: {value}")
    if value == 0:
        return 0, b""
    if value <= 0xFF:
        return 1, struct.pack("<B", value)
    if value <= 0xFFFF:
        return 2, struct.pack("<H", value)
    if value <= 0xFFFF_FFFF:
        return 4, struct.pack("<I", value)
    return 8, struct.pack("<Q", value)


def _pack_control_value(value: float, vsize: int, *, signed_int32: bool) -> bytes:
    if isinstance(value, bool):
        raise PtxValidationError(f"Control-curve value must be numeric, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise PtxValidationError(f"Control-curve value must be finite, got {value!r}")
    if vsize == 2:
        iv = max(-32768, min(32767, round(fval)))
        return struct.pack("<h", iv)
    if vsize == 4:
        if signed_int32:
            iv = max(-2147483648, min(2147483647, round(fval)))
            return struct.pack("<i", iv)
        try:
            return struct.pack("<f", fval)
        except OverflowError as exc:
            raise PtxValidationError(f"Control-curve float32 value out of range: {value!r}") from exc
    if vsize == 8:
        iv = round(fval)
        if not (0 <= iv <= 0xFFFF_FFFF_FFFF_FFFF):
            raise PtxValidationError(f"Control-curve u64 value out of range: {value!r}")
        return struct.pack("<Q", iv)
    raise PtxValidationError(f"Unsupported control-curve value width {vsize}")


def encode_control_curve(
    timebase: str,
    points: Sequence[PtxAutomationPoint],
    vsize: int,
    *,
    flags: int = 1,
    signed_int32: bool = False,
) -> bytes:
    """Serialize breakpoints into a Pro Tools ``0x46`` (``'F'``) control curve."""
    if timebase not in ("Samples", "Ticks"):
        raise PtxValidationError(f"Unsupported control-curve timebase {timebase!r} (expected 'Samples' or 'Ticks')")
    if isinstance(flags, bool) or not (0 <= flags <= 0xFF):
        raise PtxValidationError(f"Control-curve flags out of u8 range: {flags}")
    if not points:
        raise PtxValidationError("Control curve requires at least one breakpoint")
    if vsize not in (2, 4, 8):
        raise PtxValidationError(f"Unsupported control-curve value width {vsize}")
    pts = sorted(points, key=lambda p: p.position)
    base_pos = int(pts[0].position)
    if base_pos < 0:
        raise PtxValidationError(f"Control-curve breakpoint position must be non-negative, got {base_pos}")
    tb_code = 1 if timebase == "Ticks" else 0
    pos0 = (ZERO_TICKS + base_pos) if tb_code == 1 else base_pos
    if pos0 > 0xFFFF_FFFF_FFFF_FFFF:
        raise PtxValidationError(f"Control-curve base position exceeds u64 range: {pos0}")
    extra_count = len(pts) - 1
    if extra_count > 0xFFFF:
        raise PtxValidationError(f"Control curve exceeds u16 extra-point limit: {extra_count}")

    payload = bytearray(struct.pack("<HIHHQ", tb_code, 1, vsize, extra_count, pos0))
    payload.extend(_pack_control_value(pts[0].value, vsize, signed_int32=signed_int32))
    for pt in pts[1:]:
        delta_pos = int(pt.position) - base_pos
        if not (0 <= delta_pos <= 0xFFFF_FFFF):
            raise PtxValidationError(f"Control-curve delta position out of u32 range: {delta_pos}")
        payload.extend(struct.pack("<I", delta_pos))
        payload.extend(_pack_control_value(pt.value, vsize, signed_int32=signed_int32))
    return struct.pack("<BBHI", flags, 0x46, 1, len(payload)) + bytes(payload)


def serialize_clip_gain_2637(
    curves: Sequence[tuple[str, Sequence[PtxAutomationPoint], int]],
    *,
    version: int = 1,
    trailer: bytes = b"",
) -> bytes:
    """Build a top-level ``0x2637`` clip-gain curves chunk from ``(timebase, points, flags)`` tuples."""
    encoded = [
        encode_control_curve(tb, pts, 4, flags=flg, signed_int32=False)
        for tb, pts, flg in curves
    ]
    body = struct.pack("<I", len(encoded)) + b"".join(encoded) + trailer
    return build_zchunk(0x2637, body, version=version)


def serialize_folder_tree_210c(
    folders: Sequence[tuple[bytes, Sequence[bytes]]],
    *,
    version: int = 1,
) -> bytes:
    """Build a top-level ``0x210c`` folder hierarchy chunk from ``(folder_uuid16, [child_uuid16, ...])``."""
    body = bytearray(struct.pack("<I", len(folders)))
    for folder_uuid, children in folders:
        if len(folder_uuid) != 16:
            raise PtxValidationError(f"Folder UUID must be 16 bytes, got {len(folder_uuid)}")
        body.append(0)
        body.extend(folder_uuid)
        body.extend(struct.pack("<I", len(children)))
        for child_uuid in children:
            if len(child_uuid) != 16:
                raise PtxValidationError(f"Child UUID must be 16 bytes, got {len(child_uuid)}")
            body.append(0)
            body.extend(child_uuid)
    return build_zchunk(0x210C, bytes(body), version=version)


def _parse_index_entry(buf: bytes, pos: int, body_end: int) -> tuple[PtxIndexEntry, int]:
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 entry path depth")
    depth = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if depth > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 path depth {depth} exceeds chunk bounds")
    steps: list[PtxIndexPathStep] = []
    for _ in range(depth):
        tag = struct.unpack_from("<H", buf, pos)[0]
        off = struct.unpack_from("<i", buf, pos + 2)[0]
        meta = buf[pos + 6 : pos + 11]
        steps.append(PtxIndexPathStep(tag=tag, offset=off, meta=meta))
        pos += 11
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 occurrence count")
    occ_cnt = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    if occ_cnt > (body_end - pos) // 11:
        raise PtxCorruptFileError(f"0x0002 occurrence count {occ_cnt} exceeds chunk bounds")
    occs: list[PtxIndexOccurrence] = []
    for _ in range(occ_cnt):
        if pos + 5 > body_end:
            raise PtxCorruptFileError("Truncated 0x0002 occurrence header")
        flag, byte_len, off_cnt = struct.unpack_from("<BHH", buf, pos)
        pos += 5
        if byte_len != 4 * off_cnt or pos + byte_len + 6 > body_end:
            raise PtxCorruptFileError(f"Invalid 0x0002 occurrence bounds (byte_len={byte_len}, count={off_cnt})")
        offs = tuple(struct.unpack_from(f"<{off_cnt}I", buf, pos))
        tail = buf[pos + byte_len : pos + byte_len + 6]
        occs.append(PtxIndexOccurrence(flag=flag, offsets=offs, tail=tail))
        pos += byte_len + 6
    return PtxIndexEntry(path=steps, occurrences=occs), pos


def parse_index_0002(decrypted_buf: bytes) -> PtxIndex0002 | None:
    """Decode the trailing ``0x0002`` chunk-offset index into a structured :class:`PtxIndex0002`."""
    chunks = iter_top_chunks(decrypted_buf, strict=True)
    c2 = next((c for c in reversed(chunks) if c.tag == 0x0002), None)
    if c2 is None:
        return None
    body_end = c2.offset + 7 + c2.size
    pos = c2.offset + 9
    if pos + 4 > body_end:
        raise PtxCorruptFileError("Truncated 0x0002 entry count")
    entry_count = struct.unpack_from("<I", decrypted_buf, pos)[0]
    pos += 4
    if entry_count > (body_end - pos) // 8:
        raise PtxCorruptFileError(f"0x0002 entry count {entry_count} exceeds chunk bounds")
    entries: list[PtxIndexEntry] = []
    for _ in range(entry_count):
        entry, pos = _parse_index_entry(decrypted_buf, pos, body_end)
        entries.append(entry)
    if pos != body_end:
        raise PtxCorruptFileError(f"0x0002 parse ended at {pos}, expected {body_end}")
    return PtxIndex0002(version=c2.version, entries=entries)


def serialize_index_0002(index: PtxIndex0002) -> bytes:
    """Serialize a :class:`PtxIndex0002` back into a binary ``0x0002`` Z-chunk."""
    body = bytearray(struct.pack("<I", len(index.entries)))
    for entry in index.entries:
        body.extend(struct.pack("<I", len(entry.path)))
        for step in entry.path:
            if not (0 <= step.tag <= 0xFFFF):
                raise PtxValidationError(f"0x0002 path step tag out of u16 range: {step.tag}")
            if not (step.offset == -1 or 0 <= step.offset <= 0x7FFF_FFFF):
                raise PtxValidationError(f"0x0002 path step offset out of range: {step.offset}")
            if len(step.meta) != 5:
                raise PtxValidationError(f"0x0002 path step meta must be 5 bytes, got {len(step.meta)}")
            body.extend(struct.pack("<Hi", step.tag, step.offset))
            body.extend(step.meta)
        body.extend(struct.pack("<I", len(entry.occurrences)))
        for occ in entry.occurrences:
            if not (0 <= occ.flag <= 0xFF):
                raise PtxValidationError(f"0x0002 occurrence flag out of u8 range: {occ.flag}")
            if len(occ.offsets) > 0xFFFF // 4:
                raise PtxValidationError(f"0x0002 occurrence offset count exceeds u16 byte length: {len(occ.offsets)}")
            if len(occ.tail) != 6:
                raise PtxValidationError(f"0x0002 occurrence tail must be 6 bytes, got {len(occ.tail)}")
            for off in occ.offsets:
                if not (0 <= off <= 0xFFFF_FFFF):
                    raise PtxValidationError(f"0x0002 occurrence offset out of u32 range: {off}")
            byte_len = 4 * len(occ.offsets)
            body.extend(struct.pack("<BHH", occ.flag, byte_len, len(occ.offsets)))
            if occ.offsets:
                body.extend(struct.pack(f"<{len(occ.offsets)}I", *occ.offsets))
            body.extend(occ.tail)
    return build_zchunk(0x0002, bytes(body), version=index.version)


def shift_parsed_index_0002(index: PtxIndex0002, threshold: int, delta: int) -> PtxIndex0002:
    """Return a copy of *index* with all offsets ``>= threshold`` shifted by *delta*."""
    new_entries: list[PtxIndexEntry] = []
    for entry in index.entries:
        new_path: list[PtxIndexPathStep] = []
        for step in entry.path:
            off = step.offset
            if delta != 0 and off != -1 and off >= threshold:
                off += delta
                if off < 0 or off > 0x7FFF_FFFF:
                    raise PtxValidationError(f"Shifted 0x0002 path offset out of signed i32 range: {off}")
            new_path.append(PtxIndexPathStep(tag=step.tag, offset=off, meta=step.meta))
        new_occs: list[PtxIndexOccurrence] = []
        for occ in entry.occurrences:
            shifted_offs: list[int] = []
            for off in occ.offsets:
                new_off = off + delta if (delta != 0 and off >= threshold) else off
                if new_off < 0 or new_off > 0xFFFF_FFFF:
                    raise PtxValidationError(f"Shifted 0x0002 occurrence offset out of u32 range: {new_off}")
                shifted_offs.append(new_off)
            new_occs.append(PtxIndexOccurrence(flag=occ.flag, offsets=tuple(shifted_offs), tail=occ.tail))
        new_entries.append(PtxIndexEntry(path=new_path, occurrences=new_occs))
    return PtxIndex0002(version=index.version, entries=new_entries)


def replace_index_0002_chunk(buf: bytearray, index: PtxIndex0002) -> None:
    """Replace the trailing ``0x0002`` chunk in *buf* with *index* and update root pointer ``0x1B``."""
    chunks = iter_top_chunks(bytes(buf), strict=True)
    c2 = next((c for c in reversed(chunks) if c.tag == 0x0002), None)
    if c2 is None:
        raise PtxCorruptFileError("Missing trailing 0x0002 chunk in .ptx buffer")
    old_end = c2.offset + 7 + c2.size
    if old_end != len(buf):
        raise PtxCorruptFileError(f"0x0002 chunk at {c2.offset:#x}..{old_end:#x} is not trailing (len={len(buf):#x})")
    if c2.offset > 0xFFFF_FFFF:
        raise PtxValidationError(f"0x0002 chunk offset exceeds u32 range: {c2.offset:#x}")
    buf[c2.offset : old_end] = serialize_index_0002(index)
    if len(buf) >= 0x1F:
        struct.pack_into("<I", buf, 0x1B, c2.offset)
