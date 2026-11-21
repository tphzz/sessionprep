"""General-purpose RIFF (WAV) / IFF (AIFF) chunk reader and writer.

Provides functions to enumerate, read, write, and selectively remove
chunks from WAV and AIFF files without external dependencies beyond
the Python standard library.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

from .file_io import atomic_writer


@dataclass
class AudioChunk:
    """A single chunk from a RIFF or IFF container.

    Attributes:
        id: 4-character chunk identifier (e.g. ``"bext"``, ``"iXML"``).
        size: Payload size in bytes (excluding the 8-byte header).
        data: Raw chunk payload.
    """
    id: str
    size: int
    data: bytes


# Chunk IDs that are always present and generally not interesting
# for DAW interoperability diagnostics.
STANDARD_CHUNKS: frozenset[str] = frozenset({
    # WAV
    "fmt ", "data",
    # AIFF / AIFC
    "COMM", "SSND",
})


def _container_header(stream: BinaryIO) -> tuple[str, str, int]:
    header = stream.read(12)
    if len(header) != 12:
        raise ValueError("Truncated WAV/AIFF container header")
    kind, form = header[:4], header[8:]
    if kind == b"RIFF" and form == b"WAVE":
        size_format = "<I"
    elif kind == b"FORM" and form in (b"AIFF", b"AIFC"):
        size_format = ">I"
    else:
        raise ValueError(f"Not a recognised WAV/AIFF container: {kind!r} / {form!r}")

    end = struct.unpack(size_format, header[4:8])[0] + 8
    if end < 12 or end > stream.seek(0, 2):
        raise ValueError("Invalid WAV/AIFF container length")
    return form.decode("ascii"), size_format, end


def _chunk_headers(stream: BinaryIO, size_format: str, end: int) -> Iterator[tuple[str, int]]:
    """Yield ID and size with the stream positioned at each payload."""
    pos = 12
    while pos < end:
        stream.seek(pos)
        header = stream.read(8)
        if pos + 8 > end or len(header) != 8:
            raise ValueError("Truncated WAV/AIFF chunk header")
        name = header[:4].decode("ascii", errors="replace")
        size = struct.unpack(size_format, header[4:])[0]
        pos += 8 + size
        if pos > end:
            raise ValueError(f"Truncated {name!r} chunk payload")
        yield name, size
        # An odd payload is padded before the next header. Some PCM writers
        # omit the final pad and declare a container ending at the last sample.
        pos += size % 2


def chunk_sizes(filepath: str) -> list[tuple[str, int]]:
    """Read ordered chunk IDs and payload sizes without loading PCM or metadata."""
    with open(filepath, "rb") as stream:
        _, size_format, end = _container_header(stream)
        return list(_chunk_headers(stream, size_format, end))


def chunk_ids(filepath: str) -> list[str]:
    """Return chunk IDs in file order without reading the payloads."""
    return [name for name, _ in chunk_sizes(filepath)]


def read_chunks(filepath: str) -> tuple[str, list[AudioChunk]]:
    """Read a complete WAVE, AIFF or AIFC container, including chunk payloads."""
    chunks: list[AudioChunk] = []
    with open(filepath, "rb") as stream:
        container, size_format, end = _container_header(stream)
        for name, size in _chunk_headers(stream, size_format, end):
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"Truncated {name!r} chunk payload")
            chunks.append(AudioChunk(id=name, size=size, data=data))
    return container, chunks


def write_chunks(
    filepath: str,
    container: str,
    chunks: list[AudioChunk],
) -> None:
    """Write chunks to a new WAV or AIFF file.

    Rebuilds the RIFF/FORM container header from the given chunks.

    Parameters
    ----------
    filepath : str
        Destination path.
    container : str
        ``"WAVE"``, ``"AIFF"``, or ``"AIFC"``.
    chunks : list[AudioChunk]
        Chunks to write (order is preserved).
    """
    if container == "WAVE":
        container_id = b"RIFF"
        form_type = b"WAVE"
        size_fmt = "<I"
    elif container in ("AIFF", "AIFC"):
        container_id = b"FORM"
        form_type = container.encode("ascii").ljust(4)
        size_fmt = ">I"
    else:
        raise ValueError(f"Unknown container format: {container!r}")

    # Calculate total data size (form_type + all chunk headers + data + padding)
    data_size = 4  # form_type
    for ch in chunks:
        data_size += 8 + len(ch.data)
        if len(ch.data) % 2:
            data_size += 1

    with atomic_writer(filepath, binary=True) as f:
        f.write(container_id)
        f.write(struct.pack(size_fmt, data_size))
        f.write(form_type)

        for ch in chunks:
            chunk_id_bytes = ch.id.encode("ascii").ljust(4)[:4]
            f.write(chunk_id_bytes)
            f.write(struct.pack(size_fmt, len(ch.data)))
            f.write(ch.data)
            if len(ch.data) % 2:
                f.write(b"\x00")


def remove_chunks(
    src: str,
    dst: str,
    remove_ids: set[str],
) -> None:
    """Copy *src* to *dst*, omitting chunks whose IDs are in *remove_ids*.

    Convenience wrapper around :func:`read_chunks` and
    :func:`write_chunks`.

    Parameters
    ----------
    src : str
        Source audio file path.
    dst : str
        Destination audio file path. An in-place edit replaces the source only
        after the complete edited container has been written successfully.
    remove_ids : set[str]
        Chunk IDs to strip (e.g. ``{"iXML", "JUNK"}``).
    """
    container, chunks = read_chunks(src)
    filtered = [ch for ch in chunks if ch.id not in remove_ids]
    write_chunks(dst, container, filtered)


def notable_chunks(all_ids: list[str]) -> list[str]:
    """Filter a list of chunk IDs, returning only non-standard ones."""
    return [cid for cid in all_ids if cid not in STANDARD_CHUNKS]


# ---------------------------------------------------------------------------
# DAW origin detection
# ---------------------------------------------------------------------------

# Proprietary chunk IDs → DAW name (tier-1, no I/O needed)
_FINGERPRINT_CHUNKS: dict[str, str] = {
    "DGDA": "Pro Tools",
    "minf": "Pro Tools",
    "elm1": "Pro Tools",
    "regn": "Pro Tools",
    "Fake": "Cubase / Nuendo",
}


def _read_bext_originator(filepath: str) -> str | None:
    """Read the originator field from a bext chunk, if present.

    The bext chunk layout (EBU Tech 3285):
        bytes   0–255: Description (256 bytes)
        bytes 256–287: Originator (32 bytes ASCII)
    """
    with open(filepath, "rb") as stream:
        _, size_format, end = _container_header(stream)
        for name, size in _chunk_headers(stream, size_format, end):
            if name == "bext" and size >= 288:
                stream.seek(256, 1)
                return stream.read(32).decode("ascii", errors="replace").strip("\x00").strip()
    return None


def detect_origin(
    chunk_ids: list[str],
    filepath: str | None = None,
) -> str | None:
    """Identify the DAW or application that created an audio file.

    Tier 1: checks proprietary chunk IDs (no file I/O).
    Tier 2: if *filepath* is provided and a ``bext`` chunk exists,
    reads the 32-byte originator field.

    Returns a human-readable string (e.g. ``"Pro Tools"``,
    ``"Cubase / Nuendo"``) or ``None`` if the origin cannot be
    determined.
    """
    # Tier 1: fingerprint chunks
    for cid in chunk_ids:
        origin = _FINGERPRINT_CHUNKS.get(cid)
        if origin:
            return origin

    # Tier 2: bext originator field
    if filepath and "bext" in chunk_ids:
        try:
            originator = _read_bext_originator(filepath)
            if originator:
                return originator
        except (OSError, ValueError):
            pass

    return None
