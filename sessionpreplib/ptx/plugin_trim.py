"""Validated Slot A Trim plugin reader and atomic ``.ptx`` patcher.

Supports three verified AAX gain-trim plugins in Insert Slot A:
  - DMG Audio ``TrackControl`` (``DMGA/TrCt/RTST`` stereo, ``DMGA/TrCt/RTMN`` mono):
    ``kOpGain`` linear ``[-200.0, +18.0] dB`` in ``elck`` (``d_2dl ``) + ``DGCT`` zlib JSON.
  - Avid stock ``Trim`` (``Digi/Trim/TR11`` stereo/mono, default ``+6 dB`` range mode):
    10-bit Pro Tools fader taper ``[-144.0, +6.0] dB`` in fixed-size ``All `` chunk.
  - Process.Audio ``Protility`` (``PrAu/PrTy/jccc`` stereo, ``PrAu/PrTy/jcbb`` mono):
    ``GAIN`` linear ``[-60.0, +24.0] dB`` in ``elck`` (``d_GAIN``) + ``juce`` ``VC2!`` XML.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import struct
from typing import Any
import zlib

from ..signal_flow.trim_codecs import (
    PROTILITY_MAX_DB,
    PROTILITY_MIN_DB,
    TRACKCONTROL_MAX_DB,
    TRACKCONTROL_MIN_DB,
    decode_trackcontrol_dgct_zip,
    extract_trackcontrol_gain,
    norm_zero as _norm_zero,
    patch_protility_vc2_xml,
    patch_trackcontrol_dgct_zip,
    protility_db_to_norm,
    read_protility_vc2_xml,
    select_preferred_trim_donor,
    trackcontrol_norm_to_db,
)
from .codec import decrypt_ptx, encrypt_ptx, find_subchunks, shift_index_0002, validate_index_0002
from .models import PtxPluginInfo, SlotATrimInfo

log = logging.getLogger(__name__)

_AVID_OFF_I32 = -0x80000000


@dataclass(frozen=True)
class SlotATrimSpec:
    """Strict AAX identity and valid gain range for a supported Slot A trim plugin."""

    plugin_id: str
    display_name: str
    manufacturer_4cc: str
    product_4cc: str
    type_4ccs: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    min_db: float
    max_db: float


@dataclass(frozen=True)
class _SlotALocation:
    """Structurally verified offsets for a track's Insert Slot A inside ``0x2624``."""

    ancestors: tuple[int, ...]  # c2624 -> track blk -> [0x261B] -> 0x2627 -> 0x2616 -> [0x2615] -> 0x2613
    p2616_end: int
    p2613: int
    p1038: int
    len1038: int


SLOT_A_TRIM_SPECS: tuple[SlotATrimSpec, ...] = (
    SlotATrimSpec(
        plugin_id="trackcontrol",
        display_name="TrackControl",
        manufacturer_4cc="DMGA",
        product_4cc="TrCt",
        type_4ccs=("RTMN", "RTST"),
        bundle_ids=("com.dmgaudio.TrackControl.aax",),
        min_db=TRACKCONTROL_MIN_DB,
        max_db=TRACKCONTROL_MAX_DB,
    ),
    SlotATrimSpec(
        plugin_id="trim",
        display_name="Trim",
        manufacturer_4cc="Digi",
        product_4cc="Trim",
        type_4ccs=("TR11",),
        bundle_ids=("com.avid.aax.trim",),
        min_db=-144.0,
        max_db=6.0,
    ),
    SlotATrimSpec(
        plugin_id="protility",
        display_name="Protility",
        manufacturer_4cc="PrAu",
        product_4cc="PrTy",
        type_4ccs=("jcbb", "jccc"),
        bundle_ids=("com.Process.Audio.Protility",),
        min_db=PROTILITY_MIN_DB,
        max_db=PROTILITY_MAX_DB,
    ),
)

# Calibrated against Pro Tools Studio 2026.11 AAX Avid Trim (TR11, +6 dB default mode)
# with multi-segment Pro Tools 10-bit log taper below -24 dB.
_AVID_TRIM_TAPER: tuple[tuple[float, int], ...] = (
    (-144.0, 0),
    (-90.0, 18),
    (-60.0, 58),
    (-48.0, 92),
    (-36.0, 150),
    (-30.0, 194),
    (-24.0, 250),
    (-12.0, 473),
    (-6.0, 622),
    (0.0, 772),
    (6.0, 1023),
)

_MONO_ONLY_4CCS = frozenset({"RTMN", "jcbb"})
_STEREO_ONLY_4CCS = frozenset({"RTST", "jccc"})


def match_slot_a_trim_spec(pinfo: PtxPluginInfo) -> SlotATrimSpec | None:
    """Match *pinfo* strictly by AAX 4CC triple and bundle ID."""
    for spec in SLOT_A_TRIM_SPECS:
        if (
            pinfo.manufacturer_4cc == spec.manufacturer_4cc
            and pinfo.product_4cc == spec.product_4cc
            and pinfo.type_4cc in spec.type_4ccs
            and (not pinfo.bundle_id or pinfo.bundle_id in spec.bundle_ids)
        ):
            return spec
    return None


def avid_trim_db_to_step(db: float) -> int:
    """Convert dB in ``[-144.0, +6.0]`` to Avid Trim's 10-bit step ``0..1023``."""
    if db <= _AVID_TRIM_TAPER[0][0]:
        return 0
    if db >= _AVID_TRIM_TAPER[-1][0]:
        return 1023
    for i in range(len(_AVID_TRIM_TAPER) - 1):
        d0, s0 = _AVID_TRIM_TAPER[i]
        d1, s1 = _AVID_TRIM_TAPER[i + 1]
        if d0 <= db <= d1:
            t = (db - d0) / (d1 - d0)
            return max(0, min(1023, int(round(s0 + t * (s1 - s0)))))
    return 772


def avid_trim_step_to_db(step: int) -> float:
    """Convert Avid Trim's 10-bit step ``0..1023`` back to dB."""
    s = max(0, min(1023, int(step)))
    if s <= 0:
        return -144.0
    if s >= 1023:
        return 6.0
    for i in range(len(_AVID_TRIM_TAPER) - 1):
        d0, s0 = _AVID_TRIM_TAPER[i]
        d1, s1 = _AVID_TRIM_TAPER[i + 1]
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0)
            return _norm_zero(round(d0 + t * (d1 - d0), 2))
    return 6.0


def _avid_trim_step_to_i32(step: int) -> int:
    s = max(0, min(1023, int(step)))
    if s >= 1023:
        return 0x7FFFFFFE
    return ((s << 22) | (s << 12) | (s << 2)) - 0x80000000


def _avid_trim_i32_to_step(raw_i32: int) -> int:
    u = (raw_i32 + (1 << 31)) / float((1 << 32) - 1)
    return max(0, min(1023, int(round(u * 1023.0))))


def _read_trackcontrol_state(b1038: bytes, expected_tag: bytes) -> tuple[float, bool] | None:
    if expected_tag not in b1038[:64]:
        return None
    dgct = b1038.find(b"DGCT")
    ep = b1038.find(b"d_2dl \x00\x00", 0, dgct) if dgct >= 20 else -1
    if ep == -1 or ep + 16 > dgct:
        return None
    chunk_start = dgct - 20
    chunk_len = struct.unpack_from(">I", b1038, chunk_start)[0]
    zpos = b1038.find(b"ZIP ", dgct)
    z_end = chunk_start + chunk_len
    if zpos == -1 or z_end > len(b1038) or zpos + 4 >= z_end:
        return None
    try:
        js, _trailing = decode_trackcontrol_dgct_zip(b1038[zpos + 4 : z_end])
        parsed = extract_trackcontrol_gain(js)
        if parsed is not None:
            return parsed
        elck_val = struct.unpack_from(">d", b1038, ep + 8)[0]
        return (trackcontrol_norm_to_db(elck_val), False) if math.isfinite(elck_val) else None
    except (ValueError, TypeError, KeyError, zlib.error, UnicodeDecodeError):
        return None


def _read_avid_trim_db(b1038: bytes, expected_tag: bytes) -> float | None:
    tag_pos = b1038.find(expected_tag + b"All ", 0, 48)
    if tag_pos < 8 or tag_pos + 64 > len(b1038):
        return None
    raw_gain, flag1, flag2, flag3 = struct.unpack_from(">iiii", b1038, tag_pos + 48)
    # Fail closed if +12 dB output range, Mute, or Phase switch deviates from default +6 dB OFF state.
    if (flag1, flag2, flag3) != (_AVID_OFF_I32, _AVID_OFF_I32, _AVID_OFF_I32):
        return None
    return avid_trim_step_to_db(_avid_trim_i32_to_step(raw_gain))


def _read_protility_state(b1038: bytes, expected_tag: bytes) -> tuple[float, bool] | None:
    if expected_tag not in b1038[:64]:
        return None
    juce = b1038.find(b"juce")
    gp = b1038.find(b"d_GAIN\x00\x00", 0, juce) if juce >= 20 else -1
    if gp == -1 or gp + 16 > juce:
        return None
    chunk_start = juce - 20
    chunk_len = struct.unpack_from(">I", b1038, chunk_start)[0]
    chunk_end = chunk_start + chunk_len
    if chunk_end > len(b1038):
        return None
    vc2 = b1038.find(b"VC2!", juce, chunk_end)
    if vc2 == -1 or vc2 + 8 > chunk_end:
        return None
    xml_len = struct.unpack_from("<I", b1038, vc2 + 4)[0]
    xml_start = vc2 + 8
    if xml_start + xml_len > chunk_end:
        return None
    return read_protility_vc2_xml(b1038[xml_start : xml_start + xml_len])


def inspect_slot_a_trim(
    slot_blk: bytes,
    pinfo: PtxPluginInfo,
    *,
    is_active: bool,
    is_bypassed: bool,
    track_format: str = "",
    has_sidechain: bool = False,
) -> SlotATrimInfo | None:
    """Validate Slot A plugin identity and state blob, returning decoded gain info or ``None``."""
    spec = match_slot_a_trim_spec(pinfo)
    if spec is None:
        return None
    if track_format == "Mono" and pinfo.type_4cc in _STEREO_ONLY_4CCS:
        return None
    if track_format == "Stereo" and pinfo.type_4cc in _MONO_ONLY_4CCS:
        return None
    s1038 = find_subchunks(slot_blk, 0x1038)
    # Fail closed on missing or Multi-Mono (>1) 0x1038 plugin state blocks.
    if len(s1038) != 1:
        return None
    b1038 = s1038[0]
    expected_tag = f"{pinfo.manufacturer_4cc}{pinfo.product_4cc}{pinfo.type_4cc}".encode("latin1")
    is_clean = False
    if spec.plugin_id == "trackcontrol":
        state = _read_trackcontrol_state(b1038, expected_tag)
        if state is None:
            return None
        gain_db, is_clean = state
    elif spec.plugin_id == "trim":
        t_db = _read_avid_trim_db(b1038, expected_tag)
        if t_db is None:
            return None
        gain_db, is_clean = t_db, True
    elif spec.plugin_id == "protility":
        state = _read_protility_state(b1038, expected_tag)
        if state is None:
            return None
        gain_db, is_clean = state
    else:
        return None
    if has_sidechain or (track_format and track_format not in ("Mono", "Stereo")):
        is_clean = False
    fourcc = f"{pinfo.manufacturer_4cc}/{pinfo.product_4cc}/{pinfo.type_4cc}"
    return SlotATrimInfo(
        plugin_id=spec.plugin_id,
        plugin_name=spec.display_name,
        fourcc=fourcc,
        bundle_id=pinfo.bundle_id,
        gain_db=gain_db,
        min_db=spec.min_db,
        max_db=spec.max_db,
        is_active=is_active,
        is_bypassed=is_bypassed,
        is_clean_donor=is_clean,
    )


def _patch_trackcontrol_1038(b1038: bytes, gain_db: float, expected_tag: bytes) -> bytes:
    if len(b1038) < 21 or expected_tag not in b1038[:64]:
        raise ValueError("TrackControl 0x1038 chunk missing expected AAX header.")
    old_inner_len = struct.unpack_from("<I", b1038, 13)[0]
    if 17 + old_inner_len > len(b1038):
        raise ValueError("TrackControl 0x1038 inner length out of bounds.")
    buf = bytearray(b1038)
    dgct = buf.find(b"DGCT")
    ep = buf.find(b"d_2dl \x00\x00", 0, dgct) if dgct >= 20 else -1
    if ep == -1 or ep + 16 > dgct:
        raise ValueError("TrackControl 0x1038 chunk missing expected elck/DGCT structure.")

    chunk_start = dgct - 20
    old_chunk_len = struct.unpack_from(">I", buf, chunk_start)[0]
    zpos = buf.find(b"ZIP ", dgct)
    z_end = chunk_start + old_chunk_len
    if zpos == -1 or z_end > len(buf) or zpos + 4 >= z_end:
        raise ValueError("TrackControl DGCT ZIP payload out of bounds.")
    new_zip, norm = patch_trackcontrol_dgct_zip(bytes(buf[zpos + 4 : z_end]), gain_db)
    struct.pack_into(">d", buf, ep + 8, norm)

    delta = len(new_zip) - (z_end - (zpos + 4))
    buf[zpos + 4 : z_end] = new_zip
    struct.pack_into(">I", buf, chunk_start, old_chunk_len + delta)
    struct.pack_into("<I", buf, 13, old_inner_len + delta)
    struct.pack_into("<I", buf, 3, struct.unpack_from("<I", buf, 3)[0] + delta)
    return bytes(buf)


def _patch_avid_trim_1038(b1038: bytes, gain_db: float, expected_tag: bytes) -> bytes:
    tag_pos = b1038.find(expected_tag + b"All ", 0, 48)
    if tag_pos < 8 or tag_pos + 64 > len(b1038):
        raise ValueError("Avid Trim 0x1038 chunk missing expected DigiTrimTR11All header.")
    _, flag1, flag2, flag3 = struct.unpack_from(">iiii", b1038, tag_pos + 48)
    if (flag1, flag2, flag3) != (_AVID_OFF_I32, _AVID_OFF_I32, _AVID_OFF_I32):
        raise ValueError("Avid Trim has non-default switch state (+12 dB mode, Mute, or Phase); refusing to patch.")
    buf = bytearray(b1038)
    step = avid_trim_db_to_step(gain_db)
    struct.pack_into(">i", buf, tag_pos + 48, _avid_trim_step_to_i32(step))
    return bytes(buf)


def _patch_protility_1038(b1038: bytes, gain_db: float, expected_tag: bytes) -> bytes:
    if len(b1038) < 21 or expected_tag not in b1038[:64]:
        raise ValueError("Protility 0x1038 chunk missing expected AAX header.")
    old_inner_len = struct.unpack_from("<I", b1038, 13)[0]
    if 17 + old_inner_len > len(b1038):
        raise ValueError("Protility 0x1038 inner length out of bounds.")
    buf = bytearray(b1038)
    norm = protility_db_to_norm(gain_db)
    juce = buf.find(b"juce")
    gp = buf.find(b"d_GAIN\x00\x00", 0, juce) if juce >= 20 else -1
    if gp == -1 or gp + 16 > juce:
        raise ValueError("Protility 0x1038 chunk missing expected elck/juce structure.")
    struct.pack_into(">d", buf, gp + 8, norm)

    chunk_start = juce - 20
    old_chunk_len = struct.unpack_from(">I", buf, chunk_start)[0]
    chunk_end = chunk_start + old_chunk_len
    if chunk_end > len(buf):
        raise ValueError("Protility juce chunk length out of bounds.")
    vc2 = buf.find(b"VC2!", juce, chunk_end)
    if vc2 == -1 or vc2 + 8 > chunk_end:
        raise ValueError("Protility juce chunk missing VC2! header.")
    old_xml_len = struct.unpack_from("<I", buf, vc2 + 4)[0]
    xml_start = vc2 + 8
    if xml_start + old_xml_len > chunk_end:
        raise ValueError("Protility XML payload out of bounds.")
    new_xml_bytes = patch_protility_vc2_xml(bytes(buf[xml_start : xml_start + old_xml_len]), gain_db)
    delta = len(new_xml_bytes) - old_xml_len
    buf[xml_start : xml_start + old_xml_len] = new_xml_bytes
    struct.pack_into("<I", buf, vc2 + 4, len(new_xml_bytes))
    struct.pack_into(">I", buf, chunk_start, old_chunk_len + delta)
    struct.pack_into("<I", buf, 13, old_inner_len + delta)
    struct.pack_into("<I", buf, 3, struct.unpack_from("<I", buf, 3)[0] + delta)
    return bytes(buf)


def _find_child_zchunk_span(buf: bytes, parent_off: int, tag: int) -> tuple[int, int] | None:
    """Return ``(abs_start, abs_end)`` of the first subchunk with *tag* inside *buf* starting at offset 9."""
    matches = find_subchunks(buf, tag)
    if not matches:
        return None
    rel = buf.find(matches[0], 9)
    if rel == -1:
        return None
    return parent_off + rel, parent_off + rel + len(matches[0])


@dataclass(frozen=True)
class _SlotAEmptyLocation:
    """Structurally verified offsets for an empty Insert Slot A (``0x2625``) inside ``0x2624``."""

    ancestors: tuple[int, ...]  # c2624 -> track blk -> [0x261B] -> 0x2627
    slot0_start: int
    slot0_end: int


def select_preferred_slot_a_trim_donor(session: Any, track_format: str) -> Any | None:
    """Return the cleanest verified donor track for the most-used Slot A Trim matching *track_format*."""
    return select_preferred_trim_donor(session.tracks, track_format)


def _first_slot_in_2627(b2627: bytes, p2627: int) -> tuple[int, int, int] | None:
    """Return ``(abs_start, abs_end, tag)`` for exact Slot A (slot index 0) inside ``0x2627``."""
    rel = 11 if (len(b2627) > 11 and b2627[11] == 0x5A) else 9
    if rel + 9 > len(b2627) or b2627[rel] != 0x5A:
        return None
    ver = struct.unpack_from("<H", b2627, rel + 1)[0]
    csz = struct.unpack_from("<I", b2627, rel + 3)[0]
    tag = struct.unpack_from("<H", b2627, rel + 7)[0]
    end = rel + 7 + csz
    if ver >= 0x1000 or csz < 2 or end > len(b2627):
        return None
    return p2627 + rel, p2627 + end, tag


def _find_block_2627_ancestors(
    dec: bytearray,
    c2624_off: int,
    blk_start: int,
    blk_end: int,
) -> tuple[list[int], int, int] | None:
    blk = bytes(dec[blk_start:blk_end])
    r2627 = _find_child_zchunk_span(blk, blk_start, 0x2627)
    if r2627 is None:
        return None
    p2627, e2627 = r2627
    ancestors: list[int] = [c2624_off, blk_start]
    for sub_1b in find_subchunks(blk, 0x261B):
        rel_1b = blk.find(sub_1b, 9)
        if rel_1b != -1 and blk_start + rel_1b < p2627 and blk_start + rel_1b + len(sub_1b) >= e2627:
            ancestors.append(blk_start + rel_1b)
            break
    ancestors.append(p2627)
    return ancestors, p2627, e2627


def _extract_slot_a_from_block(
    dec: bytearray,
    c2624_off: int,
    blk_start: int,
    blk_end: int,
) -> _SlotALocation | None:
    """Walk the exact Z-chunk hierarchy ``c2624 -> blk -> [261B] -> 2627 -> slot0(2616) -> [2615] -> 2613 -> 1038``."""
    found = _find_block_2627_ancestors(dec, c2624_off, blk_start, blk_end)
    if found is None:
        return None
    ancestors, p2627, e2627 = found
    slot0 = _first_slot_in_2627(bytes(dec[p2627:e2627]), p2627)
    if slot0 is None or slot0[2] != 0x2616:
        return None
    p2616, e2616, _ = slot0
    ancestors.append(p2616)

    b2616 = bytes(dec[p2616:e2616])
    r2613 = _find_child_zchunk_span(b2616, p2616, 0x2613)
    if r2613 is None:
        return None
    p2613, e2613 = r2613
    for sub_15 in find_subchunks(b2616, 0x2615):
        rel_15 = b2616.find(sub_15, 9)
        if rel_15 != -1 and p2616 + rel_15 < p2613 and p2616 + rel_15 + len(sub_15) >= e2613:
            ancestors.append(p2616 + rel_15)
            break
    ancestors.append(p2613)

    b2613 = bytes(dec[p2613:e2613])
    r1038 = _find_child_zchunk_span(b2613, p2613, 0x1038)
    if r1038 is None:
        return None
    p1038, e1038 = r1038
    return _SlotALocation(tuple(ancestors), e2616, p2613, p1038, e1038 - p1038)


def _locate_track_block_bounds(
    dec: bytes | bytearray,
    c2624_off: int,
    track_index: int,
    track_uuid_hex: str = "",
) -> tuple[int, int] | None:
    if c2624_off + 13 > len(dec) or dec[c2624_off] != 0x5A:
        return None
    c2624_end = c2624_off + 7 + struct.unpack_from("<I", dec, c2624_off + 3)[0]
    if c2624_end > len(dec):
        return None
    tcount = struct.unpack_from("<I", dec, c2624_off + 9)[0]
    uuid_bytes = bytes.fromhex(track_uuid_hex) if track_uuid_hex else b""
    pos = c2624_off + 13
    fallback_bounds: tuple[int, int] | None = None
    for ti in range(tcount):
        if pos + 9 > c2624_end or dec[pos] != 0x5A:
            return None
        csz = struct.unpack_from("<I", dec, pos + 3)[0]
        end = pos + 7 + csz
        if csz < 2 or end > c2624_end:
            return None
        if uuid_bytes:
            if dec.find(uuid_bytes, pos, min(pos + 256, end)) != -1:
                return pos, end
        elif ti == track_index:
            fallback_bounds = (pos, end)
        pos = end
    return fallback_bounds


def _locate_slot_a_offsets(
    dec: bytearray,
    c2624_off: int,
    track_index: int,
    track_uuid_hex: str = "",
) -> _SlotALocation | None:
    """Locate occupied Slot A (``0x2616``) structurally inside ``0x2624``."""
    bounds = _locate_track_block_bounds(dec, c2624_off, track_index, track_uuid_hex)
    return _extract_slot_a_from_block(dec, c2624_off, *bounds) if bounds else None


def _locate_slot_a_empty_2625(
    dec: bytearray,
    c2624_off: int,
    track_index: int,
    track_uuid_hex: str = "",
) -> _SlotAEmptyLocation | None:
    """Locate an empty Insert Slot A (``0x2625``) structurally inside ``0x2624``."""
    bounds = _locate_track_block_bounds(dec, c2624_off, track_index, track_uuid_hex)
    if bounds is None:
        return None
    found = _find_block_2627_ancestors(dec, c2624_off, *bounds)
    if found is None:
        return None
    ancestors, p2627, e2627 = found
    slot0 = _first_slot_in_2627(bytes(dec[p2627:e2627]), p2627)
    if slot0 is None or slot0[2] != 0x2625:
        return None
    return _SlotAEmptyLocation(tuple(ancestors), slot0[0], slot0[1])


def _validate_and_shift_span_ancestors(
    dec: bytearray,
    ancestors: tuple[int, ...],
    target_s: int,
    target_e: int,
    delta: int,
) -> None:
    """Verify strict Z-chunk nesting for *ancestors* around ``[target_s, target_e)`` and shift lengths."""
    prev_start, prev_end = -1, len(dec)
    new_sizes: list[tuple[int, int]] = []
    for anc in ancestors:
        if anc + 9 > len(dec) or dec[anc] != 0x5A:
            raise ValueError(f"Corrupt ancestor Z-chunk header at 0x{anc:x}.")
        ver = struct.unpack_from("<H", dec, anc + 1)[0]
        old_sz = struct.unpack_from("<I", dec, anc + 3)[0]
        end = anc + 7 + old_sz
        if ver >= 0x1000 or not (prev_start < anc < target_s < target_e <= end <= prev_end):
            raise ValueError(f"Ancestor Z-chunk at 0x{anc:x} violates strict nesting.")
        new_sz = old_sz + delta
        if not (2 <= new_sz <= 0xFFFFFFFF):
            raise ValueError(f"Ancestor Z-chunk size overflow/underflow at 0x{anc:x}.")
        new_sizes.append((anc, new_sz))
        prev_start, prev_end = anc, end
    for anc, new_sz in new_sizes:
        struct.pack_into("<I", dec, anc + 3, new_sz)


def _patch_1038_for_plugin(b1038: bytes, trim_info: SlotATrimInfo, clamped_db: float) -> bytes:
    expected_tag = trim_info.fourcc.replace("/", "").encode("latin1")
    if trim_info.plugin_id == "trackcontrol":
        return _patch_trackcontrol_1038(b1038, clamped_db, expected_tag)
    if trim_info.plugin_id == "trim":
        return _patch_avid_trim_1038(b1038, clamped_db, expected_tag)
    if trim_info.plugin_id == "protility":
        return _patch_protility_1038(b1038, clamped_db, expected_tag)
    raise ValueError(f"Unsupported Slot A Trim plugin ID: {trim_info.plugin_id!r}")


def _extract_pristine_donor_2616(orig_dec: bytes, c2624_off: int, donor_trk: Any) -> bytes:
    """Extract the raw ``0x2616`` Insert Slot A chunk from *donor_trk* before any mutations."""
    bounds = _locate_track_block_bounds(orig_dec, c2624_off, donor_trk.index, donor_trk.uuid_hex)
    if bounds is None:
        raise ValueError(f"Could not locate donor track {donor_trk.name!r} in 0x2624.")
    blk = orig_dec[bounds[0] : bounds[1]]
    r2627 = _find_child_zchunk_span(blk, bounds[0], 0x2627)
    if r2627 is None:
        raise ValueError(f"Donor track {donor_trk.name!r} missing 0x2627 insert bank.")
    slot0 = _first_slot_in_2627(orig_dec[r2627[0] : r2627[1]], r2627[0])
    if slot0 is None or slot0[2] != 0x2616:
        raise ValueError(f"Donor track {donor_trk.name!r} does not have an occupied 0x2616 Slot A.")
    return orig_dec[slot0[0] : slot0[1]]


def _build_cloned_slot_a_2616(
    donor_2616: bytes,
    trim_info: SlotATrimInfo,
    clamped_db: float,
    *,
    activate_if_inactive: bool,
) -> tuple[bytes, bool]:
    """Patch a cloned ``0x2616`` Slot A chunk to *clamped_db*, activate if needed, and clear bypass."""
    buf = bytearray(donor_2616)
    r2613 = _find_child_zchunk_span(bytes(buf), 0, 0x2613)
    if r2613 is None:
        raise ValueError("Donor 0x2616 missing 0x2613 subchunk.")
    p2613, e2613 = r2613
    ancestors: list[int] = [0]
    for sub_15 in find_subchunks(bytes(buf), 0x2615):
        rel_15 = bytes(buf).find(sub_15, 9)
        if rel_15 != -1 and rel_15 < p2613 and rel_15 + len(sub_15) >= e2613:
            ancestors.append(rel_15)
            break
    ancestors.append(p2613)
    r1038 = _find_child_zchunk_span(bytes(buf[p2613:e2613]), p2613, 0x1038)
    if r1038 is None:
        raise ValueError("Donor 0x2616 missing 0x1038 plugin state.")
    p1038, e1038 = r1038
    new_1038 = _patch_1038_for_plugin(bytes(buf[p1038:e1038]), trim_info, clamped_db)
    delta = len(new_1038) - (e1038 - p1038)
    if delta != 0:
        _validate_and_shift_span_ancestors(buf, tuple(ancestors), p1038, e1038, delta)
    buf[p1038:e1038] = new_1038
    # Auto-inserted Slot A Trim with non-zero gain is always activated and unbypassed so it is never a silent no-op.
    activated = False
    if abs(clamped_db) >= 0.05 or activate_if_inactive:
        activated = _activate_slot_a_in_place(buf, p2613, len(buf), p1038 + len(new_1038))
    p13_end = p2613 + 7 + struct.unpack_from("<I", buf, p2613 + 3)[0]
    if p13_end < len(buf) and buf[p13_end] == 1:
        buf[p13_end] = 0
    return bytes(buf), activated


def _activate_slot_a_in_place(dec: bytearray, p2613: int, p2616_end: int, p1038_end: int) -> bool:
    """Flip both Pro Tools insert-active flags inside ``0x2613`` to active (``1``)."""
    if p2613 + 9 > p2616_end or dec[p2613] != 0x5A:
        return False
    p13_end = p2613 + 7 + struct.unpack_from("<I", dec, p2613 + 3)[0]
    if p13_end > p2616_end or p1038_end >= p13_end or dec[p1038_end] not in (0, 1):
        return False
    dec[p1038_end] = 1
    sub13 = bytes(dec[p2613:p13_end])
    s260e = find_subchunks(sub13, 0x260E)
    if s260e:
        rel_0e = sub13.find(s260e[0], 9)
        if rel_0e != -1:
            p0e_end = p2613 + rel_0e + len(s260e[0])
            if p0e_end < p13_end and dec[p0e_end] in (0, 1):
                dec[p0e_end] = 1
    return True


def _apply_one_slot_a_patch(
    dec: bytearray,
    c2624_off: int,
    trk_idx: int,
    trk_uuid_hex: str,
    trk_name: str,
    trim_info: SlotATrimInfo,
    clamped_db: float,
    *,
    activate_if_inactive: bool,
    unbypass_if_bypassed: bool,
) -> dict[str, Any]:
    loc = _locate_slot_a_offsets(dec, c2624_off, trk_idx, trk_uuid_hex)
    if loc is None:
        raise ValueError(f"Could not locate Slot A 0x1038 chunk on track {trk_name!r}.")
    b1038 = bytes(dec[loc.p1038 : loc.p1038 + loc.len1038])
    new_1038 = _patch_1038_for_plugin(b1038, trim_info, clamped_db)

    delta = len(new_1038) - loc.len1038
    if delta != 0:
        _validate_and_shift_span_ancestors(dec, loc.ancestors, loc.p1038, loc.p1038 + loc.len1038, delta)
        dec[loc.p1038 : loc.p1038 + loc.len1038] = new_1038
        shift_index_0002(dec, loc.p1038 + loc.len1038, delta)
    else:
        dec[loc.p1038 : loc.p1038 + loc.len1038] = new_1038

    p2616_end = loc.p2616_end + delta
    activated = False
    if activate_if_inactive and not trim_info.is_active and abs(clamped_db) >= 0.05:
        activated = _activate_slot_a_in_place(dec, loc.p2613, p2616_end, loc.p1038 + len(new_1038))
    unbypassed = False
    if unbypass_if_bypassed and trim_info.is_bypassed:
        p13_end = loc.p2613 + 7 + struct.unpack_from("<I", dec, loc.p2613 + 3)[0]
        if p13_end < p2616_end and dec[p13_end] == 1:
            dec[p13_end] = 0
            unbypassed = True
    return {
        "plugin": trim_info.plugin_name,
        "previous_db": trim_info.gain_db,
        "target_db": clamped_db,
        "activated": activated,
        "unbypassed": unbypassed,
        "auto_inserted": False,
    }


def _apply_one_slot_a_auto_insert(
    dec: bytearray,
    c2624_off: int,
    trk_idx: int,
    trk_uuid_hex: str,
    trk_name: str,
    donor_2616: bytes,
    donor_trim: SlotATrimInfo,
    clamped_db: float,
    *,
    activate_if_inactive: bool,
) -> dict[str, Any]:
    loc = _locate_slot_a_empty_2625(dec, c2624_off, trk_idx, trk_uuid_hex)
    if loc is None:
        raise ValueError(f"Track {trk_name!r} does not have an empty Insert Slot A (0x2625) for auto-insert.")
    cloned_2616, activated = _build_cloned_slot_a_2616(
        donor_2616, donor_trim, clamped_db, activate_if_inactive=activate_if_inactive,
    )
    delta = len(cloned_2616) - (loc.slot0_end - loc.slot0_start)
    _validate_and_shift_span_ancestors(dec, loc.ancestors, loc.slot0_start, loc.slot0_end, delta)
    dec[loc.slot0_start : loc.slot0_end] = cloned_2616
    if delta != 0:
        shift_index_0002(dec, loc.slot0_end, delta)
    return {
        "plugin": donor_trim.plugin_name,
        "previous_db": 0.0,
        "target_db": clamped_db,
        "activated": activated,
        "unbypassed": False,
        "auto_inserted": True,
    }


def _resolve_slot_a_targets(
    session: Any,
    track_gains_db: dict[str, float],
    *,
    allow_auto_insert: bool,
) -> list[tuple[int, str, str, SlotATrimInfo, float, Any | None]]:
    targets: list[tuple[int, str, str, SlotATrimInfo, float, Any | None]] = []
    seen_keys: set[str] = set()
    for name, raw_db in track_gains_db.items():
        if not math.isfinite(float(raw_db)):
            raise ValueError(f"Non-finite Slot A trim gain for track {name!r}: {raw_db!r}")
        trk = session.track_by_name(name)
        if trk is None:
            raise ValueError(f"Track {name!r} not found in .ptx session.")
        key = trk.uuid_hex or f"idx:{trk.index}"
        if key in seen_keys:
            raise ValueError(f"Duplicate Slot A trim target resolving to track {trk.name!r}.")
        seen_keys.add(key)
        donor_trk = None
        trim_info = trk.slot_a_trim
        if trim_info is None:
            slot_a_ins = next((ins for ins in trk.inserts if ins.slot_index == 0), None)
            if slot_a_ins is not None:
                plug = slot_a_ins.plugin_name or slot_a_ins.fourcc or "unknown plugin"
                raise ValueError(
                    f"Track {trk.name!r} does not have a validated Slot A Trim plugin "
                    f"(Insert Slot A is occupied by {plug!r})."
                )
            if not allow_auto_insert:
                raise ValueError(f"Track {trk.name!r} does not have a validated Slot A Trim plugin.")
            donor_trk = select_preferred_slot_a_trim_donor(session, trk.format)
            if donor_trk is None or donor_trk.slot_a_trim is None:
                raise ValueError(
                    f"Track {trk.name!r} ({trk.format}) has an empty Slot A, but no matching "
                    f"{trk.format} Slot A Trim donor exists in the session."
                )
            trim_info = donor_trk.slot_a_trim
        prec = 2 if trim_info.plugin_id == "trim" else 1
        clamped = _norm_zero(round(max(trim_info.min_db, min(trim_info.max_db, float(raw_db))), prec))
        targets.append((trk.index, trk.uuid_hex, trk.name, trim_info, clamped, donor_trk))
    targets.sort(key=lambda item: item[0], reverse=True)
    return targets


def patch_slot_a_trim_gains_on_session(
    raw_ptx: bytes,
    session: Any,
    track_gains_db: dict[str, float],
    *,
    activate_if_inactive: bool = True,
    unbypass_if_bypassed: bool = False,
    allow_auto_insert: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Patch (or auto-insert into empty Slot A) validated Slot A Trim gains on *session*."""
    if not track_gains_db:
        return raw_ptx, {}
    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None:
        raise ValueError("Session missing 0x2624 mixer chunk.")

    targets = _resolve_slot_a_targets(session, track_gains_db, allow_auto_insert=allow_auto_insert)
    dec_bytes, xor_delta = decrypt_ptx(raw_ptx)
    k0 = raw_ptx[0x14] ^ 0x5A
    donor_blobs: dict[str, bytes] = {}
    for _, _, _, _, _, donor_trk in targets:
        if donor_trk is not None:
            dkey = donor_trk.uuid_hex or f"idx:{donor_trk.index}"
            if dkey not in donor_blobs:
                donor_blobs[dkey] = _extract_pristine_donor_2616(dec_bytes, c2624.offset, donor_trk)

    dec = bytearray(dec_bytes)
    meta: dict[str, Any] = {}
    for trk_idx, trk_uuid_hex, trk_name, trim_info, clamped_db, donor_trk in targets:
        if donor_trk is not None:
            dkey = donor_trk.uuid_hex or f"idx:{donor_trk.index}"
            meta[trk_name] = _apply_one_slot_a_auto_insert(
                dec, c2624.offset, trk_idx, trk_uuid_hex, trk_name,
                donor_blobs[dkey], trim_info, clamped_db,
                activate_if_inactive=activate_if_inactive,
            )
        else:
            meta[trk_name] = _apply_one_slot_a_patch(
                dec, c2624.offset, trk_idx, trk_uuid_hex, trk_name, trim_info, clamped_db,
                activate_if_inactive=activate_if_inactive,
                unbypass_if_bypassed=unbypass_if_bypassed,
            )

    validate_index_0002(bytes(dec))
    return encrypt_ptx(bytes(dec), xor_delta, k0=k0), meta
