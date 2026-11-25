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
import json
import logging
import math
import re
import struct
from typing import Any
import zlib

from .codec import decrypt_ptx, encrypt_ptx, find_subchunks, shift_index_0002, validate_index_0002
from .models import PtxPluginInfo, SlotATrimInfo

log = logging.getLogger(__name__)

_MAX_DGCT_JSON_BYTES = 2 * 1024 * 1024
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
        min_db=-200.0,
        max_db=18.0,
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
        min_db=-60.0,
        max_db=24.0,
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

_PROTILITY_ACTIVE_SLOT_RE = re.compile(r'activeSlot="([AB])"')
_PROTILITY_GAIN_RE = re.compile(r'(<PARAM\s+id="GAIN"\s+value=")([^"]*)(")')
_PROTILITY_MODIFIED_RE = re.compile(r'PRESET_MODIFIED="0"')


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


def _f32_round(val: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", val))[0])


def _norm_zero(val: float) -> float:
    return 0.0 if abs(val) < 1e-9 else val


def _decompress_bounded(data: bytes, max_bytes: int = _MAX_DGCT_JSON_BYTES) -> bytes:
    """Decompress *data* enforcing a strict *max_bytes* ceiling against decompression bombs."""
    decomp = zlib.decompressobj()
    out = decomp.decompress(data, max_bytes + 1)
    if len(out) > max_bytes or not decomp.eof:
        raise ValueError(f"Compressed plugin payload exceeds {max_bytes} byte safety limit.")
    return out


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


def _find_protility_active_slot_span(xml_str: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` character span of Protility's active ``<slotA>`` or ``<slotB>`` block."""
    slot_match = _PROTILITY_ACTIVE_SLOT_RE.search(xml_str)
    slot_tag = f"slot{slot_match.group(1)}" if slot_match else "slotA"
    m = re.search(rf"<{slot_tag}\b[^>]*>.*?</{slot_tag}>", xml_str, flags=re.DOTALL)
    return (m.start(), m.end()) if m else None


def _read_trackcontrol_db(b1038: bytes, expected_tag: bytes) -> float | None:
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
        raw_json = _decompress_bounded(b1038[zpos + 4 : z_end])
        js = json.loads(raw_json.rstrip(b"\x00 ").decode("utf-8"))
        if not isinstance(js, dict):
            return None
        settings = js.get("settings")
        cur_ab = int(settings.get("currentAB", 0)) if isinstance(settings, dict) else 0
        ab_list = js.get("ab")
        if isinstance(ab_list, list) and 0 <= cur_ab < len(ab_list) and isinstance(ab_list[cur_ab], dict):
            params = ab_list[cur_ab].get("parameters")
            if params == "default":
                return 0.0
            if isinstance(params, dict) and "kOpGain" in params:
                val = float(params["kOpGain"])
                return _norm_zero(round(val * 218.0 - 200.0, 2)) if math.isfinite(val) else None
        elck_val = struct.unpack_from(">d", b1038, ep + 8)[0]
        return _norm_zero(round(elck_val * 218.0 - 200.0, 2)) if math.isfinite(elck_val) else None
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


def _read_protility_db(b1038: bytes, expected_tag: bytes) -> float | None:
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
    try:
        xml_str = b1038[xml_start : xml_start + xml_len].rstrip(b"\x00").decode("utf-8")
    except UnicodeDecodeError:
        return None
    span = _find_protility_active_slot_span(xml_str)
    if span is None:
        return None
    match = _PROTILITY_GAIN_RE.search(xml_str[span[0] : span[1]])
    if not match:
        return None
    try:
        val = float(match.group(2))
    except ValueError:
        return None
    return _norm_zero(round(val, 2)) if math.isfinite(val) else None


def inspect_slot_a_trim(
    slot_blk: bytes,
    pinfo: PtxPluginInfo,
    *,
    is_active: bool,
    is_bypassed: bool,
) -> SlotATrimInfo | None:
    """Validate Slot A plugin identity and state blob, returning decoded gain info or ``None``."""
    spec = match_slot_a_trim_spec(pinfo)
    if spec is None:
        return None
    s1038 = find_subchunks(slot_blk, 0x1038)
    if not s1038:
        return None
    b1038 = s1038[0]
    expected_tag = f"{pinfo.manufacturer_4cc}{pinfo.product_4cc}{pinfo.type_4cc}".encode("latin1")
    if spec.plugin_id == "trackcontrol":
        gain_db = _read_trackcontrol_db(b1038, expected_tag)
    elif spec.plugin_id == "trim":
        gain_db = _read_avid_trim_db(b1038, expected_tag)
    elif spec.plugin_id == "protility":
        gain_db = _read_protility_db(b1038, expected_tag)
    else:
        return None
    if gain_db is None:
        return None
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
    )


def _update_trackcontrol_json(js: dict[str, Any], norm_rounded: float) -> None:
    settings = js.get("settings")
    cur_ab = int(settings.get("currentAB", 0)) if isinstance(settings, dict) else 0
    ab_list = js.get("ab")
    if isinstance(ab_list, list) and 0 <= cur_ab < len(ab_list) and isinstance(ab_list[cur_ab], dict):
        params = ab_list[cur_ab].get("parameters")
        if not isinstance(params, dict):
            params = {}
        params["kOpGain"] = norm_rounded
        ab_list[cur_ab]["parameters"] = params
    presets = js.get("presets")
    if isinstance(presets, list) and len(presets) == 1 and isinstance(presets[0], dict):
        p_params = presets[0].get("parameters")
        if not isinstance(p_params, dict):
            p_params = {}
        p_params["kOpGain"] = norm_rounded
        presets[0]["parameters"] = p_params


def _patch_trackcontrol_1038(b1038: bytes, gain_db: float, expected_tag: bytes) -> bytes:
    if len(b1038) < 21 or expected_tag not in b1038[:64]:
        raise ValueError("TrackControl 0x1038 chunk missing expected AAX header.")
    old_inner_len = struct.unpack_from("<I", b1038, 13)[0]
    if 17 + old_inner_len > len(b1038):
        raise ValueError("TrackControl 0x1038 inner length out of bounds.")
    buf = bytearray(b1038)
    norm = _f32_round((gain_db + 200.0) / 218.0)
    dgct = buf.find(b"DGCT")
    ep = buf.find(b"d_2dl \x00\x00", 0, dgct) if dgct >= 20 else -1
    if ep == -1 or ep + 16 > dgct:
        raise ValueError("TrackControl 0x1038 chunk missing expected elck/DGCT structure.")
    struct.pack_into(">d", buf, ep + 8, norm)

    chunk_start = dgct - 20
    old_chunk_len = struct.unpack_from(">I", buf, chunk_start)[0]
    zpos = buf.find(b"ZIP ", dgct)
    z_end = chunk_start + old_chunk_len
    if zpos == -1 or z_end > len(buf) or zpos + 4 >= z_end:
        raise ValueError("TrackControl DGCT ZIP payload out of bounds.")
    raw_json = _decompress_bounded(bytes(buf[zpos + 4 : z_end]))
    trailing = raw_json[len(raw_json.rstrip(b"\x00 ")) :]
    js = json.loads(raw_json.rstrip(b"\x00 ").decode("utf-8"))
    if not isinstance(js, dict):
        raise ValueError("TrackControl DGCT JSON root is not an object.")
    _update_trackcontrol_json(js, round(norm, 10))

    new_zip = zlib.compress(json.dumps(js, indent="\t").encode("utf-8") + trailing)
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
    norm = _f32_round((gain_db + 60.0) / 84.0)
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
    xml_bytes = bytes(buf[xml_start : xml_start + old_xml_len])
    had_nul = xml_bytes.endswith(b"\x00")
    xml_str = xml_bytes.rstrip(b"\x00").decode("utf-8")
    span = _find_protility_active_slot_span(xml_str)
    if span is None:
        raise ValueError("Protility XML missing active slot element.")
    slot_xml = xml_str[span[0] : span[1]]
    val_str = f"{_norm_zero(gain_db):.1f}"
    new_slot_xml, count = _PROTILITY_GAIN_RE.subn(rf"\g<1>{val_str}\g<3>", slot_xml, count=1)
    if count != 1:
        raise ValueError("Protility active slot XML missing <PARAM id='GAIN'> element.")
    new_slot_xml = _PROTILITY_MODIFIED_RE.sub('PRESET_MODIFIED="1"', new_slot_xml, count=1)
    new_xml_str = xml_str[: span[0]] + new_slot_xml + xml_str[span[1] :]
    new_xml_bytes = new_xml_str.encode("utf-8") + (b"\x00" if had_nul else b"")
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


def _extract_slot_a_from_block(
    dec: bytearray,
    c2624_off: int,
    blk_start: int,
    blk_end: int,
) -> _SlotALocation | None:
    """Walk the exact Z-chunk hierarchy ``c2624 -> blk -> [261B] -> 2627 -> 2616 -> [2615] -> 2613 -> 1038``."""
    blk = bytes(dec[blk_start:blk_end])
    ancestors: list[int] = [c2624_off, blk_start]

    r2627 = _find_child_zchunk_span(blk, blk_start, 0x2627)
    if r2627 is None:
        return None
    p2627, e2627 = r2627
    # Optional enclosing 0x261B wrapper inside blk containing 0x2627
    for sub_1b in find_subchunks(blk, 0x261B):
        rel_1b = blk.find(sub_1b, 9)
        if rel_1b != -1 and blk_start + rel_1b < p2627 and blk_start + rel_1b + len(sub_1b) >= e2627:
            ancestors.append(blk_start + rel_1b)
            break
    ancestors.append(p2627)

    b2627 = bytes(dec[p2627:e2627])
    r2616 = _find_child_zchunk_span(b2627, p2627, 0x2616)
    if r2616 is None:
        return None
    p2616, e2616 = r2616
    ancestors.append(p2616)

    b2616 = bytes(dec[p2616:e2616])
    r2613 = _find_child_zchunk_span(b2616, p2616, 0x2613)
    if r2613 is None:
        return None
    p2613, e2613 = r2613
    # Optional enclosing 0x2615 wrapper inside 0x2616 containing 0x2613
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


def _locate_slot_a_offsets(
    dec: bytearray,
    c2624_off: int,
    track_index: int,
    track_uuid_hex: str = "",
) -> _SlotALocation | None:
    """Locate Slot A structurally inside ``0x2624`` (UUID-first; index fallback only when no UUID)."""
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
                return _extract_slot_a_from_block(dec, c2624_off, pos, end)
        elif ti == track_index:
            fallback_bounds = (pos, end)
        pos = end
    return _extract_slot_a_from_block(dec, c2624_off, *fallback_bounds) if fallback_bounds else None


def _validate_and_shift_ancestors(dec: bytearray, loc: _SlotALocation, delta: int) -> None:
    """Verify strict Z-chunk nesting for *loc.ancestors* and adjust all ancestor lengths by *delta*."""
    target_s, target_e = loc.p1038, loc.p1038 + loc.len1038
    prev_start, prev_end = -1, len(dec) + 1
    new_sizes: list[tuple[int, int]] = []
    for anc in loc.ancestors:
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
    expected_tag = trim_info.fourcc.replace("/", "").encode("latin1")
    b1038 = bytes(dec[loc.p1038 : loc.p1038 + loc.len1038])
    if trim_info.plugin_id == "trackcontrol":
        new_1038 = _patch_trackcontrol_1038(b1038, clamped_db, expected_tag)
    elif trim_info.plugin_id == "trim":
        new_1038 = _patch_avid_trim_1038(b1038, clamped_db, expected_tag)
    elif trim_info.plugin_id == "protility":
        new_1038 = _patch_protility_1038(b1038, clamped_db, expected_tag)
    else:
        raise ValueError(f"Unsupported Slot A Trim plugin ID: {trim_info.plugin_id!r}")

    delta = len(new_1038) - loc.len1038
    if delta != 0:
        _validate_and_shift_ancestors(dec, loc, delta)
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
    }


def patch_slot_a_trim_gains_on_session(
    raw_ptx: bytes,
    session: Any,
    track_gains_db: dict[str, float],
    *,
    activate_if_inactive: bool = True,
    unbypass_if_bypassed: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Patch validated Slot A Trim plugin gains using an already-parsed ``PtxSession``."""
    if not track_gains_db:
        return raw_ptx, {}
    c2624 = session.chunk_map.get(0x2624)
    if c2624 is None:
        raise ValueError("Session missing 0x2624 mixer chunk.")

    targets: list[tuple[int, str, str, SlotATrimInfo, float]] = []
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
        if trk.slot_a_trim is None:
            raise ValueError(f"Track {name!r} does not have a validated Slot A Trim plugin.")
        clamped = _norm_zero(round(max(trk.slot_a_trim.min_db, min(trk.slot_a_trim.max_db, float(raw_db))), 1))
        targets.append((trk.index, trk.uuid_hex, trk.name, trk.slot_a_trim, clamped))

    targets.sort(key=lambda item: item[0], reverse=True)
    dec_bytes, xor_delta = decrypt_ptx(raw_ptx)
    k0 = raw_ptx[0x14] ^ 0x5A
    dec = bytearray(dec_bytes)
    meta: dict[str, Any] = {}
    for trk_idx, trk_uuid_hex, trk_name, trim_info, clamped_db in targets:
        meta[trk_name] = _apply_one_slot_a_patch(
            dec, c2624.offset, trk_idx, trk_uuid_hex, trk_name, trim_info, clamped_db,
            activate_if_inactive=activate_if_inactive,
            unbypass_if_bypassed=unbypass_if_bypassed,
        )

    validate_index_0002(bytes(dec))
    return encrypt_ptx(bytes(dec), xor_delta, k0=k0), meta
