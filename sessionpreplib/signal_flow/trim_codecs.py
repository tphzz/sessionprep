"""Format-agnostic payload codecs for shared Trim plugins (TrackControl, Protility) and donor selection."""

from __future__ import annotations

from collections.abc import Iterable
import json
import math
import re
import struct
from typing import Any
import zlib

MAX_PLUGIN_PAYLOAD_BYTES = 2 * 1024 * 1024

TRACKCONTROL_MIN_DB = -200.0
TRACKCONTROL_MAX_DB = 18.0
TRACKCONTROL_SPAN_DB = TRACKCONTROL_MAX_DB - TRACKCONTROL_MIN_DB

PROTILITY_MIN_DB = -60.0
PROTILITY_MAX_DB = 24.0
PROTILITY_SPAN_DB = PROTILITY_MAX_DB - PROTILITY_MIN_DB

TRIM_PLUGIN_TIE_ORDER: dict[str, int] = {
    "trackcontrol": 0,
    "trim": 1,
    "protility": 2,
}

_PROTILITY_ACTIVE_SLOT_RE = re.compile(r'activeSlot="([AB])"')
_PROTILITY_GAIN_RE = re.compile(r'(<PARAM\s+id="GAIN"\s+value=")([^"]*)(")')
_PROTILITY_PARAM_RE = re.compile(r'<PARAM\s+id="([^"]+)"\s+value="([^"]*)"')
_PROTILITY_MODIFIED_RE = re.compile(r'PRESET_MODIFIED="0"')

PROTILITY_CLEAN_DEFAULTS: dict[str, float] = {
    "BASS_MONO_ENABLED": 0.0,
    "BASS_MONO_FREQ": 10.0,
    "DUAL_PAN_L": -100.0,
    "DUAL_PAN_R": 100.0,
    "GAIN_L": 0.0,
    "GAIN_MODE": 0.0,
    "GAIN_R": 0.0,
    "HP_ENABLED": 0.0,
    "HP_FREQ": 10.0,
    "HP_SLOPE": 1.0,
    "MID_GAIN": 100.0,
    "MONO_MODE": 0.0,
    "PAN": 0.0,
    "PAN_MODE": 0.0,
    "PHASE_INVERT_L": 0.0,
    "PHASE_INVERT_LINK": 1.0,
    "PHASE_INVERT_R": 0.0,
    "PHASE_SHIFT": 10.0,
    "PHASE_SHIFT_ENABLED": 0.0,
    "SIDE_GAIN": 100.0,
    "STEREO_MODE": 0.0,
    "TIMING_ENABLED": 0.0,
    "TIMING_LINK": 1.0,
    "TIMING_MS_L": 0.0,
    "TIMING_MS_R": 0.0,
    "TIMING_SAMPLES_L": 0.0,
    "TIMING_SAMPLES_R": 0.0,
    "TIMING_TYPE": 0.0,
    "TRIM": 100.0,
    "WIDTH": 100.0,
    "WIDTH_ENABLED": 1.0,
    "WIDTH_MODE": 0.0,
    "BALANCE_LAW": 0.0,
    "PAN_LAW": 1.0,
}


def f32_round(val: float) -> float:
    """Round *val* to IEEE-754 single precision."""
    return float(struct.unpack("<f", struct.pack("<f", val))[0])


def norm_zero(val: float) -> float:
    """Normalize tiny signed zero values to ``0.0``."""
    return 0.0 if abs(val) < 1e-9 else val


def decompress_bounded(data: bytes, max_bytes: int = MAX_PLUGIN_PAYLOAD_BYTES) -> bytes:
    """Decompress zlib *data* while enforcing *max_bytes* against decompression bombs."""
    decomp = zlib.decompressobj()
    out = decomp.decompress(data, max_bytes + 1)
    if len(out) > max_bytes or not decomp.eof:
        raise ValueError(f"Compressed plugin payload exceeds {max_bytes} byte safety limit.")
    return out


def _validate_clamped_db(gain_db: float, lo: float, hi: float, plugin_label: str) -> float:
    if isinstance(gain_db, bool):
        raise ValueError(f"Non-numeric {plugin_label} gain value: {gain_db!r}")
    val = float(gain_db)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite {plugin_label} gain value: {gain_db!r}")
    return max(lo, min(hi, val))


def trackcontrol_db_to_norm(gain_db: float) -> float:
    """Convert TrackControl gain in ``[-200.0, +18.0] dB`` to normalized IEEE-754 float32."""
    clamped = _validate_clamped_db(gain_db, TRACKCONTROL_MIN_DB, TRACKCONTROL_MAX_DB, "TrackControl")
    return f32_round((clamped - TRACKCONTROL_MIN_DB) / TRACKCONTROL_SPAN_DB)


def trackcontrol_norm_to_db(norm: float) -> float:
    """Convert normalized TrackControl ``kOpGain`` back to dB (2-decimal rounded)."""
    return norm_zero(round(float(norm) * TRACKCONTROL_SPAN_DB + TRACKCONTROL_MIN_DB, 2))


def decode_trackcontrol_dgct_zip(compressed_json: bytes) -> tuple[dict[str, Any], bytes]:
    """Decompress and parse a TrackControl ``DGCT`` zlib JSON payload, returning ``(obj, trailing_bytes)``."""
    raw_json = decompress_bounded(compressed_json)
    stripped = raw_json.rstrip(b"\x00 ")
    trailing = raw_json[len(stripped) :]
    js = json.loads(stripped.decode("utf-8"))
    if not isinstance(js, dict):
        raise ValueError("TrackControl DGCT JSON root is not an object.")
    return js, trailing


def extract_trackcontrol_gain(js: dict[str, Any]) -> tuple[float, bool] | None:
    """Return ``(gain_db, is_clean)`` from parsed TrackControl JSON, or ``None`` if ``kOpGain`` is absent.

    Raises ``ValueError`` if ``kOpGain`` is present but non-finite or non-numeric.
    """
    settings = js.get("settings")
    cur_ab = int(settings.get("currentAB", 0)) if isinstance(settings, dict) else 0
    ab_list = js.get("ab")
    if isinstance(ab_list, list) and 0 <= cur_ab < len(ab_list) and isinstance(ab_list[cur_ab], dict):
        params = ab_list[cur_ab].get("parameters")
        if params == "default":
            return 0.0, (cur_ab == 0)
        if isinstance(params, dict) and "kOpGain" in params:
            val = float(params["kOpGain"])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite TrackControl kOpGain value: {val!r}")
            clean = cur_ab == 0 and set(params.keys()) <= {"kOpGain"}
            return trackcontrol_norm_to_db(val), clean
    return None


def update_trackcontrol_json(js: dict[str, Any], norm_rounded: float) -> None:
    """Update TrackControl's active A/B slot (and single-preset entry if present) with *norm_rounded*."""
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


def patch_trackcontrol_dgct_zip(compressed_json: bytes, gain_db: float) -> tuple[bytes, float]:
    """Patch a TrackControl ``DGCT`` zlib JSON payload to *gain_db*, returning ``(new_zip_bytes, norm)``."""
    norm = trackcontrol_db_to_norm(gain_db)
    js, trailing = decode_trackcontrol_dgct_zip(compressed_json)
    update_trackcontrol_json(js, round(norm, 10))
    new_zip = zlib.compress(json.dumps(js, indent="\t", allow_nan=False).encode("utf-8") + trailing)
    return new_zip, norm


def protility_db_to_norm(gain_db: float) -> float:
    """Convert Protility gain in ``[-60.0, +24.0] dB`` to normalized IEEE-754 float32."""
    clamped = _validate_clamped_db(gain_db, PROTILITY_MIN_DB, PROTILITY_MAX_DB, "Protility")
    return f32_round((clamped - PROTILITY_MIN_DB) / PROTILITY_SPAN_DB)


def _find_protility_active_slot_span(xml_str: str) -> tuple[int, int] | None:
    slot_match = _PROTILITY_ACTIVE_SLOT_RE.search(xml_str)
    slot_tag = f"slot{slot_match.group(1)}" if slot_match else "slotA"
    m = re.search(rf"<{slot_tag}\b[^>]*>.*?</{slot_tag}>", xml_str, flags=re.DOTALL)
    return (m.start(), m.end()) if m else None


def _is_protility_slot_clean(xml_str: str, slot_xml: str) -> bool:
    slot_match = _PROTILITY_ACTIVE_SLOT_RE.search(xml_str)
    if slot_match and slot_match.group(1) != "A":
        return False
    found_any = False
    for pid, val_s in _PROTILITY_PARAM_RE.findall(slot_xml):
        found_any = True
        if pid == "GAIN":
            continue
        expected = PROTILITY_CLEAN_DEFAULTS.get(pid)
        if expected is None:
            return False
        try:
            if abs(float(val_s) - expected) > 1e-3:
                return False
        except ValueError:
            return False
    return found_any


def read_protility_vc2_xml(xml_bytes: bytes) -> tuple[float, bool] | None:
    """Parse a Protility JUCE ``VC2!`` XML payload, returning ``(gain_db, is_clean)`` or ``None``."""
    try:
        xml_str = xml_bytes.rstrip(b"\x00").decode("utf-8")
    except UnicodeDecodeError:
        return None
    span = _find_protility_active_slot_span(xml_str)
    if span is None:
        return None
    slot_xml = xml_str[span[0] : span[1]]
    match = _PROTILITY_GAIN_RE.search(slot_xml)
    if not match:
        return None
    try:
        val = float(match.group(2))
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return norm_zero(round(val, 2)), _is_protility_slot_clean(xml_str, slot_xml)


def patch_protility_vc2_xml(xml_bytes: bytes, gain_db: float) -> bytes:
    """Patch a Protility JUCE ``VC2!`` XML payload to *gain_db*, preserving trailing NUL framing."""
    clamped = _validate_clamped_db(gain_db, PROTILITY_MIN_DB, PROTILITY_MAX_DB, "Protility")
    had_nul = xml_bytes.endswith(b"\x00")
    xml_str = xml_bytes.rstrip(b"\x00").decode("utf-8")
    span = _find_protility_active_slot_span(xml_str)
    if span is None:
        raise ValueError("Protility XML missing active slot element.")
    slot_xml = xml_str[span[0] : span[1]]
    val_str = f"{norm_zero(clamped):.1f}"
    new_slot_xml, count = _PROTILITY_GAIN_RE.subn(rf"\g<1>{val_str}\g<3>", slot_xml, count=1)
    if count != 1:
        raise ValueError("Protility active slot XML missing <PARAM id='GAIN'> element.")
    new_slot_xml = _PROTILITY_MODIFIED_RE.sub('PRESET_MODIFIED="1"', new_slot_xml, count=1)
    new_xml_str = xml_str[: span[0]] + new_slot_xml + xml_str[span[1] :]
    return new_xml_str.encode("utf-8") + (b"\x00" if had_nul else b"")


def select_preferred_trim_donor(tracks: Iterable[Any], track_format: str) -> Any | None:
    """Return the cleanest verified donor track for the most-used Slot A Trim matching *track_format*."""
    if track_format not in ("Mono", "Stereo"):
        return None
    track_list = list(tracks)
    clean_candidates = [
        trk for trk in track_list
        if trk.slot_a_trim is not None and trk.slot_a_trim.is_clean_donor and trk.format == track_format
    ]
    if not clean_candidates:
        return None
    total_counts: dict[str, int] = {}
    for trk in track_list:
        if trk.slot_a_trim is not None and trk.format == track_format:
            pid = trk.slot_a_trim.plugin_id
            total_counts[pid] = total_counts.get(pid, 0) + 1
    avail_pids = {trk.slot_a_trim.plugin_id for trk in clean_candidates}
    best_pid = min(
        avail_pids,
        key=lambda pid: (-total_counts.get(pid, 0), TRIM_PLUGIN_TIE_ORDER.get(pid, 99), pid),
    )
    matching = [trk for trk in clean_candidates if trk.slot_a_trim.plugin_id == best_pid]
    matching.sort(
        key=lambda t: (
            t.slot_a_trim.is_bypassed,
            abs(t.slot_a_trim.gain_db) >= 0.05,
            getattr(t, "index", 0),
        )
    )
    return matching[0]
