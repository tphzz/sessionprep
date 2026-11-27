"""Pure-Python VST3 ``.vstpreset`` parser/writer and DAWproject Slot 1 Trim donor helper.

Supports VST3 plugins only (VST2 is intentionally excluded as legacy).
Matches verified cross-DAW Trim plugins (DMG Audio ``TrackControl`` and Process.Audio
``Protility``) from existing ``.dawproject`` template donors, clones their exact
``<Vst3Plugin>`` XML metadata and ``.vstpreset`` state blob into Slot 1 of newly
created tracks or busses, and provides a safe gain-patching hook for calibrated
``Comp`` chunk codecs (``DGCT`` zlib-JSON and JUCE ``VC2!`` XML).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import struct
from typing import Any
import zlib

from ..signal_flow.models import TrimPluginInfo
from ..signal_flow.trim_codecs import (
    MAX_PLUGIN_PAYLOAD_BYTES,
    PROTILITY_MAX_DB,
    PROTILITY_MIN_DB,
    TRACKCONTROL_MAX_DB,
    TRACKCONTROL_MIN_DB,
    TRIM_PLUGIN_TIE_ORDER,
    decode_trackcontrol_dgct_zip,
    extract_trackcontrol_gain,
    patch_protility_vc2_xml,
    patch_trackcontrol_dgct_zip,
    read_protility_vc2_xml,
)

_VST3_MAGIC = b"VST3"
_VST3_LIST_MAGIC = b"List"
_VST3_HEADER_SIZE = 48
_VST3_ENTRY_SIZE = 20
_MAX_VSTPRESET_BYTES = MAX_PLUGIN_PAYLOAD_BYTES * 2


@dataclass(frozen=True)
class Vst3TrimSpec:
    """Identity and gain range for a supported VST3 Slot 1 Trim plugin."""

    plugin_id: str
    display_name: str
    name_tokens: tuple[str, ...]
    vendor_tokens: tuple[str, ...]
    min_db: float
    max_db: float


VST3_TRIM_SPECS: tuple[Vst3TrimSpec, ...] = (
    Vst3TrimSpec(
        plugin_id="trackcontrol",
        display_name="TrackControl",
        name_tokens=("trackcontrol",),
        vendor_tokens=("dmg audio", "dmgaudio", "dmga"),
        min_db=TRACKCONTROL_MIN_DB,
        max_db=TRACKCONTROL_MAX_DB,
    ),
    Vst3TrimSpec(
        plugin_id="protility",
        display_name="Protility",
        name_tokens=("protility",),
        vendor_tokens=("process.audio", "process audio", "prau"),
        min_db=PROTILITY_MIN_DB,
        max_db=PROTILITY_MAX_DB,
    ),
)


@dataclass(frozen=True)
class VstPresetChunk:
    """One named binary chunk (e.g. ``b'Comp'``, ``b'Cont'``, ``b'Info'``) inside a ``.vstpreset``."""

    chunk_id: bytes
    data: bytes


@dataclass(frozen=True)
class VstPresetData:
    """Parsed Steinberg ``.vstpreset`` binary container."""

    version: int
    class_id: str
    chunks: tuple[VstPresetChunk, ...]


@dataclass(frozen=True)
class Vst3TrimDonor:
    """VST3 Trim plugin template donor extracted from a ``.dawproject`` archive."""

    plugin_id: str
    plugin_name: str
    device_id: str
    device_vendor: str
    plugin_version: str
    device_role: str
    track_format: str
    state_path: str
    state_bytes: bytes | None
    trim_info: TrimPluginInfo
    source_channel_id: str
    source_index: int = 0
    device_name: str = ""
    is_calibrated: bool = False
    has_valid_state: bool = False


def _local_tag(tag: Any) -> str:
    text = str(tag or "")
    return text.rsplit("}", 1)[-1] if "}" in text else text


def parse_vstpreset(raw: bytes) -> VstPresetData:
    """Parse a Steinberg ``.vstpreset`` binary stream, raising ``ValueError`` on malformed input."""
    if len(raw) < _VST3_HEADER_SIZE or len(raw) > _MAX_VSTPRESET_BYTES:
        raise ValueError("Invalid .vstpreset size.")
    if raw[:4] != _VST3_MAGIC:
        raise ValueError("Missing 'VST3' magic header in .vstpreset.")
    version = struct.unpack_from("<i", raw, 4)[0]
    try:
        class_id = raw[8:40].decode("ascii").rstrip("\x00 ")
    except UnicodeDecodeError as exc:
        raise ValueError("Non-ASCII VST3 class ID in .vstpreset header.") from exc
    list_offset = struct.unpack_from("<q", raw, 40)[0]
    if list_offset < _VST3_HEADER_SIZE or list_offset + 8 > len(raw):
        raise ValueError("VST3 chunk list offset out of bounds.")
    if raw[list_offset : list_offset + 4] != _VST3_LIST_MAGIC:
        raise ValueError("Missing 'List' chunk directory in .vstpreset.")
    count = struct.unpack_from("<i", raw, list_offset + 4)[0]
    if count < 0 or list_offset + 8 + count * _VST3_ENTRY_SIZE > len(raw):
        raise ValueError("VST3 chunk list entry count out of bounds.")
    chunks: list[VstPresetChunk] = []
    pos = list_offset + 8
    total_chunk_bytes = 0
    for _ in range(count):
        cid = raw[pos : pos + 4]
        offset, length = struct.unpack_from("<qq", raw, pos + 4)
        pos += _VST3_ENTRY_SIZE
        if offset < _VST3_HEADER_SIZE or length < 0 or offset + length > len(raw):
            raise ValueError(f"VST3 chunk {cid!r} bounds out of range.")
        total_chunk_bytes += length
        if total_chunk_bytes > _MAX_VSTPRESET_BYTES:
            raise ValueError("VST3 preset cumulative chunk size exceeds limit.")
        chunks.append(VstPresetChunk(chunk_id=cid, data=raw[offset : offset + length]))
    return VstPresetData(version=version, class_id=class_id, chunks=tuple(chunks))


def build_vstpreset(preset: VstPresetData) -> bytes:
    """Serialize *preset* into a valid Steinberg ``.vstpreset`` binary stream."""
    cid_bytes = preset.class_id.encode("ascii")[:32].ljust(32, b"\x00")
    data_parts: list[bytes] = []
    entries: list[tuple[bytes, int, int]] = []
    cursor = _VST3_HEADER_SIZE
    for ch in preset.chunks:
        if len(ch.chunk_id) != 4:
            raise ValueError(f"VST3 chunk ID must be 4 bytes, got {ch.chunk_id!r}.")
        entries.append((ch.chunk_id, cursor, len(ch.data)))
        data_parts.append(ch.data)
        cursor += len(ch.data)
    header = _VST3_MAGIC + struct.pack("<i", preset.version) + cid_bytes + struct.pack("<q", cursor)
    list_hdr = _VST3_LIST_MAGIC + struct.pack("<i", len(entries))
    list_body = b"".join(cid + struct.pack("<qq", off, ln) for cid, off, ln in entries)
    return header + b"".join(data_parts) + list_hdr + list_body


def extract_vstpreset_chunk(preset: VstPresetData, chunk_id: bytes = b"Comp") -> bytes | None:
    """Return payload bytes for *chunk_id* (default ``b'Comp'``), or ``None`` if absent."""
    for ch in preset.chunks:
        if ch.chunk_id == chunk_id:
            return ch.data
    return None


def replace_vstpreset_chunk(preset: VstPresetData, chunk_id: bytes, new_data: bytes) -> VstPresetData:
    """Return a copy of *preset* with *chunk_id* replaced by *new_data* (raises if absent)."""
    replaced = False
    out: list[VstPresetChunk] = []
    for ch in preset.chunks:
        if ch.chunk_id == chunk_id and not replaced:
            out.append(VstPresetChunk(chunk_id=chunk_id, data=new_data))
            replaced = True
        else:
            out.append(ch)
    if not replaced:
        raise ValueError(f"VST3 preset missing required {chunk_id!r} chunk.")
    return VstPresetData(version=preset.version, class_id=preset.class_id, chunks=tuple(out))


def match_vst3_trim_spec(device_name: str, device_vendor: str = "") -> Vst3TrimSpec | None:
    """Match a ``<Vst3Plugin>`` strictly by known Trim plugin name and vendor."""
    name_low = (device_name or "").strip().lower()
    if not name_low:
        return None
    vendor_low = (device_vendor or "").strip().lower()
    for spec in VST3_TRIM_SPECS:
        if any(tok in name_low for tok in spec.name_tokens):
            if not vendor_low or any(v in vendor_low for v in spec.vendor_tokens):
                return spec
    return None


def _has_trailing_zlib_bytes(compressed: bytes) -> bool:
    """Return True when *compressed* is malformed or has unexpected bytes trailing the zlib stream."""
    decomp = zlib.decompressobj()
    try:
        out = decomp.decompress(compressed, MAX_PLUGIN_PAYLOAD_BYTES + 1)
    except zlib.error:
        return True
    if len(out) > MAX_PLUGIN_PAYLOAD_BYTES or not decomp.eof:
        return True
    return bool(decomp.unused_data)


def _decode_comp_trim_state(plugin_id: str, comp_data: bytes) -> tuple[float, bool] | None:
    """Inspect a VST3 ``Comp`` chunk for known ``DGCT`` (TrackControl) or ``VC2!`` (Protility) payloads."""
    if plugin_id == "trackcontrol":
        dgct = comp_data.find(b"DGCT")
        zpos = comp_data.find(b"ZIP ", dgct) if dgct != -1 else -1
        if zpos != -1 and zpos + 4 < len(comp_data):
            zslice = comp_data[zpos + 4 :]
            if _has_trailing_zlib_bytes(zslice):
                return None
            try:
                js, _trailing = decode_trackcontrol_dgct_zip(zslice)
                return extract_trackcontrol_gain(js)
            except Exception:
                return None
    elif plugin_id == "protility":
        vc2 = comp_data.find(b"VC2!")
        if vc2 != -1 and vc2 + 8 <= len(comp_data):
            xml_len = struct.unpack_from("<I", comp_data, vc2 + 4)[0]
            xml_start = vc2 + 8
            if xml_start + xml_len <= len(comp_data):
                return read_protility_vc2_xml(comp_data[xml_start : xml_start + xml_len])
    return None


def patch_vst3_trim_preset(raw_preset: bytes, plugin_id: str, gain_db: float) -> bytes | None:
    """Patch a calibrated VST3 ``.vstpreset`` ``Comp`` chunk to *gain_db*, or return ``None`` if uncalibrated."""
    if not math.isfinite(float(gain_db)):
        raise ValueError(f"Non-finite VST3 Trim gain: {gain_db!r}")
    preset = parse_vstpreset(raw_preset)
    comp = extract_vstpreset_chunk(preset, b"Comp")
    if comp is None:
        return None
    if plugin_id == "trackcontrol":
        dgct = comp.find(b"DGCT")
        zpos = comp.find(b"ZIP ", dgct) if dgct != -1 else -1
        if zpos == -1 or zpos + 4 >= len(comp):
            return None
        zslice = comp[zpos + 4 :]
        if _has_trailing_zlib_bytes(zslice):
            return None
        new_zip, _norm = patch_trackcontrol_dgct_zip(zslice, gain_db)
        new_comp = comp[: zpos + 4] + new_zip
        return build_vstpreset(replace_vstpreset_chunk(preset, b"Comp", new_comp))
    if plugin_id == "protility":
        vc2 = comp.find(b"VC2!")
        if vc2 == -1 or vc2 + 8 > len(comp):
            return None
        old_len = struct.unpack_from("<I", comp, vc2 + 4)[0]
        xml_start = vc2 + 8
        if xml_start + old_len > len(comp):
            return None
        new_xml = patch_protility_vc2_xml(comp[xml_start : xml_start + old_len], gain_db)
        new_comp = comp[: vc2 + 4] + struct.pack("<I", len(new_xml)) + new_xml + comp[xml_start + old_len :]
        return build_vstpreset(replace_vstpreset_chunk(preset, b"Comp", new_comp))
    return None


def _xml_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    low = val.strip().lower()
    if low in ("true", "1"):
        return True
    if low in ("false", "0"):
        return False
    return default


def _inspect_state_blob(
    state_el: Any,
    plugin_id: str,
    read_member: Callable[[str], bytes | None],
) -> tuple[str, bytes | None, float, bool, bool, bool]:
    """Return ``(state_path, state_bytes, gain_db, is_clean_decoded, is_calibrated, has_valid_state)``."""
    if state_el is None:
        return "", None, 0.0, False, False, True
    if _xml_bool(state_el.get("external"), False):
        return str(state_el.get("path") or "").strip(), None, 0.0, False, False, False
    state_path = str(state_el.get("path") or "").strip()
    raw_blob = read_member(state_path) if state_path else None
    if raw_blob is None:
        return state_path, None, 0.0, False, False, False
    try:
        preset = parse_vstpreset(raw_blob)
    except ValueError:
        return state_path, None, 0.0, False, False, False
    comp = extract_vstpreset_chunk(preset, b"Comp")
    decoded = _decode_comp_trim_state(plugin_id, comp) if comp is not None else None
    if decoded is not None:
        return state_path, raw_blob, decoded[0], decoded[1], True, True
    return state_path, raw_blob, 0.0, False, False, True


def inspect_vst3_slot1_trim(
    device_elem: Any,
    read_member: Callable[[str], bytes | None],
    *,
    track_format: str = "Stereo",
    source_channel_id: str = "",
    source_index: int = 0,
) -> Vst3TrimDonor | None:
    """Validate a Slot 1 ``<Vst3Plugin>`` XML node as a supported Trim plugin (VST3 only)."""
    if not isinstance(getattr(device_elem, "tag", None), str):
        return None
    if _local_tag(device_elem.tag) != "Vst3Plugin":
        return None
    orig_dev_name = str(device_elem.get("deviceName") or "").strip()
    dev_name = orig_dev_name or str(device_elem.get("name") or "")
    dev_vendor = str(device_elem.get("deviceVendor") or "")
    spec = match_vst3_trim_spec(dev_name, dev_vendor)
    if spec is None:
        return None
    device_id = str(device_elem.get("deviceID") or "").strip()
    is_loaded = _xml_bool(device_elem.get("loaded"), True)
    enabled_el = device_elem.find("Enabled")
    is_enabled = _xml_bool(enabled_el.get("value"), True) if enabled_el is not None else True
    is_bypassed = not is_enabled

    state_path, state_bytes, gain_db, decoded_clean, is_calibrated, has_valid_state = _inspect_state_blob(
        device_elem.find("State"), spec.plugin_id, read_member,
    )
    is_clean = bool(device_id) and is_calibrated and decoded_clean
    trim_info = TrimPluginInfo(
        plugin_id=spec.plugin_id,
        plugin_name=spec.display_name,
        gain_db=gain_db,
        min_db=spec.min_db,
        max_db=spec.max_db,
        is_active=is_loaded and not is_bypassed,
        is_bypassed=is_bypassed,
        is_clean_donor=is_clean,
        precision_decimals=1,
    )
    return Vst3TrimDonor(
        plugin_id=spec.plugin_id,
        plugin_name=spec.display_name,
        device_id=device_id,
        device_vendor=dev_vendor,
        plugin_version=str(device_elem.get("pluginVersion") or ""),
        device_role=str(device_elem.get("deviceRole") or "audioFX"),
        track_format=track_format,
        state_path=state_path,
        state_bytes=state_bytes,
        trim_info=trim_info,
        source_channel_id=source_channel_id,
        source_index=source_index,
        device_name=orig_dev_name or spec.display_name,
        is_calibrated=is_calibrated,
        has_valid_state=has_valid_state,
    )


def has_calibrated_vst3_trim_donor(donors: Iterable[Vst3TrimDonor]) -> bool:
    """Return True when at least one verified clean, calibrated VST3 Trim donor exists."""
    return select_vst3_trim_donor(donors, "Stereo", require_calibrated=True) is not None


def select_vst3_trim_donor(
    donors: Iterable[Vst3TrimDonor],
    track_format: str,
    *,
    require_calibrated: bool = False,
) -> Vst3TrimDonor | None:
    """Select the cleanest VST3 Trim donor, preferring *track_format* then channel-agnostic fallback.

    When *require_calibrated* is ``True``, only verified clean calibrated donors are eligible.
    When ``False`` (Phase 1 unity insert), verified clean donors are preferred and uncalibrated
    valid-state template donors serve as a fallback (dirty calibrated donors remain excluded).
    """
    all_donors = list(donors)
    eligible = [
        d for d in all_donors
        if d.device_id and d.has_valid_state and (
            (d.is_calibrated and d.trim_info.is_clean_donor)
            if require_calibrated
            else (d.trim_info.is_clean_donor if d.is_calibrated else True)
        )
    ]
    if not eligible:
        return None
    counts: dict[str, int] = {}
    for d in all_donors:
        counts[d.plugin_id] = counts.get(d.plugin_id, 0) + 1
    avail_pids = {d.plugin_id for d in eligible}
    best_pid = min(
        avail_pids,
        key=lambda pid: (-counts.get(pid, 0), TRIM_PLUGIN_TIE_ORDER.get(pid, 99), pid),
    )
    matching = [d for d in eligible if d.plugin_id == best_pid]
    matching.sort(
        key=lambda d: (
            not d.is_calibrated,
            d.track_format != track_format,
            d.state_bytes is None,
            d.trim_info.is_bypassed,
            abs(d.trim_info.gain_db) >= 0.05,
            d.source_index,
        )
    )
    return matching[0]


def allocate_plugin_state_path(
    plugin_id: str,
    used_members: set[str],
    counter: list[int],
) -> str:
    """Allocate a unique ``plugins/<plugin_id>_<N>.vstpreset`` archive path."""
    while True:
        counter[0] += 1
        candidate = f"plugins/{plugin_id}_{counter[0]}.vstpreset"
        if candidate.lower() not in used_members:
            used_members.add(candidate.lower())
            return candidate


def insert_vst3_trim_slot1_xml(
    channel_elem: Any,
    donor: Vst3TrimDonor,
    *,
    make_element: Callable[..., Any],
    next_xml_id: Callable[[], str],
    used_members: set[str],
    state_counter: list[int],
    extra_archive_blobs: dict[str, bytes],
    apply_gain_db: float | None = None,
) -> dict[str, Any]:
    """Insert a cloned VST3 Trim ``<Vst3Plugin>`` into Slot 1 of *channel_elem* (XSD-ordered).

    When *apply_gain_db* is ``None``, writes the pristine donor ``.vstpreset`` unchanged
    (unity insertion without gain adjustment).
    """
    devices_el = channel_elem.find("Devices")
    if devices_el is None:
        devices_el = make_element("Devices")
        channel_elem.insert(0, devices_el)

    attrs: dict[str, str] = {
        "id": next_xml_id(),
        "name": donor.plugin_name,
        "deviceID": donor.device_id,
        "deviceName": donor.device_name or donor.plugin_name,
        "deviceRole": donor.device_role or "audioFX",
        "loaded": "true",
    }
    if donor.device_vendor:
        attrs["deviceVendor"] = donor.device_vendor
    if donor.plugin_version:
        attrs["pluginVersion"] = donor.plugin_version

    plugin_el = make_element("Vst3Plugin", attrs)
    enabled_el = make_element("Enabled", {"id": next_xml_id(), "value": "true"})
    plugin_el.append(enabled_el)

    patched_gain = False
    state_arcname = ""
    if donor.state_bytes is not None:
        blob = donor.state_bytes
        if apply_gain_db is not None:
            patched = patch_vst3_trim_preset(blob, donor.plugin_id, apply_gain_db)
            if patched is not None:
                blob = patched
                patched_gain = True
        state_arcname = allocate_plugin_state_path(donor.plugin_id, used_members, state_counter)
        extra_archive_blobs[state_arcname] = blob
        plugin_el.append(make_element("State", {"path": state_arcname}))

    devices_el.insert(0, plugin_el)
    return {
        "plugin": donor.plugin_name,
        "device_id": donor.device_id,
        "state_path": state_arcname,
        "gain_applied": patched_gain,
        "target_db": round(float(apply_gain_db), 1) if (apply_gain_db is not None and patched_gain) else 0.0,
    }


def patch_vst3_trim_slot1_xml(
    channel_elem: Any,
    target_db: float,
    read_member: Callable[[str], bytes | None],
    *,
    make_element: Callable[..., Any],
    next_xml_id: Callable[[], str],
    used_members: set[str],
    state_counter: list[int],
    extra_archive_blobs: dict[str, bytes],
) -> bool:
    """Patch an existing Slot 1 ``<Vst3Plugin>`` Trim on *channel_elem* to *target_db*.

    Allocates a fresh ``plugins/<id>_<N>.vstpreset`` member and updates ``<State path="..."/>``
    so shared template presets are never clobbered and duplicate ZIP entries cannot occur.
    """
    devices_el = channel_elem.find("Devices")
    if devices_el is None:
        return False
    device_elems = [el for el in devices_el if isinstance(getattr(el, "tag", None), str)]
    if not device_elems:
        return False
    plugin_el = device_elems[0]
    if _local_tag(plugin_el.tag) != "Vst3Plugin":
        return False
    dev_name = str(plugin_el.get("deviceName") or plugin_el.get("name") or "")
    spec = match_vst3_trim_spec(dev_name, str(plugin_el.get("deviceVendor") or ""))
    if spec is None:
        return False
    state_el = plugin_el.find("State")
    if state_el is None or _xml_bool(state_el.get("external"), False):
        return False
    state_path = str(state_el.get("path") or "").strip()
    raw_blob = read_member(state_path) if state_path else None
    if raw_blob is None:
        return False
    try:
        patched = patch_vst3_trim_preset(raw_blob, spec.plugin_id, target_db)
    except ValueError:
        return False
    if patched is None:
        return False
    new_arcname = allocate_plugin_state_path(spec.plugin_id, used_members, state_counter)
    extra_archive_blobs[new_arcname] = patched
    state_el.set("path", new_arcname)
    plugin_el.set("loaded", "true")
    enabled_el = plugin_el.find("Enabled")
    if enabled_el is None:
        params_el = plugin_el.find("Parameters")
        insert_idx = list(plugin_el).index(params_el) + 1 if params_el is not None else 0
        plugin_el.insert(insert_idx, make_element("Enabled", {"id": next_xml_id(), "value": "true"}))
    else:
        enabled_el.set("value", "true")
    return True
