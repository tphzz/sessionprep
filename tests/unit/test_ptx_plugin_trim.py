"""Unit tests for validated Pro Tools Slot A Trim plugin parsing and atomic ``.ptx`` patching."""

from __future__ import annotations

import json
import struct
import zlib

import pytest

from sessionpreplib.ptx import (
    PtxSession,
    encrypt_ptx,
    patch_slot_a_trim_gains,
    validate_index_0002,
)
from sessionpreplib.ptx.codec import decrypt_ptx, find_subchunks, iter_top_chunks
from sessionpreplib.ptx.models import PtxPluginInfo
from sessionpreplib.ptx.plugin_trim import avid_trim_db_to_step, avid_trim_step_to_db, inspect_slot_a_trim
from tests.unit.test_ptx_signal_flow import _build_console_ptx


def _zchunk(tag: int, body: bytes, ver: int = 1) -> bytes:
    return b"Z" + struct.pack("<HIH", ver, len(body) + 2, tag) + body


def _lp_str(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<I", len(raw)) + raw


def _aax_subchunk(mfr: bytes, prod: bytes, typ: bytes, c4cc: bytes, flags: int, name: bytes, payload: bytes) -> bytes:
    header = struct.pack(">I", flags) + mfr + prod + typ + c4cc + name.ljust(32, b"\x00")[:32]
    total = 4 + len(header) + len(payload)
    return struct.pack(">I", total) + header + payload


def _build_1038_trackcontrol(
    typ: bytes = b"RTST",
    gain_db: float = 0.0,
    *,
    extra_param: bool = False,
    current_ab: int = 0,
    multi_preset: bool = False,
) -> bytes:
    norm = struct.unpack("<f", struct.pack("<f", (gain_db + 200.0) / 218.0))[0]
    elck_payload = b"\x01\x01\x01\x01" + b"d_2dl \x00\x00" + struct.pack(">d", norm)
    elck = _aax_subchunk(b"DMGA", b"TrCt", typ, b"elck", 0xFFFFFFFF, b"Complete Controls State", elck_payload)
    params: str | dict[str, float]
    if abs(gain_db) < 1e-6 and not extra_param:
        params = "default"
    else:
        params = {"kOpGain": round(norm, 10)}
        if extra_param:
            params["kWidth"] = 0.75
    ab_slots: list[dict[str, object]] = [{"parameters": "default"}, {"parameters": "default"}]
    ab_slots[current_ab] = {"parameters": params}
    presets = [{"parameters": params, "name": "-"}]
    if multi_preset:
        presets.append({"parameters": {"kOpGain": 0.5}, "name": "SavedPreset"})
    js = {"presets": presets, "ab": ab_slots, "settings": {"currentAB": current_ab}, "v102": True}
    dgct_payload = b"ZIP " + zlib.compress((json.dumps(js, indent="\t") + " ").encode("utf-8"))
    dgct = _aax_subchunk(b"DMGA", b"TrCt", typ, b"DGCT", 0, b"Plugin Chunk", dgct_payload)
    inner = elck + dgct
    return _zchunk(0x1038, struct.pack("<II", 1, len(inner)) + inner + b"\x00\x00\x00\x00", ver=2)


def _build_1038_trim(gain_db: float = 0.0, *, flag3: int = -0x80000000) -> bytes:
    step = avid_trim_db_to_step(gain_db)
    raw_i32 = 0x7FFFFFFE if step >= 1023 else (((step << 22) | (step << 12) | (step << 2)) - 0x80000000)
    payload = struct.pack(">iiii", raw_i32, -0x80000000, -0x80000000, flag3)
    all_ch = _aax_subchunk(b"Digi", b"Trim", b"TR11", b"All ", 2, b"", payload)
    return _zchunk(0x1038, struct.pack("<II", 1, len(all_ch)) + all_ch + b"\x00\x00\x00\x00", ver=2)


def _build_1038_protility(typ: bytes = b"jccc", gain_db: float = 0.0, *, active_slot: str = "A") -> bytes:
    norm = struct.unpack("<f", struct.pack("<f", (gain_db + 60.0) / 84.0))[0]
    elck_payload = b"\x01\x01\x01\x01" + b"d_GAIN\x00\x00" + struct.pack(">d", norm)
    elck = _aax_subchunk(b"PrAu", b"PrTy", typ, b"elck", 0xFFFFFFFF, b"Complete Controls State", elck_payload)
    a_val = f"{gain_db:.1f}" if active_slot == "A" else "-2.0"
    b_val = f"{gain_db:.1f}" if active_slot == "B" else "0.0"
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?> <Protility activeSlot="{active_slot}" version="1">'
        f'<slotA USER_PRESET="0" PRESET_MODIFIED="0" version="1">'
        f'<Protility><PARAM id="GAIN" value="{a_val}"/><PARAM id="WIDTH" value="100.0"/></Protility></slotA>'
        f'<slotB USER_PRESET="1" PRESET_MODIFIED="0" version="1">'
        f'<Protility><PARAM id="GAIN" value="{b_val}"/><PARAM id="WIDTH" value="100.0"/></Protility></slotB>'
        f'</Protility>\x00'
    ).encode()
    juce = _aax_subchunk(b"PrAu", b"PrTy", typ, b"juce", 0, b"", b"VC2!" + struct.pack("<I", len(xml)) + xml)
    inner = elck + juce
    return _zchunk(0x1038, struct.pack("<II", 1, len(inner)) + inner + b"\x00\x00\x00\x00", ver=2)


def _build_1038_non_trim() -> bytes:
    dummy = _aax_subchunk(b"FabF", b"FQ4p", b"FQ4S", b"FFPs", 0, b"State", b"FFBS\x00\x00\x00\x00")
    return _zchunk(0x1038, struct.pack("<II", 1, len(dummy)) + dummy + b"\x00\x00\x00\x00", ver=2)


def _slot_2616(plug_idx: int, c1038: bytes, *, is_active: bool = True, is_bypassed: bool = False) -> bytes:
    c2626 = _zchunk(0x2626, b"\x00\x00\x00\x00")
    c260e = _zchunk(0x260E, b"\x00" * 16)
    act_byte = b"\x01" if is_active else b"\x00"
    c2613 = _zchunk(0x2613, c1038 + act_byte + b"\x00\xff\xff" + c260e + act_byte + b"\x00\x00", ver=10)
    byp_byte = b"\x01" if is_bypassed else b"\x00"
    c2615 = _zchunk(0x2615, c2626 + struct.pack("<H", plug_idx) + c2613)
    return _zchunk(0x2616, c2615 + byp_byte + b"\x00\x00")


def _wrap_slots(slots: list[bytes]) -> bytes:
    return _zchunk(0x261B, _zchunk(0x2627, b"".join(slots)), ver=15)


def _plugin_entry_1017(name: str, mfr: str, prod: str, typ: str, bundle: str) -> bytes:
    body = b"\x00" + _lp_str(name) + f"{mfr[::-1]}{prod[::-1]}{typ[::-1]}".encode("latin1") + b"\x00" * 4 + _lp_str(bundle)
    return _zchunk(0x1017, body)


def _build_plugin_trim_ptx(*, k0: int = 0, slot_overrides: dict[int, bytes] | None = None, decoy_in_blk0: bytes = b"") -> bytes:
    dec, xd = decrypt_ptx(_build_console_ptx())
    plugins = [
        _plugin_entry_1017("TrackControl", "DMGA", "TrCt", "RTST", "com.dmgaudio.TrackControl.aax"),
        _plugin_entry_1017("TrackControl", "DMGA", "TrCt", "RTMN", "com.dmgaudio.TrackControl.aax"),
        _plugin_entry_1017("Trim", "Digi", "Trim", "TR11", "com.avid.aax.trim"),
        _plugin_entry_1017("Protility", "PrAu", "PrTy", "jccc", "com.Process.Audio.Protility"),
        _plugin_entry_1017("Protility", "PrAu", "PrTy", "jcbb", "com.Process.Audio.Protility"),
        _plugin_entry_1017("FabFilter Pro-Q 4", "FabF", "FQ4p", "FQ4S", "com.fabfilter.Pro-Q.4"),
    ]
    c1018 = _zchunk(0x1018, struct.pack("<I", len(plugins)) + b"".join(plugins))
    chunks = list(iter_top_chunks(dec))
    c2624 = next(c for c in chunks if c.tag == 0x2624)
    c_body = c2624.raw
    tcount = struct.unpack_from("<I", c_body, 9)[0]
    default_wrappers = {
        0: _wrap_slots([_slot_2616(0, _build_1038_trackcontrol(b"RTST", 0.0, extra_param=True), is_active=False)]),
        2: _wrap_slots([_slot_2616(1, _build_1038_trackcontrol(b"RTMN", -6.0), is_active=True, is_bypassed=True)]),
        3: _wrap_slots([_slot_2616(2, _build_1038_trim(-6.0), is_active=True)]),
        4: _wrap_slots([_slot_2616(3, _build_1038_protility(b"jccc", 0.0), is_active=True)]),
        5: _wrap_slots([_slot_2616(5, _build_1038_non_trim(), is_active=True)]),
    }
    wrappers = slot_overrides if slot_overrides is not None else default_wrappers
    pos, new_blocks = 13, []
    for ti in range(tcount):
        csz = struct.unpack_from("<I", c_body, pos + 3)[0]
        blk_ba = bytearray(c_body[pos : pos + 7 + csz])
        pos += 7 + csz
        extra = (decoy_in_blk0 if ti == 0 else b"") + wrappers.get(ti, b"")
        if extra:
            blk_ba.extend(extra)
            struct.pack_into("<I", blk_ba, 3, struct.unpack_from("<I", blk_ba, 3)[0] + len(extra))
        new_blocks.append(bytes(blk_ba))
    new_c2624 = _zchunk(0x2624, struct.pack("<I", tcount) + b"".join(new_blocks))
    out = bytearray(dec[:0x1F])
    offsets: list[int] = []
    for ch in chunks:
        if ch.tag == 0x0002:
            continue
        for raw_ch in ([c1018, new_c2624] if ch.tag == 0x2624 else [ch.raw]):
            offsets.append(len(out))
            out.extend(raw_ch)
    idx_body = struct.pack("<III", 1, 0, 1) + struct.pack("<BHH", 0, 4, 1) + struct.pack("<I", offsets[-1]) + b"\x00" * 6
    idx_off = len(out)
    out.extend(_zchunk(0x0002, idx_body, ver=2))
    struct.pack_into("<I", out, 0x1B, idx_off)
    validate_index_0002(bytes(out))
    return encrypt_ptx(bytes(out), xd, k0=k0)


def test_avid_trim_taper_roundtrip_and_below_m24_calibration():
    for db, expected_step in [
        (-144.0, 0), (-60.0, 58), (-36.0, 150), (-24.0, 250), (-12.0, 473), (-6.0, 622), (0.0, 772), (6.0, 1023),
    ]:
        assert avid_trim_db_to_step(db) == expected_step
        assert avid_trim_step_to_db(expected_step) == pytest.approx(db, abs=0.05)
    for db in (-48.0, -30.0, -18.0, -9.5, -3.0, -1.2, 2.5, 4.5):
        assert avid_trim_step_to_db(avid_trim_db_to_step(db)) == pytest.approx(db, abs=0.15)


def test_slot_a_trim_parse_patch_activate_clamp_and_strict_refusal():
    raw = _build_plugin_trim_ptx()
    s = PtxSession.from_bytes(raw)
    t_tc, t_byp, t_trim, t_prot, t_proq = s.tracks[0], s.tracks[2], s.tracks[3], s.tracks[4], s.tracks[5]

    assert t_tc.slot_a_trim and t_tc.slot_a_trim.plugin_id == "trackcontrol" and not t_tc.slot_a_trim.is_active
    assert t_byp.slot_a_trim and t_byp.slot_a_trim.is_bypassed and t_byp.slot_a_trim.gain_db == pytest.approx(-6.0, abs=0.05)
    assert t_trim.slot_a_trim and t_trim.slot_a_trim.plugin_id == "trim"
    assert t_prot.slot_a_trim and t_prot.slot_a_trim.plugin_id == "protility"
    assert t_proq.slot_a_trim is None

    patched, meta = patch_slot_a_trim_gains(
        raw, {t_tc.name: 8.5, t_byp.name: -14.2, t_trim.name: 12.0, t_prot.name: -9.4},
        activate_if_inactive=True, unbypass_if_bypassed=False,
    )
    assert meta[t_tc.name]["activated"] is True and meta[t_byp.name]["unbypassed"] is False
    assert meta[t_trim.name]["target_db"] == 6.0
    s2 = PtxSession.from_bytes(patched)
    assert s2.track_by_name(t_tc.name).slot_a_trim.gain_db == pytest.approx(8.5, abs=0.05)
    assert s2.track_by_name(t_tc.name).slot_a_trim.is_active is True
    assert s2.track_by_name(t_byp.name).slot_a_trim.is_bypassed is True
    assert s2.track_by_name(t_trim.name).slot_a_trim.gain_db == pytest.approx(6.0, abs=0.05)
    assert s2.track_by_name(t_prot.name).slot_a_trim.gain_db == pytest.approx(-9.4, abs=0.05)

    with pytest.raises(ValueError, match="validated Slot A Trim"):
        patch_slot_a_trim_gains(raw, {t_proq.name: -3.0})
    with pytest.raises(ValueError, match="not found"):
        patch_slot_a_trim_gains(raw, {"NO_SUCH_TRACK": -3.0})
    with pytest.raises(ValueError, match="Non-finite"):
        patch_slot_a_trim_gains(raw, {t_tc.name: float("nan")})
    with pytest.raises(ValueError, match="Duplicate"):
        patch_slot_a_trim_gains(raw, {t_tc.name: 2.0, t_tc.name.lower(): -2.0})


def test_slot_a_trim_preserves_k0_decoy_bytes_and_slot_b_isolation():
    # Decoy 0x5A header inside blk 0 spanning past blk 1 must never be touched when patching track 1
    decoy = b"Z" + struct.pack("<HIH", 1, 50000, 0x9999) + b"\xde\xad\xbe\xef"
    wrappers = {
        0: _wrap_slots([_slot_2616(5, _build_1038_non_trim()), _slot_2616(0, _build_1038_trackcontrol(b"RTST", 0.0))]),
        1: _wrap_slots([_slot_2616(0, _build_1038_trackcontrol(b"RTST", 0.0, extra_param=True), is_active=False)]),
    }
    raw = _build_plugin_trim_ptx(k0=0x5A, slot_overrides=wrappers, decoy_in_blk0=decoy)
    assert (raw[0x14] ^ 0x5A) == 0x5A
    s = PtxSession.from_bytes(raw)
    # Track 0 has Pro-Q in Slot A and TrackControl in Slot B -> Slot B MUST be ignored!
    assert s.tracks[0].slot_a_trim is None
    assert s.tracks[1].slot_a_trim is not None

    patched, meta = patch_slot_a_trim_gains(raw, {s.tracks[1].name: 6.5})
    assert (patched[0x14] ^ 0x5A) == 0x5A  # k0 XOR key preserved
    dec_after, _ = decrypt_ptx(patched)
    assert decoy in dec_after  # Decoy 0x5A in track 0 was NOT corrupted!
    assert meta[s.tracks[1].name]["target_db"] == 6.5
    assert PtxSession.from_bytes(patched).tracks[1].slot_a_trim.gain_db == pytest.approx(6.5, abs=0.05)


def test_slot_a_trim_ab_slots_mono_presets_and_zero_unbypass_flags():
    wrappers = {
        0: _wrap_slots([_slot_2616(0, _build_1038_trackcontrol(b"RTST", -3.0, extra_param=True, current_ab=1, multi_preset=True), is_active=False)]),
        1: _wrap_slots([_slot_2616(4, _build_1038_protility(b"jcbb", -4.5, active_slot="B"), is_active=True, is_bypassed=True)]),
        2: _wrap_slots([_slot_2616(2, _build_1038_trim(0.0, flag3=0x7FFFFFFE), is_active=True)]),  # +12 dB mode / non-default
    }
    raw = _build_plugin_trim_ptx(slot_overrides=wrappers)
    s = PtxSession.from_bytes(raw)
    assert s.tracks[0].slot_a_trim.gain_db == pytest.approx(-3.0, abs=0.05)
    assert s.tracks[1].slot_a_trim.gain_db == pytest.approx(-4.5, abs=0.05)
    assert s.tracks[2].slot_a_trim is None  # Avid Trim non-default switch fails closed

    patched, meta = patch_slot_a_trim_gains(
        raw, {s.tracks[0].name: 0.0, s.tracks[1].name: 7.2}, activate_if_inactive=True, unbypass_if_bypassed=True,
    )
    assert meta[s.tracks[0].name]["activated"] is False  # target 0.0 leaves inactive untouched
    assert meta[s.tracks[1].name]["unbypassed"] is True
    s2 = PtxSession.from_bytes(patched)
    assert s2.tracks[0].slot_a_trim.gain_db == pytest.approx(0.0, abs=0.05)
    assert s2.tracks[0].slot_a_trim.is_active is False
    assert s2.tracks[1].slot_a_trim.gain_db == pytest.approx(7.2, abs=0.05)
    assert s2.tracks[1].slot_a_trim.is_bypassed is False

    # Verify TrackControl preserved kWidth and second saved preset without mutating it
    dec_p, _ = decrypt_ptx(patched)
    tc_1038 = next(c for c in find_subchunks(dec_p, 0x1038) if b"DMGA" in c[:64])
    dgct_off = tc_1038.find(b"DGCT")
    clen = struct.unpack_from(">I", tc_1038, dgct_off - 20)[0]
    zpos = tc_1038.find(b"ZIP ", dgct_off)
    js = json.loads(zlib.decompress(tc_1038[zpos + 4 : dgct_off - 20 + clen]).rstrip(b"\x00 ").decode("utf-8"))
    assert js["ab"][1]["parameters"]["kWidth"] == 0.75
    assert js["presets"][1]["parameters"]["kOpGain"] == 0.5


def test_slot_a_trim_rejects_corrupt_dgct_xml_and_zlib_bomb():
    tc_pinfo = PtxPluginInfo(0, "TrackControl", "DMGA", "TrCt", "RTST", "com.dmgaudio.TrackControl.aax")
    pr_pinfo = PtxPluginInfo(3, "Protility", "PrAu", "PrTy", "jccc", "com.Process.Audio.Protility")

    bad_tc = _slot_2616(0, _build_1038_trackcontrol(b"RTST", 0.0).replace(b"ZIP ", b"BAD!"))
    assert inspect_slot_a_trim(bad_tc, tc_pinfo, is_active=True, is_bypassed=False) is None

    # Malformed DGCT JSON (bad currentAB / NaN kOpGain) and >2 MiB zlib bomb fail closed
    for payload_bytes in (
        b'{"settings": {"currentAB": "bad"}, "ab": [{"parameters": {"kOpGain": 0.5}}]}',
        b'{"settings": {"currentAB": 0}, "ab": [{"parameters": {"kOpGain": NaN}}]}',
        b" " * (3 * 1024 * 1024),
    ):
        elck = _aax_subchunk(b"DMGA", b"TrCt", b"RTST", b"elck", 0xFFFFFFFF, b"", b"d_2dl \x00\x00" + struct.pack(">d", 0.5))
        dgct = _aax_subchunk(b"DMGA", b"TrCt", b"RTST", b"DGCT", 0, b"", b"ZIP " + zlib.compress(payload_bytes))
        c1038 = _zchunk(0x1038, struct.pack("<II", 1, len(elck + dgct)) + elck + dgct + b"\x00\x00\x00\x00", ver=2)
        assert inspect_slot_a_trim(_slot_2616(0, c1038), tc_pinfo, is_active=True, is_bypassed=False) is None

    bad_pr = _slot_2616(3, _build_1038_protility(b"jccc", 0.0).replace(b"VC2!", b"NOPE"))
    assert inspect_slot_a_trim(bad_pr, pr_pinfo, is_active=True, is_bypassed=False) is None
