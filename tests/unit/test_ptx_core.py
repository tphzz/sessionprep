"""Unit tests for pure-Python Pro Tools ``.ptx`` codec, parser, patcher, and PTSL calibration."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from sessionpreplib.daw_processors import ptsl_helpers
from sessionpreplib.ptx import (
    MAX_GROUP_ID,
    PtxSession,
    apply_atomic_ptx_patch,
    decrypt_ptx,
    encrypt_ptx,
    find_subchunks,
    group_id_to_label,
    patch_add_group,
    patch_clear_track_inputs,
    patch_clip_gains,
    patch_delete_group,
    patch_modify_group,
    patch_track_volumes,
)
from sessionpreplib.ptx.models import PTX_MAGIC, UUID_PREFIX


def _zchunk(tag: int, body: bytes, version: int = 1) -> bytes:
    return struct.pack("<BHIH", 0x5A, version, len(body) + 2, tag) + body


def _lp(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _uuid(seed: int) -> bytes:
    return UUID_PREFIX + struct.pack(">II", 0x1000_0000 + seed, 0x2000_0000 + seed)


def _sp_entry(name: str, kind: int, is_stereo: bool, uid: bytes, tag: int = 0x2602) -> bytes:
    chs = [1, 2] if is_stereo else [1]
    body = (
        bytes([kind, 1 if is_stereo else 0])
        + _lp(name)
        + struct.pack(f"<I{len(chs)}H", len(chs), *chs)
        + struct.pack("<HBB", 0xFFFF, 1, 0)
        + uid
        + (b"\x00" * 16 if tag == 0x2602 else b"")
    )
    return _zchunk(tag, body, version=2)


def _track_251a(name: str, type_code: int, uid: bytes, *, is_folder: bool = False, is_stereo: bool = True) -> bytes:
    tail = bytearray(15)
    tail[-11] = 1 if is_folder else 0
    tail[-7] = 1  # visible
    tail[-6] = 1  # active
    struct.pack_into("<H", tail, len(tail) - 2, 0xFFFE)
    body = bytes([type_code, 0]) + _lp(name) + bytes([1 if is_stereo else 0]) + uid + bytes(tail)
    return _zchunk(0x251A, body, version=9)


def _chunk_260a(value: int) -> bytes:
    # 39-byte chunk: 9-byte header + 30-byte body; value at len - 4
    body = bytearray(30)
    struct.pack_into("<h", body, len(body) - 4, value)
    return _zchunk(0x260A, bytes(body), version=1)


def _chunk_1029(vol_tenths: int, *, post_fader: bool = True) -> bytes:
    body = bytearray(100)
    body[0] = 1 if post_fader else 0
    struct.pack_into("<i", body, 1, vol_tenths)
    struct.pack_into("<i", body, 87, vol_tenths)
    return _zchunk(0x1029, bytes(body), version=1)


def _mixer_track_261e(uid: bytes, input_uid: bytes | None, output_uid: bytes, vol_db: float = 0.0) -> bytes:
    val = int(round(vol_db * 10.0))
    d0 = _zchunk(
        0x260D,
        _zchunk(0x260E, b"\x00\x00\x00\x01" + output_uid)
        + _chunk_1029(val)
        + _chunk_260a(val)
        + _chunk_260a(0),
    )
    pre_input = (input_uid if input_uid else b"") + _zchunk(0x4856, b"\x00" * 8)
    body = uid + pre_input + d0 + (b"\x00" * 40)
    return _zchunk(0x261E, body, version=1)


def _mixer_vca_261f(
    uid: bytes, vol_db: float = 0.0, group_name: str = "", group_id: int = 0, *, active_flag: int | None = None
) -> bytes:
    val = int(round(vol_db * 10.0))
    flag = (1 if group_name else 0) if active_flag is None else active_flag
    s2610 = _zchunk(0x2610, _chunk_260a(val) + _chunk_260a(0))
    body = uid + bytes([flag]) + struct.pack("<I", group_id) + s2610 + _lp(group_name) + b"\x00"
    return _zchunk(0x261F, body, version=1)


def _mixer_audio_261c(
    uid: bytes,
    hw_input_name: str | None,
    hw_input_uid: bytes | None,
    output_uid: bytes,
    vol_db: float = 0.0,
) -> bytes:
    val = int(round(vol_db * 10.0))
    d0 = _zchunk(
        0x260D,
        _zchunk(0x260E, b"\x00\x00\x00\x01" + output_uid)
        + _chunk_1029(val)
        + _chunk_260a(val)
        + _chunk_260a(0),
    )
    s485a = _zchunk(0x485A, b"\x00\x00\x00\x00\x01", version=2)
    in_slot = (_lp(hw_input_name) + hw_input_uid) if (hw_input_name and hw_input_uid) else (b"\x00" * 20)
    pre_input = (
        s485a
        + b"\x01\x01\x0a\x00"
        + (b"\x00" * 40)
        + b"\x00\x00\x00\x00\x08\x00\x00\x00\x00"
        + in_slot
        + (b"\x00" * 7)
        + b"\xff\xff\xff\xff\x00\xff\x7f\x0a\x00"
        + _zchunk(0x4856, b"\x00" * 8)
    )
    inner_261b = _zchunk(0x261B, uid + pre_input + d0, version=15)
    body = inner_261b + _zchunk(0x2038, b"\x00" * 16, version=1) + (b"\x00" * 24)
    return _zchunk(0x261C, body, version=6)


def _io_1021(name: str, kind: int, is_stereo: bool, uid: bytes) -> bytes:
    ch_cnt = 2 if is_stereo else 1
    body = bytes([kind, 1 if is_stereo else 0]) + _lp(name) + struct.pack("<I", ch_cnt) + (b"\x00\x00" * ch_cnt) + struct.pack("<HB", 0xFFFF, 1) + uid
    return _zchunk(0x1021, body, version=1)


def _clip_2629(name: str, file_idx: int, ch_idx: int, gain_lin: float, clip_uid: bytes) -> bytes:
    b28_body = _lp(name) + b"\x00\x00\x00\x00\x00" + struct.pack("<i", -1) + b"\x00\x00" + clip_uid
    b28 = _zchunk(0x2628, b28_body, version=5)
    body = b28 + struct.pack("<IHf", file_idx, ch_idx, gain_lin)
    return _zchunk(0x2629, body, version=1)


def _build_synthetic_ptx(*, xor_delta: int = 0x50) -> bytes:
    """Build a valid minimal encrypted ``.ptx`` with folder, audio tracks, VCA, clips, and ``0x0002`` index."""
    bus_out_uid, bus_kik_uid, hw_in_uid = _uuid(1), _uuid(2), _uuid(3)
    rf_uid, kick_uid, snare_uid, vca_uid = _uuid(10), _uuid(11), _uuid(12), _uuid(13)

    c0003 = _zchunk(0x0003, b"Pro Tools Studio\x00\x002026.11.1\x00\x00Release\x00\x00Session\x00\x00Windows\x00")
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000))
    c1022 = _zchunk(0x1022, struct.pack("<I", 1) + _io_1021("AIO Analog 1-2", 0, True, hw_in_uid), version=1)
    c2603 = _zchunk(
        0x2603,
        struct.pack("<I", 2)
        + _sp_entry("Main Out", 1, True, bus_out_uid)
        + _sp_entry("KIK", 2, True, bus_kik_uid),
        version=2,
    )
    c2519 = _zchunk(
        0x2519,
        _track_251a("KIK", 2, rf_uid, is_folder=True)
        + _track_251a("Kick", 0, kick_uid, is_stereo=False)
        + _track_251a("Snare", 0, snare_uid, is_stereo=True)
        + _track_251a("KIK-VCA", 9, vca_uid, is_stereo=False),
        version=9,
    )
    c210c = _zchunk(
        0x210C,
        struct.pack("<I", 1) + b"\x00" + rf_uid + struct.pack("<I", 2) + b"\x00" + kick_uid + b"\x00" + vca_uid,
    )
    c2624 = _zchunk(
        0x2624,
        struct.pack("<I", 4)
        + _mixer_track_261e(rf_uid, bus_kik_uid, bus_out_uid, 0.0)
        + _mixer_audio_261c(kick_uid, "AIO Analog 1-2", hw_in_uid, bus_out_uid, 0.0)
        + _mixer_audio_261c(snare_uid, "AIO Analog 1-2", hw_in_uid, bus_out_uid, 0.0)
        + _mixer_vca_261f(vca_uid, 0.0),
    )
    c202b = _zchunk(0x202B, struct.pack("<II", 0, 0) + b"\x00\x00", version=4)
    c2637 = _zchunk(0x2637, struct.pack("<I", 0), version=1)
    c262a = _zchunk(
        0x262A,
        _clip_2629("Kick_01", 0, 0, 1.0, _uuid(100))
        + _clip_2629("Snare_01.L", 1, 0, 1.0, _uuid(101))
        + _clip_2629("Snare_01.R", 1, 1, 1.0, _uuid(102)),
    )
    lane0 = _zchunk(0x1052, _lp("Kick") + _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 0) + b"\x00" + struct.pack("<Q", 0) + b"\x00" + struct.pack("<H", 0)))
    lane1 = _zchunk(0x1052, _lp("Snare.L") + _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 1) + b"\x00" + struct.pack("<Q", 0) + b"\x00" + struct.pack("<H", 0)))
    lane2 = _zchunk(0x1052, _lp("Snare.R") + _zchunk(0x104F, b"\x00\x00" + struct.pack("<I", 2) + b"\x00" + struct.pack("<Q", 0) + b"\x00" + struct.pack("<H", 0)))
    c1054 = _zchunk(0x1054, lane0 + lane1 + lane2)
    pl_kick = _zchunk(0x1014, _lp("Kick") + b"\x00" + struct.pack("<IH", 1, 0) + _uuid(200))
    pl_snare = _zchunk(0x1014, _lp("Snare") + b"\x00" + struct.pack("<IHH", 2, 1, 2) + _uuid(201))
    c1015 = _zchunk(0x1015, pl_kick + pl_snare)

    # Pad a chunk so the file spans >= 3 4KB blocks (testing rolling XOR key recovery)
    pad = _zchunk(0x7FFE, b"\x00" * 9000)
    top_chunks = [c0003, c1028, c1022, c2603, c2519, c210c, c2624, c202b, c2637, c262a, c1054, c1015, pad]

    # Assemble header + chunks + valid 0x0002 offset directory pointing at c202b & pad
    out = bytearray(PTX_MAGIC + b"\x00\x05\x90")
    out.extend(struct.pack("<BHII", 0x5A, 1, 4, 0))  # 11-byte root chunk ending at 0x1F
    assert len(out) == 0x1F
    offsets: list[int] = []
    for chunk in top_chunks:
        offsets.append(len(out))
        out.extend(chunk)
    idx_off = len(out)
    struct.pack_into("<I", out, 0x1B, idx_off)
    # Build valid 0x0002 referencing c2624, internal 0x261C/0x4856/0x260D subchunks inside c2624, c202b, and pad
    c2624_off = offsets[6]
    c2624_raw = top_chunks[6]
    sub_offs: list[int] = [c2624_off]
    for sub in find_subchunks(c2624_raw, 0x261C):
        rel = c2624_raw.find(sub)
        sub_offs.append(c2624_off + rel)
        s4856 = find_subchunks(sub, 0x4856)
        if s4856:
            sub_offs.append(c2624_off + rel + sub.find(s4856[0]))
    sub_offs.extend([offsets[7], offsets[-1]])
    idx_body = (
        struct.pack("<II", 1, 0)  # entry_count=1, depth=0
        + struct.pack("<I", 1)  # occ_cnt=1
        + struct.pack("<BHH", 0, 4 * len(sub_offs), len(sub_offs))
        + struct.pack(f"<{len(sub_offs)}I", *sub_offs)
        + b"\x00" * 6
    )
    out.extend(_zchunk(0x0002, idx_body, version=2))
    return encrypt_ptx(bytes(out), xor_delta, k0=0)


def test_codec_decrypt_encrypt_roundtrip():
    raw = _build_synthetic_ptx(xor_delta=0x50)
    dec, delta = decrypt_ptx(raw)
    assert delta == 0x50
    assert dec[0x14] == 0x5A
    assert encrypt_ptx(dec, delta) == raw


def test_parser_decodes_tracks_folders_io_clips_and_matrix():
    raw = _build_synthetic_ptx()
    sess = PtxSession.from_bytes(raw)
    assert sess.info.sample_rate == 48000
    assert sess.info.bit_depth == 24
    assert [t.name for t in sess.tracks] == ["KIK", "Kick", "Snare", "KIK-VCA"]

    kik = sess.track_by_name("KIK")
    assert kik is not None and kik.track_type == "RoutingFolder"
    assert kik.input_path is not None and kik.input_path.name == "KIK" and kik.input_path.ptsl_index == 1
    assert kik.main_output is not None and kik.main_output.name == "Main Out" and kik.main_output.ptsl_index == 0

    kick = sess.track_by_name("Kick")
    assert kick is not None and kick.parent_folder_name == "KIK"
    assert len(kick.clips) == 1 and kick.clips[0].clip_index == 0

    snare = sess.track_by_name("Snare")
    assert snare is not None and len(snare.clips) == 2

    matrix = sess.routing_matrix(ptsl_base=0x30000100)
    assert matrix["ptsl_base"] == "0x30000100"
    assert "KIK" in matrix["bus_graph"]


def test_patch_volumes_clip_gains_and_groups_roundtrip(tmp_path):
    raw = _build_synthetic_ptx()
    ptx_file = tmp_path / "Mix.ptx"
    ptx_file.write_bytes(raw)

    def _transform(buf: bytes) -> tuple[bytes, dict]:
        b0, clr = patch_clear_track_inputs(buf, ["Kick", "Snare", "KIK"])
        b1, cg = patch_clip_gains(b0, {"Kick": 8.5, "Snare": -6.0})
        b2, vol = patch_track_volumes(b1, {"Kick": -8.5, "Snare": 6.0, "KIK-VCA": -2.5})
        b3, grp = patch_add_group(b2, "KIK-GRP", ["Kick", "Snare"], vca_track="KIK-VCA")
        return b3, {"clr": clr, "cg": cg, "vol": vol, "grp": grp}

    orig, meta = apply_atomic_ptx_patch(ptx_file, _transform)
    assert orig == raw
    assert meta["clr"]["Kick"]["cleared"] is True and meta["clr"]["Kick"]["previous_input"] == "AIO Analog 1-2"
    assert meta["clr"]["Snare"]["cleared"] is True
    assert "KIK" not in meta["clr"]  # RoutingFolder input must never be cleared
    assert meta["cg"]["Snare"]["clip_count"] == 2
    assert meta["grp"]["vca_track"] == "KIK-VCA"

    reloaded = PtxSession.from_file(ptx_file)
    assert reloaded.track_by_name("KIK").input_path is not None and reloaded.track_by_name("KIK").input_path.name == "KIK"
    assert reloaded.track_by_name("Kick").input_path is None
    assert reloaded.track_by_name("Snare").input_path is None
    assert reloaded.track_by_name("Kick").volume_db == -8.5
    assert reloaded.track_by_name("Snare").volume_db == 6.0
    assert reloaded.track_by_name("KIK-VCA").volume_db == -2.5
    assert reloaded.clips[0].clip_gain_curve_index >= 0 and reloaded.clips[0].gain_db == pytest.approx(8.5, abs=0.02)
    assert reloaded.clips[1].clip_gain_curve_index >= 0 and reloaded.clips[1].gain_db == pytest.approx(-6.0, abs=0.02)
    assert reloaded.clips[2].clip_gain_curve_index >= 0 and reloaded.clips[2].gain_db == pytest.approx(-6.0, abs=0.02)
    assert reloaded.clips[1].clip_gain_curve_index != reloaded.clips[2].clip_gain_curve_index
    # Updating an existing 0x2637 clip gain in place preserves curve count and updates gain_db
    p_upd, _ = patch_clip_gains(ptx_file.read_bytes(), {"Kick": -3.25})
    s_upd = PtxSession.from_bytes(p_upd)
    assert s_upd.clips[0].gain_db == pytest.approx(-3.25, abs=0.02)
    assert s_upd.clips[0].clip_gain_curve_index == reloaded.clips[0].clip_gain_curve_index
    # Re-running patch_clear_track_inputs on already-cleared tracks is a clean no-op
    p_noop, noop_meta = patch_clear_track_inputs(ptx_file.read_bytes(), ["Kick", "Snare"])
    assert noop_meta["Kick"]["cleared"] is False and noop_meta["Snare"]["cleared"] is False
    assert PtxSession.from_bytes(p_noop).track_by_name("Kick").input_path is None
    grp = reloaded.group_by_name("KIK-GRP")
    assert grp is not None and grp.member_names == ["Kick", "Snare"] and grp.vca_track_name == "KIK-VCA"
    assert reloaded.track_by_name("KIK-VCA").vca_group_name == "KIK-GRP"
    # Verify 0x261F active-assignment flag (p10 - 5) flipped from 0 -> 1 for Pro Tools persistence
    vca_blocks = find_subchunks(reloaded.chunk_map[0x2624].raw, 0x261F)
    s2610 = find_subchunks(vca_blocks[0], 0x2610)[0]
    assert vca_blocks[0][vca_blocks[0].find(s2610) - 5] == 1

    # Modify and delete group round-trip
    p_mod, _ = patch_modify_group(ptx_file.read_bytes(), "KIK-GRP", new_name="DRUMS", remove_tracks=["Snare"])
    s_mod = PtxSession.from_bytes(p_mod)
    assert s_mod.group_by_name("DRUMS").member_names == ["Kick"]
    assert s_mod.track_by_name("KIK-VCA").vca_group_name == "DRUMS"

    p_del, del_info = patch_delete_group(p_mod, "DRUMS")
    assert del_info["cleared_vca_track"] == "KIK-VCA"
    s_del = PtxSession.from_bytes(p_del)
    assert s_del.group_by_name("DRUMS") is None
    assert s_del.track_by_name("KIK-VCA").vca_group_name is None


def test_multibank_group_ids_beyond_26():
    raw = _build_synthetic_ptx()
    cur = raw
    for i in range(1, 29):
        cur, info = patch_add_group(cur, f"Grp_{i:02d}", ["Kick"])
    assert info["id_number"] == 28
    assert info["id_letter"] == "2b"
    assert group_id_to_label(27) == "2a"
    assert group_id_to_label(MAX_GROUP_ID) == "4z"
    sess = PtxSession.from_bytes(cur)
    assert sess.group_by_name("Grp_28").id_number == 28


def test_calibrate_signalpath_base_consensus_and_drift(monkeypatch):
    sess = PtxSession.from_bytes(_build_synthetic_ptx())
    base = 0x30000180
    live = [SimpleNamespace(id=t.uuid_braced, name=t.name) for t in sess.tracks]
    mapping = {
        sess.track_by_name("KIK").uuid_braced: [hex(base + 0)],
        sess.track_by_name("Kick").uuid_braced: [hex(base + 0)],
        sess.track_by_name("Snare").uuid_braced: [hex(base + 1)],  # unsaved output drift
    }
    monkeypatch.setattr(
        ptsl_helpers,
        "get_track_main_output_assignments",
        lambda _eng, tids: mapping.get(tids[0], []),
    )
    resolved_base, verified, warnings = ptsl_helpers.calibrate_signalpath_base(object(), sess, live)
    assert resolved_base == base
    assert verified == 2
    assert len(warnings) == 1


def test_codec_and_atomic_patcher_reject_corrupt_or_truncated_ptx(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        decrypt_ptx(b"short")
    with pytest.raises(ValueError, match="magic"):
        decrypt_ptx(b"X" * 64)

    raw = _build_synthetic_ptx()
    ptx_file = tmp_path / "Guard.ptx"
    ptx_file.write_bytes(raw)
    # Corrupted transform dropping trailing chunks must be rejected without modifying disk
    with pytest.raises(RuntimeError):
        apply_atomic_ptx_patch(ptx_file, lambda buf: (buf[:128], {}))
    assert ptx_file.read_bytes() == raw


def test_patchers_reject_nan_inf_shared_clip_conflict_and_group_errors():
    raw = _build_synthetic_ptx()
    with pytest.raises(ValueError, match="finite"):
        patch_track_volumes(raw, {"Kick": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        patch_clip_gains(raw, {"Kick": float("inf")})
    with pytest.raises(ValueError, match="no placed clips"):
        patch_clip_gains(raw, {"KIK": -3.0})

    # Two tracks sharing the same clip_index with conflicting gains must fail cleanly
    dec_shared, delta_shared = decrypt_ptx(raw)
    shared_buf = bytearray(dec_shared)
    # Point Snare's first 0x104F placement (clip_idx=1) to Kick's clip_idx=0
    snare_lane = shared_buf.find(b"Snare.L")
    assert snare_lane > 0
    marker = b"\x00\x00" + struct.pack("<I", 1) + b"\x00" + struct.pack("<Q", 0)
    pos = shared_buf.find(marker, snare_lane)
    assert pos > snare_lane
    struct.pack_into("<I", shared_buf, pos + 2, 0)
    shared_raw = encrypt_ptx(bytes(shared_buf), delta_shared)
    with pytest.raises(ValueError, match="shared by"):
        patch_clip_gains(shared_raw, {"Kick": 3.0, "Snare": -3.0})

    # Group validation errors
    with pytest.raises(ValueError, match="Edit-only"):
        patch_add_group(raw, "BadEdit", ["Kick"], vca_track="KIK-VCA", group_type="Edit")
    with pytest.raises(ValueError, match="not 'Vca'"):
        patch_add_group(raw, "BadVca", ["Kick"], vca_track="Snare")
    with pytest.raises(ValueError, match="cannot be a member"):
        patch_add_group(raw, "SelfVca", ["Kick", "KIK-VCA"], vca_track="KIK-VCA")
    with pytest.raises(ValueError, match="<ALL>"):
        patch_modify_group(raw, "<ALL>", new_name="Nope")

    p1, _ = patch_add_group(raw, "G1", ["Kick"], vca_track="KIK-VCA", id_letter="a")
    p2, _ = patch_add_group(p1, "G2", ["Snare"], id_letter="b")
    with pytest.raises(ValueError, match="already in use"):
        patch_add_group(p1, "DupID", ["Snare"], id_letter="a")
    with pytest.raises(ValueError, match="already assigned"):
        patch_add_group(p1, "StealVCA", ["Snare"], vca_track="KIK-VCA")
    with pytest.raises(ValueError, match="already exists"):
        patch_modify_group(p2, "G2", new_name="G1")
    with pytest.raises(ValueError, match="Invalid group_type"):
        patch_modify_group(p2, "G2", group_type="Bogus")

    # Exhausting all 104 Pro Tools group IDs raises cleanly on the 105th
    cur = raw
    for idx in range(1, MAX_GROUP_ID + 1):
        cur, _ = patch_add_group(cur, f"Full_{idx:03d}", ["Kick"])
    with pytest.raises(ValueError, match=f"All {MAX_GROUP_ID}"):
        patch_add_group(cur, "Overflow_105", ["Kick"])


def test_calibrate_signalpath_base_rejects_ties_malformed_and_zero_matches(monkeypatch):
    sess = PtxSession.from_bytes(_build_synthetic_ptx())
    live = [SimpleNamespace(id=t.uuid_braced, name=t.name) for t in sess.tracks]

    # Strict 1-vs-1 tie must raise RuntimeError instead of guessing
    tie_map = {
        sess.track_by_name("KIK").uuid_braced: [hex(0x30000100)],
        sess.track_by_name("Kick").uuid_braced: [hex(0x30000200)],
    }
    monkeypatch.setattr(ptsl_helpers, "get_track_main_output_assignments", lambda _e, tids: tie_map.get(tids[0], []))
    with pytest.raises(RuntimeError, match="Ambiguous"):
        ptsl_helpers.calibrate_signalpath_base(object(), sess, live)

    # Malformed / non-string / empty assignments raise zero-match RuntimeError
    bad_map = {sess.track_by_name("KIK").uuid_braced: ["not-hex"], sess.track_by_name("Kick").uuid_braced: [None]}
    monkeypatch.setattr(ptsl_helpers, "get_track_main_output_assignments", lambda _e, tids: bad_map.get(tids[0], []))
    with pytest.raises(RuntimeError, match="no routed tracks matched"):
        ptsl_helpers.calibrate_signalpath_base(object(), sess, live)


def test_clip_gain_shared_curve_multipoint_zero_legacy_and_missing_2637():
    from sessionpreplib.ptx import splice_top_chunk, validate_index_0002

    raw = _build_synthetic_ptx()

    # 1) 0.0 dB on untouched clip leaves clip_gain_curve_index == -1
    p_zero, _ = patch_clip_gains(raw, {"Kick": 0.0})
    s_zero = PtxSession.from_bytes(p_zero)
    assert s_zero.clips[0].clip_gain_curve_index == -1 and s_zero.clips[0].gain_db == 0.0

    # 2) Shared 0x2637 curve across Kick & Snare.L + 0x2637 trailer bytes preserved
    dec, delta = decrypt_ptx(raw)
    buf = bytearray(dec)
    rec0 = struct.pack("<BBHIHIHHIIf", 1, 0x46, 1, 22, 0, 1, 4, 0, 48000, 0, 2.5)
    # Multi-point curve (plen=30, total_points=2, extra=1) at index 1
    rec1 = struct.pack("<BBHIHIHHIIfIf", 1, 0x46, 1, 30, 0, 2, 4, 1, 48000, 0, 1.0, 1000, -4.0)
    trailer = b"\xde\xad\xbe\xef"
    c37_body = struct.pack("<I", 2) + rec0 + rec1 + trailer
    splice_top_chunk(buf, 0x2637, _zchunk(0x2637, c37_body, version=1))
    # Point clip 0 (Kick) and clip 1 (Snare.L) both at curve 0, and clip 2 (Snare.R) at multi-point curve 1
    c262a = PtxSession(bytes(buf), xor_delta=delta).chunk_map[0x262A]
    subs = find_subchunks(c262a.raw, 0x2629)
    for ci, curve_idx in ((0, 0), (1, 0), (2, 1)):
        sub_off = c262a.offset + c262a.raw.find(subs[ci])
        b28 = find_subchunks(subs[ci], 0x2628)[0]
        b28_off = sub_off + subs[ci].find(b28, 9)
        struct.pack_into("<i", buf, b28_off + len(b28) - 22, curve_idx)
    shared_raw = encrypt_ptx(bytes(buf), delta)

    # Patching Kick must NOT alter Snare.L's shared curve 0, and must preserve trailer
    p_kick, _ = patch_clip_gains(shared_raw, {"Kick": -7.5})
    dec_k, _ = decrypt_ptx(p_kick)
    assert validate_index_0002(dec_k)
    sk = PtxSession.from_bytes(p_kick)
    assert sk.clips[0].gain_db == pytest.approx(-7.5, abs=0.02)
    assert sk.clips[0].clip_gain_curve_index == 2
    assert sk.clips[1].gain_db == pytest.approx(2.5, abs=0.02)
    assert sk.chunk_map[0x2637].raw.endswith(trailer)

    # Patching Snare (whose .R clip points to multi-point curve 1) appends fresh static curve without corrupting curve 1
    p_snr, _ = patch_clip_gains(p_kick, {"Snare": -3.0})
    ss = PtxSession.from_bytes(p_snr)
    assert ss.clips[1].gain_db == pytest.approx(-3.0, abs=0.02)
    assert ss.clips[2].gain_db == pytest.approx(-3.0, abs=0.02)
    assert ss.clips[2].clip_gain_curve_index != 1

    # 3) Legacy 0x2628 version < 5 -> parser returns 0.0 dB / -1, patcher raises RuntimeError
    dec_l, delta_l = decrypt_ptx(raw)
    buf_l = bytearray(dec_l)
    c2a_l = PtxSession(bytes(buf_l), xor_delta=delta_l).chunk_map[0x262A]
    s0 = find_subchunks(c2a_l.raw, 0x2629)[0]
    b28_0 = find_subchunks(s0, 0x2628)[0]
    b28_0_off = c2a_l.offset + c2a_l.raw.find(s0) + s0.find(b28_0, 9)
    struct.pack_into("<H", buf_l, b28_0_off + 1, 4)  # version 4
    legacy_raw = encrypt_ptx(bytes(buf_l), delta_l)
    assert PtxSession.from_bytes(legacy_raw).clips[0].clip_gain_curve_index == -1
    with pytest.raises(RuntimeError, match="unsupported 0x2628 version 4"):
        patch_clip_gains(legacy_raw, {"Kick": 3.0})

    # 4) Missing 0x2637 raises RuntimeError
    dec_m, delta_m = decrypt_ptx(raw)
    buf_m = bytearray(dec_m)
    c37_m = PtxSession(bytes(buf_m), xor_delta=delta_m).chunk_map[0x2637]
    struct.pack_into("<H", buf_m, c37_m.offset + 7, 0x7FFD)  # rename tag away from 0x2637
    with pytest.raises(RuntimeError, match="Missing 0x2637"):
        patch_clip_gains(encrypt_ptx(bytes(buf_m), delta_m), {"Kick": 3.0})


def test_patch_clear_track_inputs_shrinks_enclosing_261b_and_aligns_trailing_chunk():
    from sessionpreplib.ptx import patch_clear_track_inputs, validate_index_0002

    raw = _build_synthetic_ptx()
    cleared, meta = patch_clear_track_inputs(raw, ["Kick", "Snare"])
    assert meta["Kick"]["cleared"] is True and meta["Snare"]["cleared"] is True

    dec, _ = decrypt_ptx(cleared)
    assert validate_index_0002(dec)
    sess = PtxSession(dec)
    assert sess.track_by_name("Kick").input_path is None
    assert sess.track_by_name("Snare").input_path is None

    # Every cleared 0x261C block's enclosing 0x261B chunk MUST end directly on the trailing 0x2038 Z-chunk (0x5A)
    c2624 = sess.chunk_map[0x2624].raw
    pos = 13
    checked_audio = 0
    for _ in range(struct.unpack_from("<I", c2624, 9)[0]):
        sz = struct.unpack_from("<I", c2624, pos + 3)[0]
        tag = struct.unpack_from("<H", c2624, pos + 7)[0]
        blk = c2624[pos : pos + 7 + sz]
        if tag == 0x261C:
            checked_audio += 1
            assert blk[9] == 0x5A and struct.unpack_from("<H", blk, 16)[0] == 0x261B
            b1b_end = 9 + 7 + struct.unpack_from("<I", blk, 12)[0]
            assert b1b_end + 9 <= len(blk)
            assert blk[b1b_end] == 0x5A and struct.unpack_from("<H", blk, b1b_end + 7)[0] == 0x2038
        pos += 7 + sz
    assert checked_audio == 2
