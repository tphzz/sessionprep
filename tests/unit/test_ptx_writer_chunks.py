"""Unit tests for verified PTX chunk serializers, 0x0002 index mutators, and offline color mapper."""

from __future__ import annotations

import struct

import pytest

from sessionpreplib.ptx import (
    PT12_COLOR_PALETTE,
    PtxDirTreeEntry,
    PtxSession,
    build_zchunk,
    closest_pt_color_index,
    decrypt_ptx,
    encode_lp_str,
    encrypt_ptx,
    generate_ptx_uuid,
    iter_top_chunks,
    parse_index_0002,
    patch_session_audio_format_1028,
    replace_index_0002_chunk,
    serialize_audio_desc_1001,
    serialize_clip_core_2628_dims,
    serialize_directory_tree_103a,
    serialize_lane_1052,
    serialize_placement_104f,
    serialize_playlist_1014,
    shift_parsed_index_0002,
    validate_index_0002,
)
from sessionpreplib.ptx.parser_timeline import (
    _parse_region_core_2628,
    parse_audio_files_1004,
    parse_playlists_1015,
    parse_timeline_lanes_1054,
)
from tests.unit.test_ptx_core import _build_synthetic_ptx


def test_patch_session_audio_format_1028_roundtrip_and_guards():
    raw = _build_synthetic_ptx()
    patched = patch_session_audio_format_1028(raw, sample_rate=96000, bit_depth=32, audio_format_code=3)
    sess = PtxSession.from_bytes(patched)
    assert sess.info.sample_rate == 96000
    assert sess.info.bit_depth == 32
    assert sess.info.audio_format_code == 3
    dec, _ = decrypt_ptx(patched)
    assert validate_index_0002(dec) is True

    # Partial patch preserves untouched fields and non-zero k0 encrypts cleanly
    dec0, delta0 = decrypt_ptx(raw)
    re_k0 = encrypt_ptx(dec0, delta0, k0=0x37)
    p_sr = patch_session_audio_format_1028(re_k0, sample_rate=88200)
    assert PtxSession.from_bytes(p_sr).info.sample_rate == 88200
    assert PtxSession.from_bytes(p_sr).info.bit_depth == 24

    for kwargs in (
        {"sample_rate": 12345},
        {"sample_rate": True},
        {"bit_depth": 8},
        {"bit_depth": False},
        {"audio_format_code": 300},
        {"audio_format_code": True},
    ):
        with pytest.raises(ValueError):
            patch_session_audio_format_1028(raw, **kwargs)  # type: ignore[arg-type]

    # Missing 0x1028 raises RuntimeError
    hdr_only = encrypt_ptx(dec0[:0x1F] + build_zchunk(0x1029, b"\x00" * 8, version=1), delta0)
    with pytest.raises(RuntimeError, match="0x1028"):
        patch_session_audio_format_1028(hdr_only, sample_rate=48000)


def test_serialize_103a_1001_104f_1052_1014_roundtrip_and_variants():
    tree = serialize_directory_tree_103a([
        PtxDirTreeEntry(0, 0xFFFFFFFF, "Macintosh HD", b"\x33\x59\xad\xe3"),
        PtxDirTreeEntry(1, 0, "SessionDir"),
        PtxDirTreeEntry(1, 1, "Audio Files"),
        PtxDirTreeEntry(2, 2, "Kick.wav", b"EVAW"),
    ])
    d1001 = serialize_audio_desc_1001(48000, 1, 24, 96000, format_code=0x11)
    c1003 = build_zchunk(0x1003, struct.pack("<I", 3) + d1001, version=9)
    c1004 = build_zchunk(0x1004, struct.pack("<I", 1) + tree + c1003, version=3)
    root, files = parse_audio_files_1004(c1004)
    assert root == "/Volumes/Macintosh HD/SessionDir"
    assert len(files) == 1
    assert files[0].filename == "Kick.wav" and files[0].relative_path == "Audio Files/Kick.wav"
    assert files[0].sample_rate == 48000 and files[0].length_samples == 96000 and files[0].bit_depth == 24

    # Wrapped v3, wrapped v2, and unwrapped 0x1052 lanes
    for wrap, wver, lver in ((True, 3, 3), (True, 2, 3), (False, 2, 2)):
        lane = serialize_lane_1052("Kick", [(0, 48000, 3, 25)], wrap_1050=wrap, wrap_version=wver, lane_version=lver)
        c1054 = build_zchunk(0x1054, struct.pack("<I", 1) + lane, version=2)
        lanes = parse_timeline_lanes_1054(c1054, [], [], [], 48000)
        assert len(lanes) == 1 and lanes[0][0] == "Kick" and len(lanes[0][1]) == 1
        assert lanes[0][1][0].start_sample == 48000 and lanes[0][1][0].color_index == 25

    # v9 stereo playlist, v6 mono playlist, and empty lane list
    pl_uid = generate_ptx_uuid()
    pl_st = serialize_playlist_1014("Synth", [0, 1], pl_uid, version=9)
    pl_v6 = serialize_playlist_1014("Kick", [0], pl_uid, version=6)
    pl_empty = serialize_playlist_1014("Empty", [], pl_uid, version=9)
    c1015 = build_zchunk(0x1015, struct.pack("<I", 3) + pl_st + pl_v6 + pl_empty, version=2)
    pls = parse_playlists_1015(c1015, [], [])
    assert [(p.name, p.lane_indices) for p in pls] == [("Synth", [0, 1]), ("Kick", [0]), ("Empty", [])]
    assert len(serialize_placement_104f(0, 0)) == 44


def test_serialize_clip_core_2628_dims_parser_roundtrip():
    for start_s, length_s, sync_s in (
        (0, 96000, 0),
        (0, 96000, 500),
        (10000, 96000, 500),
        (0x1_0000_0000, 48000 * 60, 123456),
    ):
        dims = serialize_clip_core_2628_dims(start_s, length_s, sync_s)
        c2628 = build_zchunk(0x2628, encode_lp_str("Kick.L") + dims, version=7)
        parsed = _parse_region_core_2628(c2628)
        assert parsed is not None
        name, parsed_start, parsed_sync, parsed_len, *_ = parsed
        assert (name, parsed_start, parsed_sync, parsed_len) == ("Kick.L", start_s, sync_s, length_s)

    with pytest.raises(ValueError):
        serialize_clip_core_2628_dims(-1, 100)
    with pytest.raises(ValueError):
        serialize_clip_core_2628_dims(0, True)  # type: ignore[arg-type]


def test_serializer_validation_guards():
    with pytest.raises(ValueError):
        serialize_directory_tree_103a([PtxDirTreeEntry(300, 0, "bad")])
    with pytest.raises(ValueError):
        serialize_directory_tree_103a([PtxDirTreeEntry(1, 0, "bad", b"123")])
    with pytest.raises(ValueError):
        serialize_audio_desc_1001(0, 1, 24, 100)
    with pytest.raises(ValueError):
        serialize_audio_desc_1001(48000, 0, 24, 100)
    with pytest.raises(ValueError):
        serialize_audio_desc_1001(48000, 1, 8, 100)
    with pytest.raises(ValueError):
        serialize_audio_desc_1001(48000, 1, 24, 100, tail16=b"\x00" * 4)
    with pytest.raises(ValueError):
        serialize_placement_104f(-1, 0)
    with pytest.raises(ValueError):
        serialize_playlist_1014("bad", [0], b"\x00" * 8)
    with pytest.raises(ValueError):
        serialize_playlist_1014("bad", [70000], b"\x00" * 16)


def test_shift_and_replace_index_0002_full_lifecycle():
    dec, _ = decrypt_ptx(_build_synthetic_ptx())
    idx = parse_index_0002(dec)
    assert idx is not None

    # delta=0 returns an isolated deep copy
    copy0 = shift_parsed_index_0002(idx, threshold=0, delta=0)
    assert copy0 == idx and copy0 is not idx
    copy0.entries[0].occurrences.clear()
    assert len(idx.entries[0].occurrences) > 0

    # Positive shift after splicing a chunk into the stream updates 0x0002 + 0x1B cleanly
    chunks = iter_top_chunks(dec, strict=True)
    c1028 = next(c for c in chunks if c.tag == 0x1028)
    extra = build_zchunk(0x1028, c1028.raw[9:] + b"\x00" * 16, version=c1028.version)
    delta = len(extra) - len(c1028.raw)
    assert delta == 16
    buf = bytearray(dec)
    old_end = c1028.offset + 7 + c1028.size
    buf[c1028.offset : old_end] = extra
    shifted = shift_parsed_index_0002(idx, threshold=old_end, delta=delta)
    # Wildcard -1 steps stay -1; offsets below threshold stay unchanged; offsets >= threshold shift by +16
    for orig_e, new_e in zip(idx.entries, shifted.entries, strict=True):
        for orig_s, new_s in zip(orig_e.path, new_e.path, strict=True):
            if orig_s.offset == -1 or orig_s.offset < old_end:
                assert new_s.offset == orig_s.offset
            else:
                assert new_s.offset == orig_s.offset + delta
    replace_index_0002_chunk(buf, shifted)
    assert validate_index_0002(bytes(buf)) is True

    # Negative shift and upper-bound overflow guards (u32 occurrences + signed i32 path step)
    from sessionpreplib.ptx import PtxIndex0002, PtxIndexEntry, PtxIndexPathStep

    with pytest.raises(ValueError):
        shift_parsed_index_0002(idx, threshold=0, delta=-100_000)
    with pytest.raises(ValueError):
        shift_parsed_index_0002(idx, threshold=0, delta=0x1_0000_0000)
    idx_with_path = PtxIndex0002(
        version=1,
        entries=[PtxIndexEntry(path=[PtxIndexPathStep(tag=0x2519, offset=100)], occurrences=[])],
    )
    with pytest.raises(ValueError):
        shift_parsed_index_0002(idx_with_path, threshold=0, delta=0x8000_0000)

    # Non-trailing or missing 0x0002 raises RuntimeError
    non_trailing = bytearray(dec) + build_zchunk(0x1029, b"\x00\x00", version=1)
    with pytest.raises(RuntimeError, match="not trailing"):
        replace_index_0002_chunk(non_trailing, idx)
    no_0002 = bytearray(dec[:0x1F] + build_zchunk(0x1029, b"\x00\x00", version=1))
    with pytest.raises(RuntimeError, match="Missing trailing 0x0002"):
        replace_index_0002_chunk(no_0002, idx)


def test_closest_pt_color_index_valid_and_invalid_inputs():
    r, g, b = PT12_COLOR_PALETTE[25]
    assert closest_pt_color_index(f"#FF{r:02X}{g:02X}{b:02X}") == 25
    assert closest_pt_color_index(f"{r:02x}{g:02x}{b:02x}") == 25
    for bad in ("not-a-color", "", "#-10000", "#12 456", "#12345", "#123456789"):
        assert closest_pt_color_index(bad) is None
