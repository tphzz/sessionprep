"""Unit tests for PTX binary writer primitives and 0x0002 chunk-offset index roundtrip codec."""

from __future__ import annotations

import pytest

from sessionpreplib.ptx import (
    PtxAutomationPoint,
    PtxIndex0002,
    PtxIndexEntry,
    PtxIndexOccurrence,
    PtxIndexPathStep,
    PtxSession,
    build_zchunk,
    decrypt_ptx,
    encode_control_curve,
    encode_lp_str,
    encode_packed_uint,
    generate_ptx_uuid,
    iter_top_chunks,
    parse_control_curve,
    parse_index_0002,
    read_lp_str,
    serialize_clip_gain_2637,
    serialize_folder_tree_210c,
    serialize_index_0002,
)
from sessionpreplib.ptx.models import UUID_PREFIX
from sessionpreplib.ptx.parser_timeline import parse_clip_gain_curves_2637
from tests.unit.test_ptx_core import _build_synthetic_ptx


def test_zchunk_lp_str_uuid_and_packed_uint_primitives():
    chunk = build_zchunk(0x2637, b"\x01\x02\x03\x04", version=2)
    assert chunk[:1] == b"Z"
    assert len(chunk) == 13
    with pytest.raises(ValueError):
        build_zchunk(0x10000, b"")
    with pytest.raises(ValueError):
        build_zchunk(0x1000, b"", version=-1)

    encoded = encode_lp_str("Kick_Ω")
    decoded, nxt = read_lp_str(encoded, 0)
    assert decoded == "Kick_Ω" and nxt == len(encoded)

    u1, u2 = generate_ptx_uuid(), generate_ptx_uuid()
    assert len(u1) == 16 and u1.startswith(UUID_PREFIX) and u1 != u2

    assert encode_packed_uint(0) == (0, b"")
    assert encode_packed_uint(255) == (1, b"\xff")
    assert encode_packed_uint(256) == (2, b"\x00\x01")
    assert encode_packed_uint(70000)[0] == 4
    assert encode_packed_uint(1 << 40)[0] == 8
    with pytest.raises(ValueError):
        encode_packed_uint(-1)
    with pytest.raises(ValueError):
        encode_packed_uint(1 << 64)


def test_control_curve_encode_parse_roundtrip_and_guards():
    # Float32 Samples curve (clip gain style)
    pts_f = [PtxAutomationPoint(48000, 3.5), PtxAutomationPoint(0, -2.0), PtxAutomationPoint(96000, 0.0)]
    raw_f = encode_control_curve("Samples", pts_f, 4, flags=1)
    parsed_f = parse_control_curve(raw_f, 0)
    assert parsed_f is not None
    tb, dec_pts, end = parsed_f
    assert tb == "Samples" and end == len(raw_f)
    assert [(p.position, pytest.approx(p.value, abs=1e-5)) for p in dec_pts] == [
        (0, -2.0), (48000, 3.5), (96000, 0.0),
    ]

    # Int16 Ticks curve (volume tenths of dB)
    pts_i16 = [PtxAutomationPoint(0, -60.0), PtxAutomationPoint(960000, 0.0)]
    raw_i16 = encode_control_curve("Ticks", pts_i16, 2, flags=1)
    tb16, dec16, _ = parse_control_curve(raw_i16, 0)
    assert tb16 == "Ticks" and [(p.position, p.value) for p in dec16] == [(0, -60.0), (960000, 0.0)]

    # Signed int32 plugin automation & u64 warp markers
    raw_i32 = encode_control_curve("Samples", [PtxAutomationPoint(100, -1000.0)], 4, signed_int32=True)
    _, dec32, _ = parse_control_curve(raw_i32, 0, signed_int32=True)
    assert dec32[0].value == -1000.0

    raw_u64 = encode_control_curve("Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, 44100.0)], 8)
    _, dec64, _ = parse_control_curve(raw_u64, 0)
    assert [p.value for p in dec64] == [0.0, 44100.0]

    with pytest.raises(ValueError):
        encode_control_curve("Samples", [], 4)
    with pytest.raises(ValueError):
        encode_control_curve("BadTB", [PtxAutomationPoint(0, 0.0)], 4)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(0, 0.0)], 4, flags=300)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(-100, 0.0), PtxAutomationPoint(500, 1.0)], 4)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(0, float("nan"))], 4)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(0, 1e40)], 4)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(0, -1.0)], 8)
    with pytest.raises(ValueError):
        encode_control_curve("Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(1 << 33, 1.0)], 4)


def test_clip_gain_2637_and_folder_tree_210c_serializers():
    raw_2637 = serialize_clip_gain_2637([
        ("Samples", [PtxAutomationPoint(0, 4.25)], 1),
        ("Samples", [PtxAutomationPoint(0, 0.0), PtxAutomationPoint(48000, -6.0)], 1),
    ])
    curves = parse_clip_gain_curves_2637(raw_2637, 48000)
    assert len(curves) == 2
    assert curves[0].initial_gain_db == pytest.approx(4.25, abs=1e-4)
    assert curves[1].is_dynamic is True and curves[1].gain_db_at(24000) == pytest.approx(-3.0, abs=1e-4)

    f_uid, c1_uid, c2_uid = generate_ptx_uuid(), generate_ptx_uuid(), generate_ptx_uuid()
    raw_210c = serialize_folder_tree_210c([(f_uid, [c1_uid, c2_uid])])
    assert raw_210c[:1] == b"Z" and f_uid in raw_210c and c1_uid in raw_210c and c2_uid in raw_210c
    with pytest.raises(ValueError):
        serialize_folder_tree_210c([(b"short", [])])
    with pytest.raises(ValueError):
        serialize_folder_tree_210c([(f_uid, [b"short"])])


def test_index_0002_parse_and_serialize_byte_exact_roundtrip():
    dec, _ = decrypt_ptx(_build_synthetic_ptx())
    c2_orig = next(c for c in iter_top_chunks(dec, strict=True) if c.tag == 0x0002)
    parsed = parse_index_0002(dec)
    assert parsed is not None
    assert serialize_index_0002(parsed) == c2_orig.raw

    # Custom multi-step entry roundtrip & validation guards
    custom = PtxIndex0002(
        version=2,
        entries=[
            PtxIndexEntry(
                path=[PtxIndexPathStep(0x2519, 0x100, b"\x01\x02\x03\x04\x05"), PtxIndexPathStep(0x251A, -1)],
                occurrences=[PtxIndexOccurrence(flag=0, offsets=(0x120, 0x240), tail=b"\x0a\x0b\x0c\x0d\x0e\x0f")],
            )
        ],
    )
    raw_custom = serialize_index_0002(custom)
    reparsed = parse_index_0002(b"\x00" * 0x1F + raw_custom)
    assert reparsed == custom
    assert PtxSession.from_bytes(_build_synthetic_ptx()).tracks
    with pytest.raises(ValueError):
        serialize_index_0002(PtxIndex0002(entries=[PtxIndexEntry(path=[PtxIndexPathStep(1, 0, b"123")])]))
    with pytest.raises(ValueError):
        serialize_index_0002(PtxIndex0002(entries=[PtxIndexEntry(path=[PtxIndexPathStep(1, -2)])]))
    with pytest.raises(ValueError):
        serialize_index_0002(PtxIndex0002(entries=[PtxIndexEntry(occurrences=[PtxIndexOccurrence(0, (1,), b"12")])]))
    with pytest.raises(ValueError):
        serialize_index_0002(PtxIndex0002(entries=[PtxIndexEntry(occurrences=[PtxIndexOccurrence(0, tuple(range(16384)))])]))
