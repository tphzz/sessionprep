"""Unit tests for PTSL-discovered PTX reader fields, extended patchers, markers, and non-Audio track synthesis."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from sessionpreplib.ptx import (
    ZERO_TICKS,
    OfflineAudioTrackRequest,
    OfflineGroupRequest,
    OfflineMarkerRequest,
    OfflineNonAudioTrackRequest,
    OfflinePtxWriteRequest,
    PtxPatchBuilder,
    PtxSession,
    PtxValidationError,
    decrypt_ptx,
    encrypt_ptx,
    format_pull_rate,
    generate_ptx_uuid,
    patch_add_marker,
    patch_delete_marker,
    patch_rename_tracks,
    patch_session_settings,
    patch_track_states,
    validate_index_0002,
    write_offline_ptx_session,
)
from sessionpreplib.ptx.lowlevel import (
    build_aux_or_rf_mixer_strip_261e,
    build_basic_folder_mixer_strip_263b,
    build_master_mixer_strip_261d,
    build_vca_mixer_strip_261f,
    build_zchunk,
    encode_lp_str,
    iter_top_chunks,
    parse_pull_rate_codes,
    patch_session_audio_format_1028,
    patch_session_timecode_204d,
    rebuild_index_0002,
    serialize_index_0002,
    serialize_marker_2077,
)
from sessionpreplib.ptx.models import PTX_MAGIC
from sessionpreplib.ptx.writer_blueprints import (
    build_audio_mixer_strip_261c,
    build_audio_ui_entry_2589,
    build_track_entry_251a,
    build_track_summary_2519,
    build_track_uuid_entry_210b,
)
from sessionpreplib.ptx.writer_primitives import (
    PtxIndex0002,
    PtxIndexEntry,
    PtxIndexOccurrence,
    PtxIndexPathStep,
)


def _build_modern_1028(
    sample_rate: int = 48000,
    bit_depth: int = 24,
    audio_format_code: int = 3,
    is_interleaved: bool = True,
) -> bytes:
    """Build a full 126-byte modern Pro Tools ``0x1028`` audio format chunk."""
    body = bytearray(117)
    body[0] = audio_format_code
    body[1] = 24 if bit_depth == 32 else bit_depth
    struct.pack_into("<I", body, 2, sample_rate)
    body[-8] = 1 if bit_depth == 32 else 0
    body[-7] = 1 if is_interleaved else 0
    body[-6] = bit_depth
    return build_zchunk(0x1028, bytes(body), version=4)


def _build_modern_204d(fps_code: int = 1, start_frames: int = 0, ff_code: int = 1) -> bytes:
    """Build a canonical 76-byte ``0x204D`` session timecode & Feet+Frames chunk."""
    tc = b"06:00:00;00" if fps_code in (5, 8) else b"06:00:00:00"
    body = (
        struct.pack("<I", fps_code)
        + (b"\x00" * 5)
        + struct.pack("<I", start_frames)
        + (b"\x00" * 4)
        + bytes([ff_code])
        + (b"\x00" * 3)
        + struct.pack("<II", 2, 11)
        + tc
        + struct.pack("<III", fps_code, start_frames, 2)
        + struct.pack("<I", 11)
        + tc
    )
    return build_zchunk(0x204D, body, version=7)


def _build_2587_container(entries: list[bytes]) -> bytes:
    c2581 = build_zchunk(0x2581, b"\x00\x00\x00\x00", version=1)
    c258a = build_zchunk(0x258A, c2581 + struct.pack("<I", len(entries)) + b"".join(entries), version=1)
    c258b = build_zchunk(0x258B, c258a, version=1)
    c2551 = build_zchunk(0x2551, c258b, version=1)
    return build_zchunk(0x2587, c2551, version=1)


def _build_blank_modern_ptx(
    *,
    with_tracks: bool = False,
    sample_rate: int = 48000,
    bit_depth: int = 24,
) -> bytes:
    """Synthesize a self-contained modern encrypted ``.ptx`` template with valid ``0x0002``."""
    ver_body = b"\x03\x00\x00\x00" + encode_lp_str("Pro Tools") + struct.pack("<III", 26, 11, 2)
    c2067 = build_zchunk(0x2067, ver_body, version=1)
    c1028 = _build_modern_1028(sample_rate=sample_rate, bit_depth=bit_depth)
    c204d = _build_modern_204d()
    c2030 = build_zchunk(0x2030, struct.pack("<I", 0) + (b"\x00" * 8), version=5)
    c202b = build_zchunk(0x202B, struct.pack("<II", 0, 0) + b"\x00\x00", version=4)

    rows: list[bytes] = []
    entries_251a: list[bytes] = []
    items_210b: list[bytes] = []
    strips: list[bytes] = []
    ui_entries: list[bytes] = []
    if with_tracks:
        uid_a = generate_ptx_uuid()
        uid_rf = generate_ptx_uuid()
        uid_vca = generate_ptx_uuid()
        rows = [
            build_track_summary_2519("Audio_1", uid_a, 2, 1, track_type="Audio"),
            build_track_summary_2519("Bus_Folder", uid_rf, 2, 2, track_type="RoutingFolder"),
            build_track_summary_2519("VCA_1", uid_vca, 1, 3, track_type="Vca"),
        ]
        entries_251a = [
            build_track_entry_251a("Audio_1", uid_a, 2, 1, track_type="Audio", has_clips=False),
            build_track_entry_251a("Bus_Folder", uid_rf, 2, 2, track_type="RoutingFolder", has_clips=False),
            build_track_entry_251a("VCA_1", uid_vca, 1, 3, track_type="Vca", has_clips=False),
        ]
        items_210b = [
            build_track_uuid_entry_210b("Audio_1", uid_a, type_code=0),
            build_track_uuid_entry_210b("Bus_Folder", uid_rf, type_code=2),
            build_track_uuid_entry_210b("VCA_1", uid_vca, type_code=9),
        ]
        strips = [
            build_audio_mixer_strip_261c("Audio_1", uid_a, 2, None, track_ordinal_1based=1, id_base=10000),
            build_aux_or_rf_mixer_strip_261e("Bus_Folder", uid_rf, 2, None, is_routing_folder=True, track_ordinal_1based=2, id_base=10010),
            build_vca_mixer_strip_261f("VCA_1", uid_vca, track_ordinal_1based=3),
        ]
        ui_entries = [
            build_audio_ui_entry_2589(track_index=0),
            build_audio_ui_entry_2589(track_index=1, is_folder_open=True),
            build_audio_ui_entry_2589(track_index=2),
        ]

    cnt_le = struct.pack("<I", len(rows))
    c2519 = build_zchunk(
        0x2519, (b"\x00" * 10) + struct.pack("<I", 0) + cnt_le + b"".join(rows) + b"".join(entries_251a) + cnt_le + b"".join(entries_251a),
        version=10,
    )
    c2107 = build_zchunk(0x2107, (b"\x00" * 9) + struct.pack("<I", len(items_210b)) + b"".join(items_210b), version=1)
    c2624 = build_zchunk(0x2624, struct.pack("<I", len(strips)) + b"".join(strips), version=1)
    c2587 = _build_2587_container(ui_entries)

    header = bytearray(0x1F)
    header[: len(PTX_MAGIC)] = PTX_MAGIC
    header[0x12] = 0x05
    header[0x13] = 0x00
    header[0x14] = 0x5A
    header[0x15] = 0x01
    header[0x17] = 0x04
    body_chunks = [c2067, c1028, c204d, c2030, c2107, c2519, c2624, c202b, c2587]
    pre_body = b"".join(body_chunks)
    struct.pack_into("<I", header, 0x1B, 0x1F + len(pre_body))
    pre_idx = bytes(header) + pre_body

    top = iter_top_chunks(pre_idx, strict=True)
    seed_entries = [
        PtxIndexEntry(
            path=[PtxIndexPathStep(ch.tag, -1, b"\x00" * 5)],
            occurrences=[PtxIndexOccurrence(1, (ch.offset,), b"\x00" * 6)],
        )
        for ch in top
    ]
    seed_idx = PtxIndex0002(version=1, entries=seed_entries)
    rebuilt = rebuild_index_0002(seed_idx, pre_idx, pre_idx)
    dec = pre_idx + serialize_index_0002(rebuilt)
    assert validate_index_0002(dec)
    return encrypt_ptx(dec, delta=0x13, k0=0x5A)


def test_pull_rate_helpers_roundtrip_all_combinations() -> None:
    expected = {
        (0, 0): "None",
        (0, 1): "+0.1%",
        (0, 2): "-0.1%",
        (1, 0): "+4%",
        (1, 1): "+4% +0.1%",
        (1, 2): "+4% -0.1%",
        (2, 0): "-4%",
        (2, 1): "-4% +0.1%",
        (2, 2): "-4% -0.1%",
    }
    for codes, label in expected.items():
        assert format_pull_rate(*codes) == label
        assert parse_pull_rate_codes(label) == codes
    assert parse_pull_rate_codes("") == (0, 0)
    with pytest.raises(ValueError, match="Unsupported Pro Tools pull rate"):
        parse_pull_rate_codes("+99%")


def test_patch_session_settings_1028_and_204d_roundtrip_and_guards() -> None:
    raw = _build_blank_modern_ptx()
    patched, summary = patch_session_settings(
        raw,
        sample_rate=96000,
        bit_depth=32,
        audio_format="AIFF",
        is_interleaved=False,
        audio_pull_rate="-4% +0.1%",
        video_pull_rate="-0.1%",
        fps_code=5,
        start_frames=107892,
        feet_frames_rate="25",
    )
    assert summary["bit_depth"] == 32
    assert summary["audio_format"] == "AIFF"
    assert summary["feet_frames_rate"] == "25"

    dec, delta = decrypt_ptx(patched)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    assert sess.info.sample_rate == 96000
    assert sess.info.bit_depth == 32
    assert sess.info.audio_format_name == "AIFF"
    assert sess.info.is_interleaved is False
    assert sess.info.audio_pull_rate == "-4% +0.1%"
    assert sess.info.video_pull_rate == "-0.1%"
    assert sess.info.fps_code == 5
    assert sess.info.is_drop_frame is True
    assert sess.info.start_frames == 107892
    assert sess.info.feet_frames_rate == "25"

    # Toggle back to 16-bit WAV interleaved with 24fps and 23.976 Feet+Frames
    patched2 = patch_session_audio_format_1028(
        patched, bit_depth=16, audio_format="WAV", is_interleaved=True, audio_pull_rate="None", video_pull_rate="None",
    )
    patched2 = patch_session_timecode_204d(patched2, fps_code=1, start_frames=0, feet_frames_rate="23.976")
    dec2, d2 = decrypt_ptx(patched2)
    sess2 = PtxSession(dec2, xor_delta=d2)
    assert sess2.info.bit_depth == 16
    assert sess2.info.audio_format_name == "WAV"
    assert sess2.info.is_interleaved is True
    assert sess2.info.audio_pull_rate == "None"
    assert sess2.info.video_pull_rate == "None"
    assert sess2.info.fps_code == 1
    assert sess2.info.feet_frames_rate == "23.976"

    with pytest.raises(ValueError, match="Unsupported Pro Tools audio format"):
        patch_session_settings(raw, audio_format="FLAC")
    with pytest.raises(ValueError, match="Video pull rate"):
        patch_session_settings(raw, video_pull_rate="+4%")
    with pytest.raises(ValueError, match="Unsupported Pro Tools fps_code"):
        patch_session_settings(raw, fps_code=99)
    with pytest.raises(ValueError, match="Unsupported Pro Tools feet_frames_rate"):
        patch_session_settings(raw, feet_frames_rate="30")


def test_markers_serialize_add_delete_and_reader_roundtrip() -> None:
    raw = _build_blank_modern_ptx(with_tracks=True)
    p1, info1 = patch_add_marker(
        raw, "Intro", 48000, comments="Kick enters", color_index=8, color_rgb=(230, 0, 14),
    )
    assert info1 == {"action": "add_marker", "number": 1, "name": "Intro", "location": "MainRuler", "track_name": ""}

    p2, info2 = patch_add_marker(
        p1, "Chorus", 96000, end_pos=192000, time_properties="Selection", comments="Big chorus",
    )
    assert info2["number"] == 2

    p3, info3 = patch_add_marker(
        p2, "Cue1", ZERO_TICKS + 960000, number=5, reference="FollowTrackTimebase",
        comments="Track cue", color_index=5, color_rgb=(255, 124, 214), track="Audio_1",
    )
    assert info3["location"] == "Track"
    assert info3["track_name"] == "Audio_1"

    dec, delta = decrypt_ptx(p3)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    assert len(sess.markers) == 3
    m1, m2, m3 = sess.markers
    assert (m1.number, m1.name, m1.reference, m1.time_properties) == (1, "Intro", "Absolute", "Marker")
    assert (m1.start_sample, m1.comments, m1.color_index, m1.color_rgb) == (48000, "Kick enters", 8, (230, 0, 14))
    assert (m1.location, m1.track_index, m1.track_name) == ("MainRuler", None, "")
    assert (m2.number, m2.name, m2.time_properties, m2.start_sample, m2.end_sample) == (2, "Chorus", "Selection", 96000, 192000)
    assert (m3.number, m3.name, m3.reference, m3.start_sample) == (5, "Cue1", "FollowTrackTimebase", ZERO_TICKS + 960000)
    assert (m3.location, m3.track_index, m3.track_name, m3.color_rgb) == ("Track", 0, "Audio_1", (255, 124, 214))

    p4, del_info = patch_delete_marker(p3, 2)
    assert del_info == {"action": "delete_marker", "number": 2}
    dec4, d4 = decrypt_ptx(p4)
    assert validate_index_0002(dec4)
    assert [m.number for m in PtxSession(dec4, xor_delta=d4).markers] == [1, 5]

    with pytest.raises(ValueError, match="already exists"):
        patch_add_marker(p3, "Dup", 0, number=1)
    with pytest.raises(KeyError, match="not found"):
        patch_delete_marker(p3, 99)
    with pytest.raises(ValueError, match="1..999"):
        serialize_marker_2077(0, "Bad", 0)


def test_patch_track_states_and_rename_tracks_roundtrip_and_guards() -> None:
    raw = _build_blank_modern_ptx(with_tracks=True)
    patched, summary = patch_track_states(
        raw,
        {
            "Audio_1": {
                "mute": True,
                "solo": True,
                "solo_safe": True,
                "inactive": True,
                "hidden": True,
                "color_index": 36,
                "height": "Large",
            },
            "Bus_Folder": {"folder_open": False, "height": "Mini"},
            "VCA_1": {"mute": True, "height": "Extreme"},
        },
    )
    assert summary["Audio_1"]["height"] == "Large"
    dec, delta = decrypt_ptx(patched)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    t_a = sess.track_by_name("Audio_1")
    t_rf = sess.track_by_name("Bus_Folder")
    t_vca = sess.track_by_name("VCA_1")
    assert t_a is not None and t_rf is not None and t_vca is not None
    assert (t_a.is_muted, t_a.is_soloed, t_a.is_solo_safe, t_a.is_inactive, t_a.is_hidden) == (True, True, True, True, True)
    assert (t_a.color_index, t_a.track_height_px, t_a.track_height_name) == (36, 192, "Large")
    assert (t_rf.is_folder_open, t_rf.track_height_px, t_rf.track_height_name) == (False, 23, "Mini")
    assert (t_vca.is_muted, t_vca.track_height_px, t_vca.track_height_name) == (True, 600, "Extreme")

    # Toggle states back off and rename tracks
    renamed, rmap = patch_rename_tracks(patched, {"Audio_1": "Lead_Vocal_Stereo", "Bus_Folder": "Drums_RF"})
    assert rmap == {"Audio_1": "Lead_Vocal_Stereo", "Bus_Folder": "Drums_RF"}
    unhidden, _ = patch_track_states(
        renamed,
        {
            "Lead_Vocal_Stereo": {"mute": False, "inactive": False, "hidden": False},
            "Drums_RF": {"folder_open": True},
        },
    )
    dec2, d2 = decrypt_ptx(unhidden)
    assert validate_index_0002(dec2)
    sess2 = PtxSession(dec2, xor_delta=d2)
    t_lv = sess2.track_by_name("Lead_Vocal_Stereo")
    t_dr = sess2.track_by_name("Drums_RF")
    assert t_lv is not None and t_dr is not None
    assert (t_lv.is_muted, t_lv.is_inactive, t_lv.is_hidden) == (False, False, False)
    assert t_dr.is_folder_open is True

    with pytest.raises(ValueError, match="Unsupported track state key"):
        patch_track_states(raw, {"Audio_1": {"bogus": True}})
    with pytest.raises(ValueError, match="not a folder track"):
        patch_track_states(raw, {"Audio_1": {"folder_open": False}})
    with pytest.raises(ValueError, match="Track rename collision"):
        patch_rename_tracks(raw, {"Audio_1": "VCA_1"})


def test_non_audio_strip_builders_exact_structure() -> None:
    uid = generate_ptx_uuid()
    aux_st = build_aux_or_rf_mixer_strip_261e("Aux_St", uid, 2, None, is_routing_folder=False, track_ordinal_1based=1, id_base=10000)
    rf_m = build_aux_or_rf_mixer_strip_261e("RF_M", uid, 1, None, is_routing_folder=True, track_ordinal_1based=2, id_base=10010)
    mst_st = build_master_mixer_strip_261d("Mst_St", uid, 2, None, track_ordinal_1based=3, id_base=10020)
    vca = build_vca_mixer_strip_261f("Vca_1", uid, track_ordinal_1based=4)
    bf = build_basic_folder_mixer_strip_263b("BF_1", uid, track_ordinal_1based=5)
    assert struct.unpack_from("<H", aux_st, 7)[0] == 0x261E
    assert struct.unpack_from("<H", rf_m, 7)[0] == 0x261E
    assert struct.unpack_from("<H", mst_st, 7)[0] == 0x261D
    # Master 0x260D must omit mute/LFE curves and populated pan banks (canonical 1719 B with output bus)
    from sessionpreplib.ptx.codec import find_subchunks
    d0 = find_subchunks(mst_st, 0x260D)[0]
    assert len(find_subchunks(d0, 0x260A)) == 2  # main volume + trailing curve only (no mute/LFE/pan curves)
    assert struct.unpack_from("<H", vca, 7)[0] == 0x261F and len(vca) == 705
    assert struct.unpack_from("<H", bf, 7)[0] == 0x263B and len(bf) == 385


def test_review_regressions_rename_sizes_vca_inactive_204d_sep_and_instrument_0002(tmp_path: Path) -> None:
    from sessionpreplib.ptx.codec import find_subchunks, read_zchunk_at, walk_sequential_zchunks
    from sessionpreplib.ptx.writer_track_mutations import mutate_2030_markers_for_track_map

    # 1. Blocker #1: Rename with length change must update enclosing 0x261B and 0x102D Z-chunk sizes
    raw = _build_blank_modern_ptx(with_tracks=True)
    renamed, _ = patch_rename_tracks(raw, {"Audio_1": "Lead_Vocal_Stereo_Renamed"})
    dec_r, d_r = decrypt_ptx(renamed)
    assert validate_index_0002(dec_r)
    c2624 = PtxSession(dec_r, xor_delta=d_r).chunk_map[0x2624].raw
    strips, _ = walk_sequential_zchunks(c2624, 13, struct.unpack_from("<I", c2624, 9)[0])
    st0 = strips[0]
    _t, b261b, _after_b = read_zchunk_at(st0, 9, expected_tag=0x261B)
    _t2, b102d, after_2d = read_zchunk_at(b261b, 9, expected_tag=0x102D)
    _t3, _b2619, _after_19 = read_zchunk_at(b102d, 9, expected_tag=0x2619)
    # 4-byte gap then 0x2627 must align exactly after 0x102D
    _t4, _b2627, _ = read_zchunk_at(b261b, after_2d + 4, expected_tag=0x2627)

    # 2. Blocker #2: VCA inactive=True must NOT corrupt 0x2515 size header
    vca_inact, _ = patch_track_states(renamed, {"VCA_1": {"inactive": True}})
    dec_v, d_v = decrypt_ptx(vca_inact)
    assert validate_index_0002(dec_v)
    sess_v = PtxSession(dec_v, xor_delta=d_v)
    assert sess_v.track_by_name("VCA_1").is_inactive is True  # type: ignore[union-attr]
    c2624_v = sess_v.chunk_map[0x2624].raw
    vca_strip = walk_sequential_zchunks(c2624_v, 13, struct.unpack_from("<I", c2624_v, 9)[0])[0][2]
    _tv, _b102d_v, after_2d_v = read_zchunk_at(vca_strip, 9, expected_tag=0x102D)
    _tv2, b2515_v, _ = read_zchunk_at(vca_strip, after_2d_v, expected_tag=0x2515)
    assert len(b2515_v) == 339

    # 3. Blocker #4: 0x204D drop-frame separator must toggle ':' <-> ';' in both embedded timecode strings
    df_ptx = patch_session_timecode_204d(raw, fps_code=5)
    c204d_df = PtxSession(decrypt_ptx(df_ptx)[0]).chunk_map[0x204D].raw
    assert c204d_df[9 + 29 : 9 + 40] == b"06:00:00;00"
    assert c204d_df[9 + 56 : 9 + 67] == b"06:00:00;00"
    ndf_ptx = patch_session_timecode_204d(df_ptx, fps_code=2)
    c204d_ndf = PtxSession(decrypt_ptx(ndf_ptx)[0]).chunk_map[0x204D].raw
    assert c204d_ndf[9 + 29 : 9 + 40] == b"06:00:00:00"
    assert c204d_ndf[9 + 56 : 9 + 67] == b"06:00:00:00"

    # 4. Important #5/#6: BasicFolder parses is_soloed=False; BasicFolder/Master reject unsupported states
    tpl_path = tmp_path / "bf_mst.ptxt"
    out_path = tmp_path / "bf_mst_out.ptx"
    tpl_path.write_bytes(_build_blank_modern_ptx())
    write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=tpl_path, output_ptx_path=out_path, session_dir=tmp_path,
        sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
        extra_tracks=(
            OfflineNonAudioTrackRequest(track_name="BF_Test", track_type="BasicFolder"),
            OfflineNonAudioTrackRequest(track_name="Mst_Test", track_type="Master", channels=2),
        ),
    ))
    sess_bm = PtxSession.from_file(out_path)
    assert sess_bm.track_by_name("BF_Test").is_soloed is False  # type: ignore[union-attr]
    bm_raw = out_path.read_bytes()
    with pytest.raises(ValueError, match="does not support solo"):
        patch_track_states(bm_raw, {"BF_Test": {"solo": True}})
    with pytest.raises(ValueError, match="does not support mute"):
        patch_track_states(bm_raw, {"Mst_Test": {"mute": True}})

    # 5. Important #12: 0x2030 Track marker remapping and deletion
    m_main = serialize_marker_2077(1, "Main", 0, track_index=-1)
    m_keep = serialize_marker_2077(2, "Keep", 100, track_index=2)
    m_drop = serialize_marker_2077(3, "Drop", 200, track_index=1)
    c2030 = build_zchunk(0x2030, struct.pack("<I", 3) + m_main + m_keep + m_drop + (b"\x00" * 8), version=5)
    remapped = mutate_2030_markers_for_track_map(c2030, {0: 0, 2: 5})
    assert struct.unpack_from("<I", remapped, 9)[0] == 2
    subs = find_subchunks(remapped, 0x2077)
    p4826 = subs[1].find(find_subchunks(subs[1], 0x4826)[0])
    assert struct.unpack_from("<i", subs[1], p4826 - 4)[0] == 5

    # 6. Blocker #3: 3-step 0x2624/0x2621/0x2627 Instrument rack index survives rebuild_index_0002
    uid_inst = generate_ptx_uuid()
    from sessionpreplib.ptx.writer_blueprints import _build_empty_inserts_2627, _build_track_identity_102d
    c2621 = build_zchunk(
        0x2621,
        _build_track_identity_102d("Inst_1", uid_inst) + b"\x01\xff\xff\x01" + _build_empty_inserts_2627(11),
        version=1,
    )
    c2624_inst = build_zchunk(0x2624, struct.pack("<I", 1) + c2621, version=1)
    hdr = bytearray(0x1F)
    hdr[: len(PTX_MAGIC)] = PTX_MAGIC
    hdr[0x12], hdr[0x14], hdr[0x15], hdr[0x17] = 0x05, 0x5A, 0x01, 0x04
    pre_inst = bytes(hdr) + c2624_inst
    struct.pack_into("<I", hdr, 0x1B, len(pre_inst))
    pre_inst = bytes(hdr) + c2624_inst
    seed = PtxIndex0002(1, [PtxIndexEntry([PtxIndexPathStep(0x2624, -1, b"\x00" * 5)], [PtxIndexOccurrence(1, (0x1F,), b"\x00" * 6)])])
    rebuilt_inst = rebuild_index_0002(seed, pre_inst, pre_inst)
    dec_inst = pre_inst + serialize_index_0002(rebuilt_inst)
    assert validate_index_0002(dec_inst)
    assert any(tuple(s.tag for s in e.path) == (0x2624, 0x2621, 0x2627) for e in rebuilt_inst.entries)


def test_offline_writer_synthesizes_all_track_types_markers_states_and_builder(tmp_path: Path) -> None:
    tpl_path = tmp_path / "blank.ptxt"
    out_path = tmp_path / "out.ptx"
    tpl_path.write_bytes(_build_blank_modern_ptx())

    req = OfflinePtxWriteRequest(
        ptxt_path=tpl_path,
        output_ptx_path=out_path,
        session_dir=tmp_path,
        sample_rate=48000,
        bit_depth=24,
        audio_format="WAV",
        is_interleaved=False,
        audio_pull_rate="+0.1%",
        video_pull_rate="-0.1%",
        fps_code=2,
        start_frames=90000,
        feet_frames_rate="24",
        audio_files=(),
        tracks=(OfflineAudioTrackRequest(track_name="Kick_Audio", channels=2, color_index=32),),
        extra_tracks=(
            OfflineNonAudioTrackRequest(track_name="Verb_Aux", track_type="Aux", channels=2, color_index=44),
            OfflineNonAudioTrackRequest(track_name="Mono_Aux", track_type="Aux", channels=1, color_index=45),
            OfflineNonAudioTrackRequest(track_name="Master_St", track_type="Master", channels=2, color_index=56),
            OfflineNonAudioTrackRequest(track_name="Master_M", track_type="Master", channels=1, color_index=57),
            OfflineNonAudioTrackRequest(track_name="VCA_All", track_type="Vca", color_index=68),
            OfflineNonAudioTrackRequest(track_name="BF_Vocals", track_type="BasicFolder", is_folder_open=False),
            OfflineNonAudioTrackRequest(track_name="RF_Drums", track_type="RoutingFolder", channels=2, is_folder_open=False),
        ),
        track_states={"Verb_Aux": {"mute": True, "height": "Jumbo"}, "Master_M": {"inactive": True}},
        markers=(
            OfflineMarkerRequest(name="Intro", start_pos=48000, comments="Start"),
            OfflineMarkerRequest(name="Hit", start_pos=ZERO_TICKS, reference="FollowTrackTimebase", track_name="Kick_Audio"),
        ),
        group_requests=(
            OfflineGroupRequest(name="Band", action="create", member_tracks=("Kick_Audio", "Verb_Aux"), vca_track="VCA_All"),
        ),
    )
    res = write_offline_ptx_session(req)
    assert len(res.created_tracks) == 8

    sess = PtxSession.from_file(out_path)
    assert not sess.warnings
    assert [(t.name, t.track_type, t.format) for t in sess.tracks] == [
        ("Kick_Audio", "Audio", "Stereo"),
        ("Verb_Aux", "Aux", "Stereo"),
        ("Mono_Aux", "Aux", "Mono"),
        ("Master_St", "Master", "Stereo"),
        ("Master_M", "Master", "Mono"),
        ("VCA_All", "Vca", "None"),
        ("BF_Vocals", "BasicFolder", "None"),
        ("RF_Drums", "RoutingFolder", "Stereo"),
    ]
    assert sess.track_by_name("BF_Vocals").is_folder_open is False  # type: ignore[union-attr]
    assert sess.track_by_name("RF_Drums").is_folder_open is False  # type: ignore[union-attr]
    assert sess.track_by_name("Verb_Aux").is_muted is True  # type: ignore[union-attr]
    assert sess.track_by_name("Verb_Aux").track_height_name == "Jumbo"  # type: ignore[union-attr]
    assert sess.track_by_name("Master_M").is_inactive is True  # type: ignore[union-attr]
    assert len(sess.markers) == 2
    assert sess.markers[1].track_name == "Kick_Audio"

    # Fluent PtxPatchBuilder roundtrip on the written file
    _orig, bsum = (
        PtxPatchBuilder.from_file(out_path)
        .rename_tracks({"Verb_Aux": "Plate_Reverb"})
        .set_track_states({"Plate_Reverb": {"hidden": True, "mute": False}, "RF_Drums": {"folder_open": True}})
        .set_session_settings(bit_depth=32, audio_format="AIFF", is_interleaved=True)
        .delete_marker(1)
        .add_marker("End", 240000, number=9)
        .commit()
    )
    assert bsum["renamed_tracks"] == {"Verb_Aux": "Plate_Reverb"}
    sess2 = PtxSession.from_file(out_path)
    assert sess2.info.bit_depth == 32 and sess2.info.audio_format_name == "AIFF"
    assert sess2.track_by_name("Plate_Reverb").is_hidden is True  # type: ignore[union-attr]
    assert sess2.track_by_name("RF_Drums").is_folder_open is True  # type: ignore[union-attr]
    assert {m.number for m in sess2.markers} == {2, 9}

    # Validation guard on unsupported non-Audio track type
    bad_req = OfflinePtxWriteRequest(
        ptxt_path=tpl_path, output_ptx_path=tmp_path / "bad.ptx", session_dir=tmp_path,
        sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
        extra_tracks=(OfflineNonAudioTrackRequest(track_name="Bad", track_type="Video"),),
    )
    with pytest.raises(PtxValidationError, match="Unsupported non-Audio track_type"):
        write_offline_ptx_session(bad_req)
