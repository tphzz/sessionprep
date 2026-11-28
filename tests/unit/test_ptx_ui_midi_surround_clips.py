"""Unit tests for PTX UI state, marker recall flags, timebase, MIDI/Instrument, surround, sub-clips, locks, and fades."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from sessionpreplib.ptx import (
    ZERO_TICKS,
    AudioFileWriteSpec,
    OfflineAudioTrackRequest,
    OfflineFadeRequest,
    OfflineMarkerRequest,
    OfflineNonAudioTrackRequest,
    OfflinePtxWriteRequest,
    PtxPatchBuilder,
    PtxSession,
    PtxValidationError,
    build_instrument_mixer_strip_2621,
    build_midi_mixer_strip_2620,
    build_midi_track_1057,
    build_playlist_1014_v9,
    build_zchunk,
    decrypt_ptx,
    encrypt_ptx,
    find_subchunks,
    generate_ptx_uuid,
    iter_top_chunks,
    patch_add_fade,
    patch_add_marker,
    patch_clip_attributes,
    patch_rename_clips,
    patch_rename_tracks,
    patch_session_settings,
    patch_track_states,
    rebuild_index_0002,
    serialize_fade_262f,
    serialize_index_0002,
    serialize_marker_2077,
    serialize_placement_104f,
    serialize_playlist_1014,
    validate_index_0002,
    write_offline_ptx_session,
)
from sessionpreplib.ptx.codec import find_zchunk_offset
from sessionpreplib.ptx.writer_primitives import (
    PtxIndex0002,
    PtxIndexEntry,
    PtxIndexOccurrence,
    PtxIndexPathStep,
)
from tests.unit.test_ptx_extended_capabilities import _build_blank_modern_ptx


def _add_ui_chunks_to_ptx(ptx_raw: bytes, *, tick_selection: bool = False) -> bytes:
    """Inject modern Pro Tools UI chunks (including nested 0x2016->0x2056 and 0x2587->0x2066) with valid 0x0002."""
    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    top = iter_top_chunks(dec, strict=True)
    b2024 = bytearray(106)
    b2024[0], b2024[1], b2024[4] = 3, 0, 1
    b2056 = bytearray(41)
    b2056[0], b2056[13], b2056[17], b2056[21], b2056[35] = 0, 2, 3, 0, 1
    c2056 = build_zchunk(0x2056, bytes(b2056), version=1)
    b201f = bytearray(136)
    b201f[32] = 1
    b2025 = bytearray(35)
    b2016 = bytearray(1818)
    b2016[125] = 1
    if tick_selection:
        tick0 = (0x40 << 56) | ZERO_TICKS
        struct.pack_into("<QQQQQ", b2016, 80, tick0, tick0, tick0, tick0, tick0)
    b2016[1500 : 1500 + len(c2056)] = c2056
    b2066 = bytearray(300)
    b2530 = bytearray(300)
    nested_2066 = build_zchunk(0x2066, bytes(188), version=1)  # 197B nested 0x2066 (grid_rel at len-4)

    body_chunks: list[bytes] = []
    for c in top:
        if c.tag == 0x0002:
            continue
        if c.tag == 0x2587:
            body_chunks.append(build_zchunk(0x2587, c.body + c2056 + nested_2066, version=c.version))
        else:
            body_chunks.append(c.raw)
    extras = [
        build_zchunk(0x2024, bytes(b2024), version=1),
        c2056,
        build_zchunk(0x201F, bytes(b201f), version=1),
        build_zchunk(0x2025, bytes(b2025), version=1),
        build_zchunk(0x2016, bytes(b2016), version=1),
        build_zchunk(0x2066, bytes(b2066), version=1),
        build_zchunk(0x2530, bytes(b2530), version=1),
    ]
    pre_body = b"".join([*body_chunks, *extras])
    hdr = bytearray(dec[:0x1F])
    struct.pack_into("<I", hdr, 0x1B, 0x1F + len(pre_body))
    pre_idx = bytes(hdr) + pre_body
    new_top = iter_top_chunks(pre_idx, strict=True)
    seed = PtxIndex0002(
        version=1,
        entries=[
            PtxIndexEntry([PtxIndexPathStep(c.tag, -1, b"\x00" * 5)], [PtxIndexOccurrence(1, (c.offset,), b"\x00" * 6)])
            for c in new_top
        ],
    )
    rebuilt = rebuild_index_0002(seed, pre_idx, pre_idx)
    out = pre_idx + serialize_index_0002(rebuilt)
    assert validate_index_0002(out)
    return encrypt_ptx(out, delta, k0=k0)


def _dummy_audio_file(name: str, channels: int = 2, length: int = 192000) -> AudioFileWriteSpec:
    return AudioFileWriteSpec(
        filename=name,
        sample_rate=48000,
        channels=channels,
        bit_depth=24,
        length_samples=length,
        file_id8=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        smpte_umid32=b"\x11" * 32,
        filetime_u64=133_500_000_000_000_000,
    )


def test_session_length_counters_edit_mode_tool_roll_and_timeline_roundtrip() -> None:
    raw = _add_ui_chunks_to_ptx(_build_blank_modern_ptx(), tick_selection=True)
    patched, summary = patch_session_settings(
        raw,
        fps_code=5,
        session_length="12:00:00:00",
        main_counter_format="Samples",
        sub_counter_format="TimeCode",
        edit_mode="GridRelative",
        edit_tool="Pencil",
        pre_roll_enabled=True,
        pre_roll_samples=24000,
        post_roll_enabled=True,
        post_roll_samples=48000,
        timeline_in_sample=48000,
        timeline_out_sample=144000,
    )
    assert summary["session_length"] == "12:00:00:00"
    dec, delta = decrypt_ptx(patched)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    assert sess.info.session_length_timecode == "12:00:00;00"
    assert sess.info.main_counter_format == "Samples"
    assert sess.info.sub_counter_format == "TimeCode"
    assert sess.info.edit_mode == "Grid"
    assert sess.info.grid_relative is True
    assert sess.info.edit_tool == "Pencil"
    assert sess.info.smart_tool_enabled is False
    assert (sess.info.pre_roll_enabled, sess.info.pre_roll_samples) == (True, 24000)
    assert (sess.info.post_roll_enabled, sess.info.post_roll_samples) == (True, 48000)
    assert (sess.info.timeline_in_sample, sess.info.timeline_out_sample) == (48000, 144000)

    # Verify nested 0x2016->0x2056 and 0x2587->0x2066 (197B len-4) were also updated!
    c2016 = sess.chunk_map[0x2016].raw
    p56_in_16 = find_zchunk_offset(c2016, 0x2056, 9)
    assert p56_in_16 >= 0 and c2016[p56_in_16 + 22] == 8 and c2016[p56_in_16 + 30] == 4
    c2587 = sess.chunk_map[0x2587].raw
    p66_in_87 = find_zchunk_offset(c2587, 0x2066, 9)
    l66 = 7 + struct.unpack_from("<I", c2587, p66_in_87 + 3)[0]
    assert p66_in_87 >= 0 and c2587[p66_in_87 + l66 - 4] == 1

    # Partial timeline_in_sample on a tick-flagged session never mixes ticks & samples
    part_tick, _ = patch_session_settings(raw, timeline_in_sample=96000)
    s_pt = PtxSession.from_bytes(part_tick)
    assert (s_pt.info.timeline_in_sample, s_pt.info.timeline_out_sample) == (96000, 96000)

    # Partial timeline_out_sample < existing cur_in is rejected
    with pytest.raises(ValueError, match="timeline_out_sample"):
        patch_session_settings(patched, timeline_out_sample=1000)

    # Updating session_length without fps_code preserves existing drop-frame ';' separator
    df_only, _ = patch_session_settings(patched, session_length="24:00:00:00")
    assert PtxSession.from_bytes(df_only).info.session_length_timecode == "24:00:00;00"

    # Guards: > 24:00:00:00, < 06:00:00:00, frame >= fps
    with pytest.raises(ValueError, match="session_length"):
        patch_session_settings(raw, session_length="02:00:00:00")
    with pytest.raises(ValueError, match="session_length"):
        patch_session_settings(raw, session_length="24:00:00:01")
    with pytest.raises(ValueError, match="session_length"):
        patch_session_settings(raw, fps_code=1, session_length="12:00:00:25")
    with pytest.raises(ValueError, match="main_counter_format"):
        patch_session_settings(raw, main_counter_format="InvalidCounter")
    with pytest.raises(ValueError, match="edit_mode"):
        patch_session_settings(raw, edit_mode="Warp")
    with pytest.raises(ValueError, match="edit_tool"):
        patch_session_settings(raw, edit_tool="Hammer")


def test_marker_general_properties_roundtrip_and_reference_masking() -> None:
    raw = _build_blank_modern_ptx(with_tracks=True)
    p1, _ = patch_add_marker(
        raw,
        "RecallAll",
        96000,
        reference="BarBeat",
        zoom_settings=True,
        pre_post_roll_times=True,
        track_visibility=True,
        track_heights=True,
        group_enables=True,
        window_configuration=True,
    )
    p2, _ = patch_add_marker(p1, "ZoomOnly", 192000, reference="Absolute", zoom_settings=True)
    sess = PtxSession.from_bytes(p2)
    m_all, m_zoom = sess.markers
    assert m_all.reference == "BarBeat"
    assert (
        m_all.zoom_settings,
        m_all.pre_post_roll_times,
        m_all.track_visibility,
        m_all.track_heights,
        m_all.group_enables,
        m_all.window_configuration,
    ) == (True, True, True, True, True, True)
    assert m_zoom.reference == "Absolute" and m_zoom.zoom_settings is True and not m_zoom.window_configuration
    assert serialize_marker_2077(10, "FlagBit", 0, reference="Absolute", window_configuration=True)[11] == (3 | 0x80)


def test_track_timebase_toggle_2015_and_multipoint_guard(tmp_path: Path) -> None:
    tpl = tmp_path / "tpl.ptxt"
    tpl.write_bytes(_build_blank_modern_ptx(with_tracks=False))
    out_ptx = tmp_path / "out.ptx"
    write_offline_ptx_session(
        OfflinePtxWriteRequest(
            ptxt_path=tpl,
            output_ptx_path=out_ptx,
            session_dir=tmp_path,
            sample_rate=48000,
            bit_depth=24,
            audio_files=(_dummy_audio_file("Kick.wav", channels=1),),
            tracks=(OfflineAudioTrackRequest("Kick", channels=1, audio_file_index=0, length_samples=48000, clip_base_name="Kick"),),
            extra_tracks=(
                OfflineNonAudioTrackRequest("AuxBus", track_type="Aux", channels=2),
                OfflineNonAudioTrackRequest("FolderBasic", track_type="BasicFolder"),
            ),
            track_states={"Kick": {"timebase": "Ticks"}, "AuxBus": {"timebase": "Ticks"}},
        )
    )
    sess = PtxSession.from_file(out_ptx)
    assert sess.track_by_name("Kick").timebase == "Ticks"
    assert sess.track_by_name("AuxBus").timebase == "Ticks"
    assert find_subchunks(sess.chunk_map[0x1054].raw, 0x1052)[0][-2:] == b"\x02\x00"
    # Verify 0x2015[q2] was patched to 0x02 (Ticks)
    s15 = find_subchunks(sess.chunk_map[0x2624].raw, 0x2015)[0]
    assert b"\x02\xfe\xff" in s15

    # BasicFolder rejects timebase toggle
    with pytest.raises(ValueError, match="BasicFolder"):
        patch_track_states(out_ptx.read_bytes(), {"FolderBasic": {"timebase": "Ticks"}})

    # Non-origin static curve must be rejected instead of silently corrupted
    dec, delta = decrypt_ptx(out_ptx.read_bytes())
    k0 = out_ptx.read_bytes()[0x14] ^ 0x5A
    mod = bytearray(dec)
    c2624_off = sess.chunk_map[0x2624].offset
    p_curve = find_zchunk_offset(mod, 0x260A, c2624_off + 9)
    struct.pack_into("<Q", mod, p_curve + 27, ZERO_TICKS + 480000)
    with pytest.raises(ValueError, match="non-origin or multi-point"):
        patch_track_states(encrypt_ptx(bytes(mod), delta, k0=k0), {"Kick": {"timebase": "Samples"}})


def test_midi_instrument_multi_index_delete_rename_and_surround_survival(tmp_path: Path) -> None:
    tpl = tmp_path / "tpl.ptxt"
    tpl.write_bytes(_build_blank_modern_ptx(with_tracks=False))
    step1 = tmp_path / "step1.ptx"
    write_offline_ptx_session(
        OfflinePtxWriteRequest(
            ptxt_path=tpl,
            output_ptx_path=step1,
            session_dir=tmp_path,
            sample_rate=48000,
            bit_depth=24,
            audio_files=(),
            tracks=(
                OfflineAudioTrackRequest("Aud_DeleteMe", channels=2),
                OfflineAudioTrackRequest("Aud_7_1_2", track_format="7.1.2"),
                OfflineAudioTrackRequest("Aud_7_1_4", track_format="7.1.4"),
            ),
            extra_tracks=(
                OfflineNonAudioTrackRequest("Midi_DeleteMe", track_type="Midi"),
                OfflineNonAudioTrackRequest("Midi_Keep", track_type="Midi"),
                OfflineNonAudioTrackRequest("Inst_Keep", track_type="Instrument", channels=2),
                OfflineNonAudioTrackRequest("Aux_5_1", track_type="Aux", track_format="5.1"),
                OfflineNonAudioTrackRequest("Mst_5_1", track_type="Master", track_format="5.1"),
                OfflineNonAudioTrackRequest("RF_5_1", track_type="RoutingFolder", track_format="5.1"),
            ),
        )
    )
    s1 = PtxSession.from_file(step1)
    # Reader decodes full 10 & 12 lanes for 7.1.2 / 7.1.4 AND Ticks timebase for Midi!
    pl_712 = next(p for p in s1.playlists if p.name == "Aud_7_1_2")
    pl_714 = next(p for p in s1.playlists if p.name == "Aud_7_1_4")
    assert (pl_712.channel_count, len(pl_712.lane_indices)) == (10, 10)
    assert (pl_714.channel_count, len(pl_714.lane_indices)) == (12, 12)
    assert s1.track_by_name("Midi_Keep").timebase == "Ticks"

    # Delete Aud_DeleteMe (shifts 7.1.2 & 7.1.4 lanes down by 2!) AND delete Midi_DeleteMe (shifts Midi_Keep & Inst_Keep 0x1058 indices down by 1!)
    step2 = tmp_path / "step2.ptx"
    write_offline_ptx_session(
        OfflinePtxWriteRequest(
            ptxt_path=step1,
            output_ptx_path=step2,
            session_dir=tmp_path,
            sample_rate=48000,
            bit_depth=24,
            audio_files=(),
            tracks=(),
            extra_tracks=(OfflineNonAudioTrackRequest("Midi_Added", track_type="Midi"),),
            delete_track_names=("Aud_DeleteMe", "Midi_DeleteMe"),
        )
    )
    dec2, delta2 = decrypt_ptx(step2.read_bytes())
    assert validate_index_0002(dec2)
    s2 = PtxSession(dec2, xor_delta=delta2)
    # 7.1.2 lanes remapped to 0..9, 7.1.4 lanes remapped to 10..21!
    p2_712 = next(p for p in s2.playlists if p.name == "Aud_7_1_2")
    p2_714 = next(p for p in s2.playlists if p.name == "Aud_7_1_4")
    assert p2_712.lane_indices == list(range(0, 10))
    assert p2_714.lane_indices == list(range(10, 22))
    # 0x1058 compacted to 3 entries: Midi_Keep (0), Inst_Keep (1), Midi_Added (2)
    c1058 = s2.chunk_map[0x1058].raw
    assert struct.unpack_from("<I", c1058, 9)[0] == 3
    # Rename Midi_Keep -> Midi_Renamed and assert 0x1058 -> 0x1057 renamed too!
    renamed_bytes, _ = patch_rename_tracks(step2.read_bytes(), {"Midi_Keep": "Midi_Renamed", "Inst_Keep": "Inst_Renamed"})
    s3 = PtxSession.from_bytes(renamed_bytes)
    names_1057 = [
        it[13 : 13 + struct.unpack_from("<I", it, 9)[0]].decode("utf-8")
        for it in find_subchunks(s3.chunk_map[0x1058].raw, 0x1057)
    ]
    assert names_1057 == ["Midi_Renamed", "Inst_Renamed", "Midi_Added"]

    # Direct blueprint & validation guards (including 0x2620 <IIIB gap13 + tail23 midi_index layout)
    assert len(build_midi_track_1057("Midi_1", generate_ptx_uuid())) == 205
    m2_strip = build_midi_mixer_strip_2620("Midi_2", generate_ptx_uuid(), track_ordinal_1based=2, midi_index=1)
    assert len(m2_strip) == 723
    c102d_end = 9 + 7 + struct.unpack_from("<I", m2_strip, 12)[0]
    assert struct.unpack_from("<IIIB", m2_strip, c102d_end) == (1, 1, 1, 1)
    assert struct.unpack_from("<BI", m2_strip, len(m2_strip) - 23) == (1, 1)
    # Also verify Midi_Keep's remapped 0x2620 strip in step2 updated both gap13 (<III) and tail23 (<I) from 1 -> 0
    s2_midi_strips = find_subchunks(s2.chunk_map[0x2624].raw, 0x2620)
    assert len(s2_midi_strips) == 2
    for expected_idx, raw_m in enumerate((s2_midi_strips[0], s2_midi_strips[1])):
        exp_mid = 0 if expected_idx == 0 else 2
        g_off = 9 + 7 + struct.unpack_from("<I", raw_m, 12)[0]
        assert struct.unpack_from("<IIIB", raw_m, g_off) == (exp_mid, 1, exp_mid, 1)
        assert struct.unpack_from("<BI", raw_m, len(raw_m) - 23) == (1, exp_mid)
    assert len(build_instrument_mixer_strip_2621("Inst_M", generate_ptx_uuid(), 1, None, track_ordinal_1based=2, id_base=10000, midi_index=1)) > 6000
    assert build_playlist_1014_v9("S714", list(range(12)), generate_ptx_uuid(), track_format="7.1.4")[0] == 0x5A
    with pytest.raises(ValueError, match="requires 10 lane indices"):
        serialize_playlist_1014("Bad712", [0, 1], generate_ptx_uuid(), format_code=17)
    with pytest.raises(PtxValidationError, match="only supports Mono or Stereo"):
        write_offline_ptx_session(
            OfflinePtxWriteRequest(
                ptxt_path=tpl, output_ptx_path=tmp_path / "bad_inst.ptx", session_dir=tmp_path,
                sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
                extra_tracks=(OfflineNonAudioTrackRequest("Inst51", track_type="Instrument", track_format="5.1"),),
            )
        )


def test_subclips_clip_attributes_rename_and_fades_roundtrip(tmp_path: Path) -> None:
    tpl = tmp_path / "tpl.ptxt"
    tpl.write_bytes(_add_ui_chunks_to_ptx(_build_blank_modern_ptx(with_tracks=False)))
    out_ptx = tmp_path / "clips_and_fades.ptx"
    write_offline_ptx_session(
        OfflinePtxWriteRequest(
            ptxt_path=tpl,
            output_ptx_path=out_ptx,
            session_dir=tmp_path,
            sample_rate=48000,
            bit_depth=24,
            audio_files=(
                _dummy_audio_file("MonoTone.wav", channels=1, length=192000),
                _dummy_audio_file("StereoTone.wav", channels=2, length=192000),
            ),
            tracks=(
                OfflineAudioTrackRequest(
                    "Aud_Sub",
                    channels=1,
                    audio_file_index=0,
                    length_samples=96000,
                    clip_base_name="SubMono",
                    start_sample=48000,
                    clip_start_sample=48000,
                    clip_sync_point_samples=24000,
                    clip_orig_timestamp=10000,
                    clip_user_timestamp=20000,
                    clip_muted=True,
                    clip_time_locked=True,
                    clip_color_index=29,
                ),
                OfflineAudioTrackRequest(
                    "Aud_St",
                    channels=2,
                    audio_file_index=1,
                    length_samples=192000,
                    clip_base_name="FullSt",
                    start_sample=0,
                    clip_edit_locked=True,
                ),
            ),
            fades=(
                OfflineFadeRequest("Aud_Sub", "in", start_sample=48000, total_samples=12000),
                OfflineFadeRequest("Aud_Sub", "out", start_sample=144000, total_samples=12000),
                OfflineFadeRequest("Aud_St", "crossfade", start_sample=96000, total_samples=12000, pre_samples=6000),
            ),
            markers=(OfflineMarkerRequest("CueA", 48000, zoom_settings=True, track_visibility=True),),
            session_length="12:00:00:00",
            main_counter_format="Samples",
            edit_mode="Slip",
        )
    )
    raw_written = out_ptx.read_bytes()
    dec, delta = decrypt_ptx(raw_written)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    c_sub = sess.clips[0]
    assert (c_sub.name, c_sub.start_sample, c_sub.length_samples, c_sub.sync_point_samples) == ("SubMono", 48000, 96000, 24000)
    assert (c_sub.original_timestamp_samples, c_sub.user_timestamp_samples) == (10000, 20000)
    p_sub = sess.track_by_name("Aud_Sub").clips[0]
    assert (p_sub.is_muted, p_sub.is_time_locked, p_sub.is_edit_locked, p_sub.color_index) == (True, True, False, 29)
    assert [(f.fade_type, f.pre_samples, f.total_samples) for f in sess.fades] == [
        ("in", 0, 12000), ("out", 12000, 12000), ("crossfade", 6000, 12000),
    ]

    # patch_add_fade on a virgin template WITHOUT 0x262A / 0x2504 / 0x2630 keeps 0x0002 trailing!
    empty_aud_ptx = tmp_path / "empty_aud.ptx"
    write_offline_ptx_session(
        OfflinePtxWriteRequest(
            ptxt_path=tpl, output_ptx_path=empty_aud_ptx, session_dir=tmp_path,
            sample_rate=48000, bit_depth=24, audio_files=(),
            tracks=(OfflineAudioTrackRequest("EmptyAud", channels=1),),
        )
    )
    virgin_faded, _ = patch_add_fade(empty_aud_ptx.read_bytes(), "EmptyAud", "in", 0, 4800)
    dec_vf, _ = decrypt_ptx(virgin_faded)
    assert validate_index_0002(dec_vf)
    assert len(PtxSession.from_bytes(virgin_faded).fades) == 1

    # PtxPatchBuilder rename_clips, set_clip_attributes, add_fade
    patched, summary = (
        PtxPatchBuilder.from_bytes(raw_written)
        .rename_clips({"SubMono": "SubMono_Renamed_Longer"})
        .set_clip_attributes("Aud_Sub", 48000, is_muted=False, is_edit_locked=True, color_index=40)
        .add_fade("Aud_St", "in", 0, 6000)
        .build()
    )
    assert summary["renamed_clips"] == {"SubMono": "SubMono_Renamed_Longer"}
    dec2, d2 = decrypt_ptx(patched)
    assert validate_index_0002(dec2)
    sess2 = PtxSession(dec2, xor_delta=d2)
    assert sess2.clips[0].name == "SubMono_Renamed_Longer"
    p_sub2 = sess2.track_by_name("Aud_Sub").clips[0]
    assert (p_sub2.is_muted, p_sub2.is_time_locked, p_sub2.is_edit_locked, p_sub2.color_index) == (False, True, True, 40)

    # Validation & bool guards
    with pytest.raises(ValueError, match="clip_index"):
        serialize_placement_104f(True, 0)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="not found"):
        patch_rename_clips(raw_written, {"NoSuchClip": "X"})
    with pytest.raises(KeyError, match="No clip placement"):
        patch_clip_attributes(raw_written, "Aud_Sub", 999999, is_muted=True)
    with pytest.raises(ValueError, match="Unsupported fade_type"):
        serialize_fade_262f("weird", 1000)
    with pytest.raises(PtxValidationError, match="invalid for audio file length"):
        write_offline_ptx_session(
            OfflinePtxWriteRequest(
                ptxt_path=tpl, output_ptx_path=tmp_path / "bad.ptx", session_dir=tmp_path,
                sample_rate=48000, bit_depth=24,
                audio_files=(_dummy_audio_file("Small.wav", channels=1, length=1000),),
                tracks=(OfflineAudioTrackRequest("T", channels=1, audio_file_index=0, clip_start_sample=0, length_samples=0, clip_base_name="C"),),
            )
        )
