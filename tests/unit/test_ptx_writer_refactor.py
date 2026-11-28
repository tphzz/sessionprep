"""Focused regression tests for the PTX writer/patcher 3-wave architectural refactoring."""

from __future__ import annotations

import struct

import pytest

from sessionpreplib.ptx import (
    AudioFileWriteSpec,
    OfflineAudioTrackRequest,
    OfflineClipSpec,
    OfflineMarkerRequest,
    OfflineNonAudioTrackRequest,
    OfflinePtxWriteRequest,
    PtxClipAttributesPatch,
    PtxCorruptFileError,
    PtxError,
    PtxGroupAttributes,
    PtxNotFoundError,
    PtxPatchBuilder,
    PtxPatchResult,
    PtxSession,
    PtxSessionSettings,
    PtxTrackStatePatch,
    PtxUnsupportedTemplateError,
    PtxValidationError,
    build_offline_ptx_bytes,
    closest_pt_color_index,
    decrypt_ptx,
    encrypt_ptx,
    generate_ptx_uuid,
    patch_add_group,
    patch_add_marker,
    patch_clip_attributes,
    patch_rename_clips,
    patch_rename_tracks,
    patch_session_settings,
    patch_track_states,
    patch_track_volumes,
    validate_index_0002,
)
from sessionpreplib.ptx._document import PtxBuffer, validate_decrypted_ptx_pair
from sessionpreplib.ptx.lowlevel import (
    build_instrument_mixer_strip_2621,
    build_midi_track_1057,
    build_zchunk,
    find_subchunks,
    find_zchunk_offset,
    iter_top_chunks,
    mutate_2624_mixer,
    parse_midi_strip_playlists,
    plan_existing_midi_survival,
    resolve_track_format_codes,
    serialize_lane_1052,
)
from sessionpreplib.ptx.writer_track_mutations import TrackSlotOp
from tests.unit.test_ptx_core import _build_synthetic_ptx
from tests.unit.test_ptx_extended_capabilities import _build_blank_modern_ptx
from tests.unit.test_ptx_ui_midi_surround_clips import _add_ui_chunks_to_ptx


def _strip_top_chunk(ptx_raw: bytes, drop_tag: int) -> bytes:
    """Remove top-level *drop_tag* and rebuild valid ``0x0002`` via ``shift_parsed_index_0002``."""
    from sessionpreplib.ptx.lowlevel import parse_index_0002, replace_index_0002_chunk, shift_parsed_index_0002

    dec, delta = decrypt_ptx(ptx_raw)
    k0 = ptx_raw[0x14] ^ 0x5A
    chunks = iter_top_chunks(dec, strict=True)
    target = next(c for c in chunks if c.tag == drop_tag)
    idx = parse_index_0002(dec)
    assert idx is not None
    kept = [c.raw for c in chunks if c.tag != drop_tag]
    buf = bytearray(dec[:0x1F] + b"".join(kept))
    shifted = shift_parsed_index_0002(idx, target.offset + len(target.raw), -len(target.raw))
    shifted.entries = [e for e in shifted.entries if not (e.path and e.path[0].tag == drop_tag)]
    replace_index_0002_chunk(buf, shifted)
    assert validate_index_0002(bytes(buf))
    return encrypt_ptx(bytes(buf), delta, k0=k0)


def test_bug1_patch_track_volumes_rejects_multipoint_automation() -> None:
    """Multi-point volume automation in 0x260A must be refused instead of corrupting delta_pos."""
    raw = _build_synthetic_ptx()
    dec, delta = decrypt_ptx(raw)
    buf = bytearray(dec)
    c2624 = PtxSession(dec, xor_delta=delta).chunk_map[0x2624]
    c2624_end = c2624.offset + len(c2624.raw)
    p261e = c2624.offset + 13
    p260d = find_zchunk_offset(buf, 0x260D, p261e + 9, c2624_end)
    p260a = find_zchunk_offset(buf, 0x260A, p260d + 9, c2624_end)
    assert p260d >= 0 and p260a >= 0
    old_sz = 7 + struct.unpack_from("<I", buf, p260a + 3)[0]
    mp_hdr = bytearray(struct.pack("<HHIHII", 0x4601, 1, 26, 0, 2, 2))
    struct.pack_into("<H", mp_hdr, 16, 1)  # extra_count = 1 -> 2 breakpoints
    mp_curve = bytes(mp_hdr) + struct.pack("<QhIh", 0, -30, 48000, -60) + b"\x00\x00"
    mp_260a = build_zchunk(0x260A, mp_curve, version=1)
    delta_len = len(mp_260a) - old_sz
    buf[p260a : p260a + old_sz] = mp_260a
    for anc in (p260d, p261e, c2624.offset):
        struct.pack_into("<I", buf, anc + 3, struct.unpack_from("<I", buf, anc + 3)[0] + delta_len)
    patched_raw = encrypt_ptx(bytes(buf), delta)
    with pytest.raises(PtxValidationError, match="multi-point automation"):
        patch_track_volumes(patched_raw, {"KIK": -3.0})


def test_bug2_instrument_2621_inactive_flag_offsets() -> None:
    """Setting Instrument 0x2621 inactive writes exact verified Pro Tools 2026.11.2 byte offsets."""
    raw = _build_blank_modern_ptx(with_tracks=False)
    req = OfflinePtxWriteRequest(
        sample_rate=48000,
        bit_depth=24,
        extra_tracks=(OfflineNonAudioTrackRequest("Synth", "Instrument", channels=2),),
    )
    ptx_bytes, _ = build_offline_ptx_bytes(raw, req)
    patched, summary = patch_track_states(ptx_bytes, {"Synth": PtxTrackStatePatch(is_inactive=True)})
    assert summary["Synth"]["inactive"] is True

    dec, delta = decrypt_ptx(patched)
    assert validate_index_0002(dec)
    sess = PtxSession(dec, xor_delta=delta)
    synth = sess.track_by_name("Synth")
    assert synth is not None and synth.is_inactive is True

    # Verify exact binary offsets inside the Instrument 0x2621 strip
    c2624 = sess.chunk_map[0x2624].raw
    inst_strips = find_subchunks(c2624, 0x2621)
    assert len(inst_strips) == 1
    blk = inst_strips[0]
    p4856 = find_zchunk_offset(blk, 0x4856, 9)
    p102d = find_zchunk_offset(blk, 0x102D, 9)
    after_102d = p102d + 7 + struct.unpack_from("<I", blk, p102d + 3)[0]
    p2010 = find_zchunk_offset(blk, 0x2010, 9)
    assert p4856 >= 31 and blk[p4856 - 31] == 1
    assert p102d >= 0 and blk[after_102d + 3] == 0
    assert p2010 >= 1 and blk[p2010 - 1] == 0
    for s260e in find_subchunks(blk, 0x260E):
        assert s260e[12] == 0

    # Toggling back to active restores all bytes cleanly
    restored, _ = patch_track_states(patched, {"Synth": PtxTrackStatePatch(is_inactive=False)})
    assert PtxSession.from_bytes(restored).track_by_name("Synth").is_inactive is False


def test_bug3_clip_attributes_unwrapped_v3_locks_and_disambiguation() -> None:
    """patch_clip_attributes handles unwrapped v3 0x1052 lanes, rejects invalid lock combos, and filters clips."""
    raw = _build_blank_modern_ptx(with_tracks=False)
    req = OfflinePtxWriteRequest(
        sample_rate=48000,
        bit_depth=24,
        audio_files=(
            AudioFileWriteSpec("TakeA.wav", 48000, 1, 24, 48000, b"\x01" * 8, b"\x02" * 32, 1),
            AudioFileWriteSpec("TakeB.wav", 48000, 1, 24, 48000, b"\x03" * 8, b"\x04" * 32, 2),
        ),
        tracks=(
            OfflineAudioTrackRequest(
                "Vox", 1,
                clip=OfflineClipSpec(audio_file_index=0, clip_base_name="TakeA", length_samples=48000),
            ),
        ),
    )
    ptx_bytes, _ = build_offline_ptx_bytes(raw, req)

    # Contradictory lock combination (edit-locked without time-locked) must be rejected
    with pytest.raises(PtxValidationError, match="is_edit_locked=True requires is_time_locked=True"):
        patch_clip_attributes(ptx_bytes, "Vox", 0, is_edit_locked=True, is_time_locked=False)

    # Unwrapped v3 0x1052 lane with two 0x104F clip placements must patch cleanly (not skip as fade)
    from sessionpreplib.ptx.lowlevel import apply_rebuilt_index_0002, parse_index_0002

    dec, delta = decrypt_ptx(ptx_bytes)
    sess = PtxSession(dec, xor_delta=delta)
    c1054 = sess.chunk_map[0x1054]
    p4828 = find_zchunk_offset(c1054.raw, 0x4828, 9)
    trailer = c1054.raw[p4828:] if p4828 >= 0 else b""
    unwrapped_lane = serialize_lane_1052(
        "Vox", [(0, 0, 0, 0xFFFE), (1, 0, 0, 0xFFFE)], wrap_1050=False, lane_version=3,
    )
    new_1054 = build_zchunk(0x1054, struct.pack("<I", 1) + unwrapped_lane + trailer, version=c1054.version)
    tpl_idx = parse_index_0002(dec)
    assert tpl_idx is not None
    buf = bytearray(dec)
    buf[c1054.offset : c1054.offset + len(c1054.raw)] = new_1054
    apply_rebuilt_index_0002(buf, tpl_idx, dec)
    unwrapped_ptx = encrypt_ptx(bytes(buf), delta)

    # Disambiguating by clip_index=0 patches only placement 0
    patched, summary = patch_clip_attributes(
        unwrapped_ptx, "Vox", 0,
        attrs=PtxClipAttributesPatch(is_muted=True, is_time_locked=True, is_edit_locked=True, clip_index=0),
    )
    assert summary["lanes_patched"] == 1
    from sessionpreplib.ptx.parser_timeline import parse_timeline_lanes_1054

    s_out = PtxSession.from_bytes(patched)
    pls = parse_timeline_lanes_1054(
        s_out.chunk_map[0x1054].raw, s_out.clips, s_out.fades, {}, s_out.info.sample_rate,
    )[0][1]
    assert len(pls) == 2
    assert pls[0].is_muted is True and pls[0].is_edit_locked is True
    assert pls[1].is_muted is False and pls[1].is_edit_locked is False


def test_bug4_missing_ui_chunks_raise_unsupported_template() -> None:
    """Requesting UI/transport settings on a session lacking those chunks fails loudly."""
    raw = _build_blank_modern_ptx(with_tracks=True)  # has no 0x2056/0x2024/0x201F/0x2025/0x2016
    with pytest.raises(PtxUnsupportedTemplateError, match="Template lacks writable UI state chunks"):
        patch_session_settings(raw, settings=PtxSessionSettings(edit_mode="Grid"))

    with_ui = _add_ui_chunks_to_ptx(raw, tick_selection=True)
    with pytest.raises(PtxValidationError, match="tick-based"):
        patch_session_settings(with_ui, settings=PtxSessionSettings(timeline_out_sample=96000))


def test_bug5_multi_playlist_midi_and_instrument_compaction() -> None:
    """MIDI/Instrument strips with pl_cnt >= 2 remap both active and alternate playlist indices."""
    def _make_inst_multi(name: str, active: int, alts: tuple[int, int]) -> bytes:
        raw_s = build_instrument_mixer_strip_2621(
            name, generate_ptx_uuid(), 2, None, track_ordinal_1based=1, id_base=10, midi_index=active,
        )
        p0 = parse_midi_strip_playlists(raw_s)
        assert p0 is not None
        off = p0.offset
        old_len = 6 + p0.stride * 1 + (3 if p0.stride == 2 else 1)
        new_mid = struct.pack("<IHHH", active, 2, alts[0], alts[1]) + raw_s[off + 6 + p0.stride : off + old_len]
        delta_len = len(new_mid) - old_len
        mbuf = bytearray(raw_s[:off] + new_mid + raw_s[off + old_len :])
        struct.pack_into("<I", mbuf, 3, struct.unpack_from("<I", mbuf, 3)[0] + delta_len)
        return bytes(mbuf)

    strip_del = _make_inst_multi("DropKeys", 0, (0, 1))
    strip_keep = _make_inst_multi("KeepKeys", 2, (2, 4))
    c2624 = build_zchunk(0x2624, struct.pack("<I", 2) + strip_del + strip_keep, version=1)
    entries_1057 = [build_midi_track_1057(f"M{i}", generate_ptx_uuid()) for i in range(6)]
    c1058 = build_zchunk(0x1058, struct.pack("<I", 6) + b"".join(entries_1057), version=1)

    plan = plan_existing_midi_survival(c1058, c2624, deleted_old_indices=[0])
    assert len(plan.entries) == 4
    assert plan.old_midi_to_new == {2: 0, 3: 1, 4: 2, 5: 3}

    layout = (TrackSlotOp(old_index=1),)
    mutated = mutate_2624_mixer(c2624, layout, expected_old_count=2, old_midi_to_new=plan.old_midi_to_new)
    new_strips = find_subchunks(mutated, 0x2621)
    assert len(new_strips) == 1
    parsed1 = parse_midi_strip_playlists(new_strips[0])
    assert parsed1 is not None and parsed1.active_index == 0 and parsed1.playlist_indices == (0, 2)


def test_bug6_patch_add_group_preserves_caller_attributes() -> None:
    """Explicit PtxGroupAttributes(follow_globals=False, include_rulers=True) is not overwritten by defaults."""
    raw = _build_synthetic_ptx()
    attrs = PtxGroupAttributes(follow_globals=False, include_rulers=True, main_volume=True, main_mute=True)
    patched, summary = patch_add_group(raw, "CustomAttrGrp", ["Kick", "Snare"], attributes=attrs)
    assert summary["action"] == "add_group"
    grp = PtxSession.from_bytes(patched).group_by_name("CustomAttrGrp")
    assert grp is not None
    assert grp.attributes.follow_globals is False
    assert grp.attributes.include_rulers is True
    assert grp.attributes.main_volume is True
    assert grp.attributes.main_mute is True


def test_bug7_virgin_marker_synthesis_and_warning_diff_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Virgin templates lacking 0x2030 synthesize it cleanly, and validation rejects new parser warnings."""
    raw = _strip_top_chunk(_build_blank_modern_ptx(with_tracks=False), 0x2030)
    assert 0x2030 not in PtxSession.from_bytes(raw).chunk_map

    # Direct patch_add_marker on virgin template lacking 0x2030
    patched, summary = patch_add_marker(raw, "Intro", 48000, number=1)
    assert summary["number"] == 1
    dec, _ = decrypt_ptx(patched)
    assert validate_index_0002(dec)
    sess = PtxSession(dec)
    assert len(sess.markers) == 1 and sess.markers[0].name == "Intro"

    # Offline builder on virgin template with markers
    out_bytes, res = build_offline_ptx_bytes(
        raw,
        OfflinePtxWriteRequest(
            sample_rate=48000, bit_depth=24,
            markers=(OfflineMarkerRequest("Verse", 96000, number=2),),
        ),
    )
    assert res.ptx_path is None
    assert [m.name for m in PtxSession.from_bytes(out_bytes).markers] == ["Verse"]

    # validate_decrypted_ptx_pair fails if new_sess emits a warning absent from orig_sess
    orig_dec, orig_delta = decrypt_ptx(raw)
    from sessionpreplib.ptx import _document as doc_mod

    real_cls = doc_mod.PtxSession
    calls = {"n": 0}

    class _WarnOnSecond(real_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            calls["n"] += 1
            if calls["n"] >= 2:
                self.warnings.append("Injected corrupt chunk warning")

    monkeypatch.setattr(doc_mod, "PtxSession", _WarnOnSecond)
    with pytest.raises(PtxCorruptFileError, match="Injected corrupt chunk warning"):
        validate_decrypted_ptx_pair(orig_dec, orig_delta, orig_dec, orig_delta)


def test_wave2_and_wave3_typed_apis_folder_nesting_and_guards() -> None:
    """Verify PtxBuffer, PtxPatchResult, parent_folder_name nesting, PtxNotFoundError, and guards."""
    raw = _build_blank_modern_ptx(with_tracks=False)

    # 1. channels vs track_format conflict raises PtxValidationError
    with pytest.raises(PtxValidationError, match="conflicts with track_format"):
        resolve_track_format_codes(2, "5.1")

    # 2. Newly created RoutingFolder + BasicFolder nested via parent_folder_name (single write call!)
    req = OfflinePtxWriteRequest(
        sample_rate=48000,
        bit_depth=24,
        session_settings=PtxSessionSettings(audio_format="WAV", is_interleaved=True),
        audio_files=(AudioFileWriteSpec("Kick.wav", 48000, 1, 24, 48000, b"\x11" * 8, b"\x22" * 32, 1),),
        extra_tracks=(
            OfflineNonAudioTrackRequest("Drums", "RoutingFolder", channels=2),
            OfflineNonAudioTrackRequest("Shells", "BasicFolder", parent_folder_name="Drums"),
        ),
        tracks=(
            OfflineAudioTrackRequest(
                "Kick", 1, parent_folder_name="Shells",
                clip=OfflineClipSpec(audio_file_index=0, clip_base_name="Kick", length_samples=48000),
            ),
        ),
    )
    ptx_bytes, write_res = build_offline_ptx_bytes(raw, req)
    assert set(write_res.created_tracks) == {"Drums", "Shells", "Kick"}
    sess = PtxSession.from_bytes(ptx_bytes)
    assert [t.name for t in sess.tracks] == ["Drums", "Shells", "Kick"]
    assert sess.track_by_name("Shells").parent_folder_name == "Drums"
    assert sess.track_by_name("Kick").parent_folder_name == "Shells"

    # 3. Folder cycle detection
    cycle_req = OfflinePtxWriteRequest(
        sample_rate=48000, bit_depth=24,
        extra_tracks=(
            OfflineNonAudioTrackRequest("F1", "BasicFolder", parent_folder_name="F2"),
            OfflineNonAudioTrackRequest("F2", "BasicFolder", parent_folder_name="F1"),
        ),
    )
    with pytest.raises(PtxValidationError, match="Placed|Cycle"):
        build_offline_ptx_bytes(raw, cycle_req)

    # 4. PtxPatchBuilder.build_result() single-pass PtxBuffer + typed patch objects + closest_pt_color_index
    col_idx = closest_pt_color_index("#FF0000")
    assert col_idx is not None and col_idx in range(25, 96)
    result = (
        PtxPatchBuilder.from_bytes(ptx_bytes)
        .set_track_volumes({"Kick": -4.5})
        .set_track_states({"Kick": PtxTrackStatePatch(is_muted=True, color_index=col_idx)})
        .add_marker("Hook", 24000, number=1)
        .build_result()
    )
    assert isinstance(result, PtxPatchResult)
    assert result.summary["volumes"]["Kick"]["target_db"] == pytest.approx(-4.5)
    assert result.summary["track_states"]["Kick"]["mute"] is True
    assert result.summary["markers"][0]["name"] == "Hook"
    assert PtxSession.from_bytes(result.patched_bytes).track_by_name("Kick").is_muted is True

    # 5. PtxNotFoundError inherits from (PtxError, PtxValidationError, KeyError)
    with pytest.raises(PtxNotFoundError) as exc_info:
        patch_track_states(ptx_bytes, {"NoSuchTrack": PtxTrackStatePatch(is_muted=True)})
    assert isinstance(exc_info.value, (PtxError, PtxValidationError, KeyError))

    # 6. patch_rename_clips rejects case-insensitive duplicate target keys
    with pytest.raises(PtxValidationError, match="Duplicate"):
        patch_rename_clips(ptx_bytes, {"Kick": "KickA", "kick": "KickB"})

    # 7. patch_rename_tracks uses active playlist index and updates cleanly
    renamed, rsum = patch_rename_tracks(ptx_bytes, {"Kick": "Kick_In"})
    assert rsum["Kick"] == "Kick_In"
    assert PtxSession.from_bytes(renamed).track_by_name("Kick_In") is not None

    # 8. PtxBuffer single-pass roundtrip matches decrypt/encrypt
    pbuf = PtxBuffer.from_encrypted(ptx_bytes)
    assert pbuf.to_encrypted() == ptx_bytes


def test_review_fixes_smarttool_template_path_midi_2620_and_ptx_error_normalization() -> None:
    """Verify SmartTool UI patching, template_path alias, 0x2620 pl_cnt>=2, virgin delete_marker, and PtxError."""
    from pathlib import Path
    from sessionpreplib.ptx import (
        patch_add_fade,
        patch_delete_marker,
        patch_slot_a_trim_gains,
    )
    from sessionpreplib.ptx.lowlevel import build_midi_mixer_strip_2620

    # 1. edit_tool="SmartTool" / "Smart" succeeds and contradictory smart_tool_enabled=False raises PtxValidationError
    raw_ui = _add_ui_chunks_to_ptx(_build_blank_modern_ptx(with_tracks=False))
    p_smart, s_smart = patch_session_settings(raw_ui, edit_tool="SmartTool")
    assert s_smart["edit_tool"] == "SmartTool"
    assert PtxSession.from_bytes(p_smart).info.smart_tool_enabled is True
    p_smart2, _ = patch_session_settings(raw_ui, edit_tool="Smart")
    assert PtxSession.from_bytes(p_smart2).info.smart_tool_enabled is True
    with pytest.raises(PtxValidationError, match="Contradictory"):
        patch_session_settings(raw_ui, edit_tool="SmartTool", smart_tool_enabled=False)

    # 2. OfflinePtxWriteRequest(template_path=...) works and conflicting ptxt_path/template_path raises
    r_alias = OfflinePtxWriteRequest(template_path=Path("foo.ptxt"), sample_rate=48000, bit_depth=24)
    assert r_alias.ptxt_path == Path("foo.ptxt") and r_alias.template_path == Path("foo.ptxt")
    with pytest.raises(PtxValidationError, match="Conflicting"):
        OfflinePtxWriteRequest(ptxt_path=Path("a.ptxt"), template_path=Path("b.ptxt"))

    # 3. OfflineClipSpec(is_edit_locked=True) succeeds cleanly with default is_time_locked=None
    raw_blank = _build_blank_modern_ptx(with_tracks=False)
    ptx_locked, _ = build_offline_ptx_bytes(
        raw_blank,
        OfflinePtxWriteRequest(
            sample_rate=48000, bit_depth=24,
            audio_files=(AudioFileWriteSpec("A.wav", 48000, 1, 24, 48000, b"\x01" * 8, b"\x02" * 32, 1),),
            tracks=(OfflineAudioTrackRequest("T1", 1, clip=OfflineClipSpec(0, "A", 48000, is_edit_locked=True)),),
        ),
    )
    assert PtxSession.from_bytes(ptx_locked).track_by_name("T1") is not None

    # 4. 0x2620 MIDI multi-playlist pl_cnt >= 2 compaction + trailing active index remapping
    def _make_midi_multi(name: str, active: int, alts: tuple[int, int]) -> bytes:
        raw_m = build_midi_mixer_strip_2620(name, generate_ptx_uuid(), track_ordinal_1based=1, midi_index=active)
        p0 = parse_midi_strip_playlists(raw_m)
        assert p0 is not None
        off = p0.offset
        old_len = 8 + 4 + 1
        new_mid = struct.pack("<IIIIB", active, 2, alts[0], alts[1], raw_m[off + old_len - 1])
        mbuf = bytearray(raw_m[:off] + new_mid + raw_m[off + old_len :])
        struct.pack_into("<I", mbuf, 3, struct.unpack_from("<I", mbuf, 3)[0] + len(new_mid) - old_len)
        return bytes(mbuf)

    m_del = _make_midi_multi("DropM", 0, (0, 1))
    m_keep = _make_midi_multi("KeepM", 2, (2, 4))
    c2624 = build_zchunk(0x2624, struct.pack("<I", 2) + m_del + m_keep, version=1)
    c1058 = build_zchunk(0x1058, struct.pack("<I", 6) + b"".join(build_midi_track_1057(f"M{i}", generate_ptx_uuid()) for i in range(6)), version=1)
    plan = plan_existing_midi_survival(c1058, c2624, deleted_old_indices=[0])
    assert plan.old_midi_to_new == {2: 0, 3: 1, 4: 2, 5: 3}
    mutated = mutate_2624_mixer(c2624, (TrackSlotOp(old_index=1),), expected_old_count=2, old_midi_to_new=plan.old_midi_to_new)
    new_m = find_subchunks(mutated, 0x2620)[0]
    p1 = parse_midi_strip_playlists(new_m)
    assert p1 is not None and p1.active_index == 0 and p1.playlist_indices == (0, 2)
    p206f = find_zchunk_offset(new_m, 0x206F, p1.offset + 9)
    t_off = p206f + 7 + struct.unpack_from("<I", new_m, p206f + 3)[0]
    assert struct.unpack_from("<I", new_m, t_off + 1)[0] == 0

    # 5. patch_delete_marker on virgin template lacking 0x2030 raises PtxNotFoundError
    virgin_no_2030 = _strip_top_chunk(raw_blank, 0x2030)
    with pytest.raises(PtxNotFoundError):
        patch_delete_marker(virgin_no_2030, 1)

    # 6. Public patchers consistently raise PtxError subclasses
    with pytest.raises(PtxError):
        patch_session_settings(raw_ui, edit_mode="NotAMode")
    with pytest.raises(PtxError):
        patch_add_marker(raw_blank, "Bad", 0, number=1000)
    with pytest.raises(PtxError):
        patch_add_fade(ptx_locked, "T1", "bogus", 0, 1000)
    with pytest.raises(PtxError):
        patch_slot_a_trim_gains(ptx_locked, {"T1": -3.0})

