"""Unit tests for the pure offline Pro Tools .ptx session writer, blueprints, and 0x0002 rebuilder."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.daw_processors.protools_ptx_media import (
    TargetAudioSpec,
    _embed_protools_wav_identity,
    stage_session_audio_files,
)
from sessionpreplib.daw_processors.protools_ptx_offline import execute_offline_ptx_creation
from sessionpreplib.daw_processors.protools_ptx_plan import build_protools_creation_plan
from sessionpreplib.models import SessionContext, TrackContext, TransferEntry
from sessionpreplib.ptx import (
    AudioFileWriteSpec,
    OfflineAudioTrackRequest,
    OfflineGroupRequest,
    OfflinePtxWriteRequest,
    PtxIndex0002,
    PtxIndexEntry,
    PtxIndexOccurrence,
    PtxIndexPathStep,
    PtxSession,
    apply_rebuilt_index_0002,
    build_zchunk,
    decrypt_ptx,
    encode_lp_str,
    encrypt_ptx,
    generate_ptx_uuid,
    iter_top_chunks,
    parse_index_0002,
    rebuild_index_0002,
    serialize_folder_tree_210c,
    serialize_index_0002,
    validate_index_0002,
    write_offline_ptx_session,
)
from sessionpreplib.ptx.models import PTX_MAGIC
from sessionpreplib.ptx.writer_blueprints import (
    _MONO_261C_RAW,
    _STEREO_261C_RAW,
    AUDIO_UI_2589_RAW,
    build_audio_file_entry_1003,
    build_audio_mixer_strip_261c,
    build_output_bus_260e,
    build_playlist_1014_v9,
    build_track_entry_251a,
    build_track_summary_2519,
    build_track_uuid_entry_210b,
    build_volume_table_0f3d,
    build_whole_file_clip_2629,
)
from sessionpreplib.ptx.writer_chunks import serialize_directory_tree_103a
from sessionpreplib.ptx.writer_mutations import (
    NewAudioTrackSpec,
    TrackSlotOp,
    mutate_202b_groups_for_track_map,
    mutate_210c_folders,
    mutate_2519_tracks,
    mutate_2624_mixer,
)
from tests.unit.test_ptx_core import _mixer_track_261e, _mixer_vca_261f, _sp_entry, _uuid, _zchunk


def _write_wav(path: Path, sr: int, channels: int, frames: int = 4800) -> Path:
    t = np.arange(frames, dtype=np.float64) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    data = sig if channels == 1 else np.column_stack([sig, 0.5 * sig])
    sf.write(path, data, sr, format="WAV", subtype="PCM_24")
    return path


def _build_251a_template_track(
    name: str, type_code: int, uid: bytes, visible_ord: int, *, is_stereo: bool = True, is_folder: bool = False,
) -> bytes:
    st = 1 if is_stereo else 0
    c4420 = build_zchunk(0x4420, b"\x00\x00", version=1)
    body = (
        struct.pack("<H", type_code)
        + encode_lp_str(name)
        + bytes([st, 0])
        + uid
        + struct.pack("<H", visible_ord)
        + (b"\x00" * 4)
        + uid
        + bytes([st, 0])
        + c4420
        + bytes([1 if is_folder else 0, 0, st, 0, 1, 1, 0, 0, 0])
        + struct.pack("<H", 0xFFFE)
    )
    return build_zchunk(0x251A, body, version=10)


def _build_group_202a(name: str, members: list[int], id_num: int, *, vca_index: int = 0xFFFF) -> bytes:
    m_bytes = struct.pack(f"<{len(members)}H", *members) if members else b""
    tail = bytes.fromhex("feff1080010000000000") + struct.pack("<H", vca_index)
    body = (
        encode_lp_str(name)
        + struct.pack("<BiH", 2, id_num, 0)
        + struct.pack("<I", len(members))
        + m_bytes
        + tail
        + build_zchunk(0x258E, b"\x01\x01" + b"\x00" * 60, version=5)
    )
    return build_zchunk(0x202A, body, version=5)


def _build_clean_template_ptxt(
    *,
    include_0f3d: bool = True,
    include_2587: bool = True,
    include_empty_basic_folder: bool = False,
    cnt_a_values: tuple[int, ...] = (),
    with_preexisting_vca_group: bool = False,
) -> bytes:
    """Build a realistic self-contained clean .ptxt template with full 0x0002 entries."""
    bus_out, bus_kik, bus_vox = _uuid(1), _uuid(2), _uuid(3)
    kik_uid, kik_vca_uid, vox_uid, vox_vca_uid, empty_bf_uid = _uuid(10), _uuid(11), _uuid(12), _uuid(13), _uuid(14)
    tracks = [
        ("KIK", 2, kik_uid, True, True),
        ("KIK-VCA", 9, kik_vca_uid, False, False),
        ("VOX", 2, vox_uid, True, True),
        ("VOX-VCA", 9, vox_vca_uid, False, False),
    ]
    if include_empty_basic_folder:
        tracks.append(("EmptyBasic", 11, empty_bf_uid, False, True))

    c0003 = _zchunk(0x0003, b"Pro Tools Studio\x00\x002026.11.2\x00\x00Release\x00\x00Session\x00\x00Windows\x00")
    c1028 = _zchunk(0x1028, bytes([3, 24]) + struct.pack("<I", 48000) + b"\x00" * 16, version=10)
    c0f3d = _zchunk(0x0F3D, struct.pack("<I", 0), version=1)
    c1004 = _zchunk(0x1004, struct.pack("<I", 0) + serialize_directory_tree_103a([]) + b"\x00\x00\x00\x00" + serialize_directory_tree_103a([]), version=3)
    c2603 = _zchunk(
        0x2603,
        struct.pack("<I", 3) + _sp_entry("Main Out", 1, True, bus_out) + _sp_entry("KIK", 2, True, bus_kik) + _sp_entry("VOX", 3, True, bus_vox),
        version=2,
    )
    rows = [
        struct.pack("<H", tc) + encode_lp_str(nm) + bytes([1 if st else 0, 0]) + uid + struct.pack("<HB", idx, 0)
        for idx, (nm, tc, uid, st, _fl) in enumerate(tracks, start=1)
    ]
    p251a = [
        _build_251a_template_track(nm, tc, uid, idx, is_stereo=st, is_folder=fl)
        for idx, (nm, tc, uid, st, fl) in enumerate(tracks, start=1)
    ]
    cnt_le = struct.pack("<I", len(tracks))
    c251c = build_zchunk(0x251C, b"\x00" * 11, version=2)
    extra_a = struct.pack(f"<{len(cnt_a_values)}H", *cnt_a_values) if cnt_a_values else b""
    c2519_body = (
        struct.pack("<HIII", 1, 0, 0, len(cnt_a_values)) + extra_a + cnt_le + b"".join(rows)
        + b"".join(p251a) + cnt_le + b"".join(p251a)
        + b"\x00\x01\x18\x00" + b"\x00" * 8
        + build_zchunk(0x2716, b"\x00\x00\x00\x00", version=1)
        + build_zchunk(0x251B, struct.pack("<I", 1) + c251c, version=1)
    )
    c2519 = build_zchunk(0x2519, c2519_body, version=9)
    c2107 = build_zchunk(
        0x2107,
        b"\x00" * 8 + b"\x01" + cnt_le + b"".join(
            build_track_uuid_entry_210b(nm, uid, type_code=tc) for nm, tc, uid, _, _ in tracks
        ),
        version=5,
    )
    # Note: EmptyBasic is intentionally omitted from 0x210C (as Pro Tools does for empty folders).
    c210c = serialize_folder_tree_210c([(kik_uid, [kik_vca_uid]), (vox_uid, [vox_vca_uid])])
    c258a = build_zchunk(0x258A, build_zchunk(0x2581, b"\x00" * 8, version=3) + cnt_le + AUDIO_UI_2589_RAW * len(tracks), version=2)
    c2587 = build_zchunk(0x2587, build_zchunk(0x2551, build_zchunk(0x258B, c258a, version=1), version=3), version=1)
    strips = [
        _mixer_track_261e(kik_uid, bus_kik, bus_out, 0.0),
        _mixer_vca_261f(kik_vca_uid, 0.0),
        _mixer_track_261e(vox_uid, bus_vox, bus_out, 0.0),
        _mixer_vca_261f(vox_vca_uid, 0.0),
    ]
    if include_empty_basic_folder:
        strips.append(_mixer_track_261e(empty_bf_uid, None, bus_out, 0.0))
    c2624 = build_zchunk(0x2624, cnt_le + b"".join(strips), version=1)

    edit_groups = [
        _build_group_202a("<ALL>", list(range(len(tracks))), -1),
        _build_group_202a("ALL-VCA", [1, 3], 1, vca_index=(3 if with_preexisting_vca_group else 0xFFFF)),
    ]
    if with_preexisting_vca_group:
        edit_groups.append(_build_group_202a("KIK-PRE", [0], 2, vca_index=1))
    c202b = build_zchunk(0x202B, struct.pack("<I", len(edit_groups)) + b"".join(edit_groups) + struct.pack("<I", 0) + b"\x00\x00", version=4)
    c1054 = build_zchunk(0x1054, struct.pack("<I", 0) + build_zchunk(0x4828, b"\x00\x00\x00\x00", version=2), version=2)
    c1015 = build_zchunk(0x1015, struct.pack("<I", 0) + build_zchunk(0x4829, b"\x00\x00\x00\x00", version=2), version=2)
    c262a = build_zchunk(0x262A, struct.pack("<I", 0), version=1)
    c2637 = build_zchunk(0x2637, struct.pack("<I", 0), version=1)
    c7ffe_a = build_zchunk(0x7FFE, b"\x01\x02\x03\x04", version=1)
    c7ffe_b = build_zchunk(0x7FFE, b"\x05\x06\x07\x08", version=1)

    out = bytearray(PTX_MAGIC + b"\x00\x05\x40")
    out.extend(struct.pack("<BHII", 0x5A, 1, 4, 0))
    chunks = [c1028, c7ffe_a, c2519]
    if include_0f3d:
        chunks.append(c0f3d)
    chunks.extend([c1004, c1054, c1015, c2624, c262a, c2637, c2603, c202b, c2107, c210c])
    if include_2587:
        chunks.append(c2587)
    chunks.extend([c7ffe_b, c0003])

    offs: dict[int, list[int]] = {}
    for ch in chunks:
        tag = struct.unpack_from("<H", ch, 7)[0]
        offs.setdefault(tag, []).append(len(out))
        out.extend(ch)

    o2519 = offs[0x2519][0]
    m_n = b"\x01" + struct.pack("<I", len(tracks))
    idx_entries = [
        PtxIndexEntry([PtxIndexPathStep(0x0003, -1)], [PtxIndexOccurrence(1, (offs[0x0003][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x7FFE, -1)], [PtxIndexOccurrence(1, (offs[0x7FFE][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x7FFE, -1)], [PtxIndexOccurrence(1, (offs[0x7FFE][1],))]),
        PtxIndexEntry([PtxIndexPathStep(0x1004, -1)], [PtxIndexOccurrence(1, (offs[0x1004][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x1015, -1, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x1015][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x1015, -1, b"\x01\x00\x00\x00\x00"), PtxIndexPathStep(0x4829, offs[0x1015][0], b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x1015][0] + 13,))]),
        PtxIndexEntry([PtxIndexPathStep(0x1054, -1, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x1054][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x1054, -1, b"\x01\x00\x00\x00\x00"), PtxIndexPathStep(0x4828, offs[0x1054][0], b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x1054][0] + 13,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2519, -1, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (o2519,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2519, -1), PtxIndexPathStep(0x251A, o2519)], [PtxIndexOccurrence(1, (o2519,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2519, -1, m_n), PtxIndexPathStep(0x2716, o2519)], [PtxIndexOccurrence(1, (o2519,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2519, -1, m_n), PtxIndexPathStep(0x251B, o2519, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (o2519,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2519, -1, m_n), PtxIndexPathStep(0x251B, o2519, b"\x01\x01\x00\x00\x00"), PtxIndexPathStep(0x251C, o2519)], [PtxIndexOccurrence(1, (o2519,))]),
        PtxIndexEntry([PtxIndexPathStep(0x2624, -1, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x2624][0],))]),
        PtxIndexEntry([PtxIndexPathStep(0x262A, -1)], [PtxIndexOccurrence(1, (offs[0x262A][0],))]),
    ]
    if include_0f3d:
        idx_entries.insert(1, PtxIndexEntry([PtxIndexPathStep(0x0F3D, -1, b"\x01\x00\x00\x00\x00")], [PtxIndexOccurrence(1, (offs[0x0F3D][0],))]))
    struct.pack_into("<I", out, 0x1B, len(out))
    out.extend(serialize_index_0002(PtxIndex0002(version=2, entries=idx_entries)))
    base = bytes(out)
    idx0 = parse_index_0002(base)
    assert idx0 is not None
    apply_rebuilt_index_0002(out, idx0, base)
    return encrypt_ptx(bytes(out), 0x40)


def test_blueprints_are_clean_of_private_paths_and_validate_inputs():
    for blob in (_STEREO_261C_RAW, _MONO_261C_RAW, AUDIO_UI_2589_RAW):
        low = blob.lower()
        for forbidden in (b"develop", b"sessionprep_private", b"scratch", b"probe", b"e:\\", b"c:\\", b"users", b"\xb0\xcf\xee\x62"):
            assert forbidden not in low

    uid, bus_uid = generate_ptx_uuid(), generate_ptx_uuid()
    st_routed = build_audio_mixer_strip_261c("Synth", uid, 2, (118, bus_uid, "KIK"), track_ordinal_1based=2, id_base=8000, color_index=29)
    mo_unrouted = build_audio_mixer_strip_261c("Kick", uid, 1, None, track_ordinal_1based=3, id_base=8010, color_index=25)
    assert st_routed[:1] == b"Z" and b"Synth" in st_routed and b"KIK" in st_routed and st_routed.count(uid) == 3
    assert mo_unrouted[:1] == b"Z" and b"Kick" in mo_unrouted and b"BUS" not in mo_unrouted and mo_unrouted.count(uid) == 3
    assert build_output_bus_260e(118, bus_uid, "KIK")[:1] == b"Z"
    assert build_track_summary_2519("Kick", uid, 1, 2).endswith(b"\x02\x00\x00")
    assert build_track_entry_251a("Kick", uid, 1, 2, color_index=25)[:1] == b"Z"
    assert build_track_uuid_entry_210b("Kick", uid)[:1] == b"Z"
    assert build_playlist_1014_v9("Synth", [0, 1], uid, is_stereo=True, playlist_ordinal_1based=1)[:1] == b"Z"
    assert build_playlist_1014_v9("Kick", [2], uid, is_stereo=False, playlist_ordinal_1based=2)[:1] == b"Z"
    c1003 = build_audio_file_entry_1003(
        96000, 2, 24, 96000, file_id8=b"\x01" * 8, smpte_umid32=b"\x02" * 32, filetime_u64=123456, file_index=0,
    )
    assert c1003[:1] == b"Z" and len(c1003) == 319 and struct.unpack_from("<I", c1003, 9)[0] == 1
    assert struct.unpack_from("<I", build_audio_file_entry_1003(
        96000, 1, 24, 96000, file_id8=b"\x01" * 8, smpte_umid32=b"\x02" * 32, filetime_u64=123456, file_index=2,
    ), 9)[0] == 3
    assert build_whole_file_clip_2629("Kick", 96000, audio_file_index=0, channel_index=0, clip_guid16=b"\x03" * 16, pt_uuid16=uid)[:1] == b"Z"
    assert build_volume_table_0f3d(0x12345678, "Data", "e:\\")[:1] == b"Z"

    with pytest.raises(ValueError):
        build_audio_mixer_strip_261c("Bad", b"short", 1, None, track_ordinal_1based=1, id_base=100)
    with pytest.raises(ValueError):
        build_audio_mixer_strip_261c("Bad", uid, 4, None, track_ordinal_1based=1, id_base=100)
    with pytest.raises(ValueError):
        build_track_entry_251a("Bad", b"short", 1, 1)


def test_mutate_202b_remaps_members_and_vca_indices_and_210c_adds_empty_folders():
    g_all = _build_group_202a("<ALL>", [0, 1, 2, 3], -1)
    g_kept_vca = _build_group_202a("KEPT", [0, 2], 1, vca_index=3)
    g_del_vca = _build_group_202a("DEAD", [0, 1], 2, vca_index=1)
    c202b = build_zchunk(
        0x202B,
        struct.pack("<I", 3) + g_all + g_kept_vca + g_del_vca + struct.pack("<I", 1) + g_kept_vca + b"\x00\x00",
        version=4,
    )
    # Delete index 1, insert 1 track after 0 -> old 0->0, 2->3, 3->4 (new total 5)
    mut = mutate_202b_groups_for_track_map(c202b, {0: 0, 2: 3, 3: 4}, new_track_count=5)
    from sessionpreplib.ptx.parser import split_202b
    edit_l, mix_l, _ = split_202b(mut)
    assert len(edit_l) == 3 and len(mix_l) == 1

    def _read_members_and_vca(raw_g: bytes) -> tuple[list[int], int]:
        nlen = struct.unpack_from("<I", raw_g, 9)[0]
        c_off = 13 + nlen + 7
        mc = struct.unpack_from("<I", raw_g, c_off)[0]
        mems = list(struct.unpack_from(f"<{mc}H", raw_g, c_off + 4)) if mc else []
        vca = struct.unpack_from("<H", raw_g, c_off + 4 + mc * 2 + 10)[0]
        return mems, vca

    assert _read_members_and_vca(edit_l[0]) == ([0, 1, 2, 3, 4], 0xFFFF)
    assert _read_members_and_vca(edit_l[1]) == ([0, 3], 4)  # VCA shifted 3 -> 4!
    assert _read_members_and_vca(edit_l[2]) == ([0], 0xFFFF)  # Deleted VCA -> 0xFFFF!
    assert _read_members_and_vca(mix_l[0]) == ([0, 3], 4)

    # Empty folder absent from 0x210C receives its newly inserted child!
    empty_f, child_u = _uuid(50), _uuid(51)
    empty_210c = serialize_folder_tree_210c([])
    mut_210c = mutate_210c_folders(empty_210c, {empty_f: [child_u]}, set())
    assert empty_f in mut_210c and child_u in mut_210c


def test_deterministic_walkers_handle_90_tracks_0x5a_count_and_z_in_names():
    # 90 == 0x5A ('Z'): previously triggered heuristic parse_chunk_tree corruption!
    n = 90
    uids = [_uuid(100 + i) for i in range(n)]
    strips = [_mixer_vca_261f(u, 0.0) for u in uids]
    c2624_90 = build_zchunk(0x2624, struct.pack("<I", n) + b"".join(strips), version=1)
    new_uid = generate_ptx_uuid()
    new_spec = NewAudioTrackSpec("ZZZ_Track", new_uid, generate_ptx_uuid(), 1, uids[0], None, 25, (0,), (0,))
    layout = [TrackSlotOp(old_index=i) for i in range(n)] + [TrackSlotOp(new_track=new_spec)]
    mut_2624 = mutate_2624_mixer(c2624_90, layout, expected_old_count=n)
    assert struct.unpack_from("<I", mut_2624, 9)[0] == 91

    rows = [build_track_summary_2519(f"ZTrack_{i}", uids[i], 1, i + 1) for i in range(n)]
    p251a = [build_track_entry_251a(f"ZTrack_{i}", uids[i], 1, i + 1) for i in range(n)]
    c2519_90 = build_zchunk(
        0x2519,
        struct.pack("<HIII", 1, 0, 0, 0) + struct.pack("<I", n) + b"".join(rows)
        + b"".join(p251a) + struct.pack("<I", n) + b"".join(p251a) + b"\x00\x01\x18\x00",
        version=9,
    )
    mut_2519 = mutate_2519_tracks(c2519_90, layout, expected_old_count=n)
    assert struct.unpack_from("<I", mut_2519, 23)[0] == 91


def test_wav_bwf_umid_strips_preexisting_duplicates_and_rejects_rf64(tmp_path: Path):
    src = _write_wav(tmp_path / "BwfSource.wav", 48000, 1, frames=4800)
    # Pre-seed stale bext + umid + minf chunks into source WAV
    stale_id, stale_umid, _ = _embed_protools_wav_identity(src)
    assert src.read_bytes().count(b"bext") == 1

    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    staged = stage_session_audio_files(proj_root, "CleanBwf", [src], TargetAudioSpec(48000, 24))
    out_bytes = Path(staged.staged_files[0].dest_path).read_bytes()
    # Must contain EXACTLY ONE of each chunk with the NEW identity (zero duplicates, zero stale IDs)
    assert out_bytes.count(b"bext") == 1
    assert out_bytes.count(b"umid") == 1
    assert out_bytes.count(b"minf") == 1
    assert staged.staged_files[0].file_id8 in out_bytes
    assert staged.staged_files[0].file_id8 != stale_id
    assert stale_umid not in out_bytes
    # Verify soundfile still reads the rewritten WAV cleanly
    info = sf.info(staged.staged_files[0].dest_path)
    assert info.frames == 4800 and info.samplerate == 48000

    # RF64 WAV must raise ValueError rather than silently returning unembedded UMIDs
    rf64 = tmp_path / "Big.wav"
    rf64.write_bytes(b"RF64\xff\xff\xff\xffWAVEds64\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="RF64"):
        _embed_protools_wav_identity(rf64)


def test_write_offline_ptx_end_to_end_empty_basic_folder_vca_remap_and_unicode(tmp_path: Path):
    ptxt = tmp_path / "Template.ptxt"
    ptxt.write_bytes(_build_clean_template_ptxt(
        include_empty_basic_folder=True, cnt_a_values=(1, 2), with_preexisting_vca_group=True,
    ))
    src_st = _write_wav(tmp_path / "Kick_Ω.wav", 48000, 2, frames=4800)
    src_mo = _write_wav(tmp_path / "Snare.wav", 48000, 1, frames=4800)

    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    staged = stage_session_audio_files(proj_root, "Café_Mix", [src_st, src_mo], TargetAudioSpec(48000, 24))
    tpl_sess = PtxSession.from_file(ptxt)
    kik = tpl_sess.track_by_name("KIK")
    empty_bf = tpl_sess.track_by_name("EmptyBasic")
    assert kik is not None and empty_bf is not None

    req = OfflinePtxWriteRequest(
        ptxt_path=ptxt,
        output_ptx_path=staged.ptx_path,
        session_dir=staged.session_dir,
        sample_rate=48000,
        bit_depth=24,
        audio_files=tuple(
            AudioFileWriteSpec(
                filename=s.filename, sample_rate=s.sample_rate, channels=s.channels,
                bit_depth=s.bit_depth, length_samples=s.length_samples,
                file_id8=s.file_id8, smpte_umid32=s.smpte_umid32, filetime_u64=s.filetime_u64,
            )
            for s in staged.staged_files
        ),
        tracks=(
            OfflineAudioTrackRequest("Kick Ω", 2, kik.uuid_hex, "KIK", 0, 4800, "Kick_Ω", color_index=29),
            OfflineAudioTrackRequest("UnroutedMono", 1, empty_bf.uuid_hex, None, 1, 4800, "Snare", color_index=25),
        ),
        delete_track_names=("VOX", "VOX-VCA"),
        clip_gains_db={"Kick Ω": -3.0},
        track_volumes_db={"Kick Ω": 3.0},
        group_requests=(
            OfflineGroupRequest("KIK-PRE", "add_members", ("Kick Ω",), vca_track="KIK-VCA"),
        ),
    )
    res = write_offline_ptx_session(req)
    assert res.ptx_path.is_file()
    parsed = PtxSession.from_file(res.ptx_path)
    assert parsed.warnings == []
    assert [t.name for t in parsed.tracks] == ["KIK", "Kick Ω", "KIK-VCA", "EmptyBasic", "UnroutedMono"]
    t_unr = parsed.track_by_name("UnroutedMono")
    assert t_unr is not None and t_unr.parent_folder_name == "EmptyBasic" and t_unr.main_output is None
    # Verify 0x0002 emitted 0x2627 rack entries for newly added Audio tracks even though template had 0!
    dec, _ = decrypt_ptx(res.ptx_path.read_bytes())
    assert validate_index_0002(dec) is True
    idx = parse_index_0002(dec)
    assert idx is not None
    rack_entries = [e for e in idx.entries if len(e.path) == 4 and e.path[0].tag == 0x2624 and e.path[3].tag == 0x2627]
    assert len(rack_entries) == 2
    # Pre-existing VCA-linked group KIK-PRE stayed linked to shifted KIK-VCA (index 2) and ALL-VCA cleared deleted VOX-VCA
    g_pre = next(g for g in parsed.groups if g.name == "KIK-PRE")
    assert g_pre.vca_track_name == "KIK-VCA" and "Kick Ω" in [parsed.tracks[i].name for i in g_pre.member_indices]
    g_all_vca = next(g for g in parsed.groups if g.name == "ALL-VCA")
    assert g_all_vca.vca_track_name is None


def test_template_variants_validation_guards_and_rollback(tmp_path: Path):
    # Variant without optional 0x0F3D & 0x2587 chunks still writes cleanly
    ptxt_min = tmp_path / "Minimal.ptxt"
    ptxt_min.write_bytes(_build_clean_template_ptxt(include_0f3d=False, include_2587=False))
    wav = _write_wav(tmp_path / "Kick.wav", 48000, 1, frames=4800)
    proj_root = tmp_path / "out"
    proj_root.mkdir()
    staged = stage_session_audio_files(proj_root, "MinRun", [wav], TargetAudioSpec(48000, 24))
    tpl = PtxSession.from_file(ptxt_min)
    kik = tpl.track_by_name("KIK")
    assert kik is not None
    af = AudioFileWriteSpec(
        staged.staged_files[0].filename, 48000, 1, 24, 4800,
        staged.staged_files[0].file_id8, staged.staged_files[0].smpte_umid32, staged.staged_files[0].filetime_u64,
    )
    res = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=ptxt_min, output_ptx_path=staged.ptx_path, session_dir=staged.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=(af,),
        tracks=(OfflineAudioTrackRequest("Kick", 1, kik.uuid_hex, "KIK", 0, 4800, "Kick"),),
    ))
    assert res.ptx_path.is_file()

    # Reject inserting into missing or deleted folder, deleting folder with surviving child, or bad group action
    with pytest.raises(ValueError, match="missing or deleted parent folder"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=ptxt_min, output_ptx_path=tmp_path / "x.ptx", session_dir=tmp_path,
            sample_rate=48000, bit_depth=24, audio_files=(af,),
            tracks=(OfflineAudioTrackRequest("Kick", 1, kik.uuid_hex, "KIK", 0, 4800, "Kick"),),
            delete_track_names=("KIK", "KIK-VCA"),
        ))
    with pytest.raises(ValueError, match="surviving child"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=ptxt_min, output_ptx_path=tmp_path / "x.ptx", session_dir=tmp_path,
            sample_rate=48000, bit_depth=24, audio_files=(af,),
            tracks=(OfflineAudioTrackRequest("Kick", 1, kik.uuid_hex, "KIK", 0, 4800, "Kick"),),
            delete_track_names=("VOX",),  # VOX-VCA survives!
        ))
    with pytest.raises(ValueError, match="Unsupported offline group action"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=ptxt_min, output_ptx_path=tmp_path / "x.ptx", session_dir=tmp_path,
            sample_rate=48000, bit_depth=24, audio_files=(af,),
            tracks=(OfflineAudioTrackRequest("Kick", 1, kik.uuid_hex, "KIK", 0, 4800, "Kick"),),
            group_requests=(OfflineGroupRequest("Bad", "bogus", ("Kick",)),),
        ))

    # execute_offline_ptx_creation rolls back newly staged directory when writer fails, never touching pre-existing dir
    existing = proj_root / "ExistingSession"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("safe")
    tc = TrackContext("Kick.wav", str(wav), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
    sess = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="ExistingSession",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )
    plan_existing = build_protools_creation_plan(
        ptxt_min, sess, {"Kick.wav": kik.uuid_braced}, [{"id": kik.uuid_braced, "name": "KIK"}],
        compute_signal_flow=False,
    )
    with pytest.raises(FileExistsError):
        execute_offline_ptx_creation(plan_existing, proj_root)
    assert sentinel.is_file()

    # Non-clean template (already has audio files) fails writer -> staged dir rolled back cleanly!
    sess_rb = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="RollbackSong",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )
    plan_rb = build_protools_creation_plan(
        res.ptx_path, sess_rb, {"Kick.wav": kik.uuid_braced}, [{"id": kik.uuid_braced, "name": "KIK"}],
        compute_signal_flow=False,
    )
    with pytest.raises(ValueError, match="clean session template"):
        execute_offline_ptx_creation(plan_rb, proj_root)
    assert not (proj_root / "RollbackSong").exists()


def test_protools_processor_offline_transfer_end_to_end_and_empty_plan_guard(tmp_path: Path):
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(_build_clean_template_ptxt())
    wav = _write_wav(tmp_path / "Kick.wav", 48000, 1, frames=4800)
    proj_root = tmp_path / "out"
    proj_root.mkdir()

    tc = TrackContext("Kick.wav", str(wav), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
    proc = ProToolsDawProcessor(instance_index=0, instance_group="", instance_name="Mix", instance_ptxt_path=str(ptxt))
    proc.configure({
        "protools_transfer_backend": "offline_ptx",
        "protools_project_dir": str(proj_root),
        "protools_stage_bus_faders": False,
    })
    ok, msg = proc.check_connectivity()
    assert ok is True and "Offline .ptx ready" in msg

    sess = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="OfflineRun",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )
    proc.fetch(sess, ignore_cache=True)
    kik_folder = next(f for f in sess.daw_state[proc.id]["folders"] if f["name"] == "KIK")
    sess.daw_state[proc.id]["assignments"] = {"Kick.wav": kik_folder["id"]}

    results = proc.transfer(sess, "")
    assert results and all(r.success for r in results)
    ptx_out = proj_root / "OfflineRun" / "OfflineRun.ptx"
    assert ptx_out.is_file()
    parsed = PtxSession.from_file(ptx_out)
    assert [t.name for t in parsed.tracks] == ["KIK", "Kick", "KIK-VCA"]

    # Empty plan guard: invalid folder target -> fails cleanly without creating session dir!
    sess_bad = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="NoTracksRun",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
        daw_state={proc.id: {"folders": [{"id": "bogus", "name": "MissingFolder"}], "assignments": {"Kick.wav": "bogus"}}},
    )
    bad_res = proc.transfer(sess_bad, "")
    assert any(not r.success for r in bad_res)
    assert not (proj_root / "NoTracksRun").exists()


def test_rebuild_index_0002_idempotent_on_template():
    raw = _build_clean_template_ptxt()
    dec, _ = decrypt_ptx(raw)
    idx = parse_index_0002(dec)
    assert idx is not None
    rebuilt = rebuild_index_0002(idx, dec, dec)
    assert serialize_index_0002(rebuilt) == serialize_index_0002(idx)
    assert len(iter_top_chunks(dec, strict=True)) > 10


def test_audio_file_entry_1003_encodes_tree_node_index_and_channel_layout_flags():
    for file_index, channels, expected_flag in ((0, 2, 1), (1, 1, 0), (2, 1, 0), (3, 2, 1)):
        c1003 = build_audio_file_entry_1003(
            48000, channels, 24, 4800,
            file_id8=b"\x11" * 8, smpte_umid32=b"\x22" * 32, filetime_u64=123456789, file_index=file_index,
        )
        assert struct.unpack_from("<I", c1003, 9)[0] == file_index + 1
        sz1001 = struct.unpack_from("<I", c1003, 13 + 3)[0]
        p1033 = 13 + 7 + sz1001
        sz1033 = struct.unpack_from("<I", c1003, p1033 + 3)[0]
        p2106 = p1033 + 7 + sz1033
        body2106 = c1003[p2106 + 9 : p2106 + 7 + struct.unpack_from("<I", c1003, p2106 + 3)[0]]
        assert struct.unpack_from("<I", body2106, 29 + 8 + 4 + 51 + 8 + 3)[0] == expected_flag
        assert body2106[-48] == expected_flag
        assert body2106[-47:-15] == b"\x22" * 32
        assert struct.unpack_from("<I", body2106, len(body2106) - 8)[0] == expected_flag
