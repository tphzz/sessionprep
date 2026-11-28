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
    build_audio_file_entry_1003,
    build_audio_mixer_strip_261c,
    build_audio_ui_entry_2589,
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
    c258a = build_zchunk(0x258A, build_zchunk(0x2581, b"\x00" * 8, version=3) + cnt_le + build_audio_ui_entry_2589() * len(tracks), version=2)
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


def test_structured_builders_are_blob_free_and_validate_inputs():
    import inspect
    from sessionpreplib.ptx import writer_blueprints as wb

    src = inspect.getsource(wb)
    assert "b64decode" not in src and "zlib.decompress" not in src

    uid, bus_uid = generate_ptx_uuid(), generate_ptx_uuid()
    ui_raw = build_audio_ui_entry_2589()
    st_routed = build_audio_mixer_strip_261c("Synth", uid, 2, (118, bus_uid, "KIK"), track_ordinal_1based=2, id_base=8000, color_index=29)
    st_routed_2 = build_audio_mixer_strip_261c("Synth", uid, 2, (118, bus_uid, "KIK"), track_ordinal_1based=2, id_base=8000, color_index=29)
    mo_unrouted = build_audio_mixer_strip_261c("Kick", uid, 1, None, track_ordinal_1based=3, id_base=8010, color_index=25)
    assert len(ui_raw) == 283
    assert st_routed != st_routed_2  # fresh unique send-slot UUIDv4s per synthesized strip
    for chunk in (ui_raw, st_routed, mo_unrouted):
        low = chunk.lower()
        for forbidden in (b"c:\\", b"e:\\", b"/users/", b"\\users\\", b".wav", b"audio files"):
            assert forbidden not in low
    assert st_routed[:1] == b"Z" and b"Synth" in st_routed and b"KIK" in st_routed and st_routed.count(uid) == 3
    assert mo_unrouted[:1] == b"Z" and b"Kick" in mo_unrouted and b"BUS" not in mo_unrouted and mo_unrouted.count(uid) == 3
    # No-output 0x4856 must omit 0x260E and 0x485E (63 bytes exact, prevents Pro Tools GlobalRenderer assertion)
    p4856 = mo_unrouted.find(b"\x56\x48") - 7
    assert struct.unpack_from("<I", mo_unrouted, p4856 + 3)[0] + 7 == 63
    assert build_output_bus_260e(118, bus_uid, "KIK")[:1] == b"Z"
    assert build_track_summary_2519("Kick", uid, 1, 2).endswith(b"\x02\x00\x00")
    assert build_track_entry_251a("Kick", uid, 1, 2, color_index=25)[:1] == b"Z"
    assert build_track_uuid_entry_210b("Kick", uid)[:1] == b"Z"
    pl_st = build_playlist_1014_v9("Synth", (3, 4), uid, is_stereo=True)
    pl_mn = build_playlist_1014_v9("Kick", (5,), uid, is_stereo=False)
    assert pl_st[:1] == b"Z" and struct.unpack_from("<II", pl_st, len(pl_st) - 14) == (3, 4)
    assert pl_mn[:1] == b"Z" and struct.unpack_from("<I", pl_mn, len(pl_mn) - 10)[0] == 5
    c1003 = build_audio_file_entry_1003(
        96000, 2, 24, 96000, file_id8=b"\x01" * 8, smpte_umid32=b"\x02" * 32, filetime_u64=123456, tree_node_index=1,
    )
    assert c1003[:1] == b"Z" and len(c1003) == 319 and struct.unpack_from("<I", c1003, 9)[0] == 1
    assert struct.unpack_from("<I", build_audio_file_entry_1003(
        96000, 1, 24, 96000, file_id8=b"\x01" * 8, smpte_umid32=b"\x02" * 32, filetime_u64=123456, tree_node_index=14,
    ), 9)[0] == 14
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
    with pytest.raises(ValueError, match="missing, non-folder, or deleted parent folder"):
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

    # Writer failure (corrupt/truncated template missing required chunks) rolls back staged dir cleanly!
    bad_tpl = tmp_path / "Broken.ptxt"
    bad_tpl.write_bytes(res.ptx_path.read_bytes())
    sess_rb = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="RollbackSong",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
    )
    plan_rb = build_protools_creation_plan(
        ptxt_min, sess_rb, {"Kick.wav": kik.uuid_braced}, [{"id": kik.uuid_braced, "name": "KIK"}],
        compute_signal_flow=False,
    )
    # Point plan at a missing parent UUID so writer fails after media staging
    from dataclasses import replace as dc_replace
    bad_plan = dc_replace(plan_rb, tracks=(dc_replace(plan_rb.tracks[0], parent_folder_uuid="{00000000-0000-0000-0000-000000000000}"),))
    with pytest.raises(ValueError, match="missing, non-folder, or deleted parent folder"):
        execute_offline_ptx_creation(bad_plan, proj_root)
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
    for node_index, channels, expected_flag in ((1, 2, 1), (2, 1, 0), (12, 1, 0), (14, 2, 1)):
        c1003 = build_audio_file_entry_1003(
            48000, channels, 24, 4800,
            file_id8=b"\x11" * 8, smpte_umid32=b"\x22" * 32, filetime_u64=123456789, tree_node_index=node_index,
        )
        assert struct.unpack_from("<I", c1003, 9)[0] == node_index
        sz1001 = struct.unpack_from("<I", c1003, 13 + 3)[0]
        p1033 = 13 + 7 + sz1001
        sz1033 = struct.unpack_from("<I", c1003, p1033 + 3)[0]
        p2106 = p1033 + 7 + sz1033
        body2106 = c1003[p2106 + 9 : p2106 + 7 + struct.unpack_from("<I", c1003, p2106 + 3)[0]]
        assert struct.unpack_from("<I", body2106, 29 + 8 + 4 + 51 + 8 + 3)[0] == expected_flag
        assert body2106[-48] == expected_flag
        assert body2106[-47:-15] == b"\x22" * 32
        assert struct.unpack_from("<I", body2106, len(body2106) - 8)[0] == expected_flag


def test_template_with_preexisting_audio_append_and_delete(tmp_path: Path):
    """Appending new tracks onto a template with existing audio preserves existing media & compacts deleted lanes."""
    base_ptxt = tmp_path / "Clean.ptxt"
    base_ptxt.write_bytes(_build_clean_template_ptxt())
    w_ref = _write_wav(tmp_path / "RefLoop.wav", 48000, 2, frames=4800)
    w_old = _write_wav(tmp_path / "OldVox.wav", 48000, 1, frames=4800)
    w_new_st = _write_wav(tmp_path / "NewSt.wav", 48000, 2, frames=4800)
    w_new_mn = _write_wav(tmp_path / "NewMn.wav", 48000, 1, frames=4800)

    root = tmp_path / "sessions"
    root.mkdir()
    base_sess = PtxSession.from_file(base_ptxt)
    kik_uid = base_sess.track_by_name("KIK").uuid_hex
    vox_uid = base_sess.track_by_name("VOX").uuid_hex
    assert kik_uid and vox_uid

    # Step 1: Produce a template session that ALREADY contains 2 audio files, 3 clips, 2 playlists, 3 lanes.
    st1 = stage_session_audio_files(root, "TplWithAudio", [w_ref, w_old], TargetAudioSpec(48000, 24))
    af1 = tuple(
        AudioFileWriteSpec(sf.filename, sf.sample_rate, sf.channels, sf.bit_depth, sf.length_samples, sf.file_id8, sf.smpte_umid32, sf.filetime_u64)
        for sf in st1.staged_files
    )
    tpl_res = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=base_ptxt, output_ptx_path=st1.ptx_path, session_dir=st1.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=af1,
        tracks=(
            OfflineAudioTrackRequest("RefTrack", 2, kik_uid, "KIK", 0, 4800, "RefLoop"),
            OfflineAudioTrackRequest("OldVox", 1, vox_uid, "VOX", 1, 4800, "OldVox"),
        ),
        volume_serial_u32=0x11112222, volume_label="TplVol",
    ))

    # Step 2: Append 2 new tracks (1 Stereo + 1 Mono) onto TplWithAudio while KEEPING all existing tracks.
    st2 = stage_session_audio_files(root, "AppendedKeep", [w_new_st, w_new_mn], TargetAudioSpec(48000, 24))
    af2 = tuple(
        AudioFileWriteSpec(sf.filename, sf.sample_rate, sf.channels, sf.bit_depth, sf.length_samples, sf.file_id8, sf.smpte_umid32, sf.filetime_u64)
        for sf in st2.staged_files
    )
    res_keep = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=tpl_res.ptx_path, output_ptx_path=st2.ptx_path, session_dir=st2.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=af2,
        tracks=(
            OfflineAudioTrackRequest("NewSt", 2, kik_uid, "KIK", 0, 4800, "NewSt"),
            OfflineAudioTrackRequest("NewMn", 1, kik_uid, "KIK", 1, 4800, "NewMn"),
        ),
        volume_serial_u32=0x33334444, volume_label="OutVol",
    ))
    s_keep = PtxSession.from_file(res_keep.ptx_path)
    assert not s_keep.warnings
    assert [a.filename for a in s_keep.audio_files] == ["RefLoop.wav", "OldVox.wav", "NewSt.wav", "NewMn.wav"]
    assert len(s_keep.clips) == 6
    assert [(p.name, p.lane_indices) for p in s_keep.playlists] == [
        ("RefTrack", [0, 1]), ("OldVox", [2]), ("NewSt", [3, 4]), ("NewMn", [5]),
    ]

    # Step 3: Append new tracks onto TplWithAudio while DELETING existing audio track RefTrack (playlist 0)
    # so surviving OldVox shifts from playlist 1 -> 0 and lane 2 -> 0, and new tracks take playlists 1 & 2!
    st3 = stage_session_audio_files(root, "AppendedDelete", [w_new_st, w_new_mn], TargetAudioSpec(48000, 24))
    af3 = tuple(
        AudioFileWriteSpec(sf.filename, sf.sample_rate, sf.channels, sf.bit_depth, sf.length_samples, sf.file_id8, sf.smpte_umid32, sf.filetime_u64)
        for sf in st3.staged_files
    )
    res_del = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=tpl_res.ptx_path, output_ptx_path=st3.ptx_path, session_dir=st3.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=af3,
        tracks=(
            OfflineAudioTrackRequest("NewSt", 2, kik_uid, "KIK", 0, 4800, "NewSt"),
            OfflineAudioTrackRequest("NewMn", 1, kik_uid, "KIK", 1, 4800, "NewMn"),
        ),
        delete_track_names=("RefTrack",),
        volume_serial_u32=0x11112222, volume_label="TplVol",  # same serial -> deduplicated in 0x0F3D
    ))
    s_del = PtxSession.from_file(res_del.ptx_path)
    assert not s_del.warnings
    assert len(s_del.audio_files) == 4 and len(s_del.clips) == 6  # Clip list preserved like PT
    assert [(p.name, p.lane_indices) for p in s_del.playlists] == [
        ("OldVox", [0]), ("NewSt", [1, 2]), ("NewMn", [3]),
    ]
    # Verify every Audio track's 0x261C mid_gap, tail, and 0x2015 track ordinal match compacted layout!
    dec_del, _ = decrypt_ptx(res_del.ptx_path.read_bytes())
    assert validate_index_0002(dec_del) is True
    c2624_del = next(c.raw for c in iter_top_chunks(dec_del, strict=True) if c.tag == 0x2624)
    from sessionpreplib.ptx.codec import find_zchunk_offset, read_zchunk_at, walk_sequential_zchunks
    from sessionpreplib.ptx.writer_mutations import _extract_261c_playlist_indices
    strips_del, _ = walk_sequential_zchunks(c2624_del, 13, struct.unpack_from("<I", c2624_del, 9)[0])
    audio_pl_indices = []
    audio_view_ordinals = []
    for ord_1, st_raw in enumerate(strips_del, start=1):
        if struct.unpack_from("<H", st_raw, 7)[0] == 0x261C:
            audio_pl_indices.append(sorted(_extract_261c_playlist_indices(st_raw)))
            _, _, mid_off = read_zchunk_at(st_raw, 9, expected_tag=0x261B)
            _, _, tail_off = read_zchunk_at(st_raw, mid_off + 12, expected_tag=0x200B)
            assert struct.unpack_from("<I", st_raw, mid_off)[0] == struct.unpack_from("<I", st_raw, tail_off + 17)[0]
            p2434 = find_zchunk_offset(st_raw, 0x2434, mid_off + 21)
            after_2434 = p2434 + 7 + struct.unpack_from("<I", st_raw, p2434 + 3)[0]
            audio_view_ordinals.append((ord_1, struct.unpack_from("<I", st_raw, after_2434)[0]))
    assert audio_pl_indices == [[1], [2], [0]]  # track order: NewSt(pl 1), NewMn(pl 2) under KIK, OldVox(pl 0) under VOX
    assert all(expected == actual for expected, actual in audio_view_ordinals)


def test_review_edge_cases_playlist_ownership_fades_strides_and_guards(tmp_path: Path):
    """Lock down edge cases: non-Audio name collisions, renamed playlists, fade tails, alternate playlists, guards."""
    from sessionpreplib.ptx.codec import splice_top_chunk
    from sessionpreplib.ptx.writer_chunks import serialize_lane_1052
    from sessionpreplib.ptx.writer_mutations import (
        _extract_261c_playlist_indices,
        _parse_playlist_1014_meta,
        _remap_261c_strip_indices,
        _remap_playlist_1014_lanes,
        mutate_0f3d_volumes,
        plan_existing_media_survival,
    )

    # 1. Deleting a non-Audio track named 'VOX' must NOT delete a surviving Audio track's playlist named 'VOX',
    #    while deleting an Audio track whose playlist is renamed 'AltLead' MUST delete 'AltLead' via 0x261C ownership.
    pl0 = build_playlist_1014_v9("VOX", (0,), generate_ptx_uuid(), is_stereo=False)
    pl1 = build_playlist_1014_v9("AltLead", (1,), generate_ptx_uuid(), is_stereo=False)
    pl2 = build_playlist_1014_v9("SharedEnd", (1, 2), generate_ptx_uuid(), is_stereo=True)  # shares lane 1 with pl1!
    c1015 = build_zchunk(0x1015, struct.pack("<I", 3) + pl0 + pl1 + pl2 + build_zchunk(0x4829, b"\x00" * 4, version=2), version=2)
    lanes = [serialize_lane_1052(f"L{i}", [(i, 0, 3, 0xFFFE)], wrap_1050=True, wrap_version=3, lane_version=3) for i in range(3)]
    c1054 = build_zchunk(0x1054, struct.pack("<I", 3) + b"".join(lanes) + build_zchunk(0x4828, b"\x00" * 4, version=2), version=2)
    st_folder = _mixer_track_261e(_uuid(1), None, _uuid(99), 0.0)
    st_keep0 = build_audio_mixer_strip_261c("Keep0", _uuid(2), 1, None, track_ordinal_1based=2, id_base=100, playlist_index=0)
    st_del1 = build_audio_mixer_strip_261c("DelMid", _uuid(3), 1, None, track_ordinal_1based=3, id_base=110, playlist_index=1)
    st_keep2 = build_audio_mixer_strip_261c("Keep2", _uuid(4), 2, None, track_ordinal_1based=4, id_base=120, playlist_index=2)
    c2624 = build_zchunk(0x2624, struct.pack("<I", 4) + st_folder + st_keep0 + st_del1 + st_keep2, version=1)

    surv = plan_existing_media_survival(
        c1015, c1054, ("VOX", "DelMid"), c2624_raw=c2624, deleted_old_indices=(0, 2),
    )
    assert [_parse_playlist_1014_meta(p)[0] for p in surv.playlists] == ["VOX", "SharedEnd"]
    assert surv.old_playlist_to_new == {0: 0, 2: 1}
    assert len(surv.lanes) == 3  # lane 1 survived because SharedEnd still references it!

    # 2. Alternate playlists (pl_cnt == 2) for both 2-byte (<H) and 4-byte (<I) mid_gap strides + error on malformed gap
    from sessionpreplib.ptx.writer_mutations import _read_zchunk_at
    _, _, mid_off = _read_zchunk_at(st_keep2, 9, expected_tag=0x261B)
    _, raw_200b, tail_off = _read_zchunk_at(st_keep2, mid_off + 12, expected_tag=0x200B)
    st_alt_u16 = build_zchunk(
        0x261C, st_keep2[9:mid_off] + struct.pack("<IHHHHH", 2, 2, 2, 3, 0, 0xFFFF) + raw_200b + st_keep2[tail_off:], version=6,
    )
    assert _extract_261c_playlist_indices(st_alt_u16) == {2, 3}
    remapped_u16 = _remap_261c_strip_indices(st_alt_u16, {2: 0, 3: 1}, 5)
    assert _extract_261c_playlist_indices(remapped_u16) == {0, 1}
    st_alt_u32 = build_zchunk(
        0x261C, st_keep2[9:mid_off] + struct.pack("<IHIIH", 2, 2, 2, 3, 0xFFFF) + raw_200b + st_keep2[tail_off:], version=6,
    )
    assert _extract_261c_playlist_indices(st_alt_u32) == {2, 3}
    remapped_u32 = _remap_261c_strip_indices(st_alt_u32, {2: 0, 3: 1}, 5)
    assert _extract_261c_playlist_indices(remapped_u32) == {0, 1}
    with pytest.raises(RuntimeError, match="missing playlist index"):
        _remap_261c_strip_indices(st_alt_u16, {2: 0}, 5)
    st_bad_gap = build_zchunk(0x261C, st_keep2[9:mid_off] + b"\x00" * 11 + raw_200b + st_keep2[tail_off:], version=6)
    with pytest.raises(RuntimeError, match="Unrecognized 0x261C mid_gap"):
        _extract_261c_playlist_indices(st_bad_gap)

    # 3. Template with fade-file 0x1003 in 0x1004 trailing_tail + delete-only (0 new tracks) + shared audio_file_index
    ptxt = tmp_path / "FadeTpl.ptxt"
    raw_tpl = _build_clean_template_ptxt()
    dec_tpl, delta = decrypt_ptx(raw_tpl)
    buf_tpl = bytearray(dec_tpl)
    c1004_orig = next(c.raw for c in iter_top_chunks(dec_tpl, strict=True) if c.tag == 0x1004)
    fade_1003 = build_audio_file_entry_1003(48000, 1, 24, 128, file_id8=b"\x99" * 8, smpte_umid32=b"\x88" * 32, filetime_u64=1, tree_node_index=0)
    # Inject a fade-file 0x1003 in trailing_tail while main header count stays 0!
    c1004_with_fade = build_zchunk(0x1004, c1004_orig[9:-17] + struct.pack("<I", 1) + fade_1003 + c1004_orig[-13:], version=3)
    splice_top_chunk(buf_tpl, 0x1004, c1004_with_fade)
    ptxt.write_bytes(encrypt_ptx(bytes(buf_tpl), delta))

    wav_st = _write_wav(tmp_path / "SharedSt.wav", 48000, 2, frames=4800)
    wav_mn = _write_wav(tmp_path / "MonoOne.wav", 48000, 1, frames=4800)
    root = tmp_path / "out"
    root.mkdir()
    st_run = stage_session_audio_files(root, "FadeAndShare", [wav_st, wav_mn], TargetAudioSpec(48000, 24))
    afs = tuple(AudioFileWriteSpec(s.filename, s.sample_rate, s.channels, s.bit_depth, s.length_samples, s.file_id8, s.smpte_umid32, s.filetime_u64) for s in st_run.staged_files)
    kik_uid = PtxSession.from_file(ptxt).track_by_name("KIK").uuid_hex
    assert kik_uid
    res_shared = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=ptxt, output_ptx_path=st_run.ptx_path, session_dir=st_run.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=afs,
        tracks=(
            OfflineAudioTrackRequest("TrkA", 2, kik_uid, "KIK", 0, 4800, "SharedA"),
            OfflineAudioTrackRequest("TrkB", 2, kik_uid, "KIK", 0, 4800, "SharedB"),  # shares audio_file_index=0
        ),
    ))
    parsed_sh = PtxSession.from_file(res_shared.ptx_path)
    assert [c.audio_file_index for c in parsed_sh.clips] == [0, 0, 0, 0]  # NOT shifted by fade 0x1003!

    # Delete-only (0 new tracks, 0 staged files) on populated session
    res_del_only = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=res_shared.ptx_path, output_ptx_path=tmp_path / "DelOnly.ptx", session_dir=st_run.session_dir,
        sample_rate=48000, bit_depth=24, audio_files=(), tracks=(), delete_track_names=("TrkB",),
    ))
    assert [p.name for p in PtxSession.from_file(res_del_only.ptx_path).playlists] == ["TrkA"]

    # 4. Validation & malformed-chunk guards + writer_index 0x2519 non-0x5A gap bytes
    with pytest.raises(ValueError, match="requires 2 channels"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=ptxt, output_ptx_path=tmp_path / "BadCh.ptx", session_dir=st_run.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=afs,
            tracks=(OfflineAudioTrackRequest("BadSt", 2, kik_uid, "KIK", 1, 4800, "MonoOne"),),
        ))
    with pytest.raises(ValueError, match="Duplicate or colliding track name"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=ptxt, output_ptx_path=tmp_path / "DupTrk.ptx", session_dir=st_run.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=afs,
            tracks=(OfflineAudioTrackRequest("KIK", 1, kik_uid, "KIK", 1, 4800, "MonoOne"),),
        ))
    with pytest.raises(ValueError, match="Duplicate audio filename"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=res_shared.ptx_path, output_ptx_path=tmp_path / "DupWav.ptx", session_dir=st_run.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(afs[0],),
            tracks=(OfflineAudioTrackRequest("TrkC", 2, kik_uid, "KIK", 0, 4800, "SharedC"),),
        ))
    with pytest.raises(ValueError, match="exactly one of old_index or new_track"):
        TrackSlotOp()
    with pytest.raises(RuntimeError, match="Truncated 0x1014"):
        _parse_playlist_1014_meta(build_zchunk(0x1014, struct.pack("<I", 50) + b"abc", version=9))
    with pytest.raises(RuntimeError, match="missing lane"):
        _remap_playlist_1014_lanes(pl0, {99: 0})
    bad_0f3c = build_zchunk(0x0F3C, struct.pack("<I", 0) + struct.pack("<II", 1, 999), version=1)
    with pytest.raises(RuntimeError, match="Truncated 0x0F3C"):
        mutate_0f3d_volumes(build_zchunk(0x0F3D, struct.pack("<I", 1) + bad_0f3c, version=1), 2, "V", "d:\\")

    # writer_index.py:120 non-0x5A gap bytes between 0x2716 and 0x251B inside 0x2519
    from sessionpreplib.ptx.models import PtxChunk
    from sessionpreplib.ptx.writer_index import _parse_2519_offsets
    c2716 = build_zchunk(0x2716, b"\x00\x00\x00\x00", version=1)
    c251b = build_zchunk(0x251B, b"\x00\x00\x00\x00", version=1)
    # 0x2519 body: 10B hdr + cnt_a=0 + tcnt=0 + tcnt2=0 + c2716 + 4B non-0x5A gap + c251b
    fake_2519 = build_zchunk(0x2519, (b"\x00" * 10) + struct.pack("<III", 0, 0, 0) + c2716 + b"\xaa\xbb\xcc\xdd" + c251b, version=9)
    parsed_2519 = _parse_2519_offsets(fake_2519, PtxChunk(offset=0, version=9, size=len(fake_2519) - 7, tag=0x2519, raw=fake_2519))
    assert parsed_2519.c2716_offset == 9 + 22
    assert parsed_2519.c251b_offset == 9 + 22 + len(c2716) + 4


def test_virgin_template_2587_siblings_hidden_tracks_top_level_and_guards(tmp_path: Path, monkeypatch):
    """Verify virgin-template synthesis, canonical ordering, 0x2587 sibling shifts, hidden tracks, top-level/empty tracks, and guards."""
    from sessionpreplib.ptx import (
        OutputBusSpec,
        PtxUnsupportedTemplateError,
        PtxValidationError,
        normalize_ptx_uuid16,
    )
    from sessionpreplib.ptx.writer_mutations import mutate_2519_tracks

    # 1. Virgin template missing 0x0F3D, 0x1004, 0x2637, 0x262A, 0x1054, 0x1015, 0x210C PLUS 0x2587 sibling after 0x258A!
    raw_base = _build_clean_template_ptxt(include_0f3d=False, include_2587=True)
    dec_base, delta = decrypt_ptx(raw_base)
    top = iter_top_chunks(dec_base, strict=True)
    # Remove optional virgin-omitted chunks and append a sibling 0x2586 inside 0x2587->0x2551 AFTER 0x258B/0x258A!
    omit_tags = {0x0F3D, 0x1004, 0x2637, 0x262A, 0x1054, 0x1015, 0x210C}
    old_2587 = next(c for c in top if c.tag == 0x2587)
    sib_2056 = build_zchunk(0x2056, b"\x01\x02\x03\x04", version=6)
    sib_2586 = build_zchunk(0x2586, sib_2056, version=1)
    # Rebuild 0x2587 -> 0x2551 -> (0x258B, 0x2586)
    old_258b = old_2587.raw[18:]  # inside 0x2587(9) + 0x2551(9)
    new_2587 = build_zchunk(0x2587, build_zchunk(0x2551, old_258b + sib_2586, version=3), version=1)

    out_tpl = bytearray(dec_base[:0x1F])
    kept_top = []
    for ch in top:
        if ch.tag in omit_tags or ch.tag == 0x0002:
            continue
        off = len(out_tpl)
        blob = new_2587 if ch.tag == 0x2587 else ch.raw
        out_tpl.extend(blob)
        kept_top.append((ch.tag, off, blob))
    o2587 = next(off for t, off, _ in kept_top if t == 0x2587)
    o2551 = o2587 + 9
    o2586 = o2551 + 9 + len(old_258b)
    o2056 = o2586 + 9
    old_idx = parse_index_0002(dec_base)
    assert old_idx is not None
    virgin_entries = [
        e for e in old_idx.entries
        if e.path and e.path[0].tag not in omit_tags and e.path[0].tag != 0x2587
    ]
    # Add 0x2587 + post-0x258A sibling 0x2586->0x2056 entries!
    virgin_entries.append(PtxIndexEntry([PtxIndexPathStep(0x2587, -1)], [PtxIndexOccurrence(1, (o2587,))]))
    virgin_entries.append(PtxIndexEntry(
        [PtxIndexPathStep(0x2587, -1), PtxIndexPathStep(0x2551, o2587), PtxIndexPathStep(0x2586, o2551), PtxIndexPathStep(0x2056, o2586)],
        [PtxIndexOccurrence(1, (o2056,))],
    ))
    # Recompute static offsets via apply_rebuilt_index_0002
    o0002 = len(out_tpl)
    struct.pack_into("<I", out_tpl, 0x1B, o0002)
    out_tpl.extend(serialize_index_0002(PtxIndex0002(version=2, entries=virgin_entries)))
    # Fix shifted pre-existing entries once so virgin_ptxt is valid before writing
    virgin_dec = bytearray(out_tpl)
    rebuilt_v = rebuild_index_0002(PtxIndex0002(version=2, entries=virgin_entries), dec_base, bytes(virgin_dec))
    # Restore exact 0x2587 sibling offsets (since dec_base had smaller 0x2587)
    rebuilt_v.entries[-2] = PtxIndexEntry([PtxIndexPathStep(0x2587, -1)], [PtxIndexOccurrence(1, (o2587,))])
    rebuilt_v.entries[-1] = PtxIndexEntry(
        [PtxIndexPathStep(0x2587, -1), PtxIndexPathStep(0x2551, o2587), PtxIndexPathStep(0x2586, o2551), PtxIndexPathStep(0x2056, o2586)],
        [PtxIndexOccurrence(1, (o2056,))],
    )
    from sessionpreplib.ptx.writer_primitives import replace_index_0002_chunk
    replace_index_0002_chunk(virgin_dec, rebuilt_v)
    assert validate_index_0002(bytes(virgin_dec)) is True

    virgin_ptxt = tmp_path / "Virgin.ptxt"
    virgin_ptxt.write_bytes(encrypt_ptx(bytes(virgin_dec), delta))
    wav = _write_wav(tmp_path / "Lead.wav", 48000, 1, frames=4800)
    proj_root = tmp_path / "out"
    proj_root.mkdir()
    staged = stage_session_audio_files(proj_root, "VirginRun", [wav], TargetAudioSpec(48000, 24))
    af = AudioFileWriteSpec(
        staged.staged_files[0].filename, 48000, 1, 24, 4800,
        staged.staged_files[0].file_id8, staged.staged_files[0].smpte_umid32, staged.staged_files[0].filetime_u64,
    )
    kik_trk = PtxSession.from_file(virgin_ptxt).track_by_name("KIK")
    assert kik_trk is not None
    # Create: 1 foldered track (using braced UUID!), 1 top-level track with start_sample=480, 1 empty Audio track, + clip gain!
    res = write_offline_ptx_session(OfflinePtxWriteRequest(
        ptxt_path=virgin_ptxt,
        output_ptx_path=staged.ptx_path,
        session_dir=staged.session_dir,
        sample_rate=48000,
        bit_depth=24,
        audio_files=(af,),
        tracks=(
            OfflineAudioTrackRequest("FolderLead", 1, kik_trk.uuid_braced, "KIK", 0, 4800, "Lead"),
            OfflineAudioTrackRequest("TopLevel", 1, None, None, 0, 4800, "LeadTop", start_sample=480),
            OfflineAudioTrackRequest("EmptyAudio", 2, None, None, None),
        ),
        clip_gains_db={"FolderLead": -2.5},
    ))
    dec_out, _ = decrypt_ptx(res.ptx_path.read_bytes())
    assert validate_index_0002(dec_out) is True  # proves 0x2587 sibling 0x2586->0x2056 shifted accurately AND synthesized chunks indexed!
    sess_out = PtxSession.from_file(res.ptx_path)
    assert [t.name for t in sess_out.tracks] == ["KIK", "FolderLead", "KIK-VCA", "VOX", "VOX-VCA", "TopLevel", "EmptyAudio"]
    assert sess_out.track_by_name("TopLevel").parent_folder_name == ""
    assert sess_out.track_by_name("TopLevel").clips[0].start_sample == 480
    assert sess_out.track_by_name("EmptyAudio").clips == []
    out_tags = [c.tag for c in iter_top_chunks(dec_out, strict=True)]
    assert 0x0F3D in out_tags and 0x2637 in out_tags and 0x262A in out_tags
    # Canonical ordering: 0x2637 and 0x262A must appear BEFORE 0x1054, 0x1015, 0x210C, 0x2107, 0x2624
    assert out_tags.index(0x2637) < out_tags.index(0x1054)
    assert out_tags.index(0x262A) < out_tags.index(0x1054)
    assert out_tags.index(0x210C) < out_tags.index(0x2107)

    # Idempotency: rebuild_index_0002(idx, dec, dec) must NOT fabricate entries for unindexed pre-existing chunks
    raw_idx_base = parse_index_0002(dec_base)
    assert raw_idx_base is not None
    stripped_entries = [e for e in raw_idx_base.entries if not (e.path and e.path[0].tag == 0x210C)]
    stripped_idx = PtxIndex0002(version=raw_idx_base.version, entries=stripped_entries)
    idem_idx = rebuild_index_0002(stripped_idx, dec_base, dec_base)
    assert not any(e.path and e.path[0].tag == 0x210C for e in idem_idx.entries)

    # 2. Hidden track preservation in mutate_2519_tracks (visible_order == 0 stays 0!)
    u1, u2, u3 = generate_ptx_uuid(), generate_ptx_uuid(), generate_ptx_uuid()
    r1 = build_track_summary_2519("Vis1", u1, 1, 1)
    r2 = build_track_summary_2519("Hid2", u2, 1, 0)  # hidden!
    r3 = build_track_summary_2519("Vis3", u3, 1, 2)
    e1 = build_track_entry_251a("Vis1", u1, 1, 1)
    e2 = build_track_entry_251a("Hid2", u2, 1, 0)
    e3 = build_track_entry_251a("Vis3", u3, 1, 2)
    c2519_hid = build_zchunk(
        0x2519,
        (b"\x00" * 10) + struct.pack("<II", 0, 3) + r1 + r2 + r3 + e1 + e2 + e3 + struct.pack("<I", 3) + e1 + e2 + e3,
        version=9,
    )
    mut_hid = mutate_2519_tracks(c2519_hid, [TrackSlotOp(old_index=0), TrackSlotOp(old_index=1), TrackSlotOp(old_index=2)])
    from sessionpreplib.ptx.writer_mutations import _parse_2519_sections
    _, rows_h, p1_h, _, _ = _parse_2519_sections(mut_hid)
    vis_orders = [struct.unpack_from("<H", r, len(r) - 3)[0] for r in rows_h]
    assert vis_orders == [1, 0, 2]

    # 3. Mono OutputBusSpec resolves cleanly and emits Pro Tools' canonical (2, 2) 0x4856/0x485E mirror
    mono_strip = build_audio_mixer_strip_261c(
        "MonoOut", generate_ptx_uuid(), 1, OutputBusSpec(4, generate_ptx_uuid(), "Bus 1", channel_count=1),
        track_ordinal_1based=1, id_base=100,
    )
    from sessionpreplib.ptx.codec import find_subchunks
    s485e = find_subchunks(mono_strip, 0x485E)[0]
    assert s485e[9:13] == b"\x02\x00\x02\x00"

    # 4. Strict validation guards: non-folder parent, unresolved output_bus_name, empty track_name, link_vca without vca, legacy 0x2519
    vca_trk = PtxSession.from_file(virgin_ptxt).track_by_name("KIK-VCA")
    assert vca_trk is not None
    with pytest.raises(PtxValidationError, match="non-folder"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=virgin_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(af,),
            tracks=(OfflineAudioTrackRequest("BadParent", 1, vca_trk.uuid_hex, None, 0, 4800, "Lead"),),
        ))
    with pytest.raises(PtxValidationError, match="Output bus 'NoSuchBus' not found"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=virgin_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(af,),
            tracks=(OfflineAudioTrackRequest("BadBus", 1, None, "NoSuchBus", 0, 4800, "Lead"),),
        ))
    with pytest.raises(PtxValidationError, match="requires a non-empty vca_track"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=virgin_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
            group_requests=(OfflineGroupRequest("G", "link_vca", (), vca_track=None),),
        ))
    # Legacy 0x2519 v7 and missing 0x1028 rejected cleanly with PtxUnsupportedTemplateError
    legacy_dec = bytearray(dec_base)
    c2519_ch = next(c for c in top if c.tag == 0x2519)
    struct.pack_into("<H", legacy_dec, c2519_ch.offset + 1, 7)
    legacy_ptxt = tmp_path / "Legacy.ptxt"
    legacy_ptxt.write_bytes(encrypt_ptx(bytes(legacy_dec), delta))
    with pytest.raises(PtxUnsupportedTemplateError, match="0x2519 version 7"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=legacy_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
        ))
    no1028_dec = bytearray(dec_base[:0x1F])
    for ch in top:
        if ch.tag != 0x1028:
            no1028_dec.extend(ch.raw)
    no1028_ptxt = tmp_path / "No1028.ptxt"
    no1028_ptxt.write_bytes(encrypt_ptx(bytes(no1028_dec), delta))
    with pytest.raises(PtxUnsupportedTemplateError, match="0x1028"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=no1028_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(), tracks=(),
        ))
    with pytest.raises(PtxValidationError, match="sample rate"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=virgin_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=22050, bit_depth=24, audio_files=(), tracks=(),
        ))
    with pytest.raises(PtxValidationError, match="color_index"):
        write_offline_ptx_session(OfflinePtxWriteRequest(
            ptxt_path=virgin_ptxt, output_ptx_path=tmp_path / "x.ptx", session_dir=staged.session_dir,
            sample_rate=48000, bit_depth=24, audio_files=(),
            tracks=(OfflineAudioTrackRequest("BadColor", 1, None, None, None, color_index=70000),),
        ))

    # 5. _verify_written_ptx ignores pre-existing baseline warnings but fails on newly introduced parser warnings
    from sessionpreplib.ptx import PtxCorruptFileError
    from sessionpreplib.ptx import writer as writer_mod
    from sessionpreplib.ptx.writer import _verify_written_ptx
    sess_ok = PtxSession.from_file(res.ptx_path)
    raw_ok = res.ptx_path.read_bytes()
    dec_ok, delta_ok = decrypt_ptx(raw_ok)

    orig_session_cls = writer_mod.PtxSession

    class _WarnSession(orig_session_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.warnings.append("Pre-existing tempo warning")

    monkeypatch.setattr(writer_mod, "PtxSession", _WarnSession)
    _verify_written_ptx(
        raw_ok,
        expected_track_count=len(sess_ok.tracks), created_names=[t.name for t in sess_ok.tracks],
        expected_audio_file_count=1, expected_clip_count=2, expected_playlist_count=len(sess_ok.playlists),
        baseline_warnings=("Pre-existing tempo warning",),
    )
    with pytest.raises(PtxCorruptFileError, match="Pre-existing tempo warning"):
        _verify_written_ptx(
            raw_ok,
            expected_track_count=len(sess_ok.tracks), created_names=[t.name for t in sess_ok.tracks],
            expected_audio_file_count=1, expected_clip_count=2, expected_playlist_count=len(sess_ok.playlists),
            baseline_warnings=(),
        )
    monkeypatch.setattr(writer_mod, "PtxSession", orig_session_cls)

    # Corrupting 0x0002 raises PtxCorruptFileError
    bad_idx_dec = bytearray(dec_ok)
    struct.pack_into("<I", bad_idx_dec, 0x1B, 0x1F)
    with pytest.raises(PtxCorruptFileError, match="0x0002"):
        _verify_written_ptx(
            encrypt_ptx(bytes(bad_idx_dec), delta_ok),
            expected_track_count=len(sess_ok.tracks), created_names=[],
            expected_audio_file_count=1, expected_clip_count=2, expected_playlist_count=len(sess_ok.playlists),
        )
    assert len(normalize_ptx_uuid16(kik_trk.uuid_braced)) == 16
