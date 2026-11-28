"""Unit tests for offline Pro Tools .ptxt Fetch, backend dispatch, and offline Audio Files stager."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib.daw_processors.protools import ProToolsDawProcessor
from sessionpreplib.daw_processors.protools_ptx_media import (
    TargetAudioSpec,
    resolve_target_audio_spec,
    stage_session_audio_files,
)
from sessionpreplib.daw_processors.protools_ptx_preview import read_template_folders_from_ptxt
from sessionpreplib.models import SessionContext, TrackContext, TransferEntry
from tests.unit.test_protools_ptx_orchestration import _build_nested_folder_ptx


def _write_tone(path: Path, sr: int, channels: int, subtype: str = "PCM_24", fmt: str = "WAV") -> Path:
    n = sr // 10
    t = np.arange(n, dtype=np.float64) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    data = sig if channels == 1 else np.column_stack([sig, 0.5 * sig])
    sf.write(path, data, sr, format=fmt, subtype=subtype)
    return path


def test_read_template_folders_from_ptxt_and_offline_fetch_without_ptsl_or_temp_dir(tmp_path: Path, monkeypatch):
    ptxt = tmp_path / "Mix.ptxt"
    ptxt.write_bytes(_build_nested_folder_ptx(folder_has_input=True))

    folders = read_template_folders_from_ptxt(ptxt)
    by_name = {f["name"]: f for f in folders}
    assert set(by_name) == {"KIK", "Sub"}
    assert by_name["KIK"]["folder_type"] == "routing" and by_name["KIK"]["parent_id"] is None
    assert by_name["Sub"]["folder_type"] == "basic" and by_name["Sub"]["parent_id"] == by_name["KIK"]["id"]

    # ProToolsDawProcessor.fetch() reads local valid .ptxt directly offline (even without temp_dir or py-ptsl)
    proc = ProToolsDawProcessor(instance_index=0, instance_group="Mix", instance_name="Mix")
    proc.configure({"protools_temp_dir": ""})
    monkeypatch.setattr(proc, "_template_file_and_mtime", lambda: (ptxt, ptxt.stat().st_mtime))
    monkeypatch.setattr(proc, "_load_template_cache", lambda: (tmp_path / "cache.json", {}))
    monkeypatch.setattr(proc, "_fetch_folders_live", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("PTSL should not be called")))

    sess = SessionContext(tracks=[], output_tracks=[], config={})
    proc.fetch(sess, ignore_cache=True)
    fetched_names = [f["name"] for f in sess.daw_state["protools_0"]["folders"]]
    assert fetched_names == ["KIK", "Sub"]
    assert (tmp_path / "cache.json").is_file()


def test_protools_offline_backend_dispatch_does_not_require_ptsl(tmp_path: Path):
    wav = _write_tone(tmp_path / "Kick.wav", 48000, 1)
    tc = TrackContext("Kick.wav", str(wav), None, 48000, 1, 4800, "24", "PCM_24", 0.1)
    proc = ProToolsDawProcessor(instance_index=0, instance_group="Mix", instance_name="Studio")
    proc.configure({"protools_transfer_backend": "offline_ptx", "protools_project_dir": str(tmp_path)})
    sess = SessionContext(
        tracks=[tc], output_tracks=[tc], config={}, project_name="OfflineSong",
        transfer_manifest=[TransferEntry("Kick.wav", "Kick.wav", "Kick")],
        daw_state={proc.id: {"folders": [{"id": "f1", "name": "KIK"}], "assignments": {"Kick.wav": "f1"}}},
    )
    results = proc.transfer(sess, "")
    assert len(results) == 1 and results[0].success is False
    assert "template file not found or invalid" in (results[0].error or "").lower()


def test_resolve_target_audio_spec_and_stage_copy_convert_dedup_collisions(tmp_path: Path):
    tracks = [
        TrackContext("a.wav", "", None, 48000, 1, 4800, "24", "PCM_24", 0.1),
        TrackContext("b.wav", "", None, 48000, 2, 4800, "24", "PCM_24", 0.1),
        TrackContext("c.wav", "", None, 44100, 1, 4410, "16", "PCM_16", 0.1),
    ]
    spec = resolve_target_audio_spec(tracks)
    assert spec == TargetAudioSpec(sample_rate=48000, bit_depth=24)
    assert resolve_target_audio_spec([], template_sample_rate=96000, template_bit_depth=32) == TargetAudioSpec(96000, 32)

    src_dir_a = tmp_path / "src_a"
    src_dir_b = tmp_path / "src_b"
    src_dir_a.mkdir()
    src_dir_b.mkdir()
    wav_exact = _write_tone(src_dir_a / "Kick.wav", 48000, 1, subtype="PCM_24")
    wav_conv = _write_tone(src_dir_b / "Kick.aiff", 44100, 2, subtype="PCM_16", fmt="AIFF")

    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    staged = stage_session_audio_files(proj_root, "MySong", [wav_exact, wav_exact, wav_conv], spec)

    assert staged.session_dir == proj_root / "MySong"
    assert staged.ptx_path == proj_root / "MySong" / "MySong.ptx"
    assert len(staged.staged_files) == 2  # Deduplicated wav_exact!
    f0, f1 = staged.staged_files
    assert f0.filename == "Kick.wav" and f0.converted is False and f0.channels == 1 and f0.length_samples == 4800
    assert f1.filename == "Kick_2.wav" and f1.converted is True and f1.channels == 2 and f1.sample_rate == 48000
    info1 = sf.info(f1.dest_path)
    assert info1.format == "WAV" and info1.subtype == "PCM_24" and info1.samplerate == 48000 and info1.channels == 2


def test_stage_session_audio_files_guards_and_rollback_on_failure(tmp_path: Path):
    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    spec = TargetAudioSpec(48000, 24)
    good = _write_tone(tmp_path / "Good.wav", 48000, 1)
    quad = tmp_path / "Quad.wav"
    sf.write(quad, np.zeros((480, 4), dtype=np.float32), 48000, subtype="PCM_24")

    with pytest.raises(ValueError):
        stage_session_audio_files(proj_root, "Bad/Name", [good], spec)
    with pytest.raises(FileNotFoundError):
        stage_session_audio_files(tmp_path / "missing_root", "Song", [good], spec)

    # Failure on second file rolls back and removes newly created session dir completely
    with pytest.raises(ValueError, match="Unsupported channel count"):
        stage_session_audio_files(proj_root, "RollbackSong", [good, quad], spec)
    assert not (proj_root / "RollbackSong").exists()

    # Refuses to overwrite existing session directory
    (proj_root / "Existing").mkdir()
    with pytest.raises(FileExistsError):
        stage_session_audio_files(proj_root, "Existing", [good], spec)


def test_media_stager_edge_cases_reserved_names_pullup_float_nan_empty_and_readonly(tmp_path: Path):
    import os
    import stat

    with pytest.raises(ValueError):
        TargetAudioSpec(0, 24)
    with pytest.raises(ValueError):
        TargetAudioSpec(22050, 24)  # non-Pro-Tools sample rate rejected upfront
    with pytest.raises(ValueError):
        TargetAudioSpec(48000, 8)

    # Bit depth uses max(depths) so 24-bit tracks are never silently downcast to 16-bit!
    mixed_depth = [
        TrackContext("a.wav", "", None, 48000, 1, 4800, "16", "PCM_16", 0.1),
        TrackContext("b.wav", "", None, 48000, 1, 4800, "16", "PCM_16", 0.1),
        TrackContext("c.wav", "", None, 48000, 1, 4800, "24", "PCM_24", 0.1),
    ]
    assert resolve_target_audio_spec(mixed_depth) == TargetAudioSpec(48000, 24)
    # Sample-rate tie prefers template rate deterministically, else highest rate
    tied_rates = [
        TrackContext("a.wav", "", None, 44100, 1, 4410, "24", "PCM_24", 0.1),
        TrackContext("b.wav", "", None, 48000, 1, 4800, "24", "PCM_24", 0.1),
    ]
    assert resolve_target_audio_spec(tied_rates, template_sample_rate=44100) == TargetAudioSpec(44100, 24)
    assert resolve_target_audio_spec(tied_rates) == TargetAudioSpec(48000, 24)

    # Non-Pro-Tools sample rate falls back; DOUBLE maps to 32-bit
    odd_tracks = [TrackContext("x.wav", "", None, 22050, 1, 2205, "DOUBLE", "DOUBLE", 0.1)]
    assert resolve_target_audio_spec(odd_tracks, template_sample_rate=96000) == TargetAudioSpec(96000, 32)

    proj_root = tmp_path / "projects"
    proj_root.mkdir()

    # Reserved Windows names & dotted reserved names (AUX.bus.wav -> AUX_.bus.wav)
    aux_wav = _write_tone(tmp_path / "AUX.wav", 48000, 1)
    aux_dot_wav = _write_tone(tmp_path / "AUX.bus.wav", 48000, 1)
    for bad_name in ("CON", "AUX", "MySong.", "..."):
        with pytest.raises(ValueError):
            stage_session_audio_files(proj_root, bad_name, [aux_wav], TargetAudioSpec(48000, 24))

    # Pull-up 48048 Hz -> 48000 Hz actually resamples (exact 1000/1001, not 1:1!)
    pullup = _write_tone(tmp_path / "PullUp.wav", 48048, 1)
    pcm32 = _write_tone(tmp_path / "Pcm32.wav", 48000, 1, subtype="PCM_32")
    float32 = _write_tone(tmp_path / "Float32.wav", 48000, 1, subtype="FLOAT")

    staged_24 = stage_session_audio_files(proj_root, "PullUpSong", [aux_wav, aux_dot_wav, pullup], TargetAudioSpec(48000, 24))
    assert staged_24.lookup(aux_wav).filename == "AUX_.wav"
    assert staged_24.lookup(aux_dot_wav).filename == "AUX_.bus.wav"
    assert staged_24.lookup(pullup).converted is True and staged_24.lookup(pullup).length_samples == 4800

    # Truncated RIFF chunk must raise ValueError instead of silently dropping data
    from sessionpreplib.daw_processors.protools_ptx_media import _scan_wav_riff_chunks
    trunc_wav = tmp_path / "Trunc.wav"
    trunc_wav.write_bytes(b"RIFF" + (100).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") + b"\x00" * 16 + b"data" + (9999).to_bytes(4, "little") + b"\x00" * 4)
    with pytest.raises(ValueError, match="Truncated RIFF chunk"):
        _scan_wav_riff_chunks(trunc_wav)

    staged_32 = stage_session_audio_files(proj_root, "FloatSong", [pcm32, float32], TargetAudioSpec(48000, 32))
    assert staged_32.lookup(pcm32).converted is True and staged_32.lookup(float32).converted is False

    # Empty WAV & non-finite float WAV rejected (even on fast-copy float32 path!)
    empty_wav = tmp_path / "Empty.wav"
    sf.write(empty_wav, np.zeros((0, 1), dtype=np.float32), 48000, subtype="PCM_24")
    with pytest.raises(ValueError, match="empty"):
        stage_session_audio_files(proj_root, "EmptySong", [empty_wav], TargetAudioSpec(48000, 24))

    nan_wav = tmp_path / "NaN.wav"
    sf.write(nan_wav, np.array([0.0, float("nan"), 0.1], dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Non-finite"):
        stage_session_audio_files(proj_root, "NaNSong", [nan_wav], TargetAudioSpec(48000, 32))

    # Read-only source WAV rolls back cleanly on Windows when a later file fails
    ro_wav = _write_tone(tmp_path / "ReadOnly.wav", 48000, 1)
    os.chmod(ro_wav, stat.S_IREAD)
    try:
        with pytest.raises(ValueError):
            stage_session_audio_files(proj_root, "RoRollback", [ro_wav, empty_wav], TargetAudioSpec(48000, 24))
        assert not (proj_root / "RoRollback").exists()
    finally:
        os.chmod(ro_wav, stat.S_IWRITE | stat.S_IREAD)


def test_offline_fetch_fallback_and_unknown_backend_rejection(tmp_path: Path, monkeypatch):
    # Corrupt .ptxt with valid magic falls back cleanly to live PTSL fetch
    bad_ptxt = tmp_path / "Corrupt.ptxt"
    bad_ptxt.write_bytes(b"\x030010111100101011" + b"\x00" * 40)
    proc = ProToolsDawProcessor(instance_index=0, instance_group="Mix", instance_name="Corrupt")
    proc.configure({"protools_temp_dir": str(tmp_path), "protools_transfer_backend": "bogus_backend"})
    monkeypatch.setattr(proc, "_template_file_and_mtime", lambda: (bad_ptxt, bad_ptxt.stat().st_mtime))
    monkeypatch.setattr(proc, "_load_template_cache", lambda: (tmp_path / "cache.json", {}))
    monkeypatch.setattr(
        proc, "_fetch_folders_live",
        lambda _cb: [{"id": "live-1", "name": "LiveFolder", "folder_type": "routing", "index": 1, "parent_id": None}],
    )
    sess = SessionContext(tracks=[], output_tracks=[], config={}, project_name="Test")
    proc.fetch(sess, ignore_cache=True)
    assert sess.daw_state["protools_0"]["folders"][0]["name"] == "LiveFolder"

    res = proc.transfer(sess, "")
    assert len(res) == 1 and res[0].success is False and "Unknown Pro Tools transfer backend" in (res[0].error or "")


def test_streaming_resample_parity_and_cleanup_error_isolation(tmp_path: Path, monkeypatch):
    from scipy.signal import resample_poly

    from sessionpreplib.daw_processors import protools_ptx_media as media_mod
    from sessionpreplib.daw_processors.protools_ptx_media import (
        _stream_convert_same_rate,
        _stream_resample_wav,
        cleanup_staged_session_dir,
    )

    # Multi-block overlap-save streaming SRC matches one-shot resample_poly closely in steady-state
    sr_in, sr_out = 44100, 48000
    n_in = 8000
    t = np.arange(n_in, dtype=np.float64) / sr_in
    sig = np.column_stack([0.4 * np.sin(2.0 * np.pi * 440.0 * t), 0.3 * np.cos(2.0 * np.pi * 880.0 * t)])
    src = tmp_path / "Stereo44k.wav"
    dst_stream = tmp_path / "Stereo48k_stream.wav"
    sf.write(src, sig, sr_in, subtype="FLOAT")

    # Force small block size so overlap-save iterates across multiple polyphase-aligned chunks
    monkeypatch.setattr(media_mod, "_SRC_BLOCK_FRAMES", 1500)
    monkeypatch.setattr(media_mod, "_SRC_OVERLAP_FRAMES", 512)
    ch_out, frames_out = _stream_resample_wav(src, dst_stream, TargetAudioSpec(sr_out, 32), sr_in, 2)
    ref = resample_poly(sig, 160, 147, axis=0)
    got, got_sr = sf.read(dst_stream, dtype="float64", always_2d=True)
    assert got_sr == sr_out and ch_out == 2 and frames_out == got.shape[0]
    assert frames_out == ref.shape[0]
    # Exact polyphase alignment -> entire signal matches one-shot resample_poly to < 1e-4!
    assert np.max(np.abs(got - ref)) < 1e-4

    # Same-rate streaming conversion preserves audio and rejects non-finite samples
    dst_same = tmp_path / "Stereo44k_pcm24.wav"
    c_same, f_same = _stream_convert_same_rate(src, dst_same, TargetAudioSpec(sr_in, 24), 2)
    assert f_same == n_in and c_same == 2
    same_data, _ = sf.read(dst_same, dtype="float64", always_2d=True)
    assert np.max(np.abs(same_data - sig)) < 1e-5

    # Edge cases: tiny file (3 frames), downsampling (192k->44.1k with down > block), 24-bit clipping, and NaN rejection
    tiny = tmp_path / "Tiny.wav"
    tiny_sig = np.array([[0.1], [0.2], [-0.1]], dtype=np.float64)
    sf.write(tiny, tiny_sig, 44100, subtype="FLOAT")
    c_tiny, f_tiny = _stream_resample_wav(tiny, tmp_path / "Tiny48.wav", TargetAudioSpec(48000, 32), 44100, 1)
    assert c_tiny == 1 and f_tiny == resample_poly(tiny_sig, 160, 147, axis=0).shape[0]

    # Downsampling 192k -> 44.1k (up=147, down=640) with _SRC_BLOCK_FRAMES < down!
    monkeypatch.setattr(media_mod, "_SRC_BLOCK_FRAMES", 300)
    monkeypatch.setattr(media_mod, "_SRC_OVERLAP_FRAMES", 640)
    t_hi = np.arange(3000, dtype=np.float64) / 192000.0
    hi_sig = (1.3 * np.sin(2.0 * np.pi * 1000.0 * t_hi)).reshape(-1, 1)
    hi_src = tmp_path / "Hi192.wav"
    hi_dst = tmp_path / "Lo44.wav"
    sf.write(hi_src, hi_sig, 192000, subtype="FLOAT")
    c_hi, f_hi = _stream_resample_wav(hi_src, hi_dst, TargetAudioSpec(44100, 24), 192000, 1)
    ref_hi = resample_poly(hi_sig, 147, 640, axis=0)
    got_hi, _ = sf.read(hi_dst, dtype="float64", always_2d=True)
    assert c_hi == 1 and f_hi == ref_hi.shape[0] and float(np.max(np.abs(got_hi))) <= 1.0 + 1e-6

    # NaN in streaming resampler raises ValueError
    nan_src = tmp_path / "NanStream.wav"
    sf.write(nan_src, np.array([[0.0], [float("nan")], [0.1]], dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="Non-finite"):
        _stream_resample_wav(nan_src, tmp_path / "NanOut.wav", TargetAudioSpec(48000, 24), 44100, 1)

    # Tie-break without template_sample_rate strictly picks highest rate (96k over 48k), and space-before-dot reserved name
    t48 = TrackContext("a.wav", "a.wav", None, 48000, 1, 100, "24", "PCM_24", 0.01)
    t96 = TrackContext("b.wav", "b.wav", None, 96000, 1, 100, "24", "PCM_24", 0.01)
    assert resolve_target_audio_spec([t48, t96]).sample_rate == 96000
    assert media_mod._is_windows_reserved_name("AUX .bus") is True

    # cleanup_staged_session_dir swallows OSError so rollback never masks primary errors
    locked_dir = tmp_path / "LockedSession"
    locked_dir.mkdir()
    monkeypatch.setattr(media_mod.shutil, "rmtree", lambda *_a, **_kw: (_ for _ in ()).throw(PermissionError("locked")))
    cleanup_staged_session_dir(locked_dir)  # must not raise
