"""Hand-authored containers keep chunk framing independent of the writer."""

import os
import struct

import numpy as np
import pytest
import soundfile as sf

from sessionpreplib import chunks
from sessionpreplib.audio import discover_audio_files
from sessionpreplib.pipeline import load_session


def container_bytes(form, records):
    wave = form == b"WAVE"
    endian = "<" if wave else ">"
    body = bytearray(form)
    for name, data in records:
        body.extend(name + struct.pack(endian + "I", len(data)) + data)
        if len(data) % 2:
            body.append(0)
    return (b"RIFF" if wave else b"FORM") + struct.pack(endian + "I", len(body)) + body


def pcm_records(form):
    if form == b"WAVE":
        header = struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)
        audio = [(b"fmt ", header), (b"data", struct.pack("<3h", 8192, -4096, 0))]
    else:
        # 80-bit extended float encoding of the 44100 Hz sample rate.
        header = struct.pack(">hIh", 1, 3, 16) + bytes.fromhex("400eac44000000000000")
        audio = [(b"COMM", header), (b"SSND", bytes(8) + struct.pack(">3h", 8192, -4096, 0))]
    return [audio[0], (b"ZZZZ", b"abc"), (b"NOTE", b"abcd"), audio[1]]


@pytest.mark.parametrize(("form", "suffix"), [(b"WAVE", ".wav"), (b"AIFF", ".aif")])
def test_chunk_edit_preserves_sample_payload_and_order(tmp_path, form, suffix):
    source = tmp_path / ("take" + suffix)
    dest = tmp_path / ("edit" + suffix)
    records = pcm_records(form)
    original = container_bytes(form, records)
    source.write_bytes(original)

    assert chunks.chunk_ids(source) == [key.decode() for key, _ in records]
    encoding, data = chunks.read_chunks(source)
    assert encoding == form.decode()
    assert [(c.id.encode(), c.data, c.size) for c in data] == [
        (key, payload, len(payload)) for key, payload in records]
    chunks.remove_chunks(source, dest, {"ZZZZ"})

    assert source.read_bytes() == original
    assert dest.read_bytes() == container_bytes(form, [item for item in records if item[0] != b"ZZZZ"])
    pcm, sr = sf.read(dest)
    assert sr == 44100
    np.testing.assert_array_equal(pcm, [0.25, -0.125, 0])


@pytest.mark.parametrize(("fingerprint", "name"), [(None, "Tape desk"), (b"DGDA", "Pro Tools")])
def test_origin_uses_fingerprint_then_bext_originator(tmp_path, fingerprint, name):
    path = tmp_path / "origin.wav"
    records = pcm_records(b"WAVE")
    records.append((b"bext", bytes(256) + b"Tape desk".ljust(32, b"\0") + bytes(314)))
    if fingerprint:
        records.append((fingerprint, b""))
    path.write_bytes(container_bytes(b"WAVE", records))
    assert chunks.detect_origin(chunks.chunk_ids(path), path) == name


@pytest.mark.parametrize("malformed", [
    b"RIFF",
    b"RIFF" + struct.pack("<I", 40) + b"WAVEdata" + struct.pack("<I", 28) + b"ab",
    b"RIFF" + struct.pack("<I", 12) + b"WAVEJUNK" + struct.pack("<I", 8) + b"abcdefgh",
    b"FORM" + struct.pack(">I", 6) + b"AIFFSS",
])
def test_truncated_or_out_of_bounds_chunk_cannot_be_rewritten_as_a_valid_edit(tmp_path, malformed):
    src, dst = tmp_path / "bad.wav", tmp_path / "existing.wav"
    src.write_bytes(malformed)
    good = container_bytes(b"WAVE", pcm_records(b"WAVE"))
    dst.write_bytes(good)
    with pytest.raises(ValueError):
        chunks.remove_chunks(src, dst, {"JUNK"})
    assert dst.read_bytes() == good


def test_discovery_and_session_keep_the_good_file_alongside_corrupt_inputs(tmp_path):
    (tmp_path / "good.WAV").write_bytes(container_bytes(b"WAVE", pcm_records(b"WAVE")))
    (tmp_path / "broken.aiff").write_bytes(b"FORM")
    (tmp_path / "other.flac").write_bytes(bytes(10))
    assert set(discover_audio_files(str(tmp_path))) == {"good.WAV", "broken.aiff"}
    tracks = {t.filename: t for t in load_session(str(tmp_path), {}).tracks}
    assert tracks["broken.aiff"].status != "OK"
    assert tracks["good.WAV"].status == "OK"
    np.testing.assert_array_equal(tracks["good.WAV"].audio_data, [0.25, -0.125, 0])


def test_in_place_chunk_edit_can_be_retried_after_failed_publication(tmp_path, monkeypatch):
    path = tmp_path / "edit.wav"
    records = pcm_records(b"WAVE")
    original = container_bytes(b"WAVE", records)
    path.write_bytes(original)
    with monkeypatch.context() as patch:
        def fail_publish(*args):
            raise PermissionError("locked")

        patch.setattr(os, "replace", fail_publish)
        with pytest.raises(PermissionError, match="locked"):
            chunks.remove_chunks(path, path, {"ZZZZ"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    chunks.remove_chunks(path, path, {"ZZZZ"})
    assert path.read_bytes() == container_bytes(b"WAVE", [c for c in records if c[0] != b"ZZZZ"])


def test_metadata_scan_seeks_over_pcm_and_ancillary_data(tmp_path, monkeypatch):
    path = tmp_path / "long.wav"
    records = [(b"data", bytes(65536)), (b"iXML", bytes(8193)), (b"INFO", b"end")]
    path.write_bytes(container_bytes(b"WAVE", records))
    with path.open("rb") as stream:
        read = stream.read

        def read_header(size=-1):
            assert 0 <= size <= 12, f"metadata inspection requested {size} bytes"
            return read(size)

        monkeypatch.setattr(stream, "read", read_header)
        monkeypatch.setattr(chunks, "open", lambda *args: stream, raising=False)
        assert chunks.chunk_sizes(path) == [("data", 65536), ("iXML", 8193), ("INFO", 3)]


def test_file_report_shows_sizes_and_origin_without_reading_samples(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from sessionprepgui.detail.report import prepare_track_report

    path = tmp_path / "long.wav"
    records = [pcm_records(b"WAVE")[0], (b"data", bytes(65536)), (b"iXML", bytes(8193)), (b"DGDA", b"x")]
    path.write_bytes(container_bytes(b"WAVE", records))
    track, = load_session(str(tmp_path), {}).tracks
    with path.open("rb") as stream:
        read = stream.read

        def read_header(size=-1):
            assert 0 <= size <= 32, f"File requested {size} bytes when reading chunk sizes"
            return read(size)

        monkeypatch.setattr(stream, "read", read_header)
        monkeypatch.setattr(chunks, "open", lambda *args: stream, raising=False)
        header = prepare_track_report(track)["header"]
    assert "iXML (8.0 KB)" in header and "DGDA (1 B)" in header
    assert "Origin: Pro Tools" in header
    assert "fmt (" not in header and "data (" not in header
