"""Audio dependency distribution and native I/O regression coverage."""
from pathlib import Path, PurePosixPath
import tomllib
from urllib.parse import urlsplit

import numpy as np
from packaging.tags import mac_platforms
from packaging.utils import parse_wheel_filename
import pytest
import soundfile as sf

from sessionpreplib.audio import load_track, write_track


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target,compatible_platforms", [
    ("win-x64", {"win_amd64"}),
    ("win-arm64", {"win_arm64"}),
    ("linux-x64", {f"manylinux_2_{minor}_x86_64" for minor in range(17, 36)}),
    ("linux-arm64", {f"manylinux_2_{minor}_aarch64" for minor in range(17, 40)}),
    ("macos-x64", set(mac_platforms((13, 0), "x86_64"))),
    ("macos-arm64", set(mac_platforms((13, 0), "arm64"))),
])
def test_locked_soundfile_provides_bundled_native_audio_for_each_target(target, compatible_platforms):
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    soundfile = next(package for package in lock["package"] if package["name"] == "soundfile")
    platforms = {tag.platform for wheel in soundfile["wheels"]
                 for tag in parse_wheel_filename(PurePosixPath(urlsplit(wheel["url"]).path).name)[3]}
    assert platforms & compatible_platforms, (
        f"Locked SoundFile {soundfile['version']} has no bundled libsndfile wheel compatible with {target}; "
        "a pure-Python fallback requires a separately provisioned native library"
    )


@pytest.mark.parametrize("extension,subtype,channels", [
    ("wav", "PCM_16", 1), ("wav", "PCM_24", 2), ("wav", "FLOAT", 2),
    ("aiff", "PCM_24", 2), ("flac", "PCM_16", 1),
])
def test_native_audio_load_and_processed_write_preserve_format(tmp_path, extension, subtype, channels):
    # Binary fractions are exact in each chosen PCM/float format.
    samples = np.tile(np.array([0.0, 0.25, -0.5, 0.125]), 256)
    audio = samples if channels == 1 else np.column_stack([samples, -samples])
    source = tmp_path / f"source.{extension}"
    sf.write(source, audio, 48_000, subtype=subtype)

    track = load_track(str(source))
    assert track.channels == channels
    assert track.samplerate == 48_000
    assert track.total_samples == 1024
    assert track.subtype == subtype
    assert track.duration_sec == pytest.approx(1024 / 48_000)
    np.testing.assert_array_equal(track.audio_data, audio)

    track.audio_data *= 0.5
    destination = tmp_path / "processed.wav"
    write_track(track, str(destination))
    prepared = load_track(str(destination))
    assert prepared.channels == channels
    assert prepared.samplerate == 48_000
    assert prepared.subtype == subtype
    np.testing.assert_array_equal(prepared.audio_data, audio * 0.5)
