"""Offline Pro Tools session directory & ``Audio Files/`` media stager for pure ``.ptx`` creation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from ..models import TrackContext

log = logging.getLogger(__name__)

_VALID_PROTOOLS_RATES = frozenset({44100, 48000, 88200, 96000, 176400, 192000})

_BIT_DEPTH_TO_SUBTYPE: dict[int, str] = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}

_SUBTYPE_TO_BIT_DEPTH: dict[str, int] = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 32,
}

_SAFE_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

_WIN_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


@dataclass(frozen=True)
class TargetAudioSpec:
    """Resolved Pro Tools session audio format for offline media staging."""

    sample_rate: int
    bit_depth: int  # 16, 24, or 32 (float)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.bit_depth not in _BIT_DEPTH_TO_SUBTYPE:
            raise ValueError(f"Unsupported Pro Tools bit depth {self.bit_depth} (expected 16, 24, or 32)")

    @property
    def wav_subtype(self) -> str:
        return _BIT_DEPTH_TO_SUBTYPE[self.bit_depth]


@dataclass(frozen=True)
class StagedAudioFile:
    """Metadata for one WAV file staged into ``<session_dir>/Audio Files/``."""

    source_path: str
    dest_path: str
    filename: str
    relative_path: str  # POSIX relative path, e.g. "Audio Files/Kick.wav"
    sample_rate: int
    bit_depth: int
    channels: int
    length_samples: int
    converted: bool


@dataclass(frozen=True)
class StagedSessionDir:
    """Created Pro Tools session directory layout with staged audio files."""

    session_dir: Path
    audio_files_dir: Path
    ptx_path: Path
    spec: TargetAudioSpec
    staged_files: tuple[StagedAudioFile, ...]
    by_source_norm: dict[str, StagedAudioFile]

    def lookup(self, source_path: str | Path) -> StagedAudioFile | None:
        key = os.path.normcase(os.path.realpath(str(source_path)))
        return self.by_source_norm.get(key)


def resolve_target_audio_spec(
    tracks: Sequence[TrackContext],
    *,
    template_sample_rate: int | None = None,
    template_bit_depth: int | None = None,
) -> TargetAudioSpec:
    """Select the optimal Pro Tools sample rate and bit depth from *tracks* (with template fallback)."""
    rates = [t.samplerate for t in tracks if t.samplerate in _VALID_PROTOOLS_RATES]
    depths: list[int] = []
    for trk in tracks:
        bd = str(trk.bitdepth).upper()
        if "32" in bd or "64" in bd or "FLOAT" in bd or "DOUBLE" in bd:
            depths.append(32)
        elif "24" in bd:
            depths.append(24)
        elif "16" in bd:
            depths.append(16)
    fallback_sr = template_sample_rate if (template_sample_rate in _VALID_PROTOOLS_RATES) else 48000
    fallback_depth = template_bit_depth if (template_bit_depth in _BIT_DEPTH_TO_SUBTYPE) else 24
    sr = Counter(rates).most_common(1)[0][0] if rates else fallback_sr
    depth = Counter(depths).most_common(1)[0][0] if depths else fallback_depth
    return TargetAudioSpec(sample_rate=sr, bit_depth=depth)


def _is_windows_reserved_name(name: str) -> bool:
    return name.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES


def _sanitize_wav_stem(source_path: Path) -> str:
    stem = _SAFE_NAME_RE.sub("_", source_path.stem).strip(" .")
    if not stem:
        return "Audio"
    if _is_windows_reserved_name(stem):
        return f"{stem}_"
    return stem


def _validate_session_name(session_name: str) -> str:
    clean = session_name.strip()
    if (
        not clean
        or clean != clean.strip(" .")
        or clean.endswith(".")
        or clean in (".", "..")
        or _SAFE_NAME_RE.search(clean)
        or _is_windows_reserved_name(clean)
    ):
        raise ValueError(f"Invalid Pro Tools session name: {session_name!r}")
    return clean


def _unique_wav_filename(stem: str, used_lower: set[str]) -> str:
    cand = f"{stem}.wav"
    suffix = 2
    while cand.lower() in used_lower:
        cand = f"{stem}_{suffix}.wav"
        suffix += 1
    used_lower.add(cand.lower())
    return cand


def _needs_conversion(info: sf._SoundFileInfo, spec: TargetAudioSpec) -> bool:
    if str(info.format).upper() != "WAV":
        return True
    if int(info.samplerate) != spec.sample_rate:
        return True
    sub = str(info.subtype).upper()
    if _SUBTYPE_TO_BIT_DEPTH.get(sub) != spec.bit_depth:
        return True
    return spec.bit_depth == 32 and sub != "FLOAT"


def _verify_float_wav_finite(src: Path) -> None:
    with sf.SoundFile(src, "r") as handle:
        while True:
            block = handle.read(65536, dtype="float64", always_2d=False)
            if block.size == 0:
                break
            if not np.all(np.isfinite(block)):
                raise ValueError(f"Non-finite audio samples in {src.name}")


def _convert_and_write_wav(src: Path, dst: Path, spec: TargetAudioSpec) -> tuple[int, int]:
    samples, src_sr = sf.read(src, dtype="float64", always_2d=True)
    arr = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite audio samples in {src.name}")
    if int(src_sr) != spec.sample_rate:
        divisor = math.gcd(spec.sample_rate, int(src_sr))
        arr = resample_poly(arr, spec.sample_rate // divisor, int(src_sr) // divisor, axis=0)
    if spec.bit_depth in (16, 24):
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            log.warning("Resampled audio %s exceeded 0 dBFS (peak=%.4f); clipping to full scale.", src.name, peak)
            arr = np.clip(arr, -1.0, 1.0)
    out_data = arr[:, 0] if arr.shape[1] == 1 else arr
    sf.write(dst, out_data, spec.sample_rate, format="WAV", subtype=spec.wav_subtype)
    return int(arr.shape[1]), int(arr.shape[0])


def _stage_single_audio_file(
    src: Path, audio_dir: Path, spec: TargetAudioSpec, used_lower: set[str]
) -> StagedAudioFile:
    if not src.is_file():
        raise FileNotFoundError(f"Source audio file not found: {src}")
    info = sf.info(src)
    if info.frames <= 0:
        raise ValueError(f"Audio file is empty (0 frames): {src.name}")
    if info.channels < 1 or info.channels > 2:
        raise ValueError(f"Unsupported channel count {info.channels} in {src.name} (expected mono or stereo)")
    filename = _unique_wav_filename(_sanitize_wav_stem(src), used_lower)
    dst = audio_dir / filename
    if not _needs_conversion(info, spec):
        if str(info.subtype).upper() == "FLOAT":
            _verify_float_wav_finite(src)
        shutil.copyfile(src, dst)
        channels, length_samples, converted = int(info.channels), int(info.frames), False
    else:
        channels, length_samples = _convert_and_write_wav(src, dst, spec)
        converted = True
    return StagedAudioFile(
        source_path=str(src),
        dest_path=str(dst),
        filename=filename,
        relative_path=f"Audio Files/{filename}",
        sample_rate=spec.sample_rate,
        bit_depth=spec.bit_depth,
        channels=channels,
        length_samples=length_samples,
        converted=converted,
    )


def _on_rmtree_exc(func, path: str, _exc: BaseException) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except OSError:
        pass


def cleanup_staged_session_dir(session_dir: Path) -> None:
    """Remove a newly staged session directory on failure (never follows symlinks/junctions)."""
    is_link = session_dir.is_symlink() or (hasattr(session_dir, "is_junction") and session_dir.is_junction())
    if session_dir.exists() and not is_link:
        shutil.rmtree(session_dir, onexc=_on_rmtree_exc)


def stage_session_audio_files(
    project_dir: str | Path,
    session_name: str,
    source_paths: Sequence[str | Path],
    spec: TargetAudioSpec,
) -> StagedSessionDir:
    """Create ``<project_dir>/<session_name>/Audio Files/`` and stage deduplicated WAV assets.

    Atomically creates ``session_dir`` first (refusing pre-existing directories) and cleans up
    only a directory created by this call if staging fails.
    """
    clean_name = _validate_session_name(session_name)
    parent = Path(project_dir)
    if not parent.is_dir():
        raise FileNotFoundError(f"Pro Tools project directory does not exist: {parent}")

    session_dir = parent / clean_name
    session_dir.mkdir(parents=False, exist_ok=False)
    audio_dir = session_dir / "Audio Files"
    staged: list[StagedAudioFile] = []
    by_source_norm: dict[str, StagedAudioFile] = {}
    used_lower: set[str] = set()
    try:
        audio_dir.mkdir(parents=False, exist_ok=False)
        for raw_src in source_paths:
            src = Path(raw_src)
            norm_key = os.path.normcase(os.path.realpath(str(src)))
            if norm_key in by_source_norm:
                continue
            item = _stage_single_audio_file(src, audio_dir, spec, used_lower)
            staged.append(item)
            by_source_norm[norm_key] = item
    except BaseException:
        cleanup_staged_session_dir(session_dir)
        raise

    return StagedSessionDir(
        session_dir=session_dir,
        audio_files_dir=audio_dir,
        ptx_path=session_dir / f"{clean_name}.ptx",
        spec=spec,
        staged_files=tuple(staged),
        by_source_norm=by_source_norm,
    )
