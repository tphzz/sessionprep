"""Exercise the installed interpreter and CLI boundary with real PCM and reports."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf


CLI = Path(__file__).resolve().parents[2] / "sessionprep.py"


def run_cli(directory, *options):
    return subprocess.run([sys.executable, str(CLI), str(directory), *options], cwd=directory.parent,
                          env={**os.environ, "PYTHONUTF8": "1", "SP_LOG_LEVEL": "NONE"}, encoding="utf8", capture_output=True,
                          check=False, timeout=45)


def digests(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_partial_job_exit_reports_and_delivered_signal(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    wav = source / "Kick.wav"
    broken = source / "Bad.wav"
    samples = np.tile([.25, -.125, 0, 0], 1024)
    sf.write(wav, samples, 8000, subtype="PCM_16")
    broken.write_bytes(b"truncated")
    originals = digests([wav, broken])
    dry = run_cli(source, "--force_transient", "kick")
    assert dry.returncode == 1 and "Bad.wav" in dry.stdout and "File errors" in dry.stdout
    assert "Traceback" not in dry.stderr and digests(source.iterdir()) == originals

    target = tmp_path / "processed"
    execute = run_cli(source, "--force_transient", "kick", "--execute", "--output_folder", str(target))
    assert execute.returncode == 1 and "Bad.wav" in execute.stdout and "Traceback" not in execute.stderr
    assert digests(source.iterdir()) == originals
    assert {p.name for p in target.iterdir()} == {"Kick.wav", "sessionprep.json", "sessionprep.txt"}
    pcm, rate = sf.read(target / wav.name)
    assert rate == 8000
    expected_gain = (10 ** (-6 / 20)) / .25
    np.testing.assert_allclose(pcm, samples * expected_gain, atol=1 / 32768, rtol=0)
    data = json.loads((target / "sessionprep.json").read_text(encoding="utf8"))
    outcomes = {t["filename"]: t["processing"] for t in data["tracks"]}
    assert outcomes["Kick.wav"]["status"] == "written"
    assert outcomes["Kick.wav"]["gain_applied_db"] == pytest.approx(20 * np.log10(expected_gain), abs=.005)
    assert outcomes["Bad.wav"]["status"] == "failed" and outcomes["Bad.wav"]["error"]
    assert outcomes["Bad.wav"]["gain_applied_db"] is None
    report = (target / "sessionprep.txt").read_text(encoding="utf8")
    assert "Bad.wav" in report.split("FILES WITH ERRORS")[1] and "Kick.wav" in report


def test_invalid_config_and_destination_leave_intact_input(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    wav = source / "Tone.wav"
    sf.write(wav, [.25, -.25], 8000)
    original = digests([wav])
    for value in ("6", "nan"):
        invalid = run_cli(source, "--execute", "--target_peak", value)
        assert invalid.returncode != 0 and "target_peak" in (invalid.stderr + invalid.stdout)
        assert "Traceback" not in invalid.stderr and digests(source.iterdir()) == original
    blocked = tmp_path / "occupied"
    blocked.write_bytes(b"existing content")
    failure = run_cli(source, "--execute", "--output_folder", str(blocked))
    assert failure.returncode != 0 and "occupied" in (failure.stdout + failure.stderr)
    assert "Traceback" not in failure.stderr and blocked.read_bytes() == b"existing content"
    assert digests(source.iterdir()) == original


@pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
def test_redirected_legacy_encoding_can_print_diagnostics_and_unicode_filenames(tmp_path, errors):
    source = tmp_path / "inputs"
    source.mkdir()
    wav = source / "Bass_低.wav"
    sf.write(wav, [.25, -.25] * 128, 8000, subtype="PCM_16")
    result = subprocess.run(
        [sys.executable, str(CLI), str(source), "--execute"], cwd=tmp_path,
        env={**os.environ, "PYTHONIOENCODING": f"cp1252:{errors}", "SP_LOG_LEVEL": "NONE"},
        encoding="cp1252", capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "PROBLEMS" in result.stdout and "Report saved to" in result.stdout
    output = source / "processed"
    assert (output / wav.name).is_file()
    tracks = json.loads((output / "sessionprep.json").read_text(encoding="utf-8"))["tracks"]
    assert [(track["filename"], track["processing"]["status"]) for track in tracks] == [(wav.name, "written")]
