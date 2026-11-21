"""Launch the source GUI with its real widgets offscreen and require a readiness receipt."""
import os
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest

from sessionpreplib import __version__


def test_window_starts_loads_icon_and_exits(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("sounddevice")
    launcher = Path(__file__).resolve().parents[2] / "sessionprep-gui.py"
    audio_file = tmp_path / "mono.wav"
    with wave.open(str(audio_file), "wb") as source:
        source.setparams((1, 2, 16000, 3, "NONE", "not compressed"))
        source.writeframes(struct.pack("<3h", -16384, 0, 8192))
    receipt = tmp_path / "ready.json"
    env = {**os.environ, "APPDATA": str(tmp_path), "HOME": str(tmp_path),
           "XDG_CONFIG_HOME": str(tmp_path), "QT_QPA_PLATFORM": "offscreen"}
    subprocess.run([sys.executable, launcher, "--gui-smoke-test", audio_file, receipt],
                   check=True, timeout=60, env=env, cwd=tmp_path)
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "version": __version__, "rate": 16000, "frames": 3,
        "pcm_sha256": hashlib.sha256(struct.pack("<3d", -0.5, 0, 0.25)).hexdigest(),
    }
