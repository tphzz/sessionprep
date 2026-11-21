"""Run the real CLI and Qt entry points with disposable audio and preferences."""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import struct
import subprocess
import sys
import tempfile
import wave


ROOT = Path(__file__).resolve().parents[2]


def run(command, *, cwd, env, timeout=90):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def write_fixture(directory):
    rate = 8000
    samples = tuple(round(2048 * math.sin(2 * math.pi * i / 8)) for i in range(rate))
    with wave.open(str(directory / "tone.wav"), "wb") as source:
        source.setparams((1, 2, rate, len(samples), "NONE", "not compressed"))
        source.writeframes(struct.pack(f"<{rate}h", *samples))
    return rate, samples


def check_cli(command, directory, env, version):
    if run([*command, "--version"], cwd=directory, env=env) != f"sessionprep {version}":
        raise ValueError("CLI version differs from the build")
    rate, samples = write_fixture(directory)
    run([*command, str(directory), "-x", "--force_transient", "tone.wav"], cwd=directory, env=env)
    output = directory / "processed"
    with wave.open(str(output / "tone.wav")) as rendered:
        if (rendered.getnchannels(), rendered.getframerate(), rendered.getsampwidth(),
                rendered.getnframes()) != (1, rate, 2, len(samples)):
            raise ValueError("Unexpected prepared audio format or duration")
        actual = struct.unpack(f"<{rate}h", rendered.readframes(len(samples)))
    factor = 32768 * 10 ** (-6 / 20) / 2048
    if any(abs(code - original * factor) > 1.01 for code, original in zip(actual, samples)):
        raise ValueError("Prepared PCM differs from the -6 dBFS peak target")
    tracks = json.loads((output / "sessionprep.json").read_text(encoding="utf-8"))["tracks"]
    if len(tracks) != 1 or tracks[0]["filename"] != "tone.wav":
        raise ValueError("Missing prepared track in JSON report")
    receipt = tracks[0]["processing"]
    gain = 20 * math.log10(factor)
    if (receipt["status"] != "written" or abs(receipt["gain_applied_db"] - gain) > 0.005
            or abs(receipt["fader_db"] + gain) > 0.005):
        raise ValueError(f"CLI receipt disagrees with decoded PCM: {receipt}")
    print(f"CLI {version}: decoded and verified {len(samples)} PCM16 frames")


def check_gui(command, directory, env, version):
    rate, samples = write_fixture(directory)
    receipt = directory / "window-ready.json"
    run([*command, "--gui-smoke-test", directory / "tone.wav", receipt], cwd=directory, env=env, timeout=300)
    pcm = struct.pack(f"<{len(samples)}d", *(sample / 32768 for sample in samples))
    expected = dict(version=version, rate=rate, frames=len(samples), pcm_sha256=hashlib.sha256(pcm).hexdigest())
    if json.loads(receipt.read_text(encoding="utf-8")) != expected:
        raise ValueError("Qt readiness receipt differs from the expected version or decoded PCM")
    print(f"Qt {version}: displayed main window, loaded icon, decoded {len(samples)} samples and closed")


def check_portaudio_bundle(gui, platform):
    """Verify the bundled PortAudio library exists and loads cleanly."""
    if platform in ("linux-x64", "linux-arm64"):
        return
    expected = {
        "win-x64": "libportaudio64bit.dll",
        "win-arm64": "libportaudioarm64.dll",
        "macos-x64": "libportaudio.dylib",
        "macos-arm64": "libportaudio.dylib",
    }.get(platform)
    if expected is None:
        raise ValueError(f"Unsupported platform for PortAudio bundle check: {platform}")
    dll = gui.parent / "_sounddevice_data" / "portaudio-binaries" / expected
    if not dll.is_file():
        raise FileNotFoundError(f"Bundled PortAudio library missing from GUI payload: {dll}")
    lib = ctypes.CDLL(str(dll))
    try:
        if not hasattr(lib, "Pa_GetVersion"):
            raise RuntimeError(f"Bundled PortAudio library missing Pa_GetVersion symbol: {dll}")
    finally:
        if sys.platform == "win32":
            import _ctypes

            _ctypes.FreeLibrary(lib._handle)
    print(f"PortAudio bundle verified: {dll.name}")


def native_executables(dist, platform):
    if platform.startswith("macos-"):
        contents = dist / "SessionPrep.app/Contents"
        with (contents / "Info.plist").open("rb") as stream:
            executable = plistlib.load(stream)["CFBundleExecutable"]
        return None, contents / "MacOS" / executable
    suffix = platform + (".exe" if platform.startswith("win-") else "")
    cli = dist / "sessionprep.dist" / f"sessionprep-{suffix}"
    # Linux packaging merges both payloads into the installed directory.
    gui_root = cli.parent if platform.startswith("linux-") else dist / "sessionprep-gui.dist"
    return cli, gui_root / f"sessionprep-gui-{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["native", "fresh"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", help="Platform suffix of the compiled executables")
    args = parser.parse_args()
    if args.mode == "native" and not args.platform:
        parser.error("native requires --platform")
    with tempfile.TemporaryDirectory(prefix="sessionprep-smoke-") as temp:
        directory = Path(temp)
        env = {**os.environ, "HOME": temp, "USERPROFILE": temp, "XDG_CONFIG_HOME": temp,
               "APPDATA": temp, "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1", "QT_QPA_PLATFORM": "offscreen"}
        for key in ("PYTHONPATH", "VIRTUAL_ENV", "_SESSIONPREP_GUI_RELAUNCHED"):
            env.pop(key, None)
        if args.mode == "fresh":
            # The checkout itself is provided by CI. Both environments start empty.
            env["UV_CACHE_DIR"] = subprocess.check_output(["uv", "cache", "dir"], text=True).strip()
            env["UV_PROJECT_ENVIRONMENT"] = str(directory / "gui-environment")
            run(["uv", "build", str(ROOT), "--wheel", "--out-dir", temp], cwd=directory, env=env)
            wheel, = directory.glob("*.whl")
            cli_venv = directory / "cli-environment"
            run(["uv", "venv", str(cli_venv), "--python", sys.executable], cwd=directory, env=env)
            binaries = cli_venv / ("Scripts" if os.name == "nt" else "bin")
            python = binaries / ("python.exe" if os.name == "nt" else "python")
            run(["uv", "pip", "install", "--python", python, f"sessionprep[cli] @ {wheel.as_uri()}"],
                cwd=directory, env=env, timeout=300)
            run([python, "-I", "-c", "from importlib.util import find_spec; "
                 "assert find_spec('PySide6') is None and find_spec('sounddevice') is None"],
                cwd=directory, env=env)
            check_cli([binaries / ("sessionprep.exe" if os.name == "nt" else "sessionprep")],
                      directory, env, args.version)
            launcher = ROOT / "sessionprep-gui.py"
            without_uv = env.copy()
            without_uv.pop("UV", None)
            missing = subprocess.run([python, launcher], cwd=directory, env=without_uv,
                                     capture_output=True, text=True, timeout=30)
            if missing.returncode != 1 or "uv run --extra gui" not in missing.stderr:
                raise ValueError("Missing GUI prerequisites did not produce the source launch instructions")
            check_gui(["uv", "run", "--project", str(ROOT), "python", str(launcher)],
                      directory, env, args.version)
        else:
            cli, gui = native_executables(ROOT / "dist_nuitka", args.platform)
            if cli is not None:
                check_cli([cli], directory, env, args.version)
            check_gui([gui], directory, env, args.version)
            check_portaudio_bundle(gui, args.platform)
            if args.platform != "win-arm64":
                reply = json.loads(run([gui, "--ptsl-worker-self-test"], cwd=directory, env=env))
                if not reply["ok"] or not reply["result"]["ptsl_imported"]:
                    raise ValueError("Packaged PTSL worker failed its self-test")
                print("Packaged PTSL worker self-test passed")
            else:
                print("PTSL worker unavailable on win-arm64 (no grpcio wheel)")


if __name__ == "__main__":
    main()
