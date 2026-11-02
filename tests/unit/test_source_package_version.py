"""Fresh packaging metadata must work without a local install or generated version."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("ref_type,ref_name,base", [
    ("branch", "0.3.5", "0.3.5.dev0"),
    ("branch", "feature/audio", "0.3.4.dev1"),
    ("tag", "v0.3.5", "0.3.5"),
])
def test_setup_metadata_in_clean_detached_actions_checkout(tmp_path, ref_type, ref_name, base):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sessionpreplib").mkdir()
    for relative in ("setup.py", "sessionpreplib/_version.py"):
        shutil.copy2(ROOT / relative, source / relative)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

    git("init", "--quiet", "--initial-branch=0.3.5")
    git("add", ".")
    git("-c", "user.name=SessionPrep Tests", "-c", "user.email=tests@example.invalid",
        "commit", "--quiet", "-m", "Base fixture")
    git("tag", "v0.3.4")
    (source / "new-commit.txt").write_text("Detached build fixture", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=SessionPrep Tests", "-c", "user.email=tests@example.invalid",
        "commit", "--quiet", "-m", "Build fixture")
    sha = git("rev-parse", "HEAD")
    if ref_type == "tag":
        git("tag", ref_name)
    git("checkout", "--quiet", "--detach", sha)
    assert git("branch", "--show-current") == ""
    assert not (source / "sessionpreplib/_build_version.py").exists()
    # -I -S excludes installed distributions and packaging: only stdlib is available.
    # Intercept setup() to exercise its metadata/version bridge without dependencies.
    script = """
import json, runpy, sys, types
backend = types.ModuleType('setuptools')
backend.setup = lambda **kwargs: print(json.dumps(kwargs))
sys.modules['setuptools'] = backend
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    env = dict(os.environ, GITHUB_ACTIONS="true", GITHUB_REF_TYPE=ref_type,
               GITHUB_REF_NAME=ref_name, GITHUB_SHA=sha)
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(source / "setup.py")],
                            cwd=source, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    expected = base if ref_type == "tag" else base + "+g" + sha[:7]
    assert json.loads(result.stdout)["version"] == expected
    assert f'BUILD_VERSION = "{expected}"' in (source / "sessionpreplib/_build_version.py").read_text()
