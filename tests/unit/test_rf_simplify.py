"""Wrapper: runs tests/blender/test_rf_simplify_blender.py (Task 3) and test_rf_relevance_blender.py (Task 3b) inside Blender (factory startup, background, never saves
preferences) and checks every case passes. Skipped when no Blender is found (BLENDER env var, `blender` on PATH, or
the Windows Blender 5.2 install, also reachable from WSL through /mnt/c)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = {"simplify": (REPO / "tests" / "blender" / "test_rf_simplify_blender.py", 5),
           "relevance": (REPO / "tests" / "blender" / "test_rf_relevance_blender.py", 6)}
CANDIDATES = [r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
              "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"]


def _blender() -> str | None:
    for c in [os.environ.get("BLENDER"), shutil.which("blender"), *CANDIDATES]:
        if c and Path(c).exists():
            return c
    return None


def _arg_path(exe: str, p: Path) -> str:
    """A Windows Blender started from WSL needs a Windows path."""
    if exe.endswith(".exe") and str(p).startswith("/"):
        return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


BLENDER = _blender()


@pytest.mark.skipif(BLENDER is None, reason="Blender not installed")
@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_rf_simplify_in_blender(name):
    script, n = SCRIPTS[name]
    env = {**os.environ, "CYCLES_DEVICE": "NONE"}
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1",
                        "--python", _arg_path(BLENDER, script)],
                       capture_output=True, text=True, timeout=600, env=env)
    out = r.stdout + r.stderr
    lines = [ln for ln in out.splitlines() if ln.startswith(("PASS ", "FAIL ", "RESULT "))]
    assert r.returncode == 0, "\n".join(lines) or out[-4000:]
    assert any(ln == f"RESULT {n}/{n}" for ln in lines), "\n".join(lines)
