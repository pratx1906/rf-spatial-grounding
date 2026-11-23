"""Run provenance (SPEC 0.8): git hash, library versions, host. Attach to every JSON a script writes."""

from __future__ import annotations

import importlib.metadata as md
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIBS = ("torch", "vllm", "transformers", "peft", "trl", "bitsandbytes", "sionna-rt", "mitsuba", "drjit", "nvidia-ml-py")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance() -> dict:
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    status = _git("status", "--porcelain")
    return {
        "git_hash": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": versions,
    }
