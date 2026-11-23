"""Single-GPU time-slicing lock (SPEC 2.5).

Exactly one GPU-heavy state at a time. The lock is a JSON file at the repo root, created atomically,
so it is shared by Windows-side jobs (e.g. Sionna) and WSL-side jobs (training, serving) through /mnt/c.

Library:
    with gpu_lock("rl"):
        ...

CLI:
    python scripts/gpu_lock.py status
    python scripts/gpu_lock.py run <state> -- <command ...>     # hold the lock while command runs
    python scripts/gpu_lock.py release --force                 # remove a lock left by a crashed job
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

STATES = ("sionna_precompute", "surrogate_train", "sft", "rl", "serve_eval", "env_check", "bench", "df_sim")
REPO = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(os.environ.get("AKSHA_GPU_LOCK", REPO / ".gpu_lock"))


class GpuLockBusy(RuntimeError):
    pass


def _side() -> str:
    """Which OS owns a pid. WSL and Windows pids are different namespaces."""
    if sys.platform == "win32":
        return "windows"
    return "wsl" if "microsoft" in platform.uname().release.lower() else "linux"


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(path: Path | None = None) -> dict | None:
    path = path or LOCK_PATH  # resolved at call time, so tests and AKSHA_GPU_LOCK overrides apply
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return {"state": "unknown", "corrupt": True}


def _is_stale(info: dict) -> bool:
    # Staleness can only be judged for a pid in our own namespace; a lock held from the other OS
    # side is assumed live and must be released with --force if its job really died.
    if info.get("side") != _side() or info.get("host") != socket.gethostname():
        return False
    pid = info.get("pid")
    return isinstance(pid, int) and not _pid_alive(pid)


TAKEOVER_MUTEX_STALE_S = 30.0


def _take_over_stale(path: Path, judged: dict) -> None:
    """Remove a stale lock without ever deleting a live one.

    Only the holder of a small takeover mutex (created with O_EXCL) may remove the lock, and only after re-reading
    it and confirming it is still the lock judged stale (same token). While that stale lock file exists nobody can
    create a new one, and its owner is dead, so the lock cannot change between the re-check and the unlink.
    Contenders that cannot get the mutex simply retry.
    """
    mutex = path.with_name(path.name + ".takeover")
    try:
        fd = os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # A takeover is in progress. If its process died mid-way, clear the mutex after a timeout.
        with contextlib.suppress(OSError):
            if time.time() - mutex.stat().st_mtime > TAKEOVER_MUTEX_STALE_S:
                mutex.unlink()
        return
    except PermissionError:
        return  # Windows: mutex is being deleted by its holder right now; retry
    os.close(fd)
    try:
        current = read_lock(path)
        if current is not None and current.get("token") == judged.get("token") and _is_stale(current):
            path.unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows: someone is reading the lock file; retry
    finally:
        with contextlib.suppress(OSError):
            mutex.unlink()


def acquire(state: str, path: Path | None = None) -> dict:
    path = path or LOCK_PATH
    if state not in STATES:
        raise ValueError(f"unknown GPU state {state!r}; expected one of {STATES}")
    info = {
        "state": state,
        "pid": os.getpid(),
        "side": _side(),
        "host": socket.gethostname(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": " ".join(sys.argv),
        "token": uuid.uuid4().hex,
    }
    held: dict | None = None
    attempts = 20
    for attempt in range(attempts):
        if attempt:
            time.sleep(0.02 * attempt)  # back off; contenders are usually gone within milliseconds
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_lock(path)
            if held is None:
                continue  # released between our create and our read
            if held.get("corrupt") and attempt < 3:
                continue  # possibly a lock being written right now; re-read before calling it busy
            if _is_stale(held):
                _take_over_stale(path, held)
                continue
            raise GpuLockBusy(
                f"GPU is busy: state={held.get('state')} pid={held.get('pid')} side={held.get('side')} "
                f"since {held.get('started')}. Wait for it to finish, or if that job is dead run: "
                f"python scripts/gpu_lock.py release --force"
            ) from None
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
        return info
    raise GpuLockBusy(f"could not acquire GPU lock after {attempts} attempts (last holder: {held})")


def release(path: Path | None = None, token: str | None = None) -> None:
    path = path or LOCK_PATH
    held = read_lock(path)
    if not held:
        return
    mine = held.get("token") == token if token else (held.get("pid") == os.getpid() and held.get("side") == _side())
    if mine:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def gpu_lock(state: str, path: Path | None = None):
    path = path or LOCK_PATH
    info = acquire(state, path)
    # A terminate signal (kill, Ctrl-Break, WSL shutdown) skips `finally` by default; turn it into SystemExit so
    # the lock is released instead of left stale.
    prev = None
    if threading.current_thread() is threading.main_thread():
        prev = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        yield info
    finally:
        release(path, info["token"])
        if prev is not None:
            signal.signal(signal.SIGTERM, prev)


def exit_if_busy(fn):
    """Decorator for script entry points: a busy GPU is a clear one-line message and exit code 3 (SPEC 2.5)."""
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except GpuLockBusy as e:
            print(f"[gpu_lock] {e}", file=sys.stderr)
            sys.exit(3)
    return wrapper


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("state", choices=STATES)
    r.add_argument("command", nargs=argparse.REMAINDER)
    rel = sub.add_parser("release")
    rel.add_argument("--force", action="store_true", required=True)
    a = p.parse_args(argv)

    if a.cmd == "status":
        held = read_lock()
        print("idle" if held is None else json.dumps(held))
        return 0
    if a.cmd == "release":
        LOCK_PATH.unlink(missing_ok=True)
        print("lock removed")
        return 0

    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("run needs a command after --")
    try:
        with gpu_lock(a.state):
            return subprocess.call(command)
    except GpuLockBusy as e:
        print(f"[gpu_lock] {e}", file=sys.stderr)
        return 3
    except FileNotFoundError:
        print(f"[gpu_lock] command not found: {command[0]}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())
