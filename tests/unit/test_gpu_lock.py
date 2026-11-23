import json
import multiprocessing as mp
import os
import socket
import sys

import pytest

from scripts import gpu_lock as gl

DEAD_PID = 2**22 + 12345  # above Linux pid_max default and unlikely on Windows


def _stale(path, token="old"):
    path.write_text(json.dumps({"state": "rl", "pid": DEAD_PID, "side": gl._side(),
                                "host": socket.gethostname(), "token": token}))


def test_acquire_release(tmp_path):
    lock = tmp_path / ".gpu_lock"
    with gl.gpu_lock("rl", lock) as info:
        held = gl.read_lock(lock)
        assert held["state"] == "rl" and held["pid"] == os.getpid() and held["token"] == info["token"]
    assert not lock.exists()


def test_second_acquire_is_refused(tmp_path):
    lock = tmp_path / ".gpu_lock"
    with gl.gpu_lock("sft", lock):
        with pytest.raises(gl.GpuLockBusy, match="state=sft"):
            gl.acquire("rl", lock)
    assert not lock.exists()


def test_stale_lock_from_dead_pid_is_reclaimed(tmp_path):
    lock = tmp_path / ".gpu_lock"
    _stale(lock)
    info = gl.acquire("serve_eval", lock)
    assert info["state"] == "serve_eval"
    assert not (tmp_path / ".gpu_lock.takeover").exists(), "takeover mutex must be cleaned up"
    gl.release(lock, info["token"])


def test_takeover_never_removes_a_live_lock(tmp_path):
    # Contender judged the old lock stale, but before it acts, someone else already replaced it with a live lock.
    lock = tmp_path / ".gpu_lock"
    _stale(lock, token="old")
    judged = gl.read_lock(lock)
    lock.unlink()
    live = gl.acquire("rl", lock)
    gl._take_over_stale(lock, judged)
    assert gl.read_lock(lock)["token"] == live["token"], "live lock must survive"
    assert not (tmp_path / ".gpu_lock.takeover").exists()


def test_takeover_waits_while_another_takeover_is_in_progress(tmp_path):
    lock = tmp_path / ".gpu_lock"
    _stale(lock)
    (tmp_path / ".gpu_lock.takeover").write_text("")
    gl._take_over_stale(lock, gl.read_lock(lock))
    assert lock.exists(), "must not remove the lock without holding the takeover mutex"


def test_abandoned_takeover_mutex_expires(tmp_path, monkeypatch):
    lock = tmp_path / ".gpu_lock"
    _stale(lock)
    mutex = tmp_path / ".gpu_lock.takeover"
    mutex.write_text("")
    old = mutex.stat().st_mtime - gl.TAKEOVER_MUTEX_STALE_S - 5
    os.utime(mutex, (old, old))
    info = gl.acquire("rl", lock)
    assert info["state"] == "rl" and not mutex.exists()


def test_lock_from_other_os_side_is_never_reclaimed(tmp_path):
    lock = tmp_path / ".gpu_lock"
    other = "windows" if gl._side() != "windows" else "wsl"
    lock.write_text(json.dumps({"state": "sionna_precompute", "pid": 1, "side": other, "host": socket.gethostname()}))
    with pytest.raises(gl.GpuLockBusy):
        gl.acquire("rl", lock)


def test_corrupt_lock_is_treated_as_busy_not_crash(tmp_path):
    lock = tmp_path / ".gpu_lock"
    lock.write_text("{not json")
    with pytest.raises(gl.GpuLockBusy, match="release --force"):
        gl.acquire("rl", lock)


def test_release_leaves_other_holders_lock(tmp_path):
    lock = tmp_path / ".gpu_lock"
    other = gl.acquire("rl", lock)
    gl.release(lock, token="someone-else")
    assert gl.read_lock(lock)["token"] == other["token"]


def test_unknown_state_rejected(tmp_path):
    with pytest.raises(ValueError):
        gl.acquire("mining", tmp_path / ".gpu_lock")


def _contend(path, barrier, q, done):
    barrier.wait()
    try:
        info = gl.acquire("rl", path)
        q.put(info["token"])
        done.wait(60)  # a winner must stay alive, or its lock is legitimately stale and may be taken over
    except gl.GpuLockBusy:
        q.put(None)


@pytest.mark.parametrize("start_stale", [False, True])
def test_only_one_of_many_concurrent_contenders_wins(tmp_path, start_stale):
    lock = tmp_path / ".gpu_lock"
    if start_stale:
        _stale(lock)
    n = 8
    ctx = mp.get_context("spawn")
    barrier, q, done = ctx.Barrier(n), ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_contend, args=(lock, barrier, q, done)) for _ in range(n)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=60) for _ in range(n)]  # every contender has decided; winner still alive
        winners = [t for t in results if t]
        assert len(winners) == 1
        assert gl.read_lock(lock)["token"] == winners[0]
    finally:
        done.set()
        for p in procs:
            p.join(30)


def test_cli_run_returns_child_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(gl, "LOCK_PATH", tmp_path / ".gpu_lock")
    assert gl.main(["run", "bench", "--", sys.executable, "-c", "import sys; sys.exit(7)"]) == 7
    assert not (tmp_path / ".gpu_lock").exists()


def test_cli_run_missing_command_returns_127(tmp_path, monkeypatch):
    monkeypatch.setattr(gl, "LOCK_PATH", tmp_path / ".gpu_lock")
    assert gl.main(["run", "bench", "--", "definitely-not-a-command-xyz"]) == 127
    assert not (tmp_path / ".gpu_lock").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM delivery differs on Windows")
def test_sigterm_releases_the_lock(tmp_path):
    import signal
    import subprocess
    import time

    lock = tmp_path / ".gpu_lock"
    code = (f"import sys, time; sys.path.insert(0, {str(gl.REPO)!r}); from scripts import gpu_lock as gl\n"
            f"with gl.gpu_lock('rl', __import__('pathlib').Path({str(lock)!r})):\n"
            f"    print('held', flush=True); time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held" and lock.exists()
    p.send_signal(signal.SIGTERM)
    assert p.wait(10) == 143
    time.sleep(0.1)
    assert not lock.exists(), "SIGTERM must release the lock, not leave it stale"


def test_exit_if_busy_gives_exit_code_3(tmp_path):
    @gl.exit_if_busy
    def job():
        raise gl.GpuLockBusy("busy")

    with pytest.raises(SystemExit) as e:
        job()
    assert e.value.code == 3
