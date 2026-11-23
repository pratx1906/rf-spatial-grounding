"""Node path caches (air7311_<S>/spec_diff/<band>.npz) loaded once per process, prefetched in threads (Task 2b).

Each file is ~31 MB compressed / ~120 MB loaded and a v2 build reads 15 of them (5 nodes x 3 bands) from the
Windows drive; the sequential loads took ~9 s per build, and the spectrum world, the epoch loop and the labelling
in products_v2 each loaded them again. load() memoises the arrays (read-only use) for the current cache root;
prefetch() starts the loads in background threads (file reads and zlib inflation release the GIL) so they overlap
the first epoch. Switching to another root drops the previous root's arrays.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()
_MEMO: dict[str, Future] = {}
_ROOT: list = [None]
_POOL: list = [None]


def _read(f: str) -> dict:
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def _root_of(f: Path) -> str:
    return str(Path(f).parents[2])                     # <root>/air7311_<S>/spec_diff/<band>.npz (no resolve: slow on /mnt/c)


def _submit(f: Path, pool: bool) -> Future:
    key = str(f)
    with _LOCK:
        root = _root_of(f)
        if _ROOT[0] != root:
            _MEMO.clear()
            _ROOT[0] = root
        fut = _MEMO.get(key)
        if fut is None:
            if pool:
                if _POOL[0] is None:
                    _POOL[0] = ThreadPoolExecutor(4, thread_name_prefix="pathcache")
                fut = _POOL[0].submit(_read, key)
            else:
                fut = Future()
                fut.set_running_or_notify_cancel()
            _MEMO[key] = fut
            if not pool:
                try:
                    fut.set_result(_read(key))
                except BaseException as e:  # noqa: BLE001
                    fut.set_exception(e)
                    _MEMO.pop(key, None)
        return fut


def load(f: Path) -> dict:
    """All arrays of one path-cache file (shared; callers must not modify them in place)."""
    return _submit(f, pool=False).result()


def prefetch(files) -> None:
    for f in files:
        if Path(f).exists():
            _submit(f, pool=True)


def clear() -> None:
    with _LOCK:
        _MEMO.clear()
        _ROOT[0] = None
