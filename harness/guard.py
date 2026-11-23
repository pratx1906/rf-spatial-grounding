"""Runtime truth guard (SPEC 12.1; DECISIONS D-006, D-019).

While active, any Python-level file access under a forbidden root (truth/, world_cache/) raises PermissionError.
The env activates it around every reset/step, so tools, prompt building and the verifier cannot read truth even
by path. Audit hooks are process-wide and cannot be removed, so the hook is installed once and toggled by state.

Native opens (DuckDB, Arrow in C++) bypass audit hooks. They are covered separately: tools only receive the
belief-side SiteView (read-only observed.duckdb), and the SQL tool (M2) rejects ATTACH/COPY/read_* by parser.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path

FORBIDDEN_DIRS = ("truth", "world_cache")
_DATA_REAL = os.path.realpath(Path(__file__).resolve().parents[1] / "data")
_EVENTS = {"open", "os.listdir", "os.scandir", "os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.chmod",
           "shutil.copyfile", "shutil.copytree", "shutil.rmtree", "os.truncate", "os.utime", "glob.glob"}

_state = threading.local()
_installed = False
_lock = threading.Lock()


class TruthAccessError(PermissionError):
    pass


def _roots() -> tuple[str, ...]:
    return getattr(_state, "roots", ())


def _hook(event: str, args: tuple) -> None:
    roots = _roots()
    if not roots or event not in _EVENTS or not args:
        return
    target = args[0]
    if isinstance(target, int) or target is None:
        return
    try:
        path = os.path.realpath(os.fsdecode(target))
    except (TypeError, ValueError):
        return
    for root in roots:
        if path == root or path.startswith(root + os.sep):
            raise TruthAccessError(f"truth guard: access to {path} is forbidden while the agent stack runs")
    if other_truth(path):
        raise TruthAccessError(f"truth guard: access to {path} (another site's truth/) is forbidden")


def other_truth(path: str) -> bool:
    """Any truth/ dir under data/ (other seeds' and worlds' scenarios, e.g. data/scenarios_v3/kandla_s16_X/truth,
    and data/<site>/truth), not only the episode site's (AUDIT_CODE minor 6). A component match, so scenario dirs
    built after the guard was set up are covered too."""
    if not path.startswith(_DATA_REAL + os.sep):
        return False
    return "truth" in path[len(_DATA_REAL) + 1:].split(os.sep)[:3]


def install() -> None:
    global _installed
    with _lock:
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True


_DATA = Path(__file__).resolve().parents[1] / "data"


def forbidden_roots(site_dir: str | Path) -> tuple[str, ...]:
    """The episode site's truth/ and world_cache/, plus EVERY data/*/world_cache: world-variant RF lives under
    data/site_kandla/world_cache while variant scenarios live under data/scenarios/ (D-031)."""
    roots = [os.path.realpath(Path(site_dir) / d) for d in FORBIDDEN_DIRS]
    roots.append(os.path.realpath(_DATA / "site_kandla" / "world_cache"))
    roots.append(os.path.realpath(_DATA / "site_kandla" / "world_cache_v2"))      # RF phase 2 hidden worlds + spectrum
    roots.append(os.path.realpath(_DATA / "site_kandla" / "world_cache_v3"))      # v3 data (explicit: may not exist yet)
    roots += [os.path.realpath(p) for p in _DATA.glob("*/world_cache*")]
    return tuple(dict.fromkeys(roots))


@contextlib.contextmanager
def truth_guard(site_dir: str | Path):
    """Guard this thread (and tool worker threads given the same roots via `guard_roots`)."""
    install()
    prev = _roots()
    _state.roots = forbidden_roots(site_dir)
    try:
        yield _state.roots
    finally:
        _state.roots = prev


@contextlib.contextmanager
def guard_roots(roots: tuple[str, ...]):
    """Apply already-computed roots on a worker thread (thread-local state does not propagate)."""
    prev = _roots()
    _state.roots = roots
    try:
        yield
    finally:
        _state.roots = prev
