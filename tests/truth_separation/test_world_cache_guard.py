"""The runtime truth guard must cover every world_cache under data/, not only the episode's own site dir (D-031):
world-variant RF lives in data/site_kandla/world_cache while variant scenarios live in data/scenarios/."""

from pathlib import Path

import pytest

from harness.guard import TruthAccessError, forbidden_roots, truth_guard

REPO = Path(__file__).resolve().parents[2]


def test_forbidden_roots_include_the_site_world_cache(tmp_path):
    roots = forbidden_roots(tmp_path)
    assert any(r.endswith(str(Path("data") / "site_kandla" / "world_cache")) for r in roots)


def test_guard_blocks_world_cache_of_another_site_dir(tmp_path):
    target = REPO / "data" / "site_kandla" / "world_cache" / "_guard_probe.txt"
    with truth_guard(tmp_path):
        with pytest.raises(TruthAccessError):
            open(target, "rb")


def test_guard_blocks_the_v2_world_cache(tmp_path):
    """RF phase 2: hidden worlds and spectrum truth live in data/site_kandla/world_cache_v2 (and any *world_cache*)."""
    roots = forbidden_roots(tmp_path)
    assert any(r.endswith(str(Path("data") / "site_kandla" / "world_cache_v2")) for r in roots)
    target = REPO / "data" / "site_kandla" / "world_cache_v2" / "_guard_probe.txt"
    with truth_guard(tmp_path):
        with pytest.raises(TruthAccessError):
            open(target, "rb")
