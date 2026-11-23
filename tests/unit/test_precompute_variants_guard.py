"""scripts/precompute_variants.py guards (RF convergence audit 2026-10-26, P2 items)."""

import pytest

from scripts import precompute_variants as pv


@pytest.mark.parametrize("version", ["v21", "v3"])
def test_v1_stack_refused_under_versioned_data(version):
    # the v1 stack writes the legacy unversioned roots; under v21/v3 that would mix data versions
    with pytest.raises(SystemExit, match="--stack v1"):
        pv.check_stack("v1", version)


@pytest.mark.parametrize("stack,version", [("v1", "v2"), ("v2", "v2"), ("v2", "v21"), ("v2", "v3")])
def test_allowed_stack_versions(stack, version):
    pv.check_stack(stack, version)


def test_failed_label_names_the_lattice_shift():
    assert pv.failed_label("B2", None) == "B2"
    assert pv.failed_label("B2", 102) == "B2@p102"
