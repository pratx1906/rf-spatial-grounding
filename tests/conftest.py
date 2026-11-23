import pytest


@pytest.fixture(scope="session")
def stub_dir(tmp_path_factory):
    """A fresh stub site (seed 0) generated once per test session."""
    from sim.stub_site import generate

    d = tmp_path_factory.mktemp("site") / "stub_site"
    generate(d, seed=0)
    return d


@pytest.fixture(scope="session")
def site(stub_dir):
    from sim.belief.site_view import SiteView

    return SiteView(stub_dir)


@pytest.fixture()
def ctx(site):
    from tools.registry import ToolContext, load_all

    load_all()
    return ToolContext(site)


@pytest.fixture(scope="session")
def tasks(stub_dir):
    from tasks.m1_tasks import generate

    return generate(stub_dir, n=40, seed=0)


@pytest.fixture()
def env(stub_dir):
    from harness.env import AkshaEnv

    return AkshaEnv(stub_dir)


def call(ctx, name, **kw):
    """Run a tool directly; returns ('ok', ToolOutput) or ('err', ToolError)."""
    from tools.errors import ToolError
    from tools.registry import REGISTRY

    spec = REGISTRY[name]
    try:
        return "ok", spec.fn(ctx, spec.args(**kw))
    except ToolError as e:
        return "err", e

# Blender-only tests run inside Blender (tests/unit/test_rf_simplify.py wraps them); pytest must not import them
collect_ignore = ["blender"]
