import json
import re

from harness.config import EnvConfig
from harness.prompt import static_prefix
from tools.registry import REGISTRY, load_all
from tools.types import REF_RE

UNITLESS_OK = {"limit", "confidence", "min_confidence", "score", "fraction", "k", "threshold"}   # weak_zones.threshold: unit set by the metric enum (df_err_p90_deg | snr_db | detect_prob), echoed in the result
UNIT_SUFFIXES = ("_m", "_s", "_deg", "_mhz", "_dbm", "_db", "_khz", "_m2", "_range", "_ms")


def test_every_tool_has_one_line_description_and_schema():
    load_all()
    for name, spec in REGISTRY.items():
        assert spec.description and "\n" not in spec.description and len(spec.description) <= 100, name
        json.dumps(spec.args.model_json_schema())


def _numeric_fields(schema: dict, path=""):
    for k, v in (schema.get("properties") or {}).items():
        types = {v.get("type")} | {o.get("type") for o in v.get("anyOf", [])}
        if types & {"number", "integer"}:
            yield path + k
        for sub in [v] + v.get("anyOf", []):
            if sub.get("type") == "object":
                yield from _numeric_fields(sub, path + k + ".")


def test_numeric_args_carry_units():
    load_all()
    for name, spec in REGISTRY.items():
        schema = spec.args.model_json_schema()
        for d in schema.get("$defs", {}).values():
            for f in _numeric_fields(d):
                assert f.split(".")[-1] in UNITLESS_OK or f.endswith(UNIT_SUFFIXES), f"{name}: {f}"
        for f in _numeric_fields(schema):
            assert f.split(".")[-1] in UNITLESS_OK or f.endswith(UNIT_SUFFIXES), f"{name}: {f}"


def test_static_prefix_is_byte_identical_and_within_budget():
    a, b = static_prefix(), static_prefix()
    assert a[0] == b[0] and a[2] == b[2]
    cfg = EnvConfig()
    assert cfg.tokens(a[0] + a[2]) <= cfg.static_prefix_tokens, cfg.tokens(a[0] + a[2])
    assert not re.search(r"\d{4}-\d{2}-\d{2}|[0-9a-f]{12}", a[0]), "no dates or ids in the static prefix"


def test_rendered_prefix_fits_budget_with_real_tokenizer():
    """The estimate can drift; measure the prefix exactly as the model sees it when the tokenizer is cached."""
    import pytest

    try:
        import scripts.cuda_env  # noqa: F401  (HF_HOME inside the repo)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("model tokenizer not cached")
    system, tools, _ = static_prefix()
    rendered = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": "x"}],
                                       tools=tools, tokenize=False, add_generation_prompt=True)
    n = len(tok(rendered)["input_ids"])
    assert n <= EnvConfig().static_prefix_tokens, n


def test_envelopes_and_refs(env, tasks):
    from tasks.expert import solve

    for t in tasks[:12]:
        env.reset(t)
        solve(env, t)
        for ev in env.tracer.events:
            pass
    env.reset(tasks[0])
    _, _, _, info = env.step({"tool_calls": [{"name": "track_search", "arguments": {}}]})
    e = info["results"][0]
    assert set(e) >= {"ok", "tool", "data", "refs", "warnings", "cost_ms", "state_hash"}
    assert e["refs"] and all(REF_RE.match(r) for r in e["refs"])
    assert "total" in e["data"]
