"""SFT data: expert trajectories recorded exactly as the env renders them; loss only on assistant tokens."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "scenarios" / "kandla_s13"
pytestmark = pytest.mark.skipif(not (SEED / "meta.yaml").exists(), reason="scenario seed 13 not built")


@pytest.fixture(scope="module")
def episode():
    from harness.env import AkshaEnv
    from tasks.kandla.gen import Gen
    from train.sft_data import record

    g = Gen(SEED, 13, "check")
    t = next(x for x in g.rp(2) if x.template == "first_detection")
    g.close()
    return record(AkshaEnv(SEED), t)


def test_record_returns_openai_style_tool_conversation(episode):
    msgs, score = episode["messages"], episode["score"]
    assert score >= 0.9 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    roles = [m["role"] for m in msgs[2:]]
    assert roles[0] == "assistant" and "tool" in roles
    for m in msgs:
        if m["role"] == "assistant":
            assert m["tool_calls"] and all(isinstance(c["function"]["arguments"], dict) for c in m["tool_calls"])
        if m["role"] == "tool":
            assert m["tool_call_id"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["tool_calls"][0]["function"]["name"] == "submit"


def test_labels_cover_only_assistant_turns(episode):
    try:
        import scripts.cuda_env  # noqa: F401
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from harness.prompt import static_prefix
    from train.sft_data import tokenize_with_mask

    ids, labels = tokenize_with_mask(tok, episode["messages"], static_prefix()[1])
    assert len(ids) == len(labels)
    learned = tok.decode([i for i, lb in zip(ids, labels, strict=True) if lb != -100])
    assert "submit" in learned and "event_search" in learned
    assert "SceneScout" not in learned and "LEDGER" not in learned        # system prompt and tool results masked
