import pytest

torch = pytest.importorskip("torch")

from scripts.rl_dryrun_trl import LookupEnv, answer_reward, check_masking, make_table  # noqa: E402


def test_table_is_deterministic():
    assert make_table(10, seed=0) == make_table(10, seed=0)
    assert make_table(10, seed=0) != make_table(10, seed=1)


def test_env_rewards_exactly_one_correct_lookup():
    table = make_table(5, seed=0)
    env = LookupEnv(table)
    env.reset(key="K001")
    assert env.get_reward() == 0.0                       # no tool use
    assert env.lookup("k001") == f"code={table['K001']}"  # case-insensitive key
    assert env.get_reward() == 0.2
    env.lookup("K002")
    assert env.get_reward() == 0.1                       # right key but an extra call
    env.reset(key="K003")
    assert env.calls == [] and env.lookup("nope").startswith("error")


def test_answer_reward_reads_final_assistant_message():
    comp = [[{"role": "assistant", "content": "", "tool_calls": [{}]},
             {"role": "tool", "content": "code=ABC234"},
             {"role": "assistant", "content": "ABC234"}]]
    assert answer_reward(comp, answer=["ABC234"]) == [1.0]
    assert answer_reward(comp, answer=["ZZZ999"]) == [0.0]


class FakeTok:
    """Each id decodes to one word, so masks map to readable text."""
    vocab = {0: "<tool_call>", 1: "lookup", 2: "K001", 3: "</tool_call>", 4: "code=ABC234", 5: "ABC234", 6: "<pad>"}

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)


def _batch(tool_mask_row):
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    return {"completion_ids": ids, "completion_mask": torch.ones_like(ids),
            "tool_mask": torch.tensor([tool_mask_row])}


def test_masking_check_passes_when_only_tool_output_is_excluded():
    r = check_masking(_batch([1, 1, 1, 1, 0, 1]), FakeTok(), {"ABC234"})
    assert r["ok"] and r["tool_result_masked"] == 1 and r["model_tool_call_trained"] == 1


def test_masking_check_fails_when_tool_output_is_trained_on():
    r = check_masking(_batch([1, 1, 1, 0, 1, 1]), FakeTok(), {"ABC234"})  # wrong token masked
    assert not r["ok"] and r["tool_output_leaked_into_loss"]


def test_masking_check_fails_without_tool_mask():
    b = _batch([1] * 6)
    del b["tool_mask"]
    assert not check_masking(b, FakeTok(), set())["ok"]
