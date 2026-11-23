"""GRPO / RFT loop pieces that run without a GPU (train/grpo.py): fresh-LoRA config, seeding, prompt sampling, world
filter, advantages, best-checkpoint rule, resume, vLLM token capture, PPO-clip / KL / RFT token loss, episode
records with the v2 reward."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from train import grpo
from train.rewards.episode import RewardConfig

REPO = Path(__file__).resolve().parents[2]


def test_fresh_lora_matches_sft_v11():
    cfg = grpo.fresh_lora_config(REPO / "checkpoints" / "sft_v11")
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 64 and cfg["lora_dropout"] == 0.05
    assert cfg["target_modules"] == sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                            "down_proj"])


def test_defaults_follow_the_plan():
    a = grpo.parse_args(["--out", "x"])
    assert a.group == 8 and a.temperature == 0.8 and a.val == "data/rl/val400.jsonl" and a.val_every == 4
    assert 2 <= a.epochs <= 4 and a.kl_beta > 0 and a.reward == "v2" and a.algo == "grpo"
    assert grpo.reward_config(a).version == "v2" and grpo.reward_config(a).dist_bonus
    b = grpo.parse_args(["--out", "x", "--no-dist-bonus", "--reward", "v1", "--step-cost", "0.03"])
    rc = grpo.reward_config(b)
    assert rc.version == "v1" and not rc.dist_bonus and rc.step_cost == 0.03


def test_seeds_are_stable_and_distinct():
    assert grpo.stable_seed("a", 0, 1) == grpo.stable_seed("a", 0, 1)
    assert grpo.stable_seed("a", 0, 1) != grpo.stable_seed("a", 1, 1)
    assert 0 <= grpo.stable_seed("x") < 2**31


def pool_rows(n=50, worlds=()):
    rows = [{"task_id": f"t{i}", "world": None} for i in range(n)]
    for w in worlds:
        rows += [{"task_id": f"{w}{i}", "world": w} for i in range(10)]
    return rows


def test_prompt_sampling_deterministic_per_seed_and_iter():
    p = pool_rows()
    a = grpo.sample_prompts(p, 16, seed=0, it=3)
    assert a == grpo.sample_prompts(p, 16, seed=0, it=3)
    assert a != grpo.sample_prompts(p, 16, seed=0, it=4)
    assert a != grpo.sample_prompts(p, 16, seed=1, it=3)
    assert len({r["task_id"] for r in a}) == 16


def test_world_filter_and_fraction(tmp_path):
    f = tmp_path / "pool.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in pool_rows(worlds=("M1", "W6"))))
    assert {r["world"] for r in grpo.load_pool(f, None)} == {None}
    kept = grpo.load_pool(f, {"M1"})
    assert {r["world"] for r in kept} == {None, "M1"}
    s = grpo.sample_prompts(kept, 10, 0, 1, world_frac=0.3)
    assert sum(1 for r in s if r["world"]) == 3 and len(s) == 10


def eps_of(rewards_by_group, success=None):
    out = []
    for g, rs in enumerate(rewards_by_group):
        for r in rs:
            out.append({"group": g, "reward": r, "success": (success or (lambda r: r > 0.5))(r)})
    return out


def test_group_advantages_skip_flat_groups():
    eps = eps_of([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    adv, st = grpo.group_advantages(eps)
    assert adv[:3] == [0.0, 0.0, 0.0]
    assert adv[3] < 0 < adv[5] and abs(adv[4]) < 1e-9 and abs(sum(adv[3:])) < 1e-6
    assert st["groups"] == 2 and st["groups_with_spread"] == 1 and st["spread_share"] == 0.5
    radv, rst = grpo.group_advantages(eps, "rft")
    assert radv == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0] and rst["trained_episodes"] == 5


def test_group_advantages_ignore_noise_groups_and_scale_by_batch_std():
    # pilot r6: all-success groups with step-cost noise (std ~0.004) next to a real success/failure group
    eps = eps_of([[1.278, 1.270, 1.279, 1.276], [0.0, 1.2, 0.1, 1.3]])
    adv, st = grpo.group_advantages(eps)
    assert adv[:4] == [0.0] * 4
    assert st["groups_with_spread"] == 2 and st["groups_trained"] == 1
    cs = [r - 0.65 for r in (0.0, 1.2, 0.1, 1.3)]
    sd = (sum(c * c for c in cs) / 4) ** 0.5
    assert all(abs(a - c / (sd + 1e-4)) < 1e-6 for a, c in zip(adv[4:], cs, strict=True))
    gadv, _ = grpo.group_advantages(eps, norm="group", min_spread=0.0)
    assert abs(gadv[0]) > 0.5                          # the old per-group scaling blew the noise up to ~+-1
    two = eps_of([[0.0, 1.0], [0.0, 0.2]])             # batch std keeps the bigger contrast bigger
    badv, _ = grpo.group_advantages(two)
    assert badv[1] > 3 * badv[3] > 0


def test_ppo_token_loss_clip_higher():
    import torch

    new, old = torch.tensor([0.0, 0.0]), torch.log(torch.tensor([1 / 1.25, 1 / 0.7]))   # ratios 1.25, 0.7
    _, d = grpo.ppo_token_loss(new, old, None, adv=1.0, clip=0.2, beta=0.0, clip_high=0.28)
    assert d["clipped"].tolist() == [False, True]
    _, d2 = grpo.ppo_token_loss(new, old, None, adv=1.0, clip=0.2, beta=0.0)
    assert d2["clipped"].tolist() == [True, True]
    a = grpo.parse_args(["--out", "x"])
    assert (a.clip, a.clip_high, a.adv_norm, a.min_spread) == (0.2, 0.28, "batch", 0.05)


def test_best_checkpoint_rule_ties_to_earlier():
    h = [{"iter": 1, "val_score": 0.80}, {"iter": 2}, {"iter": 5, "val_score": 0.85}, {"iter": 9, "val_score": 0.85}]
    assert grpo.pick_best(h)["iter"] == 5
    assert grpo.pick_best([{"iter": 1}]) is None


def test_resume_point(tmp_path):
    assert grpo.resume_point(tmp_path) == (1, None)
    (tmp_path / "log.jsonl").write_text(
        json.dumps({"iter": 1, "adapter_after": "a/iter_001"}) + "\n" +
        json.dumps({"iter": 2, "adapter_after": "a/iter_002"}) + "\n" +
        json.dumps({"iter": 3, "final": True, "val_score": 0.9}) + "\n")
    assert grpo.resume_point(tmp_path) == (3, "a/iter_002")


def fake_resp(prompt=(1, 2, 3), comp=(7, 8), lps=(-0.1, -0.2), extra=False):
    lp = SimpleNamespace(content=[SimpleNamespace(logprob=x) for x in lps])
    if extra:                                  # openai SDK keeps unknown fields in model_extra
        ch = SimpleNamespace(logprobs=lp, model_extra={"token_ids": list(comp)})
        return SimpleNamespace(choices=[ch], model_extra={"prompt_token_ids": list(prompt)})
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=lp, token_ids=list(comp))], prompt_token_ids=list(prompt))


def test_parse_turn_reads_token_ids_and_logprobs():
    for extra in (False, True):
        t = grpo.parse_turn(fake_resp(extra=extra))
        assert t == {"prompt_ids": [1, 2, 3], "completion_ids": [7, 8], "logprobs": [-0.1, -0.2]}
    assert grpo.parse_turn(fake_resp(lps=(-0.1,))) is None            # misaligned: not trained on
    assert grpo.parse_turn(fake_resp(comp=())) is None


def test_recording_client_asks_for_tokens_and_records():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return fake_resp()

    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), models="M")
    rc = grpo.RecordingClient(inner)
    rc.chat.completions.create(model="policy", messages=[], extra_body={"x": 1})
    assert seen["logprobs"] is True and seen["extra_body"] == {"x": 1, "return_token_ids": True}
    assert len(rc.turns) == 1 and rc.bad_turns == 0 and rc.models == "M"


def test_merge_turns_chains_prefix_turns_into_one_sequence():
    t1 = {"prompt_ids": [1, 2], "completion_ids": [3, 4], "logprobs": [-0.1, -0.2]}
    t2 = {"prompt_ids": [1, 2, 3, 4, 9], "completion_ids": [5], "logprobs": [-0.3]}
    t3 = {"prompt_ids": [7, 8], "completion_ids": [6], "logprobs": [-0.4]}     # breaks the chain: new segment
    segs = grpo.merge_turns([t1, t2, t3], max_len=100)
    assert [s["ids"] for s in segs] == [[1, 2, 3, 4, 9, 5], [7, 8, 6]]
    assert segs[0]["spans"] == [(2, 4), (5, 6)] and segs[0]["logprobs"] == [-0.1, -0.2, -0.3]
    assert segs[1]["spans"] == [(2, 3)]
    # a merged sequence longer than max_len is not built: turns stay separate (each left-truncated as before)
    segs = grpo.merge_turns([t1, t2], max_len=5)
    assert [s["ids"] for s in segs] == [[1, 2, 3, 4], [2, 3, 4, 9, 5]]
    assert segs[1]["spans"] == [(4, 5)]


def test_ppo_token_loss():
    torch = pytest.importorskip("torch")
    new = torch.tensor([-1.0, -2.0], requires_grad=True)
    old = torch.tensor([-1.0, -2.0])
    loss, d = grpo.ppo_token_loss(new, old, None, adv=2.0, clip=0.2, beta=0.0)
    assert torch.allclose(loss, torch.tensor([-2.0, -2.0])) and not d["clipped"].any()
    loss.sum().backward()
    assert torch.allclose(new.grad, torch.tensor([-2.0, -2.0]))     # d(-ratio A)/dlogp = -A at ratio 1
    # ratio far above 1 + eps with A > 0: clipped, no gradient
    new2 = torch.tensor([0.0], requires_grad=True)
    loss2, d2 = grpo.ppo_token_loss(new2, torch.tensor([-1.0]), None, adv=1.0, clip=0.2, beta=0.0)
    loss2.sum().backward()
    assert d2["clipped"].all() and float(new2.grad) == 0.0
    # KL to the reference: zero when equal, positive otherwise
    _, d3 = grpo.ppo_token_loss(new.detach(), old, old, adv=0.0, clip=0.2, beta=0.1)
    assert float(d3["kl"].abs().max()) < 1e-6
    l4, d4 = grpo.ppo_token_loss(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-3.0]), 0.0, 0.2, 0.1)
    assert float(d4["kl"]) > 0 and float(l4) > 0
    # RFT: NLL weighted by the advantage
    l5, _ = grpo.ppo_token_loss(torch.tensor([-0.5]), torch.tensor([-9.0]), None, 1.0, 0.2, 0.0, algo="rft")
    assert float(l5) == pytest.approx(0.5)


def fake_env(task_row, answer=None, reason="submitted", verified=True, score=1.0):
    from tasks.base import Task

    ex = SimpleNamespace(calls=3, valid_calls=3, repeat_calls=0)
    ans = None
    if answer is not None:
        from harness.answers import Answer
        from pydantic import TypeAdapter

        ans = TypeAdapter(Answer).validate_python(answer)
    return SimpleNamespace(task=Task.from_row(task_row), answer=ans, end_reason=reason, executor=ex, turn=3,
                           gen_tokens=200, cfg=SimpleNamespace(max_turns=8, max_tokens_per_turn=512),
                           verifier_result={"ok": verified}, answer_score=score,
                           reward_parts={"total": 0.42, "answer": 1.0})


ROW = {"task_id": "k01X3-zone_incursion-00006", "area": "operator_query", "template": "zone_incursion",
       "question": "Any UAV within the jetty zone at t=50 s?", "site": "data/scenarios_v3/kandla_s1_X",
       "answer_key": {"track_refs": ["trk:0447"], "boolean": True, "zone": "zone:jetty_restricted", "t_s": 50.0},
       "pool_bucket": "target_A", "world": None}


def events():
    return [{"type": "tool_call", "turn": 1, "name": "inside_zone", "arguments": {"zone_ref": "zone:jetty_restricted"}},
            {"type": "tool_result", "turn": 1, "name": "inside_zone",
             "result": {"ok": True, "tool": "inside_zone", "data": {"tracks": ["trk:0447"]}, "warnings": []}},
            {"type": "tool_call", "turn": 2, "name": "submit", "arguments": {}},
            {"type": "tool_result", "turn": 2, "name": "submit", "result": {"ok": True, "tool": "submit"}}]


def test_episode_record_v2_and_legacy():
    ans = {"intent": "operator_query", "boolean": True, "track_refs": ["trk:0447"], "evidence_refs": ["trk:0447"],
           "confidence": 0.9}
    msgs = [{"role": "tool", "content": '{"data":{"tracks":["trk:0447"]}}'}]
    rec = grpo.episode_record(ROW, fake_env(ROW, ans), events(), msgs, RewardConfig(version="v2"), "v2")
    assert rec["success"] and rec["reward_parts"]["answer"] == 1.0 and rec["reward"] > 1.0
    assert [c["name"] for c in rec["calls"]] == ["inside_zone", "submit"] and rec["counters"]["tool_calls"] == 1
    assert rec["counters"]["unsupported_refs"] == 0 and rec["terms"]["missed_violation"] == 0
    legacy = grpo.episode_record(ROW, fake_env(ROW, ans), events(), msgs, RewardConfig(), "v1")
    assert legacy["reward"] == 0.42                  # v1: the env's own reward, unchanged
    miss = grpo.episode_record(ROW, fake_env(ROW, {**ans, "boolean": False, "track_refs": []}, score=0.0),
                               events(), msgs, RewardConfig(version="v2"), "v2")
    assert miss["reward_parts"]["missed"] == -2.0 and not miss["success"]


def test_iteration_and_val_metrics():
    ans = {"intent": "operator_query", "boolean": True, "track_refs": ["trk:0447"], "evidence_refs": ["trk:0447"],
           "confidence": 0.9}
    eps = [grpo.episode_record(ROW, fake_env(ROW, ans), events(), [], RewardConfig(version="v2"), "v2")
           for _ in range(3)]
    m = grpo.iteration_metrics(eps)
    for k in ("reward_mean", "reward_std", "reward_components", "success_rate", "abstain_rate", "mean_steps",
              "tool_calls", "truth_guard_hits", "verifier_rejects", "hacking"):
        assert k in m, k
    assert m["success_rate"] == 1.0 and m["tool_calls"]["inside_zone"] == 3
    v = grpo.val_metrics(eps)
    assert v["score"] == 1.0 and v["per_family"] == {"zone_incursion": 1.0}


def test_traj_file_drops_token_ids(tmp_path):
    from train.rewards.hacking import read_trajectories

    e = {"task_id": "x", "token_turns": [{"prompt_ids": [1], "completion_ids": [2, 3], "logprobs": [0.0, 0.0]}]}
    grpo.save_traj(tmp_path / "t.jsonl.gz", [e])
    (r,) = read_trajectories(tmp_path / "t.jsonl.gz")
    assert "token_turns" not in r and r["completion_tokens"] == 2 and r["n_token_turns"] == 1


def test_env_config_scales_tool_timeouts():
    base = grpo.env_config("none", 1.0)
    big = grpo.env_config("none", 4.0)
    assert big.tool_timeout_ms["default"] == 4 * base.tool_timeout_ms["default"]
    assert big.tool_timeout_ms["fit_candidates"] == 4 * base.tool_timeout_ms["fit_candidates"]
    assert "sensor_consistency" in grpo.env_config("with_consistency").tools_extra
    assert grpo.parse_args(["--out", "x"]).timeout_scale == 1.0


def test_episode_record_logs_timeouts_and_rejects_unverified():
    ans = {"intent": "operator_query", "boolean": True, "track_refs": ["trk:0447"], "evidence_refs": ["trk:0447"],
           "confidence": 0.9}
    ev = events()[:1] + [{"type": "tool_result", "turn": 1, "name": "inside_zone",
                          "result": {"ok": False, "tool": "inside_zone", "error": {"code": "TIMEOUT", "message": "x"}}}]
    rec = grpo.episode_record(ROW, fake_env(ROW, ans, reason="verifier_failed", verified=False), ev, [],
                              RewardConfig(version="v2"), "v2")
    assert rec["counters"]["timeouts"] == 1
    assert not rec["success"] and rec["reward_parts"]["answer"] == 0.0     # AUDIT_CODE C7


def test_rft_filter_threshold():
    eps = eps_of([[0.0, 1.1, 1.5]])
    adv, _ = grpo.group_advantages(eps, "rft")
    assert adv == [0.0, 1.0, 1.0]
    adv2, st = grpo.group_advantages(eps, "rft", rft_min_reward=1.2)
    assert adv2 == [0.0, 0.0, 1.0] and st["trained_episodes"] == 1
    assert grpo.parse_args(["--out", "x"]).rft_min_reward is None


def test_rollout_path_end_to_end_with_a_fake_server(monkeypatch):
    """serve.agent.run_task on a real v3 site through RecordingClient: token capture, episode record, v2 reward."""
    site = REPO / "data" / "scenarios_v3" / "kandla_s1_X"
    if not site.exists():
        pytest.skip("v3 scenario missing")
    from harness.env import AkshaEnv
    from serve.agent import run_task
    from tasks.base import Task

    def create(**kw):
        call = SimpleNamespace(id="c1", function=SimpleNamespace(
            name="submit", arguments=json.dumps({"answer": {"intent": "abstain", "reason": "insufficient_evidence",
                                                            "confidence": 0.2}})))
        msg = SimpleNamespace(content="", tool_calls=[call])
        lp = SimpleNamespace(content=[SimpleNamespace(logprob=-0.5)] * 3)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=lp, token_ids=[5, 6, 7])],
                               prompt_token_ids=[1, 2], usage=None)

    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), models=None)
    rc = grpo.RecordingClient(inner)
    env = AkshaEnv(site, config=grpo.env_config("none", 4.0))
    evs, msgs = [], None
    for ev in run_task(env, Task.from_row(ROW), client=rc, temperature=0.8, model="policy", seed=7):
        if ev["type"] == "transcript":
            msgs = ev["messages"]
        elif ev["type"] in ("tool_call", "tool_result"):
            evs.append(ev)
    rec = grpo.episode_record(ROW, env, evs, msgs, RewardConfig(version="v2"), "v2")
    assert rc.turns == [{"prompt_ids": [1, 2], "completion_ids": [5, 6, 7], "logprobs": [-0.5] * 3}]
    assert rec["end_reason"] == "submitted" and rec["counters"]["abstain"] == 1
    assert rec["terms"]["wrong_abstain"] and rec["reward_parts"]["wrong_abstain"] == -0.5
    assert rec["reward_parts"]["missed"] == -2.0 and rec["reward"] < 0          # a "yes" incursion abstained
    assert rec["counters"]["zero_evidence_submit"] == 1 and rec["counters"]["submit_first"] == 1


def test_wait_gpu_free_waits_for_memory_release(monkeypatch):
    seq = iter([21000, 18000, 2500])
    monkeypatch.setattr(grpo, "gpu_used_mb", lambda: next(seq))
    monkeypatch.setattr(grpo.time, "sleep", lambda s: None)
    grpo.wait_gpu_free(max_used_mb=6000, timeout_s=60)              # returns on the third reading
    monkeypatch.setattr(grpo, "gpu_used_mb", lambda: 21000)
    with pytest.raises(RuntimeError):
        grpo.wait_gpu_free(max_used_mb=6000, timeout_s=0.01)


def test_sigterm_kills_vllm_sessions():
    import signal
    import subprocess

    p = subprocess.Popen(["sleep", "60"], start_new_session=True)      # stands in for vLLM (own session)
    grpo._LIVE_VLLM.add(p.pid)
    try:
        with pytest.raises(SystemExit) as e:
            grpo._on_sigterm(signal.SIGTERM, None)
        assert e.value.code == 143
        assert p.wait(timeout=5) == -signal.SIGKILL
    finally:
        grpo._LIVE_VLLM.discard(p.pid)


def _grp(tid, fam, g, rewards):
    return [{"task_id": tid, "template": fam, "group": g, "reward": r} for r in rewards]


def test_adaptive_sampler_prefers_families_with_signal_and_skips_flat_tasks():
    pool = [{"task_id": f"{f}-{i}", "template": f} for f in ("easy", "edge") for i in range(40)]
    s = grpo.AdaptiveSampler(pool, min_spread=0.05, skip_p=1.0)
    eps = []
    for g in range(10):                                            # easy: always all-correct; edge: mixed
        eps += _grp(f"easy-{g}", "easy", g, [1.2] * 8)
        eps += _grp(f"edge-{g}", "edge", 10 + g, [1.2] * 4 + [0.2] * 4)
    s.observe(eps)
    assert s.weight("easy") == pytest.approx(1 / 12) and s.weight("edge") == pytest.approx(11 / 12)
    rows = s.sample(16, seed=0, it=3)
    assert len(rows) == 16 and len({r["task_id"] for r in rows}) == 16
    assert sum(r["template"] == "edge" for r in rows) >= 12
    assert not any(r["task_id"] in {f"easy-{g}" for g in range(10)} for r in rows)    # flat tasks skipped
    assert rows == s.sample(16, seed=0, it=3)                                          # deterministic
    more = s.sample(8, seed=0, it=3, exclude={r["task_id"] for r in rows})
    assert not {r["task_id"] for r in more} & {r["task_id"] for r in rows}


def test_unseen_family_gets_prior_weight():
    s = grpo.AdaptiveSampler([{"task_id": "a", "template": "new"}])
    assert s.weight("new") == 0.5


def test_paired_delta():
    base = [{"task_id": str(i), "score": 0.5} for i in range(100)]
    new = [{"task_id": str(i), "score": 0.5 + (0.2 if i < 30 else 0.0)} for i in range(100)]
    d = grpo.paired_delta(base, new)
    assert d["n"] == 100 and d["delta"] == pytest.approx(0.06) and d["better"] == 30 and d["worse"] == 0
    assert 0 < d["ci95"][0] <= 0.06 <= d["ci95"][1]
    assert grpo.paired_delta(base, [{"task_id": "x", "score": 1}]) == {"n": 0}


def test_rollout_offset_and_new_flags():
    a = grpo.parse_args(["--out", "x", "--sampler", "adaptive", "--min-informative", "16", "--kl-beta", "0"])
    assert a.sampler == "adaptive" and a.min_informative == 16 and a.topup_max == 12 and a.kl_beta == 0.0
    assert grpo.parse_args(["--out", "x"]).sampler == "uniform"                # old runs unchanged
