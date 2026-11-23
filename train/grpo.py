"""GRPO / RFT for SceneScout (SPEC 9.5; D-027 phased on-policy loop; paper plan P3/P4 + Revision 1 B1/B2/B3/M2/M5).

    python -m train.grpo --init checkpoints/sft_v11 --pool data/rl/pool_B.jsonl --out runs/paper/results/P4/B_s0 \
        --seed 0 [--iters 30] [--prompts 16] [--group 8] [--temperature 0.8]
    --init base                 RL-only arm: a fresh LoRA (rank / alpha / targets of --lora-like, SFT v1.1) on the base
    --algo rft                  rejection-sampling fine-tune control: same pool, same rollouts, NLL on verified-correct
    --worlds M1,M2,M3,M4[,W6]   pool rows tagged with these worlds are used (others with a world tag are dropped)
    --ablation with_consistency env tool switches (eval/ablations.py presets), e.g. sensor_consistency for arm C
    --reward v2|v1              v2 = the RL reward (train/rewards/episode.py); v1 = the legacy reward (reproducibility)
    --no-dist-bonus             v2 without the 0.3 exp(-d/80 m) distance bonus (ablation)

Per iteration (one GPU, no vLLM sleep mode under WSL2, D-018):
1. ROLLOUT: serve the current LoRA with vLLM (`--seed`) and run G episodes per prompt through the exact serving path
   (serve.agent.run_task, same prompt, tools and constrained tool calls as the UI). A recording client asks vLLM for
   the prompt token ids, the sampled token ids and their log-probs (`return_token_ids`, `logprobs`), so the update
   trains on exactly the sampled tokens in exactly the context the sampler saw (no re-tokenised transcript).
   Every episode is seeded: request seed + env seed = hash(seed, iter, prompt, sample).
   Validation (`--val`, default the fixed 400-task data/rl/val400.jsonl from x3/val_clean; greedy) runs on the policy
   before the update at iteration 1 and every --val-every iterations, and once more on the final adapter.
2. UPDATE (vLLM stopped): group-normalised advantages A = (r - mean_g) / (std_g + 1e-4) (groups with zero spread
   are skipped; their share is logged; default since 2026-10-29: group mean, batch std, groups with std <= --min-spread
   skipped, see group_advantages); `--epochs` (2-4) clipped PPO passes over minibatches of `--mb` episodes:
   ratio = exp(logp_new - logp_vllm) on the sampled tokens, surrogate -min(ratio A, clip(ratio, 1 - clip, 1 + clip_high) A), plus
   `--kl-beta` x k3-KL to the init policy (init adapter, or the bare base for --init base). Token sums are divided by
   a constant (`--len-norm`, Dr GRPO). Log-probs at temperature 1 on both sides (vLLM raw_logprobs); dropout off.
   RFT: the same passes with A = 1 on verified-correct episodes, 0 elsewhere, and plain NLL instead of the clip.
Adam state persists across iterations (optimizer.pt next to the newest adapter only).

Outputs under --out (a re-run with the same --out resumes after the last logged iteration):
- log.jsonl, one record per iteration: iter, reward_mean / reward_std, reward_components (mean per reward part),
  success_rate, abstain_rate, mean_steps, tool_calls (per tool), truth_guard_hits, verifier_rejects, hacking
  (train/rewards/hacking aggregate), groups_with_spread, val_score + val (every --val-every), loss, kl, clip_frac,
  logp_mismatch (first minibatch: mean logp_new - logp_vllm, ~0 when numerics agree), grad_norm, rollout_s /
  update_s / val_s / wall_s, peak_vram_mb (nvidia-smi during the iteration), torch_peak_mb (update).
- traj/iter_NNN.jsonl.gz (+ traj/val_iter_NNN.jsonl.gz): every episode (task, transcript, tool calls, answer, reward
  breakdown, v2 task terms, hacking counters).
- adapters/iter_NNN (validated iterations and the newest are kept), best/ (best val mean score, ties to the earlier
  iteration), best.json, manifest.json (git hash, dirty files, command, init adapter / pool / val sha256, seed,
  reward config, wall time, peak VRAM), run_config.json, final.txt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# variable-length update passes fragment the default allocator (pilot 2026-10-28 grew past the 22 GiB guard).
# Not expandable_segments: its CUDA VMM calls fail under WSL2 ("device not ready", pilot r2; cf. D-018).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
# weights are in the local HF cache; a Hub lookup at a vLLM start failed and cost arm B its run (2026-10-29 12:28)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from train.rewards.episode import RewardConfig, episode_reward, stats_from_env  # noqa: E402
from train.rewards.hacking import aggregate, counters  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
PORT = 8011
LORA_LIKE = "checkpoints/sft_v11"
LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type")


# --- small pure helpers (unit-tested) ---------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big") & 0x7FFFFFFF


def sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def fresh_lora_config(like: str | Path = LORA_LIKE) -> dict:
    """LoraConfig kwargs of a fresh adapter with the rank / alpha / dropout / targets of an existing one."""
    cfg = json.loads((Path(like) / "adapter_config.json").read_text())
    out = {k: cfg[k] for k in LORA_KEYS if k in cfg}
    out["target_modules"] = sorted(out["target_modules"])
    return out


def load_pool(path: Path, worlds: set[str] | None) -> list[dict]:
    """Pool rows; rows tagged with a world are kept only when that world is in `worlds`."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    worlds = worlds or set()
    return [r for r in rows if not r.get("world") or r["world"] in worlds]


def sample_prompts(pool: list[dict], n: int, seed: int, it: int, world_frac: float | None = None) -> list[dict]:
    """n prompts for iteration `it`, deterministic in (seed, it). With world_frac, that share comes from world-tagged
    rows (if any) and the rest from untagged ones; otherwise uniform over the pool."""
    rng = random.Random(stable_seed("prompts", seed, it))
    if world_frac is None:
        return rng.sample(pool, min(n, len(pool)))
    tagged = [r for r in pool if r.get("world")]
    plain = [r for r in pool if not r.get("world")]
    k = min(len(tagged), round(n * world_frac))
    return rng.sample(tagged, k) + rng.sample(plain, min(n - k, len(plain)))


class AdaptiveSampler:
    """Online difficulty-aware prompt selection (recipe B3, 2026-10-29). Only groups whose rewards differ carry a
    gradient; in arm B 68 % of groups did not (all 8 right or all 8 wrong). Following online difficulty filtering
    (Bae et al. 2025, arXiv 2504.03380: reward variance as the learnability proxy), GRESO (arXiv 2506.02177: skip
    prompts whose last rollout had no variance) and DAPO dynamic sampling (arXiv 2503.14476):
    - a family is drawn with weight (informative groups + 1) / (groups + 2), from every group seen so far in this run
      (Laplace prior: an unseen family has weight 1/2, a family that never gives a signal decays but is retried);
    - a task whose last group had no signal is skipped with probability `skip_p` (it may have become learnable);
    - within one draw a task is used once. Deterministic in (seed, iteration)."""

    def __init__(self, pool: list[dict], min_spread: float = 0.05, skip_p: float = 0.9):
        self.by_fam: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            self.by_fam[r["template"]].append(r)
        self.min_spread, self.skip_p = min_spread, skip_p
        self.groups, self.signal = Counter(), Counter()
        self.flat: dict[str, bool] = {}

    def observe(self, eps: list[dict]) -> None:
        by = defaultdict(list)
        for e in eps:
            by[e["group"]].append(e)
        for g in by.values():
            fam, tid = g[0]["template"], g[0]["task_id"]
            inf = len(g) > 1 and statistics.pstdev([e["reward"] for e in g]) > self.min_spread
            self.groups[fam] += 1
            self.signal[fam] += inf
            self.flat[tid] = not inf

    def weight(self, fam: str) -> float:
        return (self.signal[fam] + 1) / (self.groups[fam] + 2)

    def weights(self) -> dict[str, float]:
        return {f: round(self.weight(f), 3) for f in sorted(self.by_fam)}

    def sample(self, n: int, seed: int, it: int, exclude: set[str] | None = None) -> list[dict]:
        rng = random.Random(stable_seed("adaptive", seed, it, len(exclude or ())))
        used = set(exclude or ())
        fams = sorted(self.by_fam)
        out: list[dict] = []
        for _ in range(50 * n):
            if len(out) >= n:
                break
            live = [f for f in fams if any(r["task_id"] not in used for r in self.by_fam[f])]
            if not live:
                break
            fam = rng.choices(live, weights=[self.weight(f) for f in live])[0]
            row = rng.choice([r for r in self.by_fam[fam] if r["task_id"] not in used])
            if self.flat.get(row["task_id"]) and rng.random() < self.skip_p:
                continue
            used.add(row["task_id"])
            out.append(row)
        return out


def paired_delta(base: list[dict], new: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean task-score change new - base on the tasks both validation runs share, with a 95 % bootstrap CI over
    tasks. Greedy validation on a fixed set: pairing removes the between-task variance that made val200 means
    (SE ~0.03) unable to resolve +-0.02 (arm B, 2026-10-29)."""
    b = {e["task_id"]: e["score"] for e in base}
    d = [e["score"] - b[e["task_id"]] for e in new if e["task_id"] in b]
    if not d:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return {"n": len(d), "delta": round(statistics.mean(d), 4), "ci95": [round(boots[int(0.025 * n_boot)], 4),
            round(boots[int(0.975 * n_boot) - 1], 4)], "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d)}


def group_advantages(eps: list[dict], algo: str = "grpo", rft_min_reward: float | None = None,
                     norm: str = "batch", min_spread: float = 0.05) -> tuple[list[float], dict]:
    """Per-episode advantage (aligned with eps) and spread stats. GRPO: r - mean of its prompt group, divided by
    norm "batch" = the std of the centred rewards over all kept groups (group mean + batch std, "Lite PPO"), or
    "group" = that group's own std (original GRPO). Groups with reward std <= min_spread get 0: in the pilot 6 of
    13 groups with any spread had std 0.003-0.005 (all 8 succeed or all fail; step-cost noise), which "group"
    scaled up to +-1, the same weight as a real success/failure contrast. RFT: 1 for verified-correct episodes
    (and reward >= rft_min_reward if set), else 0."""
    adv = [0.0] * len(eps)
    by = defaultdict(list)
    for i, e in enumerate(eps):
        by[e["group"]].append(i)
    spread, kept = 0, []
    for idx in by.values():
        rs = [eps[i]["reward"] for i in idx]
        sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        if sd > 1e-6:
            spread += 1
        if sd > min_spread:
            mu = statistics.mean(rs)
            kept.append((idx, [r - mu for r in rs], sd))
    batch_sd = statistics.pstdev([c for _, cs, _ in kept for c in cs]) if kept else 0.0
    if algo == "grpo":
        for idx, cs, sd in kept:
            scale = (batch_sd if norm == "batch" else sd) + 1e-4
            for i, c in zip(idx, cs, strict=True):
                adv[i] = c / scale
    if algo == "rft":
        adv = [1.0 if e.get("success") and (rft_min_reward is None or e["reward"] >= rft_min_reward) else 0.0
               for e in eps]
    return adv, {"groups": len(by), "groups_with_spread": spread,
                 "spread_share": round(spread / len(by), 4) if by else 0.0,
                 "groups_trained": len(kept), "adv_batch_std": round(batch_sd, 4),
                 "trained_episodes": sum(1 for a in adv if a != 0.0)}


def pick_best(history: list[dict]) -> dict | None:
    """Best validated record: highest val_score, ties to the earlier iteration (Revision 1, B1)."""
    vals = [h for h in history if h.get("val_score") is not None]
    if not vals:
        return None
    return sorted(vals, key=lambda h: (-h["val_score"], h["iter"]))[0]


def read_log(out: Path) -> list[dict]:
    p = out / "log.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def resume_point(out: Path) -> tuple[int, str | None]:
    """(next iteration, adapter to continue from) from an existing log.jsonl; (1, None) for a fresh run."""
    recs = [r for r in read_log(out) if isinstance(r.get("iter"), int) and r.get("adapter_after")]
    if not recs:
        return 1, None
    last = max(recs, key=lambda r: r["iter"])
    return last["iter"] + 1, last["adapter_after"]


def merge_turns(turns: list[dict], max_len: int) -> list[dict]:
    """Episode turns as forward-pass segments: {"ids", "spans": [(start, end) of each completion in ids], "logprobs"}.
    A turn whose prompt starts with the previous segment's ids extends it, so one pass scores every turn of a chained
    episode (the pilot ran one ~10k-token pass per turn). Segments are never built past max_len; a lone turn longer
    than max_len is left-truncated as before."""
    segs: list[dict] = []
    for t in turns:
        p, c = t["prompt_ids"], t["completion_ids"]
        s = segs[-1] if segs else None
        if s is not None and len(p) + len(c) <= max_len and p[:len(s["ids"])] == s["ids"]:
            start = len(p)
            s["ids"] = p + c
        else:
            ids = (p + c)[-max_len:]
            start = len(ids) - len(c)
            s = {"ids": ids, "spans": [], "logprobs": []}
            segs.append(s)
        s["spans"].append((start, start + len(c)))
        s["logprobs"] += list(t["logprobs"])
    return segs


def ppo_token_loss(new_logp, old_logp, ref_logp, adv: float, clip: float, beta: float, algo: str = "grpo",
                   clip_high: float | None = None):
    """Per-token loss (torch tensors of one completion) and diagnostics. GRPO: clipped surrogate with ratio in
    [1 - clip, 1 + clip_high] (clip_high None = clip; DAPO "clip-higher" uses 0.28 so rare good tokens can grow);
    RFT: NLL weighted by adv. Both + beta * k3 KL(new || ref) (ref_logp None or beta 0: no KL)."""
    import torch

    if algo == "rft":
        loss = -adv * new_logp
        ratio = torch.ones_like(new_logp)
        clipped = torch.zeros_like(new_logp, dtype=torch.bool)
    else:
        hi = clip if clip_high is None else clip_high
        ratio = torch.exp(new_logp - old_logp)
        s1, s2 = ratio * adv, torch.clamp(ratio, 1 - clip, 1 + hi) * adv
        loss = -torch.minimum(s1, s2)
        clipped = (ratio < 1 - clip) | (ratio > 1 + hi)
    kl = torch.zeros_like(new_logp)
    if ref_logp is not None and beta:
        d = ref_logp - new_logp
        kl = torch.exp(d) - d - 1
        loss = loss + beta * kl
    return loss, {"kl": kl.detach(), "ratio": ratio.detach(), "clipped": clipped.detach()}


# --- rollouts ---------------------------------------------------------------------------------------------------
def parse_turn(resp) -> dict | None:
    """Prompt ids, sampled ids and their log-probs from a vLLM chat completion (return_token_ids + logprobs)."""
    def extra(obj, k):
        v = getattr(obj, k, None)
        if v is None and getattr(obj, "model_extra", None):
            v = obj.model_extra.get(k)
        return v

    ch = resp.choices[0]
    prompt, comp = extra(resp, "prompt_token_ids"), extra(ch, "token_ids")
    lp = getattr(getattr(ch, "logprobs", None), "content", None) or []
    if not prompt or not comp or len(lp) != len(comp):
        return None
    return {"prompt_ids": list(prompt), "completion_ids": list(comp), "logprobs": [float(x.logprob) for x in lp]}


class RecordingClient:
    """Wraps the OpenAI client for one episode: every chat completion also returns token ids and log-probs, recorded
    per turn. run_task sees an ordinary client."""

    def __init__(self, client):
        self._c = client
        self.turns: list[dict] = []
        self.bad_turns = 0
        self.models = client.models
        outer = self

        class _Comp:
            @staticmethod
            def create(**kw):
                eb = dict(kw.pop("extra_body", None) or {})
                eb["return_token_ids"] = True
                kw["extra_body"] = eb
                kw["logprobs"] = True
                resp = outer._c.chat.completions.create(**kw)
                t = parse_turn(resp)
                if t is None:
                    outer.bad_turns += 1
                else:
                    outer.turns.append(t)
                return resp

        class _Chat:
            completions = _Comp()

        self.chat = _Chat()


def episode_record(row: dict, env, events: list[dict], msgs: list[dict] | None, cfg: RewardConfig,
                   reward_version: str) -> dict:
    calls, results, pending = [], [], []
    for ev in events:
        if ev["type"] == "tool_call":
            pending.append(ev)
        elif ev["type"] == "tool_result":
            c = pending.pop(0) if pending else {"turn": ev.get("turn"), "name": ev["name"], "arguments": {}}
            r = ev["result"]
            results.append(r)
            err = r.get("error") or {}
            calls.append({"turn": c.get("turn"), "name": ev["name"], "args": c.get("arguments"), "ok": bool(r.get("ok")),
                          "code": err.get("code") if isinstance(err, dict) else None,
                          "message": (err.get("message") if isinstance(err, dict) else str(err)),
                          "warnings": r.get("warnings") or [],
                          "suspect_refs": (r.get("data") or {}).get("suspect_refs") if r.get("ok") else None})
    for c in pending:                      # calls without a result (episode ended in that turn)
        calls.append({"turn": c.get("turn"), "name": c["name"], "args": c.get("arguments"), "ok": None,
                      "code": None, "message": None, "warnings": []})
    stats, terms = stats_from_env(env, results, cfg, task_row=row)
    parts = episode_reward(stats, cfg) if reward_version == "v2" else dict(env.reward_parts or {"total": -1.0})
    ans = env.answer.model_dump(mode="json") if env.answer is not None else None
    rec = {"task_id": row["task_id"], "area": row["area"], "template": row["template"], "task": row,
           "answer": ans, "end_reason": env.end_reason, "turns": env.turn, "score": float(env.answer_score or 0.0),
           "reward": float(parts["total"]), "reward_parts": parts, "legacy_reward_parts": env.reward_parts,
           "terms": terms, "success": bool(stats.answered and stats.grounded and terms["correct"]),
           "calls": calls, "messages": msgs}
    rec["counters"] = counters(rec)
    return rec


def env_config(ablation: str = "none", timeout_scale: float = 1.0):
    """EnvConfig for rollouts: eval/ablations.py preset(s) + tool timeouts scaled by `timeout_scale` (AUDIT_CODE C8:
    250 ms wall-clock budgets time out nondeterministically under load; per-episode timeouts are logged either way)."""
    from eval.ablations import apply_ablation
    from harness.config import EnvConfig

    cfg = apply_ablation(EnvConfig.load(), ablation)
    if timeout_scale != 1.0:
        cfg = EnvConfig(**{**cfg.model_dump(), "tool_timeout_ms": {k: int(v * timeout_scale)
                                                                    for k, v in cfg.tool_timeout_ms.items()}})
    return cfg


def rollouts(rows: list[dict], group: int, temperature: float, conc: int, seed: int, it: int, cfg: RewardConfig,
             reward_version: str, ablation: str, tag: str = "train", timeout_scale: float = 1.0,
             offset: int = 0) -> list[dict]:
    """G episodes per row; group id (and episode seed) = offset + row index, so a top-up batch never collides."""
    from openai import OpenAI

    from harness.env import AkshaEnv
    from serve.agent import run_task
    from tasks.base import Task

    local = threading.local()
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="local", timeout=180)
    env_cfg = env_config(ablation, timeout_scale)

    def one(job):
        p, g = job
        row = rows[p - offset]
        envs = getattr(local, "envs", None) or {}
        local.envs = envs
        site = row.get("site") or "data/site_kandla"
        env = envs.setdefault(site, AkshaEnv(REPO / site, config=env_cfg))
        rc = RecordingClient(client)
        events, msgs = [], None
        ep_seed = stable_seed(tag, seed, it, p, g)
        for ev in run_task(env, Task.from_row(row), client=rc, temperature=temperature, model="policy",
                           seed=ep_seed):
            if ev["type"] == "transcript":
                msgs = ev["messages"]
            elif ev["type"] in ("tool_call", "tool_result", "error"):
                events.append(ev)
        rec = episode_record(row, env, events, msgs, cfg, reward_version)
        rec.update({"group": p, "sample": g, "episode_seed": ep_seed, "token_turns": rc.turns,
                    "bad_turns": rc.bad_turns, "timeouts": rec["counters"]["timeouts"]})
        return rec

    jobs = [(offset + p, g) for p in range(len(rows)) for g in range(group)]
    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(one, jobs))


# --- vLLM -------------------------------------------------------------------------------------------------------
def start_vllm(adapter: Path, util: float, max_len: int, seed: int, log_path: Path, tries: int = 3
               ) -> subprocess.Popen:
    """Start the rollout server, up to `tries` times: arm B iteration 6 (2026-10-29 12:28) lost its run to one failed
    start (vLLM's Hub lookup failed and the server hung), after 5 good iterations."""
    for k in range(1, tries + 1):
        try:
            return _start_vllm_once(adapter, util, max_len, seed, log_path)
        except RuntimeError as e:
            print(json.dumps({"vllm_start_failed": k, "error": str(e)}), flush=True)
            if k == tries:
                raise
            time.sleep(10)


def _start_vllm_once(adapter: Path, util: float, max_len: int, seed: int, log_path: Path) -> subprocess.Popen:
    wait_gpu_free()
    vllm = Path(sys.executable).with_name("vllm")
    cmd = [str(vllm), "serve", BASE, "--port", str(PORT), "--host", "127.0.0.1", "--served-model-name", "base",
           "--dtype", "bfloat16", "--max-model-len", str(max_len), "--gpu-memory-utilization", str(util),
           "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
           "--generation-config", "vllm", "--enable-lora", "--max-lora-rank", "64", "--seed", str(seed),
           "--lora-modules", f"policy={adapter}"]
    log = open(log_path, "a")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ), start_new_session=True)
    _LIVE_VLLM.add(p.pid)
    t0 = time.time()
    while time.time() - t0 < 600:                  # a normal start takes 2-3 min
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                stop_vllm(p)
                raise RuntimeError(f"vLLM exited during startup; see {log_path}") from None
            time.sleep(3)
    stop_vllm(p)
    raise RuntimeError("vLLM did not become ready in 600 s")


def stop_vllm(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    try:                                           # stragglers of the session (EngineCore) once the server is gone
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    _LIVE_VLLM.discard(p.pid)
    wait_gpu_free()


_LIVE_VLLM: set[int] = set()     # session ids of running vLLM servers (each is its own session, out of the job's group)


def _on_sigterm(signum, frame):
    """The queue's VRAM guard TERMs the job's process group, which does not reach vLLM (start_new_session); an
    orphaned EngineCore kept 19.6 GB and got the next job killed too (2026-10-29 10:49). Kill the servers, exit."""
    for sid in list(_LIVE_VLLM):
        try:
            os.killpg(sid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(128 + signum)


def gpu_used_mb() -> int | None:
    nvsmi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        out = subprocess.run([nvsmi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def wait_gpu_free(max_used_mb: int = 6000, timeout_s: float = 300.0) -> None:
    """Block until total GPU memory in use is below max_used_mb. Under WSL2 a killed vLLM EngineCore keeps its
    ~16 GB for a while after the process is gone; loading the update model 5 s after stop_vllm put the pilot at 24 GB
    (guard kill, 2026-10-29 02:25)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_used_mb()
        if used is not None and used < max_used_mb:
            print(json.dumps({"gpu_free_wait_s": round(time.time() - t0, 1), "gpu_used_mb": used}), flush=True)
            return
        time.sleep(3)
    raise RuntimeError(f"GPU memory still {gpu_used_mb()} MiB {timeout_s:.0f} s after stopping vLLM")


class VramSampler:
    """Peak GPU memory used (MiB, all processes) from nvidia-smi, sampled every `every` s in a thread."""

    def __init__(self, every: float = 2.0):
        self.every, self.peak, self._stop = every, 0, threading.Event()
        self.nvsmi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run([self.nvsmi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()
                if out:
                    self.peak = max(self.peak, int(out[0]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


# --- update -----------------------------------------------------------------------------------------------------
def make_fresh_lora(out_dir: Path, like: str, seed: int) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = get_peft_model(model, LoraConfig(**fresh_lora_config(REPO / like)))
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out_dir)
    (out_dir / "fresh_lora.json").write_text(json.dumps({"like": like, "seed": seed, **fresh_lora_config(REPO / like)}))
    del model
    return out_dir


def update_isolated(*args) -> dict:
    """update() in a spawned child process, so every byte of its CUDA memory (context, allocator cache, anything a
    reference keeps alive) is returned when it exits. In-process, the trainer kept GPU memory into the next
    iteration's vLLM start and the total passed the 22 GiB guard (arm B iteration 2, 2026-10-29 10:45)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from train.grpo import update as fn          # by module name: `python -m train.grpo` runs as __main__

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def update(adapter_in: Path, adapter_out: Path, ref_adapter: Path | None, eps: list[dict], adv: list[float],
           a, seed: int, it: int) -> dict:
    """`epochs` passes of clipped policy-gradient (or RFT NLL) on the sampled tokens + KL to the reference policy."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(stable_seed("update", seed, it))
    batch = [(e, A) for e, A in zip(eps, adv, strict=True) if A != 0.0 and e.get("token_turns")]
    if not batch:
        return {"updated": False, "reason": "no episode with a non-zero advantage"}
    torch.cuda.reset_peak_memory_stats()
    # cap what the caching allocator may reserve: uncapped, its cache grew past the 22 GiB guard (pilot r5, 24.0 GB
    # by nvidia-smi) although the measured need is <= 15.4 GB (memprobe, 15k tokens); an overrun is a caught OOM
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.torch_cap_gb * 2**30 / total))
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, adapter_in, adapter_name="default", is_trainable=True)
    if ref_adapter is not None:
        model.load_adapter(str(ref_adapter), adapter_name="ref", is_trainable=False)

    def set_trainable():
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n and ".default." in n

    set_trainable()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    st = Path(adapter_in) / "optimizer.pt"
    if st.exists():
        opt.load_state_dict(torch.load(st, map_location="cuda"))
    model.train()                                  # gradient checkpointing only runs in train mode (HF
    for m in model.modules():                      # GradientCheckpointingLayer); eval() stored every activation
        if isinstance(m, torch.nn.Dropout):        # and OOMed the pilot 3x. Dropout off by p = 0 instead.
            m.p = 0.0

    segs = [merge_turns(e["token_turns"], a.max_len) for e, _ in batch]

    def logp_of(s: dict):
        """Log-probs of every completion token in segment s (one forward pass, lm_head only at those positions)."""
        x = torch.tensor([s["ids"]], device="cuda")
        pos = torch.tensor([i for a0, b0 in s["spans"] for i in range(a0 - 1, b0 - 1)], device="cuda")
        tgt = torch.tensor([s["ids"][i + 1] for i in pos.tolist()], device="cuda")
        hidden = model.base_model.model.model(input_ids=x).last_hidden_state[0]
        logits = model.base_model.model.lm_head(hidden[pos]).float()
        return torch.log_softmax(logits, -1).gather(-1, tgt[:, None]).squeeze(-1)

    refs: dict[tuple[int, int], torch.Tensor] = {}
    if a.kl_beta:
        with torch.no_grad():
            for bi in range(len(batch)):
                for ti, t in enumerate(segs[bi]):
                    if ref_adapter is None:
                        with model.disable_adapter():
                            refs[(bi, ti)] = logp_of(t)
                    else:
                        model.set_adapter("ref")
                        refs[(bi, ti)] = logp_of(t)
                        model.set_adapter("default")
        set_trainable()
    L = float(a.len_norm)
    stats = defaultdict(list)
    first = None
    steps = 0
    rng = random.Random(stable_seed("mb", seed, it))
    for ep in range(a.epochs):
        order = list(range(len(batch)))
        rng.shuffle(order)
        for m0 in range(0, len(order), a.mb):
            mb = order[m0:m0 + a.mb]
            opt.zero_grad(set_to_none=True)
            try:
                for bi in mb:
                    A = batch[bi][1]
                    for ti, t in enumerate(segs[bi]):
                        new = logp_of(t)
                        old = torch.tensor(t["logprobs"], device="cuda")
                        loss_t, d = ppo_token_loss(new, old, refs.get((bi, ti)), A, a.clip, a.kl_beta, a.algo,
                                                       a.clip_high)
                        loss = loss_t.sum() / L / len(mb)
                        loss.backward()
                        if first is None:
                            first = float((new.detach() - old).mean())
                        stats["loss"].append(float(loss.detach()) * len(mb))
                        stats["kl"].append(float(d["kl"].mean()))
                        stats["ratio"].append(float(d["ratio"].mean()))
                        stats["clip_frac"].append(float(d["clipped"].float().mean()))
            except torch.OutOfMemoryError:           # the minibatch is dropped whole (no partial gradient step)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                stats["oom_minibatches"].append(1)
                continue
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
            stats["grad_norm"].append(gn)
            opt.step()
            torch.cuda.empty_cache()                   # return fragments between minibatches
            steps += 1
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_out, selected_adapters=["default"])
    if (adapter_out / "default").is_dir():         # PEFT nests non-"default"-only saves; flatten if it did
        for f in (adapter_out / "default").iterdir():
            shutil.move(str(f), adapter_out / f.name)
        (adapter_out / "default").rmdir()
    tok.save_pretrained(adapter_out)
    torch.save(opt.state_dict(), adapter_out / "optimizer.pt")
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt
    torch.cuda.empty_cache()
    mean = lambda k: round(statistics.mean(stats[k]), 5) if stats[k] else None  # noqa: E731
    return {"updated": True, "episodes": len(batch), "segments": sum(map(len, segs)), "optimizer_steps": steps,
            "loss": mean("loss"),
            "kl": mean("kl"), "ratio_mean": mean("ratio"), "clip_frac": mean("clip_frac"),
            "grad_norm": mean("grad_norm"), "logp_mismatch": None if first is None else round(first, 5),
            "torch_peak_mb": round(peak), "oom_minibatches": len(stats["oom_minibatches"])}


# --- logging ----------------------------------------------------------------------------------------------------
def iteration_metrics(eps: list[dict]) -> dict:
    rs = [e["reward"] for e in eps]
    comp = defaultdict(list)
    for e in eps:
        for k, v in e["reward_parts"].items():
            if k != "total":
                comp[k].append(v)
    tools = Counter(c["name"] for e in eps for c in e["calls"])
    hk = aggregate(e["counters"] for e in eps)
    return {"episodes": len(eps), "reward_mean": round(statistics.mean(rs), 4) if rs else None,
            "reward_std": round(statistics.pstdev(rs), 4) if rs else None,
            "reward_components": {k: round(statistics.mean(v), 4) for k, v in sorted(comp.items())},
            "score_mean": round(statistics.mean(e["score"] for e in eps), 4) if eps else None,
            "success_rate": round(sum(e["success"] for e in eps) / len(eps), 4) if eps else None,
            "abstain_rate": hk.get("abstain_rate"), "mean_steps": hk.get("turns_mean"),
            "tool_calls": dict(sorted(tools.items())), "truth_guard_hits": hk.get("truth_guard_hits"),
            "verifier_rejects": hk.get("verifier_rejects"), "tool_timeouts": hk.get("timeouts"),
            "timeout_episode_rate": hk.get("timeout_episode_rate"), "hacking": hk,
            "end_reasons": dict(Counter(e["end_reason"] for e in eps)),
            "token_capture_failures": sum(e.get("bad_turns", 0) for e in eps),
            "per_area_reward": {k: round(statistics.mean(v), 4) for k, v in sorted(_by(eps, "area", "reward").items())}}


def _by(eps, key, val):
    d = defaultdict(list)
    for e in eps:
        d[e[key]].append(e[val])
    return d


def val_metrics(eps: list[dict]) -> dict:
    fam = {k: round(statistics.mean(v), 4) for k, v in sorted(_by(eps, "template", "score").items())}
    return {"score": round(statistics.mean(e["score"] for e in eps), 4), "n": len(eps),
            "macro": round(statistics.mean(fam.values()), 4) if fam else None,
            "success_rate": round(sum(e["success"] for e in eps) / len(eps), 4),
            "abstain_rate": aggregate(e["counters"] for e in eps).get("abstain_rate"), "per_family": fam}


def save_traj(path: Path, eps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for e in eps:
            f.write(json.dumps({k: v for k, v in e.items() if k != "token_turns"}
                               | {"n_token_turns": len(e.get("token_turns") or []),
                                  "completion_tokens": sum(len(t["completion_ids"]) for t in e.get("token_turns") or [])})
                    + "\n")


def git_state() -> dict:
    def run(*c):
        try:
            return subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"git_hash": run("rev-parse", "HEAD"), "dirty": [x[3:] for x in run("status", "--porcelain").splitlines()
                                                           if x and not x.startswith("??")]}


def status(task: str | None, **kw) -> None:
    if not task:
        return
    try:
        from scripts import pipeline_status as ps

        ps.set_program("paper")
        ps.update(task, **kw)
    except Exception:  # noqa: BLE001
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/sft_v11", help="adapter dir, or 'base' for a fresh LoRA")
    ap.add_argument("--lora-like", default=LORA_LIKE, help="--init base: copy rank / alpha / targets from this adapter")
    ap.add_argument("--ref", default=None, help="KL reference adapter (default: the init adapter; base: the bare base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/rl/pool_B.jsonl")
    ap.add_argument("--worlds", default="", help="comma list of world tags to keep from the pool (e.g. M1,M2,M3,M4,W6)")
    ap.add_argument("--world-frac", type=float, default=None, help="share of world-tagged prompts per iteration")
    ap.add_argument("--val", default="data/rl/val400.jsonl")
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--no-final-val", action="store_true")
    ap.add_argument("--algo", choices=["grpo", "rft"], default="grpo")
    ap.add_argument("--rft-min-reward", type=float, default=None,
                    help="RFT filter: also require reward >= this (default: verified-correct only)")
    ap.add_argument("--reward", choices=["v2", "v1"], default="v2")
    ap.add_argument("--no-dist-bonus", action="store_true")
    ap.add_argument("--step-cost", type=float, default=None)
    ap.add_argument("--ablation", default="none", help="eval/ablations.py preset(s) for the env, e.g. with_consistency")
    ap.add_argument("--timeout-scale", type=float, default=1.0,
                    help="multiply every tool wall-clock timeout (AUDIT_CODE C8); timeouts are logged per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=16, help="episodes per optimizer step")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=0.28, help="upper ratio clip (DAPO clip-higher); = --clip: symmetric")
    ap.add_argument("--adv-norm", choices=["batch", "group"], default="batch",
                    help="advantage scale: batch std of group-centred rewards (Lite PPO) or per-group std (GRPO)")
    ap.add_argument("--min-spread", type=float, default=0.05, help="groups with reward std <= this are not trained on")
    ap.add_argument("--kl-beta", type=float, default=0.02, help="0: no KL term and no reference pass (DAPO)")
    ap.add_argument("--sampler", choices=["uniform", "adaptive"], default="uniform",
                    help="adaptive: AdaptiveSampler (family weight by informative-group share, skip flat tasks)")
    ap.add_argument("--min-informative", type=int, default=0,
                    help="top up with more prompts when fewer groups than this have reward std > --min-spread (0 = off)")
    ap.add_argument("--topup-max", type=int, default=12, help="most extra prompts in the one top-up round")
    ap.add_argument("--len-norm", type=float, default=512.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--conc", type=int, default=32, help="parallel episodes (16 used KV 35%% of 10 GB, GPU mostly idle)")
    ap.add_argument("--util", type=float, default=0.75, help="vLLM share; Windows apps use 1-2 GB WSL cannot see")
    ap.add_argument("--max-len", type=int, default=15360)
    ap.add_argument("--torch-cap-gb", type=float, default=19.5, help="update: per-process CUDA allocator cap")
    ap.add_argument("--status-task", default=os.environ.get("PAPER_TASK"))
    return ap.parse_args(argv)


def reward_config(a) -> RewardConfig:
    cfg = RewardConfig(version=a.reward)
    if a.no_dist_bonus:
        cfg = replace(cfg, dist_bonus=False)
    if a.step_cost is not None:
        cfg = replace(cfg, step_cost=a.step_cost)
    return cfg


def main(argv=None) -> int:
    from scripts import cuda_env  # noqa: F401
    from scripts.gpu_lock import gpu_lock

    a = parse_args(argv)
    signal.signal(signal.SIGTERM, _on_sigterm)
    out = REPO / a.out
    (out / "traj").mkdir(parents=True, exist_ok=True)
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    cfg = reward_config(a)
    worlds = {w for w in a.worlds.split(",") if w}
    pool = load_pool(REPO / a.pool, worlds)
    val_rows = [json.loads(x) for x in (REPO / a.val).read_text().splitlines() if x.strip()]
    if not pool:
        raise SystemExit(f"empty pool {a.pool} (worlds {sorted(worlds)})")
    history = read_log(out)
    start, cur = resume_point(out)
    sampler = AdaptiveSampler(pool, a.min_spread)
    for f in sorted((out / "traj").glob("iter_*.jsonl.gz")):       # resume: rebuild the sampler's statistics
        if int(f.name[5:8]) < start:
            sampler.observe([json.loads(x) for x in gzip.open(f, "rt", encoding="utf-8")])
    val_base = out / "traj" / "val_iter_001.jsonl.gz"              # paired validation reference (the init policy)
    run_cfg = {"argv": sys.argv, "args": vars(a), "reward_config": cfg.to_dict(), "pool_n": len(pool)}
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        **git_state(), "task": a.status_task, "job": os.environ.get("PAPER_JOB"), "cmd": " ".join(sys.argv),
        "base_model": BASE, "init": a.init, "init_sha256": None if a.init == "base" else
        sha256_file(REPO / a.init / "adapter_model.safetensors"), "pool": a.pool,
        "pool_sha256": sha256_file(REPO / a.pool), "val": a.val, "val_sha256": sha256_file(REPO / a.val),
        "seed": a.seed, "algo": a.algo, "reward_config": cfg.to_dict(), "ablation": a.ablation,
        "worlds": sorted(worlds), "started": time.time(), "wall_s": 0.0, "peak_vram_mb": 0}
    man.setdefault("resumes", []).append({"t": time.time(), "from_iter": start})
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=1, default=str))
    vllm_log = out / "vllm.log"
    t_run = time.time()
    with gpu_lock("rl"):
        if cur is None:
            if a.init == "base":
                cur = str(make_fresh_lora(out / "adapters" / "iter_000", a.lora_like, a.seed).relative_to(REPO))
            else:
                cur = a.init
        ref = None if a.init == "base" else REPO / (a.ref or a.init)
        status(a.status_task, state="running", note=f"{out.name}: iter {start}/{a.iters}")
        log = open(out / "log.jsonl", "a")
        it = start
        for it in range(start, a.iters + 1):
            t0 = time.time()
            rows = (sampler.sample(a.prompts, a.seed, it) if a.sampler == "adaptive"
                    else sample_prompts(pool, a.prompts, a.seed, it, a.world_frac if worlds else None))
            val, topup = None, 0
            with VramSampler() as vs:
                srv = start_vllm(REPO / cur, a.util, a.max_len, stable_seed("vllm", a.seed, it), vllm_log)
                try:
                    eps = rollouts(rows, a.group, a.temperature, a.conc, a.seed, it, cfg, a.reward, a.ablation,
                                   timeout_scale=a.timeout_scale)
                    inf = group_advantages(eps, a.algo, a.rft_min_reward, a.adv_norm, a.min_spread)[1]["groups_trained"]
                    if a.min_informative and inf < a.min_informative:       # DAPO dynamic sampling, one round
                        k = min(a.topup_max, 2 * (a.min_informative - inf))
                        extra = sampler.sample(k, a.seed, it, exclude={r["task_id"] for r in rows})
                        eps += rollouts(extra, a.group, a.temperature, a.conc, a.seed, it, cfg, a.reward,
                                        a.ablation, timeout_scale=a.timeout_scale, offset=len(rows))
                        rows, topup = rows + extra, len(extra)
                    sampler.observe(eps)
                    t_roll = time.time() - t0
                    t_v = time.time()
                    if it == 1 or (it - 1) % a.val_every == 0:
                        veps = rollouts(val_rows, 1, 0.0, a.conc, a.seed, it, cfg, a.reward, a.ablation, tag="val",
                                         timeout_scale=a.timeout_scale)
                        val = val_metrics(veps)
                        save_traj(out / "traj" / f"val_iter_{it:03d}.jsonl.gz", veps)
                        if it > 1 and val_base.exists():
                            val["paired_vs_iter1"] = paired_delta(
                                [json.loads(x) for x in gzip.open(val_base, "rt", encoding="utf-8")], veps)
                    t_val = time.time() - t_v
                finally:
                    stop_vllm(srv)
                save_traj(out / "traj" / f"iter_{it:03d}.jsonl.gz", eps)
                adv, spread = group_advantages(eps, a.algo, a.rft_min_reward, a.adv_norm, a.min_spread)
                nxt = out / "adapters" / f"iter_{it:03d}"
                t1 = time.time()
                upd = update_isolated(REPO / cur, nxt, ref, eps, adv, a, a.seed, it)
                t_upd = time.time() - t1
            before = cur
            if upd.get("updated"):
                cur = str(nxt.relative_to(REPO))
            rec = {"iter": it, "seed": a.seed, "algo": a.algo, "prompts": len(rows), "topup_prompts": topup,
                   "family_weights": sampler.weights() if a.sampler == "adaptive" else None, **iteration_metrics(eps),
                   **spread, **upd, "rollout_s": round(t_roll), "val_s": round(t_val), "update_s": round(t_upd),
                   "wall_s": round(time.time() - t0), "peak_vram_mb": vs.peak, "adapter_before": before,
                   "adapter_after": cur}
            if val:
                rec.update({"val_score": val["score"], "val": val, "val_adapter": before})
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: rec[k] for k in ("iter", "reward_mean", "success_rate", "abstain_rate",
                                                  "spread_share", "wall_s") if k in rec}
                             | ({"val_score": rec["val_score"]} if val else {})), flush=True)
            _keep_best(out, history)
            _prune(out, history, cur)
            man["wall_s"] = round(man.get("wall_s", 0) + rec["wall_s"], 1)
            man["peak_vram_mb"] = max(man.get("peak_vram_mb", 0), vs.peak)
            man_path.write_text(json.dumps(man, indent=1))
            walls = [h["wall_s"] for h in history if h.get("wall_s")]       # ETA from this run's own iterations
            status(a.status_task, state="running", pct=round(100 * it / a.iters),
                   eta=time.time() + statistics.mean(walls) * (a.iters - it) + rec.get("val_s", 0) if walls else None,
                   note=f"{out.name}: iter {it}/{a.iters} reward {rec['reward_mean']}")
        if not a.no_final_val and not any(h.get("final") for h in history):
            with VramSampler() as vs:
                srv = start_vllm(REPO / cur, a.util, a.max_len, stable_seed("vllm", a.seed, "final"), vllm_log)
                try:
                    veps = rollouts(val_rows, 1, 0.0, a.conc, a.seed, a.iters + 1, cfg, a.reward, a.ablation,
                                    tag="val", timeout_scale=a.timeout_scale)
                finally:
                    stop_vllm(srv)
            save_traj(out / "traj" / f"val_iter_{a.iters + 1:03d}.jsonl.gz", veps)
            val = val_metrics(veps)
            if val_base.exists():
                val["paired_vs_iter1"] = paired_delta(
                    [json.loads(x) for x in gzip.open(val_base, "rt", encoding="utf-8")], veps)
            rec = {"iter": a.iters + 1, "final": True, "val_score": val["score"], "val": val, "val_adapter": cur,
                   "peak_vram_mb": vs.peak}
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            _keep_best(out, history)
    man["ended"] = time.time()
    man["run_wall_s"] = round(time.time() - t_run, 1)
    man["best"] = json.loads((out / "best.json").read_text()) if (out / "best.json").exists() else None
    man_path.write_text(json.dumps(man, indent=1))
    (out / "final.txt").write_text(str(cur))
    status(a.status_task, state="running", note=f"{out.name}: done ({a.iters} iterations)")
    return 0


def _keep_best(out: Path, history: list[dict]) -> None:
    best = pick_best(history)
    if best is None:
        return
    cur = json.loads((out / "best.json").read_text()) if (out / "best.json").exists() else None
    if cur and cur.get("iter") == best["iter"]:
        return
    src = REPO / best["val_adapter"]
    dst = out / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt"))
    (out / "best.json").write_text(json.dumps({"iter": best["iter"], "val_score": best["val_score"],
                                               "adapter": best["val_adapter"],
                                               "rule": "best val mean score, ties to the earlier iteration"}))


def _prune(out: Path, history: list[dict], cur: str) -> None:
    """Keep validated adapters, the newest, iter_000 (fresh init) and the init; drop the optimizer state of all but
    the newest adapter."""
    keep = {cur} | {h.get("val_adapter") for h in history if h.get("val_adapter")}
    for d in sorted((out / "adapters").glob("iter_*")):
        rel = str(d.relative_to(REPO))
        if rel == cur:
            continue
        if rel in keep or d.name == "iter_000":
            (d / "optimizer.pt").unlink(missing_ok=True)
            continue
        # the next validation evaluates the adapter produced right before it: keep it until then
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
