"""E1 SOUP: exact average of LoRA weight updates (docs/paper/SOUP_PREREG.md).

Delta W_i = s * B_i A_i with s = alpha / r. Concatenating ranks: A = [A_1; ...; A_n], B = [B_1 ... B_n] / n gives
B A = (1/n) sum B_i A_i; with alpha' = n * alpha and r' = n * r the scale s' = alpha'/r' = s, so the merged update is
exactly the mean update.
    python runs/s82/soup.py checkpoints/sft_v14 checkpoints/sft_v14_s1 checkpoints/sft_v14_s2 --out checkpoints/sft_v14_soup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ins = [Path(x) for x in a.adapters]
    n = len(ins)
    cfgs = [json.loads((p / "adapter_config.json").read_text()) for p in ins]
    for c in cfgs[1:]:
        assert (c["r"], c["lora_alpha"], sorted(c["target_modules"])) == \
               (cfgs[0]["r"], cfgs[0]["lora_alpha"], sorted(cfgs[0]["target_modules"])), "configs differ"
    assert not cfgs[0].get("rank_pattern") and not cfgs[0].get("alpha_pattern")
    sds = [load_file(str(p / "adapter_model.safetensors")) for p in ins]
    keys = sorted(sds[0])
    assert all(sorted(s) == keys for s in sds)
    out = {}
    maxerr = 0.0
    for k in keys:
        if "lora_A" in k:
            out[k] = torch.cat([s[k] for s in sds], dim=0).contiguous()
        elif "lora_B" in k:
            out[k] = (torch.cat([s[k].float() for s in sds], dim=1) / n).to(sds[0][k].dtype).contiguous()
        else:
            raise ValueError(f"unexpected key {k}")
    for k in keys:  # verify on every module
        if "lora_A" in k:
            kb = k.replace("lora_A", "lora_B")
            mean = sum(s[kb].float() @ s[k].float() for s in sds) / n
            got = out[kb].float() @ out[k].float()
            maxerr = max(maxerr, float((got - mean).abs().max() / (mean.abs().max() + 1e-12)))
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg = dict(cfgs[0])
    cfg["r"] = n * cfgs[0]["r"]
    cfg["lora_alpha"] = n * cfgs[0]["lora_alpha"]
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for f in ins[0].iterdir():
        if f.name not in ("adapter_model.safetensors", "adapter_config.json", "train_meta.json") and f.is_file():
            shutil.copy(f, dst / f.name)
    meta = {"kind": "soup (exact mean of LoRA updates, rank concatenation)", "inputs": [str(p) for p in ins],
            "input_sha256_16": [sha(p / "adapter_model.safetensors") for p in ins], "r": cfg["r"],
            "lora_alpha": cfg["lora_alpha"], "max_rel_err_vs_mean_update": maxerr,
            "sha256_16": sha(dst / "adapter_model.safetensors")}
    (dst / "soup_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    assert maxerr < 1e-2, maxerr


if __name__ == "__main__":
    main()
