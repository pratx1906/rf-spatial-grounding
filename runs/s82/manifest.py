"""Append one run record to runs/s82/manifest.jsonl.
    python runs/s82/manifest.py <exp> <out> -- <command...>   (inputs hashed: any existing path args)"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
exp, out = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def h(p):
    p = R / p
    if p.is_dir() and (p / "adapter_model.safetensors").exists(): p = p / "adapter_model.safetensors"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None
ins = {a.split("=", 1)[-1]: h(a.split("=", 1)[-1]) for a in cmd if h(a.split("=", 1)[-1])}
git = lambda *a: subprocess.run(["git", *a], cwd=R, capture_output=True, text=True).stdout.strip()
rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "exp": exp, "out": out, "out_sha16": h(out), "cmd": cmd,
       "inputs_sha16": ins, "git_head": git("rev-parse", "--short", "HEAD"),
       "git_dirty": [l[3:] for l in git("status", "--porcelain", "-uno").splitlines()]}
with open(R / "runs/s82/manifest.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
