"""Collect every paper artefact into one local subfolder (gitignored; never pushed).
    python -m eval.paper.package [--out docs/paper/package]
Copies reports, LaTeX + figures, all figures under runs/paper, results (eval rows, summaries, pairing, audits,
RL logs and trajectories; no weights or optimizer states), logs, frozen eval sets, the code that produced them
(tracked + uncommitted), lock files and run scripts. Large items (model adapters, scenario databases, task pools)
are listed with size and sha256 instead of copied. Writes MANIFEST.json (every file, size, sha256) and README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

SKIP_SUFFIX = (".safetensors", ".pt", ".bin", ".pyc", ".aux", ".fls", ".fdb_latexmk", ".blg", ".out")
COPY = {
    "reports": ["docs/paper/RESULTS.md", "docs/paper/EXTRA_EVALS.md", "docs/paper/UNSEEN_AREA.md",
                "docs/paper/EXPERIMENTS.md", "docs/paper/PUBLICATION_QA.md", "docs/paper/CALIBRATED_SITES.md",
                "docs/paper/related_work_notes.md", "docs/paper/SCENESCOUT_6G.md", "docs/paper/TWIN_FIX_PREREG.md", "docs/paper/OVERNIGHT_PLAN_2026-11-01.md", "docs/paper/TWIN_FIX.md", "docs/paper/human_questions",
                "docs/superpowers/plans/2026-10-28-paper-program", "docs/DECISIONS.md", "docs/RESULTS"],
    "paper": ["docs/paper/latex", "docs/paper/refs.bib", "docs/paper/venue"],
    "results": ["runs/paper/results"],
    "logs": ["runs/paper/logs", "runs/paper/status"],
    "figures": ["runs/paper/figpreview"],
    "eval_sets": ["eval/frozen"],
    "code": ["eval", "tasks", "harness", "tools", "train", "serve", "sim", "envbuild", "scripts", "tests",
             "runs/paper/sh", "runs/paper/queue", "runs/paper/rl", "runs/paper/p8", "runs/paper/ladder",
             "runs/paper/g_runs.sh", "runs/paper/na_paired_build.sh", "runs/paper/na_paired_eval.sh",
             "runs/paper/newarea_build.sh", "runs/paper/newarea_eval.sh", "runs/paper/newarea_blender.py",
             "runs/paper/watchdog.py", "requirements", "Makefile", "CLAUDE.md", "docs/SPEC.md",
             "docs/ENVIRONMENT_SPEC.md", "docs/SETUP.md"],
}
POINTERS = {
    "models": ["checkpoints/sft_v11", "checkpoints/sft_v14", "checkpoints/p8_k3", "checkpoints/p8_k6",
               "checkpoints/p8_k12", "checkpoints/p8_k24", "runs/paper/results/P4/B3_s0/best",
               "runs/paper/results/P4/B3_s1/best", "runs/paper/results/P4/A3_s0/best"],
    "raw_data": ["data/scenarios_v3", "data/tasks"],
}


README = """# Paper package (local only; never push)

Built {created} by `python -m eval.paper.package`. All results are simulation only. Every copied file is listed in
`MANIFEST.json` with its source path, size and sha256; the items too large to copy (model adapters, scenario
databases, task pools) are listed under `pointers`, with sha256 for the weights and the scenario databases.

## Layout
| folder | contents |
|---|---|
| `reports/` | the RESULTS.md tables (generated), EXTRA_EVALS, UNSEEN_AREA, EXPERIMENTS, PUBLICATION_QA, the human-question form, the paper-program plan and LEDGER, DECISIONS, earlier milestone results |
| `paper/` | LaTeX source, main.pdf, figures in the paper, refs.bib, venue notes |
| `results/` | per-run eval rows (`eval.json`), summaries, manifests, pairing, audits, RL logs and trajectories; no weights |
| `logs/` | GPU queue, eval and server logs, status files |
| `figures/` | every figure under runs/paper (previews, convergence, maps, Blender and Sionna renders) |
| `eval_sets/` | frozen eval sets (tasks.jsonl + manifests) for every test set |
| `code/` | code that produced everything (working tree incl. uncommitted), run scripts, lock files; `code/_git` holds HEAD, status and the uncommitted diff |
| `models/` | configs of the adapters (weights stay in place; hashes in MANIFEST) |

## Claim -> evidence
| claim | numbers | files |
|---|---|---|
| A trained 4B model uses RF tools to ground answers; prompted models do not | RESULTS 1, 2, 7 | `results/.../P1`, `P2`, `P9full/holm.json` |
| GRPO (B3) does not beat the best SFT | RESULTS 4, 7 (contrast i) | `results/.../P4`, `P9full/` |
| Data scaling with number of training scenarios | RESULTS 3 | `results/.../P8` |
| A wrong twin causes failures; control twin does not (10 seeds) | RESULTS 5, 11 G1 | `results/.../P5/pairing`, `P5/G/g_summary.json` |
| Offering the consistency tool does not reduce twin failure (pre-registered, negative) | RESULTS 7, 11 G1b | `P5/G/g_summary.json`, `P9full/holm.json` |
| No measurable loss in areas never seen in training (paired, pre-registered) | RESULTS 10, UNSEEN_AREA | `results/.../NA/paired.json`, `NA/verify_paired`, figures |
| Robust to rewording (SFT), run noise and RL seed spread | RESULTS 11 G2, G4, G5 | `P5/G/` |
| Test-time compute, behaviour change after RL, failure taxonomy | RESULTS 6, 8, 9 | `results/.../P7`, `P9`, `P9full` |

## Reproduce
Analysis is CPU-only from the files here: `python -m eval.paper.freeze` (RESULTS.md), `eval.paper.na_paired`,
`eval.paper.g_analysis`, `eval.paper.holm`. Evals need the adapters and scenarios listed in MANIFEST pointers; the
commands are in `code/runs/paper/*.sh` and `reports/.../LEDGER.md`.
"""


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _files(src: Path):
    if src.is_file():
        yield src
    elif src.is_dir():
        for p in sorted(src.rglob("*")):
            try:
                p.is_file()
            except OSError:  # WSL symlinks (P5 -> P1 eval dirs; the targets are copied from P1)
                continue
            if p.is_file() and "__pycache__" not in p.parts and "adapters" not in p.parts and not p.name.endswith(SKIP_SUFFIX) \
                    and ".safetensors.part" not in p.name and p.name != "token.txt":
                yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/package")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": [], "pointers": []}
    for sec, srcs in COPY.items():
        for s in srcs:
            for p in _files(Path(s)):
                if out in p.parents or "package" in p.parts and "docs" in p.parts:
                    continue
                d = out / sec / p
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
                man["files"].append({"path": str(d.relative_to(out)).replace("\\", "/"), "src": str(p).replace("\\", "/"),
                                     "bytes": p.stat().st_size, "sha256": _sha(p)})
    for p in sorted(Path("runs/paper").rglob("*")):
        if p.suffix in (".png", ".pdf", ".jpg") and "figpreview" not in p.parts:
            d = out / "figures" / p
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                shutil.copy2(p, d)
    git = out / "code" / "_git"
    git.mkdir(parents=True, exist_ok=True)
    for name, cmd in {"HEAD.txt": ["git", "log", "-1", "--format=%H %ci %s"], "status.txt": ["git", "status", "--short"],
                      "uncommitted.diff": ["git", "diff", "HEAD"]}.items():
        (git / name).write_text(subprocess.run(cmd, capture_output=True, text=True).stdout, encoding="utf-8", newline="\n")
    for sec, srcs in POINTERS.items():
        for s in srcs:
            p = Path(s)
            if not p.exists():
                continue
            fs = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
            weights = [f for f in fs if f.name.startswith("adapter_model") or f.suffix == ".duckdb"]
            man["pointers"].append({"section": sec, "path": s, "files": len(fs), "bytes": sum(f.stat().st_size for f in fs),
                                    "sha256": {str(f).replace("\\", "/"): _sha(f) for f in weights[:200]}})
            if sec == "models":
                for f in fs:
                    if f.suffix in (".json", ".txt", ".md", ".jsonl") and f.stat().st_size < 5_000_000:
                        d = out / "models" / f
                        d.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, d)
    (out / "README.md").write_text(README.format(created=man["created"]), encoding="utf-8", newline="\n")
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    tot = sum(f["bytes"] for f in man["files"])
    print(out, len(man["files"]), "files", round(tot / 1e6, 1), "MB;", len(man["pointers"]), "pointers")


if __name__ == "__main__":
    main()
