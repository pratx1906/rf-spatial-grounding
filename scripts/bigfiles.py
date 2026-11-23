"""Keep files above GitHub's 100 MB limit in the repo as <95 MB parts, with SHA-256 checks.

    python scripts/bigfiles.py split    # before committing: writes <file>.partNN + bigfiles.json
    python scripts/bigfiles.py join     # after cloning: rebuilds every file and verifies its SHA-256

The originals are gitignored; only the parts and the manifest (bigfiles.json at the repo root) are committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "bigfiles.json"
PART = 95 * 1024 * 1024
FILES = [
    "checkpoints/sft_v1/adapter_model.safetensors",      # the trained SceneScout SFT v1 LoRA adapter
    "checkpoints/sft_v11/adapter_model.safetensors",     # SFT v1.1 (robustness continuation)
    "checkpoints/sft_v12/adapter_model.safetensors",     # SFT v1.2 (v2 data + v1.1 replay, demo-v2 S2.5)
    "checkpoints/sft_v13/adapter_model.safetensors",     # SFT v1.3 (v1.2 + height-varied siting; not demo: fails v3 gate)
    "checkpoints/sft_v14/adapter_model.safetensors",     # SFT v1.4 (v3 data, varied wording; not demo: fails held-out gate, D-037)
    "checkpoints/sft_v14_s1/adapter_model.safetensors",  # SFT v1.4 recipe, training seed 1 (paper N1, 2026-11-01)
    "checkpoints/sft_v15a/adapter_model.safetensors",    # T-FIX exploratory T4x: v1.4 + check->caveat data (2026-11-01)
    "checkpoints/parser_v1/adapter_model.safetensors",   # fine-tuned parser + expert baseline (2026-11-02)
    "checkpoints/p8_k3/adapter_model.safetensors",     # P8 data scaling: SFT from base on 3 train seeds
    "checkpoints/p8_k6/adapter_model.safetensors",     # P8 data scaling: SFT from base on 6 train seeds
    "checkpoints/p8_k12/adapter_model.safetensors",    # P8 data scaling: SFT from base on 12 train seeds
    "checkpoints/p8_k24/adapter_model.safetensors",    # P8 data scaling: SFT from base on 24 train seeds
    "video/out/demo_60.mp4",                             # the 60 s demo video (D-036)
    "data/site_kandla/blender/kandla_world.blend",       # the Blender world (source of the console's 3D world)
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split() -> None:
    out = {}
    for rel in FILES:
        src = REPO / rel
        if not src.exists():
            print(f"skip (missing): {rel}")
            continue
        for old in src.parent.glob(src.name + ".part*"):
            old.unlink()
        parts = []
        with open(src, "rb") as f:
            k = 0
            while chunk := f.read(PART):
                p = src.with_name(f"{src.name}.part{k:02d}")
                p.write_bytes(chunk)
                parts.append(p.name)
                k += 1
        out[rel] = {"sha256": sha256(src), "bytes": src.stat().st_size, "parts": parts}
        print(f"split {rel}: {len(parts)} parts")
    MANIFEST.write_text(json.dumps(out, indent=1))


def join() -> int:
    man = json.loads(MANIFEST.read_text())
    bad = 0
    for rel, m in man.items():
        dst = REPO / rel
        if dst.exists() and dst.stat().st_size == m["bytes"] and sha256(dst) == m["sha256"]:
            print(f"ok (already joined): {rel}")
            continue
        with open(dst, "wb") as f:
            for name in m["parts"]:
                f.write((dst.parent / name).read_bytes())
        ok = sha256(dst) == m["sha256"]
        bad += not ok
        print(f"{'ok' if ok else 'CHECKSUM MISMATCH'}: {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "split":
        split()
    elif cmd == "join":
        sys.exit(join())
    else:
        print(__doc__)
        sys.exit(2)
