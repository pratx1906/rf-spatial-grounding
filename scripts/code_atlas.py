"""Build the SceneScout code atlas (interactive module/symbol graph) from graphify's graph.json.

    .venv-graphify\\Scripts\\graphify update .      (offline AST extraction)
    python scripts/code_atlas.py                   -> graphify-out/code_atlas.html

Self-contained page (data embedded; libraries from cdnjs/jsdelivr). Layers are assigned from the file path; the
truth boundary (agent side must never depend on the world side, SPEC 12.1) is checked and drawn.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "graphify-out"

# layer: (match prefixes, label, side)
LAYERS = [
    ("tools", ("tools/",), "Agent tools", "agent"),
    ("harness", ("harness/",), "Harness and verifier", "agent"),
    ("belief", ("sim/belief/",), "Belief side (what the agent may know)", "agent"),
    ("serve", ("serve/",), "Serving and gateway", "agent"),
    ("shared", ("sim/geometry",), "Shared maths (no data access)", "shared"),
    ("world", ("sim/world/", "sim/obs_model/", "sim/df/", "sim/stub_site", "sim/layout", "sim/events"),
     "World simulation", "world"),
    ("tasks", ("tasks/",), "Task generators and expert", "offline"),
    ("train", ("train/",), "Training (SFT, GRPO)", "offline"),
    ("eval", ("eval/",), "Evaluation", "offline"),
    ("envbuild", ("envbuild/",), "Twin and RF build", "offline"),
    ("ui", ("ui/",), "Operator console UI", "agent"),
    ("scripts", ("scripts/",), "Scripts", "offline"),
    ("tests", ("tests/",), "Tests", "offline"),
]
CROSS = {"calls", "imports", "imports_from", "uses", "inherits", "indirect_call", "references"}


def layer_of(path: str) -> str | None:
    for key, prefixes, _, _ in LAYERS:
        if any(path.startswith(p) for p in prefixes):
            return key
    return None


def main() -> None:
    g = json.loads((OUT / "graph.json").read_text(encoding="utf8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    code = {i: n for i, n in nodes.items() if n.get("file_type") == "code" and n.get("source_file")
            and layer_of(n["source_file"]) and Path(n["source_file"]).suffix in (".py", ".js")}
    # module (file) level
    files = Counter(n["source_file"] for n in code.values())
    sym_by_file = defaultdict(list)
    for n in code.values():
        if n["label"] != Path(n["source_file"]).name:
            sym_by_file[n["source_file"]].append(n["label"])
    edges = Counter()
    rels = defaultdict(Counter)
    sym_links = []
    for e in g["links"]:
        a, b = code.get(e["source"]), code.get(e["target"])
        if not a or not b or e.get("relation") not in CROSS:
            continue
        fa, fb = a["source_file"], b["source_file"]
        if fa != fb:
            edges[(fa, fb)] += 1
            rels[(fa, fb)][e["relation"]] += 1
        sym_links.append([e["source"], e["target"], e["relation"]])
    side = {k: s for k, _, _, s in LAYERS}
    mods = [{"id": f, "name": Path(f).name, "path": f, "layer": layer_of(f), "n": c,
             "symbols": sorted(sym_by_file[f])[:60]} for f, c in files.items()]
    links = [{"source": a, "target": b, "w": w, "rel": rels[(a, b)].most_common(1)[0][0],
              "cross": side[layer_of(a)] == "agent" and side[layer_of(b)] == "world"}
             for (a, b), w in edges.items()]
    syms = [{"id": i, "name": n["label"], "path": n["source_file"], "layer": layer_of(n["source_file"]),
             "loc": n.get("source_location", "")} for i, n in code.items()]
    deg = Counter()
    for a, b, _ in sym_links:
        deg[a] += 1
        deg[b] += 1
    for s in syms:
        s["d"] = deg[s["id"]]
    data = {"layers": [{"key": k, "label": lab, "side": s} for k, _, lab, s in LAYERS],
            "modules": mods, "links": links,
            "symbols": syms, "symlinks": [{"source": a, "target": b, "rel": r} for a, b, r in sym_links],
            "commit": g.get("built_at_commit", ""), "crossing": sum(1 for x in links if x["cross"])}
    page = (REPO / "scripts" / "code_atlas_page.html").read_text(encoding="utf8")
    (OUT / "code_atlas.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))),
                                         encoding="utf8")
    print(f"modules {len(mods)}, module links {len(links)}, symbols {len(syms)}, symbol links {len(sym_links)}, "
          f"agent->world crossings {data['crossing']}")


if __name__ == "__main__":
    main()
