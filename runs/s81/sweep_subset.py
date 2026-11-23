"""Paired 51s subset for a new mismatch tag: keep the rows whose pair_id is in v6_Npair_51s (same 300 pairs)."""
import json, sys
tag, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = {json.loads(l)["pair_id"] for l in open("eval/frozen/v6_Npair_51s/tasks.jsonl")}
rows = [json.loads(l) for l in open(src)]
keep = [r for r in rows if r.get("pair_id") in ids]
seen = {r["pair_id"] for r in keep}
assert len(seen) == len(keep), "duplicate pair_id"
print(tag, "kept", len(keep), "of", len(ids), "missing", len(ids - seen))
open(out, "w").write("".join(json.dumps(r) + "\n" for r in keep))
