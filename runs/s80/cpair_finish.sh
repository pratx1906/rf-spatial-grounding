#!/usr/bin/env bash
# Finish C-PAIR evals against the already-running vLLM server (cpair.sh hung in `wait` on the server child).
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
EV=runs/eval/cpair_b; mkdir -p $EV
log() { echo "[cpairB $(date +%H:%M:%S)] $*"; }
one() { local o=$EV/$1_$2.json
  $P -m eval.run --policy model --served "$1" --model-id "cpair_$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
  log "$o $($P -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']['_all']; print('n', s['n'], 'score', s['score'])" "$o" 2>&1 | tail -1)"; }
grp() { local set=$1 f=$2; shift 2; local pids=(); for m in "$@"; do one "$m" "$set" "$f" & pids+=($!); done; wait "${pids[@]}"; }
log start
grp F runs/s80/prereg/fresh_heldout.jsonl A_s1 B_s1 C_s1
grp v6 eval/frozen/v6/tasks.jsonl A_s0 B_s0 C_s0
grp v6 eval/frozen/v6/tasks.jsonl A_s1 B_s1 C_s1
log "done"
