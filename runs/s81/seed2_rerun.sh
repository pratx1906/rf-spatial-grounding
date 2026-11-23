#!/usr/bin/env bash
# Re-run SFT v1.4 seed-2 evals after the vLLM segfault at 04:54 (queue2, 3 evals x 16 concurrent, KV 99%).
# Same flags, one eval at a time. Crashed outputs kept in runs/eval/s81/crashed/.
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
EV=runs/eval/s81; W=runs/s81
log() { echo "[s2rerun $(date +%H:%M:%S)] $*"; }
until grep -q "\[s81 .*done:" $W/queue2.log; do sleep 30; done
until ! (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; do sleep 15; done
for n in v6 heldout F; do [ -f $EV/sft14s2_$n.json ] && mv $EV/sft14s2_$n.json* $EV/crashed/ 2>/dev/null; done
lg=$W/logs/llm_s2.log; : > "$lg"
setsid nohup $P -m serve.llm --spec-ngram 4 --lora sft14s2=checkpoints/sft_v14_s2 >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
for _ in $(seq 150); do grep -q "Application startup complete" "$lg" && break; grep -q Traceback "$lg" && { log "vLLM FAILED"; exit 1; }; sleep 4; done
log "vLLM up"
for S in "v6:eval/frozen/v6/tasks.jsonl" "heldout:runs/s79/v14/heldout_tasks.jsonl" "F:runs/s80/prereg/fresh_heldout.jsonl"; do
  n=${S%%:*}; f=${S#*:}; o=$EV/sft14s2_$n.json
  $P -m eval.run --policy model --served sft14s2 --model-id sft14s2 --set "$f" --concurrency 16 --out "$o" > "$o.log" 2>&1
  log "$o $($P -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']['_all']; print('n', s['n'], 'score', s['score'], 'errors', sum(1 for r in d['rows'] if r['error']))" "$o" 2>&1 | tail -1)"
done
pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$pid" 2>/dev/null || true
log "done"
