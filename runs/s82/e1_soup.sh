#!/usr/bin/env bash
# E1 SOUP eval (docs/paper/SOUP_PREREG.md): seeds 0/1/2 + soup in ONE vLLM session; 2 adapters at a time
# (the 3x16 concurrent load segfaulted vLLM on 2026-11-07 04:54).
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
W=runs/s82; EV=runs/eval/s82; mkdir -p $W/logs $EV
log() { echo "[e1 $(date +%H:%M:%S)] $*"; }
until [ "$($P scripts/gpu_lock.py status)" = idle ]; do sleep 30; done
if (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; then log "port 8001 busy"; exit 1; fi
lg=$W/logs/llm_e1.log; : > "$lg"
setsid nohup $P -m serve.llm --spec-ngram 4 --max-loras 2 --max-lora-rank 128 \
  --lora s0=checkpoints/sft_v14 --lora s1=checkpoints/sft_v14_s1 --lora s2=checkpoints/sft_v14_s2 \
  --lora soup=checkpoints/sft_v14_soup >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
for _ in $(seq 150); do grep -q "Application startup complete" "$lg" && break; grep -q Traceback "$lg" && { log "vLLM FAILED"; exit 1; }; sleep 4; done
log "vLLM up"
one() { local o=$EV/e1_$1_$2.json
  $P -m eval.run --policy model --served "$1" --model-id "e1_$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
  $P runs/s82/manifest.py E1 "$o" -- eval.run --served "$1" --set "$3" checkpoints/$([ $1 = soup ] && echo sft_v14_soup || ([ $1 = s0 ] && echo sft_v14 || echo sft_v14_$1))
  log "$o $($P -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']['_all']; print('n', s['n'], 'score', s['score'], 'errors', sum(1 for r in d['rows'] if r['error']))" "$o" 2>&1 | tail -1)"; }
for S in "F:runs/s80/prereg/fresh_heldout.jsonl" "heldout:runs/s79/v14/heldout_tasks.jsonl" "v6:eval/frozen/v6/tasks.jsonl"; do
  n=${S%%:*}; f=${S#*:}
  one soup $n $f & a=$!; one s0 $n $f & b=$!; wait $a $b
  one s1 $n $f & a=$!; one s2 $n $f & b=$!; wait $a $b
done
pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$pid" 2>/dev/null || true
log "done"
