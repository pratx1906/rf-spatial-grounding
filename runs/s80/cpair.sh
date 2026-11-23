#!/usr/bin/env bash
# C-PAIR confirmatory run (docs/paper/CONTRAST_PREREG.md). 2026-11-06.
#   wsl -d Ubuntu-24.04 --cd "C:\Users\Pratyush Dixit\Desktop\Urbexsim" -e bash runs/s80/cpair.sh
set -euo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
STAGES=${STAGES:-"data train eval"}
W=runs/s80/prereg; EV=runs/eval/cpair; mkdir -p "$W" "$EV"
log() { echo "[cpair $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
until [ "$($P scripts/gpu_lock.py status)" = idle ]; do log "waiting for GPU lock"; sleep 60; done
log "stages: $STAGES"

if has data; then $P runs/s80/v15c_data.py 2>&1 | tee "$W/data_c.log"; fi

tr() {  # <init> <data> <out> <seed>
  [ -f "$3/adapter_model.safetensors" ] && { log "$3 exists, skip"; return; }
  log "train $3 (init $1, seed $4)"
  $P -m train.sft --init "$1" --data "$2" --val data/sft/val_v14.jsonl --out "$3" --lr 1e-5 --accum 16 --epochs 1 \
     --max-len 15360 --val-every 10 --stop-at 10 --seed "$4" 2>&1 | tee "$W/train_$(basename "$3").log"
}
if has train; then
  tr checkpoints/sft_v14_s1 data/sft/train_v15b.jsonl checkpoints/cpair_B_s1 1
  tr checkpoints/sft_v14    data/sft/train_v15c.jsonl checkpoints/cpair_C_s0 0
  tr checkpoints/sft_v14_s1 data/sft/train_v15c.jsonl checkpoints/cpair_C_s1 1
fi

if has eval; then
  lg=$W/llm_eval.log
  if (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; then log "port 8001 busy"; exit 1; fi
  : > "$lg"
  setsid nohup $P -m serve.llm --spec-ngram 4 --max-loras 3 --lora A_s0=checkpoints/sft_v14 --lora A_s1=checkpoints/sft_v14_s1 \
    --lora B_s0=checkpoints/sft_v15b --lora B_s1=checkpoints/cpair_B_s1 --lora C_s0=checkpoints/cpair_C_s0 \
    --lora C_s1=checkpoints/cpair_C_s1 >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
  for _ in $(seq 150); do
    grep -q "Application startup complete" "$lg" && break
    grep -q "Traceback" "$lg" && { log "vLLM FAILED"; exit 1; }
    sleep 4
  done
  log "vLLM up"
  one() {  # <model> <setname> <file>
    local o=$EV/$1_$2.json
    $P -m eval.run --policy model --served "$1" --model-id "cpair_$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
    log "$o $($P -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']['_all']; print('n', s['n'], 'score', s['score'])" "$o")"
  }
  for S in "F:$W/fresh_heldout.jsonl" "v6:eval/frozen/v6/tasks.jsonl"; do
    n=${S%%:*}; f=${S#*:}
    for grp in "A_s0 B_s0 C_s0" "A_s1 B_s1 C_s1"; do
      for m in $grp; do one "$m" "$n" "$f" & done; wait
    done
  done
  pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 -- -"$pid" 2>/dev/null || true; log "vLLM down"
fi
log "done: $STAGES"
