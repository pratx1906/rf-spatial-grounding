#!/usr/bin/env bash
# SFT v1.5b (2026-11-06, user "go", 3 h GPU budget): corrective SFT from SFT v1.4 for the near_object over-abstain.
#   wsl -d Ubuntu-24.04 --cd "C:\Users\Pratyush Dixit\Desktop\Urbexsim" -e bash runs/s80/sft_v15b.sh
# Stages: data (CPU, expert records) -> train (GPU, ~31 steps from checkpoints/sft_v14) -> select (held-out-LIKE val,
# no held-out family) -> eval (v15b on v3, v6, v6_W4, v6_W5, heldout; v1.1 on heldout as a same-server parity check).
set -euo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
STAGES=${STAGES:-"data train select eval"}
W=runs/s80/v15b; EV=runs/eval; SUF=_t7
RUN=checkpoints/sft_v15b_run; OUT=checkpoints/sft_v15b
mkdir -p "$W"
log() { echo "[v15b $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
serve_up() {
  local lg=$1; shift
  if (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; then log "port 8001 busy"; exit 1; fi
  : > "$lg"; setsid nohup $P -m serve.llm "$@" >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
  for _ in $(seq 150); do
    grep -q "Application startup complete" "$lg" && { log "vLLM up"; return 0; }
    grep -q "Traceback" "$lg" && { log "vLLM FAILED, see $lg"; exit 1; }
    sleep 4
  done; log "vLLM not ready"; exit 1
}
serve_down() {
  local pid; pid=$(cat "$1.pid"); kill -INT -- -"$pid" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 -- -"$pid" 2>/dev/null || true; sleep 5; log "vLLM down"
}
ev() {
  $P -m eval.run --policy model --served "$1" --model-id "$2" --set "$3" --concurrency 24 --out "$4" > "$4.log" 2>&1
  log "$4 $($P -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']['_all']; print('n', s['n'], 'score', s['score'], 'macro', s['macro_score'])" "$4")"
}
until [ "$($P scripts/gpu_lock.py status)" = idle ]; do log "waiting for GPU lock"; sleep 60; done
log "stages: $STAGES"

if has data; then $P runs/s80/v15b_data.py 2>&1 | tee "$W/data.log"; fi

if has train; then
  $P -m train.sft --init checkpoints/sft_v14 --data data/sft/train_v15b.jsonl --val data/sft/val_v14.jsonl --out "$RUN" \
     --lr 1e-5 --accum 16 --epochs 1 --max-len 15360 --val-every 10 --save-every 10 2>&1 | tee "$W/train.log"
fi

if has select; then
  SEL=(); LORAS=(); declare -A PATHS
  for d in "$RUN"/checkpoint-*; do [ -d "$d" ] && { s=${d##*-}; LORAS+=(--lora "s$s=$d"); PATHS[step$s]=$d; }; done
  LORAS+=(--lora "sfinal=$RUN"); PATHS[final]=$RUN
  serve_up "$W/llm_select.log" --spec-ngram 4 "${LORAS[@]}"
  for lab in "${!PATHS[@]}"; do
    srv=s${lab#step}; [ "$lab" = final ] && srv=sfinal
    ev "$srv" "sft_v15b_$lab" runs/s79/v14/tasks_val.jsonl "$EV/v15bsel_$lab.json"; SEL+=("$lab=$EV/v15bsel_$lab.json")
  done
  serve_down "$W/llm_select.log"
  $P scripts/sft_v14_report.py select --evals "${SEL[@]}" --out "$W/selected.json"
  pick=$($P -c "import json; print(json.load(open('$W/selected.json'))['selected'])")
  src=$RUN; [ "$pick" != final ] && src=$RUN/checkpoint-${pick#step}
  rm -rf "$OUT"; cp -r "$src" "$OUT"; cp "$RUN"/tokenizer* "$RUN"/special_tokens_map.json "$OUT"/ 2>/dev/null || true
  rm -rf "$OUT"/checkpoint-*; echo "{\"selected\": \"$pick\", \"from\": \"$src\"}" > "$OUT/selected.json"
  log "selected $pick -> $OUT"
fi

if has eval; then
  serve_up "$W/llm_eval.log" --spec-ngram 4 --lora sft15b=$OUT --lora sft11=checkpoints/sft_v11
  ev sft15b sft_v15b runs/s79/v14/heldout_tasks.jsonl "$EV/sft_v15b_heldout$SUF.json"
  ev sft11 sft_v11 runs/s79/v14/heldout_tasks.jsonl "$EV/sft_v11_heldout${SUF}_parity.json"
  for s in v6 v3 v6_W4 v6_W5; do ev sft15b sft_v15b "eval/frozen/$s/tasks.jsonl" "$EV/sft_v15b_$s$SUF.json"; done
  serve_down "$W/llm_eval.log"
fi
log "done: $STAGES"
