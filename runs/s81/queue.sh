#!/usr/bin/env bash
# GPU queue after C-PAIR (2026-11-06, user: "run all ... new things or limitations not in the paper"):
#  Q1 third SFT v1.4 training seed (review W14): same recipe as sft_v14_s1 (init v1.1, train_v14, stop at 15), seed 2.
#  Q2 twin heading-bias magnitude sweep (review W4): new worlds M5=1, M6=5, M7=15, M8=30 deg on seeds 51-60; the paired
#     300-question 51s subsets; SFT v1.4 seeds 0 and 1 + scripted expert on N, M0, M5, M1, M6, M2, M7, M8 (one server).
#   wsl -d Ubuntu-24.04 --cd "C:\Users\Pratyush Dixit\Desktop\Urbexsim" -e bash runs/s81/queue.sh
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
STAGES=${STAGES:-"wait seed2 build tasks eval pair"}
W=runs/s81; EV=runs/eval/s81; mkdir -p "$W/logs" "$EV"
log() { echo "[s81 $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
TAGS="M5 M6 M7 M8"

if has wait; then
  until grep -q "\[cpair .*done:" runs/s80/cpair.log 2>/dev/null; do sleep 60; done
  log "C-PAIR finished"
fi
until [ "$($P scripts/gpu_lock.py status)" = idle ]; do sleep 60; done
log "stages: $STAGES"

if has seed2 && [ ! -f checkpoints/sft_v14_s2/adapter_model.safetensors ]; then
  log "train SFT v1.4 seed 2"
  $P -m train.sft --init checkpoints/sft_v11 --data data/sft/train_v14.jsonl --val data/sft/val_v14.jsonl \
     --out checkpoints/sft_v14_s2 --lr 1.5e-5 --accum 16 --epochs 1 --max-len 15360 --val-every 15 --stop-at 15 --seed 2 \
     2>&1 | tee "$W/logs/train_s2.log" | grep -E "^\[sft\] (step|saved|stop)"
fi

if has build; then
  for TAG in $TAGS; do
    todo=""; for s in $(seq 51 60); do [ -d data/scenarios_v3/kandla_s${s}_X_${TAG} ] || todo="$todo,$s"; done; todo=${todo#,}
    log "build $TAG seeds ${todo:-none}"
    [ -n "$todo" ] && $P scripts/build_scenarios.py --obs v2 --cast varied --seeds "$todo" \
        --hw-override configs/hardware/mismatch/$TAG.yaml --tag $TAG >> "$W/logs/build_$TAG.log" 2>&1
    dirs=$(for s in $(seq 51 60); do echo -n "kandla_s${s}_X_${TAG} "; done)
    $P scripts/track_class_v2.py apply --dirs $dirs >> "$W/logs/build_$TAG.log" 2>&1
    log "build $TAG done ($(ls -d data/scenarios_v3/kandla_s5?_X_${TAG} data/scenarios_v3/kandla_s60_X_${TAG} 2>/dev/null | wc -l) dirs)"
  done
fi

if has tasks; then
  for TAG in $TAGS; do
    [ -f data/tasks/x3_${TAG}_51/test.jsonl ] || $P scripts/make_tasks.py --obs v2 --seeds 51-60 --k 25 --tag $TAG \
        --out-dir data/tasks/x3_${TAG}_51 > "$W/logs/tasks_$TAG.log" 2>&1
    $P runs/s81/sweep_subset.py $TAG data/tasks/x3_${TAG}_51/test.jsonl "$W/v6_${TAG}c_51s.jsonl"
    [ -d eval/frozen/v6_${TAG}c_51s ] || $P -m eval.frozen_build --version v6_${TAG}c_51s --copy "$W/v6_${TAG}c_51s.jsonl" > /dev/null
  done
fi

SW="Npair M0c M5c M1c M6c M2c M7c M8c"
setof() { [ "$1" = Npair ] && echo eval/frozen/v6_Npair_51s/tasks.jsonl || echo eval/frozen/v6_${1}_51s/tasks.jsonl; }
if has eval; then
  ( for k in $SW; do $P -m eval.run --policy expert --set "$(setof $k)" --out $EV/expert_$k.json > $EV/expert_$k.log 2>&1; done
    log "expert sweep done" ) &
  EXP=$!
  lg=$W/logs/llm.log; : > "$lg"
  setsid nohup $P -m serve.llm --spec-ngram 4 --max-loras 3 --lora sft14=checkpoints/sft_v14 \
    --lora sft14s1=checkpoints/sft_v14_s1 --lora sft14s2=checkpoints/sft_v14_s2 >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
  for _ in $(seq 150); do grep -q "Application startup complete" "$lg" && break; grep -q Traceback "$lg" && { log "vLLM FAILED"; exit 1; }; sleep 4; done
  log "vLLM up"
  one() { local o=$EV/$1_$2.json
    $P -m eval.run --policy model --served "$1" --model-id "$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
    log "$o $($P -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']['_all']; print('n', s['n'], 'score', s['score'])" "$o" 2>&1 | tail -1)"; }
  for k in $SW; do one sft14 $k "$(setof $k)" & one sft14s1 $k "$(setof $k)" & wait; done
  one sft14s2 v6 eval/frozen/v6/tasks.jsonl & one sft14s2 heldout runs/s79/v14/heldout_tasks.jsonl &
  one sft14s2 F runs/s80/prereg/fresh_heldout.jsonl & wait
  pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$pid" 2>/dev/null || true
  log "vLLM down"; wait $EXP
fi

if has pair; then
  for m in expert sft14 sft14s1; do for k in $SW; do [ $k = Npair ] && continue
    $P -m eval.pairing $EV/${m}_Npair.json $EV/${m}_$k.json --out $EV/pair_${m}_$k.json > /dev/null 2>> "$W/logs/pair.log" \
      && log "pair $m $k ok" || log "pair $m $k FAILED"
  done; done
fi
log "done: $STAGES"
