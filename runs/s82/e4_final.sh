#!/usr/bin/env bash
# E4 final one-session eval (runs/s82/PLAN.md; SOUP/GRPO14 pre-regs): every model on v6, F, old held-out, W4, W5 in ONE
# vLLM session, 2 adapters at a time (3x16 segfaulted vLLM on 2026-11-07). Outputs runs/eval/s82/e4_<model>_<set>.json.
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
W=runs/s82; EV=runs/eval/s82; L=$W/logs; mkdir -p $L $EV
log() { echo "[e4 $(date +%H:%M:%S)] $*"; }
until [ "$($P scripts/gpu_lock.py status)" = idle ] && ! (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; do sleep 30; done
G14=$($P -c "import json; print(json.load(open('runs/paper/results/S82/B3v14_s0/best.json'))['adapter'])" 2>/dev/null || true)
declare -A AD=( [v11]=checkpoints/sft_v11 [s0]=checkpoints/sft_v14 [s1]=checkpoints/sft_v14_s1 [s2]=checkpoints/sft_v14_s2
                [soup]=checkpoints/sft_v14_soup [rlb3]=runs/paper/results/P4/B3_s0/adapters/iter_010 )
[ -n "$G14" ] && [ -d "$G14" ] && AD[grpo14]=$G14 || log "GRPO14 adapter missing, skipped"
MODELS="v11 s0 s1 s2 soup rlb3 ${AD[grpo14]:+grpo14}"
LO=""; for m in $MODELS; do LO="$LO --lora $m=${AD[$m]}"; done
$P $W/manifest.py E4 - -- serve $(for m in $MODELS; do echo -n "${AD[$m]} "; done)
lg=$L/llm_e4.log; : > "$lg"
setsid nohup $P -m serve.llm --spec-ngram 4 --max-loras 2 --max-lora-rank 128 $LO >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
for _ in $(seq 150); do grep -q "Application startup complete" "$lg" && break; grep -q Traceback "$lg" && { log "vLLM FAILED"; exit 1; }; sleep 4; done
log "vLLM up: $MODELS"
one() { local o=$EV/e4_$1_$2.json
  [ -f "$o" ] && return
  $P -m eval.run --policy model --served "$1" --model-id "e4_$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
  $P $W/manifest.py E4 "$o" -- eval.run --served "$1" --set "$3" "${AD[$1]}"
  log "$o $($P -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']['_all']; print('n', s['n'], 'score', s['score'], 'errors', sum(1 for r in d['rows'] if r['error']))" "$o" 2>&1 | tail -1)"; }
SETS="F:runs/s80/prereg/fresh_heldout.jsonl heldout:runs/s79/v14/heldout_tasks.jsonl v6:eval/frozen/v6/tasks.jsonl W4:eval/frozen/v6_W4/tasks.jsonl W5:eval/frozen/v6_W5/tasks.jsonl"
for S in $SETS; do n=${S%%:*}; f=${S#*:}
  set -- $MODELS
  while [ $# -gt 0 ]; do
    if [ $# -ge 2 ]; then one $1 $n $f & a=$!; one $2 $n $f & b=$!; wait $a $b; shift 2; else one $1 $n $f; shift; fi
  done
done
pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$pid" 2>/dev/null || true
log "done"
