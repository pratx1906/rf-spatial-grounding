#!/usr/bin/env bash
# E6 SELFCAL (docs/paper/SELFCAL_PREREG.md): anchored self-calibration, 2 rebuild passes, tasks, eval, pairing.
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
STAGES=${STAGES:-"plan1 build1 plan2 build2 tasks eval pair"}
TAGS=${TAGS:-"M2 M0 M1 M6 M7 M5 M8"}
W=runs/s82; EV=runs/eval/s82; O=$W/e6/overrides; SC=data/scenarios_v3; L=$W/logs; mkdir -p $L $EV
log() { echo "[e6 $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
lockwait() { until [ "$($P scripts/gpu_lock.py status)" = idle ] && ! (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; do sleep 30; done; }
build() {  # <srctag-suffix-from> <newtag-suffix>: parallel builds (JOBS at a time) of every planned scenario, then copies
  local SUF=$1 FROM=$2 items=""
  for T in $TAGS; do NT=${T}${SUF}
    for s in $($P -c "import json; print(' '.join(str(s) for s in json.load(open('$W/e6/plan.json')).get('$NT', [])))"); do items="$items $NT:$s"; done
  done
  echo $items | tr ' ' '
' | grep . | $P scripts/gpu_lock.py run df_sim -- xargs -P ${JOBS:-4} -I{} bash -c 'bash runs/s82/e6_build_one.sh ${1%%:*} ${1#*:}' _ {}
  for T in $TAGS; do NT=${T}${SUF}; ST=${T}${FROM}
    for s in $(seq 51 60); do
      [ -d $SC/kandla_s${s}_X_${NT} ] || { [ -d $SC/kandla_s${s}_X_${ST} ] && cp -r $SC/kandla_s${s}_X_${ST} $SC/kandla_s${s}_X_${NT}; }
    done
  done
  log "build *${SUF} done"
}
if has plan1; then $P $W/e6_plan.py --pass 1 --tags $TAGS >> $L/e6_plan1.log 2>&1 && log "plan1 done" || { log "plan1 FAILED"; exit 1; }; fi
if has build1; then lockwait; build p ""; fi
if has plan2; then $P $W/e6_plan.py --pass 2 --tags $TAGS >> $L/e6_plan2.log 2>&1 && log "plan2 done" || { log "plan2 FAILED"; exit 1; }; fi
if has build2; then lockwait; build j p; fi
if has tasks; then
  for T in $TAGS; do NT=${T}j; seeds="51-60"
    for try in 1 2 3; do
      [ -f data/tasks/x3_${NT}_51/test.jsonl ] && break
      $P scripts/make_tasks.py --obs v2 --seeds "$seeds" --k 25 --tag $NT --out-dir data/tasks/x3_${NT}_51 > $L/e6_tasks_$NT.log 2>&1 && break
      last=$(grep -oE "^seed [0-9]+" $L/e6_tasks_$NT.log | tail -1 | grep -oE "[0-9]+" || true)
      read bad seeds < <($P -c "import sys; cur=[int(x) for x in sys.argv[1].replace('51-60',','.join(map(str,range(51,61)))).split(',')]; l=sys.argv[2]; i=cur.index(int(l))+1 if l else 0; b=cur[i]; print(b, ','.join(str(x) for x in cur if x!=b))" "$seeds" "$last")
      dropped="${dropped:+$dropped,}$bad"; log "tasks $NT: seed $bad has no drone tracks, retry without it"
    done
    $P runs/s81/sweep_subset.py $NT data/tasks/x3_${NT}_51/test.jsonl $W/e6/v6_${NT}c_51s.jsonl
    [ -d eval/frozen/v6_${NT}c_51s ] || $P -m eval.frozen_build --version v6_${NT}c_51s --copy $W/e6/v6_${NT}c_51s.jsonl > /dev/null
    dropped=""
  done
  log "tasks done"
fi
SW=""; for T in $TAGS; do SW="$SW ${T}jc"; done
if has eval; then
  lockwait
  ( for k in $SW; do $P -m eval.run --policy expert --set eval/frozen/v6_${k}_51s/tasks.jsonl --out $EV/e6_expert_$k.json > $EV/e6_expert_$k.log 2>&1; done; log "expert done" ) &
  EXP=$!
  lg=$L/llm_e6.log; : > "$lg"
  setsid nohup $P -m serve.llm --spec-ngram 4 --max-loras 2 --lora sft14=checkpoints/sft_v14 --lora sft14s1=checkpoints/sft_v14_s1 \
    >> "$lg" 2>&1 < /dev/null & echo $! > "$lg.pid"
  for _ in $(seq 150); do grep -q "Application startup complete" "$lg" && break; grep -q Traceback "$lg" && { log "vLLM FAILED"; exit 1; }; sleep 4; done
  log "vLLM up"
  one() { local o=$EV/e6_$1_$2.json
    $P -m eval.run --policy model --served "$1" --model-id "$1" --set "$3" --concurrency 16 --out "$o" > "$o.log" 2>&1
    $P $W/manifest.py E6 "$o" -- eval.run --served "$1" --set "$3"
    log "$o $($P -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']['_all']; print('n', s['n'], 'score', s['score'], 'errors', sum(1 for r in d['rows'] if r['error']))" "$o" 2>&1 | tail -1)"; }
  for k in $SW; do f=eval/frozen/v6_${k}_51s/tasks.jsonl; one sft14 $k $f & a=$!; one sft14s1 $k $f & b=$!; wait $a $b; done
  pid=$(cat "$lg.pid"); kill -INT -- -"$pid" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$pid" 2>/dev/null || true
  log "vLLM down"; wait $EXP
fi
if has pair; then
  for m in expert sft14 sft14s1; do for k in $SW; do
    $P -m eval.pairing runs/eval/s81/${m}_Npair.json $EV/e6_${m}_$k.json --out $EV/e6_pair_${m}_$k.json > /dev/null 2>> $L/e6_pair.log \
      && log "pair $m $k ok" || log "pair $m $k FAILED"
  done; done
fi
log "done: $STAGES"
