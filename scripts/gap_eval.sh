#!/bin/bash
# RF living world (D-031): world-variant scenarios -> paired tasks -> maps -> expert + SFT v1.1 evaluations.
set -u
P=.venv-wsl/bin/python
mkdir -p runs/dbg                                   # per-step logs
V="W0 W1t W1 W2 W3"
if [ "${SKIP_G1:-0}" != 1 ]; then echo "G1 scenarios $(date +%H:%M)"
for w in $V; do $P scripts/build_scenarios.py --seeds 15-16 --world-variant $w 2>&1 | grep -E "seed|Error|Traceback"; done
fi
echo "G2 tasks $(date +%H:%M)"
[ "${SKIP_G2:-0}" = 1 ] || for w in N $V; do $P scripts/make_tasks.py --k 25 --seeds 15-16 --world-variant $w --out data/tasks/test_$w.jsonl 2>&1 | tail -1; done
if [ "${SKIP_G2:-0}" != 1 ]; then :; fi
echo "G3 maps $(date +%H:%M)"
[ -f data/site_kandla/rf_cache/maps/ism_2400.npz ] || $P -m envbuild.rf_maps 2>&1 | tail -30
echo "G4 gap sets $(date +%H:%M)"
$P scripts/gap_sets.py --per-area 50 2>&1 | tail -1
echo "G5 expert $(date +%H:%M)"
for w in N $V; do for k in "" df_; do $P -m eval.run --policy expert --model-id scripted-expert --set eval/frozen/v3w/$k$w.jsonl --out runs/eval/expert_v3w_$k$w.json > runs/dbg/eval_expert_$k$w.log 2>&1; done; done
echo "G6 serve $(date +%H:%M)"
: > runs/dbg/llm_gap.log
$P -m serve.llm --lora sft=checkpoints/sft_v11 >> runs/dbg/llm_gap.log 2>&1 &
until grep -q "Application startup complete" runs/dbg/llm_gap.log; do sleep 5; grep -q "Traceback" runs/dbg/llm_gap.log && { echo "SERVE FAILED"; exit 1; }; done
for w in N $V; do
  for k in "" df_; do
    $P -m eval.run --policy model --served sft --model-id sft_v11 --set eval/frozen/v3w/$k$w.jsonl --concurrency 16 --out runs/eval/sft_v11_v3w_$k$w.json > runs/dbg/eval_sft_$k$w.log 2>&1
  done
  echo "  sft $w done $(date +%H:%M)"
done
pkill -f "vllm serve"; pkill -f "serve.llm"; sleep 10
echo "GAP_EVAL_DONE $(date +%H:%M)"
