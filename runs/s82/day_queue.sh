#!/usr/bin/env bash
# s82 day queue (runs/s82/PLAN.md): after E6 finishes -> E3 GRPO from v1.4 -> E4 final one-session eval (if written).
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
P=.venv-wsl/bin/python
log() { echo "[day $(date +%H:%M:%S)] $*"; }
until grep -q "\[e6 .*done:" runs/s82/e6.log 2>/dev/null; do sleep 60; done
log "E6 finished"
until [ "$($P scripts/gpu_lock.py status)" = idle ] && ! (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; do sleep 30; done
log "E3 GRPO start"
bash runs/s82/e3_grpo.sh > runs/s82/logs/e3_grpo.log 2>&1; log "E3 rc=$?"
if [ -f runs/s82/e4_final.sh ]; then log "E4 start"; bash runs/s82/e4_final.sh > runs/s82/e4.log 2>&1; log "E4 rc=$?"; fi
log "done"
