#!/usr/bin/env bash
# Run a command and tee its stdout+stderr, one timestamp per line, to runs/<program>/logs/<task>.log
# (the live pipeline dashboard, scripts/pipeline_dashboard.py, shows that log). Exit code = the command's.
# Program: rfconv (default, runs/rfconv) or paper (runs/paper), from `--program <name>` as the first two arguments
# or the env var SCENESCOUT_PIPELINE.
#
#   bash scripts/run_logged.sh T2 python scripts/build_scenarios.py --seeds 0-48
#   bash scripts/run_logged.sh --program paper P1 python -m eval.run ...
#   SCENESCOUT_PIPELINE=paper bash scripts/run_logged.sh P1 python -m eval.run ...
set -o pipefail
program=${SCENESCOUT_PIPELINE:-rfconv}
if [ "${1:-}" = "--program" ]; then program=${2:-}; shift 2; fi
case "$program" in rfconv|paper) ;; *) echo "run_logged.sh: unknown program '$program' (rfconv|paper)" >&2; exit 2;; esac
if [ $# -lt 2 ]; then echo "usage: run_logged.sh [--program rfconv|paper] <task> <cmd...>" >&2; exit 2; fi
task=$1; shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
log="${SCENESCOUT_PIPELINE_LOGS:-$root/runs/$program/logs}/$task.log"     # override: tests only
mkdir -p "$(dirname "$log")"
export PYTHONUNBUFFERED=1
printf '[%(%F %T)T] === START %s: %s\n' -1 "$task" "$*" | tee -a "$log"
"$@" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
  printf '[%(%F %T)T] %s\n' -1 "${line//$'\r'/}"
done | tee -a "$log"
rc=${PIPESTATUS[0]}
printf '[%(%F %T)T] === END %s rc=%s\n' -1 "$task" "$rc" | tee -a "$log"
exit "$rc"
