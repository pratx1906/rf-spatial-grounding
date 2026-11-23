#!/usr/bin/env bash
# one E6 scenario build: <newtag> <seed>  (called in parallel by e6.sh build stages)
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim"
export SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
P=.venv-wsl/bin/python; NT=$1; s=$2; mkdir -p runs/s82/locks; export AKSHA_GPU_LOCK=runs/s82/locks/${1}_${2}.lock; SC=data/scenarios_v3; L=runs/s82/logs
[ -d $SC/kandla_s${s}_X_${NT} ] && exit 0
$P scripts/build_scenarios.py --obs v2 --cast varied --seeds $s --hw-override runs/s82/e6/overrides/${NT}_s${s}.yaml --tag $NT \
  > $L/e6_build_${NT}_s${s}.log 2>&1 || { echo "[e6 $(date +%H:%M:%S)] build $NT s$s FAILED"; rm -rf $SC/kandla_s${s}_X_${NT}; exit 1; }
$P scripts/track_class_v2.py apply --dirs kandla_s${s}_X_${NT} >> $L/e6_build_${NT}_s${s}.log 2>&1
echo "[e6 $(date +%H:%M:%S)] built $NT s$s"
