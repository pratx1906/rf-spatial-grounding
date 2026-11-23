#!/usr/bin/env bash
# E3 GRPO from SFT v1.4 (docs/paper/GRPO14_PREREG.md): the paper's RL recipe B3 (runs/paper/rl/arm_B3.sh) unchanged
# except the init (checkpoints/sft_v14 instead of sft_v11). Reward v2 incl. penalty_wrong_abstain 0.5 and
# penalty_missed_violation 2.0 (train/rewards/episode.py). Output runs/paper/results/S82/B3v14_s0/.
export PAPER_TASK=S82
ARM=B3v14 SEED=${SEED:-0} ITERS=${ITERS:-20} POOL=data/rl/pool_B.jsonl LR=${LR:-3e-5} PROMPTS=${PROMPTS:-20} VAL=data/rl/val400.jsonl
source "/mnt/c/Users/Pratyush Dixit/Desktop/Urbexsim/runs/paper/rl/arm_common.sh"
run_arm --init checkpoints/sft_v14 --mb 8 --kl-beta 0 --sampler adaptive --min-informative 16 --topup-max 12
