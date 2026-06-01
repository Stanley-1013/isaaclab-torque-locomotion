#!/usr/bin/env bash
# Overnight multi-seed training dispatcher for one GPU.
#
# Runs a list of seeds sequentially on a single GPU, one full training at a time.
# Before each seed it waits until that GPU is free (idle memory), so it is safe to
# launch a GPU-0 dispatcher while an earlier standalone run is still finishing there.
# Idempotent: a seed whose result log already shows completion is skipped.
#
# Usage:
#   dispatch_seeds.sh <gpu_id> <task_id> <log_prefix> <seed> [seed ...]
# Example (detached):
#   nohup setsid scripts/dispatch_seeds.sh 1 Isaac-Velocity-Flat-Go2-Torque-v0 \
#     go2_torque 2 6 </dev/null > results/dispatch_gpu1.log 2>&1 & disown
set -u

GPU="$1"; TASK="$2"; PREFIX="$3"; shift 3
SEEDS="$*"

REPO=~/workspace/isaaclab-torque-locomotion
ISAACLAB="${ISAACLAB_PATH:-$HOME/workspace/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-1500}"
FREE_MIB="${FREE_MIB:-2000}"   # treat GPU as free below this used-MiB

source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5
cd "$ISAACLAB"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$FREE_MIB" ]; then break; fi
    echo "[gpu$1] busy (${used} MiB used), waiting..."; sleep 30
  done
}

echo "[gpu$GPU] dispatch start: task=$TASK seeds=[$SEEDS] $(date)"
for s in $SEEDS; do
  LOG="$REPO/results/${PREFIX}_s${s}.log"
  if grep -qE "Training time:|iteration $((MAX_ITER-1))/$MAX_ITER" "$LOG" 2>/dev/null; then
    echo "[gpu$GPU] seed $s already complete -> skip"; continue
  fi
  wait_gpu_free "$GPU"
  echo "[gpu$GPU] >>> seed $s starting $(date) -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" ./isaaclab.sh -p \
    "$REPO/scripts/train_go2.py" \
    --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" --seed "$s" \
    < /dev/null > "$LOG" 2>&1
  echo "[gpu$GPU] <<< seed $s exit=$? $(date)"
done
echo "[gpu$GPU] ALL DONE: [$SEEDS] $(date)"
