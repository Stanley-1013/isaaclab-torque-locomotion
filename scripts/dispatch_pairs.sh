#!/usr/bin/env bash
# Sequential multi-task dispatcher for ONE GPU.
#
# Unlike dispatch_seeds.sh (one task, many seeds), this runs an ordered list of (task,prefix,seed)
# triples on a single GPU, one training at a time. Use it to pack BOTH terrain variants onto the
# same GPU without two dispatchers racing for the same card (which risks two trainings on one GPU).
#
# Usage:
#   dispatch_pairs.sh <gpu_id> <task>,<prefix>,<seed> [<task>,<prefix>,<seed> ...]
# Example (detached):
#   nohup setsid scripts/dispatch_pairs.sh 0 \
#     Isaac-Velocity-Rough-Go2-Sata-v0,go2_sr,1 \
#     Isaac-Velocity-Rough-Go2-Sata-Default-v0,go2_dr,1 \
#     </dev/null > results/dispatch_gpu0.log 2>&1 & disown
set -u

GPU="$1"; shift
PAIRS="$*"

REPO=~/workspace/isaaclab-torque-locomotion
ISAACLAB="${ISAACLAB_PATH:-$HOME/workspace/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
FREE_MIB="${FREE_MIB:-2000}"

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

echo "[gpu$GPU] dispatch_pairs start: [$PAIRS] $(date)"
for pair in $PAIRS; do
  IFS=',' read -r TASK PREFIX SEED <<< "$pair"
  LOG="$REPO/results/${PREFIX}_s${SEED}.log"
  if grep -qE "Training time:|iteration $((MAX_ITER-1))/$MAX_ITER" "$LOG" 2>/dev/null; then
    echo "[gpu$GPU] $PREFIX seed $SEED already complete -> skip"; continue
  fi
  wait_gpu_free "$GPU"
  echo "[gpu$GPU] >>> $TASK seed $SEED starting $(date) -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" ./isaaclab.sh -p \
    "$REPO/scripts/train_go2.py" \
    --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" --seed "$SEED" \
    --run_name "${PREFIX}_s${SEED}" \
    < /dev/null > "$LOG" 2>&1
  echo "[gpu$GPU] <<< $PREFIX seed $SEED exit=$? $(date)"
done
echo "[gpu$GPU] ALL DONE: [$PAIRS] $(date)"
