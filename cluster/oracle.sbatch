#!/bin/bash
# Factored-improvement oracle replication (CPU, no GPU, no training).
# 27 tasks = loads {0.5, 1, 2} x seeds {7, 8, 9} x starts {1200, 1400, 1599};
# each runs the link oracle (~2 min) then the satellite oracle (~10 min).
#   sbatch cluster/oracle.sbatch                                  # replication, default w_max
#   sbatch --array=9-17 --export=ALL,W_MAX=0 cluster/oracle.sbatch   # load 1 only, mean-only reward
# Outputs: results/agent/oracle/runs/*.json; then, from the repo root:
#   python results/agent/oracle/aggregate.py
# Read-only on the source tree: safe to run alongside anything except mutants.py.
#SBATCH --job-name=agent-oracle
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME_CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LOADS=(0.5 1 2); SEEDS=(7 8 9); STARTS=(1200 1400 1599)
i=$SLURM_ARRAY_TASK_ID
LOAD=${LOADS[$((i / 9))]}; SEED=${SEEDS[$(((i / 3) % 3))]}; START=${STARTS[$((i % 3))]}
WMAX_ARG=""
if [ -n "${W_MAX:-}" ]; then WMAX_ARG="--w-max $W_MAX"; fi

for G in link sat; do
  python results/agent/oracle/oracle.py --load "$LOAD" --seed "$SEED" --start "$START" --gran "$G" $WMAX_ARG
done
