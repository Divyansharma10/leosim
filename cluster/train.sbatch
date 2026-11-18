#!/bin/bash
# Step 2 (GPU): the four training runs as one array job, one run per task.
#   0  primary    6x12 krishnan8, loads 0.5/1/2, kappa 0.01
#   1  kappa sweep, 0.001
#   2  kappa sweep, 0.1
#   3  secondary  6x12 populous20, load 1 only
# Outputs: results/agent/<tag>/ ; checkpoints: cache/agent/<tag>.pt (~330 MB,
# rewritten at every evaluation). If a task is pre-empted or times out,
# resubmit just that index (sbatch --array=N cluster/train.sbatch): it resumes from its checkpoint.
#SBATCH --job-name=agent-train
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
  0) ARGS="--kappa 0.01" ;;
  1) ARGS="--kappa 0.001" ;;
  2) ARGS="--kappa 0.1" ;;
  3) ARGS="--kappa 0.01 --gateway-sites populous20 --loads 1" ;;
  *) echo "unknown array index $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

# --auto-resume continues from cache/agent/<tag>.pt when it exists.
nvidia-smi --query-gpu=name,memory.total --format=csv
python run_agent.py --device cuda --threads "$SLURM_CPUS_PER_TASK" --auto-resume $ARGS
