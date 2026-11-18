#!/bin/bash
# Step 1 (CPU): the agent suite. It also builds and caches the 3x8 and 6x12
# stage-1 datasets, so the training jobs that depend on it never race to
# build the same cache directory at once.
#SBATCH --job-name=agent-validate
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME_CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python validate_agent.py     # must end "44/44 checks passed"; a non-zero exit cancels the training jobs
