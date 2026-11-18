#!/bin/bash
# Step 3 (CPU): the 16 agent mutants. mutants.py REWRITES agent/ files in
# place, so this job is chained to start only after every training task has
# finished (afterany) -- never run it while anything imports agent/ from this
# checkout. -u keeps partial results in the log if the job is killed.
# If it is killed anyway: check for agent/*.mutbak and restore from it
# before doing anything else (cp agent/X.py.mutbak agent/X.py; rm the .mutbak).
#SBATCH --job-name=agent-mutants
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME_CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u mutants.py A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A14 A15 A16
