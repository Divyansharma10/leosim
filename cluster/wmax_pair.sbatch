#!/bin/bash
# The w_max pair (GPU): does removing the max-queue term make the factored
# agent learnable? Same everything except the reward's max weight and kappa,
# which is scaled by the oracle's measured factor so it is the same margin
# relative to the reward:
#   0  w_max = 0, kappa = 0.01 * 0.04904085626035073 (results/agent/oracle/
#      kappa_factor.py over the 72-run replication, link granularity)
#   1  w_max = 1, kappa = 0.01 (control)
# Both: 6x12 krishnan8, load 1, Greedy, shadow reward, 100 episodes, seed 7,
# target_sync 100 (at the default 1000 a 100-episode run gets ~20 target
# copies and Q bootstraps only ~20 steps; the oracle's gains are over 40).
# Read each [eval] line's "Q vs return" as a divergence alarm only: a
# shadow-trained Q's level is not a function of the state (AgentValidationConfig).
# Compare the two on delivered and queue_max (evals.csv), not on gain: gain
# uses each run's own w_max.
#   sbatch cluster/wmax_pair.sbatch            # both, one GPU each, in parallel
#   sbatch --array=1 cluster/wmax_pair.sbatch  # one of them (e.g. resubmit after pre-emption)
# Outputs: results/agent_wmax_pair/<tag>/ ; checkpoints: cache/agent_wmax_pair/<tag>.pt
# --auto-resume continues a pre-empted task from its checkpoint.
#SBATCH --job-name=agent-wmax-pair
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
  0) ARGS="--w-max 0 --kappa 0.0004904085626035073" ;;
  1) ARGS="--w-max 1 --kappa 0.01" ;;
  *) echo "unknown array index $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv
python run_agent.py --device cuda --threads "$SLURM_CPUS_PER_TASK" --auto-resume \
  --planes 6 --per-plane 12 --gateway-sites krishnan8 --loads 1 --selectors Greedy \
  --reward-kind shadow --episodes 100 --seed 7 --target-sync 100 \
  --out results/agent_wmax_pair --ckpt-dir cache/agent_wmax_pair $ARGS
