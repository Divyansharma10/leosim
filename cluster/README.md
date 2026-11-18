# Running Phase 3 step 2 (residual DDQN) on a Slurm cluster

Written for Slurm. On PBS/LSF the job bodies are the same; only the `#SBATCH`
headers and the dependency flags in `submit_all.sh` change.

## 1. Get the code there

The agent work is not committed yet. Either commit and push it, or copy the
checkout (the local `cache/` is not needed -- the cluster builds its own):

```bash
rsync -av --exclude cache/ --exclude .venv/ --exclude __pycache__/ ./ USER@CLUSTER:~/leosim/
```

## 2. One-time setup (login node)

```bash
cd ~/leosim
bash cluster/setup.sh          # .venv + requirements (torch with CUDA 12)
```

## 3. Edit the job headers

In `validate.sbatch`, `train.sbatch` and `mutants.sbatch`, replace
`CHANGE_ME` (account) and `CHANGE_ME_CPU` / `CHANGE_ME_GPU` (partitions).
`sinfo -s` lists partitions; `sacctmgr show assoc user=$USER` your accounts.
If GPUs are requested by type, change `--gres=gpu:1` to e.g. `--gres=gpu:a100:1`.

## 4. Submit

```bash
bash cluster/submit_all.sh
```

| job | resource | time | what |
|---|---|---|---|
| `agent-validate` | CPU | ~1-5 min | `validate_agent.py`; must pass 44/44, else training never starts. Also builds the 6x12 stage-1 cache once. |
| `agent-train` 0-3 | 1 GPU each | est. <= 1.5 h (4 h requested) | 0 primary kappa 0.01 / 1 kappa 0.001 / 2 kappa 0.1 / 3 populous20 load 1 |
| `agent-mutants` | CPU | ~20 min | 16 `A*` mutants; starts after training ends, because it rewrites `agent/` in place |

Watch with `squeue -u $USER`; logs are in `logs/`. Each training log prints
one line per episode and an `[eval]` line every 10 episodes (gain over
backpressure per load, shortest-path's gain, share of links overridden).

## 5. If something stops

- **Training task pre-empted or timed out:** `sbatch --array=N cluster/train.sbatch`.
  It resumes from `cache/agent/<tag>.pt` (written at every evaluation).
- **Mutant job killed:** look for `agent/*.mutbak` FIRST. If one exists, the
  file next to it is mutated: `cp agent/X.py.mutbak agent/X.py && rm agent/X.py.mutbak`.
  `mutants.py` refuses to start while a `.mutbak` exists.
- **Validate failed:** read `logs/agent-validate-*.out`; the FAIL lines give
  the measured value and the bound. Do not train on a red suite.

## 6. Bring back

```bash
rsync -av USER@CLUSTER:~/leosim/results/agent/ results/agent/
rsync -av USER@CLUSTER:~/leosim/logs/ logs/
```

Per run: `results/agent/<tag>/summary.txt`, `evals.csv`, `curve.csv`,
`learning_curve.png` (gitignored: `git add -f` to commit it), plus the
mutant log `logs/agent-mutants-*.out`.

## Reproducibility on GPU

Same seed on CPU reproduces bit for bit (the suite checks this, on CPU). On
GPU expect agreement to float tolerance, not bytes: cuBLAS reductions are not
order-fixed. Report results with the seed and device; a rerun on another
device is a new sample, not a replay.
