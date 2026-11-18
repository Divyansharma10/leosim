#!/bin/bash
# Submit the whole chain from the repo root:  bash cluster/submit_all.sh
#   validate (CPU) -> train array (GPU, only if validate passed) -> mutants (CPU, after training)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
V=$(sbatch --parsable cluster/validate.sbatch)
T=$(sbatch --parsable --dependency=afterok:$V cluster/train.sbatch)
M=$(sbatch --parsable --dependency=afterany:$T cluster/mutants.sbatch)
echo "validate $V -> train $T (array 0-3) -> mutants $M"
echo "watch:  squeue -u \$USER     logs: logs/"
