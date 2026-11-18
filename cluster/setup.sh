#!/bin/bash
# One-time setup on the cluster login node, from the repo root:
#     bash cluster/setup.sh
# Creates .venv in the repo and installs requirements.txt (torch included;
# the default Linux torch wheel ships with CUDA 12, which A100s run).
# If your cluster wants modules first, load them before running this, e.g.
#     module load python/3.11
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| GPU visible here:", torch.cuda.is_available(), "(login nodes often have none; fine)")
EOF
mkdir -p logs
echo "setup done"
