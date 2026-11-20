"""Aggregate oracle runs: python results/agent/oracle/aggregate.py [runs_dir]

Per (granularity, load, w_max): number of runs, mean summed and joint gain,
the share of additivity states where joint < sum / 2 (composition fails), and
the oracle policy against backpressure with the spread over runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "runs"
rows = []
for f in sorted(d.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    s, j = np.array(r["sum_gain"]), np.array(r["joint_gain"])
    keep = s > 0  # states where some unit had a positive gain
    rows.append({"gran": r["gran"], "load": r["load"], "w_max": r["w_max"], "seed": r["seed"], "start": r["start"],
                 "sum": s[keep].mean() if keep.any() else np.nan, "joint": j[keep].mean() if keep.any() else np.nan,
                 "fails": float(np.mean(j[keep] < s[keep] / 2)) if keep.any() else np.nan,
                 "policy": r["policy_shadow_mean"], "delivered": r["policy_delivered_vs_bp"],
                 "deviating": r["units_deviating_frac"]})
df = pd.DataFrame(rows)
df.to_csv(d.parent / "replication.csv", index=False)
g = df.groupby(["gran", "load", "w_max"]).agg(
    runs=("policy", "size"), sum_gain=("sum", "mean"), joint_gain=("joint", "mean"),
    composition_fails=("fails", "mean"), policy=("policy", "mean"), policy_sd=("policy", "std"),
    policy_wins=("policy", lambda x: int((x > 0).sum())), delivered=("delivered", "mean"),
    deviating=("deviating", "mean"))
print(g.round(4).to_string())
