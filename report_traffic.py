"""Separation between selectors under load, from a run_traffic.py output.

    python report_traffic.py results/traffic/traffic_601sats_..._feeder-stage1.csv

For each load: every selector's seed-mean, the spread across the four selectors,
and whether that spread is real.

Real means: all selectors see identical demand for a given seed (common random
numbers), so what matters is the seed-to-seed stability of the *difference*
between two selectors, not each selector's own variance. A pairwise gap counts
as separated only if every seed agrees on its sign AND its mean exceeds
SEPARATION_SD x the standard deviation of that difference across seeds. With
three seeds this is a screen, not a significance test, and is reported as such.

Also reported: the feeder-capacity bound (no topology can deliver more), and the
timestep-paired Greedy vs DoTD-forward comparison from the _steps file.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SELECTORS = ["+Grid", "Greedy", "DoTD-forward", "DoTD-backward"]
METRICS = {
    "delivered_frac_of_routed": "delivered",
    "delay_mean_ms": "delay mean ms",
    "delay_p95_ms_median": "delay p95 ms",
}
# A gap must exceed this many SDs of its own seed-to-seed variation.
SEPARATION_SD = 2.0


def separated(a: np.ndarray, b: np.ndarray) -> bool:
    d = a - b
    if len(d) < 2:
        return False
    same_sign = np.all(d > 0) or np.all(d < 0)
    return bool(same_sign and abs(d.mean()) > SEPARATION_SD * d.std(ddof=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary_csv")
    args = p.parse_args()
    path = Path(args.summary_csv)
    df = pd.read_csv(path)
    steps_path = path.with_name(path.stem + "_steps.csv.gz")
    steps = pd.read_csv(steps_path) if steps_path.exists() else None

    print(f"# {path.name}\n")
    print(f"variants: {df['topology_variant'].iloc[0]} | {df['routing_variant'].iloc[0]}")
    print(f"          {df['traffic_variant'].iloc[0].split(', ', 1)[1]}")
    print(f"seeds: {sorted(df['seed'].unique().tolist())}; separation screen: same sign on every "
          f"seed and |mean gap| > {SEPARATION_SD:g} SD of the gap\n")

    rows = []
    for load, g in df.groupby("peak_total_gbps"):
        sel = g[g["algorithm"].isin(SELECTORS)]
        row = {"load_gbps": load}
        for col, label in METRICS.items():
            means = sel.groupby("algorithm")[col].mean()
            row[f"{label}: best"] = means.idxmax() if col.startswith("delivered") else means.idxmin()
            row[f"{label}: spread"] = means.max() - means.min()
            per_seed = sel.pivot(index="seed", columns="algorithm", values=col)
            pairs = [(a, b) for a, b in itertools.combinations(SELECTORS, 2)
                     if separated(per_seed[a].to_numpy(), per_seed[b].to_numpy())]
            row[f"{label}: separated pairs"] = len(pairs)
            gd = per_seed["Greedy"] - per_seed["DoTD-forward"]
            row[f"{label}: Greedy-DoTDf"] = gd.mean()
            row[f"{label}: Greedy-DoTDf sep"] = separated(per_seed["Greedy"].to_numpy(),
                                                          per_seed["DoTD-forward"].to_numpy())
        best_delivered = sel.groupby("algorithm")["delivered_frac_of_routed"].mean().max()
        row["feeder bound"] = g["feeder_bound_frac"].mean()
        row["headroom to bound"] = row["feeder bound"] - best_delivered
        av = g[~g["algorithm"].isin(SELECTORS)]
        row["all-visible delivered"] = av["delivered_frac_of_routed"].mean() if len(av) else np.nan
        row["no_path_frac Greedy"] = sel[sel["algorithm"] == "Greedy"]["no_path_frac"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    show = ["load_gbps", "delivered: best", "delivered: spread", "delivered: separated pairs",
            "delivered: Greedy-DoTDf", "delivered: Greedy-DoTDf sep",
            "feeder bound", "headroom to bound", "all-visible delivered"]
    print("## Delivered fraction of routed demand\n")
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    show = ["load_gbps"] + [c for c in out.columns if c.startswith("delay")]
    print("\n## Delay under load\n")
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n## Demand stranded with no path to any gateway (islands), Greedy\n")
    print(out[["load_gbps", "no_path_frac Greedy"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if steps is not None:
        print("\n## Timestep-paired: Greedy vs DoTD-forward, delivered bits (all seeds pooled)\n")
        s = steps[steps["algorithm"].isin(["Greedy", "DoTD-forward"])]
        w = s.pivot_table(index=["peak_total_gbps", "seed", "t"], columns="algorithm", values="delivered_bits")
        d = (w["DoTD-forward"] - w["Greedy"]) / w["Greedy"].where(w["Greedy"] > 0)
        tab = d.groupby(level=0).agg(
            dotd_more=lambda x: (x > 1e-9).mean(), equal=lambda x: (x.abs() <= 1e-9).mean(),
            dotd_less=lambda x: (x < -1e-9).mean(), median_rel_gap=lambda x: x.median(),
        )
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

    out.to_csv(path.with_name(path.stem + "_separation.csv"), index=False)
    print(f"\n[out] {path.with_name(path.stem + '_separation.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
