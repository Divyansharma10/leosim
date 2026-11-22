"""kappa for a w_max-matched run, scaled by the oracle's measured reward-scale factor.

    python results/agent/oracle/kappa_factor.py                 # load 1, H 40, base kappa 0.01
    python results/agent/oracle/kappa_factor.py --load 1 --base-kappa 0.01 --runs results/agent/oracle/runs

kappa is a margin in reward units (agent.config), and switching the max-queue
term off shrinks the per-unit gains by a large factor, so the same kappa is a
much looser margin at w_max = 0. This rescales it by the ratio the oracle
measured:

    factor = mean over (seed, start) of mean summed gain at w_max = 0
           / the same at w_max = 1

Per run, the statistic is the mean summed per-unit gain over its additivity
states with a positive sum, the same as aggregate.py. Only (seed, start) runs
present at BOTH w_max values are paired. Nothing is printed for a granularity
without the expected number of pairs (9 = seeds 7-9 x starts 1200/1400/1599),
so a partial set of runs cannot produce a factor silently.

The link-granularity factor is the one to use: kappa gates a per-slot
advantage. The satellite factor is printed for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def per_run(runs: list[dict], gran: str, load: float, w_max: float, horizon: int) -> dict:
    out = {}
    for r in runs:
        if r["gran"] == gran and r["load"] == load and r["w_max"] == w_max and r["horizon"] == horizon:
            s = np.asarray(r["sum_gain"], dtype=np.float64)
            s = s[s > 0]
            out[(r["seed"], r["start"])] = float(s.mean()) if s.size else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path(__file__).parent / "runs")
    p.add_argument("--load", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--base-kappa", type=float, default=0.01, help="kappa at w_max = 1")
    p.add_argument("--expected-pairs", type=int, default=9)
    a = p.parse_args()

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(a.runs.glob("*.json"))]
    if not runs:
        print(f"no runs in {a.runs}")
        return 1
    status = 1
    for gran in ("link", "sat"):
        r0 = per_run(runs, gran, a.load, 0.0, a.horizon)
        r1 = per_run(runs, gran, a.load, 1.0, a.horizon)
        common = sorted(set(r0) & set(r1))
        print(f"{gran}: runs w_max=0 {len(r0)}, w_max=1 {len(r1)}, paired {len(common)}")
        if len(common) != a.expected_pairs:
            print(f"  expected {a.expected_pairs} paired (seed, start) runs -- not computing")
            continue
        m0 = float(np.mean([r0[k] for k in common]))
        m1 = float(np.mean([r1[k] for k in common]))
        f = m0 / m1
        print(f"  mean summed gain w_max=0 {m0!r}  w_max=1 {m1!r}")
        print(f"  factor {f!r}   kappa = {a.base_kappa:g} * factor = {a.base_kappa * f!r}")
        if gran == "link":
            status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
