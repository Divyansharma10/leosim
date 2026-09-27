"""Window-to-window variance of the 601-satellite headline numbers.

    python run_windows.py --offsets-min 0 24 48 72     # four 6-minute windows at 1 s

The DoTD-vs-Greedy hop result and the scored selectors' disconnection rate were
measured on one window. Disconnection alone moved from 18% (30 min at 5 s) to 67%
(the first 6 min at 1 s) for Greedy, so a single window hides variance that the
Poisson seed rule in report_traffic.py does not cover: seeds vary demand, not
geometry. Each window here is a separate stage-1 dataset starting `offset`
minutes after the default start (median TLE epoch of the selection), spaced to
sample different geometry across roughly one orbit.

Per window and selector: fraction of timesteps connected, worst component count,
and New York -> San Francisco hops (mean, median) and latency under the hop and
latency route metrics. Everything is production code; nothing is recomputed here.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from routing import CITIES, RoutingConfig, compute_access, dataset_times, route_run
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.metrics import metrics_frame

ROOT = Path(__file__).parent


def default_start(planes: int, per_plane: int) -> datetime:
    """The start instant the default config resolves to (median TLE epoch)."""
    probe = SimConfig(
        step_s=60.0, duration_s=60.0,
        selection=SelectionConfig(n_sats=planes * per_plane, n_planes=planes, sats_per_plane=per_plane),
    ).resolve(ROOT)
    ds = run(probe, verbose=False)
    t0 = datetime.fromisoformat(ds.times_utc_iso[0].replace("Z", "+00:00"))
    return t0 if t0.tzinfo else t0.replace(tzinfo=timezone.utc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", type=int, default=34)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--duration-min", type=float, default=6.0)
    p.add_argument("--offsets-min", type=float, nargs="+", default=[0.0, 24.0, 48.0, 72.0])
    p.add_argument("--grid-gaps", choices=["strict", "bridge"], default="bridge")
    p.add_argument("--pair", nargs=2, default=["New York", "San Francisco"])
    p.add_argument("--out", default="results/scale/windows")
    args = p.parse_args()

    t0 = default_start(args.planes, args.per_plane)
    tcfg = TopologyConfig(grid_gaps=args.grid_gaps)
    rows = []
    for off in args.offsets_min:
        start = (t0 + timedelta(minutes=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Offset 0 passes start_utc=None, which resolves to the same instant but
        # hits the existing cache; an explicit string would be a new cache key.
        sim = SimConfig(
            step_s=args.step, duration_s=args.duration_min * 60.0, start_utc=None if off == 0 else start,
            selection=SelectionConfig(n_sats=args.planes * args.per_plane, n_planes=args.planes,
                                      sats_per_plane=args.per_plane),
        ).resolve(ROOT)
        dataset = run(sim, verbose=True)
        selectors = {
            "+Grid": PlusGrid(dataset, tcfg),
            "Greedy": Greedy(dataset, tcfg),
            "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
            "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
        }
        results = {name: s.run() for name, s in selectors.items()}
        conn = metrics_frame(dataset, results, tcfg.u_max, tcfg.variant_label)
        times = dataset_times(dataset)
        for metric in ("hop", "latency"):
            rcfg = RoutingConfig(route_metric=metric)
            access = {c: compute_access(dataset, CITIES[c], rcfg, times) for c in args.pair}
            for name, ls in results.items():
                paths = pd.DataFrame(route_run(dataset, ls, access, rcfg))
                live = paths[paths["reachable"]]
                g = conn[conn["algorithm"] == name]
                rows.append({
                    "offset_min": off, "start_utc": start, "algorithm": name, "route_metric": metric,
                    "connected_frac": float(g["connected"].mean()),
                    "components_max": int(g["n_components"].max()),
                    "reachable_frac": float(paths["reachable"].mean()),
                    "hops_mean": float(live["hops"].mean()) if len(live) else np.nan,
                    "hops_median": float(live["hops"].median()) if len(live) else np.nan,
                    "latency_mean_ms": float(live["latency_ms"].mean()) if len(live) else np.nan,
                    "topology_variant": tcfg.variant_label, "routing_variant": rcfg.variant_label,
                    "n_sats": dataset.n_sats, "step_s": dataset.config.step_s,
                })
        w = pd.DataFrame([r for r in rows if r["offset_min"] == off and r["route_metric"] == "hop"])
        print(f"\n[window +{off:g} min, start {start}]")
        print(w[["algorithm", "connected_frac", "components_max", "hops_mean", "hops_median",
                 "latency_mean_ms"]].round(3).to_string(index=False), flush=True)
        del dataset, selectors, results

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"windows_{args.planes}x{args.per_plane}_step{args.step:g}_{tcfg.file_tag}.csv", index=False)

    pd.set_option("display.width", 200)
    for metric in ("hop", "latency"):
        d = df[df["route_metric"] == metric]
        print(f"\n## Across windows, route metric = {metric}")
        print(d.pivot(index="offset_min", columns="algorithm", values="hops_mean").round(3).to_string())
        if metric == "hop":
            print("\nconnected fraction:")
            print(d.pivot(index="offset_min", columns="algorithm", values="connected_frac").round(3).to_string())
            gap = (d[d["algorithm"] == "DoTD-forward"].set_index("offset_min")["hops_mean"]
                   - d[d["algorithm"] == "Greedy"].set_index("offset_min")["hops_mean"])
            print(f"\nDoTD-forward minus Greedy mean hops per window: {gap.round(3).to_dict()}")
            print(f"  sign consistent: {bool((gap > 0).all() or (gap < 0).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
