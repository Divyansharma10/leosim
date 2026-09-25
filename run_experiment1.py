"""Experiment 1 of the base paper: city-to-city hop count and latency.

    python run_experiment1.py                      # New York -> San Francisco
    python run_experiment1.py --pair Darwin Singapore
    python run_experiment1.py --metric latency     # OSPF cost = propagation delay
    python run_experiment1.py --all-metrics        # all three, side by side

Reported for every selector: hops, ISL-only hops, end-to-end latency, mean and
bottleneck capacity along the path, and the fraction of timesteps with any path
at all. The paper reports 5 / 6 / 22 hops for DoTD / Greedy / +Grid on
New York -> San Francisco.

Both the stage-2 and stage-3 variant labels are stamped on every row. A hop
count without them says nothing: it depends on the objective flag, the latency
formula, the OSPF cost metric and the elevation mask alike.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from routing import CITIES, RoutingConfig, compute_access, dataset_times, route_run
from topology import DoTD, Greedy, PlusGrid, TopologyConfig

ROOT = Path(__file__).parent


def build_selectors(dataset, tcfg: TopologyConfig) -> dict:
    return {
        "+Grid": PlusGrid(dataset, tcfg),
        "Greedy": Greedy(dataset, tcfg),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    }


def summarise(rows: list[dict], name: str, tcfg, rcfg) -> dict:
    df = pd.DataFrame(rows)
    live = df[df["reachable"]]
    return {
        "algorithm": name,
        "reachable_frac": float(df["reachable"].mean()),
        "hops_mean": float(live["hops"].mean()) if len(live) else np.nan,
        "hops_median": float(live["hops"].median()) if len(live) else np.nan,
        "hops_min": int(live["hops"].min()) if len(live) else -1,
        "isl_hops_mean": float(live["isl_hops"].mean()) if len(live) else np.nan,
        "latency_mean_ms": float(live["latency_ms"].mean()) if len(live) else np.nan,
        "path_capacity_mean_gbps": float(live["path_capacity_mean_gbps"].mean())
        if len(live)
        else np.nan,
        "path_capacity_min_gbps": float(live["path_capacity_min_gbps"].mean())
        if len(live)
        else np.nan,
        "topology_variant": tcfg.variant_label,
        "routing_variant": rcfg.variant_label,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, default=["New York", "San Francisco"])
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--metric", choices=["hop", "latency", "inv_capacity"], default="hop")
    p.add_argument("--all-metrics", action="store_true")
    p.add_argument("--mask", type=float, default=RoutingConfig.min_elevation_deg)
    p.add_argument("--objective", choices=["max", "min"], default="max")
    p.add_argument("--latency-formula", choices=["d_over_c", "c_over_d"], default="d_over_c")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    for city in args.pair:
        if city not in CITIES:
            raise SystemExit(f"Unknown city {city!r}; known: {sorted(CITIES)}")

    sim = SimConfig(
        selection=SelectionConfig(n_planes=args.planes, sats_per_plane=args.per_plane)
    ).resolve(ROOT)
    dataset = run(sim, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)

    tcfg = TopologyConfig(objective=args.objective, latency_formula=args.latency_formula)
    selectors = build_selectors(dataset, tcfg)
    results = {name: s.run(n_steps) for name, s in selectors.items()}

    times = dataset_times(dataset)
    metrics = ["hop", "latency", "inv_capacity"] if args.all_metrics else [args.metric]

    frames = []
    for metric in metrics:
        rcfg = RoutingConfig(route_metric=metric, min_elevation_deg=args.mask)
        access = {
            city: compute_access(dataset, CITIES[city], rcfg, times) for city in args.pair
        }
        for city, acc in access.items():
            n = acc.n_visible[:n_steps]
            print(
                f"[access] {city}: {n.mean():.2f} satellites above {rcfg.min_elevation_deg:g} deg "
                f"(min {n.min()}, max {n.max()}); no access at {int((n == 0).sum())} of {n_steps} steps"
            )
        rows = []
        for name, linksets in results.items():
            paths = route_run(dataset, linksets, access, rcfg, n_steps)
            rows.append(summarise(paths, name, tcfg, rcfg))
        frame = pd.DataFrame(rows)
        frame.insert(1, "route_metric", metric)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    pd.set_option("display.width", 220)
    print(f"\n{args.pair[0]} -> {args.pair[1]}   ({tcfg.variant_label})")
    cols = [
        "algorithm", "route_metric", "reachable_frac", "hops_mean", "hops_median",
        "hops_min", "isl_hops_mean", "latency_mean_ms", "path_capacity_mean_gbps",
    ]
    print(out[cols].round(3).to_string(index=False))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.pair[0]}_{args.pair[1]}".replace(" ", "")
    path = out_dir / f"experiment1_{tag}.csv"
    out.to_csv(path, index=False)
    print(f"\n[out] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
