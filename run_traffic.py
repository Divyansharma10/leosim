"""Phase 2 baseline: every selector's topology under offered load.

    python run_traffic.py                               # 187 sats, default load
    python run_traffic.py --planes 34 --step 5 --grid-gaps bridge \\
        --loads 2 2.5 3 --seeds 7 8 9                   # evaluation constellation
    python run_traffic.py --gateway-sites krishnan8     # Krishnan et al.'s gateways
    python run_traffic.py --feeder-bandwidth-hz 1e9     # wider gateway links

Reports, per selector, load and seed: delivered fraction, end-to-end loss, delay
under load (mean and p95), backlog lost to link churn, and peak utilisation.
Demand no satellite can see is reported apart from loss: it belongs to the
ground segment, not the topology.

Greedy is always run alongside DoTD. The extension has to beat both: beating
DoTD alone could be the lookahead's doing, not the demand term's.

Two references, neither a selector:

- "All-visible" routes the same demand over every visible link, ignoring the
  four-terminal limit. It is NOT an upper bound on delivery: routing is fixed
  shortest path, and a hop-minimising path over a richer graph uses longer links
  and can concentrate load on the same feeders. It shows what connectivity
  alone buys under this routing.
- "feeder_bound_frac" IS a bound: at each step no topology can deliver more than
  the gateways' total feeder capacity x step, so delivered/routed can never
  exceed sum_t min(routed_t, feeder_t) / sum_t routed_t. (A 50 ms buffer can
  carry a sliver across steps; negligible against a 5 s step.)
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from routing import RoutingConfig, compute_access, dataset_times
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.base import LinkSet
from traffic import (
    SparseAccess,
    TrafficConfig,
    TrafficSim,
    expected_rate_bps,
    gateway_access,
    load_sites,
    offered_bits,
)

ROOT = Path(__file__).parent
ALL_VISIBLE = "All-visible (reference)"


def utc_hours(dataset) -> np.ndarray:
    start = datetime.fromisoformat(dataset.times_utc_iso[0].replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    h0 = start.hour + start.minute / 60.0 + start.second / 3600.0
    return h0 + np.asarray(dataset.time_offsets_s, dtype=np.float64) / 3600.0


def summarise(df: pd.DataFrame) -> dict:
    offered = df["offered_bits"].sum()
    routed = df["routed_bits"].sum()
    live = df["delay_mean_ms"].notna()
    return {
        "offered_gbps": offered / len(df) / 1e9 / df.attrs.get("step_s", 1.0),
        "no_access_frac": df["no_access_bits"].sum() / offered if offered else np.nan,
        "no_path_frac": df["no_path_bits"].sum() / offered if offered else np.nan,
        "delivered_frac_of_routed": df["delivered_bits"].sum() / routed if routed else np.nan,
        # End-to-end loss from per-city delivery. Not the sum of link drops:
        # every link on a path sees the path's full load, so one bit dropped at
        # two links is counted at both, and that sum can exceed what was routed.
        "lost_frac_of_routed": 1.0 - df["delivered_bits"].sum() / routed if routed else np.nan,
        "link_drop_bits_sum_over_routed": df["dropped_bits"].sum() / routed if routed else np.nan,
        "churn_lost_frac_of_routed": df["churn_lost_bits"].sum() / routed if routed else np.nan,
        "delay_mean_ms": float(np.average(df.loc[live, "delay_mean_ms"], weights=df.loc[live, "routed_bits"]))
        if live.any() and routed else np.nan,
        "delay_p95_ms_median": float(df["delay_p95_ms"].median()),
        "util_max": float(df["util_max"].max()),
        "links_saturated_mean": float(df["links_saturated"].mean()),
        "feeder_util_max": float(df["feeder_util_max"].max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--step", type=float, default=SimConfig.step_s)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--loads", type=float, nargs="+", default=[TrafficConfig.peak_total_gbps])
    p.add_argument("--seeds", type=int, nargs="+", default=[TrafficConfig.seed])
    p.add_argument("--grid-gaps", choices=["strict", "bridge"], default="strict")
    p.add_argument("--metric", choices=["hop", "latency", "inv_capacity"], default="hop")
    p.add_argument("--gateway-sites", choices=["populous20", "krishnan8"], default="populous20")
    p.add_argument("--feeder-bandwidth-hz", type=float, default=None)
    p.add_argument("--feeder-tx-power-dbw", type=float, default=None)
    p.add_argument("--feeder-noise-temperature-k", type=float, default=None)
    p.add_argument("--no-all-visible", action="store_true")
    p.add_argument("--out", default="results/traffic")
    args = p.parse_args()

    sim = SimConfig(
        step_s=args.step,
        selection=SelectionConfig(
            n_sats=args.planes * args.per_plane, n_planes=args.planes, sats_per_plane=args.per_plane
        ),
    ).resolve(ROOT)
    dataset = run(sim, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)
    step_s = dataset.config.step_s

    tcfg = TopologyConfig(grid_gaps=args.grid_gaps)
    rcfg = RoutingConfig(route_metric=args.metric)
    base = TrafficConfig(
        gateway_sites=args.gateway_sites,
        feeder_bandwidth_hz=args.feeder_bandwidth_hz,
        feeder_tx_power_dbw=args.feeder_tx_power_dbw,
        feeder_noise_temperature_k=args.feeder_noise_temperature_k,
    )
    sites = load_sites(base, ROOT)
    print(f"[variant] {tcfg.variant_label} | {rcfg.variant_label} | {base.variant_label}")

    t0 = time.time()
    times = dataset_times(dataset)
    city_acc = [SparseAccess.from_access(compute_access(dataset, s, rcfg, times)) for s in sites.cities]
    gw_acc = [gateway_access(dataset, s, rcfg, base, times) for s in sites.gateways]
    gw_cap = np.array([[a.at(t)[1].sum() for t in range(n_steps)] for a in gw_acc]) / 1e9
    print(f"[access] {len(city_acc)} cities, {len(gw_acc)} gateways in {time.time() - t0:.0f}s; "
          f"gateway feeder capacity per gateway: mean {gw_cap.mean():.2f} Gbps "
          f"(range {gw_cap.mean(axis=1).min():.2f}-{gw_cap.mean(axis=1).max():.2f}), "
          f"total {gw_cap.sum(axis=0).mean():.1f} Gbps")

    t0 = time.time()
    selectors = {
        "+Grid": PlusGrid(dataset, tcfg),
        "Greedy": Greedy(dataset, tcfg),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    }
    linksets = {name: s.run(n_steps) for name, s in selectors.items()}
    if not args.no_all_visible:
        linksets[ALL_VISIBLE] = [LinkSet(np.asarray(dataset.visible[t]), t) for t in range(n_steps)]
    print(f"[topology] {len(selectors)} selectors x {n_steps} steps in {time.time() - t0:.0f}s")

    feeder_bits = gw_cap.sum(axis=0) * 1e9 * step_s  # (T,) total feeder capacity per step
    hours = utc_hours(dataset)[:n_steps]
    rows, steps = [], []
    for load in args.loads:
        for seed in args.seeds:
            cfg = dataclasses.replace(base, peak_total_gbps=load, seed=seed)
            offered = offered_bits(expected_rate_bps(sites, hours, cfg), step_s, cfg)
            for name, ls in linksets.items():
                t1 = time.time()
                out = pd.DataFrame(TrafficSim(dataset, city_acc, gw_acc, cfg, rcfg).run(ls, offered))
                out.attrs["step_s"] = step_s
                routed = out["routed_bits"].to_numpy()
                bound = np.minimum(routed, feeder_bits).sum() / routed.sum() if routed.sum() else np.nan
                row = {"algorithm": name, "peak_total_gbps": load, "seed": seed, **summarise(out),
                       "feeder_bound_frac": bound,
                       "topology_variant": tcfg.variant_label, "routing_variant": rcfg.variant_label,
                       "traffic_variant": cfg.variant_label, "n_sats": dataset.n_sats, "step_s": step_s}
                rows.append(row)
                steps.append(out.assign(algorithm=name, peak_total_gbps=load, seed=seed))
                print(f"  load {load:5g} seed {seed}  {name:<22} delivered {row['delivered_frac_of_routed']:.4f}  "
                      f"delay {row['delay_mean_ms']:6.1f} ms  p95 {row['delay_p95_ms_median']:6.1f} ms  "
                      f"({time.time() - t1:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feeder = "feeder-stage1" if base.feeder_is_default else f"feeder-B{base.feeder_bandwidth_hz or 0:g}"
    stem = f"traffic_{dataset.n_sats}sats_{tcfg.file_tag}_{rcfg.route_metric}_{base.gateway_sites}_{feeder}"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    pd.concat(steps, ignore_index=True).to_csv(out_dir / f"{stem}_steps.csv.gz", index=False)
    print(f"\n[out] {out_dir / stem}.csv (+ _steps.csv.gz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
