"""Stage-2 driver: run all topology selectors, write metrics and a comparison plot.

    python run_topology.py                            # defaults: 10 planes x 20 sats
    python run_topology.py --objective min            # paper's Eq. 15 argmin
    python run_topology.py --latency-formula c_over_d # paper's written c/D

The active ambiguity variant is printed, stamped on every metrics row, written
into the output filenames, and shown in the plot title. A result without it is
not interpretable.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.metrics import metrics_frame, summary_table

ROOT = Path(__file__).parent
CHURN_SMOOTH_S = 30.0


def build_sim_config(args) -> SimConfig:
    selection = SelectionConfig(n_sats=args.planes * args.per_plane, n_planes=args.planes, sats_per_plane=args.per_plane)
    return SimConfig(selection=selection).resolve(ROOT)


def build_topology_config(args) -> TopologyConfig:
    return TopologyConfig(
        horizon_s=args.horizon,
        horizon_stride_s=args.stride,
        w_lookahead=args.w_lookahead,
        objective=args.objective,
        latency_formula=args.latency_formula,
        grid_gaps=args.grid_gaps,
    )


def run_all(dataset, tcfg: TopologyConfig, n_steps: int | None, verbose: bool = True):
    selectors = [
        PlusGrid(dataset, tcfg),
        Greedy(dataset, tcfg),
        DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    ]
    results = {}
    for sel in selectors:
        t0 = time.perf_counter()
        results[sel.name] = sel.run(n_steps)
        if verbose:
            print(f"[{sel.name:>14}] {len(results[sel.name])} steps in {time.perf_counter() - t0:.1f}s")
    return selectors, results


def plot_comparison(df: pd.DataFrame, variant: str, step_s: float, out: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    window = max(1, int(round(CHURN_SMOOTH_S / step_s)))

    for name, g in df.groupby("algorithm", sort=False):
        minutes = g["time_s"] / 60.0
        axes[0].plot(minutes, g["capacity_total_gbps"], label=name, lw=1.2)
        axes[1].plot(minutes, g["latency_mean_ms"], label=name, lw=1.2)
        churn = g["churn"].iloc[1:]
        axes[2].plot(
            minutes.iloc[1:],
            churn.rolling(window, min_periods=1).mean(),
            label=name,
            lw=1.2,
        )

    axes[0].set_ylabel("Total capacity (Gbps)")
    axes[1].set_ylabel("Mean link latency (ms)")
    axes[2].set_ylabel(f"Churn, links/step\n({CHURN_SMOOTH_S:g} s rolling mean)")
    axes[2].set_xlabel("Time (min)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4, fontsize=9)
    fig.suptitle(f"ISL topology comparison -- {variant}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", type=int, default=10, help="stage-1 SelectionConfig.n_planes")
    p.add_argument("--per-plane", type=int, default=20, help="stage-1 sats_per_plane (real plane population)")
    p.add_argument("--steps", type=int, default=None, help="default: whole dataset")
    p.add_argument("--horizon", type=float, default=TopologyConfig.horizon_s)
    p.add_argument("--stride", type=float, default=TopologyConfig.horizon_stride_s)
    p.add_argument("--w-lookahead", type=float, default=TopologyConfig.w_lookahead)
    p.add_argument("--objective", choices=["max", "min"], default="max")
    p.add_argument("--latency-formula", choices=["d_over_c", "c_over_d"], default="d_over_c")
    p.add_argument("--grid-gaps", choices=["strict", "bridge"], default="strict")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    dataset = run(build_sim_config(args))
    tcfg = build_topology_config(args)
    print(f"[variant] {tcfg.variant_label}")
    print(f"[config]  {tcfg}")

    selectors, results = run_all(dataset, tcfg, args.steps)
    grid = selectors[0]
    d = grid.structure.diagnostics
    print(
        f"[+Grid] {d['n_planes']} planes identified (separable={d['separable']}, "
        f"spread {d['max_plane_spread_deg']:.2f} deg vs gap {d['min_plane_gap_deg']:.2f} deg), "
        f"{int(grid.grid.sum() // 2)} structural links"
    )

    df = metrics_frame(dataset, results, tcfg.u_max, tcfg.variant_label)
    summary = summary_table(df)

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    tag = tcfg.file_tag
    df.to_csv(out_dir / f"topology_metrics_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"topology_summary_{tag}.csv")
    png = plot_comparison(df, tcfg.variant_label, dataset.config.step_s, out_dir / f"topology_comparison_{tag}.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print()
        print(f"Summary ({tcfg.variant_label}):")
        print(summary.round(3).to_string())
    print(f"\n[out] {out_dir / f'topology_metrics_{tag}.csv'}")
    print(f"[out] {png}")


if __name__ == "__main__":
    main()
