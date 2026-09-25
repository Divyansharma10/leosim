"""Phase-1 wrap-up: every number and figure for the reproduction report.

    python run_phase1_report.py              # everything, ~10 minutes
    python run_phase1_report.py --steps 300  # quick pass while iterating

Writes CSVs and PNGs into ``results/``. Each section is written as soon as it
finishes, so a long run that is interrupted still leaves what it had.

Sections
    1  Experiment 2  -- all five Table II city pairs x four algorithms
    2  Paired hops   -- per-timestep win/tie/loss against Greedy
    3  Overlap       -- how often DoTD-backward and Greedy agree exactly
    4  Experiment 3  -- Fig. 6's "ISLs kept" alongside our churn
    5  Variants      -- objective x latency_formula on New York -> San Francisco
    6  Lookahead     -- the w_lookahead trade-off

Nothing here changes stage 1-3 behaviour; it only runs them and tabulates.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leosim.config import SPEED_OF_LIGHT_M_S, SelectionConfig, SimConfig
from leosim.simulation import run
from routing import (
    CITIES,
    PAPER_PAIRS,
    RoutingConfig,
    chord_km,
    compute_access,
    dataset_times,
    great_circle_km,
    route_run,
)
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.metrics import metrics_frame, summary_table

N = chr(10)

ROOT = Path(__file__).parent
OUT = ROOT / "results"

ALGORITHMS = ["+Grid", "Greedy", "DoTD-forward", "DoTD-backward"]

# Validated categorical palette, light mode (dataviz reference palette slots
# 1-4). Worst adjacent CVD dE 9.1, normal-vision dE 22.9. Two slots sit under
# 3:1 contrast on the surface, so every bar carries a visible value label and
# every figure has a CSV beside it -- the relief rule.
COLORS = {
    "+Grid": "#2a78d6",
    "Greedy": "#eb6834",
    "DoTD-forward": "#1baf7a",
    "DoTD-backward": "#eda100",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"


# --- figure helpers ---------------------------------------------------------


def _style(ax, title: str, ylabel: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    pad = 8 if not subtitle else 21 + 11 * subtitle.count(N)
    ax.set_title(title, color=INK, fontsize=12, pad=pad, loc="left")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=8.5, va="bottom")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def grouped_bars(ax, categories, series: dict, fmt="{:.2f}", ylabel="", title="",
                 subtitle="", legend=True) -> None:
    """Grouped bars with a direct label on every bar (contrast relief rule)."""
    n = len(series)
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / n
    for k, (name, values) in enumerate(series.items()):
        offset = (k - (n - 1) / 2) * width
        vals = np.asarray(values, dtype=float)
        bars = ax.bar(
            x + offset, np.nan_to_num(vals), width * 0.92, label=name,
            color=COLORS.get(name, "#2a78d6"), edgecolor=SURFACE, linewidth=1.4,
        )
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height(),
                fmt.format(v), ha="center", va="bottom",
                color=INK_2, fontsize=7.2, rotation=90 if len(categories) > 4 else 0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    if not any(categories):
        ax.tick_params(axis="x", length=0)
    _style(ax, title, ylabel, subtitle)
    if legend:
        ax.legend(
            frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=min(4, n),
            loc="upper left", bbox_to_anchor=(0, -0.13),
        )


def figure_legend(fig, names) -> None:
    """One legend for the whole figure. Per-axes legends on a multi-panel figure
    repeat the same four entries and collide."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in names]
    fig.legend(handles, names, frameon=False, fontsize=8.5, labelcolor=INK_2,
               ncol=len(names), loc="lower center", bbox_to_anchor=(0.5, 0.0))


def save(fig, name: str, tight: bool = True) -> None:
    fig.patch.set_facecolor(SURFACE)
    if tight:
        fig.tight_layout()
    path = OUT / name
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  [fig] {path}")


def write(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"  [csv] {path}")


# --- sections ---------------------------------------------------------------


def path_rows(dataset, linksets, access, rcfg, n_steps) -> pd.DataFrame:
    return pd.DataFrame(route_run(dataset, linksets, access, rcfg, n_steps))


def section_experiment2(dataset, results, times, n_steps) -> tuple[pd.DataFrame, dict]:
    print("\n[1/6] Experiment 2: five Table II city pairs")
    rcfg = RoutingConfig()
    c_km_s = SPEED_OF_LIGHT_M_S / 1000.0
    rows = []
    per_pair_paths = {}

    for tag, a, b in PAPER_PAIRS:
        access = {
            c: compute_access(dataset, CITIES[c], rcfg, times) for c in (a, b)
        }
        arc = great_circle_km(CITIES[a], CITIES[b])
        chord = chord_km(CITIES[a], CITIES[b])
        for name in ALGORITHMS:
            df = path_rows(dataset, results[name], access, rcfg, n_steps)
            per_pair_paths[(tag, name)] = df
            live = df[df["reachable"]]
            rows.append({
                "pair": tag, "source": a, "destination": b,
                "great_circle_km": round(arc, 1), "chord_km": round(chord, 1),
                "algorithm": name,
                "reachable_frac": float(df["reachable"].mean()),
                "hops_mean": float(live["hops"].mean()) if len(live) else np.nan,
                "hops_median": float(live["hops"].median()) if len(live) else np.nan,
                "isl_hops_mean": float(live["isl_hops"].mean()) if len(live) else np.nan,
                "latency_mean_ms": float(live["latency_ms"].mean()) if len(live) else np.nan,
                "latency_median_ms": float(live["latency_ms"].median()) if len(live) else np.nan,
                "latency_floor_chord_ms": round(chord / c_km_s * 1e3, 4),
                "fibre_equivalent_ms": round(arc / (c_km_s * 2 / 3) * 1e3, 4),
                "path_cap_mean_gbps": float(live["path_capacity_mean_gbps"].mean())
                if len(live) else np.nan,
                "path_cap_mean_median_gbps": float(live["path_capacity_mean_gbps"].median())
                if len(live) else np.nan,
                "path_cap_bottleneck_gbps": float(live["path_capacity_min_gbps"].mean())
                if len(live) else np.nan,
            })
        print(f"  {tag} {a} -> {b}: reachable {rows[-1]['reachable_frac']*100:.1f}%")

    df = pd.DataFrame(rows)
    write(df, "experiment2_pairs.csv")
    return df, per_pair_paths


def section_s1_sensitivity(dataset, results, times, n_steps) -> pd.DataFrame:
    """S1 at a lower mask, reported separately and never mixed into the headline.

    Sydney peaks at 23.2 deg over this window, 1.8 deg under the 25 deg default,
    so S1's existence is decided entirely by an ASSUMED parameter. Reporting only
    "unreachable" would hide that; folding the lower mask into the main table
    would hide that the two rows were measured under different assumptions.
    """
    print("\n[1b] S1 mask sensitivity")
    rows = []
    for mask in (25.0, 20.0, 15.0, 10.0, 5.0):
        rcfg = RoutingConfig(min_elevation_deg=mask)
        access = {c: compute_access(dataset, CITIES[c], rcfg, times)
                  for c in ("Sydney", "Darwin")}
        for name in ALGORITHMS:
            df = path_rows(dataset, results[name], access, rcfg, n_steps)
            live = df[df["reachable"]]
            rows.append({
                "mask_deg": mask, "algorithm": name,
                "sydney_mean_sats": float(access["Sydney"].n_visible[:n_steps].mean()),
                "reachable_frac": float(df["reachable"].mean()),
                "hops_mean": float(live["hops"].mean()) if len(live) else np.nan,
                "hops_median": float(live["hops"].median()) if len(live) else np.nan,
                "latency_mean_ms": float(live["latency_ms"].mean()) if len(live) else np.nan,
            })
        print(f"  mask {mask:4.1f} deg: Sydney sees {rows[-1]['sydney_mean_sats']:.2f} sats, "
              f"reachable {rows[-1]['reachable_frac']*100:.1f}%")
    df = pd.DataFrame(rows)
    write(df, "experiment2_S1_mask_sensitivity.csv")
    return df


def section_paired(per_pair_paths, n_steps) -> pd.DataFrame:
    """Per-timestep win/tie/loss, which an average over integers cannot show."""
    print("\n[2/6] Paired hop comparison against Greedy")
    rows = []
    for tag, a, b in PAPER_PAIRS:
        base = per_pair_paths[(tag, "Greedy")]
        for name in ("DoTD-forward", "DoTD-backward"):
            other = per_pair_paths[(tag, name)]
            both = base["reachable"].to_numpy() & other["reachable"].to_numpy()
            if not both.any():
                rows.append({"pair": tag, "algorithm": name, "compared": 0,
                             "fewer_hops": np.nan, "equal": np.nan, "more_hops": np.nan,
                             "mean_hop_delta": np.nan})
                continue
            gh = base["hops"].to_numpy()[both]
            oh = other["hops"].to_numpy()[both]
            rows.append({
                "pair": tag, "algorithm": name, "compared": int(both.sum()),
                "fewer_hops": float((oh < gh).mean()),
                "equal": float((oh == gh).mean()),
                "more_hops": float((oh > gh).mean()),
                "mean_hop_delta": float((oh - gh).mean()),
            })
            print(f"  {tag} {name}: fewer {rows[-1]['fewer_hops']*100:.1f}%  "
                  f"equal {rows[-1]['equal']*100:.1f}%  more {rows[-1]['more_hops']*100:.1f}%")
    df = pd.DataFrame(rows)
    write(df, "paired_hop_comparison.csv")
    return df


def section_overlap(results, n_steps) -> pd.DataFrame:
    print("\n[3/6] Topology overlap with Greedy")
    rows = []
    greedy = results["Greedy"]
    for name in ("DoTD-forward", "DoTD-backward", "+Grid"):
        identical = 0
        jaccard = []
        for t in range(n_steps):
            a = set(map(tuple, greedy[t].edges))
            b = set(map(tuple, results[name][t].edges))
            if a == b:
                identical += 1
            union = len(a | b)
            jaccard.append(len(a & b) / union if union else 1.0)
        rows.append({
            "algorithm": name,
            "identical_frac": identical / n_steps,
            "jaccard_mean": float(np.mean(jaccard)),
            "jaccard_min": float(np.min(jaccard)),
        })
        print(f"  {name}: identical at {rows[-1]['identical_frac']*100:.2f}% of steps, "
              f"mean Jaccard {rows[-1]['jaccard_mean']:.4f}")
    df = pd.DataFrame(rows)
    write(df, "topology_overlap.csv")
    return df


def section_churn(dataset, results, tcfg, n_steps) -> pd.DataFrame:
    """Fig. 6's quantity -- ISLs KEPT between consecutive steps -- and ours.

    The paper counts links that survive a step; our churn counts links that
    change (added plus removed). They are not complements of one another: the
    link count itself moves between steps, so "kept" and "churned" do not sum
    to a constant. Both are reported rather than one derived from the other.
    """
    print("\n[4/6] Experiment 3: ISLs kept vs churn")
    df = metrics_frame(dataset, {k: v[:n_steps] for k, v in results.items()},
                       tcfg.u_max, tcfg.variant_label)
    summary = summary_table(df).reset_index()

    rows = []
    for name in ALGORITHMS:
        sets = [set(map(tuple, ls.edges)) for ls in results[name][:n_steps]]
        kept = [len(sets[t] & sets[t - 1]) for t in range(1, len(sets))]
        total = [len(sets[t]) for t in range(1, len(sets))]
        s = summary[summary["algorithm"] == name].iloc[0]
        rows.append({
            "algorithm": name,
            "isls_kept_mean": float(np.mean(kept)),
            "isls_kept_frac": float(np.mean(np.array(kept) / np.array(total))),
            "links_mean": float(s["links_mean"]),
            "churn_per_step": float(s["churn_per_step"]),
            "capacity_total_gbps_mean": float(s["capacity_total_gbps_mean"]),
            "latency_mean_ms": float(s["latency_mean_ms"]),
        })
        print(f"  {name}: keeps {rows[-1]['isls_kept_mean']:.1f} ISLs "
              f"({rows[-1]['isls_kept_frac']*100:.2f}%), churn {rows[-1]['churn_per_step']:.4f}")
    out = pd.DataFrame(rows)
    write(out, "experiment3_churn.csv")
    return out


def section_variants(dataset, times, n_steps) -> pd.DataFrame:
    """The two paper contradictions, all four combinations, on New York -> SF."""
    print("\n[5/6] Paper-variant matrix on New York -> San Francisco")
    rcfg = RoutingConfig()
    access = {c: compute_access(dataset, CITIES[c], rcfg, times)
              for c in ("New York", "San Francisco")}
    rows = []
    for objective in ("max", "min"):
        for formula in ("d_over_c", "c_over_d"):
            tcfg = TopologyConfig(objective=objective, latency_formula=formula)
            sel = {
                "+Grid": PlusGrid(dataset, tcfg),
                "Greedy": Greedy(dataset, tcfg),
                "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
                "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
            }
            for name, s in sel.items():
                links = s.run(n_steps)
                df = path_rows(dataset, links, access, rcfg, n_steps)
                live = df[df["reachable"]]
                rows.append({
                    "objective": objective, "latency_formula": formula,
                    "algorithm": name,
                    "hops_mean": float(live["hops"].mean()) if len(live) else np.nan,
                    "hops_median": float(live["hops"].median()) if len(live) else np.nan,
                    "latency_mean_ms": float(live["latency_ms"].mean()) if len(live) else np.nan,
                    "reachable_frac": float(df["reachable"].mean()),
                })
            med = {r["algorithm"]: r["hops_median"] for r in rows[-4:]}
            print(f"  objective={objective:3s} latency={formula:9s} -> median hops "
                  f"DoTD-f {med['DoTD-forward']:.0f}, Greedy {med['Greedy']:.0f}, "
                  f"+Grid {med['+Grid']:.0f}")
    df = pd.DataFrame(rows)

    # Distance from the paper's reported 5 / 6 ordering on this pair.
    def score(g):
        f = g[g["algorithm"] == "DoTD-forward"]["hops_median"].iloc[0]
        gr = g[g["algorithm"] == "Greedy"]["hops_median"].iloc[0]
        return pd.Series({"dotd_median": f, "greedy_median": gr,
                          "paper_gap": abs(f - 5) + abs(gr - 6),
                          "ordering_holds": bool(f < gr)})

    verdict = df.groupby(["objective", "latency_formula"]).apply(
        score, include_groups=False
    ).reset_index()
    write(df, "variant_matrix.csv")
    write(verdict, "variant_matrix_verdict.csv")
    print(verdict.to_string(index=False))
    return df, verdict


def section_lookahead(dataset, times, n_steps) -> pd.DataFrame:
    """w_lookahead against everything it trades off."""
    print("\n[6/6] w_lookahead trade-off")
    rcfg = RoutingConfig()
    access = {c: compute_access(dataset, CITIES[c], rcfg, times)
              for c in ("New York", "San Francisco")}
    base = TopologyConfig()
    rows = []
    for w in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0):
        cfg = dataclasses.replace(base, lookahead="forward", w_lookahead=w)
        links = DoTD(dataset, cfg).run(n_steps)
        frame = metrics_frame(dataset, {"x": links}, cfg.u_max, cfg.variant_label)
        s = summary_table(frame).iloc[0]
        df = path_rows(dataset, links, access, rcfg, n_steps)
        live = df[df["reachable"]]
        rows.append({
            "w_lookahead": w,
            "capacity_total_gbps_mean": float(s["capacity_total_gbps_mean"]),
            "latency_topology_ms": float(s["latency_mean_ms"]),
            "churn_per_step": float(s["churn_per_step"]),
            "hops_mean": float(live["hops"].mean()),
            "path_latency_ms": float(live["latency_ms"].mean()),
        })
        print(f"  w={w:<6} capacity {rows[-1]['capacity_total_gbps_mean']:7.2f}  "
              f"churn {rows[-1]['churn_per_step']:.4f}  hops {rows[-1]['hops_mean']:.3f}")
    df = pd.DataFrame(rows)
    write(df, "w_lookahead_tradeoff.csv")
    return df


# --- figures ----------------------------------------------------------------


def figures(exp2, paired, churn, variants, verdict, look, s1) -> None:
    print("\n[figures]")
    runnable = [t for t, _, _ in PAPER_PAIRS
                if exp2[(exp2["pair"] == t) & (exp2["reachable_frac"] > 0)].shape[0]]
    labels = {t: f"{t}\n{a[:3]}-{b[:3]}" for t, a, b in PAPER_PAIRS if t in runnable}

    for metric, fmt, ylab, title, sub in [
        ("hops_median", "{:.0f}", "hops (median)", "Hop count by city pair",
         "median over timesteps with a path; S1 omitted (no access at the 25 deg mask)"),
        ("latency_mean_ms", "{:.1f}", "milliseconds", "End-to-end latency by city pair",
         "mean over timesteps with a path"),
        ("path_cap_mean_gbps", "{:.2f}", "Gbit/s", "Mean link capacity along the path",
         "the Fig. 5c analogue; bottleneck capacity is in the CSV"),
    ]:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        series = {
            name: [exp2[(exp2["pair"] == t) & (exp2["algorithm"] == name)][metric].iloc[0]
                   for t in runnable]
            for name in ALGORITHMS
        }
        grouped_bars(ax, [labels[t] for t in runnable], series, fmt=fmt,
                     ylabel=ylab, title=title, subtitle=sub)
        save(fig, f"fig_{metric}.png")

    # Paired comparison: the honest view of an integer-valued difference.
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    sub = paired[paired["compared"] > 0]
    cats = [f"{r.pair}\n{r.algorithm.replace('DoTD-', '')}" for r in sub.itertuples()]
    bottom = np.zeros(len(sub))
    for col, colour, lab in [
        ("fewer_hops", "#1baf7a", "fewer hops than Greedy"),
        ("equal", "#d9d8d4", "equal"),
        ("more_hops", "#e34948", "more hops"),
    ]:
        vals = sub[col].to_numpy() * 100
        ax.bar(cats, vals, bottom=bottom, color=colour, label=lab,
               edgecolor=SURFACE, linewidth=1.4, width=0.72)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color=INK, fontsize=7.5)
        bottom += vals
    _style(ax, "DoTD vs Greedy at the same timestep", "% of timesteps",
           "paired per-timestep comparison; averages over integer hop counts hide this")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.13))
    save(fig, "fig_paired_hops.png")

    # Churn: two panels, never two y-scales on one axis.
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0))
    kept = {n: churn[churn["algorithm"] == n]["isls_kept_frac"].iloc[0] * 100
            for n in ALGORITHMS}
    spread = max(kept.values()) - min(kept.values())
    grouped_bars(axes[0], [""], {n: [v] for n, v in kept.items()},
                 fmt="{:.2f}", ylabel="% of ISLs kept",
                 title="ISLs kept between steps",
                 subtitle=f"the paper's Fig. 6 quantity, saturated: all four within "
                          f"{spread:.2f} points",
                 legend=False)
    axes[0].set_ylim(0, 116)
    grouped_bars(axes[1], [""], {n: [churn[churn["algorithm"] == n]["churn_per_step"].iloc[0]]
                                 for n in ALGORITHMS},
                 fmt="{:.3f}", ylabel="link changes per step", title="Churn per step",
                 subtitle="added plus removed; this is what separates them",
                 legend=False)
    axes[1].set_ylim(0, 0.56)
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    figure_legend(fig, ALGORITHMS)
    save(fig, "fig_churn.png", tight=False)

    # w_lookahead: small multiples, one measure per axis.
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for ax, (col, lab, title) in zip(axes.ravel(), [
        ("capacity_total_gbps_mean", "Gbit/s", "Topology capacity"),
        ("path_latency_ms", "ms", "End-to-end latency (NY->SF)"),
        ("hops_mean", "hops", "Mean hop count (NY->SF)"),
        ("churn_per_step", "changes/step", "Churn per step"),
    ]):
        ax.plot(look["w_lookahead"], look[col], color="#1baf7a", linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.4)
        for x, y in zip(look["w_lookahead"], look[col]):
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", color=INK_2, fontsize=7)
        _style(ax, title, lab)
        ax.set_xlabel("w_lookahead", color=INK_2, fontsize=9)
    fig.suptitle("What w_lookahead trades off", color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(0.02, 0.945, "DoTD-forward; w=0 is exactly Greedy. The paper pins no value.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_w_lookahead.png")

    # Variant matrix.
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    combos = [(o, f) for o in ("max", "min") for f in ("d_over_c", "c_over_d")]
    cats = [f"{o}\n{f}" for o, f in combos]
    series = {
        name: [variants[(variants["objective"] == o)
                        & (variants["latency_formula"] == f)
                        & (variants["algorithm"] == name)]["hops_median"].iloc[0]
               for o, f in combos]
        for name in ALGORITHMS
    }
    grouped_bars(ax, cats, series, fmt="{:.0f}", ylabel="hops (median)",
                 title="Paper-ambiguity matrix, New York -> San Francisco",
                 subtitle="paper reports DoTD 5 / Greedy 6; dashed lines mark those values")
    ax.axhline(5, color="#1baf7a", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(6, color="#eb6834", linestyle="--", linewidth=1, alpha=0.6)
    save(fig, "fig_variant_matrix.png")

    # S1 sensitivity.
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    masks = sorted(s1["mask_deg"].unique(), reverse=True)
    series = {n: [s1[(s1["mask_deg"] == m) & (s1["algorithm"] == n)]["reachable_frac"].iloc[0] * 100
                  for m in masks] for n in ALGORITHMS}
    grouped_bars(ax, [f"{m:g}" for m in masks], series, fmt="{:.0f}",
                 ylabel="% of timesteps with a path",
                 title="S1 Sydney -> Darwin exists only below the default mask",
                 subtitle="Sydney peaks at 23.2 deg elevation; the 25 deg default is ASSUMED, not from the paper")
    ax.set_xlabel("elevation mask (degrees)", color=INK_2, fontsize=9)
    save(fig, "fig_S1_mask_sensitivity.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--figures-only", action="store_true",
                   help="redraw from the CSVs already in results/, no recompute")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.figures_only:
        figures(
            pd.read_csv(OUT / "experiment2_pairs.csv"),
            pd.read_csv(OUT / "paired_hop_comparison.csv"),
            pd.read_csv(OUT / "experiment3_churn.csv"),
            pd.read_csv(OUT / "variant_matrix.csv"),
            pd.read_csv(OUT / "variant_matrix_verdict.csv"),
            pd.read_csv(OUT / "w_lookahead_tradeoff.csv"),
            pd.read_csv(OUT / "experiment2_S1_mask_sensitivity.csv"),
        )
        return 0
    t_start = time.perf_counter()

    cfg = SimConfig(
        selection=SelectionConfig(n_planes=args.planes, sats_per_plane=args.per_plane)
    ).resolve(ROOT)
    dataset = run(cfg, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)

    tcfg = TopologyConfig()
    selectors = {
        "+Grid": PlusGrid(dataset, tcfg),
        "Greedy": Greedy(dataset, tcfg),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    }
    results = {name: s.run(n_steps) for name, s in selectors.items()}
    times = dataset_times(dataset)
    print(f"\nconstellation: {dataset.n_sats} satellites, {n_steps} steps of "
          f"{dataset.config.step_s:g} s, epoch {dataset.times_utc_iso[0]}")
    print(f"topology variant: {tcfg.variant_label}")

    exp2, per_pair = section_experiment2(dataset, results, times, n_steps)
    s1 = section_s1_sensitivity(dataset, results, times, n_steps)
    paired = section_paired(per_pair, n_steps)
    overlap = section_overlap(results, n_steps)
    churn = section_churn(dataset, results, tcfg, n_steps)
    variants, verdict = section_variants(dataset, times, n_steps)
    look = section_lookahead(dataset, times, n_steps)

    figures(exp2, paired, churn, variants, verdict, look, s1)

    # One machine-readable bundle of the run's provenance, so a figure can
    # always be traced back to the constellation and epoch that produced it.
    meta = pd.DataFrame([{
        "n_satellites": dataset.n_sats,
        "n_planes": args.planes,
        "sats_per_plane": args.per_plane,
        "n_steps": n_steps,
        "step_s": dataset.config.step_s,
        "epoch_utc": dataset.times_utc_iso[0],
        "shell_inclination_deg": dataset.selection_info.get("shell_inclination_deg"),
        "topology_variant": tcfg.variant_label,
        "routing_variant": RoutingConfig().variant_label,
    }])
    write(meta, "run_metadata.csv")

    print(f"\ndone in {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
