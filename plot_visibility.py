"""Sanity plot: visible neighbours per satellite over the simulation window.

One figure, one measure, one axis. The median line carries the level; the two
bands carry the spread across the 100 satellites, which is what tells you
whether the constellation geometry is stable or drifting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leosim.config import SimConfig
from leosim.simulation import run

ROOT = Path(__file__).parent

# Slot 1 (blue) from the reference categorical palette, plus its text tokens.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="visible_pairs.png")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    cfg = SimConfig().resolve(ROOT)
    dataset = run(cfg, verbose=True)

    degree = np.asarray(dataset.degree())  # (T, N)
    minutes = np.asarray(dataset.time_offsets_s) / 60.0

    median = np.median(degree, axis=1)
    q1, q3 = np.percentile(degree, [25, 75], axis=1)
    lo, hi = degree.min(axis=1), degree.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(
        minutes, lo, hi, color=SERIES, alpha=0.13, linewidth=0,
        label="min-max across satellites",
    )
    ax.fill_between(
        minutes, q1, q3, color=SERIES, alpha=0.28, linewidth=0,
        label="interquartile range",
    )
    ax.plot(minutes, median, color=SERIES, linewidth=2.0, label="median", zorder=3)

    # Recessive grid and axes: the data should be the only thing with weight.
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    ax.set_xlim(minutes[0], minutes[-1])
    ax.set_ylim(0, hi.max() * 1.14)
    ax.set_xlabel("time since epoch (minutes)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("visible neighbours per satellite", color=TEXT_SECONDARY, fontsize=10)

    # Direct label on the median, rather than a number on every point.
    ax.annotate(
        f"median {median[-1]:.0f}",
        xy=(minutes[-1], median[-1]),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        color=TEXT_PRIMARY,
        fontsize=9.5,
        fontweight="medium",
    )

    shell = dataset.selection_info.get("shell_inclination_deg")
    alt = np.asarray(dataset.altitudes)
    subtitle = (
        f"{dataset.n_sats} satellites, "
        f"{dataset.selection_info.get('n_planes_found', '?')} planes"
        + (f" at {shell:.1f}° inclination" if shell is not None else "")
        + f", {alt.min():.0f}-{alt.max():.0f} km\n"
        f"median {median.mean():.1f} neighbours, range {lo.min()}-{hi.max()} "
        f"across {dataset.n_steps} timesteps of {cfg.step_s:g} s"
    )
    ax.set_title(
        "Visible neighbours per satellite over the simulation window",
        color=TEXT_PRIMARY, fontsize=13, fontweight="semibold", loc="left", pad=26,
    )
    ax.text(
        0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
        color=TEXT_SECONDARY, fontsize=9, linespacing=1.45,
    )

    legend = ax.legend(
        loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
        handlelength=1.6, borderpad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n[plot] wrote {out}")
    print(
        f"[plot] degree per satellite: min {degree.min()}, median "
        f"{np.median(degree):.1f}, max {degree.max()}"
    )
    print(
        f"[plot] total visible pairs per timestep: "
        f"{int(dataset.visible[0].sum() // 2)} at t=0, "
        f"{int(dataset.visible[-1].sum() // 2)} at t=end"
    )


if __name__ == "__main__":
    main()
