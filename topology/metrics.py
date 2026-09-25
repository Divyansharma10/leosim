"""Per-timestep topology metrics.

All metrics here are load-independent, matching the base paper's evaluation
axes. That is by design for the reproduction, and it is also why these metrics
cannot measure a traffic-aware extension -- see the project doc, section 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from leosim.dataset import LinkDataset

from .base import LinkSet


def link_churn(prev: LinkSet | None, curr: LinkSet) -> int:
    """Undirected links added plus links removed since the previous timestep.

    |E_t symmetric-difference E_{t-1}|. A link swapped for another counts 2: one
    torn down, one brought up, and both cost a terminal re-pointing. 0 at t=0.
    """
    if prev is None:
        return 0
    return int(np.triu(prev.adjacency ^ curr.adjacency, k=1).sum())


def timestep_metrics(
    dataset: LinkDataset, prev: LinkSet | None, curr: LinkSet, u_max: int
) -> dict:
    t = curr.t
    iu = np.triu(curr.adjacency, k=1)
    cap = np.asarray(dataset.capacity[t], dtype=np.float64)[iu]
    lat = np.asarray(dataset.latency[t], dtype=np.float64)[iu]
    deg = curr.degree()

    n_comp, _ = connected_components(csr_matrix(curr.adjacency), directed=False)
    hist = np.bincount(deg, minlength=u_max + 1)

    row = {
        "t": t,
        "time_s": float(dataset.time_offsets_s[t]),
        "n_links": int(iu.sum()),
        "capacity_total_gbps": float(cap.sum() / 1e9),
        "capacity_mean_gbps": float(cap.mean() / 1e9) if cap.size else np.nan,
        # Always stage-1 D/c seconds, whichever latency formula the score used:
        # the metric reports physical delay, the flag only changes selection.
        "latency_mean_ms": float(lat.mean() * 1e3) if lat.size else np.nan,
        "churn": link_churn(prev, curr),
        "degree_min": int(deg.min()),
        "degree_mean": float(deg.mean()),
        "degree_max": int(deg.max()),
        "connected": bool(n_comp == 1),
        "n_components": int(n_comp),
    }
    # Degree histogram as columns deg_0..deg_U. Anything above U would land
    # beyond the named columns, so it is summed into deg_over_U and must be 0.
    for k in range(u_max + 1):
        row[f"deg_{k}"] = int(hist[k])
    row["deg_over_U"] = int(hist[u_max + 1 :].sum())
    return row


def metrics_frame(
    dataset: LinkDataset,
    results: dict[str, list[LinkSet]],
    u_max: int,
    variant: str,
) -> pd.DataFrame:
    """One row per (algorithm, timestep). ``variant`` is stamped on every row."""
    rows = []
    for name, linksets in results.items():
        prev = None
        for ls in linksets:
            row = timestep_metrics(dataset, prev, ls, u_max)
            row["algorithm"] = name
            rows.append(row)
            prev = ls
    df = pd.DataFrame(rows)
    df["variant"] = variant
    lead = ["algorithm", "t", "time_s"]
    return df[lead + [c for c in df.columns if c not in lead]]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-algorithm aggregates over the run.

    Churn is summed over t >= 1 (t=0 is 0 by definition) and also given per
    timestep, so runs of different length stay comparable.
    """
    g = df.groupby("algorithm", sort=False)
    out = pd.DataFrame(
        {
            "links_mean": g["n_links"].mean(),
            "capacity_total_gbps_mean": g["capacity_total_gbps"].mean(),
            "capacity_mean_gbps": g["capacity_mean_gbps"].mean(),
            "latency_mean_ms": g["latency_mean_ms"].mean(),
            "churn_total": g["churn"].sum(),
            "churn_per_step": g["churn"].apply(lambda s: s.iloc[1:].mean()),
            "degree_max": g["degree_max"].max(),
            "connected_frac": g["connected"].mean(),
            "components_max": g["n_components"].max(),
        }
    )
    return out
