"""Link scoring: running normalisation (Eq. 12) and the weighted score.

The score has three terms on incomparable scales -- bits/s, seconds, and a 0/1
churn flag -- so the first two are normalised to [0, 1] before weighting.

Normalisation is by a RUNNING maximum that only ever increases, never by the
current timestep's maximum. A per-timestep max would rescale every link each
step, so an unchanged link could score differently at t and t+1 purely because
some other link moved; scores would not be comparable across time, which the
lookahead accumulation depends on.
"""

from __future__ import annotations

import numpy as np

from leosim.config import SPEED_OF_LIGHT_M_S
from leosim.dataset import LinkDataset

from .config import LatencyFormula, TopologyConfig


def link_score(
    capacity_norm: np.ndarray | float,
    latency_norm: np.ndarray | float,
    churn: np.ndarray | float,
    config: TopologyConfig,
) -> np.ndarray | float:
    """``w1 * capacity_norm + w2 * (1 - latency_norm) + w3 * churn``.

    Kept as a free function over already-normalised inputs so the arithmetic can
    be pinned against a hand-computed value with no dataset in the loop.
    """
    return (
        config.w1_capacity * capacity_norm
        + config.w2_latency * (1.0 - latency_norm)
        + config.w3_churn * churn
    )


def latency_quantity(
    dataset: LinkDataset, t: int, formula: LatencyFormula
) -> np.ndarray:
    """The latency-like quantity the score normalises, per the active formula.

    ``d_over_c``: stage-1 propagation delay in seconds, read directly.
    ``c_over_d``: the paper's written form, c / D, in 1/s. Largest for the
    *shortest* links, so ``1 - norm`` rewards long links. Exposed so its effect
    can be measured against the published numbers, not because it is correct.

    Zero wherever the pair is not visible, matching stage-1 convention.
    """
    visible = np.asarray(dataset.visible[t])
    if formula == "d_over_c":
        return np.asarray(dataset.latency[t], dtype=np.float64)
    if formula == "c_over_d":
        dist_m = np.asarray(dataset.distance[t], dtype=np.float64) * 1e3
        out = np.zeros_like(dist_m)
        np.divide(SPEED_OF_LIGHT_M_S, dist_m, out=out, where=visible & (dist_m > 0))
        return out
    raise ValueError(f"Unknown latency formula: {formula!r}")


class RunningNormaliser:
    """Monotone non-decreasing maxima for capacity and latency (paper Eq. 12).

    ``update(t)`` folds timestep t's visible links into the maxima *before*
    anything at t is scored, so the current step's own values can never
    normalise above 1.

    The history of maxima is kept so the validation suite can assert
    monotonicity at every step, not only at the endpoints -- a max that dipped
    and recovered would pass an endpoint comparison.
    """

    def __init__(self, dataset: LinkDataset, formula: LatencyFormula) -> None:
        self.dataset = dataset
        self.formula = formula
        self.max_capacity = 0.0
        self.max_latency = 0.0
        self.history: list[tuple[int, float, float]] = []

    def update(self, t: int) -> None:
        visible = np.asarray(self.dataset.visible[t])
        if visible.any():
            cap = np.asarray(self.dataset.capacity[t], dtype=np.float64)[visible]
            lat = latency_quantity(self.dataset, t, self.formula)[visible]
            self.max_capacity = max(self.max_capacity, float(cap.max()))
            self.max_latency = max(self.max_latency, float(lat.max()))
        self.history.append((t, self.max_capacity, self.max_latency))

    def normalise(
        self, capacity: np.ndarray, latency: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Divide by the running maxima.

        Values are clipped to [0, 1]. For the current timestep the clip is a
        no-op, because ``update`` already absorbed its maxima. It binds only for
        forward-lookahead terms, which read future timesteps against the
        *causal* running max; a future link better than anything yet seen is
        capped at 1 rather than being allowed to outscore the scale.
        """
        cap_n = np.zeros_like(capacity, dtype=np.float64)
        lat_n = np.zeros_like(latency, dtype=np.float64)
        if self.max_capacity > 0:
            cap_n = np.clip(capacity / self.max_capacity, 0.0, 1.0)
        if self.max_latency > 0:
            lat_n = np.clip(latency / self.max_latency, 0.0, 1.0)
        return cap_n, lat_n


def quality_matrix(
    dataset: LinkDataset,
    t: int,
    normaliser: RunningNormaliser,
    config: TopologyConfig,
) -> np.ndarray:
    """(N, N) capacity + latency part of the score, zero where not visible.

    Excludes churn, which depends on a decision rather than on geometry. Future
    timesteps have no decision yet, so the forward lookahead uses this alone.
    """
    visible = np.asarray(dataset.visible[t])
    cap = np.asarray(dataset.capacity[t], dtype=np.float64)
    lat = latency_quantity(dataset, t, config.latency_formula)
    cap_n, lat_n = normaliser.normalise(cap, lat)
    q = link_score(cap_n, lat_n, 0.0, config)
    return np.where(visible, q, 0.0)


def churn_matrix(prev_adjacency: np.ndarray | None, n_sats: int) -> np.ndarray:
    """1.0 where the pair was linked at the previous timestep, else 0.0."""
    if prev_adjacency is None:
        return np.zeros((n_sats, n_sats), dtype=np.float64)
    return prev_adjacency.astype(np.float64)
