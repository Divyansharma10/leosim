"""Score-based selectors: DoTD and its no-lookahead Greedy baseline.

Both share one scoring path and one matcher. The only difference between them
is the accumulation term, so any gap between their outputs is attributable to
that term and nothing else.

priority(i, j) = score_t(i, j) + w_lookahead * accumulation(i, j)
score_t(i, j)  = w1 * cap_norm + w2 * (1 - lat_norm) + w3 * churn

Accumulation has two implementations, selected by ``TopologyConfig.lookahead``.

Forward
    Weighted mean of link *quality* (the capacity and latency terms) over the
    window [t, t + horizon], sampled every ``horizon_stride_s``, with weight
    exp(-h / horizon) at offset h. A link that is strong now but occluded in 60 s
    contributes zero for the occluded samples and is ranked down -- this is the
    anticipation the paper claims for its DTEG.

    ASSUMED details, none pinned by the paper:
    - churn is excluded from future samples, since no future decision exists to
      compare against;
    - the window includes offset 0, so the term stays defined in the final
      ``horizon`` seconds of the dataset and degrades smoothly toward the
      current quality rather than dropping to zero (which would bias every
      link downward at the end of the run);
    - the mean is divided by (w1 + w2), the largest attainable quality, so the
      term lies in [0, 1].

Backward
    Per-satellite exponential moving average of realised link scores:

        A_t[s] = g * A_{t-1}[s] + (1 - g) * r_t[s],   g = exp(-step / horizon)
        r_t[s] = sum of s's selected link scores at t / (U * (w1 + w2 + w3))

    r_t is in [0, 1], so A is in [0, 1] for all t. A plain running *sum* would
    grow without bound, contradicting the paper's finiteness result; an EMA is
    one bounded reading of it, and the one used here.

    The spec reads "when i evaluates j, add j's accumulated score". Adding A[j]
    alone makes priority(i, j) != priority(j, i), which a bidirectional link
    cannot have: the two endpoints would rank the same edge differently. The
    edge term is therefore the mean of both endpoints, (A[i] + A[j]) / 2.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from leosim.dataset import LinkDataset

from .base import LinkSet, TopologySelector, b_matching
from .config import TopologyConfig
from .scoring import RunningNormaliser, churn_matrix, quality_matrix


class ScoredSelector(TopologySelector):
    """Shared machinery for DoTD and Greedy."""

    def reset(self) -> None:
        super().reset()
        self.normaliser = RunningNormaliser(self.dataset, self.config.latency_formula)
        self.accumulated = np.zeros(self.dataset.n_sats, dtype=np.float64)
        self.last_score: np.ndarray | None = None
        self.last_priority: np.ndarray | None = None

    # --- accumulation terms --------------------------------------------------

    def _step_offsets(self) -> np.ndarray:
        """Integer timestep offsets in the forward window, including 0."""
        step_s = self.dataset.config.step_s
        stride = max(1, int(round(self.config.horizon_stride_s / step_s)))
        horizon = max(1, int(round(self.config.horizon_s / step_s)))
        return np.arange(0, horizon + 1, stride)

    def _forward_term(self, t: int) -> np.ndarray:
        cfg = self.config
        quality_scale = cfg.w1_capacity + cfg.w2_latency
        n = self.dataset.n_sats
        if quality_scale == 0:
            return np.zeros((n, n))

        step_s = self.dataset.config.step_s
        total = np.zeros((n, n), dtype=np.float64)
        weight_sum = 0.0
        for off in self._step_offsets():
            tt = t + int(off)
            if tt >= self.dataset.n_steps:
                break
            w = float(np.exp(-(off * step_s) / cfg.horizon_s))
            total += w * quality_matrix(self.dataset, tt, self.normaliser, cfg)
            weight_sum += w
        return total / (weight_sum * quality_scale)

    def _backward_term(self) -> np.ndarray:
        a = self.accumulated
        return 0.5 * (a[:, None] + a[None, :])

    def _update_backward(self, links: LinkSet, score: np.ndarray) -> None:
        cfg = self.config
        total_weight = cfg.w1_capacity + cfg.w2_latency + cfg.w3_churn
        if total_weight == 0:
            return
        realised = np.where(links.adjacency, score, 0.0).sum(axis=1)
        r = realised / (cfg.u_max * total_weight)
        g = float(np.exp(-self.dataset.config.step_s / cfg.horizon_s))
        self.accumulated = g * self.accumulated + (1.0 - g) * r

    # --- selection -----------------------------------------------------------

    def _select(self, t: int) -> LinkSet:
        cfg = self.config
        ds = self.dataset
        visible = np.asarray(ds.visible[t])

        # Running max absorbs t before t is scored -- see RunningNormaliser.
        self.normaliser.update(t)

        prev_adj = None if self.prev is None else self.prev.adjacency
        score = quality_matrix(ds, t, self.normaliser, cfg) + cfg.w3_churn * churn_matrix(
            prev_adj, ds.n_sats
        )
        score = np.where(visible, score, 0.0)

        if cfg.lookahead == "forward":
            priority = score + cfg.w_lookahead * self._forward_term(t)
        elif cfg.lookahead == "backward":
            priority = score + cfg.w_lookahead * self._backward_term()
        elif cfg.lookahead == "none":
            priority = score
        else:
            raise ValueError(f"Unknown lookahead: {cfg.lookahead!r}")

        # argmin variant: negate, so the matcher's "largest first" takes the
        # lowest-scoring links. Candidates stay visible-only either way.
        signed = priority if cfg.objective == "max" else -priority
        links = b_matching(signed, visible, cfg.u_max, t)

        if cfg.lookahead == "backward":
            self._update_backward(links, score)

        self.last_score = score
        self.last_priority = priority
        return links


class DoTD(ScoredSelector):
    """DTEG-based Optimal Topology Design (Ron et al., INFOCOM 2025)."""

    def __init__(self, dataset: LinkDataset, config: TopologyConfig) -> None:
        if config.lookahead == "none":
            raise ValueError(
                "DoTD with lookahead='none' is Greedy. Use Greedy explicitly so "
                "the two cannot be confused in results."
            )
        super().__init__(dataset, config)
        self.name = f"DoTD-{config.lookahead}"


class Greedy(ScoredSelector):
    """DoTD's score at the current instant only -- no accumulation term.

    Forces ``lookahead='none'`` whatever the passed config says, so a config
    shared across all three algorithms cannot accidentally give Greedy a
    lookahead. Every other field, including both ambiguity flags, is inherited
    unchanged so the comparison stays like-for-like.
    """

    name = "Greedy"

    def __init__(self, dataset: LinkDataset, config: TopologyConfig) -> None:
        super().__init__(dataset, dataclasses.replace(config, lookahead="none"))
