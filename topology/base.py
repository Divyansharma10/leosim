"""Common interface, link-set container, constraint enforcement and matching.

Every selector returns a :class:`LinkSet` and every LinkSet passes through
:func:`assert_valid` before it leaves ``step()``. The constraints are enforced
twice on purpose: once by construction in :func:`b_matching`, once by an
independent check that does not share code with the construction. A bug in the
matcher cannot then also hide itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from leosim.dataset import LinkDataset

from .config import TopologyConfig


@dataclass(frozen=True)
class LinkSet:
    """Undirected selected-link set for one timestep.

    ``adjacency`` is the single source of truth: an (N, N) boolean matrix,
    symmetric, zero diagonal. Everything else is derived from it.
    """

    adjacency: np.ndarray
    t: int

    @property
    def n_sats(self) -> int:
        return self.adjacency.shape[0]

    @property
    def edges(self) -> np.ndarray:
        """(E, 2) array of i<j pairs."""
        i, j = np.nonzero(np.triu(self.adjacency, k=1))
        return np.column_stack([i, j])

    @property
    def n_links(self) -> int:
        return int(np.triu(self.adjacency, k=1).sum())

    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    @classmethod
    def empty(cls, n_sats: int, t: int) -> "LinkSet":
        return cls(np.zeros((n_sats, n_sats), dtype=bool), t)


class ConstraintViolation(AssertionError):
    """A selected topology broke a hard constraint."""


def assert_valid(links: LinkSet, dataset: LinkDataset, u_max: int) -> None:
    """Raise :class:`ConstraintViolation` unless all hard constraints hold.

    Checked independently of how the set was built:

    1. symmetric      -- i linked to j iff j linked to i
    2. no self-links  -- zero diagonal
    3. degree <= U    -- per satellite
    4. visible        -- every selected link is visible in stage-1 data, which
                         covers both Earth occlusion and the range cap, since
                         stage 1 folds both into ``visible``
    """
    a = links.adjacency
    t = links.t

    if a.shape != (dataset.n_sats, dataset.n_sats) or a.dtype != bool:
        raise ConstraintViolation(
            f"t={t}: adjacency must be bool ({dataset.n_sats},{dataset.n_sats}), "
            f"got {a.dtype} {a.shape}"
        )

    asym = np.argwhere(a != a.T)
    if asym.size:
        i, j = asym[0]
        raise ConstraintViolation(
            f"t={t}: not symmetric -- {len(asym) // 2} asymmetric pairs, "
            f"first ({i},{j}): a[i,j]={a[i, j]} a[j,i]={a[j, i]}"
        )

    if np.any(np.diagonal(a)):
        raise ConstraintViolation(f"t={t}: self-link on sat {int(np.argmax(np.diagonal(a)))}")

    deg = a.sum(axis=1)
    if deg.max(initial=0) > u_max:
        worst = int(np.argmax(deg))
        raise ConstraintViolation(
            f"t={t}: sat {worst} has degree {int(deg[worst])} > U={u_max}"
        )

    bad = a & ~np.asarray(dataset.visible[t])
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise ConstraintViolation(
            f"t={t}: {int(bad.sum()) // 2} selected links not visible in stage-1 data, "
            f"first ({i},{j}) at {float(dataset.distance[t, i, j]):.1f} km"
        )


def b_matching(
    priority: np.ndarray,
    candidates: np.ndarray,
    u_max: int,
    t: int,
) -> LinkSet:
    """Degree-constrained edge selection by descending priority.

    Walks candidate edges from highest to lowest ``priority`` and accepts an
    edge only if both endpoints still have spare terminals. Because an edge is
    written into both ``adj[i, j]`` and ``adj[j, i]`` in the same statement,
    selection is bidirectional by construction: there is no step at which i has
    picked j but j has not picked i.

    Substitution note: the paper solves this with dynamic programming (claimed
    O(M^2)). This is the standard greedy b-matching instead, which is not
    guaranteed optimal. DoTD and Greedy both use it, so it cannot account for
    any difference between them -- they differ only in ``priority``.

    Parameters
    ----------
    priority   : (N, N) float, symmetric. Larger is taken first. Callers wanting
                 argmin semantics negate their score before passing it in.
    candidates : (N, N) bool, symmetric. Edges eligible at all.
    """
    n = priority.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    mask = candidates[iu, ju]
    iu, ju = iu[mask], ju[mask]
    p = priority[iu, ju]

    # Stable descending sort with an explicit (i, j) tie-break, so identical
    # scores always resolve the same way and runs are exactly reproducible.
    order = np.lexsort((ju, iu, -p))

    adj = np.zeros((n, n), dtype=bool)
    deg = np.zeros(n, dtype=np.int64)
    for k in order:
        i, j = iu[k], ju[k]
        if deg[i] < u_max and deg[j] < u_max:
            adj[i, j] = adj[j, i] = True
            deg[i] += 1
            deg[j] += 1

    return LinkSet(adj, t)


class TopologySelector(ABC):
    """Stateful per-timestep topology selector.

    Stateful because three quantities cross timestep boundaries: the previous
    link set (churn), the running normalisation maxima, and the accumulated
    lookahead score. Steps must therefore be taken in increasing order from a
    fresh ``reset()``; :meth:`step` enforces that rather than trusting callers.
    """

    name: str = "selector"

    def __init__(self, dataset: LinkDataset, config: TopologyConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.prev: LinkSet | None = None
        self._last_t: int | None = None

    def step(self, t: int) -> LinkSet:
        if not 0 <= t < self.dataset.n_steps:
            raise IndexError(f"t={t} outside dataset of {self.dataset.n_steps} steps")
        if self._last_t is not None and t != self._last_t + 1:
            raise ValueError(
                f"{self.name}: steps must be consecutive; got t={t} after t={self._last_t}. "
                "Call reset() to start over."
            )
        if self._last_t is None and t != 0:
            raise ValueError(f"{self.name}: first step after reset() must be t=0, got t={t}")

        links = self._select(t)
        assert_valid(links, self.dataset, self.config.u_max)

        self.prev = links
        self._last_t = t
        return links

    def run(self, n_steps: int | None = None) -> list[LinkSet]:
        """Reset and step through ``n_steps`` timesteps (default: all)."""
        self.reset()
        n = self.dataset.n_steps if n_steps is None else n_steps
        return [self.step(t) for t in range(n)]

    @abstractmethod
    def _select(self, t: int) -> LinkSet:
        """Choose the link set at ``t``. ``self.prev`` holds t-1, or None."""
