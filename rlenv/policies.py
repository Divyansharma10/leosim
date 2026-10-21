"""Policies with the interface an agent will have: ``act(obs) -> action``.

Each reads only the observation, so it runs through ``env.step`` exactly as an
agent would; none touches the environment's internals.
"""

from __future__ import annotations

import numpy as np


class BackpressurePolicy:
    """Krishnan et al. Eq. 16: activate k -> m iff (Q_k - Q_m) C_km > 0.

    Every link with a positive weight is activated, not only the best one per
    satellite: the paper states activation "if s > 0" and no matching. Read on
    the normalised queue q / (q + q_ref), which is strictly increasing in q with
    the same q_ref for every satellite, so the sign -- all that decides -- is
    the raw-queue sign.
    """

    name = "Backpressure"

    def act(self, obs: dict) -> np.ndarray:
        q, nbr = obs["queue"].astype(np.float64), obs["neighbours"]
        mask = obs["action_mask"].astype(bool)
        q_nbr = q[np.maximum(nbr, 0)]
        weight = (q[:, None] - q_nbr) * obs["isl_capacity"]
        return ((weight > 0) & mask).astype(np.int8)


class ShortestPathPolicy:
    """Stage 4's routing as a policy: each satellite sends only to its parent
    in the hop-shortest forest grown from every satellite with a gateway in
    view; those satellites forward nothing and drain to the gateway.

    Built from the observation alone (``downlink_capacity > 0`` marks the
    roots), so it runs through the same path as every other policy. It exists
    to separate the queue model from the routing in the comparison against
    Phase 2: same routing as ``TrafficSim`` (hop metric), different queues.
    Ties go to the lowest neighbour slot, i.e. the lowest satellite index;
    Dijkstra's tie-break in stage 4 may differ. ASSUMED.
    """

    name = "Shortest-path"

    def act(self, obs: dict) -> np.ndarray:
        nbr, mask = obs["neighbours"], obs["action_mask"].astype(bool)
        n = nbr.shape[0]
        dist = np.where(obs["downlink_capacity"] > 0, 0, np.iinfo(np.int64).max)
        frontier = np.nonzero(dist == 0)[0]
        level = 0
        while frontier.size:
            level += 1
            nxt = np.unique(nbr[frontier][mask[frontier]])
            nxt = nxt[dist[nxt] > level]
            dist[nxt] = level
            frontier = nxt
        out = np.zeros(nbr.shape, dtype=np.int8)
        for i in range(n):
            if dist[i] in (0, np.iinfo(np.int64).max):
                continue
            k = np.nonzero(mask[i] & (dist[np.maximum(nbr[i], 0)] == dist[i] - 1))[0][0]
            out[i, k] = 1
        return out


class RandomPolicy:
    """Each slot independently on with probability ``p``, padding included:
    the environment must ignore padding, and this exercises it."""

    def __init__(self, p: float = 0.5, seed: int = 0) -> None:
        self.p, self.rng, self.name = p, np.random.default_rng(seed), "Random"

    def act(self, obs: dict) -> np.ndarray:
        return (self.rng.random(obs["action_mask"].shape) < self.p).astype(np.int8)


class ConstantPolicy:
    """All slots on, or all off. All-off is Krishnan et al.'s No-ISL baseline:
    a satellite drains only to a gateway it sees itself."""

    def __init__(self, on: bool) -> None:
        self.on, self.name = on, "All-on" if on else "No-ISL"

    def act(self, obs: dict) -> np.ndarray:
        return np.full(obs["action_mask"].shape, int(self.on), dtype=np.int8)


def rollout(env, policy, seed: int, options: dict | None = None) -> dict:
    """One episode through ``env.step``. Returns per-step arrays and totals."""
    obs, info = env.reset(seed=seed, options=options)
    rows, rewards, done = [], [], False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy.act(obs))
        rewards.append(r)
        rows.append(info)
        done = terminated or truncated
    arrived = sum(x["arrived_bits"] for x in rows)
    delivered = sum(x["delivered_bits"] for x in rows)
    dropped = sum(x["dropped_bits"] for x in rows)
    return {
        "policy": policy.name,
        "seed": seed,
        "steps": len(rows),
        "reward_mean": float(np.mean(rewards)),
        "queue_mean": float(np.mean([x["queue_mean"] for x in rows])),
        "queue_max": float(np.max([x["queue_max"] for x in rows])),
        "arrived_bits": arrived,
        "delivered_bits": delivered,
        "dropped_bits": dropped,
        "final_backlog_bits": rows[-1]["backlog_bits"],
        "delivered_frac": delivered / arrived if arrived else np.nan,
        "dropped_frac": dropped / arrived if arrived else np.nan,
        "rewards": np.asarray(rewards),
        "rows": rows,
    }
