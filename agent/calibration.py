"""Q calibration: does the network's Q match the return its own policy earns?

The diagnostic the 2026-11-19 300-episode run lacked. There, Q(s, a) climbed
to +470 and the TD targets to +2400 while per-step rewards stayed in
[-13, +0.3], and nothing reported it: the learning curve shows gain and
deviation, both of which only say the policy got worse, not why.

For a greedy rollout with rewards r_0 .. r_{T-1} (the env's, already scaled),

    G_t = sum_{k=0}^{T-1-t} gamma^k r_{t+k}            (backward recursion)

is the realised discounted return from s_t under the policy Q is greedy for,
and Q(s_t, a_t) is the network's estimate of it. Training bootstraps through
time-limit truncation, so Q's horizon is unbounded while G_t stops at the
episode end. Only states with at least ``tail_steps(gamma, tol)`` steps left
are compared, which bounds the dropped tail at ``tol`` of the full geometric
weight 1 / (1 - gamma). Choosing tol is the only judgement; it is
TrainConfig.calib_tail_tol.

Q(s, a) estimates the return of the greedy policy only once learning has
converged, so a calibrated agent is not proven correct by this. A badly
miscalibrated one is proven wrong: its Q is not the value of anything it does.
"""

from __future__ import annotations

import math

import numpy as np


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    """(T,) G_t = r_t + gamma G_{t+1}, G_T = 0. Exact backward recursion."""
    r = np.asarray(rewards, dtype=np.float64)
    out = np.empty_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        out[t] = acc
    return out


def tail_steps(gamma: float, tol: float) -> int:
    """Smallest L with gamma^L <= tol: states with >= L steps left drop at
    most tol of the geometric weight when the return is truncated."""
    if not 0 < tol < 1:
        raise ValueError("tol must be in (0, 1)")
    if gamma == 0:
        return 1
    return max(1, math.ceil(math.log(tol) / math.log(gamma)))


def calibration(q_taken, rewards, gamma: float, tol: float) -> dict:
    """Compare per-step Q(s_t, a_t) with realised G_t over the usable prefix.

    Returns n (states compared), q_mean, g_mean, bias = mean(Q - G),
    mae = mean |Q - G| and g_abs = mean |G| (the return's own scale). With
    no usable state (episode shorter than the tail) every value is NaN and
    n = 0, never a silent zero."""
    q = np.asarray(q_taken, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    if q.shape != r.shape:
        raise ValueError(f"q {q.shape} and rewards {r.shape} differ")
    n = len(r) - tail_steps(gamma, tol) + 1
    if n <= 0:
        nan = float("nan")
        return {"n": 0, "q_mean": nan, "g_mean": nan, "bias": nan, "mae": nan, "g_abs": nan}
    g = discounted_returns(r, gamma)[:n]
    d = q[:n] - g
    return {"n": int(n), "q_mean": float(q[:n].mean()), "g_mean": float(g.mean()), "bias": float(d.mean()),
            "mae": float(np.abs(d).mean()), "g_abs": float(np.abs(g).mean())}
