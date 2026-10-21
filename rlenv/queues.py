"""Per-satellite queues with directed forwarding: Krishnan et al. Eq. 9-14.

No Gym here. The environment calls ``step``; so does the validation's direct
reference loop, which is how "the wrapper changes nothing" is checked.

One aggregate queue Q_k per satellite, all traffic anycast to any gateway in
view. Within a step, every satellite acts on its start-of-step queue at once:

    v_k = min(Q_k, downlink_k * dt)                      downlink, automatic
    w_km = f_km * c_km * dt * s_k                        ISL k -> m
    s_k  = min(1, (Q_k - v_k) / sum_m f_km c_km dt)      never send what k lacks
    Q_k' = Q_k - v_k - sum_m w_km + sum_i w_ik + u_k     (Eq. 14)
    dropped_k = max(0, Q_k' - B);  Q_k' = min(Q_k', B)    B = inf by default

f is the binary directed activation. An active link carries its full capacity
(Eq. 11, 18); how a satellite holding less than its active links could carry
is not stated in the paper, so the shortfall is shared in proportion to
capacity. ASSUMED. Received bits and new arrivals are not forwarded until the
next step, as in Eq. 14.

Conservation, exactly, per step: sum Q + sum u = sum Q' + sum v + sum dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepResult:
    queue: np.ndarray  # (N,) bits after the step
    delivered: np.ndarray  # (N,) bits down to a gateway
    sent: np.ndarray  # (N, N) bits k -> m over ISLs
    dropped: np.ndarray  # (N,) bits over the buffer
    arrivals: np.ndarray  # (N,) bits from users


def forward_step(
    queue: np.ndarray,
    forward: np.ndarray,
    isl_capacity: np.ndarray,
    downlink_capacity: np.ndarray,
    arrivals: np.ndarray,
    dt: float,
    buffer_bits: float,
) -> StepResult:
    """One step from ``queue``; pure, mutates nothing.

    ``forward`` (N, N) bool, directed; ``isl_capacity`` (N, N) bits/s, zero
    where no selected link exists; ``downlink_capacity`` (N,) bits/s.
    """
    q = np.asarray(queue, dtype=np.float64)
    v = np.minimum(q, downlink_capacity * dt)
    rem = q - v
    offer = np.where(forward, isl_capacity, 0.0) * dt
    total = offer.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > rem, rem / total, 1.0)
    sent = offer * share[:, None]
    new = q - v - sent.sum(axis=1) + sent.sum(axis=0) + arrivals
    # Floating-point remainder of rem - sum(sent): never a real negative queue.
    new = np.maximum(new, 0.0)
    dropped = np.maximum(new - buffer_bits, 0.0)
    return StepResult(np.minimum(new, buffer_bits), v, sent, dropped, np.asarray(arrivals, dtype=np.float64))


class QueueModel:
    """Queue state plus ``forward_step``. ``snapshot``/``restore`` exist for the
    shadow-backpressure reward, which needs a counterfactual from the same
    pre-step state (see ``rlenv.reward``)."""

    def __init__(self, n_sats: int, dt: float, buffer_bits: float) -> None:
        self.n, self.dt, self.buffer_bits = n_sats, float(dt), float(buffer_bits)
        self.queue = np.zeros(n_sats)

    def reset(self) -> None:
        self.queue = np.zeros(self.n)

    def peek(self, forward, isl_capacity, downlink_capacity, arrivals) -> StepResult:
        return forward_step(self.queue, forward, isl_capacity, downlink_capacity, arrivals,
                            self.dt, self.buffer_bits)

    def step(self, forward, isl_capacity, downlink_capacity, arrivals) -> StepResult:
        r = self.peek(forward, isl_capacity, downlink_capacity, arrivals)
        self.queue = r.queue
        return r

    def snapshot(self) -> np.ndarray:
        return self.queue.copy()

    def restore(self, queue: np.ndarray) -> None:
        self.queue = np.asarray(queue, dtype=np.float64).copy()
