"""Rewards: the absolute queue reward and Krishnan et al.'s Eq. 20.

A reward is any callable ``reward_fn(ctx) -> float``, optionally with a
``reset(env)`` method that the env calls at every reset. The environment
builds the ``RewardContext``, never computes a reward itself, and multiplies
whatever comes back by ``EnvConfig.reward_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .queues import StepResult

if TYPE_CHECKING:
    from .env import LeoForwardingEnv


@dataclass(frozen=True)
class RewardContext:
    prev_queue: np.ndarray  # (N,) bits before the step
    result: StepResult  # the step as taken
    action: np.ndarray  # (N, D) as applied, mask already enforced
    t: int  # timestep the action was applied at
    queue_ref_bits: float
    # Steps the queue model from prev_queue with a DIFFERENT action at the same
    # t, mutating nothing. A one-step counterfactual; the shadow reward does NOT
    # use it (it runs backpressure in parallel), but it is kept for analysis.
    counterfactual: Callable[[np.ndarray], StepResult]
    isl_capacity: np.ndarray  # (N, N) bits/s at t, selected links only
    downlink_capacity: np.ndarray  # (N,) bits/s at t

    @property
    def next_queue_norm(self) -> np.ndarray:
        """Linear q / q_ref, NOT the observation's q / (q + q_ref): the
        reward must keep growing with the queue."""
        return self.result.queue / self.queue_ref_bits


class QueueReward:
    """r = -(w_mean * mean q + w_max * max q), q = queue / q_ref, after the step.

    <= 0 and finite for finite queues; unbounded below, since an overloaded
    queue grows without limit. Ignores drops, which is sound only with
    unbounded queues (see rlenv.config).
    """

    def __init__(self, w_mean: float, w_max: float) -> None:
        self.w_mean, self.w_max = float(w_mean), float(w_max)

    def __call__(self, ctx: RewardContext) -> float:
        q = ctx.next_queue_norm
        return -(self.w_mean * float(q.mean()) + self.w_max * float(q.max()))


class ShadowBackpressureReward:
    """Krishnan et al. Eq. 20 with a PARALLEL backpressure run:

        r = -[ w_mean (mean q_a - mean q_BP) + w_max (max q_a - max q_BP) ]

    q_BP comes from a second ``QueueModel`` owned by this object: empty at
    every reset, stepped once per env step with backpressure's action on ITS
    OWN queues, with the same topology, capacities and arrivals as the agent.
    It is never reset to the agent's state; that would be the one-step
    counterfactual, which the paper does not rule out but was not chosen.

    Backpressure's action is produced through the env's own observation and
    action decoding (``env.observe``, ``env.forward_for``), the identical path
    the baseline takes when it is the agent, so the shadow and a real
    backpressure rollout are the same trajectory bit for bit. Per step this
    equals QueueReward(agent) - QueueReward(backpressure): 0 when the agent is
    backpressure, positive when it beats it.
    """

    def __init__(self, w_mean: float, w_max: float, policy=None) -> None:
        from .policies import BackpressurePolicy
        self.w_mean, self.w_max = float(w_mean), float(w_max)
        self.policy = policy if policy is not None else BackpressurePolicy()
        self.env = self.shadow = self.last = None

    def reset(self, env: "LeoForwardingEnv") -> None:
        from .queues import QueueModel
        self.env = env
        self.shadow = QueueModel(env.n_sats, env.queues.dt, env.queues.buffer_bits)
        self.last = None

    def __call__(self, ctx: RewardContext) -> float:
        if self.shadow is None:
            raise RuntimeError("shadow reward used before env.reset()")
        obs = self.env.observe(self.shadow.queue, ctx.t)
        forward = self.env.forward_for(self.policy.act(obs), ctx.t)
        self.last = self.shadow.step(forward, ctx.isl_capacity, ctx.downlink_capacity, ctx.result.arrivals)
        qa = ctx.next_queue_norm
        qb = self.last.queue / ctx.queue_ref_bits
        return -(self.w_mean * float(qa.mean() - qb.mean()) + self.w_max * float(qa.max() - qb.max()))
