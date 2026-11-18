"""Phase 3 configuration: the RL forwarding environment.

Same convention as the other stages: every tunable lives here, fields marked
ASSUMED were pinned by neither the base paper nor Krishnan et al. (arXiv
2601.13662), and every modelling choice is stamped into ``variant_label``.

Nothing here assumes a constellation size. N comes from the stage-1 dataset
(``n_planes * sats_per_plane`` is an upper bound; a short plane contributes
what it has), and the per-satellite slot count D is stage 2's ``u_max``. The
observation and action shapes are (N,) and (N, D); a policy that must transfer
across N has to be size-invariant itself.

What follows Krishnan et al. (Sec. II-III), and what does not:

- One aggregate queue per satellite; traffic is user uplink, anycast to any
  gateway in view (their Eq. 14, 12, 9). Followed.
- The action is one bit per DIRECTED selected link, k -> m (their Eq. 18).
  Followed. The downlink is not in the action: it drains automatically,
  min(Q, downlink capacity x dt), through the best gateway in view (Eq. 9).
- Queues are unbounded by default, as theirs are. A finite buffer is available
  (``buffer_s``) but is a trap under the queue-only reward: dropped bits leave
  the queue, so dropping LOWERS the penalty. Measured: at load 2 with a 5 s
  buffer, Random out-scored backpressure (-1.11 vs -1.30) while delivering
  47% against 62%. Do not enable it without a drop term in the reward.
- Arrivals: they integrate population density over each satellite's
  footprint (Eq. 6-7). We reuse stage 4's 200 cities and Poisson flows; each
  city's bits join the satellite with its highest-capacity access link. ASSUMED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from traffic.config import GatewaySites

StartMode = Literal["fixed", "uniform"]
RewardKind = Literal["queue", "shadow"]
SelectorName = Literal["+Grid", "Greedy", "DoTD-forward", "DoTD-backward"]


@dataclass(frozen=True)
class EnvConfig:
    # --- constellation (stage 1) ------------------------------------------------
    # 3 x 8 = 24 satellites. Chosen from a measured sweep of 24-satellite shapes
    # (2x12 ... 12x2) under populous20 gateways: 3x8 is connected at every step
    # under Greedy, has the highest selected degree (3.71 of 4, so the most live
    # action slots), and has a gateway in view on 13.6% of satellite-steps and
    # on at least one satellite at every step. Wider shapes (4x6, 8x3, 12x2)
    # split into ~2 components. 2x12 sees more gateway (20%) but has only two
    # planes of cross-plane structure. ASSUMED default.
    n_planes: int = 3
    sats_per_plane: int = 8
    step_s: float = 1.0
    duration_s: float = 1800.0

    # --- topology (stage 2) ------------------------------------------------------
    # Fixed for the episode; the action forwards over what it chose.
    selector: SelectorName = "Greedy"

    # --- episode -----------------------------------------------------------------
    # The episode covers timesteps start_step .. start_step + episode_steps; the
    # observation at the last of those is returned with truncated=True, so that
    # index must exist in the dataset.
    start_step: int = 0
    episode_steps: int = 200

    # "fixed": every episode starts at start_step. "uniform": reset() draws the
    # start uniformly from every start that fits, from the env's seeded RNG, so
    # a seed fixes it. Needed for training and for any selector comparison: the
    # scored selectors can agree for the first hundreds of steps and diverge
    # later (5x8: first difference at step 342; 4x15: 477), so fixed episodes
    # at step 0 see no topology difference at all. A reset option
    # {"start_step": k} overrides either mode.
    start_mode: StartMode = "fixed"

    # --- load ----------------------------------------------------------------------
    # Offered load as a fraction of the subset's own downlink capacity:
    #     load = mean_t(expected demand on cities this subset can serve)
    #            / mean_t(sum over satellites of best downlink capacity)
    # averaged over the whole stage-1 window. Scaled to the subset because a
    # global 2 Gbps peak puts only ~0.5 Gbps on 24 satellites with ~3 Gbps of
    # downlink, which never congests. load >= 1 cannot all be delivered by any
    # policy; load < 1 still congests locally, since most satellites see no
    # gateway and must forward over ~1 Gbps ISLs. ASSUMED definition.
    load: float = 1.0

    # Diurnal shape and flow size: stage 4's (TrafficConfig defaults). The
    # Poisson seed is the episode seed, set at reset().
    gateway_sites: GatewaySites = "populous20"
    seed: int = 7

    # --- queues --------------------------------------------------------------------
    # Queue scale q_ref = queue_scale_s x the window's largest downlink
    # capacity: about how long the best satellite needs to drain its queue.
    # Observation: q / (q + q_ref), in [0, 1) for any finite queue, monotone,
    # so nothing is clipped away. Reward: the linear q / q_ref, which does not
    # saturate. ASSUMED: 5 s.
    queue_scale_s: float = 5.0

    # Per-satellite buffer in the same units; None = unbounded (Krishnan et al.).
    # See the warning above before setting it.
    buffer_s: float | None = None

    # --- reward --------------------------------------------------------------------
    # r = -(w_mean * mean_k q_k + w_max * max_k q_k) on normalised queues (the
    # absolute form of Krishnan's Eq. 20 alpha, beta; they give no values).
    # ASSUMED: equal weights.
    w_mean: float = 1.0
    w_max: float = 1.0

    # "queue": the absolute form above. "shadow": Krishnan et al. Eq. 20,
    #     r = -[w_mean (mean q_a - mean q_BP) + w_max (max q_a - max q_BP)],
    # with q_BP from a PARALLEL backpressure run: its own queues, from empty at
    # reset, fed the same arrivals and topology, never reset to the agent's
    # state. Equals queue-reward(agent) - queue-reward(backpressure) per step,
    # so 0 when the agent is backpressure and > 0 when it beats it. The paper
    # does not say one-step or parallel; parallel was chosen (2026-10-16).
    reward_kind: RewardKind = "queue"

    # Every reward is multiplied by this before the env returns it, whatever
    # reward_fn produced it. 1 = unscaled. Not chosen: under overload the
    # queue reward reaches ~-40..-85 per step at load 2 on 3x8, the shadow
    # reward is a difference of two such terms. ASSUMED placeholder.
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n_planes < 1 or self.sats_per_plane < 1:
            raise ValueError("n_planes and sats_per_plane must be >= 1")
        if self.episode_steps < 1 or self.start_step < 0:
            raise ValueError("episode_steps must be >= 1 and start_step >= 0")
        if self.load < 0 or self.queue_scale_s <= 0 or self.step_s <= 0:
            raise ValueError("load must be >= 0; queue_scale_s and step_s > 0")
        if self.buffer_s is not None and self.buffer_s <= 0:
            raise ValueError("buffer_s must be > 0 or None")
        if self.w_mean < 0 or self.w_max < 0 or self.w_mean + self.w_max <= 0:
            raise ValueError("reward weights must be >= 0 and not both 0")
        if not self.reward_scale > 0:
            raise ValueError("reward_scale must be > 0")

    @property
    def n_sats_max(self) -> int:
        return self.n_planes * self.sats_per_plane

    @property
    def variant_label(self) -> str:
        return (
            f"{self.n_planes}x{self.sats_per_plane}, step={self.step_s:g}s, selector={self.selector}, "
            f"load={self.load:g}, gateways={self.gateway_sites}, qscale={self.queue_scale_s:g}s, "
            f"buffer={'inf' if self.buffer_s is None else f'{self.buffer_s:g}s'}, "
            f"episode={self.start_step if self.start_mode == 'fixed' else 'uniform'}+{self.episode_steps}, "
            f"reward={self.reward_kind}({self.w_mean:g},{self.w_max:g})x{self.reward_scale:g}"
        )


@dataclass(frozen=True)
class RLValidationConfig:
    """Bounds for validate_rlenv.py. Margins, not results: each is set well
    inside the measured gap so the check discriminates without being fragile.
    ASSUMED, all of them."""

    seeds: tuple[int, ...] = (7, 8, 9)
    # Random must lose to backpressure on EVERY seed, and on the mean by at
    # least this factor in episode-mean reward (rewards are <= 0). Measured
    # at load 1, seed 7: backpressure -0.74, Random -19.5.
    random_reward_factor: float = 2.0
    # Load at which observations must reach saturation, and how close to 1 the
    # largest queue observation must get there (discrimination for the bound).
    overload: float = 2.0
    saturation: float = 0.9
    # Curve-shape comparison against stage 4 (Phase 2), same routing on both
    # sides: the delivered fraction of the env's shortest-path policy vs
    # TrafficSim's OSPF, max |difference| over the load grid. Measured 0.071.
    curve_loads: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    curve_tol: float = 0.10
    # Delivered/arrived includes bits still queued when the episode ends, so at
    # light load it wanders by ~0.01 with load; "falls with load" allows this.
    # (1e-3 failed on 2x12: 0.993 -> 0.994 between loads 0.25 and 0.5.)
    monotone_tol: float = 0.005
    # Backpressure may trail shortest path at light load (it parks bits away
    # from gateways while queues are near-equal); by at most this much.
    bp_vs_sp_tol: float = 0.01
    # Cross-machine float tolerances (2026-10-26). Exact bytes do not survive
    # a change of numpy build: a reduction's last bit depends on its SIMD
    # path. Same-machine identity is still checked exactly where no reduction
    # is involved (per-step queue and delivered vectors).
    # Relative tolerance on a summed episode total. One bit of 5.9e11 is
    # 1.7e-12, so this still catches a single misdelivered bit per episode.
    sum_rtol: float = 1e-12
    # Max |queue - reference| / max |reference| over the pinned backpressure
    # trajectory. A changed forwarding decision moves queues by ~1e-3 of their
    # scale; reduction-order noise is ~1e-16. Discriminated against seed 8.
    queue_rtol: float = 1e-9
