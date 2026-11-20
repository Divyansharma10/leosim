"""Phase 3 step 2 configuration: a residual Double DQN on backpressure.

Same convention as every other stage: every tunable lives here, fields marked
ASSUMED were pinned by neither the base paper nor Krishnan et al. (arXiv
2601.13662), and every modelling choice is stamped into ``variant_label``.

What Krishnan et al. give, and what they do not (read 2026-11-18, HTML v1):

- "3 layers, 256 hidden units each", Double DQN with a target network
  (van Hasselt et al. 2016), reward Eq. 20, "100 episodes". Followed.
- The residual: the prose says the network "outputs an additive correction
  (or residual) to the backpressure-derived action", but their Eq. 19 is a
  closed form, s_BP + lambda * C_downlink(m), with no network in it. How the
  learned output enters the decision is not stated. Ours is below (ASSUMED).
- Per-link or flat network, features, gamma, learning rate, optimiser, batch,
  buffer, target updates, epsilon schedule, loss, alpha/beta: not stated.
  All ASSUMED.

The residual (``residual_margin`` = kappa). Q is decomposed around
backpressure (agent.ddqn, agent.network):

    Q(s, a) = V(s) + sum_l 1[a_l != a_BP(s, l)] * A(s, l)
    deviate on slot l iff gain * A(s, l) > kappa

A is the learned correction to backpressure's action. With gain = 0 nothing
deviates and the action is backpressure's bit, exactly, with no float
sensitivity: that is the regression guard. Two per-slot-head designs were
tried first and failed (measured, see agent.ddqn).

kappa is the most arbitrary number here (the role w_lookahead plays in stage
2): it is the estimated gain the agent needs before leaving backpressure on a
link, in reward-scale units. Since 2026-11-20 Q's deviation term is a mean
over live slots (agent.network), so A(s, l) is what deviating on slot l would
be worth if every slot did so: its scale no longer depends on N D. Every
kappa value used before that change (0.001, 0.01, 0.1 in the cluster array)
was in summed, N-dependent units and is not comparable to any value after it.
kappa sets how often the agent can deviate at all. Sweep it before trusting any margin, and read the
deviation fraction beside every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rlenv.config import RewardKind, SelectorName

Device = str  # "cpu", "cuda", "cuda:1", ...

# Fixed reward scales, one per reward kind, chosen ONCE (2026-11-18) and never
# varied between runs, or no two runs are comparable. Measured on 6x12,
# krishnan8, loads 0.5/1/2, backpressure with each live slot flipped at
# eps 0.01-0.1, 2 seeds x 200 steps (results/agent/reward_scale.txt):
#   shadow |r| p95 0.30-3.4 (max 4.3)   -> 1.0, already order 1
#   queue  |r| p95 1.4-16   (max 17.7)  -> 0.1
REWARD_SCALE: dict[str, float] = {"shadow": 1.0, "queue": 0.1}


@dataclass(frozen=True)
class AgentConfig:
    # --- network (Krishnan et al.: 3 layers, 256 hidden units) ---------------------
    # One MLP shared by every directed slot, so the same weights run at any N
    # (see features.py for the inputs). Krishnan et al.'s state (Eq. 17) is
    # written over the whole constellation and may mean a flat network; that
    # cannot transfer across N. ASSUMED: per-link.
    hidden: tuple[int, ...] = (256, 256, 256)
    # Output layer weights ~ N(0, init_std^2), bias 0, so an untrained
    # residual is ~1e-4 of kappa: "starts near zero", not exactly zero.
    init_std: float = 1e-5
    # Hop distances to the nearest gateway-seeing satellite enter the features
    # as min(h, max_hops) / max_hops; unreachable = 1. ASSUMED.
    max_hops: int = 8

    # --- residual --------------------------------------------------------------------
    residual_margin: float = 0.01  # kappa, in reward-scale units. ASSUMED; sweep it.

    # --- Double DQN --------------------------------------------------------------------
    gamma: float = 0.95  # ~20 s horizon at 1 s steps. ASSUMED.
    lr: float = 1e-4  # Adam. ASSUMED.
    batch_size: int = 64  # transitions; each carries every slot. ASSUMED.
    buffer_size: int = 50_000  # transitions, ring, uniform sampling. ASSUMED.
    learning_starts: int = 1_000  # transitions before the first update. ASSUMED.
    train_every: int = 1  # env steps per gradient update. ASSUMED.
    target_sync: int = 1_000  # gradient updates between hard target copies. ASSUMED.
    huber_delta: float = 1.0
    grad_clip: float = 10.0  # max global grad norm. ASSUMED.

    # --- exploration -------------------------------------------------------------------
    # Each live slot's greedy bit flipped with probability eps, linear from
    # eps_start to eps_end over eps_decay_steps env steps. NOT eps = 1 at the
    # start: fully random forwarding scores -19.5 against backpressure's -0.74
    # (3x8, load 1), and exploring there would discard the prior. ASSUMED.
    eps_start: float = 0.1
    eps_end: float = 0.01
    eps_decay_steps: int = 20_000

    # --- reward ------------------------------------------------------------------------
    reward_kind: RewardKind = "shadow"

    # --- run -------------------------------------------------------------------------------
    device: Device = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.hidden or min(self.hidden) < 1:
            raise ValueError("hidden must be non-empty, all >= 1")
        if not self.residual_margin > 0:
            raise ValueError("residual_margin must be > 0: it is what makes gain=0 exactly backpressure")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if not 0 <= self.eps_end <= self.eps_start <= 1:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if min(self.batch_size, self.buffer_size, self.target_sync, self.train_every, self.eps_decay_steps) < 1:
            raise ValueError("batch_size, buffer_size, target_sync, train_every, eps_decay_steps must be >= 1")
        if self.reward_kind not in REWARD_SCALE:
            raise ValueError(f"no fixed reward scale for {self.reward_kind!r}")

    @property
    def reward_scale(self) -> float:
        return REWARD_SCALE[self.reward_kind]

    @property
    def variant_label(self) -> str:
        h = "x".join(str(x) for x in self.hidden)
        return (f"ddqn-residual(kappa={self.residual_margin:g}, mlp={h}, gamma={self.gamma:g}, "
                f"lr={self.lr:g}, eps={self.eps_start:g}->{self.eps_end:g}/{self.eps_decay_steps}, "
                f"sync={self.target_sync}, reward={self.reward_kind}x{self.reward_scale:g})")


@dataclass(frozen=True)
class TrainConfig:
    """What the agent trains and is evaluated on. Episodes draw (selector,
    load, start, demand seed) from the trainer's seeded RNG."""

    episodes: int = 100  # Krishnan et al.'s count. ASSUMED as a default only.
    # Mixed over all four selectors so one agent is compared on every
    # topology; the network is topology-agnostic. ASSUMED.
    selectors: tuple[SelectorName, ...] = ("Greedy", "DoTD-forward", "DoTD-backward", "+Grid")
    loads: tuple[float, ...] = (0.5, 1.0, 2.0)
    # Time is held out as well as demand: training starts are uniform in
    # [0, train_start_max], so training actions are applied at t <= 1198
    # (999 + 200 - 1) and every evaluation episode (starts >= 1200) sees
    # geometry the agent never trained on. Same orbit, so "held out" means
    # unseen timesteps, not an unseen constellation. ASSUMED split.
    train_start_max: int = 999
    # Training demand seeds are drawn from [train_seed_min, 2^31); evaluation
    # uses eval_seeds, all below it, so no evaluation demand is ever trained on.
    train_seed_min: int = 10_000
    eval_every: int = 10  # episodes
    eval_selectors: tuple[SelectorName, ...] = ("Greedy", "DoTD-forward", "DoTD-backward", "+Grid")
    eval_loads: tuple[float, ...] = (0.5, 1.0, 2.0)
    eval_seeds: tuple[int, ...] = (7, 8, 9)
    eval_starts: tuple[int, ...] = (1200, 1400, 1599)
    # Q calibration (agent.calibration): states with fewer than tail_steps(gamma,
    # tol) steps left are not compared, so truncating the realised return drops
    # at most tol of the geometric weight. 0.01 at gamma 0.95 is 90 steps, so a
    # 200-step evaluation episode compares its first 111 states. ASSUMED.
    calib_tail_tol: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.train_seed_min <= max(self.eval_seeds, default=0):
            raise ValueError("training demand seeds must not overlap eval_seeds")
        if not self.selectors or not self.loads:
            raise ValueError("need at least one training selector and load")


@dataclass(frozen=True)
class AgentValidationConfig:
    """Bounds for validate_agent.py. Margins, not results. ASSUMED, all."""

    # Regression guard (gain = 0 must be backpressure exactly).
    guard_loads: tuple[float, ...] = (0.5, 1.0, 2.0)
    guard_seeds: tuple[int, ...] = (7, 8)
    guard_starts: tuple[int, ...] = (0, 800, 1599)
    # Untrained agent vs backpressure, episode-mean queue reward (<= 0):
    # agent >= BP - untrained_tol * |BP|.
    untrained_tol: float = 1e-9
    guard_starts_big: tuple[int, ...] = (0, 1599)  # 6x12 section: fewer episodes
    # Learning-loop check: under a reward paying move_pay times the FRACTION of
    # live slots that differ from backpressure (the form Q's mean deviation
    # term represents exactly; per slot until 2026-11-20, at 0.1), the greedy agent must flip at least
    # move_min_frac of live slots. gamma and lr are the check's own, so the
    # values stay small and it converges in a few hundred updates.
    move_min_frac: float = 0.9
    move_pay: float = 10.0
    move_gamma: float = 0.5
    move_lr: float = 1e-3
    move_episodes: int = 10
    # Q calibration (agent.calibration). A trained agent's Q for the actions it
    # takes must match the realised discounted return within
    #     |mean(Q - G)| <= calib_rel_tol * mean|G| + calib_abs_tol
    # (return units, after reward scaling). The absolute floor is for an agent
    # that stays at backpressure, where G is ~0 under the shadow reward; 1.0 is
    # 0.05 per step at gamma 0.95. Measured 2026-11-20 on the summed-A code:
    # +54 bias against |G| 54 after 8 shadow episodes at 3x8; the paid-to-
    # deviate agent over by 29% and 66%. ASSUMED bounds.
    calib_rel_tol: float = 0.25
    calib_abs_tol: float = 1.0
    calib_episodes: int = 8  # full-length shadow-reward episodes, default AgentConfig
    calib_loads: tuple[float, ...] = (1.0, 2.0)
    calib_starts: tuple[int, ...] = (1200, 1400)
    calib_seed: int = 7
    # CPU threads for the suite (results identical for a fixed count).
    torch_threads: int = 4
    # Replay uniformity: chi-square p-value floor, bins, draws.
    chi2_p_min: float = 1e-3
    chi2_bins: int = 10
    chi2_draws: int = 200_000
    # Short training runs used by the loop checks (3x8 debug scale).
    short_episodes: int = 4
    short_episode_steps: int = 100
    tiny: AgentConfig = field(default_factory=lambda: AgentConfig(
        hidden=(64, 64, 64), learning_starts=100, target_sync=50, eps_decay_steps=300, buffer_size=2_000,
        batch_size=16))
