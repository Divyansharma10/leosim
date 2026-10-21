"""Gymnasium environment: directed forwarding over a fixed selector's topology.

One episode = one pass over ``episode_steps`` timesteps from ``start_step``,
with the selector and offered load fixed by ``EnvConfig``. reset() returns
the observation at start_step with empty queues; step(a) applies ``a`` at the
current t, advances to t + 1 and returns that observation; the
``episode_steps``-th step returns truncated=True. There is no terminal state.

Action: MultiBinary (N, D). Slot (i, k) = 1 means satellite i sends to its
k-th selected neighbour this step (``obs["neighbours"][i, k]``). Padding slots
(mask 0) are ignored whatever they hold, so every binary (N, D) array is a
legal action; anything else is rejected rather than coerced. The decoded
directed set is checked against the selector's link set at every step,
independently of the decoder, as stage 2 checks its matcher.

Observation (Dict, every float in [0, 1] by construction):
  queue              (N,)    Q / (Q + q_ref)                [0, 1) for any finite Q; monotone, so no clipping
  isl_capacity       (N, D)  slot capacity / isl_ref        isl_ref = max over VISIBLE pairs, whole window
  downlink_capacity  (N,)    best gateway / downlink_ref    0 when no gateway is in view
  neighbours         (N, D)  satellite index, -1 padding    structural, not normalised
  action_mask        (N, D)  1 where a selected link exists structural
Floats are float64, not the usual float32: in float32, q / (q + q_ref) rounds
nearly equal queues to a tie and backpressure then skips a send it makes on the
raw queues (measured: a 1 Mb end-of-episode difference at load 0.5). An agent
may cast down; the observation itself must not change the baseline.
The two capacity references are fixed for the dataset, not running maxima,
so a raw value maps to the same observation in every episode and at every t.
Taking them over the whole window uses future geometry for SCALING only,
never for a decision. ASSUMED.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .queues import QueueModel, StepResult, forward_step
from .reward import QueueReward, RewardContext, ShadowBackpressureReward
from .scenario import ROOT, Scenario, build_scenario


class LeoForwardingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, scenario: Scenario | None = None,
                 reward_fn=None, root: Path = ROOT) -> None:
        """``scenario`` may be passed to share one build across envs; it must
        have been built from a config with the same stage-1/2/load fields."""
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.scenario = scenario if scenario is not None else build_scenario(self.cfg, root)
        sc = self.scenario
        if self.cfg.start_step + self.cfg.episode_steps > sc.n_steps - 1:
            raise ValueError("episode runs past the end of the scenario")
        if reward_fn is None:
            kind = {"queue": QueueReward, "shadow": ShadowBackpressureReward}[self.cfg.reward_kind]
            reward_fn = kind(self.cfg.w_mean, self.cfg.w_max)
        self.reward_fn = reward_fn
        n, d = sc.n_sats, sc.n_slots
        self.n_sats, self.n_slots = n, d
        self.observation_space = spaces.Dict({
            "queue": spaces.Box(0.0, 1.0, (n,), np.float64),
            "isl_capacity": spaces.Box(0.0, 1.0, (n, d), np.float64),
            "downlink_capacity": spaces.Box(0.0, 1.0, (n,), np.float64),
            "neighbours": spaces.Box(-1, n - 1, (n, d), np.int64),
            "action_mask": spaces.Box(0, 1, (n, d), np.int8),
        })
        self.action_space = spaces.MultiBinary([n, d])
        self.queues = QueueModel(n, sc.dataset.config.step_s, sc.buffer_bits)
        self.t = self.k = None
        self._offered = None

    # --- episode -----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        demand_seed = seed if seed is not None else int(self.np_random.integers(2**31 - 1))
        last = self.scenario.n_steps - 1 - self.cfg.episode_steps
        start = self.cfg.start_step
        if self.cfg.start_mode == "uniform":
            start = int(self.np_random.integers(0, last + 1))
        if options and "start_step" in options:
            start = int(options["start_step"])
            if not 0 <= start <= last:
                raise ValueError(f"start_step {start} leaves no room for the episode")
        self._offered = self.scenario.offered(demand_seed)
        self.queues.reset()
        self.t, self.k = start, 0
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(self)
        return self._observation(), {"t": self.t, "demand_seed": demand_seed}

    def step(self, action):
        if self.t is None:
            raise RuntimeError("step() before reset()")
        if self.k >= self.cfg.episode_steps:
            raise RuntimeError("step() after the episode was truncated; call reset()")
        sc, t = self.scenario, self.t
        applied = self._validate_action(action) & self._mask(t)
        forward = self._decode(applied, t)
        self._assert_legal(forward, t)

        offered_t = self._offered[t]
        arrivals = sc.arrivals(offered_t, t)
        isl, dl = sc.isl_capacity(t), sc.downlink[t]
        prev = self.queues.snapshot()
        result = self.queues.step(forward, isl, dl, arrivals)

        def counterfactual(other_action) -> StepResult:
            f = self.forward_for(other_action, t)
            return forward_step(prev, f, isl, dl, arrivals, self.queues.dt, self.queues.buffer_bits)

        ctx = RewardContext(prev, result, applied, t, sc.queue_ref_bits, counterfactual, isl, dl)
        reward = self.cfg.reward_scale * float(self.reward_fn(ctx))

        self.t, self.k = t + 1, self.k + 1
        truncated = self.k >= self.cfg.episode_steps
        no_access = float(offered_t[sc.city_sat[t] < 0].sum())
        info = {
            "t": self.t,
            "applied_t": t,
            "offered_bits": float(offered_t.sum()),
            "no_access_bits": no_access,
            "arrived_bits": float(result.arrivals.sum()),
            "delivered_bits": float(result.delivered.sum()),
            "dropped_bits": float(result.dropped.sum()),
            "isl_bits": float(result.sent.sum()),
            "backlog_bits": float(result.queue.sum()),
            "queue_mean": float(result.queue.mean() / sc.queue_ref_bits),
            "queue_max": float(result.queue.max() / sc.queue_ref_bits),
            "links_active": int(forward.sum()),
        }
        return self._observation(), reward, False, truncated, info

    # --- action ------------------------------------------------------------------
    def forward_for(self, action, t: int) -> np.ndarray:
        """(N, N) directed activation an action produces at t: validated,
        masked, decoded and checked, exactly as step() does."""
        f = self._decode(self._validate_action(action) & self._mask(t), t)
        self._assert_legal(f, t)
        return f

    def _mask(self, t: int) -> np.ndarray:
        return self.scenario.neighbours[t] >= 0

    def _validate_action(self, action) -> np.ndarray:
        a = np.asarray(action)
        if a.shape != (self.n_sats, self.n_slots):
            raise ValueError(f"action shape {a.shape}, expected {(self.n_sats, self.n_slots)}")
        if not np.isin(a, (0, 1)).all():
            raise ValueError("action must be binary")
        return a.astype(bool)

    def _decode(self, applied: np.ndarray, t: int) -> np.ndarray:
        """(N, N) directed activation from masked slots."""
        nbr = self.scenario.neighbours[t]
        forward = np.zeros((self.n_sats, self.n_sats), dtype=bool)
        ii, kk = np.nonzero(applied)
        forward[ii, nbr[ii, kk]] = True
        return forward

    def _assert_legal(self, forward: np.ndarray, t: int) -> None:
        """Independent of the decoder: the directed set must sit inside the
        selector's link set, which must itself be a legal stage-2 set."""
        adj = self.scenario.linksets[t].adjacency
        if not np.array_equal(adj, adj.T) or adj.diagonal().any():
            raise AssertionError(f"t={t}: selected link set not symmetric with empty diagonal")
        if adj.sum(axis=1).max(initial=0) > self.n_slots:
            raise AssertionError(f"t={t}: selected degree above {self.n_slots}")
        if (forward & ~adj).any():
            raise AssertionError(f"t={t}: forwarding on a link the selector did not choose")
        if forward.sum(axis=1).max(initial=0) > self.n_slots:
            raise AssertionError(f"t={t}: out-degree above {self.n_slots}")

    # --- observation -------------------------------------------------------------
    def _squash(self, q: np.ndarray) -> np.ndarray:
        r = self.scenario.queue_ref_bits
        return (q / (q + r)).astype(np.float64)

    def _observation(self) -> dict:
        return self.observe(self.queues.queue, self.t)

    def observe(self, queue: np.ndarray, t: int) -> dict:
        """The observation a policy would see with ``queue`` (bits) at t. The
        shadow reward uses it to run backpressure on its own queues."""
        sc = self.scenario
        nbr = sc.neighbours[t]
        return {
            "queue": self._squash(queue),
            "isl_capacity": (sc.isl_slot_capacity[t] / sc.isl_ref_bps).astype(np.float64),
            "downlink_capacity": (sc.downlink[t] / sc.downlink_ref_bps).astype(np.float64),
            "neighbours": nbr.copy(),
            "action_mask": (nbr >= 0).astype(np.int8),
        }
