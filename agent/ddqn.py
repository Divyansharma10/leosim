"""Residual Double DQN with a backpressure baseline.

    Q(s, a) = V(s) + (1 / L) sum over live slots l of dev_l(a) * A(s, l),   dev_l = 1[a_l != a_BP(s, l)]
    greedy: deviate on l iff gain * A(s, l) > kappa                 (agent.network)

An additive (VDN-style) decomposition whose zero is backpressure: A is the
correction, and with A <= kappa everywhere the agent IS backpressure. The
joint argmax decomposes per slot, so the 2^(N D) action is never enumerated.

Two per-slot-head designs were tried first and failed, measured 2026-11-18
(3x8 and 6x12, Greedy, kappa 0.01):
1. kappa * 1[a == a_BP] + Q_theta inside the regressed Q: TD absorbs the
   fixed prior. After 600 updates 45% of live slots overrode backpressure,
   reward -20 per step against its 0.
2. The same prior outside the regression, per-slot heads all regressed on
   the step's one target (branching DQN, mean operator): 41-50% deviation,
   learned on/off gaps 0.11 +- 0.01 on every slot. Each head is fitted on
   different states for its two actions, so state value leaks into the gap
   as a near-constant offset that swamps any per-link advantage.
Here V(s) is fitted on every transition and the A's only explain what the
deviations add on top of it, in one scalar regression per transition.

Target (Double DQN):

    dev*  = deviate(A_online(s'))                     online SELECTS
    y     = r + gamma (1 - done) * [V_target(s') + mean dev* A_target(s')]  target EVALUATES

The mean (over the L' live slots of s') replaced a sum on 2026-11-20; see
agent.network for why.

loss = Huber(Q_online(s, a_taken), y), mean over the batch. The environment
has no terminal state, only time-limit truncation, so ``done`` is always 0
there and truncated episodes bootstrap. ASSUMED: decomposition, Huber, hard
target copies.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from rlenv.policies import BackpressurePolicy

from .config import AgentConfig
from .features import hop_distance, link_features
from .network import ResidualQ, joint_q
from .replay import ReplayBuffer


def epsilon(cfg: AgentConfig, step: int) -> float:
    """Linear eps_start -> eps_end over eps_decay_steps env steps, then flat."""
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    return cfg.eps_start + (step / cfg.eps_decay_steps) * (cfg.eps_end - cfg.eps_start)


def td_target(dev_next: torch.Tensor, v_next_target: torch.Tensor, adv_next_target: torch.Tensor,
              mask_next: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    """(B,) Double-DQN targets. ``dev_next`` (B, N, D) bool comes from the
    ONLINE network's advantages; V and A of s' from the TARGET network;
    ``mask_next`` the live slots of s'."""
    return reward + gamma * (1.0 - done) * joint_q(v_next_target, adv_next_target, dev_next, mask_next)


def td_loss(q: torch.Tensor, y: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber between (B,) Q(s, a_taken) and (B,) targets, batch mean."""
    return nn.functional.huber_loss(q, y, delta=delta)


class DDQNAgent:
    def __init__(self, cfg: AgentConfig, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        torch.manual_seed(cfg.seed)
        self.online = ResidualQ(cfg.hidden, cfg.init_std, cfg.residual_margin).to(self.device)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.rng = np.random.default_rng(cfg.seed)
        self.bp = BackpressurePolicy()
        self.steps = 0  # env steps taken through act(explore=True)
        self.updates = 0
        self.last_sync = 0  # self.updates at the last target copy
        self.losses: list[float] = []

    # --- observation ---------------------------------------------------------------
    def augment(self, obs: dict) -> dict:
        """The env observation plus hops and backpressure's bit (float64 path)."""
        out = dict(obs)
        out["hops"] = hop_distance(obs, self.cfg.max_hops)
        out["bp_action"] = self.bp.act(obs)
        return out

    def _tensors(self, batch: dict, prefix: str = "") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(features, bp_action, mask) on device from batched arrays."""
        g = lambda k, dt: torch.as_tensor(np.asarray(batch[prefix + k]), device=self.device).to(dt)
        mask = g("action_mask", torch.bool)
        feats = link_features(g("queue", torch.float32), g("isl_capacity", torch.float32),
                              g("downlink_capacity", torch.float32), g("neighbours", torch.int64),
                              mask, g("hops", torch.float32))
        return feats, g("bp_action", torch.int64), mask

    @torch.no_grad()
    def advantage(self, aug: dict) -> tuple[float, np.ndarray]:
        """V and (N, D) A from the online network for one augmented observation."""
        batch = {k: np.asarray(v)[None] for k, v in aug.items()}
        feats, _, mask = self._tensors(batch)
        v, adv = self.online.value_advantage(feats, mask)
        return float(v[0]), adv[0].cpu().numpy()

    # --- acting ------------------------------------------------------------------------
    def greedy(self, aug: dict) -> np.ndarray:
        """Backpressure's bit, flipped where the online network deviates; 0 on padding."""
        return self.greedy_q(aug)[0]

    def greedy_q(self, aug: dict) -> tuple[np.ndarray, float]:
        """The greedy action and the online network's Q for it, from one
        forward pass (agent.calibration compares that Q with the return)."""
        batch = {k: np.asarray(v)[None] for k, v in aug.items()}
        feats, _, mask = self._tensors(batch)
        with torch.no_grad():
            v, adv = self.online.value_advantage(feats, mask)
            dev_t = self.online.deviate(adv, mask)
            q = float(joint_q(v, adv, dev_t, mask)[0])
        dev = dev_t[0].cpu().numpy()
        m = aug["action_mask"].astype(bool)
        return ((aug["bp_action"].astype(bool) ^ dev) & m).astype(np.int8), q

    def act(self, aug: dict, explore: bool) -> np.ndarray:
        a = self.greedy(aug)
        if explore:
            eps = epsilon(self.cfg, self.steps)
            flip = (self.rng.random(a.shape) < eps) & aug["action_mask"].astype(bool)
            a = (a ^ flip).astype(np.int8)
            self.steps += 1
        return a

    # --- learning ----------------------------------------------------------------------
    def observe(self, aug: dict, action: np.ndarray, reward: float, next_aug: dict, done: bool) -> float | None:
        """Store a transition; update if due. Returns the loss when it updated."""
        self.buffer.add(aug, action, reward, next_aug, done)
        if self.buffer.size < self.cfg.learning_starts or self.steps % self.cfg.train_every:
            return None
        return self.update()

    @torch.no_grad()
    def targets(self, b: dict) -> torch.Tensor:
        """(B,) Double-DQN targets for a sampled batch: the ONLINE network picks
        which slots deviate at s', the TARGET network values that action."""
        nfeats, _, nmask = self._tensors(b, "next_")
        reward = torch.as_tensor(b["reward"], device=self.device, dtype=torch.float32)
        done = torch.as_tensor(b["done"], device=self.device, dtype=torch.float32)
        _, adv_on = self.online.value_advantage(nfeats, nmask)
        dev_next = self.online.deviate(adv_on, nmask)
        v_t, adv_t = self.target.value_advantage(nfeats, nmask)
        return td_target(dev_next, v_t, adv_t, nmask, reward, done, self.cfg.gamma)

    def update(self) -> float:
        cfg = self.cfg
        b = self.buffer.sample(cfg.batch_size, self.rng)
        feats, bp, mask = self._tensors(b)
        action = torch.as_tensor(b["action"], device=self.device, dtype=torch.int64)
        y = self.targets(b)
        v, adv = self.online.value_advantage(feats, mask)
        dev = (action != bp) & mask
        loss = td_loss(joint_q(v, adv, dev, mask), y, cfg.huber_delta)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optim.step()
        self.updates += 1
        if self.updates % cfg.target_sync == 0:
            self.sync()
        val = float(loss.detach())
        self.losses.append(val)
        return val

    def sync(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.last_sync = self.updates

    # --- state -------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "buffer": self.buffer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "steps": self.steps, "updates": self.updates, "last_sync": self.last_sync,
            "losses": list(self.losses),
        }

    def load_state_dict(self, s: dict) -> None:
        self.online.load_state_dict(s["online"])
        self.target.load_state_dict(s["target"])
        self.optim.load_state_dict(s["optim"])
        self.buffer.load_state_dict(s["buffer"])
        self.rng.bit_generator.state = s["rng"]
        self.steps, self.updates, self.last_sync = int(s["steps"]), int(s["updates"]), int(s["last_sync"])
        self.losses = list(s["losses"])


class AgentPolicy:
    """``act(obs)`` interface for ``rlenv.rollout``: greedy, no learning."""

    def __init__(self, agent: DDQNAgent, name: str = "Residual-DDQN") -> None:
        self.agent, self.name = agent, name
        self.deviations = self.live = 0
        self.q: list[float] = []  # Q(s_t, a_t) of every action taken, for agent.calibration

    def act(self, obs: dict) -> np.ndarray:
        aug = self.agent.augment(obs)
        a, q = self.agent.greedy_q(aug)
        self.q.append(q)
        m = aug["action_mask"].astype(bool)
        self.deviations += int((a.astype(bool) != aug["bp_action"].astype(bool))[m].sum())
        self.live += int(m.sum())
        return a

    @property
    def deviation_frac(self) -> float:
        return self.deviations / self.live if self.live else 0.0
