"""Training and evaluation loop, with checkpoint/resume.

Each training episode draws (selector, load) from TrainConfig, a start in
[0, train_start_max] and a demand seed >= train_seed_min, all from the
trainer's own seeded RNG, and passes the start through reset options, so the
env's own RNG is never used. Every env is built with the agent's fixed
reward scale (agent.config.REWARD_SCALE).

Checkpoints are taken at episode boundaries and hold everything a resumed
run needs to continue exactly: both networks, the optimiser, the replay
buffer, the agent's and the trainer's RNG states, the counters and the
learning curve. ``torch.load(map_location=...)`` moves them to any device,
so a CPU run can continue on GPU. Exact continuation is checked on one
device; across devices expect agreement to float tolerance, not bytes.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from rlenv import BackpressurePolicy, EnvConfig, LeoForwardingEnv, ShortestPathPolicy, build_scenario, rollout
from rlenv.scenario import ROOT, Scenario

from .calibration import calibration
from .config import AgentConfig, TrainConfig
from .ddqn import AgentPolicy, DDQNAgent, epsilon

StepHook = Callable[[LeoForwardingEnv, dict, np.ndarray, dict], None]


def env_config(base: EnvConfig, acfg: AgentConfig, selector: str, load: float) -> EnvConfig:
    return dataclasses.replace(base, selector=selector, load=load, reward_kind=acfg.reward_kind,
                               reward_scale=acfg.reward_scale, start_mode="fixed")


class ScenarioCache:
    """One stage-1 dataset, one Scenario per (selector, load), built lazily."""

    def __init__(self, base: EnvConfig, root: Path = ROOT, verbose: bool = False) -> None:
        self.base, self.root, self.verbose = base, root, verbose
        self.dataset = None
        self.scenarios: dict[tuple[str, float], Scenario] = {}

    def get(self, cfg: EnvConfig) -> Scenario:
        key = (cfg.selector, float(cfg.load))
        if key not in self.scenarios:
            sc = build_scenario(cfg, self.root, dataset=self.dataset, verbose=self.verbose and self.dataset is None)
            self.dataset = sc.dataset
            self.scenarios[key] = sc
        return self.scenarios[key]


class Trainer:
    def __init__(self, acfg: AgentConfig, tcfg: TrainConfig, base: EnvConfig, cache: ScenarioCache | None = None,
                 device: str | None = None, step_hook: StepHook | None = None, root: Path = ROOT,
                 reward_factory: Callable[[EnvConfig], object] | None = None) -> None:
        """``reward_factory(cfg)`` replaces the env's reward for TRAINING envs
        only (validation uses it to pay for deviating); evaluation always uses
        the configured reward."""
        self.acfg, self.tcfg, self.base = acfg, tcfg, base
        self.cache = cache if cache is not None else ScenarioCache(base, root)
        self.agent = DDQNAgent(acfg, device)
        self.rng = np.random.default_rng(tcfg.seed)
        self.step_hook = step_hook
        self.reward_factory = reward_factory
        self.train_envs: dict[tuple[str, float], LeoForwardingEnv] = {}
        self.envs: dict[tuple[str, float], LeoForwardingEnv] = {}
        self.episode = 0
        self.curve: list[dict] = []  # one row per training episode
        self.evals: list[dict] = []  # one row per evaluation episode
        self.baselines: dict[tuple, dict] | None = None

    def env(self, selector: str, load: float) -> LeoForwardingEnv:
        key = (selector, float(load))
        if key not in self.envs:
            cfg = env_config(self.base, self.acfg, selector, load)
            self.envs[key] = LeoForwardingEnv(cfg, self.cache.get(cfg))
        return self.envs[key]

    def train_env(self, selector: str, load: float) -> LeoForwardingEnv:
        if self.reward_factory is None:
            return self.env(selector, load)
        key = (selector, float(load))
        if key not in self.train_envs:
            cfg = env_config(self.base, self.acfg, selector, load)
            self.train_envs[key] = LeoForwardingEnv(cfg, self.cache.get(cfg), reward_fn=self.reward_factory(cfg))
        return self.train_envs[key]

    # --- training --------------------------------------------------------------------
    def run_episode(self) -> dict:
        t = self.tcfg
        selector = t.selectors[int(self.rng.integers(len(t.selectors)))]
        load = t.loads[int(self.rng.integers(len(t.loads)))]
        env = self.train_env(selector, load)
        last = min(t.train_start_max, env.scenario.n_steps - 1 - env.cfg.episode_steps)
        start = int(self.rng.integers(0, last + 1))
        seed = int(self.rng.integers(t.train_seed_min, 2**31 - 1))
        obs, _ = env.reset(seed=seed, options={"start_step": start})
        aug = self.agent.augment(obs)
        rewards, losses, flips, live, done = [], [], 0, 0, False
        eps0 = epsilon(self.acfg, self.agent.steps)
        while not done:
            a = self.agent.act(aug, explore=True)
            m = aug["action_mask"].astype(bool)
            flips += int((a.astype(bool) != aug["bp_action"].astype(bool))[m].sum())
            live += int(m.sum())
            obs, r, terminated, truncated, info = env.step(a)
            if self.step_hook is not None:
                self.step_hook(env, aug, a, info)
            next_aug = self.agent.augment(obs)
            loss = self.agent.observe(aug, a, r, next_aug, terminated)
            if loss is not None:
                losses.append(loss)
            rewards.append(r)
            aug, done = next_aug, terminated or truncated
        self.episode += 1
        row = {"episode": self.episode, "selector": selector, "load": load, "start": start, "seed": seed,
               "reward_mean": float(np.mean(rewards)), "reward_min": float(np.min(rewards)),
               "all_finite": bool(np.isfinite(rewards).all() and np.isfinite(losses).all()),
               "loss_mean": float(np.mean(losses)) if losses else np.nan, "updates": self.agent.updates,
               "eps_start": eps0, "eps_end": epsilon(self.acfg, self.agent.steps),
               "deviation_frac": flips / live if live else 0.0}
        self.curve.append(row)
        return row

    def train(self, episodes: int, eval_every: int | None = None, on_episode: Callable[[dict], None] | None = None,
              checkpoint: Path | None = None) -> None:
        eval_every = self.tcfg.eval_every if eval_every is None else eval_every
        for _ in range(episodes):
            row = self.run_episode()
            if on_episode is not None:
                on_episode(row)
            if eval_every and self.episode % eval_every == 0:
                self.evaluate()
                if checkpoint is not None:
                    self.save(checkpoint)

    # --- evaluation ----------------------------------------------------------------------
    def eval_set(self) -> list[tuple[str, float, int, int]]:
        t = self.tcfg
        return [(s, l, seed, start) for s in t.eval_selectors for l in t.eval_loads
                for seed in t.eval_seeds for start in t.eval_starts]

    def compute_baselines(self) -> dict[tuple, dict]:
        """Backpressure and shortest-path on the evaluation set, once. Run on
        the same envs (shadow reward), so their absolute queue metrics come
        from info, which is reward-independent."""
        out = {}
        for s, l, seed, start in self.eval_set():
            env = self.env(s, l)
            for pol in (BackpressurePolicy(), ShortestPathPolicy()):
                r = rollout(env, pol, seed, {"start_step": start})
                out[(s, l, seed, start, pol.name)] = _summary(r, env.cfg)
        self.baselines = out
        return out

    def evaluate(self) -> list[dict]:
        if self.baselines is None:
            self.compute_baselines()
        rows = []
        for s, l, seed, start in self.eval_set():
            env = self.env(s, l)
            pol = AgentPolicy(self.agent)
            r = rollout(env, pol, seed, {"start_step": start})
            row = {"episode": self.episode, "selector": s, "load": l, "seed": seed, "start": start,
                   **_summary(r, env.cfg), "deviation_frac": pol.deviation_frac}
            cal = calibration(pol.q, r["rewards"], self.acfg.gamma, self.tcfg.calib_tail_tol)
            row.update({f"calib_{k}": v for k, v in cal.items()})
            for name in ("Backpressure", "Shortest-path"):
                b = self.baselines[(s, l, seed, start, name)]
                tag = "bp" if name == "Backpressure" else "sp"
                for k in ("queue_reward", "queue_mean", "queue_max", "delivered_frac", "isl_per_arrived"):
                    row[f"{tag}_{k}"] = b[k]
            rows.append(row)
        self.evals.extend(rows)
        return rows

    # --- checkpoint ----------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"agent": self.agent.state_dict(), "rng": self.rng.bit_generator.state, "episode": self.episode,
                "curve": list(self.curve), "evals": list(self.evals),
                "acfg": dataclasses.asdict(self.acfg), "tcfg": dataclasses.asdict(self.tcfg),
                "base": dataclasses.asdict(self.base), "torch_rng": torch.get_rng_state(),
                "py_rng": random.getstate()}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        s = torch.load(path, map_location=self.agent.device, weights_only=False)
        for name, cfg in (("acfg", self.acfg), ("tcfg", self.tcfg), ("base", self.base)):
            mine, theirs = dataclasses.asdict(cfg), s[name]
            # device may legitimately differ (CPU -> GPU); nothing else may.
            mine.pop("device", None), theirs.pop("device", None)
            if _norm(mine) != _norm(theirs):
                raise ValueError(f"checkpoint {name} differs from this run's config")
        self.agent.load_state_dict(s["agent"])
        self.rng.bit_generator.state = s["rng"]
        self.episode = int(s["episode"])
        self.curve, self.evals = list(s["curve"]), list(s["evals"])
        torch.set_rng_state(s["torch_rng"].cpu())  # map_location may have moved it to the GPU
        random.setstate(s["py_rng"])


def _norm(d):
    """Tuples and lists compare equal (asdict keeps tuples; a reload may not)."""
    if isinstance(d, dict):
        return {k: _norm(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_norm(v) for v in d]
    return d


def _summary(r: dict, cfg: EnvConfig) -> dict:
    """Episode metrics. ``queue_reward`` is the absolute queue reward (unscaled)
    from the env's own per-step queue_mean/queue_max, whatever reward the env
    returned, so agent and baselines are compared on one absolute scale."""
    rows = r["rows"]
    arrived = r["arrived_bits"]
    isl = sum(x["isl_bits"] for x in rows)
    qr = -float(np.mean([cfg.w_mean * x["queue_mean"] + cfg.w_max * x["queue_max"] for x in rows]))
    return {"reward_mean": r["reward_mean"], "queue_reward": qr, "queue_mean": r["queue_mean"],
            "queue_max": r["queue_max"], "delivered_frac": r["delivered_frac"],
            "isl_per_arrived": isl / arrived if arrived else np.nan,
            "final_backlog_bits": r["final_backlog_bits"]}
