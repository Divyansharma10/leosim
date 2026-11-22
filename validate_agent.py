"""Phase 3 step 2 validation: the residual Double DQN on backpressure.

    python validate_agent.py            # ~2-3 min: unit pins, 3x8 loop checks, 6x12 guard

Same rules as the other suites. Every check drives production code; pinned
values are hard-coded literals; each check that could pass vacuously has a
partner showing it cannot.

The step-2 hazards are an agent that is not what it claims to be underneath
(the residual off is not backpressure), a learning loop that is silently dead
(a target network that never moves, a biased replay, exploration running the
wrong way), and a loop that cannot move at all, which would make "does not
beat backpressure" uninterpretable. The last is checked directly: under a
reward that pays for deviating, the agent must learn to deviate.

3x8 is the debug scale (its selectors are identical); the regression guard and
the untrained-agent check also run at 6x12 under krishnan8, the primary
training configuration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chisquare

from agent import (
    AgentConfig,
    AgentPolicy,
    AgentValidationConfig,
    DDQNAgent,
    ReplayBuffer,
    ResidualQ,
    ScenarioCache,
    TrainConfig,
    Trainer,
    env_config,
    epsilon,
    joint_q,
    masked_mean,
    td_target,
)
from agent.calibration import calibration, discounted_returns, tail_steps
from agent.features import hop_distance, link_features
from rlenv import BackpressurePolicy, EnvConfig, LeoForwardingEnv, rollout
from validate import Checks, section

ROOT = Path(__file__).parent
VCFG = AgentValidationConfig()

# --- pinned literals ----------------------------------------------------------
# Four ISLs per satellite (the paper's degree cap; stage 2's u_max).
MAX_DEGREE = 4
# Default schedule 0.1 -> 0.01 over 20000 steps, linear, then flat.
PIN_EPS = {0: 0.1, 10_000: 0.055, 20_000: 0.01, 40_000: 0.01}
# masked_mean([[1, 3], [5, 100]], live [[1, 1], [1, 0]]) = (1 + 3 + 5) / 3.
PIN_MASKED_MEAN = 3.0
# joint_q: V = 1, A = [[0.5, 0.25], [2, 0]], deviating [[1, 0], [1, 1]], live
# [[1, 1], [1, 0]]: the padding deviation is ignored, 3 live slots, so
# 1 + (0.5 + 2) / 3. (Summed, before 2026-11-20: 3.5.)
PIN_JOINT_Q = 11.0 / 6.0
# td_target, gamma 0.5. Sample 0: deviating slots (from ONLINE A > kappa 0.1:
# online A [[0.2, 0], [0.3, -1]] -> [[1, 0], [1, 0]]) valued by the TARGET's
# A [[0.5, 9], [0.25, 7]] and V 2, 4 live slots: 2 + (0.5 + 0.25) / 4 = 2.1875;
# y = 1 + 0.5 * 2.1875. Sample 1: done, r = -3 -> y = -3. Had the target
# selected (its A > 0.1 everywhere) sample 0 would be 1 + 0.5 * 6.1875 = 4.09375.
PIN_TD = [2.09375, -3.0]
# Ring of capacity 5 after rewards 0..6: slots hold [5, 6, 2, 3, 4].
PIN_RING = [5.0, 6.0, 2.0, 3.0, 4.0]
# Line 0-1-2-3 plus an isolated satellite 4, gateway in view of 3 only,
# max_hops 8: hops [3, 2, 1, 0, unreachable] -> [3/8, 2/8, 1/8, 0, 1].
PIN_HOPS = [0.375, 0.25, 0.125, 0.0, 1.0]
# G = r + 0.5 G': from the end, 1, 1 + 0.5, 1 + 0.75.
PIN_RETURNS = [1.75, 1.5, 1.0]
# ceil(ln 0.01 / ln 0.95) = ceil(89.78); ceil(ln 0.01 / ln 0.5) = ceil(6.64).
PIN_TAILS = [90, 7]


def _hash_module(m: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _params_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def _set_head(q: ResidualQ, v_bias: float, a_bias: float) -> None:
    """Zero the head weights and set its biases: v = v_bias, A = a_bias on every slot."""
    with torch.no_grad():
        q.net.head.weight.zero_()
        q.net.head.bias.copy_(torch.tensor([v_bias, a_bias]))


# --- independent legality check ------------------------------------------------------
def illegal(action, obs: dict, adjacency: np.ndarray) -> list[str]:
    """Reasons an action is illegal, from the selector's adjacency, the
    observation's neighbour slots and nothing of the agent's. Empty = legal."""
    out = []
    a = np.asarray(action)
    nbr = obs["neighbours"]
    if a.shape != nbr.shape:
        return [f"shape {a.shape} != {nbr.shape}"]
    if not np.isin(a, (0, 1)).all():
        out.append("non-binary")
    if (a[nbr < 0] != 0).any():
        out.append("padding slot set")
    if not np.array_equal(adjacency, adjacency.T) or adjacency.diagonal().any():
        out.append("selected set not symmetric with empty diagonal")
    if adjacency.sum(axis=1).max(initial=0) > MAX_DEGREE:
        out.append(f"selected degree above {MAX_DEGREE}")
    n = nbr.shape[0]
    forward = np.zeros((n, n), dtype=bool)
    ii, kk = np.nonzero((a == 1) & (nbr >= 0))
    forward[ii, nbr[ii, kk]] = True
    if (forward & ~adjacency).any():
        out.append("forwarding on a link the selector did not choose")
    if forward.sum(axis=1).max(initial=0) > MAX_DEGREE:
        out.append(f"out-degree above {MAX_DEGREE}")
    return out


class LegalityHook:
    def __init__(self) -> None:
        self.checked, self.bad = 0, []

    def __call__(self, env, aug, action, info) -> None:
        t = info["applied_t"]
        r = illegal(action, aug, env.scenario.linksets[t].adjacency)
        self.checked += 1
        if r:
            self.bad.append((t, r))


class DeviationReward:
    """Test reward for the learning-loop check: ``pay`` times the fraction of
    live slots whose applied bit differs from backpressure's at the same
    pre-step queues (through the env's own observe(), as the shadow reward
    does). A fraction, not a count, so its Q has exactly the form of Q's mean
    deviation term and the calibration reference is representable."""

    def __init__(self, pay: float) -> None:
        self.pay, self.env, self.bp = float(pay), None, BackpressurePolicy()

    def reset(self, env) -> None:
        self.env = env

    def __call__(self, ctx) -> float:
        obs = self.env.observe(ctx.prev_queue, ctx.t)
        dev = (ctx.action.astype(bool) != self.bp.act(obs).astype(bool)) & obs["action_mask"].astype(bool)
        return self.pay * float(dev.sum()) / max(1, int(obs["action_mask"].sum()))


def _deviating_agent() -> DDQNAgent:
    """An agent whose network says 'deviate' on every live slot (A = +2 kappa),
    so a guard that passes against it cannot be passing because A is small."""
    ag = DDQNAgent(AgentConfig())
    _set_head(ag.online, 0.0, 2 * ag.cfg.residual_margin)
    return ag


def _trajectory(env: LeoForwardingEnv, policy, seed: int, start: int):
    obs, _ = env.reset(seed=seed, options={"start_step": start})
    acts, qs, rs, done = [], [], [], False
    while not done:
        a = np.asarray(policy.act(obs), dtype=np.int8)
        obs, r, _, done, _ = env.step(a)
        acts.append(a)
        qs.append(env.queues.queue.copy())
        rs.append(r)
    return np.stack(acts), np.stack(qs), np.asarray(rs)


# --- unit pins -----------------------------------------------------------------------
def check_units(chk: Checks) -> None:
    section("Unit pins (hand-built)")
    cfg = AgentConfig()
    got = {s: epsilon(cfg, s) for s in PIN_EPS}
    chk.record("epsilon schedule: linear 0.1 -> 0.01 over 20000 steps, then flat",
               all(abs(got[s] - PIN_EPS[s]) < 1e-12 for s in PIN_EPS), f"measured {got}, pinned {PIN_EPS}")
    grid = [epsilon(cfg, s) for s in range(0, 30_001, 500)]
    chk.record("epsilon never increases", all(b <= a for a, b in zip(grid, grid[1:])),
               f"first {grid[0]:g}, last {grid[-1]:g}")

    x = torch.tensor([[[1.0, 3.0], [5.0, 100.0]]])
    m = torch.tensor([[[True, True], [True, False]]])
    mm = float(masked_mean(x, m)[0])
    chk.record("masked mean excludes padding", abs(mm - PIN_MASKED_MEAN) < 1e-6,
               f"measured {mm}, pinned {PIN_MASKED_MEAN} (plain mean would be 27.25)")
    jq = float(joint_q(torch.tensor([1.0]), torch.tensor([[[0.5, 0.25], [2.0, 0.0]]]),
                       torch.tensor([[[True, False], [True, True]]]),
                       torch.tensor([[[True, True], [True, False]]]))[0])
    chk.record("joint Q = V + mean of A over deviating live slots (padding ignored)", abs(jq - PIN_JOINT_Q) < 1e-6,
               f"measured {jq}, pinned {PIN_JOINT_Q}")

    q = ResidualQ((8,), 1e-5, kappa=0.1)
    live = torch.ones(2, 2, 2, dtype=torch.bool)
    dev_next = q.deviate(torch.tensor([[[0.2, 0.0], [0.3, -1.0]], [[5.0, 5.0], [5.0, 5.0]]]), live)
    y = td_target(dev_next, torch.tensor([2.0, 2.0]), torch.tensor([[[0.5, 9.0], [0.25, 7.0]]] * 2), live,
                  torch.tensor([1.0, -3.0]), torch.tensor([0.0, 1.0]), 0.5)
    chk.record("Double-DQN target: online selects, target values, done stops the bootstrap",
               np.allclose(y.numpy(), PIN_TD, rtol=0, atol=1e-6), f"measured {y.tolist()}, pinned {PIN_TD}")

    # deviate(): the residual's sign and the margin.
    feats = torch.zeros(1, 2, 2, 12)
    m = torch.tensor([[[True, True], [True, False]]])
    kappa = AgentConfig().residual_margin
    q = ResidualQ((8,), 1e-5, kappa)
    _set_head(q, 0.0, 2 * kappa)
    _, adv = q.value_advantage(feats, m)
    up = q.deviate(adv, m)
    _set_head(q, 0.0, -2 * kappa)
    _, adv = q.value_advantage(feats, m)
    down = q.deviate(adv, m)
    _set_head(q, 0.0, 0.5 * kappa)
    _, adv = q.value_advantage(feats, m)
    under = q.deviate(adv, m)
    chk.record("a residual worth +2 kappa overrides backpressure on every live slot, never on padding",
               bool(up[m].all()) and not bool(up[~m].any()), f"deviating {up.int().tolist()}")
    chk.record("a residual of -2 kappa or +0.5 kappa overrides nothing",
               not bool(down.any()) and not bool(under.any()), "")
    q.gain = 0.0
    _set_head(q, 0.0, 1e6)
    _, adv = q.value_advantage(feats, m)
    chk.record("gain 0 overrides nothing, whatever the network says", not bool(q.deviate(adv, m).any()), "")
    q.gain = 1.0
    _set_head(q, 1.0, 1.0)
    _, adv = q.value_advantage(feats, m)
    adv = adv.detach()
    chk.record("A is zero on padding", float(adv[~m].abs().sum()) == 0.0 and bool((adv[m] == 1.0).all()),
               f"A {adv.tolist()}")

    # V pools only live slots: garbage in padding features must not move it.
    q = ResidualQ((16, 16), 0.1, kappa)
    g = torch.Generator().manual_seed(0)
    f1 = torch.rand(1, 2, 2, 12, generator=g)
    f2 = f1.clone()
    f2[~m] = 1e3
    v1, _ = q.value_advantage(f1, m)
    v2, _ = q.value_advantage(f2, m)
    chk.record("V ignores padding slots", torch.equal(v1, v2), f"V {float(v1[0].detach()):.6f} vs {float(v2[0].detach()):.6f}")

    # Hop feature and the neighbour gather, on a hand-built line.
    nbr = np.array([[1, -1], [0, 2], [1, 3], [2, -1], [-1, -1]])
    obs = {"neighbours": nbr, "action_mask": (nbr >= 0).astype(np.int8),
           "downlink_capacity": np.array([0, 0, 0, 0.5, 0]), "queue": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
           "isl_capacity": np.where(nbr >= 0, 0.5, 0.0)}
    hops = hop_distance(obs, AgentConfig().max_hops)
    chk.record("hop feature: BFS distance to a gateway-seeing satellite, unreachable = 1",
               np.allclose(hops, PIN_HOPS, rtol=0, atol=1e-12), f"measured {hops.tolist()}, pinned {PIN_HOPS}")
    t = lambda x, dt: torch.as_tensor(np.asarray(x)[None]).to(dt)
    fe = link_features(t(obs["queue"], torch.float32), t(obs["isl_capacity"], torch.float32),
                       t(obs["downlink_capacity"], torch.float32), t(nbr, torch.int64),
                       t(obs["action_mask"], torch.bool), t(hops, torch.float32))[0]
    # slot (2, 1): i = 2, m = 3. q_i 0.3, q_m 0.4, weight (0.3 - 0.4) * 0.5, dl_m 0.5, h_m 0.
    got = [float(fe[2, 1, k]) for k in (0, 1, 4, 6, 8)]
    pin = [0.3, 0.4, -0.05, 0.5, 0.0]
    chk.record("link features read the neighbour's queue, downlink and hops", np.allclose(got, pin, atol=1e-6),
               f"slot (2,1): {got}, pinned {pin}")

    rb = ReplayBuffer(5)
    o = {"queue": np.zeros(2), "isl_capacity": np.zeros((2, 2)), "downlink_capacity": np.zeros(2),
         "neighbours": np.zeros((2, 2), np.int64), "action_mask": np.ones((2, 2), np.int8),
         "hops": np.zeros(2), "bp_action": np.zeros((2, 2), np.int8)}
    for r in range(7):
        rb.add(o, o["bp_action"], float(r), o, False)
    chk.record("replay ring overwrites the oldest first", rb.data["reward"].tolist() == PIN_RING and rb.size == 5,
               f"rewards {rb.data['reward'].tolist()}, pinned {PIN_RING}, size {rb.size}")
    rb2 = ReplayBuffer(5)
    rb2.load_state_dict(rb.state_dict())
    chk.record("replay state round-trips", rb2.size == rb.size and rb2.pos == rb.pos and all(
        np.array_equal(rb.data[k], rb2.data[k]) for k in rb.data), "")

    cap, rng = 1000, np.random.default_rng(0)
    full = ReplayBuffer(cap)
    for r in range(cap):
        full.add(o, o["bp_action"], float(r), o, False)
    idx = full.sample_indices(VCFG.chi2_draws, rng)
    counts = np.bincount(idx * VCFG.chi2_bins // cap, minlength=VCFG.chi2_bins)
    p_full = chisquare(counts).pvalue
    half = ReplayBuffer(cap)
    for r in range(cap // 2):
        half.add(o, o["bp_action"], float(r), o, False)
    hidx = half.sample_indices(VCFG.chi2_draws, rng)
    hcounts = np.bincount(hidx * VCFG.chi2_bins // (cap // 2), minlength=VCFG.chi2_bins)
    chk.record("replay samples uniformly over a full buffer (chi-square)", p_full > VCFG.chi2_p_min,
               f"p = {p_full:.3g}, floor {VCFG.chi2_p_min:g}; bin counts {counts.tolist()}")
    chk.record("replay samples uniformly over a half-filled buffer, never past its size",
               chisquare(hcounts).pvalue > VCFG.chi2_p_min and hidx.max() < cap // 2 and hidx.max() >= cap // 2 - 5,
               f"p = {chisquare(hcounts).pvalue:.3g}, max index {hidx.max()} of {cap // 2 - 1}")
    biased = np.random.default_rng(1).integers(0, cap // 2, VCFG.chi2_draws)
    pb = chisquare(np.bincount(biased * VCFG.chi2_bins // cap, minlength=VCFG.chi2_bins)).pvalue
    chk.record("discrimination: the chi-square test rejects sampling from half the buffer", pb < VCFG.chi2_p_min,
               f"p = {pb:.3g}")


# --- 3x8 loop checks -----------------------------------------------------------------
def _short(acfg: AgentConfig, cache: ScenarioCache, base: EnvConfig, seed: int = 0, **kw) -> Trainer:
    tcfg = TrainConfig(selectors=("Greedy",), loads=(1.0, 2.0), seed=seed)
    return Trainer(acfg, tcfg, base, cache, **kw)


def check_guard(chk: Checks, cache: ScenarioCache, base: EnvConfig, label: str, seeds, starts) -> None:
    section(f"Regression guard: residual off = backpressure exactly ({label})")
    agent = _deviating_agent()
    n_ep, same_a, same_q, zero_r, dev_on, diff_q = 0, True, True, True, 0.0, False
    for load in VCFG.guard_loads:
        cfg = env_config(base, agent.cfg, "Greedy", load)
        env = LeoForwardingEnv(cfg, cache.get(cfg))
        for seed in seeds:
            for start in starts:
                a_bp, q_bp, _ = _trajectory(env, BackpressurePolicy(), seed, start)
                agent.online.gain = 0.0
                a_ag, q_ag, r_ag = _trajectory(env, AgentPolicy(agent), seed, start)
                agent.online.gain = 1.0
                pol = AgentPolicy(agent)
                _, q_on, _ = _trajectory(env, pol, seed, start)
                n_ep += 1
                same_a &= np.array_equal(a_ag, a_bp)
                same_q &= np.array_equal(q_ag, q_bp)
                zero_r &= bool((r_ag == 0.0).all())
                dev_on = max(dev_on, pol.deviation_frac)
                diff_q |= not np.array_equal(q_on, q_bp)
    chk.record(f"gain 0: every action equals backpressure's ({n_ep} episodes, loads {VCFG.guard_loads})", same_a,
               "network says deviate (A = +2 kappa) on every live slot")
    chk.record("gain 0: every queue vector equals backpressure's, bit for bit", same_q, "")
    chk.record("gain 0: the shadow reward is exactly 0 at every step", zero_r, "")
    chk.record("discrimination: the same network with gain 1 deviates and changes the queues",
               dev_on > 0 and diff_q, f"max deviation fraction {dev_on:.3f}")


def check_untrained(chk: Checks, cache: ScenarioCache, base: EnvConfig, label: str, seeds, starts) -> None:
    section(f"Untrained agent vs backpressure ({label})")
    agent = DDQNAgent(AgentConfig())
    big = _deviating_agent()
    qcfg = AgentConfig(reward_kind="queue")
    worst, devs, big_dev, big_worst, ratio, rows = np.inf, 0.0, 0.0, np.inf, 0.0, []
    for load in VCFG.guard_loads:
        cfg = env_config(base, qcfg, "Greedy", load)
        env = LeoForwardingEnv(cfg, cache.get(cfg))
        for seed in seeds:
            for start in starts:
                bp = rollout(env, BackpressurePolicy(), seed, {"start_step": start})
                pol = AgentPolicy(agent)
                ag = rollout(env, pol, seed, {"start_step": start})
                bpol = AgentPolicy(big)
                bg = rollout(env, bpol, seed, {"start_step": start})
                big_worst = min(big_worst, (bg["reward_mean"] - bp["reward_mean"]) / abs(bp["reward_mean"]))
                margin = (ag["reward_mean"] - bp["reward_mean"]) / abs(bp["reward_mean"])
                worst = min(worst, margin)
                devs, big_dev = max(devs, pol.deviation_frac), max(big_dev, bpol.deviation_frac)
                obs, _ = env.reset(seed=seed, options={"start_step": start})
                _, adv = agent.advantage(agent.augment(obs))
                ratio = max(ratio, float(np.abs(adv).max()) / agent.cfg.residual_margin)
                rows.append(f"load {load:<4g} seed {seed} start {start:<5d} agent {ag['reward_mean']:.6f} "
                            f"BP {bp['reward_mean']:.6f}")
    chk.record("untrained agent scores no worse than backpressure on every episode",
               worst >= -VCFG.untrained_tol, f"worst relative margin {worst:.3g} (tolerance {VCFG.untrained_tol:g})\n"
               + "\n".join(rows[:6]) + ("\n..." if len(rows) > 6 else ""))
    chk.record("untrained residual is near zero: max |A| well below kappa, no slot deviates",
               ratio < 0.1 and devs == 0.0, f"max |A| / kappa = {ratio:.3g}, deviation fraction {devs:.3g}")
    chk.record("discrimination: an agent that deviates everywhere fails the same comparison",
               big_dev > 0.99 and big_worst < -VCFG.untrained_tol,
               f"deviation fraction {big_dev:.3f}, worst relative margin {big_worst:.3g}")


def check_loop(chk: Checks, cache: ScenarioCache, base: EnvConfig) -> None:
    section("Training loop (3x8 debug scale, short runs)")
    short = dataclasses.replace(base, episode_steps=VCFG.short_episode_steps)
    tiny = VCFG.tiny
    hook = LegalityHook()
    tr = _short(tiny, cache, short, step_hook=hook)
    init_target = _hash_module(tr.agent.target)
    tr.train(VCFG.short_episodes, eval_every=0)
    ag = tr.agent
    steps = VCFG.short_episodes * VCFG.short_episode_steps

    chk.record(f"every action legal during training, exploration included ({hook.checked} steps)",
               hook.checked == steps and not hook.bad, f"illegal: {hook.bad[:3]}")
    o, _ = tr.env("Greedy", 1.0).reset(seed=7)
    adj = tr.env("Greedy", 1.0).scenario.linksets[0].adjacency
    bad_pad = np.ones_like(o["neighbours"], dtype=np.int8)
    bad_link = np.zeros_like(bad_pad)
    nbr = o["neighbours"].copy()
    i, k = np.argwhere(nbr >= 0)[0]
    j = int(np.nonzero(~adj[i] & (np.arange(len(adj)) != i))[0][0])
    o_wrong = dict(o, neighbours=nbr)
    o_wrong["neighbours"][i, k] = j
    bad_link[i, k] = 1
    caught = [bool(illegal(bad_pad, o, adj)) if (o["neighbours"] < 0).any() else True,
              bool(illegal(np.full_like(bad_pad, 2), o, adj)), bool(illegal(bad_pad[:-1], o, adj)),
              bool(illegal(bad_link, o_wrong, adj))]
    chk.record("discrimination: the legality check flags padding, non-binary, misshapen and unselected-link actions",
               all(caught), f"caught {caught}")

    finite = all(r["all_finite"] for r in tr.curve)
    o2, _ = tr.env("Greedy", 2.0).reset(seed=7)
    v, adv = ag.advantage(ag.augment(o2))
    chk.record("reward, loss, V and A finite throughout training (loads 1 and 2)",
               finite and np.isfinite(v) and np.isfinite(adv).all() and np.isfinite(ag.losses).all()
               and len(ag.losses) > 0,
               f"{len(ag.losses)} updates; reward min {min(r['reward_min'] for r in tr.curve):.4g}")
    chk.record("time-limit truncation bootstraps: no stored transition is terminal",
               ag.buffer.size == steps and float(ag.buffer.data["done"][: ag.buffer.size].sum()) == 0.0,
               f"{ag.buffer.size} transitions")
    chk.record("exploration decays from eps_start to eps_end over the run",
               tr.curve[0]["eps_start"] == tiny.eps_start and tr.curve[-1]["eps_end"] == tiny.eps_end
               and tr.curve[-1]["deviation_frac"] < tr.curve[0]["deviation_frac"],
               f"eps {tr.curve[0]['eps_start']:g} -> {tr.curve[-1]['eps_end']:g}; training deviation "
               f"{tr.curve[0]['deviation_frac']:.3f} -> {tr.curve[-1]['deviation_frac']:.3f}")

    expect_sync = ag.updates - ag.updates % tiny.target_sync
    chk.record("target network was copied on schedule and has moved from its initial weights",
               ag.last_sync == expect_sync and expect_sync > 0 and _hash_module(ag.target) != init_target,
               f"{ag.updates} updates, last sync {ag.last_sync} (expected {expect_sync})")
    lag = ag.updates > ag.last_sync and not _params_equal(ag.online, ag.target)
    ag.sync()
    chk.record("target lags the online network between syncs and equals it after one",
               lag and _params_equal(ag.online, ag.target), "")

    # Double-DQN wiring through the agent: online selects, target values.
    b = ag.buffer.sample(8, np.random.default_rng(3))
    kappa = tiny.residual_margin
    _set_head(ag.online, 0.0, 2 * kappa)  # online deviates on every live slot
    _set_head(ag.target, 0.0, -0.3)  # target would deviate nowhere; values each deviation at -0.3
    y = ag.targets(b)
    # Every live slot deviates, so the mean deviation term is the per-slot A.
    r = torch.as_tensor(b["reward"])
    gamma = tiny.gamma
    v_t = 0.0
    y_online_selects = r + gamma * (v_t - 0.3)
    y_target_selects = r + gamma * v_t
    y_online_values = r + gamma * (v_t + 2 * kappa)
    ok = torch.allclose(y, y_online_selects, atol=1e-5)
    chk.record("agent targets: online network selects the deviations, target network values them", ok,
               f"y[0] {float(y[0]):.5f}; online-selects {float(y_online_selects[0]):.5f}, "
               f"target-selects {float(y_target_selects[0]):.5f}, online-values {float(y_online_values[0]):.5f}")


def check_reproducible(chk: Checks, cache: ScenarioCache, base: EnvConfig) -> None:
    section("Reproducibility and resume (3x8)")
    short = dataclasses.replace(base, episode_steps=VCFG.short_episode_steps)
    tiny, n = VCFG.tiny, VCFG.short_episodes

    def run(acfg, episodes, seed=0):
        tr = _short(acfg, cache, short, seed=seed)
        tr.train(episodes, eval_every=0)
        return tr

    a, b = run(tiny, n), run(tiny, n)
    c = run(dataclasses.replace(tiny, seed=1), n)
    same = _hash_module(a.agent.online) == _hash_module(b.agent.online) and \
        [r["reward_mean"] for r in a.curve] == [r["reward_mean"] for r in b.curve]
    chk.record("same seed: identical weights and learning curve", same,
               f"online sha {_hash_module(a.agent.online)[:16]}")
    chk.record("discrimination: a different agent seed gives different weights",
               _hash_module(c.agent.online) != _hash_module(a.agent.online), "")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        first = run(tiny, n // 2)
        first.save(path)
        resumed = _short(tiny, cache, short)
        resumed.load(path)
        resumed.train(n - n // 2, eval_every=0)
        perturbed = _short(tiny, cache, short)
        perturbed.load(path)
        perturbed.rng = np.random.default_rng(12345)
        perturbed.train(n - n // 2, eval_every=0)
    ok = (_hash_module(resumed.agent.online) == _hash_module(a.agent.online)
          and _hash_module(resumed.agent.target) == _hash_module(a.agent.target)
          and [r["reward_mean"] for r in resumed.curve] == [r["reward_mean"] for r in a.curve])
    chk.record(f"checkpoint after {n // 2} episodes + resume = {n} straight, bit for bit", ok,
               f"resumed {_hash_module(resumed.agent.online)[:16]} vs straight {_hash_module(a.agent.online)[:16]}")
    chk.record("discrimination: resuming with a different trainer RNG diverges",
               _hash_module(perturbed.agent.online) != _hash_module(a.agent.online), "")


def check_can_move(chk: Checks, cache: ScenarioCache, base: EnvConfig) -> Trainer:
    section("The learning loop can leave backpressure (3x8)")
    short = dataclasses.replace(base, episode_steps=VCFG.short_episode_steps)
    acfg = dataclasses.replace(VCFG.tiny, gamma=VCFG.move_gamma, lr=VCFG.move_lr)
    tr = _short(acfg, cache, short, reward_factory=lambda cfg: DeviationReward(VCFG.move_pay))
    tr.train(VCFG.move_episodes, eval_every=0)
    env = tr.env("Greedy", 1.0)
    pol = AgentPolicy(tr.agent)
    for seed in (7, 8):
        rollout(env, pol, seed, {"start_step": 1400})
    chk.record("paid to deviate, the trained agent overrides backpressure on most live slots",
               pol.deviation_frac >= VCFG.move_min_frac,
               f"greedy deviation fraction {pol.deviation_frac:.3f}, required >= {VCFG.move_min_frac:g} "
               f"after {tr.agent.updates} updates")
    return tr


# --- Q calibration -------------------------------------------------------------------
def _calibrate(agent: DDQNAgent, env: LeoForwardingEnv, seeds, starts) -> tuple[dict, list[np.ndarray]]:
    """Calibration averaged over greedy episodes, plus every action taken."""
    rows, acts = [], []
    for seed in seeds:
        for start in starts:
            pol = AgentPolicy(agent)
            obs, _ = env.reset(seed=seed, options={"start_step": start})
            rewards, done = [], False
            while not done:
                a = pol.act(obs)
                acts.append(a)
                obs, r, _, done, _ = env.step(a)
                rewards.append(r)
            rows.append(calibration(pol.q, rewards, agent.cfg.gamma, TrainConfig().calib_tail_tol)
                        | {"dev": pol.deviation_frac})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, acts


def _calibrated(c: dict) -> tuple[bool, float]:
    bound = VCFG.calib_rel_tol * c["g_abs"] + VCFG.calib_abs_tol
    return abs(c["bias"]) <= bound, bound


def _calib_detail(c: dict, bound: float) -> str:
    return (f"Q {c['q_mean']:+.3f} vs return {c['g_mean']:+.3f}: bias {c['bias']:+.3f}, bound {bound:.3f} "
            f"({c['n']:.0f} states/episode, deviation {c['dev']:.3f})")


def check_calibration_units(chk: Checks) -> None:
    section("Q calibration: unit pins")
    g = discounted_returns([1.0, 1.0, 1.0], 0.5)
    chk.record("pinned: discounted_returns([1, 1, 1], 0.5) = [1.75, 1.5, 1]", list(g) == PIN_RETURNS, f"{list(g)}")
    tails = [tail_steps(0.95, 0.01), tail_steps(0.5, 0.01)]
    chk.record("pinned: tail_steps(0.95, 0.01) = 90, tail_steps(0.5, 0.01) = 7", tails == PIN_TAILS, f"{tails}")
    r = np.ones(10)
    c = calibration(discounted_returns(r, 0.5) + 2.0, r, 0.5, 0.01)
    chk.record("pinned: Q = G + 2 over 10 steps at gamma 0.5, tol 0.01 gives bias 2 on the first 4 states",
               c["n"] == 4 and abs(c["bias"] - 2.0) < 1e-12 and abs(c["mae"] - 2.0) < 1e-12, f"{c}")
    c = calibration(np.zeros(5), np.ones(5), 0.95, 0.01)
    chk.record("an episode shorter than the tail compares nothing: n = 0 and NaN, never a silent 0",
               c["n"] == 0 and np.isnan(c["bias"]), f"{c}")


def check_calibration(chk: Checks, cache: ScenarioCache, base: EnvConfig, move: Trainer) -> None:
    """Q must be the value of what the agent does. Missing until 2026-11-20:
    the 300-episode run's Q reached +470 against rewards in [-13, +0.3] and no
    check or report showed it. Three parts: a learnable reference (the
    paid-to-deviate agent, whose Q has the form the network assumes), a
    discrimination (shift V, which moves Q and not one action), and the real
    queue reward with and without enough target syncs for the value to
    propagate. Not the shadow reward: its Q is not a function of the state
    (see AgentValidationConfig), so a greedy return cannot calibrate it."""
    section("Q calibration: Q against the realised return (3x8)")
    env = move.train_env("Greedy", 1.0)
    c, acts = _calibrate(move.agent, env, (7, 8), (1400,))
    ok, bound = _calibrated(c)
    chk.record("paid to deviate: Q is calibrated to the return the agent earns", ok, _calib_detail(c, bound))

    shift = 2.0 * (bound + abs(c["bias"]))
    with torch.no_grad():
        move.agent.online.net.head.bias[0] += shift
    c2, acts2 = _calibrate(move.agent, env, (7, 8), (1400,))
    with torch.no_grad():
        move.agent.online.net.head.bias[0] -= shift
    moved = abs((c2["bias"] - c["bias"]) - shift) <= 1e-3 * shift
    same = len(acts) == len(acts2) and all(np.array_equal(a, b) for a, b in zip(acts, acts2))
    chk.record("discrimination: V shifted by s moves the bias by s, changes no action, and is flagged",
               moved and same and not _calibrated(c2)[0],
               f"s {shift:.3f}; bias {c['bias']:+.3f} -> {c2['bias']:+.3f}; actions identical {same}")

    def queue_run(sync):
        acfg = dataclasses.replace(VCFG.tiny, reward_kind="queue", target_sync=sync, buffer_size=VCFG.calib_buffer)
        tr = Trainer(acfg, TrainConfig(selectors=("Greedy",), loads=VCFG.calib_loads), base, cache)
        tr.train(VCFG.calib_episodes, eval_every=0)
        rows = [_calibrate(tr.agent, tr.env("Greedy", load), (VCFG.calib_seed,), VCFG.calib_starts)[0]
                for load in VCFG.calib_loads]
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, tr.agent.updates

    c, upd = queue_run(VCFG.calib_target_sync)
    ok, bound = _calibrated(c)
    chk.record(f"queue reward, {VCFG.calib_episodes} episodes, target sync {VCFG.calib_target_sync}: "
               f"Q is calibrated to the return", ok, _calib_detail(c, bound) + f", {upd} updates")
    c, upd = queue_run(VCFG.calib_slow_sync)
    ok, bound = _calibrated(c)
    chk.record(f"discrimination: the same run with target sync {VCFG.calib_slow_sync} (value not propagated) "
               f"is flagged", not ok, _calib_detail(c, bound) + f", {upd} updates")


def main() -> int:
    print("=" * 72)
    print("  PHASE 3 STEP 2 VALIDATION -- RESIDUAL DDQN ON BACKPRESSURE")
    print(f"  agent: {AgentConfig().variant_label}")
    print("=" * 72)
    torch.set_num_threads(VCFG.torch_threads)
    chk = Checks()
    check_units(chk)
    check_calibration_units(chk)

    debug = EnvConfig()  # 3x8, populous20: the CPU debug scale
    cache = ScenarioCache(debug, ROOT, verbose=True)
    check_guard(chk, cache, debug, "3x8", VCFG.guard_seeds, VCFG.guard_starts)
    check_untrained(chk, cache, debug, "3x8", VCFG.guard_seeds, VCFG.guard_starts)
    check_loop(chk, cache, debug)
    check_reproducible(chk, cache, debug)
    move = check_can_move(chk, cache, debug)
    check_calibration(chk, cache, debug, move)

    primary = EnvConfig(n_planes=6, sats_per_plane=12, gateway_sites="krishnan8")
    big = ScenarioCache(primary, ROOT, verbose=True)
    check_guard(chk, big, primary, "6x12 krishnan8", VCFG.guard_seeds[:1], VCFG.guard_starts_big)
    check_untrained(chk, big, primary, "6x12 krishnan8", VCFG.guard_seeds[:1], VCFG.guard_starts_big)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
