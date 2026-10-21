"""Phase 3 validation: the Gymnasium forwarding environment.

    python validate_rlenv.py               # default 3x8, 200-step episodes
    python validate_rlenv.py --steps 300   # accepted for symmetry with the
                                           # other suites; sets episode_steps

Same rules as the other suites. Every check drives production code; pinned
values are hard-coded literals; each check that could pass vacuously has a
partner showing it cannot.

The Phase 3 hazard is a wrapper that changes what it wraps: an observation
that loses the precision a baseline decides on, an action decoder that
reaches a link the selector never chose, an episode one step long or short.
Every one of those leaves plausible numbers. So backpressure is run twice,
through env.step and through a direct loop over the queue model that knows
nothing of slots, masks or normalisation, and the two must agree exactly.

There are no Phase 2 backpressure numbers to reproduce: Phase 2 routes by
fixed shortest path. The comparison against Phase 2 is therefore a curve
shape: the env's shortest-path policy against stage 4's TrafficSim over the
same topology and the same offered bits. Same routing, different queue model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np

from routing import RoutingConfig, compute_access, dataset_times
from run_traffic import utc_hours
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from traffic import SparseAccess, TrafficSim, expected_rate_bps, gateway_access, load_sites, offered_bits
from rlenv import (
    BackpressurePolicy,
    ConstantPolicy,
    EnvConfig,
    LeoForwardingEnv,
    QueueModel,
    QueueReward,
    RandomPolicy,
    ShadowBackpressureReward,
    ShortestPathPolicy,
    build_scenario,
    forward_step,
    neighbour_slots,
    rollout,
)
from rlenv.config import RLValidationConfig
from rlenv.scenario import attach_cities, best_downlink, make_selector
from validate import Checks, section
from validate_traffic import _sparse

ROOT = Path(__file__).parent
VCFG = RLValidationConfig()

# --- pinned literals ----------------------------------------------------------
# Line 0-1-2, ISLs 1 Gbps, gateway on 2 at 1 Gbps, 1 s step, forwarding 0->1
# and 1->2, queues [2, 0.5, 0] Gb. 0 sends 1 Gb; 1 holds only 0.5 Gb so sends
# that; 2 had nothing at step start. Result [1, 1, 0.5] Gb.
PIN_LINE_QUEUE = [1.0e9, 1.0e9, 0.5e9]
# Satellite 0 holds 1 Gb with links of 1 and 3 Gbps active: shared 1:3.
PIN_SHARE = [0.25e9, 0.75e9]
# 1 Gb buffer, 1.5 Gb arrives at an empty satellite with no exit.
PIN_DROPPED = 0.5e9
# Two gateways see satellite 0 at 1 and 3 Gbps; satellite 1 sees nothing.
PIN_DOWNLINK = [3.0e9, 0.0]
# Backpressure through the env, default config, seed 7. Recorded on this code
# when the suite was written (there is no earlier version to record from), so
# a regression guard, not a correctness proof. Re-record only if stage 1-4
# inputs legitimately change, and say why in the commit.
# The same numbers came out of the direct loop, which shares only QueueModel.
PIN_BP_DELIVERED_FRAC = 0.9874144425419
PIN_BP_REWARD_MEAN = -0.7438283471610014
PIN_BP_QUEUE_SHA256 = "830c70749d136efc8a46becf07528b7f1e529193c3ab5371f388c0a20da6aa79"


class Recorder:
    """A reward_fn that records each step's context, so checks can read what
    the queue model actually did through the production path."""

    def __init__(self, inner) -> None:
        self.inner, self.ctxs = inner, []

    def reset(self, env) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset(env)

    def __call__(self, ctx) -> float:
        self.ctxs.append(ctx)
        return self.inner(ctx)


def _env(cfg, sc, record=False):
    rec = Recorder(QueueReward(cfg.w_mean, cfg.w_max)) if record else None
    return LeoForwardingEnv(cfg, sc, reward_fn=rec), rec


def _obs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _queue_sha(qs) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.stack(qs)).tobytes()).hexdigest()


# --- unit pins ------------------------------------------------------------------
def check_queue_model(chk: Checks) -> None:
    section("Queue model (hand-built)")
    isl = np.zeros((3, 3))
    isl[0, 1] = isl[1, 0] = isl[1, 2] = isl[2, 1] = 1e9
    fwd = np.zeros((3, 3), bool)
    fwd[0, 1] = fwd[1, 2] = True
    r = forward_step(np.array([2e9, 0.5e9, 0.0]), fwd, isl, np.array([0, 0, 1e9]), np.zeros(3), 1.0, np.inf)
    chk.record("line: each satellite sends what it holds, up to capacity",
               np.allclose(r.queue, PIN_LINE_QUEUE, rtol=0, atol=1e-3),
               f"queues {r.queue.tolist()}, pinned {PIN_LINE_QUEUE}")
    r2 = forward_step(r.queue, fwd, isl, np.array([0, 0, 1e9]), np.zeros(3), 1.0, np.inf)
    chk.record("line, next step: the downlink drains what arrived the step before",
               np.isclose(r2.delivered[2], 0.5e9) and np.isclose(r2.delivered[:2].sum(), 0.0),
               f"delivered {r2.delivered.tolist()}")

    isl = np.zeros((3, 3))
    isl[0, 1], isl[0, 2] = 1e9, 3e9
    isl = isl + isl.T
    fwd = np.zeros((3, 3), bool)
    fwd[0, 1] = fwd[0, 2] = True
    r = forward_step(np.array([1e9, 0, 0]), fwd, isl, np.zeros(3), np.zeros(3), 1.0, np.inf)
    chk.record("a short queue is shared over active links in proportion to capacity",
               np.allclose([r.sent[0, 1], r.sent[0, 2]], PIN_SHARE, rtol=1e-12),
               f"sent {r.sent[0, 1]:.4g}, {r.sent[0, 2]:.4g}; pinned {PIN_SHARE}")
    r = forward_step(np.zeros(1), np.zeros((1, 1), bool), np.zeros((1, 1)), np.zeros(1),
                     np.array([1.5e9]), 1.0, 1e9)
    chk.record("finite buffer drops the overflow", np.isclose(r.dropped[0], PIN_DROPPED) and r.queue[0] == 1e9,
               f"dropped {r.dropped[0]:.4g}, queue {r.queue[0]:.4g}")
    r = forward_step(np.zeros(1), np.zeros((1, 1), bool), np.zeros((1, 1)), np.zeros(1),
                     np.array([1.5e9]), 1.0, np.inf)
    chk.record("unbounded buffer (default) drops nothing", r.dropped[0] == 0 and r.queue[0] == 1.5e9,
               f"dropped {r.dropped[0]:.4g}, queue {r.queue[0]:.4g}")


def check_reshaping(chk: Checks) -> None:
    section("Slots, downlink, attachment (hand-built)")
    adj = np.zeros((4, 4), bool)
    for i, j in [(0, 3), (0, 1), (2, 1)]:
        adj[i, j] = adj[j, i] = True
    nbr = neighbour_slots(adj, 4)
    want = [[1, 3, -1, -1], [0, 2, -1, -1], [1, -1, -1, -1], [0, -1, -1, -1]]
    chk.record("neighbour slots ascending, padded with -1", nbr.tolist() == want, f"{nbr.tolist()}")
    try:
        neighbour_slots(adj, 1)
        chk.record("degree above the slot count is rejected", False, "no error")
    except ValueError as e:
        chk.record("degree above the slot count is rejected", True, str(e))

    gw = [_sparse([[0]], 1e9, 0.002), _sparse([[0]], 3e9, 0.002)]
    dl = best_downlink(gw, 1, 2)
    chk.record("downlink is the best gateway in view, 0 if none", dl[0].tolist() == PIN_DOWNLINK,
               f"{dl[0].tolist()}, pinned {PIN_DOWNLINK}")
    acc = SparseAccess(np.array([0, 2]), np.array([0, 1]), np.array([1e9, 4e9]), np.array([0.002, 0.002]))
    sat = attach_cities([acc, _sparse([[]], 1e9, 0.002)], 1)
    chk.record("a city's bits join its highest-capacity access satellite; -1 with none",
               sat[0].tolist() == [1, -1], f"{sat[0].tolist()}")


def check_reward_unit(chk: Checks, sc) -> None:
    section("Reward (hand-built contexts)")
    rew = QueueReward(1.0, 1.0)
    from rlenv.reward import RewardContext
    from rlenv.queues import StepResult

    def ctx(q):
        q = np.asarray(q, dtype=np.float64)
        z = np.zeros_like(q)
        return RewardContext(z, StepResult(q, z, np.zeros((len(q), len(q))), z, z), np.zeros((len(q), 1)),
                             0, 1.0, lambda a: None, np.zeros((len(q), len(q))), z)
    r0, r1, r2, r3 = rew(ctx([0, 0])), rew(ctx([1, 1])), rew(ctx([1, 3])), rew(ctx([2, 2]))
    chk.record("empty queues give reward 0; any queue gives a negative reward",
               r0 == 0.0 and r1 < 0, f"r(0,0) = {r0}, r(1,1) = {r1}")
    chk.record("reward falls when every queue grows (mean term)", r3 < r1, f"r(1,1) {r1}, r(2,2) {r3}")
    chk.record("reward falls when one queue grows (max term)", r2 < r1, f"r(1,1) {r1}, r(1,3) {r2}")
    chk.record("pinned: r(1, 3) = -(mean 2 + max 3) = -5", r2 == -5.0, f"{r2}")
    w = QueueReward(2.0, 0.0)
    chk.record("weights are applied: w_mean=2, w_max=0 on (1, 3) gives -4", w(ctx([1, 3])) == -4.0,
               f"{w(ctx([1, 3]))}")
    try:
        ShadowBackpressureReward(1.0, 1.0)(ctx([1, 1]))
        chk.record("shadow reward refuses to run before env.reset()", False, "returned a value")
    except RuntimeError:
        chk.record("shadow reward refuses to run before env.reset()", True, "raises RuntimeError")


# --- inputs equal stages 2-4 ----------------------------------------------------------
def check_inputs(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Inputs are stages 2-4's, unchanged")
    ds = sc.dataset
    tcfg = TopologyConfig()
    direct = {
        "+Grid": PlusGrid(ds, tcfg), "Greedy": Greedy(ds, tcfg),
        "DoTD-forward": DoTD(ds, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(ds, dataclasses.replace(tcfg, lookahead="backward")),
    }
    same, distinct = [], set()
    for name, sel in direct.items():
        want = sel.run(sc.n_steps)
        got = make_selector(name, ds, tcfg).run(sc.n_steps)
        same.append(all(np.array_equal(a.adjacency, b.adjacency) for a, b in zip(want, got)))
        distinct.add(hashlib.sha256(np.stack([w.adjacency for w in want]).tobytes()).hexdigest())
    chk.record("each selector name builds that stage-2 selector's exact topology", all(same),
               f"identical: {dict(zip(direct, same))}")
    want_kind = {"+Grid": (PlusGrid, None), "Greedy": (Greedy, "none"),
                 "DoTD-forward": (DoTD, "forward"), "DoTD-backward": (DoTD, "backward")}
    kinds = {n: (type(s_), getattr(s_.config, "lookahead", None) if n != "+Grid" else None)
             for n, s_ in ((n, make_selector(n, ds, tcfg)) for n in want_kind)}
    chk.record("each name builds its own selector class and lookahead",
               all(kinds[n][0] is want_kind[n][0] and kinds[n][1] == want_kind[n][1] for n in want_kind),
               f"{ {n: (k[0].__name__, k[1]) for n, k in kinds.items()} }")
    # Not a uniqueness assertion: at 3x8 Greedy, DoTD-forward and DoTD-backward
    # choose identical topologies at every step (visible degree averages 4.8 and
    # 42% of satellites see <= 4, so the matching is nearly forced and the
    # lookahead has nothing to decide). A FINDING about this scale, printed.
    chk.record("discrimination: the structural and scored selectors' topologies differ", len(distinct) >= 2,
               f"{len(distinct)} distinct sequences among 4 selectors")
    chk.record("the env's topology is the configured selector's",
               all(np.array_equal(a.adjacency, b.adjacency)
                   for a, b in zip(sc.linksets, direct[cfg.selector].run(sc.n_steps))), cfg.selector)

    sites = load_sites(sc.traffic, ROOT)
    want = offered_bits(expected_rate_bps(sites, utc_hours(ds), sc.traffic)[:sc.n_steps],
                        ds.config.step_s, dataclasses.replace(sc.traffic, seed=11))
    chk.record("offered bits are stage 4's offered_bits for that seed", np.array_equal(sc.offered(11), want),
               f"shape {want.shape}, total {want.sum() / 1e9:.1f} Gb")
    chk.record("discrimination: another seed gives other bits", not np.array_equal(sc.offered(12), want), "")

    served = np.where(sc.city_sat >= 0, sc.rate_bps, 0.0).sum(axis=1).mean()
    ratio = served / sc.downlink.sum(axis=1).mean()
    chk.close_to("load is served demand over the subset's downlink capacity", ratio, cfg.load, 1e-9)
    sc2 = build_scenario(dataclasses.replace(cfg, load=2 * cfg.load), dataset=ds)
    chk.close_to("doubling load doubles the scaled peak", sc2.traffic.peak_total_gbps / sc.traffic.peak_total_gbps,
                 2.0, 1e-12)

    rcfg, times = RoutingConfig(), dataset_times(ds)
    gw = [gateway_access(ds, s, rcfg, sc.traffic, times) for s in sites.gateways]
    t = cfg.start_step
    best = np.zeros(sc.n_sats)
    for a in gw:
        s_, c_, _ = a.at(t)
        np.maximum.at(best, s_, c_)
    chk.record("downlink capacity is stage 4's gateway_access", np.array_equal(sc.downlink[t], best),
               f"t={t}: {int((best > 0).sum())} satellites with a gateway in view")

    env, _ = _env(cfg, sc)
    obs, _ = env.reset(seed=7)
    nbr = obs["neighbours"]
    cap = np.asarray(ds.capacity[t], dtype=np.float64)
    want_cap = np.where(nbr >= 0, cap[np.arange(sc.n_sats)[:, None], np.maximum(nbr, 0)], 0.0)
    chk.record("isl_capacity x a FIXED reference is the stage-1 capacity of that slot's link",
               np.allclose(obs["isl_capacity"] * sc.isl_ref_bps, want_cap, rtol=1e-12),
               f"reference {sc.isl_ref_bps / 1e9:.3f} Gbps")
    vis = np.asarray(ds.visible)
    chk.close_to("the ISL reference is the window's largest visible-pair capacity",
                 sc.isl_ref_bps, float(np.asarray(ds.capacity)[vis].max()), 1e-12)
    chk.close_to("the downlink reference is the window's largest downlink",
                 sc.downlink_ref_bps, float(sc.downlink.max()), 1e-12)


# --- episode mechanics -------------------------------------------------------------
def check_determinism(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Determinism")
    a, _ = _env(cfg, sc)
    b, _ = _env(cfg, sc)
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    chk.record("two resets with the same seed give identical first observations", _obs_equal(oa, ob), "")
    pol = BackpressurePolicy()
    ra, rb = rollout(a, pol, 5), rollout(b, pol, 5)
    same = np.array_equal(ra["rewards"], rb["rewards"]) and ra["final_backlog_bits"] == rb["final_backlog_bits"]
    chk.record("same seed, same policy: identical trajectories", same,
               f"final backlog {ra['final_backlog_bits']:.6g} vs {rb['final_backlog_bits']:.6g}")
    rc = rollout(a, pol, 6)
    chk.record("discrimination: another seed gives another trajectory",
               not np.array_equal(ra["rewards"], rc["rewards"]),
               f"seed 5 reward {ra['reward_mean']:.6f}, seed 6 {rc['reward_mean']:.6f}")
    o1, _ = a.reset(seed=5)
    a.step(BackpressurePolicy().act(o1))
    o2, _ = a.reset(seed=5)
    chk.record("reset clears state: re-reset after stepping matches a fresh reset", _obs_equal(o2, oa), "")


def check_stepping(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Stepping and episode boundary")
    env, rec = _env(cfg, sc, record=True)
    obs, info = env.reset(seed=7)
    off = sc.offered(7)
    t0 = cfg.start_step
    chk.record("first observation is timestep start_step",
               info["t"] == t0 and np.array_equal(obs["neighbours"], sc.neighbours[t0])
               and np.allclose(obs["downlink_capacity"] * sc.downlink_ref_bps, sc.downlink[t0], rtol=1e-12),
               f"t={info['t']}, start_step={t0}")
    chk.record("discrimination: timesteps start_step and start_step+1 are distinguishable",
               not np.array_equal(sc.downlink[t0], sc.downlink[t0 + 1]), "downlink capacities differ")
    ts, applied, offered_ok, conserved, trunc, prev_q, n = [], [], True, True, [], 0.0, 0
    cap_ok, step_max = True, []
    cap_all = np.asarray(sc.dataset.capacity, dtype=np.float64)
    rows = np.arange(sc.n_sats)[:, None]
    pol = RandomPolicy(seed=3)
    done = False
    while not done:
        obs, r, term, done, info = env.step(pol.act(obs))
        ctx = rec.ctxs[-1]
        ts.append(info["t"])
        applied.append(info["applied_t"])
        trunc.append(done)
        offered_ok &= info["offered_bits"] == float(off[info["applied_t"]].sum())
        lhs = ctx.prev_queue.sum() + ctx.result.arrivals.sum()
        rhs = ctx.result.queue.sum() + ctx.result.delivered.sum() + ctx.result.dropped.sum()
        conserved &= abs(lhs - rhs) <= 1e-9 * max(lhs, 1.0)
        nbr = obs["neighbours"]
        want = np.where(nbr >= 0, cap_all[info["t"]][rows, np.maximum(nbr, 0)], 0.0)
        cap_ok &= np.allclose(obs["isl_capacity"] * sc.isl_ref_bps, want, rtol=1e-12)
        step_max.append(want.max())
        n += 1
    steps_ok = ts == list(range(t0 + 1, t0 + cfg.episode_steps + 1)) and applied == list(range(t0, t0 + cfg.episode_steps))
    chk.record("each step advances exactly one timestep", steps_ok,
               f"applied {applied[0]}..{applied[-1]}, returned {ts[0]}..{ts[-1]}")
    chk.record("episode is exactly episode_steps long; truncated only on the last step",
               n == cfg.episode_steps and trunc[-1] and not any(trunc[:-1]),
               f"{n} steps, episode_steps={cfg.episode_steps}")
    chk.record("final observation is timestep start_step + episode_steps",
               np.array_equal(obs["neighbours"], sc.neighbours[t0 + cfg.episode_steps]), f"t={info['t']}")
    chk.record("each step consumes that timestep's offered bits", offered_ok, "")
    chk.record("isl_capacity x the fixed reference is that step's stage-1 capacity, every step", cap_ok,
               f"reference {sc.isl_ref_bps / 1e9:.3f} Gbps")
    # Only needs to beat the 1e-12 tolerance above. At 3x8 one in-plane pair
    # sits near the window maximum throughout, so the margin is small (~5e-4).
    gap = 1.0 - min(step_max) / sc.isl_ref_bps
    chk.record("discrimination: some step's largest selected capacity is below the reference",
               gap > 1e-9, f"per-step max {min(step_max):.6g}..{max(step_max):.6g} bits/s; "
               f"largest relative gap {gap:.2e}")
    chk.record("bits conserved every step: Q + arrivals = Q' + delivered + dropped", conserved, "")
    chk.record("discrimination: traffic actually flows (arrivals, ISL transfers, delivery all > 0)",
               all(x.result.arrivals.sum() > 0 for x in rec.ctxs[:1])
               and sum(x.result.sent.sum() for x in rec.ctxs) > 0
               and sum(x.result.delivered.sum() for x in rec.ctxs) > 0, "")
    try:
        env.step(pol.act(obs))
        chk.record("step() after truncation is refused", False, "no error")
    except RuntimeError as e:
        chk.record("step() after truncation is refused", True, str(e))
    o, i = env.reset(seed=7, options={"start_step": t0 + 3})
    chk.record("reset option start_step moves the window", i["t"] == t0 + 3
               and np.array_equal(o["neighbours"], sc.neighbours[t0 + 3]), f"t={i['t']}")


def check_legality(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Every accepted action is a legal link set")
    d = sc.n_slots
    policies = [RandomPolicy(seed=1), ConstantPolicy(True), BackpressurePolicy()]
    ok_sub = ok_deg = ok_sym = True
    sent_any = False
    for pol in policies:
        env, rec = _env(cfg, sc, record=True)
        rollout(env, pol, 7)
        for ctx in rec.ctxs:
            adj = sc.linksets[ctx.t].adjacency
            used = ctx.result.sent > 0
            sent_any |= used.any()
            ok_sub &= not (used & ~adj).any()
            ok_deg &= used.sum(axis=1).max() <= d and adj.sum(axis=1).max() <= d
            ok_sym &= np.array_equal(adj, adj.T) and not adj.diagonal().any()
    chk.record("bits move only over links the selector chose", ok_sub, "Random, All-on, Backpressure")
    chk.record(f"out-degree of links carrying bits never above {d}", ok_deg, "")
    chk.record("the selected link set is symmetric with an empty diagonal", ok_sym, "")
    chk.record("discrimination: those policies did move bits over ISLs", sent_any, "")

    # Padding: ones on padding slots only must change nothing.
    env_a, rec_a = _env(cfg, sc, record=True)
    env_b, rec_b = _env(cfg, sc, record=True)
    oa, _ = env_a.reset(seed=7)
    ob, _ = env_b.reset(seed=7)
    padded = 0
    same = True
    for _ in range(cfg.episode_steps):
        pad = (oa["action_mask"] == 0).astype(np.int8)
        padded += int(pad.sum())
        oa, *_ = env_a.step(pad)
        ob, *_ = env_b.step(np.zeros_like(pad))
        same &= _obs_equal(oa, ob)
    chk.record("ones on padding slots are ignored (same trajectory as all zeros)", same,
               f"{padded} padding slot-steps set to 1")
    chk.record("discrimination: there were padding slots to test", padded > 0, f"{padded}")

    env, _ = _env(cfg, sc)
    obs, _ = env.reset(seed=7)
    rejects = []
    for bad in (np.ones((sc.n_sats, d + 1), np.int8), np.full((sc.n_sats, d), 2), np.full((sc.n_sats, d), -1)):
        try:
            env.step(bad)
            rejects.append(False)
        except ValueError:
            rejects.append(True)
    chk.record("wrong shape or non-binary actions are rejected, not coerced", all(rejects), f"{rejects}")


def check_observations(chk: Checks, cfg: EnvConfig, sc, sc_over) -> None:
    section("Observations finite and within the declared space")
    cfg_over = dataclasses.replace(cfg, load=VCFG.overload)
    all_ok, finite, qmax, raw_over = True, True, 0.0, 0.0
    n_obs = 0
    for c, s in ((cfg, sc), (cfg_over, sc_over)):
        for pol in (BackpressurePolicy(), RandomPolicy(seed=2), ConstantPolicy(True), ConstantPolicy(False)):
            env, _ = _env(c, s)
            obs, _ = env.reset(seed=7)
            done = False
            while True:
                all_ok &= env.observation_space.contains(obs)
                finite &= all(np.isfinite(v).all() for v in obs.values())
                qmax = max(qmax, float(obs["queue"].max()))
                raw_over = max(raw_over, float(env.queues.queue.max()))
                n_obs += 1
                if done:
                    break
                obs, _, _, done, _ = env.step(pol.act(obs))
    chk.record("every observation is inside observation_space", all_ok, f"{n_obs} observations, 4 policies, 2 loads")
    chk.record("every observation is finite", finite, "")
    chk.between(f"discrimination: at load {VCFG.overload:g} the queue observation nears its bound",
                qmax, VCFG.saturation, 1.0, extra="else the [0, 1] check could pass on queues that never grew")
    chk.record("discrimination: the raw queues there are far outside [0, 1]", raw_over > 1e3,
               f"largest raw queue {raw_over:.3g} bits")


def check_inside_alongside(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Backpressure inside the env = backpressure alongside it")
    env, rec = _env(cfg, sc, record=True)
    res = rollout(env, BackpressurePolicy(), 7)
    env_q = [c.result.queue for c in rec.ctxs]

    # Direct: no slots, no mask, no normalisation; raw queues, (Q_i - Q_j) C_ij > 0.
    qm = QueueModel(sc.n_sats, sc.dataset.config.step_s, sc.buffer_bits)
    off = sc.offered(7)
    direct_q, delivered = [], 0.0
    for t in range(cfg.start_step, cfg.start_step + cfg.episode_steps):
        cap = sc.isl_capacity(t)
        q = qm.queue
        fwd = (q[:, None] - q[None, :]) * cap > 0
        r = qm.step(fwd, cap, sc.downlink[t], sc.arrivals(off[t], t))
        direct_q.append(r.queue)
        delivered += r.delivered.sum()
    same = all(np.array_equal(a, b) for a, b in zip(env_q, direct_q))
    chk.record("queue trajectory identical, bit for bit", same,
               f"final backlog env {sum(env_q[-1]):.10g}, direct {sum(direct_q[-1]):.10g}")
    chk.record("delivered identical", res["delivered_bits"] == delivered,
               f"{res['delivered_bits']:.10g} vs {delivered:.10g}")
    chk.record("discrimination: backpressure differs from all-on here",
               rollout(env, ConstantPolicy(True), 7)["final_backlog_bits"] != res["final_backlog_bits"], "")

    sha = _queue_sha(env_q)
    print(f"         [record] delivered_frac={res['delivered_frac']!r} reward_mean={res['reward_mean']!r} sha={sha}")
    if cfg != EnvConfig():
        print("         [pins] not the default config: golden pins not applicable, skipped")
        return
    if PIN_BP_QUEUE_SHA256 is None:
        chk.record("backpressure golden pins recorded", False, "pins are None: record them from the line above")
        return
    chk.close_to("pinned: backpressure delivered fraction", res["delivered_frac"], PIN_BP_DELIVERED_FRAC, 1e-12)
    chk.close_to("pinned: backpressure episode-mean reward", res["reward_mean"], PIN_BP_REWARD_MEAN, 1e-12)
    chk.record("pinned: backpressure queue trajectory SHA-256", sha == PIN_BP_QUEUE_SHA256, sha)


def check_policies(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("The action space matters: Random vs backpressure")
    bp = [rollout(LeoForwardingEnv(cfg, sc), BackpressurePolicy(), s) for s in VCFG.seeds]
    rnd = [rollout(LeoForwardingEnv(cfg, sc), RandomPolicy(seed=s), s) for s in VCFG.seeds]
    off = [rollout(LeoForwardingEnv(cfg, sc), ConstantPolicy(False), s) for s in VCFG.seeds]
    b, r_, o = ([x["reward_mean"] for x in v] for v in (bp, rnd, off))
    chk.record("Random loses to backpressure on every seed", max(r_) < min(b),
               f"backpressure {np.round(b, 3).tolist()}, Random {np.round(r_, 3).tolist()}")
    chk.record(f"by at least {VCFG.random_reward_factor:g}x in mean reward",
               np.mean(r_) <= VCFG.random_reward_factor * np.mean(b),
               f"mean {np.mean(b):.3f} vs {np.mean(r_):.3f}")
    chk.record("Random also delivers less", np.mean([x["delivered_frac"] for x in rnd])
               < np.mean([x["delivered_frac"] for x in bp]),
               f"{np.mean([x['delivered_frac'] for x in bp]):.3f} vs {np.mean([x['delivered_frac'] for x in rnd]):.3f}")
    chk.record("No-ISL (all off) is worse than backpressure", max(o) < min(b), f"No-ISL {np.round(o, 3).tolist()}")
    chk.record("discrimination: backpressure's own queues are non-trivial at this load",
               min(x["queue_max"] for x in bp) > 0.1, f"max q/q_ref {[round(x['queue_max'], 3) for x in bp]}")

    res = rollout(LeoForwardingEnv(cfg, sc), ConstantPolicy(False), VCFG.seeds[0])
    rw = res["rewards"]
    q = len(rw) // 4
    chk.record("rewards finite under every policy",
               all(np.isfinite(x["rewards"]).all() for x in bp + rnd + off), "")
    chk.record("reward falls as queues grow (No-ISL: last quarter below first)",
               rw[-q:].mean() < rw[:q].mean(), f"first {rw[:q].mean():.3f}, last {rw[-q:].mean():.3f}")


def check_counterfactual(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Counterfactual hook (for the shadow reward)")
    env, rec = _env(cfg, sc, record=True)
    obs, _ = env.reset(seed=7)
    for _ in range(20):
        a = BackpressurePolicy().act(obs)
        obs, *_ = env.step(a)
    ctx = rec.ctxs[-1]
    again = ctx.counterfactual(ctx.action.astype(np.int8))
    chk.record("counterfactual with the taken action reproduces the step",
               np.array_equal(again.queue, ctx.result.queue), "")
    other = ctx.counterfactual(np.zeros_like(ctx.action, dtype=np.int8))
    chk.record("discrimination: another action gives another result",
               not np.array_equal(other.queue, ctx.result.queue), "")
    chk.record("counterfactual does not mutate the env's state",
               np.array_equal(env.queues.queue, ctx.result.queue), "")


def check_shadow(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Shadow-backpressure reward (Eq. 20, parallel run)")
    shadow_cfg = dataclasses.replace(cfg, reward_kind="shadow")
    env = LeoForwardingEnv(shadow_cfg, sc)
    r_bp = rollout(env, BackpressurePolicy(), 7)["rewards"]
    chk.record("with backpressure as the agent the shadow reward is exactly 0 every step",
               np.all(r_bp == 0.0), f"max |r| {np.abs(r_bp).max():.3g}")

    # Random agent, episode not at step 0: the shadow must be a real
    # backpressure run from empty queues at that start, untouched by the agent.
    start = cfg.start_step + 3
    shadow = ShadowBackpressureReward(cfg.w_mean, cfg.w_max)
    rec = Recorder(shadow)
    env_a = LeoForwardingEnv(shadow_cfg, sc, reward_fn=rec)
    obs, _ = env_a.reset(seed=7, options={"start_step": start})
    pol = RandomPolicy(seed=4)
    shadow_q, agent_ctx, r_shadow = [], [], []
    done = False
    while not done:
        obs, r, _, done, _ = env_a.step(pol.act(obs))
        shadow_q.append(shadow.last.queue)
        agent_ctx.append(rec.ctxs[-1])
        r_shadow.append(r)
    env_b, rec_b = _env(cfg, sc, record=True)
    rollout(env_b, BackpressurePolicy(), 7, {"start_step": start})
    bp_q = [c.result.queue for c in rec_b.ctxs]
    chk.record("the shadow is a backpressure rollout of its own, bit for bit (Random agent, start != 0)",
               len(bp_q) == len(shadow_q) and all(np.array_equal(a, b) for a, b in zip(shadow_q, bp_q)),
               f"final shadow backlog {shadow_q[-1].sum():.10g}, backpressure rollout {bp_q[-1].sum():.10g}")
    qr = QueueReward(cfg.w_mean, cfg.w_max)
    want = np.array([qr(a) - qr(b) for a, b in zip(agent_ctx, rec_b.ctxs)])
    chk.record("shadow reward = queue reward(agent) - queue reward(backpressure), every step",
               np.allclose(r_shadow, want, rtol=1e-9, atol=1e-12),
               f"max |difference| {np.abs(np.array(r_shadow) - want).max():.3g}")
    chk.record("discrimination: Random's shadow reward is negative on the mean (it loses to backpressure)",
               np.mean(r_shadow) < 0, f"mean {np.mean(r_shadow):.4f}")
    again = []
    obs, _ = env_a.reset(seed=7, options={"start_step": start})
    pol = RandomPolicy(seed=4)
    done = False
    while not done:
        obs, r, _, done, _ = env_a.step(pol.act(obs))
        again.append(r)
    chk.record("reset restarts the shadow from empty: a repeat episode gives the same rewards",
               np.array_equal(again, r_shadow), "")


def check_scaling_and_starts(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Reward scaling and episode starts")
    scale = 0.1
    for kind in ("queue", "shadow"):
        a = rollout(LeoForwardingEnv(dataclasses.replace(cfg, reward_kind=kind), sc), RandomPolicy(seed=5), 7)
        b = rollout(LeoForwardingEnv(dataclasses.replace(cfg, reward_kind=kind, reward_scale=scale), sc),
                    RandomPolicy(seed=5), 7)
        chk.record(f"reward_scale multiplies the {kind} reward exactly",
                   np.allclose(b["rewards"], scale * a["rewards"], rtol=1e-12, atol=0)
                   and not np.array_equal(a["rewards"], b["rewards"]),
                   f"mean {a['reward_mean']:.4f} -> {b['reward_mean']:.4f}")

    last = sc.n_steps - 1 - cfg.episode_steps
    uni = LeoForwardingEnv(dataclasses.replace(cfg, start_mode="uniform"), sc)
    starts = [uni.reset(seed=s)[1]["t"] for s in range(20)]
    chk.record("uniform starts fall inside the window and vary with the seed",
               all(0 <= x <= last for x in starts) and len(set(starts)) >= 10,
               f"{len(set(starts))} distinct starts of 20 seeds, range {min(starts)}..{max(starts)}, last {last}")
    chk.record("a seed fixes the uniform start", uni.reset(seed=3)[1]["t"] == starts[3], f"{starts[3]}")
    fixed = LeoForwardingEnv(cfg, sc)
    chk.record("fixed mode always starts at start_step",
               {fixed.reset(seed=s)[1]["t"] for s in range(5)} == {cfg.start_step}, f"{cfg.start_step}")


def check_phase2_curve(chk: Checks, cfg: EnvConfig, sc) -> None:
    section("Curve shape against Phase 2 (stage-4 TrafficSim, same routing)")
    ds, rcfg = sc.dataset, RoutingConfig()
    times = dataset_times(ds)
    sites = load_sites(sc.traffic, ROOT)
    ca = [SparseAccess.from_access(compute_access(ds, s, rcfg, times)) for s in sites.cities]
    ga = [gateway_access(ds, s, rcfg, sc.traffic, times) for s in sites.gateways]
    sp, ph2, bp = [], [], []
    for load in VCFG.curve_loads:
        c = dataclasses.replace(cfg, load=load)
        s = build_scenario(c, dataset=ds)
        sp.append(rollout(LeoForwardingEnv(c, s), ShortestPathPolicy(), 7)["delivered_frac"])
        bp.append(rollout(LeoForwardingEnv(c, s), BackpressurePolicy(), 7)["delivered_frac"])
        # Phase 2's traffic is to_user by default; the env's is uplink, so the
        # comparison runs stage 4 in its to_gateway direction.
        tc = dataclasses.replace(s.traffic, direction="to_gateway", seed=7)
        sim = TrafficSim(ds, ca, ga, tc, rcfg)
        off = s.offered(7)
        rows = [sim.step(s.linksets[t], t, off[t]) for t in range(c.start_step, c.start_step + c.episode_steps)]
        ph2.append(sum(x["delivered_bits"] for x in rows) / sum(x["routed_bits"] for x in rows))
    table = "\n".join(f"load {l:<5g} env shortest-path {a:.3f}  Phase 2 OSPF {b:.3f}  env backpressure {p:.3f}"
                      for l, a, b, p in zip(VCFG.curve_loads, sp, ph2, bp))
    diff = float(np.max(np.abs(np.array(sp) - np.array(ph2))))
    chk.between("same routing, two queue models: delivered fractions agree", diff, 0.0, VCFG.curve_tol, extra=table)
    mono = lambda v: all(b <= a + VCFG.monotone_tol for a, b in zip(v, v[1:]))
    chk.record("both curves fall with load", mono(sp) and mono(ph2), "")
    chk.record("discrimination: the curves actually fall (congestion reached)",
               sp[-1] < 0.8 * sp[0] and ph2[-1] < 0.8 * ph2[0], f"env {sp[0]:.3f}->{sp[-1]:.3f}, "
               f"Phase 2 {ph2[0]:.3f}->{ph2[-1]:.3f}")
    chk.record("backpressure delivers at least what shortest path does at every load",
               all(p >= a - VCFG.bp_vs_sp_tol for p, a in zip(bp, sp)), "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None, help="episode_steps (default: EnvConfig's)")
    p.add_argument("--planes", type=int, default=EnvConfig.n_planes)
    p.add_argument("--per-plane", type=int, default=EnvConfig.sats_per_plane)
    p.add_argument("--selector", default=EnvConfig.selector)
    args = p.parse_args()
    cfg = EnvConfig(n_planes=args.planes, sats_per_plane=args.per_plane, selector=args.selector)
    if args.steps is not None:
        cfg = dataclasses.replace(cfg, episode_steps=args.steps)
    print("=" * 72)
    print("  PHASE 3 VALIDATION -- RL FORWARDING ENVIRONMENT")
    print(f"  variant: {cfg.variant_label}")
    print("=" * 72)

    chk = Checks()
    check_queue_model(chk)
    check_reshaping(chk)
    sc = build_scenario(cfg, ROOT, verbose=True)
    print(f"  N={sc.n_sats}, D={sc.n_slots}, scaled peak {sc.traffic.peak_total_gbps:.3f} Gbps")
    sc_over = build_scenario(dataclasses.replace(cfg, load=VCFG.overload), dataset=sc.dataset)
    check_reward_unit(chk, sc)
    check_inputs(chk, cfg, sc)
    check_determinism(chk, cfg, sc)
    check_stepping(chk, cfg, sc)
    check_legality(chk, cfg, sc)
    check_observations(chk, cfg, sc, sc_over)
    check_inside_alongside(chk, cfg, sc)
    check_counterfactual(chk, cfg, sc)
    check_shadow(chk, cfg, sc)
    check_scaling_and_starts(chk, cfg, sc)
    check_policies(chk, cfg, sc)
    check_phase2_curve(chk, cfg, sc)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
