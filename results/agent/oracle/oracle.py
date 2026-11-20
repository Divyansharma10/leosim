"""Factored one-step improvement oracle: do per-unit gains over backpressure compose?

    python results/agent/oracle/oracle.py --load 1 --seed 7 --start 1200 --gran sat
    python results/agent/oracle/oracle.py --load 1 --seed 7 --start 1200 --gran link --w-max 0

No learning. A UNIT is one slot (flip backpressure's bit, either direction)
or one satellite (any of the 2^live - 1 non-backpressure sub-actions over its
live slots). The GAIN of a deviation at state s is the H-step discounted queue
reward (gamma 0.95) of "deviate once, then backpressure" minus that of
backpressure from s. Two measurements along one episode:

  additivity  every ``additivity_every`` steps: each unit's best deviation
              with gain > 0; the SUM of those gains against the gain of
              applying all of them in the same step (JOINT).
  policy      at every step, apply every unit's best deviation whose gain
              exceeds kappa, jointly -- what a perfectly learned factored
              advantage with margin kappa would do after one improvement step
              from backpressure -- against a parallel backpressure run:
              mean per-step queue-reward difference (= the shadow reward) and
              delivered bits relative to backpressure.

What it does NOT show: whether a factored form could converge to a better
fixed point by repeated improvement; only that one factored step from
backpressure does or does not compose. Horizon H truncates the return;
gamma^H of the discount weight lies beyond it (0.13 at H = 40).

First run 2026-11-20 at seed 7, start 1200 only (scratch); this script is the
replication over seeds and starts. Writes one JSON per run to --out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlenv import BackpressurePolicy, EnvConfig, LeoForwardingEnv, forward_step  # noqa: E402

BP = BackpressurePolicy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--gran", choices=["link", "sat"], required=True)
    p.add_argument("--w-max", type=float, default=EnvConfig.w_max, help="reward max-queue weight (default: EnvConfig's)")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--additivity-every", type=int, default=10)
    p.add_argument("--kappa", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--planes", type=int, default=6)
    p.add_argument("--per-plane", type=int, default=12)
    p.add_argument("--gateway-sites", default="krishnan8")
    p.add_argument("--selector", default="Greedy")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "agent" / "oracle" / "runs")
    a = p.parse_args()

    cfg = EnvConfig(n_planes=a.planes, sats_per_plane=a.per_plane, gateway_sites=a.gateway_sites,
                    selector=a.selector, load=a.load, w_max=a.w_max)
    env = LeoForwardingEnv(cfg)
    env.reset(seed=a.seed, options={"start_step": a.start})
    sc = env.scenario
    if a.start + a.steps + a.horizon > sc.n_steps:
        raise SystemExit("episode plus horizon runs past the scenario")

    def step(q, f, t):
        return forward_step(q, f, sc.isl_capacity(t), sc.downlink[t], sc.arrivals(env._offered[t], t),
                            env.queues.dt, env.queues.buffer_bits)

    def qr(q):
        x = q / sc.queue_ref_bits
        return -(cfg.w_mean * x.mean() + cfg.w_max * x.max())

    def ret(q, t0, first):
        acc = 0.0
        for h in range(a.horizon):
            t = t0 + h
            act = first if (h == 0 and first is not None) else BP.act(env.observe(q, t))
            q = step(q, env.forward_for(act, t), t).queue
            acc += a.gamma ** h * qr(q)
        return acc

    def candidates(act, mask):
        if a.gran == "link":
            for i, k in np.argwhere(mask):
                b = act.copy()
                b[i, k] ^= 1
                yield (int(i), int(k)), b
        else:
            for i in range(act.shape[0]):
                live = np.nonzero(mask[i])[0]
                for bits in itertools.product((0, 1), repeat=len(live)):
                    if np.array_equal(bits, act[i, live]):
                        continue
                    b = act.copy()
                    b[i, live] = bits
                    yield int(i), b

    def best_per_unit(q, t, act, mask):
        base = ret(q, t, None)
        best = {}
        for u, b in candidates(act, mask):
            g = ret(q, t, b) - base
            if u not in best or g > best[u][0]:
                best[u] = (g, b)
        return base, best

    def joint(act, best, thr):
        out = act.copy()
        for u, (g, b) in best.items():
            if g > thr:
                out[u] = b[u]  # u indexes a slot (i, k) or a satellite row i
        return out

    qa, qb = np.zeros(env.n_sats), np.zeros(env.n_sats)
    ra = rb = da = db = 0.0
    sums, joints, frac = [], [], []
    t_wall = time.time()
    for t in range(a.start, a.start + a.steps):
        obs = env.observe(qa, t)
        act = BP.act(obs)
        mask = obs["action_mask"].astype(bool)
        base, best = best_per_unit(qa, t, act, mask)
        if (t - a.start) % a.additivity_every == 0:
            pos = {u: v for u, v in best.items() if v[0] > 0}
            sums.append(float(sum(v[0] for v in pos.values())))
            joints.append(float(ret(qa, t, joint(act, pos, 0.0)) - base))
        frac.append(float(np.mean([v[0] > a.kappa for v in best.values()])) if best else 0.0)
        r = step(qa, env.forward_for(joint(act, best, a.kappa), t), t)
        qa, da = r.queue, da + float(r.delivered.sum())
        s = step(qb, env.forward_for(BP.act(env.observe(qb, t)), t), t)
        qb, db = s.queue, db + float(s.delivered.sum())
        ra += qr(qa)
        rb += qr(qb)

    out = {"gran": a.gran, "load": a.load, "seed": a.seed, "start": a.start, "w_mean": cfg.w_mean,
           "w_max": cfg.w_max, "horizon": a.horizon, "steps": a.steps, "kappa": a.kappa, "gamma": a.gamma,
           "variant": cfg.variant_label,
           "sum_gain": sums, "joint_gain": joints,
           "sum_gain_mean": float(np.mean(sums)), "joint_gain_mean": float(np.mean(joints)),
           "policy_shadow_mean": (ra - rb) / a.steps, "policy_delivered_vs_bp": da / db if db else float("nan"),
           "units_deviating_frac": float(np.mean(frac)), "wall_s": time.time() - t_wall}
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"{a.gran}_L{a.load:g}_s{a.seed}_t{a.start}_wmax{cfg.w_max:g}_H{a.horizon}.json"
    (a.out / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{name}: sum {out['sum_gain_mean']:+.3f} joint {out['joint_gain_mean']:+.3f} "
          f"policy {out['policy_shadow_mean']:+.4f} delivered {out['policy_delivered_vs_bp']:.4f} "
          f"({out['wall_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
