"""Phase 3: baseline policies through the RL environment.

    python run_rlenv.py                                  # 3x8, load 1, 200 steps
    python run_rlenv.py --loads 0.5 1 2 --seeds 7 8 9
    python run_rlenv.py --planes 10 --per-plane 20       # 187 sats
    python run_rlenv.py --selector DoTD-forward --episode-steps 600

Every policy runs inside the environment through env.step, exactly as an
agent will. Backpressure is Krishnan et al. Eq. 16; Shortest-path is stage 4's
hop routing as a policy; No-ISL is all links off; Random is p = 0.5 per slot.

Reports per policy, load and seed: episode-mean reward, mean and peak
normalised queue (q / q_ref), delivered fraction of arrived bits (bits still
queued at the episode end count as undelivered) and the final backlog.
Demand no satellite can see is excluded; it never enters a queue.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from rlenv import (
    BackpressurePolicy,
    ConstantPolicy,
    EnvConfig,
    LeoForwardingEnv,
    RandomPolicy,
    ShortestPathPolicy,
    build_scenario,
    rollout,
)


def main() -> int:
    d = EnvConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", type=int, default=d.n_planes)
    p.add_argument("--per-plane", type=int, default=d.sats_per_plane)
    p.add_argument("--step", type=float, default=d.step_s)
    p.add_argument("--duration", type=float, default=d.duration_s)
    p.add_argument("--selector", choices=["+Grid", "Greedy", "DoTD-forward", "DoTD-backward"], default=d.selector)
    p.add_argument("--start-step", type=int, default=d.start_step)
    p.add_argument("--episode-steps", type=int, default=d.episode_steps)
    p.add_argument("--loads", type=float, nargs="+", default=[d.load])
    p.add_argument("--seeds", type=int, nargs="+", default=[d.seed])
    p.add_argument("--gateway-sites", choices=["populous20", "krishnan8"], default=d.gateway_sites)
    p.add_argument("--queue-scale-s", type=float, default=d.queue_scale_s)
    p.add_argument("--buffer-s", type=float, default=d.buffer_s,
                   help="finite buffer; see rlenv.config before setting (drops lower the queue-only reward)")
    p.add_argument("--w-mean", type=float, default=d.w_mean)
    p.add_argument("--w-max", type=float, default=d.w_max)
    p.add_argument("--reward-kind", choices=["queue", "shadow"], default=d.reward_kind)
    p.add_argument("--reward-scale", type=float, default=d.reward_scale)
    p.add_argument("--start-mode", choices=["fixed", "uniform"], default=d.start_mode,
                   help="uniform: each seed draws its own episode start across the window")
    p.add_argument("--out", default="results/rlenv")
    args = p.parse_args()

    base = EnvConfig(
        n_planes=args.planes, sats_per_plane=args.per_plane, step_s=args.step, duration_s=args.duration,
        selector=args.selector, start_step=args.start_step, episode_steps=args.episode_steps,
        gateway_sites=args.gateway_sites, queue_scale_s=args.queue_scale_s, buffer_s=args.buffer_s,
        w_mean=args.w_mean, w_max=args.w_max, reward_kind=args.reward_kind,
        reward_scale=args.reward_scale, start_mode=args.start_mode,
    )
    dataset, rows = None, []
    for load in args.loads:
        cfg = dataclasses.replace(base, load=load)
        sc = build_scenario(cfg, dataset=dataset, verbose=dataset is None)
        dataset = sc.dataset
        print(f"[variant] {cfg.variant_label} | N={sc.n_sats} D={sc.n_slots} "
              f"scaled peak {sc.traffic.peak_total_gbps:.3f} Gbps")
        env = LeoForwardingEnv(cfg, sc)
        for seed in args.seeds:
            for pol in (BackpressurePolicy(), ShortestPathPolicy(), RandomPolicy(seed=seed),
                        ConstantPolicy(False)):
                r = rollout(env, pol, seed)
                rows.append({"variant": cfg.variant_label, "load": load, **{
                    k: v for k, v in r.items() if k not in ("rewards", "rows")}})
                print(f"  load {load:<5g} seed {seed:<3d} {pol.name:<14s} reward {r['reward_mean']:9.4f}  "
                      f"q mean {r['queue_mean']:7.3f} max {r['queue_max']:8.3f}  "
                      f"delivered {r['delivered_frac']:.3f}  backlog {r['final_backlog_bits'] / 1e9:8.2f} Gb")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"{base.n_planes}x{base.sats_per_plane}_{base.selector}_step{base.step_s:g}_"
           f"ep{base.start_step if base.start_mode == 'fixed' else 'uniform'}+{base.episode_steps}_"
           f"{base.gateway_sites}_{base.reward_kind}x{base.reward_scale:g}").replace("+Grid", "PlusGrid")
    path = out / f"baselines_{tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
