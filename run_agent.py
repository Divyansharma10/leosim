"""Phase 3 step 2: train the residual DDQN and report it against backpressure.

    python run_agent.py                                   # primary: 6x12, krishnan8, 100 episodes
    python run_agent.py --kappa 0.001                     # margin sweep (one value per run)
    python run_agent.py --gateway-sites populous20 --loads 1   # the secondary run
    python run_agent.py --device cuda --resume PATH       # continue a CPU run on GPU
    python run_agent.py --device cuda --auto-resume       # continue this run's own checkpoint if any
    python run_agent.py --planes 3 --per-plane 8 --episodes 5  # loop debugging ONLY

The primary result is trained AND evaluated at 6x12, where the selectors
differ; 3x8 (identical selectors) is for debugging the loop, never a result.

Every eval_every episodes the greedy agent is run on a fixed held-out set
(TrainConfig: all four selectors x eval loads x seeds 7-9 x starts 1200,
1400, 1599 -- timesteps and demand never trained on). Backpressure and
shortest-path run once on the same episodes. Reported per evaluation:
the gain in absolute queue reward over backpressure (agent - BP, so BP = 0
and higher is better), the same for shortest-path, the fraction of live
slots where the agent overrides backpressure, delivered fraction and ISL
transmissions per arrived bit. And Q calibration (agent.calibration): the network's
Q for the actions it took against the discounted return they earned. A Q far
from the return means the network is not valuing anything the agent does, and
no gain or deviation number from that evaluation is interpretable. The agent must beat BOTH baselines to count. Under krishnan8 shortest-path
beats backpressure only at load 0.25 (6x12 Greedy, eval seeds/starts, measured
2026-11-20: queue reward -0.500 vs -0.681, delivered 0.959 vs 0.776; the
2026-11-08 isl_crossing notes give 0.970 vs 0.845 delivered). From load 0.5
up it ties on delivery (0.705 vs 0.709) but loses on queue reward (-6.31 vs
-1.35): its tree funnels bits into a few gateway-seeing satellites, and the
max-queue term is 5x backpressure's. The default eval loads are 0.5, 1, 2, so
there the shortest-path half of the criterion never binds; add 0.25 to
--eval-loads for it to.

Outputs under --out/<tag>/: curve.csv (one row per training episode, rewards
include exploration), evals.csv (one row per evaluation episode),
learning_curve.png, summary.txt. Checkpoints go to --ckpt-dir (large: the
replay buffer is in them), written at every evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent import AgentConfig, ScenarioCache, TrainConfig, Trainer
from rlenv import EnvConfig

SELECTORS = ("Greedy", "DoTD-forward", "DoTD-backward", "+Grid")


def tag_for(base: EnvConfig, acfg: AgentConfig, tcfg: TrainConfig) -> str:
    loads = "-".join(f"{l:g}" for l in tcfg.loads)
    return (f"{base.n_planes}x{base.sats_per_plane}_{base.gateway_sites}_loads{loads}_"
            f"kappa{acfg.residual_margin:g}_{acfg.reward_kind}_ep{tcfg.episodes}_seed{acfg.seed}")


def summarise(evals: pd.DataFrame) -> pd.DataFrame:
    e = evals.assign(gain=evals.queue_reward - evals.bp_queue_reward,
                     sp_gain=evals.sp_queue_reward - evals.bp_queue_reward)
    return e.groupby(["episode", "load"]).agg(
        gain=("gain", "mean"), gain_sd=("gain", "std"), sp_gain=("sp_gain", "mean"),
        deviation=("deviation_frac", "mean"), delivered=("delivered_frac", "mean"),
        bp_delivered=("bp_delivered_frac", "mean"), sp_delivered=("sp_delivered_frac", "mean"),
        isl=("isl_per_arrived", "mean"), bp_isl=("bp_isl_per_arrived", "mean"),
        q=("calib_q_mean", "mean"), ret=("calib_g_mean", "mean"), q_bias=("calib_bias", "mean"),
        q_mae=("calib_mae", "mean"), n=("gain", "size")).reset_index()


def plot(summary: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    agent_c, sp_c = "#2a78d6", "#eb6834"  # categorical slots 1, 2
    loads = sorted(summary.load.unique())
    fig, axes = plt.subplots(2, len(loads), figsize=(4.2 * len(loads), 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    fig.patch.set_facecolor(surface)
    for j, load in enumerate(loads):
        s = summary[summary.load == load].sort_values("episode")
        se = s.gain_sd / np.sqrt(s.n)
        ax = axes[0, j]
        ax.axhline(0.0, color=muted, lw=1.2)
        ax.text(s.episode.min(), 0.0, " backpressure", color=muted, va="bottom", fontsize=8)
        ax.axhline(float(s.sp_gain.iloc[0]), color=sp_c, lw=2, ls=(0, (4, 3)), label="shortest-path")
        ax.fill_between(s.episode, s.gain - se, s.gain + se, color=agent_c, alpha=0.15, lw=0)
        ax.plot(s.episode, s.gain, color=agent_c, lw=2, marker="o", ms=4, label="residual DDQN")
        ax.set_title(f"load {load:g}", color=ink, fontsize=10, loc="left")
        if j == 0:
            ax.set_ylabel("queue-reward gain over backpressure\n(higher is better)", color=muted, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="best")
        ax2 = axes[1, j]
        ax2.plot(s.episode, s.deviation, color=agent_c, lw=2, marker="o", ms=4)
        ax2.set_xlabel("training episode", color=muted, fontsize=9)
        if j == 0:
            ax2.set_ylabel("share of links\noverriding BP", color=muted, fontsize=9)
        for a in (ax, ax2):
            a.set_facecolor(surface)
            a.grid(True, color=grid, lw=0.8)
            a.tick_params(colors=muted, labelsize=8)
            for sp in a.spines.values():
                sp.set_visible(False)
    fig.suptitle(title, color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=surface)
    plt.close(fig)


def main() -> int:
    a, t, e = AgentConfig(), TrainConfig(), EnvConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", type=int, default=6)
    p.add_argument("--per-plane", type=int, default=12)
    p.add_argument("--gateway-sites", choices=["populous20", "krishnan8"], default="krishnan8")
    p.add_argument("--episode-steps", type=int, default=e.episode_steps)
    p.add_argument("--episodes", type=int, default=t.episodes)
    p.add_argument("--loads", type=float, nargs="+", default=list(t.loads), help="training loads")
    p.add_argument("--eval-loads", type=float, nargs="+", default=None, help="default: the training loads")
    p.add_argument("--selectors", nargs="+", choices=SELECTORS, default=list(t.selectors))
    p.add_argument("--eval-every", type=int, default=t.eval_every)
    p.add_argument("--kappa", type=float, default=a.residual_margin)
    p.add_argument("--reward-kind", choices=["shadow", "queue"], default=a.reward_kind,
                   help="reward scale is fixed per kind (agent.config.REWARD_SCALE)")
    p.add_argument("--seed", type=int, default=a.seed, help="agent and trainer seed")
    p.add_argument("--device", default=a.device)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    p.add_argument("--auto-resume", action="store_true",
                   help="continue from this run's own checkpoint in --ckpt-dir if it exists")
    p.add_argument("--out", type=Path, default=Path("results/agent"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("cache/agent"))
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    base = EnvConfig(n_planes=args.planes, sats_per_plane=args.per_plane, gateway_sites=args.gateway_sites,
                     episode_steps=args.episode_steps)
    acfg = dataclasses.replace(a, residual_margin=args.kappa, reward_kind=args.reward_kind, seed=args.seed,
                               device=args.device)
    eval_loads = tuple(args.eval_loads) if args.eval_loads else tuple(args.loads)
    tcfg = dataclasses.replace(t, episodes=args.episodes, loads=tuple(args.loads), selectors=tuple(args.selectors),
                               eval_loads=eval_loads, eval_every=args.eval_every, seed=args.seed)
    tag = tag_for(base, acfg, tcfg)
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt_dir / f"{tag}.pt"
    print(f"[variant] env {base.variant_label}")
    print(f"[variant] agent {acfg.variant_label}")
    print(f"[variant] train loads {tcfg.loads} selectors {tcfg.selectors}; eval loads {tcfg.eval_loads}, "
          f"seeds {tcfg.eval_seeds}, starts {tcfg.eval_starts}")

    tr = Trainer(acfg, tcfg, base, ScenarioCache(base, verbose=True))
    if args.resume is None and args.auto_resume and ckpt.exists():
        args.resume = ckpt
    if args.resume:
        tr.load(args.resume)
        print(f"[resume] {args.resume} at episode {tr.episode}, {tr.agent.updates} updates")
    t0 = time.perf_counter()

    def report(row: dict) -> None:
        print(f"  ep {row['episode']:4d} {row['selector']:<14s} load {row['load']:<4g} reward {row['reward_mean']:+9.4f} "
              f"dev {row['deviation_frac']:.3f} loss {row['loss_mean']:.4g} upd {row['updates']:6d} "
              f"[{time.perf_counter() - t0:6.0f}s]", flush=True)

    if tr.episode == 0 and tcfg.eval_every:
        tr.evaluate()  # the untrained agent: episode 0 of the curve
        tr.save(ckpt)
    remaining = tcfg.episodes - tr.episode
    for _ in range(max(0, remaining)):
        row = tr.run_episode()
        report(row)
        if tcfg.eval_every and tr.episode % tcfg.eval_every == 0:
            tr.evaluate()
            tr.save(ckpt)
            s = summarise(pd.DataFrame(tr.evals))
            last = s[s.episode == tr.episode]
            print("  [eval] " + "  ".join(f"load {r.load:g}: gain {r.gain:+.4f} (SP {r.sp_gain:+.4f}) "
                                         f"dev {r.deviation:.3f} Q {r.q:+.3f} vs return {r.ret:+.3f}" for r in last.itertuples()), flush=True)

    pd.DataFrame(tr.curve).to_csv(out / "curve.csv", index=False)
    evals = pd.DataFrame(tr.evals)
    evals.insert(0, "variant", f"{base.variant_label} | {acfg.variant_label}")
    evals.to_csv(out / "evals.csv", index=False)
    s = summarise(evals)
    s.to_csv(out / "summary_by_eval.csv", index=False)
    plot(s, f"Residual DDQN vs backpressure -- {base.n_planes}x{base.sats_per_plane}, {base.gateway_sites}, "
            f"kappa {acfg.residual_margin:g}; held-out episodes, mean +- s.e.", out / "learning_curve.png")

    final = evals[evals.episode == evals.episode.max()]
    by_sel = final.assign(gain=final.queue_reward - final.bp_queue_reward,
                          sp_gain=final.sp_queue_reward - final.bp_queue_reward).groupby(["selector", "load"]).agg(
        gain=("gain", "mean"), gain_min=("gain", "min"), gain_max=("gain", "max"), sp_gain=("sp_gain", "mean"),
        deviation=("deviation_frac", "mean"), delivered=("delivered_frac", "mean"),
        bp_delivered=("bp_delivered_frac", "mean"), sp_delivered=("sp_delivered_frac", "mean"))
    lines = [f"env:   {base.variant_label}", f"agent: {acfg.variant_label}",
             f"train: {tcfg}", f"wall time {time.perf_counter() - t0:.0f}s this invocation, "
             f"{tr.agent.updates} updates total", "",
             "Per evaluation (mean over selectors x seeds x starts):", s.to_string(index=False), "",
             f"Final evaluation (episode {int(final.episode.max())}) by selector:", by_sel.to_string()]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-len(by_sel) - 2:]))
    print(f"[out] {out}  [ckpt] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
