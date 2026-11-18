"""How much routed traffic crosses an ISL at the RL evaluation scale (6x12).

Diagnostic only: changes nothing in leosim/, topology/, traffic/ or rlenv/.

    python results/rlenv/isl_crossing/measure_isl_crossing.py          # from the repo root
    python results/rlenv/isl_crossing/measure_isl_crossing.py --quick  # Greedy only, 1 seed

Seeds 7-9 for Greedy, seed 7 for the other selectors: hop shares agree across
seeds to ~0.1% (Poisson noise only), and this keeps the run ~20 min.

Three measures, because "ISL hops per bit" means different things in the two
queue models:

A. Stage 4 (TrafficSim, to_gateway, hop metric): fixed shortest-path routes,
   so every routed bit has a path and a hop count. Same instrumentation as
   results/phase2/diagnostic/diagnose_routing.py (the 76% figure at 187
   sats), every recomputed total asserted equal to TrafficSim's own. A city
   here picks whichever access satellite minimises access + path cost, so it
   uses a gateway-seeing satellite whenever it sees one.

B. RL env attachment (load-independent): a city's bits join its
   best-CAPACITY satellite (rlenv.scenario.attach_cities), whether or not
   that satellite sees a gateway. Demand-weighted BFS hop distance from that
   satellite to the nearest gateway-seeing satellite over the selected links
   = the fewest ISL hops any forwarding policy could use. "stranded" = no
   gateway reachable in its component.

C. RL env rollouts (backpressure, shortest-path): no per-bit paths, so flow
   ratios: ISL bits sent / bits arrived (mean ISL transmissions per arrived
   bit, if the backlog were drained) and delivered fraction.

Load is EnvConfig.load (served demand / subset downlink capacity), and stage
4 is fed the env scenario's own scaled offered bits, as validate_rlenv's
curve-shape check does.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlenv import EnvConfig, LeoForwardingEnv  # noqa: E402
from rlenv.policies import BackpressurePolicy, ShortestPathPolicy, rollout  # noqa: E402
from rlenv.scenario import build_dataset, build_scenario  # noqa: E402
from routing import RoutingConfig, compute_access, dataset_times  # noqa: E402
from traffic import SparseAccess, TrafficSim, gateway_access, load_sites  # noqa: E402

OUT = Path(__file__).resolve().parent
PLANES, PER_PLANE = 6, 12
GATEWAYS = ("populous20", "krishnan8")
SELECTORS = ("Greedy", "DoTD-forward", "DoTD-backward", "+Grid")
LOADS = (0.25, 0.5, 1.0, 2.0)
SEEDS = (7, 8, 9)
STARTS = (0, 400, 800, 1200, 1599)  # the 6x12 scale-selection episodes
EPISODE = 200
BINS = ("0", "1", "2", "3+")
RTOL = 1e-9


def hop_bin(h: np.ndarray) -> np.ndarray:
    return np.minimum(h, 3)


class Instrumented(TrafficSim):
    """TrafficSim that also records each city's ISL hops and delivered bits."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows: list[tuple] = []  # (t, city, isl_hops, offered, delivered)

    def routes(self, links, t):
        self._paths = super().routes(links, t)
        return self._paths

    def step(self, links, t, offered):
        b_prev = self.backlog.copy()
        res = super().step(links, t, offered)
        cap = self._capacity(links, t)
        b_prev[cap <= 0] = 0.0
        arrivals = np.zeros_like(cap)
        edges = {}
        for c, path in enumerate(self._paths):
            if not path:
                continue
            src, dst = np.asarray(path[1:]), np.asarray(path[:-1])  # to_gateway
            edges[c] = (src, dst)
            np.add.at(arrivals, (src, dst), offered[c])
        b = b_prev + arrivals - cap * self.dt
        dropped = np.maximum(b - self.tcfg.buffer_s * cap, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            drop_frac = np.where(arrivals > 0, dropped / arrivals, 0.0)
        delivered = 0.0
        for c, (src, dst) in edges.items():
            d = offered[c] * float(np.prod(1.0 - drop_frac[src, dst]))
            self.rows.append((t, c, len(src) - 2, offered[c], d))
            delivered += d
        np.testing.assert_allclose(delivered, res["delivered_bits"], rtol=RTOL, atol=1e-3)
        np.testing.assert_allclose(sum(offered[c] for c in edges), res["routed_bits"], rtol=RTOL, atol=1e-3)
        np.testing.assert_allclose(dropped.sum(), res["dropped_bits"], rtol=RTOL, atol=1e-3)
        return res


def attach_hops(sc) -> np.ndarray:
    """(T, C) BFS ISL hops from each city's attach satellite to the nearest
    gateway-seeing satellite; -1 no access, -2 stranded."""
    T, C = sc.city_sat.shape
    out = np.full((T, C), -1, dtype=np.int64)
    for t in range(T):
        gw = np.nonzero(sc.downlink[t] > 0)[0]
        sat = sc.city_sat[t]
        ok = sat >= 0
        if len(gw) == 0:
            out[t, ok] = -2
            continue
        d = shortest_path(csr_matrix(sc.linksets[t].adjacency.astype(np.float64)), unweighted=True,
                          directed=False, indices=gw).min(axis=0)
        h = d[sat[ok]]
        out[t, ok] = np.where(np.isfinite(h), h, -2).astype(np.int64)
    return out


def dist_row(hops: np.ndarray, bits: np.ndarray) -> dict:
    """Shares of bits by hop bin, over bits that have a hop count (>= 0)."""
    ok = hops >= 0
    tot = bits[ok].sum()
    b = hop_bin(hops[ok])
    row = {f"h{k}": float(bits[ok][b == i].sum() / tot) if tot else np.nan for i, k in enumerate(BINS)}
    row["isl_share"] = 1.0 - row["h0"]
    row["mean_hops"] = float((hops[ok] * bits[ok]).sum() / tot) if tot else np.nan
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")  # Greedy only, one seed
    args = ap.parse_args()
    selectors = SELECTORS[:1] if args.quick else SELECTORS
    seeds = SEEDS[:1] if args.quick else SEEDS
    rcfg = RoutingConfig()

    base = EnvConfig(n_planes=PLANES, sats_per_plane=PER_PLANE)
    ds = build_dataset(base, ROOT, verbose=True)
    times = dataset_times(ds)
    print(f"N={ds.n_sats}, steps {ds.n_steps}, step {ds.config.step_s:g} s, {rcfg.variant_label}")

    ctx, stage4, attach, env_rows = [], [], [], []
    for gws in GATEWAYS:
        for sel in selectors:
            t0 = time.time()
            cfg = dataclasses.replace(base, gateway_sites=gws, selector=sel, episode_steps=EPISODE)
            sc1 = build_scenario(cfg, ROOT, dataset=ds)
            sites = load_sites(sc1.traffic, ROOT)
            ca = [SparseAccess.from_access(compute_access(ds, s, rcfg, times)) for s in sites.cities]
            ga = [gateway_access(ds, s, rcfg, sc1.traffic, times) for s in sites.gateways]

            # Context: gateway visibility and connectivity (load-independent).
            from scipy.sparse.csgraph import connected_components
            ncomp = [connected_components(csr_matrix(ls.adjacency), directed=False)[0] for ls in sc1.linksets]
            hops_b = attach_hops(sc1)
            rate = sc1.rate_bps  # expected demand at load 1, (T, C)
            served = hops_b != -1
            ctx.append({"gateways": gws, "selector": sel,
                        "sat_steps_gw_in_view": float(np.mean(sc1.downlink > 0)),
                        "steps_no_gw_sat": float(np.mean((sc1.downlink > 0).sum(1) == 0)),
                        "steps_connected": float(np.mean(np.array(ncomp) == 1)),
                        "env_stranded_demand": float(rate[hops_b == -2].sum() / rate[served].sum()),
                        "scaled_peak_gbps_load1": sc1.traffic.peak_total_gbps})
            attach.append({"gateways": gws, "selector": sel, **dist_row(hops_b, rate)})

            for load in LOADS:
                c = dataclasses.replace(cfg, load=load)
                sc = build_scenario(c, ROOT, dataset=ds)
                for seed in (seeds if sel == "Greedy" else seeds[:1]):
                    off = sc.offered(seed)
                    # A. stage 4 over the whole window
                    sim = Instrumented(ds, ca, ga, dataclasses.replace(sc.traffic, direction="to_gateway",
                                                                       seed=seed), rcfg)
                    res = pd.DataFrame(sim.run(sc.linksets, off))
                    r = pd.DataFrame(sim.rows, columns=["t", "city", "isl_hops", "offered", "delivered"])
                    h, o, d = r["isl_hops"].to_numpy(), r["offered"].to_numpy(), r["delivered"].to_numpy()
                    routed = res["routed_bits"].sum()
                    stage4.append({
                        "gateways": gws, "selector": sel, "load": load, "seed": seed,
                        **{f"routed_{k}": v for k, v in dist_row(h, o).items()},
                        **{f"delivered_{k}": v for k, v in dist_row(h, d).items()},
                        "delivered_frac": res["delivered_bits"].sum() / routed,
                        "delivered_frac_0hop": d[h == 0].sum() / o[h == 0].sum() if (h == 0).any() else np.nan,
                        "delivered_frac_isl": d[h > 0].sum() / o[h > 0].sum() if (h > 0).any() else np.nan,
                        "no_path_share": res["no_path_bits"].sum() / (routed + res["no_path_bits"].sum()),
                    })
                    # C. env rollouts
                    for pol in (BackpressurePolicy(), ShortestPathPolicy()):
                        env = LeoForwardingEnv(c, sc)
                        arr = dlv = isl = back = 0.0
                        for st in STARTS:
                            ro = rollout(env, pol, seed, options={"start_step": st})
                            arr += ro["arrived_bits"]
                            dlv += ro["delivered_bits"]
                            isl += sum(x["isl_bits"] for x in ro["rows"])
                            back += ro["final_backlog_bits"]
                        env_rows.append({"gateways": gws, "selector": sel, "load": load, "seed": seed,
                                         "policy": pol.name, "isl_per_arrived_bit": isl / arr,
                                         "delivered_frac": dlv / arr, "backlog_frac": back / arr})
            print(f"  {gws:<10} {sel:<14} done in {time.time() - t0:.0f} s", flush=True)

    ctx, attach, stage4, env_rows = map(pd.DataFrame, (ctx, attach, stage4, env_rows))
    for name, df in (("context", ctx), ("env_attach", attach), ("stage4", stage4), ("env_rollouts", env_rows)):
        df.to_csv(OUT / f"{name}{'_quick' if args.quick else ''}.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n== context ==")
    print(ctx.to_string(index=False))
    print("\n== B. RL-env attachment: fewest ISL hops to a gateway-seeing satellite (demand-weighted) ==")
    print(attach.to_string(index=False))
    g = stage4.groupby(["gateways", "selector", "load"], sort=False).mean(numeric_only=True).drop(columns="seed")
    print("\n== A. stage 4 OSPF: ISL hops per ROUTED bit, mean over seeds ==")
    print(g[[c for c in g if c.startswith("routed_")] + ["no_path_share"]].to_string())
    print("\n== A. stage 4 OSPF: ISL hops per DELIVERED bit, and delivery by class ==")
    print(g[[c for c in g if c.startswith("delivered_")]].to_string())
    print("\n== C. env rollouts: ISL transmissions per arrived bit, delivered fraction ==")
    e = env_rows.groupby(["gateways", "selector", "load", "policy"], sort=False).mean(numeric_only=True)
    print(e.drop(columns="seed").unstack("policy").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
