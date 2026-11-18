"""Phase 2 diagnostic: did routing use the links the extension terms changed?

One run, no sweep: Greedy, terms off (0 / 0) vs on (0.2 / 0.2), 3.5 Gbps,
seed 7, 5 s step -- the PHASE2.md §4.1 configuration where the gain peaked.

TrafficSim reports per-step totals only. This wraps it (subclass, no change to
traffic/) to keep what it computes internally: the routed paths, and from
them per-link bits, drops and per-gateway delivered bits. Every quantity
recomputed here is asserted equal to TrafficSim's own per-step output before
it is used, and the run's delivered fraction must match the §4.1 table.

    python results/phase2/diagnostic/diagnose_routing.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from leosim.config import SelectionConfig, SimConfig  # noqa: E402
from leosim.simulation import run  # noqa: E402
from routing import RoutingConfig, compute_access, dataset_times  # noqa: E402
from run_traffic import utc_hours  # noqa: E402
from topology import Greedy, TopologyConfig  # noqa: E402
from traffic import (  # noqa: E402
    SparseAccess, TrafficConfig, TrafficSim, expected_rate_bps, floor_matrix,
    gateway_access, load_sites, offered_bits, predict_ground,
)

OUT = Path(__file__).resolve().parent
LOAD_GBPS, SEED, STEP_S = 3.5, 7, 5.0
WEIGHTS = {"off": (0.0, 0.0), "on": (0.2, 0.2)}
# PHASE2.md §4.1, Greedy at 3.5 Gbps: the run must reproduce these.
EXPECTED_DELIVERED = {"off": 0.9787, "on": 0.9856}
RTOL = 1e-9


class Instrumented(TrafficSim):
    """TrafficSim that also records per-link and per-gateway quantities."""

    def __init__(self, *a, gw_view: list[np.ndarray], **k):
        super().__init__(*a, **k)
        self.gw_view = gw_view  # per t: bool (N,), satellite sees >= 1 gateway
        self.link_rows: list[pd.DataFrame] = []
        self.gw_delivered = np.zeros(self.n_gw)
        self.gw_routed = np.zeros(self.n_gw)
        self.city_rows: list[tuple] = []  # (t, city, gateway, isl_hops, offered, delivered)

    def routes(self, links, t):
        self._paths = super().routes(links, t)
        return self._paths

    def step(self, links, t, offered):
        b_prev = self.backlog.copy()
        res = super().step(links, t, offered)
        n, gw0 = self.n, self.n + self.n_city
        cap = self._capacity(links, t)
        b_prev[cap <= 0] = 0.0  # lost to churn, exactly as step() does

        arrivals = np.zeros_like(cap)
        edges = {}
        for c, path in enumerate(self._paths):
            if not path:
                continue
            src, dst = np.asarray(path[:-1]), np.asarray(path[1:])
            if self.tcfg.direction == "to_gateway":
                src, dst = dst, src
            edges[c] = (src, dst, path[0] - gw0)
            np.add.at(arrivals, (src, dst), offered[c])

        buffer = self.tcfg.buffer_s * cap
        b = b_prev + arrivals - cap * self.dt
        dropped = np.maximum(b - buffer, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            drop_frac = np.where(arrivals > 0, dropped / arrivals, 0.0)
        served = np.minimum(b_prev + arrivals, cap * self.dt)

        delivered = 0.0
        for c, (src, dst, g) in edges.items():
            d = offered[c] * float(np.prod(1.0 - drop_frac[src, dst]))
            self.gw_delivered[g] += d
            self.gw_routed[g] += offered[c]
            self.city_rows.append((t, c, g, len(src) - 2, offered[c], d))
            delivered += d

        # Cross-checks against TrafficSim's own output for this step.
        gw = slice(gw0, self.m)
        np.testing.assert_allclose(arrivals[gw, :].sum(1) + arrivals[:, gw].sum(0),
                                   res["gateway_load_bits"], rtol=RTOL, atol=1e-3)
        np.testing.assert_allclose(dropped.sum(), res["dropped_bits"], rtol=RTOL, atol=1e-3)
        np.testing.assert_allclose(delivered, res["delivered_bits"], rtol=RTOL, atol=1e-3)
        np.testing.assert_allclose(sum(offered[c] for c in edges), res["routed_bits"], rtol=RTOL, atol=1e-3)

        ii, jj = np.nonzero(np.triu(links.adjacency, k=1))
        view = self.gw_view[t]
        self.link_rows.append(pd.DataFrame({
            "t": t, "i": ii, "j": jj,
            "bits_ij": arrivals[ii, jj], "bits_ji": arrivals[jj, ii],
            "routed_bits": arrivals[ii, jj] + arrivals[jj, ii],
            "served_bits": served[ii, jj] + served[jj, ii],
            "dropped_bits": dropped[ii, jj] + dropped[jj, ii],
            "cap_bits": cap[ii, jj] * self.dt,  # one direction's capacity x dt
            "gw_i": view[ii], "gw_j": view[jj],
        }))
        return res


def main() -> int:
    sim = SimConfig(step_s=STEP_S, selection=SelectionConfig(n_sats=200, n_planes=10, sats_per_plane=20)).resolve(ROOT)
    dataset = run(sim, verbose=True)
    T, step_s = dataset.n_steps, dataset.config.step_s
    rcfg = RoutingConfig(route_metric="hop")
    base = TrafficConfig()
    sites = load_sites(base, ROOT)
    times = dataset_times(dataset)
    city_acc = [SparseAccess.from_access(compute_access(dataset, s, rcfg, times)) for s in sites.cities]
    gw_acc = [gateway_access(dataset, s, rcfg, base, times) for s in sites.gateways]
    ground = predict_ground(sites, city_acc, gw_acc, utc_hours(dataset), dataset.n_sats, base, rcfg)
    gw_view = []
    for t in range(T):
        v = np.zeros(dataset.n_sats, dtype=bool)
        for a in gw_acc:
            v[a.at(t)[0]] = True
        gw_view.append(v)

    cfg = dataclasses.replace(base, peak_total_gbps=LOAD_GBPS, seed=SEED)
    offered = offered_bits(expected_rate_bps(sites, utc_hours(dataset)[:T], cfg), step_s, cfg)
    floor_km = floor_matrix(sites.cities, sites.gateways)

    links, gws, summary, cities = [], [], [], []
    for name, (we, wc) in WEIGHTS.items():
        tcfg = TopologyConfig(w_egress=we, w_congestion=wc)
        ls = Greedy(dataset, tcfg, ground).run(T)
        s = Instrumented(dataset, city_acc, gw_acc, cfg, rcfg, floor_km, gw_view=gw_view)
        res = pd.DataFrame(s.run(ls, offered))
        frac = res["delivered_bits"].sum() / res["routed_bits"].sum()
        print(f"[{name}] {tcfg.variant_label}: delivered {frac:.4f} "
              f"(PHASE2.md {EXPECTED_DELIVERED[name]}), imbalance mean {res['gateway_imbalance'].mean():.3f}")
        assert abs(frac - EXPECTED_DELIVERED[name]) < 5e-5, "does not reproduce PHASE2.md §4.1"
        links.append(pd.concat(s.link_rows).assign(config=name))
        cities.append(pd.DataFrame(s.city_rows, columns=["t", "city", "gateway", "isl_hops", "offered_bits",
                                                         "delivered_bits"]).assign(config=name))
        gws.append(pd.DataFrame({"config": name, "gateway": [g.name for g in sites.gateways],
                                 "routed_bits": s.gw_routed, "delivered_bits": s.gw_delivered}))
        summary.append({"config": name, "topology_variant": tcfg.variant_label, "delivered_frac": frac,
                        "imbalance_mean": res["gateway_imbalance"].mean(),
                        "imbalance_median": res["gateway_imbalance"].median()})

    links = pd.concat(links, ignore_index=True)
    links.to_csv(OUT / "link_steps.csv.gz", index=False)
    pd.concat(gws).to_csv(OUT / "gateways.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    cities = pd.concat(cities, ignore_index=True)
    cities.to_csv(OUT / "city_steps.csv.gz", index=False)
    report(links, pd.concat(gws), cities, [g.name for g in sites.gateways])
    return 0


def describe(x: pd.Series) -> str:
    q = x.quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_numpy() / 1e9
    return (f"n={len(x)} zero={np.mean(x <= 0):.1%} mean={x.mean() / 1e9:.2f} Gb  "
            f"p25/50/75/90/99 = {' / '.join(f'{v:.2f}' for v in q)} Gb")


def report(links: pd.DataFrame, gws: pd.DataFrame, cities: pd.DataFrame, gw_names: list[str]) -> None:
    print("\n== 0. ISL hops per routed bit (0 = city and gateway share a satellite) ==")
    for name, d in cities.groupby("config", sort=False):
        w = d.groupby("isl_hops")["offered_bits"].sum() / d["offered_bits"].sum()
        print(f"[{name}] " + ", ".join(f"{h}: {v:.1%}" for h, v in w.items())
              + f"  | mean {np.average(d['isl_hops'], weights=d['offered_bits']):.3f} hops")
    m = cities[cities.config == "off"].merge(cities[cities.config == "on"], on=["t", "city"],
                                             suffixes=("_off", "_on"))
    moved = m["gateway_off"] != m["gateway_on"]
    print(f"routed (t, city) in both: {len(m)}; gateway changed on {moved.mean():.1%} of them, "
          f"{m.loc[moved, 'offered_bits_off'].sum() / m['offered_bits_off'].sum():.1%} of bits; "
          f"ISL hops changed on {(m['isl_hops_off'] != m['isl_hops_on']).mean():.1%}")
    flows = (m[moved].assign(src=lambda x: x["gateway_off"].map(dict(enumerate(gw_names))),
                             dst=lambda x: x["gateway_on"].map(dict(enumerate(gw_names))))
             .groupby(["src", "dst"])["offered_bits_off"].sum().sort_values(ascending=False) / 1e12)
    print("largest gateway reassignments, off -> on (Tb):")
    print(flows.head(12).to_string(float_format="{:.4f}".format))

    links = links.assign(n_gw=links["gw_i"].astype(int) + links["gw_j"].astype(int),
                         util=links["routed_bits"] / (2 * links["cap_bits"]))
    print("\n== 1. ISL utilisation ==")
    for name, d in links.groupby("config", sort=False):
        print(f"[{name}] per link-step routed bits: {describe(d['routed_bits'])}")
        print(f"[{name}] per link-step utilisation (routed / two-way cap): "
              f"mean {d['util'].mean():.3f}, median {d['util'].median():.3f}, >=0.5: {np.mean(d['util'] >= 0.5):.1%}")
        pair = d.groupby(["i", "j"]).agg(bits=("routed_bits", "sum"), steps=("t", "size"))
        print(f"[{name}] per pair over run: {describe(pair['bits'])}  ({len(pair)} distinct pairs)")
        top = pair["bits"].sort_values(ascending=False)
        print(f"[{name}] share of all ISL bits on top 10% of pairs: "
              f"{top.iloc[:max(1, len(top) // 10)].sum() / top.sum():.1%}; total ISL bits {top.sum() / 1e12:.2f} Tb")

    print("\n== 2. Links added by the terms (link-steps in on, not in off, same t) ==")
    key = ["t", "i", "j"]
    off, on = (links[links.config == c].set_index(key) for c in ("off", "on"))
    added = on.loc[on.index.difference(off.index)]
    removed = off.loc[off.index.difference(on.index)]
    kept_on = on.loc[on.index.intersection(off.index)]
    print(f"on has {len(on)} link-steps, off {len(off)}; added {len(added)} ({len(added) / len(on):.1%} of on), "
          f"removed {len(removed)}")
    for label, d in (("added (on only)", added), ("kept, in on", kept_on), ("all on", on),
                     ("removed (off only)", removed), ("all off", off)):
        print(f"  {label:<20} {describe(d['routed_bits'])}")
    print(f"  added link-steps carry {added['routed_bits'].sum() / on['routed_bits'].sum():.1%} of on's ISL bits; "
          f"removed carried {removed['routed_bits'].sum() / off['routed_bits'].sum():.1%} of off's")
    for label, d in (("added", added), ("removed", removed), ("all on", on), ("all off", off)):
        print(f"  {label:<8} gateway-in-view ends: " + ", ".join(
            f"{k}: {np.mean(d['n_gw'] == k):.1%}" for k in (0, 1, 2)))
    new_pairs = set(zip(added.index.get_level_values("i"), added.index.get_level_values("j"))) - set(
        zip(off.index.get_level_values("i"), off.index.get_level_values("j")))
    print(f"  pairs never present in off: {len(new_pairs)}")

    print("\n== 3. By gateway-in-view ends (0 / 1 / 2 ends see a gateway) ==")
    for name, d in links.groupby("config", sort=False):
        tot = d["routed_bits"].sum()
        for k, g in d.groupby("n_gw"):
            print(f"[{name}] {k} ends: {len(g):6d} link-steps ({len(g) / len(d):.1%}), "
                  f"bits share {g['routed_bits'].sum() / tot:.1%}, {describe(g['routed_bits'])}")

    print("\n== 4. Per-gateway delivered bits (Tb over the run) ==")
    w = gws.pivot(index="gateway", columns="config", values="delivered_bits") / 1e12
    r = gws.pivot(index="gateway", columns="config", values="routed_bits") / 1e12
    t = pd.DataFrame({"routed off": r["off"], "routed on": r["on"],
                      "delivered off": w["off"], "delivered on": w["on"],
                      "delta delivered": w["on"] - w["off"],
                      "delivered/routed off": w["off"] / r["off"], "delivered/routed on": w["on"] / r["on"]})
    t = t.sort_values("delivered off", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(t.to_string())
        print(f"total delivered off {w['off'].sum():.4f}, on {w['on'].sum():.4f}, "
              f"sum |delta| {(w['on'] - w['off']).abs().sum():.4f} Tb")


if __name__ == "__main__":
    raise SystemExit(main())
