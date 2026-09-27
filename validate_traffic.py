"""Stage 4 validation: demand, gateway routing and fluid queues.

    python validate_traffic.py               # 187 sats, full run
    python validate_traffic.py --steps 300   # fast pass; what mutants.py drives

Same rules as the other suites. Every check drives production code; pinned
values are hard-coded literals, not re-derived; and each check that could pass
vacuously has a partner that shows it cannot.

The stage-4 hazard mirrors stage 3's: load routed over the wrong graph. Routing
over visibility instead of the selection, or letting a city relay traffic
between two satellites it sees, both give plausible delays for every selector
and destroy the comparison without failing anything obvious.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from routing import RoutingConfig, compute_access, dataset_times
from routing.paths import shortest_path
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.base import LinkSet
from traffic import (
    SparseAccess,
    TrafficConfig,
    TrafficSim,
    diurnal_factor,
    expected_rate_bps,
    feeder_budget,
    gateway_access,
    load_sites,
    local_solar_hour,
    offered_bits,
)
from validate import Checks, section
from validate_topology import _sym, _synthetic

ROOT = Path(__file__).parent

# --- pinned literals ----------------------------------------------------------
PIN_CITIES_SHA256 = "5b91796b55fa8b57441b6355e6d23f23d2da8054c7a1876f64e38469aab9c0ab"
PIN_N_CITIES = 200
PIN_TROUGH = 0.2
# 3-satellite line, ISLs 1 Gbps, 50 ms buffer, 1 s step, 1.03 Gbps offered:
# excess 3e7 bits per step on each of the two ISLs; the 5e7 buffer overflows
# on the second step by 1e7 per link.
PIN_BACKLOG_STEP0 = 6.0e7
PIN_DROPPED_STEP1 = 2.0e7
PIN_BACKLOG_STEP1 = 1.0e8
PIN_DELIVERED_STEP1 = 1.010097087378640776e9  # 1.03e9 * (1 - 1e7/1.03e9)^2
PIN_PROP_S = 0.024  # 2 x 10 ms ISL + 2 x 2 ms access
PIN_CHURN_LOST = 3.0e7
PIN_DROPPED_5S = 2.0e8
PIN_KRISHNAN8 = ["GW-Los Angeles", "GW-London", "GW-Dubai", "GW-Tokyo", "GW-Singapore",
                 "GW-Sao Paulo", "GW-Johannesburg", "GW-Sydney"]
FEEDER_BW_FACTOR = 10.0


def _sparse(visible_sats: list[list[int]], cap: float, lat: float) -> SparseAccess:
    """A station's access by hand: visible_sats[t] lists the satellites seen."""
    indptr = np.cumsum([0] + [len(v) for v in visible_sats])
    sat = np.array([s for v in visible_sats for s in v], dtype=int)
    return SparseAccess(indptr, sat, np.full(len(sat), cap), np.full(len(sat), lat))


def line_network(n_steps: int, drop_link_12_from: int | None = None):
    """Satellites 0 - 1 - 2 in a line. Gateway sees 0; the city sees 2, or 1
    once link 1-2 is removed. ISLs 1 Gbps / 10 ms; access 4 Gbps / 2 ms, so the
    ISLs are the bottleneck and the access links never queue."""
    n = 3
    cap = _sym(n, {(0, 1): 1.0e9, (1, 2): 1.0e9})
    lat = _sym(n, {(0, 1): 0.010, (1, 2): 0.010})
    vis = _sym(n, {(0, 1): 1, (1, 2): 1}).astype(bool)
    ds = _synthetic(
        capacity=np.repeat(cap[None], n_steps, 0), latency=np.repeat(lat[None], n_steps, 0),
        distance=np.repeat(cap[None], n_steps, 0), visible=np.repeat(vis[None], n_steps, 0),
    )
    links, city_sees = [], []
    for t in range(n_steps):
        adj = vis.copy()
        if drop_link_12_from is not None and t >= drop_link_12_from:
            adj[1, 2] = adj[2, 1] = False
            city_sees.append([1])
        else:
            city_sees.append([2])
        links.append(LinkSet(adj, t))
    city = _sparse(city_sees, 4.0e9, 0.002)
    gw = _sparse([[0]] * n_steps, 4.0e9, 0.002)
    return ds, links, city, gw


def check_demand(chk: Checks, tcfg: TrafficConfig) -> None:
    section("Demand: data, diurnal shape, Poisson arrivals")
    sha = hashlib.sha256((ROOT / tcfg.cities_path).read_bytes()).hexdigest()
    chk.record("cities.csv is the pinned file", sha == PIN_CITIES_SHA256, f"sha256 {sha}")

    sites = load_sites(tcfg, ROOT)
    chk.record(
        f"{PIN_N_CITIES} cities, {tcfg.n_gateways} gateways in distinct countries",
        len(sites.cities) == PIN_N_CITIES and len(set(sites.gateway_country)) == tcfg.n_gateways
        and len(sites.gateways) == tcfg.n_gateways,
        f"{len(sites.cities)} cities; gateway countries {sites.gateway_country}",
    )

    lon = np.array([0.0, 90.0, -180.0])
    h = local_solar_hour(np.array([12.0]), lon)[0]
    chk.record(
        "local solar hour is UTC + longitude/15, wrapped to [0, 24)",
        np.allclose(h, [12.0, 18.0, 0.0]),
        f"UTC 12:00 at lon 0/90/-180 -> {h.tolist()}",
    )
    peak = diurnal_factor(np.array([tcfg.diurnal_peak_hour]), tcfg)[0]
    low = diurnal_factor(np.array([tcfg.diurnal_peak_hour - 12.0]), tcfg)[0]
    grid = diurnal_factor(np.linspace(0, 24, 97), tcfg)
    chk.record(
        "diurnal factor is 1 at the peak hour and the pinned trough 12 h away",
        abs(peak - 1.0) < 1e-12 and abs(low - PIN_TROUGH) < 1e-12
        and grid.min() >= PIN_TROUGH - 1e-12 and grid.max() <= 1.0 + 1e-12,
        f"peak {peak:.6f}, trough {low:.6f} (pinned {PIN_TROUGH}), range [{grid.min():.4f}, {grid.max():.4f}]",
    )

    flat = dataclasses.replace(tcfg, diurnal_trough_fraction=1.0)
    total = expected_rate_bps(sites, np.array([3.0]), flat).sum()
    chk.close_to("with a flat day, expected total equals peak_total_gbps",
                 total, tcfg.peak_total_gbps * 1e9, 1e-12, " bit/s")

    rate = expected_rate_bps(sites, np.full(400, 3.0), tcfg)
    step = 5.0
    means = []
    for flow in (tcfg.mean_flow_bits, tcfg.mean_flow_bits * 20):
        got = offered_bits(rate, step, dataclasses.replace(tcfg, mean_flow_bits=flow)).sum()
        means.append(got / (rate.sum() * step))
    chk.record(
        "Poisson arrivals average rate x step, independent of flow size",
        all(abs(m - 1.0) < 0.01 for m in means),
        f"realised / expected bits: {[round(m, 5) for m in means]} "
        f"(flow sizes 1x and 20x; 400 steps of {step:g} s)",
    )
    one = offered_bits(rate[:1], step, tcfg)
    chk.record(
        "arrivals are random, not the expectation (Poisson check is not vacuous)",
        not np.allclose(one, rate[:1] * step),
        f"first-step spread vs expectation: max |diff| {np.abs(one - rate[:1] * step).max():.3g} bits",
    )


def check_queues(chk: Checks, rcfg: RoutingConfig) -> None:
    section("Fluid queues: closed forms on a 3-satellite line")
    tcfg = TrafficConfig(peak_total_gbps=0.0)

    ds, links, city, gw = line_network(2)
    sim = TrafficSim(ds, [city], [gw], tcfg, rcfg)
    under = sim.run(links, np.array([[5.0e8], [5.0e8]]))
    chk.record(
        "under capacity: no backlog, no drops, everything delivered",
        all(r["backlog_bits"] == 0 and r["dropped_bits"] == 0 and r["delivered_bits"] == 5.0e8
            for r in under),
        f"backlog {[r['backlog_bits'] for r in under]}, dropped {[r['dropped_bits'] for r in under]}, "
        f"delivered {[r['delivered_bits'] for r in under]}",
    )
    chk.close_to("under capacity, delay is propagation alone", under[0]["delay_mean_ms"] / 1e3,
                 PIN_PROP_S, 1e-9, " s")

    ds, links, city, gw = line_network(2)
    sim = TrafficSim(ds, [city], [gw], tcfg, rcfg)
    over = sim.run(links, np.array([[1.03e9], [1.03e9]]))
    chk.close_to("overload step 0: excess queues on both ISLs", over[0]["backlog_bits"],
                 PIN_BACKLOG_STEP0, 1e-9, " bits")
    chk.record("overload step 0: buffer not yet full, nothing dropped", over[0]["dropped_bits"] == 0,
               f"dropped {over[0]['dropped_bits']}")
    chk.close_to("overload step 1: buffer overflows by the pinned amount", over[1]["dropped_bits"],
                 PIN_DROPPED_STEP1, 1e-9, " bits")
    chk.close_to("overload step 1: backlog clipped at the buffer", over[1]["backlog_bits"],
                 PIN_BACKLOG_STEP1, 1e-9, " bits")
    chk.close_to("overload step 1: delivered is offered thinned by each link's drop fraction",
                 over[1]["delivered_bits"], PIN_DELIVERED_STEP1, 1e-9, " bits")
    chk.record(
        "overload adds queueing delay on top of propagation",
        over[1]["delay_mean_ms"] / 1e3 > PIN_PROP_S + 0.09,
        f"delay {over[1]['delay_mean_ms']:.3f} ms vs propagation {PIN_PROP_S * 1e3:.1f} ms "
        "(two full 50 ms buffers add 100 ms)",
    )

    # The two cases above use a 1 s step, where "capacity x dt" and "capacity"
    # are the same number. At 5 s the same 1.03 Gbps rate is 5.15e9 bits per
    # step: 1.5e8 excess per ISL against a 5e7 buffer, so 1e8 dropped per link.
    ds, links, city, gw = line_network(1)
    ds = dataclasses.replace(ds, config=SimConfig(step_s=5.0, duration_s=5.0))
    five = TrafficSim(ds, [city], [gw], tcfg, rcfg).run(links, np.array([[5.15e9]]))
    chk.close_to("queue drains capacity x step, not capacity (5 s step)", five[0]["dropped_bits"],
                 PIN_DROPPED_5S, 1e-9, " bits")

    ds, links, city, gw = line_network(2, drop_link_12_from=1)
    sim = TrafficSim(ds, [city], [gw], tcfg, rcfg)
    cut = sim.run(links, np.array([[1.03e9], [1.03e9]]))
    chk.close_to("a removed link loses its backlog to churn", cut[1]["churn_lost_bits"],
                 PIN_CHURN_LOST, 1e-9, " bits")
    chk.record(
        "churn loss discriminates: nothing is lost when no link is removed",
        over[1]["churn_lost_bits"] == 0 and cut[1]["routed_bits"] > 0,
        f"churn lost without removal {over[1]['churn_lost_bits']}; "
        f"with removal the city still routes via sat 1 ({cut[1]['routed_bits']:.3g} bits)",
    )

    ds2, links2, city2, gw2 = line_network(1)
    fwd = TrafficSim(ds2, [city2], [gw2], tcfg, rcfg)
    rev = TrafficSim(ds2, [city2], [gw2], dataclasses.replace(tcfg, direction="to_gateway"), rcfg)
    fwd.step(links2[0], 0, np.array([1.03e9]))
    rev.step(links2[0], 0, np.array([1.03e9]))
    n = 3
    gw_node, city_node = n + 1, n
    chk.record(
        "direction flag decides which way the queues fill",
        fwd.backlog[0, 1] > 0 and fwd.backlog[1, 0] == 0
        and rev.backlog[1, 0] > 0 and rev.backlog[0, 1] == 0,
        f"to_user: backlog 0->1 {fwd.backlog[0, 1]:.3g}, 1->0 {fwd.backlog[1, 0]:.3g}; "
        f"to_gateway: 0->1 {rev.backlog[0, 1]:.3g}, 1->0 {rev.backlog[1, 0]:.3g} "
        f"(gateway node {gw_node}, city node {city_node})",
    )


def check_no_ground_relay(chk: Checks, rcfg: RoutingConfig) -> None:
    """A city must not relay traffic between two satellites it sees.

    Two ISL islands, {0, 1} with the gateway on 0 and {2, 3}. City A sees only
    3; city B sees 1 and 2. If cities were graph nodes, A would reach the
    gateway through B (3-2-B-1-0). It must have no path at all.
    """
    section("Routing: ground stations are never relays")
    n = 4
    cap = _sym(n, {(0, 1): 1e9, (2, 3): 1e9})
    vis = cap > 0
    ds = _synthetic(capacity=cap[None], latency=(cap[None] > 0) * 0.01, distance=cap[None], visible=vis[None])
    a = _sparse([[3]], 4e9, 0.002)
    b = _sparse([[1, 2]], 4e9, 0.002)
    gw = _sparse([[0]], 4e9, 0.002)
    sim = TrafficSim(ds, [a, b], [gw], TrafficConfig(), rcfg)
    paths = sim.routes(LinkSet(vis, 0), 0)
    chk.record(
        "a city stranded on an island has no path, even via another city",
        paths[0] == [] and paths[1] is not None and len(paths[1]) == 4,
        f"city A path {paths[0]}; city B path {paths[1]} (gateway -> 0 -> 1 -> B)",
    )


def check_gateways(chk: Checks, dataset, rcfg: RoutingConfig) -> None:
    """Gateway sites and the feeder link budget.

    The feeder override changes only capacity, and Shannon fixes how: at the same
    distance, SNR scales as 1/B, so C' = kB log2(1 + SNR/k) with SNR recovered
    from the default capacity as 2^(C/B) - 1. That relation is checked on every
    real gateway link, not re-derived through the code under test.
    """
    section("Gateways: sites and feeder link budget")
    k8 = load_sites(TrafficConfig(gateway_sites="krishnan8"), ROOT)
    chk.record("krishnan8 is the eight sites named in Krishnan et al. Fig. 1",
               [g.name for g in k8.gateways] == PIN_KRISHNAN8,
               f"{[g.name for g in k8.gateways]}")
    p20 = load_sites(TrafficConfig(), ROOT)
    chk.record("gateway-sites flag changes the gateway set",
               {g.name for g in p20.gateways} != {g.name for g in k8.gateways},
               f"populous20 has {len(p20.gateways)}, krishnan8 has {len(k8.gateways)}")

    times = dataset_times(dataset)
    base = TrafficConfig()
    wide = TrafficConfig(feeder_bandwidth_hz=feeder_budget(dataset, rcfg, base).bandwidth_hz * FEEDER_BW_FACTOR)
    worst_default = worst_wide = 0.0
    links = 0
    for g in p20.gateways:
        ref = SparseAccess.from_access(compute_access(dataset, g, rcfg, times))
        d = gateway_access(dataset, g, rcfg, base, times)
        w = gateway_access(dataset, g, rcfg, wide, times)
        if len(ref.capacity):
            worst_default = max(worst_default, float(np.abs(d.capacity - ref.capacity).max()))
            b = feeder_budget(dataset, rcfg, base).bandwidth_hz
            snr = 2.0 ** (ref.capacity / b) - 1.0
            want = FEEDER_BW_FACTOR * b * np.log2(1.0 + snr / FEEDER_BW_FACTOR)
            worst_wide = max(worst_wide, float(np.abs(w.capacity / want - 1.0).max()))
            links += len(ref.capacity)
    chk.record("with no overrides, gateway capacity is exactly stage 3's",
               worst_default == 0.0 and links > 0, f"{links} gateway links, max |diff| {worst_default:g} bit/s")
    chk.record(f"feeder bandwidth x{FEEDER_BW_FACTOR:g} follows Shannon scaling",
               worst_wide < 1e-9 and links > 0,
               f"{links} gateway links, worst relative error {worst_wide:.2e} against kB log2(1 + SNR/k)")


def check_real_run(chk: Checks, dataset, results: dict, sites, city_acc, gw_acc, tcfg, rcfg, n_steps) -> None:
    section("Real run: conservation, graph, gateway choice")
    hours = np.full(n_steps, 12.0)
    offered = offered_bits(expected_rate_bps(sites, hours, tcfg), dataset.config.step_s, tcfg)
    out = {}
    for name, ls in results.items():
        out[name] = TrafficSim(dataset, city_acc, gw_acc, tcfg, rcfg).run(ls[:n_steps], offered)

    worst = 0.0
    for rows in out.values():
        for r in rows:
            parts = r["no_access_bits"] + r["no_path_bits"] + r["routed_bits"]
            worst = max(worst, abs(parts - r["offered_bits"]) / max(r["offered_bits"], 1.0))
    chk.record("offered = no-access + no-path + routed, every step, every selector",
               worst < 1e-12, f"worst relative imbalance {worst:.2e}")
    chk.record(
        "delivered never exceeds routed",
        all(r["delivered_bits"] <= r["routed_bits"] * (1 + 1e-12) for rows in out.values() for r in rows),
        "",
    )
    na = {k: round(sum(r["no_access_bits"] for r in v), 3) for k, v in out.items()}
    chk.record("no-access demand is identical for every selector (a ground-segment property)",
               len(set(na.values())) == 1, f"{na}")
    routed = sum(r["routed_bits"] for r in out["Greedy"])
    total = sum(r["offered_bits"] for r in out["Greedy"])
    chk.record("the run is not vacuous: some demand is routed, some is not",
               0 < routed < total, f"Greedy routed {routed / total:.1%} of offered")

    # Every ISL on every path is a selected link, never merely visible.
    sim = TrafficSim(dataset, city_acc, gw_acc, tcfg, rcfg)
    n, bad, checked, shorter = dataset.n_sats, 0, 0, 0
    ts = np.unique(np.linspace(0, n_steps - 1, 6).astype(int))
    greedy = results["Greedy"]
    for t in ts:
        paths = sim.routes(greedy[t], t)
        vis_paths = sim.routes(LinkSet(np.asarray(dataset.visible[t]), t), t)
        for p, pv in zip(paths, vis_paths):
            if not p:
                continue
            for u, v in zip(p[:-1], p[1:]):
                if u < n and v < n:
                    checked += 1
                    bad += not greedy[t].adjacency[u, v]
            if pv and len(pv) < len(p):
                shorter += 1
    chk.record("every ISL on a routed path is a selected link", bad == 0 and checked > 0,
               f"{checked} ISL hops checked, {bad} not in the selection")
    chk.record(
        "graph check discriminates: routing over visibility would give shorter paths",
        shorter > 0,
        f"{shorter} city-timesteps where the visibility graph is strictly shorter",
    )

    # Anycast: the chosen gateway is the cheapest, cross-checked against stage 3's
    # own two-station shortest path (independent code, undirected graph).
    mism = compared = 0
    gw_dense = [_dense(a, dataset) for a in gw_acc]
    picked = [int(ts[len(ts) // 3]), int(ts[2 * len(ts) // 3])]
    for t in picked:
        paths = sim.routes(greedy[t], t)
        for c, path in enumerate(paths):
            if not path:
                continue
            src = _dense(city_acc[c], dataset)
            best = np.inf
            for g in gw_dense:
                r = shortest_path(dataset, greedy[t], {"src": src, "dst": g}, t, rcfg)
                if r["reachable"]:
                    best = min(best, r["cost"])
            compared += 1
            mism += abs((len(path) - 1) - best) > 1e-9
    chk.record("each city egresses at its cheapest gateway (matches stage-3 routing)",
               mism == 0 and compared >= 20,
               f"t={picked}: {compared} city-timesteps compared, {mism} mismatches")


def _dense(sp: SparseAccess, dataset):
    """Rebuild a stage-3 GroundAccess from the sparse form, for cross-checking."""
    from routing.stations import GroundAccess, GroundStation
    t_n, n = dataset.n_steps, dataset.n_sats
    vis = np.zeros((t_n, n), dtype=bool)
    cap = np.zeros((t_n, n))
    lat = np.zeros((t_n, n))
    for t in range(t_n):
        s, c, l = sp.at(t)
        vis[t, s], cap[t, s], lat[t, s] = True, c, l
    return GroundAccess(GroundStation("x", 0.0, 0.0), vis, np.zeros((t_n, n)), np.zeros((t_n, n)), cap, lat)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--step", type=float, default=SimConfig.step_s)
    p.add_argument("--grid-gaps", choices=["strict", "bridge"], default="strict")
    args = p.parse_args()

    tcfg_traffic = TrafficConfig()
    rcfg = RoutingConfig()
    print("=" * 72)
    print("  LEO CONSTELLATION SIMULATOR -- STAGE 4 VALIDATION")
    print(f"  variant: {tcfg_traffic.variant_label} | {rcfg.variant_label}")
    print("=" * 72)

    chk = Checks()
    check_demand(chk, tcfg_traffic)
    check_queues(chk, rcfg)
    check_no_ground_relay(chk, rcfg)

    cfg = SimConfig(
        step_s=args.step,
        selection=SelectionConfig(n_sats=args.planes * args.per_plane, n_planes=args.planes,
                                  sats_per_plane=args.per_plane),
    ).resolve(ROOT)
    dataset = run(cfg, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)
    tcfg = TopologyConfig(grid_gaps=args.grid_gaps)
    results = {
        "+Grid": PlusGrid(dataset, tcfg).run(n_steps),
        "Greedy": Greedy(dataset, tcfg).run(n_steps),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")).run(n_steps),
    }
    sites = load_sites(tcfg_traffic, ROOT)
    times = dataset_times(dataset)
    city_acc = [SparseAccess.from_access(compute_access(dataset, s, rcfg, times)) for s in sites.cities]
    gw_acc = [gateway_access(dataset, s, rcfg, tcfg_traffic, times) for s in sites.gateways]
    check_gateways(chk, dataset, rcfg)
    check_real_run(chk, dataset, results, sites, city_acc, gw_acc, tcfg_traffic, rcfg, n_steps)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
