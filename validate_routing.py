"""Stage-3 validation suite: ground stations, access and routing.

Same standard as the other two suites: every check prints PASS/FAIL with the
measured value and the bound, every check drives the production code rather
than recomputing the quantity under test locally, and checks that exist only to
stop another passing vacuously say so.

    python validate_routing.py              # full 30-minute run
    python validate_routing.py --steps 300  # faster; used by mutants.py

The stage-3 hazard is different from stage 1's. There the errors were silent
wrong physics. Here the dangerous failure is a path that is real but routed
over the wrong graph -- over every visible link instead of the links the
selector actually chose. That produces plausible hop counts, identical for all
four algorithms, and destroys the entire comparison. It gets its own check.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from leosim.config import SPEED_OF_LIGHT_M_S, SelectionConfig, SimConfig
from leosim.dataset import LinkDataset
from leosim.simulation import run
from routing import (
    CITIES,
    NO_PATH,
    PAPER_PAIRS,
    chord_km,
    great_circle_km,
    RoutingConfig,
    compute_access,
    dataset_times,
    edge_costs,
    route_run,
    shortest_path,
)
from routing.stations import station_geometry
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from validate import Checks, section

ROOT = Path(__file__).parent

# Sidereal day: the period of Earth's rotation against the inertial frame, not
# the 86400 s solar day. A ground station's GCRS position must sweep 360 deg in
# exactly this, and the 236 s difference is far larger than the tolerance, so
# using the wrong one fails.
SIDEREAL_DAY_S = 86164.0905

# WGS84 geocentric radius runs from the polar semi-minor axis to the equatorial
# semi-major axis. A station outside this band is not on the ellipsoid.
WGS84_POLAR_KM = 6356.752
WGS84_EQUATORIAL_KM = 6378.137

# Maximum angle between the geodetic vertical and the geocentric radial, which
# is what makes an oblate-Earth vertical worth computing. Peaks near 45 deg
# latitude at 0.192 deg for WGS84's flattening.
MAX_VERTICAL_DEFLECTION_DEG = 0.20

# The deflection itself, per station, hard-coded rather than recomputed: a
# reference derived from the same code each run would drift with it.
PIN_VERTICAL_DEFLECTION_DEG = {
    "New York": 0.1901766,
    "Singapore": 0.0090481,
}

PIN_REL_TOL = 1e-12


def _pair_access(dataset, rcfg, times, pair=("New York", "San Francisco")):
    return {city: compute_access(dataset, CITIES[city], rcfg, times) for city in pair}


# --- check groups -----------------------------------------------------------


def check_station_geometry(chk: Checks, dataset: LinkDataset, times) -> None:
    section("Ground station geometry")

    for name in ("New York", "Singapore"):
        pos, up = station_geometry(CITIES[name], times)
        radius = np.linalg.norm(pos, axis=1)
        chk.between(
            f"{name}: geocentric radius lies on the WGS84 ellipsoid",
            float(radius.mean()), WGS84_POLAR_KM, WGS84_EQUATORIAL_KM, unit=" km",
            extra=f"range over the run {radius.min():.3f}..{radius.max():.3f} km",
        )
        chk.record(
            f"{name}: local vertical is a unit vector",
            np.allclose(np.linalg.norm(up, axis=1), 1.0, rtol=0, atol=1e-12),
            f"max |norm(up) - 1| = {float(np.abs(np.linalg.norm(up, axis=1) - 1).max()):.3e}",
        )
        radial = pos / radius[:, None]
        deflection = np.degrees(
            np.arccos(np.clip(np.einsum("ij,ij->i", up, radial), -1.0, 1.0))
        )
        # Pinned, not merely bounded. An upper bound alone is satisfied by
        # exactly 0, which is what using the geocentric radial as the vertical
        # gives -- and that is the mistake this check exists to catch. The
        # literals are the WGS84 deflection at each latitude, hard-coded.
        chk.close_to(
            f"{name}: geodetic vertical departs from radial by the WGS84 amount",
            float(deflection.max()), PIN_VERTICAL_DEFLECTION_DEG[name], 1e-4, unit=" deg",
            extra=f"latitude {CITIES[name].latitude_deg:.1f} deg; a spherical Earth "
                  f"would give exactly 0, and the bound alone would accept it",
        )

    # The station must rotate with the Earth *in the inertial frame*. A station
    # computed in ECEF and used as if it were inertial would not move at all,
    # and a solar day instead of a sidereal one would be off by 1 part in 365.
    #
    # Measured about the axis recovered from the trajectory, not about the GCRS
    # z axis. They are not the same: GCRS is referred to J2000 and the pole has
    # precessed ~0.15 deg since, which biases a z-axis measurement by an amount
    # that grows with the station's latitude -- 7.5263 deg at New York against
    # 7.5161 at Singapore, for one shared rotation. Three points on a circle
    # determine its plane exactly, so the axis is taken from the data.
    for name in ("New York", "Singapore", "Sydney"):
        pos, _ = station_geometry(CITIES[name], times)
        mid = len(pos) // 2
        axis = np.cross(pos[mid] - pos[0], pos[-1] - pos[0])
        axis /= np.linalg.norm(axis)
        if axis[2] < 0:
            axis = -axis
        centre = (pos[0] @ axis) * axis
        v0, v1 = pos[0] - centre, pos[-1] - centre
        swept = np.degrees(
            np.arctan2(np.cross(v0, v1) @ axis, v0 @ v1)
        ) % 360.0

        elapsed = float(np.asarray(dataset.time_offsets_s)[-1])
        expected = elapsed * 360.0 / SIDEREAL_DAY_S
        chk.close_to(
            f"{name}: sweeps 360 deg per sidereal day in the inertial frame",
            float(swept), expected, 1e-6, unit=" deg",
            extra=f"{elapsed:.0f} s elapsed; a solar day would give "
                  f"{elapsed * 360.0 / 86400.0:.6f} deg, a fixed station 0",
        )

    tilt = np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    chk.between(
        "rotation axis is tilted from the GCRS z axis by a precession-sized angle",
        float(tilt), 0.01, 0.5, unit=" deg",
        extra="GCRS is referred to J2000; exactly 0 would mean the station is not "
              "being placed in a precessing inertial frame at all",
    )


def check_access(chk: Checks, dataset: LinkDataset, rcfg: RoutingConfig, times) -> None:
    section("Ground-satellite access")
    station = CITIES["San Francisco"]
    acc = compute_access(dataset, station, rcfg, times)

    # Recompute elevation independently, from the spherical-Earth radial rather
    # than the production geodetic vertical, and require agreement to within
    # the oblateness deflection. Identical maths would prove nothing.
    gs, _ = station_geometry(station, times)
    los = np.asarray(dataset.positions, dtype=np.float64) - gs[:, None, :]
    rng = np.linalg.norm(los, axis=2)
    radial = gs / np.linalg.norm(gs, axis=1, keepdims=True)
    elev_sphere = np.degrees(
        np.arcsin(np.clip(np.einsum("tnc,tc->tn", los, radial) / rng, -1.0, 1.0))
    )
    gap = float(np.abs(elev_sphere - acc.elevation_deg).max())
    chk.between(
        "elevation agrees with a spherical-Earth recomputation to oblateness size",
        gap, 0.0, MAX_VERTICAL_DEFLECTION_DEG, unit=" deg",
        extra="a larger gap means the vertical is wrong, not merely oblate",
    )

    below = int((acc.elevation_deg[acc.visible] < rcfg.min_elevation_deg).sum())
    chk.record(
        "every access link is above the elevation mask",
        below == 0,
        f"{below} of {int(acc.visible.sum())} access links below "
        f"{rcfg.min_elevation_deg:g} deg",
    )
    chk.record(
        "access check is not vacuous: the station does reach satellites",
        acc.visible.any() and acc.n_visible.mean() > 0.5,
        f"mean {acc.n_visible.mean():.2f} satellites in view, "
        f"max {int(acc.n_visible.max())}",
    )

    # Raising the mask can only remove links. A mask that was ignored entirely
    # would leave the count unchanged, which this catches.
    strict = compute_access(
        dataset, station, dataclasses.replace(rcfg, min_elevation_deg=60.0), times
    )
    chk.record(
        "a stricter mask is a strict subset of a looser one",
        bool((strict.visible & ~acc.visible).sum() == 0)
        and int(strict.visible.sum()) < int(acc.visible.sum()),
        f"{int(acc.visible.sum())} links at {rcfg.min_elevation_deg:g} deg, "
        f"{int(strict.visible.sum())} at 60 deg, "
        f"{int((strict.visible & ~acc.visible).sum())} present only in the stricter set",
    )

    # Slant range is bounded by the geometry of the mask: with the satellite at
    # altitude h and the station on a sphere of radius R, the range at
    # elevation e is R*(sqrt(((R+h)/R)^2 - cos^2 e) - sin e).
    alt = float(np.asarray(dataset.altitudes).max())
    r = WGS84_EQUATORIAL_KM
    e = np.radians(rcfg.min_elevation_deg)
    bound = r * (np.sqrt(((r + alt) / r) ** 2 - np.cos(e) ** 2) - np.sin(e))
    worst = float(acc.distance_km[acc.visible].max())
    chk.between(
        "slant range of every access link is within the mask's geometric bound",
        worst, 0.0, bound, unit=" km",
        extra=f"bound from max altitude {alt:.1f} km at {rcfg.min_elevation_deg:g} deg elevation",
    )

    # max_access_links must actually cap, and keep the best links.
    capped = compute_access(
        dataset, station, dataclasses.replace(rcfg, max_access_links=1), times
    )
    over = int((capped.visible.sum(axis=1) > 1).sum())
    busy = acc.visible.sum(axis=1) > 1
    kept_best = True
    for t in np.nonzero(busy)[0][:50]:
        if capped.visible[t].any():
            best = int(np.argmax(np.where(acc.visible[t], acc.elevation_deg[t], -np.inf)))
            kept_best &= bool(capped.visible[t, best])
    chk.record(
        "max_access_links caps the count and keeps the highest-elevation link",
        over == 0 and kept_best and int(busy.sum()) > 0,
        f"{over} timesteps above the cap of 1; "
        f"{int(busy.sum())} timesteps had more than one candidate",
    )


def check_graph(
    chk: Checks, dataset: LinkDataset, results: dict, access: dict, rcfg: RoutingConfig
) -> None:
    section("Routing graph is built on the selected topology")
    probes = np.unique(np.linspace(0, min(len(results["Greedy"]), 300) - 1, 12).astype(int))
    n = dataset.n_sats

    # THE stage-3 check. Routing over the visibility set instead of the chosen
    # links would give plausible hop counts that are identical for every
    # algorithm and would silently answer a question nobody asked.
    for name, linksets in results.items():
        bad = 0
        for t in probes:
            graph, _ = edge_costs(dataset, linksets[t], access, int(t), rcfg)
            sub = graph[:n, :n].tocoo()
            got = {(min(i, j), max(i, j)) for i, j in zip(sub.row, sub.col)}
            want = {(int(i), int(j)) for i, j in linksets[t].edges}
            if got != want:
                bad += 1
        chk.record(
            f"{name}: ISL edges in the graph are exactly the selected links",
            bad == 0,
            f"{len(probes)} timesteps checked, {bad} mismatched against "
            f"LinkSet.edges (visibility at t=0 has "
            f"{int(np.triu(np.asarray(dataset.visible[0]), 1).sum())} pairs, the "
            f"selection has {results[name][0].n_links})",
        )

    graph, names = edge_costs(dataset, results["Greedy"][0], access, 0, rcfg)
    chk.record(
        "graph is symmetric and labelled satellites-then-stations",
        (abs(graph - graph.T) > 0).nnz == 0
        and len(names) == n + 2
        and names[n:] == list(access),
        f"{graph.shape[0]} nodes, asymmetric entries "
        f"{(abs(graph - graph.T) > 0).nnz}, last two labels {names[n:]}",
    )

    nz = graph.tocoo().data
    chk.record(
        "every edge cost is strictly positive (a zero weight would vanish)",
        nz.size > 0 and float(nz.min()) > 0.0,
        f"{nz.size} directed entries, smallest {float(nz.min()):.6g}",
    )

    hop_graph, _ = edge_costs(
        dataset, results["Greedy"][0], access,
        0, dataclasses.replace(rcfg, route_metric="hop"),
    )
    lat_graph, _ = edge_costs(
        dataset, results["Greedy"][0], access,
        0, dataclasses.replace(rcfg, route_metric="latency"),
    )
    chk.record(
        "hop metric weights every edge 1; latency metric does not",
        bool(np.allclose(hop_graph.tocoo().data, 1.0))
        and not bool(np.allclose(lat_graph.tocoo().data, 1.0)),
        f"hop costs unique values {np.unique(hop_graph.tocoo().data)[:3]}, "
        f"latency costs range {lat_graph.tocoo().data.min():.3e}..{lat_graph.tocoo().data.max():.3e} s",
    )


def check_paths(
    chk: Checks, dataset: LinkDataset, results: dict, access: dict, rcfg: RoutingConfig
) -> None:
    section("Paths are real, and are the shortest")
    n = dataset.n_sats
    linksets = results["DoTD-forward"]
    probes = np.unique(np.linspace(0, min(len(linksets), 300) - 1, 20).astype(int))

    broken = latency_bad = hop_bad = ends_bad = 0
    checked = 0
    for t in probes:
        res = shortest_path(dataset, linksets[int(t)], access, int(t), rcfg)
        if not res["reachable"]:
            continue
        checked += 1
        path = res["path"]
        if path[0] != n or path[-1] != n + 1:
            ends_bad += 1
        # Every consecutive pair must be a real edge of the graph it came from.
        graph, _ = edge_costs(dataset, linksets[int(t)], access, int(t), rcfg)
        dense = graph.toarray()
        if any(dense[a, b] == 0 for a, b in zip(path[:-1], path[1:])):
            broken += 1
        if res["hops"] != len(path) - 1:
            hop_bad += 1
        # Latency recomputed from stage-1 arrays and the access arrays directly.
        total = 0.0
        stations = list(access.values())
        for a, b in zip(path[:-1], path[1:]):
            if a < n and b < n:
                total += float(dataset.latency[int(t), a, b])
            else:
                g, s = (a, b) if a >= n else (b, a)
                total += float(stations[g - n].latency[int(t), s])
        if abs(total * 1e3 - res["latency_ms"]) > 1e-9:
            latency_bad += 1

    chk.record(
        "returned paths are connected walks from source to destination",
        broken == 0 and ends_bad == 0 and checked > 0,
        f"{checked} paths checked: {broken} with a non-edge step, "
        f"{ends_bad} with wrong endpoints",
    )
    chk.record(
        "hop count equals the number of edges in the returned path",
        hop_bad == 0,
        f"{hop_bad} of {checked} paths disagree with len(path) - 1",
    )
    chk.record(
        "path latency equals the sum of its edges' stage-1 latencies",
        latency_bad == 0,
        f"{latency_bad} of {checked} paths disagree with independent summation",
    )

    # Optimality, against a Dijkstra run that does not share the production
    # call: same graph, distance only, no predecessor walk.
    worst = 0.0
    for t in probes:
        res = shortest_path(dataset, linksets[int(t)], access, int(t), rcfg)
        if not res["reachable"]:
            continue
        graph, _ = edge_costs(dataset, linksets[int(t)], access, int(t), rcfg)
        d = dijkstra(graph, indices=n, directed=False)[n + 1]
        worst = max(worst, abs(d - res["cost"]))
    chk.record(
        "path cost equals the independently computed shortest-path distance",
        worst < 1e-9,
        f"largest disagreement {worst:.3e} over {len(probes)} timesteps",
    )

    # The hop metric must minimise hops. Any other metric can only do as well
    # or worse on that axis -- if latency ever beat it, the hop search is wrong.
    hop_cfg = dataclasses.replace(rcfg, route_metric="hop")
    lat_cfg = dataclasses.replace(rcfg, route_metric="latency")
    worse = 0
    differ = 0
    for t in probes:
        a = shortest_path(dataset, linksets[int(t)], access, int(t), hop_cfg)
        b = shortest_path(dataset, linksets[int(t)], access, int(t), lat_cfg)
        if not (a["reachable"] and b["reachable"]):
            continue
        if a["hops"] > b["hops"]:
            worse += 1
        if a["hops"] != b["hops"] or a["latency_ms"] != b["latency_ms"]:
            differ += 1
    chk.record(
        "hop-metric paths are never longer in hops than latency-metric paths",
        worse == 0,
        f"{worse} of {len(probes)} timesteps where minimising hops gave more hops",
    )
    chk.record(
        "metric choice check is not vacuous: the two metrics do differ",
        differ > 0,
        f"{differ} of {len(probes)} timesteps route differently under hop vs latency",
    )

    # Unreachability must be reported, not faked. A mask no satellite can meet
    # removes the ground segment entirely.
    times = dataset_times(dataset)
    blind = dataclasses.replace(rcfg, min_elevation_deg=89.99)
    blind_access = _pair_access(dataset, blind, times)
    res = shortest_path(dataset, linksets[0], blind_access, 0, blind)
    # The hop count matters as much as the flag: a zero there is a number that
    # averages into the summary and pulls every mean down, whereas the sentinel
    # is excluded. Both are pinned.
    chk.record(
        "no path is reported as unreachable rather than as a zero-hop path",
        res["reachable"] is False
        and res["hops"] == NO_PATH
        and res["isl_hops"] == NO_PATH
        and np.isnan(res["latency_ms"]),
        f"at an 89.99 deg mask: reachable={res['reachable']}, hops={res['hops']} "
        f"(sentinel {NO_PATH}), isl_hops={res['isl_hops']}, "
        f"latency_ms={res['latency_ms']}, access counts "
        f"{[int(a.n_visible[0]) for a in blind_access.values()]}",
    )


def check_latency_floor(
    chk: Checks, dataset: LinkDataset, linksets, rcfg: RoutingConfig, times, n_steps: int
) -> None:
    """No path may be faster than light on the straight line between the cities.

    The bound is the CHORD through the Earth, not the great-circle arc. Any
    route from A to B has length at least |A - B| by the triangle inequality,
    whatever it does in between; the arc is longer than the chord and therefore
    cannot bound a path from below. For the five short Table II pairs the arc
    happens to hold too -- climbing to ~580 km costs far more than the 30-80 km
    by which arc exceeds chord at these separations -- and that is asserted
    separately, explicitly scoped. It is NOT a general law: at a large enough
    separation a space path beats the ground arc traversed at c, which is the
    mechanism behind the paper's "47% faster than fibre" claim, and asserting
    the arc globally would amount to asserting that claim is impossible.
    """
    section("Reported latency is physically possible")
    c_km_s = SPEED_OF_LIGHT_M_S / 1000.0

    worst_chord = worst_arc = -np.inf
    tested = 0
    chord_viol = arc_viol = 0
    detail_rows = []
    for tag, a, b in PAPER_PAIRS:
        rows = route_run(
            dataset, linksets,
            {a: compute_access(dataset, CITIES[a], rcfg, times),
             b: compute_access(dataset, CITIES[b], rcfg, times)},
            rcfg, n_steps,
        )
        live = [r for r in rows if r["reachable"]]
        if not live:
            detail_rows.append(f"{tag} {a}->{b}: no path at any of {n_steps} steps")
            continue
        floor_chord_ms = chord_km(CITIES[a], CITIES[b]) / c_km_s * 1e3
        floor_arc_ms = great_circle_km(CITIES[a], CITIES[b]) / c_km_s * 1e3
        got = np.array([r["latency_ms"] for r in live])
        tested += len(live)
        chord_viol += int((got < floor_chord_ms - 1e-9).sum())
        arc_viol += int((got < floor_arc_ms - 1e-9).sum())
        worst_chord = max(worst_chord, float((floor_chord_ms - got).max()))
        worst_arc = max(worst_arc, float((floor_arc_ms - got).max()))
        detail_rows.append(
            f"{tag} {a}->{b}: fastest {got.min():.3f} ms vs chord floor "
            f"{floor_chord_ms:.3f} ms (margin {got.min() - floor_chord_ms:+.3f}), "
            f"arc {floor_arc_ms:.3f} ms; {len(live)}/{n_steps} steps with a path"
        )

    chk.record(
        "no path beats light on the straight line between the two cities",
        chord_viol == 0 and tested > 0,
        f"{chord_viol} violations over {tested} paths; tightest margin "
        f"{-worst_chord:+.4f} ms" + chr(10) + chr(10).join(detail_rows),
    )
    chk.record(
        "no path beats light along the ground arc either (these five pairs only)",
        arc_viol == 0,
        f"{arc_viol} violations over {tested} paths; tightest margin "
        f"{-worst_arc:+.4f} ms. Empirical for separations up to ~4100 km, not a "
        f"general law -- see the docstring.",
    )

    # The scoring flag must not reach the reported metric. Routing never reads
    # it, so this is a regression guard rather than a mutation target, and it
    # is written as one deliberately.
    pair = {c: compute_access(dataset, CITIES[c], rcfg, times)
            for c in ("New York", "San Francisco")}
    probes = np.unique(np.linspace(0, n_steps - 1, 15).astype(int))
    mismatch = 0
    compared = 0
    for formula in ("d_over_c", "c_over_d"):
        sel = DoTD(dataset, TopologyConfig(lookahead="forward", latency_formula=formula))
        runs = sel.run(int(probes.max()) + 1)
        for t in probes:
            res = shortest_path(dataset, runs[int(t)], pair, int(t), rcfg)
            if not res["reachable"]:
                continue
            compared += 1
            total = 0.0
            stations = list(pair.values())
            for x, y in zip(res["path"][:-1], res["path"][1:]):
                if x < dataset.n_sats and y < dataset.n_sats:
                    total += float(dataset.latency[int(t), x, y])
                else:
                    g, sat = (x, y) if x >= dataset.n_sats else (y, x)
                    total += float(stations[g - dataset.n_sats].latency[int(t), sat])
            if abs(total * 1e3 - res["latency_ms"]) > 1e-9:
                mismatch += 1
    chk.record(
        "reported latency is physical D/c under both latency_formula settings",
        mismatch == 0 and compared > 0,
        f"{mismatch} of {compared} paths disagree with independent D/c summation; "
        f"the flag changes which topology is selected, never how delay is measured",
    )


def check_experiment(
    chk: Checks, dataset: LinkDataset, results: dict, access: dict,
    rcfg: RoutingConfig, n_steps: int
) -> None:
    section("Experiment 1 summary quantities")
    summary = {}
    for name, linksets in results.items():
        rows = route_run(dataset, linksets, access, rcfg, n_steps)
        live = [r for r in rows if r["reachable"]]
        summary[name] = {
            "hops": float(np.mean([r["hops"] for r in live])),
            "isl": float(np.mean([r["isl_hops"] for r in live])),
            "reach": len(live) / len(rows),
        }

    # Every path leaves the ground and comes back, so exactly two of its edges
    # are access links whatever the topology. This pins isl_hops against hops.
    gap_ok = all(abs((v["hops"] - v["isl"]) - 2.0) < 1e-9 for v in summary.values())
    chk.record(
        "every path uses exactly two ground links (one up, one down)",
        gap_ok,
        "; ".join(f"{k}: hops {v['hops']:.3f} - isl {v['isl']:.3f}" for k, v in summary.items()),
    )

    chk.record(
        "+Grid needs more hops than every scored algorithm",
        all(summary["+Grid"]["hops"] > summary[k]["hops"] for k in summary if k != "+Grid"),
        "; ".join(f"{k} {v['hops']:.3f}" for k, v in summary.items()),
    )
    chk.record(
        "all algorithms are compared over the same reachable timesteps",
        len({round(v["reach"], 12) for v in summary.values()}) == 1,
        f"reachable fractions {[round(v['reach'], 4) for v in summary.values()]}; "
        "reachability is a property of the ground segment, not of the topology",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    args = p.parse_args()

    tcfg = TopologyConfig()
    rcfg = RoutingConfig()
    print("=" * 72)
    print("  LEO CONSTELLATION SIMULATOR -- STAGE 3 VALIDATION")
    print(f"  variant: {tcfg.variant_label} | {rcfg.variant_label}")
    print("=" * 72)

    cfg = SimConfig(
        selection=SelectionConfig(n_planes=args.planes, sats_per_plane=args.per_plane)
    ).resolve(ROOT)
    dataset = run(cfg, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)

    selectors = {
        "+Grid": PlusGrid(dataset, tcfg),
        "Greedy": Greedy(dataset, tcfg),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    }
    results = {name: s.run(n_steps) for name, s in selectors.items()}
    times = dataset_times(dataset)
    access = _pair_access(dataset, rcfg, times)
    print(f"\n  ran {len(selectors)} selectors x {n_steps} timesteps")

    chk = Checks()
    check_station_geometry(chk, dataset, times)
    check_access(chk, dataset, rcfg, times)
    check_graph(chk, dataset, results, access, rcfg)
    check_paths(chk, dataset, results, access, rcfg)
    check_latency_floor(chk, dataset, results['DoTD-forward'], rcfg, times, n_steps)
    check_experiment(chk, dataset, results, access, rcfg, n_steps)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
