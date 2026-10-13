"""Mutation harness for the stage-2 validation suite.

Each mutant is one deliberate bug: an exact string replacement in one source
file. For each, the file is mutated in place, ``validate_topology.py`` is run,
and the file is restored and its hash verified. A mutant is KILLED if the suite
exits non-zero (a failed check or a crash), SURVIVED if the suite still passes.

A survivor is not automatically a gap. It can be an *equivalent* mutant, one
that changes the source but not the behaviour on any input the suite can reach.
Mutants believed equivalent are marked ``expect_equivalent`` with the reason,
and are reported separately, so a believed-equivalent survivor cannot be
mistaken for a coverage gap and an unexpected survivor stands out.

Scope: stage-2 (M*), stage-3 (R*), stage-4 (T*) and Phase 2 extension (X*)
modules. Each mutant names the suite that
is supposed to catch it, and only that suite is run, so a stage-3 mutant cannot
be credited to a stage-2 check. The stage-1 mutation sweep referred to in the
project doc (48 mutants) was never committed to this repository, so there was
nothing to port; ``target`` is any repo-relative path, so stage-1 targets work
the same way.

Safety:
- a ``.mutbak`` copy of the original is written before mutating, and the run
  refuses to start while any ``.mutbak`` exists (a previous run was interrupted)
- the original bytes are restored in ``finally`` and the SHA-256 re-verified
- ``__pycache__`` is removed before every run and bytecode writing disabled:
  a mutant the same size as the original, restored within the same second,
  would otherwise be served stale bytecode, silently

    python mutants.py                 # all mutants
    python mutants.py M03 M16         # a subset
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
TOPOLOGY_SUITE = (sys.executable, "validate_topology.py", "--steps", "300")
ROUTING_SUITE = (sys.executable, "validate_routing.py", "--steps", "300")
TRAFFIC_SUITE = (sys.executable, "validate_traffic.py", "--steps", "300")
EXTENSION_SUITE = (sys.executable, "validate_extension.py", "--steps", "300")
TIMEOUT_S = 900


@dataclass(frozen=True)
class Mutant:
    id: str
    target: str
    old: str
    new: str
    description: str
    expect_equivalent: str = ""  # non-empty = reason it is believed equivalent
    suite: tuple[str, ...] = TOPOLOGY_SUITE


MUTANTS: list[Mutant] = [
    # --- scoring.py ------------------------------------------------------------
    Mutant("M01", "topology/scoring.py",
           "+ config.w2_latency * (1.0 - latency_norm)",
           "+ config.w2_latency * latency_norm",
           "latency term rewards high latency"),
    Mutant("M02", "topology/scoring.py",
           "+ config.w3_churn * churn",
           "+ 0.0 * churn",
           "churn term dropped from link_score"),
    Mutant("M03", "topology/scoring.py",
           "self.max_capacity = max(self.max_capacity, float(cap.max()))",
           "self.max_capacity = float(cap.max())",
           "capacity normalised by per-timestep max, not running max"),
    Mutant("M04", "topology/scoring.py",
           "self.max_latency = max(self.max_latency, float(lat.max()))",
           "self.max_latency = float(lat.max())",
           "latency normalised by per-timestep max, not running max"),
    Mutant("M05", "topology/scoring.py",
           "np.divide(SPEED_OF_LIGHT_M_S, dist_m, out=out",
           "np.divide(dist_m, SPEED_OF_LIGHT_M_S, out=out",
           "c/D variant silently computes D/c"),
    Mutant("M06", "topology/scoring.py",
           "return prev_adjacency.astype(np.float64)",
           "return np.zeros((n_sats, n_sats), dtype=np.float64)",
           "churn matrix ignores previous topology"),
    Mutant("M07", "topology/scoring.py",
           "cap_n = np.clip(capacity / self.max_capacity, 0.0, 1.0)",
           "cap_n = np.clip(capacity / self.max_capacity, 0.0, 2.0)",
           "normalised capacity may exceed 1"),
    Mutant("M08", "topology/scoring.py",
           'if formula == "d_over_c":',
           "if True:",
           "latency_formula flag ignored"),
    # --- dotd.py ---------------------------------------------------------------
    Mutant("M09", "topology/dotd.py",
           "        self.normaliser.update(t)\n",
           "        pass\n",
           "running max never updated"),
    Mutant("M10", "topology/dotd.py",
           "priority = score + cfg.w_lookahead * self._forward_term(t)",
           "priority = score",
           "forward lookahead term dropped"),
    Mutant("M11", "topology/dotd.py",
           "priority = score + cfg.w_lookahead * self._backward_term()",
           "priority = score",
           "backward accumulation term dropped"),
    Mutant("M12", "topology/dotd.py",
           'signed = priority if cfg.objective == "max" else -priority',
           "signed = priority",
           "objective flag ignored (argmin never applied)"),
    Mutant("M13", "topology/dotd.py",
           'super().__init__(dataset, dataclasses.replace(config, lookahead="none"), ground)',
           "super().__init__(dataset, config, ground)",
           "Greedy inherits the lookahead from a shared config"),
    Mutant("M14", "topology/dotd.py",
           "w = float(np.exp(-(off * step_s) / cfg.horizon_s))",
           "w = float(np.exp((off * step_s) / cfg.horizon_s))",
           "forward window weights the far future most"),
    Mutant("M15", "topology/dotd.py",
           "return np.arange(0, horizon + 1, stride)",
           "return np.arange(0, 1, stride)",
           "forward window contains only the current step"),
    Mutant("M16", "topology/dotd.py",
           "self.accumulated = g * self.accumulated + (1.0 - g) * r",
           "self.accumulated = g * self.accumulated + r",
           "backward accumulation unbounded (plain discounted sum)"),
    Mutant("M17", "topology/dotd.py",
           "return 0.5 * (a[:, None] + a[None, :])",
           "return np.broadcast_to(a[None, :], (len(a), len(a))).copy()",
           "edge term uses only j's accumulation (asymmetric)"),
    Mutant("M18", "topology/dotd.py",
           "links = b_matching(signed, visible, cfg.u_max, t)",
           "links = b_matching(signed, np.ones_like(visible), cfg.u_max, t)",
           "candidates not restricted to visible pairs"),
    Mutant("M19", "topology/dotd.py",
           "prev_adj = None if self.prev is None else self.prev.adjacency",
           "prev_adj = None",
           "selector never sees its previous topology"),
    Mutant("M20", "topology/dotd.py",
           "score = np.where(visible, score, 0.0)",
           "score = score",
           "score not masked to visible pairs",
           expect_equivalent=(
               "quality_matrix already zeroes invisible pairs; the only unmasked term is "
               "churn on a pair linked at t-1 and invisible at t, which b_matching excludes "
               "as a candidate and _update_backward never reads (it sums selected links only)"
           )),
    # --- base.py ---------------------------------------------------------------
    Mutant("M21", "topology/base.py",
           "if deg[i] < u_max and deg[j] < u_max:",
           "if deg[i] < u_max:",
           "b_matching checks only one endpoint's degree"),
    Mutant("M22", "topology/base.py",
           "adj[i, j] = adj[j, i] = True",
           "adj[i, j] = True",
           "b_matching writes links in one direction only"),
    Mutant("M23", "topology/base.py",
           "order = np.lexsort((ju, iu, -p))",
           "order = np.lexsort((ju, iu, p))",
           "b_matching takes lowest priority first"),
    Mutant("M24", "topology/base.py",
           "if deg.max(initial=0) > u_max:",
           "if deg.max(initial=0) > u_max + 1:",
           "assert_valid degree check off by one"),
    Mutant("M25", "topology/base.py",
           "bad = a & ~np.asarray(dataset.visible[t])",
           "bad = a & False",
           "assert_valid skips visibility"),
    Mutant("M26", "topology/base.py",
           "asym = np.argwhere(a != a.T)",
           "asym = np.argwhere(a != a)",
           "assert_valid skips symmetry"),
    Mutant("M27", "topology/base.py",
           "if np.any(np.diagonal(a)):",
           "if False:",
           "assert_valid skips self-links"),
    Mutant("M28", "topology/base.py",
           "        assert_valid(links, self.dataset, self.config.u_max)\n",
           "        pass\n",
           "step() no longer calls assert_valid",
           expect_equivalent=(
               "the in-step guard is defence in depth; on correct selectors it never fires, "
               "and the suite re-derives every constraint independently of it, which is the "
               "point -- M21/M22/M18/M34 show those independent checks still catch the bugs "
               "the guard exists for"
           )),
    Mutant("M29", "topology/base.py",
           "if self._last_t is None and t != 0:",
           "if False:",
           "step() accepts a first step other than t=0"),
    # --- grid.py ---------------------------------------------------------------
    Mutant("M30", "topology/grid.py",
           "i, j = mem[a], mem[(a + 1) % k]",
           "i, j = mem[a], mem[(a + 2) % k]",
           "in-plane ring skips a satellite"),
    Mutant("M31", "topology/grid.py",
           "rows, cols = linear_sum_assignment(cost)",
           "rows, cols = linear_sum_assignment(-cost)",
           "cross-plane assignment maximises phase difference"),
    Mutant("M32", "topology/grid.py",
           "            if g <= limit:\n",
           "            if True:\n",
           "planes adjacent across an unsampled plane"),
    Mutant("M33", "topology/grid.py",
           "PLANE_GAP_DEG = 3.0",
           "PLANE_GAP_DEG = 30.0",
           "plane clustering merges neighbouring planes"),
    Mutant("M34", "topology/grid.py",
           "return LinkSet(self.grid & visible, t)",
           "return LinkSet(self.grid, t)",
           "+Grid ignores visibility"),
    Mutant("M35", "topology/grid.py",
           "if len(planes) >= 3 and wrap_gap <= limit:",
           "if len(planes) >= 3:",
           "first and last plane always wrap"),
    Mutant("M36", "topology/grid.py",
           "np.arctan2(normals[..., 0], -normals[..., 1])",
           "np.arctan2(normals[..., 1], normals[..., 0])",
           "RAAN measured from the wrong axis",
           # Was believed equivalent: clustering and adjacency use only circular
           # RAAN differences, which a constant rotation preserves. That held
           # only while nothing pinned absolute RAAN. check_plane_adjacency now
           # places planes at known RAANs and sorts by recovered RAAN, and the
           # 90 deg rotation moves one across the 0/360 wrap -- so it is killed.
           ),
    # 0.02, not 0.1. Members of one plane on this shell sit within 0.067 deg of
    # each other in consecutive RAAN, so 0.1 leaves the clustering bit-identical
    # and mutates nothing. It bit only while the selection was scattering
    # satellites across real planes, which widened the intra-plane scatter --
    # i.e. the mutant was testing the broken selection, not the constant.
    Mutant("M37", "topology/grid.py",
           "PLANE_GAP_DEG = 3.0",
           "PLANE_GAP_DEG = 0.02",
           "plane clustering splits real planes"),
    # grid_gaps. The suite runs +Grid under one policy only, so each of these
    # needs its own check: the synthetic full-shell layout (M43, M44), the
    # PlusGrid-uses-the-flag ratio (M45), the label pin (M46), and the real-run
    # connectivity check (M47, which splits the 10-plane baseline in the middle).
    Mutant("M43", "topology/grid.py",
           'GAP_FACTORS = {"strict": ADJACENT_GAP_FACTOR, "bridge": BRIDGED_GAP_FACTOR}',
           'GAP_FACTORS = {"strict": ADJACENT_GAP_FACTOR, "bridge": ADJACENT_GAP_FACTOR}',
           "bridge policy silently behaves as strict"),
    Mutant("M44", "topology/grid.py",
           "BRIDGED_GAP_FACTOR = 2.5",
           "BRIDGED_GAP_FACTOR = 10.0",
           "bridge policy joins planes across any gap"),
    Mutant("M45", "topology/grid.py",
           "build_plane_structure(dataset, GAP_FACTORS[config.grid_gaps])",
           "build_plane_structure(dataset)",
           "PlusGrid ignores grid_gaps"),
    Mutant("M46", "topology/config.py",
           'f"lookahead={self.lookahead}, grid_gaps={self.grid_gaps}"',
           'f"lookahead={self.lookahead}"',
           "variant label omits the grid gap policy"),
    Mutant("M47", "topology/grid.py",
           "            if g <= limit:\n",
           "            if g <= limit and (len(planes) < 6 or a != planes[len(planes) // 2]):\n",
           "+Grid split in two: middle plane pair never adjacent"),
    # --- metrics.py ------------------------------------------------------------
    Mutant("M38", "topology/metrics.py",
           "prev.adjacency ^ curr.adjacency",
           "prev.adjacency & curr.adjacency",
           "churn counts kept links instead of changed links"),
    Mutant("M39", "topology/metrics.py",
           '"connected": bool(n_comp == 1),',
           '"connected": bool(n_comp >= 1),',
           "connected flag true for any graph"),
    Mutant("M40", "topology/metrics.py",
           '"capacity_total_gbps": float(cap.sum() / 1e9),',
           '"capacity_total_gbps": float(cap.sum() / 1e6),',
           "total capacity reported in Mbps under a Gbps label"),
    Mutant("M41", "topology/metrics.py",
           "float(lat.mean() * 1e3)",
           "float(lat.mean() * 1e6)",
           "latency reported in us under an ms label"),
    Mutant("M42", "topology/metrics.py",
           "    if prev is None:\n        return 0\n",
           "    if prev is None:\n        return curr.n_links\n",
           "churn at t=0 counts every link as new"),
    # --- routing/stations.py ---------------------------------------------------
    Mutant("R01", "routing/stations.py",
           "visible = elevation >= cfg.min_elevation_deg",
           "visible = elevation >= 0.0",
           "elevation mask ignored (horizon used instead)",
           suite=ROUTING_SUITE),
    Mutant("R02", "routing/stations.py",
           "    up = above.at(times).position.km.T - pos",
           "    up = pos.copy()",
           "geocentric radial used as the local vertical",
           suite=ROUTING_SUITE),
    Mutant("R03", "routing/stations.py",
           "    pos = site.at(times).position.km.T",
           "    pos = np.repeat(site.at(times[0]).position.km[None, :], len(times), axis=0)",
           "station frozen at t=0 (does not rotate with the Earth)",
           suite=ROUTING_SUITE),
    Mutant("R04", "routing/stations.py",
           "        ranked = np.argsort(-np.where(visible, elevation, -np.inf), axis=1)",
           "        ranked = np.argsort(np.where(visible, elevation, np.inf), axis=1)",
           "access cap keeps the lowest-elevation links",
           suite=ROUTING_SUITE),
    # --- routing/paths.py ------------------------------------------------------
    Mutant("R05", "routing/paths.py",
           "    ii, jj = np.nonzero(np.triu(adj, k=1))",
           "    ii, jj = np.nonzero(np.triu(np.asarray(dataset.visible[t]), k=1))",
           "routes over every visible link, not the selected topology",
           suite=ROUTING_SUITE),
    Mutant("R06", "routing/paths.py",
           "        isl_cost = np.ones(len(ii))",
           "        isl_cost = np.zeros(len(ii))",
           "hop metric weights ISL edges 0 (free hops)",
           suite=ROUTING_SUITE),
    Mutant("R07", "routing/paths.py",
           "    return graph + graph.T, names",
           "    return graph, names",
           "graph left directed (upper triangle only)",
           suite=ROUTING_SUITE),
    Mutant("R08", "routing/paths.py",
           "            isl_hops += 1",
           "            pass",
           "isl_hops never counted",
           suite=ROUTING_SUITE),
    Mutant("R09", "routing/paths.py",
           "            latency += float(acc.latency[t, sat])",
           "            pass",
           "ground-link latency omitted from the path total",
           suite=ROUTING_SUITE),
    Mutant("R10", "routing/paths.py",
           '            "hops": NO_PATH,',
           '            "hops": 0,',
           "unreachable reported as a zero-hop path",
           suite=ROUTING_SUITE),
    # Killed by check_paths before the floor sees it, and that is the honest
    # ordering: the floor's real job is the case this harness cannot express --
    # delays that are internally consistent but wrong at the source, e.g. a
    # unit error in stage 1's latency_s. That would need a leosim/ mutation,
    # which invalidates the cache and so is out of scope for this sweep.
    Mutant("R11", "routing/paths.py",
           "            latency += float(dataset.latency[t, a, b])",
           "            latency += float(dataset.latency[t, a, b]) * 0.5",
           "ISL latency halved (path faster than light)",
           suite=ROUTING_SUITE),
    Mutant("R12", "routing/paths.py",
           "            path_km += float(acc.distance_km[t, sat])\n",
           "            pass\n",
           "stretch path length omits both ground links",
           suite=ROUTING_SUITE),
    Mutant("R13", "routing/paths.py",
           "path_km += float(dataset.distance[t, a, b])",
           "path_km += 0.0",
           "stretch path length omits the ISLs",
           suite=ROUTING_SUITE),
    Mutant("R14", "routing/stations.py",
           "return float(min(np.linalg.norm(va), np.linalg.norm(vb)) * angle)",
           "return float(6371.0 * angle)",
           "stretch floor on the mean-radius sphere, not the stations' own radius",
           suite=ROUTING_SUITE),
    Mutant("R15", "routing/stations.py",
           "return float(min(np.linalg.norm(va), np.linalg.norm(vb)) * angle)",
           "return float(np.linalg.norm(va - vb))",
           "stretch floor is the chord, not the arc",
           suite=ROUTING_SUITE),
    Mutant("R16", "routing/stations.py",
           "va, vb = np.asarray(geocentric_xyz_km(a)), np.asarray(geocentric_xyz_km(b))",
           "return great_circle_km(a, b, radius_km=min(np.linalg.norm(geocentric_xyz_km(a)), "
           "np.linalg.norm(geocentric_xyz_km(b))))",
           "stretch floor angle from geodetic latitude, not geocentric vectors",
           suite=ROUTING_SUITE),
    Mutant("R17", "routing/paths.py",
           "    if not stretch >= 1.0:",
           "    if False:",
           "stretch below 1 reported instead of raising",
           suite=ROUTING_SUITE),
    Mutant("R18", "routing/paths.py",
           '"stretch": path_stretch(path_km, floor_km) if floor_km > 0 else np.nan,',
           '"stretch": path_stretch(path_km, floor_km),',
           "co-located endpoints crash instead of reporting an undefined stretch",
           suite=ROUTING_SUITE),
    # --- traffic/demand.py -----------------------------------------------------
    Mutant("T01", "traffic/demand.py",
           "(local_hour - cfg.diurnal_peak_hour)",
           "(local_hour + cfg.diurnal_peak_hour)",
           "diurnal peak at the wrong hour",
           suite=TRAFFIC_SUITE),
    Mutant("T02", "traffic/demand.py",
           "rng.poisson(rate_bps * step_s / cfg.mean_flow_bits)",
           "rng.poisson(rate_bps / cfg.mean_flow_bits)",
           "arrivals ignore the step length",
           suite=TRAFFIC_SUITE),
    Mutant("T03", "traffic/demand.py",
           "+ np.asarray(longitude_deg) / DEG_PER_HOUR",
           "+ 0.0 * np.asarray(longitude_deg)",
           "local time ignores longitude (every city peaks together)",
           suite=TRAFFIC_SUITE),
    # --- traffic/network.py ----------------------------------------------------
    Mutant("T04", "traffic/network.py",
           "gone = (self.backlog > 0) & (cap <= 0)",
           "gone = np.zeros_like(cap, dtype=bool)",
           "a removed link keeps its backlog (churn costs nothing)",
           suite=TRAFFIC_SUITE),
    Mutant("T05", "traffic/network.py",
           "dropped = np.maximum(b - buffer, 0.0)",
           "dropped = np.zeros_like(b)",
           "buffer overflow vanishes uncounted",
           suite=TRAFFIC_SUITE),
    Mutant("T06", "traffic/network.py",
           "b = self.backlog + arrivals - cap * self.dt",
           "b = self.backlog + arrivals - cap",
           "queue drains capacity per step, not capacity x step",
           suite=TRAFFIC_SUITE),
    Mutant("T07", "traffic/network.py",
           "ii, jj = np.nonzero(np.triu(links.adjacency, k=1))",
           "ii, jj = np.nonzero(np.triu(np.asarray(self.dataset.visible[t]), k=1))",
           "load routed over visibility, not the selected topology",
           suite=TRAFFIC_SUITE),
    Mutant("T08", "traffic/network.py",
           'if self.tcfg.direction == "to_gateway":',
           "if False:",
           "direction flag ignored",
           suite=TRAFFIC_SUITE),
    Mutant("T09", "traffic/network.py",
           "keep = float(np.prod(1.0 - drop_frac[src, dst]))",
           "keep = 1.0",
           "drops never reduce delivered traffic",
           suite=TRAFFIC_SUITE),
    Mutant("T10", "traffic/network.py",
           "delays.append(prop + float(qdelay[src, dst].sum()))",
           "delays.append(prop)",
           "queueing delay omitted from path delay",
           suite=TRAFFIC_SUITE),
    Mutant("T11", "traffic/network.py",
           "            k = int(np.argmin(total))",
           "            k = 0",
           "city uses its first access satellite, not its cheapest route",
           suite=TRAFFIC_SUITE),
    Mutant("T12", "traffic/network.py",
           "    if tcfg.feeder_is_default:\n",
           "    if True:\n",
           "feeder overrides silently ignored",
           suite=TRAFFIC_SUITE),
    Mutant("T13", "traffic/network.py",
           '"bandwidth_hz": tcfg.feeder_bandwidth_hz,',
           '"bandwidth_hz": None,',
           "feeder bandwidth override dropped from the budget",
           suite=TRAFFIC_SUITE),
    Mutant("T14", "traffic/demand.py",
           'if cfg.gateway_sites == "krishnan8":',
           "if False:",
           "gateway-sites flag ignored",
           suite=TRAFFIC_SUITE),
    Mutant("T15", "traffic/network.py",
           "return float(util.max() / aggregate), int((load > 0).sum())",
           "return float(util.mean() / aggregate), int((load > 0).sum())",
           "gateway imbalance from mean, not peak, utilisation",
           suite=TRAFFIC_SUITE),
    Mutant("T16", "traffic/network.py",
           "serving = capacity > 0",
           "serving = capacity >= 0",
           "gateways with no feeder capacity counted as serving",
           suite=TRAFFIC_SUITE),
    Mutant("T17", "traffic/network.py",
           "                path_km += acc.distance_at(t, sat)",
           "                path_km += 0.0",
           "stretch omits the access and feeder links from path length",
           suite=TRAFFIC_SUITE),
    Mutant("T18", "traffic/network.py",
           "floor = float(self.floor_km[c, path[0] - gw_base])",
           "floor = float(self.floor_km[c].max())",
           "stretch measured against the farthest gateway, not the one reached",
           suite=TRAFFIC_SUITE),
    # --- Phase 2 extension: topology/scoring.py ----------------------------------
    Mutant("X01", "topology/scoring.py",
           "config.w_egress * egress_edge_norm",
           "-config.w_egress * egress_edge_norm",
           "egress rewarded as a penalty",
           suite=EXTENSION_SUITE),
    Mutant("X02", "topology/scoring.py",
           "+ config.w_congestion * (1.0 - demand_edge_norm)",
           "+ config.w_congestion * demand_edge_norm",
           "congestion rewarded instead of penalised",
           suite=EXTENSION_SUITE),
    Mutant("X03", "topology/scoring.py",
           "return np.maximum(egress_norm[:, None], egress_norm[None, :])",
           "return 0.5 * (egress_norm[:, None] + egress_norm[None, :])",
           "egress combined by mean of endpoints, not max",
           suite=EXTENSION_SUITE),
    Mutant("X04", "topology/scoring.py",
           "return 0.5 * (demand_norm[:, None] + demand_norm[None, :])",
           "return np.broadcast_to(demand_norm[None, :], (len(demand_norm), len(demand_norm))).copy()",
           "congestion uses only j's demand (asymmetric)",
           suite=EXTENSION_SUITE),
    Mutant("X05", "topology/scoring.py",
           "self.max_egress = max(self.max_egress, float(self.ground.egress_bps[t].max()))",
           "self.max_egress = float(self.ground.egress_bps[t].max())",
           "egress normalised by per-timestep max, not running max",
           suite=EXTENSION_SUITE),
    Mutant("X06", "topology/scoring.py",
           "self.max_demand = max(self.max_demand, float(self.ground.demand_bps[t].max()))",
           "self.max_demand = float(self.ground.demand_bps[t].max())",
           "demand normalised by per-timestep max, not running max",
           suite=EXTENSION_SUITE),
    Mutant("X07", "topology/scoring.py",
           "return np.clip(e / self.max_egress, 0.0, 1.0)",
           "return e / self.max_egress",
           "future egress above the causal max not clipped",
           suite=EXTENSION_SUITE),
    Mutant("X20", "topology/scoring.py",
           "return np.clip(d / self.max_demand, 0.0, 1.0)",
           "return d / self.max_demand",
           "demand above the causal max not clipped",
           suite=EXTENSION_SUITE),
    # --- Phase 2 extension: topology/dotd.py -------------------------------------
    Mutant("X08", "topology/dotd.py",
           "egress_total += w * np.where(",
           "egress_total += 0.0 * np.where(",
           "egress missing from the forward window",
           suite=EXTENSION_SUITE),
    Mutant("X09", "topology/dotd.py",
           "egress_edge(self.ground_norm.egress(tt))",
           "egress_edge(self.ground_norm.egress(t))",
           "forward egress samples read the current step, not t + h",
           suite=EXTENSION_SUITE),
    Mutant("X10", "topology/dotd.py",
           "np.asarray(self.dataset.visible[tt]), egress_edge",
           "np.ones_like(np.asarray(self.dataset.visible[tt])), egress_edge",
           "forward egress credited to links that will be occluded",
           suite=EXTENSION_SUITE),
    Mutant("X11", "topology/dotd.py",
           "        if cfg.uses_ground:\n            score = score + np.where(",
           "        if False:\n            score = score + np.where(",
           "ground terms never added to the score",
           suite=EXTENSION_SUITE),
    Mutant("X12", "topology/dotd.py",
           "self._update_backward(links, phase1_score)",
           "self._update_backward(links, score)",
           "backward accumulator absorbs the extension terms",
           suite=EXTENSION_SUITE),
    Mutant("X13", "topology/dotd.py",
           "if config.uses_ground and ground is None:",
           "if False:",
           "nonzero weight without a prediction silently accepted",
           suite=EXTENSION_SUITE),
    Mutant("X14", "topology/dotd.py",
           "return geo + self.config.w_egress * egress",
           "return geo + egress",
           "forward egress part not weighted by w_egress",
           suite=EXTENSION_SUITE),
    # --- Phase 2 extension: topology/config.py -----------------------------------
    Mutant("X15", "topology/config.py",
           "return self.w_egress > 0 or self.w_congestion > 0",
           "return self.w_egress > 0",
           "congestion weight alone does not switch the ground terms on",
           suite=EXTENSION_SUITE),
    Mutant("X16", "topology/config.py",
           "        if self.uses_ground:\n            label +=",
           "        if True:\n            label +=",
           "Phase 1 label changed by the extension",
           suite=EXTENSION_SUITE),
    # --- Phase 2 extension: traffic/predict.py -----------------------------------
    Mutant("X17", "traffic/predict.py",
           "np.maximum.at(egress, (tt, acc.sat), acc.capacity)",
           "np.add.at(egress, (tt, acc.sat), acc.capacity)",
           "egress sums all gateways instead of taking the best",
           suite=EXTENSION_SUITE),
    Mutant("X18", "traffic/predict.py",
           "rate[tt, c] / k[tt]",
           "rate[tt, c]",
           "city demand counted in full on every satellite it sees",
           suite=EXTENSION_SUITE),
    Mutant("X19", "traffic/predict.py",
           "rate = expected_rate_bps(sites, np.asarray(utc_hours, dtype=np.float64), tcfg)",
           'rate = __import__("traffic.demand", fromlist=["_"]).offered_bits('
           "expected_rate_bps(sites, np.asarray(utc_hours, dtype=np.float64), tcfg), 1.0, tcfg)",
           "prediction uses the Poisson realisation, not the expected rate",
           suite=EXTENSION_SUITE),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clear_bytecode() -> None:
    dirs = [ROOT / "__pycache__"]
    for pkg in ("topology", "routing", "traffic", "leosim"):
        dirs.extend((ROOT / pkg).rglob("__pycache__"))
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def run_suite(suite: tuple[str, ...]) -> tuple[int, float, str]:
    clear_bytecode()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            list(suite), cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TIMEOUT_S,
        )
        code, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        code, out = -1, "TIMEOUT"
    return code, time.perf_counter() - t0, out


def first_failure(out: str) -> str:
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "[FAIL]" in line:
            return line.split("[FAIL]", 1)[1].strip()
    for line in reversed(lines):
        if "Error" in line or "Exception" in line:
            return "crash: " + line.strip()[:110]
    return "(non-zero exit, no FAIL line)"


def apply_and_run(m: Mutant) -> dict:
    path = ROOT / m.target
    original = path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    text = original.decode("utf-8")

    count = text.count(m.old)
    if count != 1:
        return {"id": m.id, "status": "INVALID", "detail": f"pattern occurs {count} times, need 1", "secs": 0.0}

    backup = path.with_suffix(path.suffix + ".mutbak")
    backup.write_bytes(original)
    try:
        path.write_bytes(text.replace(m.old, m.new).encode("utf-8"))
        code, secs, out = run_suite(m.suite)
    finally:
        path.write_bytes(original)
        if sha(path) != original_hash:
            raise RuntimeError(f"{m.target} NOT restored after {m.id}; original at {backup}")
        backup.unlink()

    if code == 0:
        status = "SURVIVED-EQUIV" if m.expect_equivalent else "SURVIVED"
        detail = m.expect_equivalent or "suite passed with the bug present"
    else:
        status, detail = "KILLED", first_failure(out)
    return {"id": m.id, "status": status, "detail": detail, "secs": secs}


def main(argv: list[str]) -> int:
    stale = list(ROOT.rglob("*.mutbak"))
    if stale:
        print("Refusing to run: backups from an interrupted run exist. Restore them first:")
        for s in stale:
            print(f"  {s}")
        return 2

    chosen = [m for m in MUTANTS if not argv or m.id in argv]
    # Both suites must be green before any mutant means anything. A mutant runs
    # only against the suite meant to catch it, so a suite that was already red
    # would turn every one of its mutants into a false KILL.
    for suite in (TOPOLOGY_SUITE, ROUTING_SUITE, TRAFFIC_SUITE, EXTENSION_SUITE):
        if not any(m.suite == suite for m in chosen):
            continue
        print(f"Baseline: unmutated suite must pass ({' '.join(suite[1:])})")
        code, secs, out = run_suite(suite)
        if code != 0:
            print(f"  baseline FAILED in {secs:.0f}s -- mutation results would be meaningless")
            print(out[-3000:])
            return 2
        print(f"  baseline passed in {secs:.0f}s")
    print()

    results = []
    for m in chosen:
        r = apply_and_run(m)
        results.append(r)
        print(f"{m.id} {r['status']:<15} {r['secs']:5.0f}s  {m.target:<22} {m.description}")
        print(f"     -> {r['detail']}")

    killed = sum(r["status"] == "KILLED" for r in results)
    equiv = [r for r in results if r["status"] == "SURVIVED-EQUIV"]
    surv = [r for r in results if r["status"] == "SURVIVED"]
    invalid = [r for r in results if r["status"] == "INVALID"]

    print()
    print("=" * 72)
    print(f"  {killed}/{len(results)} killed; {len(equiv)} survived as believed-equivalent; "
          f"{len(surv)} UNEXPECTED survivors; {len(invalid)} invalid")
    for r in surv:
        print(f"    UNEXPECTED SURVIVOR {r['id']}")
    for r in invalid:
        print(f"    INVALID {r['id']}: {r['detail']}")
    print("=" * 72)
    return 1 if (surv or invalid) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
