"""Phase 2 extension validation: the egress and congestion score terms.

    python validate_extension.py              # 187 sats, full 30-minute run
    python validate_extension.py --steps 300  # fast pass; what mutants.py drives

Same rules as the other suites: every check drives production code, pinned
values are hard-coded literals, and each check that could pass vacuously has a
partner that shows it cannot.

The extension's two hazards are opposite. A term that silently does nothing
(a zero weight path, a normaliser stuck at 0, a forward sample read at the
wrong time) leaves every Phase 1 number intact and makes the extension look
inert; a term that leaks into the default path changes Phase 1 numbers that
have already been reported. The regression guard pins the second against
hashes recorded BEFORE the extension was written; the activity checks pin the
first.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leosim.config import SelectionConfig, SimConfig
from leosim.simulation import run
from routing import GroundStation, RoutingConfig, compute_access, dataset_times
from run_traffic import utc_hours
from topology import DoTD, Greedy, GroundPrediction, PlusGrid, TopologyConfig
from topology.scoring import GroundNormaliser, congestion_edge, egress_edge, ground_score, link_score
from traffic import (
    DemandSites,
    SparseAccess,
    TrafficConfig,
    expected_rate_bps,
    gateway_access,
    gateway_pair_share,
    load_sites,
    offered_bits,
    predict_ground,
)
from validate import Checks, section
from validate_topology import TopologyValidationConfig, _sym, _synthetic, synthetic_score_dataset

ROOT = Path(__file__).parent

# --------------------------------------------------------------------------
# Regression guard. SHA-256 over every timestep's adjacency (np.packbits) and,
# for the scored selectors, every timestep's priority matrix (float64 bytes),
# recorded on the Phase 1 code BEFORE the extension existed (commit edfde1c,
# default 187-satellite constellation, stage-1 cache 54a1268e38b278ba). Not
# recomputed: a reference derived from the current code would move with it.
# --------------------------------------------------------------------------
GOLDEN = {
    300: {
        "+Grid": ("f4ed656f1bbb547d33364c97e4785234781f338f9620bdf7e054e7a9829a9526", None),
        "Greedy": ("af0b1a9517ec3e6a56bb43a50ffdcc865a047a1d1ab034726fa3f0667b23a7b1",
                   "efe8893dd825a5a79b57533b0a8248057e51a10bb9f7a1faf11d1d255c483967"),
        "DoTD-forward": ("426c8a0850b101c6faaaac8ef2f62bd95a36a10c8056a20b425b0954a5b59631",
                         "e6666d8720586897c38cf27f90b57935352c96fc29ff2831821861123a88a5ec"),
        "DoTD-backward": ("05f3eb04124a2691743a4c88386df2020df1a9f3d3cf51f227716936ebd34b54",
                          "681f694aec0291e23f9cf8a5090f2630834d4cae01ce9d5b9cf5468b629a6d7d"),
    },
    1800: {
        "+Grid": ("a184a2d67c17ff7818d55da3c57e6ec3051026ae617bec29e62f7600b7162261", None),
        "Greedy": ("6ed43da6ade256ada09e9b43fa793ec4712aec2785707b94416b0d923e24c24e",
                   "4f84820bb8bfdad22bda0a273c119a9f0ab7cc8034c94dec6636304644f16f4a"),
        "DoTD-forward": ("b6026e9ef4feebb305ab89dbcc43c0e144c5120d28fcdd0956c87ee8446b677b",
                         "1a83988ff55ad93f7f0eafda731fb19a2db25f1faf5b53ad12ef7cbbf8e8fbad"),
        "DoTD-backward": ("da4db45cbb8dba0831403555eac3338874f84da2c793b4b56954135d84bb7f05",
                          "a46c7bf957f527999c36bba2db2998b886a188483074871092209a0e28c6eecb"),
    },
}
PHASE1_LABEL = "objective=max, latency=d_over_c, lookahead=forward, grid_gaps=strict"
PHASE1_FILE_TAG = "max_d_over_c"

# --------------------------------------------------------------------------
# Hand-checkable pins. Weights here are pin inputs, chosen to make the
# arithmetic legible: w_egress = 0.5, w_congestion = 0.25.
#
# Free function, pair (i, j) with per-satellite normalised egress [0.4, 0.8]
# and demand [0.2, 0.6]:
#   egress edge     = max(0.4, 0.8)          = 0.8   (a mean would give 0.6)
#   congestion edge = mean(0.2, 0.6)         = 0.4   (a max would give 0.6)
#   link_score(0.75, 0.25, 1)                = 0.80  (Phase 1 pin)
#   + 0.5 * 0.8 + 0.25 * (1 - 0.4)           = 0.80 + 0.40 + 0.15 = 1.35
#
# End to end through Greedy on synthetic_score_dataset (Phase 1 score of pair
# (0, 2): 0.60 at t=0, 0.65 at t=1), with
#   egress  t=0 [0.5, 4.0, 1.0] Gbps -> running max 4.0
#           t=1 [0.5, 0.0, 1.0] Gbps -> running max stays 4.0
#   demand  t=0 [1.0, 2.0, 0.0] Gbps -> running max 2.0
#           t=1 [1.0, 0.0, 0.0] Gbps -> running max stays 2.0
#   t=0  egress edge max(0.125, 0.25) = 0.25; congestion mean(0.5, 0) = 0.25
#        0.60 + 0.5 * 0.25 + 0.25 * 0.75 = 0.60 + 0.125 + 0.1875 = 0.9125
#   t=1  same normalised inputs, by the running maxima:
#        0.65 + 0.125 + 0.1875                                    = 0.9625
#   A per-timestep max would normalise t=1 by 1.0 and 1.0 Gbps instead:
#        egress max(0.5, 1.0) = 1.0, congestion mean(1.0, 0) = 0.5
#        0.65 + 0.5 * 1.0 + 0.25 * 0.5                            = 1.275
# --------------------------------------------------------------------------
PIN_W_EGRESS = 0.5
PIN_W_CONGESTION = 0.25
PIN_FREE = 1.35
PIN_E2E_T0 = 0.9125
PIN_E2E_T1 = 0.9625
PIN_E2E_T1_PER_STEP_MAX = 1.275
PIN_REL_TOL = 1e-12

# Predictor by hand: 3 satellites, 2 timesteps, flat diurnal curve, 1 Gbps
# total, cities A (population 3) and B (population 1) -> 0.75 and 0.25 Gbps.
#   t=0  A sees {0, 1}, B sees {1}:  demand [0.375, 0.375 + 0.25, 0] Gbps
#   t=1  A sees {1},    B sees {}:   demand [0, 0.75, 0] Gbps
#   t=0  G1 sees 1 @ 2 Gbps, 2 @ 3 Gbps; G2 sees 2 @ 1 Gbps
#        egress [0, 2, max(3, 1) = 3] Gbps   (a sum would give 4 on sat 2)
#   t=1  G1 sees 0 @ 0.5 Gbps; G2 sees nothing: egress [0.5, 0, 0] Gbps
# Occluded future: link (0, 2) visible at t=0 only, satellite 2 sees a gateway
# throughout, horizon 10 s at a 1 s stride (offsets 0..10). Only offset 0 can
# count, so the egress part is w_0 / sum_h w_h = 1 / sum_{h=0}^{10} e^{-h/10}.
# Unmasked, every offset would count and the part would be 1.
PIN_OCCLUDED_EGRESS = 0.1426449665704087
# The assembled forward term on the same link at w_egress = 0.5. Link quality
# there is 0.4 * 1 + 0.4 * (1 - 1) = 0.4 at offset 0 only (both links share the
# capacity and latency maxima), so the geometric part is 0.4 / 0.8 * P = 0.5 P
# and the egress part adds 0.5 * P: total P, where P = PIN_OCCLUDED_EGRESS.
# Unweighted egress would give 1.5 P.
PIN_OCCLUDED_FORWARD_TERM = 0.1426449665704087

PIN_PRED_DEMAND =np.array([[0.375e9, 0.625e9, 0.0], [0.0, 0.75e9, 0.0]])
PIN_PRED_EGRESS = np.array([[0.0, 2.0e9, 3.0e9], [0.5e9, 0.0, 0.0]])

# Names the open-loop predictor must never touch: the simulator, the Poisson
# realisation of demand, and queue state.
FORBIDDEN_IN_PREDICTOR = {"TrafficSim", "offered_bits", "backlog"}


@dataclass(frozen=True)
class ExtensionValidationConfig:
    # Weight used to switch a term ON in the activity checks. A VALIDATION
    # PROBE, not a tuned or recommended value: it only has to be large enough
    # to move some selections. No extension weight has been tuned.
    probe_weight: float = 0.2

    # Seeds for the open-loop check: the prediction must not depend on them.
    seeds: tuple[int, int] = (7, 8)

    # Load levels for the scale check: normalised demand must not depend on them.
    loads_gbps: tuple[float, float] = (1.0, 4.0)

    # Relative tolerance for checks that compare float sums taken in a
    # different order.
    sum_rel_tol: float = 1e-12


# --- helpers -----------------------------------------------------------------


def _ground(egress, demand, label="synthetic") -> GroundPrediction:
    return GroundPrediction(np.asarray(egress, dtype=np.float64), np.asarray(demand, dtype=np.float64), label)


def choice_dataset(n_steps: int):
    """Satellite 0 chooses between 1 and 2 (u_max = 1): links (0, 1) and (0, 2)
    identical in every stage-1 quantity, (1, 2) not visible. Satellite 3 has no
    links; it exists to hold a running max where a test needs one.

    With identical scores the matcher's (i, j) tie-break takes (0, 1), so every
    check below that expects (0, 2) cannot pass by accident."""
    n = 4
    cap = _sym(n, {(0, 1): 1.0e9, (0, 2): 1.0e9})
    lat = _sym(n, {(0, 1): 0.005, (0, 2): 0.005})
    dist = _sym(n, {(0, 1): 1500.0, (0, 2): 1500.0})
    vis = cap > 0
    rep = lambda a: np.repeat(a[None], n_steps, 0)  # noqa: E731
    return _synthetic(rep(cap), rep(lat), rep(dist), rep(vis))


def make_selector(kind: str, dataset, cfg: TopologyConfig, ground: GroundPrediction | None):
    if kind == "Greedy":
        return Greedy(dataset, cfg, ground)
    if kind == "DoTD-forward":
        return DoTD(dataset, dataclasses.replace(cfg, lookahead="forward"), ground)
    if kind == "DoTD-backward":
        return DoTD(dataset, dataclasses.replace(cfg, lookahead="backward"), ground)
    if kind == "+Grid":
        return PlusGrid(dataset, cfg)
    raise ValueError(kind)


@dataclass
class Run:
    linksets: list
    prefix: dict  # n -> (adjacency hash, priority hash or None) over the first n steps
    score_hash: str | None
    symmetric: bool  # priority symmetric at every step
    acc_range: tuple[float, float] | None  # backward accumulation range


def run_hashed(sel, n_steps: int, prefixes=(300, 1800)) -> Run:
    """Step a selector, hashing exactly as the golden recording did."""
    sel.reset()
    ha, hp, hs = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    scored = hasattr(sel, "normaliser")
    links, prefix, symmetric = [], {}, True
    lo, hi = np.inf, -np.inf
    for t in range(n_steps):
        ls = sel.step(t)
        links.append(ls)
        ha.update(np.packbits(ls.adjacency).tobytes())
        if scored:
            hp.update(np.ascontiguousarray(sel.last_priority, dtype=np.float64).tobytes())
            hs.update(np.ascontiguousarray(sel.last_score, dtype=np.float64).tobytes())
            symmetric &= bool(np.array_equal(sel.last_priority, sel.last_priority.T))
            if sel.config.lookahead == "backward":
                lo, hi = min(lo, float(sel.accumulated.min())), max(hi, float(sel.accumulated.max()))
        if t + 1 in prefixes:
            prefix[t + 1] = (ha.copy().hexdigest(), hp.copy().hexdigest() if scored else None)
    return Run(links, prefix, hs.hexdigest() if scored else None, symmetric,
               (lo, hi) if np.isfinite(lo) else None)


def differ_fraction(a: list, b: list) -> float:
    return float(np.mean([not np.array_equal(x.adjacency, y.adjacency) for x, y in zip(a, b)]))


def _forbidden_names(path: Path) -> set[str]:
    """Forbidden identifiers imported, named or accessed as attributes in a file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            found |= {a.name for a in node.names} & FORBIDDEN_IN_PREDICTOR
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_IN_PREDICTOR:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IN_PREDICTOR:
            found.add(node.attr)
    return found


def _imports_traffic(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "traffic":
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "traffic" for a in node.names):
            return True
    return False


# --- checks: synthetic ---------------------------------------------------------


def check_config(chk: Checks) -> None:
    section("Configuration: defaults, labels, refusals")
    cfg = TopologyConfig()
    chk.record("both extension weights default to 0",
               cfg.w_egress == 0.0 and cfg.w_congestion == 0.0 and not cfg.uses_ground,
               f"w_egress={cfg.w_egress}, w_congestion={cfg.w_congestion}")
    chk.record("default variant label and file tag are Phase 1's, character for character",
               cfg.variant_label == PHASE1_LABEL and cfg.file_tag == PHASE1_FILE_TAG,
               f"label {cfg.variant_label!r}\nfile tag {cfg.file_tag!r}")
    on = dataclasses.replace(cfg, w_egress=0.3)
    chk.record("an active extension weight appears in the label and the file tag",
               "w_egress=0.3" in on.variant_label and "egress0.3" in on.file_tag,
               f"label {on.variant_label!r}\nfile tag {on.file_tag!r}")

    ds = choice_dataset(2)
    try:
        Greedy(ds, dataclasses.replace(cfg, w_congestion=0.1))
        chk.record("a nonzero weight with no GroundPrediction is refused", False, "accepted")
    except ValueError as e:
        chk.record("a nonzero weight with no GroundPrediction is refused", True, f"ValueError: {e}")
    try:
        Greedy(ds, cfg, _ground(np.zeros((3, 4)), np.zeros((3, 4))))
        chk.record("a GroundPrediction of the wrong shape is refused", False, "accepted")
    except ValueError as e:
        chk.record("a GroundPrediction of the wrong shape is refused", True, f"ValueError: {e}")
    try:
        TopologyConfig(w_egress=-0.1)
        chk.record("a negative extension weight is refused", False, "accepted")
    except ValueError as e:
        chk.record("a negative extension weight is refused", True, f"ValueError: {e}")


def check_pins(chk: Checks) -> None:
    section("Score arithmetic: hand-checkable pins")
    cfg = TopologyConfig(u_max=4, w_egress=PIN_W_EGRESS, w_congestion=PIN_W_CONGESTION)
    e = egress_edge(np.array([0.4, 0.8]))[0, 1]
    d = congestion_edge(np.array([0.2, 0.6]))[0, 1]
    chk.close_to(
        "link_score + ground_score, egress [0.4, 0.8], demand [0.2, 0.6]",
        float(link_score(0.75, 0.25, 1.0, cfg) + ground_score(e, d, cfg)), PIN_FREE, PIN_REL_TOL,
        extra="0.80 + 0.5*max(0.4, 0.8) + 0.25*(1 - mean(0.2, 0.6)) = 0.80 + 0.40 + 0.15",
    )

    ds = synthetic_score_dataset()
    ground = _ground([[0.5e9, 4.0e9, 1.0e9], [0.5e9, 0.0, 1.0e9]],
                     [[1.0e9, 2.0e9, 0.0], [1.0e9, 0.0, 0.0]])
    sel = Greedy(ds, cfg, ground)
    sel.reset()
    sel.step(0)
    chk.close_to("end-to-end score with both terms, pair (0,2), t=0",
                 float(sel.last_score[0, 2]), PIN_E2E_T0, PIN_REL_TOL,
                 extra="0.60 + 0.5*0.25 + 0.25*(1 - 0.25)")
    sel.step(1)
    chk.close_to(
        "end-to-end score with both terms, pair (0,2), t=1 (running maxima)",
        float(sel.last_score[0, 2]), PIN_E2E_T1, PIN_REL_TOL,
        extra=f"0.65 + 0.125 + 0.1875; a per-timestep max would give {PIN_E2E_T1_PER_STEP_MAX}",
    )


def check_predictor_pin(chk: Checks) -> None:
    section("Predictor by hand: best gateway, equal demand split")
    tr = TrafficConfig(peak_total_gbps=1.0, diurnal_trough_fraction=1.0)
    sites = DemandSites(
        cities=[GroundStation("A", 0.0, 0.0), GroundStation("B", 0.0, 0.0)],
        population=np.array([3.0, 1.0]),
        gateways=[GroundStation("G1", 0.0, 0.0), GroundStation("G2", 0.0, 0.0)],
        gateway_country=["x", "y"],
    )

    def acc(per_t: list[list[tuple[int, float]]]) -> SparseAccess:
        indptr = np.cumsum([0] + [len(v) for v in per_t])
        sat = np.array([s for v in per_t for s, _ in v], dtype=int)
        cap = np.array([c for v in per_t for _, c in v], dtype=np.float64)
        return SparseAccess(indptr, sat, cap, np.full(len(sat), 0.002))

    cities = [acc([[(0, 4e9), (1, 4e9)], [(1, 4e9)]]), acc([[(1, 4e9)], []])]
    gws = [acc([[(1, 2.0e9), (2, 3.0e9)], [(0, 0.5e9)]]), acc([[(2, 1.0e9)], []])]
    g = predict_ground(sites, cities, gws, np.array([0.0, 1.0]), 3, tr, RoutingConfig())
    chk.record("predicted demand: each city split equally over the satellites it sees",
               np.array_equal(g.demand_bps, PIN_PRED_DEMAND),
               f"got {g.demand_bps.tolist()}\npinned {PIN_PRED_DEMAND.tolist()}")
    chk.record("predicted egress: each satellite's best gateway, 0 where it sees none",
               np.array_equal(g.egress_bps, PIN_PRED_EGRESS),
               f"got {g.egress_bps.tolist()}\npinned {PIN_PRED_EGRESS.tolist()}")


def check_direction(chk: Checks, vcfg: ExtensionValidationConfig) -> None:
    section("Each term points the right way (synthetic, all else equal)")
    w = vcfg.probe_weight
    base_cfg = TopologyConfig(u_max=1)

    ds = choice_dataset(2)
    base = Greedy(ds, base_cfg)
    base.reset()
    bl = base.step(0)
    chk.record("discrimination: with no ground terms, the tie-break takes (0,1)",
               bool(bl.adjacency[0, 1]) and not bl.adjacency[0, 2],
               f"edges {sorted(map(tuple, bl.edges.tolist()))}")

    # Egress: only satellite 2 sees a gateway (3 holds an equal one, unlinked).
    egress = np.zeros((2, 4))
    egress[:, 2] = egress[:, 3] = 1.0e9
    sel = Greedy(ds, dataclasses.replace(base_cfg, w_egress=w), _ground(egress, np.zeros((2, 4))))
    sel.reset()
    ls = sel.step(0)
    s01, s02 = float(sel.last_score[0, 1]), float(sel.last_score[0, 2])
    chk.record("egress: a neighbour with a gateway in view scores higher than an identical one without",
               s02 > s01, f"score (0,1) no gateway {s01:.6f}; (0,2) gateway {s02:.6f}")
    chk.record("egress: ... and is the one selected", bool(ls.adjacency[0, 2]),
               f"edges {sorted(map(tuple, ls.edges.tolist()))}")

    # Congestion: satellite 1 carries the demand.
    demand = np.zeros((2, 4))
    demand[:, 1] = 1.0e9
    sel = Greedy(ds, dataclasses.replace(base_cfg, w_congestion=w), _ground(np.zeros((2, 4)), demand))
    sel.reset()
    ls = sel.step(0)
    s01, s02 = float(sel.last_score[0, 1]), float(sel.last_score[0, 2])
    chk.record("congestion: a neighbour under demand scores lower than an identical idle one",
               s01 < s02 and bool(ls.adjacency[0, 2]),
               f"score (0,1) loaded {s01:.6f}; (0,2) idle {s02:.6f}; "
               f"edges {sorted(map(tuple, ls.edges.tolist()))}")

    # Monotone in demand: satellite 3 pins the running max at 1 Gbps.
    curve = []
    for d1 in (0.0, 0.25e9, 0.5e9, 1.0e9):
        demand = np.zeros((2, 4))
        demand[:, 3], demand[:, 1] = 1.0e9, d1
        sel = Greedy(ds, dataclasses.replace(base_cfg, w_congestion=w), _ground(np.zeros((2, 4)), demand))
        sel.reset()
        sel.step(0)
        curve.append(float(sel.last_score[0, 1]))
    chk.record("congestion: score falls strictly as the neighbour's demand rises",
               all(a > b for a, b in zip(curve, curve[1:])),
               f"score (0,1) at demand 0, 0.25, 0.5, 1 Gbps: {[round(v, 6) for v in curve]}")

    # Horizon: satellite 2 gains a gateway only from t=5. Greedy cannot see it;
    # forward DoTD must, through the same window as the geometric lookahead.
    n_steps = 30
    ds = choice_dataset(n_steps)
    egress = np.zeros((n_steps, 4))
    egress[:, 3] = 1.0e9
    egress[5:, 2] = 1.0e9
    hcfg = TopologyConfig(u_max=1, horizon_s=10.0, horizon_stride_s=1.0, w_egress=w)
    g = Greedy(ds, hcfg, _ground(egress, np.zeros((n_steps, 4))))
    g.reset()
    gl = g.step(0)
    f = DoTD(ds, dataclasses.replace(hcfg, lookahead="forward"), _ground(egress, np.zeros((n_steps, 4))))
    f.reset()
    fl = f.step(0)
    chk.record("egress horizon: Greedy, blind to the gateway arriving at t=5, keeps the tie-break",
               bool(gl.adjacency[0, 1]), f"Greedy edges {sorted(map(tuple, gl.edges.tolist()))}")
    chk.record("egress horizon: forward DoTD anticipates it and takes (0,2)",
               bool(fl.adjacency[0, 2]), f"DoTD-forward edges {sorted(map(tuple, fl.edges.tolist()))}")

    # A link that will be occluded earns nothing from its neighbour's egress
    # at the occluded samples, however good that egress is.
    occ = choice_dataset(n_steps)
    occ.visible[1:, 0, 2] = occ.visible[1:, 2, 0] = False
    egress = np.zeros((n_steps, 4))
    egress[:, 2] = 1.0e9
    o = DoTD(occ, dataclasses.replace(hcfg, lookahead="forward"), _ground(egress, np.zeros((n_steps, 4))))
    o.reset()
    o.step(0)
    chk.close_to("egress horizon: an occluded future link gets egress credit only while visible",
                 float(o._forward_parts(0)[1][0, 2]), PIN_OCCLUDED_EGRESS, PIN_REL_TOL,
                 extra="visible at offset 0 only: 1 / sum_{h=0}^{10} exp(-h/10)")
    o = DoTD(occ, dataclasses.replace(hcfg, lookahead="forward", w_egress=PIN_W_EGRESS),
             _ground(egress, np.zeros((n_steps, 4))))
    o.reset()
    o.step(0)
    chk.close_to("forward term = geometric part + w_egress x egress part (w_egress = 0.5)",
                 float(o._forward_term(0)[0, 2]), PIN_OCCLUDED_FORWARD_TERM, PIN_REL_TOL,
                 extra="0.4/0.8 * P + 0.5 * P = P; unweighted egress would give 1.5 P")

    # Clip: a future value above the causal max must normalise to 1, not above.
    # On the real shell the egress max climbs only ~2%, too little for the
    # discounted mean to cross 1 -- so the bound check there cannot see this.
    g2 = _ground([[1.0e9, 0.0], [2.0e9, 0.0]], [[1.0e9, 0.0], [2.0e9, 0.0]])
    gn = GroundNormaliser(g2)
    gn.update(0)
    chk.record("a future egress or demand at twice the running max normalises to 1, not above",
               float(gn.egress(1).max()) == 1.0 and float(gn.demand(1).max()) == 1.0,
               f"egress -> {float(gn.egress(1).max())}, demand -> {float(gn.demand(1).max())}")

    # Discount curve of the egress part, read out of the production loop.
    probes = [1, 5, 10, 15, 20]
    dcfg = TopologyConfig(u_max=1, horizon_s=20.0, horizon_stride_s=1.0, lookahead="forward", w_egress=w)
    curve = []
    for off in probes:
        egress = np.zeros((n_steps, 4))
        egress[0, 3] = 1.0e9
        egress[off, 1] = 1.0e9
        p = DoTD(ds, dcfg, _ground(egress, np.zeros((n_steps, 4))))
        p.reset()
        p.step(0)
        curve.append(float(p._forward_parts(0)[1][0, 1]))
    chk.record("egress forward weight falls strictly with offset (near future weighted most)",
               all(a > b for a, b in zip(curve, curve[1:])) and curve[0] > 0,
               f"egress part at offsets {probes}: {[round(v, 6) for v in curve]}")


# --- checks: real shell --------------------------------------------------------


def check_regression(chk: Checks, dataset, ground: GroundPrediction, n_steps: int) -> dict:
    section("Regression guard: weights 0 reproduce Phase 1 exactly")
    tcfg = TopologyConfig()
    baseline = {}
    for kind in ("+Grid", "Greedy", "DoTD-forward", "DoTD-backward"):
        r = run_hashed(make_selector(kind, dataset, tcfg, None), n_steps)
        baseline[kind] = r
        for n, pins in GOLDEN.items():
            if n > n_steps:
                continue
            adj_pin, prio_pin = pins[kind]
            adj, prio = r.prefix[n]
            ok = adj == adj_pin and (prio_pin is None or prio == prio_pin)
            chk.record(f"{kind}: first {n} steps match the pre-extension recording",
                       ok, f"adjacency {adj[:16]}.. (pinned {adj_pin[:16]}..)"
                       + ("" if prio_pin is None else f"\npriority  {prio[:16]}.. (pinned {prio_pin[:16]}..)"))
    if n_steps < min(GOLDEN):
        chk.record("regression guard ran", False, f"needs >= {min(GOLDEN)} steps, got {n_steps}")

    for kind in ("Greedy", "DoTD-forward", "DoTD-backward"):
        r = run_hashed(make_selector(kind, dataset, tcfg, ground), n_steps)
        b = baseline[kind]
        chk.record(
            f"{kind}: a GroundPrediction at weight 0 changes nothing, bit for bit",
            r.prefix == b.prefix and r.score_hash == b.score_hash
            and all(np.array_equal(x.adjacency, y.adjacency) for x, y in zip(r.linksets, b.linksets)),
            f"adjacency, priority and score hashes over {n_steps} steps "
            f"{'equal' if r.score_hash == b.score_hash else 'DIFFER'}",
        )
    return baseline


def check_terms_active(chk: Checks, dataset, ground: GroundPrediction, baseline: dict, n_steps: int,
                       vcfg: ExtensionValidationConfig, tvcfg: TopologyValidationConfig) -> dict:
    section("Each term changes the topology, and the two differ")
    w = vcfg.probe_weight
    tcfg = TopologyConfig()
    zeros = _ground(np.zeros_like(ground.egress_bps), np.zeros_like(ground.demand_bps), "zeros")
    out = {}
    for kind in ("Greedy", "DoTD-forward", "DoTD-backward"):
        if kind == "DoTD-backward":
            # Backward is here only for the discrimination run below: its
            # accumulator must ignore the extension terms, so over all-zero
            # predictions nothing at all may change.
            z = run_hashed(make_selector(kind, dataset, dataclasses.replace(tcfg, w_egress=w, w_congestion=w),
                                         zeros), n_steps)
            chk.record(f"{kind}: all-zero predictions change nothing (accumulator ignores the new terms)",
                       differ_fraction(z.linksets, baseline[kind].linksets) == 0.0,
                       f"{differ_fraction(z.linksets, baseline[kind].linksets) * n_steps:.0f} of "
                       f"{n_steps} timesteps differ")
            continue
        eg = run_hashed(make_selector(kind, dataset, dataclasses.replace(tcfg, w_egress=w), ground), n_steps)
        co = run_hashed(make_selector(kind, dataset, dataclasses.replace(tcfg, w_congestion=w), ground), n_steps)
        out[kind] = {"egress": eg, "congestion": co}
        base = baseline[kind].linksets
        for term, r in (("egress", eg), ("congestion", co)):
            chk.between(f"{kind}: the {term} term alone changes the topology",
                        differ_fraction(r.linksets, base), tvcfg.min_differ_fraction, 1.0,
                        extra=f"fraction of {n_steps} timesteps that differ from Phase 1 (probe weight {w:g})")
        chk.between(f"{kind}: egress-only and congestion-only topologies differ from each other",
                    differ_fraction(eg.linksets, co.linksets), tvcfg.min_differ_fraction, 1.0,
                    extra=f"fraction of {n_steps} timesteps")

        # Discrimination for the activity checks: the same weights over all-zero
        # predictions must change nothing. Egress then adds 0; congestion adds
        # the same headroom constant to every candidate, which cannot reorder
        # them -- so this also verifies the headroom form selects exactly what
        # a subtracted penalty would.
        both = dataclasses.replace(tcfg, w_egress=w, w_congestion=w)
        z = run_hashed(make_selector(kind, dataset, both, zeros), n_steps)
        chk.record(f"{kind}: discrimination -- both weights on over all-zero predictions change nothing",
                   differ_fraction(z.linksets, base) == 0.0,
                   f"{differ_fraction(z.linksets, base) * n_steps:.0f} of {n_steps} timesteps differ")

    section("Bounds with the terms on")
    both = dataclasses.replace(tcfg, w_egress=w, w_congestion=w)
    for kind in ("Greedy", "DoTD-forward", "DoTD-backward"):
        r = run_hashed(make_selector(kind, dataset, both, ground), n_steps)
        out.setdefault(kind, {})["both"] = r
        chk.record(f"{kind}, both terms: priority symmetric at every timestep", r.symmetric,
                   f"{n_steps} timesteps")
        if r.acc_range is not None:
            lo, hi = r.acc_range
            chk.record(f"{kind}, both terms: backward accumulation stays within [0, 1]",
                       lo >= 0.0 and hi <= 1.0, f"range {lo:.4f} .. {hi:.4f}")

    # The egress part of the forward window, swept over the run. Its clip binds
    # only while the running max is still climbing, so binding steps are
    # counted, not assumed.
    sel = make_selector("DoTD-forward", dataset, dataclasses.replace(tcfg, w_egress=w), ground)
    sel.reset()
    worst, binds = 0.0, 0
    for t in range(n_steps):
        sel.step(t)
        worst = max(worst, float(sel._forward_parts(t)[1].max()))
        ahead = sel._step_offsets() + t
        ahead = ahead[ahead < dataset.n_steps]
        if float(ground.egress_bps[ahead].max()) > sel.ground_norm.max_egress:
            binds += 1
    chk.record("egress forward part stays within [0, 1] over the real shell",
               worst <= 1.0 + PIN_REL_TOL, f"largest over {n_steps} timesteps: {worst:.6f}")
    chk.record("clip check discriminates: future egress does exceed the causal max here",
               binds > 0, f"{binds} of {n_steps} timesteps have a sample above the running max")
    return out


def check_ground_normaliser(chk: Checks, ground: GroundPrediction, n_steps: int) -> None:
    section("Ground running maxima: monotone, causal, live from t=0")
    norm = GroundNormaliser(ground)
    for t in range(n_steps):
        norm.update(t)
    hist = np.array([(h[1], h[2]) for h in norm.history])
    arrays = (ground.egress_bps, ground.demand_bps)
    for col, label in ((0, "egress"), (1, "demand")):
        drops = int((np.diff(hist[:, col]) < 0).sum())
        chk.record(f"running max {label} non-decreasing at every step", drops == 0,
                   f"{drops} decreases; range {hist[0, col]:.6g} -> {hist[-1, col]:.6g}")
        checkpoints = np.unique(np.linspace(0, n_steps - 1, 10).astype(int))
        worst = max(abs(hist[t, col] - arrays[col][: t + 1].max()) / arrays[col][: t + 1].max()
                    for t in checkpoints)
        chk.record(f"running max {label} at t equals the max over [0, t] (causal)", worst < 1e-12,
                   f"worst relative mismatch {worst:.2e} over {len(checkpoints)} checkpoints")
        dips = int((np.diff(arrays[col][:n_steps].max(axis=1)) < 0).sum())
        chk.record(f"{label} monotonicity check discriminates: per-timestep max does decrease",
                   dips > 0, f"per-timestep max decreases at {dips} of {n_steps - 1} steps")
    chk.record("egress term is live from t=0: some satellite sees a gateway at the first step",
               hist[0, 0] > 0, f"max egress at t=0 {hist[0, 0] / 1e9:.3f} Gbps")


def check_predictor_real(chk: Checks, dataset, ground: GroundPrediction, sites, city_acc, gw_acc,
                         hours, tr: TrafficConfig, rcfg: RoutingConfig, vcfg: ExtensionValidationConfig,
                         n_steps: int) -> None:
    section("Predictor on the real shell: conservation, coverage, open loop")
    rate = expected_rate_bps(sites, hours, tr)
    worst = 0.0
    for t in range(n_steps):
        seen = np.array([acc.indptr[t + 1] > acc.indptr[t] for acc in city_acc])
        want = rate[t, seen].sum()
        worst = max(worst, abs(ground.demand_bps[t].sum() - want) / want)
    chk.record("demand is conserved: per step, satellites receive exactly what visible cities offer",
               worst < vcfg.sum_rel_tol, f"worst relative mismatch {worst:.2e} over {n_steps} steps")

    bad = 0
    for t in range(n_steps):
        seen = np.zeros(dataset.n_sats, dtype=bool)
        for acc in gw_acc:
            seen[acc.at(t)[0]] = True
        bad += int(((ground.egress_bps[t] > 0) != seen).sum())
    chk.record("egress is positive exactly where some gateway sees the satellite", bad == 0,
               f"{bad} mismatches over {n_steps} steps x {dataset.n_sats} satellites")

    a, b = (predict_ground(sites, city_acc, gw_acc, hours, dataset.n_sats,
                           dataclasses.replace(tr, seed=s), rcfg) for s in vcfg.seeds)
    chk.record("open loop: the prediction is identical for every traffic seed",
               np.array_equal(a.demand_bps, b.demand_bps) and np.array_equal(a.egress_bps, b.egress_bps),
               f"seeds {vcfg.seeds}")
    draws = [offered_bits(rate[:n_steps], dataset.config.step_s, dataclasses.replace(tr, seed=s))
             for s in vcfg.seeds]
    chk.record("seed check discriminates: the realised demand does differ between those seeds",
               not np.array_equal(*draws), f"seeds {vcfg.seeds}")

    lo, hi = (predict_ground(sites, city_acc, gw_acc, hours, dataset.n_sats,
                             dataclasses.replace(tr, peak_total_gbps=g), rcfg) for g in vcfg.loads_gbps)
    nl, nh = GroundNormaliser(lo), GroundNormaliser(hi)
    worst = 0.0
    for t in range(n_steps):
        nl.update(t)
        nh.update(t)
        worst = max(worst, float(np.abs(nl.demand(t) - nh.demand(t)).max()))
    chk.record("normalised demand does not depend on the offered-load level",
               worst < 1e-12 and not np.array_equal(lo.demand_bps, hi.demand_bps),
               f"loads {vcfg.loads_gbps} Gbps: raw demand differs, normalised max |diff| {worst:.2e}")

    pred = _forbidden_names(ROOT / "traffic" / "predict.py")
    ref = _forbidden_names(ROOT / "run_traffic.py")
    chk.record("open loop: the predictor never touches TrafficSim, offered_bits or backlog",
               not pred, f"found {sorted(pred) or 'none'} in traffic/predict.py")
    chk.record("scan discriminates: it does find those names in run_traffic.py", bool(ref),
               f"found {sorted(ref)}")
    leaks = [p for p in ("dotd.py", "scoring.py", "base.py", "config.py")
             if _imports_traffic(ROOT / "topology" / p)]
    chk.record("the topology package does not import the traffic layer", not leaks,
               f"importing modules: {leaks or 'none'}")


def report_pair_share(baseline: dict, active: dict, ground: GroundPrediction) -> None:
    section("Reported, not checked: links with a gateway in view at both ends")
    for kind in ("Greedy", "DoTD-forward"):
        b = gateway_pair_share(baseline[kind].linksets, ground)
        e = gateway_pair_share(active[kind]["egress"].linksets, ground)
        print(f"  {kind:<14} Phase 1 {b:.4f}   egress probe {e:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None, help="default: whole dataset")
    args = p.parse_args()

    vcfg = ExtensionValidationConfig()
    tr, rcfg = TrafficConfig(), RoutingConfig()
    print("=" * 72)
    print("  LEO CONSTELLATION SIMULATOR -- PHASE 2 EXTENSION VALIDATION")
    print(f"  probe weight {vcfg.probe_weight:g} (validation only; nothing is tuned)")
    print("=" * 72)

    chk = Checks()
    check_config(chk)
    check_pins(chk)
    check_predictor_pin(chk)
    check_direction(chk, vcfg)

    planes, per_plane = 10, 20
    cfg = SimConfig(selection=SelectionConfig(n_sats=planes * per_plane, n_planes=planes,
                                              sats_per_plane=per_plane)).resolve(ROOT)
    dataset = run(cfg, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)

    sites = load_sites(tr, ROOT)
    times = dataset_times(dataset)
    city_acc = [SparseAccess.from_access(compute_access(dataset, s, rcfg, times)) for s in sites.cities]
    gw_acc = [gateway_access(dataset, s, rcfg, tr, times) for s in sites.gateways]
    hours = utc_hours(dataset)
    ground = predict_ground(sites, city_acc, gw_acc, hours, dataset.n_sats, tr, rcfg)
    print(f"  {ground.label}")

    baseline = check_regression(chk, dataset, ground, n_steps)
    active = check_terms_active(chk, dataset, ground, baseline, n_steps, vcfg, TopologyValidationConfig())
    check_ground_normaliser(chk, ground, n_steps)
    check_predictor_real(chk, dataset, ground, sites, city_acc, gw_acc, hours, tr, rcfg, vcfg, n_steps)
    report_pair_share(baseline, active, ground)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
