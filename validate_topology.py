"""Stage-2 validation suite: topology selectors and metrics.

Same standard as ``validate.py``: every check prints PASS/FAIL with the measured
value and the bound, and every check drives the production code rather than
recomputing the quantity under test locally.

Several checks exist only to stop another check passing vacuously -- the same
role the grazing-pair check plays in stage 1. Each says so in its docstring.

    python validate_topology.py              # full 30-minute run
    python validate_topology.py --steps 300  # faster; used by mutants.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leosim.config import SPEED_OF_LIGHT_M_S, SelectionConfig, SimConfig
from leosim.dataset import LinkDataset
from leosim.geometry import segment_min_radius
from leosim.simulation import run
from topology import (
    ConstraintViolation,
    DoTD,
    Greedy,
    LinkSet,
    PlusGrid,
    TopologyConfig,
    assert_valid,
)
from topology.metrics import link_churn, metrics_frame, summary_table, timestep_metrics
from topology.grid import GAP_FACTORS, build_plane_structure, structural_pairs
from topology.scoring import RunningNormaliser, link_score
from validate import Checks, section

ROOT = Path(__file__).parent

# --------------------------------------------------------------------------
# Pinned scores, hand-computed from the synthetic dataset built in
# synthetic_score_dataset(). Hard-coded literals, not recomputed: a reference
# derived from the same code each run would drift along with that code.
#
# Pair (0, 2). Running maxima after t=0: capacity 2.0e9 bps (pair 0-1),
# latency 0.020 s (pair 0-1). Weights 0.4 / 0.4 / 0.2.
#
#   t=0  cap 1.50e9 -> 0.75   lat 0.005 -> 0.25   churn 0
#        0.4*0.75 + 0.4*(1-0.25) + 0.2*0          = 0.30 + 0.30 + 0.00 = 0.60
#
#   t=1  cap 0.75e9 -> 0.375  lat 0.005 -> 0.25   churn 1 (linked at t=0)
#        0.4*0.375 + 0.4*(1-0.25) + 0.2*1         = 0.15 + 0.30 + 0.20 = 0.65
#
#   The t=1 pin is what distinguishes a RUNNING max from a per-timestep max.
#   Every capacity halves at t=1, so a per-timestep max would renormalise
#   (0, 2) back to 0.75 and score it 0.80, not 0.65.
#
#   c/D variant, t=0. D = 1000 / 2000 / 4000 km for pairs 0-1 / 0-2 / 1-2, so
#   c/D peaks at the shortest link and (0, 2) normalises to 2000 -> 1000/2000
#   = 0.5 of the max:
#        0.4*0.75 + 0.4*(1-0.5) + 0.2*0           = 0.30 + 0.20 + 0.00 = 0.50
# --------------------------------------------------------------------------
PIN_SCORE_T0 = 0.60
PIN_SCORE_T1 = 0.65
PIN_SCORE_T1_PER_STEP_MAX = 0.80  # what the wrong normalisation would give
PIN_SCORE_C_OVER_D_T0 = 0.50
PIN_LINK_SCORE_FREE = 0.80  # link_score(0.75, 0.25, 1) at default weights
PIN_REL_TOL = 1e-12


@dataclass(frozen=True)
class TopologyValidationConfig:
    """Test tolerances. Separate from TopologyConfig for the same reason stage 1
    keeps ValidationConfig out of SimConfig: tuning a threshold must not look
    like a change to the algorithm."""

    # DoTD must differ from Greedy on at least this fraction of timesteps.
    # "At least one" is too weak: 1 differing step in 1800 would pass while the
    # lookahead is effectively inert.
    min_differ_fraction: float = 0.01

    # +Grid churn per step must be at most this multiple of the lowest churn
    # among the scored algorithms. "Far more stable" made numeric.
    grid_churn_ratio_max: float = 0.1

    # Window for re-running the ambiguity-flag variants; they only need to show
    # the flag is not inert, not reproduce the whole run.
    flag_window_steps: int = 120

    # +Grid plane grouping must satisfy spread < this fraction of the gap.
    plane_separability_fraction: float = 0.5

    # Timesteps at which metrics are independently recomputed.
    metric_spot_checks: int = 25


# --- synthetic datasets ------------------------------------------------------


def _synthetic(
    capacity: np.ndarray,
    latency: np.ndarray,
    distance: np.ndarray,
    visible: np.ndarray,
    step_s: float = 1.0,
) -> LinkDataset:
    """A LinkDataset with hand-chosen link values and no orbits behind it."""
    t, n = visible.shape[0], visible.shape[1]
    return LinkDataset(
        distance=distance.astype(np.float64),
        visible=visible.astype(bool),
        capacity=np.where(visible, capacity, 0.0),
        latency=np.where(visible, latency, 0.0),
        positions=np.zeros((t, n, 3)),
        altitudes=np.zeros((t, n)),
        sat_names=[f"S{i}" for i in range(n)],
        norad_ids=list(range(n)),
        plane_index=[-1] * n,
        times_utc_iso=["synthetic", "synthetic"],
        time_offsets_s=np.arange(t) * step_s,
        config=SimConfig(step_s=step_s, duration_s=t * step_s),
        selection_info={},
    )


def _sym(n: int, pairs: dict[tuple[int, int], float]) -> np.ndarray:
    m = np.zeros((n, n))
    for (i, j), v in pairs.items():
        m[i, j] = m[j, i] = v
    return m


def synthetic_score_dataset() -> LinkDataset:
    """3 satellites, 2 timesteps, all pairs visible. See the PIN_ block."""
    n = 3
    cap0 = _sym(n, {(0, 1): 2.0e9, (0, 2): 1.5e9, (1, 2): 1.0e9})
    lat = _sym(n, {(0, 1): 0.020, (0, 2): 0.005, (1, 2): 0.010})
    dist = _sym(n, {(0, 1): 1000.0, (0, 2): 2000.0, (1, 2): 4000.0})
    vis = _sym(n, {(0, 1): 1, (0, 2): 1, (1, 2): 1}).astype(bool)
    return _synthetic(
        capacity=np.stack([cap0, 0.5 * cap0]),
        latency=np.stack([lat, lat]),
        distance=np.stack([dist, dist]),
        visible=np.stack([vis, vis]),
    )


def synthetic_lookahead_dataset(n_steps: int = 30) -> LinkDataset:
    """Satellite 0 has one terminal and two candidates.

    - (0, 1): the better link now (2.0 Gbps), occluded from t=2 onward
    - (0, 2): slightly worse (1.8 Gbps), visible throughout
    - (1, 2): never visible

    An instantaneous scorer must take (0, 1) at t=0. Anything that looks ahead
    must take (0, 2), because (0, 1) is worth nothing for most of the horizon.
    """
    n = 3
    cap = _sym(n, {(0, 1): 2.0e9, (0, 2): 1.8e9})
    lat = _sym(n, {(0, 1): 0.005, (0, 2): 0.005})
    dist = _sym(n, {(0, 1): 1500.0, (0, 2): 1500.0})
    vis = np.zeros((n_steps, n, n), dtype=bool)
    vis[:, 0, 2] = vis[:, 2, 0] = True
    vis[:2, 0, 1] = vis[:2, 1, 0] = True
    return _synthetic(
        capacity=np.broadcast_to(cap, (n_steps, n, n)).copy(),
        latency=np.broadcast_to(lat, (n_steps, n, n)).copy(),
        distance=np.broadcast_to(dist, (n_steps, n, n)).copy(),
        visible=vis,
    )


def synthetic_plane_dataset(
    raans_deg: tuple[float, ...],
    sats_per_plane: int = 4,
    inclination_deg: float = 70.0,
    radius_km: float = 6900.0,
    n_steps: int = 10,
) -> LinkDataset:
    """Circular orbits at hand-chosen RAANs, for testing plane structure alone.

    Plane adjacency and the wrap decision are geometric rules over recovered
    RAAN. On the real dataset the sampled planes happen to sit at regular
    spacing, so a rule that ignored its own gap limit would produce the same
    answer and nothing would notice. Placing the planes by hand is what makes
    the limit testable: one plane is put far enough out that it must NOT be
    adjacent, and the span is kept narrow so the run must NOT wrap.

    Positions only -- link quantities are zero, since nothing here selects.
    """
    n = len(raans_deg) * sats_per_plane
    pos = np.zeros((n_steps, n, 3))
    inc = np.radians(inclination_deg)
    s_idx = 0
    for raan in raans_deg:
        om = np.radians(raan)
        # Ascending-node direction, and the in-plane axis 90 deg ahead of it.
        e1 = np.array([np.cos(om), np.sin(om), 0.0])
        h = np.array([np.sin(om) * np.sin(inc), -np.cos(om) * np.sin(inc), np.cos(inc)])
        e2 = np.cross(h, e1)
        for k in range(sats_per_plane):
            u0 = 2.0 * np.pi * k / sats_per_plane
            for t in range(n_steps):
                u = u0 + 0.01 * t  # advances with time, so r x v fixes the normal sign
                pos[t, s_idx] = radius_km * (np.cos(u) * e1 + np.sin(u) * e2)
            s_idx += 1

    zeros = np.zeros((n_steps, n, n))
    ds = _synthetic(
        capacity=zeros, latency=zeros, distance=zeros,
        visible=np.zeros((n_steps, n, n), dtype=bool),
    )
    return dataclasses.replace(ds, positions=pos)


def synthetic_discount_dataset(n_steps: int = 40) -> LinkDataset:
    """Satellite 0 has one terminal and two candidates, tied at t=0.

    - (0, 1): 2.0 Gbps over t = 1..9, 0.1 Gbps from t = 10
    - (0, 2): 0.1 Gbps over t = 1..9, 2.0 Gbps from t = 10
    - both 1.0 Gbps at t = 0, so the instantaneous score is a tie

    The tie is the point. With different values at t=0 the present-step score
    dominates the accumulation term and the choice is made before the discount
    is consulted at all. Tied, the two links are also equal over the window as
    an unweighted mean, so only the shape of the discount separates them:
    weights that fall with offset take (0, 1), weights that rise take (0, 2).
    """
    n = 3
    tie = _sym(n, {(0, 1): 1.0e9, (0, 2): 1.0e9})
    near = _sym(n, {(0, 1): 2.0e9, (0, 2): 0.1e9})
    far = _sym(n, {(0, 1): 0.1e9, (0, 2): 2.0e9})
    lat = _sym(n, {(0, 1): 0.005, (0, 2): 0.005})
    dist = _sym(n, {(0, 1): 1500.0, (0, 2): 1500.0})
    vis = _sym(n, {(0, 1): 1, (0, 2): 1}).astype(bool)

    cap = np.stack([tie if t == 0 else (near if t < 10 else far) for t in range(n_steps)])
    return _synthetic(
        capacity=cap,
        latency=np.broadcast_to(lat, (n_steps, n, n)).copy(),
        distance=np.broadcast_to(dist, (n_steps, n, n)).copy(),
        visible=np.broadcast_to(vis, (n_steps, n, n)).copy(),
    )


def synthetic_impulse_dataset(offset: int, n_steps: int = 30) -> LinkDataset:
    """Pair (0, 1) is visible at exactly one timestep and nowhere else.

    Feeding this to the forward term at t=0 returns that offset's normalised
    weight times a constant that does not depend on the offset, so sweeping the
    offset reads the discount curve straight out of the production code rather
    than recomputing exp(-h/H) in the test -- which would pass whatever the
    production code did.
    """
    n = 3
    cap = _sym(n, {(0, 1): 2.0e9})
    lat = _sym(n, {(0, 1): 0.005})
    dist = _sym(n, {(0, 1): 1500.0})
    vis = np.zeros((n_steps, n, n), dtype=bool)
    vis[offset, 0, 1] = vis[offset, 1, 0] = True
    return _synthetic(
        capacity=np.broadcast_to(cap, (n_steps, n, n)).copy(),
        latency=np.broadcast_to(lat, (n_steps, n, n)).copy(),
        distance=np.broadcast_to(dist, (n_steps, n, n)).copy(),
        visible=vis,
    )


# --- independent reference computations --------------------------------------


def edge_set(ls: LinkSet) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in ls.edges}


def bfs_components(adjacency: np.ndarray) -> int:
    """Connected components by breadth-first search. Deliberately not scipy,
    which the metrics module uses."""
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        seen[s] = True
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.nonzero(adjacency[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
    return count


# --- check groups ---------------------------------------------------------------


def check_constraints(chk: Checks, dataset: LinkDataset, results: dict, tcfg: TopologyConfig) -> None:
    section("Hard constraints, every algorithm, every timestep")
    geo = dataset.config.geometry

    for name, linksets in results.items():
        n_links = n_asym = n_self = n_over = n_invisible = n_through_earth = n_over_range = 0
        min_clearance = np.inf
        max_len = 0.0
        max_deg = 0
        for ls in linksets:
            a = ls.adjacency
            n_links += ls.n_links
            n_asym += int((a != a.T).sum())
            n_self += int(np.diagonal(a).sum())
            deg = a.sum(axis=1)
            max_deg = max(max_deg, int(deg.max()))
            n_over += int((deg > tcfg.u_max).sum())
            n_invisible += int((a & ~np.asarray(dataset.visible[ls.t])).sum() // 2)

            # Occlusion and range recomputed from raw positions, not read from
            # stage-1 `visible`, so this is independent of the array the
            # selectors were given.
            e = ls.edges
            if len(e):
                pos = np.asarray(dataset.positions[ls.t], dtype=np.float64)
                p1, p2 = pos[e[:, 0]], pos[e[:, 1]]
                clearance = segment_min_radius(p1, p2)
                length = np.linalg.norm(p2 - p1, axis=1)
                n_through_earth += int((clearance < geo.blocking_radius_km).sum())
                n_over_range += int((length > geo.max_range_km).sum())
                min_clearance = min(min_clearance, float(clearance.min()))
                max_len = max(max_len, float(length.max()))

        chk.record(
            f"{name}: every link visible in stage-1 data",
            n_invisible == 0,
            f"{n_invisible} of {n_links} link-timesteps not visible",
        )
        chk.record(
            f"{name}: no link passes through the Earth (recomputed from positions)",
            n_through_earth == 0,
            f"{n_through_earth} links below {geo.blocking_radius_km:.0f} km; "
            f"lowest segment clearance {min_clearance:.1f} km",
        )
        chk.record(
            f"{name}: no link beyond max range",
            n_over_range == 0,
            f"{n_over_range} links over {geo.max_range_km:.0f} km; longest {max_len:.1f} km",
        )
        chk.record(
            f"{name}: degree <= U={tcfg.u_max} for every satellite and timestep",
            n_over == 0,
            f"max degree {max_deg}; {n_over} sat-timesteps over U",
        )
        chk.record(
            f"{name}: link set symmetric, no self-links",
            n_asym == 0 and n_self == 0,
            f"{n_asym // 2} asymmetric pairs, {n_self} self-links",
        )


def check_constraints_non_vacuous(chk: Checks, results: dict, tcfg: TopologyConfig) -> None:
    """Degree <= U passes trivially for an empty topology. Require that the
    scored algorithms actually fill terminals, so that check has teeth."""
    section("Constraint checks are not vacuous")
    for name, linksets in results.items():
        if name == "+Grid":
            continue
        at_u = np.mean([np.mean(ls.degree() == tcfg.u_max) for ls in linksets])
        chk.between(
            f"{name}: satellites at full degree U, averaged over run",
            at_u, 0.5, 1.0,
            extra="below 0.5 would mean the degree cap is rarely exercised",
        )


def check_enforcement_rejects(chk: Checks, dataset: LinkDataset, tcfg: TopologyConfig) -> None:
    """assert_valid runs inside every step(). If it were a no-op, every
    constraint check above would still pass on well-behaved selectors and
    nobody would learn the guard was broken. Feed it known-bad sets."""
    section("Constraint enforcement rejects invalid link sets")
    n = dataset.n_sats
    vis0 = np.asarray(dataset.visible[0])
    vi, vj = np.argwhere(np.triu(vis0, k=1))[0]
    ii, ij = np.argwhere(np.triu(~vis0, k=1))[0]

    def rejects(label: str, adj: np.ndarray, because: str | None = None) -> None:
        try:
            assert_valid(LinkSet(adj, 0), dataset, tcfg.u_max)
            chk.record(f"rejects {label}", False, "assert_valid accepted it")
        except ConstraintViolation as e:
            msg = str(e).splitlines()[0]
            ok = because is None or because in msg
            detail = f"raised: {msg}"
            if because is not None and not ok:
                detail += f" -- rejected, but not for {because!r}"
            chk.record(f"rejects {label}", ok, detail)

    a = np.zeros((n, n), dtype=bool)
    a[vi, vj] = True
    rejects("asymmetric link", a)

    # A self-pair is also never visible, so a visibility complaint proves
    # nothing about the diagonal check: the reason is pinned, not just the
    # raise. Sat 0 is linked to a genuinely visible partner as well, so the
    # only thing left to object to is the diagonal itself.
    a = np.zeros((n, n), dtype=bool)
    a[0, 0] = True
    a[vi, vj] = a[vj, vi] = True
    rejects("self-link", a, because="self-link")

    a = np.zeros((n, n), dtype=bool)
    a[ii, ij] = a[ij, ii] = True
    rejects("link not visible in stage-1 data", a)

    # Degree U+1 using only visible partners, so degree is the sole violation.
    hub = int(np.argmax(vis0.sum(axis=1)))
    partners = np.nonzero(vis0[hub])[0][: tcfg.u_max + 1]
    a = np.zeros((n, n), dtype=bool)
    a[hub, partners] = a[partners, hub] = True
    if len(partners) == tcfg.u_max + 1:
        rejects(f"degree U+1={tcfg.u_max + 1}", a)
    else:
        chk.record("rejects degree U+1", False, f"sat {hub} has only {len(partners)} visible partners")

    a = np.zeros((n, n), dtype=bool)
    for k, p in enumerate(partners[: tcfg.u_max]):
        a[hub, p] = a[p, hub] = True
    try:
        assert_valid(LinkSet(a, 0), dataset, tcfg.u_max)
        chk.record("accepts a valid degree-U set", True, f"sat {hub} at degree {tcfg.u_max}")
    except ConstraintViolation as e:
        chk.record("accepts a valid degree-U set", False, str(e))

    sel = Greedy(dataset, tcfg)
    sel.reset()
    try:
        sel.step(5)
        chk.record("step() refuses a non-consecutive timestep", False, "t=5 accepted after reset")
    except ValueError:
        chk.record("step() refuses a non-consecutive timestep", True, "t=5 after reset raised ValueError")


def check_score_pins(chk: Checks) -> None:
    section("Score arithmetic: hand-checkable pins")
    tcfg = TopologyConfig(u_max=4)

    chk.close_to(
        "link_score(cap_norm=0.75, lat_norm=0.25, churn=1)",
        float(link_score(0.75, 0.25, 1.0, tcfg)), PIN_LINK_SCORE_FREE, PIN_REL_TOL,
        extra="0.4*0.75 + 0.4*(1-0.25) + 0.2*1 = 0.30 + 0.30 + 0.20",
    )

    ds = synthetic_score_dataset()
    sel = Greedy(ds, tcfg)
    sel.reset()
    sel.step(0)
    chk.close_to(
        "end-to-end score, pair (0,2), t=0 (no churn)",
        float(sel.last_score[0, 2]), PIN_SCORE_T0, PIN_REL_TOL,
        extra="cap 1.5/2.0 Gbps, lat 5/20 ms: 0.4*0.75 + 0.4*0.75 + 0",
    )
    linked = bool(sel.prev.adjacency[0, 2])
    sel.step(1)
    chk.record(
        "synthetic pair (0,2) was linked at t=0, so churn term applies at t=1",
        linked, f"adjacency[0,2] at t=0 = {linked}",
    )
    chk.close_to(
        "end-to-end score, pair (0,2), t=1 (running max, churn=1)",
        float(sel.last_score[0, 2]), PIN_SCORE_T1, PIN_REL_TOL,
        extra=(
            "cap 0.75/2.0 Gbps (running max holds 2.0), lat 5/20 ms, churn 1:\n"
            "0.4*0.375 + 0.4*0.75 + 0.2 = 0.15 + 0.30 + 0.20\n"
            f"a per-timestep max would give {PIN_SCORE_T1_PER_STEP_MAX:.2f}"
        ),
    )

    sel = Greedy(ds, dataclasses.replace(tcfg, latency_formula="c_over_d"))
    sel.reset()
    sel.step(0)
    chk.close_to(
        "end-to-end score, pair (0,2), t=0, paper's c/D latency",
        float(sel.last_score[0, 2]), PIN_SCORE_C_OVER_D_T0, PIN_REL_TOL,
        extra=(
            f"c/D: {SPEED_OF_LIGHT_M_S / 1e6:.3f} (1000 km, max) vs "
            f"{SPEED_OF_LIGHT_M_S / 2e6:.3f} (2000 km) -> 0.5\n"
            "0.4*0.75 + 0.4*(1-0.5) = 0.30 + 0.20"
        ),
    )


def check_running_max(chk: Checks, dataset: LinkDataset, selectors: dict, n_steps: int) -> None:
    section("Running maximum (Eq. 12)")
    sel = selectors["DoTD-forward"]
    hist = np.array([(h[1], h[2]) for h in sel.normaliser.history])

    for col, label in [(0, "capacity"), (1, "latency")]:
        drops = np.diff(hist[:, col])
        n_drop = int((drops < 0).sum())
        chk.record(
            f"running max {label} non-decreasing at every step",
            n_drop == 0,
            f"{n_drop} decreases over {len(hist)} steps; "
            f"range {hist[0, col]:.6g} -> {hist[-1, col]:.6g}",
        )

    # A constant equal to the global max over the whole run is also
    # non-decreasing, and would leak the future into normalisation. Pin
    # causality: at each checkpoint the max must equal the max over [0, t].
    checkpoints = np.unique(np.linspace(0, n_steps - 1, 10).astype(int))
    worst = 0.0
    for t in checkpoints:
        vis = np.asarray(dataset.visible[: t + 1])
        cap = np.asarray(dataset.capacity[: t + 1], dtype=np.float64)[vis].max()
        worst = max(worst, abs(hist[t, 0] - cap) / cap)
    chk.record(
        "running max capacity at t equals max over timesteps [0, t] (causal)",
        worst < 1e-12,
        f"worst relative mismatch {worst:.2e} over {len(checkpoints)} checkpoints",
    )

    # Discrimination: the check above is only meaningful if a per-timestep max
    # would actually behave differently on this data.
    per_step = np.array([
        np.asarray(dataset.capacity[t], dtype=np.float64)[np.asarray(dataset.visible[t])].max()
        for t in range(n_steps)
    ])
    n_dips = int((np.diff(per_step) < 0).sum())
    chk.record(
        "monotonicity check discriminates: per-timestep max does decrease here",
        n_dips > 0,
        f"per-timestep max capacity decreases at {n_dips} of {n_steps - 1} steps",
    )


def check_normalisation_bounds(
    chk: Checks, dataset: LinkDataset, tcfg: TopologyConfig, n_steps: int
) -> None:
    """The clip in RunningNormaliser.normalise is load-bearing for the forward
    lookahead only. At the current timestep the running max already absorbed
    that step, so the clip is a no-op and no check on present-step scores can
    see it. Future samples are read against the *causal* max, so a link better
    than anything yet seen normalises above 1 and outscores the scale it is
    measured on -- which is what these two checks pin."""
    section("Normalisation stays inside [0, 1]")

    norm = RunningNormaliser(dataset, tcfg.latency_formula)
    norm.update(0)
    cap_n, lat_n = norm.normalise(
        np.array([[2.0 * norm.max_capacity]]), np.array([[2.0 * norm.max_latency]])
    )
    chk.record(
        "a value at twice the running max normalises to 1, not above",
        float(cap_n.max()) <= 1.0 and float(lat_n.max()) <= 1.0,
        f"capacity -> {float(cap_n.max()):.6f}, latency -> {float(lat_n.max()):.6f}, "
        "both required <= 1",
    )

    # End to end: the assembled forward term over the real dataset. The clip
    # only binds while the running max is still climbing -- on this shell that
    # is the first ~600 timesteps -- so the whole available run is swept rather
    # than a window, and the number of binding timesteps is asserted, not
    # assumed.
    sel = DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward"))
    sel.reset()
    worst = 0.0
    binds = 0
    for t in range(n_steps):
        sel.step(t)
        worst = max(worst, float(sel._forward_term(t).max()))
        ahead = sel._step_offsets() + t
        ahead = ahead[ahead < dataset.n_steps]
        future_cap = float(np.asarray(dataset.capacity[ahead], dtype=np.float64).max())
        if future_cap > sel.normaliser.max_capacity:
            binds += 1
    chk.record(
        "forward lookahead term stays within [0, 1] over the real dataset",
        worst <= 1.0 + PIN_REL_TOL,
        f"largest forward term over {n_steps} timesteps: {worst:.6f}",
    )
    chk.record(
        "clip check discriminates: future links do exceed the causal max here",
        binds > 0,
        f"{binds} of {n_steps} timesteps have a sample inside the horizon above "
        f"the running max at that step",
    )


def check_lookahead(
    chk: Checks, results: dict, selectors: dict, vcfg: TopologyValidationConfig
) -> None:
    section("Lookahead is doing something")
    greedy = results["Greedy"]
    for name in ("DoTD-forward", "DoTD-backward"):
        differ = np.array([
            not np.array_equal(a.adjacency, b.adjacency) for a, b in zip(results[name], greedy)
        ])
        chk.between(
            f"{name} differs from Greedy on a fraction of timesteps",
            float(differ.mean()), vcfg.min_differ_fraction, 1.0,
            extra=f"{int(differ.sum())} of {len(differ)} timesteps differ",
        )

    ds = synthetic_lookahead_dataset()
    cfg = TopologyConfig(u_max=1, horizon_s=10.0, horizon_stride_s=1.0)
    g = Greedy(ds, cfg)
    g.reset()
    gl = g.step(0)
    d = DoTD(ds, dataclasses.replace(cfg, lookahead="forward"))
    d.reset()
    dl = d.step(0)
    chk.record(
        "synthetic: Greedy takes the link that is better now but occluded from t=2",
        bool(gl.adjacency[0, 1]),
        f"Greedy edges at t=0: {sorted(edge_set(gl))}",
    )
    chk.record(
        "synthetic: forward DoTD anticipates and takes the persistent link",
        bool(dl.adjacency[0, 2]),
        f"DoTD-forward edges at t=0: {sorted(edge_set(dl))}",
    )

    # Direction of the discount. The occlusion dataset above is satisfied by
    # any weighting that looks past the current step at all, including one that
    # weights the far future most -- so it cannot see an inverted discount.
    dcfg = TopologyConfig(u_max=1, horizon_s=20.0, horizon_stride_s=1.0,
                          lookahead="forward")
    dd = synthetic_discount_dataset()
    near = DoTD(dd, dcfg)
    near.reset()
    nl = near.step(0)
    chk.record(
        "synthetic: tied now, forward DoTD takes the link that is good soonest",
        bool(nl.adjacency[0, 1]),
        f"edges at t=0: {sorted(edge_set(nl))}; (0,1) strong over t=1..9, "
        f"(0,2) strong from t=10, equal at t=0 and over the window unweighted",
    )

    # The discount curve itself, read out of the production term one offset at
    # a time. Recomputing exp(-h/H) in the test would pass whatever the
    # production code weighted by.
    probes = [1, 5, 10, 15, 20]
    curve = []
    for off in probes:
        ids = synthetic_impulse_dataset(off)
        probe = DoTD(ids, dcfg)
        probe.reset()
        probe.normaliser.update(0)
        curve.append(float(probe._forward_term(0)[0, 1]))
    chk.record(
        "forward discount falls strictly with offset (near future weighted most)",
        all(a > b for a, b in zip(curve, curve[1:])) and curve[0] > 0,
        f"term at offsets {probes}: {[round(v, 6) for v in curve]}",
    )

    sel = selectors["Greedy"]
    chk.record(
        "Greedy has no accumulation term (priority == score at last step)",
        sel.last_priority is not None and np.array_equal(sel.last_priority, sel.last_score),
        f"lookahead={sel.config.lookahead}",
    )

    acc = selectors["DoTD-backward"]
    ds_real = acc.dataset
    acc.reset()
    lo, hi = np.inf, -np.inf
    for t in range(min(len(results["DoTD-backward"]), ds_real.n_steps)):
        acc.step(t)
        lo, hi = min(lo, acc.accumulated.min()), max(hi, acc.accumulated.max())
    chk.record(
        "backward accumulation stays within [0, 1] (bounded, per paper's finiteness)",
        lo >= 0.0 and hi <= 1.0,
        f"accumulated score range over run: {lo:.4f} .. {hi:.4f}",
    )

    for name, s in selectors.items():
        if getattr(s, "last_priority", None) is None:
            continue
        p = s.last_priority
        chk.record(
            f"{name}: priority matrix symmetric (i and j rank the link identically)",
            np.allclose(p, p.T, rtol=0, atol=0),
            f"max |p - p.T| = {float(np.abs(p - p.T).max()):.3g}",
        )


def check_matching(chk: Checks, dataset: LinkDataset, results: dict, selectors: dict, tcfg: TopologyConfig) -> None:
    section("Matching behaves as specified")
    for name in ("Greedy", "DoTD-forward"):
        # Maximality: greedy b-matching never leaves a visible pair unused when
        # both endpoints still have spare terminals.
        violations = 0
        for ls in results[name]:
            spare = ls.degree() < tcfg.u_max
            vis = np.asarray(dataset.visible[ls.t])
            violations += int(np.triu(vis & ~ls.adjacency & spare[:, None] & spare[None, :], 1).sum())
        chk.record(
            f"{name}: maximal -- no unused visible pair with both endpoints below U",
            violations == 0,
            f"{violations} such pairs across run",
        )

    # The single highest-priority visible edge must always be selected under
    # "max": nothing can have blocked it. Checks the sort direction.
    sel = selectors["Greedy"]
    sel.reset()
    misses = 0
    steps = len(results["Greedy"])
    for t in range(steps):
        ls = sel.step(t)
        vis = np.asarray(dataset.visible[t])
        p = np.where(np.triu(vis, 1), sel.last_priority, -np.inf)
        i, j = np.unravel_index(np.argmax(p), p.shape)
        misses += int(not ls.adjacency[i, j])
    chk.record(
        "Greedy (max): top-priority visible edge selected at every timestep",
        misses == 0,
        f"missed at {misses} of {steps} timesteps",
    )


def check_ambiguity_flags(chk: Checks, dataset: LinkDataset, tcfg: TopologyConfig, vcfg: TopologyValidationConfig) -> None:
    """Each paper-ambiguity flag must change the output, or it is not really
    exposing the alternative reading."""
    section("Paper-ambiguity flags are live")
    n = min(vcfg.flag_window_steps, dataset.n_steps)
    base_sel = Greedy(dataset, tcfg)
    base = base_sel.run(n)
    base_mean = np.mean([base_sel_score for base_sel_score in _selected_scores(dataset, base, tcfg)])

    amin = Greedy(dataset, dataclasses.replace(tcfg, objective="min"))
    amin_links = amin.run(n)
    differ = np.mean([not np.array_equal(a.adjacency, b.adjacency) for a, b in zip(amin_links, base)])
    amin_mean = np.mean(_selected_scores(dataset, amin_links, dataclasses.replace(tcfg, objective="min")))
    chk.record(
        "objective=min selects a different topology than max",
        differ > 0.5,
        f"differs on {differ:.0%} of {n} timesteps",
    )
    chk.record(
        "objective=min selects lower-scoring links than max",
        amin_mean < base_mean,
        f"mean selected instantaneous score: min {amin_mean:.4f} vs max {base_mean:.4f}",
    )

    cod = Greedy(dataset, dataclasses.replace(tcfg, latency_formula="c_over_d")).run(n)
    differ = np.mean([not np.array_equal(a.adjacency, b.adjacency) for a, b in zip(cod, base)])
    len_base = np.mean([np.asarray(dataset.distance[ls.t])[ls.adjacency].mean() for ls in base])
    len_cod = np.mean([np.asarray(dataset.distance[ls.t])[ls.adjacency].mean() for ls in cod])
    chk.record(
        "latency_formula=c_over_d selects a different topology than d_over_c",
        differ > 0.5,
        f"differs on {differ:.0%} of {n} timesteps",
    )
    chk.record(
        "c_over_d rewards longer links, as the inversion predicts",
        len_cod > len_base,
        f"mean selected link length: c/D {len_cod:.0f} km vs D/c {len_base:.0f} km",
    )


def _selected_scores(dataset: LinkDataset, linksets: list[LinkSet], tcfg: TopologyConfig) -> list[float]:
    """Mean instantaneous score of selected links per step, rescored with a
    fresh max-objective Greedy so both variants are measured on one scale."""
    ref = Greedy(dataset, dataclasses.replace(tcfg, objective="max"))
    ref.reset()
    out = []
    for ls in linksets:
        ref.step(ls.t)
        if ls.n_links:
            out.append(float(ref.last_score[ls.adjacency].mean()))
    return out


def check_grid(
    chk: Checks, dataset: LinkDataset, results: dict, selectors: dict, summary, tcfg: TopologyConfig,
    vcfg: TopologyValidationConfig,
) -> None:
    section("+Grid structure")
    g: PlusGrid = selectors["+Grid"]
    s = g.structure
    d = s.diagnostics

    chk.record(
        "orbital planes are cleanly separable",
        d["max_plane_spread_deg"] < vcfg.plane_separability_fraction * d["min_plane_gap_deg"],
        f"{d['n_planes']} planes; max intra-plane spread {d['max_plane_spread_deg']:.3f} deg "
        f"vs min inter-plane gap {d['min_plane_gap_deg']:.3f} deg "
        f"(required spread < {vcfg.plane_separability_fraction:g} x gap)\n"
        f"stage-1 plane_index clusters contain {d['stage1_labels']} real planes",
    )

    # The two layers identify planes independently -- stage 1 from catalogue
    # RAAN at each satellite's own epoch, stage 2 from orbit normals fitted to
    # the propagated trajectory. Agreement one-for-one is the check that both
    # are resolving real planes and not some clustering of their own. It also
    # pins the selection: this is exactly what a forced RAAN-cluster count got
    # wrong, reporting 5 planes of 20 for what were really 28 planes of 1 to 13.
    per_label = d["stage1_labels"]
    sizes = sorted((len(m) for m in s.members.values()), reverse=True)
    chk.record(
        "each stage-1 plane label resolves to exactly one real plane",
        set(per_label.values()) == {1}
        and d["n_planes"] == len(per_label),
        f"{d['n_planes']} planes recovered from normals, {len(per_label)} stage-1 labels; "
        f"real planes per label {per_label}; "
        f"recovered plane populations {sizes}",
    )

    plane = np.asarray(d["plane_of_sat"])
    built = int(g.grid.sum() // 2)
    expected = s.expected_link_count()
    chk.record(
        "structural link count matches count derived from plane sizes and adjacency",
        built == expected,
        f"grid has {built} links, plane sizes and adjacency imply {expected}",
    )

    ii, jj = np.nonzero(np.triu(g.grid, 1))
    wrong_in = wrong_cross = 0
    for i, j in zip(ii, jj):
        pi_, pj_ = int(plane[i]), int(plane[j])
        if g.kind[i, j] == 1 and pi_ != pj_:
            wrong_in += 1
        if g.kind[i, j] == 2 and pj_ not in s.adjacent[pi_]:
            wrong_cross += 1
    chk.record(
        "in-plane links join same-plane sats; cross-plane links join adjacent planes only",
        wrong_in == 0 and wrong_cross == 0,
        f"{wrong_in} in-plane links across planes, {wrong_cross} cross links to non-adjacent planes",
    )

    per_sat_in = ((g.kind == 1) & g.grid).sum(axis=1)
    per_sat_cross_by_plane_ok = True
    for sidx in range(dataset.n_sats):
        partners = np.nonzero((g.kind[sidx] == 2) & g.grid[sidx])[0]
        planes_hit = plane[partners].tolist()
        if len(planes_hit) != len(set(planes_hit)):
            per_sat_cross_by_plane_ok = False
    chk.record(
        "structural slots per sat: <= 2 in-plane, <= 1 per adjacent plane",
        int(per_sat_in.max()) <= 2 and per_sat_cross_by_plane_ok,
        f"max in-plane per sat {int(per_sat_in.max())}; one partner per adjacent plane: {per_sat_cross_by_plane_ok}",
    )

    # "Exactly 4 links where geometry allows", made precise: realised degree
    # equals the number of the satellite's grid slots that are visible now.
    mismatches = 0
    four_possible = 0
    for ls in results["+Grid"]:
        vis_slots = (g.grid & np.asarray(dataset.visible[ls.t])).sum(axis=1)
        mismatches += int((ls.degree() != vis_slots).sum())
        four_possible += int((vis_slots == 4).sum())
    chk.record(
        "+Grid realised degree == visible structural slots, every sat, every timestep",
        mismatches == 0,
        f"{mismatches} mismatches",
    )
    frac4 = four_possible / (len(results["+Grid"]) * dataset.n_sats)
    chk.record(
        "non-vacuous: some satellites have all 4 grid slots visible",
        four_possible > 0,
        f"{four_possible} sat-timesteps ({frac4:.1%}) where geometry allows 4; "
        f"all reached degree 4 per check above",
    )

    churn_grid = float(summary.loc["+Grid", "churn_per_step"])
    others = {k: float(summary.loc[k, "churn_per_step"]) for k in summary.index if k != "+Grid"}
    lowest_name = min(others, key=others.get)
    ratio = churn_grid / others[lowest_name] if others[lowest_name] > 0 else np.inf
    chk.record(
        "+Grid churn far below every scored algorithm",
        ratio <= vcfg.grid_churn_ratio_max,
        f"+Grid {churn_grid:.4f}/step vs lowest scored ({lowest_name}) {others[lowest_name]:.4f}/step; "
        f"ratio {ratio:.4f}, required <= {vcfg.grid_churn_ratio_max:g}",
    )

    # No scoring at all: changing every scoring knob must leave +Grid unchanged.
    # Built from tcfg, not a fresh config: grid_gaps is structural, and a fresh
    # config would silently reset it to the default and compare two grids.
    n = min(vcfg.flag_window_steps, dataset.n_steps)
    alt = PlusGrid(
        dataset,
        dataclasses.replace(tcfg, w1_capacity=0.9, w2_latency=0.05, w3_churn=0.05, objective="min",
                            latency_formula="c_over_d", lookahead="backward"),
    ).run(n)
    same = all(np.array_equal(a.adjacency, b.adjacency) for a, b in zip(alt, results["+Grid"][:n]))
    chk.record(
        "+Grid output invariant to every scoring parameter",
        same,
        f"compared {n} timesteps under changed weights, objective, latency formula, lookahead",
    )


def check_plane_adjacency(chk: Checks, dataset: LinkDataset) -> None:
    """Plane adjacency and the wrap decision, on planes placed by hand.

    Both rules are gap comparisons, and on the real sample the planes sit at
    near-uniform spacing where a rule that ignored its limit would give the
    same answer. So the limit is tested on a constellation built to need it:
    four planes 10 degrees apart plus one far out at 120, spanning only a third
    of the shell.

      - 30 -> 120 is a 90 degree gap against a 15 degree limit: NOT adjacent
      - the span does not close, so the first and last plane do NOT wrap

    Both are what the real sample also needs, and neither is visible there.
    """
    section("Plane adjacency respects its gap limit")
    raans = (0.0, 10.0, 20.0, 30.0, 120.0)
    st = build_plane_structure(synthetic_plane_dataset(raans, sats_per_plane=4))

    chk.record(
        "hand-placed planes are recovered one-for-one",
        len(st.plane_ids) == len(raans),
        f"{len(st.plane_ids)} planes recovered from {len(raans)} placed; "
        f"RAANs {[round(st.plane_raan_deg[p], 1) for p in st.plane_ids]}",
    )

    by_raan = sorted(st.plane_ids, key=lambda p: st.plane_raan_deg[p])
    far = by_raan[-1]
    near = by_raan[-2]
    chk.record(
        "a plane across a 90 degree gap is not adjacent",
        far not in st.adjacent[near] and near not in st.adjacent[far],
        f"gap limit {st.diagnostics['adjacent_gap_limit_deg']:.1f} deg; "
        f"plane at {st.plane_raan_deg[far]:.0f} deg adjacency: "
        f"{[round(st.plane_raan_deg[q]) for q in st.adjacent[far]]}",
    )
    chk.record(
        "first and last plane do not wrap across an unsampled arc",
        st.wraps is False,
        f"wrap gap {360.0 - (st.plane_raan_deg[by_raan[-1]] - st.plane_raan_deg[by_raan[0]]):.0f} deg "
        f"against limit {st.diagnostics['adjacent_gap_limit_deg']:.1f} deg; wraps={st.wraps}",
    )
    expected = {
        (min(a, b), max(a, b))
        for a, b in zip(by_raan[:3], by_raan[1:4])
    }
    got = {(min(p, q), max(p, q)) for p in st.plane_ids for q in st.adjacent[p]}
    got_r = sorted((round(st.plane_raan_deg[a]), round(st.plane_raan_deg[b])) for a, b in got)
    chk.record(
        "adjacency is exactly the consecutive pairs inside the limit",
        got == expected,
        f"adjacent pairs by RAAN: {got_r}, expected the three 10 degree steps",
    )


def check_baseline_connected(
    chk: Checks, dataset: LinkDataset, results: dict, selectors: dict, tcfg: TopologyConfig
) -> None:
    """Every selector must produce one connected network at every timestep.

    A disconnected topology fails nothing else. Every path is still over
    selected links and every hop count is still correct -- the paths just
    detour around, or cannot cross, a partition. On the full 34-plane shell
    under grid_gaps="strict" +Grid split in two and its New York -> San
    Francisco count went to 17 hops, falling to 5 once the halves were joined;
    no check noticed. The same blind spot would hide stranded demand in Phase 2.

    So this FAILS whenever any selector is disconnected, on purpose. At 601
    satellites the scored selectors do fail it: they form closed 5-cliques from
    counter-rotating planes (each other's four best candidates). That is a
    finding about per-link scored selection, reported as such -- not a reason to
    relax the check or to change selection. Components are counted by the
    suite's own BFS, not by the metrics module.
    """
    section("Every selector's topology is connected")
    s = selectors["+Grid"].structure
    for name, linksets in results.items():
        comps = [bfs_components(ls.adjacency) for ls in linksets]
        bad = [t for t, c in enumerate(comps) if c != 1]
        extra = (f"; grid_gaps={tcfg.grid_gaps}, wraps={s.wraps}, "
                 f"gap limit {s.diagnostics['adjacent_gap_limit_deg']:.1f} deg") if name == "+Grid" else ""
        chk.record(
            f"{name}: one connected component at every timestep",
            not bad,
            f"{len(bad)} of {len(comps)} timesteps disconnected (max {max(comps)} components){extra}",
        )

    # Discrimination. On the 10-plane window the chosen planes avoid both empty
    # slots of the shell, so the check above passes whether or not the gap
    # policy does anything. Rebuild the real shell's layout by hand: planes
    # every 10 deg with the slots at 160 and 0 deg left empty, the two 20 deg
    # gaps that split the real 34-plane +Grid.
    raans = tuple(float(r) for r in range(10, 360, 10) if r != 160)
    ds = synthetic_plane_dataset(raans, sats_per_plane=4)
    parts = {}
    for policy in ("strict", "bridge"):
        st = build_plane_structure(ds, GAP_FACTORS[policy])
        grid, _ = structural_pairs(st)
        parts[policy] = (bfs_components(grid), st.wraps)
    chk.record(
        "strict gap policy splits a full shell with two empty slots",
        parts["strict"] == (2, False),
        f"{len(raans)} planes, empty slots at 160 and 0 deg: "
        f"strict gives {parts['strict'][0]} components, wraps={parts['strict'][1]}",
    )
    chk.record(
        "bridge gap policy joins the same shell into one wrapped grid",
        parts["bridge"] == (1, True),
        f"bridge gives {parts['bridge'][0]} components, wraps={parts['bridge'][1]}",
    )

    # Bridging is for one empty slot, not any gap: it must still refuse the
    # 90 deg step that check_plane_adjacency uses.
    st = build_plane_structure(
        synthetic_plane_dataset((0.0, 10.0, 20.0, 30.0, 120.0), sats_per_plane=4),
        GAP_FACTORS["bridge"],
    )
    by_raan = sorted(st.plane_ids, key=lambda p: st.plane_raan_deg[p])
    chk.record(
        "bridge gap policy still rejects a 90 degree gap",
        by_raan[-1] not in st.adjacent[by_raan[-2]] and st.wraps is False,
        f"gap limit {st.diagnostics['adjacent_gap_limit_deg']:.1f} deg; "
        f"plane at 120 deg adjacency {[round(st.plane_raan_deg[q]) for q in st.adjacent[by_raan[-1]]]}",
    )
    # PlusGrid must take its limit from the flag. Everything above calls
    # build_plane_structure directly, and the suite's own +Grid runs one policy
    # only, so a PlusGrid that ignored grid_gaps would pass all of it.
    other = "bridge" if tcfg.grid_gaps == "strict" else "strict"
    alt = PlusGrid(dataset, dataclasses.replace(tcfg, grid_gaps=other)).structure
    ratio = alt.diagnostics["adjacent_gap_limit_deg"] / s.diagnostics["adjacent_gap_limit_deg"]
    want = GAP_FACTORS[other] / GAP_FACTORS[tcfg.grid_gaps]
    chk.record(
        "PlusGrid takes its adjacency limit from grid_gaps",
        abs(ratio - want) < 1e-9,
        f"limit {s.diagnostics['adjacent_gap_limit_deg']:.2f} deg ({tcfg.grid_gaps}) vs "
        f"{alt.diagnostics['adjacent_gap_limit_deg']:.2f} deg ({other}); ratio {ratio:.3f}, want {want:.3f}",
    )
    chk.record(
        "variant label records the grid gap policy",
        "grid_gaps=bridge" in TopologyConfig(grid_gaps="bridge").variant_label
        and "grid_gaps=strict" in TopologyConfig(grid_gaps="strict").variant_label,
        f"bridge label: {TopologyConfig(grid_gaps='bridge').variant_label!r}",
    )
    chk.record(
        "gap factors are the pinned values",
        GAP_FACTORS == {"strict": 1.5, "bridge": 2.5},
        f"GAP_FACTORS = {GAP_FACTORS}",
    )


def check_metrics(chk: Checks, dataset: LinkDataset, results: dict, df, tcfg: TopologyConfig, vcfg: TopologyValidationConfig) -> None:
    section("Metrics agree with independent recomputation")
    for name, linksets in results.items():
        g = df[df["algorithm"] == name].set_index("t")
        idx = np.unique(np.linspace(1, len(linksets) - 1, vcfg.metric_spot_checks).astype(int))
        churn_bad = comp_bad = cap_bad = lat_bad = 0
        for t in idx:
            prev, curr = edge_set(linksets[t - 1]), edge_set(linksets[t])
            if len(prev ^ curr) != g.loc[t, "churn"]:
                churn_bad += 1
            ncomp = bfs_components(linksets[t].adjacency)
            if ncomp != g.loc[t, "n_components"] or (ncomp == 1) != bool(g.loc[t, "connected"]):
                comp_bad += 1
            cap = sum(float(dataset.capacity[t, i, j]) for i, j in curr) / 1e9
            if not np.isclose(cap, g.loc[t, "capacity_total_gbps"], rtol=1e-9):
                cap_bad += 1
            if curr:
                lat = np.mean([float(dataset.latency[t, i, j]) for i, j in curr]) * 1e3
                if not np.isclose(lat, g.loc[t, "latency_mean_ms"], rtol=1e-9):
                    lat_bad += 1
        chk.record(
            f"{name}: churn, components, capacity, latency match recomputation",
            churn_bad == comp_bad == cap_bad == lat_bad == 0,
            f"{len(idx)} timesteps: mismatches churn {churn_bad}, components {comp_bad}, "
            f"capacity {cap_bad}, latency {lat_bad}",
        )
        chk.record(
            f"{name}: degree histogram has nothing above U",
            int(g["deg_over_U"].sum()) == 0 and int(g["degree_max"].max()) <= tcfg.u_max,
            f"deg_over_U total {int(g['deg_over_U'].sum())}, degree_max {int(g['degree_max'].max())}",
        )

    chk.record(
        "churn metric is the undirected symmetric difference; identical sets give 0",
        link_churn(LinkSet(np.eye(3, k=1, dtype=bool) | np.eye(3, k=-1, dtype=bool), 1),
                   LinkSet(np.eye(3, k=2, dtype=bool) | np.eye(3, k=-2, dtype=bool), 2)) == 3
        and link_churn(results["Greedy"][0], results["Greedy"][0]) == 0,
        "path 0-1-2 -> single edge 0-2: remove 2, add 1 = 3; identical sets = 0",
    )
    # t=0 has no previous topology, so its churn is 0 by definition -- not the
    # link count. The spot checks above start at t=1 and the per-step summary
    # drops t=0, so nothing else here would notice; churn_total would absorb a
    # whole topology's worth of phantom reconnections.
    first = {name: int(df[(df["algorithm"] == name) & (df["t"] == 0)]["churn"].iloc[0])
             for name in results}
    chk.record(
        "churn at t=0 is 0 for every algorithm (no previous topology to differ from)",
        all(v == 0 for v in first.values()) and link_churn(None, results["Greedy"][0]) == 0,
        f"t=0 churn by algorithm: {first}; link_churn(None, ...) = "
        f"{link_churn(None, results['Greedy'][0])}",
    )
    chk.record(
        "t=0 churn check discriminates: the topologies there are not empty",
        all(ls[0].n_links > 0 for ls in results.values()),
        f"links at t=0: { {n: ls[0].n_links for n, ls in results.items()} }",
    )

    # Every algorithm is connected at every timestep on this constellation, so
    # recomputation cannot tell "n_comp == 1" from "n_comp >= 1" -- agreement
    # everywhere is exactly what a flag stuck at True produces. Feed the metric
    # a topology that is definitely not connected.
    n = dataset.n_sats
    split = np.zeros((n, n), dtype=bool)
    vis0 = np.asarray(dataset.visible[0])
    pairs = np.argwhere(np.triu(vis0, k=1))[:2]
    for i, j in pairs:
        split[i, j] = split[j, i] = True
    row = timestep_metrics(dataset, None, LinkSet(split, 0), tcfg.u_max)
    chk.record(
        "connected flag is False for a topology that is not connected",
        row["connected"] is False and row["n_components"] == bfs_components(split),
        f"two disjoint links over {n} satellites: connected={row['connected']}, "
        f"n_components={row['n_components']} (independent BFS: {bfs_components(split)})",
    )
    # The real runs must supply the True side: the check above covers a flag
    # stuck at True, and only recomputation against connected real timesteps
    # covers one stuck at False. This used to demand that EVERY real timestep be
    # connected, which was a fact about the 10-plane window rather than the
    # property being protected; at 34 planes it failed with the flag correct.
    chk.record(
        "connected-flag check discriminates: the flag is exercised with both values",
        row["connected"] is False and bool(df["connected"].any()),
        f"{int(df['connected'].sum())} of {len(df)} real algorithm-timesteps connected; "
        f"synthetic split connected={row['connected']}",
    )

    variants = df["variant"].unique().tolist()
    chk.record(
        "every metrics row carries the active variant label",
        len(variants) == 1 and variants[0] == tcfg.variant_label,
        f"variant column: {variants}",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None, help="default: whole dataset")
    p.add_argument("--planes", type=int, default=10)
    p.add_argument("--per-plane", type=int, default=20)
    p.add_argument("--step", type=float, default=SimConfig.step_s, help="stage-1 timestep, seconds")
    p.add_argument("--duration-min", type=float, default=SimConfig.duration_s / 60.0, help="stage-1 window, minutes")
    p.add_argument("--grid-gaps", choices=["strict", "bridge"], default="strict")
    args = p.parse_args()

    tcfg = TopologyConfig(grid_gaps=args.grid_gaps)
    vcfg = TopologyValidationConfig()
    print("=" * 72)
    print("  LEO CONSTELLATION SIMULATOR -- STAGE 2 VALIDATION")
    print(f"  variant: {tcfg.variant_label}")
    print("=" * 72)

    cfg = SimConfig(step_s=args.step, duration_s=args.duration_min * 60.0, selection=SelectionConfig(n_sats=args.planes * args.per_plane, n_planes=args.planes, sats_per_plane=args.per_plane)).resolve(ROOT)
    dataset = run(cfg, verbose=True)
    n_steps = dataset.n_steps if args.steps is None else min(args.steps, dataset.n_steps)

    selectors = {
        "+Grid": PlusGrid(dataset, tcfg),
        "Greedy": Greedy(dataset, tcfg),
        "DoTD-forward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward")),
        "DoTD-backward": DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward")),
    }
    results = {name: s.run(n_steps) for name, s in selectors.items()}
    df = metrics_frame(dataset, results, tcfg.u_max, tcfg.variant_label)
    summary = summary_table(df)
    print(f"\n  ran {len(selectors)} selectors x {n_steps} timesteps")

    chk = Checks()
    check_constraints(chk, dataset, results, tcfg)
    check_constraints_non_vacuous(chk, results, tcfg)
    check_enforcement_rejects(chk, dataset, tcfg)
    check_score_pins(chk)
    check_running_max(chk, dataset, selectors, n_steps)
    check_normalisation_bounds(chk, dataset, tcfg, n_steps)
    check_lookahead(chk, results, selectors, vcfg)
    check_matching(chk, dataset, results, selectors, tcfg)
    check_ambiguity_flags(chk, dataset, tcfg, vcfg)
    check_grid(chk, dataset, results, selectors, summary, tcfg, vcfg)
    check_plane_adjacency(chk, dataset)
    check_baseline_connected(chk, dataset, results, selectors, tcfg)
    check_metrics(chk, dataset, results, df, tcfg, vcfg)
    return chk.summarise()


if __name__ == "__main__":
    sys.exit(main())
