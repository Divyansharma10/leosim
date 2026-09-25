"""+Grid: the structural baseline. Two in-plane neighbours, two cross-plane.

No link scoring is involved at any point. The grid is a fixed structure defined
once, from orbital-plane membership and in-plane phase; at each timestep the
active topology is that structure intersected with stage-1 visibility.

Fixed-once is deliberate. +Grid's defining property is stability, and it gets
that from the physical fact that satellites in a plane keep their order. A grid
re-derived every timestep would inherit churn from assignment ties flipping,
which is an artefact of the implementation rather than the baseline.

Plane identification
--------------------
Planes are identified here, from each satellite's orbit normal recovered from
its propagated trajectory, by clustering RAAN with a gap tolerance.

Stage 1's ``dataset.plane_index`` is NOT used for membership. Despite the name
it is not an orbital-plane label: ``leosim/tle.py`` partitions the whole shell
into exactly ``SelectionConfig.n_planes`` clusters, so asking for 5 planes on
the 70-degree shell merges roughly 33 real planes (10 degrees apart in RAAN)
into 5 clusters, one of them spanning 10..120 degrees. A +Grid built on that
would link satellites in different planes as "in-plane" neighbours. The stage-1
label is kept only as a cross-check in the diagnostics.

Why trajectory RAAN rather than TLE RAAN: TLE elements are at each satellite's
own epoch, and catalog epochs span days. At 70 degrees inclination J2 precesses
RAAN by a few degrees per day -- a sizeable fraction of the 10-degree plane
spacing. Normals fitted to positions propagated to a common time do not have
that error.

Plane adjacency
---------------
Two planes are adjacent only if they are consecutive in RAAN *and* the gap
between them is no more than ``ADJACENT_GAP_FACTOR`` times the median gap.
A 100-satellite sample does not populate every plane of the shell, so
consecutive-in-the-sample is not consecutive-in-the-shell: two sampled planes
20 degrees apart have an unsampled plane between them and are not grid
neighbours. The same rule decides whether the first and last plane wrap.

Consequence worth knowing: a plane with an unsampled neighbour, or at the end of
a non-wrapping span, has fewer than two adjacent planes, and a plane with one
sampled member has no in-plane partner. Such satellites are entitled to fewer
than 4 structural slots. Validation compares realised degree against that
entitlement, not against a flat 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from leosim.dataset import LinkDataset

from .base import LinkSet, TopologySelector
from .config import TopologyConfig

# Satellites whose recovered RAAN differ by less than this belong to the same
# plane. On the 70-degree shell, members of one plane agree to ~0.4 degrees and
# planes are ~10 degrees apart, so any value in roughly 1..8 separates them
# identically. ASSUMED: shell-dependent; re-check against the diagnostics'
# spread and gap for any other shell.
PLANE_GAP_DEG = 3.0

# Consecutive planes are adjacent if their RAAN gap is at most this multiple of
# the median consecutive gap. 1.5 admits normal spacing jitter but rejects a
# doubled gap, i.e. an unsampled plane in between.
# ASSUMED: structural tolerance, not from the paper.
ADJACENT_GAP_FACTOR = 1.5

# Planes are reported as not cleanly separable when a member's normal lies
# further from its plane's mean than this fraction of the smallest gap to a
# neighbouring plane. ASSUMED diagnostic threshold.
SEPARABILITY_WARN_FRACTION = 0.5

# Trajectory samples used to fit each satellite's orbit normal.
NORMAL_FIT_SAMPLES = 60


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def orbit_normals(dataset: LinkDataset) -> np.ndarray:
    """(N, 3) unit angular-momentum direction per satellite, from its trajectory.

    Fitted as the least-variance direction of the satellite's positions (the
    trajectory is a plane through Earth's centre), with the sign fixed by the
    direction of motion. Uses the whole run rather than one step, so a single
    noisy sample cannot tilt the plane.
    """
    idx = np.unique(np.linspace(0, dataset.n_steps - 1, NORMAL_FIT_SAMPLES).astype(int))
    pos = np.asarray(dataset.positions[idx], dtype=np.float64)  # (k, N, 3)
    if len(idx) < 3:
        raise ValueError("Need at least 3 timesteps to fit orbit normals")

    normals = np.empty((dataset.n_sats, 3))
    for s in range(dataset.n_sats):
        traj = pos[:, s, :]
        _, _, vt = np.linalg.svd(traj, full_matrices=False)
        n = vt[-1]
        # Orient along r x v, approximated by r(first) x r(second sample).
        if np.dot(np.cross(traj[0], traj[1]), n) < 0:
            n = -n
        normals[s] = n
    return _unit(normals)


def raan_inclination_deg(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RAAN and inclination from unit orbit normals.

    Ascending-node direction is z_hat x n = (-n_y, n_x, 0), so
    RAAN = atan2(n_x, -n_y). Inclination = arccos(n_z).
    """
    raan = np.degrees(np.arctan2(normals[..., 0], -normals[..., 1])) % 360.0
    inc = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    return raan, inc


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


@dataclass
class PlaneStructure:
    """Everything +Grid derives from plane membership, computed once."""

    plane_ids: list[int]  # distinct plane labels, sorted by recovered RAAN
    members: dict[int, np.ndarray]  # plane -> satellite indices, in phase order
    adjacent: dict[int, list[int]]  # plane -> adjacent planes
    plane_raan_deg: dict[int, float]
    plane_inclination_deg: dict[int, float]
    wraps: bool  # whether the first and last plane are treated as adjacent
    diagnostics: dict

    def expected_link_count(self) -> int:
        """Structural link count derived from plane sizes and adjacency alone.

        Computed without looking at the grid itself, so validation can compare
        it to what :func:`structural_pairs` built:

        - a plane of k >= 3 members is a ring of k in-plane links; k = 2 gives
          1; k <= 1 gives 0
        - each adjacent plane pair (p, q) gives min(|p|, |q|) cross links, the
          size of a one-to-one assignment
        """
        in_plane = sum(
            len(m) if len(m) >= 3 else (1 if len(m) == 2 else 0)
            for m in self.members.values()
        )
        pairs = {
            (min(p, q), max(p, q)) for p in self.plane_ids for q in self.adjacent[p]
        }
        cross = sum(min(len(self.members[p]), len(self.members[q])) for p, q in pairs)
        return in_plane + cross


def plane_diagnostics(
    plane_index: np.ndarray, normals: np.ndarray
) -> dict:
    """How cleanly the stage-1 plane grouping separates, measured independently.

    For each plane: population, mean RAAN/inclination, and the largest angle
    between a member's orbit normal and the plane's mean normal (the plane's
    *spread*). For the grouping as a whole: the smallest angle between any two
    plane mean normals (the tightest *gap*), and whether every plane's spread is
    small relative to that gap.
    """
    planes = sorted(set(plane_index.tolist()))
    means = {p: _unit(normals[plane_index == p].mean(axis=0)) for p in planes}

    per_plane = {}
    for p in planes:
        mem = normals[plane_index == p]
        spread = float(_angle_deg(mem, means[p][None, :]).max())
        raan, inc = raan_inclination_deg(means[p])
        per_plane[int(p)] = {
            "n": int(len(mem)),
            "raan_deg": float(raan),
            "inclination_deg": float(inc),
            "spread_deg": spread,
        }

    gaps = [
        float(_angle_deg(means[a], means[b]))
        for ai, a in enumerate(planes)
        for b in planes[ai + 1 :]
    ]
    min_gap = min(gaps) if gaps else float("inf")
    max_spread = max(v["spread_deg"] for v in per_plane.values())
    counts = np.array([v["n"] for v in per_plane.values()])

    return {
        "per_plane": per_plane,
        "n_planes": len(planes),
        "min_plane_gap_deg": min_gap,
        "max_plane_spread_deg": max_spread,
        "separable": bool(max_spread < SEPARABILITY_WARN_FRACTION * min_gap),
        "population_min": int(counts.min()),
        "population_max": int(counts.max()),
        "unassigned": int((plane_index < 0).sum()),
    }


def identify_planes(normals: np.ndarray, gap_deg: float = PLANE_GAP_DEG) -> np.ndarray:
    """(N,) plane label per satellite, by gap clustering on circular RAAN.

    Sort by RAAN and cut wherever consecutive satellites are more than
    ``gap_deg`` apart. The number of planes is an *output*, not an input -- the
    opposite of stage 1's forced-count partition. The cluster straddling
    0/360 degrees is merged across the wrap. Labels are assigned in RAAN order
    starting from the largest gap, so they are deterministic.

    Assumes one inclination shell; RAAN alone does not separate planes that
    share a node but differ in inclination. :func:`plane_diagnostics` reports
    per-plane inclination spread, which would expose that case.
    """
    raan, _ = raan_inclination_deg(normals)
    n = len(raan)
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(raan, kind="stable")
    r = raan[order]
    gaps = np.diff(np.concatenate([r, [r[0] + 360.0]]))  # gaps[k]: r[k] -> r[k+1]

    cuts = np.nonzero(gaps > gap_deg)[0]
    labels = np.empty(n, dtype=int)
    if len(cuts) == 0:
        labels[:] = 0  # one plane, or a shell with no clean gaps anywhere
        return labels

    # Start walking just after the first cut, so no cluster is split by the wrap.
    start = (cuts[0] + 1) % n
    label = 0
    for step in range(n):
        k = (start + step) % n
        labels[order[k]] = label
        if gaps[k] > gap_deg and step < n - 1:
            label += 1
    return labels


def build_plane_structure(dataset: LinkDataset) -> PlaneStructure:
    normals = orbit_normals(dataset)
    plane_index = identify_planes(normals)
    diag = plane_diagnostics(plane_index, normals)

    # Cross-check against stage 1's cluster label: how many real planes each
    # stage-1 "plane" actually contains. 1 everywhere would mean the two agree.
    stage1 = np.asarray(dataset.plane_index)
    diag["stage1_labels"] = {
        int(s): int(len(set(plane_index[stage1 == s].tolist())))
        for s in sorted(set(stage1.tolist()))
    }
    diag["plane_of_sat"] = plane_index.tolist()

    planes = list(diag["per_plane"].keys())
    raan = {p: diag["per_plane"][p]["raan_deg"] for p in planes}
    inc = {p: diag["per_plane"][p]["inclination_deg"] for p in planes}
    planes.sort(key=lambda p: raan[p])

    # --- plane adjacency from recovered RAAN ---------------------------------
    adjacent: dict[int, list[int]] = {p: [] for p in planes}
    wraps = False
    if len(planes) >= 2:
        r = np.array([raan[p] for p in planes])
        inner_gaps = np.diff(r)
        wrap_gap = 360.0 - (r[-1] - r[0])
        # Median over every consecutive gap including the wrap, so a sample
        # covering one arc of the shell does not let the long wrap gap vote.
        all_gaps = np.concatenate([inner_gaps, [wrap_gap]]) if len(planes) >= 3 else inner_gaps
        limit = ADJACENT_GAP_FACTOR * float(np.median(all_gaps))
        for a, b, g in zip(planes[:-1], planes[1:], inner_gaps):
            if g <= limit:
                adjacent[a].append(b)
                adjacent[b].append(a)
        if len(planes) >= 3 and wrap_gap <= limit:
            wraps = True
            adjacent[planes[0]].append(planes[-1])
            adjacent[planes[-1]].append(planes[0])
        diag["adjacent_gap_limit_deg"] = limit
    diag["wraps"] = wraps
    diag["plane_adjacency"] = {int(p): [int(q) for q in adjacent[p]] for p in planes}

    # --- in-plane phase order at t=0 -----------------------------------------
    # Argument of latitude in each plane's own frame. Satellites in one plane at
    # near-equal altitude keep their order, so it is computed once.
    pos0 = np.asarray(dataset.positions[0], dtype=np.float64)
    members: dict[int, np.ndarray] = {}
    phase = np.zeros(dataset.n_sats)
    for p in planes:
        idx = np.nonzero(plane_index == p)[0]
        n_hat = _unit(normals[idx].mean(axis=0))
        node = np.cross([0.0, 0.0, 1.0], n_hat)
        if np.linalg.norm(node) < 1e-9:  # equatorial plane: node undefined
            node = np.array([1.0, 0.0, 0.0])
        e1 = _unit(node)
        e2 = np.cross(n_hat, e1)
        u = np.degrees(np.arctan2(pos0[idx] @ e2, pos0[idx] @ e1)) % 360.0
        phase[idx] = u
        members[p] = idx[np.argsort(u, kind="stable")]

    diag["phase_deg"] = phase.tolist()

    return PlaneStructure(
        plane_ids=planes,
        members=members,
        adjacent=adjacent,
        plane_raan_deg=raan,
        plane_inclination_deg=inc,
        wraps=wraps,
        diagnostics=diag,
    )


def structural_pairs(structure: PlaneStructure) -> tuple[np.ndarray, np.ndarray]:
    """(N, N) bool grid, plus (N, N) int8 kind: 1 in-plane, 2 cross-plane.

    In-plane: each satellite linked to its phase-order predecessor and
    successor, circularly. A 2-member plane yields one link; a 1-member plane
    none.

    Cross-plane: for each adjacent plane pair, a one-to-one assignment
    minimising total circular phase difference (Hungarian algorithm). One-to-one
    is what makes the grid a grid -- nearest-neighbour picking instead lets
    several satellites claim the same partner, which then overflows U=4 and has
    to be broken by a rule that is no longer structural.
    """
    n = sum(len(m) for m in structure.members.values())
    phase = np.asarray(structure.diagnostics["phase_deg"])
    grid = np.zeros((n, n), dtype=bool)
    kind = np.zeros((n, n), dtype=np.int8)

    for mem in structure.members.values():
        k = len(mem)
        if k < 2:
            continue
        for a in range(k):
            i, j = mem[a], mem[(a + 1) % k]
            if i != j:
                grid[i, j] = grid[j, i] = True
                kind[i, j] = kind[j, i] = 1

    done: set[tuple[int, int]] = set()
    for p in structure.plane_ids:
        for q in structure.adjacent[p]:
            key = (min(p, q), max(p, q))
            if key in done:
                continue
            done.add(key)
            mp, mq = structure.members[p], structure.members[q]
            if len(mp) == 0 or len(mq) == 0:
                continue
            d = np.abs(phase[mp][:, None] - phase[mq][None, :]) % 360.0
            cost = np.minimum(d, 360.0 - d)
            rows, cols = linear_sum_assignment(cost)
            for r_, c_ in zip(rows, cols):
                i, j = mp[r_], mq[c_]
                grid[i, j] = grid[j, i] = True
                kind[i, j] = kind[j, i] = 2

    return grid, kind


class PlusGrid(TopologySelector):
    """+Grid. Structure fixed at reset; active links = grid AND visible(t)."""

    name = "+Grid"

    def __init__(self, dataset: LinkDataset, config: TopologyConfig) -> None:
        self.structure = build_plane_structure(dataset)
        self.grid, self.kind = structural_pairs(self.structure)
        deg = self.grid.sum(axis=1)
        if deg.max() > config.u_max:
            raise ValueError(
                f"Structural grid gives sat {int(np.argmax(deg))} degree {int(deg.max())} "
                f"> U={config.u_max}; plane adjacency is inconsistent."
            )
        super().__init__(dataset, config)

    def _select(self, t: int) -> LinkSet:
        visible = np.asarray(self.dataset.visible[t])
        return LinkSet(self.grid & visible, t)

    def slot_fill(self, links: LinkSet) -> dict:
        """Realised vs structurally-entitled links at one timestep.

        Entitlement is the fixed grid's degree; a satellite is short when some
        of its grid partners are not visible at this timestep.
        """
        expected = self.grid.sum(axis=1)
        realised = links.degree()
        return {
            "expected_links": int(self.grid.sum() // 2),
            "realised_links": links.n_links,
            "sats_at_4": int((realised == 4).sum()),
            "sats_full": int((realised == expected).sum()),
            "sats_short": int((realised < expected).sum()),
            "missing_in_plane": int(((self.kind == 1) & ~links.adjacency).sum() // 2),
            "missing_cross_plane": int(((self.kind == 2) & ~links.adjacency).sum() // 2),
        }
