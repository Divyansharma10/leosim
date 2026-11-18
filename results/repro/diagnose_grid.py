"""Cross-machine +Grid diagnosis. Diagnostic only: changes nothing.

    python results/repro/diagnose_grid.py dump  OUT.npz    # on each machine, from the repo root
    python results/repro/diagnose_grid.py compare A.npz B.npz

The dump holds every intermediate +Grid derives (orbit normals, RAAN, plane
labels, in-plane phase, member order, plane adjacency, the structural grid,
the per-step active link sets) on the default 187-satellite run that
validate_extension.py hashes, plus how close each discrete decision came to
flipping, and the library versions. `compare` says which stage first differs.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path.cwd()))

from leosim.config import SelectionConfig, SimConfig  # noqa: E402
from leosim.simulation import run  # noqa: E402
from topology import PlusGrid, TopologyConfig  # noqa: E402
from topology.grid import PLANE_GAP_DEG, raan_inclination_deg  # noqa: E402


def _blas() -> str:
    try:
        from threadpoolctl import threadpool_info
        return "; ".join(f"{d.get('internal_api')} {d.get('version')} {d.get('architecture', '')}"
                         for d in threadpool_info())
    except Exception:
        cfg = getattr(np, "show_config", None)
        return "threadpoolctl unavailable"


def lsa_ties(cost: np.ndarray) -> tuple[float, float]:
    """(optimal cost, smallest increase from forbidding one optimal edge).

    0 increase means another assignment is exactly as good: the solver's
    choice between them is an implementation detail.
    """
    r, c = linear_sum_assignment(cost)
    best = float(cost[r, c].sum())
    margin = np.inf
    for a, b in zip(r, c):
        k = cost.copy()
        k[a, b] = 1e9
        r2, c2 = linear_sum_assignment(k)
        margin = min(margin, float(k[r2, c2].sum()) - best)
    return best, margin


def dump(out: str) -> None:
    planes, per_plane = 10, 20
    cfg = SimConfig(selection=SelectionConfig(n_sats=planes * per_plane, n_planes=planes,
                                              sats_per_plane=per_plane)).resolve(Path.cwd())
    ds = run(cfg, verbose=False)
    sel = PlusGrid(ds, TopologyConfig())
    st = sel.structure
    d = st.diagnostics
    phase = np.asarray(d["phase_deg"])
    normals = None
    from topology.grid import orbit_normals
    normals = orbit_normals(ds)
    raan, inc = raan_inclination_deg(normals)

    # Closeness of each discrete decision to flipping.
    r = np.sort(raan)
    gaps = np.diff(np.concatenate([r, [r[0] + 360.0]]))
    gap_margin = float(np.min(np.abs(gaps - PLANE_GAP_DEG)))
    phase_margin = min(float(np.min(np.diff(np.sort(phase[m])))) if len(m) > 1 else np.inf
                       for m in st.members.values())
    ties = {}
    for p in st.plane_ids:
        for q in st.adjacent[p]:
            if p < q:
                mp, mq = st.members[p], st.members[q]
                dd = np.abs(phase[mp][:, None] - phase[mq][None, :]) % 360.0
                ties[(p, q)] = lsa_ties(np.minimum(dd, 360.0 - dd))

    sel.reset()
    adj = np.stack([sel.step(t).adjacency for t in range(ds.n_steps)])
    h = {}
    ha = hashlib.sha256()
    for t in range(ds.n_steps):
        ha.update(np.packbits(adj[t]).tobytes())
        if t + 1 in (300, 1800):
            h[t + 1] = ha.copy().hexdigest()

    np.savez_compressed(
        out,
        versions=np.array([sys.version.split()[0], np.__version__, scipy.__version__,
                           platform.platform(), platform.processor() or platform.machine(), _blas()]),
        positions_sha=np.array(hashlib.sha256(np.ascontiguousarray(ds.positions).tobytes()).hexdigest()),
        visible_sha=np.array(hashlib.sha256(np.packbits(np.asarray(ds.visible)).tobytes()).hexdigest()),
        normals=normals, raan=raan, inc=inc,
        plane_of_sat=np.asarray(d["plane_of_sat"]), phase=phase,
        members=np.array([np.asarray(st.members[p]) for p in st.plane_ids], dtype=object),
        plane_ids=np.asarray(st.plane_ids),
        adjacency_planes=np.array([str(d["plane_adjacency"]), str(st.wraps)]),
        grid=sel.grid, kind=sel.kind, adj=np.packbits(adj, axis=-1),
        hashes=np.array([h.get(300, ""), h.get(1800, "")]),
        gap_margin=gap_margin, phase_margin=phase_margin,
        ties=np.array([(p, q, c, m) for (p, q), (c, m) in ties.items()]),
    )
    print(f"versions: {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}")
    print(f"+Grid hash 300 {h.get(300, '')[:16]}  1800 {h.get(1800, '')[:16]}")
    print(f"closest RAAN gap to the {PLANE_GAP_DEG} deg cut: {gap_margin:.3g} deg")
    print(f"closest in-plane phase pair: {phase_margin:.3g} deg")
    for (p, q), (c, m) in ties.items():
        print(f"planes {p}-{q}: Hungarian cost {c:.6f} deg, margin to next assignment {m:.3g}"
              + ("   <-- TIE" if m < 1e-9 else ""))
    print(f"wrote {out}")


def compare(a_path: str, b_path: str) -> None:
    a = np.load(a_path, allow_pickle=True)
    b = np.load(b_path, allow_pickle=True)
    print("A:", list(a["versions"]))
    print("B:", list(b["versions"]))
    print("stage-1 positions identical:", str(a["positions_sha"]) == str(b["positions_sha"]))
    print("stage-1 visibility identical:", str(a["visible_sha"]) == str(b["visible_sha"]))
    for k in ("normals", "raan", "phase"):
        print(f"{k}: max |A-B| = {np.max(np.abs(a[k] - b[k])):.3g}")
    print("plane labels identical:", np.array_equal(a["plane_of_sat"], b["plane_of_sat"]))
    print("plane adjacency identical:", list(a["adjacency_planes"]) == list(b["adjacency_planes"]))
    ma, mb = a["members"], b["members"]
    same_order = len(ma) == len(mb) and all(np.array_equal(x, y) for x, y in zip(ma, mb))
    print("in-plane member order identical:", same_order)
    ga, gb = a["grid"], b["grid"]
    diff = np.argwhere(np.triu(ga != gb))
    print(f"structural grid: {len(diff)} pairs differ")
    for i, j in diff[:40]:
        print(f"   ({i},{j}) A={bool(ga[i, j])} kind {a['kind'][i, j]}, B={bool(gb[i, j])} kind {b['kind'][i, j]}")
    aa, ab = np.unpackbits(a["adj"], axis=-1), np.unpackbits(b["adj"], axis=-1)
    per_t = (aa != ab).reshape(len(aa), -1).sum(axis=1) // 2
    bad = np.nonzero(per_t)[0]
    print(f"active link set differs at {len(bad)} of {len(per_t)} timesteps", end="")
    print(f"; first {bad[0]}, last {bad[-1]}, max {per_t.max()} links per step" if len(bad) else "")
    print("hashes A:", list(a["hashes"]), "\nhashes B:", list(b["hashes"]))
    for tag, z in (("A", a), ("B", b)):
        print(f"{tag} margins: RAAN cut {float(z['gap_margin']):.3g} deg, phase {float(z['phase_margin']):.3g} deg, "
              f"ties {[(int(p), int(q), float(m)) for p, q, c, m in z['ties'] if m < 1e-9]}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
