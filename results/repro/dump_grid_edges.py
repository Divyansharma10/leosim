"""+Grid edge set as plain text, for a cross-machine diff. Changes nothing.

    python results/repro/dump_grid_edges.py results/repro/grid_laptop.txt    # from the repo root
    python results/repro/dump_grid_edges.py results/repro/grid_cluster.txt
    diff <(grep -v '^#' results/repro/grid_laptop.txt) <(grep -v '^#' results/repro/grid_cluster.txt)

Same run validate_extension.py hashes: 10 x 20 selection (187 sats), default
TopologyConfig. Lines starting with '#' are machine-specific (versions, source
digest) and are meant to be excluded from the diff; everything else should be
identical on two machines that agree.

Contents:
- the structural grid, fixed at reset, as sorted i-j pairs (i < j): in-plane
  links grouped by plane, cross-plane links grouped by plane pair. A different
  tie-break shows up here.
- for a few timesteps, the active link count and the structural pairs NOT
  active (grid AND visible), so the whole active set is recoverable without
  listing ~330 edges per step.
- a SHA-256 over every step's adjacency, hashed exactly as GOLDEN in
  validate_extension.py, compared to the stored +Grid pin. If that line
  matches on both machines, all 1800 steps agree, not just the sampled ones.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path.cwd()))

from leosim.config import SelectionConfig, SimConfig  # noqa: E402
from leosim.simulation import run  # noqa: E402
from topology import PlusGrid, TopologyConfig  # noqa: E402
from validate_extension import GOLDEN  # noqa: E402

SAMPLE_STEPS = (0, 1, 450, 900, 1350, 1799)
PER_LINE = 10


def _wrap(pairs: list[tuple[int, int]], indent: str = "    ") -> list[str]:
    toks = [f"{i}-{j}" for i, j in pairs]
    return [indent + " ".join(toks[k:k + PER_LINE]) for k in range(0, len(toks), PER_LINE)] or [indent + "(none)"]


def main(out: str) -> None:
    planes, per_plane = 10, 20
    cfg = SimConfig(selection=SelectionConfig(n_sats=planes * per_plane, n_planes=planes,
                                              sats_per_plane=per_plane)).resolve(Path.cwd())
    ds = run(cfg, verbose=False)
    tcfg = TopologyConfig()
    sel = PlusGrid(ds, tcfg)
    plane_of = np.asarray(sel.structure.diagnostics["plane_of_sat"])
    grid, kind = sel.grid, sel.kind
    n = ds.n_sats

    pairs = [(int(i), int(j)) for i, j in np.argwhere(np.triu(grid, 1))]
    in_plane: dict[int, list] = {}
    cross: dict[tuple[int, int], list] = {}
    for i, j in pairs:
        p, q = int(plane_of[i]), int(plane_of[j])
        if kind[i, j] == 1:
            in_plane.setdefault(p, []).append((i, j))
        else:
            cross.setdefault((min(p, q), max(p, q)), []).append((i, j))

    src = hashlib.sha256(Path("topology/grid.py").read_bytes()).hexdigest()
    lines = [
        f"# python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}",
        f"# {platform.platform()}, {platform.processor() or platform.machine()}",
        f"# topology/grid.py sha256 {src}",
        f"variant {tcfg.variant_label}",
        f"n_sats {n}, steps {ds.n_steps}, structural pairs {len(pairs)} "
        f"(in-plane {sum(map(len, in_plane.values()))}, cross-plane {sum(map(len, cross.values()))})",
        "",
        "== in-plane pairs, by plane ==",
    ]
    for p in sorted(in_plane):
        lines.append(f"  plane {p} ({len(in_plane[p])}):")
        lines += _wrap(sorted(in_plane[p]))
    lines += ["", "== cross-plane pairs, by plane pair =="]
    for key in sorted(cross):
        lines.append(f"  planes {key[0]}-{key[1]} ({len(cross[key])}):")
        lines += _wrap(sorted(cross[key]))

    lines += ["", "== sampled steps: active = structure AND visible; listed = structural pairs NOT active =="]
    sel.reset()
    ha = hashlib.sha256()
    hashes = {}
    for t in range(ds.n_steps):
        adj = sel.step(t).adjacency
        ha.update(np.packbits(adj).tobytes())
        if t + 1 in GOLDEN:
            hashes[t + 1] = ha.copy().hexdigest()
        if t in SAMPLE_STEPS:
            missing = [(i, j) for i, j in pairs if not adj[i, j]]
            extra = int(np.triu(adj & ~grid, 1).sum())  # must be 0 by construction
            lines.append(f"  t={t}: active {int(np.triu(adj, 1).sum())}, inactive {len(missing)}"
                         + (f", NON-STRUCTURAL {extra}" if extra else ""))
            lines += _wrap(missing, "      ")

    lines += ["", "== whole-run adjacency hash (as validate_extension.py GOLDEN) =="]
    for k in sorted(hashes):
        pin = GOLDEN[k]["+Grid"][0]
        lines.append(f"  first {k} steps: {hashes[k]}  {'matches' if hashes[k] == pin else 'DIFFERS FROM'} "
                     f"stored pin {pin[:16]}..")

    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid_edges.txt")
