"""City-to-city paths over a selected ISL topology.

The graph at one timestep has N + 2 nodes: the satellites, plus a source and a
destination ground station. Satellite-satellite edges are exactly the links the
stage-2 selector chose -- never the full visibility set, which is the whole
point of the comparison. Ground-satellite edges are the access links above the
elevation mask.

Hop count is reported as the number of *edges* in the path, so New York ->
sat -> sat -> San Francisco is 3. The base paper reports 5 for DoTD on this
pair; whether that counts ground links or only ISLs is not stated, so both are
reported and neither is quietly folded into the other.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from leosim.dataset import LinkDataset
from topology.base import LinkSet

from .config import RoutingConfig
from .stations import GroundAccess

# Sentinel for "no path exists at this timestep". Distinguished from a real
# metric value everywhere; rows carrying it are excluded from averages rather
# than counted as zero.
NO_PATH = -1


def edge_costs(
    dataset: LinkDataset,
    links: LinkSet,
    access: dict[str, GroundAccess],
    t: int,
    cfg: RoutingConfig,
) -> tuple[csr_matrix, list[str]]:
    """Weighted graph for one timestep, and its node labels.

    Node order is [0..N-1] satellites, then N = source, N+1 = destination.
    """
    n = dataset.n_sats
    names = [str(s) for s in dataset.sat_names] + list(access.keys())
    size = n + len(access)

    adj = links.adjacency
    ii, jj = np.nonzero(np.triu(adj, k=1))

    if cfg.route_metric == "hop":
        isl_cost = np.ones(len(ii))
    elif cfg.route_metric == "latency":
        isl_cost = np.asarray(dataset.latency[t], dtype=np.float64)[ii, jj]
    elif cfg.route_metric == "inv_capacity":
        cap = np.asarray(dataset.capacity[t], dtype=np.float64)[ii, jj]
        isl_cost = cfg.ospf_reference_bps / np.maximum(cap, 1.0)
    else:
        raise ValueError(f"Unknown route metric: {cfg.route_metric!r}")

    rows = list(ii)
    cols = list(jj)
    data = list(isl_cost)

    for k, acc in enumerate(access.values()):
        sats = np.nonzero(acc.visible[t])[0]
        if cfg.route_metric == "hop":
            cost = np.ones(len(sats))
        elif cfg.route_metric == "latency":
            cost = acc.latency[t][sats]
        else:
            cost = cfg.ospf_reference_bps / np.maximum(acc.capacity[t][sats], 1.0)
        rows.extend([n + k] * len(sats))
        cols.extend(sats.tolist())
        data.extend(cost.tolist())

    # Costs must be strictly positive: a zero-weight edge is indistinguishable
    # from an absent one in a sparse matrix, so a zero-latency link would be
    # silently deleted rather than treated as free.
    data = np.asarray(data, dtype=np.float64)
    if data.size and data.min() <= 0.0:
        raise ValueError(
            f"t={t}: {int((data <= 0).sum())} edges have non-positive cost under "
            f"metric {cfg.route_metric!r}; a zero weight would vanish from the graph"
        )

    graph = csr_matrix(
        (data, (rows, cols)), shape=(size, size)
    )
    return graph + graph.T, names


def shortest_path(
    dataset: LinkDataset,
    links: LinkSet,
    access: dict[str, GroundAccess],
    t: int,
    cfg: RoutingConfig,
) -> dict:
    """One timestep: the path between the two stations, and what it costs.

    ``access`` must hold exactly two stations, source first.
    """
    if len(access) != 2:
        raise ValueError(f"expected exactly 2 stations, got {list(access)}")

    n = dataset.n_sats
    graph, _ = edge_costs(dataset, links, access, t, cfg)
    src, dst = n, n + 1

    dist, pred = dijkstra(graph, indices=src, return_predecessors=True, directed=False)
    if not np.isfinite(dist[dst]):
        return {
            "t": t,
            "reachable": False,
            "hops": NO_PATH,
            "isl_hops": NO_PATH,
            "latency_ms": np.nan,
            "path_capacity_mean_gbps": np.nan,
            "path_capacity_min_gbps": np.nan,
            "cost": np.nan,
        }

    # Walk the predecessor chain back from the destination.
    path = [dst]
    while path[-1] != src:
        path.append(int(pred[path[-1]]))
    path.reverse()

    latency = 0.0
    caps = []
    isl_hops = 0
    stations = list(access.values())
    for a, b in zip(path[:-1], path[1:]):
        if a < n and b < n:
            latency += float(dataset.latency[t, a, b])
            caps.append(float(dataset.capacity[t, a, b]))
            isl_hops += 1
        else:
            ground, sat = (a, b) if a >= n else (b, a)
            acc = stations[ground - n]
            latency += float(acc.latency[t, sat])
            caps.append(float(acc.capacity[t, sat]))

    caps_arr = np.asarray(caps)
    return {
        "t": t,
        "reachable": True,
        "hops": len(path) - 1,
        "isl_hops": isl_hops,
        "latency_ms": latency * 1e3,
        "path_capacity_mean_gbps": float(caps_arr.mean() / 1e9),
        "path_capacity_min_gbps": float(caps_arr.min() / 1e9),
        "cost": float(dist[dst]),
        "path": path,
    }


def route_run(
    dataset: LinkDataset,
    linksets: list[LinkSet],
    access: dict[str, GroundAccess],
    cfg: RoutingConfig,
    n_steps: int | None = None,
) -> list[dict]:
    """Shortest path at every timestep of a run."""
    steps = len(linksets) if n_steps is None else min(n_steps, len(linksets))
    return [shortest_path(dataset, linksets[t], access, t, cfg) for t in range(steps)]
