"""Offered load routed over a selected topology, and fluid queues on every link.

Routing is fixed shortest path, as in stage 3 and as the project doc requires
("keep OSPF fixed"): it is load-oblivious, so any difference in delay or loss
between selectors comes from the topology they chose, not from routing
reacting to congestion. Each city egresses through whichever gateway its
cheapest path reaches (anycast).

Cities are deliberately NOT graph nodes. A city sees several satellites, so as
a node it would be a free relay -- sat -> city -> sat -- which no real ground
terminal is. Instead the shortest-path forest is grown from every gateway at
once over satellites only, with gateway nodes that have outgoing edges and no
incoming ones, so no path can transit a gateway either; each city then attaches
through its cheapest access link.

Fluid queue per directed link e, capacity c_e, buffer B_e = buffer_s * c_e:

    b <- b + A_e - c_e * dt;  dropped = max(0, b - B_e);  b <- clip(b, 0, B_e)

Two simplifications, stated rather than hidden:

- Every link on a path sees the path's full offered load. Upstream drops do not
  thin what arrives downstream, so downstream congestion is overstated, and
  equally for every selector.
- A link that disappears (churn, or a satellite leaving a station's view)
  loses its backlog. That is the mechanism by which churn costs anything here,
  and it is reported separately from buffer overflow.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from leosim.dataset import LinkDataset
from leosim.linkbudget import capacity_bps
from routing.config import RoutingConfig
from routing.paths import path_stretch
from routing.stations import GroundAccess, GroundStation, compute_access, geodesic_floor_km, ground_link_budget
from topology.base import LinkSet

from .config import TrafficConfig


@dataclass(frozen=True)
class SparseAccess:
    """One station's access links, ragged by timestep (CSR over t).

    Full GroundAccess arrays are (T, N) float64 x 5; for 220 stations at 601
    satellites that is ~2 GB, almost all of it zeros.
    """

    indptr: np.ndarray  # (T+1,)
    sat: np.ndarray
    capacity: np.ndarray  # bits/s
    latency: np.ndarray  # s
    # Slant range, for the stretch factor. Optional so hand-built test access
    # without geometry still works; stretch is then not computed.
    distance_km: np.ndarray | None = None

    @classmethod
    def from_access(cls, acc: GroundAccess) -> "SparseAccess":
        tt, ss = np.nonzero(acc.visible)
        indptr = np.searchsorted(tt, np.arange(acc.visible.shape[0] + 1))
        return cls(indptr, ss, acc.capacity[tt, ss], acc.latency[tt, ss], acc.distance_km[tt, ss])

    def distance_at(self, t: int, sat: int) -> float:
        lo, hi = self.indptr[t], self.indptr[t + 1]
        return float(self.distance_km[lo + np.nonzero(self.sat[lo:hi] == sat)[0][0]])

    def at(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.indptr[t], self.indptr[t + 1]
        return self.sat[lo:hi], self.capacity[lo:hi], self.latency[lo:hi]


def feeder_budget(dataset: LinkDataset, rcfg: RoutingConfig, tcfg: TrafficConfig):
    """Link budget for gateway links: the stage-3 ground budget (stage-1 budget
    with the paper's ground gain) with any feeder overrides applied."""
    budget = ground_link_budget(dataset, rcfg)
    overrides = {
        "bandwidth_hz": tcfg.feeder_bandwidth_hz,
        "tx_power_dbw": tcfg.feeder_tx_power_dbw,
        "noise_temperature_k": tcfg.feeder_noise_temperature_k,
    }
    return dataclasses.replace(budget, **{k: v for k, v in overrides.items() if v is not None})


def gateway_access(
    dataset: LinkDataset, station: GroundStation, rcfg: RoutingConfig, tcfg: TrafficConfig, times=None
) -> SparseAccess:
    """A gateway's access links. Which satellites it sees, and the latency, are
    stage 3's; only the capacity is recomputed, from the same distances, under
    the feeder budget. With no overrides this is exactly stage 3's capacity."""
    acc = compute_access(dataset, station, rcfg, times)
    sp = SparseAccess.from_access(acc)
    if tcfg.feeder_is_default:
        return sp
    tt, ss = np.nonzero(acc.visible)
    cap = capacity_bps(acc.distance_km[tt, ss], feeder_budget(dataset, rcfg, tcfg))
    return dataclasses.replace(sp, capacity=np.asarray(cap, dtype=np.float64))


def floor_matrix(cities: list[GroundStation], gateways: list[GroundStation]) -> np.ndarray:
    """(cities, gateways) geodesic floor in km, the stretch denominator."""
    return np.array([[geodesic_floor_km(c, g) for g in gateways] for c in cities])


def _cost(metric: str, capacity: np.ndarray, latency: np.ndarray, rcfg: RoutingConfig) -> np.ndarray:
    # Same three metrics as routing.paths.edge_costs, on the same inputs.
    if metric == "hop":
        return np.ones(len(capacity))
    if metric == "latency":
        return np.asarray(latency, dtype=np.float64)
    if metric == "inv_capacity":
        return rcfg.ospf_reference_bps / np.maximum(capacity, 1.0)
    raise ValueError(f"Unknown route metric: {metric!r}")


class TrafficSim:
    """Runs one selector's topology sequence under load. Node indices for the
    queue state are [satellites | cities | gateways]."""

    def __init__(
        self,
        dataset: LinkDataset,
        city_access: list[SparseAccess],
        gateway_access: list[SparseAccess],
        tcfg: TrafficConfig,
        rcfg: RoutingConfig,
        floor_km: np.ndarray | None = None,
    ) -> None:
        """``floor_km`` is (cities, gateways) of ``routing.geodesic_floor_km``;
        when given, each routed path's stretch factor is reported, exactly as
        stage 3 defines it, with the gateway the city egressed through as the
        far end. A co-located pair (floor 0) has no stretch and is counted
        apart, as in stage 3."""
        self.dataset = dataset
        self.floor_km = floor_km
        self.city_access = city_access
        self.gateway_access = gateway_access
        self.tcfg = tcfg
        self.rcfg = rcfg
        self.n = dataset.n_sats
        self.n_city = len(city_access)
        self.n_gw = len(gateway_access)
        self.m = self.n + self.n_city + self.n_gw
        self.dt = float(dataset.config.step_s)
        self.backlog = np.zeros((self.m, self.m))

    # --- routing -------------------------------------------------------------
    def _forest(self, links: LinkSet, t: int):
        """Shortest-path forest from all gateways over the satellites.

        Graph nodes are [satellites | gateways]; gateway nodes have outgoing
        edges only. Returns (dist, pred) over those nodes.
        """
        n, metric = self.n, self.rcfg.route_metric
        ii, jj = np.nonzero(np.triu(links.adjacency, k=1))
        cap = np.asarray(self.dataset.capacity[t], dtype=np.float64)[ii, jj]
        lat = np.asarray(self.dataset.latency[t], dtype=np.float64)[ii, jj]
        c = _cost(metric, cap, lat, self.rcfg)
        rows, cols, data = [ii, jj], [jj, ii], [c, c]
        for g, acc in enumerate(self.gateway_access):
            sats, gcap, glat = acc.at(t)
            rows.append(np.full(len(sats), n + g))
            cols.append(sats)
            data.append(_cost(metric, gcap, glat, self.rcfg))
        data = np.concatenate(data)
        if data.size and data.min() <= 0.0:
            raise ValueError(f"t={t}: non-positive edge cost would vanish from the graph")
        size = n + self.n_gw
        graph = csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(size, size))
        # min_only=True also returns each node's nearest source; the chain
        # walk recovers it anyway.
        dist, pred, _ = dijkstra(
            graph, directed=True, indices=np.arange(n, size), min_only=True, return_predecessors=True
        )
        return dist, pred

    def routes(self, links: LinkSet, t: int) -> list:
        """Per city: None if it sees no satellite, [] if it sees one but no
        gateway is reachable, else the path as queue-node indices in the
        gateway -> city direction."""
        n = self.n
        dist, pred = self._forest(links, t)
        out = []
        for c, acc in enumerate(self.city_access):
            sats, ccap, clat = acc.at(t)
            if len(sats) == 0:
                out.append(None)
                continue
            total = _cost(self.rcfg.route_metric, ccap, clat, self.rcfg) + dist[sats]
            k = int(np.argmin(total))
            if not np.isfinite(total[k]):
                out.append([])
                continue
            node = int(sats[k])
            chain = [node]
            while pred[node] >= 0:
                node = int(pred[node])
                chain.append(node)
            # chain ends at the gateway's graph node n + g; map it to its queue
            # node, then append the city.
            chain[-1] = n + self.n_city + (chain[-1] - n)
            chain.reverse()
            chain.append(n + c)
            out.append(chain)
        return out

    # --- queues --------------------------------------------------------------
    def _capacity(self, links: LinkSet, t: int) -> np.ndarray:
        """(M, M) directed capacity of every link that exists at t; 0 elsewhere."""
        n, cap = self.n, np.zeros((self.m, self.m))
        isl = np.asarray(self.dataset.capacity[t], dtype=np.float64)
        cap[:n, :n] = np.where(links.adjacency, isl, 0.0)
        for offset, group in ((n, self.city_access), (n + self.n_city, self.gateway_access)):
            for k, acc in enumerate(group):
                sats, c, _ = acc.at(t)
                cap[offset + k, sats] = c
                cap[sats, offset + k] = c
        return cap

    def step(self, links: LinkSet, t: int, offered: np.ndarray) -> dict:
        """Advance the queues one timestep. ``offered`` is (C,) bits."""
        cap = self._capacity(links, t)

        # Backlog on a link that no longer exists is lost with it.
        gone = (self.backlog > 0) & (cap <= 0)
        churn_lost = float(self.backlog[gone].sum())
        self.backlog[gone] = 0.0

        paths = self.routes(links, t)
        arrivals = np.zeros_like(cap)
        edges_of = {}
        no_access = no_path = 0.0
        for c, path in enumerate(paths):
            if path is None:
                no_access += offered[c]
                continue
            if not path:
                no_path += offered[c]
                continue
            src, dst = np.asarray(path[:-1]), np.asarray(path[1:])
            if self.tcfg.direction == "to_gateway":
                src, dst = dst, src
            edges_of[c] = (src, dst)
            np.add.at(arrivals, (src, dst), offered[c])

        if np.any(arrivals[cap <= 0] > 0):
            raise AssertionError(f"t={t}: load routed onto a link that does not exist")

        buffer = self.tcfg.buffer_s * cap
        b = self.backlog + arrivals - cap * self.dt
        dropped = np.maximum(b - buffer, 0.0)
        self.backlog = np.clip(b, 0.0, buffer)

        live = cap > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            drop_frac = np.where(arrivals > 0, dropped / arrivals, 0.0)
            util = np.where(live, arrivals / (cap * self.dt), 0.0)
            qdelay = np.where(live, self.backlog / cap, 0.0)

        # Per-city delivery and delay, from the links on its path.
        delivered, delays, weights = 0.0, [], []
        stretch_sum = stretch_bits = stretch_undefined = km_sum = 0.0
        lat_isl = np.asarray(self.dataset.latency[t], dtype=np.float64)
        for c, (src, dst) in edges_of.items():
            keep = float(np.prod(1.0 - drop_frac[src, dst]))
            delivered += offered[c] * keep
            prop = 0.0
            for u, v in zip(src, dst):
                if u < self.n and v < self.n:
                    prop += lat_isl[u, v]
                else:
                    ground, sat = (u, v) if u >= self.n else (v, u)
                    acc = (self.city_access[ground - self.n] if ground < self.n + self.n_city
                           else self.gateway_access[ground - self.n - self.n_city])
                    sats, _, lat = acc.at(t)
                    prop += float(lat[np.nonzero(sats == sat)[0][0]])
            delays.append(prop + float(qdelay[src, dst].sum()))
            weights.append(offered[c])
            if self.floor_km is not None:
                km, s = self._path_km_stretch(paths[c], c, t)
                km_sum += offered[c] * km
                if np.isnan(s):
                    stretch_undefined += offered[c]
                else:
                    stretch_sum += offered[c] * s
                    stretch_bits += offered[c]

        delays_arr, w = np.asarray(delays), np.asarray(weights)
        routed = float(w.sum())
        if routed > 0:
            order = np.argsort(delays_arr)
            cum = np.cumsum(w[order]) / routed
            p95 = float(delays_arr[order][np.searchsorted(cum, 0.95)])
            mean = float((delays_arr * w).sum() / routed)
        else:
            p95 = mean = np.nan

        gw = slice(self.n + self.n_city, self.m)
        gw_imbalance, gw_loaded = self._gateway_imbalance(arrivals, cap, gw)
        return {
            "t": t,
            "offered_bits": float(offered.sum()),
            "no_access_bits": no_access,
            "no_path_bits": no_path,
            "routed_bits": routed,
            "delivered_bits": delivered,
            "dropped_bits": float(dropped.sum()),
            "churn_lost_bits": churn_lost,
            "backlog_bits": float(self.backlog.sum()),
            "delay_mean_ms": mean * 1e3,
            "delay_p95_ms": p95 * 1e3,
            "util_max": float(util.max()) if live.any() else 0.0,
            "links_saturated": int((util[live] >= 1.0).sum()),
            # Gateway feeder links, in whichever direction the traffic runs.
            "feeder_util_max": float(max(util[gw, :].max(), util[:, gw].max())) if self.n_gw else 0.0,
            "gateway_imbalance": gw_imbalance,
            "gateways_loaded": gw_loaded,
            # (G,) bits on each gateway's feeder links this step, for per-gateway shares.
            "gateway_load_bits": arrivals[gw, :].sum(axis=1) + arrivals[:, gw].sum(axis=0),
            # Demand-weighted over routed cities whose stretch is defined.
            "stretch_mean": stretch_sum / stretch_bits if stretch_bits > 0 else np.nan,
            "stretch_bits": stretch_bits,
            "stretch_undefined_bits": stretch_undefined,
            # Demand-weighted routed path length. Stretch alone misleads here: a
            # city a few km from its gateway still climbs ~1100 km to a satellite
            # and back, so its stretch is in the hundreds and dominates a mean.
            "path_km_mean": km_sum / routed if routed > 0 and self.floor_km is not None else np.nan,
        }

    def _path_km_stretch(self, path: list, c: int, t: int) -> tuple[float, float]:
        """Stretch of one routed path (queue nodes, gateway -> city): its length
        over the geodesic floor between the city and the gateway it reached.
        Same definition and same guard as ``routing.paths.path_stretch``."""
        n, gw_base = self.n, self.n + self.n_city
        path_km = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if u < n and v < n:
                path_km += float(self.dataset.distance[t, u, v])
            else:
                ground, sat = (u, v) if u >= n else (v, u)
                acc = (self.city_access[ground - n] if ground < gw_base
                       else self.gateway_access[ground - gw_base])
                path_km += acc.distance_at(t, sat)
        floor = float(self.floor_km[c, path[0] - gw_base])
        return path_km, (path_stretch(path_km, floor) if floor > 0 else np.nan)

    def _gateway_imbalance(self, arrivals: np.ndarray, cap: np.ndarray, gw: slice) -> tuple[float, int]:
        """Peak gateway utilisation over aggregate gateway utilisation.

        Per gateway g: load_g = bits arriving on its feeder links this step (in
        whichever direction traffic runs), capacity_g = its feeder capacity x
        dt. Over gateways with any feeder capacity at all:

            imbalance = max_g(load_g / capacity_g) / (sum load / sum capacity)

        1 when every serving gateway is equally utilised; larger when load
        piles onto some feeders while others idle. A topology cannot change
        feeder capacity -- ground links are not selected -- so redistributing
        load across gateways is the only gain available to it there. NaN when
        no gateway carries load.
        """
        load = arrivals[gw, :].sum(axis=1) + arrivals[:, gw].sum(axis=0)
        capacity = cap[gw, :].sum(axis=1) * self.dt
        serving = capacity > 0
        if load.sum() <= 0 or not serving.any():
            return float("nan"), 0
        util = load[serving] / capacity[serving]
        aggregate = load.sum() / capacity[serving].sum()
        return float(util.max() / aggregate), int((load > 0).sum())

    def run(self, linksets: list[LinkSet], offered: np.ndarray) -> list[dict]:
        self.backlog[:] = 0.0
        return [self.step(ls, t, offered[t]) for t, ls in enumerate(linksets)]
