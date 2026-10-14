"""Open-loop ground predictions for the Phase 2 score terms.

Builds the GroundPrediction that ``topology`` scores against: per satellite and
timestep, the best feeder capacity to a gateway (egress) and the expected user
demand landing on it (congestion). Both come from geometry -- which station
sees which satellite above the mask -- and from population times the diurnal
curve. Nothing here reads, or can read, the queue state of a TrafficSim: this
module imports neither ``network.TrafficSim`` nor the Poisson realisation
``offered_bits``, and the suite checks both by parsing this file.

Perfect-forecast caveat. Demand is predicted with exactly the model stage 4
evaluates against, so the congestion term is fed a perfect forecast. That is
the best case for the extension and the main threat to its validity; CLAUDE.md
requires a demand-perturbation sensitivity run before any result is reported.
"""

from __future__ import annotations

import numpy as np

from routing.config import RoutingConfig
from topology.base import LinkSet
from topology.scoring import GroundPrediction

from .config import TrafficConfig
from .demand import DemandSites, expected_rate_bps
from .network import SparseAccess


def predict_ground(
    sites: DemandSites,
    city_access: list[SparseAccess],
    gateway_access: list[SparseAccess],
    utc_hours: np.ndarray,
    n_sats: int,
    tcfg: TrafficConfig,
    rcfg: RoutingConfig,
) -> GroundPrediction:
    """Egress and demand per satellite for every timestep in ``utc_hours``.

    egress[t, s] = max over gateways g seeing s at t of g's feeder capacity to s.
        The best gateway, not the sum: a satellite downlinks through one feeder
        at a time in this model's routing (shortest-path forest), and capacity
        is set by slant range, so "best" is the nearest gateway above the mask.

    demand[t, s] = sum over cities c seeing s at t of rate[t, c] / k[t, c],
        where k[t, c] is how many satellites c sees. ASSUMED: an equal split.
        Giving each satellite the city's whole demand would count it k times;
        the satellite the traffic layer actually attaches to depends on the
        selected topology, so it cannot be predicted open-loop. Demand is the
        EXPECTED rate, never the Poisson draw: the prediction must not depend
        on the traffic seed.
    """
    n_t = len(utc_hours)
    for acc in [*city_access, *gateway_access]:
        if len(acc.indptr) != n_t + 1:
            raise ValueError(f"access covers {len(acc.indptr) - 1} steps, hours cover {n_t}")
    if len(city_access) != len(sites.cities):
        raise ValueError(f"{len(city_access)} city accesses for {len(sites.cities)} cities")

    egress = np.zeros((n_t, n_sats))
    for acc in gateway_access:
        tt = np.repeat(np.arange(n_t), np.diff(acc.indptr))
        np.maximum.at(egress, (tt, acc.sat), acc.capacity)

    rate = expected_rate_bps(sites, np.asarray(utc_hours, dtype=np.float64), tcfg)  # (T, C)
    sd = tcfg.predict_rate_noise_sd
    if sd > 0:
        # Predictor error only; the evaluator never sees this (TrafficConfig).
        z = np.random.default_rng(tcfg.predict_noise_seed).standard_normal(rate.shape[1])
        rate = rate * np.exp(sd * z - 0.5 * sd * sd)[None, :]
    demand = np.zeros((n_t, n_sats))
    for c, acc in enumerate(city_access):
        k = np.diff(acc.indptr)
        tt = np.repeat(np.arange(n_t), k)
        np.add.at(demand, (tt, acc.sat), rate[tt, c] / k[tt])

    label = (
        f"ground: mask={rcfg.min_elevation_deg:g}deg, {tcfg.gateway_label}, "
        + ("demand=expected population x diurnal (perfect forecast), split=equal" if sd == 0 else
           f"demand=expected population x diurnal x lognormal(sd={sd:g}, seed={tcfg.predict_noise_seed}) "
           "per city, split=equal")
    )
    return GroundPrediction(egress_bps=egress, demand_bps=demand, label=label)


def gateway_pair_share(linksets: list[LinkSet], ground: GroundPrediction) -> float:
    """Fraction of selected links whose BOTH ends see a gateway, pooled over the
    run. The risk the max-combination of the egress term is meant to avoid:
    gateway-visible satellites spending their terminals on each other."""
    both = total = 0
    for ls in linksets:
        e = ground.egress_bps[ls.t] > 0
        upper = np.triu(ls.adjacency, k=1)
        both += int((upper & e[:, None] & e[None, :]).sum())
        total += int(upper.sum())
    return both / total if total else float("nan")
