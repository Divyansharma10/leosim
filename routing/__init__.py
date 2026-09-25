"""Stage 3: ground stations and city-to-city routing over a chosen topology.

Outside the ``leosim`` package for the same reason ``topology`` is: stage 1's
cache key hashes every ``*.py`` under ``leosim/``, so a module placed there
would invalidate the link dataset on every edit.

Stage 3 is what makes the base paper's Experiment 1 measurable. Stages 1 and 2
produce only load-independent, topology-wide quantities -- total capacity, mean
latency, churn -- and the paper's headline numbers are per-path: hop count and
end-to-end delay between two cities. Those cannot be read off a topology
without a ground segment and a path.
"""

from .config import RoutingConfig, RouteMetric
from .paths import NO_PATH, edge_costs, route_run, shortest_path
from .stations import (
    CITIES,
    PAPER_PAIRS,
    GroundAccess,
    GroundStation,
    chord_km,
    compute_access,
    dataset_times,
    great_circle_km,
)

__all__ = [
    "CITIES",
    "PAPER_PAIRS",
    "GroundAccess",
    "GroundStation",
    "NO_PATH",
    "RouteMetric",
    "RoutingConfig",
    "chord_km",
    "compute_access",
    "dataset_times",
    "edge_costs",
    "great_circle_km",
    "route_run",
    "shortest_path",
]
