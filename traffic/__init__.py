"""Stage 4: offered load, gateways and queues over a selected topology.

Outside the ``leosim`` package for the same reason ``topology`` and ``routing``
are: stage 1's cache key hashes every ``*.py`` under ``leosim/``.

Stages 1-3 measure nothing that depends on load, so a demand-aware extension to
DoTD would be unmeasurable on them. This stage supplies delivered throughput,
delay under load and loss, and is the axis Phase 2 is evaluated on.
"""

from .config import GatewaySites, TrafficConfig, TrafficDirection
from .demand import (
    DemandSites,
    diurnal_factor,
    expected_rate_bps,
    load_sites,
    local_solar_hour,
    offered_bits,
)
from .network import SparseAccess, TrafficSim, feeder_budget, floor_matrix, gateway_access
from .predict import gateway_pair_share, predict_ground

__all__ = [
    "DemandSites",
    "GatewaySites",
    "SparseAccess",
    "TrafficConfig",
    "TrafficDirection",
    "TrafficSim",
    "diurnal_factor",
    "expected_rate_bps",
    "feeder_budget",
    "floor_matrix",
    "gateway_access",
    "gateway_pair_share",
    "load_sites",
    "local_solar_hour",
    "offered_bits",
    "predict_ground",
]
