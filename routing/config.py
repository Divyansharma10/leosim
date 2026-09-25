"""Stage-3 configuration: ground stations and routing.

Same convention as the other two stages. Every tunable lives here, fields
marked ASSUMED were not pinned by the base paper, and where the paper leaves a
genuine choice it is exposed as a flag rather than resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Cost assigned to each edge when running the shortest path. The paper says it
# routes with OSPF and reports hop counts *and* end-to-end latency, but not
# which quantity OSPF minimises -- and the three answers differ:
#
#   "hop"           unit cost per edge. Minimises hops directly, which is the
#                   quantity Experiment 1 reports.
#   "latency"       propagation delay per edge. What a delay-optimised LEO
#                   network would do; hop count is then an output, not a target.
#   "inv_capacity"  OSPF's own default metric, reference bandwidth over link
#                   bandwidth. Prefers short (high-capacity) links, so it tends
#                   to agree with latency but is not identical to it.
#
# Every result carries the metric that produced it, for the same reason the
# stage-2 ambiguity flags do.
RouteMetric = Literal["hop", "latency", "inv_capacity"]


@dataclass(frozen=True)
class RoutingConfig:
    """Ground segment and path selection."""

    # Minimum elevation for a usable ground-satellite link. ASSUMED: the paper
    # gives a ground antenna gain but no mask. 25 deg is the figure commonly
    # quoted for Starlink user terminals; gateways are usually cited lower.
    # It is consequential -- it sets how many satellites a city can reach at
    # all, and at 187 satellites that is a small number.
    min_elevation_deg: float = 25.0

    # Ground station antenna gain, dBi. Table I of the base paper.
    ground_gain_dbi: float = 33.2

    # Ground stations connect to every satellite above the mask. ASSUMED: the
    # paper does not constrain the ground segment's terminal count, and the
    # satellite-side U = 4 cap is explicitly about optical ISL terminals, not
    # about feeder links. None means unlimited; an integer keeps the N best by
    # elevation.
    max_access_links: int | None = None

    # OSPF reference bandwidth for the inv_capacity metric, bits/s. Cisco's
    # historical default is 100 Mbps; every ISL here is faster than that, so a
    # 100 Mbps reference would floor every cost at 1 and silently turn the
    # metric into a hop count. Set above the fastest link instead.
    ospf_reference_bps: float = 10.0e9

    route_metric: RouteMetric = "hop"

    def __post_init__(self) -> None:
        if not -90.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError(f"min_elevation_deg out of range: {self.min_elevation_deg}")
        if self.max_access_links is not None and self.max_access_links < 1:
            raise ValueError("max_access_links must be >= 1 or None")
        if self.ospf_reference_bps <= 0:
            raise ValueError("ospf_reference_bps must be > 0")

    @property
    def variant_label(self) -> str:
        return (
            f"metric={self.route_metric}, mask={self.min_elevation_deg:g}deg, "
            f"access={'all' if self.max_access_links is None else self.max_access_links}"
        )
