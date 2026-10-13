"""Stage-2 configuration.

Every tunable of the topology-selection layer lives here, mirroring the stage-1
convention that no compute module holds a bare literal for a meaningful
quantity.

Fields marked ASSUMED were not pinned by the base paper (Ron et al., INFOCOM
2025). The value is a documented default, not a reading of the paper. Where the
paper is internally inconsistent, both readings are exposed as a flag rather
than resolved silently -- see ``objective`` and ``latency_formula``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Which extremum of the score to select. The paper's prose says to take the
# highest-scoring links; Eq. 15 and Algorithm 1 line 19 both say argmin. They
# cannot both hold. Default follows the prose.
Objective = Literal["max", "min"]

# Propagation delay is distance / c. The paper writes c / D (Eq. 5, Eq. 9),
# which is dimensionally inverted: fed into the w2 * (1 - L_norm) term it
# rewards *longer* links where it means to reward shorter ones. Default is the
# dimensionally correct form.
LatencyFormula = Literal["d_over_c", "c_over_d"]

# "forward"  -- read timesteps t+1..t+H from the stage-1 dataset and fold
#               discounted future link quality into the present decision. This
#               is what the paper's DTEG claims to do and what makes `horizon_s`
#               mean anything.
# "backward" -- accumulate each satellite's realised link scores from previous
#               timesteps. A stability/memory effect, not anticipation.
# "none"     -- no accumulation term at all; this is the Greedy baseline.
LookaheadDirection = Literal["forward", "backward", "none"]

# How +Grid treats a doubled RAAN gap between consecutive planes. NOT from the
# paper, which does not say how its grid handles missing planes; both are ours.
# "strict" -- a gap over topology.grid.ADJACENT_GAP_FACTOR x the median (15 deg
#             on the 70 deg shell) is not adjacent: the slot is assumed to hold
#             an unsampled plane, so linking across it would skip a neighbour.
#             Right for a partial window of the shell.
# "bridge" -- gaps up to topology.grid.BRIDGED_GAP_FACTOR x the median are
#             adjacent: an empty slot in the real shell has no plane to skip.
#             On the full 34-plane shell, whose two empty slots leave 20 deg
#             gaps, "strict" splits +Grid into two disconnected halves.
GridGapPolicy = Literal["strict", "bridge"]


@dataclass(frozen=True)
class TopologyConfig:
    """Scoring weights, constraints and the two paper-ambiguity flags."""

    # --- hard constraints ---------------------------------------------------
    # Optical terminals per satellite. U in the paper.
    u_max: int = 4

    # --- score weights ------------------------------------------------------
    # score = w1 * capacity_norm + w2 * (1 - latency_norm) + w3 * churn
    w1_capacity: float = 0.4
    w2_latency: float = 0.4
    w3_churn: float = 0.2

    # --- lookahead ----------------------------------------------------------
    lookahead: LookaheadDirection = "forward"

    # Horizon over which future (or past) link quality is accumulated.
    # ASSUMED: the paper says "a few minutes" without pinning a value.
    horizon_s: float = 120.0

    # Sub-sample the horizon rather than visiting every timestep in it. With a
    # 1 s step and a 120 s horizon this turns 120 array passes per timestep into
    # 12. ASSUMED: not in the paper; it is a cost control. Set to the dataset
    # step to disable sub-sampling.
    horizon_stride_s: float = 10.0

    # Weight on the accumulation term relative to the instantaneous score. Both
    # are normalised to [0, 1], so 1.0 gives the lookahead parity with the
    # entire base score.
    #
    # ASSUMED, and consequential: the paper adds the accumulated score directly,
    # which only defines a relative magnitude once you know how the accumulation
    # is normalised -- and it does not say. This is the single most arbitrary
    # number in stage 2. Vary it before trusting any DoTD-vs-Greedy margin.
    w_lookahead: float = 1.0

    # --- paper ambiguities, both exposed rather than resolved ----------------
    objective: Objective = "max"
    latency_formula: LatencyFormula = "d_over_c"

    # --- +Grid structure: our choice, not the paper's -------------------------
    grid_gaps: GridGapPolicy = "strict"

    # --- Phase 2 extension: ground-aware terms. OURS, not the paper's. -------
    # Both are open-loop predictions from geometry and demographics (a
    # GroundPrediction handed to the selector); neither reads queue state.
    # Default 0 reproduces Phase 1 exactly. NOT TUNED: no value other than 0
    # has been justified yet.
    #
    # Egress: reward for links that reach a satellite with a gateway in view,
    # integrated over the same forward window as the geometric lookahead. Per
    # link it is the MAX of the two endpoints' normalised egress: a mean would
    # rank gateway-to-gateway pairs highest and spend gateway satellites'
    # terminals on each other.
    w_egress: float = 0.0
    # Congestion: penalty for links into satellites under heavy predicted
    # demand, at the current timestep only. Written as headroom,
    # w * (1 - mean of endpoints' normalised demand), as latency is written
    # w2 * (1 - latency): every candidate edge gets the same constant w, so the
    # selection equals that of subtracting w * demand, while every score term
    # stays non-negative. The backward accumulator does not include either
    # extension term (topology.dotd), so the constant cannot leak into it.
    w_congestion: float = 0.0

    def __post_init__(self) -> None:
        if self.u_max < 1:
            raise ValueError(f"u_max must be >= 1, got {self.u_max}")
        if self.horizon_s <= 0:
            raise ValueError(f"horizon_s must be > 0, got {self.horizon_s}")
        if self.horizon_stride_s <= 0:
            raise ValueError("horizon_stride_s must be > 0")
        for name in ("w1_capacity", "w2_latency", "w3_churn", "w_lookahead",
                     "w_egress", "w_congestion"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def uses_ground(self) -> bool:
        """True when either extension term is active and needs a GroundPrediction."""
        return self.w_egress > 0 or self.w_congestion > 0

    @property
    def variant_label(self) -> str:
        """Short description of the active ambiguity flags, for logs and plots.

        Any output that does not say which variant produced it is not
        interpretable, so this string is printed by the driver and carried into
        the metrics frame.
        """
        label = (
            f"objective={self.objective}, latency={self.latency_formula}, "
            f"lookahead={self.lookahead}, grid_gaps={self.grid_gaps}"
        )
        # Extension weights appear only when active, so Phase 1 labels -- and
        # every result row stamped with them -- are unchanged.
        if self.uses_ground:
            label += f", w_egress={self.w_egress:g}, w_congestion={self.w_congestion:g}"
        return label

    @property
    def file_tag(self) -> str:
        """Filename suffix for the active flags. ``grid_gaps`` and the extension
        weights appear only when not the default, so Phase 1 output names are
        unchanged."""
        tag = f"{self.objective}_{self.latency_formula}"
        if self.grid_gaps != "strict":
            tag += f"_grid-{self.grid_gaps}"
        if self.uses_ground:
            tag += f"_egress{self.w_egress:g}_cong{self.w_congestion:g}"
        return tag
