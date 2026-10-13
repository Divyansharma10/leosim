"""Stage 2: ISL topology selection on top of the stage-1 link dataset.

Lives outside the ``leosim`` package on purpose. Stage 1's cache key hashes
every ``*.py`` under ``leosim/`` recursively (``leosim.config.source_digest``),
so a module placed there would invalidate the ~237 MB link dataset on every
edit to a scoring function.
"""

from .base import ConstraintViolation, LinkSet, TopologySelector, assert_valid, b_matching
from .config import TopologyConfig
from .dotd import DoTD, Greedy
from .grid import PlusGrid
from .scoring import GroundPrediction

__all__ = [
    "ConstraintViolation",
    "DoTD",
    "Greedy",
    "GroundPrediction",
    "LinkSet",
    "PlusGrid",
    "TopologyConfig",
    "TopologySelector",
    "assert_valid",
    "b_matching",
]
