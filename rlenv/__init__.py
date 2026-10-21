"""Phase 3: a Gymnasium forwarding environment over stages 1-4.

Outside ``leosim`` for the same reason as the other stages: stage 1's cache
key hashes every ``*.py`` under ``leosim/``. Wraps stages 1-4 without changing
them; the only new physics is the per-satellite queue model in ``queues.py``,
because stage 4's queues are per link with fixed routes and have no forwarding
decision for an action to make.
"""

from .config import EnvConfig
from .env import LeoForwardingEnv
from .policies import BackpressurePolicy, ConstantPolicy, RandomPolicy, ShortestPathPolicy, rollout
from .queues import QueueModel, StepResult, forward_step
from .reward import QueueReward, RewardContext, ShadowBackpressureReward
from .scenario import Scenario, build_dataset, build_scenario, neighbour_slots

__all__ = [
    "BackpressurePolicy",
    "ConstantPolicy",
    "EnvConfig",
    "LeoForwardingEnv",
    "QueueModel",
    "QueueReward",
    "RandomPolicy",
    "RewardContext",
    "Scenario",
    "ShortestPathPolicy",
    "ShadowBackpressureReward",
    "StepResult",
    "build_dataset",
    "build_scenario",
    "forward_step",
    "neighbour_slots",
    "rollout",
]
