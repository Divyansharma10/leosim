"""Phase 3 step 2: a residual Double DQN on backpressure (Krishnan et al.).

Outside ``rlenv`` so the validated environment is untouched, and outside
``leosim`` for the stage-1 cache-key reason every other stage gives.
"""

from .config import REWARD_SCALE, AgentConfig, AgentValidationConfig, TrainConfig
from .ddqn import AgentPolicy, DDQNAgent, epsilon, td_loss, td_target
from .network import ResidualQ, joint_q, masked_mean
from .replay import ReplayBuffer
from .train import ScenarioCache, Trainer, env_config

__all__ = [
    "REWARD_SCALE", "AgentConfig", "AgentPolicy", "AgentValidationConfig", "DDQNAgent", "ReplayBuffer",
    "ResidualQ", "ScenarioCache", "joint_q", "TrainConfig", "Trainer", "env_config", "epsilon", "masked_mean", "td_loss", "td_target",
]
