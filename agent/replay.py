"""Ring replay buffer of whole-constellation transitions, sampled uniformly.

A transition is one env step: the observation fields the features need, the
backpressure bit for every slot (taken from ``BackpressurePolicy`` on the
float64 observation when the step was taken, so the prior in the TD target is
exactly the prior the agent acted with), the applied action, the reward, and
the same fields for the next observation. Arrays are allocated on the first
``add``, since N and D come from the scenario. Uniform sampling with
replacement; the oldest entry is overwritten first.
"""

from __future__ import annotations

import numpy as np

# Per-observation fields and their storage dtypes. Float fields are stored as
# float32: they only feed the network, never backpressure's decision.
OBS_FIELDS = {
    "queue": np.float32,
    "isl_capacity": np.float32,
    "downlink_capacity": np.float32,
    "neighbours": np.int16,
    "action_mask": np.bool_,
    "hops": np.float32,
    "bp_action": np.int8,
}


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.size = self.pos = 0
        self.data: dict[str, np.ndarray] | None = None

    def _allocate(self, obs: dict) -> None:
        c, self.data = self.capacity, {}
        for prefix in ("", "next_"):
            for k, dt in OBS_FIELDS.items():
                self.data[prefix + k] = np.zeros((c, *np.shape(obs[k])), dtype=dt)
        self.data["action"] = np.zeros((c, *np.shape(obs["bp_action"])), dtype=np.int8)
        self.data["reward"] = np.zeros(c, dtype=np.float32)
        self.data["done"] = np.zeros(c, dtype=np.float32)

    def add(self, obs: dict, action: np.ndarray, reward: float, next_obs: dict, done: bool) -> None:
        """``obs``/``next_obs`` carry every OBS_FIELDS key (agent.ddqn adds
        hops and bp_action to the env observation)."""
        if self.data is None:
            self._allocate(obs)
        p = self.pos
        for k in OBS_FIELDS:
            self.data[k][p] = obs[k]
            self.data["next_" + k][p] = next_obs[k]
        self.data["action"][p] = action
        self.data["reward"][p] = reward
        self.data["done"][p] = float(done)
        self.pos = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        if self.size == 0:
            raise RuntimeError("sampling an empty buffer")
        return rng.integers(0, self.size, batch)

    def sample(self, batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = self.sample_indices(batch, rng)
        return {k: v[idx] for k, v in self.data.items()}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "size": self.size, "pos": self.pos,
                "data": None if self.data is None else {k: v[: self.size].copy() for k, v in self.data.items()}}

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(f"buffer capacity {state['capacity']} != {self.capacity}")
        self.size, self.pos, self.data = int(state["size"]), int(state["pos"]), None
        if state["data"] is not None:
            self.data = {}
            for k, v in state["data"].items():
                full = np.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype)
                full[: self.size] = v
                self.data[k] = full
