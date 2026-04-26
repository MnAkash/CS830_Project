"""
Phase 1 POGEMA wrapper for decentralized MAPF experiments.

This file gives the rest of the project one clean interface to POGEMA.
It does not depend on Stable-Baselines3. The project trains a custom PPO
policy later, so this wrapper focuses on the basics we need first:

- named experiment configurations: Quick, Main, and Scale;
- reset and step methods with predictable NumPy outputs;
- helper methods for obstacles, agent positions, and goal positions;
- a random-action rollout used by the Phase 1 visual preview.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pogema import GridConfig, pogema_v0


Position = Tuple[int, int]


@dataclass(frozen=True)
class MapfConfig:
    """Small, explicit configuration for one MAPF environment size."""

    name: str
    num_agents: int
    map_size: int
    density: float
    obs_radius: int
    max_episode_steps: int
    seed: Optional[int] = 42
    on_target: str = "finish"

    @property
    def obs_size(self) -> int:
        """Local observation width/height."""
        return 2 * self.obs_radius + 1

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """POGEMA observation shape for one agent: channels, height, width."""
        return 3, self.obs_size, self.obs_size

    def to_grid_config(self, seed: Optional[int] = None) -> GridConfig:
        """Convert this project config into POGEMA's `GridConfig`."""
        return GridConfig(
            num_agents=self.num_agents,
            size=self.map_size,
            density=self.density,
            seed=self.seed if seed is None else seed,
            max_episode_steps=self.max_episode_steps,
            obs_radius=self.obs_radius,
            on_target=self.on_target,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly configuration dictionary."""
        data = asdict(self)
        data["obs_size"] = self.obs_size
        data["observation_shape"] = list(self.observation_shape)
        return data


QUICK_CONFIG = MapfConfig(
    name="quick",
    num_agents=8,
    map_size=16,
    density=0.3,
    obs_radius=2,
    max_episode_steps=128,
    seed=42,
)

MAIN_CONFIG = MapfConfig(
    name="main",
    num_agents=16,
    map_size=20,
    density=0.3,
    obs_radius=2,
    max_episode_steps=192,
    seed=42,
)

SCALE_CONFIG = MapfConfig(
    name="scale",
    num_agents=32,
    map_size=32,
    density=0.3,
    obs_radius=2,
    max_episode_steps=256,
    seed=42,
)

CONFIGS: Dict[str, MapfConfig] = {
    QUICK_CONFIG.name: QUICK_CONFIG,
    MAIN_CONFIG.name: MAIN_CONFIG,
    SCALE_CONFIG.name: SCALE_CONFIG,
}


@dataclass
class MapfState:
    """Snapshot of the grid state used by visualizers and tests."""

    step: int
    obstacles: np.ndarray
    agent_positions: List[Position]
    goal_positions: List[Position]
    terminated: List[bool]
    truncated: List[bool]

    @property
    def num_agents(self) -> int:
        return len(self.agent_positions)

    @property
    def success_count(self) -> int:
        return int(sum(self.terminated))

    @property
    def done_count(self) -> int:
        return int(sum(t or u for t, u in zip(self.terminated, self.truncated)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly state summary. Obstacle arrays are summarized."""
        return {
            "step": self.step,
            "grid_shape": list(self.obstacles.shape),
            "obstacle_cells": int(np.asarray(self.obstacles).sum()),
            "agent_positions": [list(p) for p in self.agent_positions],
            "goal_positions": [list(p) for p in self.goal_positions],
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
            "num_agents": self.num_agents,
            "success_count": self.success_count,
            "done_count": self.done_count,
        }


def make_mapf_config(name: str = "quick", **overrides: Any) -> MapfConfig:
    """Return a named config, optionally replacing selected fields."""
    key = name.lower()
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown MAPF config '{name}'. Valid configs: {valid}")

    data = CONFIGS[key].to_dict()
    data.pop("obs_size", None)
    data.pop("observation_shape", None)
    data.update(overrides)
    return MapfConfig(**data)


class MultiAgentPogemaEnv:
    """
    Thin project wrapper around one POGEMA multi-agent environment.

    The wrapper keeps POGEMA's simultaneous-action semantics: every call to
    `step()` must provide one action per agent, and the environment advances
    all agents together.
    """

    def __init__(self, config: Union[str, MapfConfig] = "quick", seed: Optional[int] = None):
        base_config = make_mapf_config(config) if isinstance(config, str) else config
        self.config = replace(base_config, seed=seed) if seed is not None else base_config
        self.seed_value = self.config.seed
        self.rng = np.random.default_rng(self.seed_value)

        self._env = None
        self._obs: Optional[np.ndarray] = None
        self._terminated = np.zeros(self.config.num_agents, dtype=bool)
        self._truncated = np.zeros(self.config.num_agents, dtype=bool)
        self._step_count = 0
        self._make_env(self.seed_value)

    def _make_env(self, seed: Optional[int]) -> None:
        self._env = pogema_v0(grid_config=self.config.to_grid_config(seed=seed))

    @property
    def num_agents(self) -> int:
        return self.config.num_agents

    @property
    def action_count(self) -> int:
        return 5

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return `(observations, info)`.

        If a seed is provided, the POGEMA environment is recreated so the map
        is reproducible. This makes visual feedback and tests deterministic.
        """
        if seed is not None:
            self.config = replace(self.config, seed=seed)
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)
            self.close()
            self._make_env(seed)

        obs, info = self._env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        self._terminated = np.zeros(self.num_agents, dtype=bool)
        self._truncated = np.zeros(self.num_agents, dtype=bool)
        self._step_count = 0
        return self._obs.copy(), info

    def step(
        self,
        actions: Iterable[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Advance all agents by one simultaneous step."""
        actions_array = np.asarray(list(actions), dtype=np.int64)
        if actions_array.shape != (self.num_agents,):
            raise ValueError(
                f"Expected {self.num_agents} actions, got shape {actions_array.shape}."
            )
        if np.any(actions_array < 0) or np.any(actions_array >= self.action_count):
            raise ValueError("Actions must be integer IDs in [0, 4].")

        obs, rewards, terminated, truncated, infos = self._env.step(actions_array.tolist())
        self._obs = np.asarray(obs, dtype=np.float32)
        self._terminated = np.asarray(terminated, dtype=bool)
        self._truncated = np.asarray(truncated, dtype=bool)
        self._step_count += 1

        return (
            self._obs.copy(),
            np.asarray(rewards, dtype=np.float32),
            self._terminated.copy(),
            self._truncated.copy(),
            list(infos),
        )

    def sample_actions(self) -> np.ndarray:
        """Sample one random action per agent using the wrapper RNG."""
        return self.rng.integers(0, self.action_count, size=self.num_agents, dtype=np.int64)

    def all_done(
        self,
        terminated: Optional[np.ndarray] = None,
        truncated: Optional[np.ndarray] = None,
    ) -> bool:
        """Return True when every agent has terminated or truncated."""
        term = self._terminated if terminated is None else np.asarray(terminated, dtype=bool)
        trunc = self._truncated if truncated is None else np.asarray(truncated, dtype=bool)
        return bool(np.all(term | trunc))

    def get_observations(self) -> np.ndarray:
        """Return the latest observation batch."""
        if self._obs is None:
            raise RuntimeError("Call reset() before reading observations.")
        return self._obs.copy()

    def _grid(self):
        return self._env.unwrapped.grid

    def get_obstacles(self) -> np.ndarray:
        """Return the obstacle map as a 2D NumPy array."""
        return np.asarray(self._grid().get_obstacles(), dtype=np.float32)

    def get_agent_positions(self) -> List[Position]:
        """Return current agent positions as `(row, col)` tuples."""
        return [tuple(map(int, p)) for p in self._grid().get_agents_xy()]

    def get_goal_positions(self) -> List[Position]:
        """Return goal positions as `(row, col)` tuples."""
        return [tuple(map(int, p)) for p in self._grid().finishes_xy]

    def get_state(self) -> MapfState:
        """Return a visualization-friendly snapshot of the current grid."""
        if self._obs is None:
            raise RuntimeError("Call reset() before reading state.")
        return MapfState(
            step=self._step_count,
            obstacles=self.get_obstacles(),
            agent_positions=self.get_agent_positions(),
            goal_positions=self.get_goal_positions(),
            terminated=self._terminated.tolist(),
            truncated=self._truncated.tolist(),
        )

    def run_random_episode(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run one random-action episode and return a compact summary."""
        self.reset(seed=self.seed_value)
        total_rewards = np.zeros(self.num_agents, dtype=np.float32)
        limit = self.config.max_episode_steps if max_steps is None else max_steps

        states = [self.get_state()]
        for _ in range(limit):
            obs, rewards, terminated, truncated, _ = self.step(self.sample_actions())
            del obs
            total_rewards += rewards
            states.append(self.get_state())
            if self.all_done(terminated, truncated):
                break

        return {
            "config": self.config.to_dict(),
            "steps": self.step_count,
            "success_count": self.get_state().success_count,
            "mean_reward": float(total_rewards.mean()),
            "total_rewards": total_rewards.tolist(),
            "states": states,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def __enter__(self) -> "MultiAgentPogemaEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_pogema_env(
    name: str = "quick",
    seed: Optional[int] = None,
    **overrides: Any,
) -> MultiAgentPogemaEnv:
    """Convenience factory used by tests and visualizations."""
    config = make_mapf_config(name, **overrides)
    return MultiAgentPogemaEnv(config=config, seed=seed)


if __name__ == "__main__":
    env = make_pogema_env("quick", seed=7)
    obs, _ = env.reset(seed=7)
    print("Phase 1 POGEMA wrapper smoke test")
    print(f"  config: {env.config.to_dict()}")
    print(f"  observation batch shape: {obs.shape}")
    print(f"  initial positions: {env.get_agent_positions()}")
    print(f"  goal positions: {env.get_goal_positions()}")
    for _ in range(5):
        obs, rewards, terminated, truncated, _ = env.step(env.sample_actions())
    print(f"  after 5 random steps: obs={obs.shape}, mean_reward={rewards.mean():.3f}")
    print(f"  all_done: {env.all_done(terminated, truncated)}")
    env.close()
