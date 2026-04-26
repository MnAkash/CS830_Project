"""Phase 1 tests for the POGEMA environment wrapper.

Run from the project root:

    conda activate grasp_splats
    python src/test_pogema.py

The script validates the wrapper and creates visual feedback artifacts under
`results/phase1_environment/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_visualize import create_phase1_preview
from pogema_wrapper import CONFIGS, MultiAgentPogemaEnv, make_mapf_config, make_pogema_env


def _print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def test_named_configs() -> None:
    """Check that Quick, Main, and Scale configs match the proposal."""
    _print_header("Test 1: Named MAPF configurations")
    expected = {
        "quick": (8, 16, 128),
        "main": (16, 20, 192),
        "scale": (32, 32, 256),
    }
    for name, (agents, size, max_steps) in expected.items():
        cfg = CONFIGS[name]
        print(
            f"  {name:>5}: agents={cfg.num_agents}, size={cfg.map_size}, "
            f"max_steps={cfg.max_episode_steps}, obs={cfg.observation_shape}"
        )
        assert cfg.num_agents == agents
        assert cfg.map_size == size
        assert cfg.max_episode_steps == max_steps
        assert cfg.observation_shape == (3, 5, 5)
    print("  [PASS] Named configs are correct.\n")


def test_wrapper_reset_and_step() -> None:
    """Check reset, state inspection, action validation, and one step."""
    _print_header("Test 2: Wrapper reset, step, and state helpers")
    env = make_pogema_env("quick", seed=11)
    obs, info = env.reset(seed=11)
    del info

    print(f"  Observation batch shape: {obs.shape}")
    print(f"  Initial positions: {env.get_agent_positions()}")
    print(f"  Goal positions: {env.get_goal_positions()}")
    assert obs.shape == (env.num_agents, 3, 5, 5)
    assert len(env.get_agent_positions()) == env.num_agents
    assert len(env.get_goal_positions()) == env.num_agents
    expected_grid_shape = (
        env.config.map_size + 2 * env.config.obs_radius,
        env.config.map_size + 2 * env.config.obs_radius,
    )
    assert env.get_obstacles().shape == expected_grid_shape

    actions = env.sample_actions()
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    del infos
    print(f"  Random actions: {actions.tolist()}")
    print(f"  Next obs shape: {next_obs.shape}")
    print(f"  Rewards shape: {rewards.shape}, mean reward={rewards.mean():.3f}")
    assert next_obs.shape == obs.shape
    assert rewards.shape == (env.num_agents,)
    assert terminated.shape == (env.num_agents,)
    assert truncated.shape == (env.num_agents,)

    try:
        env.step([0, 1])
        raise AssertionError("Action-length validation did not trigger.")
    except ValueError:
        print("  Action-length validation works.")

    env.close()
    print("  [PASS] Wrapper reset and step are correct.\n")


def test_seed_reproducibility() -> None:
    """Check that resetting with the same seed gives the same map state."""
    _print_header("Test 3: Seed reproducibility")
    env_a = MultiAgentPogemaEnv(make_mapf_config("quick"), seed=21)
    env_b = MultiAgentPogemaEnv(make_mapf_config("quick"), seed=21)
    env_a.reset(seed=21)
    env_b.reset(seed=21)

    same_obstacles = np.array_equal(env_a.get_obstacles(), env_b.get_obstacles())
    same_positions = env_a.get_agent_positions() == env_b.get_agent_positions()
    same_goals = env_a.get_goal_positions() == env_b.get_goal_positions()
    print(f"  Same obstacles: {same_obstacles}")
    print(f"  Same starts:    {same_positions}")
    print(f"  Same goals:     {same_goals}")
    assert same_obstacles
    assert same_positions
    assert same_goals
    env_a.close()
    env_b.close()
    print("  [PASS] Seeded resets are reproducible.\n")


def test_phase1_visualization() -> None:
    """Create visual feedback artifacts for the user."""
    _print_header("Test 4: Phase 1 visual report")
    summary = create_phase1_preview(config_name="quick", seed=7, rollout_steps=32)
    for label, path in summary["files"].items():
        print(f"  {label:>12}: {path}")
        assert Path(path).exists(), f"Missing visualization artifact: {path}"
    print("  [PASS] Visual report and artifacts were created.\n")


if __name__ == "__main__":
    test_named_configs()
    test_wrapper_reset_and_step()
    test_seed_reproducibility()
    test_phase1_visualization()
    print("All Phase 1 POGEMA wrapper tests passed!")
