"""Visualization helpers for Phase 1 environment-wrapper feedback.

The goal is not to show a trained policy yet. Phase 1 only verifies that the
environment wrapper works, observations have the right shape, and grid state
can be inspected. The generated HTML report, PNGs, and GIF make that visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pogema_wrapper import MapfConfig, MapfState, MultiAgentPogemaEnv, make_mapf_config
from utils import PHASE1_RESULTS_DIR, ensure_dir, save_json, set_global_seed, timestamp


AGENT_COLORS = [
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#e6194B",
]


def _agent_color(index: int) -> str:
    return AGENT_COLORS[index % len(AGENT_COLORS)]


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    return array


def render_state(
    state: MapfState,
    title: str,
    trajectories: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    cell_px: int = 42,
):
    """Render one grid state as a Matplotlib figure."""
    obstacles = np.asarray(state.obstacles)
    height, width = obstacles.shape
    dpi = 100
    fig_width = width * cell_px / dpi + 2.2
    fig_height = height * cell_px / dpi + 1.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    for row in range(height):
        for col in range(width):
            is_obstacle = obstacles[row, col] > 0.5
            face = "#4b4b4b" if is_obstacle else "#f7f7f7"
            edge = "#3a3a3a" if is_obstacle else "#dddddd"
            ax.add_patch(
                plt.Rectangle(
                    (col, height - 1 - row),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.45,
                    zorder=0,
                )
            )

    if trajectories is not None:
        for agent_id, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue
            color = _agent_color(agent_id)
            for point_id in range(1, len(trajectory)):
                r0, c0 = trajectory[point_id - 1]
                r1, c1 = trajectory[point_id]
                alpha = 0.15 + 0.6 * point_id / max(1, len(trajectory) - 1)
                ax.plot(
                    [c0 + 0.5, c1 + 0.5],
                    [height - 1 - r0 + 0.5, height - 1 - r1 + 0.5],
                    color=color,
                    linewidth=2.0,
                    alpha=alpha,
                    solid_capstyle="round",
                    zorder=1,
                )

    for agent_id, (goal_row, goal_col) in enumerate(state.goal_positions):
        color = _agent_color(agent_id)
        x = goal_col + 0.5
        y = height - 1 - goal_row + 0.5
        ax.plot(
            x,
            y,
            marker="D",
            markersize=10,
            color=color,
            alpha=0.28,
            markeredgecolor=color,
            markeredgewidth=1.5,
            zorder=2,
        )
        ax.text(x, y, "G", ha="center", va="center", fontsize=6, color=color, fontweight="bold", zorder=3)

    for agent_id, (agent_row, agent_col) in enumerate(state.agent_positions):
        color = _agent_color(agent_id)
        x = agent_col + 0.5
        y = height - 1 - agent_row + 0.5
        done = state.terminated[agent_id] or state.truncated[agent_id]
        if done:
            ax.add_patch(plt.Circle((x, y), 0.42, facecolor="#00cc44", alpha=0.18, edgecolor="none", zorder=3))
            edge = "#008833"
            label = "✓"
        else:
            edge = "#222222"
            label = str(agent_id)
        ax.add_patch(
            plt.Circle((x, y), 0.34, facecolor=color, edgecolor=edge, linewidth=1.2, alpha=0.95, zorder=4)
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=5)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    status = f"Step {state.step} | {state.success_count}/{state.num_agents} agents reached goals"
    ax.set_title(f"{title}\n{status}", fontsize=10, fontweight="bold", pad=8)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=8, label="Agent"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888", markersize=8, alpha=0.5, label="Goal"),
        Line2D([0], [0], color="#888888", linewidth=2, alpha=0.6, label="Trajectory"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4b4b4b", markersize=8, label="Obstacle"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.9)
    fig.subplots_adjust(right=0.78)
    return fig


def save_state_png(
    state: MapfState,
    path: Path,
    title: str,
    trajectories: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Path:
    """Save one state as a PNG image."""
    ensure_dir(path.parent)
    fig = render_state(state, title=title, trajectories=trajectories)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _trajectory_until(states: Sequence[MapfState], frame_index: int) -> List[List[Tuple[int, int]]]:
    trajectories: List[List[Tuple[int, int]]] = [[] for _ in range(states[0].num_agents)]
    for state in states[: frame_index + 1]:
        for agent_id, position in enumerate(state.agent_positions):
            trajectories[agent_id].append(position)
    return trajectories


def write_html_report(summary: Dict[str, Any], output_path: Path) -> Path:
    """Write a small browser-friendly report for visual feedback."""
    ensure_dir(output_path.parent)
    initial = Path(summary["files"]["initial_png"]).name
    final = Path(summary["files"]["final_png"]).name
    gif = Path(summary["files"]["rollout_gif"]).name
    json_name = Path(summary["files"]["summary_json"]).name
    cfg = summary["config"]

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Phase 1 Environment Wrapper Preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f6f8fb; color: #17202a; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    .card {{ background: white; border: 1px solid #dde3ee; border-radius: 12px; padding: 18px; margin: 16px 0; box-shadow: 0 2px 8px rgba(20, 30, 50, 0.06); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d4dbe8; border-radius: 10px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e9f2; padding: 8px; text-align: left; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
    .ok {{ color: #087f23; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Phase 1 Environment Wrapper Preview</h1>
  <p>Generated at {summary['generated_at']} using the <code>{summary['environment']}</code> conda environment.</p>

  <div class=\"card\">
    <h2>What this implements</h2>
    <p>The Phase 1 wrapper creates named POGEMA configurations, returns consistent observation tensors, exposes grid state for debugging, and can run reproducible random-action rollouts.</p>
    <p class=\"ok\">Status: wrapper test and visualization generation completed.</p>
  </div>

  <div class=\"card\">
    <h2>Configuration used for this preview</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Name</td><td>{cfg['name']}</td></tr>
      <tr><td>Agents</td><td>{cfg['num_agents']}</td></tr>
    <tr><td>Requested map size</td><td>{cfg['map_size']}×{cfg['map_size']}</td></tr>
    <tr><td>Rendered grid with POGEMA padding</td><td>{summary['obstacle_grid_shape']}</td></tr>
      <tr><td>Obstacle density</td><td>{cfg['density']}</td></tr>
      <tr><td>Observation shape</td><td>{summary['observation_shape']}</td></tr>
      <tr><td>Seed</td><td>{summary['seed']}</td></tr>
      <tr><td>Random rollout steps</td><td>{summary['steps']}</td></tr>
      <tr><td>Mean random-policy reward</td><td>{summary['mean_reward']:.3f}</td></tr>
    </table>
  </div>

  <div class=\"card\">
    <h2>Visual feedback</h2>
    <div class=\"grid\">
      <div><h3>Initial state</h3><img src=\"{initial}\" alt=\"Initial grid state\"></div>
      <div><h3>Random rollout GIF</h3><img src=\"{gif}\" alt=\"Random rollout animation\"></div>
      <div><h3>Final state</h3><img src=\"{final}\" alt=\"Final grid state\"></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>How to give feedback</h2>
    <p>Open this report and check whether the grid, obstacles, agents, goals, and trajectories are understandable. Raw numbers are in <code>{json_name}</code>.</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def create_phase1_preview(
    config_name: str = "quick",
    seed: int = 7,
    rollout_steps: int = 32,
    output_dir: Optional[Path] = None,
    environment_name: str = "grasp_splats",
) -> Dict[str, Any]:
    """Run a deterministic random rollout and save PNG/GIF/HTML artifacts."""
    set_global_seed(seed)
    out_dir = ensure_dir(PHASE1_RESULTS_DIR if output_dir is None else output_dir)
    config: MapfConfig = make_mapf_config(config_name, seed=seed)

    with MultiAgentPogemaEnv(config=config, seed=seed) as env:
        obs, _ = env.reset(seed=seed)
        states: List[MapfState] = [env.get_state()]
        total_rewards = np.zeros(env.num_agents, dtype=np.float32)

        for _ in range(rollout_steps):
            _, rewards, terminated, truncated, _ = env.step(env.sample_actions())
            total_rewards += rewards
            states.append(env.get_state())
            if env.all_done(terminated, truncated):
                break

    initial_png = out_dir / "phase1_initial_state.png"
    final_png = out_dir / "phase1_final_state.png"
    rollout_gif = out_dir / "phase1_random_rollout.gif"
    summary_json = out_dir / "phase1_environment_summary.json"
    report_html = out_dir / "phase1_environment_report.html"

    save_state_png(states[0], initial_png, title="Initial Quick MAPF State")
    save_state_png(
        states[-1],
        final_png,
        title="Final State After Random Actions",
        trajectories=_trajectory_until(states, len(states) - 1),
    )

    frames = []
    for frame_id, state in enumerate(states):
        frame_trajectories = _trajectory_until(states, frame_id)
        fig = render_state(state, title="Random-Action Phase 1 Rollout", trajectories=frame_trajectories)
        frames.append(_figure_to_array(fig))
    imageio.mimsave(rollout_gif, frames, duration=0.35)

    summary: Dict[str, Any] = {
        "generated_at": timestamp(),
        "environment": environment_name,
        "seed": seed,
        "config": config.to_dict(),
        "observation_shape": list(obs.shape),
        "obstacle_grid_shape": list(states[0].obstacles.shape),
        "steps": states[-1].step,
        "success_count": states[-1].success_count,
        "mean_reward": float(total_rewards.mean()),
        "final_state": states[-1].to_dict(),
        "files": {
            "initial_png": str(initial_png),
            "final_png": str(final_png),
            "rollout_gif": str(rollout_gif),
            "summary_json": str(summary_json),
            "report_html": str(report_html),
        },
    }
    save_json(summary, summary_json)
    write_html_report(summary, report_html)
    return summary


if __name__ == "__main__":
    result = create_phase1_preview()
    print("Phase 1 visual preview created:")
    for name, path in result["files"].items():
        print(f"  {name}: {path}")
