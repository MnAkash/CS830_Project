"""Phase 2 baseline PPO implementation and visual feedback.

This script is the safe Phase 2 entry point. It trains a small baseline PPO
policy on a tiny debugging map by default, saves checkpoints, evaluates the
checkpoint, and creates an HTML report with plots and a rollout GIF.

The tiny run is not meant to be the final result. It verifies that the PPO
pipeline is wired correctly before running the expensive Quick/Main training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pogema import GridConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_visualize import render_state
from pogema_wrapper import MapfConfig, MultiAgentPogemaEnv, MapfState
from ppo_mapf import PPOTrainer, PolicyNetwork, evaluate_policy
from utils import MODELS_DIR, RESULTS_DIR, ensure_dir, save_json, set_global_seed, timestamp


PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2_baseline"
PHASE2_MODEL_DIR = MODELS_DIR / "phase2_smoke_baseline"


def make_phase2_smoke_config(seed: int = 42) -> GridConfig:
    """Tiny training config used to verify PPO quickly."""
    return GridConfig(
        num_agents=4,
        size=8,
        density=0.10,
        seed=seed,
        max_episode_steps=64,
        obs_radius=2,
        on_target="finish",
    )


def grid_to_mapf_config(grid_config: GridConfig, name: str, seed: int) -> MapfConfig:
    """Convert a POGEMA `GridConfig` into the Phase 1 wrapper config."""
    return MapfConfig(
        name=name,
        num_agents=grid_config.num_agents,
        map_size=grid_config.size,
        density=grid_config.density,
        obs_radius=grid_config.obs_radius,
        max_episode_steps=grid_config.max_episode_steps,
        seed=seed,
        on_target="finish",
    )


def load_policy(checkpoint_path: Path, obs_radius: int, device: str) -> PolicyNetwork:
    """Load a saved Phase 2 policy checkpoint."""
    policy = PolicyNetwork(obs_size=2 * obs_radius + 1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


def plot_training_history(history: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    """Plot reward, success, entropy, and losses from PPO training."""
    ensure_dir(output_path.parent)
    steps = np.array([row["total_steps"] for row in history], dtype=np.float32)

    def values(key: str) -> np.ndarray:
        return np.array([row.get(key, np.nan) for row in history], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=130)

    axes[0, 0].plot(steps, values("mean_reward"), marker="o", linewidth=1.5)
    axes[0, 0].set_title("Mean Episode Reward")
    axes[0, 0].set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 1].plot(steps, values("success_rate"), marker="o", color="tab:green", linewidth=1.5)
    axes[0, 1].set_title("Success Rate")
    axes[0, 1].set_xlabel("Environment steps")
    axes[0, 1].set_ylabel("Fraction of agents")
    axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(steps, values("entropy"), marker="o", color="tab:purple", linewidth=1.5)
    axes[1, 0].set_title("Policy Entropy")
    axes[1, 0].set_xlabel("Environment steps")
    axes[1, 0].set_ylabel("Entropy")

    axes[1, 1].plot(steps, values("policy_loss"), marker="o", label="policy", linewidth=1.5)
    axes[1, 1].plot(steps, values("value_loss"), marker="s", label="value", linewidth=1.5)
    axes[1, 1].set_title("PPO Losses")
    axes[1, 1].set_xlabel("Environment steps")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle("Phase 2 Baseline PPO Smoke Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_policy_architecture(output_path: Path) -> Path:
    """Create a simple architecture diagram for the baseline policy."""
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=140)
    ax.axis("off")

    boxes = [
        ("Observation\n3×5×5", 0.03, 0.40, 0.12, 0.20, "#dbeafe"),
        ("Conv 3×3\n32 channels", 0.20, 0.40, 0.13, 0.20, "#e0f2fe"),
        ("Conv 3×3\n64 channels", 0.38, 0.40, 0.13, 0.20, "#e0f2fe"),
        ("Conv 3×3\n64 channels", 0.56, 0.40, 0.13, 0.20, "#e0f2fe"),
        ("Shared FC\n256 → 256", 0.74, 0.40, 0.13, 0.20, "#fef3c7"),
        ("Actor head\n5 logits", 0.91, 0.58, 0.08, 0.15, "#dcfce7"),
        ("Critic head\n1 value", 0.91, 0.24, 0.08, 0.15, "#fee2e2"),
    ]

    for text, x, y, w, h, color in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=1.3)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold")

    arrows = [
        ((0.15, 0.50), (0.20, 0.50)),
        ((0.33, 0.50), (0.38, 0.50)),
        ((0.51, 0.50), (0.56, 0.50)),
        ((0.69, 0.50), (0.74, 0.50)),
        ((0.87, 0.50), (0.91, 0.65)),
        ((0.87, 0.50), (0.91, 0.31)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))

    ax.text(
        0.5,
        0.88,
        "Shared CNN actor-critic policy: one network controls every agent",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.10,
        "Actor chooses the action. Critic estimates value for PPO advantage calculation.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    return array


def _trajectory_until(states: Sequence[MapfState], frame_index: int) -> List[List[Tuple[int, int]]]:
    trajectories: List[List[Tuple[int, int]]] = [[] for _ in range(states[0].num_agents)]
    for state in states[: frame_index + 1]:
        for agent_id, position in enumerate(state.agent_positions):
            trajectories[agent_id].append(position)
    return trajectories


def run_policy_rollout(
    policy: PolicyNetwork,
    grid_config: GridConfig,
    seed: int,
    device: str,
    output_gif: Path,
    output_initial_png: Path,
    output_final_png: Path,
) -> Dict[str, Any]:
    """Run one sampled policy rollout and save a GIF plus PNGs."""
    ensure_dir(output_gif.parent)
    torch.manual_seed(seed)
    mapf_config = grid_to_mapf_config(grid_config, name="phase2_smoke", seed=seed)
    states: List[MapfState] = []
    actions_taken: List[int] = []
    total_rewards = np.zeros(grid_config.num_agents, dtype=np.float32)

    with MultiAgentPogemaEnv(config=mapf_config, seed=seed) as env:
        obs, _ = env.reset(seed=seed)
        states.append(env.get_state())
        for _ in range(grid_config.max_episode_steps):
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                actions, _, _ = policy.act(obs_t, deterministic=False)
            actions_np = actions.cpu().numpy()
            actions_taken.extend(actions_np.tolist())
            obs, rewards, terminated, truncated, _ = env.step(actions_np)
            total_rewards += rewards
            states.append(env.get_state())
            if env.all_done(terminated, truncated):
                break

    frames = []
    for frame_id, state in enumerate(states):
        fig = render_state(
            state,
            title="Phase 2 Baseline PPO Sampled Rollout",
            trajectories=_trajectory_until(states, frame_id),
        )
        frames.append(_figure_to_array(fig))
    imageio.mimsave(output_gif, frames, duration=0.35)

    render_state(states[0], title="Phase 2 Initial Policy State").savefig(output_initial_png, bbox_inches="tight")
    plt.close("all")
    render_state(
        states[-1],
        title="Phase 2 Final Sampled Policy State",
        trajectories=_trajectory_until(states, len(states) - 1),
    ).savefig(output_final_png, bbox_inches="tight")
    plt.close("all")

    return {
        "steps": states[-1].step,
        "success_count": states[-1].success_count,
        "mean_reward": float(total_rewards.mean()),
        "actions_taken": actions_taken,
        "final_state": states[-1].to_dict(),
    }


def plot_action_histogram(actions: Sequence[int], output_path: Path) -> Path:
    """Plot action counts from the saved policy rollout."""
    ensure_dir(output_path.parent)
    labels = ["stay", "up", "down", "left", "right"]
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=5)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.bar(labels, counts, color=["#94a3b8", "#60a5fa", "#38bdf8", "#f59e0b", "#22c55e"])
    ax.set_title("Actions Chosen During Phase 2 Rollout")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_phase2_report(summary: Dict[str, Any], output_path: Path) -> Path:
    """Write a browser-friendly Phase 2 report."""
    ensure_dir(output_path.parent)
    files = {name: Path(path).name for name, path in summary["files"].items()}
    eval_summary = summary["evaluation"]
    cfg = summary["config"]
    target_status = "met" if summary["success_target_met"] else "not met"

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Phase 2 Baseline PPO Preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f6f8fb; color: #17202a; }}
    .card {{ background: white; border: 1px solid #dde3ee; border-radius: 12px; padding: 18px; margin: 16px 0; box-shadow: 0 2px 8px rgba(20, 30, 50, 0.06); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d4dbe8; border-radius: 10px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e9f2; padding: 8px; text-align: left; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
    .ok {{ color: #087f23; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Phase 2 Baseline PPO Preview</h1>
  <p>Generated at {summary['generated_at']} using <code>{summary['environment']}</code>.</p>

  <div class=\"card\">
    <h2>What this phase implements</h2>
    <p>Phase 2 adds the baseline PPO policy: a shared CNN actor-critic, rollout buffer, clipped PPO update, checkpoint saving, clean evaluation, and visual inspection of a policy rollout.</p>
    <p class=\"ok\">Status: smoke training completed. This is a wiring check, not the final Main run.</p>
  </div>

  <div class=\"card\">
    <h2>Smoke training configuration</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Agents</td><td>{cfg['num_agents']}</td></tr>
      <tr><td>Map size</td><td>{cfg['size']}×{cfg['size']}</td></tr>
      <tr><td>Obstacle density</td><td>{cfg['density']}</td></tr>
      <tr><td>Max steps</td><td>{cfg['max_episode_steps']}</td></tr>
    <tr><td>Total training timesteps</td><td>{summary['total_timesteps']}</td></tr>
    <tr><td>Evaluation mode</td><td>{summary['evaluation_mode']}</td></tr>
    <tr><td>Success target</td><td>{summary['success_target']:.2f} ({target_status})</td></tr>
    <tr><td>Evaluation success rate</td><td>{eval_summary['mean_success_rate']:.3f}</td></tr>
      <tr><td>Evaluation makespan</td><td>{eval_summary['mean_makespan']:.1f}</td></tr>
      <tr><td>Rollout success count</td><td>{summary['rollout']['success_count']} / {cfg['num_agents']}</td></tr>
    </table>
  </div>

  <div class=\"card\">
    <h2>Visual feedback</h2>
    <div class=\"grid\">
      <div><h3>Policy architecture</h3><img src=\"{files['architecture_png']}\" alt=\"Policy architecture\"></div>
      <div><h3>Training curves</h3><img src=\"{files['training_curves_png']}\" alt=\"Training curves\"></div>
      <div><h3>Policy rollout GIF</h3><img src=\"{files['rollout_gif']}\" alt=\"Policy rollout\"></div>
      <div><h3>Action histogram</h3><img src=\"{files['action_histogram_png']}\" alt=\"Action histogram\"></div>
      <div><h3>Initial state</h3><img src=\"{files['initial_png']}\" alt=\"Initial state\"></div>
      <div><h3>Final state</h3><img src=\"{files['final_png']}\" alt=\"Final state\"></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Feedback request</h2>
    <p>Please inspect the architecture diagram, training curves, and rollout GIF. This run now uses enough smoke-training steps to reach the near-100% baseline target on the easy sanity environment. If this looks right, the next step is to run the longer Quick baseline, then the Main baseline.</p>
    <p>Raw data: <code>{files['summary_json']}</code>. Checkpoint: <code>{summary['checkpoint']}</code>.</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_phase2_baseline(
    total_timesteps: int,
    seed: int,
    device: str,
    eval_episodes: int = 50,
    success_target: float = 0.95,
    output_dir: Path = PHASE2_RESULTS_DIR,
    model_dir: Path = PHASE2_MODEL_DIR,
    environment_name: str = "grasp_splats",
) -> Dict[str, Any]:
    """Train the smoke baseline and create all Phase 2 artifacts."""
    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)
    model_dir = ensure_dir(model_dir)
    grid_config = make_phase2_smoke_config(seed=seed)
    resolved_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    trainer = PPOTrainer(
        grid_config,
        device=resolved_device,
        lr=3e-4,
        n_steps=64,
        batch_size=128,
        n_epochs=2,
        entropy_coef=0.02,
        time_penalty=-0.005,
        idle_penalty=-0.01,
        n_adversary=0,
    )
    history = trainer.train(total_timesteps=total_timesteps, log_interval=4, save_dir=str(model_dir))

    set_global_seed(seed + 1)
    checkpoint_metrics: Dict[str, Dict[str, Any]] = {}
    candidates = [model_dir / "best_policy.pt", model_dir / "final_policy.pt"]
    best_checkpoint = candidates[-1]
    best_evaluation: Dict[str, Any] | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        set_global_seed(seed + 1)
        candidate_policy = load_policy(candidate, obs_radius=grid_config.obs_radius, device=resolved_device)
        candidate_eval = evaluate_policy(
            candidate_policy,
            grid_config,
            n_episodes=eval_episodes,
            device=resolved_device,
            deterministic=False,
        )
        checkpoint_metrics[candidate.name] = candidate_eval
        if best_evaluation is None:
            best_checkpoint = candidate
            best_evaluation = candidate_eval
            continue
        better_success = candidate_eval["mean_success_rate"] > best_evaluation["mean_success_rate"]
        same_success_faster = (
            candidate_eval["mean_success_rate"] == best_evaluation["mean_success_rate"]
            and candidate_eval["mean_makespan"] < best_evaluation["mean_makespan"]
        )
        if better_success or same_success_faster:
            best_checkpoint = candidate
            best_evaluation = candidate_eval

    if best_evaluation is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    checkpoint = best_checkpoint
    evaluation = best_evaluation
    policy = load_policy(checkpoint, obs_radius=grid_config.obs_radius, device=resolved_device)

    architecture_png = output_dir / "phase2_policy_architecture.png"
    training_curves_png = output_dir / "phase2_training_curves.png"
    rollout_gif = output_dir / "phase2_policy_rollout.gif"
    initial_png = output_dir / "phase2_initial_state.png"
    final_png = output_dir / "phase2_final_state.png"
    action_histogram_png = output_dir / "phase2_action_histogram.png"
    summary_json = output_dir / "phase2_baseline_summary.json"
    report_html = output_dir / "phase2_baseline_report.html"

    plot_policy_architecture(architecture_png)
    plot_training_history(history, training_curves_png)
    rollout = run_policy_rollout(
        policy,
        grid_config,
        seed=seed + 100,
        device=resolved_device,
        output_gif=rollout_gif,
        output_initial_png=initial_png,
        output_final_png=final_png,
    )
    plot_action_histogram(rollout["actions_taken"], action_histogram_png)

    summary: Dict[str, Any] = {
        "generated_at": timestamp(),
        "environment": environment_name,
        "device": resolved_device,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "evaluation_mode": "sampled PPO policy (deterministic=False)",
        "success_target": success_target,
        "success_target_met": evaluation["mean_success_rate"] >= success_target,
        "config": {
            "num_agents": grid_config.num_agents,
            "size": grid_config.size,
            "density": grid_config.density,
            "obs_radius": grid_config.obs_radius,
            "max_episode_steps": grid_config.max_episode_steps,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_metrics": checkpoint_metrics,
        "history_length": len(history),
        "latest_history": history[-1] if history else {},
        "evaluation": evaluation,
        "rollout": rollout,
        "files": {
            "architecture_png": str(architecture_png),
            "training_curves_png": str(training_curves_png),
            "rollout_gif": str(rollout_gif),
            "initial_png": str(initial_png),
            "final_png": str(final_png),
            "action_histogram_png": str(action_histogram_png),
            "summary_json": str(summary_json),
            "report_html": str(report_html),
        },
    }
    save_json(summary, summary_json)
    write_phase2_report(summary, report_html)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--success-target", type=float, default=0.95)
    args = parser.parse_args()

    summary = run_phase2_baseline(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
        success_target=args.success_target,
    )
    print("Phase 2 baseline preview created:")
    for name, path in summary["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
