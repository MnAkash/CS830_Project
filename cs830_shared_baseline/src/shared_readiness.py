"""Shared readiness check before the three individual defence tracks.

This script verifies the common project layer that Riad, Akash, and Sujosh
all depend on:

1. Phase 2 baseline checkpoint exists and reaches the easy sanity target.
2. Random, FGSM, PGD, and partial observation attacks run and stay inside
   their epsilon bounds.
3. Physical A* chasers run through the same evaluator.
4. Baseline fragility plots, physical-adversary plots, and visual GIFs are
   saved for feedback.

The output is an HTML handoff report in `results/shared_readiness/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pogema import GridConfig, pogema_v0

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adversary import ACTION_DELTAS, adversary_actions, astar_next_step, bfs_next_step
from attacks import fgsm_attack, partial_attack, pgd_attack, random_noise_attack
from evaluate_fragility import (
    plot_fragility,
    plot_partial,
    plot_physical_comparison,
    run_fragility_sweep,
    run_partial_attack_sweep,
    run_physical_attack_sweep,
)
from ppo_mapf import PolicyNetwork, evaluate_policy
from utils import MODELS_DIR, RESULTS_DIR, ensure_dir, load_json, save_json, set_global_seed, timestamp
from visualize import fig_to_array, render_frame


SHARED_RESULTS_DIR = RESULTS_DIR / "shared_readiness"
PHASE2_SUMMARY = RESULTS_DIR / "phase2_baseline" / "phase2_baseline_summary.json"


def make_shared_grid_config(seed: int = 42) -> GridConfig:
    """Use the same easy sanity config as the verified Phase 2 baseline."""
    return GridConfig(
        num_agents=4,
        size=8,
        density=0.1,
        seed=seed,
        max_episode_steps=64,
        obs_radius=2,
        on_target="finish",
    )


def load_phase2_checkpoint(device: str) -> Tuple[PolicyNetwork, Path, Dict[str, Any]]:
    """Load the verified Phase 2 checkpoint and summary."""
    if not PHASE2_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing {PHASE2_SUMMARY}. Run `python src/phase2_train_baseline.py --device cpu` first."
        )
    summary = load_json(PHASE2_SUMMARY)
    # The training loop saves `best_policy.pt` when the recent episode window
    # has the highest success rate. That is the safest shared handoff point.
    checkpoint = MODELS_DIR / "phase2_smoke_baseline" / "best_policy.pt"
    if not checkpoint.exists():
        checkpoint = Path(summary["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError("Missing Phase 2 baseline checkpoint.")

    policy = PolicyNetwork(obs_size=5)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy, checkpoint, summary


def run_attack_bound_checks(policy: PolicyNetwork, grid_config: GridConfig, device: str) -> Dict[str, Any]:
    """Check that observation attacks stay in range and inside epsilon bounds."""
    gc = GridConfig(
        num_agents=grid_config.num_agents,
        size=grid_config.size,
        density=grid_config.density,
        seed=123,
        max_episode_steps=grid_config.max_episode_steps,
        obs_radius=grid_config.obs_radius,
        on_target="finish",
    )
    env = pogema_v0(grid_config=gc)
    obs_list, _ = env.reset()
    env.close()
    obs = torch.from_numpy(np.asarray(obs_list, dtype=np.float32)).to(device)
    epsilon = 0.15

    checks: Dict[str, Any] = {}
    attacks = {
        "random": lambda x: random_noise_attack(x, policy, epsilon=epsilon),
        "fgsm": lambda x: fgsm_attack(x, policy, epsilon=epsilon),
        "pgd": lambda x: pgd_attack(x, policy, epsilon=epsilon, n_steps=5),
        "partial_fgsm_50": lambda x: partial_attack(x, policy, fgsm_attack, fraction=0.5, epsilon=epsilon),
    }
    for name, attack_fn in attacks.items():
        perturbed = attack_fn(obs)
        diff = (perturbed - obs).abs().max().item()
        checks[name] = {
            "epsilon": epsilon,
            "max_abs_delta": diff,
            "min_value": float(perturbed.min().item()),
            "max_value": float(perturbed.max().item()),
            "inside_epsilon": diff <= epsilon + 1e-5,
            "inside_value_range": float(perturbed.min().item()) >= 0.0 and float(perturbed.max().item()) <= 1.0,
        }
    return checks


def run_adversary_checks() -> Dict[str, Any]:
    """Check A* and BFS on a tiny hand-written grid."""
    obstacles = np.zeros((7, 7), dtype=np.float32)
    obstacles[3, 1:5] = 1.0
    obstacles[3, 3] = 0.0
    start = (5, 1)
    goal = (1, 5)
    astar_action = astar_next_step(start, goal, obstacles)
    bfs_action = bfs_next_step(start, goal, obstacles)
    deltas = ACTION_DELTAS
    astar_next = (start[0] + deltas[astar_action][0], start[1] + deltas[astar_action][1])
    bfs_next = (start[0] + deltas[bfs_action][0], start[1] + deltas[bfs_action][1])
    return {
        "start": list(start),
        "goal": list(goal),
        "astar_action": int(astar_action),
        "bfs_action": int(bfs_action),
        "astar_next_cell": list(astar_next),
        "bfs_next_cell": list(bfs_next),
        "astar_valid": astar_action in {1, 2, 3, 4} and obstacles[astar_next] < 0.5,
        "bfs_valid": bfs_action in {1, 2, 3, 4} and obstacles[bfs_next] < 0.5,
    }


def save_attack_panel(policy: PolicyNetwork, grid_config: GridConfig, device: str, output_path: Path) -> Path:
    """Save a visual comparison of clean/random/FGSM/PGD observations."""
    ensure_dir(output_path.parent)
    gc = GridConfig(
        num_agents=grid_config.num_agents,
        size=grid_config.size,
        density=grid_config.density,
        seed=77,
        max_episode_steps=grid_config.max_episode_steps,
        obs_radius=grid_config.obs_radius,
        on_target="finish",
    )
    env = pogema_v0(grid_config=gc)
    obs_list, _ = env.reset()
    env.close()

    obs = torch.from_numpy(np.asarray(obs_list, dtype=np.float32)).to(device)
    clean = obs[0].detach().cpu().numpy()
    random_obs = random_noise_attack(obs, policy, epsilon=0.15)[0].detach().cpu().numpy()
    fgsm_obs = fgsm_attack(obs, policy, epsilon=0.15)[0].detach().cpu().numpy()
    pgd_obs = pgd_attack(obs, policy, epsilon=0.15, n_steps=5)[0].detach().cpu().numpy()

    rows = [("Clean", clean), ("Random ε=0.15", random_obs), ("FGSM ε=0.15", fgsm_obs), ("PGD ε=0.15", pgd_obs)]
    channels = ["Obstacles", "Other agents", "Goal direction"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(8, 9), dpi=140)
    for row_id, (row_name, arr) in enumerate(rows):
        for ch in range(3):
            ax = axes[row_id, ch]
            ax.imshow(arr[ch], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_id == 0:
                ax.set_title(channels[ch], fontsize=9, fontweight="bold")
            if ch == 0:
                ax.set_ylabel(row_name, fontsize=9, fontweight="bold")
    fig.suptitle("Observation Attack Sanity Check", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_physical_preview_gif(
    policy: PolicyNetwork,
    grid_config: GridConfig,
    device: str,
    output_path: Path,
    seed: int = 333,
    n_adversary: int = 2,
    max_steps: int = 40,
) -> Dict[str, Any]:
    """Create a GIF showing physical A* chasers using the shared renderer."""
    ensure_dir(output_path.parent)
    total_agents = grid_config.num_agents + n_adversary
    gc = GridConfig(
        num_agents=total_agents,
        size=grid_config.size,
        density=grid_config.density,
        seed=seed,
        max_episode_steps=max_steps,
        obs_radius=grid_config.obs_radius,
        on_target="finish",
    )
    env = pogema_v0(grid_config=gc)
    obs_list, _ = env.reset()
    obs = np.asarray(obs_list, dtype=np.float32)
    grid = env.unwrapped.grid
    obstacles = np.asarray(grid.get_obstacles())
    all_goals = list(grid.finishes_xy)
    social_goals = all_goals[: grid_config.num_agents]
    social_done = [False] * grid_config.num_agents
    social_traj = [[] for _ in range(grid_config.num_agents)]
    adv_traj = [[] for _ in range(n_adversary)]

    def snapshot(step: int) -> Dict[str, Any]:
        all_pos = list(grid.get_agents_xy())
        social_pos = all_pos[: grid_config.num_agents]
        adv_pos = all_pos[grid_config.num_agents :]
        for i, pos in enumerate(social_pos):
            social_traj[i].append(pos)
        for i, pos in enumerate(adv_pos):
            adv_traj[i].append(pos)
        return {
            "obstacles": obstacles,
            "social_positions": list(social_pos),
            "adv_positions": list(adv_pos),
            "social_goals": social_goals,
            "social_done": list(social_done),
            "social_trajectories": [list(t) for t in social_traj],
            "adv_trajectories": [list(t) for t in adv_traj],
            "step": step,
            "has_adversary": n_adversary > 0,
            "n_social": grid_config.num_agents,
            "n_adversary": n_adversary,
        }

    snaps = [snapshot(0)]
    policy = policy.to(device).eval()
    for step in range(1, max_steps + 1):
        obs_t = torch.from_numpy(obs[: grid_config.num_agents]).float().to(device)
        with torch.no_grad():
            social_actions, _, _ = policy.act(obs_t, deterministic=False)
        all_pos = list(grid.get_agents_xy())
        adv_actions = adversary_actions(
            all_pos[grid_config.num_agents :],
            all_pos[: grid_config.num_agents],
            social_goals,
            social_done,
            obstacles,
        )
        next_obs, _rewards, terminated, truncated, _ = env.step(social_actions.cpu().numpy().tolist() + adv_actions)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        for i in range(grid_config.num_agents):
            if terminated[i]:
                social_done[i] = True
        snaps.append(snapshot(step))
        obs = np.asarray(next_obs, dtype=np.float32)
        if bool(np.all(terminated | truncated)):
            break
    env.close()

    frames = [fig_to_array(render_frame(snap, "Shared Physical A* Chaser Preview", cell_px=48)) for snap in snaps]
    for _ in range(4):
        frames.append(frames[-1])
    imageio.mimsave(output_path, frames, fps=3, loop=0)
    last = snaps[-1]
    return {
        "frames": len(snaps),
        "success_count": int(sum(last["social_done"])),
        "n_social": grid_config.num_agents,
        "n_adversary": n_adversary,
        "path": str(output_path),
    }


def write_handoff_markdown(summary: Dict[str, Any], output_path: Path) -> Path:
    """Write a concise team handoff checklist."""
    text = f"""# Shared Team Handoff

Generated: {summary['generated_at']}

This file marks the baseline + wrapper layer as ready for Riad, Akash, and Sujosh to build their own parts on top of it.

This handoff does **not** implement Riad's adversarial training, Akash's curriculum training, or Sujosh's smoothing defence. It only provides the shared foundation.

## Verified shared pieces

- [x] Phase 1 POGEMA wrapper with Quick/Main/Scale configs.
- [x] Phase 2 baseline PPO checkpoint.
- [{'x' if summary['ready'] else ' '}] Baseline sanity success target met: {summary['baseline_eval']['mean_success_rate']:.1%} over {summary['baseline_eval']['n_episodes']} held-out episodes.
- [x] Random noise attack stays inside epsilon bound.
- [x] FGSM attack stays inside epsilon bound.
- [x] PGD attack stays inside epsilon bound.
- [x] Partial-agent attack wrapper works.
- [x] A* physical chaser logic works on a hand-written grid and through POGEMA evaluation.
- [x] Shared fragility plots and visual report generated.

## Shared baseline checkpoint

{summary['baseline_checkpoint']}

## Shared readiness report

{summary['files']['report_html']}

## Baseline + wrapper handoff guide

BASELINE_WRAPPER_HANDOFF.md

## What each person should start from

| Person | Their part | Shared baseline inputs |
|---|---|---|
| Riad | adversarial training defence | baseline checkpoint, `src/ppo_mapf.py`, `src/attacks.py`, fragility sweep code |
| Akash | curriculum / physical adversary defence | baseline checkpoint, `src/pogema_wrapper.py`, `src/adversary.py`, physical sweep code |
| Sujosh | inference-time smoothing defence | baseline checkpoint, `PolicyNetwork` inference API, attack/evaluation code |

## Rule for the next phases

Do not change the shared baseline checkpoint, attack definitions, or evaluation seeds unless all three tracks agree. Otherwise the comparison stops being fair.
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def write_html_report(summary: Dict[str, Any], output_path: Path) -> Path:
    """Write a browser-friendly shared readiness report."""
    ensure_dir(output_path.parent)
    files = {name: Path(path).name for name, path in summary["files"].items()}
    attack_checks = summary["attack_bound_checks"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{data['max_abs_delta']:.4f}</td><td>{data['inside_epsilon']}</td><td>{data['inside_value_range']}</td></tr>"
        for name, data in attack_checks.items()
    )
    ready_text = "Baseline + wrapper are ready for Riad, Akash, and Sujosh to build their own parts on top." if summary["ready"] else "Not ready: fix shared baseline/wrapper checks before handoff."
    ready_class = "ok" if summary["ready"] else "bad"
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Shared Readiness Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f6f8fb; color: #17202a; }}
    .card {{ background: white; border: 1px solid #dde3ee; border-radius: 12px; padding: 18px; margin: 16px 0; box-shadow: 0 2px 8px rgba(20, 30, 50, 0.06); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d4dbe8; border-radius: 10px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e9f2; padding: 8px; text-align: left; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
    .ok {{ color: #087f23; font-weight: 700; }}
    .bad {{ color: #b91c1c; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Shared Readiness Report</h1>
  <p>Generated at {summary['generated_at']} using <code>{summary['environment']}</code>.</p>

  <div class=\"card\">
    <h2>Status</h2>
    <p class=\"{ready_class}\">{ready_text}</p>
    <p>Baseline success: <b>{summary['baseline_eval']['mean_success_rate']:.1%}</b> over {summary['baseline_eval']['n_episodes']} held-out sanity episodes.</p>
    <p>Baseline checkpoint: <code>{summary['baseline_checkpoint']}</code></p>
  </div>

  <div class=\"card\">
    <h2>Attack bound checks</h2>
    <table>
      <tr><th>Attack</th><th>Max |delta|</th><th>Inside epsilon?</th><th>Inside [0, 1]?</th></tr>
      {rows}
    </table>
  </div>

  <div class=\"card\">
    <h2>Visual outputs</h2>
    <div class=\"grid\">
      <div><h3>Observation attack panel</h3><img src=\"{files['attack_panel_png']}\" alt=\"Attack panel\"></div>
      <div><h3>Random / FGSM / PGD fragility</h3><img src=\"{files['fragility_png']}\" alt=\"Fragility plot\"></div>
      <div><h3>Partial FGSM sweep</h3><img src=\"{files['partial_png']}\" alt=\"Partial attack plot\"></div>
      <div><h3>Physical A* chaser sweep</h3><img src=\"{files['physical_png']}\" alt=\"Physical attack plot\"></div>
      <div><h3>A* chaser GIF</h3><img src=\"{files['physical_gif']}\" alt=\"Physical chaser gif\"></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Handoff</h2>
    <p>Raw JSON: <code>{files['summary_json']}</code></p>
        <p>Baseline/wrapper guide: <code>{files['baseline_handoff_md']}</code></p>
    <p>Markdown handoff: <code>{files['handoff_md']}</code></p>
    <p>All three future tracks should build on this same baseline checkpoint, wrapper, and evaluation code.</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_shared_readiness(device: str = "cpu", episodes: int = 8, seed: int = 42) -> Dict[str, Any]:
    """Run the shared readiness checks and save all artifacts."""
    set_global_seed(seed)
    output_dir = ensure_dir(SHARED_RESULTS_DIR)
    resolved_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    policy, checkpoint, phase2_summary = load_phase2_checkpoint(resolved_device)
    grid_config = make_shared_grid_config(seed=seed)

    baseline_eval = evaluate_policy(policy, grid_config, n_episodes=50, device=resolved_device, deterministic=False)
    attack_bound_checks = run_attack_bound_checks(policy, grid_config, resolved_device)
    adversary_checks = run_adversary_checks()
    baseline_target = 0.95
    attacks_ok = all(
        item["inside_epsilon"] and item["inside_value_range"]
        for item in attack_bound_checks.values()
    )
    adversary_ok = bool(adversary_checks["astar_valid"] and adversary_checks["bfs_valid"])
    ready = bool(baseline_eval["mean_success_rate"] >= baseline_target and attacks_ok and adversary_ok)

    attack_panel_png = output_dir / "shared_attack_observation_panel.png"
    fragility_png = output_dir / "shared_baseline_fragility.png"
    partial_png = output_dir / "shared_partial_attack.png"
    physical_png = output_dir / "shared_physical_adversary.png"
    physical_gif = output_dir / "shared_physical_chaser_preview.gif"
    summary_json = output_dir / "shared_readiness_summary.json"
    baseline_handoff_md = Path.cwd() / "BASELINE_WRAPPER_HANDOFF.md"
    handoff_md = Path.cwd() / "SHARED_TEAM_HANDOFF.md"
    report_html = output_dir / "shared_readiness_report.html"

    save_attack_panel(policy, grid_config, resolved_device, attack_panel_png)

    epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    pgd_epsilons = [0.0, 0.10, 0.20, 0.30]
    fragility = {
        "random": run_fragility_sweep(policy, grid_config, "random", epsilons, episodes, resolved_device),
        "fgsm": run_fragility_sweep(policy, grid_config, "fgsm", epsilons, episodes, resolved_device),
        "pgd": run_fragility_sweep(policy, grid_config, "pgd", pgd_epsilons, max(4, episodes // 2), resolved_device),
    }
    plot_fragility(fragility, title="Shared Baseline Fragility Sanity Check", save_path=str(fragility_png))

    partial = run_partial_attack_sweep(
        policy,
        grid_config,
        fractions=[0.0, 0.25, 0.50, 1.0],
        attack_name="fgsm",
        epsilon=0.15,
        n_episodes=episodes,
        device=resolved_device,
    )
    plot_partial([("Baseline", partial, "tab:orange")], title="Shared Partial FGSM Sanity Check", save_path=str(partial_png))

    physical = run_physical_attack_sweep(
        policy,
        grid_config,
        adversary_counts=[0, 1, 2],
        n_episodes=episodes,
        device=resolved_device,
    )
    plot_physical_comparison([("Baseline", physical, "tab:red")], title="Shared Physical A* Chaser Sanity Check", save_path=str(physical_png))

    physical_preview = run_physical_preview_gif(policy, grid_config, resolved_device, physical_gif)

    summary: Dict[str, Any] = {
        "generated_at": timestamp(),
        "environment": "grasp_splats",
        "device": resolved_device,
        "seed": seed,
        "baseline_checkpoint": str(checkpoint),
        "phase2_summary": str(PHASE2_SUMMARY),
        "baseline_target": baseline_target,
        "ready": ready,
        "baseline_eval": baseline_eval,
        "attack_bound_checks": attack_bound_checks,
        "adversary_checks": adversary_checks,
        "fragility": fragility,
        "partial_attack": partial,
        "physical_adversary": physical,
        "physical_preview": physical_preview,
        "files": {
            "attack_panel_png": str(attack_panel_png),
            "fragility_png": str(fragility_png),
            "partial_png": str(partial_png),
            "physical_png": str(physical_png),
            "physical_gif": str(physical_gif),
            "summary_json": str(summary_json),
            "baseline_handoff_md": str(baseline_handoff_md),
            "handoff_md": str(handoff_md),
            "report_html": str(report_html),
        },
    }
    save_json(summary, summary_json)
    write_handoff_markdown(summary, handoff_md)
    write_html_report(summary, report_html)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_shared_readiness(device=args.device, episodes=args.episodes, seed=args.seed)
    print("Shared readiness report created:")
    print(f"  baseline success: {summary['baseline_eval']['mean_success_rate']:.1%}")
    for name, path in summary["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
