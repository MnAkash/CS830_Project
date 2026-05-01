#!/usr/bin/env python3
"""Akash PACT physical-adversary curriculum training and evaluation.

This script owns Moniruzzaman Akash's defence track: continue PPO training
from the shared baseline while gradually increasing physical adversary agents.
It also creates paired baseline-vs-curriculum evaluations for paper/poster
figures.

The implementation lives outside ``cs830_shared_baseline``. PACT-local PPO and
evaluation helpers are preferred, with the shared baseline source kept as a
fallback for unchanged utilities such as attacks and JSON helpers.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pogema import GridConfig

try:
    from .path_utils import PACT_DIR, configure_paths, shared_path
except ImportError:  # Support direct execution: python pact/curriculum_train.py
    from path_utils import PACT_DIR, configure_paths, shared_path

configure_paths()

from attacks import fgsm_attack
from evaluate_fragility import (
    plot_fragility,
    plot_physical_comparison,
    run_fragility_sweep,
    run_physical_attack_sweep,
)
from ppo_mapf import PPOTrainer, PolicyNetwork
from utils import ensure_dir, save_json, set_global_seed, timestamp


DEFAULT_BASELINE = shared_path("models", "phase2_smoke_baseline", "best_policy.pt")
DEFAULT_MODEL_DIR = PACT_DIR / "models" / "main_akash_curriculum"
DEFAULT_RESULTS_DIR = PACT_DIR / "results" / "main" / "akash"


def make_grid_config(name: str, seed: int) -> GridConfig:
    """Return the experiment grid config by name."""
    configs = {
        "smoke": dict(num_agents=4, size=8, density=0.10, max_episode_steps=64),
        "quick": dict(num_agents=8, size=16, density=0.30, max_episode_steps=128),
        "main": dict(num_agents=16, size=20, density=0.30, max_episode_steps=192),
        "scale": dict(num_agents=32, size=32, density=0.30, max_episode_steps=256),
    }
    key = name.lower()
    if key not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown config '{name}'. Valid configs: {valid}")
    return GridConfig(seed=seed, obs_radius=2, on_target="finish", **configs[key])


def grid_config_to_dict(grid_config: GridConfig) -> Dict[str, Any]:
    """Convert a POGEMA GridConfig subset to JSON-friendly values."""
    return {
        "num_agents": grid_config.num_agents,
        "size": grid_config.size,
        "density": grid_config.density,
        "obs_radius": grid_config.obs_radius,
        "max_episode_steps": grid_config.max_episode_steps,
        "seed": grid_config.seed,
        "on_target": grid_config.on_target,
    }


def curriculum_n_adversary(
    iteration: int,
    total_iterations: int,
    max_adversary: int,
    warmup_fraction: float,
) -> int:
    """Ramp from 0 adversaries to max_adversary over the warmup window."""
    if max_adversary <= 0:
        return 0
    if total_iterations <= 1:
        return max_adversary

    warmup_iters = max(2, int(math.ceil(total_iterations * warmup_fraction)))
    if iteration <= 1:
        return 0
    if iteration >= warmup_iters:
        return max_adversary

    progress = (iteration - 1) / max(1, warmup_iters - 1)
    return int(round(progress * max_adversary))


def load_policy(checkpoint_path: Path, grid_config: GridConfig, device: str) -> PolicyNetwork:
    """Load a policy checkpoint for the given observation radius."""
    resolved_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    policy = PolicyNetwork(obs_size=2 * grid_config.obs_radius + 1)
    state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    policy.load_state_dict(state)
    policy.to(resolved_device)
    policy.eval()
    return policy


def plot_curriculum_history(history: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    """Plot Akash curriculum training curves."""
    ensure_dir(output_path.parent)
    steps = np.array([row["total_steps"] for row in history], dtype=np.float32)

    def values(key: str) -> np.ndarray:
        return np.array([row.get(key, np.nan) for row in history], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    axes[0, 0].plot(steps, values("success_rate"), marker="o", color="tab:green")
    axes[0, 0].set_title("Training Success Rate")
    axes[0, 0].set_ylim(-0.05, 1.05)

    axes[0, 1].plot(steps, values("mean_reward"), marker="o", color="tab:blue")
    axes[0, 1].set_title("Mean Episode Reward")

    axes[1, 0].plot(steps, values("entropy"), marker="o", color="tab:purple")
    axes[1, 0].set_title("Policy Entropy")

    axes[1, 1].step(steps, values("n_adversary"), where="post", color="tab:red")
    axes[1, 1].set_title("Curriculum Chaser Count")
    axes[1, 1].set_ylabel("Physical adversaries")

    for ax in axes.flat:
        ax.set_xlabel("Social-agent environment steps")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Akash Physical-Adversary Curriculum Training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def train_curriculum(args: argparse.Namespace) -> Dict[str, Any]:
    """Run curriculum PPO training from the shared baseline checkpoint."""
    set_global_seed(args.seed)
    model_dir = ensure_dir(args.save_dir)
    results_dir = ensure_dir(args.results_dir)
    grid_config = make_grid_config(args.config, args.seed)
    resolved_device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    trainer = PPOTrainer(
        grid_config,
        device=resolved_device,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        obs_noise_std=args.obs_noise_std,
        n_adversary=0,
        adversary_strategy=args.adversary_strategy,
        adversary_seed=args.seed,
    )

    baseline_checkpoint = Path(args.baseline_checkpoint)
    if baseline_checkpoint.exists():
        state = torch.load(baseline_checkpoint, map_location=resolved_device, weights_only=True)
        trainer.policy.load_state_dict(state)
        print(f"Loaded shared baseline checkpoint: {baseline_checkpoint}")
    else:
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")

    steps_per_iter = trainer.n_steps * trainer.n_agents
    n_iters = max(1, args.total_timesteps // steps_per_iter)
    history: List[Dict[str, Any]] = []
    all_episodes: List[Dict[str, Any]] = []
    total_steps = 0
    best_curriculum_success = -1.0
    start = time.time()

    run_config = {
        "generated_at": timestamp(),
        "mode": "train",
        "config_name": args.config,
        "grid_config": grid_config_to_dict(grid_config),
        "seed": args.seed,
        "device": resolved_device,
        "baseline_checkpoint": str(baseline_checkpoint),
        "total_timesteps": args.total_timesteps,
        "n_iters": n_iters,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "max_adversary": args.max_adversary,
        "warmup_fraction": args.warmup_fraction,
        "adversary_strategy": args.adversary_strategy,
        "obs_noise_std": args.obs_noise_std,
    }
    save_json(run_config, model_dir / "run_config.json")

    print(
        f"Akash curriculum: {n_iters} iters, {steps_per_iter} steps/iter, "
        f"strategy={args.adversary_strategy}, max_adv={args.max_adversary}, "
        f"warmup={args.warmup_fraction:.0%}, device={resolved_device}"
    )
    print("-" * 80)

    for iteration in range(1, n_iters + 1):
        n_adv = curriculum_n_adversary(
            iteration, n_iters, args.max_adversary, args.warmup_fraction,
        )
        trainer.set_n_adversary(n_adv)
        (obs, act, lp, adv, ret, mask), episodes = trainer.collect_rollout()
        loss_info = trainer.train_step(obs, act, lp, adv, ret, mask)
        all_episodes.extend(episodes)
        total_steps += steps_per_iter

        recent = all_episodes[-20:] if len(all_episodes) >= 20 else all_episodes
        if recent:
            mean_reward = float(np.mean([e["mean_reward"] for e in recent]))
            mean_success = float(np.mean([e["success_rate"] for e in recent]))
            mean_length = float(np.mean([e["length"] for e in recent]))
        else:
            mean_reward = mean_success = mean_length = float("nan")

        row = {
            "iteration": iteration,
            "total_steps": total_steps,
            "n_adversary": n_adv,
            "adversary_strategy": args.adversary_strategy,
            "warmup_fraction": args.warmup_fraction,
            "mean_reward": mean_reward,
            "success_rate": mean_success,
            "mean_length": mean_length,
            "policy_loss": loss_info["policy"],
            "value_loss": loss_info["value"],
            "entropy": loss_info["entropy"],
            "episodes_completed": len(all_episodes),
        }
        history.append(row)

        robust_phase = n_adv == args.max_adversary or args.max_adversary == 0
        if recent and robust_phase and mean_success > best_curriculum_success:
            best_curriculum_success = mean_success
            torch.save(trainer.policy.state_dict(), model_dir / "best_policy.pt")

        if iteration % args.log_interval == 0 or iteration == 1 or iteration == n_iters:
            fps = total_steps / max(1e-6, time.time() - start)
            print(
                f"Iter {iteration:4d}/{n_iters:<4d} | Steps {total_steps:>8d} | "
                f"Adv {n_adv:2d} | Succ {mean_success:.1%} | Rew {mean_reward:+.3f} | "
                f"Len {mean_length:5.1f} | Ent {loss_info['entropy']:.3f} | FPS {fps:.0f}"
            )

    torch.save(trainer.policy.state_dict(), model_dir / "final_policy.pt")
    if not (model_dir / "best_policy.pt").exists():
        shutil.copyfile(model_dir / "final_policy.pt", model_dir / "best_policy.pt")
    save_json(history, model_dir / "training_history.json")
    plot_curriculum_history(history, results_dir / "akash_curriculum_training.png")
    trainer.env.close()

    summary = {
        "run_config": run_config,
        "best_curriculum_success": best_curriculum_success,
        "episodes_completed": len(all_episodes),
        "files": {
            "best_policy": str(model_dir / "best_policy.pt"),
            "final_policy": str(model_dir / "final_policy.pt"),
            "history": str(model_dir / "training_history.json"),
            "training_plot": str(results_dir / "akash_curriculum_training.png"),
        },
    }
    save_json(summary, results_dir / "akash_training_summary.json")
    print("-" * 80)
    print(f"Saved Akash curriculum checkpoint to {model_dir}")
    return summary


def area_under_curve(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute normalized area under a curve for compact robustness summaries."""
    if len(xs) < 2:
        return float(ys[0]) if ys else float("nan")
    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    span = float(xs_arr.max() - xs_arr.min())
    if span <= 0:
        return float(np.mean(ys_arr))
    return float(np.trapz(ys_arr, xs_arr) / span)


def summarize_eval(baseline_physical, akash_physical, baseline_fgsm, akash_fgsm) -> Dict[str, Any]:
    """Create paper/poster-friendly derived metrics."""
    def by_key(rows, key_name, key_value):
        return next(row for row in rows if row[key_name] == key_value)

    baseline_clean = by_key(baseline_physical, "n_adversary", 0)["success_rate"]
    akash_clean = by_key(akash_physical, "n_adversary", 0)["success_rate"]
    baseline_phys4 = by_key(baseline_physical, "n_adversary", 4)["success_rate"]
    akash_phys4 = by_key(akash_physical, "n_adversary", 4)["success_rate"]

    baseline_fgsm_015 = by_key(baseline_fgsm, "epsilon", 0.15)["success_rate"]
    akash_fgsm_015 = by_key(akash_fgsm, "epsilon", 0.15)["success_rate"]

    physical_x = [row["n_adversary"] for row in baseline_physical]
    fgsm_x = [row["epsilon"] for row in baseline_fgsm]
    return {
        "baseline_clean_success": baseline_clean,
        "akash_clean_success": akash_clean,
        "baseline_physical_4_success": baseline_phys4,
        "akash_physical_4_success": akash_phys4,
        "physical_4_absolute_gain": akash_phys4 - baseline_phys4,
        "baseline_physical_drop_0_to_4": baseline_clean - baseline_phys4,
        "akash_physical_drop_0_to_4": akash_clean - akash_phys4,
        "baseline_fgsm_0_15_success": baseline_fgsm_015,
        "akash_fgsm_0_15_success": akash_fgsm_015,
        "fgsm_0_15_absolute_gain": akash_fgsm_015 - baseline_fgsm_015,
        "baseline_physical_auc": area_under_curve(
            physical_x, [row["success_rate"] for row in baseline_physical],
        ),
        "akash_physical_auc": area_under_curve(
            physical_x, [row["success_rate"] for row in akash_physical],
        ),
        "baseline_fgsm_auc": area_under_curve(
            fgsm_x, [row["success_rate"] for row in baseline_fgsm],
        ),
        "akash_fgsm_auc": area_under_curve(
            fgsm_x, [row["success_rate"] for row in akash_fgsm],
        ),
    }


def evaluate_curriculum(args: argparse.Namespace) -> Dict[str, Any]:
    """Run paired baseline-vs-Akash evaluations and save plots/JSON."""
    set_global_seed(args.seed)
    results_dir = ensure_dir(args.results_dir)
    grid_config = make_grid_config(args.config, args.seed)
    resolved_device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    baseline_policy = load_policy(Path(args.baseline_checkpoint), grid_config, resolved_device)
    akash_policy = load_policy(Path(args.akash_checkpoint), grid_config, resolved_device)
    adversary_counts = [0, 1, 2, 4]
    combined_counts = [0, 2, 4]
    epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    print("Evaluating physical adversary sweep: baseline")
    baseline_physical = run_physical_attack_sweep(
        baseline_policy,
        grid_config,
        adversary_counts=adversary_counts,
        n_episodes=args.eval_episodes,
        device=resolved_device,
        adversary_strategy=args.adversary_strategy,
    )
    print("Evaluating physical adversary sweep: Akash curriculum")
    akash_physical = run_physical_attack_sweep(
        akash_policy,
        grid_config,
        adversary_counts=adversary_counts,
        n_episodes=args.eval_episodes,
        device=resolved_device,
        adversary_strategy=args.adversary_strategy,
    )

    print("Evaluating FGSM sweep: baseline")
    baseline_fgsm = run_fragility_sweep(
        baseline_policy,
        grid_config,
        attack_name="fgsm",
        epsilons=epsilons,
        n_episodes=args.eval_episodes,
        device=resolved_device,
    )
    print("Evaluating FGSM sweep: Akash curriculum")
    akash_fgsm = run_fragility_sweep(
        akash_policy,
        grid_config,
        attack_name="fgsm",
        epsilons=epsilons,
        n_episodes=args.eval_episodes,
        device=resolved_device,
    )

    print("Evaluating combined physical + FGSM threat")
    combined_kwargs = {"epsilon": args.combined_epsilon}
    baseline_combined = run_physical_attack_sweep(
        baseline_policy,
        grid_config,
        adversary_counts=combined_counts,
        n_episodes=args.eval_episodes,
        device=resolved_device,
        attack_fn=fgsm_attack,
        attack_kwargs=combined_kwargs,
        adversary_strategy=args.adversary_strategy,
    )
    akash_combined = run_physical_attack_sweep(
        akash_policy,
        grid_config,
        adversary_counts=combined_counts,
        n_episodes=args.eval_episodes,
        device=resolved_device,
        attack_fn=fgsm_attack,
        attack_kwargs=combined_kwargs,
        adversary_strategy=args.adversary_strategy,
    )

    summary_metrics = summarize_eval(
        baseline_physical, akash_physical, baseline_fgsm, akash_fgsm,
    )
    results = {
        "generated_at": timestamp(),
        "config_name": args.config,
        "grid_config": grid_config_to_dict(grid_config),
        "seed": args.seed,
        "device": resolved_device,
        "eval_episodes": args.eval_episodes,
        "adversary_strategy": args.adversary_strategy,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "akash_checkpoint": str(args.akash_checkpoint),
        "physical": {
            "baseline": baseline_physical,
            "akash_curriculum": akash_physical,
        },
        "fgsm": {
            "baseline": baseline_fgsm,
            "akash_curriculum": akash_fgsm,
        },
        "combined_fgsm_physical": {
            "epsilon": args.combined_epsilon,
            "baseline": baseline_combined,
            "akash_curriculum": akash_combined,
        },
        "summary_metrics": summary_metrics,
    }

    save_json(results, results_dir / "akash_eval_results.json")
    save_json(summary_metrics, results_dir / "akash_summary.json")
    plot_physical_comparison(
        [
            ("Baseline", baseline_physical, "tab:red"),
            ("Akash curriculum", akash_physical, "tab:blue"),
        ],
        title=f"Physical Adversary Robustness ({args.adversary_strategy})",
        save_path=results_dir / "akash_physical_comparison.png",
    )
    plot_fragility(
        {
            "Baseline FGSM": baseline_fgsm,
            "Akash curriculum FGSM": akash_fgsm,
        },
        title="Cross-Robustness under FGSM Observation Attack",
        save_path=results_dir / "akash_fgsm_comparison.png",
    )
    save_json(
        {
            "physical_plot": str(results_dir / "akash_physical_comparison.png"),
            "fgsm_plot": str(results_dir / "akash_fgsm_comparison.png"),
        },
        results_dir / "akash_plot_files.json",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "full"], default="train")
    parser.add_argument("--config", choices=["smoke", "quick", "main", "scale"], default="smoke")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--akash-checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "best_policy.pt")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--total-timesteps", type=int, default=65_536)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-adversary", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.40)
    parser.add_argument(
        "--adversary-strategy",
        choices=["astar_pursuit", "random_walk", "goal_blocking", "mixed"],
        default="astar_pursuit",
    )
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--combined-epsilon", type=float, default=0.15)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.save_dir)
    ensure_dir(args.results_dir)

    if args.mode in {"train", "full"}:
        train_summary = train_curriculum(args)
        if args.mode == "full":
            args.akash_checkpoint = Path(train_summary["files"]["best_policy"])

    if args.mode in {"evaluate", "full"}:
        if not Path(args.akash_checkpoint).exists():
            raise FileNotFoundError(f"Akash checkpoint not found: {args.akash_checkpoint}")
        evaluate_curriculum(args)


if __name__ == "__main__":
    main()