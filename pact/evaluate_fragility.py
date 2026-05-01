"""
Fragility evaluation: Sweep over epsilon values for different attacks.
Generates plots comparing baseline vs robust policies.

Includes:
  - Epsilon sweep for each attack type
  - Partial agent compromise sweep
  - Randomized smoothing evaluation
  - Multi-policy comparison plots (baseline vs robust vs smoothed)
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json, os
from pogema import pogema_v0, GridConfig

from ppo_mapf import PolicyNetwork, evaluate_policy
from attacks import ATTACKS, fgsm_attack, pgd_attack, random_noise_attack, partial_attack


def run_fragility_sweep(
    policy, grid_config, attack_name="fgsm",
    epsilons=None, n_episodes=30, device="cuda",
    n_adversary=0,
    adversary_strategy="astar_pursuit",
):
    """
    Evaluate policy under increasing attack strength.
    When n_adversary > 0, physical adversary chasers are also present.
    Returns list of dicts: [{epsilon, success_rate, makespan}, ...].
    """
    if epsilons is None:
        epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    attack_fn = ATTACKS.get(attack_name)
    results = []

    for eps in epsilons:
        if eps == 0.0 or attack_fn is None:
            res = evaluate_policy(policy, grid_config, n_episodes=n_episodes,
                                  device=device, n_adversary=n_adversary,
                                  adversary_strategy=adversary_strategy)
        else:
            res = evaluate_policy(
                policy, grid_config, n_episodes=n_episodes, device=device,
                attack_fn=attack_fn, attack_kwargs={"epsilon": eps},
                n_adversary=n_adversary,
                adversary_strategy=adversary_strategy,
            )

        entry = {
            "epsilon": eps,
            "attack": attack_name,
            "success_rate": res["mean_success_rate"],
            "std_success_rate": res["std_success_rate"],
            "makespan": res["mean_makespan"],
        }
        results.append(entry)
        print(f"  {attack_name} eps={eps:.2f}: "
              f"success={entry['success_rate']:.1%}, "
              f"makespan={entry['makespan']:.1f}")

    return results


def run_partial_attack_sweep(
    policy, grid_config, fractions=None,
    attack_name="fgsm", epsilon=0.15, n_episodes=30, device="cuda",
    n_adversary=0,
    adversary_strategy="astar_pursuit",
):
    """Sweep fraction of agents attacked."""
    if fractions is None:
        fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    base_fn = ATTACKS.get(attack_name)
    if base_fn is None:
        raise ValueError(f"Unknown attack: {attack_name}")

    results = []
    for frac in fractions:
        if frac == 0.0:
            res = evaluate_policy(policy, grid_config, n_episodes=n_episodes,
                                  device=device, n_adversary=n_adversary,
                                  adversary_strategy=adversary_strategy)
        elif frac >= 1.0:
            res = evaluate_policy(
                policy, grid_config, n_episodes=n_episodes, device=device,
                attack_fn=base_fn, attack_kwargs={"epsilon": epsilon},
                n_adversary=n_adversary,
                adversary_strategy=adversary_strategy,
            )
        else:
            res = evaluate_policy(
                policy, grid_config, n_episodes=n_episodes, device=device,
                attack_fn=partial_attack,
                attack_kwargs={
                    "base_attack_fn": base_fn,
                    "fraction": frac,
                    "epsilon": epsilon,
                },
                n_adversary=n_adversary,
                adversary_strategy=adversary_strategy,
            )

        entry = {
            "fraction": frac,
            "success_rate": res["mean_success_rate"],
            "std_success_rate": res["std_success_rate"],
            "makespan": res["mean_makespan"],
        }
        results.append(entry)
        print(f"  Partial {attack_name} frac={frac:.0%}: "
              f"success={entry['success_rate']:.1%}")

    return results


def plot_fragility(all_results, title="Fragility to Observation Perturbation",
                   save_path=None):
    """Plot success rate vs epsilon for multiple attack types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate
    ax = axes[0]
    for label, data in all_results.items():
        eps = [d["epsilon"] for d in data]
        sr = [d["success_rate"] for d in data]
        std = [d.get("std_success_rate", 0) for d in data]
        ax.errorbar(eps, sr, yerr=std, marker='o', label=label, capsize=3)
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs Attack Strength")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Makespan
    ax = axes[1]
    for label, data in all_results.items():
        eps = [d["epsilon"] for d in data]
        mk = [d["makespan"] for d in data]
        ax.plot(eps, mk, marker='s', label=label)
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Mean Makespan")
    ax.set_title("Makespan vs Attack Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.close()


def plot_comparison(baseline_results, robust_results, attack_name="fgsm",
                    save_path=None):
    """Plot baseline vs robust policy under same attack."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data, color, marker in [
        ("Baseline", baseline_results, "tab:red", "o"),
        ("Robust (Adv-trained)", robust_results, "tab:blue", "s"),
    ]:
        eps = [d["epsilon"] for d in data]
        sr = [d["success_rate"] for d in data]
        std = [d.get("std_success_rate", 0) for d in data]
        ax.errorbar(eps, sr, yerr=std, marker=marker, color=color,
                     label=label, capsize=3, linewidth=2, markersize=8)

    ax.set_xlabel("Perturbation ε", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Baseline vs Robust Policy under {attack_name.upper()} Attack",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved to {save_path}")
    plt.close()


def plot_partial(results, title="Effect of Partial Agent Compromise",
                 save_path=None):
    """Plot success rate vs fraction of compromised agents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in results:
        fracs = [d["fraction"] for d in data]
        srs = [d["success_rate"] for d in data]
        ax.plot(fracs, srs, marker='o', label=label, color=color, linewidth=2)

    ax.set_xlabel("Fraction of Agents Attacked", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Partial attack plot saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Physical adversary sweep
# ─────────────────────────────────────────────────────────────────────
def run_physical_attack_sweep(
    policy, grid_config, adversary_counts=None,
    n_episodes=30, device="cuda",
    attack_fn=None, attack_kwargs=None,
    adversary_strategy="astar_pursuit",
):
    """
    Evaluate policy under varying numbers of physical adversary agents
    (BFS pursuit chasers).

    Args:
        adversary_counts: list of int, number of chaser agents to test
        attack_fn / attack_kwargs: optional observation-space attack
            applied simultaneously (for combined threat evaluation)

    Returns list of dicts: [{n_adversary, success_rate, makespan}, ...].
    """
    if adversary_counts is None:
        adversary_counts = [0, 2, 4, 6, 8]

    results = []
    for n_adv in adversary_counts:
        if attack_fn is not None:
            res = evaluate_policy(
                policy, grid_config, n_episodes=n_episodes,
                device=device, n_adversary=n_adv,
                attack_fn=attack_fn, attack_kwargs=attack_kwargs,
                adversary_strategy=adversary_strategy,
            )
        else:
            res = evaluate_policy(
                policy, grid_config, n_episodes=n_episodes,
                device=device, n_adversary=n_adv,
                adversary_strategy=adversary_strategy,
            )

        entry = {
            "n_adversary": n_adv,
            "adversary_strategy": adversary_strategy,
            "success_rate": res["mean_success_rate"],
            "std_success_rate": res["std_success_rate"],
            "makespan": res["mean_makespan"],
        }
        results.append(entry)
        atk_label = ""
        if attack_fn is not None:
            atk_label = " + obs attack"
        print(f"  {n_adv} adversaries{atk_label}: "
              f"success={entry['success_rate']:.1%}, "
              f"makespan={entry['makespan']:.1f}")

    return results


def plot_physical_comparison(results_list, title="Physical Adversary Resilience",
                             save_path=None):
    """
    Plot baseline vs robust under varying physical adversary counts.

    results_list: list of (label, sweep_data, color)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for label, data, color in results_list:
        xs = [d["n_adversary"] for d in data]
        srs = [d["success_rate"] for d in data]
        stds = [d.get("std_success_rate", 0) for d in data]
        ax.errorbar(xs, srs, yerr=stds, marker='o', color=color,
                     label=label, capsize=3, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Physical Adversary Agents", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate vs Physical Adversaries", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, data, color in results_list:
        xs = [d["n_adversary"] for d in data]
        mks = [d["makespan"] for d in data]
        ax.plot(xs, mks, marker='s', color=color, label=label, linewidth=2)
    ax.set_xlabel("Number of Physical Adversary Agents", fontsize=12)
    ax.set_ylabel("Mean Makespan", fontsize=13)
    ax.set_title("Makespan vs Physical Adversaries", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Physical attack plot saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Smoothed evaluation (randomized smoothing defense)
# ─────────────────────────────────────────────────────────────────────
def evaluate_policy_smoothed(
    policy, grid_config, n_episodes=30, device="cuda",
    n_smooth_samples=10, smooth_sigma=0.1,
    attack_fn=None, attack_kwargs=None,
):
    """
    Evaluate with randomized smoothing: at each step, average softmax
    predictions over n_smooth_samples noisy copies of the observation.

    This provides a principled defense against L2-bounded attacks.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    policy.eval()
    if attack_kwargs is None:
        attack_kwargs = {}

    results = []
    for ep in range(n_episodes):
        gc = GridConfig(
            num_agents=grid_config.num_agents,
            size=grid_config.size,
            density=grid_config.density,
            seed=ep + 1000,
            max_episode_steps=grid_config.max_episode_steps,
            obs_radius=grid_config.obs_radius,
            on_target='finish',
        )
        env = pogema_v0(grid_config=gc)
        obs_list, _ = env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        n_agents = gc.num_agents
        arrival = np.full(n_agents, -1, dtype=np.int32)
        step = 0

        while True:
            obs_t = torch.from_numpy(obs).float().to(device)

            # Apply attack if provided (before smoothing)
            if attack_fn is not None:
                obs_t = attack_fn(obs_t, policy, **attack_kwargs)

            # Randomized smoothing: average predictions over noisy copies
            avg_probs = torch.zeros(n_agents, 5, device=device)
            for _ in range(n_smooth_samples):
                noisy = (obs_t + torch.randn_like(obs_t) * smooth_sigma).clamp(0, 1)
                with torch.no_grad():
                    logits, _ = policy(noisy)
                avg_probs += F.softmax(logits, dim=-1)
            avg_probs /= n_smooth_samples

            actions = avg_probs.argmax(dim=-1)
            actions_np = actions.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = env.step(
                actions_np.tolist()
            )
            terminated = np.array(terminated, dtype=bool)
            truncated = np.array(truncated, dtype=bool)
            step += 1

            newly = terminated & (arrival < 0)
            arrival[newly] = step

            if all(terminated) or all(truncated):
                break
            obs = np.array(next_obs, dtype=np.float32)

        succeeded = (arrival > 0).sum()
        sr = succeeded / n_agents
        if succeeded > 0:
            makespan = int(arrival[arrival > 0].max())
        else:
            makespan = step

        results.append({
            "episode": ep, "success_rate": float(sr),
            "makespan": makespan, "agents_succeeded": int(succeeded),
        })
        env.close()

    srs = [r["success_rate"] for r in results]
    mks = [r["makespan"] for r in results]
    return {
        "mean_success_rate": float(np.mean(srs)),
        "std_success_rate": float(np.std(srs)),
        "mean_makespan": float(np.mean(mks)),
        "std_makespan": float(np.std(mks)),
        "n_episodes": n_episodes,
        "per_episode": results,
    }


def run_smoothed_fragility_sweep(
    policy, grid_config, attack_name="fgsm",
    epsilons=None, n_episodes=20, device="cuda",
    n_smooth_samples=10, smooth_sigma=0.1,
):
    """Fragility sweep with randomized smoothing defense."""
    if epsilons is None:
        epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    attack_fn = ATTACKS.get(attack_name)
    results = []

    for eps in epsilons:
        if eps == 0.0 or attack_fn is None:
            res = evaluate_policy_smoothed(
                policy, grid_config, n_episodes=n_episodes, device=device,
                n_smooth_samples=n_smooth_samples, smooth_sigma=smooth_sigma,
            )
        else:
            res = evaluate_policy_smoothed(
                policy, grid_config, n_episodes=n_episodes, device=device,
                n_smooth_samples=n_smooth_samples, smooth_sigma=smooth_sigma,
                attack_fn=attack_fn, attack_kwargs={"epsilon": eps},
            )

        entry = {
            "epsilon": eps, "attack": attack_name,
            "success_rate": res["mean_success_rate"],
            "std_success_rate": res["std_success_rate"],
            "makespan": res["mean_makespan"],
        }
        results.append(entry)
        print(f"  smoothed_{attack_name} eps={eps:.2f}: "
              f"success={entry['success_rate']:.1%}")

    return results


def plot_multi_comparison(policies_data, attack_name="fgsm", save_path=None):
    """
    Plot multiple policies on same chart.
    policies_data: list of (label, sweep_data, color, marker)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data, color, marker in policies_data:
        eps = [d["epsilon"] for d in data]
        sr = [d["success_rate"] for d in data]
        std = [d.get("std_success_rate", 0) for d in data]
        ax.errorbar(eps, sr, yerr=std, marker=marker, color=color,
                     label=label, capsize=3, linewidth=2, markersize=7)

    ax.set_xlabel("Perturbation ε", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Defense Comparison under {attack_name.upper()} Attack",
                 fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Multi-comparison plot saved to {save_path}")
    plt.close()
