"""
Observation-space attacks for POGEMA MAPF policies.

Attacks perturb the observation tensor before it's fed to the policy network.
All attacks take (obs_tensor, policy, **kwargs) and return a perturbed tensor.

POGEMA observations are binary {0, 1} grids of shape (batch, 3, H, W).
After perturbation we clamp to [0, 1] to stay in the valid range.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────
# 1. Random Noise Attack
# ─────────────────────────────────────────────────────────────────────
def random_noise_attack(obs, policy, epsilon=0.1, **kwargs):
    """
    Add uniform random noise in [-epsilon, epsilon] to observations.
    This simulates sensor noise or random corruption.

    Args:
        obs: (batch, 3, H, W) float tensor
        policy: unused (kept for API consistency)
        epsilon: noise magnitude

    Returns:
        Perturbed observation tensor clamped to [0, 1].
    """
    noise = torch.empty_like(obs).uniform_(-epsilon, epsilon)
    perturbed = obs + noise
    return perturbed.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 2. FGSM Attack (Fast Gradient Sign Method)
# ─────────────────────────────────────────────────────────────────────
def fgsm_attack(obs, policy, epsilon=0.1, **kwargs):
    """
    One-step gradient-based attack that maximizes policy loss.

    Computes the gradient of the negative log-probability of the
    action the policy would have taken (i.e., tries to make the
    policy's preferred action less likely).

    Args:
        obs: (batch, 3, H, W) float tensor (will be detached+cloned)
        policy: the policy network
        epsilon: L-inf perturbation budget

    Returns:
        Perturbed observation tensor clamped to [0, 1].
    """
    obs_adv = obs.detach().clone().requires_grad_(True)

    logits, _ = policy(obs_adv)
    # The action the policy would take on clean obs
    with torch.no_grad():
        clean_actions = logits.argmax(dim=-1)

    # Loss = negative log-prob of the clean action (we want to maximize this)
    dist = Categorical(logits=logits)
    loss = -dist.log_prob(clean_actions).mean()

    policy.zero_grad()
    loss.backward()

    # FGSM: step in the direction of the gradient sign
    grad_sign = obs_adv.grad.sign()
    perturbed = obs.detach() + epsilon * grad_sign
    return perturbed.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 3. PGD Attack (Projected Gradient Descent) — Tier 3
# ─────────────────────────────────────────────────────────────────────
def pgd_attack(obs, policy, epsilon=0.1, n_steps=10, step_size=None, **kwargs):
    """
    Multi-step PGD attack. Iteratively applies FGSM-like steps and
    projects back onto the L-inf epsilon ball.

    Args:
        obs: (batch, 3, H, W) float tensor
        policy: the policy network
        epsilon: L-inf perturbation budget
        n_steps: number of PGD iterations
        step_size: per-step size (default: 2*epsilon/n_steps)

    Returns:
        Perturbed observation tensor clamped to [0, 1].
    """
    if step_size is None:
        step_size = 2 * epsilon / n_steps

    # Start from random point within epsilon ball
    delta = torch.empty_like(obs).uniform_(-epsilon, epsilon)
    delta = delta.clamp(-epsilon, epsilon)

    # Get the clean action for the attack objective
    with torch.no_grad():
        clean_logits, _ = policy(obs)
        clean_actions = clean_logits.argmax(dim=-1)

    for _ in range(n_steps):
        delta.requires_grad_(True)
        obs_adv = (obs.detach() + delta).clamp(0.0, 1.0)

        logits, _ = policy(obs_adv)
        dist = Categorical(logits=logits)
        loss = -dist.log_prob(clean_actions).mean()

        policy.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        # Step in gradient direction
        grad_sign = delta.grad.sign()
        delta = delta.detach() + step_size * grad_sign
        # Project back onto epsilon ball
        delta = delta.clamp(-epsilon, epsilon)

    perturbed = (obs.detach() + delta.detach()).clamp(0.0, 1.0)
    return perturbed


# ─────────────────────────────────────────────────────────────────────
# 4. Partial attack wrapper
# ─────────────────────────────────────────────────────────────────────
def partial_attack(obs, policy, base_attack_fn, fraction=0.25, **kwargs):
    """
    Apply an attack to only a fraction of agents.

    Args:
        obs: (n_agents, 3, H, W)
        policy: policy network
        base_attack_fn: the underlying attack function
        fraction: fraction of agents to perturb (0.0 to 1.0)
        **kwargs: passed to base_attack_fn

    Returns:
        Perturbed observation tensor (only some agents affected).
    """
    n_agents = obs.shape[0]
    n_attack = max(1, int(n_agents * fraction))

    # Randomly select which agents to attack
    attack_indices = torch.randperm(n_agents)[:n_attack]

    perturbed = obs.clone()
    obs_subset = obs[attack_indices]
    perturbed_subset = base_attack_fn(obs_subset, policy, **kwargs)
    perturbed[attack_indices] = perturbed_subset

    return perturbed


# ─────────────────────────────────────────────────────────────────────
# Registry for easy lookup
# ─────────────────────────────────────────────────────────────────────
ATTACKS = {
    "none": None,
    "random": random_noise_attack,
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


# ─────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from ppo_mapf import PolicyNetwork
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork().to(device)
    policy.eval()

    # Fake observation batch
    obs = torch.rand(8, 3, 5, 5, device=device)

    print("Testing attacks on random policy...")

    # Test random noise
    perturbed = random_noise_attack(obs, policy, epsilon=0.1)
    diff = (perturbed - obs).abs().max().item()
    assert diff <= 0.1 + 1e-6, f"Random noise exceeded epsilon: {diff}"
    assert perturbed.min() >= 0 and perturbed.max() <= 1
    print(f"  [PASS] Random noise: max perturbation = {diff:.4f}")

    # Test FGSM
    policy.train()  # need gradients
    perturbed = fgsm_attack(obs, policy, epsilon=0.1)
    diff = (perturbed - obs).abs().max().item()
    assert diff <= 0.1 + 1e-6, f"FGSM exceeded epsilon: {diff}"
    assert perturbed.min() >= 0 and perturbed.max() <= 1
    print(f"  [PASS] FGSM: max perturbation = {diff:.4f}")

    # Test PGD
    perturbed = pgd_attack(obs, policy, epsilon=0.1, n_steps=5)
    diff = (perturbed - obs).abs().max().item()
    assert diff <= 0.1 + 1e-6, f"PGD exceeded epsilon: {diff}"
    assert perturbed.min() >= 0 and perturbed.max() <= 1
    print(f"  [PASS] PGD: max perturbation = {diff:.4f}")

    # Test partial attack
    perturbed = partial_attack(obs, policy, fgsm_attack, fraction=0.25, epsilon=0.1)
    n_changed = (perturbed != obs).any(dim=(1, 2, 3)).sum().item()
    print(f"  [PASS] Partial FGSM (25%): {n_changed}/8 agents perturbed")

    print("\nAll attack tests passed!")
