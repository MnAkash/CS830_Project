"""
Custom PPO for multi-agent POGEMA pathfinding.

All agents share one policy (parameter sharing).
Each agent's experience is collected independently and batched for training.

Key design decisions:
  - seed=None during training → random maps each episode for generalization
  - Per-agent done masking in GAE (agent frozen after reaching goal)
  - Small time penalty to encourage fast completion
  - Orthogonal weight initialization for better gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from pogema import pogema_v0, GridConfig
import os, json, time
from pathlib import Path

from pogema_compat import apply_pogema_compat_patch
from adversary import adversary_actions

apply_pogema_compat_patch()


# ─────────────────────────────────────────────────────────────────────
# 1. Policy Network
# ─────────────────────────────────────────────────────────────────────
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (standard in PPO implementations)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """CNN actor-critic for 3×W×W POGEMA observations."""

    def __init__(self, obs_channels=3, obs_size=5, n_actions=5, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * obs_size * obs_size
        self.shared = nn.Sequential(
            layer_init(nn.Linear(conv_out, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x):
        feat = self.shared(self.conv(x))
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate(self, obs, actions):
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ─────────────────────────────────────────────────────────────────────
# 2. Rollout Buffer (proper per-agent done handling)
# ─────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []  # per-agent done flag after this transition
        self.active = []  # per-agent active flag before this transition

    def add(self, obs, actions, log_probs, rewards, values, dones, active=None):
        self.obs.append(obs.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.rewards.append(rewards.copy())
        self.values.append(values.copy())
        self.dones.append(dones.copy())
        if active is None:
            active = 1.0 - dones.astype(np.float32)
        self.active.append(active.copy())

    def compute_returns(self, last_values, last_dones, gamma=0.99, lam=0.95):
        """
        Compute GAE with per-agent done masking.
        When agent is done its future value is 0.
        """
        n_steps = len(self.rewards)
        n_agents = len(self.rewards[0])

        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        last_gae = np.zeros(n_agents, dtype=np.float32)

        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - dones[t]
            if t == n_steps - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        obs = np.concatenate(self.obs, axis=0)  # (T*N, 3, H, W)
        actions = np.concatenate(self.actions, axis=0)
        log_probs = np.concatenate(self.log_probs, axis=0)
        active_mask = np.array(self.active, dtype=np.float32).reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        return obs, actions, log_probs, advantages, returns, active_mask

    def clear(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────────
# 3. PPO Trainer
# ─────────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(
        self,
        grid_config: GridConfig,
        device="cuda",
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=512,
        n_steps=256,
        time_penalty=-0.01,
        idle_penalty=-0.03,   # extra penalty for choosing idle action (0) — prevents stalling
        obs_noise_std=0.0,    # Gaussian noise added to obs during training (data augmentation)
        n_adversary=0,        # number of physical adversary chasers in the env
        adversary_strategy="astar_pursuit",
        adversary_seed=None,
    ):
        self.gc = grid_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.time_penalty = time_penalty
        self.idle_penalty = idle_penalty
        self.obs_noise_std = obs_noise_std
        self.adversary_strategy = adversary_strategy
        self.adversary_rng = np.random.default_rng(adversary_seed)

        # Social vs adversary agent counts
        self.n_social = grid_config.num_agents
        self.n_adversary = n_adversary
        self.n_total = self.n_social + self.n_adversary
        # n_agents is used for buffer dimensions — only social agents
        self.n_agents = self.n_social

        obs_size = 2 * grid_config.obs_radius + 1
        self.policy = PolicyNetwork(obs_channels=3, obs_size=obs_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self._make_env()

    def _make_env(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
        gc = GridConfig(
            num_agents=self.n_total,  # social + adversary
            size=self.gc.size,
            density=self.gc.density,
            seed=None,  # RANDOM map each reset
            max_episode_steps=self.gc.max_episode_steps,
            obs_radius=self.gc.obs_radius,
            on_target='finish',
        )
        self.env = pogema_v0(grid_config=gc)

    def set_n_adversary(self, n_adversary):
        """Change physical adversary count and recreate the environment."""
        n_adversary = int(n_adversary)
        if n_adversary < 0:
            raise ValueError("n_adversary must be non-negative")
        if n_adversary == self.n_adversary:
            return
        self.n_adversary = n_adversary
        self.n_total = self.n_social + self.n_adversary
        self._make_env()

    def set_adversary_strategy(self, strategy):
        """Set the physical adversary strategy used in future rollouts."""
        self.adversary_strategy = strategy

    def _to_tensor(self, arr):
        return torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)

    def _read_env_state(self):
        """Read obstacle map, agent positions, and goal positions from the env."""
        grid = self.env.unwrapped.grid
        obstacles = np.array(grid.get_obstacles())
        all_pos = list(grid.get_agents_xy())
        all_goals = list(grid.finishes_xy)
        return obstacles, all_pos, all_goals

    def _get_adversary_actions(self, social_done):
        """Compute greedy-pursuit actions for adversary agents."""
        if self.n_adversary == 0:
            return []
        obstacles, all_pos, all_goals = self._read_env_state()
        social_pos = all_pos[:self.n_social]
        adv_pos = all_pos[self.n_social:]
        social_goals = all_goals[:self.n_social]
        return adversary_actions(
            adv_pos, social_pos, social_goals,
            social_done.tolist(), obstacles,
            strategy=self.adversary_strategy,
            rng=self.adversary_rng,
        )

    def collect_rollout(self):
        buffer = RolloutBuffer()
        obs_list, _ = self.env.reset()
        all_obs = np.array(obs_list, dtype=np.float32)

        n_s = self.n_social
        social_done = np.zeros(n_s, dtype=bool)
        adv_done = np.zeros(self.n_adversary, dtype=bool)

        episode_rewards = np.zeros(n_s, dtype=np.float64)
        episode_success = np.zeros(n_s, dtype=bool)
        completed_episodes = []
        steps_in_episode = 0

        for step in range(self.n_steps):
            # ── social agent actions (PPO) ────────────────────────
            social_obs = all_obs[:n_s]
            obs_t = self._to_tensor(social_obs)
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(obs_t)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()
            values_np[social_done] = 0.0

            # ── adversary agent actions (greedy pursuit) ─────────
            adv_acts = self._get_adversary_actions(social_done)

            # ── step environment with all agents ─────────────────
            combined_actions = actions_np.tolist() + adv_acts
            next_obs_list, rewards, terminated, truncated, infos = self.env.step(
                combined_actions
            )

            # Only use social-agent rewards / termination for PPO
            rewards_social = np.array(rewards[:n_s], dtype=np.float32)
            term_all = np.array(terminated, dtype=bool)
            trunc_all = np.array(truncated, dtype=bool)
            term_social = term_all[:n_s]
            trunc_social = trunc_all[:n_s]

            shaped_rewards = rewards_social.copy()
            active = ~social_done
            shaped_rewards[active] += self.time_penalty
            # Penalize voluntary idling (action 0) for active agents
            if self.idle_penalty != 0:
                idle_mask = (actions_np == 0) & active
                shaped_rewards[idle_mask] += self.idle_penalty
            shaped_rewards[social_done] = 0.0

            active_before_step = (~social_done).astype(np.float32)
            done_after_step = social_done | term_social | trunc_social
            buffer.add(social_obs, actions_np, log_probs_np, shaped_rewards,
                        values_np, done_after_step.copy(), active_before_step)

            newly_terminated = term_social & ~social_done
            episode_success |= newly_terminated
            episode_rewards += rewards_social
            social_done = done_after_step

            # Track adversary termination
            for i in range(self.n_adversary):
                if term_all[n_s + i]:
                    adv_done[i] = True

            steps_in_episode += 1

            all_done = all(term_all) or all(trunc_all)
            if all_done:
                completed_episodes.append({
                    "mean_reward": float(episode_rewards.mean()),
                    "success_rate": float(episode_success.mean()),
                    "length": steps_in_episode,
                })
                obs_list, _ = self.env.reset()
                all_obs = np.array(obs_list, dtype=np.float32)
                social_done[:] = False
                adv_done[:] = False
                episode_rewards[:] = 0.0
                episode_success[:] = False
                steps_in_episode = 0
            else:
                all_obs = np.array(next_obs_list, dtype=np.float32)

        # Last values for GAE (social agents only)
        social_obs = all_obs[:n_s]
        with torch.no_grad():
            _, _, last_values = self.policy.act(self._to_tensor(social_obs))
        last_values_np = last_values.cpu().numpy()
        last_values_np[social_done] = 0.0

        flat = buffer.compute_returns(last_values_np, social_done, self.gamma, self.lam)
        buffer.clear()
        return flat, completed_episodes

    def train_step(self, obs, actions, old_log_probs, advantages, returns, active_mask):
        obs_t = self._to_tensor(obs)
        # Observation augmentation: add noise for inherent robustness
        if self.obs_noise_std > 0:
            obs_t = (obs_t + torch.randn_like(obs_t) * self.obs_noise_std).clamp(0, 1)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        old_lp_t = self._to_tensor(old_log_probs)
        adv_t = self._to_tensor(advantages)
        ret_t = self._to_tensor(returns)
        mask_t = self._to_tensor(active_mask)

        active_adv = adv_t[mask_t > 0.5]
        if len(active_adv) > 1:
            adv_t = (adv_t - active_adv.mean()) / (active_adv.std() + 1e-8)

        n_samples = len(obs)
        info = {"policy": 0, "value": 0, "entropy": 0, "n_updates": 0}

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]
                m = mask_t[idx]
                if m.sum() < 2:
                    continue

                new_lp, ent, new_val = self.policy.evaluate(obs_t[idx], actions_t[idx])
                ratio = torch.exp(new_lp - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_t[idx]
                p_loss = -(torch.min(surr1, surr2) * m).sum() / m.sum()
                v_loss = ((new_val - ret_t[idx]).pow(2) * m).sum() / m.sum()
                e_loss = -(ent * m).sum() / m.sum()

                loss = p_loss + self.value_coef * v_loss + self.entropy_coef * e_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                info["policy"] += p_loss.item()
                info["value"] += v_loss.item()
                info["entropy"] += (-e_loss.item())
                info["n_updates"] += 1

        n = max(1, info["n_updates"])
        for k in ["policy", "value", "entropy"]:
            info[k] /= n
        return info

    def train(self, total_timesteps=1_000_000, log_interval=10, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
        steps_per_iter = self.n_steps * self.n_agents
        n_iters = max(1, total_timesteps // steps_per_iter)
        total_steps = 0
        history = []
        all_episodes = []

        print(f"PPO Training: {n_iters} iters, {steps_per_iter} steps/iter, "
              f"{self.n_social} social + {self.n_adversary} adversary agents, "
              f"device={self.device}")
        print(f"  Map: {self.gc.size}x{self.gc.size}, density={self.gc.density}, "
              f"max_steps={self.gc.max_episode_steps}")
        print("-" * 70)

        best_success = -1.0
        start = time.time()

        for it in range(1, n_iters + 1):
            (obs, act, lp, adv, ret, mask), episodes = self.collect_rollout()
            loss_info = self.train_step(obs, act, lp, adv, ret, mask)
            all_episodes.extend(episodes)
            total_steps += steps_per_iter

            recent = all_episodes[-20:] if len(all_episodes) >= 20 else all_episodes
            if recent:
                mr = np.mean([e["mean_reward"] for e in recent])
                ms = np.mean([e["success_rate"] for e in recent])
                ml = np.mean([e["length"] for e in recent])
            else:
                mr = ms = ml = float("nan")

            history.append({
                "iteration": it, "total_steps": total_steps,
                "mean_reward": float(mr), "success_rate": float(ms),
                "mean_length": float(ml),
                "policy_loss": loss_info["policy"],
                "value_loss": loss_info["value"],
                "entropy": loss_info["entropy"],
                "episodes_completed": len(all_episodes),
                "n_adversary": self.n_adversary,
                "adversary_strategy": self.adversary_strategy,
            })

            if it % log_interval == 0 or it == 1:
                fps = total_steps / (time.time() - start)
                print(
                    f"Iter {it:4d} | Steps {total_steps:>9d} | "
                    f"Rew {mr:+.3f} | Succ {ms:.1%} | Len {ml:5.0f} | "
                    f"PL {loss_info['policy']:+.4f} VL {loss_info['value']:.4f} "
                    f"Ent {loss_info['entropy']:.3f} | "
                    f"Eps {len(all_episodes):>4d} | FPS {fps:.0f}"
                )

            if recent and ms > best_success:
                best_success = ms
                torch.save(self.policy.state_dict(),
                           os.path.join(save_dir, "best_policy.pt"))

        torch.save(self.policy.state_dict(),
                   os.path.join(save_dir, "final_policy.pt"))
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print("-" * 70)
        print(f"Done. Best success: {best_success:.1%}, "
              f"episodes: {len(all_episodes)}")
        self.env.close()
        return history


# ─────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────────────────────────────
def evaluate_policy(
    policy, grid_config, n_episodes=100, device="cuda",
    deterministic=False, attack_fn=None, attack_kwargs=None,
    n_adversary=0,
    adversary_strategy="astar_pursuit",
    adversary_seed=12345,
):
    """
    Evaluate policy on diverse random maps.

    When n_adversary > 0, extra adversary agents are added to the
    environment and controlled by the greedy-pursuit heuristic.
    Only social agents' success/makespan are reported.

    Returns dict with success_rate, makespan, per-episode data.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    policy.eval()
    if attack_kwargs is None:
        attack_kwargs = {}

    n_social = grid_config.num_agents
    n_total = n_social + n_adversary

    results = []
    for ep in range(n_episodes):
        gc = GridConfig(
            num_agents=n_total,
            size=grid_config.size,
            density=grid_config.density,
            seed=ep + 1000,
            max_episode_steps=grid_config.max_episode_steps,
            obs_radius=grid_config.obs_radius,
            on_target='finish',
        )
        env = pogema_v0(grid_config=gc)
        obs_list, _ = env.reset()
        all_obs = np.array(obs_list, dtype=np.float32)

        grid = env.unwrapped.grid
        obstacles = np.array(grid.get_obstacles())
        social_goals = list(grid.finishes_xy)[:n_social]

        arrival = np.full(n_social, -1, dtype=np.int32)
        social_done = [False] * n_social
        step = 0
        adv_rng = np.random.default_rng(adversary_seed + ep)

        while True:
            # Social agent actions (policy)
            social_obs = all_obs[:n_social]
            obs_t = torch.from_numpy(social_obs).float().to(device)
            if attack_fn is not None:
                obs_t = attack_fn(obs_t, policy, **attack_kwargs)
            with torch.no_grad():
                s_actions, _, _ = policy.act(obs_t, deterministic=deterministic)
            s_actions_list = s_actions.cpu().numpy().tolist()

            # Adversary actions (greedy pursuit)
            if n_adversary > 0:
                all_pos = list(grid.get_agents_xy())
                adv_acts = adversary_actions(
                    all_pos[n_social:], all_pos[:n_social],
                    social_goals, social_done, obstacles,
                    strategy=adversary_strategy,
                    rng=adv_rng,
                )
            else:
                adv_acts = []

            combined = s_actions_list + adv_acts
            next_obs, rewards, terminated, truncated, _ = env.step(combined)
            term_all = np.array(terminated, dtype=bool)
            trunc_all = np.array(truncated, dtype=bool)
            step += 1

            # Track social agent arrivals
            for i in range(n_social):
                if term_all[i] and arrival[i] < 0:
                    arrival[i] = step
                    social_done[i] = True

            if all(term_all) or all(trunc_all):
                break
            all_obs = np.array(next_obs, dtype=np.float32)

        succeeded = (arrival > 0).sum()
        sr = succeeded / n_social
        if succeeded > 0:
            makespan = int(arrival[arrival > 0].max())
        else:
            makespan = step

        results.append({
            "episode": ep, "success_rate": float(sr),
            "makespan": makespan, "agents_succeeded": int(succeeded),
            "total_steps": step,
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
        "n_adversary": n_adversary,
        "adversary_strategy": adversary_strategy,
        "per_episode": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--obs-radius", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="models/baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gc = GridConfig(
        num_agents=args.num_agents, size=args.map_size,
        density=args.density, seed=args.seed,
        max_episode_steps=args.max_steps, obs_radius=args.obs_radius,
    )

    trainer = PPOTrainer(gc, lr=args.lr, n_steps=args.n_steps,
                         batch_size=args.batch_size)
    trainer.train(total_timesteps=args.total_timesteps, save_dir=args.save_dir)

    print("\nEvaluating...")
    policy = PolicyNetwork(obs_size=2*args.obs_radius+1)
    mp = os.path.join(args.save_dir, "best_policy.pt")
    if not os.path.exists(mp):
        mp = os.path.join(args.save_dir, "final_policy.pt")
    policy.load_state_dict(torch.load(mp, weights_only=True))
    res = evaluate_policy(policy, gc, n_episodes=args.eval_episodes)
    print(f"  Success: {res['mean_success_rate']:.1%} ± {res['std_success_rate']:.1%}")
    print(f"  Makespan: {res['mean_makespan']:.1f} ± {res['std_makespan']:.1f}")

    with open(os.path.join(args.save_dir, "eval_results.json"), "w") as f:
        json.dump(res, f, indent=2)
