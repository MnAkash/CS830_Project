#!/usr/bin/env python3
"""
MAPF Adversarial Visualization — physical adversary agents vs cooperative social agents.

Visual language
───────────────
• **Social agents** (coloured circles with ID) cooperate to reach their goals.
• **Adversary agents** (red triangles marked 'A') **chase** social agents
  using greedy pursuit — they are actual agents on the grid that pursue the
  nearest social agent to physically block its path.
• **FGSM observation attacks** (invisible) perturb each social agent's sensor
  reading to confuse the policy — FAST policies are trained to resist this.
• Coloured **diamonds** mark each social agent's goal.
• Fading **trajectory trails** show the path every agent has walked.
  – Social trails: solid coloured lines.
  – Adversary trails: dashed red lines.
• Green halo + ✓ = social agent reached its goal.

Generated GIFs
──────────────
  1. baseline_clean.gif          — Baseline policy, no adversary
  2. baseline_attacked.gif       — Baseline policy  + adversary chasers + FGSM obs noise
  3. fast_clean.gif              — FAST policy, no adversary
  4. fast_attacked.gif           — FAST policy      + adversary chasers + FGSM obs noise
  5. comparison_attacked.gif     — side‑by‑side  (baseline vs FAST)  under attack
"""

import sys, os, copy, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import imageio
from pogema import pogema_v0, GridConfig

from ppo_mapf import PolicyNetwork
from attacks import fgsm_attack
from adversary import bfs_next_step, adversary_actions, ACTION_DELTAS

# ─── colours ──────────────────────────────────────────────────────
SOCIAL_COLORS = [
    "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#e6194B",
]
ADV_COLOR       = "#d62728"   # vivid red
ADV_TRAIL_COLOR = "#8b0000"   # dark red


def _sc(i):
    """Social‑agent colour."""
    return SOCIAL_COLORS[i % len(SOCIAL_COLORS)]


# ─── single‑frame renderer ────────────────────────────────────────
def render_frame(snap, title, cell_px=48):
    """Draw one animation frame from a snapshot dict."""
    obstacles      = snap["obstacles"]
    social_pos     = snap["social_positions"]
    adv_pos        = snap["adv_positions"]
    social_goals   = snap["social_goals"]
    social_done    = snap["social_done"]
    social_traj    = snap["social_trajectories"]
    adv_traj       = snap["adv_trajectories"]
    step           = snap["step"]
    n_social       = snap["n_social"]
    n_adv          = snap["n_adversary"]
    has_adv        = snap["has_adversary"]

    H, W = obstacles.shape
    dpi = 100
    fw = W * cell_px / dpi + 2.4      # extra for legend
    fh = H * cell_px / dpi + 1.2
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)

    # ── grid background ──────────────────────────────────────────
    for r in range(H):
        for c in range(W):
            fc = "#505050" if obstacles[r, c] > 0.5 else "#f5f5f5"
            ec = "#404040" if obstacles[r, c] > 0.5 else "#e0e0e0"
            ax.add_patch(plt.Rectangle(
                (c, H - 1 - r), 1, 1, facecolor=fc,
                edgecolor=ec, linewidth=0.4, zorder=0))

    # ── helper: draw trajectory trail ────────────────────────────
    def _draw_trail(traj, color, ls='-', lw=2.0):
        n_pts = len(traj)
        if n_pts < 2:
            return
        for j in range(1, n_pts):
            r0, c0 = traj[j - 1]
            r1, c1 = traj[j]
            alpha = 0.10 + 0.60 * (j / n_pts)
            ax.plot([c0 + 0.5, c1 + 0.5],
                    [H - 1 - r0 + 0.5, H - 1 - r1 + 0.5],
                    color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls, solid_capstyle='round', zorder=1)

    # ── social agent trajectories (solid) ────────────────────────
    for i, traj in enumerate(social_traj):
        _draw_trail(traj, _sc(i), ls='-', lw=2.0)

    # ── adversary trajectories (dashed red) ──────────────────────
    for traj in adv_traj:
        _draw_trail(traj, ADV_TRAIL_COLOR, ls='--', lw=1.8)

    # ── social goals (diamonds) ──────────────────────────────────
    for i, (gr, gc_) in enumerate(social_goals):
        color = _sc(i)
        x, y = gc_ + 0.5, H - 1 - gr + 0.5
        ax.plot(x, y, marker='D', markersize=10, color=color,
                alpha=0.30, markeredgecolor=color,
                markeredgewidth=1.5, zorder=2)
        ax.text(x, y, "G", ha='center', va='center',
                fontsize=6, color=color, alpha=0.55,
                fontweight='bold', zorder=2)

    # ── social agents (circles) ──────────────────────────────────
    for i, (ar, ac) in enumerate(social_pos):
        color = _sc(i)
        x, y = ac + 0.5, H - 1 - ar + 0.5
        if social_done[i]:
            # reached goal — green glow
            ax.add_patch(plt.Circle((x, y), 0.42,
                         fc='#00ff00', alpha=0.22, ec='none', zorder=3))
            ax.add_patch(plt.Circle((x, y), 0.34,
                         fc=color, ec='#00aa00', linewidth=2.2,
                         alpha=0.92, zorder=4))
            ax.text(x, y, "✓", ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=5)
        else:
            ax.add_patch(plt.Circle((x, y), 0.34,
                         fc=color, ec='#333333', linewidth=1.0,
                         alpha=0.92, zorder=4))
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=5)

    # ── adversary agents (red triangles) ─────────────────────────
    for j, (ar, ac) in enumerate(adv_pos):
        x, y = ac + 0.5, H - 1 - ar + 0.5
        # red glow
        ax.add_patch(plt.Circle((x, y), 0.44,
                     fc=ADV_COLOR, alpha=0.15, ec='none', zorder=3))
        ax.plot(x, y, marker='^', markersize=14,
                color=ADV_COLOR, markeredgecolor='#400000',
                markeredgewidth=1.5, zorder=4)
        ax.text(x, y - 0.02, "A", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=5)

    # ── axes ─────────────────────────────────────────────────────
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── title ────────────────────────────────────────────────────
    reached = sum(social_done)
    status = f"Step {step}  |  {reached}/{n_social} social agents reached goal"
    if has_adv:
        status += f"  |  {n_adv} adversary chaser(s)"
    title_color = '#cc0000' if has_adv else '#000000'
    ax.set_title(f"{title}\n{status}", fontsize=10, fontweight='bold',
                 pad=6, color=title_color)

    # ── legend ───────────────────────────────────────────────────
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=8, label='Social Agent (cooperative)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#888888',
               markersize=8, alpha=0.4, label='Goal'),
        Line2D([0], [0], color='#888888', linewidth=2,
               alpha=0.5, label='Social Trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00cc00',
               markersize=8, label='Reached Goal ✓'),
    ]
    if has_adv:
        legend_elems += [
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=ADV_COLOR, markersize=10,
                   markeredgecolor='#400000', markeredgewidth=1,
                   label='Adversary Agent (chaser)'),
            Line2D([0], [0], color=ADV_TRAIL_COLOR, linewidth=1.8,
                   linestyle='--', alpha=0.6, label='Adversary Trajectory'),
        ]
    ax.legend(handles=legend_elems, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=7,
              framealpha=0.85, edgecolor='#cccccc')
    fig.subplots_adjust(right=0.75)
    return fig


def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    return buf


# ─── episode runner (mixed social + adversary agents) ─────────────
def run_episode(social_policy, n_social, n_adversary,
                size, density, seed, max_steps, obs_radius,
                device="cuda", attack_fn=None, attack_kwargs=None):
    """
    Run one POGEMA episode.

    Agents 0 … n_social−1      → controlled by *social_policy* (PPO).
    Agents n_social … total−1   → controlled by greedy pursuit heuristic.

    When n_adversary == 0 (clean run) only social agents are present.
    attack_fn:     optional observation perturbation (e.g. fgsm_attack).
    attack_kwargs: keyword arguments forwarded to attack_fn.
    """
    if attack_kwargs is None:
        attack_kwargs = {}
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    social_policy = social_policy.to(device).eval()

    total = n_social + n_adversary
    gc = GridConfig(
        num_agents=total, size=size, density=density, seed=seed,
        max_episode_steps=max_steps, obs_radius=obs_radius,
        on_target='finish',
    )
    env = pogema_v0(grid_config=gc)
    obs_list, _ = env.reset()

    grid = env.unwrapped.grid
    obstacles = np.array(grid.get_obstacles())
    all_goals = list(grid.finishes_xy)
    social_goals = all_goals[:n_social]

    social_done = [False] * n_social
    adv_done    = [False] * n_adversary

    social_traj = [[] for _ in range(n_social)]
    adv_traj    = [[] for _ in range(n_adversary)]

    def _snap(step):
        all_pos = list(grid.get_agents_xy())
        s_pos = all_pos[:n_social]
        a_pos = all_pos[n_social:]
        for i in range(n_social):
            social_traj[i].append(s_pos[i])
        for i in range(n_adversary):
            adv_traj[i].append(a_pos[i])
        return {
            "obstacles": obstacles,
            "social_positions":  list(s_pos),
            "adv_positions":     list(a_pos),
            "social_goals":      social_goals,
            "social_done":       list(social_done),
            "social_trajectories": [list(t) for t in social_traj],
            "adv_trajectories":    [list(t) for t in adv_traj],
            "step": step,
            "has_adversary": n_adversary > 0,
            "n_social": n_social,
            "n_adversary": n_adversary,
        }

    snaps = [_snap(0)]
    obs = np.array(obs_list, dtype=np.float32)

    for step in range(1, max_steps + 1):
        # ── social actions (PPO) ─────────────────────────────────
        social_obs_t = torch.from_numpy(obs[:n_social]).float().to(device)
        if attack_fn is not None:
            social_obs_t = attack_fn(social_obs_t, social_policy,
                                     **attack_kwargs)
        with torch.no_grad():
            s_actions, _, _ = social_policy.act(social_obs_t, deterministic=True)
        s_actions = s_actions.cpu().numpy().tolist()

        # ── adversary actions (smart intercept + block) ──────────
        if n_adversary > 0:
            all_pos = list(grid.get_agents_xy())
            a_actions = adversary_actions(
                all_pos[n_social:], all_pos[:n_social],
                social_goals, social_done, obstacles)
        else:
            a_actions = []

        all_actions = s_actions + a_actions
        nxt, _rew, term, trunc, _ = env.step(all_actions)
        term  = np.array(term,  dtype=bool)
        trunc = np.array(trunc, dtype=bool)

        for i in range(n_social):
            if term[i] and not social_done[i]:
                social_done[i] = True
        for i in range(n_adversary):
            if term[n_social + i] and not adv_done[i]:
                adv_done[i] = True

        snaps.append(_snap(step))

        if all(term) or all(trunc):
            break
        obs = np.array(nxt, dtype=np.float32)

    env.close()
    return snaps


# ─── GIF writers ──────────────────────────────────────────────────
def snaps_to_gif(snaps, path, title="", fps=3, cell_px=48):
    imgs = []
    for s in snaps:
        fig = render_frame(s, title, cell_px=cell_px)
        imgs.append(fig_to_array(fig))
    for _ in range(fps * 2):        # hold last frame
        imgs.append(imgs[-1])
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    print(f"    saved {path}  ({len(snaps)} frames)")


def side_by_side_gif(snaps_L, snaps_R, title_L, title_R,
                     path, fps=3, cell_px=44):
    """Render two episodes side‑by‑side into one GIF."""
    max_len = max(len(snaps_L), len(snaps_R))
    while len(snaps_L) < max_len:
        snaps_L.append(snaps_L[-1])
    while len(snaps_R) < max_len:
        snaps_R.append(snaps_R[-1])

    imgs = []
    for i in range(max_len):
        aL = fig_to_array(render_frame(snaps_L[i], title_L, cell_px=cell_px))
        aR = fig_to_array(render_frame(snaps_R[i], title_R, cell_px=cell_px))
        h = max(aL.shape[0], aR.shape[0])
        if aL.shape[0] < h:
            aL = np.vstack([aL, np.full((h - aL.shape[0], aL.shape[1], 3),
                                        255, dtype=np.uint8)])
        if aR.shape[0] < h:
            aR = np.vstack([aR, np.full((h - aR.shape[0], aR.shape[1], 3),
                                        255, dtype=np.uint8)])
        sep = np.full((h, 6, 3), 30, dtype=np.uint8)
        imgs.append(np.hstack([aL, sep, aR]))

    for _ in range(fps * 2):
        imgs.append(imgs[-1])
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    print(f"    saved {path}  ({max_len} frames, side‑by‑side)")


# ─── main ─────────────────────────────────────────────────────────
def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--baseline-model", default="models/quick_v5_baseline/best_policy.pt")
    pa.add_argument("--robust-model",   default="models/quick_v5_robust/final_policy.pt")
    pa.add_argument("--output-dir",     default="results/animations")
    pa.add_argument("--num-social",  type=int, default=8)
    pa.add_argument("--num-adversary", type=int, default=4)
    pa.add_argument("--map-size",    type=int, default=16)
    pa.add_argument("--seed",        type=int, default=1025)
    pa.add_argument("--max-steps",   type=int, default=80)
    pa.add_argument("--fps",         type=int, default=3)
    pa.add_argument("--fgsm-epsilon", type=float, default=0.15,
                    help="FGSM observation perturbation budget for attacked scenarios")
    pa.add_argument("--device",      default="cuda")
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    def load(path):
        net = PolicyNetwork(obs_size=5)
        net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        return net.to(dev).eval()

    print("Loading policies …")
    base = load(args.baseline_model)
    fast = load(args.robust_model)

    common = dict(size=args.map_size, density=0.3, seed=args.seed,
                  max_steps=args.max_steps, obs_radius=2, device=dev)

    print(f"\nRunning episodes  ({args.num_social} social + "
          f"{args.num_adversary} adversary agents on {args.map_size}×{args.map_size} grid)…")

    atk_kw = {"epsilon": args.fgsm_epsilon}

    print("  1/4  Baseline — no adversary")
    ep_bc = run_episode(base, args.num_social, 0, **common)
    print(f"  2/4  Baseline — chasers + FGSM (ε={args.fgsm_epsilon})")
    ep_ba = run_episode(base, args.num_social, args.num_adversary,
                        attack_fn=fgsm_attack, attack_kwargs=atk_kw, **common)
    print("  3/4  FAST     — no adversary")
    ep_fc = run_episode(fast, args.num_social, 0, **common)
    print(f"  4/4  FAST     — chasers + FGSM (ε={args.fgsm_epsilon})")
    ep_fa = run_episode(fast, args.num_social, args.num_adversary,
                        attack_fn=fgsm_attack, attack_kwargs=atk_kw, **common)

    for tag, ep in [("Baseline clean", ep_bc), ("Baseline attacked", ep_ba),
                    ("FAST clean",     ep_fc), ("FAST attacked",     ep_fa)]:
        last = ep[-1]
        sr = sum(last["social_done"])
        print(f"    {tag:20s}  {sr}/{args.num_social} social reached goal  "
              f"({sr/args.num_social:.0%})  in {last['step']} steps")

    print("\nGenerating individual GIFs …")
    snaps_to_gif(ep_bc, f"{args.output_dir}/baseline_clean.gif",
                 "Baseline Policy — No Adversary", args.fps)
    snaps_to_gif(ep_ba, f"{args.output_dir}/baseline_attacked.gif",
                 f"Baseline — {args.num_adversary} Chasers + FGSM ε={args.fgsm_epsilon}", args.fps)
    snaps_to_gif(ep_fc, f"{args.output_dir}/fast_clean.gif",
                 "FAST Policy — No Adversary", args.fps)
    snaps_to_gif(ep_fa, f"{args.output_dir}/fast_attacked.gif",
                 f"FAST — {args.num_adversary} Chasers + FGSM ε={args.fgsm_epsilon}", args.fps)

    print("\nGenerating side‑by‑side comparisons …")
    side_by_side_gif(
        copy.deepcopy(ep_ba), copy.deepcopy(ep_fa),
        f"Baseline + {args.num_adversary} Chasers + FGSM",
        f"FAST + {args.num_adversary} Chasers + FGSM",
        f"{args.output_dir}/comparison_attacked.gif", args.fps)

    print(f"\n✓ All animations saved to {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.gif'):
            kb = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
            print(f"    {f:42s} {kb:6.0f} KB")


if __name__ == "__main__":
    main()
