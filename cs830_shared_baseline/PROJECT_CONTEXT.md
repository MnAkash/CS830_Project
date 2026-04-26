# Project Context

This file tracks all project-specific details: what the project is, who
does what, what has been built, and what state things are in. Update this
file whenever the project state changes.

For writing tone and guidelines, see `CONTEXT.md`.

---

## Course

CS 730/830 — Introduction to Artificial Intelligence, UNH.

Relevant course schedule:
- Weeks 1–3: Search (A*, BFS, heuristics)
- Week 4: Game playing (minimax, adversarial)
- Week 6: Uncertainty and probabilistic reasoning
- Weeks 7–8: Planning
- Weeks 9–10: Markov Decision Processes (MDPs, Bellman equation, value iteration)
- Weeks 11–14: Reinforcement learning (policy gradient, PPO)

---

## Team

| Person | Role |
|--------|------|
| Riad Ahmed | Adversarial training defence (modifies training loss) |
| Mohammad Moniruzzaman Akash | Curriculum training defence (modifies training environment) |
| Sujosh Nag | Inference-time smoothing defence (modifies how the policy is used at test time) |

---

## Problem

Multi-agent pathfinding (MAPF): N agents on a grid, each with a start
position and a goal position. They must all reach their goals without
colliding with each other or with obstacles.

We use a decentralized approach — each agent sees only a small local window
(5×5 grid) around itself and picks an action using a shared neural network
policy trained with PPO. No central coordinator.

The question: these learned policies break badly when observations are
corrupted (sensor noise, communication errors, or deliberate adversarial
attacks). We study this fragility and compare three simple ways to fix it.

---

## Environment

**POGEMA benchmark** — a grid-world MAPF simulator.

- Grid with random obstacles (density 0.3)
- Each agent observes a 5×5 local window with 3 channels:
  - Channel 0: obstacles
  - Channel 1: other agents
  - Channel 2: goal direction
- 5 discrete actions: stay, up, down, left, right
- Episode ends when all agents reach goals or time limit is hit
- Reward: +1 for reaching goal, small per-step penalty

Configurations:

| Name | Agents | Grid | Max Steps | Purpose |
|------|--------|------|-----------|---------|
| Quick | 8 | 16×16 | 128 | Fast debugging |
| Main | 16 | 20×20 | 192 | Primary results |
| Scale | 32 | 32×32 | 256 | Scalability test |

---

## Baseline Policy

- Architecture: CNN with 3 conv layers (32→64→64 channels), 2 fully connected layers (256 units), then two heads:
  - Actor head: 5 logits → action probabilities
  - Critic head: 1 scalar → estimated value
- Training: PPO, 2M timesteps, lr=3e-4, 4 epochs per update, clip ratio 0.2
- Parameter sharing: one network for all agents
- Target clean performance: ≥95% on the Phase 2 easy sanity environment before
  the defence tracks begin; larger Quick/Main experiments can use the proposal
  target of ~70%+ on harder settings.

---

## Attacks

| Attack | How it works |
|--------|-------------|
| Random noise | Adds uniform random values to the observation tensor. Baseline for measuring sensitivity. |
| FGSM (Fast Gradient Sign Method) | Computes the gradient of the policy loss with respect to the observation, then steps in the worst direction by ε. One step, one perturbation. The simplest gradient-based attack. |
| Physical A* chasers | Extra agents placed in the environment that chase cooperative agents using A* search with Manhattan distance heuristic. These are not perturbations — they are actual agents that block and collide. |

Attack strength sweep: ε ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}

---

## Three Defences

### Riad — Adversarial training with frozen-model FGSM

Modifies the training loss. During PPO training, generate adversarial
observations using FGSM applied to a frozen copy of the old baseline policy
(not the policy currently being trained). Train on a mix: 70% clean
observations, 30% adversarial observations. Also add a KL-divergence
smoothness penalty between the policy's output on clean vs. slightly noisy
observations.

The frozen model is the key idea — without it, the policy and the attack
co-evolve and the policy collapses to always outputting the same action
(entropy collapse).

Implementation owner: Riad. This is not part of the baseline/wrapper handoff.
Riad should build this on top of `src/ppo_mapf.py`, `src/attacks.py`, and the
shared baseline checkpoint.

### Akash — Curriculum training with A* adversary agents

Modifies the training environment. During PPO training, gradually introduce
physical adversary agents that chase cooperative agents using A* search.
Start with zero adversaries and ramp up to the maximum over the first 40%
of training (curriculum schedule). Also add small Gaussian noise to
observations during training.

The curriculum matters — introducing all adversaries from the start
overwhelms the policy before it has learned basic navigation.

Implementation owner: Akash. This is not part of the baseline/wrapper handoff.
Akash should build this on top of `src/pogema_wrapper.py`, `src/adversary.py`,
and the shared baseline checkpoint.

### Sujosh — Inference-time noise averaging

Modifies how the policy is used at test time. Instead of feeding the raw
observation into the policy once, add random Gaussian noise N times, get N
action probability distributions, average them, and pick the action with the
highest average probability. This dilutes adversarial perturbations because
they are crafted for one specific direction — random noise on top washes
them out on average.

Requires zero additional training. Can be applied on top of any trained
policy (baseline, Riad's, or Akash's).

Implementation owner: Sujosh. This is not part of the baseline/wrapper
handoff. Sujosh should build this as an inference wrapper around the shared
baseline policy.

---

## Codebase

| File | What it does |
|------|-------------|
| `src/ppo_mapf.py` | Baseline PPO implementation: PolicyNetwork (CNN), RolloutBuffer, PPOTrainer, evaluate_policy() |
| `src/phase2_train_baseline.py` | Phase 2 baseline PPO smoke trainer with architecture plot, training curves, rollout GIF, and HTML report |
| `src/shared_readiness.py` | Shared pre-handoff validation: baseline checkpoint, attack bounds, fragility sweeps, physical A* checks, HTML report, and team handoff |
| `src/adv_train.py` | Team-owned area for Riad; not part of the baseline/wrapper handoff |
| `src/curriculum_train.py` | Team-owned area for Akash; not part of the baseline/wrapper handoff |
| `src/smoothing_defense.py` | Team-owned area for Sujosh; not part of the baseline/wrapper handoff |
| `src/attacks.py` | FGSM, PGD, random noise, partial attack implementations |
| `src/adversary.py` | A* and BFS adversary agent logic |
| `src/pogema_wrapper.py` | POGEMA environment wrapper for multi-agent RL |
| `src/phase1_visualize.py` | Phase 1 environment visualization: initial PNG, final PNG, random-rollout GIF, HTML preview |
| `src/utils.py` | Shared project paths, seeding, device selection, JSON helpers |
| `src/evaluate_fragility.py` | Evaluation: runs attack sweeps, computes success rate and makespan |
| `src/run_experiments.py` | Full 12-step experiment pipeline |
| `src/visualize.py` | GIF animation generation |
| `src/plot_training.py` | Training curve comparison plots |
| `src/scalability.py` | Scalability analysis across agent counts |
| `./` | Self-contained baseline + wrapper handoff package for all three team members |

---

## Model Checkpoints

Located in `models/`. Key ones:

| Directory | What it is |
|-----------|-----------|
| `full_baseline/` | Full-run baseline policy |
| `full_robust/` | Full-run adversarially trained policy (Riad's) |
| `quick_v6_baseline/` | Latest quick-run baseline |
| `quick_v6_robust/` | Latest quick-run robust |
| `phase2_smoke_baseline/` | Verified easy-environment baseline for shared handoff; use `best_policy.pt` |

Each contains: `best_policy.pt`, `final_policy.pt`, `training_history.json`

---

## Results

Located in `results/`. Key outputs:

| Directory | Contents |
|-----------|---------|
| `full/` | Full-run fragility results (baseline_fragility.json, config.json) |
| `quick_v6/` | Latest quick-run results |
| `animations_best/` | Best GIF animations |
| `phase1_environment/` | Phase 1 wrapper preview: HTML report, PNGs, GIF, and JSON summary |
| `phase2_baseline/` | Phase 2 baseline PPO preview: HTML report, training curves, rollout GIF, architecture plot, action histogram |
| `shared_readiness/` | Shared readiness report: attack panel, fragility plots, physical A* GIF, JSON summary |

Phase 1 visual feedback report:
- `results/phase1_environment/phase1_environment_report.html`
- `results/phase1_environment/phase1_random_rollout.gif`
- `results/phase1_environment/phase1_initial_state.png`
- `results/phase1_environment/phase1_final_state.png`

Phase 2 baseline feedback report:
- `results/phase2_baseline/phase2_baseline_report.html`
- `results/phase2_baseline/phase2_training_curves.png`
- `results/phase2_baseline/phase2_policy_rollout.gif`
- `results/phase2_baseline/phase2_baseline_summary.json`

Latest Phase 2 smoke result: 96.5% sampled-policy success over 50 held-out
episodes on the easy sanity environment (4 agents, 8×8 grid, density 0.1).
This meets the near-100% smoke target before moving to longer Quick/Main runs.

Shared readiness handoff:
- `README.md`
- `README.md`
- this copied package with code, wrappers, checkpoint, reports, and docs
- `BASELINE_WRAPPER_HANDOFF.md`
- `SHARED_TEAM_HANDOFF.md`
- `results/shared_readiness/shared_readiness_report.html`
- `results/shared_readiness/shared_readiness_summary.json`
- `results/shared_readiness/shared_attack_observation_panel.png`
- `results/shared_readiness/shared_baseline_fragility.png`
- `results/shared_readiness/shared_partial_attack.png`
- `results/shared_readiness/shared_physical_adversary.png`
- `results/shared_readiness/shared_physical_chaser_preview.gif`

Latest shared readiness result: ready = true. The shared handoff uses
`models/phase2_smoke_baseline/best_policy.pt`, which reached 96.5% sampled
policy success over 50 held-out episodes and passed attack-bound and A* chaser
checks.

---

## Running the Pipeline

```bash
conda activate grasp_splats
cd /home/carl_ma/Riad/MS/CS830/project
python src/run_experiments.py --quick    # smoke test (~15 min)
python src/run_experiments.py            # full run
python src/shared_readiness.py --device cpu --episodes 8
```

---

## Files to Keep in Sync

When something changes, update these in order:

1. `PROJECT_CONTEXT.md` (this file) — update project state
2. `PROJECT_PLAN.md` — update plan/timeline if needed
3. `CONTEXT.md` — only if the writing rules need adjustment
4. Report/paper files — propagate any changed details

---

## Current Status

- [x] Phase 1 POGEMA wrapper rebuilt with named Quick/Main/Scale configs
- [x] Phase 1 visual feedback report generated in `results/phase1_environment/`
- [x] Phase 2 baseline PPO smoke trainer and visual report generated
- [x] Phase 2 easy baseline reached near-100% success target (96.5% over 50 held-out episodes)
- [x] Shared readiness report generated before Riad/Akash/Sujosh start defence work
- [x] Shared handoff file generated: `SHARED_TEAM_HANDOFF.md`
- [x] Baseline/wrapper handoff guide generated: `BASELINE_WRAPPER_HANDOFF.md`
- [x] Self-contained shared package created in `shared_baseline/`
- [x] Shared package README created in `shared_baseline/README.md`
- [x] Root README created with quick handoff instructions
- [x] Shared best baseline checkpoint selected: `models/phase2_smoke_baseline/best_policy.pt`
- [x] Attack-bound checks passed for random, FGSM, PGD, and partial FGSM
- [x] A* physical adversary checks and visual preview generated
- [x] Baseline PPO implemented and trained
- [x] Attack implementations (FGSM, random noise, physical A*)
- [ ] Riad's adversarial training defence — to be implemented by Riad on top of the baseline
- [ ] Akash's curriculum/physical-adversary defence — to be implemented by Akash on top of the baseline
- [ ] Sujosh's smoothing defence — to be implemented by Sujosh on top of the baseline
- [ ] Full experiment pipeline after all three team defences are added
- [x] PROJECT_PLAN.md with PPO justification
- [ ] Full experiment run on Main config (16 agents, 20×20)
- [ ] Cross-evaluation of all three defences
- [ ] Individual papers
- [ ] Presentation
