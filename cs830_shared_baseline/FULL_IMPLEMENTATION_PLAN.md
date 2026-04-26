# Full Implementation Plan: Adversarial Robustness in Decentralized MAPF

**Project:** Adversarial Robustness in Decentralized Multi-Agent Pathfinding  
**Course:** CS 730/830 — Introduction to Artificial Intelligence  
**Team:** Riad Ahmed, Mohammad Moniruzzaman Akash, Sujosh Nag  
**Plan date:** April 26, 2026  
**Source:** proposal_final.tex

---

## Goal

Build the full project from the beginning: train a decentralized multi-agent pathfinding policy, attack it, measure how it fails, implement three different defences, and compare them under the same evaluation protocol.

The project studies Multi-Agent Pathfinding (MAPF), where multiple agents move on a grid from start cells to goal cells without collisions. The policy is decentralized: each agent only sees a local observation window and chooses one of five actions. All agents share one neural network policy trained with PPO (Proximal Policy Optimization), a reinforcement learning method that updates the policy in small controlled steps.

The main question is simple: when observations are corrupted by noise, gradient-based attacks, or physical adversaries, how badly does the learned policy fail, and which defence helps most?

---

## Final Deliverables

By the end, the project should have:

1. A working POGEMA-based MAPF environment wrapper.
2. A shared PPO baseline policy trained on the main configuration.
3. Attack implementations:
   - random observation noise,
   - FGSM observation attack,
   - PGD observation attack,
   - physical A* chaser agents.
4. A baseline fragility profile showing success rate vs. attack strength.
5. Three defence implementations:
   - Riad: adversarial training,
   - Akash: physical adversary curriculum training,
   - Sujosh: observation denoising before policy inference.
6. Cross-evaluation of all defences under the same attacks and metrics.
7. Plots, tables, and GIF animations for the report and presentation.
8. Individual writeups and one combined presentation.

---

## Shared Experimental Setup

All phases use the same environment, model architecture, attacks, seeds, and metrics. This keeps the comparison fair.

### Environment Configurations

| Name | Agents | Grid | Max steps | Purpose |
|---|---:|---:|---:|---|
| Quick | 8 | 16×16 | 128 | Fast debugging and smoke tests |
| Main | 16 | 20×20 | 192 | Primary experiments |
| Scale | 32 | 32×32 | 256 | Scalability check |

### Observation and Action Space

- Observation: local 5×5 window with 3 channels.
  - Channel 0: obstacles.
  - Channel 1: other agents.
  - Channel 2: goal direction.
- Actions: stay, up, down, left, right.
- Reward: positive reward for reaching the goal and a small per-step penalty.

### Metrics

| Metric | Meaning |
|---|---|
| Success rate | Percentage of agents that reach their goals. |
| Makespan | Number of steps until all successful agents finish, or timeout. |
| Collision/blocking count | Optional diagnostic for physical adversary runs. |
| Policy entropy | Measures how spread out action probabilities are. Low entropy can indicate collapse. |
| Runtime | Used for denoising and smoothing-style inference overhead. |

### Attack Strengths

Use the same epsilon sweep for observation attacks:

`ε = {0.00, 0.05, 0.10, 0.15, 0.20, 0.30}`

Use the same physical adversary sweep:

`chasers = {0, 1, 2, 4}`

---

## Phase Overview

| Phase | Name | Main owner | Dependency | Output |
|---:|---|---|---|---|
| 0 | Project setup | All | None | Reproducible repo and environment |
| 1 | Environment wrapper | All | Phase 0 | POGEMA wrapper and configs |
| 2 | Baseline PPO | All | Phase 1 | Trained clean baseline |
| 3 | Attack suite | All | Phase 2 | Random, FGSM, PGD, A* attacks |
| 4 | Baseline fragility | All | Phase 3 | Baseline attack plots and JSON results |
| 5A | Riad defence | Riad | Phase 4 | Adversarially trained policies |
| 5B | Akash defence | Akash | Phase 4 | Curriculum-trained policies |
| 5C | Sujosh defence | Sujosh | Phase 4 | Trained denoiser models |
| 6 | Cross-evaluation | All | Phases 5A–5C | Comparison tables and plots |
| 7 | Visualizations | All | Phase 6 | GIFs and final figures |
| 8 | Reports and presentation | All | Phase 7 | Final papers and slides |

Phases 5A, 5B, and 5C should run in parallel after the baseline and attack suite are stable.

---

## Phase 0 — Project Setup

### Purpose

Create a reproducible starting point so every team member can run the same code and get comparable results.

### Tasks

- Create a clean project structure:
  - `src/` for code,
  - `models/` for checkpoints,
  - `results/` for JSON results and plots,
  - `results/animations/` for GIFs,
  - `docs/` or root markdown files for notes and plans.
- Set up the Python environment.
- Install required packages:
  - PyTorch,
  - POGEMA,
  - NumPy,
  - Matplotlib,
  - tqdm,
  - imageio or Pillow for GIFs.
- Add a fixed seed utility used by training and evaluation.
- Decide naming conventions for runs:
  - `quick_baseline`,
  - `main_baseline`,
  - `main_riad_advtrain`,
  - `main_akash_curriculum`,
  - `main_sujosh_denoiser`.

### Files to create or verify

| File | Purpose |
|---|---|
| `requirements.txt` or environment file | Reproduce dependencies. |
| `src/utils.py` | Seeding, logging, device selection. |
| `PROJECT_CONTEXT.md` | Tracks project state. |
| `FULL_IMPLEMENTATION_PLAN.md` | This plan. |

### Done when

- Every member can run a small script that imports POGEMA and PyTorch.
- Results and model folders are created automatically when needed.
- The random seed is set in one place and reused everywhere.

---

## Phase 1 — Environment Wrapper

### Purpose

Wrap POGEMA so the rest of the code sees a clean multi-agent reinforcement learning interface.

### Tasks

- Implement the Quick, Main, and Scale configurations.
- Convert POGEMA observations into tensors with shape expected by the CNN policy.
- Standardize action IDs for stay/up/down/left/right.
- Return per-agent observations, rewards, done flags, and info dictionaries.
- Add helper methods for:
  - resetting with a seed,
  - stepping all agents at once,
  - reading agent positions,
  - reading goal positions,
  - rendering frames for GIFs.
- Add a minimal smoke test that runs random actions for one episode.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/pogema_wrapper.py` | Main environment wrapper. |
| `src/test_pogema.py` | Smoke test for reset, step, and observation shapes. |

### Done when

- Random agents can run a full Quick episode without crashing.
- Observation tensors have consistent shape and dtype.
- Episode termination works when all agents finish or max steps is reached.

---

## Phase 2 — Baseline PPO Policy

### Purpose

Train one shared decentralized policy that all defences use as the starting point.

### Method

Use PPO with parameter sharing. Parameter sharing means one neural network controls every agent. This is useful because all agents solve the same kind of local navigation problem, but it also means one bad update affects all agents.

### Tasks

- Implement the CNN actor-critic policy:
  - 3 convolution layers,
  - fully connected trunk,
  - actor head with 5 action logits,
  - critic head with one value estimate.
- Implement rollout collection from all agents.
- Implement PPO update:
  - clipped policy loss,
  - value loss,
  - entropy bonus,
  - advantage normalization.
- Save checkpoints:
  - `best_policy.pt`,
  - `final_policy.pt`,
  - `training_history.json`.
- Train first on Quick, then train on Main.
- Track success rate, episode return, makespan, entropy, and loss curves.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/ppo_mapf.py` | Policy network, PPO trainer, rollout buffer, evaluation. |
| `src/plot_training.py` | Training curves. |
| `models/main_baseline/` | Main baseline checkpoints. |

### Hyperparameters to start with

| Setting | Value |
|---|---:|
| Learning rate | 3e-4 |
| PPO epochs | 4 |
| Clip ratio | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Main timesteps | 2M |

### Done when

- The baseline reaches a reasonable clean success rate on Main.
- The same checkpoint is used by all three team members.
- Training curves are saved and can be plotted.

---

## Phase 3 — Attack Suite

### Purpose

Implement all attacks before building defences. The baseline must be attacked first so the team knows what failure looks like.

### Attack 1: Random Noise

Random noise adds uniform perturbations to observations. This is not a smart attack; it measures sensitivity to generic sensor corruption.

Tasks:

- Add uniform noise within an epsilon bound.
- Clip observations back to the valid range.
- Evaluate across the full epsilon sweep.

### Attack 2: FGSM

FGSM (Fast Gradient Sign Method) takes one gradient step in the direction that most increases the policy loss. It is a one-step adversarial observation attack.

Tasks:

- Enable gradients with respect to observations.
- Define the attack loss using the current policy output.
- Compute `sign(gradient)` and perturb by epsilon.
- Clip to the valid observation range.

### Attack 3: PGD

PGD (Projected Gradient Descent) repeats the FGSM idea for multiple small steps and projects the result back into the allowed epsilon ball. It is stronger than FGSM and helps detect gradient masking.

Tasks:

- Implement multi-step perturbation.
- Support configurable step size and number of steps.
- Use PGD in evaluation, not just training.

### Attack 4: Physical A* Chasers

A* is a graph search algorithm that expands paths based on actual cost so far plus a heuristic estimate to the goal. Here the heuristic is Manhattan distance on the grid.

Tasks:

- Implement A* pathfinding on the grid.
- Create adversary agents that chase the nearest cooperator.
- Add random-walk and goal-blocking variants for comparison.
- Make sure chasers occupy cells and affect movement physically, not just observations.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/attacks.py` | Random, FGSM, PGD, and partial observation attacks. |
| `src/adversary.py` | A*, random chasers, goal blockers. |
| `src/evaluate_fragility.py` | Shared attack evaluation loop. |

### Done when

- Each attack runs on the baseline checkpoint.
- Random noise, FGSM, and PGD use the same epsilon list.
- Physical chaser evaluation supports 0, 1, 2, and 4 chasers.

---

## Phase 4 — Baseline Fragility Profile

### Purpose

Measure how the clean baseline fails before adding any defence.

### Tasks

- Run the baseline on clean maps with fixed seeds.
- Run the baseline under random noise for every epsilon.
- Run the baseline under FGSM for every epsilon.
- Run the baseline under PGD for selected epsilon values.
- Run the baseline with physical A* chasers.
- Save results as JSON and plots.

### Required outputs

| Output | Purpose |
|---|---|
| `baseline_fragility.json` | Raw numbers for success rate and makespan. |
| `baseline_fragility.png` | Success rate vs. epsilon. |
| `physical_baseline.png` | Success rate vs. number of chasers. |

### Done when

- The team can point to one plot that shows how much FGSM hurts the baseline.
- Clean performance at epsilon 0 is reported beside attacked performance.
- These results become the reference point for all defences.

---

## Phase 5A — Riad Defence: Adversarial Training

### Purpose

Train policies that are exposed to adversarial observations during PPO training.

### Methods to implement

| Method | Description |
|---|---|
| SA-PPO | Replace observations with PGD worst-case observations during PPO updates. |
| Frozen-model FGSM | Generate FGSM attacks from a frozen copy of the baseline, then train the current policy on a clean/adversarial mix. |
| TRADES-style regularization | Keep the clean PPO loss and add a KL penalty that encourages similar outputs on clean and perturbed observations. |

KL divergence measures how different two probability distributions are. Here it measures how much the policy's action distribution changes when the observation is slightly corrupted.

### Main tasks

- Start from the shared baseline checkpoint.
- Implement adversarial observation generation inside the PPO update.
- Freeze a copy of the baseline for frozen-model FGSM.
- Mix clean and attacked observations.
- Add entropy tracking to detect policy collapse.
- Add ablations:
  - adversarial mix ratio alpha,
  - smoothness coefficient lambda,
  - FGSM vs. PGD training attack.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/adv_train.py` | Riad's adversarial training code. |
| `models/main_riad_advtrain/` | Best and final checkpoints. |
| `results/main/riad_ablation.json` | Ablation results. |

### Done when

- At least one adversarially trained policy improves FGSM success rate over the baseline.
- Clean performance is not destroyed.
- Entropy curves show whether collapse happened.
- FGSM and PGD evaluation are both reported.

---

## Phase 5B — Akash Defence: Physical Adversary Curriculum

### Purpose

Train the policy in a harder environment where physical adversary agents chase or block cooperative agents.

### Methods to implement

| Strategy | Description |
|---|---|
| Random walk | Chaser moves randomly. Lower baseline. |
| A* pursuit | Chaser computes shortest path to nearest cooperator and follows it. |
| Goal blocking | Chaser moves to the cooperator's goal and occupies it. |

### Curriculum

Curriculum learning means increasing difficulty gradually. Here the difficulty is the number of adversary agents. Start with zero chasers and ramp up to the maximum during the first part of training.

### Main tasks

- Start from the shared baseline checkpoint.
- Add physical adversaries to the training environment.
- Implement curriculum schedule:
  - 0 chasers at the start,
  - gradually increase to max chasers,
  - compare warmup fractions of 20%, 40%, and 60%.
- Compare random walk, A* pursuit, and goal blocking.
- Add optional small Gaussian observation noise during training.
- Save training histories and checkpoints.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/curriculum_train.py` | Akash's curriculum training code. |
| `src/adversary.py` | A*, random walk, and goal-blocking logic. |
| `models/main_akash_curriculum/` | Best and final checkpoints. |
| `results/main/akash_ablation.json` | Curriculum ablation results. |

### Done when

- The curriculum-trained policy is evaluated under physical chasers.
- The same policy is also evaluated under FGSM to test cross-robustness.
- The report can compare A* chasers against random chasers and goal blockers.

---

## Phase 5C — Sujosh Defence: Observation Denoising

### Purpose

Train a preprocessing model that cleans corrupted observations before the policy sees them.

The policy itself stays frozen. The denoiser sits between the raw observation and the policy:

`raw observation -> denoiser -> policy -> action`

### Dataset

Create paired observations:

| Input | Target |
|---|---|
| Corrupted observation | Original clean observation |

Corruption sources:

- random noise,
- FGSM,
- mixed random + FGSM.

### Denoiser models to compare

| Model | Description |
|---|---|
| Shallow CNN | Small 2-layer convolutional network. |
| Deep residual CNN | Deeper model with skip connections. |
| Bottleneck autoencoder | Compresses observation then reconstructs it. |

An autoencoder is a network trained to copy its input through a compressed internal representation. A denoising autoencoder receives a corrupted input and learns to reconstruct the clean version.

### Main tasks

- Collect clean observations from baseline rollouts.
- Create corrupted copies with random noise and FGSM.
- Train denoiser models with mean squared error loss.
- Insert denoiser before the frozen baseline policy at test time.
- Measure:
  - success rate under attack,
  - reconstruction MSE,
  - inference latency.
- Test denoising on top of:
  - baseline policy,
  - Riad's best policy,
  - Akash's best policy.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/denoising_defense.py` | Denoiser models, training, and evaluation wrapper. |
| `models/main_sujosh_denoiser/` | Trained denoiser checkpoints. |
| `results/main/sujosh_ablation.json` | Denoiser ablation results. |

### Done when

- At least one denoiser improves attacked success rate over the raw baseline.
- Reconstruction MSE and success rate are both reported.
- Inference overhead is measured.

---

## Phase 6 — Cross-Evaluation

### Purpose

Evaluate every defence under the same attacks so the comparison is fair.

### Policies to evaluate

| Policy | Source |
|---|---|
| Baseline | Phase 2 |
| Riad best | Phase 5A |
| Akash best | Phase 5B |
| Sujosh denoised baseline | Phase 5C |
| Sujosh denoiser + Riad policy | Phase 5C stacked test |
| Sujosh denoiser + Akash policy | Phase 5C stacked test |

### Attacks to run

| Attack | Conditions |
|---|---|
| Clean | epsilon = 0 |
| Random noise | all epsilon values |
| FGSM | all epsilon values |
| PGD | selected epsilon values, especially 0.10 and 0.15 |
| Partial FGSM | attack 25%, 50%, and 100% of agents |
| Physical A* | 0, 1, 2, and 4 chasers |
| Combined attack | FGSM plus physical chasers, if time allows |

### Required tables

1. Clean success rate for all policies.
2. FGSM success rate at epsilon 0.15 for all policies.
3. Performance drop from clean to attacked.
4. Makespan under clean and attacked conditions.
5. Runtime overhead for denoising.

### Required plots

| Plot | Purpose |
|---|---|
| `three_way_fgsm.png` | Main comparison: success rate vs. epsilon. |
| `three_way_pgd.png` | Stronger gradient attack comparison. |
| `physical_attack.png` | Success rate vs. number of chasers. |
| `partial_attack.png` | Success rate vs. fraction of agents attacked. |
| `training_curves.png` | Baseline and defence training curves. |

### Done when

- All policies are evaluated with the same seeds.
- Results are saved as JSON before plotting.
- Every figure in the report can be regenerated from saved data.

---

## Phase 7 — Visualizations and Qualitative Analysis

### Purpose

Use animations and short case studies to explain what is happening in the grid.

### Tasks

- Generate GIFs for:
  - clean baseline,
  - baseline under FGSM,
  - Riad policy under FGSM,
  - Akash policy with physical chasers,
  - Sujosh denoiser under FGSM.
- Pick 2–3 episodes where failure is easy to see.
- Write short notes for each animation:
  - what the agents tried to do,
  - where the failure started,
  - how the defence changed behavior.

### Files to create or verify

| File | Purpose |
|---|---|
| `src/visualize.py` | Render frames and GIFs. |
| `results/animations/` | Final GIFs. |

### Done when

- Each team member has at least one GIF that supports their method section.
- The presentation has one clean failure example and one defended example.

---

## Phase 8 — Reports and Presentation

### Purpose

Turn the implementation and results into final course deliverables.

### Shared report content

All papers should use the same shared description for:

- problem setup,
- POGEMA environment,
- PPO baseline,
- attacks,
- evaluation metrics,
- seeds and configurations.

### Individual report content

| Person | Main section |
|---|---|
| Riad | Adversarial training, entropy collapse, frozen-model fix. |
| Akash | A* physical adversaries, curriculum schedule, physical attack results. |
| Sujosh | Denoising dataset, denoiser architecture, inference-time results. |

### Presentation structure

| Part | Time | Speaker |
|---|---:|---|
| Problem and setup | 2 min | Any |
| Baseline fragility | 2 min | Any |
| Riad method/results | 3–4 min | Riad |
| Akash method/results | 3–4 min | Akash |
| Sujosh method/results | 3–4 min | Sujosh |
| Final comparison and takeaway | 2 min | Any |

### Done when

- Every figure in the report has a caption that tells the reader what to notice.
- Every result number in the report matches a saved JSON result.
- The presentation includes at least one animation.
- The final conclusion says plainly which defence worked best under which attack.

---

## Suggested Implementation Order

This is the shortest safe path from nothing to full results.

1. Build and test the POGEMA wrapper on Quick.
2. Train a tiny PPO run on Quick to verify the training loop.
3. Train the full baseline on Main.
4. Implement random noise, FGSM, and PGD.
5. Run the baseline fragility sweep.
6. Implement A* chasers and physical attack evaluation.
7. Split into the three defence tracks:
   - Riad starts adversarial training.
   - Akash starts curriculum training.
   - Sujosh starts collecting denoiser data.
8. Run all defences on Quick first.
9. Run only the best settings on Main.
10. Run cross-evaluation using fixed seeds.
11. Generate plots and animations.
12. Write reports and presentation.

---

## Minimum Viable Version

If time becomes tight, finish this reduced version first:

1. Baseline PPO on Main.
2. Random noise and FGSM attacks.
3. One Riad defence: frozen-model FGSM training.
4. One Akash defence: A* curriculum training with one warmup schedule.
5. One Sujosh defence: shallow CNN denoiser trained on mixed random + FGSM corruption.
6. One comparison plot: success rate vs. FGSM epsilon.
7. One summary table: clean success, FGSM epsilon 0.15 success, and performance drop.

This version is enough to answer the core project question.

---

## Risk Plan

| Risk | Symptom | Response |
|---|---|---|
| PPO does not learn | Clean success stays near random | Reduce map size, verify rewards, overfit Quick first. |
| FGSM code gives no effect | Success does not drop as epsilon increases | Check observation gradients and attack loss. Compare against random noise. |
| Adversarial training collapses | Entropy goes near zero and policy always picks same action | Lower adversarial weight, use frozen model, keep clean loss. |
| A* chasers break environment | Episodes crash or agents overlap incorrectly | Test chasers separately before training with them. |
| Curriculum too hard | Success drops immediately after chasers appear | Increase warmup fraction or reduce max chasers. |
| Denoiser improves MSE but not success | Reconstructed observations look clean but policy still fails | Train on policy-relevant corruptions and report both MSE and success rate. |
| Full runs take too long | Main training cannot finish | Use Quick for ablations and run only best settings on Main. |

---

## Definition of Done

The project is fully implemented when these checks pass:

- [ ] A clean baseline checkpoint exists and can be evaluated.
- [ ] Random, FGSM, PGD, and physical A* attacks run through one shared evaluator.
- [ ] Baseline fragility results are saved and plotted.
- [ ] Riad's best adversarially trained checkpoint is saved and evaluated.
- [ ] Akash's best curriculum-trained checkpoint is saved and evaluated.
- [ ] Sujosh's best denoiser checkpoint is saved and evaluated.
- [ ] All methods are evaluated with the same seeds and metrics.
- [ ] The main comparison plot includes baseline plus all three defences.
- [ ] Raw JSON results exist for every plot and table.
- [ ] At least three GIFs are generated for qualitative explanation.
- [ ] Individual reports and the presentation are consistent with the saved results.
