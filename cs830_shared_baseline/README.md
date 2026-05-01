# Shared Baseline + Wrapper Package

This folder is the clean handoff package for Riad, Akash, and Sujosh.

It contains the shared MAPF foundation only:

- POGEMA environment wrapper
- PPO baseline policy code
- verified baseline checkpoint
- shared attack/evaluation utilities
- visual validation reports
- handoff documentation

It does **not** implement Riad's, Akash's, or Sujosh's defence systems. Each member should build their part on top of this package.

---

## 1. Folder layout

```text
shared_baseline/
├── README.md
├── requirements.txt
├── BASELINE_WRAPPER_HANDOFF.md
├── SHARED_TEAM_HANDOFF.md
├── PROJECT_CONTEXT.md
├── FULL_IMPLEMENTATION_PLAN.md
├── src/
│   ├── pogema_wrapper.py
│   ├── ppo_mapf.py
│   ├── utils.py
│   ├── attacks.py
│   ├── adversary.py
│   ├── evaluate_fragility.py
│   ├── visualize.py
│   ├── phase1_visualize.py
│   ├── phase2_train_baseline.py
│   ├── shared_readiness.py
│   └── test_pogema.py
├── models/
│   └── phase2_smoke_baseline/
│       ├── best_policy.pt
│       ├── final_policy.pt
│       └── training_history.json
└── results/
    ├── phase1_environment/
    ├── phase2_baseline/
    └── shared_readiness/
```

Run commands from inside this folder unless stated otherwise.

---

## 2. Environment setup

Use the verified conda environment:

```bash
source /home/carl_ma/miniconda3/etc/profile.d/conda.sh
conda activate grasp_splats
cd /home/carl_ma/Riad/MS/CS830/project/shared_baseline
```

If another machine does not already have the packages, install:

```bash
pip install -r requirements.txt
```

Required packages:

- numpy
- torch
- pogema
- gymnasium
- matplotlib
- imageio
- tqdm

---

## 3. Verified baseline checkpoint

Use this checkpoint as the common starting point for all three members:

```text
models/phase2_smoke_baseline/best_policy.pt
```

Validation result:

- environment: 4 agents, 8×8 grid, obstacle density 0.1, 64 max steps
- policy evaluation mode: sampled PPO policy
- success rate: **96.5%** over 50 held-out sanity episodes
- readiness status: `ready = true`

Do not replace this checkpoint unless all three team members agree, because otherwise defence comparisons will not be fair.

---

## 4. Core system pieces

### 4.1 POGEMA wrapper

File:

```text
src/pogema_wrapper.py
```

Main interface:

- `MapfConfig`: project-level environment configuration
- `QUICK_CONFIG`: 8 agents, 16×16 grid, 128 steps
- `MAIN_CONFIG`: 16 agents, 20×20 grid, 192 steps
- `SCALE_CONFIG`: 32 agents, 32×32 grid, 256 steps
- `MultiAgentPogemaEnv`: wrapper around POGEMA
- `make_pogema_env(config)`: environment factory

Important wrapper methods:

- `reset(seed=None)`: returns observation batch
- `step(actions)`: takes one action per agent
- `sample_actions()`: random action batch for debugging
- `get_agent_positions()`: current agent positions
- `get_goal_positions()`: current goal positions
- `get_obstacles()`: obstacle grid
- `get_state()`: full visualization/debug state

Observation shape for `obs_radius=2`:

```text
(num_agents, 3, 5, 5)
```

Channels:

1. obstacles
2. nearby agents
3. goal direction

### 4.2 PPO baseline

File:

```text
src/ppo_mapf.py
```

Main interface:

- `PolicyNetwork`: shared CNN actor-critic policy
- `RolloutBuffer`: PPO rollout storage
- `PPOTrainer`: baseline PPO trainer
- `evaluate_policy(policy, grid_config, ...)`: clean evaluation

Architecture:

- CNN encoder over 3×5×5 observations
- actor head outputs 5 action logits
- critic head outputs a value estimate
- one shared policy is used by all agents

Action space:

```text
0 = stay
1 = up
2 = down
3 = left
4 = right
```

### 4.3 Attack/evaluation utilities

These utilities are included so future defences can be evaluated consistently.

They are not defence implementations.

Files:

```text
src/attacks.py
src/adversary.py
src/evaluate_fragility.py
src/visualize.py
```

Available observation attacks:

- `random_noise_attack()`
- `fgsm_attack()`
- `pgd_attack()`
- `partial_attack()`

Available physical adversary helpers:

- `astar_next_step()`
- `bfs_next_step()`
- `adversary_actions()`

Available evaluation sweeps:

- `run_fragility_sweep()`
- `run_partial_attack_sweep()`
- `run_physical_attack_sweep()`
- `plot_fragility()`
- `plot_partial()`
- `plot_physical_comparison()`

---

## 5. How to run validation

### 5.1 Test the wrapper

```bash
python src/test_pogema.py
```

This checks:

- named configs
- reset/step API
- seed reproducibility
- phase-1 visual report generation

Outputs:

```text
results/phase1_environment/
```

### 5.2 Re-run baseline training/report

Use this only if the baseline must be retrained:

```bash
python src/phase2_train_baseline.py --device cpu --total-timesteps 65536 --eval-episodes 50 --success-target 0.95
```

Outputs:

```text
models/phase2_smoke_baseline/
results/phase2_baseline/
```

Expected sanity target:

```text
success rate >= 95%
```

### 5.3 Re-run shared readiness check

```bash
python src/shared_readiness.py --device cpu --episodes 8
```

This checks:

- baseline checkpoint loads correctly
- baseline success is at least 95% on the easy sanity environment
- random/FGSM/PGD/partial-FGSM attacks stay inside epsilon bounds
- A* and BFS chaser logic works
- fragility plots are generated
- physical chaser GIF is generated

Outputs:

```text
results/shared_readiness/shared_readiness_report.html
results/shared_readiness/shared_readiness_summary.json
results/shared_readiness/shared_attack_observation_panel.png
results/shared_readiness/shared_baseline_fragility.png
results/shared_readiness/shared_partial_attack.png
results/shared_readiness/shared_physical_adversary.png
results/shared_readiness/shared_physical_chaser_preview.gif
```

---

## 6. How each team member should build on top

### 6.1 Riad: adversarial training defence

Riad should build training-time robustness on top of the shared PPO baseline.

Recommended starting files:

```text
src/ppo_mapf.py
src/attacks.py
src/evaluate_fragility.py
models/phase2_smoke_baseline/best_policy.pt
```

Suggested new file name:

```text
src/riad_adv_training.py
```

Do not change the shared `PolicyNetwork` API unless all members agree.

### 6.2 Akash: curriculum / physical-adversary defence

Akash should build environment-side robustness on top of the wrapper and A* adversary utilities.

Recommended starting files:

```text
src/pogema_wrapper.py
src/ppo_mapf.py
src/adversary.py
src/evaluate_fragility.py
models/phase2_smoke_baseline/best_policy.pt
```

Suggested new file name:

```text
src/akash_curriculum_training.py
```

Do not change the wrapper return shapes or action mapping unless all members agree.

### 6.3 Sujosh: inference-time smoothing defence

Sujosh should build an inference wrapper around the trained policy.

Recommended starting files:

```text
src/ppo_mapf.py
src/attacks.py
src/evaluate_fragility.py
models/phase2_smoke_baseline/best_policy.pt
```

Suggested new file name:

```text
src/sujosh_smoothing.py
```

This part should not require retraining the baseline policy.

---

## 7. Minimal policy-loading example

```python
from pathlib import Path

import torch

from ppo_mapf import PolicyNetwork


checkpoint = Path("models/phase2_smoke_baseline/best_policy.pt")
policy = PolicyNetwork(obs_size=5)
policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
policy.eval()
```

When running a new script inside `src/`, use:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

---

## 8. Fair comparison rules

All members should keep these fixed unless the full team agrees:

- baseline checkpoint
- action mapping
- observation shape
- wrapper API
- attack definitions
- evaluation seeds/configs
- success-rate metric

Each defence can add its own training or inference logic, but final comparisons should be run with the shared evaluation utilities.

---

## 9. Reports for visual understanding

Open these files in a browser:

```text
results/phase1_environment/phase1_environment_report.html
results/phase2_baseline/phase2_baseline_report.html
results/shared_readiness/shared_readiness_report.html
```

They show:

- the wrapper/environment behavior
- the PPO baseline training result
- attack sensitivity and physical-adversary sanity checks
