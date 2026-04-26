# Baseline + Wrapper Handoff

This is the package to hand over before Riad, Akash, and Sujosh build their own parts.

This folder is the self-contained handoff package. Start with:

```text
README.md
```

## Scope

This handoff contains only the shared foundation:

- POGEMA MAPF wrapper
- PPO baseline policy and checkpoint
- baseline training/evaluation scripts
- shared attack/evaluation utilities for testing future defences
- visual validation reports

It does **not** implement Riad's adversarial training, Akash's curriculum training, or Sujosh's smoothing defence. Those should be built by them on top of this baseline.

## Stable wrapper interface

Use `src/pogema_wrapper.py` for environment access.

Important objects/functions:

- `MapfConfig`: project-level environment config
- `QUICK_CONFIG`, `MAIN_CONFIG`, `SCALE_CONFIG`: named configs
- `MultiAgentPogemaEnv`: project wrapper around POGEMA
- `make_pogema_env(config)`: factory function
- `reset()`: returns observations with shape `(num_agents, 3, 5, 5)` for `obs_radius=2`
- `step(actions)`: steps all agents together
- `get_state()`: returns positions, goals, obstacles, done flags, and step count for visualization/debugging

## Stable baseline interface

Use `src/ppo_mapf.py` for the baseline policy and PPO tools.

Important objects/functions:

- `PolicyNetwork`: shared CNN actor-critic policy
- `PPOTrainer`: baseline PPO trainer
- `evaluate_policy(policy, grid_config, ...)`: clean-policy evaluation

Verified checkpoint for all three tracks:

```text
models/phase2_smoke_baseline/best_policy.pt
```

This checkpoint reached **96.5% sampled-policy success** over 50 held-out easy sanity episodes.

## Shared evaluation utilities

These are for measuring future defence work, not defence implementations themselves.

- `src/attacks.py`: random noise, FGSM, PGD, partial-agent attack utilities
- `src/adversary.py`: A* / BFS chaser utilities for physical stress tests
- `src/evaluate_fragility.py`: attack and physical-adversary sweep functions
- `src/shared_readiness.py`: verifies the baseline handoff and regenerates the readiness report
- `src/phase1_visualize.py`: wrapper/environment visualization
- `src/phase2_train_baseline.py`: baseline training and baseline report generation

## Reports to share

- `README.md`
- `results/phase1_environment/phase1_environment_report.html`
- `results/phase2_baseline/phase2_baseline_report.html`
- `results/shared_readiness/shared_readiness_report.html`
- `SHARED_TEAM_HANDOFF.md`

## Reproduce the handoff check

From the project root:

```bash
source /home/carl_ma/miniconda3/etc/profile.d/conda.sh
conda activate grasp_splats
python src/shared_readiness.py --device cpu --episodes 8
```

Expected result:

- `ready = true` in `results/shared_readiness/shared_readiness_summary.json`
- baseline success at or above 95% on the easy sanity setting
- all attack epsilon-bound checks pass
- A* chaser sanity checks pass

## How each person should build on top

| Person | Their task | Start from shared baseline |
|---|---|---|
| Riad | adversarial training defence | load the baseline checkpoint, reuse `PolicyNetwork`, `PPOTrainer`, and `src/attacks.py` |
| Akash | curriculum/physical adversary defence | load the baseline checkpoint, reuse `src/pogema_wrapper.py` and `src/adversary.py` |
| Sujosh | inference-time smoothing defence | load the baseline checkpoint and wrap policy inference without changing training |

## Fairness rule

All three tracks should use the same baseline checkpoint, wrapper, evaluation seeds, and attack definitions unless the whole team agrees to change them.
