# PACT: Physical-Adversary Curriculum Training

This folder contains Moniruzzaman Akash's PACT implementation, separated from the shared baseline code.

## What lives here

- `curriculum_train.py`: trains/evaluates PACT and PACT-Mix style physical-adversary curricula.
- `adversary.py`: physical adversary strategies: A* pursuit, goal blocking, random walk, and mixed.
- `ppo_mapf.py`: PACT-local PPO trainer/evaluator with curriculum adversary controls.
- `evaluate_fragility.py`: PACT-local robustness sweep helpers that understand adversary strategy.
- `pogema_compat.py`: local POGEMA/Gymnasium compatibility patch used by PACT runs.
- `visualize.py`: rollout GIF generation for baseline-vs-PACT physical adversary comparisons.
- `path_utils.py`: explicit path bridge to `../cs830_shared_baseline/src`.

## Dependency boundary

PACT keeps the Akash-specific runtime code here. It may still import unchanged shared utilities, such as attack functions and JSON/path helpers, from:

```text
../cs830_shared_baseline/src
```

The shared baseline source is treated as a library fallback, not as the place to run PACT scripts. PACT-owned checkpoints and outputs live in this folder:

```text
pact/models/
pact/results/
```

The shared baseline keeps only the baseline checkpoint and original baseline/readiness artifacts. Akash/Pact experiment models and results belong under `pact/models/` and `pact/results/`.

## Example commands

Train a short PACT curriculum from the shared baseline checkpoint:

```bash
python cs830_final_project/pact/curriculum_train.py --mode train --config smoke --total-timesteps 4096 --n-steps 128 --batch-size 128 --device cpu
```

Evaluate an existing PACT checkpoint:

```bash
python cs830_final_project/pact/curriculum_train.py --mode evaluate --config quick --akash-checkpoint cs830_final_project/pact/models/main_akash_curriculum/best_policy.pt --device cpu
```
