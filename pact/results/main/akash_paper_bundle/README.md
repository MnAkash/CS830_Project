# Akash Physical Adversary Results Bundle

Generated on 2026-04-26 for Moniruzzaman Akash's physical adversary robustness track.

## Main Claim

Akash's curriculum-trained PPO policy improves robustness against physical adversary agents that act as A* pursuit chasers in POGEMA MAPF.

Quick-scale evidence is the strongest result and should be the main paper/report result:

| Metric | Shared baseline | Akash curriculum | Change |
|---|---:|---:|---:|
| Clean success | 64.58% | 70.83% | +6.25 pp |
| Success with 4 physical chasers | 52.50% | 69.58% | +17.08 pp |
| Drop from clean to 4 chasers | 12.08 pp | 1.25 pp | 10.83 pp smaller drop |
| Physical robustness AUC | 0.5859 | 0.7104 | +0.1245 |

## Experimental Setup

- Config: quick
- Social agents: 8
- Grid size: 16 x 16
- Obstacle density: 0.30
- Max episode steps: 128
- Evaluation episodes per sweep point: 30
- Physical adversary strategy: astar_pursuit
- Curriculum max adversaries: 4
- Curriculum warmup fraction: 0.60
- Training timesteps: 524,288 social-agent environment steps
- Device: CPU

## Saved Models

- Best Akash quick policy: ../akash_quick is results; model is ../../../models/quick_akash_curriculum/best_policy.pt
- Final Akash quick policy: ../../../models/quick_akash_curriculum/final_policy.pt
- Shared baseline checkpoint: ../../../models/phase2_smoke_baseline/best_policy.pt

## Ready-To-Use Figures

Use these in the paper/report:

- Physical robustness plot: ../akash_quick/akash_physical_comparison.png
- FGSM cross-robustness plot: ../akash_quick/akash_fgsm_comparison.png
- Curriculum training plot: ../akash_quick/akash_curriculum_training.png

## Ready-To-Use Tables

- Physical adversary sweep: quick_physical_sweep.csv
- FGSM cross-robustness sweep: quick_fgsm_sweep.csv
- Combined FGSM + physical sweep: quick_combined_sweep.csv

## Animations

Physical-only quick-scale animations were generated with seed 1000, where the shared baseline reaches 2/8 goals and Akash curriculum reaches 7/8 goals under 4 chasers.

- Baseline clean: ../akash_quick/animations_physical/baseline_clean.gif
- Baseline attacked: ../akash_quick/animations_physical/baseline_attacked.gif
- Akash clean: ../akash_quick/animations_physical/fast_clean.gif
- Akash attacked: ../akash_quick/animations_physical/fast_attacked.gif
- Static final-frame comparison: ../akash_quick/animations_physical/final_frame_comparison.png

The full side-by-side GIF render was intentionally stopped because it was slow on CPU after the individual GIFs had already been written. Use the four GIFs plus the static final-frame comparison for presentation/paper material.

## Important Limitation

Do not claim broad adversarial robustness. The method improves physical-adversary robustness, but FGSM observation-space robustness is weaker at higher epsilon values. Present FGSM as a tradeoff/limitation and keep the main claim focused on physical adversary agents.

Suggested claim text:

> Curriculum training with physical A* pursuit adversaries substantially improves MAPF policy performance under physical adversary pressure, increasing success under 4 chasers from 52.5% to 69.6% on the quick benchmark while reducing the clean-to-attacked success drop from 12.1 percentage points to 1.3 percentage points.
