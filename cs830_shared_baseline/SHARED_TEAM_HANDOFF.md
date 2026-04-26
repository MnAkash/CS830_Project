# Shared Team Handoff

Generated: 2026-04-26 11:25:54

This file marks the baseline + wrapper layer as ready for Riad, Akash, and Sujosh to build their own parts on top of it.

This handoff does **not** implement Riad's adversarial training, Akash's curriculum training, or Sujosh's smoothing defence. It only provides the shared foundation.

## Verified shared pieces

- [x] Phase 1 POGEMA wrapper with Quick/Main/Scale configs.
- [x] Phase 2 baseline PPO checkpoint.
- [x] Baseline sanity success target met: 96.5% over 50 held-out episodes.
- [x] Random noise attack stays inside epsilon bound.
- [x] FGSM attack stays inside epsilon bound.
- [x] PGD attack stays inside epsilon bound.
- [x] Partial-agent attack wrapper works.
- [x] A* physical chaser logic works on a hand-written grid and through POGEMA evaluation.
- [x] Shared fragility plots and visual report generated.

## Shared baseline checkpoint

/home/carl_ma/Riad/MS/CS830/project/shared_baseline/models/phase2_smoke_baseline/best_policy.pt

## Shared readiness report

/home/carl_ma/Riad/MS/CS830/project/shared_baseline/results/shared_readiness/shared_readiness_report.html

## Baseline + wrapper handoff guide

BASELINE_WRAPPER_HANDOFF.md

## What each person should start from

| Person | Their part | Shared baseline inputs |
|---|---|---|
| Riad | adversarial training defence | baseline checkpoint, `src/ppo_mapf.py`, `src/attacks.py`, fragility sweep code |
| Akash | curriculum / physical adversary defence | baseline checkpoint, `src/pogema_wrapper.py`, `src/adversary.py`, physical sweep code |
| Sujosh | inference-time smoothing defence | baseline checkpoint, `PolicyNetwork` inference API, attack/evaluation code |

## Rule for the next phases

Do not change the shared baseline checkpoint, attack definitions, or evaluation seeds unless all three tracks agree. Otherwise the comparison stops being fair.
