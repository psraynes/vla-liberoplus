# SmolVLA analysis

From the repo root:

```bash
python experiments/smolvla/validate_results.py
python experiments/smolvla/analyze_paired.py --input results/smolvla/source_exports --out /tmp/smolvla_analysis
```

These scripts use the standard library, not a simulator or GPU. The first checks original completed run coverage/totals. The second validates new full/holdout/standard keys, unique paired outcomes, supplied suite/category rates and shared speed query counts, then writes paired gains/losses, nominal exact tests and a six-overall-test Holm correction. Task-family correlation and single-seed limitations remain; do not turn exploratory subgroup p values into broad claims.

Methods: docs/plans/PLAN_smolvla_results_and_spatial_repair.md. Protocol cards distinguish raw per-task/per-episode evidence from remaining aggregate-only historical records. No times or outcomes are reconstructed from percentages.
