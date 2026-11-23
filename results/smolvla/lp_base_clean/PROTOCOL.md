# lp_base_clean

| Setting | Value |
|---|---|
| Model/checkpoint | lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d |
| Task set | 10030 LIBERO+ variants; suite + zero-based task_id |
| Initial state / settling | first saved state;10 no-op steps |
| Max steps Spatial/Object/Goal/10 | 220/280/300/520 |
| Actions executed/query | 10, policy cache; one returned action per env step |
| Instructions | clean; intentional Language perturbations preserved |
| Env and policy seed | 7 reset before each task |
| Stack/machine | MuJoCo3.8.1, robosuite1.4.1; WPI Turing A30 |
| Trials | one per variant |
| Outcome | 5372/10030 = 53.5593% |
| Reused outcomes | 1537 Language-task outcomes from the corresponding official run |

## Comparability and provenance
Within this SmolVLA stack, compare matching task IDs and instruction conditions. Cross-model pairing with the team's MuJoCo3.2.0 runs is **different protocol / aggregate reference only**, not an exact causal paired comparison.

`seconds` is blank because exported CSVs lack timing. Blank instruction fields occur in the Language category because exact BDDL-derived text was absent from that export; they are unknown, not empty prompts fed to the policy. Non-Language text comes from the frozen instruction map checked against task.language. Error/skip flags are0, supported by the completed status reports. Results are not reconstructed from aggregate percentages.

The task plan preserves the exported row order; round labels were not preserved and are set0. Full benchmark rows flag membership in the original4,295 holdout. New teacher training excluded those4,295 exact variants, not the underlying base-task families. The10,030 full benchmark includes some original-v2 teacher source variants, so it is not entirely held out. No raw rollout timing, logs, weights, videos or datasets are included.
