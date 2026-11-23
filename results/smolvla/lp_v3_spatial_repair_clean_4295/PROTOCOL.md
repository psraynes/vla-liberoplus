# Spatial-repaired v3, clean instructions — frozen subset

| Setting | Value |
|---|---|
| Checkpoint | `training/smol_v3_spatial_continuation_full_2540524/output/checkpoints/020000/pretrained_model`; v3 step 400000 plus 20000 continuation updates |
| Tasks | 4295 unique suite + zero-based task_id pairs; frozen plan included |
| Init / settle | First saved initial state; ten no-op steps |
| Horizons Spatial/Object/Goal/10 | 220 / 280 / 300 / 520 |
| Executed actions/query | 10; internal policy queue, one action returned per step |
| Instructions | Canonical clean wording; intentional Language variants preserved |
| Seed / trials | 7 reset per task; one trial per variant |
| Stack | MuJoCo 3.8.1, robosuite 1.4.1, LeRobot 0.6.1; Turing GPU jobs |
| Input/action | Two 256px images rotated 180 degrees; 8D state, 7D relative action |
| Outcome | 2098/4295 = 48.8475% |
| Errors/skips | Zero, confirmed by completed worker reports |

The repaired checkpoint reran all 10030 clean tasks, including Language. The 4295 folder is a subset of that same evaluation, not a separate run. Earlier base/v3 clean results reuse 1537 unchanged Language outcomes. Within-SmolVLA matching-prompt comparisons are supported; cross-model rates remain a different-simulator reference. Full benchmark contains some teacher source variants; the frozen subset excludes collection source IDs but shares families and was inspected during development. It is not a final blind test.

`seconds` is blank because the exported comparison CSV omits per-rollout timing. Instruction strings for Language remain unknown where absent from prior exports; blanks are not empty instructions fed to the model. Other instruction text is joined from the verified frozen map. No action labels, training episodes, weights, videos or logs are distributed.
