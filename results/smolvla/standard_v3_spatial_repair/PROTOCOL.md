# Original standard LIBERO: v3_spatial_repair

Checkpoint: `training/smol_v3_spatial_continuation_full_2540524/output/checkpoints/020000/pretrained_model`. 40 original tasks x 50 saved initial states = 2000 episodes. Trial e uses state e and seed 7+e; ten settling steps; ten executed actions/query. Horizons 280/280/300/520 (Spatial differs from LIBERO+'s 220). Original instructions, two rotated 256px images, 8D state and 7D relative actions. LIBERO commit 8f1084e3132a39270c3a13ebe37270a43ece2a01; MuJoCo 3.8.1, robosuite 1.4.1; separate original-LIBERO evaluator on Turing.

Retrieved raw per-episode comparison export: 1425/2000 = 71.25%. `results.csv` has actual recorded outcomes, not reconstructions from suite rates. `seconds` is unavailable in the export and remains blank. Standard results for noise/clean/v2 remain aggregate-only in their separate cards. Cross-team evaluations are different-protocol references.
