# SmolVLA results and Spatial recovery experiments

**Evidence status, 22 November 2026:** collection, fine-tuning, official/clean evaluations, Spatial continuation and shared speed measurements are complete. Downloaded per-task/per-episode exports validate the recovery results and paired comparisons.

## Evaluation protocol

The base is `lerobot/smolvla_libero` at `a24f1f380461b665158d52acd140152c79e8b36d`, approximately 450M parameters. LIBERO+ uses first saved initial state, 10 settling steps, horizons 220/280/300/520 for Spatial/Object/Goal/10, seed 7, one trial per variant, two 256px images rotated 180 degrees, 8-D state, 7-D relative actions, and 10 internally cached executed actions per query. MuJoCo 3.8.1, robosuite 1.4.1 and LeRobot 0.6.1 were used on Turing.

Standard LIBERO uses the separate original checkout, 40 tasks x 50 initial states, seed `7+episode`, and horizons **280**/280/300/520. Values are our measurements, not published model-card scores. Run cards under `results/smolvla/` record the protocol and missing fields.

The team's main simulator was MuJoCo 3.2.0. This is compatibility **grade B / different-protocol aggregate reference** for cross-team comparisons, not an exact cross-model paired result. Within-SmolVLA comparisons require identical task IDs, prompts and settings. These are single-trial variant outcomes, with no measured same-stack rerun floor or multi-seed uncertainty here.

## Training and data

Noise and clean controls start from the same base and LIBERO demonstration source: 4,000 updates, batch 2, seed 42. The noise arm adds Gaussian readout noise, motion blur and JPEG corruption. The clean control does not. The teacher-v2 student used 1,263 train / 149 validation successful pi0.5 episodes and 100,000 updates.

Combined v3 starts from released base, not v2: 5,105 train / 586 validation episodes, 839,358 training frames, 400,000 updates, batch 2, seed 42, corrected 400k scheduler, training run 2473800. Its recipe totals are original v2 1,412 successes; new view 1,744; mild blur including retry 1,747; dual blur 788. The teacher is `lerobot/pi05_libero_finetuned_v044@7152f13e34c6c115d808aea0e5d08aca74eac14b`.

Teacher success selection is not uniform robustness sampling. The dataset excludes the original 4,295 exact evaluation IDs, but shares underlying task families. Some original teacher source variants are in the later 10,030-task benchmark, so its score is **not wholly held out**. Spatial was absent from v3 teacher training. Dedicated lighting/background augmentation was not studied.

## Clean-instruction ablation

For each checkpoint, 8,493 non-Language variants were rerun with canonical parent-task wording; 1,537 intentional Language-task outcomes were reused unchanged. Checkpoint, seed and environment were otherwise fixed.

| Set | Base official | Base clean | V3 official | V3 clean |
|---|---:|---:|---:|---:|
| Full 10,030 | 26.73 | 53.56 | 45.57 | 39.89 |
| Frozen 4,295 holdout | 29.01 | 53.02 | 33.29 | 29.38 |

The held-out official gain is +4.28 percentage points, but clean v3 trails clean base by 23.63 points. Spatial regression persists after cleaning. Instruction sensitivity is observed; metadata reliance and catastrophic forgetting are hypotheses, not established mechanisms. Per-task CSVs support the rates; missing timings/instruction text have not been invented.

## Historical timing boundary

A30, batch 1, fixed real LIBERO+ Spatial/0 post-settle observation, 20 warmups and 100 synchronized fresh queries. End-to-end includes preprocessing, inference, postprocessing and host action transfer; excludes the simulator. Per-action time is query latency divided by 10. This differs from the shared plan's OFT observation and 50 warmups / 300 queries. All reported 958 MiB peaks are **PyTorch reserved memory**, not process/device peaks.

The model-only timing used the locally modified LeRobot snapshot. No complete `git format-patch` series has yet been reconstructed; the public contribution does not claim a clean upstream-only container rebuild. The snapshot's upstream commit and versions are recorded in `cluster/smolvla/environment.json`.

## Spatial recovery evidence

| Stage | Verified evidence |
|---|---|
| Clean Spatial collection | 349 successes: 310 train / 39 validation, all ten Spatial tasks covered; no infrastructure errors |
| Balanced replay data | 932 train / 123,184 frames; 122 validation / 15,955 frames; suite frame totals within 5% of Spatial |
| Base-start v4 | Training 2538431 completed 20,000 updates, batch 8, LR 1e-5; evaluation jobs 2540494–2540496 completed; final rates for this separate branch not supplied |
| Direct v3 continuation | Pilot 2540523 and full 2540524 completed; v3 step 400000 weights, fresh optimizer/scheduler, 20,000 additional updates on the same balanced data |
| Continuation evaluation | Jobs 2542992–2542994 completed; clean full 54.17%, clean holdout 48.85%, standard LIBERO 71.25% |

Spatial sources are 181 non-held-out Light-category IDs per recipe; lighting variation is discarded before applying new views/mild blur. This is variant-level separation. During replay conversion, successful older demonstration instructions are relabeled to canonical wording while images/states/actions remain unchanged. Coverage, prompt distribution, learning rate and training budget change together, so the repair is not an isolated ablation of Spatial coverage alone.

## Completed recovery and paired analysis

| Protocol | Base | V3 | Repaired V3 | Repair minus V3 | Repair minus base |
|---|---:|---:|---:|---:|---:|
| Clean full, 10,030 | 53.56 | 39.89 | 54.17 | +14.28 | +0.61 |
| Clean frozen holdout, 4,295 | 53.02 | 29.38 | 48.85 | +19.46 | -4.17 |
| Original LIBERO, 2,000 | 82.05 | 52.90 | 71.25 | +18.35 | -10.80 |

All four suites improve over V3. Spatial standard recovers 0.40% to 54.20%, below base 76.00%. Full clean Spatial recovers 1.42% to 39.88%, below base 55.62%. This is substantial partial recovery with residual retention losses. Repaired official-instruction and speed scores remain unmeasured.

Full-clean repaired versus base: 1,734 repair-only successes, 1,673 base-only, exact conditional McNemar p=0.304. Frozen clean: 640 repair-only, 819 base-only (p=3.08e-6). Standard: 184 repair-only, 400 base-only (p=2.23e-19). All six overall contrasts and Holm-adjusted values are in `results/smolvla/reports/paired_overall.csv`. These are nominal, single-seed conditional tests. Related variants and episodes share task families; no cluster-aware uncertainty or independent multi-seed replication is claimed. The frozen set was repeatedly inspected during development and is not a blind final test.

## Shared speed measurement

Completed job 2568735 uses NVIDIA A30, batch 1, the team's fixed OFT sample, 50 warmups and 300 fresh synchronized queries in separate processes. Base: median 197.06 ms/query, model-forward 191.38 ms, 19.71 ms/executed action. V3: 208.77 ms/query, model-forward 201.58 ms, 20.88 ms/action. Both peak at approximately 0.899 GiB PyTorch allocated. Each query generates 50 actions but executes 10. Model-forward includes vision/language/action generation, not LLM-only. Peak allocated is not process/device memory. The fixed timing sample is not a manipulation rollout.

Reproduction code is in `cluster/smolvla/shared_speed/`, with a pinned, hash-checked upstream sample downloader. Raw timing samples and metadata are in the corresponding result cards. The shared timing procedure does not align the differing success-evaluation simulator stacks.

## Publication and evidence limits

Original wide CSVs, redacted raw speed JSON and checksums are in `results/smolvla/source_exports/`. Normalized per-task files include the six earlier completed runs and repaired clean evaluations. Original standard LIBERO per-episode outcomes are available for base, V3 and repaired V3. Noise, unaltered-demo control and teacher-V2 standard results remain aggregate-only. Missing episode timings or instruction strings remain blank. The analysis checks exact plan keys, unique records, full/holdout agreement, reported rates and timing counts.

The public contribution contains source and evaluation outcomes only. Slides, videos, weights, training episodes, logs, secrets and private archives are shared separately or excluded. Per `THIRD_PARTY_LICENSES.md`, teacher-demonstration-trained weight distributions must comply with applicable Gemma terms and notices. No weights are published here. A complete patch series for the captured locally modified LeRobot image remains unavailable; the pinned environment and external-image reproduction boundary are explicit.

Next: retention-focused validation across all suites, a fresh confirmation protocol or additional seeds, and profiling the repaired checkpoint before making speed claims. No dedicated lighting/background augmentation is planned.
