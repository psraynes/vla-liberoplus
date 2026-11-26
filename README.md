# vla-liberoplus

Robustness of compact vision-language-action (VLA) robot policies on
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), with standard [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
as a regression check and [OpenVLA-OFT](https://github.com/moojink/openvla-oft) as the large-model reference. Everything
ran on WPI's Turing cluster (NVIDIA A30 GPUs).

A **VLA** looks at camera images, reads a typed instruction, and outputs robot arm movements. **LIBERO-Plus** takes the
40 standard LIBERO tasks and generates 10,030 perturbed variants: camera, robot start pose, wording, lighting,
background, image noise and object layout.

## Main results
| Model | Params | LIBERO-Plus (official) | LIBERO-Plus (clean instructions) | Standard LIBERO | Per executed action (A30) |
|---|---|---|---|---|---|
| OpenVLA-OFT (LIBERO-trained) | 7.5B | 68.7 | +3.0 on 1,000 tasks | 96.1 | 17.5 ms |
| OpenVLA-OFT+ (LIBERO-Plus-trained) | 7.5B | 80.5 | 80.5 (+0.1 on the affected tasks) | 96.6 | 17.9 ms |
| TurboVLA (released) | 216M | 45.5 | +15.1 on 1,000 tasks | 93.8 | ≈ 2.6 ms |
| **TurboVLA fine-tuned on LIBERO-Plus (ours)** | 216M | **68.9** | **≈ 82.1** | **96.5** | **2.56 ms** |

1. **Fine-tuning TurboVLA works.** One epoch on the LIBERO-Plus training mix (3 h 51 min on 2 A30s) lifts it from 45.5
   to 68.9 on LIBERO-Plus. Standard LIBERO improves too, 93.8 → 96.5.
2. **The benchmark leaks file-name codes into the instructions.** 85 % of LIBERO-Plus instructions carry codes such
   as "view 317 15 100 0 0 initstate 0". TurboVLA's short-context text encoder is confused by them; with the parent
   task's clean instruction it reaches ≈ 82.1 %.
3. **TurboVLA is fast.** It is 7× faster per executed action than OpenVLA-OFT+ and needs 0.43 GB of GPU memory instead
   of about 15 GB.

Details, sources and caveats: [docs/findings/RESULTS.md](docs/findings/RESULTS.md),
[docs/findings/GAPS.md](docs/findings/GAPS.md).

## Layout
To add work to the repo, see [CONTRIBUTING.md](CONTRIBUTING.md).

| Folder | Contents |
|---|---|
| [`docs/findings/`](docs/findings) | results with sources, methods and glossary, data map, timeline, caveats |
| [`docs/plans/`](docs/plans) | plans and logs of the robustness tests and reference runs |
| [`docs/oft_reference.md`](docs/oft_reference.md) | OpenVLA-OFT/OFT+ setup, baselines and latency |
| [`docs/turing/`](docs/turing) | Turing rules for coding agents, first-session checklist |
| [`docs/data/`](docs/data) | headline numbers as CSV, for plotting |
| [`cluster/`](cluster) | containers, eval kit, fine-tuning pipeline, OFT setup and jobs, teammates' starter kit |
| [`experiments/`](experiments) | robustness tests (instruction leak, OFT+ gap-fill runs), video reruns, report scripts |
| [`results/`](results) | per-task results of every run ([format](results/README.md)) |
| [`patches/`](patches) | our changes to TurboVLA and OpenVLA-OFT |
| [`third_party/VERSIONS.md`](third_party/VERSIONS.md) | pinned upstream commits, model revisions and data |
| [`report/`](report) | final report in IEEE conference format (LaTeX; build instructions inside) |

## Running on Turing
Turing is a shared cluster. Read [docs/turing/AGENTS.md](docs/turing/AGENTS.md) and the
[Turing docs](https://docs.turing.wpi.edu/llms.txt) first. All heavy work goes through SLURM with a `--time` limit.

Scripts expect their folder at a fixed place in your Turing home directory:

| Repo folder | Copy to |
|---|---|
| `cluster/liberoplus_container/` (+ `patches/TurboVLA/*.patch` → `patches/`) | `~/liberoplus_container/` |
| `cluster/oft_container/` (+ `patches/openvla-oft/*.patch` → `patches/`) | `~/oft_container/` |
| `cluster/eval_kit/` | `~/turing_eval_kit/` |
| `cluster/finetune_lp/` | `~/finetune_lp/` |
| `experiments/robustness/` | `~/robustness_tests/` |

**Setup order:**
1. **LIBERO-Plus + TurboVLA:** `setup_code.sh`, then `build_image.sbatch`, `make_assets.sbatch`,
   `download_weights.sbatch` (see `cluster/liberoplus_container/README.md`).
2. **Evaluation:** `cluster/eval_kit/README.md`.
3. **TurboVLA fine-tuning:** `cluster/finetune_lp/README.md`.
4. **OpenVLA-OFT:** `setup_oft_code.sh` and `build_oft_image.sbatch`, then `download_oftplus.sbatch` or
   `download_oft_libero.sbatch`, then `setup_oft_eval.sh`.

**Job emails:** SLURM emails the submitting user. To use another address, set `MAIL_USER`; the submit scripts pass it
as `--mail-user`.

## Protocol in one paragraph
- **LIBERO-Plus runs:** all 10,030 tasks, one trial each, in one fixed task order for every model (`plan.tsv`).
- **Episodes:** initial state 0, 10 settling steps, step limits 220/280/300/520 (spatial/object/goal/10); each
  predicted action chunk is executed open loop.
- **Env seed:** 7 for TurboVLA, 0 for OpenVLA-OFT (OFT's own protocol).
- **Instructions:** official, unless a run is marked "clean".
- **Comparisons:** paired on the same tasks with McNemar tests (`cluster/finetune_lp/compare_evals.py`).
- **Standard LIBERO:** 4 suites × 10 tasks × 50 trials, with each model's own evaluator.
- **Full write-up:** [docs/findings/METHODS.md](docs/findings/METHODS.md).

## License
- **This repo:** Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
- **Third-party terms:** model weights and upstream code used by the scripts have their own terms. In particular,
  TurboVLA checkpoints fall under the DINOv3 License, and OpenVLA-OFT weights under the Llama 2 Community License.
  See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **Acknowledgement:** results obtained with TurboVLA use DINOv3 (Meta).
