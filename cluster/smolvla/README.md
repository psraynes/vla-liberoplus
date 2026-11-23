# SmolVLA work on Turing

Historical workspace layout: copy this folder's children into your own `~/smolvla_turing/`. Existing scripts intentionally retain that layout and use $HOME and /scratch/$USER; no personal email or token is embedded. Do not run submissions blindly: historical job IDs/checkpoints are inputs, and scripts may expect completed raw collections, plans and cached assets. Read each experiment card first. Heavy work uses Slurm with time limits.

Includes the SmolVLA adapter, established evaluator, standard benchmark/profile scripts, teacher collection/conversion, clean-instruction ablation, Spatial repair and continuation. Protocols and completed measurement data are under results/smolvla. Study scripts under experiments/smolvla validate exports without importing a simulator.

## Reproduction boundary
The recorded environment is in environment.json. It used an Apptainer image made from a captured, locally modified LeRobot snapshot. The source archive, full upstream modeling file, weights, assets and dataset episodes are NOT included. The exact local patch series has not been reconstructed as git format-patch, so this contribution does not claim a clean reproducible container rebuild. Use the already provisioned matching image or first reconstruct the patches separately, preserving upstream license headers. Scripts clone LIBERO+ at its pinned commit; no LIBERO+ source is vendored.

The original container/image is external to Git. Set LP_DIR/LP_CODE and model/cache roots for your account. Model files include saved processors and normalization statistics; keep them with their checkpoint. Training uses PyAV due to TorchCodec library loading failures in the recorded image.

The matching standalone adapter is also supplied at cluster/eval_kit/adapters/smolvla.py (class Adapter, image_size256, predict returns one7D action while its policy caches10). The historical model-specific workspace mirrors it as eval_kit/smolvla_adapter.py.

No new jobs are submitted by installing this tree. Submission scripts create jobs only when explicitly run.

Source kit manifests have been refreshed for the public copy after account-path sanitization. They verify these published files, not the byte identity of historical cluster deployments. Frozen result cards retain the original provenance where recorded.
