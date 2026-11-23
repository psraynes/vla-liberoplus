# Shared SmolVLA speed benchmark

Completed measurements: `results/smolvla/shared_speed_base/` and `shared_speed_combined_v3/`.

Use the matching recorded image and frozen evaluation adapters described in `../README.md`. Copy this directory to your provisioned project root as `smolvla_shared_speed`, set `SMOLVLA_PROJECT_ROOT` when needed, and inspect checkpoint paths in `run.sbatch`. Run `bash submit.sh <account>` from that folder. The submission requests one A30 and runs base then V3 sequentially. No job is launched by downloading this code.

`prepare_sample.py` fetches the pinned MIT-licensed OFT observation and checks its exact SHA-256 before restricted decoding. It creates a local NPZ fixture (not committed), consumed with `allow_pickle=False`. Internet access is needed for this small fixture unless prepared in advance. Existing matching fixture files are verified by the preparer.

Each measurement uses 50 warmups and 300 fresh queries, CUDA synchronization, batch 1, and ten executed actions per query. End-to-end includes preprocess, policy, postprocess and host transfer, excluding the simulator. Generation produces 50 actions; amortized latency divides by ten executed actions. The script checks that each query actually generates a new chunk. Model timing includes vision/language/action generation, not LLM-only. GPU memory is PyTorch allocated, not full process/device memory. Do not mix these medians with older 100-query means or reserved-memory peaks.

The sample's MIT license is retained alongside the code. No observations, datasets or model weights are committed. Shared timing does not make success evaluation settings identical across model families.
