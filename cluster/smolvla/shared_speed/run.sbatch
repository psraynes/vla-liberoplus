#!/usr/bin/env bash
#SBATCH --job-name=smol-shared-speed
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=00:30:00
#SBATCH --output=shared_speed_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark folder}"
PROJECT=${SMOLVLA_PROJECT_ROOT:-$HOME/smolvla_turing}
source "$PROJECT/eval_kit/config.sh"
"$PY" prepare_sample.py --output-dir "$PWD"
export SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SMOLVLA_PROFILE_CSV
BASE_MODEL=${BASE_MODEL:-$HF_HUB_CACHE/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d}
V3_MODEL=${V3_MODEL:-$LP_DIR/training/smol_combined_full_fixed_2473800/output/checkpoints/400000/pretrained_model}
for MODEL in "$BASE_MODEL" "$V3_MODEL"; do
  test -f "$MODEL/model.safetensors" || { echo "Missing checkpoint: $MODEL" >&2; exit 2; }
done
for ARM in base combined_v3; do
  test -f "$PROJECT/smolvla_rollout_eval/results/$ARM/smolvla_adapter.py" || {
    echo "Missing frozen evaluation adapter for $ARM; stop and inspect the results folder" >&2; exit 2;
  }
done
RUN="$PWD/results/$SLURM_JOB_ID"
mkdir -p "$RUN"
printf 'job=%s\nnode=%s\nbase=%s\nv3=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$BASE_MODEL" "$V3_MODEL" > "$RUN/manifest.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate --format=csv > "$RUN/gpu_before.csv"
# Sequential fresh processes: only one resident checkpoint occupies the GPU.
for ARM in base combined_v3; do
  if [[ "$ARM" == base ]]; then MODEL=$BASE_MODEL; else MODEL=$V3_MODEL; fi
  export SMOLVLA_MODEL="$MODEL"
  echo "[$(date -Is)] benchmarking $ARM on $(hostname)"
  "$PY" "$PWD/benchmark_smolvla.py" --arm "$ARM" \
    --adapter-dir "$PROJECT/smolvla_rollout_eval/results/$ARM" \
    --sample "$PWD/oft_sample_observation.npz" \
    --warmup 50 --iters 300 --out "$RUN/${ARM}.json"
done
python3 report.py "$RUN"
printf 'Completed shared speed benchmark: %s\n' "$RUN"
