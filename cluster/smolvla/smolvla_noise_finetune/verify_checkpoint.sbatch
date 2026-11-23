#!/bin/bash
# Seven real LIBERO+ smoke rollouts for a newly trained checkpoint.
#SBATCH --job-name=smol-noise-check
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=noise_verify_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARM=${ARM:?Expected ARM=noise or ARM=clean}
TRAIN_JOB_ID=${TRAIN_JOB_ID:?Expected training job ID}
case "$ARM" in noise|clean) ;; *) echo "Invalid ARM=$ARM" >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"

TRAIN="$LP_DIR/training/smol_noise_${ARM}_${TRAIN_JOB_ID}"
export SMOLVLA_MODEL="$TRAIN/output/checkpoints/last/pretrained_model"
test -f "$SMOLVLA_MODEL/model.safetensors"
OUT="$LP_DIR/verification/smol_noise_${ARM}_${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
"$PY" "$HOME/smolvla_turing/eval_kit/verify_smolvla.py" "$OUT"
printf 'Checkpoint smoke result: %s/verification_report.json\n' "$OUT"
