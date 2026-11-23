#!/usr/bin/env bash
#SBATCH --job-name=smol-v4-verify
#SBATCH --partition=academic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=v4_verify_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
export SMOLVLA_MODEL="$LP_DIR/training/smol_spatial_balanced_v4_full_2538431/output/checkpoints/020000/pretrained_model"
test -s "$SMOLVLA_MODEL/model.safetensors"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV
"$PY" "$HOME/smolvla_turing/eval_kit/verify_smolvla.py" "$PWD/results/verification_$SLURM_JOB_ID"
