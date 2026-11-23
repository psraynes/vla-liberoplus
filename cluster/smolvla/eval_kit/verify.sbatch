#!/bin/bash
#SBATCH --job-name=smol-verify
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=smol_verify_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./config.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV
nvidia-smi
"$PY" verify_smolvla.py "$LP_DIR/verification/$SLURM_JOB_ID"
