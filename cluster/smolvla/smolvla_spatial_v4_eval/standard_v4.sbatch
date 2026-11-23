#!/usr/bin/env bash
#SBATCH --job-name=smol-v4-standard
#SBATCH --partition=academic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=v4_standard_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
export SMOLVLA_MODEL="$LP_DIR/training/smol_spatial_balanced_v4_full_2538431/output/checkpoints/020000/pretrained_model"
test -s "$SMOLVLA_MODEL/model.safetensors"
cmp -s "$HOME/smolvla_turing/eval_kit/smolvla_adapter.py" "$HOME/smolvla_turing/smolvla_rollout_eval/results/base/smolvla_adapter.py"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SMOLVLA_ACTION_STEPS=10
unset SMOLVLA_PROFILE_CSV
MEASURE="$HOME/smolvla_turing/smolvla_measurements"
exec bash "$MEASURE/standard-exec.sh" "$MEASURE/standard_eval.py" --output "$PWD/results/standard_v4" --episodes 50
