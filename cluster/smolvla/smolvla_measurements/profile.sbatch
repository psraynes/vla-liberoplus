#!/usr/bin/env bash
#SBATCH --job-name=smol-profile
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=profile_%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
ARM=${ARM:?Set ARM=base,noise,clean,teacher_v2,combined_v3}
case "$ARM" in
  base|noise|clean) MODEL=$(sed -n "s/^${ARM}=//p" "$HOME/smolvla_turing/smolvla_rollout_eval/models.txt") ;;
  teacher_v2) MODEL="$LP_DIR/training/smol_teacher_full_2448430/output/checkpoints/100000/pretrained_model" ;;
  combined_v3) MODEL="$LP_DIR/training/smol_combined_full_fixed_2473800/output/checkpoints/400000/pretrained_model" ;;
  *) echo "Unknown ARM=$ARM" >&2; exit 2 ;;
esac
test -f "$MODEL/model.safetensors"
export SMOLVLA_MODEL="$MODEL" SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset SMOLVLA_PROFILE_CSV
"$PY" profile_smolvla.py --output "$PWD/results/profile_${ARM}_${SLURM_JOB_ID}.json"
