#!/usr/bin/env bash
#SBATCH --job-name=smol-standard
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=standard_%x_%j.log
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
EPISODES=${EPISODES:-50}
if [[ "$EPISODES" = 1 ]]; then
  # Four-suite smoke check: task 0, first state in each suite.
  exec bash standard-exec.sh standard_eval.py --output "$PWD/results/standard_pilot_$ARM" --episodes 1 --task-id 0
fi
if [[ "$EPISODES" != 50 ]]; then
  echo "Expected EPISODES=1 (pilot) or 50 (full standard benchmark)" >&2
  exit 2
fi
exec bash standard-exec.sh standard_eval.py --output "$PWD/results/standard_full_$ARM" --episodes 50
