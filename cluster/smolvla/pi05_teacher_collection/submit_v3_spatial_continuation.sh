#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
test ! -e v3_spatial_continuation_jobs.txt || { cat v3_spatial_continuation_jobs.txt; echo 'Submission already recorded' >&2; exit 2; }
source "$HOME/smolvla_turing/eval_kit/config.sh"
MODEL="$LP_DIR/training/smol_combined_full_fixed_2473800/output/checkpoints/400000/pretrained_model"
test -s "$MODEL/model.safetensors"
python3 balanced_spatial_v4.py verify --paired "$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv" \
  --dataset-base "$LP_DIR/datasets"
bash -n train_v3_spatial_continuation.sbatch
P=$(sbatch --parsable --account="$ACCOUNT" --export=ALL,TRAIN_MODE=pilot train_v3_spatial_continuation.sbatch)
printf 'Pilot=%s\n' "$P" | tee v3_spatial_continuation_jobs.txt
F=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$P" \
  --export=ALL,TRAIN_MODE=full train_v3_spatial_continuation.sbatch)
printf 'Full=%s\n' "$F" | tee -a v3_spatial_continuation_jobs.txt
printf 'Starting checkpoint=%s\n' "$MODEL"
squeue --me
