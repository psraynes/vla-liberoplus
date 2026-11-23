#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
test ! -e spatial_clean_v4_jobs.txt || { cat spatial_clean_v4_jobs.txt; echo 'Submission already recorded' >&2; exit 2; }
source "$HOME/smolvla_turing/eval_kit/config.sh"
test -s spatial_expansion_plans/spatial_full_new_view.tsv
test -s spatial_expansion_plans/spatial_full_new_view_blur.tsv
bash -n collect_spatial_clean_v4.sbatch
for recipe in new_view new_view_blur; do
  test ! -e "$LP_DIR/teacher_pi05/spatial_repair_v4_full_$recipe"
  test ! -e "$LP_DIR/teacher_pi05/spatial_repair_v4_pilot_$recipe"
done
touch spatial_clean_v4_jobs.txt
for recipe in new_view new_view_blur; do
  P=$(sbatch --parsable --account="$ACCOUNT" --time=02:00:00 \
    --export="ALL,EXPAND_SET=pilot,EXPAND_RECIPE=$recipe" collect_spatial_clean_v4.sbatch)
  printf '%sPilot=%s\n' "$recipe" "$P" | tee -a spatial_clean_v4_jobs.txt
  F=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$P" \
    --export="ALL,EXPAND_SET=full,EXPAND_RECIPE=$recipe" collect_spatial_clean_v4.sbatch)
  printf '%sFull=%s\n' "$recipe" "$F" | tee -a spatial_clean_v4_jobs.txt
done
squeue --me
