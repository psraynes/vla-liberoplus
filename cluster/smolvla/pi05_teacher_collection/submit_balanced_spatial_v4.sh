#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
test ! -e balanced_spatial_v4_submission.txt || { cat balanced_spatial_v4_submission.txt; exit 2; }
bash -n convert_balanced_spatial_v4.sbatch train_balanced_spatial_v4.sbatch
C=$(sbatch --parsable --account="$ACCOUNT" convert_balanced_spatial_v4.sbatch)
printf 'Convert=%s\n' "$C" | tee balanced_spatial_v4_submission.txt
P=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$C" \
  --export=ALL,TRAIN_MODE=pilot train_balanced_spatial_v4.sbatch)
printf 'Pilot=%s\n' "$P" | tee -a balanced_spatial_v4_submission.txt
squeue --me
