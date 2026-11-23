#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
ARM=${2:-base}
case "$ARM" in base|combined_v3) ;; *) echo 'Use base or combined_v3' >&2; exit 2 ;; esac
sha256sum -c frozen.sha256
bash -n run_clean.sbatch
SUBMISSION="submission_${ARM}_clean.txt"
if [[ -e "$SUBMISSION" ]]; then
  cat "$SUBMISSION"
  echo 'Existing submission recorded; inspect it before any resubmission' >&2
  exit 2
fi
JOB=$(sbatch --parsable --account="$ACCOUNT" --array=0-1%2 \
  --export="ALL,ARM=$ARM" run_clean.sbatch)
printf '%sCleanArray=%s\n' "$ARM" "$JOB" | tee "$SUBMISSION"
squeue -j "$JOB"
