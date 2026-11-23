#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ACCOUNT=${1:-cs595}
bash -n run.sbatch
if [[ -e submission.txt ]]; then
  cat submission.txt
  echo 'A submission is already recorded; inspect that job before submitting again.' >&2
  exit 1
fi
JOB=$(sbatch --parsable --account="$ACCOUNT" run.sbatch)
printf 'SharedSpeed=%s\nAccount=%s\n' "$JOB" "$ACCOUNT" | tee submission.txt
squeue -j "$JOB"
