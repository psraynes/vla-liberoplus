#!/bin/bash
# Submit exactly one pilot and two after-success full jobs; never overwrite results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ACCOUNT=${1:-cs595}
STEPS=${STEPS:-4000}
case "$STEPS" in ''|*[!0-9]*) echo "STEPS must be a positive integer" >&2; exit 2 ;; esac
test "$STEPS" -gt 0
test ! -e submission_ids.txt || { echo "submission_ids.txt already exists; inspect it before submitting again" >&2; exit 2; }

PILOT=$(sbatch --parsable --account="$ACCOUNT" --export=ALL pilot.sbatch)
printf 'Pilot=%s\n' "$PILOT" > submission_ids.txt
NOISE=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$PILOT" \
  --export="ALL,ARM=noise,PILOT_JOB_ID=$PILOT,STEPS=$STEPS" train_pair.sbatch)
printf 'Noise=%s\n' "$NOISE" >> submission_ids.txt
CLEAN=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$PILOT" \
  --export="ALL,ARM=clean,PILOT_JOB_ID=$PILOT,STEPS=$STEPS" train_pair.sbatch)
printf 'Clean=%s\n' "$CLEAN" >> submission_ids.txt
NOISE_VERIFY=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$NOISE" \
  --export="ALL,ARM=noise,TRAIN_JOB_ID=$NOISE" verify_checkpoint.sbatch)
printf 'NoiseVerify=%s\n' "$NOISE_VERIFY" >> submission_ids.txt
CLEAN_VERIFY=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$CLEAN" \
  --export="ALL,ARM=clean,TRAIN_JOB_ID=$CLEAN" verify_checkpoint.sbatch)
printf 'CleanVerify=%s\n' "$CLEAN_VERIFY" >> submission_ids.txt
printf 'Account=%s\nSteps=%s\n' "$ACCOUNT" "$STEPS" >> submission_ids.txt
cat submission_ids.txt
squeue --me
