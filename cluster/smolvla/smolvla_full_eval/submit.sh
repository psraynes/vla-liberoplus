#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
ARM=${2:-base}
case "$ARM" in base|combined_v3) ;; *) echo 'Use base or combined_v3' >&2; exit 2 ;; esac
SUBMISSION="submission_${ARM}.txt"

if [[ -e "$SUBMISSION" ]]; then
  echo "$SUBMISSION already exists; inspect the current job before resubmitting" >&2
  cat "$SUBMISSION"
  exit 2
fi
if [[ "$ARM" == combined_v3 && -e submission_id.txt ]]; then
  echo 'An earlier combined-v3 extension submission exists; inspect it before starting another' >&2
  cat submission_id.txt
  exit 2
fi
shopt -s nullglob
existing=(results/"${ARM}"_remaining_w*/results*.jsonl)
if ((${#existing[@]})); then
  echo "Existing $ARM extension results found; inspect/resume the current run" >&2
  exit 2
fi
sha256sum -c plans.sha256
PAIRED="$HOME/smolvla_turing/smolvla_rollout_eval/results/$ARM"
test -s "$PAIRED/run_manifest.txt"
test "$(sha256sum "$PAIRED/plan.tsv" | cut -d' ' -f1)" = 123a6e5979e3eda29f601383236515f3527f653463fd1f91437f4ce7231af98d
python3 "$HOME/smolvla_turing/smolvla_rollout_eval/status.py" "$PAIRED"
bash -n run_remaining.sbatch

JOB=$(sbatch --parsable --account="$ACCOUNT" --array=0-1%2 --export="ALL,ARM=$ARM" run_remaining.sbatch)
printf '%sRemainingArray=%s\n' "$ARM" "$JOB" | tee "$SUBMISSION"
squeue -j "$JOB"
