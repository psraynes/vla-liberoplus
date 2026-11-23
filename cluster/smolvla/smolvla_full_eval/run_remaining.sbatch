#!/usr/bin/env bash
#SBATCH --job-name=smol-full-extra
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=remaining_%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
W=${SLURM_ARRAY_TASK_ID:?Submit with --array=0-1}
case "$W" in 0|1) ;; *) echo 'Expected worker 0 or 1' >&2; exit 2 ;; esac
ARM=${ARM:?Submit with ARM=base or ARM=combined_v3}
case "$ARM" in base|combined_v3) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"

PAIRED="$HOME/smolvla_turing/smolvla_rollout_eval/results/$ARM"
test -s "$PAIRED/run_manifest.txt"
MODEL=$(sed -n 's/^model=//p' "$PAIRED/run_manifest.txt")
test -n "$MODEL"
RUN="$PWD/results/${ARM}_remaining_w$W"
PLAN="$PWD/plans/remaining_worker$W.tsv"
test -s "$MODEL/model.safetensors"
test -s "$PLAN"
test -s "$PAIRED/run_manifest.txt"
for expected in "arm=$ARM" "model=$MODEL" "seed=7" "action_steps=10" "save_video=0"; do
  grep -Fxq "$expected" "$PAIRED/run_manifest.txt" || {
    echo "Original paired run manifest differs: $expected" >&2; exit 2;
  }
done
mkdir -p "$RUN"
for name in eval_tasks.py smolvla_adapter.py kitconfig.py; do
  cp -n "$PAIRED/$name" "$RUN/$name"
  cmp -s "$PAIRED/$name" "$RUN/$name" || {
    echo "Existing $RUN/$name differs from paired run" >&2; exit 2;
  }
done
cp -n "$PLAN" "$RUN/plan.tsv"
cmp -s "$PLAN" "$RUN/plan.tsv" || {
  echo 'Existing run plan differs from frozen remaining plan' >&2; exit 2;
}
printf 'source "$HOME/smolvla_turing/eval_kit/config.sh"\n' > "$RUN/config.sh"
for expected in "arm=${ARM}_remaining_w$W" "model=$MODEL" "seed=7" "action_steps=10" "save_video=0"; do
  if [[ -e "$RUN/run_manifest.txt" ]]; then
    grep -Fxq "$expected" "$RUN/run_manifest.txt" || {
      echo "Existing run manifest differs: $expected" >&2; exit 2;
    }
  fi
done
if [[ ! -e "$RUN/run_manifest.txt" ]]; then
  printf 'arm=%s_remaining_w%s\nmodel=%s\nseed=7\naction_steps=10\nsave_video=0\n' "$ARM" "$W" "$MODEL" > "$RUN/run_manifest.txt"
fi
sha256sum "$RUN"/eval_tasks.py "$RUN"/smolvla_adapter.py "$RUN"/kitconfig.py "$RUN"/plan.tsv > "$RUN/code_and_plan_sha256.txt"

export SMOLVLA_MODEL="$MODEL" SAVE_VIDEO=0 SEED=7 SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV
for pass in $(seq 1 10); do
  echo "[$(date -Is)] remaining worker $W pass $pass"
  set +e
  "$PY" "$RUN/eval_tasks.py"
  rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 2 ]]; then exit "$rc"; fi
  if python3 status_remaining.py "$RUN"; then
    echo "[$(date -Is)] remaining worker $W complete"
    exit 0
  fi
done
echo "Worker $W incomplete after ten passes; inspect errors" >&2
exit 1
