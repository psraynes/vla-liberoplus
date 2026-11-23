#!/usr/bin/env bash
#SBATCH --job-name=smol-clean-input
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=clean_%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
W=${SLURM_ARRAY_TASK_ID:?Submit as a two-worker array}
case "$W" in 0|1) ;; *) exit 2 ;; esac
ARM=${ARM:?Set ARM=base or ARM=combined_v3}
case "$ARM" in base|combined_v3) ;; *) exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"

sha256sum -c frozen.sha256
CLASSIFICATION="$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"
echo 'faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6  '"$CLASSIFICATION" | sha256sum -c -
PAIRED="$HOME/smolvla_turing/smolvla_rollout_eval/results/$ARM"
test -s "$PAIRED/run_manifest.txt"
test "$(sha256sum "$PAIRED/eval_tasks.py" | cut -d' ' -f1)" = "$(cat original_evaluator_sha256.txt)"
MODEL=$(sed -n 's/^model=//p' "$PAIRED/run_manifest.txt")
test -s "$MODEL/model.safetensors"
for expected in "arm=$ARM" "model=$MODEL" 'seed=7' 'action_steps=10'; do
  grep -Fxq "$expected" "$PAIRED/run_manifest.txt"
done
if grep -q '^save_video=' "$PAIRED/run_manifest.txt"; then
  grep -Fxq 'save_video=0' "$PAIRED/run_manifest.txt"
fi

RUN="$PWD/results/${ARM}_clean_w$W"
PLAN="$PWD/frozen/affected_worker$W.tsv"
mkdir -p "$RUN"
cp -n eval_tasks.py "$RUN/eval_tasks.py"
cmp -s eval_tasks.py "$RUN/eval_tasks.py"
cp -n frozen/instruction_map.csv "$RUN/instruction_map.csv"
cmp -s frozen/instruction_map.csv "$RUN/instruction_map.csv"
cp -n "$PLAN" "$RUN/plan.tsv"
cmp -s "$PLAN" "$RUN/plan.tsv"
for name in smolvla_adapter.py kitconfig.py; do
  cp -n "$PAIRED/$name" "$RUN/$name"
  cmp -s "$PAIRED/$name" "$RUN/$name"
done
printf 'source "$HOME/smolvla_turing/eval_kit/config.sh"\n' > "$RUN/config.sh"
if [[ ! -e "$RUN/run_manifest.txt" ]]; then
  printf 'arm=%s_clean_w%s\nmodel=%s\nseed=7\naction_steps=10\nsave_video=0\ninstruction_mode=clean\n' \
    "$ARM" "$W" "$MODEL" > "$RUN/run_manifest.txt"
fi
for expected in "arm=${ARM}_clean_w$W" "model=$MODEL" 'seed=7' 'action_steps=10' 'save_video=0' 'instruction_mode=clean'; do
  grep -Fxq "$expected" "$RUN/run_manifest.txt"
done
sha256sum "$RUN"/eval_tasks.py "$RUN"/smolvla_adapter.py "$RUN"/kitconfig.py \
  "$RUN"/instruction_map.csv "$RUN"/plan.tsv > "$RUN/code_and_plan_sha256.txt"

export SMOLVLA_MODEL="$MODEL" SAVE_VIDEO=0 SEED=7 SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV
for pass in $(seq 1 10); do
  echo "[$(date -Is)] $ARM clean worker $W pass $pass"
  set +e
  "$PY" "$RUN/eval_tasks.py"
  rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 2 ]]; then exit "$rc"; fi
  if python3 status_clean.py "$RUN"; then
    echo "[$(date -Is)] $ARM clean worker $W complete"
    exit 0
  fi
done
echo "$ARM clean worker $W incomplete after ten passes" >&2
exit 1
