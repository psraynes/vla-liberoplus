#!/usr/bin/env bash
#SBATCH --job-name=pi05-collect
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=collect_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SET=${COLLECT_SET:-pilot}
case "$SET" in pilot|full) ;; *) echo 'COLLECT_SET must be pilot or full' >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"
export PI05_MODEL="$LP_DIR/models/pi05_libero_finetuned_v044"
test -s "$PI05_MODEL/model.safetensors"
REVISION=$(cat "$PI05_MODEL/pinned_revision.txt")
test "${#REVISION}" -eq 40
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_HUB_OFFLINE=0 APPTAINERENV_TRANSFORMERS_OFFLINE=0
export APPTAINERENV_PI05_MODEL="$PI05_MODEL"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
if [[ "$SET" == pilot ]]; then
  PLAN="$PWD/teacher_collection_plan_pilot.tsv"
else
  PLAN="$PWD/teacher_collection_plan.tsv"
fi
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
RUN="$LP_DIR/teacher_pi05/collection_${SET}_v2"
[[ -s "$PLAN" ]] || { echo "Missing collection plan: $PLAN (run bash prepare_plan.sh first)" >&2; exit 2; }
[[ -s "$PAIRED" ]] || { echo "Missing held-out plan: $PAIRED" >&2; exit 2; }
"$PY" check_access.py
for pass in $(seq 1 10); do
  echo "[$(date -Is)] $SET collection pass $pass"
  set +e
  "$PY" collect_teacher_rollouts.py \
    --plan "$PLAN" --paired-plan "$PAIRED" \
    --adapter pi05_adapter:Adapter \
    --teacher-checkpoint "lerobot/pi05_libero_finetuned_v044@$REVISION" \
    --output "$RUN" --seed 7 --attempts-per-task 1
  rc=$?
  set -e
  if [[ "$rc" == 0 ]]; then break; fi
  if [[ "$rc" != 2 ]]; then exit "$rc"; fi
  if [[ "$pass" == 10 ]]; then echo 'Too many file-descriptor restarts' >&2; exit 1; fi
done
python3 status.py --plan "$PLAN" --paired-plan "$PAIRED" --run "$RUN"
