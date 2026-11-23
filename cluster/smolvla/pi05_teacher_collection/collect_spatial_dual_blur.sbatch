#!/usr/bin/env bash
#SBATCH --job-name=pi05-spdual
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=collect_spatial_dual_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SET=${EXPAND_SET:?Set EXPAND_SET=pilot or full}
RECIPE=new_view_dual_blur
case "$SET" in pilot|full) ;; *) echo 'Invalid EXPAND_SET' >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"
export PI05_MODEL="$LP_DIR/models/pi05_libero_finetuned_v044"
test -s "$PI05_MODEL/model.safetensors"
REVISION=$(cat "$PI05_MODEL/pinned_revision.txt")
test "${#REVISION}" -eq 40
PLAN="$PWD/spatial_dual_blur_plans/spatial_${SET}_${RECIPE}.tsv"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
RUN="$LP_DIR/teacher_pi05/spatial_expansion_v3_${SET}_${RECIPE}"
test -s "$PLAN"
test -s "$PAIRED"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_HUB_OFFLINE=0 APPTAINERENV_TRANSFORMERS_OFFLINE=0
export APPTAINERENV_PI05_MODEL="$PI05_MODEL"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
"$PY" check_access.py
for pass in $(seq 1 10); do
  echo "[$(date -Is)] $SET/$RECIPE pass $pass"
  set +e
  "$PY" collect_spatial_dual_blur.py --plan "$PLAN" --paired-plan "$PAIRED" \
    --teacher-checkpoint "lerobot/pi05_libero_finetuned_v044@$REVISION" \
    --output "$RUN" --seed 7
  rc=$?
  set -e
  if [[ "$rc" == 0 ]]; then break; fi
  if [[ "$rc" != 2 ]]; then exit "$rc"; fi
  if [[ "$pass" == 10 ]]; then echo 'Too many file-descriptor restarts' >&2; exit 1; fi
done
python3 status.py --plan "$PLAN" --paired-plan "$PAIRED" --run "$RUN"
