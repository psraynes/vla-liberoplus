#!/usr/bin/env bash
#SBATCH --job-name=pi05-convert
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=convert_%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MODE=${CONVERT_SET:?Set CONVERT_SET=pilot or full}
case "$MODE" in pilot|full) ;; *) echo 'Invalid CONVERT_SET' >&2; exit 2 ;; esac

source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer
RUN="$LP_DIR/teacher_pi05/collection_${MODE}_v2"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
if [[ "$MODE" == pilot ]]; then
  PLAN="$PWD/teacher_collection_plan_pilot.tsv"
else
  PLAN="$PWD/teacher_collection_plan.tsv"
fi
DATASETS="$LP_DIR/datasets"
mkdir -p "$DATASETS"

python3 status.py --plan "$PLAN" --paired-plan "$PAIRED" --run "$RUN"
test -f "$LP_DIR/liberoplus.sif"
for SPLIT in train validation; do
  OUT="$DATASETS/pi05_camera_sensor_${MODE}_${SPLIT}_v2"
  if [[ -e "$OUT" ]]; then
    echo "Dataset path exists; refusing to overwrite: $OUT" >&2
    exit 2
  fi
  echo "[$(date -Is)] converting $MODE/$SPLIT to $OUT"
  apptainer exec -B /scratch "$LP_DIR/liberoplus.sif" python "$PWD/convert_teacher_episodes.py" \
    --attempt-root "$RUN" --paired-plan "$PAIRED" --split "$SPLIT" \
    --dataset-root "$OUT" \
    --repo-id "capstone/pi05_camera_sensor_${MODE}_${SPLIT}_v2"
  test -s "$OUT/meta/info.json"
  test -s "$OUT/collection_manifest.jsonl"
done
echo "[$(date -Is)] conversion complete: $MODE"
