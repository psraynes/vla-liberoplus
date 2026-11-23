#!/usr/bin/env bash
#SBATCH --job-name=pi05-combine
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --output=convert_combined_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SPLIT=${COMBINE_SPLIT:?Set COMBINE_SPLIT=train or validation}
case "$SPLIT" in train|validation) ;; *) echo 'Invalid COMBINE_SPLIT' >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer
SCRIPT="$PWD/convert_combined_teacher_v3.py"
RAW="$LP_DIR/teacher_pi05"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
OUT="$LP_DIR/datasets/pi05_camera_sensor_combined_v3_${SPLIT}"
REPO="capstone/pi05_camera_sensor_combined_v3_${SPLIT}"
test -f "$LP_DIR/liberoplus.sif"
test ! -e "$OUT" || { echo "Dataset exists; refusing overwrite: $OUT" >&2; exit 2; }
python3 "$SCRIPT" --raw-base "$RAW" --paired-plan "$PAIRED" --audit-only
sha256sum "$SCRIPT" "$PAIRED" > "$PWD/combined_${SPLIT}_${SLURM_JOB_ID}_input_sha256.txt"
apptainer exec -B /scratch "$LP_DIR/liberoplus.sif" python "$SCRIPT" \
  --raw-base "$RAW" --paired-plan "$PAIRED" \
  --split "$SPLIT" --dataset-root "$OUT" --repo-id "$REPO"
test -s "$OUT/meta/info.json"
test -s "$OUT/collection_manifest.jsonl"
echo "Combined dataset complete: $OUT"
