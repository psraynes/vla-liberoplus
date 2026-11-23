#!/usr/bin/env bash
#SBATCH --job-name=spatial-v4-data
#SBATCH --partition=academic
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=balanced_spatial_convert_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
python3 balanced_spatial_v4.py prepare --raw "$LP_DIR/teacher_pi05" --paired "$PAIRED" \
  --instructions "$HOME/smolvla_turing/smolvla_clean_instruction_eval/frozen/instruction_map.csv"
for SPLIT in train validation; do
  apptainer exec -B /scratch "$LP_DIR/liberoplus.sif" python balanced_spatial_v4.py convert \
    --paired "$PAIRED" --split "$SPLIT" --output "$LP_DIR/datasets/spatial_balanced_clean_v4_$SPLIT"
done
python3 balanced_spatial_v4.py verify --paired "$PAIRED" --dataset-base "$LP_DIR/datasets"
