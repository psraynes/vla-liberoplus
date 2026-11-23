#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
test ! -e submission_ids.txt || { cat submission_ids.txt; exit 2; }
source "$HOME/smolvla_turing/eval_kit/config.sh"
MODEL="$LP_DIR/training/smol_v3_spatial_continuation_full_2540524/output/checkpoints/020000/pretrained_model"
test -s "$MODEL/model.safetensors" || { echo "Missing final checkpoint: $MODEL" >&2; exit 2; }
echo "Evaluating v3-plus-Spatial: $MODEL"
sha256sum -c kit.sha256
bash -n verify_v3_repair.sbatch run_v3_repair.sbatch standard_v3_repair.sbatch
V=$(sbatch --parsable --account="$ACCOUNT" verify_v3_repair.sbatch)
echo "Verify=$V" | tee submission_ids.txt
E=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$V" --array=0-1%2 run_v3_repair.sbatch)
echo "CleanArray=$E" | tee -a submission_ids.txt
S=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$V" standard_v3_repair.sbatch)
echo "Standard=$S" | tee -a submission_ids.txt
squeue --me
