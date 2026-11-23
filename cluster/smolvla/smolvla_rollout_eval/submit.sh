#!/usr/bin/env bash
# Submit the same 4,295 LIBERO+ rollouts for all three checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ACCOUNT=${1:-cs595}
test ! -e submission_ids.txt || { echo 'submission_ids.txt exists; inspect existing jobs first' >&2; exit 2; }
test -f "$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
test -f "$HOME/smolvla_turing/eval_kit/eval_tasks.py"
test -f "$HOME/smolvla_turing/eval_kit/smolvla_adapter.py"
test -f "$HOME/smolvla_turing/eval_kit/kitconfig.py"

source "$HOME/smolvla_turing/eval_kit/config.sh"
NOISE="$LP_DIR/training/smol_noise_noise_2405981/output/checkpoints/last/pretrained_model"
CLEAN="$LP_DIR/training/smol_noise_clean_2405982/output/checkpoints/last/pretrained_model"
BASE="$SMOLVLA_MODEL"
for model in "$BASE" "$NOISE" "$CLEAN"; do
  test -f "$model/model.safetensors" || { echo "Missing checkpoint: $model" >&2; exit 2; }
done

rows=$(awk 'BEGIN{FS="\t"} !/^#/ && NF>=3 {n++} END{print n+0}' "$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv")
test "$rows" -eq 4295 || { echo "Expected 4295 plan rows, found $rows" >&2; exit 2; }
mkdir -p results
printf 'base=%s\nnoise=%s\nclean=%s\n' "$BASE" "$NOISE" "$CLEAN" > models.txt
sha256sum "$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv" > plan_sha256.txt

for arm in base noise clean; do
  job=$(sbatch --parsable --account="$ACCOUNT" --export="ALL,ARM=$arm" run_eval.sbatch)
  printf '%s=%s\n' "$arm" "$job" | tee -a submission_ids.txt
done
squeue --me
