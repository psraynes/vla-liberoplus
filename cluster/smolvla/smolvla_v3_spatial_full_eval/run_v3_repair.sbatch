#!/usr/bin/env bash
#SBATCH --job-name=smol-v3_repair-clean
#SBATCH --partition=academic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=v3_repair_clean_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
W=${SLURM_ARRAY_TASK_ID:?}
case "$W" in 0|1) ;; *) exit 2 ;; esac
sha256sum -c kit.sha256
MODEL="$LP_DIR/training/smol_v3_spatial_continuation_full_2540524/output/checkpoints/020000/pretrained_model"
test -s "$MODEL/model.safetensors"
BASE="$HOME/smolvla_turing/smolvla_rollout_eval/results/base"
MAP="$HOME/smolvla_turing/smolvla_clean_instruction_eval/frozen/instruction_map.csv"
RUN="$PWD/results/clean_v3_repair_w$W"
mkdir -p "$RUN"
for NAME in smolvla_adapter.py kitconfig.py; do
  cp -n "$BASE/$NAME" "$RUN/$NAME"; cmp -s "$BASE/$NAME" "$RUN/$NAME"
done
for NAME in eval_tasks.py; do cp -n "$NAME" "$RUN/$NAME"; cmp -s "$NAME" "$RUN/$NAME"; done
cp -n "$MAP" "$RUN/instruction_map.csv"; cmp -s "$MAP" "$RUN/instruction_map.csv"
cp -n "worker$W.tsv" "$RUN/plan.tsv"; cmp -s "worker$W.tsv" "$RUN/plan.tsv"
printf 'source "$HOME/smolvla_turing/eval_kit/config.sh"\n' > "$RUN/config.sh"
if [[ ! -e "$RUN/run_manifest.txt" ]]; then
 printf 'arm=spatial_balanced_v3_repair\nmodel=%s\nseed=7\naction_steps=10\ninstruction_mode=clean\n' "$MODEL" > "$RUN/run_manifest.txt"
fi
grep -Fxq "model=$MODEL" "$RUN/run_manifest.txt"
sha256sum "$RUN"/*.py "$RUN/plan.tsv" "$RUN/instruction_map.csv" > "$RUN/code_and_plan_sha256.txt"
export SMOLVLA_MODEL="$MODEL" SEED=7 SAVE_VIDEO=0 SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV
for pass in $(seq 1 10); do
 set +e
 "$PY" "$RUN/eval_tasks.py"
 rc=$?
 set -e
 if [[ "$rc" != 0 && "$rc" != 2 ]]; then exit "$rc"; fi
 if python3 status_clean.py "$RUN"; then exit 0; fi
done
exit 1
