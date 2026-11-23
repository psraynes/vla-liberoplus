#!/usr/bin/env bash
#SBATCH --job-name=smol-paired-eval
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=rollout_%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ARM=${ARM:?Expected ARM=base, noise, or clean}
case "$ARM" in base|noise|clean) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"
MODEL=$(sed -n "s/^${ARM}=//p" models.txt)
test -f "$MODEL/model.safetensors"

RUN="$PWD/results/$ARM"
mkdir -p "$RUN"
for name in eval_tasks.py smolvla_adapter.py kitconfig.py; do
  cp -n "$HOME/smolvla_turing/eval_kit/$name" "$RUN/$name"
done
cp -n "$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv" "$RUN/plan.tsv"
cat > "$RUN/config.sh" <<'EOF'
source "$HOME/smolvla_turing/eval_kit/config.sh"
EOF
sha256sum "$RUN"/eval_tasks.py "$RUN"/smolvla_adapter.py "$RUN"/kitconfig.py "$RUN"/plan.tsv > "$RUN/code_and_plan_sha256.txt"
printf 'arm=%s\nmodel=%s\nseed=7\naction_steps=10\nsave_video=0\n' "$ARM" "$MODEL" > "$RUN/run_manifest.txt"

export SMOLVLA_MODEL="$MODEL"
export SAVE_VIDEO=0
export SEED=7
export SMOLVLA_ACTION_STEPS=10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4
unset SMOLVLA_PROFILE_CSV

# eval_tasks.py writes each result immediately. A new pass retries a first exception
# once and continues after an intentional file-descriptor restart.
for pass in $(seq 1 10); do
  echo "[$(date -Is)] $ARM pass $pass"
  set +e
  "$PY" "$RUN/eval_tasks.py"
  rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 2 ]]; then
    echo "Evaluator exited with $rc" >&2
    exit "$rc"
  fi
  if python3 status.py "$RUN"; then
    echo "[$(date -Is)] $ARM all 4295 rollouts complete"
    exit 0
  fi
done
echo 'Ten passes ended without completing the plan; inspect results for hangs.' >&2
exit 1
