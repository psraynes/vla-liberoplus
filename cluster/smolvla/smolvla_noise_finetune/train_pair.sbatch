#!/bin/bash
# Full controlled comparison, one arm per job. Only runs after a successful pilot.
#SBATCH --job-name=smol-noise-train
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=noise_train_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer

ARM=${ARM:?Expected ARM=noise or ARM=clean}
PILOT_JOB_ID=${PILOT_JOB_ID:?Expected the successful pilot job ID}
STEPS=${STEPS:-4000}
case "$ARM" in noise|clean) ;; *) echo "Invalid ARM=$ARM" >&2; exit 2 ;; esac
case "$STEPS" in ''|*[!0-9]*) echo "STEPS must be a positive integer" >&2; exit 2 ;; esac
test "$STEPS" -gt 0

source "$HOME/smolvla_turing/eval_kit/config.sh"
SIF="$LP_DIR/liberoplus.sif"
MODEL="$SMOLVLA_MODEL"
PILOT="$LP_DIR/training/smol_noise_pilot_${PILOT_JOB_ID}"
RUN="$LP_DIR/training/smol_noise_${ARM}_${SLURM_JOB_ID}"
test -f "$SIF"
test -f "$MODEL/model.safetensors"
test -f "$PILOT/dataset_revision.txt"
test -f "$PILOT/noise_recipe.json"
REVISION=$(cat "$PILOT/dataset_revision.txt")
mkdir -p "$RUN"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_LEROBOT_HOME="$LP_DIR/hf/lerobot"
cp noise_recipe.json "$RUN/noise_recipe.json"
sha256sum train_noise.py split_compat.py check_noise.py check_dataset.py noise_recipe.json > "$RUN/code_sha256.txt"
printf 'arm=%s\nstudent=%s\nsource_dataset=lerobot/libero@%s\nsteps=%s\nseed=42\npilot_job=%s\n' \
  "$ARM" "$MODEL" "$REVISION" "$STEPS" "$PILOT_JOB_ID" > "$RUN/run_manifest.txt"

apptainer exec -B /scratch "$SIF" python check_dataset.py --revision "$REVISION"
if [ "$ARM" = noise ]; then
  apptainer exec --nv -B /scratch "$SIF" python check_noise.py
  TRANSFORMS=$(python3 -c 'import json; print(json.dumps(json.load(open("noise_recipe.json"))["transforms"], separators=(",", ":")))')
  export SMOLVLA_NOISE_RECIPE="$PWD/noise_recipe.json"
  export APPTAINERENV_SMOLVLA_NOISE_RECIPE="$SMOLVLA_NOISE_RECIPE"
  AUG_ARGS=(--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=1 --dataset.image_transforms.tfs="$TRANSFORMS")
else
  unset SMOLVLA_NOISE_RECIPE APPTAINERENV_SMOLVLA_NOISE_RECIPE
  AUG_ARGS=(--dataset.image_transforms.enable=false)
fi

apptainer exec --nv -B /scratch "$SIF" python train_noise.py \
  --policy.path="$MODEL" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision="$REVISION" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.1 \
  "${AUG_ARGS[@]}" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --seed=42 \
  --batch_size=2 \
  --num_workers=4 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=500 \
  --eval_steps=500 \
  --max_eval_samples=400 \
  --env_eval_freq=0 \
  --log_freq=50 \
  --wandb.enable=false \
  --output_dir="$RUN/output" \
  --job_name="smol_noise_${ARM}_${SLURM_JOB_ID}"

find "$RUN/output/checkpoints" -name model.safetensors -print -quit | grep -q .
printf 'Training complete: %s\n' "$RUN"
