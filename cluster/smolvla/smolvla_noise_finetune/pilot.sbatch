#!/bin/bash
# Ten optimizer steps and a checkpoint. Full jobs depend on this job succeeding.
#SBATCH --job-name=smol-noise-pilot
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=noise_pilot_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load apptainer
source "$HOME/smolvla_turing/eval_kit/config.sh"

SIF="$LP_DIR/liberoplus.sif"
MODEL="$SMOLVLA_MODEL"
RUN="$LP_DIR/training/smol_noise_pilot_${SLURM_JOB_ID}"
test -f "$SIF"
test -f "$MODEL/model.safetensors"
mkdir -p "$RUN"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_LEROBOT_HOME="$LP_DIR/hf/lerobot"
REVISION=${DATASET_REVISION:-}
if [ -z "$REVISION" ]; then
  REVISION=$(apptainer exec -B /scratch "$SIF" python -c \
    'from huggingface_hub import HfApi; print(HfApi().dataset_info("lerobot/libero").sha)')
fi
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid dataset revision: $REVISION" >&2; exit 2; }
printf '%s\n' "$REVISION" > "$RUN/dataset_revision.txt"
cp noise_recipe.json "$RUN/noise_recipe.json"
sha256sum train_noise.py split_compat.py check_noise.py check_dataset.py noise_recipe.json > "$RUN/code_sha256.txt"

apptainer exec --nv -B /scratch "$SIF" python check_noise.py
apptainer exec -B /scratch "$SIF" python check_dataset.py --revision "$REVISION"
TRANSFORMS=$(python3 -c 'import json; print(json.dumps(json.load(open("noise_recipe.json"))["transforms"], separators=(",", ":")))')
export SMOLVLA_NOISE_RECIPE="$PWD/noise_recipe.json"
export APPTAINERENV_SMOLVLA_NOISE_RECIPE="$SMOLVLA_NOISE_RECIPE"

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
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs="$TRANSFORMS" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --seed=42 \
  --batch_size=2 \
  --num_workers=2 \
  --steps=10 \
  --save_checkpoint=true \
  --save_freq=0 \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --log_freq=1 \
  --wandb.enable=false \
  --output_dir="$RUN/output" \
  --job_name="smol_noise_pilot_${SLURM_JOB_ID}"

test -f "$RUN/output/checkpoints/last/pretrained_model/model.safetensors" || \
  find "$RUN/output/checkpoints" -name model.safetensors -print -quit | grep -q .
printf 'Pilot complete: %s\n' "$RUN"
