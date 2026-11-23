#!/usr/bin/env bash
#SBATCH --job-name=smol-teacher-pilot
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=train_teacher_pilot_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer

TRAIN="$LP_DIR/datasets/pi05_camera_sensor_full_train_v2"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
SIF="$LP_DIR/liberoplus.sif"
MODEL="$SMOLVLA_MODEL"
RUN="$LP_DIR/training/smol_teacher_pilot_${SLURM_JOB_ID}"
test -f "$SIF"
test -f "$MODEL/model.safetensors"
python3 verify_teacher_datasets.py --dataset-base "$LP_DIR/datasets" --paired-plan "$PAIRED"
mkdir -p "$RUN"
sha256sum "$TRAIN/meta/info.json" "$TRAIN/collection_manifest.jsonl" train_teacher_pilot.sbatch verify_teacher_datasets.py > "$RUN/input_sha256.txt"
printf 'student=%s\nteacher_dataset=%s\nseed=42\nsteps=10\nbatch_size=2\nvideo_backend=pyav\n' \
  "$MODEL" "$TRAIN" > "$RUN/run_manifest.txt"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_LEROBOT_HOME="$LP_DIR/hf/lerobot"
apptainer exec --nv -B /scratch "$SIF" lerobot-train \
  --policy.path="$MODEL" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --dataset.repo_id=capstone/pi05_camera_sensor_full_train_v2 \
  --dataset.root="$TRAIN" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.0 \
  --dataset.image_transforms.enable=false \
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
  --job_name="smol_teacher_pilot_${SLURM_JOB_ID}"

find "$RUN/output/checkpoints" -name model.safetensors -print -quit | grep -q .
echo "Training pilot complete: $RUN"
