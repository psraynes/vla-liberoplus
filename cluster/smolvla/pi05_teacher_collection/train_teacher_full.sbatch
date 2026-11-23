#!/usr/bin/env bash
#SBATCH --job-name=smol-teacher-full
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=train_teacher_full_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer

TRAIN="$LP_DIR/datasets/pi05_camera_sensor_full_train_v2"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
SIF="$LP_DIR/liberoplus.sif"
MODEL="$SMOLVLA_MODEL"
RUN="$LP_DIR/training/smol_teacher_full_${SLURM_JOB_ID}"
STEPS=100000
BATCH_SIZE=2
test -f "$SIF"
test -f "$MODEL/model.safetensors"
find "$LP_DIR/training/smol_teacher_pilot_2448330/output/checkpoints" -name model.safetensors -print -quit | grep -q .
python3 verify_teacher_datasets.py --dataset-base "$LP_DIR/datasets" --paired-plan "$PAIRED"
mkdir -p "$RUN"
sha256sum "$TRAIN/meta/info.json" "$TRAIN/collection_manifest.jsonl" train_teacher_full.sbatch verify_teacher_datasets.py > "$RUN/input_sha256.txt"
printf 'student_start=%s\nteacher_dataset=%s\nseed=42\nsteps=%s\nbatch_size=%s\nvideo_backend=pyav\nvalidation_dataset=%s\n' \
  "$MODEL" "$TRAIN" "$STEPS" "$BATCH_SIZE" \
  "$LP_DIR/datasets/pi05_camera_sensor_full_validation_v2" > "$RUN/run_manifest.txt"

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
  --batch_size="$BATCH_SIZE" \
  --num_workers=4 \
  --steps="$STEPS" \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps="$STEPS" \
  --scheduler.peak_lr=0.0001 \
  --scheduler.decay_lr=0.0000025 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --log_freq=500 \
  --wandb.enable=false \
  --output_dir="$RUN/output" \
  --job_name="smol_teacher_full_${SLURM_JOB_ID}"

find "$RUN/output/checkpoints" -name model.safetensors -print -quit | grep -q .
echo "Full teacher-data fine-tune complete: $RUN"
