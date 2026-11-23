#!/usr/bin/env bash
#SBATCH --job-name=smol-combined-fixed
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=train_combined_full_fixed_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer
TRAIN="$LP_DIR/datasets/pi05_camera_sensor_combined_v3_train"
VALID="$LP_DIR/datasets/pi05_camera_sensor_combined_v3_validation"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
SIF="$LP_DIR/liberoplus.sif"
MODEL="$SMOLVLA_MODEL"
RUN="$LP_DIR/training/smol_combined_full_fixed_${SLURM_JOB_ID}"
STEPS=400000
BATCH_SIZE=2
test -f "$SIF"
test -f "$MODEL/model.safetensors"
PILOT=$(sed -n 's/^CombinedPilot=//p' combined_training_pilot_id.txt)
[[ "$PILOT" =~ ^[0-9]+$ ]] || { echo 'Missing combined pilot job ID' >&2; exit 2; }
find "$LP_DIR/training/smol_combined_pilot_${PILOT}/output/checkpoints" \
  -name model.safetensors -print -quit | grep -q .
python3 verify_combined_v3.py --dataset-base "$LP_DIR/datasets" --paired-plan "$PAIRED"
mkdir -p "$RUN"
sha256sum "$TRAIN/meta/info.json" "$TRAIN/collection_manifest.jsonl" \
  "$VALID/meta/info.json" "$VALID/collection_manifest.jsonl" \
  train_combined_full_fixed.sbatch verify_combined_v3.py > "$RUN/input_sha256.txt"
printf 'student_start=%s\ntrain=%s\nvalidation=%s\nseed=42\nsteps=%s\nbatch_size=%s\nvideo_backend=pyav\n' \
  "$MODEL" "$TRAIN" "$VALID" "$STEPS" "$BATCH_SIZE" > "$RUN/run_manifest.txt"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_LEROBOT_HOME="$LP_DIR/hf/lerobot"
apptainer exec --nv -B /scratch "$SIF" lerobot-train \
  --policy.path="$MODEL" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id=capstone/pi05_camera_sensor_combined_v3_train \
  --dataset.root="$TRAIN" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.0 \
  --dataset.image_transforms.enable=false \
  --seed=42 \
  --batch_size="$BATCH_SIZE" \
  --num_workers=4 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=100000 \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --log_freq=1000 \
  --wandb.enable=false \
  --output_dir="$RUN/output" \
  --job_name="smol_combined_full_fixed_${SLURM_JOB_ID}"
test -f "$RUN/output/checkpoints/400000/pretrained_model/model.safetensors"
echo "Combined-data SmolVLA full fine-tune complete: $RUN"
