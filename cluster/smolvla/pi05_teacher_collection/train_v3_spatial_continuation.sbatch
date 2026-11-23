#!/usr/bin/env bash
#SBATCH --job-name=smol-v3-spatial
#SBATCH --partition=academic
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train_v3_spatial_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MODE=${TRAIN_MODE:-pilot}
case "$MODE" in pilot) STEPS=10; SAVE=0; LOG=1 ;; full) STEPS=20000; SAVE=5000; LOG=100 ;; *) exit 2 ;; esac
source "$HOME/smolvla_turing/eval_kit/config.sh"
module load apptainer
MODEL="$LP_DIR/training/smol_combined_full_fixed_2473800/output/checkpoints/400000/pretrained_model"
test -s "$MODEL/model.safetensors"
TRAIN="$LP_DIR/datasets/spatial_balanced_clean_v4_train"
PAIRED="$HOME/smolvla_turing/eval_kit/paired_plan_4295.tsv"
python3 balanced_spatial_v4.py verify --paired "$PAIRED" --dataset-base "$LP_DIR/datasets"
if [[ "$MODE" == full ]]; then
  PILOT=$(sed -n 's/^Pilot=//p' v3_spatial_continuation_jobs.txt)
  [[ "$PILOT" =~ ^[0-9]+$ ]]
  find "$LP_DIR/training/smol_v3_spatial_continuation_pilot_$PILOT/output/checkpoints" \
    -name model.safetensors -print -quit | grep -q .
fi
RUN="$LP_DIR/training/smol_v3_spatial_continuation_${MODE}_${SLURM_JOB_ID}"
mkdir -p "$RUN"
cp balanced_spatial_v4_plan.json "$RUN/"
sha256sum balanced_spatial_v4.py train_v3_spatial_continuation.sbatch \
  "$TRAIN/meta/info.json" "$TRAIN/collection_manifest.jsonl" > "$RUN/input_sha256.txt"
printf 'model=%s\nmode=%s\nsteps=%s\nbatch_size=8\nseed=42\nlr=0.00001\ninstruction_mode=clean\n' \
  "$MODEL" "$MODE" "$STEPS" > "$RUN/run_manifest.txt"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_LEROBOT_HOME="$LP_DIR/hf/lerobot"
apptainer exec --nv -B /scratch "$LP_DIR/liberoplus.sif" lerobot-train \
  --policy.path="$MODEL" --policy.device=cuda --policy.push_to_hub=false \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.train_state_proj=true \
  --policy.optimizer_lr=0.00001 --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps="$STEPS" --policy.scheduler_decay_lr=0.000001 \
  --dataset.repo_id=capstone/spatial_balanced_clean_v4_train --dataset.root="$TRAIN" \
  --dataset.video_backend=pyav --dataset.eval_split=0.0 --dataset.image_transforms.enable=false \
  --seed=42 --batch_size=8 --num_workers=4 --steps="$STEPS" \
  --save_checkpoint=true --save_freq="$SAVE" --eval_steps=0 --env_eval_freq=0 --log_freq="$LOG" \
  --wandb.enable=false --output_dir="$RUN/output" --job_name="v3_spatial_continuation_${MODE}_${SLURM_JOB_ID}"
find "$RUN/output/checkpoints" -name model.safetensors -print -quit | grep -q .
echo "V3 Spatial continuation $MODE complete: $RUN"
