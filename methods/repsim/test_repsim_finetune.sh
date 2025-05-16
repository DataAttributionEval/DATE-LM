#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=test_repsim_finetune
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/test_repsim_finetune_%A_%a.out
#SBATCH --error=logs/test_repsim_finetune_%A_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu # change to your email address

# Usage:
# cd data-attribution-evaluation
# conda activate myenv
# sbatch methods/repsim/test_repsim_finetune.sh

set -e
RANK=$SLURM_ARRAY_TASK_ID

step_dir="/data/group_data/cx_group/date-models/pythia-1b"
#train_dir="methods/repsim/mathqa.json"
train_dir="/data/group_data/cx_group/data/mathqa/mathqa.json"
val_dir="/data/group_data/cx_group/data/mathqa/mathqa-valid.json"
out_dir="/data/group_data/cx_group/data/mathqa"

# Run the script for the current shard
echo "Processing shard $RANK"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python methods/repsim/repsim_utils_litgpt_finetune.py \
    --model_name pythia-1b \
    --train_json_path "${train_dir}" \
    --val_json_path "${val_dir}" \
    --out_dir "${out_dir}" \
    --initial_checkpoint_dir "${step_dir}" \
    --rank $RANK \
    --shard_size 3730 \
    --devices 1
