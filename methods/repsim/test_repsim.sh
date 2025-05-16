#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=test_repsim
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/test_repsim_%A_%a.out
#SBATCH --error=logs/test_repsim_%A_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu # change to your email address

# Usage:
# cd data-attribution-evaluation
# conda activate myenv
# sbatch methods/repsim/test_repsim.sh

set -e
RANK=$SLURM_ARRAY_TASK_ID

step_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000"
train_dir="/data/group_data/cx_group/data/fineweb/train/0"
out_dir="${train_dir}/repsim"

# Run the script for the current shard
echo "Processing shard $RANK"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python methods/repsim/repsim_utils_litgpt.py \
    --model_name pythia-1b \
    --train_data_dir /data/group_data/cx_group/data/fineweb/train/0 \
    --out_dir "${out_dir}" \
    --initial_checkpoint_dir "${step_dir}" \
    --rank $RANK \
    --shard_size 64000 \
    --devices 1