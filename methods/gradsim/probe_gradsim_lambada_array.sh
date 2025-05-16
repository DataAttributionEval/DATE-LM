#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=probe_gradsim_lambada
#SBATCH --gres=gpu:L40S:1              
#SBATCH --output=logs/probe_gradsim_lambada_%A_%a.out
#SBATCH --error=logs/probe_gradsim_lambada_%A_%a.err
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emilyx@andrew.cmu.edu
#SBATCH --requeue

# Description: Runs probe_gradient_similarity.py for 16 shards using SLURM array jobs.
# Usage:
# cd data-attribution-evaluation
# conda activate myenv
# sbatch methods/gradsim/probe_gradsim_lambada_array.sh

# Exit immediately on any error
set -ex

# Define the step and shard rank
STEP="step-00030000"
RANK=$SLURM_ARRAY_TASK_ID

# Run the script for the current shard
echo "Processing shard $RANK"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python methods/gradsim/probe_gradient_similarity.py \
    --model_name pythia-1b \
    --train_data_dir /data/group_data/cx_group/data/fineweb/train/0 \
    --reference_data_dir /data/user_data/emilyx/lambada_openai/train-1024.pt  \
    --reference_data_size 1024 \
    --resume /data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/${STEP}/lit_model.pth \
    --out_dir /data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/${STEP}/gradsim_lambada_preempt \
    --rank $RANK \
    --devices 1