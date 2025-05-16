#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=predict_influence
#SBATCH --gres=gpu:L40S:1              
#SBATCH --output=logs/predict_influence_%A_%a.out
#SBATCH --error=logs/predict_influence_%A_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00

# Description: Runs predict_data_influence.py for 8 shards using SLURM array jobs.
# Usage:
# sbatch methods/mates/run_mates_predict.sh

# Exit immediately on any error
set -e

# Define the sets of parameters
declare -a steps=("30000" "30000")
declare -a refdatas=("lambada" "flan")

# Loop through each set of parameters
for i in "${!steps[@]}"; do
    step=${steps[$i]}
    refdata=${refdatas[$i]}

    # Define parameters
    model_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/${step}-data_influence_model-${refdata}"
    output_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-000${step}/mates_${refdata}"
    select_from_size=1024000

    # Get the shard index from the SLURM array task ID
    SHARD=$SLURM_ARRAY_TASK_ID

    # Run the script for the current shard
    echo "Processing shard $SHARD with step $step and refdata $refdata"
    CUDA_VISIBLE_DEVICES=0 python methods/mates/predict_data_influence.py \
        --model_dir $model_dir \
        --output_dir $output_dir \
        --select_from_size $select_from_size \
        --shard $SHARD 8
done