#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=bm25_instruct
#SBATCH --gres=gpu:0                
#SBATCH --output=logs/probe_bm25_instruct_%A_%a.out
#SBATCH --error=logs/probe_bm25_instruct_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --array=0-2  # Array job for 3 tasks

# Usage
# cd data-attribution-evaluation
# conda activate mmft
# sbatch methods/bm25/bm25_instruct.sh 

# Parameterized variables
OUT_DIR="/data/user_data/emilyx/scores"
VAL_DATASETS=("mmlu" "gsm8k" "bbh")  # List of validation datasets

# Get the current dataset based on the SLURM_ARRAY_TASK_ID
VAL_DATASET=${VAL_DATASETS[$SLURM_ARRAY_TASK_ID]}

# Run the task
CUDA_VISIBLE_DEVICES=0 python -m methods.bm25.bm25_instruct \
      --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
      --val_dataset_name ${VAL_DATASET} \
      --val_num_samples 100 \
      --subset_size 200000 \
      --out_dir ${OUT_DIR}/${VAL_DATASET}_bm25.npy \
      --model_ckpt "meta-llama/Llama-3.1-8B"

# test
# python -m methods.bm25.bm25_instruct \
#       --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
#       --val_dataset_name mmlu \
#       --val_num_samples 100 \
#       --subset_size 200000 \
#       --out_dir out/test_bm25.npy \
#       --model_ckpt "meta-llama/Llama-3.1-8B"