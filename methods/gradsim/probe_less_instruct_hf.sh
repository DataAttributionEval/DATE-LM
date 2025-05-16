#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=less
#SBATCH --gres=gpu:L40S:1                
#SBATCH --output=logs/probe_less_instruct_%A_%a.out
#SBATCH --error=logs/probe_less_instruct_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --exclude=babel-13-13,babel-2-13,babel-13-1,babel-4-13
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address
#SBATCH --array=0-2  # Array job for 3 tasks

# Usage
# cd data-att
# conda activate mmft
# sbatch methods/gradsim/probe_less_instruct_hf.sh 

### Parameterized variables
OUT_DIR="/data/user_data/emilyx/scores"
# VAL_DATASETS=("mmlu" "gsm8k" "bbh")  # List of validation datasets
VAL_DATASETS=("mmlu_shots" "gsm8k_shots" "bbh_shots")  # List of validation datasets


### Get the current dataset based on the SLURM_ARRAY_TASK_ID
VAL_DATASET=${VAL_DATASETS[$SLURM_ARRAY_TASK_ID]}

### Run the task
CUDA_VISIBLE_DEVICES=0 python -m methods.gradsim.probe_less_instruct_hf \
      --base_model_name meta-llama/Llama-3.1-8B \
      --lora_ckpt /data/user_data/emilyx/lora_llama3 \
      --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
      --out_dir ${OUT_DIR}/${VAL_DATASET}_less_hf.npy \
      --subset_size 200000 \
      --val_dataset_name ${VAL_DATASET} \
      --val_num_samples -1

#### test
# CUDA_VISIBLE_DEVICES=0 python -m methods.gradsim.probe_less_instruct_hf \
#       --base_model_name meta-llama/Llama-3.1-8B \
#       --lora_ckpt /data/user_data/emilyx/lora_llama3 \
#       --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
#       --out_dir out/less_hf.npy \
#       --subset_size 100 \
#       --val_dataset_name gsm8k_shots \
#       --val_num_samples -1