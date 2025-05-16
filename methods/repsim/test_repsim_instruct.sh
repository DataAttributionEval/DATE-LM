#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=instruct_test
#SBATCH --gres=gpu:L40S:1                
#SBATCH --output=probe_instruct_%J.out
#SBATCH --error=probe_instruct_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --exclude=babel-13-13,babel-2-13,babel-13-1,babel-4-13
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address

# Usage
# cd data-att
# conda activate mmft
# sbatch methods/repsim/test_repsim_instruct.sh 

# out_dir /data/user_data/emilyx/scores/instruct_mmlu_gradsim.npy

CUDA_VISIBLE_DEVICES=0 python -m methods.repsim.probe_repsim_instruct \
      --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
      --resume /data/group_data/cx_group/date-models/Llama-3.1-8B-tulu3-opt/final/ \
      --out_dir out/scores/mmlu_repsim.npy \
      --subset_size 100 \
      --val_dataset_name mmlu \
      --val_num_samples 100 \

# CUDA_VISIBLE_DEVICES=0 python -m methods.gradsim.probe_gradient_similarity_instruct_simple \
#       --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
#       --resume /data/group_data/cx_group/date-models/Llama-3.1-8B-tulu3-opt/final/ \
#       --out_dir out/scores/instruct_gsm8k_gradsim.npy \
#       --subset_size 100 \
#       --val_dataset_name gsm8k \
#       --val_num_samples 100 \


# CUDA_VISIBLE_DEVICES=0 python -m methods.gradsim.probe_gradient_similarity_instruct_simple \
#       --train_data_dir /data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl \
#       --resume /data/group_data/cx_group/date-models/Llama-3.1-8B-tulu3-opt/final/ \
#       --out_dir out/scores/instruct_bbh_gradsim.npy \
#       --subset_size 100 \
#       --val_dataset_name bbh \
#       --val_num_samples 100 \