#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=pretrain_decay
#SBATCH --gres=gpu:L40S:8                
#SBATCH --output=logs/pretrain_%J.out
#SBATCH --error=logs/pretrain_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G

# Usage: sbatch train/run_pretrain.sh configs/default.yaml

set -ex

export NCCL_P2P_DISABLE=1

CONFIG_FILE=$1

train_config=$(yq .train_config "$CONFIG_FILE" | tr -d '"')
selected_train_data_dir=$(yq .selected_train_data_dir "$CONFIG_FILE" | tr -d '"')
resume=$(yq .resume "$CONFIG_FILE" | tr -d '"')
out_dir=$(yq .out_dir "$CONFIG_FILE" | tr -d '"')

# Run Training
python train/pretrain.py \
  --config $train_config \
  --data_path $selected_train_data_dir \
  --resume $resume \
  --out_dir $out_dir \

# Run Evaluation
echo "Running evaluation for Gumbel temperature $gumbel_temp"
python3 evaluation/evaluate.py \
  "${out_dir}/final" \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "sciq,arc_easy,arc_challenge,logiqa,mmlu,boolq,hellaswag,piqa,winogrande,openbookqa,lambada,lambada_openai" \
