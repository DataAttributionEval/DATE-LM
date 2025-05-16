#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=finetune
#SBATCH --gres=gpu:L40S:1      
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB


# usage :
# [data-attribution-evaluation]$ bash train/run_finetune.sh configs/finetune_llama3-8b-tulu3.yaml # configs/finetune_llama2-7b-tulu2.yaml
set -ex

export NCCL_P2P_DISABLE=1

CONFIG_FILE=$1

train_config=$(yq .train_config "$CONFIG_FILE" | tr -d '"')
train_data_path=$(yq .train_data_path "$CONFIG_FILE" | tr -d '"')
out_dir=$(yq .out_dir "$CONFIG_FILE" | tr -d '"')
checkpoint_dir=$(yq .checkpoint_dir "$CONFIG_FILE" | tr -d '"')

# run training
python train/finetune.py \
  --config $train_config \
  --train_data_path $train_data_path \
  --out_dir $out_dir

python evaluation/convert.py $out_dir/final # convert ckpt to huggingface

# run evaluate
data_base_dir="/data/hf_cache/datasets/lsds_data/data"
MODEL_PATH=$out_dir/final

# mmlu 50 min
# 0shot eval
# NB: bsz 1 due to a bug in the tokenizer for llama 2.
python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
    --ntrain 0 \
    --data_dir ${data_base_dir}/eval/mmlu \
    --save_dir results_mmlu \
    --model_name_or_path ${MODEL_PATH} \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# gsm8k 10 min
# cot, 8 shot.
python -m minimal_multitask.eval.gsm.run_eval \
    --data_dir ${data_base_dir}/eval/gsm \
    --save_dir results_gsm8k \
    --model_name_or_path ${MODEL_PATH} \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# bbh 20 min
# cot, 3-shot
python -m minimal_multitask.eval.bbh.run_eval \
    --data_dir ${data_base_dir}/eval/bbh \
    --save_dir results_bbh \
    --model_name_or_path ${MODEL_PATH} \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

echo "Done evaluation!"

