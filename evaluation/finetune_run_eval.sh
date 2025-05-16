#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=eval
#SBATCH --gres=gpu:L40S:2    
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G

# bash evaluation/finetune_run_eval.sh meta-llama/Llama-3.1-8B
set -ex

MODEL_PATH=$1

data_base_dir="/data/hf_cache/datasets/lsds_data/data"

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
