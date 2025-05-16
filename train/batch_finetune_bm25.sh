#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=finetune_bm25
#SBATCH --gres=gpu:L40S:1      
#SBATCH --output=logs/finetune_bm25_%A_%a.out
#SBATCH --error=logs/finetune_bm25_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address
#SBATCH --array=0  # Array job for 3 tasks: mmlu, gsm8k, bbh

# usage:
# (mmft) [@babel-15-20 data-attribution-evaluation]$ sbatch train/batch_finetune_bm25.sh

set -ex

export NCCL_P2P_DISABLE=1

# Define tasks and corresponding metric paths
TASKS=("mmlu" "gsm8k" "bbh")
METRIC_PATHS=(
    "/data/user_data/emilyx/scores/mmlu_bm25.npy"
    "/data/user_data/emilyx/scores/gsm8k_bm25.npy"
    "/data/user_data/emilyx/scores/bbh_bm25.npy"
)

# Get the current task and metric path based on SLURM_ARRAY_TASK_ID
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
METRIC_PATH=${METRIC_PATHS[$SLURM_ARRAY_TASK_ID]}

# Define parameters
train_config="configs/train_finetune_llama3-8b.yaml"
train_data_path="/data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl"
checkpoint_dir="/data/group_data/cx_group/date-models/Llama-3.1-8B"
out_dir="/data/group_data/cx_group/date-models/Llama-3.1-8B-tulu3-bm25-${TASK}/"

echo "Running finetuning for task: $TASK"
echo "Metric Path: $METRIC_PATH"
echo "Output Directory: $out_dir"

# Run the finetune script
# Check if the final output directory already exists
if [ -d "$out_dir/final" ]; then
    echo "Output directory $out_dir/final already exists. Skipping finetuning for task: $TASK"
else
    # Run the finetune script
    python train/finetune.py \
      --config $train_config \
      --train_data_path $train_data_path \
      --out_dir $out_dir \
      --metric_path $METRIC_PATH \

    echo "Finetuning completed for task: $TASK"
fi

# Convert the checkpoint to HuggingFace format
python evaluation/convert.py $out_dir/final
echo "Checkpoint converted to HuggingFace format for task: $TASK"

# Run evaluation based on the task
data_base_dir="/data/hf_cache/datasets/lsds_data/data"
MODEL_PATH=$out_dir/final

if [ "$TASK" == "mmlu" ]; then
    echo "Running MMLU evaluation..."
    python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
        --ntrain 0 \
        --data_dir ${data_base_dir}/eval/mmlu \
        --save_dir out/bm25/results_mmlu \
        --model_name_or_path ${MODEL_PATH} \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format
elif [ "$TASK" == "gsm8k" ]; then
    echo "Running GSM8K evaluation..."
    python -m minimal_multitask.eval.gsm.run_eval \
        --data_dir ${data_base_dir}/eval/gsm \
        --save_dir out/bm25/results_gsm8k \
        --model_name_or_path ${MODEL_PATH} \
        --n_shot 8 \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
        --use_vllm
elif [ "$TASK" == "bbh" ]; then
    echo "Running BBH evaluation..."
    python -m minimal_multitask.eval.bbh.run_eval \
        --data_dir ${data_base_dir}/eval/bbh \
        --save_dir out/bm25/results_bbh \
        --model_name_or_path ${MODEL_PATH} \
        --use_vllm \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format
else
    echo "Unknown task: $TASK"
    exit 1
fi

echo "Evaluation completed for task: $TASK"