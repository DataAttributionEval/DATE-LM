#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=gradsim
#SBATCH --gres=gpu:L40S:1      
#SBATCH --output=logs/finetune_%x_%A_%a.out
#SBATCH --error=logs/finetune_%x_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --array=0-2  # Array job for 3 tasks: mmlu, gsm8k, bbh

# usage:
# sbatch train/batch_finetune_shots.sh --method gradsim_hf --base_dir /path/to/base

set -ex

export NCCL_P2P_DISABLE=1

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift ;;
        --base_dir) BASE_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$METHOD" ]; then
    echo "Error: --method flag is required (e.g., bm25, repsim)"
    exit 1
fi

if [ -z "$BASE_DIR" ]; then
    echo "Error: --base_dir flag is required (e.g., /path/to/base)"
    exit 1
fi

# Define tasks and corresponding metric paths
TASKS=("mmlu_shots" "gsm8k_shots" "bbh_shots")
METRIC_PATHS=(
    "${BASE_DIR}/scores/mmlu_shots_${METHOD}.npy"
    "${BASE_DIR}/scores/gsm8k_shots_${METHOD}.npy"
    "${BASE_DIR}/scores/bbh_shots_${METHOD}.npy"
)

# Get the current task and metric path based on SLURM_ARRAY_TASK_ID
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
METRIC_PATH=${METRIC_PATHS[$SLURM_ARRAY_TASK_ID]}

# Define parameters
train_config="configs/train_finetune_llama3-8b.yaml"
train_data_path="${BASE_DIR}/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl"
checkpoint_dir="${BASE_DIR}/date-models/Llama-3.1-8B"
out_dir="${BASE_DIR}/date-models/Llama-3.1-8B-tulu3-${METHOD}-${TASK}/"

echo "Running finetuning for task: $TASK with method: $METHOD"
echo "Metric Path: $METRIC_PATH"
echo "Output Directory: $out_dir"

# Run the finetune script
python train/finetune.py \
    --config $train_config \
    --train_data_path $train_data_path \
    --out_dir $out_dir \
    --metric_path $METRIC_PATH \

echo "Finetuning completed for task: $TASK"

# Convert the checkpoint to HuggingFace format
python evaluation/convert.py $out_dir/final
echo "Checkpoint converted to HuggingFace format for task: $TASK"

# Run evaluation based on the task
data_base_dir="${BASE_DIR}/lsds_data/data"
MODEL_PATH=$out_dir/final

if [ "$TASK" == "mmlu_shots" ]; then
    echo "Running MMLU evaluation..."
    python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
        --ntrain 0 \
        --data_dir ${data_base_dir}/eval/mmlu \
        --save_dir out/${METHOD}/results_mmlu \
        --model_name_or_path ${MODEL_PATH} \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format
elif [ "$TASK" == "gsm8k_shots" ]; then
    echo "Running GSM8K evaluation..."
    python -m minimal_multitask.eval.gsm.run_eval \
        --data_dir ${data_base_dir}/eval/gsm \
        --save_dir out/${METHOD}/results_gsm8k \
        --model_name_or_path ${MODEL_PATH} \
        --n_shot 8 \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
        --use_vllm
elif [ "$TASK" == "bbh_shots" ]; then
    echo "Running BBH evaluation..."
    python -m minimal_multitask.eval.bbh.run_eval \
        --data_dir ${data_base_dir}/eval/bbh \
        --save_dir out/${METHOD}/results_bbh \
        --model_name_or_path ${MODEL_PATH} \
        --use_vllm \
        --use_chat_format \
        --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format
else
    echo "Unknown task: $TASK"
    exit 1
fi

echo "Evaluation completed for task: $TASK with method: $METHOD"