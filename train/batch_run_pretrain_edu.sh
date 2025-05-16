#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=batch_pretrain_decay
#SBATCH --gres=gpu:L40S:8                
#SBATCH --output=logs/pretrain_%J.out
#SBATCH --error=logs/pretrain_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu  # Replace with your email address

# Usage: sbatch train/batch_run_pretrain_edu.sh

export NCCL_P2P_DISABLE=1
set -e

# Base parameters
resume="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth"
base_out_dir="/data/user_data/hanzhanz/test"
config_path="configs/train_pretrain_decay_pythia-1b.yaml"
tasks="sciq,arc_easy,arc_challenge,logiqa,mmlu,boolq,hellaswag,piqa,winogrande,openbookqa,lambada,lambada_openai"

# Define the data paths to test
data_paths=(
    "/data/group_data/cx_group/data/fineweb/train/0/edu_lambada_pythia1b_step10000_gumbel_0.1"
    "/data/group_data/cx_group/data/fineweb/train/0/edu_lambada_pythia1b_step10000_gumbel_0.5"
    "/data/group_data/cx_group/data/fineweb/train/0/edu_lambada_pythia1b_step10000_gumbel_1.0"
    "/data/group_data/cx_group/data/fineweb/train/0/edu_lambada_pythia1b_step10000_gumbel_2.0"
)

# Iterate over data paths
for data_path in "${data_paths[@]}"; do
  # Extract the Gumbel temperature from the data path for naming
  gumbel_temp=$(basename "$data_path" | grep -oP 'gumbel_\K[0-9.]+')

  # Define the output directory and experiment name
  out_dir="${base_out_dir}/edu_gumbel_${gumbel_temp}"
  exp_name="pretrain_gumbel_${gumbel_temp}"

  # Run the pretraining script
  echo "Running pretraining with data_path: $data_path, out_dir: $out_dir, exp_name: $exp_name"
  python3 train/pretrain.py \
    --config "$config_path" \
    --data_path "$data_path" \
    --resume "$resume" \
    --out_dir "$out_dir" \
    --exp_name "$exp_name"

  # Check if the pretraining run was successful
  if [ $? -eq 0 ]; then
    echo "Pretraining with Gumbel temperature $gumbel_temp completed successfully."

    # Run evaluation
    echo "Running evaluation for Gumbel temperature $gumbel_temp"
    python3 evaluation/evaluate.py \
      "${out_dir}/final" \
      --num_fewshot 0 \
      --batch_size 4 \
      --tasks "$tasks" \
      --out_dir "out/edu_gumbel_${gumbel_temp}" \

    # Check if the evaluation run was successful
    if [ $? -eq 0 ]; then
      echo "Evaluation for Gumbel temperature $gumbel_temp completed successfully."
    else
      echo "Evaluation for Gumbel temperature $gumbel_temp failed."
    fi
  else
    echo "Pretraining with Gumbel temperature $gumbel_temp failed."
  fi
done