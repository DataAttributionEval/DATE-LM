#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=batch_pretrain_decay
#SBATCH --gres=gpu:L40S:8                
#SBATCH --output=logs/pretrain_%J.out
#SBATCH --error=logs/pretrain_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address

# Usage: sbatch train/batch_run_pretrain_mates.sh

export NCCL_P2P_DISABLE=1
set -ex

# Base parameters
base_out_dir="/data/hf_cache/datasets/data/date"
config_path="configs/train_pretrain_decay_pythia-1b.yaml"
tasks="sciq,arc_easy,arc_challenge,logiqa,boolq,hellaswag,piqa,winogrande,openbookqa,lambada,lambada_openai"

# Define the sets of steps, reference datasets, and Gumbel temperatures
steps=("30000" "30000")
refdatasets=("lambada" "flan")
gumbel_settings=(2.0 1.0)

# Iterate over each set of parameters
for i in "${!steps[@]}"; do
  step=${steps[$i]}
  refdataset=${refdatasets[$i]}

  # Format the step with leading zeros (e.g., step-00030000)
  formatted_step=$(printf "%08d" "$step")
  resume="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-${formatted_step}/lit_model.pth"

  # Iterate over Gumbel temperature settings
  for gumbel_temp in "${gumbel_settings[@]}"; do
    # Define the data path, output directory, and experiment name
    data_path="/data/hf_cache/datasets/data/fineweb/train/0/mates_${refdataset}_pythia1b_step${step}_gumbel_${gumbel_temp}"
    out_dir="${base_out_dir}/mates_${refdataset}_pythia1b_step${step}_gumbel_${gumbel_temp}"
    exp_name="pretrain_mates_${refdataset}_step${step}_gumbel_${gumbel_temp}"

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
      echo "Pretraining with step $step, dataset $refdataset, and Gumbel temperature $gumbel_temp completed successfully."

      # Run evaluation
      echo "Running evaluation for step $step, dataset $refdataset, and Gumbel temperature $gumbel_temp"
      python3 evaluation/evaluate.py \
        "${out_dir}/final" \
        --num_fewshot 0 \
        --batch_size 4 \
        --tasks "$tasks" \
        --out_dir "out/mates_${refdataset}_step${step}_gumbel_${gumbel_temp}"

      # Check if the evaluation run was successful
      if [ $? -eq 0 ]; then
        echo "Evaluation for step $step, dataset $refdataset, and Gumbel temperature $gumbel_temp completed successfully."
      else
        echo "Evaluation for step $step, dataset $refdataset, and Gumbel temperature $gumbel_temp failed."
      fi
    else
      echo "Pretraining with step $step, dataset $refdataset, and Gumbel temperature $gumbel_temp failed."
    fi
  done
done