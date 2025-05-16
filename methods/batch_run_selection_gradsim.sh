#!/bin/bash

# Usage: bash methods/batch_run_selection.sh

# Define the base parameters from gradsim_pretrain.yaml
base_train_data_dir="/data/group_data/cx_group/data/fineweb/train/0"
base_metrics_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00030000/gradsim_flan"
base_output_dir="/data/group_data/cx_group/data/fineweb/train/0/gradsim_flan_pythia1b_step30000"
select_from_size=1024000
selection_size=102400

# Define the Gumbel temperature settings to test
gumbel_settings=(0.1 0.5 1.0 2.0)

# Iterate over Gumbel temperature settings
for gumbel_temp in "${gumbel_settings[@]}"; do
  # Define the output directory for this run
  run_output_dir="${base_output_dir}_gumbel_${gumbel_temp}"
  echo "Output directory: $run_output_dir"

  # Run the selection script with command-line arguments
  echo "Running selection with Gumbel temperature: $gumbel_temp"
  python methods/select_data.py \
    --gumbel_temp "$gumbel_temp" \
    --select_from_size "$select_from_size" \
    --selection_size "$selection_size" \
    --train_data_dir "$base_train_data_dir" \
    --metrics_dir "$base_metrics_dir" \
    --selected_train_data_dir "$run_output_dir"

  # Check if the run was successful
  if [ $? -eq 0 ]; then
    echo "Run with Gumbel temperature $gumbel_temp completed successfully."
  else
    echo "Run with Gumbel temperature $gumbel_temp failed."
  fi
done