#!/bin/bash

# usage: bash methods/run_selection.sh configs/gradsim_pretrain.yaml

# Check if config path is provided
if [ -z "$1" ]; then
  echo "Error: No config path provided."
  exit 1
fi

# Read method from config
config_path="$1"
echo "config_path: $config_path"
method=$(yq .method "$config_path" | tr -d '"')
echo "method: $method"

# for now, this takes the score for each datapoint, and select the topk with a diversity metric
python methods/select_data.py --config "$config_path"