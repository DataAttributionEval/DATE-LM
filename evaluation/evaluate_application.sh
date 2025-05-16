#!/bin/bash

# usage: bash evaluation/evaluate_application.sh configs/toxicity-bias.yaml /path/to/score

# Check if config path is provided
if [ -z "$1" ]; then
  echo "Error: No config path provided."
  echo "Usage: ./run.sh <config_path> [score_path]"
  exit 1
fi

# Set config path and optional score path
config_path="$1"
score_path="${2:-./results/}"  # Default to ./results/ if not provided

# Read method from config using yq
echo "Config path: $config_path"
echo "Score path: $score_path"

# for now, this takes the score for each datapoint, and select the topk with a diversity metric
python evaluation/evaluate_application.py --config "$config_path" --score_path "$score_path"