#!/bin/bash
#SBATCH --partition=preempt             
#SBATCH --job-name=run_preprocess            
#SBATCH --mem=200G                  
#SBATCH --gres=gpu:0                 
#SBATCH --cpus-per-task=64           
#SBATCH --time=1-00:00:00            
#SBATCH --output=logs/process_%j.out    
#SBATCH --error=logs/process_%j.err    

# Usage
# conda activate myenv
# cd data-attribution-evaluation
# sbatch data_processing/run_preprocess.sh configs/default.yaml

# Check if config path is provided
if [ -z "$1" ]; then
  echo "Error: No config path provided."
  exit 1
fi

# TODO download tokenizer or include in the repo -- included in repo
# litgpt download \
#   --repo_id EleutherAI/pythia-160m \
#   --tokenizer_only True

# Run the preprocessing script with the provided config path
python data_processing/run_preprocess.py --config "$1"