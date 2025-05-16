#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=test_edu
#SBATCH --gres=gpu:L40S:8                
#SBATCH --output=logs/test_edu_%J.out
#SBATCH --error=logs/test_edu_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu


# Usage
# cd data-attribution-evaluation
# conda activate myenv
# sbatch methods/edu_classifier/run_edu.sh 


set -e

export NCCL_P2P_DISABLE=1
python methods/edu_classifier/edu_litgpt.py \
    --train_data_dir "/data/group_data/cx_group/data/fineweb/train/0" \
    --output_dir "/data/group_data/cx_group/data/fineweb/train/0/edu" \
    --dataset_tokenizer_path "checkpoints/EleutherAI/pythia-1b"

wait
echo "Done"
