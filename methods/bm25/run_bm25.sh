#!/bin/bash
# Usage: bash methods/bm25/run_bm25.sh

python methods/bm25/bm25_litgpt.py \
    --train_data_dir "/data/group_data/cx_group/data/fineweb/train/0" \
    --reference_pt_path "/data/group_data/cx_group/data/lambada_openai/train/train-1024.pt" \
    --output_dir "/data/group_data/cx_group/data/fineweb/train/0/bm25" \
    --reference_tokenizer_path "checkpoints/EleutherAI/pythia-1b" \
    --dataset_tokenizer_path "checkpoints/EleutherAI/pythia-1b"