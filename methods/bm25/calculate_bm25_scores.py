import sys
import argparse
import yaml
import torch
import json
import os
from pathlib import Path
from datasets import Dataset

# Support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from litgpt import Tokenizer
from bm25_utils import (
    decode_texts_from_pt,
    build_bm25_index,
    load_user_data_jsonl,
    process_user_data_with_scores,
    compute_average_bm25_scores
)

def main():
    parser = argparse.ArgumentParser(
        description="Compute detailed and average BM25 scores for user data."
    )
    parser.add_argument(
        "--reference_pt",
        type=str,
        default="datasets/data/lambada_openai/train-1024.pt",
        help="Path to the reference .pt file."
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="tokenizer/togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
        help="Path to the tokenizer directory for lit-GPT."
    )
    parser.add_argument(
        "--user_data",
        type=str,
        required=True,
        help="Path to the user training data in JSONL format."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="methods/bm25/sample_bm25_score/bm25_score_detail.pt",
        help="File to save the detailed BM25 scores."
    )
    parser.add_argument(
        "--average_output",
        type=str,
        default="methods/bm25/sample_bm25_score/average_bm25_scores",
        help="File to save the average BM25 scores."
    )
    args = parser.parse_args()

    # Load the tokenizer.
    tokenizer = Tokenizer(Path(args.tokenizer_dir))

    # Load and decode the reference dataset.
    print(f"Loading reference dataset from: {args.reference_pt}")
    reference_texts, reference_ids = decode_texts_from_pt(Path(args.reference_pt), tokenizer)
    print(f"Reference set has {len(reference_texts)} items.")

    # Build the BM25 index.
    print("Building BM25 index from reference texts...")
    bm25 = build_bm25_index(reference_texts)

    # Load user data.
    print(f"Loading user data from: {args.user_data}")
    user_data = load_user_data_jsonl(Path(args.user_data))
    print(f"User data has {len(user_data)} samples.")

    # Compute detailed BM25 scores for user data.
    print("Computing BM25 scores for user data...")
    user_data_with_scores = process_user_data_with_scores(user_data, bm25, reference_ids)

    # Save the detailed BM25 scores.
    print(f"Saving detailed BM25 scores to: {args.output_file}")
    torch.save(user_data_with_scores, args.output_file)

    # Calculate and save average BM25 scores for each datapoint.
    average_scores = compute_average_bm25_scores(user_data_with_scores)
    # save as json for visualization now, can be changed to pt later for data selection
    with open(args.average_output+".json", "w", encoding="utf-8") as f:
        json.dump(average_scores, f, indent=2)

    # the MATES used a dataset containing maps like {"index": 123, "prediction": 0.0}
    # where index is the original index of the example in the source dataset
    # save a output in similar format
    dataset = Dataset.from_list(average_scores)
    dataset.save_to_disk(args.average_output)
    print(f"Saved average BM25 scores to: {args.average_output}")
    print("All Done")

if __name__ == "__main__":
    main()