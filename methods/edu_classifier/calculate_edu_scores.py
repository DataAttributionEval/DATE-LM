import sys
import argparse
import torch
import json
from pathlib import Path
from datasets import Dataset

# Support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from edu_utils import (
    load_edu_model,
    load_user_data_jsonl,
    process_user_data_with_edu_scores,
)

def main():
    parser = argparse.ArgumentParser(
        description="Compute EDU classifier scores for user data."
    )
    parser.add_argument(
        "--user_data",
        type=str,
        required=True,
        help="Path to the user training data in JSONL format."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="HuggingFaceTB/fineweb-edu-classifier",
        help="Hugging Face model name or local path for the FineWeb-EDU classifier."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="methods/edu_classifier/sample_edu_score/edu_scores",
        help="File to save the EDU scores."
    )
    args = parser.parse_args()

    # Load the EDU classifier (model & tokenizer).
    print(f"Loading FineWeb-EDU model from: {args.model_name_or_path}")
    model, tokenizer = load_edu_model(args.model_name_or_path)

    # Load user data.
    print(f"Loading user data from: {args.user_data}")
    user_data = load_user_data_jsonl(Path(args.user_data))
    print(f"User data has {len(user_data)} samples.")

    # Compute EDU scores for user data.
    print("Computing EDU scores for user data...")
    user_data_with_scores = process_user_data_with_edu_scores(user_data, model, tokenizer)

    # Save the detailed EDU scores in pt.
    print(f"Saving detailed EDU scores to: {args.output_file}.pt")
    torch.save(user_data_with_scores, args.output_file + ".pt")
    
    # Save as JSON for quick inspection
    json_file = args.output_file + ".json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(user_data_with_scores, f, indent=2)
    print(f"Saved EDU scores JSON to: {json_file}")

    # Also save a Hugging Face `Dataset` to disk, the same format as MATES
    dataset = Dataset.from_list(user_data_with_scores)
    dataset.save_to_disk(args.output_file)
    print(f"Saved HF Dataset to: {args.output_file}")

    print("All Done")

if __name__ == "__main__":
    main()