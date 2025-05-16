import argparse
import os

from datasets import load_dataset
from litdata import optimize
from litgpt.tokenizer import Tokenizer

def process_fineweb(split, base_dir, tokenizer_dir="checkpoints/EleutherAI/pythia-1b"):
    rank = split
    dataset_name = "fineweb"

    start = rank * 10
    end = (rank + 1) * 10
    print(f"loading data from {start}% to {end}%")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        num_proc=10,
        name="sample-350BT",
        split=f"train[{start}%:{end}%]",
    )
    total_samples = len(dataset)
    print(f"Total examples: {total_samples}")
    train_samples = int(total_samples * 0.997)
    print(f"Train examples: {train_samples} Val examples: {total_samples - train_samples}")
    
    # TODO add as argument
    tokenizer = Tokenizer(tokenizer_dir)

    # TODO move the following part to run_preprocess.py if more datasets are added
    # TODO make output dirs inputs instead of hardcoding 
    # optimize train
    train_output_dir = os.path.join(base_dir, "data", dataset_name, "train", str(rank))
    optimize(
        fn=lambda index: tokenizer.encode(dataset[index]["text"], eos=True),
        inputs=list(range(train_samples)),
        output_dir=train_output_dir,
        num_workers=16,
        chunk_bytes="200MB",
    )
    
    # optimize val
    val_output_dir = os.path.join(base_dir, "data", dataset_name, "val", str(rank))
    optimize(
        fn=lambda index: tokenizer.encode(dataset[index]["text"], eos=True),
        inputs=list(range(train_samples, total_samples)),
        output_dir=val_output_dir,
        num_workers=16,
        chunk_bytes="200MB",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    process_fineweb(args.split, args.base_dir)