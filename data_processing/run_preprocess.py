import argparse
import yaml

from process_fineweb import process_fineweb
from prepare_lambada import prepare_lambada
import os

def main(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    train_dataset_name = config.get("train_dataset_name")
    split = config.get("train_data_split", 0)
    base_dir = config.get("base_dir")

    train_path = config.get("train_data_dir", os.path.join(base_dir, "data", train_dataset_name, "train", str(split)))
    val_path = os.path.join(base_dir, "data", train_dataset_name, "val", str(split))
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")

    # Check if data already exists, returns if it does
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data already exists at {train_path} and {val_path}")
        return
    
    # Process the training dataset
    if train_dataset_name == "fineweb":
        process_fineweb(split, base_dir)
        print("Processing fineweb dataset...")
    else:
        print(f"Error: No processing function found for dataset '{train_dataset_name}'")

    # Process the reference dataset
    # TODO: Implement this
    print("Processing lambada dataset...")
    prepare_lambada(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)