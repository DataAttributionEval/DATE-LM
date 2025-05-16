import argparse
import os
import numpy as np
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader
import yaml
import datasets

def select(metric_path, metrics_format, select_from_size, selection_size, gumbel_temp, random_selection_seed=42):
    if not metric_path and not gumbel_temp:
        # random selection for finetuning
        rng = np.random.default_rng(random_selection_seed)
        metrics = rng.random(select_from_size)
    elif not metric_path:
        metrics = np.zeros(select_from_size)
        print(">> Random selection Metrics shape:", metrics.shape)
    else:
        if metrics_format == "npy":
            if os.path.isfile(metric_path):
                metrics = np.load(metric_path)
            elif os.path.isdir(metric_path):
                subdirs = [
                    d for d in os.listdir(metric_path)
                    if os.path.isdir(os.path.join(metric_path, d))
                ]
                subdirs = sorted(subdirs, key=int)
                parts = []
                for sd in subdirs:
                    npy_file = os.path.join(metric_path, sd, "metrics.npy")
                    parts.append(np.load(npy_file))
                metrics = np.concatenate(parts, axis=0)
            else:
                raise ValueError(f"metrics_format=='npy' but {metric_path} is neither file nor dir")

            metrics = metrics[:select_from_size]
            print(">> Metrics shape:", metrics.shape)
        else:
            max_splits = len([name for name in os.listdir(metric_path) if os.path.isdir(os.path.join(metric_path, name))])
            metrics_dataset = datasets.concatenate_datasets(
                [
                    datasets.load_from_disk(
                        os.path.join(metric_path, str(i))
                    )
                    for i in range(max_splits)
                ]
            )
            print(metrics_dataset)
            metrics = np.array(metrics_dataset["scores"]).reshape(-1)[:select_from_size] # TODO "scores" should be column name
            print(">> Metrics shape:", metrics.shape)

    if gumbel_temp is not None:
        # Run Gumbel-Top-$k$ algorithm
        # TODO if new diversity method is added separate to helper function
        
        # Normalize metrics (Z-Score)
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)
        metrics = (metrics - mean_metric) / std_metric
        print(">> Normalized metrics (Z-Score):", metrics)

        # Scale metrics by Gumbel temperature
        metrics = metrics / gumbel_temp
        print(">>metric after gumbel temp scaled", metrics)

        # Add Gumbel noise 
        rng = np.random.default_rng(42) # set the seed
        gumbel_noise = rng.gumbel(size=len(metrics))
        metrics += gumbel_noise
        print(">>metrics after gumbel", metrics)

    # Select top-k (selection size) indices
    selected_indices = np.argpartition(metrics, -selection_size)[-selection_size:] # array of indices for top k scores
    topk_scores = metrics[selected_indices]
    avg_topk_scores = np.mean(topk_scores)
    avg_whole_scores = np.mean(metrics)
    print(f">> Average of top-{selection_size} scores: {avg_topk_scores}")
    print(f">> Average of whole scores: {avg_whole_scores}")
    return selected_indices


def main(config_path=None, args=None):
    if config_path:
        # Load configuration from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        print("config", config)
    else:
        # Use command-line arguments
        config = {
            "gumbel_temp": args.gumbel_temp,
            "select_from_size": args.select_from_size,
            "selection_size": args.selection_size,
            "train_data_dir": args.train_data_dir,
            "metrics_dir": args.metrics_dir,
            "metrics_format": args.metrics_format,
            "selected_train_data_dir": args.selected_train_data_dir,
        }
        print("args", config)

    # Extract arguments from config
    dataset_name = config.get("train_dataset_name")
    split = config.get("train_data_split", 0)
    base_dir = config.get("base_dir")

    method = config.get("method") # method name
    gumbel_temp = config.get("gumbel_temp", 0.5)
    select_from_size = config.get("select_from_size")
    selection_size = config.get("selection_size")

    train_dir = config.get("train_data_dir") or os.path.join(base_dir, "data", dataset_name, "train", str(split))
    metrics_dir = config.get("metrics_dir")
    metrics_format = config.get("metrics_format", "npy")
    output_dir = config.get("selected_train_data_dir") or os.path.join(base_dir, "data", dataset_name, "train", str(split), method) # TODO add selection size
    print(f"train data path: {train_dir}")
    print(f"metrics path: {metrics_dir}")
    print(f"selected train data path: {output_dir}")

    # Load dataset to select from
    dataset = StreamingDataset(
        input_dir=train_dir,
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    if not select_from_size or not selection_size or len(dataset) < select_from_size: 
        print(">> Default selecting 10% from the whole dataset")
        select_from_size = len(dataset)
        selection_size = int(0.1 * len(dataset))
    print(f"Selecting {selection_size} from {select_from_size} samples using [ {method} ] selection method")
    
    # Run selection
    indices = select(metrics_dir, metrics_format, select_from_size, selection_size, gumbel_temp)
    indices = list(map(int, indices))
    
    print("writing selected data to {}".format(output_dir))
    # Save selected dataset
    optimize(
        fn=lambda index: dataset[index],
        inputs=indices,
        output_dir=output_dir,
        num_workers=8,
        chunk_bytes="200MB",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--gumbel_temp", type=float, default=0.5, help="Gumbel temperature")
    parser.add_argument("--select_from_size", type=int, help="Number of samples to select from")
    parser.add_argument("--selection_size", type=int, help="Number of samples to select")
    parser.add_argument("--train_data_dir", type=str, help="Directory of training data")
    parser.add_argument("--metrics_dir", type=str, help="Directory of metrics data")
    parser.add_argument("--metrics_format", type=str, help="Format of metrics data")
    parser.add_argument("--selected_train_data_dir", type=str, help="Output directory for selected data")
    args = parser.parse_args()

    if args.config:
        main(config_path=args.config)
    else:
        main(args=args)