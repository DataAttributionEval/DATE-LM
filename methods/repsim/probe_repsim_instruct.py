import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy, instantiate_torch_optimizer, num_parameters, parse_devices
from litdata.streaming import StreamingDataset, TokensLoader
import os
from minimal_multitask.utils import encode_with_messages_format
from minimal_multitask.data import DATASETS, FileDataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from datamodules.instruct_json_data_module import InstructJsonDataModule
from train.model_utils import *
import numpy as np



def get_val_dataset(
    eval_dataset_name: str,
    tokenizer,
    seed: int,
    prompt_only: bool,
    label_only: bool,
    val_num_samples: int,  # Renamed parameter
) -> Dataset:
    if eval_dataset_name in DATASETS:
        return DATASETS[eval_dataset_name](tokenizer).get_all_test_prompts(
            num_samples=val_num_samples, seed=seed, prompt_only=prompt_only, response_only=label_only
        )
    else:
        raise ValueError(f"Invalid evaluation dataset: {eval_dataset_name}")


def setup(
    model_name: Optional[str] = None,
    train_data_dir: Path = Path("/data/datasets/hf_cache/data/fineweb/sample-10BT/val"),
    resume: Union[bool, Path] = "/data/group_data/cx_group/date-models/Llama-3.1-8B-tulu3-opt/final",
    out_dir: Path = Path("out/pretrain"),
    rank: int = 0,
    devices: Union[int, str] = "auto",
    subset_size: int = 200000,  # Parameterized subset size
    val_dataset_name: str = "mmlu",  # Parameterized validation dataset name
    val_num_samples: int = 100,  # Renamed parameter for number of samples in validation dataset
) -> None:
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed everything
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Load model and optimizer
    checkpoint_dir = Path(resume)
    pretrained_checkpoint_dir = Path("/data/group_data/cx_group/date-models/Llama-3.1-8B")
    optimizer_config = {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 2e-5, "weight_decay": 0.0},
    }
    model, optimizer = load_lora_model_and_optimizer(
        None,  # Fabric is no longer used
        checkpoint_dir=checkpoint_dir,
        pretrained_checkpoint_dir=pretrained_checkpoint_dir,
        optimizer_config=optimizer_config,
        load_optimizer=False,
    )

    print("partial loaded model")
    model = model.to(device)
    print(f"Loaded model and optimizer from {resume}")

    # Prepare train dataset
    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
    train_data_module = InstructJsonDataModule(
        train_data_path=str(train_data_dir),
        batch_size=1,
        tokenizer=tokenizer,
        max_seq_length=2048,
        subset_size=subset_size, 
    )
    train_data_module.setup()
    train_dataset = train_data_module.train_dataset
    train_dataloader = train_data_module.train_dataloader()

    # Prepare validation dataloader
    val_dataset = get_val_dataset(
        val_dataset_name, tokenizer, seed=42, prompt_only=False, label_only=False, val_num_samples=val_num_samples
    )  # Use renamed parameter
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, pin_memory=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    # Compute representation similarity
    scores = compute_representation_similarity(device, model, train_dataloader, val_dataloader)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_dir, scores)
    print(f"Probing completed. results saved to {out_dir}")



def compute_representation_similarity(device, model, train_dataloader, val_dataloader):
    """Compute representation similarity between train and validation data."""
    # Cache validation representations
    print("Caching validation representations...")
    val_reps = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Val Reps"):
            input_ids = batch["input_ids"].to(device)
            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :]  # shape [batch_size, hidden_size]
            val_reps.append(last_token_reps)
    val_reps = torch.cat(val_reps, dim=0)  # [total_val_samples, hidden_size]
    print(f"Validation representation shape: {val_reps.shape}")

    # Compute train-to-validation representation similarity
    print("Computing train-to-validation representation similarity...")
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Train Similarities"):
            input_ids = batch["input_ids"].to(device)
            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :]  # shape [batch_size, hidden_size]

            # Compute cosine similarity
            influence_scores = torch.matmul(last_token_reps, val_reps.transpose(0, 1))
            mean_scores = influence_scores.mean(dim=1)

            all_scores.extend(mean_scores.cpu().tolist())

    return np.array(all_scores, dtype=np.float32)

def get_tokenizer(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)