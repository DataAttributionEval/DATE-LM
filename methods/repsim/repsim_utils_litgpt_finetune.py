# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from functools import partial
import math
import pprint
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import lightning as L
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule
from litgpt.model import GPT, Config, LLaMAMLP, CausalSelfAttention
from litgpt.utils import (
    capture_hparams,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)
from lightning.fabric.strategies import FSDPStrategy
from litdata.streaming import StreamingDataset, TokensLoader
from transformers import default_data_collator, AutoTokenizer

def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    # For finetuning:
    train_json_path: Optional[str] = None, 
    val_json_path: Optional[str] = None,
    out_dir: Path = Path("/data/group_data/cx_group/data/fineweb/train/0/repsim"),
    initial_checkpoint_dir: Optional[Path] = Path(
        "/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000"
    ),
    shard_size: int = 64000,
    rank: int = 0, 
    precision: Optional[str] = None,
    train: TrainArgs = TrainArgs(
        global_batch_size=16,
        micro_batch_size=4,
    ),
    devices: Union[int, str] = "auto",
    seed: int = 42,
) -> np.ndarray:
    """
    Compute forward representation similarity between train and validation data for a GPT-style model.
        - Train data: JSON file with "text" and "target" fields
        - Val/Reference data: JSON file with same structure
    """
    hparams = capture_hparams()

    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        available_models = "\n".join(sorted(name_to_config))
        raise ValueError(
            f"Please specify --model_name <model_name>. Available values:\n{available_models}"
        )

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=False)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision)
    fabric.launch()
    fabric.print(pprint.pformat(hparams))
    fabric.seed_everything(seed)

    fabric.print(f"Loading model config:\n{config}")
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)
    fabric.print(f"Total model parameters: {num_parameters(model):,}")

    model = fabric.setup(model)

    checkpoint_path = initial_checkpoint_dir / "lit_model.pth"
    state = fabric.load(checkpoint_path)

    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    fabric.print(f"Loading checkpoint from {initial_checkpoint_dir}")

    model.eval()

    fabric.print("==== Prepare finetuning dataset ====")
    if not train_json_path or not val_json_path:
        raise ValueError("In finetune mode, please specify --train_json_path and --val_json_path")

    # tokenizer = Tokenizer(initial_checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(initial_checkpoint_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    from datasets import load_dataset
    raw_train = load_dataset("json", data_files=train_json_path, split="train")
    raw_val = load_dataset("json", data_files=val_json_path, split="train")

    # shard the train split by (rank, shard_size)
    total = len(raw_train)
    start = rank * shard_size
    end = min((rank + 1) * shard_size, total)
    shards = list(range(start, end))
    raw_train = raw_train.select(shards)

    # helper to encode a list of records into a list of 1D torch tensors
    def encode_records(records):
        out = []
        for rec in records:
            text = rec["text"]
            target = rec.get("target", "")
            merged = f"{text}\n{target}"
            toks = tokenizer(merged, padding='max_length', max_length=model.max_seq_length, truncation=True, return_tensors="pt")
            out.append(toks["input_ids"].squeeze(0))
        return out

    train_tensors = encode_records(raw_train)
    val_tensors = encode_records(raw_val)

    # tiny Dataset wrapper over those tensors
    class InputIDsDataset(Dataset):
        def __init__(self, tensors): self.tensors = tensors
        def __len__(self): return len(self.tensors)
        def __getitem__(self, idx): return self.tensors[idx]

    train_ds = InputIDsDataset(train_tensors)
    val_ds = InputIDsDataset(val_tensors)

    # collate: pad to longest in batch
    def collate_fn(batch: List[torch.Tensor]):
        return {
            "input_ids": pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        }

    train_loader = DataLoader(
        train_ds,
        batch_size = train.global_batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        pin_memory = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = train.micro_batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        pin_memory = True,
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_loader, val_loader)

    fabric.print("Caching validation representations…")
    model.eval()
    val_reps: List[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(val_dataloader, desc="Val Reps"):
            input_ids = batch["input_ids"].to(fabric.device)
            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :]  # [B, H]
            # move off GPU right away
            val_reps.append(last_token_reps.cpu())
    val_reps = torch.cat(val_reps, dim=0)  # CPU tensor [N_val, H]
    torch.cuda.empty_cache()  # free up everything else
    fabric.print(f"Val representation shape: {val_reps.shape}")


    fabric.print("Computing train->val representation similarity...")
    all_scores = []

    with torch.inference_mode():
        for batch in tqdm(train_dataloader, desc="Train Similarities"):
            input_ids = batch["input_ids"][:, :-1].to(fabric.device)

            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :].cpu()  # shape [batch_size, hidden_size]

            # compute influence scores
            influence_scores = torch.matmul(last_token_reps, val_reps.transpose(0, 1))
            mean_scores = influence_scores.mean(dim=1)
            all_scores.extend(mean_scores.tolist())

    all_scores = np.array(all_scores)
    fabric.print(f"all_scores shape = {all_scores.shape}")

    # save the final scores
    shard_out_dir = out_dir / f"{rank}"
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_out_dir / f"metrics.npy", all_scores)
    fabric.print("Done!")


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama
    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(
                init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd)
            )

    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(
                init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer)
            )

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(setup)