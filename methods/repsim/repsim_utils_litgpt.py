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
from datasets import Dataset, Features, Sequence, Value
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
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
from litdata.streaming import StreamingDataset, TokensLoader

def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    train_data_dir: Path = Path("/data/group_data/cx_group/data/fineweb/train/0"),
    out_dir: Path = Path("/data/group_data/cx_group/data/fineweb/train/0/repsim"),
    precision: Optional[str] = None,
    initial_checkpoint_dir: Optional[Path] = Path("/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000"),
    shard_size: int = 64000,
    rank: int = 0, 
    train: TrainArgs = TrainArgs(
        global_batch_size=32,
        micro_batch_size=4,
    ),
    devices: Union[int, str] = 1,
    seed: int = 42,
):
    """Compute forward representation similarity between train and validation data for a GPT-style model."""
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

    train_dataset = StreamingDataset(
        input_dir=str(train_data_dir),
        item_loader=TokensLoader(block_size=model.max_seq_length + 1),
        drop_last=True,
    )

    train_dataset = train_dataset[
        rank * shard_size : min((rank + 1) * shard_size, len(train_dataset))
    ]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train.global_batch_size,
        pin_memory=True,
    )

    def val_collate_fn(batch):
        input_ids = [
            torch.tensor(sample["input_ids"], device="cuda") for sample in batch
        ]
        labels = [torch.tensor(sample["labels"], device="cuda") for sample in batch]

        x = pad_sequence(input_ids, batch_first=True, padding_value=0)
        y = pad_sequence(labels, batch_first=True, padding_value=-1)

        max_seq_length = model.max_seq_length
        if max_seq_length:
            x = x[:, :max_seq_length]
            y = y[:, :max_seq_length]

        return {"input_ids": x, "labels": y}

    val_dataloader = DataLoader(
        torch.load("/data/group_data/cx_group/data/lambada_openai/train/train-1024.pt"),
        batch_size=train.micro_batch_size,
        collate_fn=val_collate_fn,
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # First compute the final layer representations for the entire val set
    fabric.print("Caching validation representations...")
    val_reps = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Val Reps"):
            # final hidden: shape [batch_size, seq_len, hidden_size]
            input_ids = batch["input_ids"]
            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :]  # shape [batch_size, hidden_size]
            val_reps.append(last_token_reps)
    val_reps = torch.cat(val_reps, dim=0)  # [total_val_samples, hidden_size]
    fabric.print(f"Val representation shape: {val_reps.shape}")


    # For each train sample, compute forward pass, get last layer rep, average cos sim w.r.t val_reps
    fabric.print("Computing train->val representation similarity...")
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Train Similarities"):
            input_ids = batch[:, :-1].contiguous().long()
            hidden_states = model(input_ids)
            last_token_reps = hidden_states[:, -1, :]  # shape [batch_size, hidden_size]

            # [batch_size, hidden_size] @ [hidden_size, val_size] = [batch_size, val_size]
            influence_scores = torch.matmul(last_token_reps, val_reps.transpose(0, 1))
            mean_scores = influence_scores.mean(dim=1)

            all_scores.extend(mean_scores.cpu().tolist())

    all_scores = np.array(all_scores)
    fabric.print(f"all_scores shape = {all_scores.shape}")

    # save the final scores
    shard_out_dir = out_dir / f"{rank}"
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_out_dir / f"metrics.npy", all_scores)
    fabric.print("Done!")
    # return all_scores


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

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
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