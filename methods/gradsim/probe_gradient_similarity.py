import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from datasets import Dataset, Features, Sequence, Value
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy, instantiate_torch_optimizer, num_parameters, parse_devices
from litdata.streaming import StreamingDataset, TokensLoader

# Global cache for validation gradients
validate_grad_cache = {}


def setup(
    model_name: Optional[str] = None,
    train_data_dir: Path = None,
    reference_data_dir: Optional[Path] = None,
    reference_data_size: Optional[int] = None,  # parameter to limit validation data size
    resume: Union[bool, Path] = False,
    out_dir: Path = Path("out/pretrain"),
    shard_size: int = 64000,
    rank: int = 0,
    devices: Union[int, str] = "auto",
) -> None:
    devices = parse_devices(devices)
    fabric = L.Fabric(devices=devices, precision="bf16-mixed")
    fabric.launch()

    fabric.seed_everything(42)

    # Initialize model
    config = Config.from_name(model_name)
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Total parameters: {num_parameters(model):,}")
    model = fabric.setup(model)

    # Initialize optimizer
    optimizer = instantiate_torch_optimizer(
        {"class_path": "torch.optim.AdamW", "init_args": {"lr": 0.0001, "weight_decay": 0.1}}, 
        model.parameters()
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Load checkpoint
    fabric.print(f"Resuming training from {resume}")
    state = fabric.load(resume)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    fabric.print(f"Loaded model from {resume}")

    # Prepare train dataset
    train_dataset = StreamingDataset(
        input_dir=str(train_data_dir),
        item_loader=TokensLoader(block_size=config.block_size + 1),
        drop_last=True,
    )
    train_dataset = train_dataset[
        rank * shard_size : ((rank + 1) * shard_size if rank + 1 < 128 else len(train_dataset))
    ]
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Prepare validation dataloader
    if reference_data_dir is None:
        raise ValueError("reference_data_dir must be provided for validation data.")
    val_dataset = torch.load(reference_data_dir, map_location="cpu")

    # Limit the size of the validation dataset if val_data_size is specified
    if reference_data_size is not None:
        val_dataset = val_dataset[:reference_data_size]
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16, # TODO 16 for lamabada, 4 for FLAN
        collate_fn=lambda batch: {
            "input_ids": pad_sequence([torch.tensor(sample["input_ids"]) for sample in batch], batch_first=True),
            "labels": pad_sequence([torch.tensor(sample["labels"]) for sample in batch], batch_first=True, padding_value=-1),
        },
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Training loop
    train_iterator = iter(train_dataloader)
    oracle = []
    for cnt, train_data in enumerate(tqdm(train_iterator), start=1):
        scores = fit(fabric, {"model": model, "optimizer": optimizer}, train_data, val_dataloader)
        fabric.print(f"Scores: {scores}")
        oracle.append({"input_ids": train_data[0].cpu().numpy().tolist(), "scores": scores})

        if cnt % 2000 == 0 or cnt == len(train_dataset):
            processed_ds = Dataset.from_list(
                oracle,
                features=Features({"input_ids": Sequence(Value("int32")), "scores": Sequence(Value("float32"))}),
            )
            processed_ds.save_to_disk(f"{out_dir}/{rank}")

    fabric.print(f"Training completed.")


def fit(
    fabric: L.Fabric,
    state: dict,
    train_data: torch.Tensor,
    val_dataloader: DataLoader,
) -> torch.Tensor:
    model = state["model"]
    val_iter = iter(val_dataloader)
    return gradient_cos(fabric, train_data, train_data, model, val_iter)


def gradient_cos(fabric, input_ids, labels, model, val_iter):
    def collect_grad(model):
        return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

    def get_validate_grad():
        cache_key = "validate_grad"
        if cache_key in validate_grad_cache:
            return validate_grad_cache[cache_key]

        model.zero_grad()
        accumulated_loss = 0.0
        cnt = 0
        for i, inputs in enumerate(val_iter):
            input_ids, targets = inputs["input_ids"], inputs["labels"]
            logits = model(input_ids)
            accumulated_loss += chunked_cross_entropy(
                logits[..., :-1, :], targets[..., 1:], chunk_size=0, ignore_index=-1
            )
            cnt += 1

            if (i + 1) % 4 == 0:
                accumulated_loss /= 4
                fabric.backward(accumulated_loss)
                accumulated_loss = 0.0

        if accumulated_loss > 0:
            accumulated_loss /= (cnt % 4)
            fabric.backward(accumulated_loss)

        val_grad = collect_grad(model)
        validate_grad_cache[cache_key] = val_grad
        return val_grad

    model.zero_grad()
    val_grad = get_validate_grad()
    model.zero_grad()

    logits = model(input_ids[:, :-1])
    shift_logits = logits.contiguous().float()
    shift_labels = labels[:, 1:].contiguous().long()
    pertoken_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    perseq_loss = torch.mean(pertoken_loss, dim=1)

    weights = torch.zeros(input_ids.size(0), device=fabric.device)
    for i, sample_loss in enumerate(perseq_loss):
        fabric.backward(sample_loss, retain_graph=True)
        sample_grad = collect_grad(model)
        weights[i] = torch.nn.functional.cosine_similarity(sample_grad, val_grad, dim=-1)
        model.zero_grad()

    return weights


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)