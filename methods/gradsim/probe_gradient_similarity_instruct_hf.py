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
from train.model_utils import load_lora_model_and_optimizer
import numpy as np
from train.model_utils import *


# Global cache for validation gradients
validate_grad_cache = {}


def get_val_dataset(
    eval_dataset_name: str,
    tokenizer,
    seed: int,
    prompt_only: bool,
    label_only: bool,
    val_num_samples: int,  # Renamed parameter
) -> Dataset:
    print("getting val dataset with prompt_only", prompt_only, "label_only", label_only)
    if eval_dataset_name in DATASETS:
        if val_num_samples <= 0:
            print("val_num_samples <= 0, using all test prompts")
            return DATASETS[eval_dataset_name](tokenizer).get_all_test_prompts(
                seed=seed, prompt_only=prompt_only, response_only=label_only
            )
        return DATASETS[eval_dataset_name](tokenizer).get_all_test_prompts(
            num_samples=val_num_samples, seed=seed, prompt_only=prompt_only, response_only=label_only
        )
    else:
        raise ValueError(f"Invalid evaluation dataset: {eval_dataset_name}")

def setup(
    base_model_name: Optional[str] = None,
    lora_ckpt: Optional[str] = None,
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
    # checkpoint_dir = Path(resume)
    # pretrained_checkpoint_dir = Path("/data/group_data/cx_group/date-models/Llama-3.1-8B")
    # optimizer_config = {
    #     "class_path": "torch.optim.AdamW",
    #     "init_args": {"lr": 2e-5, "weight_decay": 0.0},
    # }
    # model, optimizer = load_lora_model_and_optimizer(
    #     None,  # Fabric is no longer used
    #     checkpoint_dir=checkpoint_dir,
    #     pretrained_checkpoint_dir=pretrained_checkpoint_dir,
    #     optimizer_config=optimizer_config,
    #     load_optimizer=True,
    # )
    # print("partial loaded model")
    # model = model.to(device)

    model, tokenizer = checkpoints_load_func( # pad token is [PAD]
        None,
        lora_ckpt,
        base_model_name#"meta-llama/Llama-3.1-8B"
    )
    # model = model.to(device)


    # print(f"Loaded model and optimizer from {resume}")

    # Prepare train dataset
    # tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
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

    # Training loop
    train_iterator = iter(train_dataloader)
    output = []
    for cnt, train_data in enumerate(tqdm(train_iterator), start=1):
        scores = fit(device, {"model": model}, train_data, val_dataloader)
        # print(f"Scores: {scores}")
        # oracle.append({"input_ids": train_data["input_ids"][0].cpu().numpy().tolist(), "scores": scores})
        output.append(scores[0].cpu().numpy())
        # if cnt % 2000 == 0 or cnt == len(train_dataset):
        #     processed_ds = Dataset.from_list(
        #         oracle,
        #         features=Features({"input_ids": Sequence(Value("int32")), "scores": Sequence(Value("float32"))}),
        #     )
        #     processed_ds.save_to_disk(f"{out_dir}/{rank}")

    output_array = np.array(output, dtype=np.float32)
    print(output_array.shape, output_array[:100])
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_dir, output_array)

    print(f"Probing completed. results saved to {out_dir}")


def fit(
    device: torch.device,
    state: dict,
    train_data: torch.Tensor,
    val_dataloader: DataLoader,
) -> torch.Tensor:
    model = state["model"]
    val_iter = iter(val_dataloader)
    return gradient_cos(device, train_data["input_ids"].to('cuda'), train_data["labels"].to('cuda'), model, val_iter)


def gradient_cos(device, input_ids, labels, model, val_iter):
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
            input_ids, targets = inputs["input_ids"].to('cuda'), inputs["labels"].to('cuda')
            logits = model(input_ids)[0] # change for hf model
            accumulated_loss += chunked_cross_entropy(
                logits[..., :-1, :], targets[..., 1:], chunk_size=0, ignore_index=-100
            )
            cnt += 1

            if (i + 1) % 2 == 0: # less = slower but saves memory
                accumulated_loss /= 2
                accumulated_loss.backward()
                accumulated_loss = 0.0

        if accumulated_loss > 0:
            accumulated_loss /= (cnt % 2)
            accumulated_loss.backward()

        val_grad = collect_grad(model)
        validate_grad_cache[cache_key] = val_grad
        return val_grad

    model.zero_grad()
    val_grad = get_validate_grad() # 100M
    model.zero_grad()
    torch.cuda.empty_cache()
    logits = model(input_ids[:, :-1])[0]
    shift_logits = logits.contiguous().float()
    shift_labels = labels[:, 1:].contiguous().long()
    pertoken_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    perseq_loss = torch.mean(pertoken_loss, dim=1)

    weights = torch.zeros(input_ids.size(0), device=device)
    for i, sample_loss in enumerate(perseq_loss):
        sample_loss.backward(retain_graph=True)
        sample_grad = collect_grad(model)
        weights[i] = torch.nn.functional.cosine_similarity(sample_grad, val_grad, dim=-1)
        model.zero_grad()
        del sample_grad
        torch.cuda.empty_cache()

    return weights


# def get_tokenizer(model_ckpt):
#     tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     return tokenizer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)