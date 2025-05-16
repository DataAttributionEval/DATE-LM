import os
from pathlib import Path
from tqdm import tqdm
import random
import torch

from litgpt import Tokenizer
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.tasks import TaskManager

def encode_pair(tokenizer: Tokenizer, context: str, continuation: str):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation, bos=False, eos=False).tolist()
    context_enc = tokenizer.encode(context, bos=False, eos=False).tolist()
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc


def prepare_sample(
    task: Task,
    example: dict,
    tokenizer: Tokenizer,
    ignore_index: int,
) -> dict:
    context = task.doc_to_text(example)
    target = task.doc_to_target(example)
    context_enc, continuation_enc = encode_pair(tokenizer, context, target)
    return {
        "input_ids": context_enc + continuation_enc,
        "labels": [ignore_index] * len(context_enc) + continuation_enc,
    }

task_map = {
    "lambada_openai": "lambada_openai",
    "lambada": "lambada_standard",
    "openbookqa": "openbookqa",
    "hellaswag": "hellaswag",
    "wikitext": "wikitext",
    "squad": "squad",
    "arce": "arc_easy",
    "piqa": "piqa",
    "sciq": "sciq",
    "copa": "copa",
}


def prepare_lambada(
    # destination_path: Path = Path("/data/user_data/hanzhanz"),

    # for local testing
    # add to config later
    base_dir,
    tokenizer_dir: Path = Path(
        "checkpoints/EleutherAI/pythia-1b"
    ),
    ignore_index: int = -1,
    task_name: str = "lambada_openai",
) -> None:
    random.seed(1234)

    train_destination_path = Path(base_dir) / "data" / task_name / "train"
    val_destination_path = Path(base_dir) / "data" / task_name / "val"
    train_destination_path.mkdir(parents=True, exist_ok=True)
    val_destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading data file...")
    
    tm = TaskManager()
    config_dict = tm._get_config(task_map[task_name])
    task = ConfigurableTask(config=config_dict)
    print("Config file loaded...")

    if task_name == "lambada_openai":
        train_set = list(task.test_docs())
        val_set = list(task.test_docs())
    elif task_name == "lambada":
        train_set = list(task.validation_docs())
        val_set = list(task.test_docs())
    else:
        train_set = list(task.training_docs())
        val_set = list(task.validation_docs())

    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(val_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            task,
            example=sample,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    random.shuffle(train_set)
    train_set = train_set[:1024]
    torch.save(train_set, train_destination_path / "train-1024.pt")

    print("Processing val split ...")
    val_set = [
        prepare_sample(
            task,
            example=sample,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        for sample in tqdm(val_set)
    ]
    torch.save(val_set, val_destination_path / "val.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare_lambada)
