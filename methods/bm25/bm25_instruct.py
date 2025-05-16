import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import bm25s
from jsonargparse import CLI

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from minimal_multitask.data import DATASETS
from datamodules.instruct_json_data_module import InstructJsonDataModule


def get_tokenizer(model_ckpt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_val_dataset(
    eval_dataset_name: str,
    tokenizer,
    seed: int,
    prompt_only: bool,
    label_only: bool,
    val_num_samples: int,
):
    if eval_dataset_name in DATASETS:
        return DATASETS[eval_dataset_name](tokenizer).get_all_test_prompts(
            num_samples=val_num_samples, seed=seed, prompt_only=prompt_only, response_only=label_only
        )
    else:
        raise ValueError(f"Invalid evaluation dataset: {eval_dataset_name}")


def decode_dataset_to_texts(
    dataset,
    tokenizer,
    batch_size: int = 8,
) -> list[str]:
    """
    Turn a huggingface Dataset or list-of-dicts of tokenized examples
    into a list of decoded strings.
    """
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    texts = []
    for batch in tqdm(loader, desc="Decoding dataset"):
        for seq in batch["input_ids"]:
            texts.append(tokenizer.decode(seq, skip_special_tokens=True))
    return texts


def compute_bm25_scores(
    train_loader: DataLoader,
    val_texts: list[str],
    tokenizer,
    stopwords: str = "en",
    k: int = 1024,
) -> np.ndarray:
    # 1) tokenize & index the entire val set
    print(f"Indexing {len(val_texts)} validation examples in BM25…")
    val_tokens = bm25s.tokenize(val_texts, stopwords=stopwords)
    retriever = bm25s.BM25()
    retriever.index(val_tokens)

    # 2) for each train example, decode → tokenize → retrieve → average
    scores: list[float] = []
    for batch in tqdm(train_loader, desc="Scoring train examples"):
        # train batch_size==1 by design, so squeeze dim 0
        ids = batch["input_ids"].squeeze(0)
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        tok = bm25s.tokenize([txt], stopwords=stopwords)
        result = retriever.retrieve(tok, k=min(k, len(val_texts)))
        avg_score = float(result.scores.mean())
        scores.append(avg_score)

    return np.array(scores, dtype=np.float32)


def main(
    train_data_dir: str = "/data/hf_cache/datasets/lsds_data/data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl",
    val_dataset_name: str = "mmlu",
    val_num_samples: int = 100,
    subset_size: int = 200000,
    out_dir: str = "out/bm25_output.npy",
    model_ckpt: str = "meta-llama/Llama-3.1-8B",
):
    tokenizer = get_tokenizer(model_ckpt)

    # Prepare train dataset
    train_dm = InstructJsonDataModule(
        train_data_path=train_data_dir,
        batch_size=1,
        tokenizer=tokenizer,
        max_seq_length=2048,
        subset_size=subset_size,
    )
    train_dm.setup()
    train_loader = train_dm.train_dataloader()

    # Prepare validation dataset
    val_ds = get_val_dataset(val_dataset_name, tokenizer, seed=42, prompt_only=False, label_only=False, val_num_samples=val_num_samples)
    val_texts = decode_dataset_to_texts(val_ds, tokenizer, batch_size=8)

    # Compute and save BM25 scores
    bm25_scores = compute_bm25_scores(train_loader, val_texts, tokenizer)
    os.makedirs(Path(out_dir).parent, exist_ok=True)
    np.save(out_dir, bm25_scores)
    print(f"Saved {bm25_scores.shape[0]} scores → {out_dir}")


if __name__ == "__main__":
    CLI(main)